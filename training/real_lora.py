from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.lora import (
    adapter_hashes,
    inject_lora,
    load_adapter,
    save_adapter,
    trainable_parameter_receipt,
)
from training.lora_study import gradient_receipt, loss_receipt
from training.model_loading import load_training_model
from training.real_data import sha256_file

VARIANT = "backbone_depth_projection"
_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the selected real Breeze LoRA")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument(
        "--text-encoder-attention-implementation",
        choices=("inherit", "flash_attention_2"),
        default="inherit",
        help=(
            "Optional text-encoder-only override. Global FlashAttention is not "
            "offered because the causal Breeze backbone/depth path produces "
            "non-finite training gradients with FA2."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-regions",
        choices=("none", "backbone", "depth", "backbone-depth"),
        default="none",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs", "reduce-overhead"),
        default="default",
    )
    parser.add_argument(
        "--compile-static",
        action="store_true",
        help="Specialize compiled regions to input shapes instead of dynamic shapes.",
    )
    parser.add_argument("--compile-optimizer", action="store_true")
    parser.add_argument(
        "--sequence-length-bucket",
        type=int,
        default=0,
        help="Right-pad model sequence tensors to this multiple; zero disables it.",
    )
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def load_cache(cache_root: Path) -> tuple[dict[str, Any], list[Path], list[Path]]:
    receipt_path = cache_root / "cache-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "deterministic_supervised_cache_complete":
        raise RuntimeError(f"cache is not complete: {receipt.get('status')}")
    train = [cache_root / row["artifact"] for row in receipt["splits"]["train"]]
    validation = [
        cache_root / row["artifact"] for row in receipt["splits"]["validation"]
    ]
    if not all(path.is_file() for path in train + validation):
        raise FileNotFoundError("cache receipt points to missing example artifacts")
    return receipt, train, validation


def example_order(length: int, *, epoch: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def example_index(micro_step: int, length: int, *, seed: int) -> int:
    epoch, offset = divmod(micro_step, length)
    return example_order(length, epoch=epoch, seed=seed)[offset]


def move_example(path: Path, device: str) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return {name: tensor.to(device) for name, tensor in value.items()}


def bucket_example_sequence(
    value: dict[str, torch.Tensor], multiple: int
) -> dict[str, torch.Tensor]:
    if multiple <= 0:
        return value
    sequence_length = int(value["input_ids"].shape[1])
    target_length = math.ceil(sequence_length / multiple) * multiple
    padding = target_length - sequence_length
    if padding == 0:
        return value

    result = dict(value)
    pad_values = {
        "input_ids": 0,
        "attention_mask": 0,
        "labels": -100,
        "text_ids_mask": False,
    }
    for name, pad_value in pad_values.items():
        tensor = result[name]
        if tensor.ndim != 2 or tensor.shape[1] != sequence_length:
            raise ValueError(f"unexpected {name} shape for sequence bucketing")
        result[name] = torch.nn.functional.pad(
            tensor, (0, padding), value=pad_value
        )
    return result


def ensure_windows_msvc_environment() -> None:
    if sys.platform != "win32":
        return
    include_paths = os.environ.get("INCLUDE", "").split(os.pathsep)
    if os.environ.get("LIB") and any(
        (Path(path) / "omp.h").is_file() for path in include_paths if path
    ):
        return

    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]
    candidates = [
        path
        for root in roots
        for path in root.glob(
            "Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"
        )
    ]
    if not candidates:
        raise RuntimeError(
            "torch.compile on Windows requires Visual Studio C++ build tools"
        )

    vcvars = max(candidates, key=lambda path: path.stat().st_mtime)
    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            os.environ[name] = value


def compile_training_regions(
    model: torch.nn.Module,
    *,
    regions: str,
    mode: str,
    dynamic: bool,
) -> int:
    if regions == "none":
        return 0
    if sys.platform == "win32":
        ensure_windows_msvc_environment()
        torch._inductor.config.use_static_cuda_launcher = False

    layers: list[torch.nn.Module] = []
    if regions in {"backbone", "backbone-depth"}:
        layers.extend(model.backbone_model.layers)
    if regions in {"depth", "backbone-depth"}:
        layers.extend(model.depth_decoder.model.layers)

    for layer in layers:
        layer.forward = torch.compile(
            layer.forward,
            mode=mode,
            fullgraph=False,
            dynamic=dynamic,
        )
    return len(layers)


def lr_multiplier(step: int, *, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def validate(
    model: torch.nn.Module,
    paths: list[Path],
    *,
    device: str,
    limit: int,
) -> dict[str, float]:
    if limit <= 0 or limit > len(paths):
        raise ValueError("validation limit must fit the cached validation split")
    sums = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
    model.eval()
    with torch.no_grad():
        for path in paths[:limit]:
            losses = loss_receipt(
                model(**move_example(path, device), use_cache=False, return_dict=True)
            )
            for name, value in losses.items():
                sums[name] += value
    model.train()
    return {name: value / limit for name, value in sums.items()}


def rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(path: Path) -> None:
    value = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state_all(value["torch_cuda"])


def checkpoint_role_hashes(checkpoint: Path) -> dict[str, str]:
    return {
        name: sha256_file(checkpoint / name)
        for name in (
            "adapter.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng.pt",
            "trainer-state.json",
        )
    }


def save_checkpoint(
    *,
    output_root: Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: dict[str, Any],
) -> Path:
    checkpoint = output_root / f"checkpoint-step-{global_step:06d}"
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    if checkpoint.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    partial.mkdir(parents=True, exist_ok=False)
    hashes = save_adapter(model, partial / "adapter.safetensors")
    if hashes != adapter_hashes(model):
        raise RuntimeError("adapter changed while saving checkpoint")
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(scheduler.state_dict(), partial / "scheduler.pt")
    torch.save(rng_state(), partial / "rng.pt")
    atomic_write_json(partial / "trainer-state.json", state)
    roles = checkpoint_role_hashes(partial)
    atomic_write_json(
        partial / "checkpoint-receipt.json",
        {
            "schema_version": 1,
            "status": "five_role_checkpoint_complete",
            "global_step": global_step,
            "roles": roles,
            "adapter_tensor_hashes": hashes,
        },
    )
    partial.rename(checkpoint)
    atomic_write_json(
        output_root / "latest.json",
        {
            "checkpoint": str(checkpoint),
            "global_step": global_step,
            "checkpoint_receipt_sha256": sha256_file(
                checkpoint / "checkpoint-receipt.json"
            ),
        },
    )
    return checkpoint


def run_configuration(
    args: argparse.Namespace, cache_receipt: dict[str, Any]
) -> dict[str, Any]:
    configuration = {
        "variant": VARIANT,
        "rank": args.rank,
        "alpha": args.alpha,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "save_every": args.save_every,
        "validation_examples": args.validation_examples,
        "seed": args.seed,
        "max_gradient_norm": args.max_gradient_norm,
        "cache_receipt_sha256": sha256_file(args.cache_root / "cache-receipt.json"),
        "train_manifest_sha256": cache_receipt["source"]["train_manifest_sha256"],
        "validation_manifest_sha256": cache_receipt["source"][
            "validation_manifest_sha256"
        ],
    }
    if args.attention_implementation != "eager":
        configuration["attention_implementation"] = args.attention_implementation
    if args.text_encoder_attention_implementation != "inherit":
        configuration["text_encoder_attention_implementation"] = (
            args.text_encoder_attention_implementation
        )
    if not args.gradient_checkpointing:
        configuration["gradient_checkpointing"] = False
    if args.compile_regions != "none":
        configuration["compile_regions"] = args.compile_regions
        configuration["compile_mode"] = args.compile_mode
        configuration["compile_dynamic"] = not args.compile_static
    if args.compile_optimizer:
        configuration["compile_optimizer"] = True
    if args.sequence_length_bucket:
        configuration["sequence_length_bucket"] = args.sequence_length_bucket
    return configuration


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real LoRA training requires an available CUDA device")
    if args.max_steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("step counts must be positive")
    if args.sequence_length_bucket < 0:
        raise ValueError("sequence-length-bucket must not be negative")
    if args.save_every <= 0 or args.max_steps % args.save_every:
        raise ValueError("save-every must divide max-steps")
    if (
        args.stop_after_step is not None
        and not 0 < args.stop_after_step < args.max_steps
    ):
        raise ValueError("stop-after-step must be inside the training interval")
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cache_receipt, train_paths, validation_paths = load_cache(args.cache_root)
    configuration = run_configuration(args, cache_receipt)
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).splitlines()
    if dirty_paths:
        raise RuntimeError(f"source repository must be clean: {dirty_paths}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "run-config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise RuntimeError("resume configuration differs from the original run")
    else:
        if args.resume_checkpoint is not None:
            raise FileNotFoundError("resume requested before run-config.json exists")
        atomic_write_json(config_path, configuration)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    model = load_training_model(
        args.model_root,
        device=args.device,
        attention_implementation=args.attention_implementation,
        text_encoder_attention_implementation=(
            None
            if args.text_encoder_attention_implementation == "inherit"
            else args.text_encoder_attention_implementation
        ),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    families = inject_lora(
        model,
        variant=VARIANT,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    parameters = trainable_parameter_receipt(model, families)
    compiled_region_count = compile_training_regions(
        model,
        regions=args.compile_regions,
        mode=args.compile_mode,
        dynamic=not args.compile_static,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
        foreach=False,
    )
    optimizer_step = optimizer.step
    if args.compile_optimizer:
        if sys.platform == "win32":
            ensure_windows_msvc_environment()
            torch._inductor.config.use_static_cuda_launcher = False
        optimizer_step = torch.compile(
            optimizer.step,
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(
            step, warmup_steps=args.warmup_steps, max_steps=args.max_steps
        ),
    )

    global_step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    initial_validation = None
    if args.resume_checkpoint is not None:
        checkpoint_receipt = json.loads(
            (args.resume_checkpoint / "checkpoint-receipt.json").read_text()
        )
        actual_roles = checkpoint_role_hashes(args.resume_checkpoint)
        if actual_roles != checkpoint_receipt["roles"]:
            raise RuntimeError("resume checkpoint role hashes do not match")
        load_adapter(model, args.resume_checkpoint / "adapter.safetensors")
        optimizer.load_state_dict(
            torch.load(
                args.resume_checkpoint / "optimizer.pt",
                map_location=args.device,
                weights_only=True,
            )
        )
        scheduler.load_state_dict(
            torch.load(
                args.resume_checkpoint / "scheduler.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        state = json.loads((args.resume_checkpoint / "trainer-state.json").read_text())
        global_step = int(state["global_step"])
        micro_step = int(state["micro_step"])
        history = list(state["history"])
        initial_validation = state["initial_validation"]
        restore_rng_state(args.resume_checkpoint / "rng.pt")
    else:
        initial_validation = validate(
            model,
            validation_paths,
            device=args.device,
            limit=args.validation_examples,
        )

    first_gradients = None
    started_at = time.time()
    model.train()
    while global_step < args.max_steps:
        torch.cuda.synchronize(args.device)
        iteration_started_at = time.time()
        iteration_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
        for _ in range(args.gradient_accumulation):
            path = train_paths[
                example_index(micro_step, len(train_paths), seed=args.seed)
            ]
            example = bucket_example_sequence(
                move_example(path, args.device), args.sequence_length_bucket
            )
            outputs = model(
                **example, use_cache=False, return_dict=True
            )
            losses = loss_receipt(outputs)
            for name, value in losses.items():
                accumulated[name] += value / args.gradient_accumulation
            (outputs.loss / args.gradient_accumulation).backward()
            micro_step += 1
        if first_gradients is None:
            first_gradients = gradient_receipt(model, families)
            for family, row in first_gradients.items():
                if row["gradient_tensors"] != row["finite_gradient_tensors"]:
                    raise RuntimeError(f"non-finite first gradients in {family}: {row}")
                if row["nonzero_gradient_tensors"] == 0:
                    raise RuntimeError(f"zero first gradients in {family}: {row}")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=args.max_gradient_norm,
            )
            .detach()
            .float()
            .cpu()
            .item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {global_step + 1}")
        optimizer_step()
        scheduler.step()
        torch.cuda.synchronize(args.device)
        iteration_seconds = time.perf_counter() - iteration_started
        global_step += 1
        row: dict[str, Any] = {
            "step": global_step,
            "micro_step": micro_step,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm_before_clip": gradient_norm,
            "training_loss": accumulated,
            "performance": {
                "started_at_unix": iteration_started_at,
                "finished_at_unix": time.time(),
                "seconds": iteration_seconds,
                "iterations_per_second": 1.0 / iteration_seconds,
            },
        }

        should_save = (
            global_step % args.save_every == 0 or global_step == args.max_steps
        )
        should_pause = args.stop_after_step == global_step or _STOP_REQUESTED
        if should_save or should_pause:
            row["validation_loss"] = validate(
                model,
                validation_paths,
                device=args.device,
                limit=args.validation_examples,
            )
        history.append(row)
        if should_save or should_pause:
            status = "paused_for_fresh_process_resume" if should_pause else "running"
            if global_step == args.max_steps:
                status = "training_complete"
            state = {
                "schema_version": 1,
                "status": status,
                "global_step": global_step,
                "micro_step": micro_step,
                "initial_validation": initial_validation,
                "history": history,
                "source_revision": revision,
                "trainable_parameters": parameters,
                "target_families": families,
                "first_step_gradients": first_gradients,
                "runtime": {
                    "elapsed_seconds_this_process": time.time() - started_at,
                    "peak_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated(args.device)
                    ),
                    "peak_cuda_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(args.device)
                    ),
                    "python": sys.version,
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(args.device),
                    "compiled_region_count": compiled_region_count,
                },
            }
            checkpoint = save_checkpoint(
                output_root=args.output_root,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
            )
            print(
                json.dumps(
                    {
                        "status": status,
                        "step": global_step,
                        "checkpoint": str(checkpoint),
                    }
                ),
                flush=True,
            )
            if should_pause and global_step < args.max_steps:
                return 75

    final_checkpoint = args.output_root / f"checkpoint-step-{args.max_steps:06d}"
    atomic_write_json(
        args.output_root / "training-receipt.json",
        {
            "schema_version": 1,
            "status": "real_lora_training_complete",
            "source": {
                "repository": str(source_root),
                "revision": revision,
                "model_root": str(args.model_root),
                "model_index_sha256": sha256_file(
                    args.model_root / "model.safetensors.index.json"
                ),
            },
            "configuration": configuration,
            "trainable_parameters": parameters,
            "target_families": families,
            "initial_validation": initial_validation,
            "final_validation": history[-1]["validation_loss"],
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_receipt_sha256": sha256_file(
                final_checkpoint / "checkpoint-receipt.json"
            ),
            "interpretation_boundary": (
                "This run establishes one bounded multi-example optimization and "
                "held-out objective result. Loss is not a substitute for blind "
                "speaker identity, accent, cadence, pronunciation, monotony, or "
                "listening-fatigue judgments."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
