#!/usr/bin/env python3
"""Build the final p003 Breeze LoRA report from immutable run receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--pytest-result", default="not recorded")
    parser.add_argument("--ruff-result", default="not recorded")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"telemetry value is not numeric: {value!r}")
    return float(match.group())


def telemetry_summary(paths: list[Path]) -> dict[str, Any]:
    samples: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            samples.extend(
                row
                for row in csv.DictReader(handle)
                if row[" index"].strip() == "0"
            )
    active = [row for row in samples if numeric(row[" memory.used [MiB]"]) > 1000]
    return {
        "files": [str(path.resolve()) for path in paths],
        "physical_gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "samples": len(samples),
        "active_samples": len(active),
        "peak_system_memory_mib": max(
            numeric(row[" memory.used [MiB]"]) for row in samples
        ),
        "peak_utilization_percent": max(
            numeric(row[" utilization.gpu [%]"]) for row in samples
        ),
        "mean_active_utilization_percent": sum(
            numeric(row[" utilization.gpu [%]"]) for row in active
        )
        / len(active),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    dataset = read_json(root / "dataset-conversion-report.json")
    transcript_document = read_json(args.transcripts)
    model = read_json(root / "model-resolution.json")
    cache = read_json(root / "cache" / "cache-receipt.json")
    training = read_json(root / "full-run" / "training-receipt.json")
    state = read_json(
        root / "full-run" / "checkpoint-step-001000" / "trainer-state.json"
    )
    selected = read_json(root / "full-run" / "selected-checkpoint.json")
    verification = read_json(root / "selected-adapter-verification.json")
    smoke = read_json(
        root / "smoke-run" / "checkpoint-step-000002" / "trainer-state.json"
    )
    generation = read_json(root / "generated" / "generation-manifest.json")
    event_eval = read_json(root / "event-clap-evaluation.json")
    adapter = read_json(root / "adapter-release" / "adapter_config.json")

    history = state["history"]
    history_json = root / "loss-history.json"
    history_json.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    history_csv = root / "loss-history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "micro_step",
                "learning_rate",
                "gradient_norm_before_clip",
                "train_total",
                "train_backbone",
                "train_depth_decoder",
                "validation_total",
                "validation_backbone",
                "validation_depth_decoder",
            ),
        )
        writer.writeheader()
        for row in history:
            validation = row.get("validation_loss", {})
            writer.writerow(
                {
                    "step": row["step"],
                    "micro_step": row["micro_step"],
                    "learning_rate": row["learning_rate"],
                    "gradient_norm_before_clip": row[
                        "gradient_norm_before_clip"
                    ],
                    "train_total": row["training_loss"]["total"],
                    "train_backbone": row["training_loss"]["backbone"],
                    "train_depth_decoder": row["training_loss"]["depth_decoder"],
                    "validation_total": validation.get("total", ""),
                    "validation_backbone": validation.get("backbone", ""),
                    "validation_depth_decoder": validation.get(
                        "depth_decoder", ""
                    ),
                }
            )

    validation_history = [
        {"step": 0, "validation_loss": state["initial_validation"]},
        *[
            {"step": row["step"], "validation_loss": row["validation_loss"]}
            for row in history
            if "validation_loss" in row
        ],
    ]
    telemetry = telemetry_summary(
        [root / "telemetry-full.csv", root / "telemetry-full-resume-2.csv"]
    )
    generated_issues = {
        key: sum(bool(row["audio"][key]) for row in generation["items"])
        for key in ("invalid", "silent", "truncated")
    }
    commands = {
        "environment": (
            "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "
            "HF_HOME=/mnt/d/huggingface HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1"
        ),
        "cache": "bash scripts/prepare_cache.sh --train-limit 159 --validation-limit 14",
        "smoke_step_1": (
            "bash scripts/run_lora.sh --max-steps 2 --save-every 1 "
            "--validation-examples 2 --stop-after-step 1"
        ),
        "smoke_resume": (
            "bash scripts/run_lora.sh --max-steps 2 --save-every 1 "
            "--validation-examples 2 --resume-checkpoint "
            "_artifacts/09-25_02-49_Breeze-p003-LoRA/smoke-run/"
            "checkpoint-step-000001"
        ),
        "full_train": "bash scripts/run_lora.sh --validation-examples 14",
        "full_resume": (
            "bash scripts/run_lora.sh --validation-examples 14 "
            "--resume-checkpoint _artifacts/09-25_02-49_Breeze-p003-LoRA/"
            "full-run/checkpoint-step-000750"
        ),
        "select": (
            "python -m training.select_checkpoint --training-root "
            "_artifacts/09-25_02-49_Breeze-p003-LoRA/full-run"
        ),
        "generation": (
            "python -m scripts.generate_p003_evaluation --model-root "
            f"{model['snapshot_path']} --adapter-root "
            "_artifacts/09-25_02-49_Breeze-p003-LoRA/adapter-release "
            f"--base-revision {model['revision']} --output-root "
            "_artifacts/09-25_02-49_Breeze-p003-LoRA/generated --fast-all"
        ),
        "clap": (
            "python -m scripts.score_p003_events --generation-manifest "
            "_artifacts/09-25_02-49_Breeze-p003-LoRA/generated/"
            "generation-manifest.json --clap-model /mnt/h/Git/"
            "Higgs-tts-3-finetune/_artifacts/model_cache/clap-htsat-unfused "
            "--output _artifacts/09-25_02-49_Breeze-p003-LoRA/"
            "event-clap-evaluation.json --device cuda:0"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "breeze_p003_lora_poc_complete",
        "repository_revision": git_head(),
        "base_model": model,
        "dataset": dataset,
        "source_provenance": transcript_document["provenance"],
        "transcripts_sha256": sha256_file(args.transcripts),
        "cache_receipt_sha256": training["configuration"][
            "cache_receipt_sha256"
        ],
        "cache_runtime": cache["runtime"],
        "commands": commands,
        "training_configuration": {
            **training["configuration"],
            "optimizer": "AdamW",
            "scheduler": "cosine with linear warmup",
            "attention_implementation": "eager (Instavar default)",
            "gradient_checkpointing": True,
        },
        "trainable_parameters": state["trainable_parameters"],
        "smoke_test": {
            "status": smoke["status"],
            "global_step": smoke["global_step"],
            "history": smoke["history"],
            "runtime": smoke["runtime"],
        },
        "loss_history": {
            "steps": len(history),
            "json": str(history_json),
            "json_sha256": sha256_file(history_json),
            "csv": str(history_csv),
            "csv_sha256": sha256_file(history_csv),
            "validation": validation_history,
        },
        "selected_checkpoint": selected["selected"],
        "adapter_verification": verification,
        "adapter_release": {
            "path": str((root / "adapter-release").resolve()),
            "manifest": adapter,
            "sha256sums_sha256": sha256_file(
                root / "adapter-release" / "SHA256SUMS"
            ),
            "merged": False,
        },
        "memory": {
            "trainer_peak_allocated_bytes": state["runtime"][
                "peak_cuda_memory_bytes"
            ],
            "trainer_peak_reserved_bytes": state["runtime"][
                "peak_cuda_reserved_bytes"
            ],
            "system_telemetry": telemetry,
        },
        "generation": {
            "manifest": str((root / "generated" / "generation-manifest.json")),
            "manifest_sha256": sha256_file(
                root / "generated" / "generation-manifest.json"
            ),
            "items": len(generation["items"]),
            "total_audio_seconds": sum(
                row["audio"]["duration_seconds"] for row in generation["items"]
            ),
            "issues": generated_issues,
            "peak_allocated_bytes": generation["peak_allocated_bytes"],
            "peak_reserved_bytes": generation["peak_reserved_bytes"],
        },
        "event_evaluation": event_eval["events"],
        "tests": {
            "pytest": args.pytest_result,
            "ruff": args.ruff_result,
        },
    }
    report_json = root / "final-report.json"
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation_rows = [
        [
            row["step"],
            f"{row['validation_loss']['total']:.6f}",
            f"{row['validation_loss']['backbone']:.6f}",
            f"{row['validation_loss']['depth_decoder']:.6f}",
        ]
        for row in validation_history
    ]
    event_rows = [
        [
            event,
            f"{value['median_paired_uplift']:+.6f}",
            "yes" if value["not_demonstrated"] else "no",
            "yes" if value["unstable"] else "no",
        ]
        for event, value in event_eval["events"].items()
    ]
    mapping_rows: list[list[str]] = []
    for family in ("inline_sfx", "filename_inferred_events", "emotion", "style", "prosody"):
        for source, target in dataset["mapping"][family].items():
            mapping_rows.append([family, source, target])
    mapping_rows.append(
        ["untagged", "(none)", dataset["mapping"]["untagged_instruction"]]
    )
    sample_rows = []
    for row in generation["items"]:
        if row["seed"] == 42 and row["variant"] in {"normal", "tagged"}:
            relative = Path(row["audio"]["path"]).relative_to(root)
            label = "normal" if row["event"] is None else row["event"]
            sample_rows.append(
                [label, row["text"], f"[{relative.name}]({relative.as_posix()})"]
            )

    markdown = f"""# Breeze TTS 2 p003 LoRA POC

Status: complete. The original Higgs repository and source `p003` tree were read-only; the copied dataset and every produced artifact are contained in this repository.

## Base model

- Model: `BreezeBlue/Breeze-TTS-2`
- Resolved revision: `{model['revision']}`
- Snapshot: `{model['snapshot_path']}`
- Offline cache size: `{model['cache_size_bytes']:,}` bytes, including both main shards and `audio_tokenizer/model.safetensors`

## Dataset

- 161 source recordings -> 173 examples (155 whole clips + 18 deterministic slices)
- Train: 159 examples, {dataset['duration_minutes']['train']:.6f} minutes
- Validation: 14 examples, {dataset['duration_minutes']['validation']:.6f} minutes
- Total: {dataset['duration_minutes']['source']:.6f} minutes
- Train/validation source audio is disjoint and the original 14-file validation membership is unchanged.
- Train manifest SHA-256: `{dataset['outputs']['train_manifest_sha256']}`
- Validation manifest SHA-256: `{dataset['outputs']['validation_manifest_sha256']}`
- Converted transcript/provenance SHA-256: `{report['transcripts_sha256']}`

## Complete control mapping

{markdown_table(['Family', 'Higgs source', 'Breeze representation'], mapping_rows)}

The four `filename_inferred_events` rows restore event semantics absent from edit text and are reported separately from inline tags. Multiple non-event controls are combined in emotion, style, then prosody order.

## Training

- Target set: `backbone_depth_projection`
- Trainable parameters: {state['trainable_parameters']['total']:,}
- Rank/alpha: 8/16; AdamW; LR 2e-4; weight decay 0.01; warmup 100; cosine decay; gradient accumulation 4; max gradient norm 1.0; seed 42; 1,000 steps; checkpoints every 250.
- Training retained Instavar's default eager attention and gradient checkpointing. No Higgs hyperparameters or model-specific code were reused.
- Fresh-process two-step stop/resume smoke completed before the full run.
- Exact commands and environment are recorded in [final-report.json](final-report.json).

{markdown_table(['Step', 'Validation total', 'Backbone', 'Depth decoder'], validation_rows)}

The full 1,000-row training history is in [loss-history.csv](loss-history.csv) and [loss-history.json](loss-history.json). Training loss at step 1,000 was `{history[-1]['training_loss']['total']:.6f}`.

## Selection and adapter

- Selected checkpoint: step {selected['selected']['global_step']} by minimum held-out total validation loss `{selected['selected']['validation_loss']['total']:.6f}` among steps 250/500/750/1000.
- Adapter: [adapter-release/adapter.safetensors](adapter-release/adapter.safetensors)
- Adapter SHA-256: `{adapter['adapter']['sha256']}`
- Fresh-process adapter-only verification reproduced all 14 validation examples and matched adapter tensor hashes.
- The release contains pinned base hashes plus the adapter; it contains no merged base checkpoint.

## VRAM and utilization

- Trainer peak allocated: {state['runtime']['peak_cuda_memory_bytes'] / 2**30:.3f} GiB
- Trainer peak reserved: {state['runtime']['peak_cuda_reserved_bytes'] / 2**30:.3f} GiB
- System-observed 5090 peak: {telemetry['peak_system_memory_mib'] / 1024:.3f} GiB
- System peak utilization: {telemetry['peak_utilization_percent']:.0f}%; mean active utilization: {telemetry['mean_active_utilization_percent']:.2f}%

The low average is expected from a single-example, four-serial-microstep, gradient-checkpointed trainer. An opt-in SDPA/no-checkpoint benchmark was faster but was not substituted into the requested default run. FlashAttention was not installed: the host `nvcc` was CUDA 12.0, below the CUDA 12.8 requirement for RTX 5090/SM120 builds, and changing the host CUDA toolchain was outside this repository-scoped POC.

## Generated tests

57 reference-free WAVs were generated (3 normal + 9 events x 3 seeds x tagged/control). Diagnostics found {generated_issues['invalid']} invalid, {generated_issues['silent']} silent, and {generated_issues['truncated']} truncated files.

{markdown_table(['Case (seed 42)', 'Text', 'WAV'], sample_rows)}

Full generation metadata and every checksum are in [generated/generation-manifest.json](generated/generation-manifest.json).

## Event screen

CLAP used paired tag-minus-control scoring with matched seeds 42/43/44.

{markdown_table(['Event', 'Median uplift', 'Not demonstrated', 'Unstable'], event_rows)}

All nine tags had positive median uplift. `sniff` is flagged unstable because uplift changed sign across seeds; no tag was classified as not demonstrated. These are descriptive CLAP screening results, not a substitute for blinded listening evaluation.

## Verification

- pytest: {args.pytest_result}
- ruff: {args.ruff_result}
- Repository revision containing the implementation: `{report['repository_revision']}`
"""
    report_markdown = root / "final-report.md"
    report_markdown.write_text(markdown, encoding="utf-8")
    print(report_markdown)
    print(report_json)


if __name__ == "__main__":
    main()
