#!/usr/bin/env python3
"""Generate the fixed reference-free p003 adapter evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

SEEDS = (42, 43, 44)
BASE_TEXT = "I did not expect that to happen today."
NORMAL_TEXT = "The evening light settled softly across the quiet room."
EVENTS = (
    ("laugh", "(laugh)"),
    ("sigh", "(sigh)"),
    ("cough", "(cough)"),
    ("sniff", "(sniff)"),
    ("sneeze", "(sneeze)"),
    ("clears_throat", "(clears throat)"),
    ("yawn", "(yawn)"),
    ("crying", "(crying)"),
    ("scream", "(scream)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--instruction", default="Speak clearly and naturally.")
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_requests(instruction: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(
            {
                "id": f"normal_seed-{seed}",
                "family": "normal",
                "event": None,
                "variant": "normal",
                "seed": seed,
                "text": NORMAL_TEXT,
                "instruction": instruction,
            }
        )
    for event, tag in EVENTS:
        for seed in SEEDS:
            for variant, text in (
                ("tagged", f"{tag} {BASE_TEXT}"),
                ("control", BASE_TEXT),
            ):
                rows.append(
                    {
                        "id": f"{event}_{variant}_seed-{seed}",
                        "family": "event",
                        "event": event,
                        "tag": tag,
                        "variant": variant,
                        "seed": seed,
                        "text": text,
                        "instruction": instruction,
                    }
                )
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def audio_receipt(
    path: Path,
    *,
    codec_frames: int,
    max_new_tokens: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    duration = float(audio.size / sample_rate) if sample_rate else 0.0
    invalid = bool(
        audio.size == 0
        or sample_rate <= 0
        or not np.isfinite(audio).all()
        or duration < 0.25
    )
    silent = bool(peak < 1e-3 or rms < 1e-4)
    truncated = bool(codec_frames >= max_new_tokens)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_rate": int(sample_rate),
        "samples": int(audio.size),
        "duration_seconds": duration,
        "peak_absolute": peak,
        "rms": rms,
        "codec_frames": codec_frames,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "invalid": invalid,
        "silent": silent,
        "truncated": truncated,
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.max_seq_len <= args.max_new_tokens:
        raise ValueError("max token limits must be positive and fit max sequence length")
    requests = evaluation_requests(args.instruction)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        requests = requests[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "generation-manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        existing = {str(row["id"]): row for row in manifest.get("items", [])}

    device = args.device or resolve_device()
    tokenizer, model, audio_tokenizer = load_runtime(
        args.model_root,
        device=device,
        attn_implementation="eager",
        adapter_root=args.adapter_root,
        base_revision=args.base_revision,
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            fast_all=args.fast_all,
            repetition_penalty=1.1,
        ),
        tokenizer=tokenizer,
    )

    completed = dict(existing)
    for index, row in enumerate(requests, start=1):
        output_path = args.output_root / f"{row['id']}.wav"
        old = completed.get(row["id"])
        if old and output_path.is_file() and old.get("audio", {}).get("sha256") == sha256_file(output_path):
            print(f"[{index}/{len(requests)}] retained {row['id']}", flush=True)
            continue

        set_all_seeds(int(row["seed"]))
        request = {
            "id": row["id"],
            "text": row["text"],
            "instruction": row["instruction"],
            "speaker": "S0",
        }
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template("tts_instruction"),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        started = time.perf_counter()
        codec_frames = 0
        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=runtime.sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as output_file:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=row["id"]):
                output_file.write(chunk.audio)
                codec_frames += int(chunk.codec_frames)
        elapsed = time.perf_counter() - started
        completed[row["id"]] = {
            **row,
            "audio": audio_receipt(
                output_path,
                codec_frames=codec_frames,
                max_new_tokens=args.max_new_tokens,
                elapsed_seconds=elapsed,
            ),
        }
        ordered = [completed[item["id"]] for item in evaluation_requests(args.instruction) if item["id"] in completed]
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "in_progress",
                "base_model": str(args.model_root.resolve()),
                "base_revision": args.base_revision,
                "adapter": str(args.adapter_root.resolve()),
                "device": device,
                "device_name": torch.cuda.get_device_name(device),
                "max_new_tokens": args.max_new_tokens,
                "fast_all": args.fast_all,
                "items": ordered,
            },
        )
        print(
            f"[{index}/{len(requests)}] generated {row['id']} "
            f"({completed[row['id']]['audio']['duration_seconds']:.2f}s audio, "
            f"{elapsed:.2f}s wall)",
            flush=True,
        )

    all_requests = evaluation_requests(args.instruction)
    ordered = [completed[item["id"]] for item in all_requests if item["id"] in completed]
    status = "complete" if len(ordered) == len(all_requests) else "partial"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": status,
            "base_model": str(args.model_root.resolve()),
            "base_revision": args.base_revision,
            "adapter": str(args.adapter_root.resolve()),
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "max_new_tokens": args.max_new_tokens,
            "fast_all": args.fast_all,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "items": ordered,
        },
    )
    print(f"wrote {manifest_path} ({status}, {len(ordered)}/{len(all_requests)})")


if __name__ == "__main__":
    main()
