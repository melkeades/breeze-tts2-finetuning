#!/usr/bin/env python3
"""Generate a large paired quality matrix for compile-training comparisons."""

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

STYLE_CASES = (
    ("whisper", "Please leave the small package beside the blue door before noon.", "Whisper the text."),
    ("shout", "The last train is leaving now, so we need to cross the platform.", "Shout the text."),
    ("sing", "A silver moon is rising slowly above the quiet harbor tonight.", "Sing the text."),
    ("fast", "We checked the tickets, packed the bags, and hurried toward the gate.", "Speak quickly."),
    ("slow", "Each wave rolled gently onto the sand and faded into silence.", "Speak slowly."),
    ("angry", "Someone moved my notes again and left the office door unlocked.", "Speak angrily."),
    ("sad", "The old photograph was still on the shelf where she had left it.", "Speak sadly."),
    ("amused", "He tried to balance three oranges and immediately dropped every one.", "Speak with amusement."),
    ("high_pitch", "A tiny bird landed on the railing and chirped at the window.", "Use a high speaking pitch."),
    ("low_pitch", "The distant thunder echoed across the valley just before dawn.", "Use a low speaking pitch."),
)

EVENT_CASES = (
    ("laugh", "(laugh)", "I honestly did not expect the meeting to end that way."),
    ("sigh", "(sigh)", "There is still another hour of paperwork waiting on the desk."),
    ("cough", "(cough)", "Could you pass me the glass of water by the lamp?"),
    ("sniff", "(sniff)", "The spring flowers filled the entire hallway with perfume."),
    ("sneeze", "(sneeze)", "Someone must have opened the dusty box in the attic."),
    ("clears_throat", "(clears throat)", "May I have everyone's attention for one brief announcement?"),
    ("yawn", "(yawn)", "We have been awake since four o'clock this morning."),
    ("crying", "(crying)", "I kept the final letter because it was all I had left."),
    ("scream", "(scream)", "Get away from the edge and come back here right now!"),
    ("cheering", "(cheering)", "They crossed the finish line and won the championship."),
)

SEEDS = (101, 202)
NEUTRAL_INSTRUCTION = "Speak clearly and naturally."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def evaluation_requests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for case_id, text, style_instruction in STYLE_CASES:
            pair_id = f"style_{case_id}_seed-{seed}"
            rows.extend(
                (
                    {
                        "id": f"neutral_{case_id}_seed-{seed}",
                        "group": "neutral",
                        "pair_id": pair_id,
                        "seed": seed,
                        "text": text,
                        "instruction": NEUTRAL_INSTRUCTION,
                    },
                    {
                        "id": f"styled_{case_id}_seed-{seed}",
                        "group": "styled",
                        "pair_id": pair_id,
                        "seed": seed,
                        "text": text,
                        "instruction": style_instruction,
                    },
                )
            )
        for case_id, tag, text in EVENT_CASES:
            pair_id = f"event_{case_id}_seed-{seed}"
            rows.extend(
                (
                    {
                        "id": f"sound_control_{case_id}_seed-{seed}",
                        "group": "sound_control",
                        "pair_id": pair_id,
                        "event": case_id,
                        "tag": tag,
                        "seed": seed,
                        "text": text,
                        "instruction": NEUTRAL_INSTRUCTION,
                    },
                    {
                        "id": f"sound_tagged_{case_id}_seed-{seed}",
                        "group": "sound_tagged",
                        "pair_id": pair_id,
                        "event": case_id,
                        "tag": tag,
                        "seed": seed,
                        "text": f"{tag} {text}",
                        "instruction": NEUTRAL_INSTRUCTION,
                    },
                )
            )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "invalid": bool(
            audio.size == 0
            or sample_rate <= 0
            or not np.isfinite(audio).all()
            or duration < 0.25
        ),
        "silent": bool(peak < 1e-3 or rms < 1e-4),
        "truncated": bool(codec_frames >= max_new_tokens),
    }


def main() -> None:
    args = parse_args()
    requests = evaluation_requests()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        requests = requests[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "generation-manifest.json"
    completed: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        completed = {str(row["id"]): row for row in manifest.get("items", [])}

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

    for index, row in enumerate(requests, start=1):
        output_path = args.output_root / row["group"] / f"{row['id']}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        old = completed.get(row["id"])
        if (
            old
            and output_path.is_file()
            and old.get("audio", {}).get("sha256") == sha256_file(output_path)
        ):
            print(f"[{index}/{len(requests)}] retained {row['id']}", flush=True)
            continue
        if output_path.exists():
            raise FileExistsError(f"unreceipted output requires review: {output_path}")

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
            mode="x",
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
        ordered = [completed[item["id"]] for item in requests if item["id"] in completed]
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
                "matrix": {"total": 80, "per_group": 20, "seeds": list(SEEDS)},
                "items": ordered,
            },
        )
        print(
            f"[{index}/{len(requests)}] generated {row['id']} "
            f"({completed[row['id']]['audio']['duration_seconds']:.2f}s audio, "
            f"{elapsed:.2f}s wall)",
            flush=True,
        )

    ordered = [completed[item["id"]] for item in requests if item["id"] in completed]
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete" if len(ordered) == len(requests) else "partial",
            "base_model": str(args.model_root.resolve()),
            "base_revision": args.base_revision,
            "adapter": str(args.adapter_root.resolve()),
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "max_new_tokens": args.max_new_tokens,
            "fast_all": args.fast_all,
            "matrix": {"total": 80, "per_group": 20, "seeds": list(SEEDS)},
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "items": ordered,
        },
    )
    print(f"wrote {manifest_path} ({len(ordered)}/{len(requests)})")


if __name__ == "__main__":
    main()
