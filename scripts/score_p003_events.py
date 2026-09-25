#!/usr/bin/env python3
"""Score paired p003 event/control WAVs with a local CLAP checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EVENT_LABELS = {
    "laugh": "laughter",
    "sigh": "sighing",
    "cough": "coughing",
    "sniff": "sniffing",
    "sneeze": "sneezing",
    "clears_throat": "throat clearing",
    "yawn": "yawning",
    "crying": "crying",
    "scream": "screaming",
}
CLAP_CANDIDATES = (*EVENT_LABELS.values(), "normal speech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--clap-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_clap_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 48_000:
        audio = resample_poly(audio, 48_000, sample_rate).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def clap_scores(classifier: Any, wav_path: Path) -> dict[str, float]:
    output = classifier(
        load_clap_audio(wav_path),
        candidate_labels=list(CLAP_CANDIDATES),
        hypothesis_template="This audio contains {}.",
    )
    return {str(item["label"]): float(item["score"]) for item in output}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.generation_manifest.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("generation manifest must be complete before scoring")
    if not args.clap_model.is_dir():
        raise FileNotFoundError(args.clap_model)

    from transformers import pipeline

    device_index = int(args.device.split(":", 1)[1]) if args.device.startswith("cuda:") else -1
    classifier = pipeline(
        task="zero-shot-audio-classification",
        model=str(args.clap_model),
        device=device_index,
        local_files_only=True,
    )
    items = {str(row["id"]): row for row in manifest["items"]}
    score_cache: dict[str, dict[str, float]] = {}
    details: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for event, target_label in EVENT_LABELS.items():
        event_rows: list[dict[str, Any]] = []
        for seed in (42, 43, 44):
            tagged_id = f"{event}_tagged_seed-{seed}"
            control_id = f"{event}_control_seed-{seed}"
            tagged = items[tagged_id]
            control = items[control_id]
            for row_id, row in ((tagged_id, tagged), (control_id, control)):
                if row_id not in score_cache:
                    score_cache[row_id] = clap_scores(
                        classifier, Path(row["audio"]["path"])
                    )
            tagged_scores = score_cache[tagged_id]
            control_scores = score_cache[control_id]
            uplift = tagged_scores[target_label] - control_scores[target_label]
            invalid = any(
                bool(item["audio"].get(flag))
                for item in (tagged, control)
                for flag in ("invalid", "silent", "truncated")
            )
            detail = {
                "event": event,
                "target_label": target_label,
                "seed": seed,
                "tagged_id": tagged_id,
                "control_id": control_id,
                "tagged_target_score": tagged_scores[target_label],
                "control_target_score": control_scores[target_label],
                "uplift": uplift,
                "tagged_top_label": max(tagged_scores, key=tagged_scores.get),
                "control_top_label": max(control_scores, key=control_scores.get),
                "invalid_silent_or_truncated": invalid,
                "tagged_scores": tagged_scores,
                "control_scores": control_scores,
            }
            details.append(detail)
            event_rows.append(detail)
        uplifts = [float(row["uplift"]) for row in event_rows]
        signs = {0 if value == 0 else (1 if value > 0 else -1) for value in uplifts}
        median_uplift = float(median(uplifts))
        summaries[event] = {
            "target_label": target_label,
            "uplifts": uplifts,
            "median_paired_uplift": median_uplift,
            "not_demonstrated": median_uplift <= 0,
            "unstable": bool(
                (-1 in signs and 1 in signs)
                or any(row["invalid_silent_or_truncated"] for row in event_rows)
            ),
        }
        print(
            f"{event}: median uplift={median_uplift:+.6f}, "
            f"not_demonstrated={summaries[event]['not_demonstrated']}, "
            f"unstable={summaries[event]['unstable']}",
            flush=True,
        )

    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "complete",
            "method": {
                "classifier": "laion/clap-htsat-unfused",
                "model_path": str(args.clap_model.resolve()),
                "candidate_labels": list(CLAP_CANDIDATES),
                "hypothesis_template": "This audio contains {}.",
                "decision": {
                    "not_demonstrated": "median paired uplift <= 0",
                    "unstable": "uplift changes sign across seeds or a pair is invalid, silent, or truncated",
                },
            },
            "generation_manifest": str(args.generation_manifest.resolve()),
            "details": details,
            "events": summaries,
        },
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
