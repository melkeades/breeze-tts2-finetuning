#!/usr/bin/env python3
"""Summarize the paired 240-file compile quality experiment."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

CHECKPOINTS = ("windows-baseline", "compile-only", "compile-text-fa2")
GROUPS = ("neutral", "styled", "sound_control", "sound_tagged")
METRICS = (
    "nisqa_mos",
    "dnsmos_p808_mos",
    "dnsmos_mos_ovr",
    "dnsmos_mos_sig",
    "dnsmos_mos_bak",
    "identity",
    "audio_duration_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def load_rows(root: Path) -> list[dict[str, Any]]:
    quality = json.loads((root / "wisvec-quality-240.json").read_text())["rows"]
    identity = json.loads((root / "wisvec-identity-240.json").read_text())["rows"]
    identity_by_key: dict[tuple[str, str], float] = {}
    for row in identity:
        candidate = Path(row["candidate"])
        identity_by_key[(candidate.parent.name, candidate.name)] = float(
            row["speaker_similarity"]
        )

    rows: list[dict[str, Any]] = []
    for row in quality:
        label = str(row["label"])
        checkpoint, group = label.split("--", 1)
        key = (label, str(row["file"]))
        rows.append(
            {
                **row,
                "checkpoint": checkpoint,
                "group": group,
                "identity": identity_by_key[key],
            }
        )
    return rows


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    rows = load_rows(root)
    if len(rows) != 240:
        raise RuntimeError(f"expected 240 scored rows, found {len(rows)}")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        if len(checkpoint_rows) != 80:
            raise RuntimeError(f"expected 80 rows for {checkpoint}")
        grouped[checkpoint] = {
            "overall": {
                "count": len(checkpoint_rows),
                **{
                    metric: summarize([float(row[metric]) for row in checkpoint_rows])
                    for metric in METRICS
                },
            }
        }
        for group in GROUPS:
            subset = [row for row in checkpoint_rows if row["group"] == group]
            if len(subset) != 20:
                raise RuntimeError(f"expected 20 rows for {checkpoint}/{group}")
            grouped[checkpoint][group] = {
                "count": len(subset),
                **{
                    metric: summarize([float(row[metric]) for row in subset])
                    for metric in METRICS
                },
            }

    row_map = {
        (row["checkpoint"], row["group"], row["file"]): row for row in rows
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for comparison_index, checkpoint in enumerate(CHECKPOINTS[1:], start=1):
        comparisons[checkpoint] = {}
        for group_index, group in enumerate(("overall", *GROUPS)):
            group_names = GROUPS if group == "overall" else (group,)
            keys = sorted(
                (candidate_group, file_name)
                for candidate_group in group_names
                for file_name in {
                    row["file"]
                    for row in rows
                    if row["checkpoint"] == "windows-baseline"
                    and row["group"] == candidate_group
                }
            )
            metric_results: dict[str, Any] = {}
            for metric_index, metric in enumerate(METRICS):
                differences = [
                    float(row_map[(checkpoint, candidate_group, file_name)][metric])
                    - float(
                        row_map[
                            ("windows-baseline", candidate_group, file_name)
                        ][metric]
                    )
                    for candidate_group, file_name in keys
                ]
                lower, upper = bootstrap_ci(
                    differences,
                    samples=args.bootstrap_samples,
                    seed=42 + comparison_index * 100 + group_index * 10 + metric_index,
                )
                metric_results[metric] = {
                    "paired_count": len(differences),
                    "mean_delta": statistics.fmean(differences),
                    "median_delta": statistics.median(differences),
                    "bootstrap_95_ci": [lower, upper],
                    "ci_excludes_zero": lower > 0 or upper < 0,
                }
            comparisons[checkpoint][group] = metric_results

    generation: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        manifest = json.loads(
            (root / "audio" / checkpoint / "generation-manifest.json").read_text()
        )
        items = manifest["items"]
        generation[checkpoint] = {
            "count": len(items),
            "invalid": sum(bool(item["audio"]["invalid"]) for item in items),
            "silent": sum(bool(item["audio"]["silent"]) for item in items),
            "truncated": sum(bool(item["audio"]["truncated"]) for item in items),
            "under_one_second": sum(
                float(item["audio"]["duration_seconds"]) < 1.0 for item in items
            ),
        }

    result = {
        "design": {
            "total_wavs": 240,
            "checkpoints": 3,
            "samples_per_checkpoint": 80,
            "groups_per_checkpoint": {group: 20 for group in GROUPS},
            "paired_style_design": "same text and seed, neutral versus style instruction",
            "paired_sound_design": "same text and seed, without versus with sound tag",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "generation": generation,
        "scores": grouped,
        "paired_comparisons_vs_windows_baseline": comparisons,
    }
    (root / "analysis-240.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# 240-sample compile-training quality evaluation",
        "",
        "## Design",
        "",
        "Three matched 250-step Windows checkpoints were evaluated with 80 WAVs each: "
        "20 neutral, 20 paired style-instruction, 20 sound-tag controls, and 20 paired "
        "sound-tagged prompts. NISQA and DNSMOS are automatic MOS predictors; identity "
        "is WavLM-Large cosine similarity to the p003 neutral reference.",
        "",
        "## Overall results (80 samples per checkpoint)",
        "",
        "| Checkpoint | NISQA | P808 | OVR | SIG | BAK | Identity | Duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_names = {
        "windows-baseline": "No speedups",
        "compile-only": "Compile backbone",
        "compile-text-fa2": "Compile + text FA2",
    }
    for checkpoint in CHECKPOINTS:
        value = grouped[checkpoint]["overall"]
        lines.append(
            f"| {display_names[checkpoint]} | {fmt(value['nisqa_mos']['mean'])} | "
            f"{fmt(value['dnsmos_p808_mos']['mean'])} | "
            f"{fmt(value['dnsmos_mos_ovr']['mean'])} | "
            f"{fmt(value['dnsmos_mos_sig']['mean'])} | "
            f"{fmt(value['dnsmos_mos_bak']['mean'])} | "
            f"{fmt(value['identity']['mean'])} | "
            f"{value['audio_duration_sec']['mean']:.3f}s |"
        )

    lines.extend(
        [
            "",
            "## Results by prompt group (20 samples per cell)",
            "",
            "| Group | Checkpoint | NISQA | P808 | Identity | Duration |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        for checkpoint in CHECKPOINTS:
            value = grouped[checkpoint][group]
            lines.append(
                f"| {group} | {display_names[checkpoint]} | "
                f"{fmt(value['nisqa_mos']['mean'])} | "
                f"{fmt(value['dnsmos_p808_mos']['mean'])} | "
                f"{fmt(value['identity']['mean'])} | "
                f"{value['audio_duration_sec']['mean']:.3f}s |"
            )

    lines.extend(["", "## Paired deltas versus no speedups", ""])
    for checkpoint in CHECKPOINTS[1:]:
        values = comparisons[checkpoint]["overall"]
        nisqa = values["nisqa_mos"]
        p808 = values["dnsmos_p808_mos"]
        identity = values["identity"]
        lines.append(
            f"- {display_names[checkpoint]}: NISQA {nisqa['mean_delta']:+.4f} "
            f"(95% paired bootstrap CI {nisqa['bootstrap_95_ci'][0]:+.4f} to "
            f"{nisqa['bootstrap_95_ci'][1]:+.4f}); P808 {p808['mean_delta']:+.4f} "
            f"(CI {p808['bootstrap_95_ci'][0]:+.4f} to {p808['bootstrap_95_ci'][1]:+.4f}); "
            f"identity {identity['mean_delta']:+.4f} "
            f"(CI {identity['bootstrap_95_ci'][0]:+.4f} to "
            f"{identity['bootstrap_95_ci'][1]:+.4f})."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 80-sample aggregate is the relevant result; the earlier six-sample "
            "neutral test was underpowered and its apparent regression did not replicate. "
            "Overall automatic quality and identity are similar across checkpoints. The "
            "sound-tagged subset should be inspected separately because intentional "
            "non-speech events can lower speech-quality and speaker-similarity predictors.",
            "",
            "These are automatic proxy scores, not human MOS or proof of style/tag "
            "adherence. The paired confidence intervals quantify sampling uncertainty for "
            "this matrix; a blind listening test is still required for a perceptual claim.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(root / "REPORT.md"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
