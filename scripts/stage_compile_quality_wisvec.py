#!/usr/bin/env python3
"""Stage nested quality-matrix WAVs for Wisvec's one-directory label format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CHECKPOINTS = ("windows-baseline", "compile-only", "compile-text-fa2")
GROUPS = ("neutral", "styled", "sound_control", "sound_tagged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    return parser.parse_args()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def main() -> None:
    args = parse_args()
    source_root = args.artifact_root / "audio"
    target_root = args.artifact_root / "wisvec-input"
    target_root.mkdir(parents=True, exist_ok=True)
    reference = args.reference.resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)

    pairs: list[str] = []
    labels: dict[str, int] = {}
    for checkpoint in CHECKPOINTS:
        for group in GROUPS:
            label = f"{checkpoint}--{group}"
            target_dir = target_root / label
            target_dir.mkdir(parents=True, exist_ok=True)
            sources = sorted((source_root / checkpoint / group).glob("*.wav"))
            if len(sources) != 20:
                raise RuntimeError(f"expected 20 WAVs for {label}, found {len(sources)}")
            for source in sources:
                target = target_dir / source.name
                if target.exists():
                    if not os.path.samefile(source, target):
                        raise RuntimeError(f"staged file differs from source: {target}")
                else:
                    os.link(source, target)
                pairs.append(f"{wsl_path(target)}|{wsl_path(reference)}")
            labels[label] = len(sources)

    pair_manifest = args.artifact_root / "wisvec-identity-pairs.lst"
    pair_manifest.write_text("\n".join(pairs) + "\n", encoding="utf-8")
    receipt = {
        "status": "complete",
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "reference": str(reference),
        "labels": labels,
        "total": sum(labels.values()),
        "identity_manifest": str(pair_manifest.resolve()),
    }
    (args.artifact_root / "wisvec-staging.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
