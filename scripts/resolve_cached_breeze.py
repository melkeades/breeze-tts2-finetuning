#!/usr/bin/env python3
"""Resolve and validate a complete offline Breeze TTS 2 cache snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import scan_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--repo-id", default="BreezeBlue/Breeze-TTS-2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = scan_cache_dir(args.hf_home / "hub")
    matching = [
        repo
        for repo in cache.repos
        if repo.repo_id == args.repo_id and repo.repo_type == "model"
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one cached model repository for {args.repo_id}, got {len(matching)}"
        )
    main_revisions = [
        revision
        for revision in matching[0].revisions
        if "main" in revision.refs
    ]
    if len(main_revisions) != 1:
        raise ValueError(
            f"expected one cached main revision for {args.repo_id}, got {len(main_revisions)}"
        )
    revision = main_revisions[0]
    snapshot = revision.snapshot_path.resolve()
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        *shard_names,
        "audio_tokenizer/config.json",
        "audio_tokenizer/configuration.json",
        "audio_tokenizer/preprocessor_config.json",
        "audio_tokenizer/model.safetensors",
    ]
    missing = [name for name in required if not (snapshot / name).is_file()]
    empty = [
        name
        for name in required
        if (snapshot / name).is_file() and (snapshot / name).stat().st_size == 0
    ]
    if missing or empty:
        raise FileNotFoundError(
            f"cached snapshot is incomplete: missing={missing} empty={empty}"
        )
    receipt = {
        "schema_version": 1,
        "status": "offline_breeze_snapshot_resolved",
        "repo_id": args.repo_id,
        "revision": revision.commit_hash,
        "refs": sorted(revision.refs),
        "snapshot_path": str(snapshot),
        "cache_size_bytes": revision.size_on_disk,
        "required_files": {
            name: (snapshot / name).stat().st_size for name in required
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
