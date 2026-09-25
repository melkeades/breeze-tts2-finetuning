#!/usr/bin/env python3
"""Convert the p003 Higgs corpus into deterministic Breeze JSONL manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

TAG_RE = re.compile(r"<\|(?P<kind>[^:|]+):(?P<value>[^|]+)\|>")
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
LONG_RECORDINGS = {f"freeform_speech_{index:02d}.wav" for index in range(1, 7)}

EVENTS = {
    "cough": "(cough)",
    "crying": "(crying)",
    "laughter": "(laugh)",
    "screaming": "(scream)",
    "sigh": "(sigh)",
    "sneeze": "(sneeze)",
    "sniff": "(sniff)",
}
INFERRED_EVENTS = {
    "vegetative_eating": "(eating)",
    "vegetative_throat": "(clears throat)",
    "vegetative_yawning": "(yawn)",
    "nonverbal_cheering": "(cheering)",
}
EMOTIONS = {
    "affection": "Speak affectionately.",
    "amusement": "Speak with amusement.",
    "anger": "Speak angrily.",
    "arousal": "Speak with an eager, desirous tone.",
    "bitterness": "Speak bitterly.",
    "confusion": "Speak with confusion.",
    "contemplation": "Speak thoughtfully.",
    "contentment": "Speak contentedly.",
    "disgust": "Speak with disgust.",
    "elation": "Speak with elation.",
    "fear": "Speak fearfully.",
    "helplessness": "Speak with a helpless tone.",
    "pride": "Speak proudly.",
    "relief": "Speak with relief.",
    "sadness": "Speak sadly.",
    "shame": "Speak with embarrassment and shame.",
    "surprise": "Speak with surprise.",
}
STYLES = {
    "shouting": "Shout the text.",
    "singing": "Sing the text.",
    "whispering": "Whisper the text.",
}
PROSODY = {
    "expressive_low": "Use a restrained, minimally expressive delivery.",
    "pitch_high": "Use a high speaking pitch.",
    "pitch_low": "Use a low speaking pitch.",
    "speed_fast": "Speak quickly.",
    "speed_slow": "Speak slowly.",
}
CONTROL_MAPS = {"emotion": EMOTIONS, "style": STYLES, "prosody": PROSODY}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_words(value: str) -> list[str]:
    value = TAG_RE.sub(" ", value.lower().replace("’", "'"))
    return WORD_RE.findall(value)


def basename_any_platform(value: str) -> str:
    return re.split(r"[\\/]", value)[-1]


def extract_tags(text: str) -> list[dict[str, str]]:
    return [match.groupdict() for match in TAG_RE.finditer(text)]


def convert_text_and_instruction(
    text: str, *, recording_stem: str
) -> tuple[str, str, list[dict[str, str]], str | None]:
    tags = extract_tags(text)
    instructions: dict[str, list[str]] = {
        "emotion": [],
        "style": [],
        "prosody": [],
    }

    def replace_tag(match: re.Match[str]) -> str:
        kind = match.group("kind")
        value = match.group("value")
        if kind == "sfx":
            try:
                return f" {EVENTS[value]} "
            except KeyError as exc:
                raise ValueError(f"unmapped SFX tag: {value}") from exc
        if kind not in CONTROL_MAPS:
            raise ValueError(f"unmapped control kind: {kind}")
        try:
            direction = CONTROL_MAPS[kind][value]
        except KeyError as exc:
            raise ValueError(f"unmapped {kind} tag: {value}") from exc
        if direction not in instructions[kind]:
            instructions[kind].append(direction)
        return ""

    converted = TAG_RE.sub(replace_tag, text)
    converted = " ".join(converted.split())
    inferred = INFERRED_EVENTS.get(recording_stem)
    if inferred is not None:
        if recording_stem in {"vegetative_eating", "vegetative_throat"}:
            if converted:
                raise ValueError(
                    f"expected empty edit text for inferred-only event {recording_stem}"
                )
            converted = inferred
        else:
            converted = f"{inferred} {converted}".strip()
    if not converted:
        raise ValueError(f"empty converted text for {recording_stem}")
    if "<|" in converted or "|>" in converted:
        raise ValueError(f"Higgs control token remains in {recording_stem}")

    ordered = [
        *instructions["emotion"],
        *instructions["style"],
        *instructions["prosody"],
    ]
    instruction = " ".join(ordered) if ordered else "Speak clearly and naturally."
    return converted, instruction, tags, inferred


def whisper_words(timestamp_entry: dict[str, Any]) -> list[dict[str, Any]]:
    words = [
        word
        for segment in timestamp_entry["segments"]
        for word in segment.get("words", [])
    ]
    if not words:
        raise ValueError("timestamp entry contains no word timestamps")
    return words


def unique_slice_start(
    edit_text: str,
    timestamp_words: list[dict[str, Any]],
    *,
    minimum_index: int,
) -> tuple[int, dict[str, Any]]:
    """Find a unique timestamp-word start using the edit's leading words.

    A small leading offset is allowed only for transcription fillers or corrections.
    The returned timestamp is always rewound to the first normalized edit word when
    that word occurs directly before the unique anchor; otherwise conversion fails.
    """

    edit_words = normalized_words(edit_text)
    raw_words = [normalized_words(str(word["word"])) for word in timestamp_words]
    raw_tokens = [tokens[0] if len(tokens) == 1 else "" for tokens in raw_words]
    max_offset = min(8, max(0, len(edit_words) - 5))
    for offset in range(max_offset + 1):
        max_width = min(14, len(edit_words) - offset)
        for width in range(max_width, 4, -1):
            needle = edit_words[offset : offset + width]
            matches = [
                index
                for index in range(minimum_index, len(raw_tokens) - width + 1)
                if raw_tokens[index : index + width] == needle
            ]
            if len(matches) != 1:
                continue
            anchor = matches[0]
            start_index = anchor - offset
            if start_index < minimum_index:
                continue
            if raw_tokens[start_index:anchor] != edit_words[:offset]:
                continue
            return start_index, {
                "edit_word_offset": offset,
                "anchor_width": width,
                "anchor_words": needle,
                "raw_word_index": start_index,
            }
    raise ValueError(
        "slice boundary is not uniquely aligned to Whisper words: "
        f"minimum_index={minimum_index} leading_words={edit_words[:14]}"
    )


def slice_boundaries(
    edits: list[dict[str, Any]], timestamp_entry: dict[str, Any]
) -> tuple[list[float], list[dict[str, Any]]]:
    if len(edits) != 3:
        raise ValueError(f"expected three edit slices, got {len(edits)}")
    words = whisper_words(timestamp_entry)
    starts = [0]
    matches: list[dict[str, Any]] = [
        {
            "edit_word_offset": 0,
            "anchor_width": None,
            "anchor_words": [],
            "raw_word_index": 0,
        }
    ]
    minimum = 1
    for edit in edits[1:]:
        start_index, match = unique_slice_start(
            edit["text"], words, minimum_index=minimum
        )
        if start_index <= starts[-1]:
            raise ValueError("slice starts are not strictly increasing")
        starts.append(start_index)
        matches.append(match)
        minimum = start_index + 1

    boundaries = [0.0]
    for start_index in starts[1:]:
        previous_end = float(words[start_index - 1]["end"])
        next_start = float(words[start_index]["start"])
        boundary = (previous_end + next_start) / 2.0
        if boundary <= boundaries[-1]:
            raise ValueError("derived slice boundaries are not strictly increasing")
        boundaries.append(boundary)
    boundaries.append(float(timestamp_entry["duration"]))
    return boundaries, matches


def load_validation_membership(validation_manifest: Path) -> set[str]:
    names: set[str] = set()
    with validation_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            name = basename_any_platform(str(value["audio"]))
            if not name:
                raise ValueError(f"missing validation basename at line {line_number}")
            names.add(name)
    if len(names) != 14:
        raise ValueError(f"expected 14 validation recordings, got {len(names)}")
    return names


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            value = {
                "audio": record["audio"],
                "text": record["text"],
                "instruction": record["instruction"],
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--higgs-train-manifest", type=Path, required=True)
    parser.add_argument("--higgs-validation-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    artifact_root = args.artifact_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    for output in (
        artifact_root / "dataset",
        artifact_root / "train.jsonl",
        artifact_root / "validation.jsonl",
        artifact_root / "dataset-conversion-report.json",
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite converter output: {output}")
    slice_audio_root = artifact_root / "dataset" / "audio"
    slice_audio_root.mkdir(parents=True, exist_ok=False)

    edit_path = dataset_root / "p003-emotion-style-sfx-dataset-trial1-edits.json"
    timestamp_path = (
        dataset_root / "transcripts_freeform_whisper_fastapi_raw-20260501-175154.json"
    )
    edit_document = json.loads(edit_path.read_text(encoding="utf-8"))
    timestamps = json.loads(timestamp_path.read_text(encoding="utf-8"))
    edits: list[dict[str, Any]] = edit_document["edits"]
    if len(edits) != 173:
        raise ValueError(f"expected 173 edit rows, got {len(edits)}")
    validation_names = load_validation_membership(args.higgs_validation_manifest)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edit in edits:
        grouped[edit["relative_path"]].append(edit)
    if len(grouped) != 161:
        raise ValueError(f"expected 161 source recordings, got {len(grouped)}")

    slice_metadata: dict[str, dict[str, Any]] = {}
    for relative_path in sorted(LONG_RECORDINGS):
        source_path = dataset_root / relative_path
        source_stem = Path(relative_path).stem
        source_edits = grouped[relative_path]
        boundaries, matches = slice_boundaries(source_edits, timestamps[source_stem])
        audio, sample_rate = sf.read(source_path, dtype="int16", always_2d=True)
        source_frames = len(audio)
        frame_boundaries = [0]
        frame_boundaries.extend(
            round(boundary * sample_rate) for boundary in boundaries[1:-1]
        )
        frame_boundaries.append(source_frames)
        if not all(
            left < right
            for left, right in zip(frame_boundaries, frame_boundaries[1:])
        ):
            raise ValueError(f"invalid frame boundaries for {relative_path}")
        source_info = sf.info(source_path)
        for index, edit in enumerate(source_edits):
            output_path = slice_audio_root / f"{edit['slice_id']}.wav"
            sf.write(
                output_path,
                audio[frame_boundaries[index] : frame_boundaries[index + 1]],
                sample_rate,
                subtype=source_info.subtype,
            )
            slice_metadata[edit["slice_id"]] = {
                "audio": output_path,
                "start_seconds": frame_boundaries[index] / sample_rate,
                "end_seconds": frame_boundaries[index + 1] / sample_rate,
                "start_frame": frame_boundaries[index],
                "end_frame": frame_boundaries[index + 1],
                "alignment": matches[index],
            }

    records: list[dict[str, Any]] = []
    source_duration_by_name: dict[str, float] = {}
    observed_tags: Counter[str] = Counter()
    inferred_counts: Counter[str] = Counter()
    for edit in edits:
        relative_path = edit["relative_path"]
        source_path = (dataset_root / relative_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if relative_path not in source_duration_by_name:
            source_duration_by_name[relative_path] = float(sf.info(source_path).duration)
        stem = Path(relative_path).stem
        text, instruction, tags, inferred = convert_text_and_instruction(
            edit["text"], recording_stem=stem
        )
        for tag in tags:
            observed_tags[f"{tag['kind']}:{tag['value']}"] += 1
        if inferred is not None:
            inferred_counts[stem] += 1

        if relative_path in LONG_RECORDINGS:
            sliced = slice_metadata[edit["slice_id"]]
            audio_path = Path(sliced["audio"])
            start_seconds = sliced["start_seconds"]
            end_seconds = sliced["end_seconds"]
            alignment = sliced["alignment"]
        else:
            audio_path = source_path
            start_seconds = 0.0
            end_seconds = source_duration_by_name[relative_path]
            alignment = None
        split = "validation" if relative_path in validation_names else "train"
        records.append(
            {
                "slice_id": edit["slice_id"],
                "source_relative_path": relative_path,
                "source_audio": str(source_path),
                "audio": str(audio_path.resolve()),
                "audio_sha256": sha256_file(audio_path),
                "split": split,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "text": text,
                "instruction": instruction,
                "original_text": edit["text"],
                "original_tags": tags,
                "inferred_event": inferred,
                "alignment": alignment,
            }
        )

    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    if (len(train), len(validation)) != (159, 14):
        raise ValueError(
            f"unexpected split counts: train={len(train)} validation={len(validation)}"
        )
    if {record["source_relative_path"] for record in train} & {
        record["source_relative_path"] for record in validation
    }:
        raise ValueError("source-audio leakage between train and validation")
    if {record["source_relative_path"] for record in validation} != validation_names:
        raise ValueError("validation membership differs from the Higgs manifest")
    if any(not Path(record["audio"]).is_file() for record in records):
        raise FileNotFoundError("one or more converted audio paths do not exist")

    train_manifest = artifact_root / "train.jsonl"
    validation_manifest = artifact_root / "validation.jsonl"
    write_jsonl(train_manifest, train)
    write_jsonl(validation_manifest, validation)
    transcript_document = {
        "schema_version": 1,
        "dataset": "p003",
        "provenance": {
            "edit_manifest": str(edit_path.resolve()),
            "edit_manifest_sha256": sha256_file(edit_path),
            "timestamp_manifest": str(timestamp_path.resolve()),
            "timestamp_manifest_sha256": sha256_file(timestamp_path),
            "higgs_train_manifest": str(args.higgs_train_manifest.resolve()),
            "higgs_train_manifest_sha256": sha256_file(args.higgs_train_manifest),
            "higgs_validation_manifest": str(
                args.higgs_validation_manifest.resolve()
            ),
            "higgs_validation_manifest_sha256": sha256_file(
                args.higgs_validation_manifest
            ),
        },
        "mapping": {
            "inline_sfx": EVENTS,
            "filename_inferred_events": INFERRED_EVENTS,
            "emotion": EMOTIONS,
            "style": STYLES,
            "prosody": PROSODY,
            "untagged_instruction": "Speak clearly and naturally.",
        },
        "records": records,
    }
    transcript_path = artifact_root / "transcripts.breeze.json"
    transcript_path.write_text(
        json.dumps(transcript_document, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    source_total = sum(source_duration_by_name.values())
    train_total = sum(record["duration_seconds"] for record in train)
    validation_total = sum(record["duration_seconds"] for record in validation)
    if abs((train_total + validation_total) - source_total) > 1e-3:
        raise ValueError("slice durations do not preserve total source duration")
    report = {
        "schema_version": 1,
        "status": "p003_breeze_conversion_complete",
        "counts": {
            "source_recordings": len(grouped),
            "items": len(records),
            "train": len(train),
            "validation": len(validation),
            "whole_clips": len(records) - len(slice_metadata),
            "derived_slices": len(slice_metadata),
        },
        "duration_seconds": {
            "source": source_total,
            "train": train_total,
            "validation": validation_total,
        },
        "duration_minutes": {
            "source": source_total / 60.0,
            "train": train_total / 60.0,
            "validation": validation_total / 60.0,
        },
        "validation_source_files": sorted(validation_names),
        "observed_source_tags": dict(sorted(observed_tags.items())),
        "inferred_event_counts": dict(sorted(inferred_counts.items())),
        "mapping": transcript_document["mapping"],
        "outputs": {
            "transcripts": str(transcript_path.resolve()),
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest": str(validation_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
        },
    }
    report_path = artifact_root / "dataset-conversion-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
