from __future__ import annotations

import pytest

from scripts.prepare_p003_breeze import (
    convert_text_and_instruction,
    slice_boundaries,
)


def test_inline_events_and_controls_convert_in_place() -> None:
    text, instruction, tags, inferred = convert_text_and_instruction(
        "<|emotion:amusement|>Well <|sfx:laughter|>ha! "
        "<|style:whispering|><|prosody:speed_slow|>Done.",
        recording_stem="example",
    )
    assert text == "Well (laugh) ha! Done."
    assert instruction == "Speak with amusement. Whisper the text. Speak slowly."
    assert [f"{tag['kind']}:{tag['value']}" for tag in tags] == [
        "emotion:amusement",
        "sfx:laughter",
        "style:whispering",
        "prosody:speed_slow",
    ]
    assert inferred is None


@pytest.mark.parametrize(
    ("stem", "source_text", "expected"),
    [
        ("vegetative_eating", "", "(eating)"),
        ("vegetative_throat", "", "(clears throat)"),
        ("vegetative_yawning", "oh, oh.", "(yawn) oh, oh."),
        ("nonverbal_cheering", "woo!", "(cheering) woo!"),
    ],
)
def test_filename_inferred_events(stem: str, source_text: str, expected: str) -> None:
    text, instruction, _tags, inferred = convert_text_and_instruction(
        source_text, recording_stem=stem
    )
    assert text == expected
    assert instruction == "Speak clearly and naturally."
    assert inferred is not None


def test_timestamp_slice_alignment_uses_midpoint() -> None:
    words = [
        {"word": " alpha", "start": 0.1, "end": 0.4},
        {"word": " beta", "start": 0.5, "end": 0.8},
        {"word": " gamma", "start": 1.0, "end": 1.3},
        {"word": " delta", "start": 1.4, "end": 1.7},
        {"word": " epsilon", "start": 1.8, "end": 2.1},
        {"word": " zeta", "start": 2.2, "end": 2.5},
        {"word": " eta", "start": 2.6, "end": 2.9},
        {"word": " theta", "start": 3.0, "end": 3.3},
        {"word": " iota", "start": 3.4, "end": 3.7},
        {"word": " kappa", "start": 3.8, "end": 4.1},
        {"word": " lambda", "start": 4.2, "end": 4.5},
        {"word": " mu", "start": 4.6, "end": 4.9},
        {"word": " nu", "start": 5.0, "end": 5.3},
        {"word": " xi", "start": 5.4, "end": 5.7},
        {"word": " omicron", "start": 5.8, "end": 6.1},
        {"word": " pi", "start": 6.2, "end": 6.5},
    ]
    edits = [
        {"text": "alpha beta"},
        {"text": "gamma delta epsilon zeta eta"},
        {"text": "lambda mu nu xi omicron pi"},
    ]
    boundaries, matches = slice_boundaries(
        edits, {"duration": 7.0, "segments": [{"words": words}]}
    )
    assert boundaries == pytest.approx([0.0, 0.9, 4.15, 7.0])
    assert [match["raw_word_index"] for match in matches] == [0, 2, 10]


def test_timestamp_slice_alignment_fails_when_not_unique() -> None:
    phrase = "one two three four five"
    words = [
        {"word": token, "start": index, "end": index + 0.5}
        for index, token in enumerate((phrase + " " + phrase).split())
    ]
    edits = [{"text": "zero"}, {"text": phrase}, {"text": "missing words here"}]
    with pytest.raises(ValueError, match="not uniquely aligned"):
        slice_boundaries(
            edits, {"duration": 12.0, "segments": [{"words": words}]}
        )
