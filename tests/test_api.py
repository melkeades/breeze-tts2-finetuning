from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from breeze_infer.api import (
    DEFAULT_CFG_SCALE,
    WEB_ROOT,
    ApiSettings,
    _pcm16,
    _resolve_adapter,
    _resolve_fast_all_default,
    app,
    speech,
)
from breeze_infer.api import (
    MAX_NEW_TOKENS as API_MAX_NEW_TOKENS,
)
from breeze_infer.api import (
    MAX_SEQ_LEN as API_MAX_SEQ_LEN,
)
from infer import MAX_NEW_TOKENS as CLI_MAX_NEW_TOKENS
from infer import MAX_SEQ_LEN as CLI_MAX_SEQ_LEN


def test_api_exposes_streaming_ui_and_adapter_control() -> None:
    paths = {route.path for route in app.routes if route.path.startswith("/")}

    assert "/" in paths
    assert "/health" in paths
    assert "/v1/audio/speech" in paths
    assert "/v1/adapters" in paths
    assert "/v1/adapters/reload" in paths
    assert "/v1/adapters/reload/{job_id}" in paths
    assert "/api/ref-audio-codes" not in paths
    assert (WEB_ROOT / "index.html").is_file()
    assert (WEB_ROOT / "app.css").is_file()
    assert (WEB_ROOT / "app.js").is_file()


def test_speech_request_parameters_are_minimal() -> None:
    assert list(inspect.signature(speech).parameters) == [
        "text",
        "instruction",
        "cfg_scale",
        "ref_audio",
        "ref_text",
        "seed",
    ]


def test_api_cfg_defaults_to_one() -> None:
    cfg_parameter = inspect.signature(speech).parameters["cfg_scale"]

    assert DEFAULT_CFG_SCALE == 1.0
    assert cfg_parameter.default.default == 1.0


def test_cli_and_api_support_1500_generated_tokens() -> None:
    assert CLI_MAX_NEW_TOKENS == API_MAX_NEW_TOKENS == 1500
    assert CLI_MAX_SEQ_LEN == API_MAX_SEQ_LEN == 2048


def test_api_settings_include_adapter_identity() -> None:
    assert "adapter" in ApiSettings.__dataclass_fields__
    assert "adapters_dir" in ApiSettings.__dataclass_fields__
    assert "base_revision" in ApiSettings.__dataclass_fields__


def test_api_defaults_to_fast_all_without_masking_stage_profiling() -> None:
    stages = (False, False, False, False, False)

    assert _resolve_fast_all_default(None, stages) is True
    assert _resolve_fast_all_default(False, stages) is False
    assert _resolve_fast_all_default(None, (True, *stages[1:])) is None


def test_adapter_resolution_stays_inside_configured_root(tmp_path: Path) -> None:
    adapters_dir = tmp_path / "adapters"
    adapter = adapters_dir / "voice-a"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    settings = ApiSettings(
        model=tmp_path / "model",
        adapter=None,
        adapters_dir=adapters_dir,
        base_revision=None,
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
    )

    assert _resolve_adapter(settings, "base") is None
    assert _resolve_adapter(settings, "voice-a") == adapter.resolve()
    with pytest.raises(HTTPException) as raised:
        _resolve_adapter(settings, "../outside")
    assert raised.value.status_code == 400


def test_pcm16_clips_and_encodes_little_endian() -> None:
    encoded = _pcm16(np.array([-2.0, 0.0, 2.0], dtype=np.float32))

    assert np.frombuffer(encoded, dtype="<i2").tolist() == [-32767, 0, 32767]
