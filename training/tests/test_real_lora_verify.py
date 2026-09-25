from argparse import Namespace
from pathlib import Path

import pytest

from training.real_lora_verify import validate_output_mode


def test_adapter_only_verification_requires_no_merged_output() -> None:
    validate_output_mode(Namespace(adapter_only=True, merged_output=None))


def test_merge_verification_requires_output() -> None:
    with pytest.raises(ValueError, match="required unless"):
        validate_output_mode(Namespace(adapter_only=False, merged_output=None))


def test_adapter_only_rejects_merge_output() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_output_mode(
            Namespace(adapter_only=True, merged_output=Path("merged"))
        )
