from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig


def configure_attention(config, implementation: str):
    """Propagate one supported attention backend through every sub-config."""

    if implementation not in {"eager", "sdpa"}:
        raise ValueError(f"unsupported training attention backend: {implementation}")

    config._attn_implementation = implementation
    config.use_cache = False
    for nested_name in (
        "backbone_config",
        "depth_decoder_config",
        "text_encoder_config",
    ):
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        if isinstance(nested, dict):
            nested["_attn_implementation"] = implementation
            nested["preferred_attn_implementation"] = implementation
            nested["use_cache"] = False
            continue
        nested._attn_implementation = implementation
        if hasattr(nested, "preferred_attn_implementation"):
            nested.preferred_attn_implementation = implementation
        if hasattr(nested, "use_cache"):
            nested.use_cache = False
    return config


def configure_eager_attention(config):
    """Backward-compatible eager-attention configuration helper."""

    return configure_attention(config, "eager")


def load_eager_config(model_root: str | Path) -> BreezeConfig:
    """Load Breeze while propagating eager attention through every nested model."""

    config = BreezeConfig.from_pretrained(model_root)
    configure_eager_attention(config)
    return config


def load_training_model(
    model_root: str | Path,
    *,
    device: str,
    attention_implementation: str = "eager",
) -> BreezeForConditionalGeneration:
    config = BreezeConfig.from_pretrained(model_root)
    configure_attention(config, attention_implementation)
    model = BreezeForConditionalGeneration.from_pretrained(
        model_root,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    return model


def load_tokenizer(model_root: str | Path):
    # Transformers 4.57.3 treats any local 4.57.3 model config as a possible
    # Mistral tokenizer. Breeze uses GemmaTokenizerFast with one whitespace
    # Split pre-tokenizer, so the Mistral sequence-index patch is inapplicable.
    return AutoTokenizer.from_pretrained(model_root, fix_mistral_regex=False)
