"""Shared CPU fixtures: the base config, the real Qwen3 tokenizer and a tiny randomly initialised Qwen3.

The tokenizer is fetched once from the HF Hub at the revision pinned in
configs/base.yaml (tokenizer_reference) and cached by huggingface_hub. No real
weights are ever loaded in tests.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3Config, Qwen3ForCausalLM

BASE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


@pytest.fixture(scope="session")
def base_config() -> dict[str, Any]:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text())


@pytest.fixture(scope="session")
def tokenizer(base_config: dict[str, Any]) -> PreTrainedTokenizerBase:
    # Qwen3-0.6B-Base and Qwen3-1.7B-Base share one tokenizer; tests/test_encode.py checks it.
    reference = base_config["tokenizer_reference"]
    return AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])


@pytest.fixture(scope="session")
def tiny_config(tokenizer: PreTrainedTokenizerBase) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: Qwen3Config) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config)
    model.eval()
    return model
