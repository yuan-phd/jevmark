"""Shared CPU fixtures: the real Qwen3 tokenizer and a tiny randomly initialised Qwen3.

The tokenizer is fetched once from the HF Hub at a pinned revision and cached by
huggingface_hub. No real weights are ever loaded in tests.
"""

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3Config, Qwen3ForCausalLM

# Qwen3-0.6B-Base and Qwen3-1.7B-Base share one tokenizer.
TOKENIZER_ID = "Qwen/Qwen3-0.6B-Base"
TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)


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
