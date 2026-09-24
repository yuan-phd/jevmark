"""Readout on the tiny random Qwen3 model, docs/API_SPEC.md section 5.

Permutation behaviour is measured in evaluation, not unit-tested on a random model.
"""

import copy
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3ForCausalLM

from jevmark.encode import encode
from jevmark.model import JevMark, collate, flat_index
from jevmark.schema import Request

REQUEST_A = {
    "state": "I was charged twice, please refund me.",
    "questions": {
        "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {"billing": "Charges, invoices, refunds", "technical": None, "other": None},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the reported issue?",
            "criteria": ["Cosmetic", "Degraded with a workaround", "Blocking"],
        },
    },
}

REQUEST_B = {
    "state": {"ticket": 42, "text": "App crashes on start"},
    "questions": {
        "component": {
            "type": "choice",
            "instructions": "Which component is affected?",
            "criteria": {"ios": None, "android": None, "web": None, "backend": "APIs and jobs", "other": None},
        },
    },
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


@pytest.fixture(scope="module")
def encoded_pair(tokenizer):
    a = encode(Request.from_dict(REQUEST_A), tokenizer, max_tokens=2048)
    b = encode(Request.from_dict(REQUEST_B), tokenizer, max_tokens=2048)
    assert len(a.input_ids) != len(b.input_ids), "the batch test needs different lengths to exercise padding"
    return a, b


def randomise_lora(model, seed=0):
    """Fresh LoRA has lora_B = 0 and changes nothing; random B makes the adapter visible."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
    return model


def make_lora(tiny_model):
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=LORA_TARGETS)
    return randomise_lora(get_peft_model(copy.deepcopy(tiny_model), config)).eval()


@pytest.fixture(scope="module")
def plain(tiny_model, tokenizer):
    return JevMark(tiny_model, tokenizer, max_tokens=2048)


@pytest.fixture(scope="module")
def lora_model(tiny_model):
    return make_lora(tiny_model)


@pytest.fixture(scope="module")
def lora(lora_model, tokenizer):
    return JevMark(lora_model, tokenizer, max_tokens=2048)


@pytest.fixture(scope="module")
def untied(tiny_config, tokenizer):
    config = copy.deepcopy(tiny_config)
    config.tie_word_embeddings = False
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(config).eval()
    assert model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
    return JevMark(model, tokenizer, max_tokens=2048)


# Shapes, order, sums


def test_flat_output_shapes_and_order(plain, encoded_pair):
    distributions = plain.forward_distributions(list(encoded_pair))
    assert flat_index(list(encoded_pair)) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert [tuple(d.shape) for d in distributions] == [(2,), (3,), (3,), (5,)]


@pytest.mark.parametrize("name", ["plain", "lora"])
def test_distributions_sum_to_one(name, request, encoded_pair):
    jev = request.getfixturevalue(name)
    for distribution in jev.forward_distributions(list(encoded_pair)):
        assert distribution.dtype == torch.float32
        assert torch.all(distribution >= 0)
        assert distribution.sum().item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ["plain", "lora"])
def test_batch_of_two_equals_two_single_calls(name, request, encoded_pair):
    jev = request.getfixturevalue(name)
    a, b = encoded_pair
    batched = jev.forward_distributions([a, b])
    singles = jev.forward_distributions([a]) + jev.forward_distributions([b])
    assert len(batched) == len(singles)
    for x, y in zip(batched, singles):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


# Hidden-state path against the full-logit path


def full_logit_columns(causal_lm, encoded):
    with torch.no_grad():
        logits = causal_lm(input_ids=torch.tensor([encoded.input_ids])).logits[0].float()
    return [logits[slot, list(ids)] for slot, ids in zip(encoded.slot_positions, encoded.letter_ids)]


@pytest.mark.parametrize("name", ["plain", "lora", "untied"])
def test_hidden_state_letter_logits_equal_full_logit_columns(name, request, encoded_pair):
    jev = request.getfixturevalue(name)
    hidden_path = jev.slot_logits(list(encoded_pair))
    full_path = full_logit_columns(jev.model, encoded_pair[0]) + full_logit_columns(jev.model, encoded_pair[1])
    for x, y in zip(hidden_path, full_path):
        torch.testing.assert_close(x.detach(), y, atol=1e-4, rtol=1e-4)


def test_lora_changes_the_letter_logits(plain, lora, encoded_pair):
    # Guards against the hidden-state path bypassing the adapter.
    for x, y in zip(plain.slot_logits(list(encoded_pair)), lora.slot_logits(list(encoded_pair))):
        assert not torch.allclose(x, y, atol=1e-3)


# Gradients and temperature


def test_slot_logits_carry_gradients_to_lora(lora, encoded_pair):
    lora.model.zero_grad()
    logits = lora.slot_logits(list(encoded_pair))
    assert all(x.requires_grad and x.dtype == torch.float32 for x in logits)
    torch.stack([x.logsumexp(0) - x[0] for x in logits]).sum().backward()
    grads = [p.grad for n, p in lora.model.named_parameters() if "lora_" in n]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    lora.model.zero_grad()


def test_forward_distributions_has_no_grad(lora, encoded_pair):
    assert not any(d.requires_grad for d in lora.forward_distributions(list(encoded_pair)))


def test_temperature_divides_letter_logits(tiny_model, tokenizer, encoded_pair):
    warm = JevMark(tiny_model, tokenizer, max_tokens=2048, temperature=2.0)
    logits = warm.slot_logits(list(encoded_pair))
    for d, x in zip(warm.forward_distributions(list(encoded_pair)), logits):
        torch.testing.assert_close(d, torch.softmax(x.detach() / 2.0, dim=-1))


# Padding


def test_collate_pads_on_the_right_with_mask(tokenizer, encoded_pair):
    a, b = encoded_pair
    batch = collate([a, b], pad_id=tokenizer.pad_token_id)
    longest = max(len(a.input_ids), len(b.input_ids))
    assert tuple(batch.input_ids.shape) == (2, longest)
    for row, encoded in enumerate((a, b)):
        n = len(encoded.input_ids)
        assert batch.input_ids[row, :n].tolist() == list(encoded.input_ids)
        assert batch.attention_mask[row, :n].tolist() == [1] * n
        assert batch.attention_mask[row, n:].tolist() == [0] * (longest - n)
        assert batch.input_ids[row, n:].tolist() == [tokenizer.pad_token_id] * (longest - n)
    assert batch.slot_positions == (a.slot_positions, b.slot_positions)


# Loading


@pytest.fixture
def saved_run(tmp_path, tiny_model, tokenizer, lora_model, base_config):
    backbone_dir = tmp_path / "backbone"
    tiny_model.save_pretrained(backbone_dir)
    tokenizer.save_pretrained(backbone_dir)
    run_dir = tmp_path / "runs" / "tiny_run"
    lora_model.save_pretrained(run_dir / "adapter")
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 1.5}))
    (run_dir / "model_id.txt").write_text("jevmark-tiny_run\n")
    config = copy.deepcopy(base_config)
    config["backbone"] = {"id": str(backbone_dir), "revision": None}
    return config, run_dir


def test_load_backbone_adapter_and_calibration(saved_run, lora_model, tokenizer, encoded_pair):
    config, run_dir = saved_run
    loaded = JevMark.load(config, checkpoint=run_dir, device="cpu")
    assert loaded.temperature == 1.5
    assert loaded.model_id == "jevmark-tiny_run"
    assert loaded.max_tokens == config["max_tokens"] == 2048
    reference = JevMark(lora_model, tokenizer, max_tokens=2048, temperature=1.5)
    for x, y in zip(loaded.forward_distributions(list(encoded_pair)), reference.forward_distributions(list(encoded_pair))):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_load_without_checkpoint_uses_defaults(saved_run):
    config, _ = saved_run
    loaded = JevMark.load(config, device="cpu")
    assert loaded.temperature == 1.0
    assert loaded.model_id == f"jevmark-{config['run_name']}"


def test_load_missing_checkpoint_raises_runtime_error(saved_run, tmp_path):
    config, _ = saved_run
    with pytest.raises(RuntimeError, match="checkpoint"):
        JevMark.load(config, checkpoint=tmp_path / "runs" / "does_not_exist", device="cpu")
