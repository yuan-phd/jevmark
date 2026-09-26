"""scripts/train_rlcd.py on the tiny model, CPU (task 2.2)."""

import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import load_file

from jevmark.config import load_config
from jevmark.data.assemble import build_all
from jevmark.data.form import FORM_KINDS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_rlcd", REPO / "scripts" / "train_rlcd.py")
rl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rl
spec.loader.exec_module(rl)

TRAINING = {
    "data_dir": "data",
    "max_tokens": 1024,
    "steps": 10,
    "lr": 5.0e-5,
    "weight_decay": 0.0,
    "warmup_steps": 2,
    "effective_batch": 4,
    "micro_batch": 2,
    "grad_clip": 1.0,
    "eval_every": 5,
    "valid_subset": 8,
    "gradient_checkpointing": False,
}


# Rewards, advantages and weights by hand


def t(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_rewards_by_hand():
    r, p = t(1.0, 0.0, 1.0, 0.0), t(0.8, 0.3, 0.2, 0.9)
    torch.testing.assert_close(rl.reward("outcome", r, p), t(1.0, 0.0, 1.0, 0.0))
    torch.testing.assert_close(rl.reward("outcome_minus_p", r, p), t(0.2, -0.3, 0.8, -0.9))
    torch.testing.assert_close(rl.reward("brier", r, p), t(1 - 0.04, 1 - 0.09, 1 - 0.64, 1 - 0.81))
    torch.testing.assert_close(rl.reward("log", r, p), t(math.log(0.8), math.log(0.7), math.log(0.2), math.log(0.1)))


def test_log_reward_clips_probabilities():
    r, p = t(1.0, 0.0), t(0.0, 1.0)
    torch.testing.assert_close(rl.reward("log", r, p), t(math.log(1e-6), math.log(1e-6)))


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        rl.reward("ppo", t(1.0), t(0.5))


def test_advantages_have_zero_group_mean():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.3, 0.9, 0.1, 0.5], [2.0, 2.0, 2.0, 2.0]])
    for normalize in (False, True):
        a = rl.advantages(rewards, normalize)
        torch.testing.assert_close(a.mean(dim=-1), torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(rl.advantages(rewards, False)[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))
    torch.testing.assert_close(rl.advantages(rewards, True)[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4, rtol=0)
    assert torch.all(rl.advantages(rewards, True)[2] == 0)  # equal rewards carry no signal


def test_behaviour_mixes_in_uniform_and_the_importance_weight_is_clipped():
    p = t(0.97, 0.01, 0.01, 0.01)
    q = rl.behaviour(p, 0.1)
    torch.testing.assert_close(q, t(0.9 * 0.97 + 0.025, 0.9 * 0.01 + 0.025, 0.9 * 0.01 + 0.025, 0.9 * 0.01 + 0.025))
    assert float(q.sum()) == pytest.approx(1.0)
    w = rl.importance_weights(p, q, 5.0)
    torch.testing.assert_close(w, p / q)  # all under the clip here
    torch.testing.assert_close(rl.importance_weights(t(0.9), t(0.1), 5.0), t(5.0))  # 9 clipped to 5


def settings(arm, **kw):
    base = dict(arm=arm, group_size=4, epsilon=0.1, importance_weight=True, importance_clip=5.0, normalize_std=False, beta=0.02)
    base.update(kw)
    return rl.Settings(**base)


def test_policy_gradient_loss_by_hand():
    z = torch.tensor([2.0, 0.0, -1.0], requires_grad=True)
    z_ref = torch.tensor([1.0, 0.5, 0.0])
    s = settings("outcome", beta=0.0)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    loss, stats = rl.question_loss(z, z_ref, 0, s, g1)
    p = torch.softmax(z.detach(), -1)
    q = rl.behaviour(p, 0.1)
    actions = torch.multinomial(q, 4, replacement=True, generator=g2)
    r = (actions == 0).float()
    expected = -((r - r.mean()) * (p[actions] / q[actions]).clamp(max=5.0) * torch.log_softmax(z.detach(), -1)[actions]).mean()
    assert float(loss.detach()) == pytest.approx(float(expected), abs=1e-6)
    assert stats["reward"] == pytest.approx(float(r.mean()))
    loss.backward()
    assert z.grad is not None


def test_kl_term_is_zero_at_the_reference_and_sft_cont_is_cross_entropy():
    z = torch.tensor([1.0, -0.5, 0.2, 0.0], requires_grad=True)
    _, stats = rl.question_loss(z, z.detach().clone(), 2, settings("brier"), torch.Generator().manual_seed(0))
    assert stats["kl"] == pytest.approx(0.0, abs=1e-7)
    loss, _ = rl.question_loss(z, z.detach() + 1.0, 2, settings("sft_cont"), torch.Generator().manual_seed(0))
    assert float(loss.detach()) == pytest.approx(float(-torch.log_softmax(z.detach(), -1)[2]))


def test_direct_bandit_differentiates_the_score_of_the_sampled_action():
    z = torch.tensor([0.5, 0.0], requires_grad=True)
    s = settings("direct_bandit", beta=0.0)
    loss, _ = rl.question_loss(z, z.detach(), 0, s, torch.Generator().manual_seed(1))
    actions = torch.multinomial(rl.behaviour(torch.softmax(z.detach(), -1), 0.1), 4, replacement=True, generator=torch.Generator().manual_seed(1))
    p = torch.softmax(z.detach(), -1)
    assert float(loss.detach()) == pytest.approx(float((((actions == 0).float() - p[actions]) ** 2).mean()), abs=1e-6)
    loss.backward()
    assert torch.any(z.grad != 0)


# The tiny model end to end


@pytest.fixture(scope="module")
def records():
    built = build_all(load_config(REPO / "configs" / "data.yaml"), ["train", "valid"])
    train_split = built.splits["train"]
    clinc = [r for r in train_split if r["source"].startswith("clinc")][:24]
    sst5 = [r for r in train_split if r["source"] == "SetFit/sst5"][:8]
    return clinc + sst5, built.splits["valid"][:12]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, tiny_model, tokenizer, base_config, records):
    from peft import LoraConfig, get_peft_model

    root = tmp_path_factory.mktemp("rlcd")
    backbone = root / "backbone"
    tiny_model.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    data_dir = root / "data"
    data_dir.mkdir()
    train_records, valid_records = records
    for name, rows in (("train", train_records), ("valid", valid_records)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # A stand-in SFT run: a LoRA with random non-zero weights, so the policy differs from the backbone.
    torch.manual_seed(0)
    peft_model = get_peft_model(copy.deepcopy(tiny_model), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=False))
    init = root / "sft_tiny"
    peft_model.save_pretrained(init / "adapter")
    config = yaml.safe_load((REPO / "configs" / "rlcd_06b.yaml").read_text())
    config["backbone"] = {"id": str(backbone), "revision": None}
    config["tokenizer_reference"] = base_config["tokenizer_reference"]
    config["training"] = dict(TRAINING)
    config["rlcd"]["init"] = str(init)
    config_path = root / "rlcd_tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return SimpleNamespace(root=root, config_path=config_path, data_dir=data_dir, init=init)


def run_training(setup, runs_dir, *extra):
    code = rl.main(["--config", str(setup.config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", *extra])
    assert code == 0


def log_of(run_dir):
    return [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines()]


def adapter_tensors(path):
    return load_file(str(path / "adapter_model.safetensors"))


@pytest.mark.parametrize("arm", rl.ARMS)
def test_ten_steps_of_every_arm_complete_on_cpu(setup, tmp_path, arm):
    run_training(setup, tmp_path, f"arm={arm}")
    run_dir = tmp_path / f"rlcd_06b_{arm}_s0"
    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["steps"] == summary["total_steps"] == 10 and summary["arm"] == arm
    assert summary["best_step"] in (5, 10)
    assert (run_dir / "adapter" / "adapter_model.safetensors").is_file() and (run_dir / "adapter_last" / "adapter_model.safetensors").is_file()
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert config["rlcd"]["init_adapter_sha256"] == summary["init_adapter_sha256"] == rl.adapter_sha256(setup.init / "adapter")
    log = log_of(run_dir)
    steps = [e for e in log if "loss" in e]
    assert [e["step"] for e in steps] == list(range(1, 11)) and all(math.isfinite(e["loss"]) for e in steps)
    valids = [e for e in log if e.get("event") == "valid"]
    assert [e["step"] for e in valids] == [0, 5, 10]
    assert valids[0]["kl_to_ref"] == pytest.approx(0.0, abs=1e-5)  # step 0 is the SFT adapter itself
    for key in ("accuracy", "ece", "nll", "kl_to_ref", "expected_reward", "expected_p_chosen"):
        assert key in valids[-1]
    assert any(not torch.equal(a, b) for a, b in zip(adapter_tensors(run_dir / "adapter_last").values(), adapter_tensors(setup.init / "adapter").values()))


def test_form_nouls_are_never_sampled_or_trained(setup, monkeypatch):
    from jevmark.model import JevMark

    config = load_config(setup.config_path)
    jev = JevMark.load(config, device="cpu")
    train_records = [json.loads(line) for line in (setup.data_dir / "train.jsonl").read_text().splitlines()]
    examples, _ = rl.epoch_examples(train_records, jev, 1024, 0, 0)
    with_form = [e for e in examples if not all(e.gold_dependent)]
    assert with_form, "the fixture should hold records with form nouls"
    seen = []
    real = rl.question_loss

    def spy(z, z_ref, gold, s, generator):
        seen.append(len(z))
        return real(z, z_ref, gold, s, generator)

    monkeypatch.setattr(rl, "question_loss", spy)
    batch = with_form[:2]
    rl.batch_loss(jev, jev, batch, settings("brier"), torch.Generator().manual_seed(0))
    assert len(seen) == sum(sum(e.gold_dependent) for e in batch) < sum(len(e.targets) for e in batch)
    record = next(r for r in train_records if any(q in FORM_KINDS for q in r["questions"]))
    mask = rl.gold_dependent_mask(record)
    assert [keep for keep in mask] == [qid not in FORM_KINDS for qid in record["questions"]]


def test_resume_matches_an_uninterrupted_run(setup, tmp_path):
    uninterrupted = tmp_path / "a"
    run_training(setup, uninterrupted, "arm=log")
    resumed = tmp_path / "b"
    run_training(setup, resumed, "arm=log", "--limit-steps", "5")
    run_dir = resumed / "rlcd_06b_log_s0"
    run_training(setup, resumed, "arm=log", "--resume")
    log = log_of(run_dir)
    assert [e["step"] for e in log if "loss" in e] == list(range(1, 11))
    assert any(e.get("event") == "resume" and e["step"] == 5 for e in log)
    a = adapter_tensors(uninterrupted / "rlcd_06b_log_s0" / "adapter_last")
    b = adapter_tensors(run_dir / "adapter_last")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    rewards_a = [e["reward"] for e in log_of(uninterrupted / "rlcd_06b_log_s0") if "loss" in e]
    rewards_b = [e["reward"] for e in log if "loss" in e]
    assert rewards_a == pytest.approx(rewards_b)  # the same actions were sampled after the resume


def test_a_second_fresh_start_is_refused_and_a_changed_init_adapter_is_caught(setup, tmp_path):
    run_training(setup, tmp_path, "arm=outcome", "--limit-steps", "2")
    with pytest.raises(SystemExit, match="already holds a training run"):
        rl.main(["--config", str(setup.config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path), "--device", "cpu", "arm=outcome"])
    moved = tmp_path / "other_init"
    shutil.copytree(setup.init, moved)
    run_dir = tmp_path / "rlcd_06b_outcome_s0"
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    config["rlcd"]["init_adapter_sha256"] = "0" * 64
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit, match="sha256 differs"):
        rl.main(["--config", str(setup.config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path), "--device", "cpu", "arm=outcome", "--resume"])


def test_run_name_and_overrides():
    config = load_config(REPO / "configs" / "rlcd_17b.yaml", ["arm=direct_bandit", "seed=2"])
    assert rl.resolve_run_name(config) == "rlcd_17b_direct_bandit_s2"
    assert config["training"]["gradient_checkpointing"] is True and config["training"]["steps"] == 500
    config6 = load_config(REPO / "configs" / "rlcd_06b.yaml")
    assert config6["training"]["lr"] == 5e-5 and config6["training"]["warmup_steps"] == 20 and config6["rlcd"]["beta"] == 0.02
    assert config6["training"]["micro_batch"] == load_config(REPO / "configs" / "sft_06b.yaml")["training"]["micro_batch"]
    with pytest.raises(ValueError):
        rl.Settings.from_config({**config6, "arm": "ppo"})
