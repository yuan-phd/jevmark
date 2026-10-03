"""v3 learners from the deployment-feedback log: train_rlcd.py log mode and calibrate.py --fit-log, tiny model on CPU (task 3.3)."""

import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file

from jevmark.calibration import fit_outcome_temperature
from jevmark.config import load_config
from jevmark.data import unseen
from jevmark.data.description_loader import load_descriptions
from jevmark.feedback import Interaction, file_sha256, interactions, write_log
from jevmark.metrics import QuestionResult, write_results
from jevmark.model import JevMark

REPO = Path(__file__).resolve().parents[1]
DATA_CONFIG = load_config(REPO / "configs" / "data.yaml")
DESCRIPTIONS = load_descriptions("banking77")
NAMES = sorted(DESCRIPTIONS)
N = 40  # 36 training interactions, 4 for selection


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_v3", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rl = load_script("train_rlcd")
calibrate = load_script("calibrate")


def make_log(path: Path, records, seed: int, data_sha: str, outcome_override=None) -> list[Interaction]:
    """A synthetic log over the records: random distributions, actions and flips from feedback.interactions."""
    rng = np.random.default_rng(seed)
    rows = []
    for record in records:
        (qid, question), = record["questions"].items()
        labels = list(question["criteria"])
        rows.append((record["id"], qid, labels, rng.dirichlet(np.ones(len(labels))).tolist(), record["gold"][qid]))
    log = interactions(rows, seed=seed, model="jevmark-tiny", commit="test")
    if outcome_override is not None:
        log = [outcome_override(i) for i in log]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_log(path, log)
    (path.parent / "metrics.json").write_text(json.dumps({"sampling": {"seed": seed}, "data_files_sha256": {"v3_banking77_train.jsonl": data_sha}}))
    return log


@pytest.fixture(scope="module")
def setup(tmp_path_factory, tiny_model, tokenizer, base_config):
    from peft import LoraConfig, get_peft_model

    root = tmp_path_factory.mktemp("v3")
    backbone = root / "backbone"
    tiny_model.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    torch.manual_seed(0)
    init = root / "sft_tiny"
    get_peft_model(copy.deepcopy(tiny_model), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=False)).save_pretrained(init / "adapter")
    data_dir = root / "data"
    data_dir.mkdir()
    rows = [(i, f"message {i} about my card payment", NAMES[(3 * i) % len(NAMES)]) for i in range(N + 4)]
    records = unseen.banking77_records(unseen.V3_TRAIN, rows, "train", NAMES, DESCRIPTIONS, DATA_CONFIG)
    (data_dir / "v3_banking77_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    data_sha = file_sha256(data_dir / "v3_banking77_train.jsonl")
    log_path = root / "v3_log_s0" / "log.jsonl"
    log = make_log(log_path, records, 0, data_sha)
    config = yaml.safe_load((REPO / "configs" / "v3_06b.yaml").read_text())
    config["backbone"] = {"id": str(backbone), "revision": None}
    config["tokenizer_reference"] = base_config["tokenizer_reference"]
    config["training"].update({"micro_batch": 2, "effective_batch": 4, "eval_every": 5, "warmup_steps": 2})
    config["v3"].update({"init": str(init), "epochs": 1, "min_steps": 10})  # max(10, ceil(36 / 4)) = 10 steps
    config_path = root / "v3_tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return SimpleNamespace(root=root, config_path=config_path, data_dir=data_dir, init=init, records=records, log_path=log_path, log=log, data_sha=data_sha)


def train(setup, runs_dir, *extra, log_path=None):
    argv = ["--config", str(setup.config_path), "--log", str(log_path or setup.log_path), "--n", str(N), "--data-dir", str(setup.data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", *extra]
    assert rl.main(argv) == 0


def events(run_dir):
    return [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines()]


def tensors(path):
    return load_file(str(path / "adapter_model.safetensors"))


@pytest.mark.parametrize("arm", rl.LOG_ARMS)
def test_each_arm_trains_ten_steps_from_the_log(setup, tmp_path, arm):
    train(setup, tmp_path, f"arm={arm}")
    run_dir = tmp_path / f"v3_06b_{arm}_n{N}_s0"
    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["steps"] == summary["total_steps"] == 10 and summary["arm"] == arm and summary["mode"] == "log"
    assert summary["log_sha256"] == file_sha256(setup.log_path) and summary["n"] == N and summary["n_train"] == 36 and summary["n_select"] == 4
    assert [point["step"] for point in summary["selection_curve"]] == [0, 5, 10] and summary["best_step"] in (5, 10)
    assert (run_dir / "adapter" / "adapter_model.safetensors").is_file() and (run_dir / "adapter_last" / "adapter_model.safetensors").is_file()
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert config["v3"]["log_sha256"] == summary["log_sha256"] and config["v3"]["steps"] == 10 and config["v3"]["noisy"] is False
    assert config["v3"]["init_adapter_sha256"] == rl.adapter_sha256(setup.init / "adapter")
    steps = [e for e in events(run_dir) if "loss" in e]
    assert [e["step"] for e in steps] == list(range(1, 11)) and all(math.isfinite(e["loss"]) for e in steps)


def test_positive_sft_trains_only_on_outcome_1_interactions(setup, tmp_path, monkeypatch):
    seen = []
    real = rl.log_batch_loss

    def spy(jev, batch, arm):
        seen.extend(e.outcome for e in batch)
        return real(jev, batch, arm)

    monkeypatch.setattr(rl, "log_batch_loss", spy)
    train(setup, tmp_path, "arm=positive_sft")
    positives = sum(i.outcome for i in setup.log[:36])
    start = next(e for e in events(tmp_path / f"v3_06b_positive_sft_n{N}_s0") if e.get("event") == "start")
    assert 0 < positives < 36 and start["n_used"] == positives
    assert seen and set(seen) == {1}


def test_full_sft_never_reads_the_logged_outcome_to_train(setup, tmp_path):
    inverted = setup.root / "v3_log_inverted" / "log.jsonl"
    make_log(inverted, setup.records, 0, setup.data_sha, outcome_override=lambda i: i if i.index >= 36 else Interaction(**{**i.__dict__, "outcome": 1 - i.outcome, "flipped_outcome": 1 - i.flipped_outcome}))
    for arm in ("full_sft", "positive_sft"):
        train(setup, tmp_path / "a", f"arm={arm}", "--limit-steps", "3")
        train(setup, tmp_path / "b", f"arm={arm}", "--limit-steps", "3", log_path=inverted)
        a = tensors(tmp_path / "a" / f"v3_06b_{arm}_n{N}_s0" / "last" / "adapter")
        b = tensors(tmp_path / "b" / f"v3_06b_{arm}_n{N}_s0" / "last" / "adapter")
        same = all(torch.equal(a[k], b[k]) for k in a)
        assert same is (arm == "full_sft")  # full_sft is blind to training outcomes; positive_sft is not


def test_noisy_reads_the_flipped_outcome_and_names_the_run(setup, tmp_path, base_config):
    config = load_config(setup.config_path)
    jev = JevMark.load(config, device="cpu")
    records = {r["id"]: r for r in setup.records}
    clean = rl.log_examples(setup.log[:10], records, jev, 1024, noisy=False, with_gold=False)
    noisy = rl.log_examples(setup.log[:10], records, jev, 1024, noisy=True, with_gold=False)
    assert [e.outcome for e in clean] == [i.outcome for i in setup.log[:10]]
    assert [e.outcome for e in noisy] == [i.flipped_outcome for i in setup.log[:10]]
    assert all(e.gold is None for e in clean)
    train(setup, tmp_path, "arm=direct_brier", "--noisy", "--limit-steps", "1")
    assert (tmp_path / f"v3_06b_direct_brier_n{N}_s0_noisy" / "config.yaml").is_file()


def test_full_sft_with_noisy_and_other_arms_are_refused(setup, tmp_path):
    base = ["--config", str(setup.config_path), "--log", str(setup.log_path), "--n", str(N), "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path), "--device", "cpu"]
    with pytest.raises(SystemExit, match="full_sft"):
        rl.main([*base, "arm=full_sft", "--noisy"])
    with pytest.raises(SystemExit, match="log mode"):
        rl.main([*base, "arm=outcome"])
    with pytest.raises(SystemExit):
        rl.parse_args(["--config", "x", "--n", "5"])


def test_a_log_drawn_with_seed_1_names_the_run(setup, tmp_path):
    log1 = setup.root / "v3_log_s1" / "log.jsonl"
    make_log(log1, setup.records, 1, setup.data_sha)
    train(setup, tmp_path, "arm=direct_brier", "--limit-steps", "1", log_path=log1)
    assert (tmp_path / f"v3_06b_direct_brier_n{N}_s0_log1").is_dir()


class FixedLogits:
    """A stand-in for JevMark whose letter logits are given, so the criterion can be computed by hand."""

    def __init__(self, logits):
        self.logits = logits
        self.model = SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)

    def slot_logits(self, encoded):
        return [self.logits[e] for e in encoded]


def test_selection_criterion_matches_a_hand_computation():
    logits = {0: torch.tensor([2.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0]), 2: torch.tensor([0.5, 0.5, 3.0])}
    examples = [rl.LogExample(0, 0, 1, None), rl.LogExample(1, 1, 0, None), rl.LogExample(2, 0, 0, None)]
    result = rl.select_score(FixedLogits(logits), examples, batch_size=2)
    p0 = math.exp(2) / (math.exp(2) + 2)
    p1 = math.exp(1) / (1 + math.exp(1))
    p2 = math.exp(0.5) / (2 * math.exp(0.5) + math.exp(3))
    assert result["criterion"] == pytest.approx((math.log(p0) + math.log(1 - p1) + math.log(1 - p2)) / 3)
    assert result["mean_p_action"] == pytest.approx((p0 + p1 + p2) / 3) and result["outcome_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("n, steps", [(500, 200), (2000, 563), (5000, 1407)])
def test_step_rule_at_the_three_n(n, steps):
    config = load_config(REPO / "configs" / "v3_06b.yaml")
    n_train = math.floor(config["v3"]["train_share"] * n + 1e-9)
    assert rl.log_steps(n_train, config["v3"]["epochs"], config["v3"]["min_steps"], config["training"]["effective_batch"]) == steps
    assert config["training"]["warmup_steps"] == 20 and config["training"]["lr"] == 5e-5 and config["v3"]["beta"] == 0.0
    assert config["training"]["micro_batch"] * 4 == config["training"]["effective_batch"] == 32 and config["training"]["eval_every"] == 50


def test_resume_matches_an_uninterrupted_run(setup, tmp_path):
    train(setup, tmp_path / "a", "arm=direct_brier")
    train(setup, tmp_path / "b", "arm=direct_brier", "--limit-steps", "5")
    run_dir = tmp_path / "b" / f"v3_06b_direct_brier_n{N}_s0"
    train(setup, tmp_path / "b", "arm=direct_brier", "--resume")
    log = events(run_dir)
    assert [e["step"] for e in log if "loss" in e] == list(range(1, 11)) and any(e.get("event") == "resume" and e["step"] == 5 for e in log)
    a = tensors(tmp_path / "a" / f"v3_06b_direct_brier_n{N}_s0" / "adapter_last")
    b = tensors(run_dir / "adapter_last")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    losses_a = [e["loss"] for e in events(tmp_path / "a" / f"v3_06b_direct_brier_n{N}_s0") if "loss" in e]
    assert losses_a == pytest.approx([e["loss"] for e in log if "loss" in e], abs=1e-6)
    with pytest.raises(SystemExit, match="another log"):
        rl.main(["--config", str(setup.config_path), "--log", str(setup.log_path), "--n", "39", "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path / "b"), "--device", "cpu", "arm=direct_brier", "run_name=" + run_dir.name, "--resume"])


def test_temperature_fit_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    true_t = 2.0
    probs, actions, outcomes = [], [], []
    for _ in range(20000):
        k = int(rng.integers(2, 11))
        p = rng.dirichlet(np.full(k, 0.3))
        z = np.log(np.maximum(p, 1e-45)) / true_t
        q = np.exp(z - z.max())
        q /= q.sum()
        a = int(rng.integers(k))
        probs.append(p.tolist())
        actions.append(a)
        outcomes.append(int(rng.random() < q[a]))
    assert fit_outcome_temperature(probs, actions, outcomes) == pytest.approx(true_t, rel=0.08)


def test_calibrate_fit_log_writes_the_temperature_run(setup, tmp_path):
    source = tmp_path / "v3_06b_zeroshot"
    source.mkdir()
    results = [QuestionResult(i.record_id, "label", "choice", i.probs, i.labels.index(next(r["gold"]["label"] for r in setup.records if r["id"] == i.record_id)), i.labels, split=unseen.V3_TEST_FULL) for i in setup.log]
    write_results(source / "results.jsonl.gz", results)
    (source / "metrics.json").write_text(json.dumps({"run_name": source.name, "splits": {}}))
    (source / "config.yaml").write_text("run_name: v3_06b_zeroshot\n")
    assert calibrate.main([str(source), "--fit-log", str(setup.log_path), "--n", str(N)]) == 0
    out = tmp_path / f"v3_06b_temp_n{N}"
    expected = fit_outcome_temperature([i.probs for i in setup.log[:36]], [i.action for i in setup.log[:36]], [i.outcome for i in setup.log[:36]])
    cal = json.loads((out / "calibration.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert cal["temperature"] == pytest.approx(expected) and metrics["calibration"]["n_fit"] == 36
    assert metrics["calibration"]["log_sha256"] == file_sha256(setup.log_path) and set(metrics["splits"]) == {unseen.V3_TEST_FULL}
    with pytest.raises(SystemExit):
        calibrate.main([str(source), "--fit-log", str(setup.log_path)])


def test_a_stored_adapter_becomes_a_run_that_evaluates(setup, tmp_path):
    """scripts/prepare_adapter_run.py (task 3.6): adapter_last of a trained run, evaluated as a new run."""
    prepare = load_script("prepare_adapter_run")
    evaluate = load_script("evaluate")
    train(setup, tmp_path, "arm=direct_brier")
    source = tmp_path / f"v3_06b_direct_brier_n{N}_s0"
    last = source / "adapter_last"
    expected = file_sha256(last / "adapter_model.safetensors")
    with pytest.raises(SystemExit, match="expected"):
        prepare.main(["--adapter", str(last), "--source-run", str(source), "--name", "v3_tiny_last", "--runs-dir", str(tmp_path), "--sha256", "0" * 64])
    assert not (tmp_path / "v3_tiny_last").exists()
    with pytest.raises(SystemExit, match="adapter_model.safetensors"):
        prepare.main(["--adapter", str(tmp_path), "--source-run", str(source), "--name", "v3_tiny_last", "--runs-dir", str(tmp_path)])
    assert prepare.main(["--adapter", str(last), "--source-run", str(source), "--adapter-dir", "adapter_last", "--name", "v3_tiny_last", "--runs-dir", str(tmp_path), "--sha256", expected]) == 0
    run = tmp_path / "v3_tiny_last"
    config = yaml.safe_load((run / "config.yaml").read_text())
    source_config = yaml.safe_load((source / "config.yaml").read_text())
    assert config["run_name"] == "v3_tiny_last" and config["adapter_eval"] == {"source_run": source.name, "adapter_dir": "adapter_last", "adapter_sha256": expected}
    assert config["v3"] == source_config["v3"] and (run / "model_id.txt").read_text() == "jevmark-v3_tiny_last\n"
    assert file_sha256(run / "adapter" / "adapter_model.safetensors") == expected
    with pytest.raises(SystemExit, match="never overwritten"):
        prepare.main(["--adapter", str(last), "--source-run", str(source), "--name", "v3_tiny_last", "--runs-dir", str(tmp_path)])
    argv = ["--ckpt", str(run), "--splits", "v3_banking77_train", "--limit", "4", "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path), "--device", "cpu"]
    assert evaluate.main(argv) == 0
    metrics = json.loads((tmp_path / "v3_tiny_last_limit4" / "metrics.json").read_text())
    assert metrics["adapter_sha256"] == expected and metrics["lora_merged"] is True and metrics["model_id"] == "jevmark-v3_tiny_last"
    assert metrics["v3"]["log_sha256"] == source_config["v3"]["log_sha256"]
