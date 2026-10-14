"""v3 learners from the deployment-feedback log: train_rlcd.py log mode and calibrate.py --fit-log, tiny model on CPU (task 3.3)."""

import copy
import importlib.util
import json
import math
import random
import sys
from dataclasses import replace
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
from jevmark.metrics import QuestionResult, read_results, write_results
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
    clean = rl.log_examples(setup.log[:10], records, jev, 1024, "clean", with_gold=False)
    noisy = rl.log_examples(setup.log[:10], records, jev, 1024, "flipped", with_gold=False)
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
    emotion = tmp_path / "v3_emotion_06b_zeroshot"  # an emotion zero-shot run names its temperature runs after its domain
    emotion.mkdir()
    for name in ("results.jsonl.gz", "metrics.json", "config.yaml"):
        (emotion / name).write_bytes((source / name).read_bytes())
    assert calibrate.main([str(emotion), "--fit-log", str(setup.log_path), "--n", str(N)]) == 0
    assert json.loads((tmp_path / f"v3_emotion_06b_temp_n{N}" / "calibration.json").read_text())["temperature"] == pytest.approx(expected)


def test_a_temperature_on_a_trained_learner_uses_its_own_probabilities_on_the_log_prefix(setup, tmp_path):
    """evaluate.py --log-prefix, then calibrate.py --fit-log --fit-probs (task 3.5, temperature on RLCD)."""
    evaluate = load_script("evaluate")
    train(setup, tmp_path, "arm=direct_brier")
    run = tmp_path / f"v3_06b_direct_brier_n{N}_s0"
    argv = ["--ckpt", str(run), "--log-prefix", str(setup.log_path), "--n", str(N), "--no-probes", "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path), "--device", "cpu"]
    assert evaluate.main(argv) == 0
    prefix_run = tmp_path / f"{run.name}_prefix"
    metrics = json.loads((prefix_run / "metrics.json").read_text())
    assert metrics["log_prefix"] == {"log": str(setup.log_path), "log_sha256": file_sha256(setup.log_path), "n": N, "n_interactions": 36}
    assert metrics["latency"] is None and metrics["batching_precision"] is None and list(metrics["splits"]) == [unseen.V3_TRAIN]
    prefix_results = {r.record_id: r for r in read_results(prefix_run / "results.jsonl.gz")}
    assert set(prefix_results) == {i.record_id for i in setup.log[:36]}

    # The learner's own evaluation, standing in for its test-split results.
    write_results(run / "results.jsonl.gz", [replace(r, split=unseen.V3_TEST_FULL) for r in prefix_results.values()])
    run_metrics = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").is_file() else {"run_name": run.name, "splits": {}}
    run_metrics["adapter_sha256"] = metrics["adapter_sha256"]
    (run / "metrics.json").write_text(json.dumps(run_metrics))
    assert calibrate.main([str(run), "--fit-log", str(setup.log_path), "--n", str(N), "--fit-probs", str(prefix_run)]) == 0
    fit = setup.log[:36]
    expected = fit_outcome_temperature([prefix_results[i.record_id].probs for i in fit], [i.action for i in fit], [i.outcome for i in fit])
    cal = json.loads((tmp_path / f"{run.name}_temp" / "metrics.json").read_text())["calibration"]
    assert cal["temperature"] == pytest.approx(expected) and cal["probs_from"] == str(prefix_run) and cal["n_fit"] == 36
    assert cal["temperature"] != pytest.approx(fit_outcome_temperature([i.probs for i in fit], [i.action for i in fit], [i.outcome for i in fit]))

    with pytest.raises(SystemExit, match="first 0.9 N"):
        calibrate.main([str(run), "--fit-log", str(setup.log_path), "--n", "20", "--fit-probs", str(prefix_run), "--out", str(tmp_path / "x")])
    run_metrics["adapter_sha256"] = "0" * 64
    (run / "metrics.json").write_text(json.dumps(run_metrics))
    with pytest.raises(SystemExit, match="another adapter"):
        calibrate.main([str(run), "--fit-log", str(setup.log_path), "--n", str(N), "--fit-probs", str(prefix_run), "--out", str(tmp_path / "y")])


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


# --noise soft (task 3.6, decision 59)


def test_noise_mode_of_old_and_new_configs_and_the_run_names():
    assert rl.noise_of({"noisy": True}) == "fixed" and rl.noise_of({"noisy": False}) is None and rl.noise_of({}) is None
    assert rl.noise_of({"noisy": True, "noise": "soft"}) == "soft"
    base = {"size": "06b", "arm": "direct_brier", "seed": 0}
    assert rl.log_run_name(base, 5000, "soft", 0) == "v3_06b_direct_brier_n5000_s0_noisy_soft"
    assert rl.log_run_name(base, 5000, "fixed", 0) == rl.log_run_name(base, 5000, True, 0) == "v3_06b_direct_brier_n5000_s0_noisy"
    assert rl.log_run_name(base, 5000, None, 1) == rl.log_run_name(base, 5000, False, 1) == "v3_06b_direct_brier_n5000_s0_log1"
    assert rl.parse_args(["--config", "x", "--log", "l", "--n", "5", "--noisy"]).noise == "fixed"
    assert rl.parse_args(["--config", "x", "--log", "l", "--n", "5", "--noise", "soft"]).noise == "soft"
    with pytest.raises(SystemExit):
        rl.parse_args(["--config", "x", "--log", "l", "--n", "5", "--noisy", "--noise", "soft"])
    with pytest.raises(SystemExit):
        rl.parse_args(["--config", "x", "--noise", "soft"])


def test_soft_targets_are_the_expected_flipped_outcome_and_the_loss_uses_them(setup):
    config = load_config(setup.config_path)
    jev = JevMark.load(config, device="cpu")
    records = {r["id"]: r for r in setup.records}
    soft = rl.log_examples(setup.log[:10], records, jev, 1024, "soft", with_gold=False)
    assert [e.outcome for e in soft] == [pytest.approx(0.2 + 0.6 * i.outcome) for i in setup.log[:10]]
    assert {round(e.outcome, 6) for e in soft} <= {0.2, 0.8}
    assert [e.outcome for e in rl.log_examples(setup.log[:4], records, jev, 1024, "soft", with_gold=False, flip_rate=0.1)] == [pytest.approx(0.1 + 0.8 * i.outcome) for i in setup.log[:4]]
    with pytest.raises(ValueError, match="outcome column"):
        rl.log_examples(setup.log[:1], records, jev, 1024, "noisy", with_gold=False)
    loss, stats = rl.log_batch_loss(jev, soft[:3], "direct_brier")
    with torch.no_grad():
        p = [float(torch.softmax(z.float(), dim=-1)[e.action]) for z, e in zip(jev.slot_logits([e.encoded for e in soft[:3]]), soft[:3])]
    assert float(loss.detach()) == pytest.approx(sum((e.outcome - q) ** 2 for e, q in zip(soft[:3], p)) / 3, rel=1e-5)
    assert stats["outcome_rate"] == pytest.approx(sum(e.outcome for e in soft[:3]) / 3)  # the mean training target


def test_a_soft_run_trains_on_soft_targets_and_selects_on_the_stored_flipped_outcomes(setup, tmp_path, monkeypatch):
    targets = []
    real = rl.log_batch_loss

    def spy(jev, batch, arm):
        targets.extend(e.outcome for e in batch)
        return real(jev, batch, arm)

    monkeypatch.setattr(rl, "log_batch_loss", spy)
    train(setup, tmp_path, "arm=direct_brier", "--noise", "soft", "--limit-steps", "2")
    run_dir = tmp_path / f"v3_06b_direct_brier_n{N}_s0_noisy_soft"
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert config["v3"]["noise"] == "soft" and config["v3"]["noisy"] is True and config["v3"]["flip_rate"] == 0.2
    assert targets and {round(x, 6) for x in targets} <= {0.2, 0.8}
    start = next(e for e in events(run_dir) if e.get("event") == "start")
    assert start["noise"] == "soft" and start["n_used"] == 36
    select = next(e for e in events(run_dir) if e.get("event") == "select" and e["step"] == 0)
    flipped = [i.flipped_outcome for i in setup.log[36:40]]
    assert select["outcome_rate"] == pytest.approx(sum(flipped) / len(flipped))  # selection never sees the soft target or the clean column
    with pytest.raises(SystemExit, match="noise mode"):
        train(setup, tmp_path, "arm=direct_brier", "run_name=" + run_dir.name, "--noise", "fixed", "--resume")
    train(setup, tmp_path, "arm=direct_brier", "--noise", "soft", "--resume")
    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["steps"] == 10 and summary["noise"] == "soft" and summary["noisy"] is True


def test_soft_noise_is_refused_for_positive_sft_and_full_sft(setup, tmp_path):
    for arm in ("positive_sft", "full_sft"):
        with pytest.raises(SystemExit, match="soft"):
            train(setup, tmp_path, f"arm={arm}", "--noise", "soft")



# Off-policy arms: direct_log, outcome_minus_p_log, ips_log, ips_positive


def test_ips_weights_and_their_normaliser_by_hand():
    # 1 / q = 2, 10, 4, 50; clipped at 5: 2, 5, 4, 5; mean 4; normalised .5, 1.25, 1, 1.25.
    weights, info = rl.ips_weights([0.5, 0.1, 0.25, 0.02])
    assert weights == pytest.approx([0.5, 1.25, 1.0, 1.25])
    assert info["normaliser"] == pytest.approx(4.0) and info["clipped"] == 2 and info["clipped_share"] == 0.5 and info["n"] == 4 and info["clip"] == 5.0
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_off_policy_losses_match_a_hand_computation():
    logits = {0: torch.tensor([2.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0]), 2: torch.tensor([0.5, 0.5, 3.0])}
    p = [math.exp(2) / (math.exp(2) + 2), math.exp(1) / (1 + math.exp(1)), math.exp(0.5) / (2 * math.exp(0.5) + math.exp(3))]
    # encoded, action, outcome, gold, propensity, weight
    batch = [rl.LogExample(0, 0, 1, None, 0.5, 0.8), rl.LogExample(1, 1, 0, None, 0.05, 1.4), rl.LogExample(2, 0, 0, None, 0.02, 0.8)]
    fixed = FixedLogits(logits)
    log_score = [-(math.log(p[0])), -math.log(1 - p[1]), -math.log(1 - p[2])]  # r log p + (1 - r) log(1 - p), negated

    loss, stats = rl.log_batch_loss(fixed, batch, "direct_log")
    assert float(loss) == pytest.approx(sum(log_score) / 3) and "weight_mean" not in stats

    loss, stats = rl.log_batch_loss(fixed, batch, "ips_log")
    assert float(loss) == pytest.approx((0.8 * log_score[0] + 1.4 * log_score[1] + 0.8 * log_score[2]) / 3)
    assert stats["weight_mean"] == pytest.approx(1.0) and stats["weight_clipped_share"] == pytest.approx(2 / 3)  # 1 / q = 2, 20, 50

    loss, _ = rl.log_batch_loss(fixed, batch, "ips_positive")
    assert float(loss) == pytest.approx((-0.8 * math.log(p[0]) - 1.4 * math.log(p[1]) - 0.8 * math.log(p[2])) / 3)

    # outcome_minus_p_log: reward r - p_a, baseline its micro-batch mean, weight min(p / q, 5).
    rewards = [1 - p[0], -p[1], -p[2]]
    baseline = sum(rewards) / 3
    w = [min(p[0] / 0.5, 5.0), min(p[1] / 0.05, 5.0), min(p[2] / 0.02, 5.0)]
    assert w[1] == 5.0 and w[0] < 5.0  # p[1] / .05 = 14.6 is clipped
    expected = sum(-(rewards[k] - baseline) * w[k] * math.log(p[k]) for k in range(3)) / 3  # p[k] is each logged action's probability
    loss, stats = rl.log_batch_loss(fixed, batch, "outcome_minus_p_log")
    assert float(loss) == pytest.approx(expected)
    assert stats["weight_mean"] == pytest.approx(sum(w) / 3) and stats["weight_clipped_share"] == pytest.approx(sum(p[k] / q > 5 for k, q in enumerate((0.5, 0.05, 0.02))) / 3)


def test_ips_runs_record_the_prefix_normaliser_and_ips_positive_trains_on_positives_only(setup, tmp_path, monkeypatch):
    seen = []
    real = rl.log_batch_loss

    def spy(jev, batch, arm):
        seen.extend((e.outcome, e.weight) for e in batch)
        return real(jev, batch, arm)

    monkeypatch.setattr(rl, "log_batch_loss", spy)
    train(setup, tmp_path, "arm=ips_positive")
    prefix_log = setup.log[:36]
    raw = [min(1 / i.propensity, 5.0) for i in prefix_log]
    normaliser = sum(raw) / len(raw)
    positives = [w / normaliser for w, i in zip(raw, prefix_log) if i.outcome == 1]
    run_dir = tmp_path / f"v3_06b_ips_positive_n{N}_s0"
    summary = json.loads((run_dir / "train_summary.json").read_text())
    start = next(e for e in events(run_dir) if e.get("event") == "start")
    for ips in (summary["ips"], start["ips"], yaml.safe_load((run_dir / "config.yaml").read_text())["v3"]["ips"]):
        assert ips["normaliser"] == pytest.approx(normaliser) and ips["n"] == 36 and ips["n_used"] == len(positives)
        assert ips["mean_weight_used"] == pytest.approx(sum(positives) / len(positives))
        assert ips["clipped"] == sum(1 / i.propensity > 5.0 for i in prefix_log) and ips["clipped_used"] == sum(1 / i.propensity > 5.0 for i in prefix_log if i.outcome == 1)
    assert seen and {o for o, _ in seen} == {1} and {round(w, 9) for _, w in seen} <= {round(w, 9) for w in positives}
    steps = [e for e in events(run_dir) if "loss" in e]
    assert all("weight_mean" in e and "weight_clipped_share" in e for e in steps)


def test_outcome_minus_p_log_records_its_weights_and_resumes_exactly(setup, tmp_path):
    train(setup, tmp_path / "a", "arm=outcome_minus_p_log")
    train(setup, tmp_path / "b", "arm=outcome_minus_p_log", "--limit-steps", "5")
    train(setup, tmp_path / "b", "arm=outcome_minus_p_log", "--resume")
    name = f"v3_06b_outcome_minus_p_log_n{N}_s0"
    a = json.loads((tmp_path / "a" / name / "train_summary.json").read_text())["weights"]
    b = json.loads((tmp_path / "b" / name / "train_summary.json").read_text())["weights"]
    assert a["n"] == b["n"] == 10 * 4 and a["clip"] == 5.0 and 0 < a["mean"] <= 5.0 and 0 <= a["clipped_share"] <= 1
    assert a["mean"] == pytest.approx(b["mean"], rel=1e-5) and a["clipped_share"] == pytest.approx(b["clipped_share"])
    steps = [e for e in events(tmp_path / "a" / name) if "loss" in e]
    assert sum(e["weight_mean"] for e in steps) / len(steps) == pytest.approx(a["mean"], rel=1e-6)


def test_weighted_arms_refuse_a_log_without_propensities_and_the_others_train_on_it(setup, tmp_path):
    bare = setup.root / "v3_log_bare" / "log.jsonl"
    make_log(bare, setup.records, 0, setup.data_sha)
    rows = [json.loads(line) for line in bare.read_text().splitlines()]
    bare.write_text("".join(json.dumps({k: v for k, v in row.items() if k != "propensity"}) + "\n" for row in rows))
    for arm in rl.PROPENSITY_ARMS:
        with pytest.raises(SystemExit, match="propensity"):
            train(setup, tmp_path, f"arm={arm}", log_path=bare)
    for arm in ("direct_log", "positive_sft"):
        train(setup, tmp_path, f"arm={arm}", "--limit-steps", "1", log_path=bare)


def test_fixed_noise_is_allowed_for_the_off_policy_arms_and_soft_noise_is_not(setup, tmp_path):
    for arm in rl.OFF_POLICY_ARMS:
        train(setup, tmp_path, f"arm={arm}", "--noise", "fixed", "--limit-steps", "1")
        assert (tmp_path / f"v3_06b_{arm}_n{N}_s0_noisy" / "config.yaml").is_file()
        with pytest.raises(SystemExit, match="soft"):
            train(setup, tmp_path, f"arm={arm}", "--noise", "soft")


# Sibling logs at other flip rates (the noise-rate sweep)


def test_run_names_carry_a_flip_rate_other_than_0_2():
    base = {"size": "06b", "arm": "direct_brier", "seed": 0}
    assert rl.log_run_name(base, 5000, "fixed", 0, 0.1) == "v3_06b_direct_brier_n5000_s0_noisy_flip0.1"
    assert rl.log_run_name(base, 5000, "fixed", 0, 0.3) == "v3_06b_direct_brier_n5000_s0_noisy_flip0.3"
    assert rl.log_run_name(base, 5000, "soft", 1, 0.3) == "v3_06b_direct_brier_n5000_s0_noisy_soft_flip0.3_log1"
    assert rl.log_run_name(base, 5000, "fixed", 0, 0.2) == rl.log_run_name(base, 5000, "fixed", 0) == "v3_06b_direct_brier_n5000_s0_noisy"
    assert rl.log_run_name(base, 5000, None, 0, 0.1) == "v3_06b_direct_brier_n5000_s0"


def test_a_sibling_log_trains_on_its_own_flips_and_refuses_a_clean_run(setup, tmp_path, monkeypatch):
    derive = load_script("derive_flip_log")
    logs = tmp_path / "logs"
    source = logs / "v3_log_s0"
    source.mkdir(parents=True)
    digest = write_log(source / "log.jsonl", setup.log)
    (source / "metrics.json").write_text(json.dumps({"sampling": {"seed": 0, "flip_rate": 0.2}, "data_files_sha256": {"v3_banking77_train.jsonl": setup.data_sha}, "log_sha256": digest}))
    assert derive.main(["--log-dir", str(source), "--flip-rate", "0.3"]) == 0
    sibling = logs / "v3_log_s0_flip0.3" / "log.jsonl"
    flips = [i.flipped_outcome for i in rl.read_log(sibling)]
    assert flips != [i.flipped_outcome for i in setup.log]  # some interaction flips at 0.3 that did not at 0.2
    seen = []
    original = rl.log_examples

    def spy(interactions, *args, **kwargs):
        examples = original(interactions, *args, **kwargs)
        seen.append(([i.index for i in interactions], [e.outcome for e in examples]))
        return examples

    monkeypatch.setattr(rl, "log_examples", spy)
    runs = tmp_path / "runs"
    train(setup, runs, "arm=direct_brier", "--noisy", "--limit-steps", "2", log_path=sibling)
    run = runs / f"v3_06b_direct_brier_n{N}_s0_noisy_flip0.3"
    config = yaml.safe_load((run / "config.yaml").read_text())
    summary = json.loads((run / "train_summary.json").read_text())
    assert config["v3"]["flip_rate"] == 0.3 and summary["flip_rate"] == 0.3 and summary["log_sha256"] == file_sha256(sibling)
    sibling_flips = {i.index: i.flipped_outcome for i in rl.read_log(sibling)}
    assert len(seen) == 2  # the training prefix, then the selection interactions
    for indices, outcomes in seen:
        assert outcomes == [sibling_flips[k] for k in indices]  # trained and selected on the sibling's flips
    with pytest.raises(SystemExit, match="differs from v3_log_s0 only in its flipped outcomes"):
        train(setup, runs, "arm=direct_brier", "--limit-steps", "1", log_path=sibling)


# 1.7B: the config and the logging-policy check


def test_the_1_7b_config_differs_from_0_6b_only_in_backbone_size_start_and_checkpointing():
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    small, large = flat(load_config(REPO / "configs" / "v3_06b.yaml")), flat(load_config(REPO / "configs" / "v3_17b.yaml"))
    differ = {k for k in small.keys() | large.keys() if small.get(k) != large.get(k)}
    assert differ == {"size", "backbone.id", "backbone.revision", "v3.init", "training.gradient_checkpointing"}
    sft = flat(load_config(REPO / "runs" / "sft_17b" / "config.yaml"))
    assert large["v3.init"] == "runs/sft_17b" and large["training.gradient_checkpointing"] is True
    for key in ("backbone.id", "backbone.revision", "tokenizer_reference.id", "tokenizer_reference.revision", "lora.r", "lora.alpha", "lora.target_modules", "max_tokens"):
        assert large[key] == sft[key], key


def test_the_emotion_config_differs_from_banking77_only_in_domain_and_data_file():
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    banking, emotion = flat(load_config(REPO / "configs" / "v3_06b.yaml")), flat(load_config(REPO / "configs" / "v3_emotion_06b.yaml"))
    assert {k for k in banking.keys() | emotion.keys() if banking.get(k) != emotion.get(k)} == {"domain", "training.data_file"}
    assert emotion["domain"] == "emotion" and emotion["training.data_file"] == load_config(REPO / "configs" / "v3_data.yaml")["emotion"]["files"]["train"]
    base = {"size": "06b", "arm": "direct_brier", "seed": 0}
    assert rl.log_run_name({**base, "domain": "emotion"}, 5000, None, 0) == "v3_emotion_06b_direct_brier_n5000_s0"
    assert rl.log_run_name({**base, "domain": "emotion", "arm": "direct_log", "seed": 2}, 5000, None, 0) == "v3_emotion_06b_direct_log_n5000_s2"
    assert rl.log_run_name(base, 5000, None, 0) == rl.log_run_name({**base, "domain": "banking77"}, 5000, None, 0) == "v3_06b_direct_brier_n5000_s0"


def test_a_log_drawn_from_another_data_file_is_refused(setup, tmp_path):
    logs = tmp_path / "logs" / "v3_log_emotion_s0"
    logs.mkdir(parents=True)
    digest = write_log(logs / "log.jsonl", setup.log)
    (logs / "metrics.json").write_text(json.dumps({"sampling": {"seed": 0}, "data_files_sha256": {"v3_emotion_train.jsonl": setup.data_sha}, "log_sha256": digest}))
    with pytest.raises(SystemExit, match="use the config of the log's domain"):
        train(setup, tmp_path / "runs", "arm=direct_brier", "--limit-steps", "1", log_path=logs / "log.jsonl")
    assert not (tmp_path / "runs" / f"v3_06b_direct_brier_n{N}_s0" / "train_summary.json").exists()


def test_a_log_drawn_by_another_adapter_is_refused(setup, tmp_path):
    logs = tmp_path / "logs" / "v3_log_17b_s0"
    logs.mkdir(parents=True)
    digest = write_log(logs / "log.jsonl", setup.log)
    metrics = {"sampling": {"seed": 0}, "data_files_sha256": {"v3_banking77_train.jsonl": setup.data_sha}, "log_sha256": digest,
               "logging_policy": {"adapter_sha256": "0" * 64}}
    (logs / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(SystemExit, match="a v3 learner adapts the model that logged its feedback"):
        train(setup, tmp_path / "runs", "arm=direct_brier", "--limit-steps", "1", log_path=logs / "log.jsonl")
    metrics["logging_policy"]["adapter_sha256"] = rl.adapter_sha256(setup.init / "adapter")
    (logs / "metrics.json").write_text(json.dumps(metrics))
    train(setup, tmp_path / "runs", "arm=direct_brier", "--limit-steps", "1", log_path=logs / "log.jsonl")
    assert (tmp_path / "runs" / f"v3_06b_direct_brier_n{N}_s0" / "train_summary.json").is_file()

