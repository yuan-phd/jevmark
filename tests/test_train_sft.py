"""scripts/train_sft.py on the tiny model, CPU."""

import copy
import importlib.util
import json
import random
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
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_sft", REPO / "scripts" / "train_sft.py")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

TRAINING = {
    "data_dir": "data",
    "max_tokens": 1024,
    "epochs": 2,
    "lr": 2.0e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "effective_batch": 4,
    "micro_batch": 2,
    "grad_clip": 1.0,
    "eval_every": 5,
    "valid_subset": 8,
    "gradient_checkpointing": False,
}


@pytest.fixture(scope="module")
def records():
    built = build_all(load_config(REPO / "configs" / "data.yaml"), ["train", "valid"])
    train_split = built.splits["train"]
    clinc = [r for r in train_split if r["source"].startswith("clinc")][:24]
    sst5 = [r for r in train_split if r["source"] == "SetFit/sst5"][:8]
    return clinc + sst5, built.splits["valid"][:12]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, tiny_model, tokenizer, base_config, records):
    root = tmp_path_factory.mktemp("train")
    backbone = root / "backbone"
    tiny_model.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    data_dir = root / "data"
    data_dir.mkdir()
    train_records, valid_records = records
    for name, rows in (("train", train_records), ("valid", valid_records)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = copy.deepcopy(base_config)
    config["backbone"] = {"id": str(backbone), "revision": None}
    config["run_name"] = "tiny_sft"
    config["training"] = dict(TRAINING)
    config_path = root / "tiny_sft.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return SimpleNamespace(root=root, config_path=config_path, data_dir=data_dir)


def run_training(setup, runs_dir, *extra):
    code = train.main(["--config", str(setup.config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", *extra])
    assert code == 0
    return runs_dir / "tiny_sft"


def log_of(run_dir):
    return [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines()]


def adapter_tensors(path):
    return load_file(str(path / "adapter_model.safetensors"))


# Shuffle and gold remap


def test_prepare_reshuffles_options_and_remaps_gold(records):
    rng = random.Random(0)
    changed = 0
    for record in records[0]:
        request, targets = train.prepare(record, rng)
        stored = Request.from_dict({"state": record["state"], "questions": record["questions"]})
        for question, stored_question, target in zip(request.questions, stored.questions, targets):
            gold = record["gold"][question.id]
            if question.type == "choice":
                assert question.labels[target] == gold
                assert sorted(question.options) == sorted(stored_question.options)
                changed += question.labels != stored_question.labels
            elif question.type == "noul":
                assert target == (0 if gold == "true" else 1) and question == stored_question
            else:
                assert target == gold and question == stored_question
    assert changed > 0


def test_epochs_have_different_orders_and_are_reproducible(records, tokenizer):
    jev = SimpleNamespace(tokenizer=tokenizer)
    first, _ = train.epoch_examples(records[0], jev, 1024, seed=0, epoch=0)
    again, _ = train.epoch_examples(records[0], jev, 1024, seed=0, epoch=0)
    second, _ = train.epoch_examples(records[0], jev, 1024, seed=0, epoch=1)
    assert [e.encoded.input_ids for e in first] == [e.encoded.input_ids for e in again]
    assert [e.encoded.input_ids for e in first] != [e.encoded.input_ids for e in second]
    assert len(first) == len(second) == len(records[0])


# End to end


def test_twelve_steps_end_to_end(setup):
    run_dir = run_training(setup, setup.root / "runs_12", "--limit-steps", "12")
    log = log_of(run_dir)
    steps = [e for e in log if "loss" in e]
    assert [e["step"] for e in steps] == list(range(1, 13))
    assert [e["epoch"] for e in steps] == [0] * 8 + [1] * 4  # 32 records / micro 2 / accumulation 2 = 8 steps per epoch
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in steps)
    assert steps[0]["lr"] > 0 and steps[-1]["lr"] < steps[1]["lr"]  # warmup, then linear decay
    assert [e["step"] for e in log if e.get("event") == "valid"] == [5, 10, 12]
    start = next(e for e in log if e.get("event") == "start")
    assert start["total_steps"] == 16 and start["accumulation"] == 2 and start["fp32_fallback_used"] is False
    final = next(e for e in log if e.get("event") == "final_valid")
    assert final["n"] > 0
    for name in ("config.yaml", "model_id.txt", "calibration.json", "train_summary.json", "adapter/adapter_model.safetensors", "last/state.pt"):
        assert (run_dir / name).exists(), name
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["training"]["micro_batch"] == 2 and saved["backbone"]["id"] and saved["tokenizer_reference"] and saved["lora"]["r"] == 16
    summary = json.loads((run_dir / "train_summary.json").read_text())
    assert summary["steps"] == 12 and summary["best_step"] in (5, 10, 12)


def test_trained_adapter_loads_for_evaluation(setup):
    from jevmark.model import JevMark

    run_dir = setup.root / "runs_12" / "tiny_sft"
    jev = JevMark.load(load_config(run_dir / "config.yaml"), checkpoint=run_dir, device="cpu")
    assert jev.model_id == "jevmark-tiny_sft" and jev.temperature == 1.0


def test_existing_run_is_not_overwritten(setup):
    with pytest.raises(SystemExit, match="--resume"):
        run_training(setup, setup.root / "runs_12", "--limit-steps", "2")


def test_loss_decreases_on_four_records(setup, tmp_path):
    # Four SST-5 records: score levels keep a fixed order, so the optimizer and loss
    # can be checked in isolation. Choice records are reshuffled every epoch, and a
    # random 2-layer model cannot learn which label text goes with which letter in
    # 30 steps (measured: 1.27 to 1.13); that learning is what the Kaggle run measures.
    config = load_config(setup.config_path)
    config["training"].update({"epochs": 30, "lr": 5.0e-3, "warmup_ratio": 0.0, "micro_batch": 4, "effective_batch": 4, "eval_every": 100})
    config["lora"]["dropout"] = 0.0
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [r for r in (json.loads(line) for line in (setup.data_dir / "train.jsonl").read_text().splitlines()) if r["source"] == "SetFit/sst5"][:4]
    assert len(rows) == 4
    (data_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy(setup.data_dir / "valid.jsonl", data_dir / "valid.jsonl")
    config_path = tmp_path / "four.yaml"
    config_path.write_text(yaml.safe_dump(config))
    code = train.main(["--config", str(config_path), "--data-dir", str(data_dir), "--runs-dir", str(tmp_path / "runs"), "--device", "cpu"])
    assert code == 0
    losses = [e["loss"] for e in log_of(tmp_path / "runs" / "tiny_sft") if "loss" in e]
    assert len(losses) == 30
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < 0.6 * first, (first, last)  # measured 1.24 to 0.65


def test_save_at_step_6_and_resume_matches_an_uninterrupted_run(setup, tmp_path):
    uninterrupted = setup.root / "runs_12" / "tiny_sft"
    resumed_runs = tmp_path / "runs"
    run_dir = run_training(setup, resumed_runs, "--limit-steps", "6")
    snapshot = tmp_path / "last_at_6"
    shutil.copytree(run_dir / "last", snapshot)
    run_training(setup, resumed_runs, "--resume", "--limit-steps", "12")

    log = log_of(run_dir)
    assert [e["step"] for e in log if "loss" in e] == list(range(1, 13))
    assert any(e.get("event") == "resume" and e["step"] == 6 for e in log)
    assert json.loads((run_dir / "train_summary.json").read_text())["steps"] == 12

    state = torch.load(snapshot / "state.pt", weights_only=False)
    assert state["progress"]["step"] == 6 and state["scheduler"]["last_epoch"] == 6
    assert state["optimizer"]["state"] and all(int(s["step"]) == 6 for s in state["optimizer"]["state"].values())

    a, b = adapter_tensors(uninterrupted / "last" / "adapter"), adapter_tensors(run_dir / "last" / "adapter")
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)


def test_load_state_restores_optimizer_and_scheduler(setup, tmp_path):
    from jevmark.model import JevMark

    run_dir = run_training(setup, tmp_path / "runs", "--limit-steps", "6")
    config = load_config(run_dir / "config.yaml")
    jev = JevMark.load(config, device="cpu")
    train.attach_lora(jev, config)
    params = [p for p in jev.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scheduler = train.get_linear_schedule_with_warmup(optimizer, 1, 16)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = train.load_state(run_dir, jev, optimizer, scheduler, scaler)
    assert progress["step"] == 6 and scheduler.last_epoch == 6
    assert len(optimizer.state) == len(params)
    saved = adapter_tensors(run_dir / "last" / "adapter")
    current = {k: v for k, v in jev.model.state_dict().items() if "lora_" in k}
    assert any(not torch.all(v == 0) for k, v in saved.items() if "lora_B" in k)
    assert sum(1 for k in current if "lora_B" in k) == sum(1 for k in saved if "lora_B" in k)


def test_max_hours_saves_last_and_exits_cleanly(setup, tmp_path):
    run_dir = run_training(setup, tmp_path / "runs", "--max-hours", "0")
    log = log_of(run_dir)
    assert [e["step"] for e in log if "loss" in e] == [1]
    assert log[-1]["event"] == "time_limit"
    assert (run_dir / "last" / "state.pt").exists() and not (run_dir / "train_summary.json").exists()


def test_gradient_checkpointing_trains(setup, tmp_path):
    config = load_config(setup.config_path)
    config["training"]["gradient_checkpointing"] = True
    config_path = tmp_path / "gc.yaml"
    config_path.write_text(yaml.safe_dump(config))
    code = train.main(["--config", str(config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(tmp_path / "runs"), "--device", "cpu", "--limit-steps", "2"])
    assert code == 0
    losses = [e["loss"] for e in log_of(tmp_path / "runs" / "tiny_sft") if "loss" in e]
    assert len(losses) == 2 and all(loss == loss for loss in losses)


# Pre-flight check (decision 45)


def test_preflight_runs_the_longest_records_and_is_logged(setup, tmp_path):
    run_dir = run_training(setup, tmp_path / "runs", "--limit-steps", "2")
    events = [e for e in log_of(run_dir) if e.get("event") == "preflight"]
    assert len(events) == 1 and events[0]["passed"] is True
    check = events[0]
    assert check["records"] == TRAINING["micro_batch"] and check["device"] == "cpu" and "peak_gib" not in check
    # The worst case is the longest records of the training set.
    from jevmark.encode import encode

    train_records = [json.loads(l) for l in (setup.data_dir / "train.jsonl").read_text().splitlines()]
    tokenizer = train.JevMark.load(yaml.safe_load(setup.config_path.read_text()), device="cpu").tokenizer
    longest = max(len(encode(Request.from_dict({"state": r["state"], "questions": r["questions"]}), tokenizer, 1024).input_ids) for r in train_records)
    assert abs(check["longest_tokens"] - longest) <= 1  # the reshuffled order can differ by a token
    log = log_of(run_dir)
    assert [e.get("event") for e in log[:2]] == ["start", "preflight"]


def test_preflight_out_of_memory_exits_nonzero_before_any_step(setup, tmp_path, monkeypatch):
    real = train.batch_loss
    calls = []

    def exploding(jev, batch):
        calls.append(len(batch))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return real(jev, batch)

    monkeypatch.setattr(train, "batch_loss", exploding)
    runs_dir = tmp_path / "runs"
    with pytest.raises(SystemExit, match="PREFLIGHT FAIL") as excinfo:
        train.main(["--config", str(setup.config_path), "--data-dir", str(setup.data_dir), "--runs-dir", str(runs_dir), "--device", "cpu"])
    assert excinfo.value.code != 0
    log = log_of(runs_dir / "tiny_sft")
    assert log[-1]["event"] == "preflight" and log[-1]["passed"] is False and "out of memory" in log[-1]["error"]
    assert not any("step" in e and "event" not in e for e in log)  # no training step ran
    assert calls == [TRAINING["micro_batch"]]


def test_preflight_leaves_no_gradients_and_the_same_random_state(setup):
    jev = train.JevMark.load(yaml.safe_load(setup.config_path.read_text()), device="cpu")
    config = yaml.safe_load(setup.config_path.read_text())
    train.attach_lora(jev, config)
    records = [json.loads(l) for l in (setup.data_dir / "train.jsonl").read_text().splitlines()]
    records, _, lengths = train.fits(records, jev, 1024)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    report = train.preflight(jev, records, lengths, 2, 1024, 0, torch.amp.GradScaler("cuda", enabled=False))
    assert torch.equal(torch.get_rng_state(), before)
    assert all(p.grad is None for p in jev.model.parameters())
    assert report["records"] == 2 and report["longest_tokens"] >= report["shortest_tokens"] == sorted(lengths)[-2]
