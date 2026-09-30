"""v3 data and the deployment-feedback log (tasks 3.1 and 3.2, decision 56)."""

import copy
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from jevmark.config import load_config
from jevmark.data import dedup, unseen
from jevmark.data.description_loader import load_descriptions
from jevmark.encode import encode
from jevmark.feedback import (
    FlipChannel,
    Interaction,
    behaviour,
    bernoulli_log_likelihood,
    file_sha256,
    interactions,
    prefix,
    read_log,
    sample_action,
    write_log,
)
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "configs" / "data.yaml")
NAMES = sorted(load_descriptions("banking77"))
DESCRIPTIONS = load_descriptions("banking77")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(n, offset=0):
    """Synthetic Banking77 rows: (source index, text, gold label)."""
    return [(offset + i, f"message number {offset + i} about my card", NAMES[(offset + i) % len(NAMES)]) for i in range(n)]


# 3.1 data


def test_dedupe_drops_test_texts_and_later_repeats_and_records_every_drop():
    texts = ["Where is my card?", "where is my CARD", "top up failed", "Lost my phone.", "declined", "Top-up failed!"]
    kept, dropped = dedup.dedupe(texts, frozenset({dedup.normalise("lost my phone")}))
    assert kept == [0, 2, 4]
    assert dropped == [{"index": 1, "reason": "repeat", "first": 0}, {"index": 3, "reason": "test_text"}, {"index": 5, "reason": "repeat", "first": 2}]


def test_dedupe_prefers_the_test_reason_and_keeps_the_first_copy_that_survives():
    kept, dropped = dedup.dedupe(["x", "x", "y"], frozenset({"x"}))
    assert kept == [2] and [d["reason"] for d in dropped] == ["test_text", "test_text"]


def test_banking77_records_are_valid_encode_and_follow_the_test_construction(tokenizer):
    records = unseen.banking77_records(unseen.V3_TRAIN, rows(50), "train", NAMES, DESCRIPTIONS, CONFIG)
    assert len(records) == 50 and len({r["id"] for r in records}) == 50
    assert sum(isinstance(r["state"], dict) for r in records) == 10  # the 20 percent JSON-state share
    for record in records:
        question = record["questions"][unseen.QUESTION_ID]
        gold = record["gold"][unseen.QUESTION_ID]
        assert question["type"] == "choice" and question["instructions"] == unseen.SETS["test_banking77"].instructions
        assert len(question["criteria"]) == 10 and gold in question["criteria"] and unseen.OTHER_LABEL in question["criteria"] and gold != unseen.OTHER_LABEL
        assert question["criteria"][gold] == DESCRIPTIONS[gold].canonical
        assert record["meta"]["source_split"] == "train" and record["split"] == unseen.V3_TRAIN
        encoded = encode(Request.from_dict({"state": record["state"], "questions": record["questions"]}), tokenizer, 1024)
        assert len(encoded.slot_positions) == 1
    again = unseen.banking77_records(unseen.V3_TRAIN, rows(50), "train", NAMES, DESCRIPTIONS, CONFIG)
    assert again == records  # seeded
    other_stream = unseen.banking77_records(unseen.V3_TEST_FULL, rows(50), "test", NAMES, DESCRIPTIONS, CONFIG)
    assert [r["questions"] for r in other_stream] != [r["questions"] for r in records]  # each file has its own stream


def test_full_test_keeps_the_given_records_verbatim_first_and_adds_every_unused_row():
    given = unseen.banking77_records("test_banking77", [rows(30)[i] for i in range(0, 30, 3)], "test", NAMES, DESCRIPTIONS, CONFIG)
    full = unseen.full_test_records(given, rows(30), NAMES, DESCRIPTIONS, CONFIG)
    assert full[: len(given)] == given and len(full) == 30
    assert sorted(r["meta"]["source_index"] for r in full) == list(range(30))
    new = full[len(given) :]
    assert all(r["split"] == unseen.V3_TEST_FULL for r in new) and sum(isinstance(r["state"], dict) for r in new) == 4  # 20 percent of 20
    assert len({r["id"] for r in full}) == 30


def test_the_v3_hashes_are_recorded_and_the_files_have_their_own_names():
    v3 = load_config(REPO / "configs" / "v3_data.yaml")
    assert set(v3["sha256"]) == {v3["files"]["train"], v3["files"]["test_full"]}
    assert all(isinstance(h, str) and len(h) == 64 for h in v3["sha256"].values())
    assert not {f"{s}.jsonl" for s in unseen.SETS} & set(v3["sha256"])  # never a v1.3 file name


# 3.2 feedback log


def fake_log(n):
    return [Interaction(i, f"r{i}", "label", ("a", "b"), (0.7, 0.3), 0, 0.68, i % 2, i % 2, "m", "c") for i in range(n)]


@pytest.mark.parametrize("n, n_train, n_select", [(500, 450, 50), (2000, 1800, 200), (5000, 4500, 500)])
def test_prefix_splits_the_first_n_into_training_and_selection(n, n_train, n_select):
    log = fake_log(6000)
    split = prefix(log, n)
    assert len(split.train) == n_train and len(split.select) == n_select
    assert split.train[0].index == 0 and split.select[-1].index == n - 1 and split.select[0].index == n_train


def test_prefix_refuses_a_log_shorter_than_n():
    with pytest.raises(ValueError):
        prefix(fake_log(100), 500)


def test_propensity_is_the_mixture_probability_and_actions_follow_it():
    p = [0.6, 0.25, 0.1, 0.05]
    q = behaviour(p, 0.1)
    assert q == pytest.approx([0.9 * x + 0.025 for x in p]) and math.fsum(q) == pytest.approx(1.0)
    rng = random.Random(0)
    draws = [sample_action(p, rng, 0.1) for _ in range(40000)]
    assert all(prop == pytest.approx(q[a]) for a, prop in draws)
    counts = Counter(a for a, _ in draws)
    for action, mass in enumerate(q):
        sd = math.sqrt(mass * (1 - mass) / 40000)
        assert abs(counts[action] / 40000 - mass) < 4 * sd


def test_flip_rate_is_about_0_2_and_seeded():
    flips = FlipChannel(0)
    n = 20000
    changed = sum(flips(i % 2) != i % 2 for i in range(n))
    assert abs(changed / n - 0.2) < 4 * math.sqrt(0.2 * 0.8 / n)
    a, b = FlipChannel(3), FlipChannel(3)
    assert [a(1) for _ in range(50)] == [b(1) for _ in range(50)]


def test_interactions_reveal_only_the_chosen_option_and_are_seeded():
    data = [(f"r{i}", "label", ["x", "y", "z"], [0.5, 0.3, 0.2], "y") for i in range(300)]
    log = interactions(data, seed=0, model="m", commit="c")
    assert [i.index for i in log] == list(range(300))
    assert all(i.outcome == int(i.labels[i.action] == "y") for i in log)
    assert all(i.propensity == pytest.approx(behaviour([0.5, 0.3, 0.2])[i.action]) for i in log)
    assert interactions(data, seed=0, model="m", commit="c") == log
    assert interactions(data, seed=1, model="m", commit="c") != log


def test_bernoulli_log_likelihood_by_hand():
    value = bernoulli_log_likelihood([0.8, 0.3, 1.0], [1, 0, 0])
    assert value == pytest.approx((math.log(0.8) + math.log(0.7) + math.log(1e-6)) / 3)
    with pytest.raises(ValueError):
        bernoulli_log_likelihood([0.5], [])


def test_log_round_trips_and_its_sha256_is_the_file_hash(tmp_path):
    log = interactions([("r0", "label", ["x", "y"], [0.9, 0.1], "x")], seed=0, model="m", commit="c")
    path = tmp_path / "log.jsonl"
    digest = write_log(path, log)
    assert digest == file_sha256(path) and read_log(path) == log


# collect_log.py on the tiny model


@pytest.fixture(scope="module")
def tiny_logging_policy(tmp_path_factory, tiny_model, tokenizer, base_config):
    import torch
    from peft import LoraConfig, get_peft_model

    root = tmp_path_factory.mktemp("v3_log")
    backbone = root / "backbone"
    tiny_model.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    config = copy.deepcopy(base_config)
    config["backbone"] = {"id": str(backbone), "revision": None}
    config["run_name"] = "tiny_policy"
    run_dir = root / "tiny_policy"
    model = get_peft_model(copy.deepcopy(tiny_model), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        generator = torch.Generator().manual_seed(0)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
    model.save_pretrained(run_dir / "adapter")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    data_dir = root / "data"
    data_dir.mkdir()
    records = unseen.banking77_records(unseen.V3_TRAIN, rows(40), "train", NAMES, DESCRIPTIONS, CONFIG)
    (data_dir / "v3_banking77_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    v3_config = root / "v3_data.yaml"
    v3_config.write_text(yaml.safe_dump({"files": {"train": "v3_banking77_train.jsonl"}, "sha256": {"v3_banking77_train.jsonl": file_sha256(data_dir / "v3_banking77_train.jsonl")}}))
    return root, run_dir, data_dir, v3_config, records


def test_collect_log_writes_the_log_and_its_metrics(tiny_logging_policy, tmp_path):
    collect = load_script("collect_log")
    _, run_dir, data_dir, v3_config, records = tiny_logging_policy
    argv = ["--ckpt", str(run_dir), "--seed", "0", "--data-dir", str(data_dir), "--runs-dir", str(tmp_path), "--device", "cpu", "--batch-size", "8", "--v3-config", str(v3_config)]
    assert collect.main(argv) == 0
    out = tmp_path / "v3_log_s0"
    log = read_log(out / "log.jsonl")
    metrics = json.loads((out / "metrics.json").read_text())
    assert len(log) == 40 and metrics["n"] == 40 and metrics["log_sha256"] == file_sha256(out / "log.jsonl")
    order = list(range(40))
    random.Random("v3_log_order:0").shuffle(order)
    assert [i.record_id for i in log] == [records[j]["id"] for j in order]  # the seeded message order
    gold = {r["id"]: r["gold"]["label"] for r in records}
    for interaction in log:
        assert interaction.propensity == pytest.approx(behaviour(interaction.probs)[interaction.action])
        assert interaction.outcome == int(interaction.labels[interaction.action] == gold[interaction.record_id])
        assert sum(interaction.probs) == pytest.approx(1.0) and len(interaction.labels) == 10
    assert metrics["outcome_rate"] == pytest.approx(sum(i.outcome for i in log) / 40)
    assert set(metrics["git"]) == {"commit", "dirty"} and metrics["data_files_sha256"]["v3_banking77_train.jsonl"] == file_sha256(data_dir / "v3_banking77_train.jsonl")
    assert "gold" not in json.loads((out / "log.jsonl").read_text().splitlines()[0])
    assert collect.main(argv) == 1  # never overwritten
    other_seed = collect.main([*argv[:2], "--seed", "1", *argv[4:]])
    assert other_seed == 0 and [i.record_id for i in read_log(tmp_path / "v3_log_s1" / "log.jsonl")] == [i.record_id for i in log]  # same order for every log seed


def test_collect_log_refuses_a_data_file_with_another_hash(tiny_logging_policy, tmp_path):
    collect = load_script("collect_log")
    root, run_dir, data_dir, _, _ = tiny_logging_policy
    wrong = root / "wrong.yaml"
    wrong.write_text(yaml.safe_dump({"files": {"train": "v3_banking77_train.jsonl"}, "sha256": {"v3_banking77_train.jsonl": "0" * 64}}))
    argv = ["--ckpt", str(run_dir), "--data-dir", str(data_dir), "--runs-dir", str(tmp_path), "--device", "cpu", "--v3-config", str(wrong)]
    assert collect.main(argv) == 1 and not (tmp_path / "v3_log_s0").exists()
