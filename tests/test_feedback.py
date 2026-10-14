"""v3 data and the deployment-feedback log (tasks 3.1 and 3.2, decision 56)."""

import copy
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from dataclasses import replace
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
    log_bytes,
    prefix,
    read_log,
    reflip,
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


EMOTIONS = sorted(load_descriptions("emotion"))
EMOTION_DESCRIPTIONS = load_descriptions("emotion")


def emotion_rows(n, offset=0):
    """Synthetic emotion rows: (source index, text, gold label)."""
    return [(offset + i, f"i feel like message number {offset + i}", EMOTIONS[(offset + i) % len(EMOTIONS)]) for i in range(n)]


def test_emotion_records_ask_every_label_and_never_other_like_test_emotion(tokenizer):
    records = unseen.emotion_records(unseen.V3_EMOTION_TRAIN, emotion_rows(50), "train", EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG, nouls=False)
    assert len(records) == 50 and len({r["id"] for r in records}) == 50 and all(r["id"].startswith("emotion-v3_emotion_train-") for r in records)
    assert sum(isinstance(r["state"], dict) for r in records) == 10  # the 20 percent JSON-state share
    for record in records:
        assert list(record["questions"]) == [unseen.QUESTION_ID]  # the log domain asks the choice only
        question, gold = record["questions"][unseen.QUESTION_ID], record["gold"][unseen.QUESTION_ID]
        assert question["type"] == "choice" and question["instructions"] == unseen.SETS["test_emotion"].instructions
        assert sorted(question["criteria"]) == EMOTIONS and unseen.OTHER_LABEL not in question["criteria"] and gold in question["criteria"]
        assert question["criteria"][gold] == EMOTION_DESCRIPTIONS[gold].canonical
        assert record["source"] == "dair-ai/emotion" and record["meta"]["source_split"] == "train" and record["split"] == unseen.V3_EMOTION_TRAIN
        assert len(encode(Request.from_dict({"state": record["state"], "questions": record["questions"]}), tokenizer, 1024).slot_positions) == 1
    assert len({tuple(r["questions"][unseen.QUESTION_ID]["criteria"]) for r in records}) > 1  # the option order is shuffled per record
    assert unseen.emotion_records(unseen.V3_EMOTION_TRAIN, emotion_rows(50), "train", EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG, nouls=False) == records  # seeded
    other_stream = unseen.emotion_records(unseen.V3_EMOTION_TEST_FULL, emotion_rows(50), "test", EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG, nouls=False)
    order = lambda rs: [list(r["questions"][unseen.QUESTION_ID]["criteria"]) for r in rs]  # dict equality ignores key order
    assert order(other_stream) != order(records)  # each file has its own stream


def test_emotion_full_test_keeps_test_emotion_verbatim_first_and_asks_the_noul_like_it():
    given = unseen.emotion_records("test_emotion", [emotion_rows(40)[i] for i in range(0, 40, 2)], "test", EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG, nouls=True)
    full = unseen.emotion_full_test_records(given, emotion_rows(40), EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG)
    assert full[: len(given)] == given and len(full) == 40 and len({r["id"] for r in full}) == 40
    assert sorted(r["meta"]["source_index"] for r in full) == list(range(40))
    new = full[len(given) :]
    assert all(r["split"] == unseen.V3_EMOTION_TEST_FULL for r in new) and sum(isinstance(r["state"], dict) for r in new) == 4  # 20 percent of 20
    # Choice first, then the expresses_emotion noul, half of them asking the gold emotion, as in test_emotion.
    assert all(list(r["questions"]) == [unseen.QUESTION_ID, "expresses_emotion"] for r in new)
    asks_gold = [r["meta"]["nouls"]["expresses_emotion"]["asked_emotion"] == r["gold"][unseen.QUESTION_ID] for r in new]
    assert sum(asks_gold) == len(new) // 2
    for r, yes in zip(new, asks_gold):  # a negated phrasing flips the answer
        assert (r["gold"]["expresses_emotion"] == "true") == (yes != r["meta"]["nouls"]["expresses_emotion"]["negated"])


def test_the_v3_hashes_are_recorded_and_the_files_have_their_own_names():
    v3 = load_config(REPO / "configs" / "v3_data.yaml")
    assert set(v3["sha256"]) == {v3["files"]["train"], v3["files"]["test_full"], v3["emotion"]["files"]["train"], v3["emotion"]["files"]["test_full"]}
    assert v3["emotion"]["files"] == {"train": "v3_emotion_train.jsonl", "test_full": "v3_emotion_test_full.jsonl"}
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


def test_collect_log_names_a_log_by_its_policys_size():
    collect = load_script("collect_log")
    assert collect.log_run_name(Path("runs/sft_06b"), 0) == "v3_log_s0" and collect.log_run_name(Path("runs/sft_06b"), 1, 50) == "v3_log_s1_limit50"
    assert collect.log_run_name(Path("runs/sft_17b"), 0) == "v3_log_17b_s0" and collect.log_run_name(Path("runs/sft_17b"), 0, 50) == "v3_log_17b_s0_limit50"
    assert collect.log_run_name(Path("/tmp/tiny_policy"), 2) == "v3_log_s2"  # no size in the name: the phase 3 name
    assert collect.log_run_name(Path("runs/sft_06b"), 0, domain="emotion") == "v3_log_emotion_s0"
    assert collect.log_run_name(Path("runs/sft_17b"), 0, 50, "emotion") == "v3_log_emotion_17b_s0_limit50"


def test_collect_log_on_the_emotion_domain_reads_its_own_file_and_names(tiny_logging_policy, tmp_path):
    collect = load_script("collect_log")
    root, run_dir, data_dir, _, _ = tiny_logging_policy
    records = unseen.emotion_records(unseen.V3_EMOTION_TRAIN, emotion_rows(24), "train", EMOTIONS, EMOTION_DESCRIPTIONS, CONFIG, nouls=False)
    (data_dir / "v3_emotion_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    v3_config = root / "v3_both.yaml"
    v3_config.write_text(yaml.safe_dump({"files": {"train": "v3_banking77_train.jsonl"}, "emotion": {"files": {"train": "v3_emotion_train.jsonl"}},
                                         "sha256": {"v3_emotion_train.jsonl": file_sha256(data_dir / "v3_emotion_train.jsonl")}}))
    argv = ["--ckpt", str(run_dir), "--seed", "0", "--domain", "emotion", "--data-dir", str(data_dir), "--runs-dir", str(tmp_path), "--device", "cpu", "--batch-size", "8", "--v3-config", str(v3_config)]
    assert collect.main(argv) == 0 and not (tmp_path / "v3_log_s0").exists()
    out = tmp_path / "v3_log_emotion_s0"
    log, metrics = read_log(out / "log.jsonl"), json.loads((out / "metrics.json").read_text())
    assert len(log) == 24 and all(len(i.labels) == 6 and unseen.OTHER_LABEL not in i.labels for i in log)
    assert set(metrics["data_files_sha256"]) == {"v3_emotion_train.jsonl"} and metrics["chosen_other_rate"] == 0


# Sibling logs at other flip rates: feedback.reflip and scripts/derive_flip_log.py (the noise-rate sweep)


def synthetic_log(n=3000, seed=0, rate=0.2):
    rng = random.Random(seed)
    data = [(f"r{i}", "label", ["x", "y", "z"], [0.6, 0.3, 0.1], rng.choice("xyz")) for i in range(n)]
    return interactions(data, seed=seed, model="m", commit="c", flip_rate=rate)


def test_reflip_at_the_logs_rate_reproduces_it_and_other_rates_nest():
    log = synthetic_log()
    assert reflip(log, 0, 0.2) == log
    flipped = {rate: [i.outcome != i.flipped_outcome for i in reflip(log, 0, rate)] for rate in (0.1, 0.2, 0.3)}
    assert all(not low or high for low, high in zip(flipped[0.1], flipped[0.2]))
    assert all(not low or high for low, high in zip(flipped[0.2], flipped[0.3]))
    for rate, flips in flipped.items():
        assert abs(sum(flips) / len(flips) - rate) < 4 * math.sqrt(rate * (1 - rate) / len(flips))
    other = reflip(log, 0, 0.1)
    assert [(i.action, i.outcome, i.propensity, i.probs) for i in other] == [(i.action, i.outcome, i.propensity, i.probs) for i in log]
    assert reflip(log, 1, 0.2) != log  # another seed's stream gives other flips
    assert synthetic_log(rate=0.1) == other  # what collect_log.py would have written at that rate


def write_source(tmp_path, log, rate=0.2):
    source = tmp_path / "v3_log_s0"
    source.mkdir(parents=True)
    digest = write_log(source / "log.jsonl", log)
    (source / "metrics.json").write_text(json.dumps({"run_name": "v3_log_s0", "decision": 56, "sampling": {"seed": 0, "flip_rate": rate},
                                                     "data_files_sha256": {"v3_banking77_train.jsonl": "d" * 64}, "n": len(log), "log_sha256": digest}))
    return source


def test_derive_flip_log_writes_the_sibling_and_its_metrics(tmp_path):
    derive = load_script("derive_flip_log")
    log = synthetic_log()
    source = write_source(tmp_path, log)
    before = (source / "log.jsonl").read_bytes()
    assert derive.main(["--log-dir", str(source), "--flip-rate", "0.1"]) == 0
    out = tmp_path / "v3_log_s0_flip0.1"
    assert (out / "log.jsonl").read_bytes() == log_bytes(reflip(log, 0, 0.1))
    assert (source / "log.jsonl").read_bytes() == before
    m = json.loads((out / "metrics.json").read_text())
    assert m["log_sha256"] == file_sha256(out / "log.jsonl") and m["run_name"] == "v3_log_s0_flip0.1"
    assert m["sampling"] == {"seed": 0, "flip_rate": 0.1} and m["data_files_sha256"] == {"v3_banking77_train.jsonl": "d" * 64}
    assert m["derived_from"]["run"] == "v3_log_s0" and m["derived_from"]["log_sha256"] == file_sha256(source / "log.jsonl") and m["derived_from"]["flip_rate"] == 0.2
    derived = read_log(out / "log.jsonl")
    assert m["flip_share"] == sum(i.outcome != i.flipped_outcome for i in derived) / len(derived)
    assert m["flipped_outcome_rate"] == sum(i.flipped_outcome for i in derived) / len(derived)
    assert "decision" not in m


def test_derive_flip_log_refuses_what_it_cannot_derive(tmp_path):
    derive = load_script("derive_flip_log")
    log = synthetic_log(n=200)
    source = write_source(tmp_path / "a", log)
    for rate, match in (("0.2", "same log"), ("0.5", "flip-rate"), ("0", "flip-rate")):
        with pytest.raises(SystemExit, match=match):
            derive.main(["--log-dir", str(source), "--flip-rate", rate])
    assert derive.main(["--log-dir", str(source), "--flip-rate", "0.3"]) == 0
    with pytest.raises(SystemExit, match="never overwritten"):
        derive.main(["--log-dir", str(source), "--flip-rate", "0.3"])
    with pytest.raises(SystemExit, match="itself derived"):
        derive.main(["--log-dir", str(tmp_path / "a" / "v3_log_s0_flip0.3"), "--flip-rate", "0.1"])
    tampered = write_source(tmp_path / "b", [replace(i, flipped_outcome=1 - i.flipped_outcome) if i.index == 5 else i for i in log])
    with pytest.raises(SystemExit, match="does not reproduce"):
        derive.main(["--log-dir", str(tampered), "--flip-rate", "0.1"])
    edited = write_source(tmp_path / "c", log)
    (edited / "log.jsonl").write_bytes((edited / "log.jsonl").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="records"):
        derive.main(["--log-dir", str(edited), "--flip-rate", "0.1"])

