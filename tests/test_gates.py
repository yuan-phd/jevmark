"""CPU data gates (decision 42): scripts/leak_probe.py and scripts/check_duplicates.py on small synthetic records."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from jevmark.data.dedup import normalise
from jevmark.data.negation import render

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


leak_probe = load_script("leak_probe")
check_duplicates = load_script("check_duplicates")


def sentiment_record(i, label, noul_first, with_noul, negated=False):
    """An SST-5 style record: a score question and, optionally, a sentiment noul before or after it."""
    score = {"type": "score", "instructions": "How positive is the sentiment of this text?", "criteria": ["neg", "neutral", "pos"]}
    questions, gold, nouls = {"sentiment": score}, {"sentiment": label}, {}
    if with_noul:
        underlying = label == 2
        noul = {"type": "noul", "instructions": render("is_positive", 0, negated)}
        questions = {"is_positive": noul, "sentiment": score} if noul_first else {"sentiment": score, "is_positive": noul}
        gold = {"is_positive": "true" if underlying != negated else "false", "sentiment": label}
        nouls = {"is_positive": {"template": 0, "negated": negated, "slot": None}}
    return {"id": f"r{i}", "state": "text", "questions": questions, "gold": {q: gold[q] for q in questions}, "meta": {"scale": "s3", "nouls": nouls}}


def test_structure_probe_detects_a_noul_that_reveals_the_score():
    # The v1.2 SST-5 leak in miniature: neutral records have no noul, so a noul before the score means "not neutral".
    rng = random.Random(0)
    records = []
    for i in range(600):
        label = rng.choice([0, 1, 1, 1, 2])
        records.append(sentiment_record(i, label, noul_first=rng.random() < 0.5, with_noul=label != 1, negated=rng.random() < 0.5))
    report = leak_probe.probe_split(records, folds=5, min_n=50, permutations=20)
    structure = report["score/sentiment/s3"]["logistic"]["structure"]
    assert structure["lift"] > 0.10
    assert structure["gate"] == {"null_percentile": None, "p_value": None, "failed": True, "reason": "above the hard limit"}  # no permutation test


def test_later_questions_are_invisible_to_the_probe():
    # The same records with the noul always after the score: causal attention hides it, so no probe may use it.
    rng = random.Random(0)
    records = []
    for i in range(600):
        label = rng.choice([0, 1, 1, 1, 2])
        records.append(sentiment_record(i, label, noul_first=False, with_noul=label != 1, negated=rng.random() < 0.5))
    report = leak_probe.probe_split(records, folds=5, min_n=50, permutations=20)
    assert report["score/sentiment/s3"]["max_lift"] < 0.03  # both models


def test_noul_probes_use_the_underlying_answer_and_phrasing_the_stored_one():
    rng = random.Random(1)
    records = [sentiment_record(i, rng.choice([0, 2]), noul_first=False, with_noul=True, negated=i % 2 == 0) for i in range(400)]
    rows = [row for r in records for row in leak_probe.Features().rows(r) if row.group == "noul/is_positive"]
    for record, row in zip(records, rows):
        assert row.stored == record["gold"]["is_positive"]
        assert row.underlying == ("true" if record["gold"]["sentiment"] == 2 else "false")


def test_gate_by_hand():
    null = [i / 1000 for i in range(200)]  # lifts 0.000 to 0.199
    assert leak_probe.gate(0.12, null, 0.03, 0.10, 0.99)["failed"]  # above the hard limit
    assert leak_probe.gate(0.12, null, 0.03, 0.10, 0.99)["reason"] == "above the hard limit"
    passing = leak_probe.gate(0.05, null, 0.03, 0.10, 0.99)
    assert not passing["failed"] and passing["null_percentile"] == pytest.approx(0.197)
    assert passing["p_value"] == pytest.approx((1 + 150) / 201)
    small_null = [0.0] * 200
    assert leak_probe.gate(0.05, small_null, 0.03, 0.10, 0.99) == {"null_percentile": 0.0, "p_value": 1 / 201, "failed": True, "reason": "above chance"}


def test_probe_flags_only_lifts_over_three_points():
    rng = random.Random(2)
    records = [sentiment_record(i, rng.choice([0, 2]), noul_first=False, with_noul=True, negated=rng.random() < 0.5) for i in range(300)]
    report = leak_probe.probe_split(records, folds=5, min_n=50, permutations=20)
    for entry in report.values():
        for model in leak_probe.MODELS:
            for probe in leak_probe.PROBES:
                lift = entry[model][probe]["lift"]
                assert ("gate" in entry[model][probe]) == (lift is not None and lift > 0.03)


def test_boosting_probe_finds_an_interaction_the_linear_probe_cannot():
    # The answer is the exclusive or of two features: no linear function of them beats chance.
    rng = random.Random(3)
    features, targets = [], []
    for _ in range(800):
        a, b = rng.random() < 0.5, rng.random() < 0.5
        features.append({"bias": 1.0, f"a={a}": 1.0, f"b={b}": 1.0})
        targets.append(a != b)
    base = leak_probe.majority(targets)
    assert leak_probe.probe_accuracy(features, targets, 5, model="logistic") - base < 0.05
    assert leak_probe.probe_accuracy(features, targets, 5, model="boosting") - base > 0.4


# Duplicate check


def test_normalise():
    assert normalise("  What's   the TIME?\n") == "what s the time"
    assert normalise("ｆｕｌｌ width") == "full width"


def test_compare_finds_exact_matches_and_ngram_overlap():
    train = [{"id": "a", "state": "One two three four five six seven eight nine"}]
    test = [
        {"id": "t1", "state": {"text": "one two, three four five six seven eight nine!", "channel": "sms"}},  # exact after normalising
        {"id": "t2", "state": "zero one two three four five six seven eight"},  # shares an 8-gram
        {"id": "t3", "state": "short text"},
    ]
    row = check_duplicates.compare(train, test)
    assert row["exact_matches"] == 1 and row["exact_examples"][0]["id"] == "t1"
    assert row["ngram_overlap_rate"] == pytest.approx(2 / 3)
    assert row["test_records_with_an_8gram"] == 2
