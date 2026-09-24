"""Every metric against values computed by hand on tiny inputs."""

import math

import pytest

from jevmark.metrics import (
    QuestionResult,
    brier,
    coverage_curve,
    ece,
    letter_bias,
    macro_f1,
    max_abs_difference,
    mean_absolute_error,
    nll,
    reliability,
    split_metrics,
    symmetry,
    timing_summary,
)


def noul(p_true, gold_true, rid="r"):
    return QuestionResult(rid, "q", "noul", (p_true, 1 - p_true), 0 if gold_true else 1, ("true", "false"))


def choice(probs, gold, labels=None, rid="r"):
    labels = labels or tuple(f"o{i}" for i in range(len(probs)))
    return QuestionResult(rid, "intent", "choice", tuple(probs), gold, tuple(labels))


def score(probs, gold, rid="r"):
    return QuestionResult(rid, "sentiment", "score", tuple(probs), gold, tuple(str(i) for i in range(len(probs))))


# Per-question fields


def test_noul_prediction_threshold_and_confidence():
    assert noul(0.5, True).correct  # P(true) >= 0.5 predicts true
    assert not noul(0.49, True).correct
    assert noul(0.2, False).correct
    assert noul(0.2, False).top1 == pytest.approx(0.8)
    assert noul(0.2, False).confidence == pytest.approx(0.8)  # max(p, 1 - p)


def test_choice_and_score_confidence_is_normalised_entropy():
    r = choice([0.5, 0.5, 0.0], 0)
    assert r.confidence == pytest.approx(1 - math.log(2) / math.log(3))
    assert r.top1 == 0.5 and r.prediction == 0  # first index wins a tie
    assert score([0.2, 0.3, 0.5], 2).expected_level == pytest.approx(1.3)


# Scalar metrics


def test_ece_by_hand():
    # Bin [0.8, 0.8667): confidences 0.8, 0.8 with accuracy 1/2 -> gap 0.3, weight 2/4.
    # Bin [0.9333, 1.0]: confidences 1.0, 1.0 with accuracy 1 -> gap 0, weight 2/4.
    assert ece([0.8, 0.8, 1.0, 1.0], [True, False, True, True]) == pytest.approx(0.15)


def test_ece_perfectly_calibrated_is_zero():
    assert ece([0.5, 0.5], [True, False]) == pytest.approx(0.0)


def test_ece_puts_confidence_one_in_the_last_bin():
    bins = reliability([1.0], [True])
    assert bins[-1]["count"] == 1 and len(bins) == 15


def test_reliability_bins_by_hand():
    bins = reliability([0.05, 0.06, 0.95], [True, False, True], n_bins=10)
    assert bins[0] == {"lower": 0.0, "upper": 0.1, "count": 2, "mean_confidence": pytest.approx(0.055), "mean_accuracy": 0.5}
    assert bins[9]["count"] == 1 and bins[9]["mean_accuracy"] == 1.0
    assert bins[5] == {"lower": 0.5, "upper": 0.6, "count": 0, "mean_confidence": None, "mean_accuracy": None}


def test_brier_by_hand():
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.14 ; noul p=0.6 gold false: 0.6^2 + 0.6^2 = 0.72
    assert brier([choice([0.7, 0.2, 0.1], 0)]) == pytest.approx(0.14)
    assert brier([noul(0.6, False)]) == pytest.approx(0.72)
    assert brier([choice([0.7, 0.2, 0.1], 0), noul(0.6, False)]) == pytest.approx(0.43)


def test_nll_by_hand_and_clipped():
    assert nll([choice([0.5, 0.25, 0.25], 1)]) == pytest.approx(math.log(4))
    assert nll([choice([1.0, 0.0], 1)]) == pytest.approx(-math.log(1e-12))


def test_macro_f1_by_hand():
    # gold a, a, b; predicted a, b, b.
    # a: tp 1, fp 0, fn 1 -> 2/3 ; b: tp 1, fp 1, fn 0 -> 2/3 ; macro 2/3
    results = [
        choice([0.9, 0.1], 0, ("a", "b")),
        choice([0.2, 0.8], 0, ("a", "b")),
        choice([0.3, 0.7], 1, ("a", "b")),
    ]
    assert macro_f1(results) == pytest.approx(2 / 3)


def test_macro_f1_counts_only_gold_labels_but_false_positives_on_them():
    # gold a, a; predicted a, c. Class c is never gold, so it is not averaged: a -> tp 1, fn 1 -> 2/3.
    results = [choice([0.9, 0.1], 0, ("a", "c")), choice([0.1, 0.9], 0, ("a", "c"))]
    assert macro_f1(results) == pytest.approx(2 / 3)


def test_mean_absolute_error_uses_expected_level():
    # expected 1.3 vs gold 2 -> 0.7 ; expected 0.0 vs gold 1 -> 1.0
    assert mean_absolute_error([score([0.2, 0.3, 0.5], 2), score([1.0, 0.0, 0.0], 1)]) == pytest.approx(0.85)


def test_coverage_curve_by_hand():
    results = [noul(0.9, True), noul(0.3, True), noul(0.55, False)]  # confidences 0.9, 0.7, 0.55
    curve = {row["threshold"]: row for row in coverage_curve(results)}
    assert len(curve) == 21
    assert curve[0.0] == {"threshold": 0.0, "coverage": 1.0, "n": 3, "accuracy": pytest.approx(1 / 3)}
    assert curve[0.6] == {"threshold": 0.6, "coverage": pytest.approx(2 / 3), "n": 2, "accuracy": 0.5}
    assert curve[0.75] == {"threshold": 0.75, "coverage": pytest.approx(1 / 3), "n": 1, "accuracy": 1.0}
    assert curve[0.95] == {"threshold": 0.95, "coverage": 0.0, "n": 0, "accuracy": None}


def test_letter_bias_by_hand():
    results = [
        choice([0.6, 0.4], 0),  # K2, gold pos 0, correct, p_gold 0.6
        choice([0.3, 0.7], 0),  # K2, gold pos 0, wrong, p_gold 0.3
        choice([0.2, 0.3, 0.5], 2),  # K3, gold pos 2, correct, p_gold 0.5
    ]
    bias = letter_bias(results)
    assert bias["by_position"]["0"] == {"n": 2, "accuracy": 0.5, "mean_gold_probability": pytest.approx(0.45)}
    assert bias["by_position"]["2"] == {"n": 1, "accuracy": 1.0, "mean_gold_probability": 0.5}
    assert bias["by_k"]["2"] == {"n": 2, "accuracy": 0.5, "mean_gold_probability": pytest.approx(0.45)}
    assert bias["by_k"]["3"]["n"] == 1


def test_symmetry_by_hand():
    # sums 1.0 and 1.2 -> mean 1.1, population std 0.1.
    # pair 1: q says yes (0.7), not-q says no (0.3) -> consistent. pair 2: both yes -> inconsistent.
    result = symmetry([(0.7, 0.3), (0.6, 0.6)])
    assert result == {"n": 2, "mean_sum": pytest.approx(1.1), "std_sum": pytest.approx(0.1), "argmax_consistent": 0.5}


def test_max_abs_difference():
    assert max_abs_difference([[0.1, 0.9], [0.5, 0.5]], [[0.1, 0.9], [0.4, 0.6]]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        max_abs_difference([[0.1, 0.9]], [[0.1, 0.8, 0.1]])


def test_timing_summary():
    assert timing_summary([0.3, 0.1, 0.2]) == {"n": 3, "median_ms": pytest.approx(200.0), "mean_ms": pytest.approx(200.0)}


# Aggregation


def test_split_metrics_structure_and_values():
    results = [noul(0.8, True), noul(0.4, True), choice([0.7, 0.3], 0, ("a", "b")), score([0.1, 0.9], 1)]
    metrics = split_metrics(results)
    assert set(metrics) == {"overall", "noul", "choice", "score", "letter_bias"}
    assert metrics["overall"]["n"] == 4 and metrics["overall"]["accuracy"] == pytest.approx(0.75)
    assert metrics["noul"]["accuracy"] == 0.5
    assert "macro_f1" in metrics["choice"] and "mae" in metrics["score"]
    assert "coverage" in metrics["noul"] and "coverage" not in metrics["overall"]
    assert len(metrics["noul"]["reliability"]) == 15


def test_split_metrics_omits_absent_types():
    assert set(split_metrics([score([0.5, 0.5], 0)])) == {"overall", "score"}


def test_result_line_round_trip():
    from jevmark.metrics import result_from_line, result_to_line

    r = QuestionResult("id1", "about_domain", "noul", (0.7, 0.3), 1, ("true", "false"), split="valid", kind="about_domain", negated_p_yes=0.4)
    line = result_to_line(r)
    assert line["type"] == "noul" and line["confidence"] == pytest.approx(0.7) and line["prediction"] == 0
    assert result_from_line(line) == r
