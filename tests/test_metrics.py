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
    assert plain(bias["by_position"]["0"]) == {"n": 2, "accuracy": 0.5, "mean_gold_probability": pytest.approx(0.45)}
    assert plain(bias["by_position"]["2"]) == {"n": 1, "accuracy": 1.0, "mean_gold_probability": 0.5}
    assert plain(bias["by_k"]["2"]) == {"n": 2, "accuracy": 0.5, "mean_gold_probability": pytest.approx(0.45)}
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


# Breakdowns added in task 1.6b


def plain(block):
    """A breakdown entry without its bootstrap intervals and nested breakdowns, for comparisons by hand."""
    return {k: v for k, v in block.items() if not k.endswith("_ci") and k != "by_question_position"}


def kinded(p_true, gold_true, kind):
    return QuestionResult("r", kind, "noul", (p_true, 1 - p_true), 0 if gold_true else 1, ("true", "false"), kind=kind)


def test_noul_by_kind_by_hand():
    from jevmark.metrics import noul_by_kind

    results = [
        kinded(0.9, True, "about_domain"),  # predicts yes, correct, top1 0.9
        kinded(0.6, False, "about_domain"),  # predicts yes, wrong, top1 0.6
        kinded(0.2, False, "out_of_scope"),  # predicts no, correct, top1 0.8
    ]
    by_kind = noul_by_kind(results)
    # about_domain ECE: confidence 0.9 (correct, gap 0.1) and 0.6 (wrong, gap 0.6) in separate bins, weight 1/2 each.
    assert plain(by_kind["about_domain"]) == {"n": 2, "accuracy": 0.5, "ece": pytest.approx(0.35), "yes_rate": 1.0}
    assert plain(by_kind["out_of_scope"]) == {"n": 1, "accuracy": 1.0, "ece": pytest.approx(0.2), "yes_rate": 0.0}


def test_noul_block_reports_overall_yes_rate():
    metrics = split_metrics([kinded(0.9, True, "about_domain"), kinded(0.2, False, "out_of_scope")])
    assert metrics["noul"]["yes_rate"] == 0.5
    assert set(metrics["noul"]["by_kind"]) == {"about_domain", "out_of_scope"}


def test_choice_by_gold_other_by_hand():
    from jevmark.metrics import choice_by_gold_other

    labels = ("a", "other", "b")
    results = [
        choice([0.2, 0.7, 0.1], 1, labels),  # gold other, predicts other: correct
        choice([0.6, 0.3, 0.1], 1, labels),  # gold other, predicts a: wrong
        choice([0.1, 0.6, 0.3], 0, labels),  # gold a, predicts other: wrong
        choice([0.1, 0.2, 0.7], 2, labels),  # gold b, predicts b: correct
        choice([0.5, 0.5], 0, ("x", "y")),  # no other option offered: excluded
    ]
    split = choice_by_gold_other(results)
    assert plain(split["gold_other"]) == {"n": 2, "accuracy": 0.5, "predicted_other_rate": 0.5}
    assert plain(split["gold_named"]) == {"n": 2, "accuracy": 0.5, "predicted_other_rate": 0.5}
    assert split["predicted_other_rate"] == 0.5 and split["n_offering_other"] == 4


def test_choice_by_gold_other_absent_without_other_option():
    metrics = split_metrics([choice([0.7, 0.3], 0, ("a", "b"))])
    assert "by_gold_other" not in metrics["choice"]


def test_letter_bias_by_k_and_position_by_hand():
    results = [
        choice([0.6, 0.4], 0),  # K2 pos0 correct p 0.6
        choice([0.3, 0.7], 0),  # K2 pos0 wrong p 0.3
        choice([0.4, 0.6], 1),  # K2 pos1 correct p 0.6
        choice([0.2, 0.3, 0.5], 2),  # K3 pos2 correct p 0.5
    ]
    joint = letter_bias(results)["by_k_position"]
    assert plain(joint["2"]["0"]) == {"n": 2, "accuracy": 0.5, "mean_gold_probability": pytest.approx(0.45)}
    assert plain(joint["2"]["1"]) == {"n": 1, "accuracy": 1.0, "mean_gold_probability": 0.6}
    assert {k: plain(v) for k, v in joint["3"].items()} == {"2": {"n": 1, "accuracy": 1.0, "mean_gold_probability": 0.5}}


# Bootstrap intervals, question position and order sensitivity (decision 42)


def test_bootstrap_ci_is_deterministic_and_contains_the_estimate():
    from jevmark.metrics import bootstrap_ci

    results = [noul(0.9 if i % 3 else 0.3, True, rid=f"r{i}") for i in range(60)]
    ci = bootstrap_ci(results)
    assert ci == bootstrap_ci(results)
    accuracy = sum(r.correct for r in results) / len(results)
    assert ci["accuracy_ci"][0] <= accuracy <= ci["accuracy_ci"][1]
    calibration = ece([r.top1 for r in results], [r.correct for r in results])
    assert ci["ece_ci"][0] <= calibration + 1e-9 and ci["ece_ci"][0] < ci["ece_ci"][1]


def test_bootstrap_ci_degenerate_and_shrinking():
    from jevmark.metrics import bootstrap_ci

    all_correct = [noul(0.8, True, rid=f"r{i}") for i in range(20)]
    assert bootstrap_ci(all_correct) == {"accuracy_ci": [1.0, 1.0], "ece_ci": [pytest.approx(0.2), pytest.approx(0.2)]}

    def width(n):
        ci = bootstrap_ci([noul(0.7, i % 2 == 0, rid=f"r{i}") for i in range(n)])["accuracy_ci"]
        return ci[1] - ci[0]

    assert width(400) < width(40) / 2


def test_bootstrap_resamples_records_not_questions():
    from jevmark.metrics import bootstrap_ci

    # The same 40 answers as 40 independent records, or as 4 records of 10 questions:
    # clustering must widen the interval.
    answers = [i % 4 == 0 for i in range(40)]
    separate = [noul(0.7, a, rid=f"r{i}") for i, a in enumerate(answers)]
    clustered = [noul(0.7, a, rid=f"r{i % 4}") for i, a in enumerate(answers)]
    lo_s, hi_s = bootstrap_ci(separate)["accuracy_ci"]
    lo_c, hi_c = bootstrap_ci(clustered)["accuracy_ci"]
    assert hi_c - lo_c > hi_s - lo_s


def test_every_block_and_breakdown_has_intervals():
    results = [
        QuestionResult(f"r{i}", "about_domain", "noul", (0.8, 0.2), i % 2, ("true", "false"), kind="about_domain", position=i % 2) for i in range(10)
    ] + [choice([0.6, 0.4], i % 2, ("a", "other"), rid=f"r{i}") for i in range(10)]
    metrics = split_metrics(results)
    for block in ("overall", "noul", "choice"):
        assert len(metrics[block]["accuracy_ci"]) == 2 and len(metrics[block]["ece_ci"]) == 2
    assert "accuracy_ci" in metrics["noul"]["by_kind"]["about_domain"] and "ece_ci" in metrics["noul"]["by_kind"]["about_domain"]
    assert "accuracy_ci" in metrics["choice"]["by_gold_other"]["gold_other"]
    assert "accuracy_ci" in metrics["letter_bias"]["by_k"]["2"]


def test_by_question_position_by_hand():
    from jevmark.metrics import by_question_position

    def at(position, correct, rid):
        return QuestionResult(rid, "q", "noul", (0.9, 0.1), 0 if correct else 1, ("true", "false"), kind="about_intent", position=position)

    results = [at(0, True, "a"), at(0, False, "b"), at(1, True, "c"), at(2, True, "d")]
    table = by_question_position(results)
    assert plain(table["first"]) == {"n": 2, "accuracy": 0.5, "ece": pytest.approx(0.4)}
    assert plain(table["later"]) == {"n": 2, "accuracy": 1.0, "ece": pytest.approx(0.1)}
    assert "accuracy_ci" in table["later"] and "ece_ci" in table["later"]
    metrics = split_metrics(results)
    assert set(metrics["noul"]["by_question_position"]) == {"first", "later"}
    assert set(metrics["noul"]["by_kind"]["about_intent"]["by_question_position"]) == {"first", "later"}
    assert by_question_position([noul(0.9, True)]) is None  # results without positions (older files)


def test_order_sensitivity_by_hand():
    from jevmark.metrics import order_sensitivity, split_report

    results = [
        QuestionResult("a", "q", "noul", (0.8, 0.2), 0, ("true", "false"), kind="k", shuffled_probs=(0.4, 0.6)),  # flips to wrong
        choice([0.6, 0.4], 0, rid="b"),  # not reordered: ignored
    ]
    results[1] = QuestionResult("b", "intent", "choice", (0.6, 0.4), 0, ("x", "y"), shuffled_probs=(0.7, 0.3))
    table = order_sensitivity(results)
    assert table["overall"] == {
        "n": 2,
        "accuracy": 1.0,
        "accuracy_shuffled": 0.5,
        "prediction_agreement": 0.5,
        "mean_max_abs_difference": pytest.approx(0.25),
        "max_abs_difference": pytest.approx(0.4),
    }
    assert table["noul"]["n"] == 1 and table["choice"]["accuracy_shuffled"] == 1.0
    assert order_sensitivity([noul(0.5, True)]) is None
    assert "order_sensitivity" in split_report(results)


def test_result_line_round_trip_with_position_and_shuffle():
    from jevmark.metrics import result_from_line, result_to_line

    r = QuestionResult("id1", "intent", "choice", (0.7, 0.3), 0, ("a", "b"), split="valid", position=1, shuffled_probs=(0.6, 0.4), shuffled_position=0)
    assert result_from_line(result_to_line(r)) == r
    old = result_to_line(r)
    for key in ("position", "shuffled_probs", "shuffled_position"):
        del old[key]
    assert result_from_line(old).position is None and result_from_line(old).shuffled_probs is None


def test_form_nouls_are_kept_out_of_the_headline_metrics():
    # Decision 44: overall and the noul block cover gold-dependent questions; form nouls get their own block.
    def q(kind, p_true, gold_true, rid):
        return QuestionResult(rid, kind, "noul", (p_true, 1 - p_true), 0 if gold_true else 1, ("true", "false"), kind=kind, negated_p_yes=1 - p_true)

    results = [
        q("about_domain", 0.9, True, "a"),  # correct
        q("about_domain", 0.8, True, "b"),  # correct
        q("char_count_over", 0.9, False, "a"),  # wrong, form
        q("word_count_over", 0.7, False, "b"),  # wrong, form
        choice([0.6, 0.4], 0, rid="a"),
    ]
    metrics = split_metrics(results)
    assert metrics["overall"]["n"] == 3 and metrics["overall"]["accuracy"] == 1.0
    assert metrics["noul"]["n"] == 2 and set(metrics["noul"]["by_kind"]) == {"about_domain"}
    assert metrics["form"]["n"] == 2 and metrics["form"]["accuracy"] == 0.0
    assert set(metrics["form"]["by_kind"]) == {"char_count_over", "word_count_over"}
    assert {"ece", "brier", "nll", "coverage", "accuracy_ci", "ece_ci", "yes_rate"} <= set(metrics["form"])
    from jevmark.metrics import split_report

    report = split_report(results)
    assert report["symmetry"]["n"] == 2 and report["form"]["symmetry"]["n"] == 2
    assert "form" not in split_metrics(results[:2])
