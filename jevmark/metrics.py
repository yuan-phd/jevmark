"""Evaluation metrics on unrounded answer distributions (docs/TASKS.md 1.6).

Two confidence quantities are used on purpose, and every report states this once:

- ECE and the reliability diagram use the top-1 probability, max_k p_k, for every
  question type (for noul this is max(p, 1 - p)).
- Coverage curves use the response confidence field: 1 - H(p) / ln(K) for choice
  and score (the same normalised_confidence the API returns), and max(p, 1 - p)
  for noul, which has no confidence field (decision 18).

A noul distribution is (P(true), P(false)); its prediction is true when
P(true) >= 0.5. Choice and score predictions are the argmax, first index on ties.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jevmark.systemone import normalised_confidence

N_BINS = 15
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(21))
NLL_FLOOR = 1e-12
QUESTION_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class QuestionResult:
    record_id: str
    question_id: str
    qtype: str
    probs: tuple[float, ...]  # in option order; noul is (P(true), P(false))
    gold: int  # index of the gold option
    labels: tuple[str, ...]  # option labels in the same order

    @property
    def k(self) -> int:
        return len(self.probs)

    @property
    def prediction(self) -> int:
        if self.qtype == "noul":
            return 0 if self.probs[0] >= 0.5 else 1
        return max(range(self.k), key=self.probs.__getitem__)

    @property
    def correct(self) -> bool:
        return self.prediction == self.gold

    @property
    def top1(self) -> float:
        return max(self.probs)

    @property
    def confidence(self) -> float:
        """The response confidence field: max(p, 1 - p) for noul, normalised entropy otherwise."""
        if self.qtype == "noul":
            return max(self.probs[0], 1 - self.probs[0])
        return normalised_confidence(self.probs)

    @property
    def expected_level(self) -> float:
        return sum(i * p for i, p in enumerate(self.probs))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bin_index(confidence: float, n_bins: int) -> int:
    return min(int(confidence * n_bins), n_bins - 1)


def reliability(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Equal-width bins on [0, 1]; confidence 1.0 falls in the last bin. Empty bins have None means."""
    members: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        members[_bin_index(c, n_bins)].append(i)
    bins = []
    for b, idx in enumerate(members):
        bins.append(
            {
                "lower": round(b / n_bins, 6),
                "upper": round((b + 1) / n_bins, 6),
                "count": len(idx),
                "mean_confidence": _mean([confidences[i] for i in idx]),
                "mean_accuracy": _mean([float(correct[i]) for i in idx]),
            }
        )
    return bins


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = N_BINS) -> float:
    """Expected calibration error: sum over bins of (bin share) x |accuracy - mean confidence|."""
    n = len(confidences)
    return sum(
        b["count"] / n * abs(b["mean_accuracy"] - b["mean_confidence"])
        for b in reliability(confidences, correct, n_bins)
        if b["count"]
    )


def brier(results: Sequence[QuestionResult]) -> float:
    """Multi-class Brier score: mean over questions of sum_k (p_k - 1[k = gold])^2."""
    return sum(sum((p - (k == r.gold)) ** 2 for k, p in enumerate(r.probs)) for r in results) / len(results)


def nll(results: Sequence[QuestionResult]) -> float:
    return sum(-math.log(max(r.probs[r.gold], NLL_FLOOR)) for r in results) / len(results)


def macro_f1(results: Sequence[QuestionResult]) -> float:
    """F1 averaged over the labels that occur as gold; predictions of other labels still count as false positives."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for r in results:
        gold, pred = r.labels[r.gold], r.labels[r.prediction]
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    gold_labels = {r.labels[r.gold] for r in results}
    return sum(2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in gold_labels) / len(gold_labels)


def mean_absolute_error(results: Sequence[QuestionResult]) -> float:
    """Mean |expected level - gold level| for score questions."""
    return sum(abs(r.expected_level - r.gold) for r in results) / len(results)


def coverage_curve(results: Sequence[QuestionResult], thresholds: Sequence[float] = THRESHOLDS) -> list[dict[str, Any]]:
    """Share of questions with confidence >= threshold, and their accuracy (None when none remain)."""
    rows = []
    for t in thresholds:
        kept = [r for r in results if r.confidence >= t - 1e-12]
        rows.append({"threshold": t, "coverage": len(kept) / len(results), "n": len(kept), "accuracy": _mean([float(r.correct) for r in kept])})
    return rows


def letter_bias(results: Sequence[QuestionResult]) -> dict[str, dict[str, dict[str, Any]]]:
    """Choice questions: accuracy and mean probability on the gold option, by gold position and by K."""
    groups: dict[str, dict[str, list[QuestionResult]]] = {"by_position": defaultdict(list), "by_k": defaultdict(list)}
    for r in results:
        groups["by_position"][str(r.gold)].append(r)
        groups["by_k"][str(r.k)].append(r)
    return {
        name: {
            key: {
                "n": len(members),
                "accuracy": _mean([float(r.correct) for r in members]),
                "mean_gold_probability": _mean([r.probs[r.gold] for r in members]),
            }
            for key, members in sorted(by.items(), key=lambda item: int(item[0]))
        }
        for name, by in groups.items()
    }


def symmetry(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Pairs of (P(yes | q), P(yes | not q)).

    mean_sum and std_sum (population) of the two probabilities, which a consistent
    model keeps near 1. argmax_consistent is the share of pairs whose argmax answers
    are logically consistent: yes to q exactly when no to not q.
    """
    sums = [a + b for a, b in pairs]
    consistent = [(a >= 0.5) != (b >= 0.5) for a, b in pairs]
    return {
        "n": len(pairs),
        "mean_sum": statistics.fmean(sums),
        "std_sum": statistics.pstdev(sums),
        "argmax_consistent": _mean([float(c) for c in consistent]),
    }


def max_abs_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Largest absolute difference between two lists of distributions of matching shapes."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("distributions to compare must have matching shapes")
    return max(abs(p - q) for x, y in zip(a, b) for p, q in zip(x, y))


def timing_summary(seconds: Sequence[float]) -> dict[str, Any]:
    return {"n": len(seconds), "median_ms": statistics.median(seconds) * 1000, "mean_ms": statistics.fmean(seconds) * 1000}


def summarize(results: Sequence[QuestionResult], with_coverage: bool) -> dict[str, Any]:
    top1 = [r.top1 for r in results]
    correct = [r.correct for r in results]
    summary: dict[str, Any] = {
        "n": len(results),
        "accuracy": _mean([float(c) for c in correct]),
        "ece": ece(top1, correct),
        "brier": brier(results),
        "nll": nll(results),
        "reliability": reliability(top1, correct),
    }
    if with_coverage:
        summary["coverage"] = coverage_curve(results)
    return summary


def split_metrics(results: Sequence[QuestionResult]) -> dict[str, Any]:
    """overall plus one block per question type present; coverage per type only, since its confidence differs by type."""
    metrics: dict[str, Any] = {"overall": summarize(results, with_coverage=False)}
    for qtype in QUESTION_TYPES:
        of_type = [r for r in results if r.qtype == qtype]
        if not of_type:
            continue
        block = summarize(of_type, with_coverage=True)
        if qtype == "choice":
            block["macro_f1"] = macro_f1(of_type)
            metrics["letter_bias"] = letter_bias(of_type)
        if qtype == "score":
            block["mae"] = mean_absolute_error(of_type)
        metrics[qtype] = block
    return metrics
