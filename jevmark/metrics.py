"""Evaluation metrics on unrounded answer distributions (docs/TASKS.md 1.6).

Two confidence quantities are used on purpose, and every report states this once:

- ECE and the reliability diagram use the top-1 probability, max_k p_k, for every
  question type (for noul this is max(p, 1 - p)).
- Coverage curves use the response confidence field: 1 - H(p) / ln(K) for choice
  and score (the same normalised_confidence the API returns), and max(p, 1 - p)
  for noul, which has no confidence field (decision 18).

A noul distribution is (P(true), P(false)); its prediction is true when
P(true) >= 0.5. Choice and score predictions are the argmax, first index on ties.

Headline metrics (overall and the per-type blocks) cover the gold-dependent
questions only; form nouls, which ask about the text's form rather than a label,
are reported in their own "form" block with the same metrics (decision 44).

Accuracy and ECE carry 95 percent bootstrap confidence intervals (1000 resamples,
percentile method, seed 0) on every split and breakdown. Resampling is by record,
not by question, because the questions of one record share a state and are not
independent: a resample draws records with replacement and keeps all of their
questions in the block being measured.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from jevmark.data.form import FORM_KINDS
from jevmark.systemone import normalised_confidence

N_BINS = 15
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(21))
NLL_FLOOR = 1e-12
QUESTION_TYPES = ("noul", "choice", "score")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 100


@dataclass(frozen=True)
class QuestionResult:
    record_id: str
    question_id: str
    qtype: str
    probs: tuple[float, ...]  # in option order; noul is (P(true), P(false))
    gold: int  # index of the gold option
    labels: tuple[str, ...]  # option labels in the same order
    split: str = ""
    kind: str | None = None  # the noul kind (its question id) for noul questions, else None
    negated_p_yes: float | None = None  # noul only: P(yes) for the negated instruction
    position: int | None = None  # 0-based index of the question in its record's question order
    shuffled_probs: tuple[float, ...] | None = None  # the distribution with the record's questions reordered (--shuffle-questions)
    shuffled_position: int | None = None  # the question's index in that reordered record

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


def bootstrap_ci(
    results: Sequence[QuestionResult], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED, n_bins: int = N_BINS
) -> dict[str, list[float]]:
    """95 percent percentile intervals for accuracy and ECE (top-1, n_bins bins), resampling records with replacement."""
    if not results:
        return {}
    return bootstrap_intervals(
        [r.record_id for r in results], [r.correct for r in results], [r.top1 for r in results], n_resamples, seed, n_bins
    )


def bootstrap_intervals(
    record_ids: Sequence[str],
    correct: Sequence[bool],
    confidences: Sequence[float] | None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    n_bins: int = N_BINS,
) -> dict[str, list[float]]:
    """bootstrap_ci on plain sequences, one entry per question; the baselines use it too (task 1.8).

    Each resample gives every record a multiplicity from a multinomial draw; a
    question's weight is its record's multiplicity. ECE with weights is
    sum over bins of |sum of weighted correct - sum of weighted confidence| / total weight.
    Without confidences only the accuracy interval is returned.
    """
    if not record_ids:
        return {}
    record_index: dict[str, int] = {}
    rec = np.array([record_index.setdefault(r, len(record_index)) for r in record_ids])
    hit = np.array([float(c) for c in correct])
    top1 = np.array(confidences if confidences is not None else np.zeros(len(hit)), dtype=float)
    bins = np.minimum((top1 * n_bins).astype(int), n_bins - 1)
    one_hot = np.zeros((len(hit), n_bins))
    one_hot[np.arange(len(hit)), bins] = 1.0
    n_records = len(record_index)
    rng = np.random.default_rng(seed)
    accuracies, eces = [], []
    for start in range(0, n_resamples, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, n_resamples - start)
        weights = rng.multinomial(n_records, np.full(n_records, 1.0 / n_records), size=size)[:, rec].astype(float)
        total = weights.sum(axis=1)
        total[total == 0] = np.nan  # a resample that drew none of this block's records
        accuracies.append((weights @ hit) / total)
        gap = np.abs((weights * hit) @ one_hot - (weights * top1) @ one_hot).sum(axis=1)
        eces.append(gap / total)
    accuracy, calibration = np.concatenate(accuracies), np.concatenate(eces)

    def interval(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]

    out = {"accuracy_ci": interval(accuracy)}
    if confidences is not None:
        out["ece_ci"] = interval(calibration)
    return out


def _with_ci(block: dict[str, Any], results: Sequence[QuestionResult], ece_too: bool = True) -> dict[str, Any]:
    ci = bootstrap_ci(results)
    block["accuracy_ci"] = ci.get("accuracy_ci")
    if ece_too:
        block["ece_ci"] = ci.get("ece_ci")
    return block


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


def _gold_stats(members: Sequence[QuestionResult]) -> dict[str, Any]:
    stats = {
        "n": len(members),
        "accuracy": _mean([float(r.correct) for r in members]),
        "mean_gold_probability": _mean([r.probs[r.gold] for r in members]),
    }
    return _with_ci(stats, members, ece_too=False)


def _by_int_key(groups: Mapping[str, Sequence[QuestionResult]]) -> dict[str, dict[str, Any]]:
    return {key: _gold_stats(members) for key, members in sorted(groups.items(), key=lambda item: int(item[0]))}


def letter_bias(results: Sequence[QuestionResult]) -> dict[str, Any]:
    """Choice questions: accuracy and mean probability on the gold option by gold position, by K, and by K then position.

    Positions beyond 2 exist only for larger K, where accuracy is lower anyway, so
    by_k_position is the table that separates position bias from the effect of K.
    """
    by_position: dict[str, list[QuestionResult]] = defaultdict(list)
    by_k: dict[str, list[QuestionResult]] = defaultdict(list)
    joint: dict[str, dict[str, list[QuestionResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_position[str(r.gold)].append(r)
        by_k[str(r.k)].append(r)
        joint[str(r.k)][str(r.gold)].append(r)
    return {
        "by_position": _by_int_key(by_position),
        "by_k": _by_int_key(by_k),
        "by_k_position": {k: _by_int_key(joint[k]) for k in sorted(joint, key=int)},
    }


def noul_by_kind(results: Sequence[QuestionResult]) -> dict[str, dict[str, Any]]:
    """Noul questions by kind (question id): accuracy, ECE on top-1 and the share of yes predictions."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.kind or "unknown"].append(r)
    out = {}
    for kind, members in sorted(groups.items()):
        entry = {
            "n": len(members),
            "accuracy": _mean([float(r.correct) for r in members]),
            "ece": ece([r.top1 for r in members], [r.correct for r in members]),
            "yes_rate": _mean([float(r.prediction == 0) for r in members]),
        }
        out[kind] = _with_ci(entry, members)
        by_position = by_question_position(members)
        if by_position:
            out[kind]["by_question_position"] = by_position
    return out


def by_question_position(results: Sequence[QuestionResult]) -> dict[str, dict[str, Any]] | None:
    """Accuracy and ECE with intervals for questions first in their record versus later ones; None without positions."""
    placed = [r for r in results if r.position is not None]
    if not placed:
        return None
    out = {}
    for name, members in (("first", [r for r in placed if r.position == 0]), ("later", [r for r in placed if r.position > 0])):
        if members:
            entry = {"n": len(members), "accuracy": _mean([float(r.correct) for r in members]), "ece": ece([r.top1 for r in members], [r.correct for r in members])}
            out[name] = _with_ci(entry, members)
    return out


def order_sensitivity(results: Sequence[QuestionResult]) -> dict[str, Any] | None:
    """Questions evaluated a second time with their record's questions reordered: how much the answers moved.

    Per question type and overall: accuracy in the stored and in the reordered order,
    the share of questions whose prediction is unchanged, and the mean and maximum
    over questions of max_k |p_k - p'_k|. None when nothing was reordered.
    """
    shuffled = [r for r in results if r.shuffled_probs is not None]
    if not shuffled:
        return None

    def block(members: Sequence[QuestionResult]) -> dict[str, Any]:
        moved = [replace(r, probs=r.shuffled_probs) for r in members]
        diffs = [max(abs(a - b) for a, b in zip(r.probs, r.shuffled_probs)) for r in members]
        return {
            "n": len(members),
            "accuracy": _mean([float(r.correct) for r in members]),
            "accuracy_shuffled": _mean([float(r.correct) for r in moved]),
            "prediction_agreement": _mean([float(a.prediction == b.prediction) for a, b in zip(members, moved)]),
            "mean_max_abs_difference": statistics.fmean(diffs),
            "max_abs_difference": max(diffs),
        }

    out = {"overall": block(shuffled)}
    for qtype in QUESTION_TYPES:
        of_type = [r for r in shuffled if r.qtype == qtype]
        if of_type:
            out[qtype] = block(of_type)
    return out


def choice_by_gold_other(results: Sequence[QuestionResult], other: str = "other") -> dict[str, Any] | None:
    """Choice questions that offer `other`, split by whether the gold answer is `other` or a named label.

    Reports accuracy and the rate of predicting `other` in each group and overall.
    None when no question offers `other`.
    """
    offering = [r for r in results if other in r.labels]
    if not offering:
        return None

    def stats(members: Sequence[QuestionResult]) -> dict[str, Any]:
        return _with_ci(
            {
                "n": len(members),
                "accuracy": _mean([float(r.correct) for r in members]),
                "predicted_other_rate": _mean([float(r.labels[r.prediction] == other) for r in members]),
            },
            members,
            ece_too=False,
        )

    split = {
        "n_offering_other": len(offering),
        "accuracy_ci": bootstrap_ci(offering).get("accuracy_ci"),
        "predicted_other_rate": _mean([float(r.labels[r.prediction] == other) for r in offering]),
    }
    gold_other = [r for r in offering if r.labels[r.gold] == other]
    gold_named = [r for r in offering if r.labels[r.gold] != other]
    if gold_other:
        split["gold_other"] = stats(gold_other)
    if gold_named:
        split["gold_named"] = stats(gold_named)
    return split


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
    _with_ci(summary, results)
    if with_coverage:
        summary["coverage"] = coverage_curve(results)
    return summary


def split_report(results: Sequence[QuestionResult]) -> dict[str, Any]:
    """Everything metrics.json holds for one split: split_metrics, n_records, noul symmetry and order sensitivity when measured.

    Symmetry and order sensitivity follow the same split as split_metrics: the
    headline entries cover gold-dependent questions, and form nouls get their own
    inside the "form" block.
    """
    report = split_metrics(results)
    report["n_records"] = len({r.record_id for r in results})
    for members, target in ((gold_dependent(results), report), (form_nouls(results), report.get("form"))):
        if target is None:
            continue
        pairs = [(r.probs[0], r.negated_p_yes) for r in members if r.qtype == "noul" and r.negated_p_yes is not None]
        if pairs:
            target["symmetry"] = symmetry(pairs)
        sensitivity = order_sensitivity(members)
        if sensitivity is not None:
            target["order_sensitivity"] = sensitivity
    return report


# results.jsonl.gz: one line per question, enough to rebuild metrics.json exactly.


def result_to_line(r: QuestionResult) -> dict[str, Any]:
    line = asdict(r)
    line["type"] = line.pop("qtype")
    line["confidence"] = r.confidence
    line["prediction"] = r.prediction
    return line


def result_from_line(line: Mapping[str, Any]) -> QuestionResult:
    """Inverse of result_to_line; confidence and prediction are derived, so they are recomputed, not read."""
    return QuestionResult(
        record_id=line["record_id"],
        question_id=line["question_id"],
        qtype=line["type"],
        probs=tuple(line["probs"]),
        gold=int(line["gold"]),
        labels=tuple(line["labels"]),
        split=line["split"],
        kind=line["kind"],
        negated_p_yes=line["negated_p_yes"],
        position=line.get("position"),  # absent in results files written before these fields existed
        shuffled_probs=tuple(line["shuffled_probs"]) if line.get("shuffled_probs") is not None else None,
        shuffled_position=line.get("shuffled_position"),
    )


def write_results(path: Path, results: Sequence[QuestionResult]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(result_to_line(r)) + "\n")


def read_results(path: Path) -> list[QuestionResult]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [result_from_line(json.loads(line)) for line in f if line.strip()]


def reports_by_split(results: Sequence[QuestionResult]) -> dict[str, dict[str, Any]]:
    """split -> split_report, splits in order of first appearance."""
    order: dict[str, list[QuestionResult]] = {}
    for r in results:
        order.setdefault(r.split, []).append(r)
    return {split: split_report(members) for split, members in order.items()}


def is_form_noul(r: QuestionResult) -> bool:
    """A form noul (label-independent, answered from the text's form; data v1.3), recognised by its kind."""
    return r.qtype == "noul" and r.kind in FORM_KINDS


def gold_dependent(results: Sequence[QuestionResult]) -> list[QuestionResult]:
    return [r for r in results if not is_form_noul(r)]


def form_nouls(results: Sequence[QuestionResult]) -> list[QuestionResult]:
    return [r for r in results if is_form_noul(r)]


def split_metrics(results: Sequence[QuestionResult]) -> dict[str, Any]:
    """Headline metrics on the gold-dependent questions, plus a "form" block for form nouls (decision 44).

    overall and one block per question type cover the questions whose answer
    depends on a gold label, the capability the model is for; form nouls (word
    and character counts and the like) are reported apart, in "form", with the
    same metrics, so they never move a split's headline accuracy or ECE. Coverage
    is per type only, since its confidence differs by type.
    """
    gold = gold_dependent(results)
    metrics: dict[str, Any] = {"overall": summarize(gold, with_coverage=False)}
    for qtype in QUESTION_TYPES:
        of_type = [r for r in gold if r.qtype == qtype]
        if not of_type:
            continue
        block = summarize(of_type, with_coverage=True)
        if qtype == "noul":
            block["yes_rate"] = _mean([float(r.prediction == 0) for r in of_type])
            block["by_kind"] = noul_by_kind(of_type)
        if qtype == "choice":
            block["macro_f1"] = macro_f1(of_type)
            by_gold_other = choice_by_gold_other(of_type)
            if by_gold_other is not None:
                block["by_gold_other"] = by_gold_other
            metrics["letter_bias"] = letter_bias(of_type)
        if qtype == "score":
            block["mae"] = mean_absolute_error(of_type)
        by_position = by_question_position(of_type)
        if by_position:
            block["by_question_position"] = by_position
        metrics[qtype] = block
    form = form_nouls(results)
    if form:
        block = summarize(form, with_coverage=True)
        block["yes_rate"] = _mean([float(r.prediction == 0) for r in form])
        block["by_kind"] = noul_by_kind(form)
        by_position = by_question_position(form)
        if by_position:
            block["by_question_position"] = by_position
        metrics["form"] = block
    return metrics
