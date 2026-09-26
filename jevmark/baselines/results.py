"""Per-question baseline answers and their metrics, in the metrics.json layout of evaluate.py (task 1.8).

A baseline run writes replies.jsonl, one line per request with the raw reply and
its tokens (plus cost and latency for the API baseline). answers_from_replies turns
the replies into one BaselineAnswer per question with the shared parser, and
baseline_reports_by_split builds, per split, the same blocks evaluate.py writes:
overall and one block per question type on the gold-dependent questions, form nouls
apart in "form" (decision 44).

Every block reports:

- n, n_parsed, parse_failure_rate and parse_failures (count per failure status)
- accuracy on the parsed answers, and accuracy_all counting failures as wrong;
  jevmark itself never fails to parse, so accuracy_all is the like-for-like number
- ECE, reliability and (per type) coverage on the verbalized confidence, over the
  parsed answers that carry one (n_with_confidence); there is no distribution, so
  the stated confidence stands in for both the top-1 probability and the response
  confidence field, and the coverage share is of those answers
- 95 percent bootstrap intervals by record for accuracy, accuracy_all and ECE,
  with the same resampling as metrics.bootstrap_ci
- a secondary "lenient" reading (decision 48): accuracy_all, parse failure rate
  and the number of answers recovered when an echoed option line such as
  "true: yes" counts as the label before its first colon; the strict numbers
  above stay the headline
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevmark.baselines.prompt import FAILURES, parse_reply
from jevmark.data.form import FORM_KINDS
from jevmark.metrics import QUESTION_TYPES, THRESHOLDS, bootstrap_intervals, ece, reliability
from jevmark.schema import Request


@dataclass(frozen=True)
class BaselineAnswer:
    record_id: str
    question_id: str
    split: str
    qtype: str
    kind: str | None  # the noul kind (its question id) for noul questions
    labels: tuple[str, ...]
    gold: int
    status: str
    prediction: int | None
    confidence: float | None
    position: int
    lenient_status: str | None = None  # the lenient reading (decision 48); None means the same as the strict one
    lenient_prediction: int | None = None

    @property
    def lenient_correct(self) -> bool:
        status = self.status if self.lenient_status is None else self.lenient_status
        prediction = self.prediction if self.lenient_status is None else self.lenient_prediction
        return status == "ok" and prediction == self.gold

    @property
    def lenient_parsed(self) -> bool:
        return (self.status if self.lenient_status is None else self.lenient_status) == "ok"

    @property
    def parsed(self) -> bool:
        return self.status == "ok"

    @property
    def correct(self) -> bool:
        return self.parsed and self.prediction == self.gold

    @property
    def is_form(self) -> bool:
        return self.qtype == "noul" and self.kind in FORM_KINDS


def gold_index(question: Mapping[str, Any], gold: Any) -> int:
    if question["type"] == "noul":
        return {"true": 0, "false": 1}[gold]
    if question["type"] == "choice":
        return list(question["criteria"]).index(gold)
    return int(gold)


def option_labels(question: Mapping[str, Any]) -> tuple[str, ...]:
    if question["type"] == "noul":
        return ("true", "false")
    if question["type"] == "choice":
        return tuple(question["criteria"])
    return tuple(str(i) for i in range(len(question["criteria"])))


def request_of(record: Mapping[str, Any]) -> Request:
    return Request.from_dict({"state": record["state"], "questions": record["questions"]})


def answers_for_record(record: Mapping[str, Any], split: str, reply: str | None) -> list[BaselineAnswer]:
    request = request_of(record)
    parsed = parse_reply(reply, request)
    lenient = parse_reply(reply, request, lenient=True)
    out = []
    for position, (qid, question) in enumerate(record["questions"].items()):
        p, q = parsed[qid], lenient[qid]
        out.append(
            BaselineAnswer(
                record_id=record["id"],
                question_id=qid,
                split=split,
                qtype=question["type"],
                kind=qid if question["type"] == "noul" else None,
                labels=option_labels(question),
                gold=gold_index(question, record["gold"][qid]),
                status=p.status,
                prediction=p.answer,
                confidence=p.confidence,
                position=position,
                lenient_status=None if q == p else q.status,
                lenient_prediction=None if q == p else q.answer,
            )
        )
    return out


def answers_from_replies(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]], replies: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[BaselineAnswer]:
    """One answer per question of every record that has a reply, keyed by (split, record id); records without a reply are skipped."""
    out: list[BaselineAnswer] = []
    for split, records in records_by_split.items():
        for record in records:
            line = replies.get((split, record["id"]))
            if line is not None:
                out += answers_for_record(record, split, line.get("reply"))
    return out


# replies.jsonl


def read_replies(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """(split, record_id) -> reply line; a later line for the same request replaces an earlier one."""
    replies: dict[tuple[str, str], dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                replies[(row["split"], row["record_id"])] = row
    return replies


def append_reply(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# Metrics


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _accuracy_block(answers: Sequence[BaselineAnswer]) -> dict[str, Any]:
    parsed = [a for a in answers if a.parsed]
    with_confidence = [a for a in parsed if a.confidence is not None]
    counts = Counter(a.status for a in answers)
    block: dict[str, Any] = {
        "n": len(answers),
        "n_parsed": len(parsed),
        "parse_failure_rate": 1 - len(parsed) / len(answers),
        "parse_failures": {status: counts.get(status, 0) for status in FAILURES},
        "accuracy": _mean([float(a.correct) for a in parsed]),
        "accuracy_ci": bootstrap_intervals([a.record_id for a in parsed], [a.correct for a in parsed], None).get("accuracy_ci"),
        "accuracy_all": _mean([float(a.correct) for a in answers]),
        "accuracy_all_ci": bootstrap_intervals([a.record_id for a in answers], [a.correct for a in answers], None).get("accuracy_ci"),
        "n_with_confidence": len(with_confidence),
        "ece": None,
        "ece_ci": None,
    }
    block["lenient"] = {
        "accuracy_all": _mean([float(a.lenient_correct) for a in answers]),
        "parse_failure_rate": 1 - sum(a.lenient_parsed for a in answers) / len(answers),
        "recovered": sum(a.lenient_parsed and not a.parsed for a in answers),
    }
    if with_confidence:
        confidences = [a.confidence for a in with_confidence]
        correct = [a.correct for a in with_confidence]
        block["ece"] = ece(confidences, correct)
        block["ece_ci"] = bootstrap_intervals([a.record_id for a in with_confidence], correct, confidences).get("ece_ci")
        block["reliability"] = reliability(confidences, correct)
    return block


def coverage_curve(answers: Sequence[BaselineAnswer], thresholds: Sequence[float] = THRESHOLDS) -> list[dict[str, Any]] | None:
    """Over parsed answers with a verbalized confidence: share with confidence >= threshold and their accuracy."""
    rated = [a for a in answers if a.parsed and a.confidence is not None]
    if not rated:
        return None
    rows = []
    for t in thresholds:
        kept = [a for a in rated if a.confidence >= t - 1e-12]
        rows.append({"threshold": t, "coverage": len(kept) / len(rated), "n": len(kept), "accuracy": _mean([float(a.correct) for a in kept])})
    return rows


def _yes_rate(answers: Sequence[BaselineAnswer]) -> float | None:
    return _mean([float(a.prediction == 0) for a in answers if a.parsed])


def _by_kind(answers: Sequence[BaselineAnswer]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[BaselineAnswer]] = defaultdict(list)
    for a in answers:
        groups[a.kind or "unknown"].append(a)
    return {kind: {**_accuracy_block(members), "yes_rate": _yes_rate(members)} for kind, members in sorted(groups.items())}


def _macro_f1(answers: Sequence[BaselineAnswer]) -> float | None:
    """metrics.macro_f1 on parsed answers: F1 averaged over labels that occur as gold."""
    parsed = [a for a in answers if a.parsed]
    if not parsed:
        return None
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for a in parsed:
        gold, pred = a.labels[a.gold], a.labels[a.prediction]
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    gold_labels = {a.labels[a.gold] for a in parsed}
    return sum(2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in gold_labels) / len(gold_labels)


def baseline_split_metrics(answers: Sequence[BaselineAnswer]) -> dict[str, Any]:
    gold = [a for a in answers if not a.is_form]
    metrics: dict[str, Any] = {"overall": _accuracy_block(gold)}
    for qtype in QUESTION_TYPES:
        of_type = [a for a in gold if a.qtype == qtype]
        if not of_type:
            continue
        block = _accuracy_block(of_type)
        block["coverage"] = coverage_curve(of_type)
        if qtype == "noul":
            block["yes_rate"] = _yes_rate(of_type)
            block["by_kind"] = _by_kind(of_type)
        if qtype == "choice":
            block["macro_f1"] = _macro_f1(of_type)
        if qtype == "score":
            parsed = [a for a in of_type if a.parsed]
            block["mae"] = _mean([abs(a.prediction - a.gold) for a in parsed])
        metrics[qtype] = block
    form = [a for a in answers if a.is_form]
    if form:
        block = _accuracy_block(form)
        block["coverage"] = coverage_curve(form)
        block["yes_rate"] = _yes_rate(form)
        block["by_kind"] = _by_kind(form)
        metrics["form"] = block
    metrics["n_records"] = len({a.record_id for a in answers})
    return metrics


def baseline_reports_by_split(answers: Iterable[BaselineAnswer]) -> dict[str, dict[str, Any]]:
    """split -> baseline_split_metrics, splits in order of first appearance."""
    order: dict[str, list[BaselineAnswer]] = {}
    for a in answers:
        order.setdefault(a.split, []).append(a)
    return {split: baseline_split_metrics(members) for split, members in order.items()}


CONFIDENCE_NOTE = (
    "Baseline answers carry a verbalized confidence instead of a distribution. ECE, reliability and coverage use it, "
    "over parsed answers that state one (n_with_confidence). accuracy is on parsed answers; accuracy_all counts parse "
    "failures as wrong and is the number to compare with jevmark runs, which never fail to parse."
)
