"""Shared record helpers and dataset checks for the data builders (docs/DATA.md).

A record is one systemone request plus gold answers. make_record validates it with
Request.from_dict, so a builder can never emit a record the API would reject.
The check functions are used by scripts/build_data.py (on every record) and by
tests/test_data.py.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jevmark.schema import Request

SPLITS = (
    "train",
    "valid",
    "test_indomain",
    "test_unseen_intents",
    "test_sst5",
    "test_agnews",
    "test_emotion",
    "test_banking77",
)


def split_rng(seed: int, name: str) -> random.Random:
    """One generator per split (or purpose), so rebuilding one never changes another."""
    return random.Random(f"{seed}:{name}")


def make_record(
    record_id: str,
    source: str,
    split: str,
    state: str,
    questions: dict[str, dict[str, Any]],
    gold: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    Request.from_dict({"state": state, "questions": questions})
    if set(gold) != set(questions):
        raise ValueError(f"{record_id}: gold keys {sorted(gold)} differ from question ids {sorted(questions)}")
    return {"id": record_id, "source": source, "split": split, "state": state, "questions": questions, "gold": gold, "meta": meta}


def choice_question(instructions: str, options: Sequence[tuple[str, str | None]], rng: random.Random) -> dict[str, Any]:
    """A choice question with its options in a seeded random order, fixed from here on."""
    shuffled = list(options)
    rng.shuffle(shuffled)
    return {"type": "choice", "instructions": instructions, "criteria": dict(shuffled)}


# Checks


def noul_balance(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Noul question id (its kind) -> counts of "true" and "false" gold answers."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "noul":
                counts[qid][record["gold"][qid]] += 1
    return {kind: {"true": c["true"], "false": c["false"]} for kind, c in counts.items()}


def gold_positions(records: Iterable[Mapping[str, Any]]) -> dict[int, Counter]:
    """K -> Counter of the gold option's 0-based position, over every choice question."""
    positions: dict[int, Counter] = defaultdict(Counter)
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "choice":
                labels = list(question["criteria"])
                positions[len(labels)][labels.index(record["gold"][qid])] += 1
    return dict(positions)


def position_deviations(positions: Mapping[int, Counter], min_count: int = 100) -> dict[int, float]:
    """K -> the largest deviation of a position's share from 1/K, in binomial standard deviations.

    Only K with at least min_count questions are checked; smaller groups are too noisy to judge.
    """
    result = {}
    for k, counter in positions.items():
        n = sum(counter.values())
        if n < min_count:
            continue
        sd = math.sqrt((1 / k) * (1 - 1 / k) / n)
        result[k] = max(abs(counter[p] / n - 1 / k) / sd for p in range(k))
    return result


def held_out_leaks(records: Iterable[Mapping[str, Any]], held_out: Iterable[str]) -> list[str]:
    """Ids of records whose gold intent or any option label is a held-out intent."""
    held = set(held_out)
    leaks = []
    for record in records:
        labels = {label for q in record["questions"].values() if q["type"] == "choice" for label in q["criteria"]}
        if record["meta"].get("gold_intent") in held or labels & held:
            leaks.append(record["id"])
    return leaks


def noul_phrasing(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per noul kind: negated share, and gold "true" / "false" counts in each phrasing (data v1.1).

    A kind whose answer is predictable from its phrasing alone is a shortcut a model
    can learn without reading the message; this table makes that visible.
    """
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: {"positive": Counter(), "negated": Counter()})
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "noul":
                phrasing = "negated" if record["meta"].get("negated") else "positive"
                table[qid][phrasing][record["gold"][qid]] += 1
    result = {}
    for kind, by in table.items():
        n_pos, n_neg = sum(by["positive"].values()), sum(by["negated"].values())
        result[kind] = {
            "n": n_pos + n_neg,
            "negated_share": n_neg / (n_pos + n_neg),
            "positive": {"true": by["positive"]["true"], "false": by["positive"]["false"]},
            "negated": {"true": by["negated"]["true"], "false": by["negated"]["false"]},
        }
    return result


def phrasing_only_accuracy(train_table: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, float]:
    """Accuracy per kind of answering each phrasing with its majority answer in train, without reading the message."""
    accuracy = {}
    for kind, row in table.items():
        if kind not in train_table:
            continue
        correct = 0
        for phrasing in ("positive", "negated"):
            train_counts = train_table[kind][phrasing]
            majority = "true" if train_counts["true"] >= train_counts["false"] else "false"
            correct += row[phrasing][majority]
        accuracy[kind] = correct / row["n"]
    return accuracy
