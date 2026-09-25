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
    "test_yelp",
)


def split_rng(seed: int, name: str) -> random.Random:
    """One generator per split (or purpose), so rebuilding one never changes another."""
    return random.Random(f"{seed}:{name}")


def state_text(record: Mapping[str, Any]) -> str:
    """The record's text: the state itself, or its text field when the state is a JSON object (data v1.3)."""
    state = record["state"]
    return state["text"] if isinstance(state, dict) else state


def make_record(
    record_id: str,
    source: str,
    split: str,
    state: str | dict[str, Any],
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
    """Ids of records whose gold intent, any option label or any asked intent is a held-out intent."""
    held = set(held_out)
    leaks = []
    for record in records:
        labels = {label for q in record["questions"].values() if q["type"] == "choice" for label in q["criteria"]}
        asked = {n.get("asked_intent") for n in record["meta"].get("nouls", {}).values()}
        if record["meta"].get("gold_intent") in held or labels & held or asked & held:
            leaks.append(record["id"])
    return leaks


def is_form(record: Mapping[str, Any], qid: str) -> bool:
    """True for a form noul (label-independent, data v1.3); every other question depends on the record's gold."""
    return bool(record["meta"].get("nouls", {}).get(qid, {}).get("form"))


def order_violations(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Ids of records that break the data v1.3 order rule (decision 42).

    Among a record's gold-dependent questions (all but form nouls), the choice or
    score question comes first and at most one noul follows it. Form nouls may be
    anywhere.
    """
    bad = []
    for record in records:
        dependent = [q["type"] for qid, q in record["questions"].items() if not is_form(record, qid)]
        if dependent[0] == "noul" or dependent.count("noul") > 1 or len(dependent) - dependent.count("noul") != 1:
            bad.append(record["id"])
    return bad


# Noul phrasing (decisions 40 and 41)


def noul_question(kind: str, gold: bool, slot: str | None = None, **meta: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """A noul question in template 0, positive phrasing, with its underlying gold; assign_phrasings sets the final phrasing."""
    from jevmark.data.negation import render

    question = {"type": "noul", "instructions": render(kind, 0, False, slot)}
    return question, "true" if gold else "false", {"template": 0, "negated": False, "slot": slot, **meta}


def assign_phrasings(records: list[dict[str, Any]], rng: random.Random, kinds: set[str] | None = None) -> None:
    """Give every noul question (of the given kinds, default all) its final template and polarity, in place.

    Within each kind and underlying answer, questions get (template, negated) pairs
    from consecutive shuffled blocks that hold each pair once. Every phrasing then
    has about the same share of underlying yes and no answers as the kind, and
    exactly half of each group is negated, which flips its gold answer.
    """
    from jevmark.data.negation import TEMPLATES, render

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "noul" and (kinds is None or qid in kinds):
                groups[(qid, record["gold"][qid])].append((record, qid))
    for (kind, underlying), members in sorted(groups.items()):
        pairs = [(t, negated) for t in range(len(TEMPLATES[kind])) for negated in (False, True)]
        sequence: list[tuple[int, bool]] = []
        while len(sequence) < len(members):
            block = list(pairs)
            rng.shuffle(block)
            sequence += block
        for (record, qid), (template, negated) in zip(members, sequence):
            info = record["meta"]["nouls"][qid]
            record["questions"][qid]["instructions"] = render(kind, template, negated, info["slot"])
            record["gold"][qid] = {"true": "false", "false": "true"}[underlying] if negated else underlying
            info["template"], info["negated"] = template, negated


def noul_phrasing(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per noul kind: n, yes share, negated share, gold counts per phrasing, and phrasing-only accuracy.

    A phrasing is a (template, polarity) pair. Phrasing-only accuracy answers each
    phrasing with its own majority answer in the same records, the best any rule
    that ignores the message can do; a kind well above 50 percent has a shortcut.
    """
    table: dict[str, dict[tuple[int, bool], Counter]] = defaultdict(lambda: defaultdict(Counter))
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "noul":
                info = record["meta"]["nouls"][qid]
                table[qid][(info["template"], info["negated"])][record["gold"][qid]] += 1
    result = {}
    for kind, phrasings in table.items():
        n = sum(sum(c.values()) for c in phrasings.values())
        yes = sum(c["true"] for c in phrasings.values())
        negated = sum(sum(c.values()) for (_, neg), c in phrasings.items() if neg)
        result[kind] = {
            "n": n,
            "yes_share": yes / n,
            "negated_share": negated / n,
            "phrasings": {f"t{t}{'-neg' if neg else ''}": {"true": c["true"], "false": c["false"]} for (t, neg), c in sorted(phrasings.items())},
            "phrasing_only_accuracy": sum(max(c["true"], c["false"]) for c in phrasings.values()) / n,
        }
    return result


def score_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[int, int]]:
    """Scale name (meta.scale) -> Counter of gold levels, over score questions."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for qid, question in record["questions"].items():
            if question["type"] == "score":
                counts[record["meta"].get("scale", "unknown")][record["gold"][qid]] += 1
    return {scale: dict(sorted(c.items())) for scale, c in sorted(counts.items())}
