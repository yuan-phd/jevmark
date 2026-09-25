"""Form nouls and state formats (docs/DATA.md section 2, data v1.3, decisions 42 and 43).

Form nouls are label-independent noul questions answered from the state text alone
(its length, its digits, its punctuation), never from a gold label. Which records
get them, which kinds and where they go are drawn at random, independently of the
text and of every gold answer, so their presence and position carry no information.

Thresholds are set per split and source, on the texts of that split's records from
that source: the threshold whose yes share is closest to one half. A kind is used
for a (split, source) only if that share (for a yes or no property, its yes share)
is within max_imbalance of one half; otherwise it is skipped there. Records are
never selected by their answer, because a selection that balances a text property
would tie the form noul's presence to the text and so, through the text, to the
labels.

Placement: each form noul goes before the record's first gold-dependent question
with probability p_before_first, at a uniform slot among the form nouls already
there, and otherwise at a uniform slot after it. A fixed probability keeps "a form
noul precedes the choice or score question" independent of how many gold-dependent
questions follow, which would otherwise reveal, for example, a neutral SST-5 record.

State formats: a seeded share of every split wraps the state text in a JSON object
with one to three label-independent fields (channel, timestamp, ids, locale) plus
the text field; the rest stay plain text. Form answers are always computed on the
text itself, never on the JSON rendering.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jevmark.data.build import assign_phrasings, make_record, noul_question, split_rng, state_text


@dataclass(frozen=True)
class FormKind:
    name: str
    measure: Callable[[str], int]  # a count, or 0 and 1 for a yes or no property
    threshold: bool  # True: yes means measure > threshold; False: yes means measure == 1


def _longest_word(text: str) -> int:
    return max((len(w) for w in re.findall(r"[A-Za-z]+", text)), default=0)


FORM_KINDS = {
    k.name: k
    for k in (
        FormKind("word_count_over", lambda t: len(t.split()), True),
        FormKind("char_count_over", len, True),
        FormKind("longest_word_over", _longest_word, True),
        FormKind("contains_number", lambda t: int(bool(re.search(r"[0-9]", t))), False),
        FormKind("contains_comma", lambda t: int("," in t), False),
        FormKind("ends_with_question_mark", lambda t: int(t.rstrip().endswith("?")), False),
    )
}
FORM_KINDS_ORDER = {name: i for i, name in enumerate(FORM_KINDS)}

# Label-independent JSON fields; values are drawn at random, never from the record.
STATE_FIELDS = ("channel", "timestamp", "message_id", "user_id", "thread_id", "locale")
CHANNELS = ("chat", "email", "sms", "web", "app", "phone")
LOCALES = ("en-US", "en-GB", "en-AU", "en-CA", "en-IE", "en-NZ")
TEXT_FIELD = "text"


def best_threshold(values: Sequence[int]) -> tuple[int, float]:
    """The threshold t whose share of values > t is closest to one half, and that share."""
    ordered = sorted(set(values))
    n = len(values)
    counts: dict[int, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    above, best = n, (ordered[0] - 1, 1.0)
    for t in ordered:
        above -= counts[t]
        share = above / n
        if abs(share - 0.5) < abs(best[1] - 0.5):
            best = (t, share)
    return best


def kind_settings(texts: Sequence[str], max_imbalance: float) -> dict[str, dict[str, Any]]:
    """Per form kind: threshold (or None), yes share on the texts, and whether the kind is used."""
    settings = {}
    for kind in FORM_KINDS.values():
        values = [kind.measure(t) for t in texts]
        if kind.threshold:
            threshold, share = best_threshold(values)
        else:
            threshold, share = None, sum(values) / len(values)
        settings[kind.name] = {"threshold": threshold, "yes_share": share, "used": abs(share - 0.5) <= max_imbalance}
    return settings


def split_settings(records: Sequence[Mapping[str, Any]], max_imbalance: float) -> dict[str, dict[str, dict[str, Any]]]:
    """Source -> kind -> settings, on the texts of these records (one split)."""
    texts: dict[str, list[str]] = defaultdict(list)
    for record in records:
        texts[record["source"]].append(state_text(record))
    return {source: kind_settings(t, max_imbalance) for source, t in sorted(texts.items())}


def answer(kind: str, text: str, threshold: int | None) -> bool:
    value = FORM_KINDS[kind].measure(text)
    return value > threshold if threshold is not None else bool(value)


def add_form_nouls(
    records: list[dict[str, Any]], settings: Mapping[str, Mapping[str, Mapping[str, Any]]], p_before_first: float, rng: random.Random, phrasing_rng: random.Random
) -> None:
    """Give each record 0, 1 or 2 form nouls of distinct used kinds, in place, then phrase them (assign_phrasings)."""
    for record in records:
        used = [k for k, s in settings[record["source"]].items() if s["used"]]
        kinds = rng.sample(used, min(rng.choice((0, 1, 2)), len(used)))
        before: list[str] = []
        after = list(record["questions"])  # the gold-dependent questions, first one at index 0
        for kind in kinds:
            threshold = settings[record["source"]][kind]["threshold"]
            question, gold, info = noul_question(kind, answer(kind, state_text(record), threshold), None if threshold is None else str(threshold), form=True, threshold=threshold)
            record["questions"][kind], record["gold"][kind], record["meta"]["nouls"][kind] = question, gold, info
            if rng.random() < p_before_first:
                before.insert(rng.randint(0, len(before)), kind)
            else:
                after.insert(rng.randint(1, len(after)), kind)
        order = before + after
        record["questions"] = {qid: record["questions"][qid] for qid in order}
        record["gold"] = {qid: record["gold"][qid] for qid in order}
    assign_phrasings(records, phrasing_rng, kinds=set(FORM_KINDS))


def wrap_states(records: list[dict[str, Any]], share: float, rng: random.Random) -> None:
    """Wrap the state of exactly round(share * n) records in a JSON object, in place; meta records the format."""
    wrapped = set(rng.sample(range(len(records)), round(share * len(records))))
    for j, record in enumerate(records):
        if j not in wrapped:
            record["meta"]["state_format"], record["meta"]["state_fields"] = "plain", []
            continue
        fields = rng.sample(STATE_FIELDS, rng.randint(1, 3))
        keys = list(fields)
        keys.insert(rng.randint(0, len(keys)), TEXT_FIELD)
        values = {name: _field_value(name, rng) for name in fields}
        values[TEXT_FIELD] = record["state"]
        record["state"] = {key: values[key] for key in keys}
        record["meta"]["state_format"], record["meta"]["state_fields"] = "json", keys


def _field_value(name: str, rng: random.Random) -> str:
    if name == "channel":
        return rng.choice(CHANNELS)
    if name == "locale":
        return rng.choice(LOCALES)
    if name == "timestamp":
        return (
            f"{rng.randint(2019, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        )
    if name == "message_id":
        return f"msg-{rng.getrandbits(32):08x}"
    if name == "user_id":
        return f"u{rng.randint(0, 999999):06d}"
    return f"t-{rng.getrandbits(24):06x}"  # thread_id


def apply(split: str, records: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Form nouls, then state formats, for one split, each from its own seeded stream; returns the form settings used."""
    seed = int(config["seed"])
    settings = split_settings(records, float(config["form"]["max_imbalance"]))
    add_form_nouls(records, settings, float(config["form"]["p_before_first"]), split_rng(seed, f"{split}:form"), split_rng(seed, f"{split}:form:phrasing"))
    wrap_states(records, float(config["state_format"]["p_json"]), split_rng(seed, f"{split}:state_format"))
    for record in records:
        make_record(record["id"], record["source"], record["split"], record["state"], record["questions"], record["gold"], record["meta"])
    return settings
