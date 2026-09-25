"""Form nouls and state formats (docs/DATA.md section 2, data v1.3, decision 42).

Form nouls are label-independent noul questions answered from the state text alone
(its length, its digits, its punctuation), never from a gold label, so they may sit
anywhere in a record, before or after the gold-dependent questions. Kinds with a
threshold get it per source, set on the source's full pinned texts so the kind is
as close to balanced as it can be; a kind whose best split of a source is outside
[0.5 - max_imbalance, 0.5 + max_imbalance] is skipped for that source. Within each
split, source and kind, assignments are then trimmed to exactly as many yes as no
answers.

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
from functools import cache
from typing import Any

from datasets import load_dataset

from jevmark.data.build import assign_phrasings, make_record, noul_question, split_rng, state_text
from jevmark.data.sources import AG_NEWS, BANKING77, CLINC, EMOTION, SST5, YELP, DatasetSource


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

# Record source field -> (dataset, HF splits whose texts set the thresholds).
SOURCE_TEXTS: dict[str, tuple[DatasetSource, tuple[str, ...]]] = {
    f"{CLINC.id}/{CLINC.config}": (CLINC, ("train", "validation", "test")),
    SST5.id: (SST5, ("train", "validation", "test")),
    AG_NEWS.id: (AG_NEWS, ("test",)),
    EMOTION.id: (EMOTION, ("test",)),
    BANKING77.id: (BANKING77, ("test",)),
    YELP.id: (YELP, ("test",)),
}

# Label-independent JSON fields; values are drawn at random, never from the record.
STATE_FIELDS = ("channel", "timestamp", "message_id", "user_id", "thread_id", "locale")
CHANNELS = ("chat", "email", "sms", "web", "app", "phone")
LOCALES = ("en-US", "en-GB", "en-AU", "en-CA", "en-IE", "en-NZ")
TEXT_FIELD = "text"


@cache
def source_texts(source: str, max_chars: int | None = None) -> tuple[str, ...]:
    dataset, hf_splits = SOURCE_TEXTS[source]
    texts: list[str] = []
    for hf_split in hf_splits:
        texts += load_dataset(dataset.id, dataset.config, revision=dataset.revision, split=hf_split)["text"]
    return tuple(t for t in texts if max_chars is None or len(t) <= max_chars)


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


def form_settings(sources: Sequence[str], config: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Source -> kind -> settings, from each source's full pinned texts (independent of which splits are built)."""
    max_imbalance = float(config["form"]["max_imbalance"])
    yelp_max = int(config["unseen"]["yelp_max_chars"])
    return {s: kind_settings(source_texts(s, yelp_max if s == YELP.id else None), max_imbalance) for s in sources}


def answer(kind: str, text: str, threshold: int | None) -> bool:
    value = FORM_KINDS[kind].measure(text)
    return value > threshold if threshold is not None else bool(value)


def add_form_nouls(records: list[dict[str, Any]], settings: Mapping[str, Mapping[str, Mapping[str, Any]]], rng: random.Random, phrasing_rng: random.Random) -> None:
    """Give each record 0, 1 or 2 form nouls at random positions, in place, then phrase them.

    Every record draws how many and which of its source's used kinds; within each
    (source, kind), assignments on the larger answer side are dropped at random until
    yes and no are equal, so the underlying answers are exactly balanced.
    """
    tentative: dict[tuple[str, str], dict[bool, list[int]]] = defaultdict(lambda: {True: [], False: []})
    for j, record in enumerate(records):
        used = [k for k, s in settings[record["source"]].items() if s["used"]]
        n = rng.choice((0, 1, 2))
        for kind in rng.sample(used, min(n, len(used))):
            threshold = settings[record["source"]][kind]["threshold"]
            tentative[(record["source"], kind)][answer(kind, state_text(record), threshold)].append(j)
    chosen: dict[int, list[str]] = defaultdict(list)
    for (source, kind), sides in sorted(tentative.items()):
        keep = min(len(sides[True]), len(sides[False]))
        for side in (True, False):
            for j in sorted(rng.sample(sides[side], keep)):
                chosen[j].append(kind)
    for j, record in enumerate(records):
        order = list(record["questions"])
        for kind in sorted(chosen.get(j, []), key=lambda k: FORM_KINDS_ORDER[k]):
            threshold = settings[record["source"]][kind]["threshold"]
            slot = None if threshold is None else str(threshold)
            question, gold, info = noul_question(kind, answer(kind, state_text(record), threshold), slot, form=True, threshold=threshold)
            record["questions"][kind] = question
            record["gold"][kind] = gold
            record["meta"]["nouls"][kind] = info
            order.insert(rng.randint(0, len(order)), kind)
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


def apply(split: str, records: list[dict[str, Any]], settings: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Form nouls, then state formats, for one split, each from its own seeded stream."""
    seed = int(config["seed"])
    add_form_nouls(records, settings, split_rng(seed, f"{split}:form"), split_rng(seed, f"{split}:form:phrasing"))
    wrap_states(records, float(config["state_format"]["p_json"]), split_rng(seed, f"{split}:state_format"))
    for record in records:
        make_record(record["id"], record["source"], record["split"], record["state"], record["questions"], record["gold"], record["meta"])
