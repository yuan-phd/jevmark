"""Unseen-schema evaluation sets: AG News, emotion, Banking77 and Yelp (docs/DATA.md section 2, data v1.3).

Evaluation only: none of these datasets or labels appear in training. Labels are
used exactly as the datasets give them; descriptions are the canonical texts from
the description loader. test_emotion also asks an expresses_emotion noul per
record, and test_yelp is the unseen score schema, with five hand-written star
levels that live only here.

v3 (decision 56, docs/V3_DESIGN.md) adds two Banking77 files built with the same
question construction as test_banking77, each from its own seeded stream: the log
domain, every Banking77 train message left after dedup.dedupe (v3_banking77_train),
and the full Banking77 test split, the 1000 test_banking77 records verbatim followed
by the other 2080 test messages (v3_banking77_test_full). Both get the same share
of JSON states as every v1.3 split. They are written by scripts/build_v3_data.py
and never change the nine v1.3 files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from jevmark.data import dedup
from jevmark.data.build import assign_phrasings, choice_question, make_record, noul_question, split_rng
from jevmark.data.description_loader import load_descriptions
from jevmark.data.form import wrap_states
from jevmark.data.sources import AG_NEWS, BANKING77, EMOTION, YELP, DatasetSource

QUESTION_ID = "label"
from jevmark.data.build import V3_TEST_FULL, V3_TRAIN  # noqa: E402  (re-exported for the v3 builders)
OTHER_LABEL = "other"
OTHER_DESCRIPTION = "None of the listed options"


@dataclass(frozen=True)
class UnseenSet:
    split: str
    short: str
    source: DatasetSource
    instructions: str
    subset: bool  # True: gold plus distractors plus "other"; False: every label


SETS = {
    "test_agnews": UnseenSet("test_agnews", "agnews", AG_NEWS, "Which topic is this news article about?", False),
    "test_emotion": UnseenSet("test_emotion", "emotion", EMOTION, "Which emotion does this message express most strongly?", False),
    "test_banking77": UnseenSet("test_banking77", "banking77", BANKING77, "Which banking request does this message make?", True),
}


YELP_INSTRUCTIONS = "How many stars does this review give the business?"
# Hand-written, level 0 (1 star) to level 4 (5 stars).
YELP_LEVELS = (
    "1 star: very dissatisfied; would not return or recommend",
    "2 stars: dissatisfied, with a few redeeming points",
    "3 stars: mixed or average experience",
    "4 stars: satisfied, with minor complaints",
    "5 stars: very satisfied; an enthusiastic recommendation",
)
YELP_SCALE = "yelp_5_stars"


def build_yelp(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """test_yelp: a seeded, stratified sample of Yelp test reviews, n_per_star per star, one score question each.

    Only reviews of at most yelp_max_chars characters are sampled, so every record
    encodes within the training max_tokens without truncation.
    """
    params = config["unseen"]
    rng = split_rng(int(config["seed"]), "test_yelp")
    part = load_dataset(YELP.id, YELP.config, revision=YELP.revision, split="test")
    names = part.features[YELP.label_column].names
    max_chars = int(params["yelp_max_chars"])
    by_star: dict[int, list[int]] = {star: [] for star in range(len(names))}
    for index, (text, star) in enumerate(zip(part["text"], part[YELP.label_column])):
        if len(text) <= max_chars:
            by_star[star].append(index)
    chosen = sorted(i for star in sorted(by_star) for i in rng.sample(by_star[star], int(params["yelp_per_star"])))
    question = {"type": "score", "instructions": YELP_INSTRUCTIONS, "criteria": list(YELP_LEVELS)}
    records = []
    for index in chosen:
        row = part[index]
        star = int(row[YELP.label_column])
        records.append(
            make_record(
                f"yelp-test_yelp-{len(records):06d}",
                YELP.id,
                "test_yelp",
                row["text"],
                {"stars": question},
                {"stars": star},
                {"source_split": "test", "source_index": index, "label_text": names[star], "scale": YELP_SCALE, "nouls": {}},
            )
        )
    return records


def subset_question(instructions: str, gold: str, names: Sequence[str], descriptions: Mapping[str, Any], n_options: int, rng) -> dict[str, Any]:
    """Gold, n_options - 2 distractors drawn from the other labels, and "other", in a seeded order (test_banking77's construction)."""
    distractors = rng.sample([n for n in names if n != gold], n_options - 2)
    options = [(label, descriptions[label].canonical) for label in [gold, *distractors]] + [(OTHER_LABEL, OTHER_DESCRIPTION)]
    return choice_question(instructions, options, rng)


def build_split(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if split == "test_yelp":
        return build_yelp(config)
    spec = SETS[split]
    params = config["unseen"]
    rng = split_rng(int(config["seed"]), split)
    part = load_dataset(spec.source.id, spec.source.config, revision=spec.source.revision, split="test")
    names = part.features[spec.source.label_column].names
    descriptions = load_descriptions(spec.source.key)
    if set(descriptions) != set(names):
        raise RuntimeError(f"{split}: descriptions do not cover exactly the dataset's labels")
    if OTHER_LABEL in names:
        raise RuntimeError(f"{split}: the dataset already has a label named {OTHER_LABEL!r}")

    indices = sorted(rng.sample(range(part.num_rows), int(params["n_per_set"])))
    records = []
    for index in indices:
        row = part[index]
        gold = names[row[spec.source.label_column]]
        if spec.subset:
            question = subset_question(spec.instructions, gold, names, descriptions, int(params["banking77_options"]), rng)
        else:
            question = choice_question(spec.instructions, [(label, descriptions[label].canonical) for label in names], rng)
        records.append(
            {
                "id": f"{spec.short}-{split}-{len(records):06d}",
                "source": spec.source.id,
                "split": split,
                "state": row["text"],
                "questions": {QUESTION_ID: question},
                "gold": {QUESTION_ID: gold},
                "meta": {"source_split": "test", "source_index": index, "nouls": {}},
            }
        )
    if split == "test_emotion":
        add_emotion_nouls(records, names, rng)
        assign_phrasings(records, split_rng(int(config["seed"]), f"{split}:phrasing"))
    return [make_record(r["id"], r["source"], r["split"], r["state"], r["questions"], r["gold"], r["meta"]) for r in records]


def add_emotion_nouls(records: list[dict[str, Any]], emotions: list[str], rng) -> None:
    """An expresses_emotion noul after the choice: exactly half ask the gold emotion, the rest another emotion.

    The "no" questions ask each emotion exactly as often as the "yes" questions do
    (matched_asks), so the asked emotion says nothing about the answer. Asking a
    uniformly drawn other emotion would not do that: the gold emotions are skewed
    (joy and sadness are most of the test split), so a rare asked emotion would mean no.
    """
    asks_gold = set(rng.sample(range(len(records)), len(records) // 2))
    golds = [record["gold"][QUESTION_ID] for record in records]
    no_records = [j for j in range(len(records)) if j not in asks_gold]
    asked_no = matched_asks([golds[j] for j in no_records], [golds[j] for j in sorted(asks_gold)], rng)
    asked = {**{j: golds[j] for j in asks_gold}, **dict(zip(no_records, asked_no))}
    for j, record in enumerate(records):
        question, gold, info = noul_question("expresses_emotion", asked[j] == golds[j], asked[j], asked_emotion=asked[j])
        # Choice first, then the noul (decision 42).
        record["questions"]["expresses_emotion"] = question
        record["gold"]["expresses_emotion"] = gold
        record["meta"]["nouls"]["expresses_emotion"] = info


def matched_asks(golds: list[str], pool: list[str], rng) -> list[str]:
    """One asked label per gold, never equal to it, using the labels in pool exactly once each.

    The pool is shuffled and dealt in order; a gold that would get itself swaps with an
    earlier assignment where both sides stay different. Fails if no valid dealing exists.
    """
    if len(pool) < len(golds):
        raise ValueError("pool must hold at least one label per gold")
    deck = list(pool)
    rng.shuffle(deck)
    deck = deck[: len(golds)]
    asked: list[str] = []
    for i, gold in enumerate(golds):
        label = deck[i]
        if label == gold:
            swap = next((k for k in rng.sample(range(len(asked)), len(asked)) if asked[k] != gold and golds[k] != label), None)
            if swap is None:
                later = next((k for k in range(i + 1, len(deck)) if deck[k] != gold), None)
                if later is None:
                    raise RuntimeError("cannot assign asked labels that differ from every gold")
                deck[i], deck[later] = deck[later], deck[i]
                label = deck[i]
            else:
                label, asked[swap] = asked[swap], label
        asked.append(label)
    return asked


# v3: the Banking77 log domain and the full Banking77 test split (decision 56)


def banking77_labels() -> tuple[list[str], dict[str, Any]]:
    """Banking77's label names in dataset order and their descriptions, checked as build_split checks them."""
    names = load_dataset(BANKING77.id, BANKING77.config, revision=BANKING77.revision, split="test").features[BANKING77.label_column].names
    descriptions = load_descriptions(BANKING77.key)
    if set(descriptions) != set(names) or OTHER_LABEL in names:
        raise RuntimeError("banking77: descriptions do not cover exactly the dataset's labels, or a label is named other")
    return names, descriptions


def banking77_records(
    split: str,
    rows: Sequence[tuple[int, str, str]],
    source_split: str,
    names: Sequence[str],
    descriptions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Records built as test_banking77's are, one per (source index, text, gold label) row, from the split's own seeded streams.

    Options come from split_rng(seed, split), JSON states from
    split_rng(seed, f"{split}:state_format") with the configured share, as
    form.apply gives every v1.3 split; each record is validated by make_record.
    """
    seed = int(config["seed"])
    rng = split_rng(seed, split)
    n_options = int(config["unseen"]["banking77_options"])
    instructions = SETS["test_banking77"].instructions
    records = [
        {
            "id": f"banking77-{split}-{number:06d}",
            "source": BANKING77.id,
            "split": split,
            "state": text,
            "questions": {QUESTION_ID: subset_question(instructions, gold, names, descriptions, n_options, rng)},
            "gold": {QUESTION_ID: gold},
            "meta": {"source_split": source_split, "source_index": index, "nouls": {}},
        }
        for number, (index, text, gold) in enumerate(rows)
    ]
    wrap_states(records, float(config["state_format"]["p_json"]), split_rng(seed, f"{split}:state_format"))
    return [make_record(r["id"], r["source"], r["split"], r["state"], r["questions"], r["gold"], r["meta"]) for r in records]


def build_v3_train(config: Mapping[str, Any], excluded: frozenset[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The log domain: every Banking77 train message kept by dedup.dedupe against `excluded`, in dataset order, and the dropped rows."""
    names, descriptions = banking77_labels()
    part = load_dataset(BANKING77.id, BANKING77.config, revision=BANKING77.revision, split="train")
    texts, labels = part["text"], part[BANKING77.label_column]
    kept, dropped = dedup.dedupe(texts, excluded)
    for row in dropped:
        row["text"] = texts[row["index"]]
    rows = [(i, texts[i], names[labels[i]]) for i in kept]
    return banking77_records(V3_TRAIN, rows, "train", names, descriptions, config), dropped


def full_test_records(
    test_banking77: Sequence[dict[str, Any]],
    rows: Sequence[tuple[int, str, str]],
    names: Sequence[str],
    descriptions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """test_banking77's records verbatim, then a record for every row whose source index they do not use, from V3_TEST_FULL's own streams."""
    used = {r["meta"]["source_index"] for r in test_banking77}
    if len(used) != len(test_banking77):
        raise RuntimeError("test_banking77 uses a source index twice")
    rest = [row for row in rows if row[0] not in used]
    return [dict(r) for r in test_banking77] + banking77_records(V3_TEST_FULL, rest, "test", names, descriptions, config)


def build_v3_test_full(config: Mapping[str, Any], test_banking77: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full Banking77 test split: the given test_banking77 records verbatim, then the other test messages in dataset order."""
    names, descriptions = banking77_labels()
    part = load_dataset(BANKING77.id, BANKING77.config, revision=BANKING77.revision, split="test")
    rows = [(i, text, names[label]) for i, (text, label) in enumerate(zip(part["text"], part[BANKING77.label_column]))]
    return full_test_records(test_banking77, rows, names, descriptions, config)
