"""Unseen-schema evaluation sets: AG News, emotion, Banking77 and Yelp (docs/DATA.md section 2, data v1.2).

Evaluation only: none of these datasets or labels appear in training. Labels are
used exactly as the datasets give them; descriptions are the canonical texts from
the description loader. test_emotion also asks an expresses_emotion noul per
record, and test_yelp is the unseen score schema, with five hand-written star
levels that live only here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from jevmark.data.build import assign_phrasings, choice_question, make_record, noul_question, split_rng
from jevmark.data.description_loader import load_descriptions
from jevmark.data.sources import AG_NEWS, BANKING77, EMOTION, YELP, DatasetSource

QUESTION_ID = "label"
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
            distractors = rng.sample([n for n in names if n != gold], int(params["banking77_options"]) - 2)
            options = [(label, descriptions[label].canonical) for label in [gold, *distractors]] + [(OTHER_LABEL, OTHER_DESCRIPTION)]
        else:
            options = [(label, descriptions[label].canonical) for label in names]
        records.append(
            {
                "id": f"{spec.short}-{split}-{len(records):06d}",
                "source": spec.source.id,
                "split": split,
                "state": row["text"],
                "questions": {QUESTION_ID: choice_question(spec.instructions, options, rng)},
                "gold": {QUESTION_ID: gold},
                "meta": {"source_split": "test", "source_index": index, "nouls": {}},
            }
        )
    if split == "test_emotion":
        add_emotion_nouls(records, names, rng)
        assign_phrasings(records, split_rng(int(config["seed"]), f"{split}:phrasing"))
    return [make_record(r["id"], r["source"], r["split"], r["state"], r["questions"], r["gold"], r["meta"]) for r in records]


def add_emotion_nouls(records: list[dict[str, Any]], emotions: list[str], rng) -> None:
    """An expresses_emotion noul per record: exactly half ask the gold emotion, the rest another emotion."""
    asks_gold = set(rng.sample(range(len(records)), len(records) // 2))
    for j, record in enumerate(records):
        gold_emotion = record["gold"][QUESTION_ID]
        asked = gold_emotion if j in asks_gold else rng.choice([e for e in emotions if e != gold_emotion])
        question, gold, info = noul_question("expresses_emotion", asked == gold_emotion, asked, asked_emotion=asked)
        order = [QUESTION_ID, "expresses_emotion"]
        rng.shuffle(order)
        questions = {**record["questions"], "expresses_emotion": question}
        answers = {**record["gold"], "expresses_emotion": gold}
        record["questions"] = {qid: questions[qid] for qid in order}
        record["gold"] = {qid: answers[qid] for qid in order}
        record["meta"]["nouls"]["expresses_emotion"] = info
