"""Unseen-schema evaluation sets from AG News, emotion and Banking77 (docs/DATA.md section 2).

Evaluation only: none of these datasets or labels appear in training. Labels are
used exactly as the datasets give them; descriptions are the canonical texts from
the description loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from jevmark.data.build import choice_question, make_record, split_rng
from jevmark.data.description_loader import load_descriptions
from jevmark.data.sources import AG_NEWS, BANKING77, EMOTION, DatasetSource

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


def build_split(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
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
            make_record(
                f"{spec.short}-{split}-{len(records):06d}",
                spec.source.id,
                split,
                row["text"],
                {QUESTION_ID: choice_question(spec.instructions, options, rng)},
                {QUESTION_ID: gold},
                {"source_split": "test", "source_index": index},
            )
        )
    return records
