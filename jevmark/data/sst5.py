"""SST-5 -> records with one score question each (docs/DATA.md section 2).

The five level descriptions are written by hand and live only here.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from jevmark.data.build import make_record
from jevmark.data.sources import SST5

QUESTION_ID = "sentiment"
INSTRUCTIONS = "How positive is the sentiment of this text?"
# Hand-written, level 0 (very negative) to level 4 (very positive).
LEVELS = (
    "Very negative: harsh criticism, strong dislike or deep disappointment",
    "Negative: mostly unfavourable, with clear criticism",
    "Neutral or mixed: neither clearly positive nor clearly negative",
    "Positive: mostly favourable, with clear praise",
    "Very positive: enthusiastic, glowing praise",
)
# SST-5 label_text for each level, checked against the data so a relabelled revision fails loudly.
LABEL_TEXTS = ("very negative", "negative", "neutral", "positive", "very positive")
HF_SPLITS = {"train": "train", "valid": "validation", "test_sst5": "test"}
SOURCE = SST5.id


def build_split(split: str) -> list[dict[str, Any]]:
    part = load_dataset(SST5.id, SST5.config, revision=SST5.revision, split=HF_SPLITS[split])
    question = {"type": "score", "instructions": INSTRUCTIONS, "criteria": list(LEVELS)}
    records = []
    for index, row in enumerate(part):
        level = int(row["label"])
        if row["label_text"] != LABEL_TEXTS[level]:
            raise RuntimeError(f"SST-5 {split} row {index}: label {level} has label_text {row['label_text']!r}")
        records.append(
            make_record(
                f"sst5-{split}-{index:06d}",
                SOURCE,
                split,
                row["text"],
                {QUESTION_ID: question},
                {QUESTION_ID: level},
                {"source_split": HF_SPLITS[split], "source_index": index, "label_text": row["label_text"]},
            )
        )
    return records
