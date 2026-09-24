"""SST-5 -> records with a score question, and a sentiment noul on non-neutral records (docs/DATA.md section 2, data v1.2).

Two hand-written scales live only here: the 5-level scale, and a 3-level scale
(negative, neutral or mixed, positive) used on a seeded 30 percent of records in
every split, with labels 0 and 1 mapped to 0, 2 to 1, and 3 and 4 to 2. meta.scale
records which scale a record uses. Every non-neutral record (labels 0, 1, 3, 4)
also gets an is_positive or is_negative noul question, chosen at random; neutral
records keep only their score question.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import load_dataset

from jevmark.data.build import assign_phrasings, make_record, noul_question, split_rng
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
# Hand-written 3-level scale, and the map from the 5 dataset labels onto it.
LEVELS_3 = (
    "Negative: unfavourable or critical",
    "Neutral or mixed: neither clearly positive nor clearly negative",
    "Positive: favourable or approving",
)
TO_3_LEVELS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
SCALE_5, SCALE_3 = "sst5_5_levels", "sst5_3_levels"
NEUTRAL = 2
# SST-5 label_text for each level, checked against the data so a relabelled revision fails loudly.
LABEL_TEXTS = ("very negative", "negative", "neutral", "positive", "very positive")
HF_SPLITS = {"train": "train", "valid": "validation", "test_sst5": "test"}
SOURCE = SST5.id


def build_split(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = split_rng(int(config["seed"]), f"{split}:sst5")
    part = load_dataset(SST5.id, SST5.config, revision=SST5.revision, split=HF_SPLITS[split])
    three_level = set(rng.sample(range(part.num_rows), round(float(config["sst5"]["p_three_levels"]) * part.num_rows)))
    records = []
    for index, row in enumerate(part):
        label = int(row["label"])
        if row["label_text"] != LABEL_TEXTS[label]:
            raise RuntimeError(f"SST-5 {split} row {index}: label {label} has label_text {row['label_text']!r}")
        if index in three_level:
            score = {"type": "score", "instructions": INSTRUCTIONS, "criteria": list(LEVELS_3)}
            level, scale = TO_3_LEVELS[label], SCALE_3
        else:
            score = {"type": "score", "instructions": INSTRUCTIONS, "criteria": list(LEVELS)}
            level, scale = label, SCALE_5
        questions, gold, nouls = {QUESTION_ID: score}, {QUESTION_ID: level}, {}
        if label != NEUTRAL:
            kind = rng.choice(("is_positive", "is_negative"))
            answer = label > NEUTRAL if kind == "is_positive" else label < NEUTRAL
            noul_q, noul_gold, info = noul_question(kind, answer)
            order = [QUESTION_ID, kind]
            rng.shuffle(order)
            questions = {qid: {QUESTION_ID: score, kind: noul_q}[qid] for qid in order}
            gold = {qid: {QUESTION_ID: level, kind: noul_gold}[qid] for qid in order}
            nouls = {kind: info}
        records.append(
            {
                "id": f"sst5-{split}-{index:06d}",
                "source": SOURCE,
                "split": split,
                "state": row["text"],
                "questions": questions,
                "gold": gold,
                "meta": {"source_split": HF_SPLITS[split], "source_index": index, "label_text": row["label_text"], "scale": scale, "nouls": nouls},
            }
        )
    assign_phrasings(records, split_rng(int(config["seed"]), f"{split}:sst5:phrasing"))
    return [make_record(r["id"], r["source"], r["split"], r["state"], r["questions"], r["gold"], r["meta"]) for r in records]
