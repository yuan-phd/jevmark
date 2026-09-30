"""Texts that train and valid must not contain (docs/DATA.md section 8, decision 43).

A few texts occur in both the training and the test splits of the source datasets
(for example "where did you grow up" in CLINC150, "no" in SST-5). train and valid
drop every utterance whose normalised text occurs in any text that a test split can
be drawn from, so the test splits keep their pinned contents and the duplicate check
(scripts/check_duplicates.py) finds no exact match. The set is built from whole
source splits, not from the sampled test sets, so it does not depend on which splits
a build includes.

The v3 log domain (decision 56) uses the same set: dedupe() drops every Banking77
train message whose normalised text is in it, and keeps only the first copy of a
text that repeats within the split, recording every dropped row and why.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import cache
from typing import Any

from datasets import load_dataset

from jevmark.data.sources import AG_NEWS, BANKING77, CLINC, EMOTION, SST5, YELP


def normalise(text: str) -> str:
    """NFKC, lower case, every run of characters other than letters and digits as one space, edges stripped."""
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


@cache
def test_texts(held_out: tuple[str, ...]) -> frozenset[str]:
    """Normalised texts of every source split a test split draws from.

    CLINC test, and every utterance of the held-out intents in any CLINC split
    (test_unseen_intents); SST-5 test; the test splits of AG News, emotion, Banking77
    and Yelp.
    """
    texts: list[str] = []
    clinc = load_dataset(CLINC.id, CLINC.config, revision=CLINC.revision)
    names = clinc["train"].features[CLINC.label_column].names
    held = set(held_out)
    for hf_split, part in clinc.items():
        texts += [row["text"] for row in part if hf_split == "test" or names[row[CLINC.label_column]] in held]
    for source in (SST5, AG_NEWS, EMOTION, BANKING77, YELP):
        texts += load_dataset(source.id, source.config, revision=source.revision, split="test")["text"]
    return frozenset(normalise(t) for t in texts)


def keep(texts: Iterable[str], excluded: frozenset[str]) -> list[bool]:
    return [normalise(t) not in excluded for t in texts]


def dedupe(texts: Sequence[str], excluded: frozenset[str]) -> tuple[list[int], list[dict[str, Any]]]:
    """The indices to keep, in order, and every dropped index with its reason.

    A text is dropped as "test_text" when its normalised form is in `excluded`, and as
    "repeat" when an earlier kept text has the same normalised form ("first" names
    that earlier index), so the first copy of a repeated text is the one kept.
    """
    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    first: dict[str, int] = {}
    for index, text in enumerate(texts):
        key = normalise(text)
        if key in excluded:
            dropped.append({"index": index, "reason": "test_text"})
        elif key in first:
            dropped.append({"index": index, "reason": "repeat", "first": first[key]})
        else:
            first[key] = index
            kept.append(index)
    return kept, dropped
