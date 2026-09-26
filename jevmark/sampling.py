"""Seeded, stratified record samples: the --limit sample of evaluate.py and the baseline subset (task 1.8).

evaluate.py --limit N draws sample_records(records, N, random.Random(f"limit:{split}"));
the baseline subset (jevmark/baselines/subset.py) draws the same sample with N 500,
so a --limit 500 evaluation and the baselines see the same records.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split total across groups in proportion to their sizes (largest remainder), at least one per group while total allows."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    counts = {k: min(sizes[k], max(1, int(x))) for k, x in exact.items()}
    for k in sorted(sizes, key=lambda k: exact[k] - int(exact[k]), reverse=True):
        if sum(counts.values()) >= total:
            break
        if counts[k] < sizes[k]:
            counts[k] += 1
    while sum(counts.values()) > total:  # the "at least one" floor overshot
        k = max(counts, key=lambda k: counts[k] - exact[k])
        counts[k] -= 1
    return counts


def sample_records(records: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    """A seeded sample of `limit` records, in file order (docs/KAGGLE.md section 8).

    Records are dataset-ordered, so the first N would cover only a few CLINC intents
    and no out-of-scope utterance. Instead the limit is split across sources in
    proportion to their size; within CLINC, out-of-scope utterances get their
    proportional share (at least one) and the rest is dealt round-robin over the
    intents in a seeded order, so the sample holds as many intents as it can;
    other sources are sampled uniformly.
    """
    if limit >= len(records):
        return records
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_source[record["source"]].append(i)
    chosen: list[int] = []
    for source, count in _allocate({s: len(ix) for s, ix in by_source.items()}, limit).items():
        indices = by_source[source]
        intents: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            intent = records[i]["meta"].get("gold_intent")
            if intent is not None:
                intents[intent].append(i)
        if not intents:
            chosen += rng.sample(indices, count)
            continue
        oos = intents.pop("oos", [])
        n_oos = min(len(oos), max(1, round(count * len(oos) / len(indices)))) if oos else 0
        chosen += rng.sample(oos, n_oos)
        pools = {k: rng.sample(v, len(v)) for k, v in sorted(intents.items())}
        order = sorted(pools)
        rng.shuffle(order)
        taken, depth = 0, 0
        while taken < count - n_oos:
            progressed = False
            for intent in order:
                if depth < len(pools[intent]) and taken < count - n_oos:
                    chosen.append(pools[intent][depth])
                    taken += 1
                    progressed = True
            if not progressed:
                break
            depth += 1
    return [records[i] for i in sorted(chosen)]
