"""The fixed evaluation subset every baseline comparison uses (task 1.8).

500 records per split, drawn with the fast-cycle sampler (jevmark.sampling) and the
same seed evaluate.py --limit uses, random.Random(f"limit:{split}"), so the subset
is exactly what `evaluate.py --limit 500` evaluates. A 200-record sub-subset per
split, drawn from the 500 with the same sampler and a seed of its own, serves the
optional stronger API model. The record ids are written once to
data/baseline_subset.json and committed, together with the sha256 of each data file
they were drawn from; loading a split through the subset fails if the file changed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jevmark.sampling import sample_records

REPO = Path(__file__).resolve().parents[2]
SUBSET_PATH = REPO / "data" / "baseline_subset.json"
PER_SPLIT = 500
SUB_PER_SPLIT = 200


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def build_subset(data_dir: Path, splits: Sequence[str], per_split: int = PER_SPLIT, sub_per_split: int = SUB_PER_SPLIT) -> dict[str, Any]:
    subset: dict[str, Any] = {
        "per_split": per_split,
        "sub_per_split": sub_per_split,
        "sampler": 'jevmark.sampling.sample_records, random.Random(f"limit:{split}"); sub-subset from the subset with random.Random(f"baseline-sub:{split}")',
        "data_files_sha256": {},
        "splits": {},
        "sub_splits": {},
    }
    for split in splits:
        records, digest = read_jsonl(data_dir / f"{split}.jsonl")
        chosen = sample_records(records, per_split, random.Random(f"limit:{split}"))
        sub = sample_records(chosen, sub_per_split, random.Random(f"baseline-sub:{split}"))
        subset["data_files_sha256"][f"{split}.jsonl"] = digest
        subset["splits"][split] = [r["id"] for r in chosen]
        subset["sub_splits"][split] = [r["id"] for r in sub]
    return subset


def load_subset(path: Path = SUBSET_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def subset_ids(subset: dict[str, Any], split: str, sub: bool = False) -> list[str]:
    return list(subset["sub_splits" if sub else "splits"][split])


def subset_records(data_dir: Path, subset: dict[str, Any], split: str, sub: bool = False) -> tuple[list[dict[str, Any]], str]:
    """The split's subset records in file order, and the file's sha256; RuntimeError if the file is not the one the subset was drawn from."""
    records, digest = read_jsonl(data_dir / f"{split}.jsonl")
    expected = subset["data_files_sha256"][f"{split}.jsonl"]
    if digest != expected:
        raise RuntimeError(f"{split}.jsonl has sha256 {digest}, but the baseline subset was drawn from {expected}; rebuild the data at the frozen version")
    wanted = set(subset_ids(subset, split, sub))
    chosen = [r for r in records if r["id"] in wanted]
    if len(chosen) != len(wanted):
        raise RuntimeError(f"{split}: {len(wanted) - len(chosen)} subset ids are missing from the data file")
    return chosen, digest
