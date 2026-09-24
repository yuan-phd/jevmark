"""Build every split in memory (docs/DATA.md section 2); scripts/build_data.py and the tests both use this."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jevmark.data import sst5, unseen
from jevmark.data.build import SPLITS
from jevmark.data.clinc import ClincBuilder


@dataclass(frozen=True)
class BuiltData:
    splits: dict[str, list[dict[str, Any]]]
    held_out: tuple[str, ...]
    seen: tuple[str, ...]


def build_all(config: Mapping[str, Any], splits: Sequence[str] = SPLITS) -> BuiltData:
    """Records per split. train and valid hold CLINC records followed by SST-5 records."""
    clinc = ClincBuilder(config)
    built: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        if split in ("train", "valid"):
            built[split] = clinc.build_split(split) + sst5.build_split(split, config)
        elif split in ("test_indomain", "test_unseen_intents"):
            built[split] = clinc.build_split(split)
        elif split == "test_sst5":
            built[split] = sst5.build_split(split, config)
        else:
            built[split] = unseen.build_split(split, config)
    return BuiltData(built, tuple(clinc.held_out), tuple(clinc.seen))
