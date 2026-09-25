"""Build every split in memory (docs/DATA.md section 2); scripts/build_data.py and the tests both use this."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jevmark.data import dedup, form, sst5, unseen
from jevmark.data.build import SPLITS
from jevmark.data.clinc import ClincBuilder


@dataclass(frozen=True)
class BuiltData:
    splits: dict[str, list[dict[str, Any]]]
    held_out: tuple[str, ...]
    seen: tuple[str, ...]
    form_settings: dict[str, dict[str, dict[str, dict[str, Any]]]]  # split -> source -> form kind -> threshold, yes share, used


def build_all(config: Mapping[str, Any], splits: Sequence[str] = SPLITS) -> BuiltData:
    """Records per split. train and valid hold CLINC records followed by SST-5 records.

    After each split is built, form nouls and state formats are added (form.apply),
    with thresholds from that split's own texts, so a split's records do not depend
    on which other splits are built.
    """
    clinc = ClincBuilder(config)
    built: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        if split in ("train", "valid"):
            # Drop texts that also occur where the test splits are drawn from (decision 43).
            excluded = dedup.test_texts(tuple(clinc.held_out))
            built[split] = clinc.build_split(split, excluded) + sst5.build_split(split, config, excluded)
        elif split in ("test_indomain", "test_unseen_intents"):
            built[split] = clinc.build_split(split)
        elif split == "test_sst5":
            built[split] = sst5.build_split(split, config)
        else:
            built[split] = unseen.build_split(split, config)
    settings = {split: form.apply(split, records, config) for split, records in built.items()}
    return BuiltData(built, tuple(clinc.held_out), tuple(clinc.seen), settings)
