"""Pinned dataset sources (docs/DATA.md section 3). Every loader reads ids and revisions from here."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import ClassLabel, load_dataset_builder


@dataclass(frozen=True)
class DatasetSource:
    key: str
    id: str
    config: str | None
    revision: str
    label_column: str

    @property
    def pinned(self) -> str:
        suffix = f"/{self.config}" if self.config else ""
        return f"{self.id}{suffix}@{self.revision}"


CLINC = DatasetSource("clinc", "clinc/clinc_oos", "plus", "155b9c710419136e17307b80d0a13e68cd46b4ec", "intent")
SST5 = DatasetSource("sst5", "SetFit/sst5", None, "e51bdcd8cd3a30da231967c1a249ba59361279a3", "label")
AG_NEWS = DatasetSource("ag_news", "fancyzhx/ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4", "label")
EMOTION = DatasetSource("emotion", "dair-ai/emotion", None, "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "label")
BANKING77 = DatasetSource("banking77", "legacy-datasets/banking77", None, "f54121560de48f2852f90be299010d1d6dc612ec", "label")

SOURCES = {s.key: s for s in (CLINC, SST5, AG_NEWS, EMOTION, BANKING77)}

CLINC_OUT_OF_SCOPE = "oos"


def label_names(source: DatasetSource) -> list[str]:
    """ClassLabel names from the pinned dataset's metadata; no data files are downloaded."""
    builder = load_dataset_builder(source.id, source.config, revision=source.revision)
    feature = builder.info.features[source.label_column] if builder.info.features else None
    if not isinstance(feature, ClassLabel):
        raise RuntimeError(f"{source.pinned}: column {source.label_column!r} is not a ClassLabel")
    return list(feature.names)
