"""Check that every pinned dataset in jevmark/data/sources.py loads at its pinned revision.

Prints one block per dataset with splits, sizes and label names, notes when the
Hub repo has moved past the pin, and exits non-zero if any dataset fails to load.

TASKS.md names clinc_oos and PolyAI/banking77. datasets 5 accepts only namespaced
ids, and the Hub redirects clinc_oos to clinc/clinc_oos. PolyAI/banking77 is a
loading script with no parquet conversion; its mirror legacy-datasets/banking77
was checked row by row against the original PolyAI CSVs (docs/DATA.md section 3).
"""

from __future__ import annotations

import sys

import datasets
import huggingface_hub
from datasets import ClassLabel, load_dataset
from huggingface_hub import HfApi

from jevmark.data.sources import SOURCES, DatasetSource


def check(source: DatasetSource, api: HfApi) -> bool:
    print(f"{source.id} config={source.config}")
    try:
        current = api.dataset_info(source.id).sha
        note = "" if current == source.revision else f"  (Hub is now at {current}; the pin stays)"
        print(f"  pinned   {source.revision}{note}")
        dataset = load_dataset(source.id, source.config, revision=source.revision)
    except Exception as err:  # noqa: BLE001 - report every failure, then continue
        print(f"  FAILED   {type(err).__name__}: {str(err).splitlines()[0][:200]}")
        return False
    for split, part in dataset.items():
        print(f"  split    {split}: {part.num_rows} rows, columns {part.column_names}")
    feature = next(iter(dataset.values())).features[source.label_column]
    if isinstance(feature, ClassLabel):
        names = feature.names
        shown = names if len(names) <= 12 else names[:6] + ["..."] + names[-3:]
        print(f"  labels   {source.label_column}: {len(names)} classes {shown}")
    else:
        print(f"  labels   {source.label_column}: {feature}")
    return True


def main() -> int:
    print(f"datasets {datasets.__version__}, huggingface_hub {huggingface_hub.__version__}\n")
    api = HfApi()
    failures = [s.id for s in SOURCES.values() if not check(s, api)]
    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all datasets loaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
