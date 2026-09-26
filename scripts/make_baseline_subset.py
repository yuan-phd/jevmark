"""Write data/baseline_subset.json once: the fixed baseline evaluation subset (task 1.8).

    uv run python scripts/make_baseline_subset.py

500 records per split with the fast-cycle sampler (jevmark/baselines/subset.py),
plus the 200-record sub-subset, with the sha256 of each data file. The file is
committed; the script refuses to overwrite it unless --force is given, because
every baseline number is tied to these exact ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jevmark.baselines.subset import PER_SPLIT, SUB_PER_SPLIT, SUBSET_PATH, build_subset
from jevmark.data.build import SPLITS

REPO = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--out", default=str(SUBSET_PATH))
    parser.add_argument("--force", action="store_true", help="overwrite an existing subset file")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; the subset is fixed once written (--force to overwrite)")
        return 1
    subset = build_subset(Path(args.data_dir), SPLITS, PER_SPLIT, SUB_PER_SPLIT)
    out.write_text(json.dumps(subset, indent=1) + "\n")
    for split in SPLITS:
        print(f"{split:20} {len(subset['splits'][split]):4} records, sub-subset {len(subset['sub_splits'][split])}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
