"""Duplicate check between training and test data (docs/DATA.md section 8).

    uv run python scripts/check_duplicates.py [--config configs/data.yaml] [--data-dir data]

Compares the state texts (the `text` field of a JSON state) of train with every
test split, and of valid with every test split. Texts are normalised (Unicode NFKC,
lower case, every run of characters other than letters and digits becomes one
space, edge spaces stripped; dedup.normalise, the same function the builders use to
drop such texts from train and valid). Fails on any normalised exact match; reports, per
pair, the share of test records that share at least one word 8-gram with the
reference split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jevmark.config import load_config
from jevmark.data.build import SPLITS, state_text
from jevmark.data.dedup import normalise

REPO = Path(__file__).resolve().parents[1]
REFERENCES = ("train", "valid")
TESTS = tuple(s for s in SPLITS if s.startswith("test_"))
N = 8


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compare(reference: Iterable[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exact normalised matches (test record ids and texts) and the 8-gram overlap rate of the test records."""
    ref_texts = {normalise(state_text(r)) for r in reference}
    ref_grams = set().union(*(ngrams(t) for t in ref_texts))
    exact = []
    overlapping = 0
    unique_test = set()
    for record in test:
        text = normalise(state_text(record))
        if text in ref_texts:
            exact.append({"id": record["id"], "text": text})
        overlapping += bool(ngrams(text) & ref_grams)
        unique_test.add(text)
    with_grams = sum(1 for r in test if ngrams(normalise(state_text(r))))
    return {
        "n_test": len(test),
        "exact_matches": len(exact),
        "exact_examples": exact[:5],
        "ngram_overlap_rate": overlapping / len(test),
        "test_records_with_an_8gram": with_grams,
    }


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "data.yaml"))
    parser.add_argument("--data-dir", default=None, help="default: out_dir from the config")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else REPO / config["out_dir"]
    splits = {s: read_split(data_dir, s) for s in (*REFERENCES, *TESTS)}
    print(f"== duplicate check: normalised exact matches (fail on any) and word {N}-gram overlap")
    failures = []
    results = {}
    for reference in REFERENCES:
        for test in TESTS:
            row = compare(splits[reference], splits[test])
            results[f"{reference}->{test}"] = row
            print(
                f"{reference:6} vs {test:20} exact {row['exact_matches']:4}  "
                f"{N}-gram overlap {row['ngram_overlap_rate']:6.2%} of {row['n_test']} records "
                f"({row['test_records_with_an_8gram']} have an {N}-gram)"
            )
            if row["exact_matches"]:
                failures.append(f"{reference} and {test} share {row['exact_matches']} normalised texts, first {row['exact_examples'][:2]}")
    (data_dir / "duplicates.json").write_text(json.dumps(results, indent=2) + "\n")
    if failures:
        print("\nFAILED: duplicate check\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("duplicate check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
