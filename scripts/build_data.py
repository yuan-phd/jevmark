"""Build every JSONL split under data/ (docs/DATA.md) and check it.

    uv run python scripts/build_data.py --config configs/data.yaml [key=value ...]

Every record is validated with Request.from_dict and must encode within max_tokens
with the pinned reference tokenizer. The build fails if, for any noul kind in any
split, the yes share is outside the configured range or the phrasing-only accuracy
(answering each phrasing with its own majority answer) is above
max_phrasing_only_accuracy; if a held-out intent reaches train or valid as an
utterance, an option or an asked intent; if test_unseen_intents offers a seen
intent; if a gold option position deviates from uniform by more than 4 standard
deviations for some K; or if record ids repeat. The report prints split sizes, noul
balance and phrasing-only accuracy per kind, score levels per scale, option-count
and answer-letter histograms, and gold positions per K.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from jevmark.config import load_config
from jevmark.data.assemble import build_all
from jevmark.data.build import SPLITS, gold_positions, held_out_leaks, noul_phrasing, position_deviations, score_distribution
from jevmark.encode import LETTERS, encode, letter_token_ids
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
MAX_POSITION_SD = 4.0


class BuildError(RuntimeError):
    pass


def gold_letter(question: dict[str, Any], gold: Any) -> str:
    if question["type"] == "noul":
        return "A" if gold == "true" else "B"
    if question["type"] == "choice":
        return LETTERS[list(question["criteria"]).index(gold)]
    return LETTERS[int(gold)]


def check_and_report(built: Any, config: dict[str, Any], tokenizer: Any) -> None:
    low, high = config["noul_balance"]
    max_tokens = int(config["max_tokens"])
    failures: list[str] = []
    ids: Counter = Counter()

    print("== split sizes (records by source), max encoded length")
    for split, records in built.splits.items():
        lengths = [len(encode(Request.from_dict(r), tokenizer, max_tokens).input_ids) for r in records]
        ids.update(r["id"] for r in records)
        sources = Counter(r["source"] for r in records)
        print(f"{split:20} {len(records):6}  {dict(sources)}  max tokens {max(lengths)}")

    max_shortcut = float(config["max_phrasing_only_accuracy"])
    print(f"\n== noul balance per kind (fail: yes share outside {low:.0%}-{high:.0%}, or phrasing-only accuracy above {max_shortcut:.0%})")
    for split, records in built.splits.items():
        table = noul_phrasing(records)
        if not table:
            print(f"{split:20} no noul questions")
            continue
        for kind in sorted(table):
            row = table[kind]
            print(
                f"{split:20} {kind:17} n {row['n']:5}  yes {row['yes_share']:.1%}  negated {row['negated_share']:.1%}  "
                f"phrasing-only acc {row['phrasing_only_accuracy']:.1%}  ({len(row['phrasings'])} phrasings)"
            )
            if not low <= row["yes_share"] <= high:
                failures.append(f"{split}/{kind}: yes share {row['yes_share']:.1%} outside {low:.0%}-{high:.0%}")
            if row["phrasing_only_accuracy"] > max_shortcut:
                failures.append(f"{split}/{kind}: phrasing-only accuracy {row['phrasing_only_accuracy']:.1%} above {max_shortcut:.0%}")

    print("\n== score gold level distribution per scale")
    for split, records in built.splits.items():
        for scale, counts in score_distribution(records).items():
            total = sum(counts.values())
            print(f"{split:20} {scale:15} n {total:5}  " + "  ".join(f"{level}: {n} ({n / total:.0%})" for level, n in counts.items()))

    print("\n== options per choice question (K: count)")
    for split, records in built.splits.items():
        ks = Counter(len(q["criteria"]) for r in records for q in r["questions"].values() if q["type"] == "choice")
        print(f"{split:20} {dict(sorted(ks.items())) or 'no choice questions'}")

    print("\n== gold answer letter, all question types (unconditional; see gold position per K below)")
    for split, records in built.splits.items():
        letters = Counter(gold_letter(q, r["gold"][qid]) for r in records for qid, q in r["questions"].items())
        print(f"{split:20} {dict(sorted(letters.items()))}")

    print(f"\n== gold option position per K (choice questions); fail above {MAX_POSITION_SD} sd from uniform")
    for split, records in built.splits.items():
        positions = gold_positions(records)
        deviations = position_deviations(positions)
        for k, sd in deviations.items():
            if sd > MAX_POSITION_SD:
                failures.append(f"{split}: K={k} gold position deviates {sd:.2f} sd from uniform")
        if split in ("train", "test_indomain"):
            print(f"{split}:")
            for k in sorted(positions):
                counts = [positions[k][p] for p in range(k)]
                print(f"  K={k:2} n={sum(counts):5} positions {counts}  max dev {deviations.get(k, float('nan')):.2f} sd")
        elif deviations:
            print(f"{split}: max deviation per K " + ", ".join(f"K={k}: {sd:.2f} sd" for k, sd in sorted(deviations.items())))

    leaks = held_out_leaks(built.splits.get("train", []) + built.splits.get("valid", []), built.held_out)
    if leaks:
        failures.append(f"held-out intents reach train or valid in {len(leaks)} records, first {leaks[:3]}")
    allowed = set(built.held_out) | {"other"}
    for record in built.splits.get("test_unseen_intents", []):
        labels = set(record["questions"]["intent"]["criteria"])
        if not labels <= allowed:
            failures.append(f"{record['id']}: test_unseen_intents offers seen intents {sorted(labels - allowed)}")
            break
    repeated = [i for i, n in ids.items() if n > 1]
    if repeated:
        failures.append(f"{len(repeated)} record ids repeat, first {repeated[:3]}")

    print(f"\n== held-out intents ({len(built.held_out)}): {', '.join(built.held_out)}")
    print(f"held-out leaks into train or valid: {len(leaks)}")
    if failures:
        raise BuildError("build checks failed:\n  " + "\n  ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "data.yaml"))
    parser.add_argument("overrides", nargs="*", help="key=value overrides, for example seed=1")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    reference = load_config(REPO / config["base_config"])["tokenizer_reference"]
    tokenizer = AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])
    letter_token_ids(tokenizer)

    print(f"seed {config['seed']}, max_tokens {config['max_tokens']}, tokenizer {reference['id']}@{reference['revision'][:8]}\n")
    built = build_all(config, SPLITS)
    try:
        check_and_report(built, config, tokenizer)
    except (BuildError, ValueError) as err:
        print(f"\nFAILED: {err}", file=sys.stderr)
        return 1

    out_dir = REPO / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in built.splits.items():
        with (out_dir / f"{split}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(built.splits)} files to {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
