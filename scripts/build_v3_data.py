"""Build the v3 data files (task 3.1, decision 56, docs/V3_DESIGN.md) and check them.

    uv run python scripts/build_v3_data.py [--config configs/v3_data.yaml]

Writes, next to the nine v1.3 files and without changing any of them:

- data/v3_banking77_train.jsonl, the log domain: every Banking77 train message
  except those whose normalised text occurs in a source split a test split draws
  from (dedup.test_texts, the decision 43 set, which holds the whole Banking77 test
  split) and except later copies of a text repeated within train, one choice
  question each, built as test_banking77 records are, from its own seeded streams;
- data/v3_banking77_test_full.jsonl: the 1000 test_banking77 records verbatim,
  then the other 2080 Banking77 test messages from their own seeded streams;
- data/v3_manifest.json: counts, every dropped train row with its reason, and the
  sha256 of the two new files and of the nine v1.3 files.

Checks, any failure stops the build before a file is written: the nine v1.3 files
match the sha256 recorded in data/baseline_subset.json, before and after the
build; test_banking77 rebuilt in memory equals data/test_banking77.jsonl byte for
byte, and is the first 1000 records of the full test; every record validates as a
request and encodes within max_tokens; ids are unique across both files; every
question has 10 options with gold among them and never other; the gold position is
within 4 standard deviations of uniform; about 20 percent of states are JSON; no
log-domain text is in the excluded set or repeats; the full test uses every
Banking77 test message once; and a file whose sha256 is recorded in the config
matches it. The leak probes run on the two files afterwards (make data-v3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from jevmark.config import load_config
from jevmark.data import dedup, unseen
from jevmark.data.assemble import build_all
from jevmark.data.build import SPLITS, gold_positions, position_deviations, state_text
from jevmark.data.sources import BANKING77
from jevmark.baselines.subset import SUBSET_PATH, load_subset
from jevmark.encode import encode, letter_token_ids
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
MAX_POSITION_SD = 4.0


class BuildError(RuntimeError):
    pass


def serialise(records: Sequence[Mapping[str, Any]]) -> bytes:
    """A split file's bytes, exactly as build_data.py writes them."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def v13_hashes(data_dir: Path, recorded: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Each v1.3 file's sha256 now, against the one recorded when the baseline subset was drawn."""
    out = {}
    for split in SPLITS:
        name = f"{split}.jsonl"
        digest = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        out[name] = {"sha256": digest, "unchanged": digest == recorded[name]}
    return out


def check_records(name: str, records: Sequence[Mapping[str, Any]], tokenizer: Any, max_tokens: int, n_options: int, failures: list[str]) -> dict[str, Any]:
    lengths, ks, other_gold, missing_gold = [], Counter(), 0, 0
    for record in records:
        lengths.append(len(encode(Request.from_dict({"state": record["state"], "questions": record["questions"]}), tokenizer, max_tokens).input_ids))
        question = record["questions"][unseen.QUESTION_ID]
        gold = record["gold"][unseen.QUESTION_ID]
        ks[len(question["criteria"])] += 1
        other_gold += gold == unseen.OTHER_LABEL
        missing_gold += gold not in question["criteria"]
    deviations = position_deviations(gold_positions(records))
    json_share = sum(isinstance(r["state"], dict) for r in records) / len(records)
    if set(ks) != {n_options}:
        failures.append(f"{name}: option counts {dict(ks)}, expected {n_options} everywhere")
    if other_gold or missing_gold:
        failures.append(f"{name}: gold is other in {other_gold} records and missing from the options in {missing_gold}")
    worst = max(deviations.values(), default=0.0)
    if worst > MAX_POSITION_SD:
        failures.append(f"{name}: gold position deviates {worst:.2f} sd from uniform")
    if abs(json_share - 0.2) > 0.005:
        failures.append(f"{name}: JSON-state share {json_share:.3f}, expected 0.2")
    return {"records": len(records), "max_tokens": max(lengths), "options": dict(ks), "json_share": json_share, "gold_position_max_sd": worst}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "v3_data.yaml"))
    args = parser.parse_args(argv)
    v3 = load_config(args.config)
    config = load_config(REPO / v3["data_config"])
    data_dir = REPO / v3["out_dir"]
    reference = load_config(REPO / config["base_config"])["tokenizer_reference"]
    tokenizer = AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])
    letter_token_ids(tokenizer)
    max_tokens, n_options = int(config["max_tokens"]), int(config["unseen"]["banking77_options"])
    recorded = load_subset(SUBSET_PATH)["data_files_sha256"]
    failures: list[str] = []

    before = v13_hashes(data_dir, recorded)
    if not all(v["unchanged"] for v in before.values()):
        print("FAILED: v1.3 files differ from data/baseline_subset.json before the build: " + ", ".join(n for n, v in before.items() if not v["unchanged"]), file=sys.stderr)
        return 1

    # test_banking77 rebuilt from its own streams must equal the file, so the full test starts from it verbatim.
    rebuilt = build_all(config, ["test_banking77"])
    on_disk = (data_dir / "test_banking77.jsonl").read_bytes()
    if serialise(rebuilt.splits["test_banking77"]) != on_disk:
        failures.append("test_banking77 rebuilt in memory differs from data/test_banking77.jsonl")
    test_banking77 = [json.loads(line) for line in on_disk.decode("utf-8").splitlines()]

    excluded = dedup.test_texts(tuple(rebuilt.held_out))
    train, dropped = unseen.build_v3_train(config, excluded)
    test_full = unseen.build_v3_test_full(config, test_banking77)
    banking_test = set(dedup.normalise(t) for t in load_dataset(BANKING77.id, BANKING77.config, revision=BANKING77.revision, split="test")["text"])

    reports = {
        "train": check_records(unseen.V3_TRAIN, train, tokenizer, max_tokens, n_options, failures),
        "test_full": check_records(unseen.V3_TEST_FULL, test_full, tokenizer, max_tokens, n_options, failures),
        "test_full_new": check_records(f"{unseen.V3_TEST_FULL} (new records)", test_full[len(test_banking77) :], tokenizer, max_tokens, n_options, failures),
    }
    ids = Counter(r["id"] for r in [*train, *test_full])
    if any(n > 1 for n in ids.values()):
        failures.append(f"record ids repeat: {[i for i, n in ids.items() if n > 1][:3]}")
    if serialise(test_full[: len(test_banking77)]) != on_disk:
        failures.append("the full test does not start with test_banking77 verbatim")
    indices = [r["meta"]["source_index"] for r in test_full]
    total_test = len(load_dataset(BANKING77.id, BANKING77.config, revision=BANKING77.revision, split="test"))
    if sorted(indices) != list(range(total_test)):
        failures.append(f"the full test does not use each of the {total_test} Banking77 test messages exactly once")
    train_texts = [dedup.normalise(state_text(r)) for r in train]
    if any(t in excluded for t in train_texts) or len(set(train_texts)) != len(train_texts):
        failures.append("a log-domain text is excluded or repeats")

    reasons = Counter(d["reason"] for d in dropped)
    banking_matches = sum(d["reason"] == "test_text" and dedup.normalise(d["text"]) in banking_test for d in dropped)
    print(f"== v3 log domain ({unseen.V3_TRAIN})")
    print(f"Banking77 train messages {len(train) + len(dropped)}; dropped {len(dropped)}: {reasons['test_text']} matching a test text "
          f"({banking_matches} Banking77 test, {reasons['test_text'] - banking_matches} other test sources), {reasons['repeat']} later copies of a repeated text; kept {len(train)}")
    for d in dropped:
        if d["reason"] == "test_text" and dedup.normalise(d["text"]) not in banking_test:
            print(f"  dropped, matches another test source: train index {d['index']}: {d['text']!r}")
    print(f"\n== full Banking77 test ({unseen.V3_TEST_FULL}): {len(test_banking77)} test_banking77 records verbatim + {len(test_full) - len(test_banking77)} new = {len(test_full)}")
    print("\n== checks per file")
    for name, report in reports.items():
        print(f"{name:14} records {report['records']:5}  options {report['options']}  json states {report['json_share']:.1%}  "
              f"gold position max {report['gold_position_max_sd']:.2f} sd  max tokens {report['max_tokens']}")

    after = v13_hashes(data_dir, recorded)
    if not all(v["unchanged"] for v in after.values()):
        failures.append("v1.3 files changed during the build")
    new_files = {v3["files"]["train"]: serialise(train), v3["files"]["test_full"]: serialise(test_full)}
    new_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in new_files.items()}
    expected = v3.get("sha256") or {}
    for name, digest in new_hashes.items():
        if expected.get(name) and expected[name] != digest:
            failures.append(f"{name}: sha256 {digest} differs from the recorded {expected[name]}")

    print("\n== v1.3 files against data/baseline_subset.json")
    for name, row in after.items():
        print(f"{name:26} {row['sha256']}  {'unchanged' if row['unchanged'] else 'CHANGED'}")
    print("\n== v3 files")
    for name, digest in new_hashes.items():
        state = "matches the config" if expected.get(name) == digest else "not yet recorded in the config"
        print(f"{name:30} {digest}  {state}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    for name, raw in new_files.items():
        (data_dir / name).write_bytes(raw)
    manifest = {
        "decision": 56,
        "files": {name: {"sha256": new_hashes[name], "records": len(raw.splitlines())} for name, raw in new_files.items()},
        "log_domain": {"source_rows": len(train) + len(dropped), "kept": len(train), "dropped": reasons, "dropped_banking77_test_matches": banking_matches, "dropped_rows": dropped},
        "test_full": {"test_banking77_records": len(test_banking77), "new_records": len(test_full) - len(test_banking77)},
        "checks": reports,
        "v1_3_files": after,
    }
    (data_dir / v3["manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {', '.join(new_files)} and {v3['manifest']} to {data_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
