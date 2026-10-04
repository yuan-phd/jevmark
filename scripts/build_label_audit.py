"""The label-noise audit sheet (closing plan, step 3): questions for a human to judge, the model answers kept apart.

    uv run python scripts/build_label_audit.py [--out docs/audit]

Samples, all seeded from seed 0:
- a: 100 questions of v3_banking77_test_full that both full_sft (N 5000, seed 0) and direct_brier
  (N 5000, seed 0) answer wrongly, from their results.jsonl.gz;
- b: 50 questions drawn uniformly from all of v3_banking77_test_full, independently of a, so a
  question can be in both (its group is then "a+b" and it appears once);
- c: 50 intent questions drawn uniformly from test_indomain (the choice question of each record), for
  contrast, with sft_06b's answers.

Writes, under --out:
- label_audit_sheet.csv: one row per question in a seeded shuffled order: group, record id, question
  id, message text, gold label and its description, the full option list, and the empty columns
  verdict (label_wrong, ambiguous or model_wrong) and note. No model answer is in it.
- label_audit_answers.csv: the model answers, keyed by record id, to open only after judging.
The first-pass verdicts (label_audit_cc_pass.csv) are written by hand from the sheet alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jevmark.metrics import QuestionResult, read_results  # noqa: E402

SEED = 0
V3 = "v3_banking77_test_full"
FULL = "v3_06b_full_sft_n5000_s0"
RLCD = "v3_06b_direct_brier_n5000_s0"
ZERO = "v3_06b_zeroshot"
SFT = "sft_06b"
SIZES = {"a": 100, "b": 50, "c": 50}
SHEET_FIELDS = ["row", "group", "record_id", "question_id", "message", "state_format", "gold_label", "gold_description", "options", "verdict", "note"]
ANSWER_FIELDS = ["record_id", "question_id", "group", "gold_label", "zero_shot", "full_sft_s0", "direct_brier_s0", "sft_06b"]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def by_key(results: Sequence[QuestionResult], split: str) -> dict[tuple[str, str], QuestionResult]:
    return {(r.record_id, r.question_id): r for r in results if r.split == split}


def message_of(record: dict[str, Any]) -> tuple[str, str]:
    state = record["state"]
    if isinstance(state, dict):
        return str(state.get("text", json.dumps(state, ensure_ascii=False))), "json"
    return str(state), "plain"


def answer(r: QuestionResult | None) -> str:
    if r is None:
        return ""
    return f"{r.labels[r.prediction]} ({r.probs[r.prediction]:.3f})"


def build(runs: Path, data: Path, out: Path) -> dict[str, int]:
    banking = read_jsonl(data / f"{V3}.jsonl")
    indomain = read_jsonl(data / "test_indomain.jsonl")
    full = by_key(read_results(runs / FULL / "results.jsonl.gz"), V3)
    rlcd = by_key(read_results(runs / RLCD / "results.jsonl.gz"), V3)
    zero = by_key(read_results(runs / ZERO / "results.jsonl.gz"), V3)
    sft = by_key(read_results(runs / SFT / "results.jsonl.gz"), "test_indomain")

    both_wrong = sorted(k for k in full if not full[k].correct and k in rlcd and not rlcd[k].correct)
    group_a = random.Random(f"label-audit:{SEED}:a").sample(both_wrong, SIZES["a"])
    group_b = random.Random(f"label-audit:{SEED}:b").sample(sorted(full), SIZES["b"])
    intent_keys = sorted(k for k in sft if indomain[k[0]]["questions"][k[1]]["type"] == "choice")
    group_c = random.Random(f"label-audit:{SEED}:c").sample(intent_keys, SIZES["c"])

    groups: dict[tuple[str, str], str] = {}
    for name, keys in (("a", group_a), ("b", group_b), ("c", group_c)):
        for k in keys:
            groups[k] = f"{groups[k]}+{name}" if k in groups else name
    order = sorted(groups)
    random.Random(f"label-audit:{SEED}:order").shuffle(order)

    sheet, answers = [], []
    for row, key in enumerate(order, start=1):
        record_id, qid = key
        record = (banking if key in full else indomain)[record_id]
        question = record["questions"][qid]
        gold = record["gold"][qid]
        message, state_format = message_of(record)
        options = " | ".join(f"{label}: {desc}" if desc else label for label, desc in question["criteria"].items())
        sheet.append({"row": row, "group": groups[key], "record_id": record_id, "question_id": qid, "message": message, "state_format": state_format,
                      "gold_label": gold, "gold_description": question["criteria"].get(gold) or "", "options": options, "verdict": "", "note": ""})
        answers.append({"record_id": record_id, "question_id": qid, "group": groups[key], "gold_label": gold, "zero_shot": answer(zero.get(key)),
                        "full_sft_s0": answer(full.get(key)), "direct_brier_s0": answer(rlcd.get(key)), "sft_06b": answer(sft.get(key))})

    out.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (("label_audit_sheet.csv", SHEET_FIELDS, sheet), ("label_audit_answers.csv", ANSWER_FIELDS, answers)):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    counts = {name: sum(name in g.split("+") for g in groups.values()) for name in SIZES}
    counts["rows"] = len(order)
    counts["both_wrong_pool"] = len(both_wrong)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--out", default=str(REPO / "docs" / "audit"))
    args = parser.parse_args(argv)
    counts = build(Path(args.runs_dir), Path(args.data_dir), Path(args.out))
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
