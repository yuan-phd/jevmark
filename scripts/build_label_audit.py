"""The label-noise audit sheet (closing plan, step 3): questions for a human to judge, the model answers kept apart.

    uv run python scripts/build_label_audit.py [--domain banking77|emotion] [--out docs/audit]

Banking77 (the default) samples, all seeded from seed 0:
- a: 100 questions of v3_banking77_test_full that both full_sft (N 5000, seed 0) and direct_brier
  (N 5000, seed 0) answer wrongly, from their results.jsonl.gz;
- b: 50 questions drawn uniformly from all of v3_banking77_test_full, independently of a, so a
  question can be in both (its group is then "a+b" and it appears once);
- c: 50 intent questions drawn uniformly from test_indomain (the choice question of each record), for
  contrast, with sft_06b's answers.

Writes, under --out:
- label_audit_sheet.csv: the blind sheet, one row per question in a seeded shuffled order: record
  id, question id, message text, state format, gold label and its description, the full option
  list, and the empty columns verdict (label_correct, ambiguous, label_wrong or convention, defined
  in docs/audit/REVIEW_GUIDE.md) and note. Neither the sample group nor any model answer is in it.
- label_audit_answers.csv: the sample group and the model answers, keyed by record id, to open only
  after judging.
The first-pass verdicts (label_audit_cc_pass.csv) are written by hand from the sheet alone.

--domain emotion builds the same audit for the emotion log domain, on the choice question
(label) of v3_emotion_test_full, seeded from "emotion-audit:0":
- a: 100 questions that both full_sft and positive_sft (N 5000, seed 0) answer wrongly
  (direct_brier on emotion had not run when the sheet was built);
- b: 50 questions drawn uniformly from the same split, independently of a.
It writes emotion_audit_sheet.csv (same columns as the Banking77 sheet) and
emotion_audit_answers.csv (group, zero-shot, full_sft and positive_sft answers); the first pass
is emotion_audit_cc_pass.csv, written by hand from the sheet alone.
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
SHEET_FIELDS = ["record_id", "question_id", "message", "state_format", "gold_label", "gold_description", "options", "verdict", "note"]
ANSWER_FIELDS = ["record_id", "question_id", "group", "gold_label", "zero_shot", "full_sft_s0", "direct_brier_s0", "sft_06b"]
EMOTION = "v3_emotion_test_full"
EMOTION_FULL = "v3_emotion_06b_full_sft_n5000_s0"
EMOTION_POSITIVE = "v3_emotion_06b_positive_sft_n5000_s0"
EMOTION_ZERO = "v3_emotion_06b_zeroshot"
EMOTION_QUESTION = "label"
EMOTION_SIZES = {"a": 100, "b": 50}
EMOTION_ANSWER_FIELDS = ["record_id", "question_id", "group", "gold_label", "zero_shot", "full_sft_s0", "positive_sft_s0"]


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


def sheet_row(record: dict[str, Any], record_id: str, qid: str) -> dict[str, str]:
    question = record["questions"][qid]
    gold = record["gold"][qid]
    message, state_format = message_of(record)
    options = " | ".join(f"{label}: {desc}" if desc else label for label, desc in question["criteria"].items())
    return {"record_id": record_id, "question_id": qid, "message": message, "state_format": state_format,
            "gold_label": gold, "gold_description": question["criteria"].get(gold) or "", "options": options, "verdict": "", "note": ""}


def write(out: Path, files: Sequence[tuple[str, list[str], list[dict[str, str]]]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in files:
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


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
    for key in order:
        record_id, qid = key
        record = (banking if key in full else indomain)[record_id]
        row = sheet_row(record, record_id, qid)
        sheet.append(row)
        answers.append({"record_id": record_id, "question_id": qid, "group": groups[key], "gold_label": row["gold_label"], "zero_shot": answer(zero.get(key)),
                        "full_sft_s0": answer(full.get(key)), "direct_brier_s0": answer(rlcd.get(key)), "sft_06b": answer(sft.get(key))})

    write(out, (("label_audit_sheet.csv", SHEET_FIELDS, sheet), ("label_audit_answers.csv", ANSWER_FIELDS, answers)))
    counts = {name: sum(name in g.split("+") for g in groups.values()) for name in SIZES}
    counts["rows"] = len(order)
    counts["both_wrong_pool"] = len(both_wrong)
    return counts


def build_emotion(runs: Path, data: Path, out: Path) -> dict[str, int]:
    emotion = read_jsonl(data / f"{EMOTION}.jsonl")
    full, positive, zero = ({k: r for k, r in by_key(read_results(runs / run / "results.jsonl.gz"), EMOTION).items() if k[1] == EMOTION_QUESTION}
                            for run in (EMOTION_FULL, EMOTION_POSITIVE, EMOTION_ZERO))

    both_wrong = sorted(k for k in full if not full[k].correct and k in positive and not positive[k].correct)
    group_a = random.Random(f"emotion-audit:{SEED}:a").sample(both_wrong, EMOTION_SIZES["a"])
    group_b = random.Random(f"emotion-audit:{SEED}:b").sample(sorted(full), EMOTION_SIZES["b"])

    groups: dict[tuple[str, str], str] = {}
    for name, keys in (("a", group_a), ("b", group_b)):
        for k in keys:
            groups[k] = f"{groups[k]}+{name}" if k in groups else name
    order = sorted(groups)
    random.Random(f"emotion-audit:{SEED}:order").shuffle(order)

    sheet, answers = [], []
    for key in order:
        record_id, qid = key
        row = sheet_row(emotion[record_id], record_id, qid)
        sheet.append(row)
        answers.append({"record_id": record_id, "question_id": qid, "group": groups[key], "gold_label": row["gold_label"], "zero_shot": answer(zero.get(key)),
                        "full_sft_s0": answer(full.get(key)), "positive_sft_s0": answer(positive.get(key))})

    write(out, (("emotion_audit_sheet.csv", SHEET_FIELDS, sheet), ("emotion_audit_answers.csv", EMOTION_ANSWER_FIELDS, answers)))
    counts = {name: sum(name in g.split("+") for g in groups.values()) for name in EMOTION_SIZES}
    counts["a+b"] = sum(g == "a+b" for g in groups.values())
    counts["rows"] = len(order)
    counts["both_wrong_pool"] = len(both_wrong)
    counts["questions"] = len(full)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", choices=("banking77", "emotion"), default="banking77")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--out", default=str(REPO / "docs" / "audit"))
    args = parser.parse_args(argv)
    builder = build_emotion if args.domain == "emotion" else build
    counts = builder(Path(args.runs_dir), Path(args.data_dir), Path(args.out))
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
