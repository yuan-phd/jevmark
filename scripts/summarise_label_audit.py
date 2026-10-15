"""Summary of the label-noise audit: verdict shares per group for two passes, their agreement and disagreements.

    uv run python scripts/summarise_label_audit.py [--domain banking77|emotion] [--audit docs/audit]

Reads label_audit_sheet.csv (the filled pass), label_audit_cc_pass.csv (the first pass) and
label_audit_answers.csv (the sample groups), and writes summary.json (per group and overall: counts,
shares with 95 percent Wilson intervals for each pass, Cohen's kappa over the four verdicts) and
disagreements.csv (every row where the two passes differ, with message, gold and both verdicts and notes).

--domain emotion reads emotion_audit_sheet.csv, emotion_audit_cc_pass.csv and
emotion_audit_answers.csv (groups a and b), refuses a sheet whose header differs from the built
sheet's, whose rows differ from the first pass's, whose gold labels differ from the answers file's,
or with an empty or unknown verdict, and writes emotion_summary.json and emotion_disagreements.csv.
emotion_summary.json also records full_sft's accuracy on the choice question of
v3_emotion_test_full (from its committed metrics.json) and the size of the shared-error pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VERDICTS = ("label_correct", "ambiguous", "label_wrong", "convention")
GROUPS = ("a", "b", "c")
SHEET_FIELDS = ["record_id", "question_id", "message", "state_format", "gold_label", "gold_description", "options", "verdict", "note"]
DOMAINS: dict[str, dict[str, Any]] = {
    "banking77": {"prefix": "label_audit_", "summary": "summary.json", "disagreements": "disagreements.csv", "groups": GROUPS, "strict": False},
    "emotion": {"prefix": "emotion_audit_", "summary": "emotion_summary.json", "disagreements": "emotion_disagreements.csv", "groups": ("a", "b"), "strict": True},
}
EMOTION_FULL_SFT = "runs/v3_emotion_06b_full_sft_n5000_s0/metrics.json"
EMOTION_POOL = 197  # questions both full_sft and positive_sft (N 5000, seed 0) miss, printed by build_label_audit.py --domain emotion
EMOTION_QUESTIONS = 2000


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    n = len(first)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(first, second)) / n
    ca, cb = Counter(first), Counter(second)
    expected = sum(ca[v] * cb[v] for v in VERDICTS) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def shares(verdicts: Sequence[str]) -> dict[str, Any]:
    n = len(verdicts)
    counts = Counter(verdicts)
    out: dict[str, Any] = {"n": n, "empty": counts.get("", 0)}
    for v in VERDICTS:
        out[v] = {"count": counts.get(v, 0), "share": counts.get(v, 0) / n if n else None, "ci": wilson(counts.get(v, 0), n)}
    problems = sum(counts.get(v, 0) for v in ("ambiguous", "label_wrong", "convention"))
    out["any_label_problem"] = {"count": problems, "share": problems / n if n else None, "ci": wilson(problems, n)}
    return out


def summarise(audit: Path, domain: str = "banking77") -> dict[str, Any]:
    d = DOMAINS[domain]
    with (audit / f"{d['prefix']}sheet.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        sheet = list(reader)
        header = reader.fieldnames
    first = {r["record_id"]: r for r in csv.DictReader((audit / f"{d['prefix']}cc_pass.csv").open())}
    answers = {r["record_id"]: r for r in csv.DictReader((audit / f"{d['prefix']}answers.csv").open())}
    bad = [r["record_id"] for r in sheet if r["verdict"] not in (*VERDICTS, "")]
    if bad:
        raise SystemExit(f"verdicts outside {VERDICTS}: {bad}")
    if [r["record_id"] for r in sheet] != list(first) or set(first) != set(answers):
        raise SystemExit("the sheet, the first pass and the answers file do not hold the same rows in the same order")
    if d["strict"]:
        if header != SHEET_FIELDS:
            raise SystemExit(f"sheet columns {header}, expected {SHEET_FIELDS}")
        changed = [r["record_id"] for r in sheet if r["gold_label"] != answers[r["record_id"]]["gold_label"]]
        if changed:
            raise SystemExit(f"gold labels differ from the answers file: {changed}")
        empty = [r["record_id"] for r in sheet if not r["verdict"]]
        if empty:
            raise SystemExit(f"{len(empty)} rows without a verdict: {empty}")
    rows = [{"record_id": r["record_id"], "group": answers[r["record_id"]]["group"], "second": r["verdict"], "first": first[r["record_id"]]["cc_verdict"],
             "second_note": r["note"], "first_note": first[r["record_id"]]["cc_note"], "message": r["message"], "gold": r["gold_label"]} for r in sheet]
    out: dict[str, Any] = {"passes": {"first": f"{d['prefix']}cc_pass.csv (Claude, from the sheet alone)", "second": f"{d['prefix']}sheet.csv (filled by GPT)"},
                           "groups": {}}
    for name in (*d["groups"], "all"):
        members = [r for r in rows if name == "all" or name in r["group"].split("+")]
        judged = [r for r in members if r["second"]]
        out["groups"][name] = {
            "first": shares([r["first"] for r in members]),
            "second": shares([r["second"] for r in members]),
            "agreement": sum(r["first"] == r["second"] for r in judged) / len(judged) if judged else None,
            "kappa": kappa([r["first"] for r in judged], [r["second"] for r in judged]),
            "both_label_wrong": sum(r["first"] == r["second"] == "label_wrong" for r in members),
            "both_label_correct": sum(r["first"] == r["second"] == "label_correct" for r in members),
            "both_any_problem": sum(r["first"] != "label_correct" and r["second"] != "label_correct" for r in judged),
        }
    out["disagreements"] = sum(r["first"] != r["second"] for r in rows)
    if domain == "emotion":
        metrics = json.loads((REPO / EMOTION_FULL_SFT).read_text())
        out["context"] = {"full_sft_choice_accuracy": metrics["splits"]["v3_emotion_test_full"]["choice"]["accuracy"],
                          "full_sft_source": f"{EMOTION_FULL_SFT}:splits.v3_emotion_test_full.choice.accuracy",
                          "shared_error_pool": EMOTION_POOL, "questions": EMOTION_QUESTIONS, "shared_error_share": EMOTION_POOL / EMOTION_QUESTIONS}
    (audit / d["summary"]).write_text(json.dumps(out, indent=2) + "\n")
    with (audit / d["disagreements"]).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "group", "message", "gold", "first", "second", "first_note", "second_note"], lineterminator="\n")
        writer.writeheader()
        for r in rows:
            if r["first"] != r["second"]:
                writer.writerow({k: r[k] for k in ("record_id", "group", "message", "gold", "first", "second", "first_note", "second_note")})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", choices=sorted(DOMAINS), default="banking77")
    parser.add_argument("--audit", default=str(REPO / "docs" / "audit"))
    args = parser.parse_args(argv)
    out = summarise(Path(args.audit), args.domain)
    print(json.dumps({g: {"kappa": v["kappa"], "agreement": v["agreement"]} for g, v in out["groups"].items()}), "disagreements", out["disagreements"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
