"""Summary of the label-noise audit: verdict shares per group for two passes, their agreement and disagreements.

    uv run python scripts/summarise_label_audit.py [--audit docs/audit]

Reads label_audit_sheet.csv (the filled pass), label_audit_cc_pass.csv (the first pass) and
label_audit_answers.csv (the sample groups), and writes summary.json (per group and overall: counts,
shares with 95 percent Wilson intervals for each pass, Cohen's kappa over the four verdicts) and
disagreements.csv (every row where the two passes differ, with message, gold and both verdicts and notes).
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


def summarise(audit: Path) -> dict[str, Any]:
    sheet = list(csv.DictReader((audit / "label_audit_sheet.csv").open()))
    first = {r["record_id"]: r for r in csv.DictReader((audit / "label_audit_cc_pass.csv").open())}
    answers = {r["record_id"]: r for r in csv.DictReader((audit / "label_audit_answers.csv").open())}
    bad = [r["record_id"] for r in sheet if r["verdict"] not in (*VERDICTS, "")]
    if bad:
        raise SystemExit(f"verdicts outside {VERDICTS}: {bad}")
    if [r["record_id"] for r in sheet] != list(first) or set(first) != set(answers):
        raise SystemExit("the sheet, the first pass and the answers file do not hold the same rows in the same order")
    rows = [{"record_id": r["record_id"], "group": answers[r["record_id"]]["group"], "second": r["verdict"], "first": first[r["record_id"]]["cc_verdict"],
             "second_note": r["note"], "first_note": first[r["record_id"]]["cc_note"], "message": r["message"], "gold": r["gold_label"]} for r in sheet]
    out: dict[str, Any] = {"passes": {"first": "label_audit_cc_pass.csv (Claude, from the sheet alone)", "second": "label_audit_sheet.csv (filled by GPT)"},
                           "groups": {}}
    for name in (*GROUPS, "all"):
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
    (audit / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    with (audit / "disagreements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "group", "message", "gold", "first", "second", "first_note", "second_note"], lineterminator="\n")
        writer.writeheader()
        for r in rows:
            if r["first"] != r["second"]:
                writer.writerow({k: r[k] for k in ("record_id", "group", "message", "gold", "first", "second", "first_note", "second_note")})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", default=str(REPO / "docs" / "audit"))
    args = parser.parse_args(argv)
    out = summarise(Path(args.audit))
    print(json.dumps({g: {"kappa": v["kappa"], "agreement": v["agreement"]} for g, v in out["groups"].items()}), "disagreements", out["disagreements"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
