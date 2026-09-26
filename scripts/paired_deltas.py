"""Paired accuracy differences between two runs on the same questions (task 1.9).

    uv run python scripts/paired_deltas.py runs/sft_17b runs/sft_06b --out runs/paired_17b_vs_06b/metrics.json

Both runs' results.jsonl.gz must hold the same questions (the same split, record
and question ids). For every split, every question type, and every noul kind, it
reports the accuracy of each run, the difference (first minus second), a 95
percent paired bootstrap interval (1000 resamples by record, seed 0, the same
record resampling as metrics.bootstrap_ci), and the share of questions whose
prediction differs. Gold-dependent questions only; form nouls are reported as
their own group, as everywhere else (decision 44). Several pairs can be given
with --pair, and each pair's block is keyed by "<first>_vs_<second>".
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevmark.metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, QuestionResult, is_form_noul, read_results

REPO = Path(__file__).resolve().parents[1]
TYPES = ("noul", "choice", "score")


def paired_block(pairs: Sequence[tuple[QuestionResult, QuestionResult]], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    record_index: dict[str, int] = {}
    rec = np.array([record_index.setdefault(a.record_id, len(record_index)) for a, _ in pairs])
    delta = np.array([float(a.correct) - float(b.correct) for a, b in pairs])
    n_records = len(record_index)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_records, np.full(n_records, 1.0 / n_records), size=BOOTSTRAP_RESAMPLES)[:, rec].astype(float)
    boot = (weights @ delta) / weights.sum(axis=1)
    return {
        "n": len(pairs),
        "accuracy_first": float(np.mean([a.correct for a, _ in pairs])),
        "accuracy_second": float(np.mean([b.correct for _, b in pairs])),
        "delta": float(delta.mean()),
        "delta_ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "prediction_disagreement": float(np.mean([a.prediction != b.prediction for a, b in pairs])),
    }


def paired_report(first: Path, second: Path) -> dict[str, Any]:
    a = {(r.split, r.record_id, r.question_id): r for r in read_results(first / "results.jsonl.gz")}
    b = {(r.split, r.record_id, r.question_id): r for r in read_results(second / "results.jsonl.gz")}
    if a.keys() != b.keys():
        raise SystemExit(f"{first} and {second} do not hold the same questions")
    by_split: dict[str, list[tuple[QuestionResult, QuestionResult]]] = defaultdict(list)
    for key in a:
        by_split[key[0]].append((a[key], b[key]))
    splits: dict[str, Any] = {}
    for split, pairs in by_split.items():
        gold = [p for p in pairs if not is_form_noul(p[0])]
        block: dict[str, Any] = {"overall": paired_block(gold)}
        for qtype in TYPES:
            of_type = [p for p in gold if p[0].qtype == qtype]
            if of_type:
                block[qtype] = paired_block(of_type)
        kinds = sorted({p[0].kind for p in gold if p[0].qtype == "noul"})
        if kinds:
            block["noul_by_kind"] = {kind: paired_block([p for p in gold if p[0].kind == kind]) for kind in kinds}
        form = [p for p in pairs if is_form_noul(p[0])]
        if form:
            block["form"] = paired_block(form)
        splits[split] = block

    def provenance(run: Path) -> dict[str, Any]:
        metrics = json.loads((run / "metrics.json").read_text())
        return {"run": str(run), "git": metrics.get("git"), "data_files_sha256": metrics.get("data_files_sha256")}

    return {"first": provenance(first), "second": provenance(second), "splits": splits}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", metavar=("FIRST", "SECOND"), required=True, help="two run directories; repeat for more pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "note": "paired accuracy differences, first minus second, on identical questions; 95 percent bootstrap intervals by record (1000 resamples, seed 0)",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pairs": {},
    }
    for first, second in args.pair:
        report["pairs"][f"{Path(first).name}_vs_{Path(second).name}"] = paired_report(Path(first), Path(second))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}: {', '.join(report['pairs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
