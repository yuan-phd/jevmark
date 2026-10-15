"""Rebuild a run's metrics.json from its results.jsonl.gz, without the model.

    uv run python scripts/recompute_metrics.py runs/base_06b [--out path]
    uv run python scripts/recompute_metrics.py runs/sft_06b --subset [--sub]

The per-split metrics are recomputed from the per-question lines with the same
code evaluate.py uses (metrics.reports_by_split). Run-level fields that are not
per question (git, precision, latency, batching precision, data file hashes) are
kept from the existing metrics.json when there is one. Use this after adding a
metric to metrics.py, so old runs gain it without another GPU session.

A temperature-scaled run (scripts/calibrate.py: calibration.json, no results file
of its own) is rebuilt from its source run's results.jsonl.gz, named in its
metrics.json, scaled by the temperature its calibration.json holds; nothing is
refitted.

--subset [PATH] restricts the questions to the records of the fixed baseline
subset (default data/baseline_subset.json, task 1.8), 500 per split, or with --sub
to its 200-record sub-subset, so a jevmark run is compared with the baselines on
exactly their records. It writes metrics_subset.json (metrics_subset_sub.json with
--sub) next to metrics.json, never over it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_ids
from jevmark.calibration import scale_results
from jevmark.metrics import QuestionResult, read_results, reports_by_split

REPO = Path(__file__).resolve().parents[1]


def restrict_to_subset(results: Sequence[QuestionResult], subset: dict, sub: bool = False) -> list[QuestionResult]:
    """The results whose (split, record id) is in the subset; splits the subset does not cover are dropped."""
    wanted = {(split, rid) for split in subset["splits"] for rid in subset_ids(subset, split, sub)}
    return [r for r in results if (r.split, r.record_id) in wanted]


def scaled_source(run_dir: Path, metrics: dict) -> tuple[list[QuestionResult], str]:
    """A temperature-scaled run's results: its source run's, scaled by the temperature in its calibration.json."""
    t = float(json.loads((run_dir / "calibration.json").read_text())["temperature"])
    source = Path(metrics["source"]["run"])
    source = source if source.is_absolute() else REPO / source
    return scale_results(read_results(source / "results.jsonl.gz"), t), f"{metrics['source']['run']}/results.jsonl.gz scaled by temperature {t}"


def recompute(run_dir: Path, subset: dict | None = None, sub: bool = False) -> dict:
    existing_path = run_dir / "metrics.json"
    metrics = json.loads(existing_path.read_text()) if existing_path.is_file() else {"run_name": run_dir.name}
    if (run_dir / "results.jsonl.gz").is_file():
        results, origin = read_results(run_dir / "results.jsonl.gz"), "results.jsonl.gz"
    elif (run_dir / "calibration.json").is_file() and "source" in metrics:
        results, origin = scaled_source(run_dir, metrics)
    else:
        raise SystemExit(f"{run_dir}: no results.jsonl.gz, and not a temperature-scaled run with a source")
    if subset is not None:
        results = restrict_to_subset(results, subset, sub)
    metrics["splits"] = reports_by_split(results)
    metrics["recomputed"] = {
        "from": origin,
        "questions": len(results),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if subset is not None:
        metrics["recomputed"]["subset"] = {"records": "sub_splits (200 per split)" if sub else "splits (500 per split)", "per_split": subset["sub_per_split" if sub else "per_split"]}
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a run directory holding results.jsonl.gz")
    parser.add_argument("--out", default=None, help="where to write; default: metrics.json in the run directory (metrics_subset.json with --subset)")
    parser.add_argument("--subset", nargs="?", const=str(SUBSET_PATH), default=None, metavar="PATH", help="restrict to the baseline subset (default path data/baseline_subset.json)")
    parser.add_argument("--sub", action="store_true", help="with --subset: its 200-record sub-subset")
    args = parser.parse_args(argv)
    if args.sub and args.subset is None:
        parser.error("--sub needs --subset")
    run_dir = Path(args.run_dir)
    subset = load_subset(Path(args.subset)) if args.subset else None
    metrics = recompute(run_dir, subset, args.sub)
    default = "metrics.json" if subset is None else ("metrics_subset_sub.json" if args.sub else "metrics_subset.json")
    out = Path(args.out) if args.out else run_dir / default
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"recomputed {len(metrics['splits'])} splits from {metrics['recomputed']['questions']} questions; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
