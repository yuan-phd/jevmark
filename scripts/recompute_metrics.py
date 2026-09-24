"""Rebuild a run's metrics.json from its results.jsonl.gz, without the model.

    uv run python scripts/recompute_metrics.py runs/base_06b [--out path]

The per-split metrics are recomputed from the per-question lines with the same
code evaluate.py uses (metrics.reports_by_split). Run-level fields that are not
per question (git, precision, latency, batching precision, data file hashes) are
kept from the existing metrics.json when there is one. Use this after adding a
metric to metrics.py, so old runs gain it without another GPU session.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jevmark.metrics import read_results, reports_by_split


def recompute(run_dir: Path) -> dict:
    results = read_results(run_dir / "results.jsonl.gz")
    existing_path = run_dir / "metrics.json"
    metrics = json.loads(existing_path.read_text()) if existing_path.is_file() else {"run_name": run_dir.name}
    metrics["splits"] = reports_by_split(results)
    metrics["recomputed"] = {
        "from": "results.jsonl.gz",
        "questions": len(results),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a run directory holding results.jsonl.gz")
    parser.add_argument("--out", default=None, help="where to write; default: metrics.json in the run directory")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    metrics = recompute(run_dir)
    out = Path(args.out) if args.out else run_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"recomputed {len(metrics['splits'])} splits from {metrics['recomputed']['questions']} questions; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
