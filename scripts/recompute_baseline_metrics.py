"""Rebuild a baseline run's per-split metrics from its replies.jsonl, without calling the model (task 1.8).

    uv run python scripts/recompute_baseline_metrics.py runs/b1_qwen17b_json [--out path]

The baseline counterpart of recompute_metrics.py: the replies are parsed again with
the shared parser (strict, plus the lenient secondary reading of decision 48) on the
records of the baseline subset, or of its sub-subset when the run's config.yaml has
`sub: true`, and the per-split blocks are rebuilt with the same code the baseline
scripts use. Run-level fields (provenance, latency, usage, spend) are kept from the
existing metrics.json. Use it after a change to the parser or to the baseline
metrics, so finished runs gain it without new generation or API spend.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from jevmark.baselines.results import answers_from_replies, baseline_reports_by_split, read_replies
from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_records

REPO = Path(__file__).resolve().parents[1]


def recompute(run_dir: Path, subset: dict, data_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config_path = run_dir / "config.yaml"
    sub = bool(yaml.safe_load(config_path.read_text()).get("sub", False)) if config_path.is_file() else False
    replies = read_replies(run_dir / "replies.jsonl")
    splits = [split for split in subset["splits"] if split in metrics["splits"]]
    records = {split: subset_records(data_dir, subset, split, sub)[0] for split in splits}
    metrics["splits"] = baseline_reports_by_split(answers_from_replies(records, replies))
    metrics["recomputed"] = {
        "from": "replies.jsonl",
        "requests": sum((split, r["id"]) in replies for split, rs in records.items() for r in rs),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a baseline run directory holding replies.jsonl and metrics.json")
    parser.add_argument("--out", default=None, help="where to write; default: metrics.json in the run directory")
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    metrics = recompute(run_dir, load_subset(Path(args.subset)), Path(args.data_dir))
    out = Path(args.out) if args.out else run_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"recomputed {len(metrics['splits'])} splits from {metrics['recomputed']['requests']} replies; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
