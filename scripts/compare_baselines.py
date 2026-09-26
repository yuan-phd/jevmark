"""One table: jevmark runs and the generative baselines on the fixed baseline subset (task 1.8).

    uv run python scripts/compare_baselines.py
    uv run python scripts/compare_baselines.py --sub runs/sft_17b runs/b2_<stronger model>

Every number is recomputed here from per-question evidence restricted to the subset
(data/baseline_subset.json, or its 200-record sub-subset with --sub): a jevmark run
from its results.jsonl.gz (the restriction recompute_metrics.py --subset applies),
a baseline from its replies.jsonl with the shared parser. Nothing is written.

jevmark columns show accuracy and ECE (top-1 probability). Baseline columns show
accuracy counting parse failures as wrong, ECE on the verbalized confidence of the
parsed answers that state one, and the parse failure rate. Rows are the gold-
dependent questions of each split, overall and per question type (decision 44);
n is the number of questions in the subset. A baseline cell marked * covers fewer
questions than the subset (an incomplete run). Below the table: batch-1 latency
and throughput from each run's metrics.json, and the cost per 1000 requests for
the API baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from recompute_metrics import restrict_to_subset  # noqa: E402

from jevmark.baselines.results import answers_from_replies, baseline_reports_by_split, read_replies  # noqa: E402
from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_records  # noqa: E402
from jevmark.data.build import SPLITS  # noqa: E402
from jevmark.metrics import read_results, reports_by_split  # noqa: E402

DEFAULT_RUNS = ("runs/base_06b", "runs/sft_06b", "runs/sft_17b", "runs/b1_qwen17b_json", "runs/b2_gpt-4.1-mini")
BLOCKS = ("overall", "noul", "choice", "score")
WIDTH = 23


def run_reports(run_dir: Path, subset: dict[str, Any], sub: bool, data_dir: Path) -> tuple[str, dict[str, Any] | None]:
    """("jevmark" | "baseline" | "missing", per-split reports on the subset)."""
    if (run_dir / "results.jsonl.gz").is_file():
        return "jevmark", reports_by_split(restrict_to_subset(read_results(run_dir / "results.jsonl.gz"), subset, sub))
    if (run_dir / "replies.jsonl").is_file():
        records = {split: subset_records(data_dir, subset, split, sub)[0] for split in subset["splits"]}
        return "baseline", baseline_reports_by_split(answers_from_replies(records, read_replies(run_dir / "replies.jsonl")))
    return "missing", None


def cell(kind: str, block: dict[str, Any] | None, n_expected: int | None) -> str:
    if block is None:
        return "-"
    if kind == "jevmark":
        return f"{block['accuracy']:.3f} / {block['ece']:.3f}"
    ece = "n/a" if block["ece"] is None else f"{block['ece']:.3f}"
    mark = "*" if n_expected is not None and block["n"] < n_expected else ""
    return f"{block['accuracy_all']:.3f} / {ece} / {block['parse_failure_rate']:.2f}{mark}"


def table(runs: Sequence[tuple[str, str, dict[str, Any] | None]]) -> list[str]:
    header = f"{'split / type':28}{'n':>6}  " + "".join(f"{name[:WIDTH - 1]:<{WIDTH}}" for name, _, _ in runs)
    legend = f"{'':36}" + "".join(f"{'acc / ECE' if kind == 'jevmark' else 'acc_all / ECE / fail' if kind == 'baseline' else 'missing':<{WIDTH}}" for _, kind, _ in runs)
    lines = [header, legend]
    reference = next((reports for _, kind, reports in runs if kind == "jevmark"), None)
    splits = [s for s in SPLITS if any(reports and s in reports for _, _, reports in runs)]
    for split in splits:
        for block_name in BLOCKS:
            if not any(reports and split in reports and block_name in reports[split] for _, _, reports in runs):
                continue
            ref_block = reference.get(split, {}).get(block_name) if reference else None
            n_expected = ref_block["n"] if ref_block else None
            label = split if block_name == "overall" else f"  {block_name}"
            row = f"{label:28}{n_expected if n_expected is not None else '':>6}  "
            for _, kind, reports in runs:
                block = reports.get(split, {}).get(block_name) if reports else None
                row += f"{cell(kind, block, n_expected):<{WIDTH}}"
            lines.append(row)
    return lines


def footer(run_dirs: Sequence[Path]) -> list[str]:
    lines = ["", "latency and cost (from each run's metrics.json)"]
    for run_dir in run_dirs:
        path = run_dir / "metrics.json"
        if not path.is_file():
            lines.append(f"  {run_dir.name}: no metrics.json")
            continue
        metrics = json.loads(path.read_text())
        latency = metrics.get("latency") or {}
        parts = []
        if latency.get("batch_1"):
            parts.append(f"batch 1 median {latency['batch_1']['median_ms']:.1f} ms")
        throughput = [(k, v) for k, v in latency.items() if k.endswith("_requests_per_second")]
        parts += [f"{k.removesuffix('_requests_per_second').replace('_', ' ')}: {v:.1f} requests/s" for k, v in throughput]
        if latency.get("per_request"):
            parts.append(f"per request median {latency['per_request']['median_ms']:.0f} ms (API, sequential)")
        usage = metrics.get("usage") or {}
        if usage.get("cost_per_1000_requests_usd") is not None:
            parts.append(f"{usage['cost_per_1000_requests_usd']:.4f} USD per 1000 requests ({usage['cost_usd']:.4f} USD total)")
        if metrics.get("complete") is False:
            parts.append(f"INCOMPLETE: {metrics['n_requests']} of {metrics['n_requests_expected']} requests")
        lines.append(f"  {run_dir.name}: " + ("; ".join(parts) if parts else "no latency recorded"))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", default=list(DEFAULT_RUNS), help="run directories, in column order")
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--sub", action="store_true", help="the 200-record sub-subset")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    args = parser.parse_args(argv)
    subset = load_subset(Path(args.subset))
    run_dirs = [Path(r) for r in args.runs]
    runs = [(d.name, *run_reports(d, subset, args.sub, Path(args.data_dir))) for d in run_dirs]
    print(f"baseline subset: {'200' if args.sub else subset['per_split']} records per split, gold-dependent questions")
    print("\n".join(table(runs) + footer(run_dirs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
