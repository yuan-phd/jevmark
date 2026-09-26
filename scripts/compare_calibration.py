"""One table for temperature scaling (task 2.1): SFT, SFT plus a global T, SFT plus a per-split oracle T.

    uv run python scripts/compare_calibration.py

Reads runs/sft_<size>/metrics.json, runs/sft_<size>_temp/metrics.json and
runs/sft_<size>_temp/metrics_oracle.json for both sizes, written by calibrate.py.
Rows are the nine splits, overall and per question type (gold-dependent questions,
decision 44); columns are ECE and NLL for each variant at each size, and the oracle
temperature of each split. Above the table: the fitted temperatures, global and the
per-type diagnostics, for the base and SFT runs. Below it: coverage and accuracy at
confidence thresholds 0.5, 0.8, 0.9 and 0.95 for SFT and SFT plus the global T on
four splits. Accuracy is the same in every variant (a temperature changes no
prediction), so it is not repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SIZES = ("06b", "17b")
BLOCKS = ("overall", "noul", "choice", "score")
COVERAGE_SPLITS = ("test_indomain", "test_unseen_intents", "test_emotion", "test_yelp")
THRESHOLDS = (0.5, 0.8, 0.9, 0.95)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def temperatures(runs_dir: Path) -> list[str]:
    lines = ["fitted temperatures (valid, gold-dependent questions; per type is a diagnostic only)"]
    for kind in ("base", "sft"):
        for size in SIZES:
            cal = load(runs_dir / f"{kind}_{size}_temp" / "metrics.json")["calibration"]
            per_type = ", ".join(f"{k} {v['temperature']:.3f}" for k, v in cal["per_type_diagnostic"].items())
            lines.append(f"  {kind}_{size}: global T {cal['temperature']:.4f} (valid NLL {cal['valid_nll_at_t1']:.4f} -> {cal['valid_nll_at_t']:.4f}); per type {per_type}")
    return lines


def table(runs_dir: Path) -> list[str]:
    variants = {}
    for size in SIZES:
        variants[size] = (
            load(runs_dir / f"sft_{size}" / "metrics.json")["splits"],
            load(runs_dir / f"sft_{size}_temp" / "metrics.json")["splits"],
            load(runs_dir / f"sft_{size}_temp" / "metrics_oracle.json"),
        )
    head = f"{'split / type':28}{'n':>6}"
    for size in SIZES:
        head += f" | sft_{size}: ECE    NLL  | +T: ECE    NLL  | oracle: ECE   NLL    T  "
    lines = [head]
    for split in variants[SIZES[0]][0]:
        for block in BLOCKS:
            first = variants[SIZES[0]][0][split].get(block)
            if first is None:
                continue
            label = split if block == "overall" else f"  {block}"
            row = f"{label:28}{first['n']:>6}"
            for size in SIZES:
                sft, temp, oracle = variants[size]
                a, b, c = sft[split][block], temp[split][block], oracle["splits"][split][block]
                t = f"{oracle['temperatures'][split]:5.2f}" if block == "overall" else "     "
                row += f" |      {a['ece']:.3f}  {a['nll']:.3f} |    {b['ece']:.3f}  {b['nll']:.3f} |      {c['ece']:.3f} {c['nll']:.3f} {t}"
            lines.append(row)
    return lines


def coverage(runs_dir: Path) -> list[str]:
    lines = ["", "coverage / accuracy at confidence thresholds (response confidence field), SFT and SFT plus global T"]
    header = f"{'split / type':28}{'run':14}" + "".join(f"{f't={t}':>15}" for t in THRESHOLDS)
    lines.append(header)
    for split in COVERAGE_SPLITS:
        for size in SIZES:
            for name in (f"sft_{size}", f"sft_{size}_temp"):
                blocks = load(runs_dir / name / "metrics.json")["splits"][split]
                for block in ("noul", "choice", "score"):
                    if block not in blocks:
                        continue
                    rows = {round(r["threshold"], 2): r for r in blocks[block]["coverage"]}
                    cells = "".join(f"{rows[t]['coverage']:>8.3f} / {rows[t]['accuracy'] if rows[t]['accuracy'] is not None else float('nan'):.3f}" for t in THRESHOLDS)
                    lines.append(f"{split + ' ' + block:28}{name:14}{cells}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    print("\n".join(temperatures(runs_dir) + [""] + table(runs_dir) + coverage(runs_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
