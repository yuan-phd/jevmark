"""Temperature scaling of a finished run, on CPU, from its results.jsonl.gz (task 2.1).

    uv run python scripts/calibrate.py runs/sft_06b

Fits one scalar temperature T on the gold-dependent questions of valid by minimising
the mean NLL of softmax(log p / T), which equals softmax(z / T) for the letter logits
z because log p and z differ by a constant per question (jevmark/calibration.py).
No model is loaded and no GPU is needed.

Writes runs/<run>_temp/, next to the source run, which is never modified:

- calibration.json: {"temperature": T, ...} in the checkpoint format JevMark.load reads
- config.yaml: copied from the source run
- source.txt: the source run; its adapter/ is the weights this temperature belongs to
- metrics.json: every split recomputed from the scaled probabilities, in the layout
  evaluate.py writes (metrics.reports_by_split), with the fit and two diagnostics
  under "calibration": a temperature per question type fitted on valid, and the
  valid NLL before and after scaling. Run-level fields that do not depend on the
  temperature (latency, batching precision, data hashes, precision) are the
  source run's and are copied with a note.
- metrics_oracle.json: a diagnostic only, never a result: for each split, a
  temperature fitted on that split's own gold-dependent questions and the split's
  metrics under it, the best any single temperature could do there.

--temperature applies a given T instead of fitting one (the tests use it with 1.0
to check that the scaled metrics equal evaluate.py's).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jevmark.calibration import fit_temperature, mean_nll, scale_results
from jevmark.metrics import QUESTION_TYPES, gold_dependent, read_results, reports_by_split, split_report
from jevmark.provenance import git_state

FIT_SPLIT = "valid"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def calibrate(run_dir: Path, temperature: float | None = None, out_dir: Path | None = None) -> Path:
    git = git_state()
    results = read_results(run_dir / "results.jsonl.gz")
    source_metrics = json.loads((run_dir / "metrics.json").read_text())
    fit_set = [r for r in gold_dependent(results) if r.split == FIT_SPLIT]
    if not fit_set:
        raise SystemExit(f"{run_dir} has no gold-dependent {FIT_SPLIT} questions to fit on")
    fitted = temperature is None
    t = fit_temperature(fit_set) if fitted else float(temperature)
    per_type = {}
    for qtype in QUESTION_TYPES:
        of_type = [r for r in fit_set if r.qtype == qtype]
        if of_type:
            per_type[qtype] = {"n": len(of_type), "temperature": fit_temperature(of_type)}
    calibration = {
        "temperature": t,
        "fitted": fitted,
        "fit_on": f"{FIT_SPLIT}, gold-dependent questions",
        "n_fit": len(fit_set),
        "valid_nll_at_t1": mean_nll(fit_set, 1.0),
        "valid_nll_at_t": mean_nll(fit_set, t),
        "per_type_diagnostic": per_type,
    }

    out_dir = out_dir or run_dir.parent / f"{run_dir.name}_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps({"temperature": t, "source_run": str(run_dir), "fit_on": calibration["fit_on"]}, indent=2) + "\n")
    shutil.copyfile(run_dir / "config.yaml", out_dir / "config.yaml")
    (out_dir / "source.txt").write_text(f"{run_dir}\ntemperature scaling of this run; the weights are its adapter/ (or the frozen backbone for a base run)\n")

    source = {"run": str(run_dir), "git": source_metrics.get("git"), "temperature": source_metrics.get("temperature")}
    metrics = {k: v for k, v in source_metrics.items() if k not in ("splits", "recomputed")}
    metrics.update(
        {
            "run_name": out_dir.name,
            "model_id": f"jevmark-{out_dir.name}",
            "ckpt": str(out_dir),
            "git": git,
            "created": _now(),
            "temperature": t,
            "calibration": calibration,
            "source": source,
            "source_fields_note": "latency, batching_precision, precision, data_files_sha256, device and wall_clock_seconds are the source run's; a temperature does not change them",
            "splits": reports_by_split(scale_results(results, t)),
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    by_split: dict[str, list] = {}
    for r in results:
        by_split.setdefault(r.split, []).append(r)
    oracle: dict[str, Any] = {
        "run_name": out_dir.name,
        "note": "diagnostic only: each split's temperature is fitted on that split's own gold-dependent questions, the best a single temperature could do there",
        "git": git,
        "created": _now(),
        "source": source,
        "temperatures": {},
        "splits": {},
    }
    for split, members in by_split.items():
        t_split = fit_temperature(gold_dependent(members))
        oracle["temperatures"][split] = t_split
        oracle["splits"][split] = split_report(scale_results(members, t_split))
    (out_dir / "metrics_oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a run directory holding results.jsonl.gz, metrics.json and config.yaml")
    parser.add_argument("--temperature", type=float, default=None, help="apply this temperature instead of fitting one on valid")
    parser.add_argument("--out", default=None, help="output directory; default runs/<run>_temp next to the source")
    args = parser.parse_args(argv)
    out = calibrate(Path(args.run_dir), args.temperature, Path(args.out) if args.out else None)
    metrics = json.loads((out / "metrics.json").read_text())
    cal = metrics["calibration"]
    per_type = ", ".join(f"{k} {v['temperature']:.3f}" for k, v in cal["per_type_diagnostic"].items())
    print(f"{out}: T {cal['temperature']:.4f} ({'fitted' if cal['fitted'] else 'given'}), valid NLL {cal['valid_nll_at_t1']:.4f} -> {cal['valid_nll_at_t']:.4f}; per type on valid: {per_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
