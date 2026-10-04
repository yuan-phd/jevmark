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

--fit-log (v3, task 3.3, decision 56) is the temperature learner:

    uv run python scripts/calibrate.py runs/v3_06b_zeroshot --fit-log runs/v3_log_s0/log.jsonl --n 5000

One scalar T is fitted by maximum likelihood of the logged outcomes under
softmax(log p / T) of the logged action, on the first 0.9 N interactions of the log
(jevmark/feedback.py; the same training part the trained learners use; no gold),
from the probabilities the log stores. It is applied to every split in the source
run's results.jsonl.gz, which must be the logging policy's own evaluation (the
adapter sha256 in its metrics.json must equal the log's), and written to
runs/v3_<size>_temp_n<N>/ unless --out is given.

With --fit-probs PREFIX_RUN the temperature goes on top of a trained learner:

    uv run python scripts/calibrate.py runs/v3_06b_direct_brier_n5000_s0 --fit-log runs/v3_log_s0/log.jsonl --n 5000 \
        --fit-probs runs/v3_06b_direct_brier_n5000_s0_prefix

The probabilities of the logged actions are the learner's own on the records it
trained on (evaluate.py --log-prefix, same adapter, same log, same N), not the
logging policy's stored in the log; the actions and outcomes are the log's. The
result is applied to the learner's results.jsonl.gz and written to runs/<run>_temp/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jevmark.calibration import fit_outcome_temperature, fit_temperature, mean_nll, outcome_log_likelihood, scale_results
from jevmark.feedback import TRAIN_SHARE, Interaction, file_sha256, prefix, read_log
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

    return write_scaled_run(run_dir, results, source_metrics, t, calibration, out_dir or run_dir.parent / f"{run_dir.name}_temp", git)


def write_scaled_run(run_dir: Path, results: list, source_metrics: dict[str, Any], t: float, calibration: dict[str, Any], out_dir: Path, git: dict[str, Any]) -> Path:
    """runs/<out>/: calibration.json, config.yaml, source.txt, metrics.json and metrics_oracle.json for temperature t."""
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


def learner_probs(fit: Sequence[Interaction], probs_dir: Path, source_metrics: dict[str, Any], log_sha256: str, n: int) -> list[tuple[float, ...]]:
    """The learner's probabilities over each interaction's options, from its evaluation on the log prefix (evaluate.py --log-prefix)."""
    probs_metrics = json.loads((probs_dir / "metrics.json").read_text())
    if probs_metrics.get("adapter_sha256") != source_metrics.get("adapter_sha256"):
        raise SystemExit(f"{probs_dir} evaluates another adapter than the source run")
    on = probs_metrics.get("log_prefix") or {}
    if on.get("log_sha256") != log_sha256 or on.get("n") != n:
        raise SystemExit(f"{probs_dir} was not evaluated on the first 0.9 N interactions of this log at N {n}")
    by_record = {(r.record_id, r.question_id): r for r in read_results(probs_dir / "results.jsonl.gz")}
    out = []
    for i in fit:
        r = by_record.get((i.record_id, i.question_id))
        if r is None or tuple(r.labels) != tuple(i.labels):
            raise SystemExit(f"{probs_dir}: no result with the logged option order for {i.record_id}")
        out.append(tuple(r.probs))
    return out


def calibrate_from_log(run_dir: Path, log_path: Path, n: int, out_dir: Path | None = None, train_share: float = TRAIN_SHARE, probs_dir: Path | None = None) -> Path:
    """The v3 temperature learner: T fitted on the logged outcomes of the first train_share x N interactions, applied to run_dir's results.

    Without probs_dir the probabilities are the logging policy's, stored in the log, and run_dir must
    evaluate that policy. With probs_dir they are run_dir's own learner's on the same interactions.
    """
    git = git_state()
    source_metrics = json.loads((run_dir / "metrics.json").read_text())
    log_sha256 = file_sha256(log_path)
    fit = prefix(read_log(log_path), n, train_share).train
    if probs_dir is None:
        log_metrics_path = log_path.parent / "metrics.json"
        log_metrics = json.loads(log_metrics_path.read_text()) if log_metrics_path.is_file() else {}
        logged_adapter = log_metrics.get("logging_policy", {}).get("adapter_sha256")
        if logged_adapter and source_metrics.get("adapter_sha256") and logged_adapter != source_metrics["adapter_sha256"]:
            raise SystemExit(f"{run_dir} evaluates another adapter than the policy that drew {log_path}; the temperature belongs to the logging policy (pass --fit-probs for a trained learner)")
        probs = [i.probs for i in fit]
    else:
        probs = learner_probs(fit, probs_dir, source_metrics, log_sha256, n)
    actions, outcomes = [i.action for i in fit], [i.outcome for i in fit]
    t = fit_outcome_temperature(probs, actions, outcomes)
    calibration = {
        "temperature": t,
        "fitted": True,
        "fit_on": f"the first {len(fit)} interactions of {log_path} (N {n}), Bernoulli log-likelihood of the logged outcomes of the chosen actions"
        + ("" if probs_dir is None else f", under the learner's probabilities from {probs_dir}"),
        "log": str(log_path),
        "log_sha256": log_sha256,
        "probs_from": "the log (the logging policy)" if probs_dir is None else str(probs_dir),
        "n": n,
        "n_fit": len(fit),
        "outcome_log_likelihood_at_t1": outcome_log_likelihood(probs, actions, outcomes, 1.0),
        "outcome_log_likelihood_at_t": outcome_log_likelihood(probs, actions, outcomes, t),
    }
    if out_dir is None and probs_dir is not None:
        out_dir = run_dir.parent / f"{run_dir.name}_temp"
    if out_dir is None:
        size = re.search(r"(06b|17b)", run_dir.name)
        if size is None:
            raise SystemExit(f"cannot tell the backbone size from {run_dir.name}; pass --out")
        out_dir = run_dir.parent / f"v3_{size.group(1)}_temp_n{n}"
    return write_scaled_run(run_dir, read_results(run_dir / "results.jsonl.gz"), source_metrics, t, calibration, out_dir, git)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a run directory holding results.jsonl.gz, metrics.json and config.yaml")
    parser.add_argument("--temperature", type=float, default=None, help="apply this temperature instead of fitting one on valid")
    parser.add_argument("--out", default=None, help="output directory; default runs/<run>_temp next to the source (runs/v3_<size>_temp_n<N> with --fit-log)")
    parser.add_argument("--fit-log", default=None, help="v3: fit T on the logged outcomes of this deployment-feedback log instead of on valid")
    parser.add_argument("--n", type=int, default=None, help="with --fit-log: the learner's N; T is fitted on the first 0.9 N interactions")
    parser.add_argument("--fit-probs", default=None, help="with --fit-log: a trained learner's evaluation on the log prefix (evaluate.py --log-prefix); its probabilities replace the logging policy's")
    args = parser.parse_args(argv)
    if args.fit_log is not None:
        if args.n is None or args.temperature is not None:
            parser.error("--fit-log needs --n and does not take --temperature")
        out = calibrate_from_log(Path(args.run_dir), Path(args.fit_log), args.n, Path(args.out) if args.out else None, probs_dir=Path(args.fit_probs) if args.fit_probs else None)
        cal = json.loads((out / "metrics.json").read_text())["calibration"]
        print(f"{out}: T {cal['temperature']:.4f} fitted on {cal['n_fit']} logged outcomes; log-likelihood {cal['outcome_log_likelihood_at_t1']:.4f} -> {cal['outcome_log_likelihood_at_t']:.4f}")
        return 0
    elif args.n is not None or args.fit_probs is not None:
        parser.error("--n and --fit-probs are --fit-log options")
    out = calibrate(Path(args.run_dir), args.temperature, Path(args.out) if args.out else None)
    metrics = json.loads((out / "metrics.json").read_text())
    cal = metrics["calibration"]
    per_type = ", ".join(f"{k} {v['temperature']:.3f}" for k, v in cal["per_type_diagnostic"].items())
    print(f"{out}: T {cal['temperature']:.4f} ({'fitted' if cal['fitted'] else 'given'}), valid NLL {cal['valid_nll_at_t1']:.4f} -> {cal['valid_nll_at_t']:.4f}; per type on valid: {per_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
