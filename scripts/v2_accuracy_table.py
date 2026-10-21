"""Full-label accuracy table at 0.6B: sft_06b, the broken REINFORCE-Brier run and the five full-label arms (review item 1).

    uv run python scripts/v2_accuracy_table.py

Reads runs/sft_06b, runs/rlcd_06b_brier_s0 and runs/rlcd_06b_<arm>_s<seed> for the
five arms of RESULTS_v2 section 3 (seeds 0, 1 and 2) from their results.jsonl.gz;
no model is loaded. For test_indomain, test_sst5 and test_yelp, and for the overall
block (gold-dependent questions, decision 44) and every question type present:

- each run's accuracy with its 95 percent record bootstrap interval (1000 multinomial
  resamples of the block's records, seed 0, the draws of metrics.bootstrap_intervals,
  so each interval equals the run's own metrics.json accuracy_ci, which is checked);
- per arm, the seed mean, its range over seeds, and an interval for the seed mean in
  which each draw averages the seeds' accuracies under the same record resample (so it
  covers record sampling for these seeds; the range shows seed-to-seed variation);
- each run and each seed mean minus sft_06b, paired on the same draws.

Score questions are the ordinal questions (qtype "score"): SST-5's `sentiment` on
test_sst5 (five levels in 70 percent of records, three levels in 30 percent, docs/DATA.md)
and Yelp's `stars` on test_yelp (five levels; the unseen score schema). test_indomain
has no score question.

Writes runs/v2_accuracy_06b/metrics.json and prints the table.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevmark.metrics import BOOTSTRAP_CHUNK, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, QUESTION_TYPES, QuestionResult, gold_dependent, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("test_indomain", "test_sst5", "test_yelp")
ARMS = ("sft_cont", "outcome", "outcome_minus_p", "direct_brier", "direct_log")
SEEDS = (0, 1, 2)
BROKEN = "rlcd_06b_brier_s0"
CI_TOL = 1e-12
SCORE_NOTE = (
    "score questions are the ordinal questions (qtype score): SST-5 sentiment on test_sst5 (five levels in 70 percent of records, "
    "three in 30 percent) and Yelp stars on test_yelp (five levels, the unseen score schema); test_indomain has none. overall is every gold-dependent question of the split (noul, choice and score; "
    "form nouls excluded, decision 44)."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(r: QuestionResult) -> tuple[str, str, str]:
    return (r.split, r.record_id, r.question_id)


def record_draws(results: Sequence[QuestionResult]) -> np.ndarray:
    """Question weights for every resample, drawn exactly as metrics.bootstrap_intervals draws them."""
    record_index: dict[str, int] = {}
    rec = np.array([record_index.setdefault(r.record_id, len(record_index)) for r in results])
    n_records = len(record_index)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = []
    for start in range(0, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, BOOTSTRAP_RESAMPLES - start)
        blocks.append(rng.multinomial(n_records, np.full(n_records, 1.0 / n_records), size=size)[:, rec].astype(float))
    return np.concatenate(blocks)


def interval(draws: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(draws, [2.5, 97.5])]


def table(runs_dir: Path) -> dict[str, Any]:
    names = ["sft_06b", BROKEN] + [f"rlcd_06b_{arm}_s{seed}" for arm in ARMS for seed in SEEDS]
    results = {name: {key(r): r for r in gold_dependent(read_results(runs_dir / name / "results.jsonl.gz")) if r.split in SPLITS} for name in names}
    reference = [r for r in results["sft_06b"].values()]
    for name, rs in results.items():
        if rs.keys() != results["sft_06b"].keys():
            raise SystemExit(f"{name}: its questions differ from sft_06b's on {', '.join(SPLITS)}")
    stored = {name: json.loads((runs_dir / name / "metrics.json").read_text()) for name in names}

    splits: dict[str, Any] = {}
    for split in SPLITS:
        blocks: dict[str, Any] = {}
        for block in ("overall", *QUESTION_TYPES):
            ref = [r for r in reference if r.split == split and (block == "overall" or r.qtype == block)]
            if not ref:
                continue
            weights = record_draws(ref)
            total = weights.sum(axis=1)
            correct = {name: np.array([float(results[name][key(r)].correct) for r in ref]) for name in names}
            draws = {name: weights @ c / total for name, c in correct.items()}
            point = {name: float(c.mean()) for name, c in correct.items()}
            runs: dict[str, Any] = {}
            for name in names:
                ci = interval(draws[name])
                own = stored[name]["splits"][split][block]
                if not math.isclose(own["accuracy"], point[name], abs_tol=CI_TOL) or any(not math.isclose(a, b, abs_tol=CI_TOL) for a, b in zip(own["accuracy_ci"], ci)):
                    raise SystemExit(f"{name}: {split}.{block} accuracy {point[name]} {ci} differs from its metrics.json {own['accuracy']} {own['accuracy_ci']}")
                runs[name] = {"accuracy": point[name], "accuracy_ci": ci}
                if name != "sft_06b":
                    delta = draws[name] - draws["sft_06b"]
                    runs[name].update({"minus_sft": point[name] - point["sft_06b"], "minus_sft_ci": interval(delta)})
            arms: dict[str, Any] = {}
            for arm in ARMS:
                members = [f"rlcd_06b_{arm}_s{seed}" for seed in SEEDS]
                mean_draws = np.mean([draws[m] for m in members], axis=0)
                values = [point[m] for m in members]
                mean = sum(values) / len(values)
                arms[arm] = {
                    "seeds": list(SEEDS),
                    "accuracy": mean,
                    "accuracy_range": [min(values), max(values)],
                    "accuracy_ci": interval(mean_draws),
                    "minus_sft": mean - point["sft_06b"],
                    "minus_sft_ci": interval(mean_draws - draws["sft_06b"]),
                    "by_seed": {str(seed): runs[m] for seed, m in zip(SEEDS, members)},
                }
            blocks[block] = {
                "n": len(ref),
                "n_records": len({r.record_id for r in ref}),
                "sft_06b": runs["sft_06b"],
                "reinforce_brier_s0": runs[BROKEN],
                "arms": arms,
            }
        splits[split] = blocks
    return {
        "runs": {name: str((runs_dir / name).relative_to(REPO)) if (runs_dir / name).resolve().is_relative_to(REPO) else str(runs_dir / name) for name in names},
        "results_sha256": {name: sha256(runs_dir / name / "results.jsonl.gz") for name in names},
        "run_git": {name: stored[name].get("git") for name in names},
        "splits": splits,
    }


def fmt(v: dict[str, Any]) -> str:
    return f"{v['accuracy']:.3f} [{v['accuracy_ci'][0]:.3f}, {v['accuracy_ci'][1]:.3f}]"


def print_table(result: dict[str, Any]) -> None:
    print("accuracy [95 percent record interval]; arms: seed mean [interval of the seed mean] (range over seeds 0, 1, 2)")
    print("split | block | n | sft_06b | REINFORCE-Brier s0 | " + " | ".join(ARMS))
    print("|".join(["---"] * (5 + len(ARMS))))
    for split, blocks in result["splits"].items():
        for block in ("overall", "score"):
            if block not in blocks:
                continue
            b = blocks[block]
            cells = [fmt(b["sft_06b"]), fmt(b["reinforce_brier_s0"])]
            cells += [f"{fmt(a)} ({a['accuracy_range'][0]:.3f}-{a['accuracy_range'][1]:.3f})" for a in b["arms"].values()]
            print(f"{split} | {block} | {b['n']} | " + " | ".join(cells))
    print("\nminus sft_06b, paired [95 percent record interval]")
    print("split | block | REINFORCE-Brier s0 | " + " | ".join(ARMS))
    for split, blocks in result["splits"].items():
        for block in ("overall", "score"):
            if block not in blocks:
                continue
            b = blocks[block]
            cells = [b["reinforce_brier_s0"]] + list(b["arms"].values())
            print(f"{split} | {block} | " + " | ".join(f"{c['minus_sft']:+.3f} [{c['minus_sft_ci'][0]:+.3f}, {c['minus_sft_ci'][1]:+.3f}]" for c in cells))
    print(f"\nnote: {result['note']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=str(REPO / "runs" / "v2_accuracy_06b"))
    args = parser.parse_args(argv)
    git = git_state()
    out_dir = Path(args.out)
    result = {
        "run_name": out_dir.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "code_sha256": {"v2_accuracy_table.py": sha256(Path(__file__))},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "record", "method": "percentile, 95 percent"},
        "note": SCORE_NOTE,
        "broken_run": f"{BROKEN}: REINFORCE with the Brier score of the sampled action as reward, p detached (RESULTS_v2 section 2, decision 52)",
        **table(Path(args.runs_dir)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print_table(result)
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
