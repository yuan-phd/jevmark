"""Score a run against the stochastic-outcome environment, offline from its results.jsonl.gz (task 2.5, decision 54).

    uv run python scripts/evaluate_env.py runs/sft_06b
    uv run python scripts/evaluate_env.py runs/rlcd_06b_noisy_direct_brier_s0

No model is loaded. On every split, gold-dependent questions only (decision 44),
every metric of jevmark.environment.env_metrics against the environment's target
distribution theta: expected Brier and cross-entropy against theta (with their
minima, reached at p = theta), accuracy against gold, the calibration table by K
(mean top-1 probability, mean probability on gold, 1 - eta(K), count, gap) with its
count-weighted gap, and ECE against outcomes sampled once from theta with a fixed
seed per split (the same outcomes for every variant and every run).

Three variants of the run's distributions:

- raw: as evaluated
- global_T: one temperature fitted on valid from bandit outcomes the run's own
  policy collects in the environment (G actions per question from
  (1 - epsilon) p + epsilon uniform, seed 0; the binary log loss of p_T(a) against
  each outcome), applied to every split; for sft_<size> this is the post-hoc
  competitor that sees the same kind of feedback RLCD sees
- oracle_T_by_K: one temperature per K fitted on each split's own theta; an
  oracle, since it reads theta on the split it is scored on

and the reference theta itself (the best any distribution can do).

Writes runs/<run>_env/metrics.json (--out overrides) with the git state and the
sha256 of the results file and of this script.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from jevmark import environment
from jevmark.calibration import scale_results
from jevmark.environment import ETA_BASE, ETA_CAP, ETA_SLOPE, env_metrics, fit_bandit_temperature, oracle_temperatures_by_k, scale_by_k, theta
from jevmark.metrics import QuestionResult, gold_dependent, read_results
from jevmark.provenance import git_state

FIT_SPLIT = "valid"


def by_split(results: Sequence[QuestionResult]) -> dict[str, list[QuestionResult]]:
    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(r.split, []).append(r)
    return groups


def outcome_seed(split: str) -> str:
    """The seed text of a split's sampled outcomes, shared by every variant and every run."""
    return f"env-outcomes:{split}"


def theta_results(results: Sequence[QuestionResult]) -> list[QuestionResult]:
    return [replace(r, probs=tuple(float(x) for x in theta(r.k, r.gold))) for r in results]


def evaluate(results: Sequence[QuestionResult], group_size: int, epsilon: float, seed: int) -> dict[str, Any]:
    splits = by_split(results)
    if FIT_SPLIT not in splits:
        raise SystemExit(f"no gold-dependent {FIT_SPLIT} questions to fit the global temperature on")
    global_t = fit_bandit_temperature(splits[FIT_SPLIT], group_size, epsilon, seed)
    out: dict[str, Any] = {
        "global_T": {"temperature": global_t, "fit": {"split": FIT_SPLIT, "group_size": group_size, "epsilon": epsilon, "seed": seed, "objective": "mean binary log loss of p_T(a) against bandit outcomes of the run's own policy"}, "splits": {}},
        "raw": {"splits": {}},
        "oracle_T_by_K": {"splits": {}},
        "theta": {"splits": {}},
    }
    for split, rs in splits.items():
        seed_text = outcome_seed(split)
        out["raw"]["splits"][split] = env_metrics(rs, seed_text)
        out["global_T"]["splits"][split] = env_metrics(scale_results(rs, global_t), seed_text)
        temps = oracle_temperatures_by_k(rs)
        out["oracle_T_by_K"]["splits"][split] = {"temperatures": {str(k): t for k, t in temps.items()}, **env_metrics(scale_by_k(rs, temps), seed_text)}
        out["theta"]["splits"][split] = env_metrics(theta_results(rs), seed_text)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run directory holding results.jsonl.gz")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="default runs/<run>_env next to the run")
    args = parser.parse_args(argv)
    git = git_state()
    run_dir = Path(args.run)
    source = run_dir / "results.jsonl.gz"
    results = gold_dependent(read_results(source))
    variants = evaluate(results, args.group_size, args.epsilon, args.seed)
    out_dir = Path(args.out) if args.out else run_dir.parent / f"{run_dir.name}_env"
    metrics = {
        "run_name": out_dir.name,
        "source_run": str(run_dir),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("evaluate_env.py", Path(__file__)), ("jevmark/environment.py", Path(environment.__file__)))},
        "environment": {"eta": f"min({ETA_CAP}, {ETA_BASE} + {ETA_SLOPE} (K - 2))", "theta": "1 - eta on gold, eta / (K - 1) on each other option", "decision": 54},
        "note": "Gold-dependent questions only. Every variant's ECE uses the same outcomes, sampled once per split from theta with seed text env-outcomes:<split>. oracle_T_by_K reads theta on the split it is scored on and is a bound, not a method.",
        "variants": variants,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{run_dir.name}: global T {variants['global_T']['temperature']:.3f} (bandit fit on {FIT_SPLIT})")
    print("split | variant | expected Brier (min) | cross-entropy (entropy) | accuracy | calibration gap | ECE (sampled)")
    for split in variants["raw"]["splits"]:
        for name in ("raw", "global_T", "oracle_T_by_K", "theta"):
            m = variants[name]["splits"][split]
            print(f"{split} | {name} | {m['expected_brier']:.4f} ({m['expected_brier_min']:.4f}) | {m['cross_entropy']:.4f} ({m['entropy_theta']:.4f}) | {m['accuracy']:.4f} | {m['calibration_gap']:.4f} | {m['ece_sampled']:.4f}")
    print(f"wrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
