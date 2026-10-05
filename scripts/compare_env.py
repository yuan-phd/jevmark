"""Stage 3 table: SFT, its post-hoc temperatures and the noisy RLCD arms, scored against the environment's theta (task 2.5, decision 54).

    uv run python scripts/compare_env.py --size 06b
    uv run python scripts/compare_env.py --size 06b --seeds 0 1 2

Reads runs/sft_<size>/results.jsonl.gz and every runs/rlcd_<size>_noisy_<arm>_s<seed>/
with a results.jsonl.gz (all seeds found unless --seeds). No model is loaded.
Gold-dependent questions only (decision 44), on test_indomain, test_unseen_intents,
test_sst5 and the four unseen schemas. The columns:

- sft: sft_<size> as evaluated
- sft_global_T: sft_<size> with one temperature fitted on valid from the SFT
  policy's own bandit outcomes in the environment (G 4, epsilon 0.1, seed 0;
  jevmark.environment.fit_bandit_temperature), the post-hoc competitor that sees
  the kind of feedback RLCD sees
- sft_oracle_T: sft_<size> with one temperature per K fitted on the split's own
  theta; an oracle bound, since it reads theta on the split it is scored on
- each noisy arm: the seed mean with the range across seeds

and the metrics: expected Brier and cross-entropy against theta, accuracy against
gold, and the calibration gap by K (count-weighted mean over K of |mean top-1
probability - mean expected hit rate of the top-1 option|; jevmark.environment).

Paired bootstrap by record (1000 resamples of a split's records, the same draws
for every column): for every column, the differences in expected Brier,
cross-entropy and calibration gap against sft_global_T and against sft_oracle_T,
and the accuracy difference against sft; for an arm, each draw averages its seeds'
differences under that draw (the interval covers record sampling for these seeds;
the range shows seed-to-seed variation).

Writes runs/rlcd_stage<stage>_<size>/metrics.json (default stage 3; --out
overrides) with the git state and the sha256 of every results file read and of
this script, and prints one table per split.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevmark import environment
from jevmark.calibration import scale_results
from jevmark.environment import env_metrics, fit_bandit_temperature, oracle_temperatures_by_k, question_arrays, scale_by_k
from jevmark.metrics import QuestionResult, gold_dependent, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("test_indomain", "test_unseen_intents", "test_sst5", "test_agnews", "test_emotion", "test_banking77", "test_yelp")
ARM_ORDER = ("direct_brier", "direct_log", "outcome_minus_p", "outcome")
FIT_SPLIT = "valid"
METRICS = ("expected_brier", "cross_entropy", "accuracy", "calibration_gap")
REFERENCES = ("sft_global_T", "sft_oracle_T")


def discover(runs_dir: Path, size: str, seeds: Sequence[int] | None) -> dict[tuple[str, int], Path]:
    pattern = re.compile(rf"^rlcd_{re.escape(size)}_noisy_(.+)_s(\d+)$")
    found = {}
    for path in runs_dir.glob(f"rlcd_{size}_noisy_*_s*"):
        match = pattern.match(path.name)
        if match and path.is_dir() and (path / "results.jsonl.gz").is_file():
            seed = int(match.group(2))
            if seeds is None or seed in seeds:
                found[(match.group(1), seed)] = path
    order = lambda run: (ARM_ORDER.index(run[0]) if run[0] in ARM_ORDER else len(ARM_ORDER), run[0], run[1])
    return {run: found[run] for run in sorted(found, key=order)}


def key(r: QuestionResult) -> tuple[str, str, str]:
    return (r.split, r.record_id, r.question_id)


def aligned(reference: Sequence[QuestionResult], other: Sequence[QuestionResult], name: str) -> list[QuestionResult]:
    index = {key(r): r for r in other}
    if len(index) != len(other) or len(other) != len(reference) or any(key(r) not in index for r in reference):
        raise SystemExit(f"{name}: its questions differ from the SFT run's; the runs are not comparable")
    return [index[key(r)] for r in reference]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: Path) -> str:
    """The path as recorded in metrics.json: relative to the repository when inside it, so no local directory is committed."""
    resolved = path.resolve()
    return str(resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else path)


def interval(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


class Weighted:
    """The four metrics of one column under rows of record-resample weights."""

    def __init__(self, results: Sequence[QuestionResult]):
        a = question_arrays(results)
        self.brier, self.ce, self.correct = a["brier"], a["ce"], a["correct"]
        self.diff = a["top1"] - a["top1_hit"]
        ks = sorted(set(a["k"]))
        self.members = np.stack([a["k"] == k for k in ks], axis=1).astype(float)  # questions x K

    def __call__(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        total = weights.sum(axis=1)
        gap = np.abs(weights @ (self.members * self.diff[:, None])).sum(axis=1) / total
        return {"expected_brier": weights @ self.brier / total, "cross_entropy": weights @ self.ce / total, "accuracy": weights @ self.correct / total, "calibration_gap": gap}


def record_weights(results: Sequence[QuestionResult], resamples: int, seed_text: str, chunk: int = 100):
    records: dict[str, int] = {}
    record_of = np.array([records.setdefault(r.record_id, len(records)) for r in results])
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    for start in range(0, resamples, chunk):
        draws = rng.integers(0, len(records), size=(min(chunk, resamples - start), len(records)))
        counts = np.stack([np.bincount(d, minlength=len(records)) for d in draws]).astype(np.float64)
        yield counts[:, record_of]


def compare(runs_dir: Path, size: str, seeds: Sequence[int] | None, resamples: int, group_size: int, epsilon: float, seed: int) -> dict[str, Any]:
    sft_dir = runs_dir / f"sft_{size}"
    sft_all = gold_dependent(read_results(sft_dir / "results.jsonl.gz"))
    valid = [r for r in sft_all if r.split == FIT_SPLIT]
    if not valid:
        raise SystemExit(f"{sft_dir}: no gold-dependent {FIT_SPLIT} questions to fit the global temperature on")
    global_t = fit_bandit_temperature(valid, group_size, epsilon, seed)
    found = discover(runs_dir, size, seeds)
    run_all = {run: aligned(sft_all, gold_dependent(read_results(path / "results.jsonl.gz")), path.name) for run, path in found.items()}
    arms: dict[str, list[int]] = {}
    for arm, s in found:
        arms.setdefault(arm, []).append(s)

    splits: dict[str, Any] = {}
    for split in SPLITS:
        idx = [i for i, r in enumerate(sft_all) if r.split == split]
        if not idx:
            continue
        sft = [sft_all[i] for i in idx]
        temps = oracle_temperatures_by_k(sft)
        columns: dict[Any, list[QuestionResult]] = {"sft": sft, "sft_global_T": scale_results(sft, global_t), "sft_oracle_T": scale_by_k(sft, temps)}
        for run, rs in run_all.items():
            columns[run] = [rs[i] for i in idx]
        seed_text = f"env-outcomes:{split}"
        points = {name: env_metrics(rs, seed_text) for name, rs in columns.items()}
        weighted = {name: Weighted(rs) for name, rs in columns.items()}
        draws: dict[Any, dict[str, list[np.ndarray]]] = {name: {m: [] for m in METRICS} for name in columns}
        for weights in record_weights(sft, resamples, f"stage3:{split}"):
            for name, w in weighted.items():
                for m, values in w(weights).items():
                    draws[name][m].append(values)
        draws = {name: {m: np.concatenate(v) for m, v in d.items()} for name, d in draws.items()}

        def deltas(names: Sequence[Any]) -> dict[str, Any]:
            """Seed-mean differences and their intervals for the given columns (one name, or an arm's seeds)."""
            out: dict[str, Any] = {}
            mean = lambda name, m: np.mean([draws[n][m] for n in names], axis=0)
            point = lambda m: float(np.mean([points[n][m] for n in names]))
            for ref in REFERENCES:
                for m in ("expected_brier", "cross_entropy", "calibration_gap"):
                    out[f"{m}_minus_{ref}"] = point(m) - points[ref][m]
                    out[f"{m}_minus_{ref}_ci"] = interval(mean(names, m) - draws[ref][m])
            out["accuracy_minus_sft"] = point("accuracy") - points["sft"]["accuracy"]
            out["accuracy_minus_sft_ci"] = interval(mean(names, "accuracy") - draws["sft"]["accuracy"])
            return out

        block: dict[str, Any] = {"global_T": global_t, "oracle_T_by_K": {str(k): t for k, t in temps.items()}}
        for name in ("sft", "sft_global_T", "sft_oracle_T"):
            block[name] = {**points[name], **deltas([name])}
        block["arms"] = {}
        for arm, arm_seeds in arms.items():
            names = [(arm, s) for s in arm_seeds]
            per_seed = {str(s): points[(arm, s)] for s in arm_seeds}
            mean = {m: float(np.mean([p[m] for p in per_seed.values()])) for m in METRICS}
            ranges = {f"{m}_range": [min(p[m] for p in per_seed.values()), max(p[m] for p in per_seed.values())] for m in METRICS}
            block["arms"][arm] = {"seeds": arm_seeds, "mean": {**mean, **ranges, **deltas(names)}, "by_seed": per_seed}
        splits[split] = block

    return {
        "size": size,
        "seeds": sorted({s for _, s in found}),
        "arms": arms,
        "runs": {"sft": shown(sft_dir), **{f"{a}_s{s}": shown(p) for (a, s), p in found.items()}},
        "results_sha256": {"sft": sha256(sft_dir / "results.jsonl.gz"), **{f"{a}_s{s}": sha256(p / "results.jsonl.gz") for (a, s), p in found.items()}},
        "global_T": {"temperature": global_t, "fit": {"split": FIT_SPLIT, "group_size": group_size, "epsilon": epsilon, "seed": seed}},
        "bootstrap": {"resamples": resamples, "unit": "record", "interval": "95 percent percentile", "seed": "zlib.crc32 of 'stage3:<split>', the same draws for every column"},
        "note": "Gold-dependent questions only; metrics against the environment's theta (decision 54). sft_oracle_T reads theta on the split it is scored on and is a bound, not a method. ECE fields use outcomes sampled once per split (seed text env-outcomes:<split>).",
        "splits": splits,
    }


def print_tables(result: dict[str, Any]) -> None:
    arms = list(result["arms"])
    print(f"size {result['size']}, seeds {result['seeds']}, global T {result['global_T']['temperature']:.3f}; arm cells are seed means [range of the calibration gap]")
    print("marks: vs sft_global_T then vs sft_oracle_T on cross-entropy: + lower, x higher (95 percent interval excludes 0)")
    for split, block in result["splits"].items():
        print(f"\n{split}")
        print("column | expected Brier | cross-entropy | accuracy | calibration gap | marks")
        rows = [(name, block[name]) for name in ("sft", "sft_global_T", "sft_oracle_T")] + [(arm, block["arms"][arm]["mean"]) for arm in arms]
        for name, v in rows:
            marks = ""
            for ref in ("sft_global_T", "sft_oracle_T"):
                lo, hi = v[f"cross_entropy_minus_{ref}_ci"]
                marks += "+" if hi < 0 else "x" if lo > 0 else "."
            spread = f" [{v['calibration_gap_range'][0]:.4f}-{v['calibration_gap_range'][1]:.4f}]" if "calibration_gap_range" in v and len(block["arms"][name]["seeds"]) > 1 else ""
            print(f"{name} | {v['expected_brier']:.4f} | {v['cross_entropy']:.4f} | {v['accuracy']:.4f} | {v['calibration_gap']:.4f}{spread} | {marks}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="06b")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="seed of the bandit samples the global temperature is fitted on")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    git = git_state()
    runs_dir = Path(args.runs_dir)
    result = compare(runs_dir, args.size, args.seeds, args.resamples, args.group_size, args.epsilon, args.seed)
    out_dir = Path(args.out) if args.out else runs_dir / f"rlcd_stage{args.stage}_{args.size}"
    result = {
        "run_name": out_dir.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "code_sha256": {"compare_env.py": sha256(Path(__file__)), "jevmark/environment.py": sha256(Path(environment.__file__))},
        **result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print_tables(result)
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
