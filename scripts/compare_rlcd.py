"""One table for an RLCD stage: SFT, SFT plus temperature and every RLCD arm of one size and seed (task 2.3).

    uv run python scripts/compare_rlcd.py --size 06b --seed 0

Reads runs/sft_<size>/results.jsonl.gz, the temperature of runs/sft_<size>_temp/
calibration.json (applied to the SFT probabilities, decision 50), and every
runs/rlcd_<size>_<arm>_s<seed>/ with a results.jsonl.gz. No model is loaded.

For every split and for its overall block and each question type (gold-dependent
questions only, decision 44):

- accuracy, ECE and NLL of every run, computed with jevmark.metrics from the stored
  probabilities and checked against each run's metrics.json
- per RLCD arm, a paired bootstrap by record (resamples of the split's records,
  the same draws for every arm): the accuracy difference against sft_<size> and the
  ECE difference against sft_<size>_temp, with 95 percent percentile intervals

and, per arm, the mean ECE over the four unseen schemas (test_agnews, test_emotion,
test_banking77, test_yelp) with the paired difference to sft_<size>_temp, each
bootstrap draw of the mean combining the same draw index of the four splits.

Per arm, the validation summary from training_log.jsonl on the fixed valid subset:
step 0 (the SFT adapter), the best step and the last step, each with accuracy,
ECE, NLL, KL to the SFT policy, the expected p of the chosen action (sum of p
squared) and the expected reward; plus train_summary.json's full-valid passes.

Writes runs/rlcd_stage<stage>_<size>/metrics.json (--out overrides) with the git
state, the sha256 of every results file read and of this script, and prints the
table.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevmark.calibration import scale_results
from jevmark.metrics import N_BINS, QUESTION_TYPES, QuestionResult, ece, gold_dependent, nll, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
ARM_ORDER = ("sft_cont", "outcome", "outcome_minus_p", "direct_brier", "direct_log", "brier", "log")
KNOWN_BROKEN = ("brier", "log")  # REINFORCE proper-score arms, decision 52
UNSEEN = ("test_agnews", "test_emotion", "test_banking77", "test_yelp")
VALID_KEYS = ("accuracy", "ece", "nll", "kl_to_ref", "expected_p_chosen", "expected_reward")
POINT_TOL = 1e-9


# Runs


def discover(runs_dir: Path, size: str, seed: int) -> list[tuple[str, Path]]:
    """(arm, run directory) for every rlcd_<size>_<arm>_s<seed> with results, in ARM_ORDER, unknown arms last by name."""
    prefix, suffix = f"rlcd_{size}_", f"_s{seed}"
    found = {}
    for path in runs_dir.glob(f"{prefix}*{suffix}"):
        if path.is_dir() and (path / "results.jsonl.gz").is_file():
            found[path.name[len(prefix) : -len(suffix)]] = path
    return sorted(found.items(), key=lambda item: (ARM_ORDER.index(item[0]) if item[0] in ARM_ORDER else len(ARM_ORDER), item[0]))


def key(r: QuestionResult) -> tuple[str, str, str]:
    return (r.split, r.record_id, r.question_id)


def aligned(reference: Sequence[QuestionResult], other: Sequence[QuestionResult], name: str) -> list[QuestionResult]:
    """`other` in the order of `reference`; both must hold exactly the same questions."""
    index = {key(r): r for r in other}
    if len(index) != len(other) or len(other) != len(reference) or any(key(r) not in index for r in reference):
        raise SystemExit(f"{name}: its questions differ from the SFT run's; the runs are not comparable")
    return [index[key(r)] for r in reference]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Metrics


def point(results: Sequence[QuestionResult]) -> dict[str, float]:
    correct = [r.correct for r in results]
    return {"n": len(results), "accuracy": sum(correct) / len(results), "ece": ece([r.top1 for r in results], correct), "nll": nll(results)}


def weighted_ece(conf: np.ndarray, correct: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ECE (equal-width bins on top-1 probability, as jevmark.metrics.ece) for each row of question weights."""
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    total = np.zeros(weights.shape[0])
    for b in range(N_BINS):
        members = bins == b
        if members.any():
            total += np.abs(weights[:, members] @ (conf[members] - correct[members]))
    return total / weights.sum(axis=1)


def record_weights(results: Sequence[QuestionResult], resamples: int, seed_text: str, chunk: int = 100):
    """Yield blocks of question weights: each row counts how often the question's record was drawn in one resample of the records."""
    records: dict[str, int] = {}
    record_of = np.array([records.setdefault(r.record_id, len(records)) for r in results])
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    for start in range(0, resamples, chunk):
        draws = rng.integers(0, len(records), size=(min(chunk, resamples - start), len(records)))
        counts = np.stack([np.bincount(d, minlength=len(records)) for d in draws]).astype(np.float64)
        yield counts[:, record_of]


def paired_deltas(sft: Sequence[QuestionResult], temp: Sequence[QuestionResult], arms: dict[str, Sequence[QuestionResult]], resamples: int, seed_text: str) -> dict[str, dict[str, np.ndarray]]:
    """Per arm, bootstrap draws of accuracy minus sft's and ECE minus sft_temp's, the same record draws for every arm."""
    arrays = lambda rs: (np.array([r.top1 for r in rs]), np.array([float(r.correct) for r in rs]))
    sft_conf, sft_correct = arrays(sft)
    temp_conf, temp_correct = arrays(temp)
    arm_arrays = {arm: arrays(rs) for arm, rs in arms.items()}
    draws: dict[str, dict[str, list[np.ndarray]]] = {arm: {"accuracy": [], "ece": []} for arm in arms}
    for weights in record_weights(sft, resamples, seed_text):
        total = weights.sum(axis=1)
        temp_ece = weighted_ece(temp_conf, temp_correct, weights)
        sft_accuracy = weights @ sft_correct / total
        for arm, (conf, correct) in arm_arrays.items():
            draws[arm]["accuracy"].append(weights @ correct / total - sft_accuracy)
            draws[arm]["ece"].append(weighted_ece(conf, correct, weights) - temp_ece)
    return {arm: {k: np.concatenate(v) for k, v in d.items()} for arm, d in draws.items()}


def interval(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def check_against_metrics(run_dir: Path, split: str, block: str, values: dict[str, float]) -> None:
    """The recomputed point estimates must equal the run's own metrics.json (when it has that block)."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return
    stored = json.loads(path.read_text())["splits"].get(split, {}).get(block)
    if not stored:
        return
    for k in ("accuracy", "ece", "nll"):
        if not math.isclose(stored[k], values[k], rel_tol=0, abs_tol=POINT_TOL):
            raise SystemExit(f"{path}: {split}.{block}.{k} is {stored[k]}, recomputed {values[k]}")


# Validation curves


def validation_summary(run_dir: Path) -> dict[str, Any]:
    log = [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines() if line.strip()]
    valids = [e for e in log if e.get("event") == "valid"]
    summary = json.loads((run_dir / "train_summary.json").read_text())
    pick = lambda e: {"step": e["step"], **{k: e[k] for k in VALID_KEYS}}
    best = next((e for e in valids if e["step"] == summary["best_step"]), None)
    return {
        "step_0": pick(valids[0]),
        "best": pick(best) if best else None,
        "last": pick(valids[-1]),
        "curve": [pick(e) for e in valids],
        "full_valid_best": {k: summary["final_valid"][k] for k in VALID_KEYS},
        "full_valid_last": {k: summary["final_valid_last"][k] for k in VALID_KEYS},
        "best_step": summary["best_step"],
        "steps": summary["steps"],
    }


# Main


def compare(runs_dir: Path, size: str, seed: int, resamples: int) -> dict[str, Any]:
    sft_dir, temp_dir = runs_dir / f"sft_{size}", runs_dir / f"sft_{size}_temp"
    temperature = float(json.loads((temp_dir / "calibration.json").read_text())["temperature"])
    sft_all = gold_dependent(read_results(sft_dir / "results.jsonl.gz"))
    temp_all = scale_results(sft_all, temperature)
    found = discover(runs_dir, size, seed)
    if not found:
        raise SystemExit(f"no rlcd_{size}_*_s{seed} run with results.jsonl.gz under {runs_dir}")
    arm_all = {arm: aligned(sft_all, gold_dependent(read_results(path / "results.jsonl.gz")), path.name) for arm, path in found}
    arm_dirs = dict(found)

    splits: dict[str, Any] = {}
    unseen_draws: dict[str, list[np.ndarray]] = {arm: [] for arm in arm_all}
    unseen_points: dict[str, list[float]] = {arm: [] for arm in [*arm_all, "sft", "sft_temp"]}
    for split in dict.fromkeys(r.split for r in sft_all):
        in_split = [i for i, r in enumerate(sft_all) if r.split == split]
        blocks = {}
        for block in ("overall", *QUESTION_TYPES):
            idx = [i for i in in_split if block == "overall" or sft_all[i].qtype == block]
            if not idx:
                continue
            sft = [sft_all[i] for i in idx]
            temp = [temp_all[i] for i in idx]
            arms = {arm: [rs[i] for i in idx] for arm, rs in arm_all.items()}
            values = {"sft": point(sft), "sft_temp": point(temp), **{arm: point(rs) for arm, rs in arms.items()}}
            check_against_metrics(sft_dir, split, block, values["sft"])
            check_against_metrics(temp_dir, split, block, values["sft_temp"])
            for arm in arms:
                check_against_metrics(arm_dirs[arm], split, block, values[arm])
            draws = paired_deltas(sft, temp, arms, resamples, f"{split}:{block}")
            for arm, d in draws.items():
                values[arm]["accuracy_minus_sft"] = values[arm]["accuracy"] - values["sft"]["accuracy"]
                values[arm]["accuracy_minus_sft_ci"] = interval(d["accuracy"])
                values[arm]["ece_minus_sft_temp"] = values[arm]["ece"] - values["sft_temp"]["ece"]
                values[arm]["ece_minus_sft_temp_ci"] = interval(d["ece"])
                if split in UNSEEN and block == "overall":
                    unseen_draws[arm].append(d["ece"])
            if split in UNSEEN and block == "overall":
                for name in unseen_points:
                    unseen_points[name].append(values[name]["ece"])
            blocks[block] = values
        splits[split] = blocks

    present = [s for s in UNSEEN if s in splits]
    unseen = {"splits": present}
    if len(present) == len(UNSEEN):
        unseen["mean_ece"] = {name: sum(v) / len(v) for name, v in unseen_points.items()}
        unseen["mean_ece_minus_sft_temp"] = {arm: unseen["mean_ece"][arm] - unseen["mean_ece"]["sft_temp"] for arm in arm_all}
        unseen["mean_ece_minus_sft_temp_ci"] = {arm: interval(np.mean(np.stack(d), axis=0)) for arm, d in unseen_draws.items()}

    return {
        "run_name": f"rlcd_stage1_{size}",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "size": size,
        "seed": seed,
        "runs": {
            "sft": str(sft_dir),
            "sft_temp": {"run": str(temp_dir), "temperature": temperature, "note": "the SFT probabilities scaled by this temperature (decision 50)"},
            **{arm: str(path) for arm, path in found},
        },
        "known_broken_arms": [arm for arm in arm_all if arm in KNOWN_BROKEN],
        "results_sha256": {"sft": sha256(sft_dir / "results.jsonl.gz"), **{arm: sha256(path / "results.jsonl.gz") for arm, path in found}},
        "bootstrap": {"resamples": resamples, "unit": "record", "interval": "95 percent percentile", "seed": "zlib.crc32 of '<split>:<block>', the same draws for every arm"},
        "note": "Gold-dependent questions only. accuracy_minus_sft is against sft; ece_minus_sft_temp against sft_temp (one global temperature fitted on valid). Point estimates equal each run's metrics.json.",
        "splits": splits,
        "unseen_schemas": unseen,
        "validation": {arm: validation_summary(path) for arm, path in found},
    }


def print_table(result: dict[str, Any]) -> None:
    arms = [a for a in result["runs"] if a not in ("sft", "sft_temp")]
    mark = lambda v: ("+" if v["ece_minus_sft_temp_ci"][1] < 0 else "x" if v["ece_minus_sft_temp_ci"][0] > 0 else " ") + ("v" if v["accuracy_minus_sft_ci"][1] < 0 else "^" if v["accuracy_minus_sft_ci"][0] > 0 else " ")
    print("accuracy / ECE per run; marks per arm: + ECE below sft_temp, x above it, v accuracy below sft, ^ above it (95 percent interval excludes 0)")
    print("split | block | n | sft | sft_temp | " + " | ".join(arms))
    for split, blocks in result["splits"].items():
        for block, v in blocks.items():
            cells = [f"{v['sft']['accuracy']:.3f}/{v['sft']['ece']:.3f}", f"{v['sft_temp']['accuracy']:.3f}/{v['sft_temp']['ece']:.3f}"]
            cells += [f"{v[a]['accuracy']:.3f}/{v[a]['ece']:.3f} {mark(v[a])}" for a in arms]
            print(f"{split} | {block} | {v['sft']['n']} | " + " | ".join(cells))
    unseen = result["unseen_schemas"]
    if "mean_ece" in unseen:
        print("\nmean ECE over the four unseen schemas; difference to sft_temp [95 percent interval]")
        print(f"sft {unseen['mean_ece']['sft']:.4f}, sft_temp {unseen['mean_ece']['sft_temp']:.4f}")
        for arm in arms:
            lo, hi = unseen["mean_ece_minus_sft_temp_ci"][arm]
            print(f"{arm}: {unseen['mean_ece'][arm]:.4f}, {unseen['mean_ece_minus_sft_temp'][arm]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nvalidation subset, step 0 -> last step (best step): KL to SFT, sum of p squared, expected reward, NLL")
    for arm in arms:
        v = result["validation"][arm]
        s0, last = v["step_0"], v["last"]
        print(f"{arm}: KL {last['kl_to_ref']:.4f}, sum p^2 {s0['expected_p_chosen']:.4f} -> {last['expected_p_chosen']:.4f}, expected reward {s0['expected_reward']:.4f} -> {last['expected_reward']:.4f}, "
              f"NLL {s0['nll']:.4f} -> {last['nll']:.4f} (best step {v['best_step']}, NLL {v['best']['nll']:.4f})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="06b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, default=1, help="names the output directory, runs/rlcd_stage<stage>_<size>")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    git = git_state()
    runs_dir = Path(args.runs_dir)
    result = compare(runs_dir, args.size, args.seed, args.resamples)
    out_dir = Path(args.out) if args.out else runs_dir / f"rlcd_stage{args.stage}_{args.size}"
    result["run_name"] = out_dir.name
    result = {"run_name": result.pop("run_name"), "created": result.pop("created"), "git": git, "code_sha256": {"compare_rlcd.py": sha256(Path(__file__))}, **result}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print_table(result)
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
