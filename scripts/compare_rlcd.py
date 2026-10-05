"""One table for an RLCD stage: SFT, SFT plus temperature and every RLCD arm of one size, over one or more seeds (task 2.3).

    uv run python scripts/compare_rlcd.py --size 06b --seeds 0            # stage 1
    uv run python scripts/compare_rlcd.py --size 06b --seeds 0 1 2 --stage 2
    uv run python scripts/compare_rlcd.py --size 17b --seeds 0 --stage 2

Reads runs/sft_<size>/results.jsonl.gz, the temperature of runs/sft_<size>_temp/
calibration.json (applied to the SFT probabilities, decision 50), and every
runs/rlcd_<size>_<arm>_s<seed>/ with a results.jsonl.gz for the chosen seeds (all
seeds found when --seeds is not given). No model is loaded.

For every split and for its overall block and each question type (gold-dependent
questions only, decision 44):

- accuracy, ECE and NLL of every run, computed with jevmark.metrics from the stored
  probabilities and checked against each run's metrics.json
- per arm and seed, a paired bootstrap by record (resamples of the split's records,
  the same draws for every run): the accuracy difference against sft_<size> and the
  ECE difference against sft_<size>_temp, with 95 percent percentile intervals
- per arm over its seeds: the mean and the range (min, max) of accuracy and ECE,
  and the same two differences for the seed mean, each bootstrap draw averaging the
  seeds' differences under that draw (so the interval covers record sampling for
  these seeds, not seed-to-seed variation, which the range shows)

and, per arm, the mean ECE over the four unseen schemas (test_agnews, test_emotion,
test_banking77, test_yelp) with the paired difference to sft_<size>_temp, each
bootstrap draw of the mean combining the same draw index of the four splits.

Temperature ablation: for every arm whose runs all have runs/<run>_temp/
calibration.json (scripts/calibrate.py, one T fitted on valid), the arm's
probabilities scaled by its own T, and its unseen-schema ECE before and after
against sft_<size>_temp, per split and as the four-split mean, with paired
intervals. An arm is back at sft_temp's level when the interval of its seed-mean
difference after scaling contains 0, below it when the interval lies under 0,
and above it otherwise.

Per arm and seed, the validation summary from training_log.jsonl on the fixed
valid subset: step 0 (the SFT adapter), the best step and the last step, each with
accuracy, ECE, NLL, KL to the SFT policy, the expected p of the chosen action (sum
of p squared) and the expected reward; plus train_summary.json's full-valid passes.

Writes runs/rlcd_stage<stage>_<size>/metrics.json (--out overrides) with the git
state, the sha256 of every results file read and of this script, and prints the
tables.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
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

Run = tuple[str, int]  # (arm, seed)


# Runs


def discover(runs_dir: Path, size: str, seeds: Sequence[int] | None) -> dict[Run, Path]:
    """(arm, seed) -> run directory for every rlcd_<size>_<arm>_s<seed> with results, arms in ARM_ORDER (unknown arms last by name), then seeds."""
    pattern = re.compile(rf"^rlcd_{re.escape(size)}_(.+)_s(\d+)$")
    found = {}
    for path in runs_dir.glob(f"rlcd_{size}_*_s*"):
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
    """`other` in the order of `reference`; both must hold exactly the same questions."""
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


def label(run: Run) -> str:
    return f"{run[0]}_s{run[1]}"


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


def paired_deltas(sft: Sequence[QuestionResult], temp: Sequence[QuestionResult], others: dict[Any, Sequence[QuestionResult]], resamples: int, seed_text: str) -> dict[Any, dict[str, np.ndarray]]:
    """Per named result list, bootstrap draws of accuracy minus sft's and ECE minus sft_temp's, the same record draws for every list."""
    arrays = lambda rs: (np.array([r.top1 for r in rs]), np.array([float(r.correct) for r in rs]))
    sft_correct = arrays(sft)[1]
    temp_conf, temp_correct = arrays(temp)
    other_arrays = {name: arrays(rs) for name, rs in others.items()}
    draws: dict[Any, dict[str, list[np.ndarray]]] = {name: {"accuracy": [], "ece": []} for name in others}
    for weights in record_weights(sft, resamples, seed_text):
        total = weights.sum(axis=1)
        temp_ece = weighted_ece(temp_conf, temp_correct, weights)
        sft_accuracy = weights @ sft_correct / total
        for name, (conf, correct) in other_arrays.items():
            draws[name]["accuracy"].append(weights @ correct / total - sft_accuracy)
            draws[name]["ece"].append(weighted_ece(conf, correct, weights) - temp_ece)
    return {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in draws.items()}


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


def level(ci: Sequence[float]) -> str:
    return "below" if ci[1] < 0 else "above" if ci[0] > 0 else "level"


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


def seed_aggregate(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Mean and range over seeds of accuracy and ECE."""
    out: dict[str, Any] = {"seeds": sorted(per_seed)}
    for k in ("accuracy", "ece", "nll"):
        values = [v[k] for v in per_seed.values()]
        out[k] = sum(values) / len(values)
        out[f"{k}_range"] = [min(values), max(values)]
    return out


# Main


def compare(runs_dir: Path, size: str, seeds: Sequence[int] | None, resamples: int) -> dict[str, Any]:
    sft_dir, temp_dir = runs_dir / f"sft_{size}", runs_dir / f"sft_{size}_temp"
    temperature = float(json.loads((temp_dir / "calibration.json").read_text())["temperature"])
    sft_all = gold_dependent(read_results(sft_dir / "results.jsonl.gz"))
    temp_all = scale_results(sft_all, temperature)
    found = discover(runs_dir, size, seeds)
    if not found:
        raise SystemExit(f"no rlcd_{size}_*_s<seed> run with results.jsonl.gz under {runs_dir}" + (f" for seeds {list(seeds)}" if seeds else ""))
    run_all = {run: aligned(sft_all, gold_dependent(read_results(path / "results.jsonl.gz")), path.name) for run, path in found.items()}
    arms: dict[str, list[int]] = {}
    for arm, seed in found:
        arms.setdefault(arm, []).append(seed)

    # Temperature ablation: arms whose every seed has a fitted temperature.
    scaled_dirs = {run: path.parent / f"{path.name}_temp" for run, path in found.items()}
    ablated = [arm for arm, arm_seeds in arms.items() if arm not in KNOWN_BROKEN and all((scaled_dirs[(arm, s)] / "calibration.json").is_file() for s in arm_seeds)]
    run_temperature = {(arm, s): float(json.loads((scaled_dirs[(arm, s)] / "calibration.json").read_text())["temperature"]) for arm in ablated for s in arms[arm]}
    scaled_all = {run: scale_results(run_all[run], t) for run, t in run_temperature.items()}

    splits: dict[str, Any] = {}
    unseen_ece: dict[str, list[float]] = {}  # name -> per unseen split ECE, in UNSEEN order
    unseen_draws: dict[Any, list[np.ndarray]] = {}  # (arm, seed) or (arm, seed, "T") -> per unseen split ECE-minus-sft_temp draws
    for split in dict.fromkeys(r.split for r in sft_all):
        in_split = [i for i, r in enumerate(sft_all) if r.split == split]
        blocks = {}
        for block in ("overall", *QUESTION_TYPES):
            idx = [i for i in in_split if block == "overall" or sft_all[i].qtype == block]
            if not idx:
                continue
            take = lambda rs: [rs[i] for i in idx]
            sft, temp = take(sft_all), take(temp_all)
            runs = {run: take(rs) for run, rs in run_all.items()}
            values: dict[str, Any] = {"sft": point(sft), "sft_temp": point(temp), "arms": {}}
            check_against_metrics(sft_dir, split, block, values["sft"])
            check_against_metrics(temp_dir, split, block, values["sft_temp"])
            per_run = {run: point(rs) for run, rs in runs.items()}
            for run, v in per_run.items():
                check_against_metrics(found[run], split, block, v)
            unseen_block = split in UNSEEN and block == "overall"
            others: dict[Any, Sequence[QuestionResult]] = dict(runs)
            if unseen_block:
                for run, rs in scaled_all.items():
                    scaled = take(rs)
                    others[(*run, "T")] = scaled
                    per_run[(*run, "T")] = point(scaled)
                    check_against_metrics(scaled_dirs[run], split, block, per_run[(*run, "T")])
            draws = paired_deltas(sft, temp, others, resamples, f"{split}:{block}")
            for arm, arm_seeds in arms.items():
                seed_values = {}
                for s in arm_seeds:
                    v, d = dict(per_run[(arm, s)]), draws[(arm, s)]
                    v["accuracy_minus_sft"] = v["accuracy"] - values["sft"]["accuracy"]
                    v["accuracy_minus_sft_ci"] = interval(d["accuracy"])
                    v["ece_minus_sft_temp"] = v["ece"] - values["sft_temp"]["ece"]
                    v["ece_minus_sft_temp_ci"] = interval(d["ece"])
                    seed_values[s] = v
                agg = seed_aggregate(seed_values)
                agg["accuracy_minus_sft"] = agg["accuracy"] - values["sft"]["accuracy"]
                agg["accuracy_minus_sft_ci"] = interval(np.mean([draws[(arm, s)]["accuracy"] for s in arm_seeds], axis=0))
                agg["ece_minus_sft_temp"] = agg["ece"] - values["sft_temp"]["ece"]
                agg["ece_minus_sft_temp_ci"] = interval(np.mean([draws[(arm, s)]["ece"] for s in arm_seeds], axis=0))
                values["arms"][arm] = {"mean": agg, "by_seed": {str(s): v for s, v in seed_values.items()}}
            if unseen_block:
                for name, v in (("sft", values["sft"]), ("sft_temp", values["sft_temp"])):
                    unseen_ece.setdefault(name, []).append(v["ece"])
                for name, v in per_run.items():
                    unseen_ece.setdefault(name, []).append(v["ece"])
                    unseen_draws.setdefault(name, []).append(draws[name]["ece"])
            blocks[block] = values
        splits[split] = blocks

    unseen: dict[str, Any] = {"splits": [s for s in UNSEEN if s in splits]}
    temperature_ablation: dict[str, Any] = {"arms": {}}
    if len(unseen["splits"]) == len(UNSEEN):
        mean_of = lambda name: sum(unseen_ece[name]) / len(UNSEEN)
        mean_draws = lambda name: np.mean(np.stack(unseen_draws[name]), axis=0)
        unseen["mean_ece"] = {"sft": mean_of("sft"), "sft_temp": mean_of("sft_temp")}
        unseen["arms"] = {}
        for arm, arm_seeds in arms.items():
            by_seed = {str(s): mean_of((arm, s)) for s in arm_seeds}
            seed_mean_draws = np.mean([mean_draws((arm, s)) for s in arm_seeds], axis=0)
            mean = sum(by_seed.values()) / len(by_seed)
            unseen["arms"][arm] = {
                "mean_ece": mean,
                "mean_ece_range": [min(by_seed.values()), max(by_seed.values())],
                "mean_ece_by_seed": by_seed,
                "mean_ece_minus_sft_temp": mean - unseen["mean_ece"]["sft_temp"],
                "mean_ece_minus_sft_temp_ci": interval(seed_mean_draws),
            }
        for arm in ablated:
            arm_seeds = arms[arm]
            per_split = {}
            for i, split in enumerate(UNSEEN):
                before = [unseen_ece[(arm, s)][i] for s in arm_seeds]
                after = [unseen_ece[(arm, s, "T")][i] for s in arm_seeds]
                per_split[split] = {
                    "ece_before": sum(before) / len(before),
                    "ece_after": sum(after) / len(after),
                    "ece_after_minus_sft_temp": sum(after) / len(after) - unseen_ece["sft_temp"][i],
                    "ece_after_minus_sft_temp_ci": interval(np.mean([unseen_draws[(arm, s, "T")][i] for s in arm_seeds], axis=0)),
                }
            before = sum(mean_of((arm, s)) for s in arm_seeds) / len(arm_seeds)
            after = sum(mean_of((arm, s, "T")) for s in arm_seeds) / len(arm_seeds)
            after_ci = interval(np.mean([mean_draws((arm, s, "T")) for s in arm_seeds], axis=0))
            temperature_ablation["arms"][arm] = {
                "temperature_by_seed": {str(s): run_temperature[(arm, s)] for s in arm_seeds},
                "mean_ece_before": before,
                "mean_ece_after": after,
                "mean_ece_after_minus_sft_temp": after - unseen["mean_ece"]["sft_temp"],
                "mean_ece_after_minus_sft_temp_ci": after_ci,
                "level_vs_sft_temp": level(after_ci),
                "splits": per_split,
            }
        levels = {arm: v["level_vs_sft_temp"] for arm, v in temperature_ablation["arms"].items()}
        temperature_ablation["sft_temp_temperature"] = temperature
        temperature_ablation["sft_temp_mean_ece"] = unseen["mean_ece"]["sft_temp"]
        temperature_ablation["every_arm_back_to_sft_temp_level"] = bool(levels) and all(v in ("level", "below") for v in levels.values())
        temperature_ablation["note"] = "Each arm scaled by its own temperature fitted on valid (scripts/calibrate.py, runs/<run>_temp/). level: the interval of the seed-mean difference to sft_temp contains 0; below: it lies under 0; above: over 0."

    return {
        "size": size,
        "seeds": sorted({s for _, s in found}),
        "runs": {
            "sft": shown(sft_dir),
            "sft_temp": {"run": shown(temp_dir), "temperature": temperature, "note": "the SFT probabilities scaled by this temperature (decision 50)"},
            **{label(run): shown(path) for run, path in found.items()},
        },
        "arms": {arm: arm_seeds for arm, arm_seeds in arms.items()},
        "known_broken_arms": [arm for arm in arms if arm in KNOWN_BROKEN],
        "results_sha256": {"sft": sha256(sft_dir / "results.jsonl.gz"), **{label(run): sha256(path / "results.jsonl.gz") for run, path in found.items()}},
        "bootstrap": {
            "resamples": resamples,
            "unit": "record",
            "interval": "95 percent percentile",
            "seed": "zlib.crc32 of '<split>:<block>', the same draws for every run",
            "seed_mean": "each draw averages the seeds' differences under that draw; the interval covers record sampling for these seeds, and the range shows seed-to-seed variation",
        },
        "note": "Gold-dependent questions only. accuracy_minus_sft is against sft; ece_minus_sft_temp against sft_temp (one global temperature fitted on valid). Point estimates equal each run's metrics.json.",
        "splits": splits,
        "unseen_schemas": unseen,
        "temperature_ablation": temperature_ablation,
        "validation": {arm: {str(s): validation_summary(found[(arm, s)]) for s in arm_seeds} for arm, arm_seeds in arms.items()},
    }


def print_tables(result: dict[str, Any]) -> None:
    arms = list(result["arms"])
    multi = any(len(s) > 1 for s in result["arms"].values())
    mark = lambda v: ("+" if v["ece_minus_sft_temp_ci"][1] < 0 else "x" if v["ece_minus_sft_temp_ci"][0] > 0 else " ") + ("v" if v["accuracy_minus_sft_ci"][1] < 0 else "^" if v["accuracy_minus_sft_ci"][0] > 0 else " ")
    print(f"size {result['size']}, seeds {result['seeds']}" + (" (arm cells are seed means)" if multi else ""))
    print("accuracy / ECE per run; marks per arm: + ECE below sft_temp, x above it, v accuracy below sft, ^ above it (95 percent interval excludes 0)")
    print("split | block | n | sft | sft_temp | " + " | ".join(arms))
    for split, blocks in result["splits"].items():
        for block, v in blocks.items():
            cells = [f"{v['sft']['accuracy']:.3f}/{v['sft']['ece']:.3f}", f"{v['sft_temp']['accuracy']:.3f}/{v['sft_temp']['ece']:.3f}"]
            for a in arms:
                m = v["arms"][a]["mean"]
                cell = f"{m['accuracy']:.3f}/{m['ece']:.3f} {mark(m)}"
                if len(m["seeds"]) > 1:
                    cell += f" (ECE {m['ece_range'][0]:.3f}-{m['ece_range'][1]:.3f})"
                cells.append(cell)
            print(f"{split} | {block} | {v['sft']['n']} | " + " | ".join(cells))
    unseen = result["unseen_schemas"]
    if "mean_ece" in unseen:
        print("\nmean ECE over the four unseen schemas; difference to sft_temp [95 percent interval]")
        print(f"sft {unseen['mean_ece']['sft']:.4f}, sft_temp {unseen['mean_ece']['sft_temp']:.4f}")
        for arm in arms:
            u = unseen["arms"][arm]
            lo, hi = u["mean_ece_minus_sft_temp_ci"]
            spread = f" (seeds {u['mean_ece_range'][0]:.4f}-{u['mean_ece_range'][1]:.4f})" if len(u["mean_ece_by_seed"]) > 1 else ""
            print(f"{arm}: {u['mean_ece']:.4f}{spread}, {u['mean_ece_minus_sft_temp']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    ablation = result["temperature_ablation"]
    if ablation.get("arms"):
        print(f"\ntemperature ablation: each arm scaled by its own T fitted on valid; unseen-schema ECE before -> after, against sft_temp (T {ablation['sft_temp_temperature']:.3f}, mean ECE {ablation['sft_temp_mean_ece']:.4f})")
        print("arm | T | " + " | ".join(f"{s} before -> after [after - sft_temp, CI]" for s in UNSEEN) + " | mean before -> after | after - sft_temp [CI] | level")
        for arm, a in ablation["arms"].items():
            ts = ", ".join(f"{t:.3f}" for t in a["temperature_by_seed"].values())
            cells = [f"{p['ece_before']:.3f} -> {p['ece_after']:.3f} [{p['ece_after_minus_sft_temp']:+.3f}, {p['ece_after_minus_sft_temp_ci'][0]:+.3f} to {p['ece_after_minus_sft_temp_ci'][1]:+.3f}]" for p in a["splits"].values()]
            lo, hi = a["mean_ece_after_minus_sft_temp_ci"]
            print(f"{arm} | {ts} | " + " | ".join(cells) + f" | {a['mean_ece_before']:.4f} -> {a['mean_ece_after']:.4f} | {a['mean_ece_after_minus_sft_temp']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {a['level_vs_sft_temp']}")
        print(f"every arm back to sft_temp's level: {ablation['every_arm_back_to_sft_temp_level']}")
    print("\nvalidation subset, step 0 -> last step (best step): KL to SFT, sum of p squared, expected reward, NLL")
    for arm in arms:
        for seed, v in result["validation"][arm].items():
            s0, last = v["step_0"], v["last"]
            print(f"{arm} s{seed}: KL {last['kl_to_ref']:.4f}, sum p^2 {s0['expected_p_chosen']:.4f} -> {last['expected_p_chosen']:.4f}, expected reward {s0['expected_reward']:.4f} -> {last['expected_reward']:.4f}, "
                  f"NLL {s0['nll']:.4f} -> {last['nll']:.4f} (best step {v['best_step']}, NLL {v['best']['nll']:.4f})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="06b", help="06b or 17b: selects sft_<size>, sft_<size>_temp and rlcd_<size>_*")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: every seed found")
    parser.add_argument("--stage", type=int, default=1, help="names the output directory, runs/rlcd_stage<stage>_<size>")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    git = git_state()
    runs_dir = Path(args.runs_dir)
    result = compare(runs_dir, args.size, args.seeds, args.resamples)
    out_dir = Path(args.out) if args.out else runs_dir / f"rlcd_stage{args.stage}_{args.size}"
    result = {
        "run_name": out_dir.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "code_sha256": {"compare_rlcd.py": sha256(Path(__file__))},
        **result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print_tables(result)
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
