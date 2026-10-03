"""The v3 comparison: learners from the deployment-feedback log on Banking77 (task 3.4, decision 56, docs/V3_DESIGN.md).

    uv run python scripts/compare_v3.py [--size 06b] [--runs-dir runs] [--resamples 1000]

No model is loaded. It reads, under --runs-dir:

- v3_<size>_zeroshot: the logging policy (sft_<size>) evaluated on the v3 splits,
  the zero-shot learner and the reference of the forgetting check;
- v3_<size>_temp_n<N>: the temperature learner (scripts/calibrate.py --fit-log);
  its probabilities are the zero-shot results scaled by its calibration.json;
- every v3_<size>_<arm>_n<N>_s<seed>[_noisy][_log<k>] with a results.jsonl.gz, for
  the arms positive_sft, full_sft and direct_brier (scripts/train_rlcd.py log mode,
  then scripts/evaluate.py);
- b2_gpt-4.1-mini/metrics.json for the B2 reference on the 500-record baseline
  subset of test_banking77 (data/baseline_subset.json).

On v3_banking77_test_full it reports accuracy, ECE, Brier and NLL with 95 percent
bootstrap intervals by record (resamples of the split's records, the same draws for
every run, so differences are paired), and writes runs/v3_stage_<size>/metrics.json:

- n_curve: rows N 500, 2000, 5000; columns zero_shot, temperature, positive_sft,
  full_sft and direct_brier (seed 0, the seed 0 log, clean outcomes), and the paired
  differences direct_brier minus positive_sft and direct_brier minus full_sft;
- seeds: direct_brier at N 5000 for training seeds 0, 1 and 2 (mean and range) and
  on the seed 1 log (logging variance), with each run's difference from seed 0, and
  the seed mean minus positive_sft and minus full_sft at N 5000 (each draw averages
  the seeds' differences under the same record resample, decision 53); full_sft at
  N 5000 for training seeds 0, 1 and 2 (mean and range, each seed minus seed 0) and
  direct_brier's seed mean minus full_sft's seed mean, paired the same way;
- noisy: direct_brier and positive_sft at N 5000 with flipped outcomes, each one's
  change from its clean run, and the difference of those changes (prediction 4);
- forgetting: every learner's accuracy and ECE on test_indomain and
  test_unseen_intents minus the zero-shot run's, paired;
- other: each run's predicted-other rate (other is never gold in these questions)
  and its accuracy on the questions where no run predicted other;
- coverage: share kept and accuracy at confidence thresholds 0.8, 0.9 and 0.95;
- b2_reference: gpt-4.1-mini on the 500-record subset of test_banking77 next to
  every run on the same records.

Every run must hold exactly the same questions as the zero-shot run on each split,
and the data hashes of the splits must agree across runs; otherwise it stops.
Adapted learners' Banking77 numbers are never mixed into the v1 or v2
unseen-schema means (decision 56).
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
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_ids
from jevmark.calibration import scale_results
from jevmark.data.build import V3_TEST_FULL
from jevmark.metrics import N_BINS, NLL_FLOOR, QuestionResult, gold_dependent, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
NS = (500, 2000, 5000)
ARMS = ("positive_sft", "full_sft", "direct_brier")
LEARNERS = ("zero_shot", "temperature", *ARMS)
FORGETTING_SPLITS = ("test_indomain", "test_unseen_intents")
COVERAGE_THRESHOLDS = (0.8, 0.9, 0.95)
METRICS = ("accuracy", "ece", "brier", "nll")
OTHER = "other"
B2_RUN = "b2_gpt-4.1-mini"
RUN_PATTERN = re.compile(r"^v3_(?P<size>06b|17b)_(?P<arm>positive_sft|full_sft|direct_brier)_n(?P<n>\d+)_s(?P<seed>\d+)(?P<noisy>_noisy)?(?:_log(?P<log>\d+))?$")


# Runs


def discover(runs_dir: Path, size: str) -> dict[tuple[str, int, int, bool, int], Path]:
    """(arm, N, seed, noisy, log seed) -> run directory, for every learner run with results."""
    found = {}
    for path in sorted(runs_dir.glob(f"v3_{size}_*")):
        m = RUN_PATTERN.match(path.name)
        if m and m["size"] == size and (path / "results.jsonl.gz").is_file():
            found[(m["arm"], int(m["n"]), int(m["seed"]), bool(m["noisy"]), int(m["log"] or 0))] = path
    return found


def by_split(results: Sequence[QuestionResult]) -> dict[str, list[QuestionResult]]:
    out: dict[str, list[QuestionResult]] = {}
    for r in gold_dependent(results):
        out.setdefault(r.split, []).append(r)
    return out


def aligned(reference: Sequence[QuestionResult], other: Sequence[QuestionResult], name: str) -> list[QuestionResult]:
    """`other` in the order of `reference`; both must hold exactly the same questions."""
    index = {(r.record_id, r.question_id): r for r in other}
    if len(index) != len(reference) or any((r.record_id, r.question_id) not in index for r in reference):
        raise SystemExit(f"{name}: its questions differ from the zero-shot run's on {reference[0].split}; the runs are not comparable")
    return [index[(r.record_id, r.question_id)] for r in reference]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Metrics with paired bootstrap draws


def arrays(results: Sequence[QuestionResult]) -> dict[str, np.ndarray]:
    return {
        "correct": np.array([float(r.correct) for r in results]),
        "top1": np.array([r.top1 for r in results]),
        "brier": np.array([sum((p - (k == r.gold)) ** 2 for k, p in enumerate(r.probs)) for r in results]),
        "nll": np.array([-math.log(max(r.probs[r.gold], NLL_FLOOR)) for r in results]),
    }


def weighted_ece(conf: np.ndarray, correct: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ECE (15 equal-width bins on the top-1 probability, as jevmark.metrics.ece) for each row of question weights."""
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    total = np.zeros(weights.shape[0])
    for b in range(N_BINS):
        members = bins == b
        if members.any():
            total += np.abs(weights[:, members] @ (conf[members] - correct[members]))
    return total / weights.sum(axis=1)


def values(a: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    total = weights.sum(axis=1)
    return {
        "accuracy": weights @ a["correct"] / total,
        "ece": weighted_ece(a["top1"], a["correct"], weights),
        "brier": weights @ a["brier"] / total,
        "nll": weights @ a["nll"] / total,
    }


def record_weights(results: Sequence[QuestionResult], resamples: int, seed_text: str, chunk: int = 100) -> Iterator[np.ndarray]:
    """Blocks of question weights: each row counts how often the question's record was drawn in one resample of the records."""
    records: dict[str, int] = {}
    record_of = np.array([records.setdefault(r.record_id, len(records)) for r in results])
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    for start in range(0, resamples, chunk):
        draws = rng.integers(0, len(records), size=(min(chunk, resamples - start), len(records)))
        counts = np.stack([np.bincount(d, minlength=len(records)) for d in draws]).astype(np.float64)
        yield counts[:, record_of]


def interval(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


class Split:
    """Every run's results on one split, aligned to the zero-shot run, with the same bootstrap draws for every run."""

    def __init__(self, split: str, runs: Mapping[str, Sequence[QuestionResult]], resamples: int):
        self.split = split
        self.results = dict(runs)
        self.arrays = {name: arrays(rs) for name, rs in runs.items()}
        reference = next(iter(runs.values()))
        draws: dict[str, dict[str, list[np.ndarray]]] = {name: {m: [] for m in METRICS} for name in runs}
        for weights in record_weights(reference, resamples, f"v3:{split}"):
            for name, a in self.arrays.items():
                for metric, value in values(a, weights).items():
                    draws[name][metric].append(value)
        self.draws = {name: {m: np.concatenate(v) for m, v in d.items()} for name, d in draws.items()}

    def point(self, name: str) -> dict[str, Any]:
        a = self.arrays[name]
        out: dict[str, Any] = {"n": len(a["correct"])}
        for metric, value in values(a, np.ones((1, len(a["correct"])))).items():
            out[metric] = float(value[0])
            out[f"{metric}_ci"] = interval(self.draws[name][metric])
        return out

    def delta(self, first: str, second: str) -> dict[str, Any]:
        """first minus second, point and paired interval, for every metric."""
        p, q = self.point(first), self.point(second)
        return {metric: {"delta": p[metric] - q[metric], "ci": interval(self.draws[first][metric] - self.draws[second][metric])} for metric in METRICS}

    def mean_delta(self, names: Sequence[str], second: str) -> dict[str, Any]:
        """The mean of names minus second, paired: each draw averages the runs' differences under the same record resample (decision 53)."""
        out = {}
        for metric in METRICS:
            draws = np.mean([self.draws[name][metric] for name in names], axis=0) - self.draws[second][metric]
            point = sum(self.point(name)[metric] for name in names) / len(names) - self.point(second)[metric]
            out[metric] = {"delta": point, "ci": interval(draws)}
        return out

    def mean_minus_mean(self, first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
        """The mean of first minus the mean of second, paired: each draw averages each group under the same record resample."""
        out = {}
        for metric in METRICS:
            draws = np.mean([self.draws[name][metric] for name in first], axis=0) - np.mean([self.draws[name][metric] for name in second], axis=0)
            point = sum(self.point(name)[metric] for name in first) / len(first) - sum(self.point(name)[metric] for name in second) / len(second)
            out[metric] = {"delta": point, "ci": interval(draws)}
        return out

    def delta_of_deltas(self, a: str, a_ref: str, b: str, b_ref: str) -> dict[str, Any]:
        """(a - a_ref) - (b - b_ref), paired."""
        out = {}
        for metric in METRICS:
            draws = (self.draws[a][metric] - self.draws[a_ref][metric]) - (self.draws[b][metric] - self.draws[b_ref][metric])
            point = (self.point(a)[metric] - self.point(a_ref)[metric]) - (self.point(b)[metric] - self.point(b_ref)[metric])
            out[metric] = {"delta": point, "ci": interval(draws)}
        return out


def predicted_other_rate(results: Sequence[QuestionResult]) -> float:
    return sum(r.labels[r.prediction] == OTHER for r in results) / len(results)


def coverage(results: Sequence[QuestionResult]) -> dict[str, dict[str, Any]]:
    out = {}
    for t in COVERAGE_THRESHOLDS:
        kept = [r for r in results if r.confidence >= t - 1e-12]
        out[f"{t:.2f}"] = {"coverage": len(kept) / len(results), "n": len(kept), "accuracy": sum(r.correct for r in kept) / len(kept) if kept else None}
    return out


# The comparison


def learner_name(arm: str, n: int, seed: int = 0, noisy: bool = False, log: int = 0) -> str:
    return f"{arm}_n{n}_s{seed}" + ("_noisy" if noisy else "") + (f"_log{log}" if log else "")


def compare(runs_dir: Path, size: str, resamples: int, subset_path: Path = SUBSET_PATH) -> dict[str, Any]:
    zero_dir = runs_dir / f"v3_{size}_zeroshot"
    if not (zero_dir / "results.jsonl.gz").is_file():
        raise SystemExit(f"{zero_dir}: no results.jsonl.gz; evaluate sft_{size} on the v3 splits first (evaluate.py --ckpt runs/sft_{size} --run-name v3_{size}_zeroshot)")
    zero_all = read_results(zero_dir / "results.jsonl.gz")
    zero_metrics = json.loads((zero_dir / "metrics.json").read_text())
    zero = by_split(zero_all)
    for needed in (V3_TEST_FULL, *FORGETTING_SPLITS):
        if needed not in zero:
            raise SystemExit(f"{zero_dir} has no {needed} results")

    sources: dict[str, dict[str, Any]] = {"zero_shot": {"dir": str(zero_dir), "git": zero_metrics.get("git"), "results_sha256": sha256(zero_dir / "results.jsonl.gz"), "adapter_sha256": zero_metrics.get("adapter_sha256")}}
    per_run: dict[str, dict[str, list[QuestionResult]]] = {"zero_shot": zero}
    data_hashes = {k: v for k, v in zero_metrics.get("data_files_sha256", {}).items()}

    temperatures = {}
    for n in NS:
        temp_dir = runs_dir / f"v3_{size}_temp_n{n}"
        if (temp_dir / "calibration.json").is_file():
            t = float(json.loads((temp_dir / "calibration.json").read_text())["temperature"])
            temperatures[n] = t
            per_run[f"temperature_n{n}"] = by_split(scale_results(zero_all, t))
            sources[f"temperature_n{n}"] = {"dir": str(temp_dir), "temperature": t}

    runs = discover(runs_dir, size)
    for (arm, n, seed, noisy, log), path in runs.items():
        name = learner_name(arm, n, seed, noisy, log)
        metrics = json.loads((path / "metrics.json").read_text())
        for file, digest in metrics.get("data_files_sha256", {}).items():
            if file in data_hashes and data_hashes[file] != digest:
                raise SystemExit(f"{path}: {file} has sha256 {digest}, the zero-shot run read {data_hashes[file]}")
        per_run[name] = by_split(read_results(path / "results.jsonl.gz"))
        sources[name] = {"dir": str(path), "git": metrics.get("git"), "results_sha256": sha256(path / "results.jsonl.gz"), "v3": metrics.get("v3")}

    splits = {}
    for split in (V3_TEST_FULL, *FORGETTING_SPLITS):
        reference = zero[split]
        members = {name: aligned(reference, results[split], name) if name != "zero_shot" else reference for name, results in per_run.items() if split in results}
        missing = [name for name, results in per_run.items() if split not in results]
        if missing:
            raise SystemExit(f"{', '.join(missing)}: no {split} results")
        splits[split] = Split(split, members, resamples)
    main = splits[V3_TEST_FULL]
    present = set(main.results)

    def column(n: int, learner: str) -> str | None:
        name = {"zero_shot": "zero_shot", "temperature": f"temperature_n{n}"}.get(learner, learner_name(learner, n))
        return name if name in present else None

    n_curve = {}
    for n in NS:
        row: dict[str, Any] = {learner: (main.point(name) if (name := column(n, learner)) else None) for learner in LEARNERS}
        brier, pos, full = column(n, "direct_brier"), column(n, "positive_sft"), column(n, "full_sft")
        row["direct_brier_minus_positive_sft"] = main.delta(brier, pos) if brier and pos else None
        row["direct_brier_minus_full_sft"] = main.delta(brier, full) if brier and full else None
        n_curve[str(n)] = row

    seed_runs = {f"seed_{s}": learner_name("direct_brier", 5000, s) for s in (0, 1, 2)}
    seed_runs["log_seed_1"] = learner_name("direct_brier", 5000, 0, log=1)
    seeds: dict[str, Any] = {key: (main.point(name) if name in present else None) for key, name in seed_runs.items()}
    trained = [seeds[f"seed_{s}"] for s in (0, 1, 2) if seeds[f"seed_{s}"]]
    if trained:
        seeds["training_seeds"] = {m: {"mean": sum(p[m] for p in trained) / len(trained), "range": [min(p[m] for p in trained), max(p[m] for p in trained)], "n_seeds": len(trained)} for m in METRICS}
    trained_names = [seed_runs[f"seed_{s}"] for s in (0, 1, 2) if seed_runs[f"seed_{s}"] in present]
    for other_arm in ("positive_sft", "full_sft"):
        other_name = learner_name(other_arm, 5000)
        seeds[f"training_seed_mean_minus_{other_arm}"] = main.mean_delta(trained_names, other_name) if trained_names and other_name in present else None
    base = seed_runs["seed_0"]
    seeds["minus_seed_0"] = {key: main.delta(name, base) for key, name in seed_runs.items() if key != "seed_0" and name in present and base in present}
    full_names = {f"seed_{s}": learner_name("full_sft", 5000, s) for s in (0, 1, 2)}
    full_trained = [name for name in full_names.values() if name in present]
    full: dict[str, Any] = {key: (main.point(name) if name in present else None) for key, name in full_names.items()}
    if full_trained:
        points = [main.point(name) for name in full_trained]
        full["training_seeds"] = {m: {"mean": sum(p[m] for p in points) / len(points), "range": [min(p[m] for p in points), max(p[m] for p in points)], "n_seeds": len(points)} for m in METRICS}
    full["minus_seed_0"] = {key: main.delta(name, full_names["seed_0"]) for key, name in full_names.items() if key != "seed_0" and name in present and full_names["seed_0"] in present}
    seeds["full_sft"] = full
    seeds["training_seed_mean_minus_full_sft_seed_mean"] = main.mean_minus_mean(trained_names, full_trained) if trained_names and full_trained else None

    noisy: dict[str, Any] = {}
    for arm in ("direct_brier", "positive_sft"):
        clean, flipped = learner_name(arm, 5000), learner_name(arm, 5000, noisy=True)
        noisy[arm] = {"noisy": main.point(flipped) if flipped in present else None, "clean": main.point(clean) if clean in present else None,
                      "noisy_minus_clean": main.delta(flipped, clean) if flipped in present and clean in present else None}
    names = [learner_name(a, 5000, noisy=f) for a in ("direct_brier", "positive_sft") for f in (True, False)]
    noisy["direct_brier_change_minus_positive_sft_change"] = main.delta_of_deltas(*names) if all(n in present for n in names) else None

    forgetting = {}
    for split in FORGETTING_SPLITS:
        s = splits[split]
        forgetting[split] = {"zero_shot": s.point("zero_shot"), "minus_zero_shot": {name: {m: s.delta(name, "zero_shot")[m] for m in ("accuracy", "ece")} for name in s.results if name != "zero_shot"}}

    never_other = [i for i in range(len(main.results["zero_shot"])) if not any(main.results[name][i].labels[main.results[name][i].prediction] == OTHER for name in main.results)]
    other = {name: {"predicted_other_rate": predicted_other_rate(rs), "accuracy_where_no_run_predicted_other": sum(rs[i].correct for i in never_other) / len(never_other) if never_other else None} for name, rs in main.results.items()}
    other["n_questions_where_no_run_predicted_other"] = len(never_other)

    b2_reference: dict[str, Any] = {}
    b2_path = runs_dir / B2_RUN / "metrics.json"
    if b2_path.is_file():
        b2 = json.loads(b2_path.read_text())["splits"]["test_banking77"]["overall"]
        ids = set(subset_ids(load_subset(subset_path), "test_banking77"))
        b2_reference = {"subset": "data/baseline_subset.json, test_banking77, 500 records", "b2": {"accuracy_all": b2["accuracy_all"], "ece": b2["ece"], "n": b2["n"]}, "runs": {}}
        for name, results in per_run.items():
            on_subset = [r for r in results.get("test_banking77", []) if r.record_id in ids]
            if on_subset:
                a = arrays(on_subset)
                b2_reference["runs"][name] = {"n": len(on_subset), **{m: float(v[0]) for m, v in values(a, np.ones((1, len(on_subset)))).items()}}

    return {
        "decision": 56,
        "size": size,
        "split": V3_TEST_FULL,
        "git": git_state(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": {"resamples": resamples, "by": "record", "seed_text": "v3:<split>", "paired": "the same record draws for every run on a split"},
        "note": "Adapted learners' Banking77 numbers are never mixed into the v1 or v2 unseen-schema means (decision 56).",
        "data_files_sha256": data_hashes,
        "temperatures": {str(n): t for n, t in temperatures.items()},
        "sources": sources,
        "n_curve": n_curve,
        "seeds": seeds,
        "noisy": noisy,
        "forgetting": forgetting,
        "other": other,
        "coverage": {name: coverage(rs) for name, rs in main.results.items()},
        "b2_reference": b2_reference,
        "missing": sorted({learner_name(a, n) for a in ARMS for n in NS} - present),
    }


def fmt(block: Mapping[str, Any] | None, metric: str) -> str:
    if not block:
        return "-"
    lo, hi = block[f"{metric}_ci"]
    return f"{block[metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def print_tables(result: Mapping[str, Any]) -> None:
    for metric in METRICS:
        print(f"\n== {metric} on {result['split']} by N")
        print("N      " + "  ".join(f"{name:>24}" for name in LEARNERS))
        for n, row in result["n_curve"].items():
            print(f"{n:6} " + "  ".join(f"{fmt(row[name], metric):>24}" for name in LEARNERS))
    print("\n== direct_brier minus positive_sft / minus full_sft (accuracy, paired)")
    for n, row in result["n_curve"].items():
        cells = []
        for key in ("direct_brier_minus_positive_sft", "direct_brier_minus_full_sft"):
            d = row[key]
            cells.append("-" if d is None else f"{d['accuracy']['delta']:+.3f} [{d['accuracy']['ci'][0]:+.3f}, {d['accuracy']['ci'][1]:+.3f}]")
        print(f"{n:6} " + "   ".join(cells))
    if result["missing"]:
        print(f"\nmissing runs: {', '.join(result['missing'])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="06b")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", default=None, help="default: runs/v3_stage_<size>/metrics.json")
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    result = compare(runs_dir, args.size, args.resamples, Path(args.subset))
    out = Path(args.out) if args.out else runs_dir / f"v3_stage_{args.size}" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print_tables(result)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
