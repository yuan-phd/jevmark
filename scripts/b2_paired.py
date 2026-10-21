"""Paired accuracy differences against gpt-4.1-mini (B2) on the baseline subset, CPU only (review item 6).

    uv run python scripts/b2_paired.py

B2's per-question answers are parsed again from runs/b2_gpt-4.1-mini/replies.jsonl with
the shared parser (jevmark.baselines.results.answers_from_replies), on the records of
data/baseline_subset.json; a reply that does not parse counts as wrong, as in B2's
accuracy_all, the like-for-like number. Its accuracy is checked against B2's
metrics.json. For each log domain, the choice question of the subset split
(test_banking77, 500 records; test_emotion, 500 records, choice question `label`
only, the one a v3 learner answers), and every learner below on the same questions:

- banking77: CAPS-Brier (direct_brier) at N 5000 on clean outcomes, training seeds 0,
  1 and 2 on the seed 0 log and seed 0 on the seed 1 log; full_sft at N 5000, seeds 0,
  1 and 2; the zero-shot logging policy for reference;
- emotion: CAPS-Brier at N 5000, seeds 0, 1 and 2; full_sft at N 5000, seed 0; the
  zero-shot run.

Per learner: its accuracy, B2's, the difference (learner minus B2) with a 95 percent
paired bootstrap interval by record (1000 multinomial resamples, seed 0, the
resampling of paired_deltas.py), and the share of questions where the two predictions
differ; per arm with several training seeds on the seed 0 log, the seed mean minus
B2, each draw averaging the seeds' differences under the same record resample.

Writes runs/b2_paired_v3/metrics.json and prints the table.
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

from jevmark.baselines.results import answers_from_replies, read_replies
from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_records
from jevmark.metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
B2_RUN = "b2_gpt-4.1-mini"
CHOICE_QUESTION = "label"
DOMAINS = {
    "banking77": {
        "split": "test_banking77",
        "runs": {
            "zero_shot": "v3_06b_zeroshot",
            "caps_brier_n5000_s0": "v3_06b_direct_brier_n5000_s0",
            "caps_brier_n5000_s1": "v3_06b_direct_brier_n5000_s1",
            "caps_brier_n5000_s2": "v3_06b_direct_brier_n5000_s2",
            "caps_brier_n5000_s0_log1": "v3_06b_direct_brier_n5000_s0_log1",
            "full_sft_n5000_s0": "v3_06b_full_sft_n5000_s0",
            "full_sft_n5000_s1": "v3_06b_full_sft_n5000_s1",
            "full_sft_n5000_s2": "v3_06b_full_sft_n5000_s2",
        },
        "seed_means": {"caps_brier_n5000": ("caps_brier_n5000_s0", "caps_brier_n5000_s1", "caps_brier_n5000_s2"), "full_sft_n5000": ("full_sft_n5000_s0", "full_sft_n5000_s1", "full_sft_n5000_s2")},
    },
    "emotion": {
        "split": "test_emotion",
        "runs": {
            "zero_shot": "v3_emotion_06b_zeroshot",
            "caps_brier_n5000_s0": "v3_emotion_06b_direct_brier_n5000_s0",
            "caps_brier_n5000_s1": "v3_emotion_06b_direct_brier_n5000_s1",
            "caps_brier_n5000_s2": "v3_emotion_06b_direct_brier_n5000_s2",
            "full_sft_n5000_s0": "v3_emotion_06b_full_sft_n5000_s0",
        },
        "seed_means": {"caps_brier_n5000": ("caps_brier_n5000_s0", "caps_brier_n5000_s1", "caps_brier_n5000_s2")},
    },
}
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(draws: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def domain_block(runs_dir: Path, data_dir: Path, subset: dict[str, Any], spec: dict[str, Any], b2_metrics: dict[str, Any], replies: dict) -> dict[str, Any]:
    split = spec["split"]
    records, data_sha = subset_records(data_dir, subset, split)
    b2 = {a.record_id: a for a in answers_from_replies({split: records}, replies) if a.question_id == CHOICE_QUESTION and a.qtype == "choice"}
    if len(b2) != len(records):
        raise SystemExit(f"{split}: {len(b2)} B2 choice answers for {len(records)} subset records")
    order = [r["id"] for r in records]
    b2_correct = np.array([float(b2[i].correct) for i in order])
    stored = b2_metrics["splits"][split]["choice"]
    if not math.isclose(stored["accuracy_all"], float(b2_correct.mean()), abs_tol=TOL):
        raise SystemExit(f"{split}: B2 accuracy_all recomputed {b2_correct.mean()}, metrics.json has {stored['accuracy_all']}")

    # One question per record, so resampling records is resampling questions here; the draws are paired_deltas.paired_block's.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(order), np.full(len(order), 1.0 / len(order)), size=BOOTSTRAP_RESAMPLES).astype(float)
    b2_draws = weights @ b2_correct / weights.sum(axis=1)

    learners: dict[str, Any] = {}
    draws: dict[str, np.ndarray] = {}
    sources: dict[str, Any] = {}
    for name, run in spec["runs"].items():
        path = runs_dir / run
        results = {r.record_id: r for r in read_results(path / "results.jsonl.gz") if r.split == split and r.question_id == CHOICE_QUESTION}
        missing = [i for i in order if i not in results]
        if missing:
            raise SystemExit(f"{run}: no {split} {CHOICE_QUESTION} result for {len(missing)} subset records")
        for i in order:
            if results[i].labels != b2[i].labels or results[i].gold != b2[i].gold:
                raise SystemExit(f"{run}: record {i} has other options or gold than the B2 request")
        correct = np.array([float(results[i].correct) for i in order])
        draws[name] = weights @ correct / weights.sum(axis=1)
        learners[name] = {
            "run": f"runs/{run}",
            "n": len(order),
            "accuracy": float(correct.mean()),
            "b2_accuracy": float(b2_correct.mean()),
            "minus_b2": float(correct.mean() - b2_correct.mean()),
            "minus_b2_ci": interval(draws[name] - b2_draws),
            "prediction_disagreement": float(np.mean([results[i].prediction != b2[i].prediction for i in order])),
        }
        sources[name] = {"run": f"runs/{run}", "results_sha256": sha256(path / "results.jsonl.gz")}
    seed_means = {}
    for name, members in spec["seed_means"].items():
        accuracy = sum(learners[m]["accuracy"] for m in members) / len(members)
        seed_means[name] = {
            "members": list(members),
            "accuracy": accuracy,
            "accuracy_range": [min(learners[m]["accuracy"] for m in members), max(learners[m]["accuracy"] for m in members)],
            "minus_b2": accuracy - float(b2_correct.mean()),
            "minus_b2_ci": interval(np.mean([draws[m] for m in members], axis=0) - b2_draws),
        }
    return {
        "split": split,
        "question": CHOICE_QUESTION,
        "n_records": len(order),
        "data_sha256": data_sha,
        "b2": {"accuracy_all": float(b2_correct.mean()), "n_parsed": sum(b2[i].parsed for i in order), "parse_failures": sum(not b2[i].parsed for i in order),
               "stored_accuracy_all_ci": stored["accuracy_all_ci"]},
        "learners": learners,
        "seed_means": seed_means,
        "sources": sources,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--out", default=str(REPO / "runs" / "b2_paired_v3"))
    args = parser.parse_args(argv)
    git = git_state()
    runs_dir, data_dir = Path(args.runs_dir), Path(args.data_dir)
    subset = load_subset(Path(args.subset))
    b2_dir = runs_dir / B2_RUN
    b2_metrics = json.loads((b2_dir / "metrics.json").read_text())
    replies = read_replies(b2_dir / "replies.jsonl")
    out_dir = Path(args.out)
    result = {
        "run_name": out_dir.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "code_sha256": {"b2_paired.py": sha256(Path(__file__))},
        "b2": {"run": f"runs/{B2_RUN}", "model": b2_metrics.get("model"), "replies_sha256": sha256(b2_dir / "replies.jsonl")},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "record (one choice question per record)", "paired": True},
        "note": ("learner minus gpt-4.1-mini (B2) accuracy on the same subset questions; a B2 reply that does not parse counts as wrong (accuracy_all). "
                 "caps_brier is the direct_brier arm (CAPS-Brier) on clean logged outcomes."),
        "domains": {domain: domain_block(runs_dir, data_dir, subset, spec, b2_metrics, replies) for domain, spec in DOMAINS.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for domain, d in result["domains"].items():
        print(f"\n== {domain}: {d['split']}, {d['n_records']} records; gpt-4.1-mini {d['b2']['accuracy_all']:.3f} ({d['b2']['parse_failures']} parse failures)")
        print("learner | accuracy | minus gpt-4.1-mini [95 percent paired interval] | predictions differ")
        for name, v in d["learners"].items():
            print(f"{name} | {v['accuracy']:.3f} | {v['minus_b2']:+.3f} [{v['minus_b2_ci'][0]:+.3f}, {v['minus_b2_ci'][1]:+.3f}] | {v['prediction_disagreement']:.3f}")
        for name, v in d["seed_means"].items():
            print(f"{name} seed mean ({len(v['members'])} seeds) | {v['accuracy']:.3f} ({v['accuracy_range'][0]:.3f}-{v['accuracy_range'][1]:.3f}) | {v['minus_b2']:+.3f} [{v['minus_b2_ci'][0]:+.3f}, {v['minus_b2_ci'][1]:+.3f}] |")
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
