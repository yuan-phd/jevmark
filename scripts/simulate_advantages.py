"""The decision 52 simulation: RLCD advantages under each reward, on CPU, from a run's stored probabilities.

    uv run python scripts/simulate_advantages.py runs/sft_06b

Reads the gold-dependent questions of one split (default train) from the run's
results.jsonl.gz, and for each question draws G actions from the behaviour
distribution q = (1 - epsilon) p + epsilon / K with one seeded torch generator, in
file order. The same actions serve every reward. Rewards, advantages (group mean,
no std division), the behaviour mix and the clipped importance weights are the
functions of scripts/train_rlcd.py itself, so the simulation checks the training
code rather than a copy of it. No model is loaded.

For the REINFORCE rewards outcome, outcome_minus_p, brier and log, overall and per
question type:

- gold_negative_advantage: the share of gold samples with A < 0
- wrong_positive_advantage: the share of wrong samples with A > 0
- mean_advantage_gold, mean_advantage_wrong, and the same times the importance
  weight w (mean_weighted_advantage_*), the coefficient on log p(a) in the loss
- gold_logit_down, gold_logit_up: the share of questions whose update direction
  mean over samples of A w (e_a - p) lowers or raises the gold logit, and its mean
- mixed_groups: over groups holding both gold and wrong samples, the share where
  some wrong sample is rewarded below some gold sample, the share where every
  reward is equal, and the share where the gold samples' mean advantage is negative

- gold_logit_zero: the share of questions whose update leaves the gold logit unchanged
  (1 - down - up), split into all_samples_identical (all G sampled actions the same,
  so every reward and advantage is equal), distinct_samples_equal_rewards (different
  actions, every advantage zero; for brier and log every noul group) and
  nonzero_advantage (the rest: some advantage non-zero, the step below the 1e-12
  tolerance, as when gold is sampled at p_g near 1 next to a wrong action of tiny p)

the expected gold-logit gradient: with R detached and the group mean of all G
rewards as the baseline, E[mean_i A_i w_i (1[a_i = g] - p_g)] =
(1 - 1/G) p_g (R(g) - sum_a p_a R(a)) whenever the weight clip does not bind (p / q <
1 / (1 - epsilon) under the epsilon mix), because the baseline's own-sample term
removes 1/G of the expectation and its other samples are independent of a_i, and
E_p[1[a = g] - p_g] = 0. For brier and log every wrong action's reward is at least
gold's (p_a <= 1 - p_g), so the value is at most 0 for every K and exactly 0 at K 2.
It is evaluated in closed form on every question, and checked against an exact
enumeration of all K^G action tuples under q, with train_rlcd.py's reward, advantage
and weight functions, on the first questions of every (type, K),

plus a sign check: question_loss of train_rlcd.py on K 3, p = (0.6, 0.3, 0.1),
gold 0, actions forced to (0, 1, 2), beta 0, for every arm, with the loss and
gradient computed by hand next to it and p after one plain descent step at lr 1.

Writes runs/advantage_simulation_<size>/metrics.json (size from the run name, for
example 06b from sft_06b; --out overrides), with the source file's sha256, the git
state, and the sha256 of this script and of train_rlcd.py, which pin the code even
when the script is newer than the recorded commit.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from jevmark.metrics import QUESTION_TYPES, QuestionResult, gold_dependent, read_results
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
REWARDS = ("outcome", "outcome_minus_p", "brier", "log")  # the REINFORCE rewards of decision 52
SIGN_CHECK_P = (0.6, 0.3, 0.1)
SIGN_CHECK_ACTIONS = (0, 1, 2)
TOL = 1e-12
HAND_TOL = 1e-6  # question_loss computes in fp32 (z.float()), the hand values in float64
ENUM_TOL = 1e-12


def load_train_rlcd() -> Any:
    """scripts/train_rlcd.py as a module, so its own reward and advantage functions are the ones simulated."""
    spec = importlib.util.spec_from_file_location("train_rlcd", REPO / "scripts" / "train_rlcd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def draw(results: Sequence[QuestionResult], rl: Any, group_size: int, epsilon: float, seed: int) -> list[tuple[str, torch.Tensor, int, torch.Tensor, torch.Tensor]]:
    """(type, p, gold, q, actions) per question, in order, from one generator."""
    generator = torch.Generator().manual_seed(seed)
    draws = []
    for r in results:
        p = torch.tensor(r.probs, dtype=torch.float64)
        p = p / p.sum()
        q = rl.behaviour(p, epsilon)
        actions = torch.multinomial(q.float(), group_size, replacement=True, generator=generator)
        draws.append((r.qtype, p, r.gold, q, actions))
    return draws


def _share(x: list[bool]) -> float | None:
    return sum(x) / len(x) if x else None


def _mean(x: list[float]) -> float | None:
    return sum(x) / len(x) if x else None


def simulate(draws: Sequence[tuple[str, torch.Tensor, int, torch.Tensor, torch.Tensor]], rl: Any, arm: str, clip: float) -> dict[str, Any]:
    a_gold, a_wrong, aw_gold, aw_wrong, step, mixed_below, mixed_equal, mixed_gold_negative = [], [], [], [], [], [], [], []
    zero_identical, zero_equal, zero_other = [], [], []
    for _, p, gold, q, actions in draws:
        r = (actions == gold).double()
        rewards = rl.reward(arm, r, p[actions])
        a = rl.advantages(rewards, False)
        w = rl.importance_weights(p[actions], q[actions], clip)
        for ai, wi, ri in zip(a.tolist(), w.tolist(), r.tolist()):
            (a_gold if ri else a_wrong).append(ai)
            (aw_gold if ri else aw_wrong).append(ai * wi)
        step.append(float(((a * w) * (r - p[gold])).mean()))
        if abs(step[-1]) <= TOL:
            identical = bool((actions == actions[0]).all())
            all_zero = bool(a.abs().max() < TOL)
            zero_identical.append(identical)
            zero_equal.append(all_zero and not identical)
            zero_other.append(not all_zero)
        n_gold = int(r.sum())
        if 0 < n_gold < len(r):
            mixed_below.append(bool(rewards[r == 0].min() < rewards[r == 1].max() - TOL))
            mixed_equal.append(bool((rewards - rewards.mean()).abs().max() < TOL))
            mixed_gold_negative.append(bool(a[r == 1].mean() < -TOL))
    return {
        "questions": len(draws),
        "gold_samples": len(a_gold),
        "wrong_samples": len(a_wrong),
        "gold_negative_advantage": _share([x < -TOL for x in a_gold]),
        "wrong_positive_advantage": _share([x > TOL for x in a_wrong]),
        "mean_advantage_gold": _mean(a_gold),
        "mean_advantage_wrong": _mean(a_wrong),
        "mean_weighted_advantage_gold": _mean(aw_gold),
        "mean_weighted_advantage_wrong": _mean(aw_wrong),
        "gold_logit_down": _share([x < -TOL for x in step]),
        "gold_logit_up": _share([x > TOL for x in step]),
        "mean_gold_logit_step": _mean(step),
        "gold_logit_zero": {
            "share": len(zero_identical) / len(draws) if draws else None,
            "all_samples_identical": sum(zero_identical) / len(draws) if draws else None,
            "distinct_samples_equal_rewards": sum(zero_equal) / len(draws) if draws else None,
            "nonzero_advantage": sum(zero_other) / len(draws) if draws else None,
        },
        "mixed_groups": {
            "n": len(mixed_below),
            "some_wrong_rewarded_below_gold": _share(mixed_below),
            "all_rewards_equal": _share(mixed_equal),
            "gold_mean_advantage_negative": _share(mixed_gold_negative),
        },
    }


def expected_gold_step(rl: Any, arm: str, p: torch.Tensor, gold: int, group_size: int) -> float:
    """(1 - 1/G) p_g (R(g) - sum_a p_a R(a)): the expected gold-logit step with R detached, unclipped weights."""
    k = p.shape[-1]
    options = torch.arange(k)
    rewards = rl.reward(arm, (options == gold).double(), p)
    return float((1.0 - 1.0 / group_size) * p[gold] * (rewards[gold] - (p * rewards).sum()))


def enumerated_gold_step(rl: Any, arm: str, p: torch.Tensor, gold: int, q: torch.Tensor, group_size: int, clip: float) -> float:
    """E over all K^G action tuples drawn from q of mean_i A_i w_i (1[a_i = g] - p_g), with train_rlcd.py's functions."""
    k = p.shape[-1]
    tuples = torch.cartesian_prod(*[torch.arange(k)] * group_size).reshape(-1, group_size)
    r = (tuples == gold).double()
    a = rl.advantages(rl.reward(arm, r, p[tuples]), False)
    w = rl.importance_weights(p[tuples], q[tuples], clip)
    probability = q[tuples].prod(dim=-1)
    return float((probability * ((a * w) * (r - p[gold])).mean(dim=-1)).sum())


def expected_gradient(draws: Sequence[tuple[str, torch.Tensor, int, torch.Tensor, torch.Tensor]], rl: Any, group_size: int, epsilon: float, clip: float, per_cell: int) -> dict[str, Any]:
    """The closed-form expected gold-logit step on every question, by K, and its check against exact enumeration."""
    max_weight = max(float((p / q).max()) for _, p, _, q, _ in draws)
    out: dict[str, Any] = {
        "formula": "E[mean_i A_i w_i (1[a_i = g] - p_g)] = (1 - 1/G) p_g (R(g) - sum_a p_a R(a)), R detached, group-mean baseline over all G samples, unclipped weights",
        "group_size": group_size,
        "max_weight_p_over_q": max_weight,
        "weight_bound_1_over_1_minus_epsilon": 1.0 / (1.0 - epsilon),
        "clip": clip,
        "clip_binds": max_weight > clip,
        "rewards": {},
    }
    by_k: dict[int, list[int]] = {}
    for i, (_, p, _, _, _) in enumerate(draws):
        by_k.setdefault(p.shape[-1], []).append(i)
    checked: dict[tuple[str, int], list[int]] = {}
    for i, (qtype, p, _, _, _) in enumerate(draws):
        cell = checked.setdefault((qtype, p.shape[-1]), [])
        if len(cell) < per_cell:
            cell.append(i)
    for arm in REWARDS:
        values = [expected_gold_step(rl, arm, p, gold, group_size) for _, p, gold, _, _ in draws]
        blocks = {}
        for k, idx in sorted(by_k.items()):
            v = [values[i] for i in idx]
            blocks[str(k)] = {
                "questions": len(v),
                "max": max(v),
                "min": min(v),
                "mean": sum(v) / len(v),
                "share_negative": sum(x < -TOL for x in v) / len(v),
                "share_zero": sum(abs(x) <= TOL for x in v) / len(v),
                "share_positive": sum(x > TOL for x in v) / len(v),
            }
        differences = [abs(enumerated_gold_step(rl, arm, draws[i][1], draws[i][2], draws[i][3], group_size, clip) - values[i]) for idx in checked.values() for i in idx]
        out["rewards"][arm] = {
            "overall": {
                "max": max(values),
                "share_negative": sum(x < -TOL for x in values) / len(values),
                "share_zero": sum(abs(x) <= TOL for x in values) / len(values),
                "share_positive": sum(x > TOL for x in values) / len(values),
            },
            "by_k": blocks,
            "never_positive": max(values) <= TOL,
            "zero_at_k2": all(abs(values[i]) <= TOL for i in by_k.get(2, [])),
            "enumeration_check": {
                "questions": len(differences),
                "cells": {f"{t}_k{k}": len(idx) for (t, k), idx in sorted(checked.items())},
                "max_abs_difference": max(differences) if differences else None,
                "matches": bool(differences) and max(differences) < ENUM_TOL,
            },
        }
    return out


def hand_loss_and_grad(arm: str, p: Sequence[float], actions: Sequence[int], gold: int, epsilon: float, clip: float) -> tuple[float, list[float]]:
    """The REINFORCE loss -mean A w log p(a) and its gradient in the logits, in plain Python."""
    k, g = len(p), len(actions)
    r = [1.0 if a == gold else 0.0 for a in actions]
    pa = [p[a] for a in actions]
    if arm == "outcome":
        rewards = r
    elif arm == "outcome_minus_p":
        rewards = [ri - x for ri, x in zip(r, pa)]
    elif arm == "brier":
        rewards = [1 - (ri - x) ** 2 for ri, x in zip(r, pa)]
    else:
        rewards = [ri * math.log(x) + (1 - ri) * math.log(1 - x) for ri, x in zip(r, pa)]
    mean = sum(rewards) / g
    adv = [x - mean for x in rewards]
    w = [min(x / ((1 - epsilon) * x + epsilon / k), clip) for x in pa]
    loss = -sum(ai * wi * math.log(x) for ai, wi, x in zip(adv, w, pa)) / g
    grad = [-sum(ai * wi * ((1.0 if a == j else 0.0) - p[j]) for ai, wi, a in zip(adv, w, actions)) / g for j in range(k)]
    return loss, grad


def sign_check(rl: Any, epsilon: float, clip: float) -> dict[str, Any]:
    """train_rlcd.question_loss against the hand computation on K 3 with forced actions."""
    p = torch.tensor(SIGN_CHECK_P, dtype=torch.float64)
    forced = torch.tensor(SIGN_CHECK_ACTIONS)
    out: dict[str, Any] = {"p": list(SIGN_CHECK_P), "gold": 0, "actions": list(SIGN_CHECK_ACTIONS), "beta": 0.0, "arms": {}}
    real = torch.multinomial
    torch.multinomial = lambda *args, **kwargs: forced
    try:
        for arm in REWARDS:
            settings = rl.Settings(arm, len(SIGN_CHECK_ACTIONS), epsilon, True, clip, False, 0.0)
            z = p.log().requires_grad_(True)
            loss, _ = rl.question_loss(z, p.log(), 0, settings, torch.Generator())
            loss.backward()
            hand_loss, hand_grad = hand_loss_and_grad(arm, SIGN_CHECK_P, SIGN_CHECK_ACTIONS, 0, epsilon, clip)
            r = (forced == 0).double()
            out["arms"][arm] = {
                "rewards": rl.reward(arm, r, p[forced]).tolist(),
                "advantages": rl.advantages(rl.reward(arm, r, p[forced]), False).tolist(),
                "loss": float(loss.detach()),
                "loss_by_hand": hand_loss,
                "grad": z.grad.tolist(),
                "grad_by_hand": hand_grad,
                "max_abs_difference": max(abs(float(loss.detach()) - hand_loss), *(abs(a - b) for a, b in zip(z.grad.tolist(), hand_grad))),
                "code_matches_hand": abs(float(loss.detach()) - hand_loss) < HAND_TOL and max(abs(a - b) for a, b in zip(z.grad.tolist(), hand_grad)) < HAND_TOL,
                "p_after_one_descent_step_lr_1": torch.softmax(z.detach() - z.grad, -1).tolist(),
                "gold_logit_lowered": z.grad[0].item() > 0,
            }
    finally:
        torch.multinomial = real
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run directory holding results.jsonl.gz, for example runs/sft_06b")
    parser.add_argument("--split", default="train")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--importance-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enumerate-per-cell", type=int, default=20, help="questions per (type, K) checked by exact enumeration of the K^G action tuples")
    parser.add_argument("--out", default=None, help="output directory; default runs/advantage_simulation_<size> next to the run")
    args = parser.parse_args(argv)

    git = git_state()
    run_dir = Path(args.run)
    source = run_dir / "results.jsonl.gz"
    results = [r for r in gold_dependent(read_results(source)) if r.split == args.split]
    if not results:
        raise SystemExit(f"{source} has no gold-dependent {args.split} questions")
    size = re.search(r"\d+b", run_dir.name)
    out_dir = Path(args.out) if args.out else run_dir.parent / f"advantage_simulation_{size.group(0) if size else run_dir.name}"

    rl = load_train_rlcd()
    draws = draw(results, rl, args.group_size, args.epsilon, args.seed)
    by_type = {t: [d for d in draws if d[0] == t] for t in QUESTION_TYPES}
    rewards = {arm: {"overall": simulate(draws, rl, arm, args.importance_clip), **{t: simulate(d, rl, arm, args.importance_clip) for t, d in by_type.items() if d}} for arm in REWARDS}
    metrics = {
        "run_name": out_dir.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256((REPO / "scripts" / name).read_bytes()).hexdigest() for name in ("simulate_advantages.py", "train_rlcd.py")},
        "settings": {"split": args.split, "group_size": args.group_size, "epsilon": args.epsilon, "importance_clip": args.importance_clip, "normalize_std": False, "seed": args.seed},
        "questions": {"overall": len(draws), **{t: len(d) for t, d in by_type.items()}},
        "note": "Advantages of the REINFORCE rewards of scripts/train_rlcd.py on actions sampled from the stored probabilities; decision 52. brier and log are the known broken arms.",
        "rewards": rewards,
        "expected_gold_gradient": expected_gradient(draws, rl, args.group_size, args.epsilon, args.importance_clip, args.enumerate_per_cell),
        "sign_check": sign_check(rl, args.epsilon, args.importance_clip),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{len(draws)} gold-dependent {args.split} questions ({', '.join(f'{t} {len(d)}' for t, d in by_type.items())}); wrote {out_dir / 'metrics.json'}")
    print("reward | gold A<0 | wrong A>0 | mean A gold | mean A wrong | gold logit down / up")
    for arm, blocks in rewards.items():
        m = blocks["overall"]
        z = m["gold_logit_zero"]
        print(f"{arm} | {m['gold_negative_advantage']:.3f} | {m['wrong_positive_advantage']:.3f} | {m['mean_advantage_gold']:+.3f} | {m['mean_advantage_wrong']:+.3f} | {m['gold_logit_down']:.3f} / {m['gold_logit_up']:.3f}"
              f" | zero {z['share']:.3f} = identical {z['all_samples_identical']:.3f} + equal rewards {z['distinct_samples_equal_rewards']:.3f} + other {z['nonzero_advantage']:.3f}")
    e = metrics["expected_gold_gradient"]
    print(f"\nexpected gold-logit step {e['formula']}; max p/q {e['max_weight_p_over_q']:.4f} (clip {e['clip']} binds: {e['clip_binds']})")
    print("reward | max | share < 0 / = 0 / > 0 | never positive | zero at K 2 | enumeration max |diff| (questions)")
    for arm, v in e["rewards"].items():
        o, c = v["overall"], v["enumeration_check"]
        print(f"{arm} | {o['max']:+.2e} | {o['share_negative']:.3f} / {o['share_zero']:.3f} / {o['share_positive']:.3f} | {v['never_positive']} | {v['zero_at_k2']} | {c['max_abs_difference']:.1e} ({c['questions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
