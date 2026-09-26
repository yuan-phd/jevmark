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
    for _, p, gold, q, actions in draws:
        r = (actions == gold).double()
        rewards = rl.reward(arm, r, p[actions])
        a = rl.advantages(rewards, False)
        w = rl.importance_weights(p[actions], q[actions], clip)
        for ai, wi, ri in zip(a.tolist(), w.tolist(), r.tolist()):
            (a_gold if ri else a_wrong).append(ai)
            (aw_gold if ri else aw_wrong).append(ai * wi)
        step.append(float(((a * w) * (r - p[gold])).mean()))
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
        "mixed_groups": {
            "n": len(mixed_below),
            "some_wrong_rewarded_below_gold": _share(mixed_below),
            "all_rewards_equal": _share(mixed_equal),
            "gold_mean_advantage_negative": _share(mixed_gold_negative),
        },
    }


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
        "sign_check": sign_check(rl, args.epsilon, args.importance_clip),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{len(draws)} gold-dependent {args.split} questions ({', '.join(f'{t} {len(d)}' for t, d in by_type.items())}); wrote {out_dir / 'metrics.json'}")
    print("reward | gold A<0 | wrong A>0 | mean A gold | mean A wrong | gold logit down / up")
    for arm, blocks in rewards.items():
        m = blocks["overall"]
        print(f"{arm} | {m['gold_negative_advantage']:.3f} | {m['wrong_positive_advantage']:.3f} | {m['mean_advantage_gold']:+.3f} | {m['mean_advantage_wrong']:+.3f} | {m['gold_logit_down']:.3f} / {m['gold_logit_up']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
