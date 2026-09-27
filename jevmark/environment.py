"""The stochastic-outcome environment of RLCD stage 3 (task 2.5, decision 54).

The data stay frozen; the environment is a layer on top of them. A question with
K options (K = 2 for a noul) has a noise rate

    eta(K) = min(0.40, 0.05 + 0.03 (K - 2))

and a target distribution theta: 1 - eta on the gold option and eta / (K - 1) on
each other option. On every visit the environment draws its accepted answer from
theta (gold with probability 1 - eta, otherwise a uniformly random other option)
with its own seeded generator, and an action's revealed outcome is 1 when it equals
the accepted answer. A policy that reports theta is then exactly calibrated
against the outcomes, and the right confidence depends on K, which a single
temperature cannot express.

Metrics against theta, per question with distribution p:

- expected Brier: E over y ~ theta of sum_k (p_k - 1[k = y])^2 = sum p^2 - 2 sum p theta + 1,
  minimised by p = theta at 1 - sum theta^2
- cross-entropy: -sum_k theta_k log p_k, minimised by p = theta at the entropy of theta;
  its excess over that entropy is KL(theta || p)
- accuracy against gold (theta's argmax is always gold)
- a calibration table by K: the count, mean top-1 probability, mean probability on
  gold, 1 - eta(K), and the expected hit rate of the top-1 option (theta of the
  predicted option: 1 - eta when it is gold, eta / (K - 1) otherwise); the gap of a
  K is mean top-1 probability minus mean expected hit rate, the calibration of the
  reported confidence against the environment's outcomes, and the calibration gap
  of a set of questions is the count-weighted mean of |gap| over K. gap_gold, mean
  p(gold) - (1 - eta(K)), is reported beside it; it is not a calibration measure
  for a model that is sometimes wrong, since softening lowers p(gold) on the
  questions it gets right and raises it on the ones it gets wrong
- ECE (15 bins, top-1 probability) against outcomes sampled once from theta

Two post-hoc temperatures for comparison: a global T fitted on bandit outcomes the
policy itself collects in the environment (G actions from (1 - epsilon) p +
epsilon uniform, the pathwise binary log loss of p_T(a) against the outcome), and
an oracle T per K fitted on a split's own theta (minimum mean cross-entropy).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from jevmark.calibration import LOG_T_BOUNDS, PROB_FLOOR, scale_probs
from jevmark.metrics import QuestionResult, ece

ETA_BASE = 0.05
ETA_SLOPE = 0.03
ETA_CAP = 0.40
P_CLIP = 1e-6  # as the log reward and direct_log in scripts/train_rlcd.py


def eta(k: int) -> float:
    """The environment's noise rate for a question with k options."""
    if k < 2:
        raise ValueError(f"a question has at least 2 options, got {k}")
    return min(ETA_CAP, ETA_BASE + ETA_SLOPE * (k - 2))


def theta(k: int, gold: int) -> np.ndarray:
    """The target distribution: 1 - eta(k) on gold, eta(k) / (k - 1) on each other option."""
    if not 0 <= gold < k:
        raise ValueError(f"gold {gold} out of range for {k} options")
    e = eta(k)
    out = np.full(k, e / (k - 1))
    out[gold] = 1.0 - e
    return out


def env_seed(seed: int) -> int:
    """The environment's generator seed for a run seed, apart from the sampler's."""
    return zlib.crc32(f"environment:{seed}".encode())


class NoisyEnvironment:
    """Draws the accepted answer of each visit from theta with its own torch generator."""

    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(env_seed(seed))

    def accepted(self, gold: int, k: int) -> int:
        if torch.rand((), generator=self.generator).item() < 1.0 - eta(k):
            return gold
        other = int(torch.randint(k - 1, (), generator=self.generator).item())
        return other if other < gold else other + 1

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)


# Metrics against theta


def expected_brier(p: np.ndarray, target: np.ndarray) -> float:
    return float((p * p).sum() - 2.0 * (p * target).sum() + 1.0)


def cross_entropy(p: np.ndarray, target: np.ndarray) -> float:
    return float(-(target * np.log(np.maximum(p, PROB_FLOOR))).sum())


def question_arrays(results: Sequence[QuestionResult]) -> dict[str, np.ndarray]:
    """Per question: expected Brier and cross-entropy against theta, their minima, correctness against gold, p(gold), top-1, the top-1 option's expected hit rate, 1 - eta and K."""
    out: dict[str, list[float]] = {k: [] for k in ("brier", "ce", "brier_min", "entropy", "correct", "p_gold", "top1", "top1_hit", "one_minus_eta", "k")}
    for r in results:
        p = np.asarray(r.probs, dtype=float)
        p = p / p.sum()
        t = theta(r.k, r.gold)
        out["brier"].append(expected_brier(p, t))
        out["ce"].append(cross_entropy(p, t))
        out["brier_min"].append(1.0 - float((t * t).sum()))
        out["entropy"].append(cross_entropy(t, t))
        out["correct"].append(float(r.correct))
        out["p_gold"].append(float(p[r.gold]))
        out["top1"].append(float(p.max()))
        out["top1_hit"].append(float(t[r.prediction]))
        out["one_minus_eta"].append(1.0 - eta(r.k))
        out["k"].append(float(r.k))
    return {k: np.asarray(v) for k, v in out.items()}


def by_k(results: Sequence[QuestionResult]) -> dict[str, dict[str, float]]:
    """The calibration table by K: count, mean top-1 probability, mean p(gold), 1 - eta(K), the top-1 option's mean expected hit rate, the gap (mean top-1 minus that rate) and gap_gold (mean p(gold) - (1 - eta(K)))."""
    a = question_arrays(results)
    table = {}
    for k in sorted(set(int(x) for x in a["k"])):
        m = a["k"] == k
        table[str(k)] = {
            "count": int(m.sum()),
            "mean_top1": float(a["top1"][m].mean()),
            "mean_p_gold": float(a["p_gold"][m].mean()),
            "one_minus_eta": 1.0 - eta(k),
            "expected_top1_hit": float(a["top1_hit"][m].mean()),
            "gap": float(a["top1"][m].mean() - a["top1_hit"][m].mean()),
            "gap_gold": float(a["p_gold"][m].mean()) - (1.0 - eta(k)),
        }
    return table


def calibration_gap(table: dict[str, dict[str, float]]) -> float:
    """Count-weighted mean over K of |mean top-1 probability - mean expected hit rate of the top-1 option|."""
    n = sum(row["count"] for row in table.values())
    return sum(row["count"] * abs(row["gap"]) for row in table.values()) / n


def sampled_outcomes(results: Sequence[QuestionResult], seed_text: str) -> list[int]:
    """One accepted answer per question drawn from theta, from a generator seeded by seed_text."""
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    return [int(rng.choice(r.k, p=theta(r.k, r.gold))) for r in results]


def env_metrics(results: Sequence[QuestionResult], seed_text: str) -> dict[str, Any]:
    """Every metric against theta for one set of questions; ECE uses outcomes sampled once with seed_text."""
    a = question_arrays(results)
    table = by_k(results)
    outcomes = sampled_outcomes(results, seed_text)
    top1 = [r.top1 for r in results]
    hits = [r.prediction == y for r, y in zip(results, outcomes)]
    return {
        "n": len(results),
        "expected_brier": float(a["brier"].mean()),
        "expected_brier_min": float(a["brier_min"].mean()),
        "cross_entropy": float(a["ce"].mean()),
        "entropy_theta": float(a["entropy"].mean()),
        "kl_theta": float((a["ce"] - a["entropy"]).mean()),
        "accuracy": float(a["correct"].mean()),
        "calibration_gap": calibration_gap(table),
        "by_k": table,
        "ece_sampled": ece(top1, hits),
        "accuracy_sampled": sum(hits) / len(hits),
    }


# Post-hoc temperatures


def _padded_log_probs(results: Sequence[QuestionResult]) -> np.ndarray:
    """log p per question, padded with -inf to the largest K."""
    logp = np.full((len(results), max(r.k for r in results)), -np.inf)
    for i, r in enumerate(results):
        logp[i, : r.k] = np.log(np.maximum(np.asarray(r.probs, dtype=float), PROB_FLOOR))
    return logp


def _log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log softmax of padded logits; padded entries stay -inf."""
    top = z.max(axis=1, keepdims=True)
    return z - (top + np.log(np.exp(z - top).sum(axis=1, keepdims=True)))


def bandit_samples(results: Sequence[QuestionResult], group_size: int, epsilon: float, seed: int) -> list[tuple[int, int, float]]:
    """(question index, action, outcome) for G actions per question from (1 - epsilon) p + epsilon uniform, outcomes from a NoisyEnvironment.

    Actions use a torch generator seeded with `seed` (as the RLCD sampler), the
    accepted answers the environment of that seed, one per question visit.
    """
    sampler = torch.Generator().manual_seed(seed)
    env = NoisyEnvironment(seed)
    samples = []
    for i, r in enumerate(results):
        p = torch.tensor(r.probs, dtype=torch.float64)
        p = p / p.sum()
        q = (1.0 - epsilon) * p + epsilon / r.k
        actions = torch.multinomial(q.float(), group_size, replacement=True, generator=sampler).tolist()
        accepted = env.accepted(r.gold, r.k)
        samples.extend((i, a, 1.0 if a == accepted else 0.0) for a in actions)
    return samples


def fit_bandit_temperature(results: Sequence[QuestionResult], group_size: int = 4, epsilon: float = 0.1, seed: int = 0) -> float:
    """The global T in [0.05, 20] minimising the mean binary log loss of softmax(log p / T)[a] against the revealed outcome, over bandit samples of the policy itself."""
    if not results:
        raise ValueError("cannot fit a temperature on no questions")
    samples = bandit_samples(results, group_size, epsilon, seed)
    logp = _padded_log_probs(results)
    index = np.array([s[0] for s in samples])
    action = np.array([s[1] for s in samples])
    outcome = np.array([s[2] for s in samples])

    def loss(u: float) -> float:
        p = np.clip(np.exp(_log_softmax(logp / math.exp(u))[index, action]), P_CLIP, 1.0 - P_CLIP)
        return float(-np.mean(outcome * np.log(p) + (1.0 - outcome) * np.log(1.0 - p)))

    found = minimize_scalar(loss, bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6})
    return math.exp(float(found.x))


def fit_oracle_temperature(results: Sequence[QuestionResult]) -> float:
    """The T in [0.05, 20] minimising the mean cross-entropy of softmax(log p / T) against theta; an oracle, since it reads theta."""
    if not results:
        raise ValueError("cannot fit a temperature on no questions")
    logp = _padded_log_probs(results)
    target = np.zeros_like(logp)
    for i, r in enumerate(results):
        target[i, : r.k] = theta(r.k, r.gold)
    live = target > 0

    def loss(u: float) -> float:
        return float(-np.where(live, target * _log_softmax(logp / math.exp(u)), 0.0).sum(axis=1).mean())

    found = minimize_scalar(loss, bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6})
    return math.exp(float(found.x))


def oracle_temperatures_by_k(results: Sequence[QuestionResult]) -> dict[int, float]:
    """One oracle temperature per K."""
    groups: dict[int, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(r.k, []).append(r)
    return {k: fit_oracle_temperature(rs) for k, rs in sorted(groups.items())}


def scale_by_k(results: Sequence[QuestionResult], temperatures: dict[int, float]) -> list[QuestionResult]:
    """Each question's distribution scaled by the temperature of its K."""
    return [replace(r, probs=scale_probs(r.probs, temperatures[r.k])) for r in results]
