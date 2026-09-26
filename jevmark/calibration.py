"""Temperature scaling on stored answer distributions, without the model (task 2.1).

The readout computes p = softmax(z) over a question's letter logits z, and a
temperature T gives softmax(z / T) (API_SPEC section 5). log p = z - logsumexp(z),
so log p equals z up to a constant per question, and a softmax is unchanged by
adding a constant to every input: softmax(log p / T) = softmax(z / T). Temperature
scaling can therefore be fitted and applied to the probabilities in a run's
results.jsonl.gz exactly as if the model had been rerun with that temperature, up
to the fp32 rounding of the stored probabilities (tested against the model on the
tiny backbone). Stored probabilities are floored at 1e-45, the smallest positive
fp32 value, before the logarithm, so an underflowed probability stays finite.
At T = 1 the result is the stored distribution renormalised: fp32 probabilities sum
to 1 only within fp32 rounding, so metrics can move in about the eighth decimal.

Scaling never changes a prediction (the argmax, and for noul P(yes) >= 0.5, are
invariant under a positive temperature), so accuracy is unchanged and only the
calibration metrics and the confidence-based coverage move.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from jevmark.metrics import QuestionResult

PROB_FLOOR = 1e-45
LOG_T_BOUNDS = (math.log(0.05), math.log(20.0))


def scale_probs(probs: Sequence[float], temperature: float) -> tuple[float, ...]:
    """softmax(log p / T), computed stably."""
    z = np.log(np.maximum(np.asarray(probs, dtype=float), PROB_FLOOR)) / temperature
    z -= z.max()
    e = np.exp(z)
    return tuple(float(x) for x in e / e.sum())


def scale_result(r: QuestionResult, temperature: float) -> QuestionResult:
    """The result with every stored distribution scaled: its probabilities, the negated noul's P(yes) and the reordered distribution."""
    negated = None if r.negated_p_yes is None else scale_probs((r.negated_p_yes, 1.0 - r.negated_p_yes), temperature)[0]
    shuffled = None if r.shuffled_probs is None else scale_probs(r.shuffled_probs, temperature)
    return replace(r, probs=scale_probs(r.probs, temperature), negated_p_yes=negated, shuffled_probs=shuffled)


def scale_results(results: Sequence[QuestionResult], temperature: float) -> list[QuestionResult]:
    return [scale_result(r, temperature) for r in results]


def _padded_log_probs(results: Sequence[QuestionResult]) -> tuple[np.ndarray, np.ndarray]:
    k_max = max(r.k for r in results)
    logp = np.full((len(results), k_max), -np.inf)
    for i, r in enumerate(results):
        logp[i, : r.k] = np.log(np.maximum(np.asarray(r.probs, dtype=float), PROB_FLOOR))
    return logp, np.array([r.gold for r in results])


def mean_nll(results: Sequence[QuestionResult], temperature: float) -> float:
    """Mean over questions of -log softmax(log p / T)[gold], every question weighted equally (as in training)."""
    logp, gold = _padded_log_probs(results)
    return _mean_nll(logp, gold, temperature)


def _mean_nll(logp: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    z = logp / temperature
    top = z.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(z - top).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(gold)), gold]))


def fit_temperature(results: Sequence[QuestionResult]) -> float:
    """The T in [0.05, 20] minimising mean NLL, searched over log T (bounded Brent)."""
    if not results:
        raise ValueError("cannot fit a temperature on no questions")
    logp, gold = _padded_log_probs(results)
    found = minimize_scalar(lambda u: _mean_nll(logp, gold, math.exp(u)), bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6})
    return math.exp(float(found.x))
