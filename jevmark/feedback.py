"""The v3 deployment-feedback log (task 3.2, decision 56, docs/V3_DESIGN.md sections 3 to 5).

A logging policy answers each message of the log domain once. It samples one action
from q = (1 - epsilon) p + epsilon / K over its own distribution p (epsilon 0.1), and
only the correctness of that action is revealed. Every interaction records the
record id, the question id, the option labels in the order the model read them, p,
the chosen action, its propensity q(a), the revealed outcome, the outcome after a
symmetric flip with probability 0.2 (the noisy condition), the model id and the
commit. Gold is never stored: only the full-label learner reads it, from the data
file.

Learners see the first N interactions. The first 0.9 N are for training (and for
fitting the temperature learner), and the last 0.1 N are for checkpoint selection by
the mean Bernoulli log-likelihood of the logged outcomes under the model's
probability of the logged action.

Randomness: actions come from random.Random(f"v3_log:{seed}") and flips from
random.Random(f"v3_flip:{seed}"), one draw of each per interaction, so the flips
do not depend on which actions were drawn.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EPSILON = 0.1
FLIP_RATE = 0.2
TRAIN_SHARE = 0.9
P_CLIP = 1e-6


@dataclass(frozen=True)
class Interaction:
    index: int  # position in the log
    record_id: str
    question_id: str
    labels: tuple[str, ...]  # option labels in the order the model read them
    probs: tuple[float, ...]  # the logging policy's p over those labels
    action: int  # index into labels
    propensity: float  # q(action) under (1 - epsilon) p + epsilon / K
    outcome: int  # 1 if the chosen option is gold
    flipped_outcome: int  # outcome flipped with probability flip_rate
    model: str
    commit: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["labels"], row["probs"] = list(self.labels), list(self.probs)
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Interaction:
        return cls(**{**row, "labels": tuple(row["labels"]), "probs": tuple(row["probs"])})


@dataclass(frozen=True)
class Prefix:
    train: tuple[Interaction, ...]  # the first 0.9 N
    select: tuple[Interaction, ...]  # the last 0.1 N


def behaviour(probs: Sequence[float], epsilon: float = EPSILON) -> list[float]:
    """q = (1 - epsilon) p + epsilon / K, with p renormalised first."""
    total = math.fsum(probs)
    return [(1.0 - epsilon) * p / total + epsilon / len(probs) for p in probs]


def sample_action(probs: Sequence[float], rng: random.Random, epsilon: float = EPSILON) -> tuple[int, float]:
    """One action from q by inverse transform on a single uniform draw, and its propensity q(action)."""
    q = behaviour(probs, epsilon)
    u = rng.random()
    cumulative = 0.0
    for action, mass in enumerate(q):
        cumulative += mass
        if u < cumulative:
            return action, q[action]
    return len(q) - 1, q[-1]  # u within rounding of 1


class FlipChannel:
    """Flips a revealed outcome with probability rate, from its own seeded stream: one draw per call."""

    def __init__(self, seed: int, rate: float = FLIP_RATE):
        self.rng = random.Random(f"v3_flip:{seed}")
        self.rate = rate

    def __call__(self, outcome: int) -> int:
        return 1 - outcome if self.rng.random() < self.rate else outcome


def interactions(
    rows: Iterable[tuple[str, str, Sequence[str], Sequence[float], str]],
    seed: int,
    model: str,
    commit: str,
    epsilon: float = EPSILON,
    flip_rate: float = FLIP_RATE,
) -> list[Interaction]:
    """The log for (record id, question id, labels, p, gold) rows in log order."""
    actions = random.Random(f"v3_log:{seed}")
    flip = FlipChannel(seed, flip_rate)
    out = []
    for index, (record_id, question_id, labels, probs, gold) in enumerate(rows):
        action, propensity = sample_action(probs, actions, epsilon)
        outcome = int(labels[action] == gold)
        out.append(Interaction(index, record_id, question_id, tuple(labels), tuple(float(p) for p in probs), action, propensity, outcome, flip(outcome), model, commit))
    return out


def write_log(path: Path, log: Sequence[Interaction]) -> str:
    """Write the log as JSONL and return its sha256."""
    raw = "".join(json.dumps(i.to_dict(), ensure_ascii=False) + "\n" for i in log).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def read_log(path: Path) -> list[Interaction]:
    return [Interaction.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix(log: Sequence[Interaction], n: int, train_share: float = TRAIN_SHARE) -> Prefix:
    """The first n interactions, split into the first floor(train_share * n) for training and the rest for selection."""
    if not 0 < n <= len(log):
        raise ValueError(f"prefix of {n} interactions requested from a log of {len(log)}")
    cut = math.floor(train_share * n + 1e-9)
    if not 0 < cut < n:
        raise ValueError(f"a prefix of {n} leaves no training or no selection interactions at train share {train_share}")
    return Prefix(tuple(log[:cut]), tuple(log[cut:n]))


def bernoulli_log_likelihood(p_action: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean of r log p + (1 - r) log(1 - p) over interactions, p clipped to [1e-6, 1 - 1e-6]; higher is better."""
    if len(p_action) != len(outcomes) or not outcomes:
        raise ValueError("need one probability per outcome, and at least one")
    total = 0.0
    for p, r in zip(p_action, outcomes):
        p = min(max(p, P_CLIP), 1.0 - P_CLIP)
        total += r * math.log(p) + (1 - r) * math.log(1.0 - p)
    return total / len(outcomes)
