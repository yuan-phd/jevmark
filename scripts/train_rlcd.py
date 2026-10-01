"""RLCD: bandit training of the letter readout from the SFT adapter (docs/TASKS.md 2.2).

    python scripts/train_rlcd.py --config configs/rlcd_06b.yaml --init runs/sft_06b arm=direct_brier seed=0
    python scripts/train_rlcd.py --config configs/rlcd_06b.yaml arm=direct_brier seed=0 --resume
    python scripts/train_rlcd.py --config configs/rlcd_06b.yaml --env noisy arm=direct_brier seed=0
    python scripts/train_rlcd.py --config configs/v3_06b.yaml --log runs/v3_log_s0/log.jsonl --n 5000 arm=direct_brier seed=0

The trainable LoRA starts as the SFT adapter of --init (a run directory holding
adapter/); a second copy of the backbone with that adapter merged is the frozen
reference policy p_ref. The sha256 of the SFT adapter's weights is recorded in the
run's config.yaml and train_summary.json.

Every optimizer step takes an effective batch of training records (the same seeded
order and per-epoch option reshuffles as train_sft.py). For every gold-dependent
question (form nouls are excluded from sampling and from the loss, decision 44):

    p = softmax(letter logits), q = (1 - epsilon) p + epsilon / K
    G actions a ~ q; only r_a = 1[a is the accepted answer] is revealed, for sampled actions only

The accepted answer depends on --env (decision 54):

- deterministic (the default, stages 1 and 2): the gold option.
- noisy (stage 3): drawn on each visit by jevmark.environment.NoisyEnvironment from
  theta, 1 - eta(K) on gold and eta(K) / (K - 1) on each other option, with
  eta(K) = min(0.40, 0.05 + 0.03 (K - 2)); the environment's generator is seeded
  from the run seed and saved with the resume state. Only outcome,
  outcome_minus_p, direct_brier and direct_log run here (sft_cont is refused: its
  cross-entropy on gold is not an answer to noisy outcomes, and the known broken
  arms are refused too). A fresh noisy run takes rlcd.beta and training.steps from
  the config's noisy section (beta 0, 1000 steps); overriding rlcd.beta or
  training.steps directly is refused, use noisy.beta and noisy.steps. The run name
  is rlcd_<size>_noisy_<arm>_s<seed> unless set.

and the arm decides the loss:

- outcome, outcome_minus_p (policy gradient): the reward R(a) is r_a or r_a - p_a,
  with p detached. The advantage is R(a) minus the mean over the G samples of that
  question (divided by their standard deviation when rlcd.normalize_std), the
  importance weight w = p(a) / q(a) clipped at rlcd.importance_clip corrects for
  sampling from q, and the loss is mean over samples of -A w log p(a), plus
  beta KL(p || p_ref) summed over options.
- direct_brier, direct_log (pathwise proper score): no policy gradient; the loss is
  the mean over the same samples of the sampled action's score with p_a
  differentiable, (r_a - p_a)^2 or -(r_a log p_a + (1 - r_a) log(1 - p_a)) with p
  clipped to [1e-6, 1 - 1e-6], plus beta KL(p || p_ref).
- sft_cont: the control for extra training: cross-entropy on the gold option of the
  same gold-dependent questions, same records, steps, learning rate and schedule, no
  sampling and no KL term.

Known broken, kept only to reproduce the negative result runs/rlcd_06b_brier_s0
(decision 52): brier and log, the same proper scores as detached REINFORCE rewards,
R(a) = 1 - (r_a - p_a)^2 or r_a log p_a + (1 - r_a) log(1 - p_a). A wrong action a
has p_a <= 1 - p_gold, so it never scores below the gold action: the policy gradient
moves probability away from gold for K > 2 and carries no signal for K = 2. They run
only with rlcd.reinforce_proper_score: true.

The loss is the mean over the micro-batch's gold-dependent questions. Sampling uses
its own torch generator, seeded from the config seed and saved with the resume state.

Validation every training.eval_every steps (and at step 0, the SFT adapter itself,
logged as the reference) on the same seeded 1000-record subset of valid as
train_sft.py, gold-dependent questions only: accuracy, ECE, NLL, Brier, mean
KL(p || p_ref), the mean expected reward under p (sum over a of p_a R(a); the Brier
reward for sft_cont, the arm's own score otherwise) and the mean expected probability of the
chosen action (sum of p_a squared). In noisy mode validation also reports, against
theta, the expected Brier score and the cross-entropy (and its excess over theta's
entropy), the expected reward uses outcomes distributed as theta, and accuracy stays
against gold. The best adapter by validation NLL (deterministic) or by cross-entropy
against theta (noisy) among the trained steps goes to adapter/, the final one to adapter_last/; the final full-valid
pass reports both. Pre-flight memory check, stale-state guard, --max-hours and
--resume behave as in train_sft.py (decision 45); the first-batch NaN check runs on
the backbone before the adapter is attached.

Run directory runs/<run_name>/ (run_name rlcd_<size>_<arm>_s<seed>, or
rlcd_<size>_noisy_<arm>_s<seed>, unless set):
config.yaml, model_id.txt, training_log.jsonl, adapter/, adapter_last/, last/,
train_summary.json, calibration.json.

Log mode (v3, task 3.3, decision 56, docs/V3_DESIGN.md section 5): --log
runs/v3_log_s<k>/log.jsonl --n N with configs/v3_06b.yaml replaces sampling by the
logged interactions of a deployment-feedback log (jevmark/feedback.py). Each logged
record's question is rebuilt in its stored option order, which is the order the
logging policy read; the logged action and its outcome (the flipped outcome with
--noisy) are the only feedback, with no behaviour mixture, no importance weights,
no KL term and no reference model. Training uses the first 0.9 N interactions,
every epoch in a seeded order. The arms:

- positive_sft: cross-entropy on the logged action of every interaction whose
  outcome is 1; outcome-0 interactions are skipped.
- full_sft: cross-entropy on gold for every interaction, gold read from the data
  file (training.data_file), never from the log; it never reads an outcome to
  train, and --noisy is refused for it.
- direct_brier: (r - p_a)^2 on the logged action's probability for every
  interaction, positives and negatives.

Every arm takes steps = max(v3.min_steps, ceil(v3.epochs x 0.9 N / effective
batch)), warmup and linear decay as configured, from the --init adapter (default
v3.init). Checkpoint selection, the same for every arm, is the mean Bernoulli
log-likelihood of the logged outcomes under the model's probability of the logged
action on the last 0.1 N interactions, every training.eval_every steps and at the
last step; step 0 (the starting adapter) is logged as the reference and is not a
candidate, and gold is never read for it. config.yaml and train_summary.json record
the log's sha256 and seed, N, the arm, the noisy flag and the selection curve.
Run names: v3_<size>_<arm>_n<N>_s<seed>, plus _noisy, plus _log<k> for a log drawn
with seed k > 0. --resume needs the same --log, --n and --noisy, which name the run.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import get_linear_schedule_with_warmup

from jevmark.config import load_config
from jevmark.data.form import FORM_KINDS
from jevmark.encode import Encoded, encode
from jevmark.environment import NoisyEnvironment
from jevmark.environment import cross_entropy as theta_cross_entropy
from jevmark.environment import expected_brier as theta_expected_brier
from jevmark.environment import theta
from jevmark.feedback import Interaction, bernoulli_log_likelihood, file_sha256, prefix, read_log
from jevmark.metrics import QuestionResult, split_metrics
from jevmark.model import JevMark, keep_lora_fp32
from jevmark.schema import ChoiceQuestion, Request
from jevmark.training import (
    PREFLIGHT_WARN_SHARE,
    STALE_STATE,
    PreflightError,
    fits,
    load_adapter_weights,
    load_state,
    log_line,
    prepare,
    preflight,
    read_jsonl,
    save_state,
    say,
    targets_for,
)

REPO = Path(__file__).resolve().parents[1]
PG_ARMS = ("outcome", "outcome_minus_p")
DIRECT_ARMS = ("direct_brier", "direct_log")
ARMS = ("sft_cont", *PG_ARMS, *DIRECT_ARMS)
REINFORCE_PROPER_ARMS = ("brier", "log")  # known broken (decision 52); only with rlcd.reinforce_proper_score
ENVS = ("deterministic", "noisy")
NOISY_ARMS = ("outcome", "outcome_minus_p", "direct_brier", "direct_log")  # decision 54
RL_STALE_STATE = (*STALE_STATE, "adapter_last")
P_CLIP = 1e-6


# Rewards, advantages, weights


def reward(arm: str, r: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """R(a) for revealed outcomes r (1 when the sampled action is gold) and the policy's probabilities p_a of those actions."""
    if arm == "outcome":
        return r
    if arm == "outcome_minus_p":
        return r - p_a
    if arm in ("brier", "direct_brier", "sft_cont"):
        return 1.0 - (r - p_a) ** 2
    if arm in ("log", "direct_log"):
        p = p_a.clamp(P_CLIP, 1.0 - P_CLIP)
        return r * torch.log(p) + (1.0 - r) * torch.log(1.0 - p)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS + REINFORCE_PROPER_ARMS}")


def advantages(rewards: torch.Tensor, normalize_std: bool) -> torch.Tensor:
    """Reward minus the group mean (the last dimension holds one question's G samples); optionally divided by the group standard deviation."""
    centred = rewards - rewards.mean(dim=-1, keepdim=True)
    if normalize_std:
        centred = centred / (rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-6)
    return centred


def behaviour(p: torch.Tensor, epsilon: float) -> torch.Tensor:
    """q = (1 - epsilon) p + epsilon uniform."""
    return (1.0 - epsilon) * p + epsilon / p.shape[-1]


def importance_weights(p_a: torch.Tensor, q_a: torch.Tensor, clip: float) -> torch.Tensor:
    """p(a) / q(a), clipped at `clip`."""
    return (p_a / q_a).clamp(max=clip)


# Data


@dataclass(frozen=True)
class RLExample:
    encoded: Encoded
    targets: tuple[int, ...]
    gold_dependent: tuple[bool, ...]  # False for form nouls, which are never sampled or trained on


def gold_dependent_mask(record: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(not (q["type"] == "noul" and qid in FORM_KINDS) for qid, q in record["questions"].items())


def epoch_examples(records: list[dict[str, Any]], jev: JevMark, max_tokens: int, seed: int, epoch: int) -> tuple[list[RLExample], int]:
    """The epoch's examples in a seeded order with seeded option shuffles, as train_sft.epoch_examples, with the form-noul mask."""
    rng = random.Random(f"{seed}:epoch{epoch}")
    order = list(range(len(records)))
    rng.shuffle(order)
    examples, dropped = [], 0
    for index in order:
        request, targets = prepare(records[index], rng)
        try:
            encoded = encode(request, jev.tokenizer, max_tokens)
        except ValueError:
            dropped += 1
            continue
        examples.append(RLExample(encoded, tuple(targets), gold_dependent_mask(records[index])))
    return examples, dropped


# Loss


@dataclass(frozen=True)
class Settings:
    arm: str
    group_size: int
    epsilon: float
    importance_weight: bool
    importance_clip: float
    normalize_std: bool
    beta: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Settings:
        rl = config["rlcd"]
        arm = config["arm"]
        if config.get("env", "deterministic") == "noisy" and arm not in NOISY_ARMS:
            reason = "sft_cont's cross-entropy on gold is not a response to noisy outcomes" if arm == "sft_cont" else "it is not a stage 3 arm"
            raise ValueError(f"arm {arm!r} does not run in the noisy environment ({reason}); expected one of {NOISY_ARMS} (decision 54)")
        if arm in REINFORCE_PROPER_ARMS:
            if not rl.get("reinforce_proper_score", False):
                raise ValueError(f"arm {arm!r} is the REINFORCE proper-score arm, known broken (decision 52); use direct_{arm}, or set rlcd.reinforce_proper_score=true to reproduce the negative result")
        elif arm not in ARMS:
            raise ValueError(f"arm {arm!r}: expected one of {ARMS}")
        return cls(arm, int(rl["group_size"]), float(rl["epsilon"]), bool(rl["importance_weight"]), float(rl["importance_clip"]), bool(rl["normalize_std"]), float(rl["beta"]))


def question_loss(z: torch.Tensor, z_ref: torch.Tensor, gold: int, s: Settings, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    """The loss of one gold-dependent question from its letter logits z (with gradients) and the reference's z_ref, and its statistics."""
    logp = F.log_softmax(z.float(), dim=-1)
    p = logp.exp()
    kl = (p * (logp - F.log_softmax(z_ref.float(), dim=-1))).sum()
    if s.arm == "sft_cont":
        loss = -logp[gold]
        return loss, {"kl": float(kl.detach()), "reward": float("nan"), "p_chosen": float(p.detach()[gold])}
    q = behaviour(p.detach(), s.epsilon)
    actions = torch.multinomial(q.cpu(), s.group_size, replacement=True, generator=generator).to(z.device)
    r = (actions == gold).float()
    p_a = p.detach()[actions]
    rewards = reward(s.arm, r, p_a)
    if s.arm == "direct_brier":
        loss = ((r - p[actions]) ** 2).mean() + s.beta * kl
    elif s.arm == "direct_log":
        loss = -reward("log", r, p[actions]).mean() + s.beta * kl
    else:
        weight = importance_weights(p_a, q[actions], s.importance_clip) if s.importance_weight else torch.ones_like(p_a)
        loss = -(advantages(rewards, s.normalize_std) * weight * logp[actions]).mean() + s.beta * kl
    return loss, {"kl": float(kl.detach()), "reward": float(rewards.mean()), "p_chosen": float(p_a.mean())}


def batch_loss(jev: JevMark, ref: JevMark, batch: Sequence[RLExample], s: Settings, generator: torch.Generator, env: NoisyEnvironment | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean loss over the micro-batch's gold-dependent questions, and the mean of their statistics; with env, each question's outcome is the environment's accepted answer."""
    encoded = [e.encoded for e in batch]
    logits = jev.slot_logits(encoded)
    with torch.no_grad():
        ref_logits = ref.slot_logits(encoded)
    losses, stats = [], []
    flat = iter(zip(logits, ref_logits))
    for example in batch:
        for target, keep in zip(example.targets, example.gold_dependent):
            z, z_ref = next(flat)
            if not keep:
                continue
            if env is not None:
                target = env.accepted(target, len(z))
            loss, stat = question_loss(z, z_ref, target, s, generator)
            losses.append(loss)
            stats.append(stat)
    if not losses:
        raise RuntimeError("a micro-batch without gold-dependent questions")
    mean = {k: sum(x[k] for x in stats) / len(stats) for k in ("kl", "reward", "p_chosen")}
    return torch.stack(losses).mean(), mean


# Validation


def validate(jev: JevMark, ref: JevMark, records: list[dict[str, Any]], batch_size: int, max_tokens: int, arm: str, noisy: bool = False) -> dict[str, Any]:
    """Accuracy, ECE, NLL, Brier, mean KL to p_ref, expected reward and expected p of the chosen action, on gold-dependent questions; with noisy, also the expected Brier and cross-entropy against theta."""
    was_training = jev.model.training
    jev.model.eval()
    results, kls, rewards, chosen, briers, ces, entropies = [], [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            requests = [Request.from_dict({"state": r["state"], "questions": r["questions"]}) for r in chunk]
            encoded = [encode(q, jev.tokenizer, max_tokens) for q in requests]
            flat = iter(zip(jev.slot_logits(encoded), ref.slot_logits(encoded)))
            for record, request in zip(chunk, requests):
                for question, target, keep in zip(request.questions, targets_for(request, record["gold"]), gold_dependent_mask(record)):
                    z, z_ref = next(flat)
                    if not keep:
                        continue
                    logp = F.log_softmax(z.float() / jev.temperature, dim=-1)
                    p = logp.exp()
                    kls.append(float((p * (logp - F.log_softmax(z_ref.float(), dim=-1))).sum()))
                    probs = tuple(float(x) for x in p.double().cpu())
                    if noisy:
                        target_dist = theta(len(p), target)
                        rate = torch.tensor(target_dist, dtype=p.dtype, device=p.device)
                        ones, zeros = torch.ones_like(p), torch.zeros_like(p)
                        rewards.append(float((p * (rate * reward(arm, ones, p) + (1 - rate) * reward(arm, zeros, p))).sum()))
                        dist = np.asarray(probs)
                        briers.append(theta_expected_brier(dist, target_dist))
                        ces.append(theta_cross_entropy(dist, target_dist))
                        entropies.append(theta_cross_entropy(target_dist, target_dist))
                    else:
                        outcomes = (torch.arange(len(p), device=p.device) == target).float()
                        rewards.append(float((p * reward(arm, outcomes, p)).sum()))
                    chosen.append(float((p * p).sum()))
                    labels = question.labels if isinstance(question, ChoiceQuestion) else tuple(str(i) for i in range(len(probs)))
                    results.append(QuestionResult(record["id"], question.id, question.type, probs, target, labels, kind=question.id if question.type == "noul" else None))
    if was_training:
        jev.model.train()
    overall = split_metrics(results)["overall"]
    n = len(kls)
    out = {
        "n": overall["n"],
        "accuracy": overall["accuracy"],
        "ece": overall["ece"],
        "nll": overall["nll"],
        "brier": overall["brier"],
        "kl_to_ref": sum(kls) / n,
        "expected_reward": sum(rewards) / n,
        "expected_p_chosen": sum(chosen) / n,
    }
    if noisy:
        out["expected_brier_theta"] = sum(briers) / n
        out["cross_entropy_theta"] = sum(ces) / n
        out["kl_theta"] = (sum(ces) - sum(entropies)) / n
    return out


# Setup


def adapter_sha256(adapter_dir: Path) -> str:
    return hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()


def first_batch_finite(jev: JevMark, batch: Sequence[RLExample]) -> bool:
    with torch.no_grad():
        return all(torch.isfinite(x).all().item() for x in jev.slot_logits([e.encoded for e in batch]))


def attach_sft_adapter(jev: JevMark, adapter_dir: Path, config: dict[str, Any]) -> None:
    """Wrap the backbone with the SFT adapter as the trainable LoRA."""
    jev.model = PeftModel.from_pretrained(jev.model, adapter_dir, is_trainable=True)
    keep_lora_fp32(jev.model)
    if config["training"].get("gradient_checkpointing"):
        jev.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        jev.model.enable_input_require_grads()
    jev.model.train()


def load_reference(config: dict[str, Any], init_dir: Path, device: torch.device, half: bool) -> JevMark:
    """The frozen reference policy: a second backbone with the SFT adapter merged, no gradients."""
    ref = JevMark.load(config, checkpoint=init_dir, device=device, half=half)
    ref.merge_lora()
    ref.model.eval()
    for param in ref.model.parameters():
        param.requires_grad_(False)
    ref.temperature = 1.0
    return ref


def resolve_run_name(config: dict[str, Any]) -> str:
    env = "noisy_" if config.get("env", "deterministic") == "noisy" else ""
    return config.get("run_name") or f"rlcd_{config['size']}_{env}{config['arm']}_s{config['seed']}"


def apply_noisy_defaults(config: dict[str, Any], overrides: Sequence[str]) -> None:
    """A fresh noisy run takes rlcd.beta and training.steps from the noisy section; overriding them directly would be silently replaced, so it is refused."""
    direct = [o.partition("=")[0] for o in overrides if o.partition("=")[0] in ("rlcd.beta", "training.steps")]
    if direct:
        raise SystemExit(f"--env noisy takes beta and steps from the noisy section; override noisy.beta or noisy.steps instead of {', '.join(direct)}")
    config["rlcd"]["beta"] = float(config["noisy"]["beta"])
    config["training"]["steps"] = int(config["noisy"]["steps"])


# Log mode (v3, task 3.3, decision 56)

LOG_ARMS = ("positive_sft", "full_sft", "direct_brier")


@dataclass(frozen=True)
class LogExample:
    encoded: Encoded
    action: int  # the logged action's index in the stored option order
    outcome: int  # the logged outcome, or the flipped one with --noisy
    gold: int | None  # the gold index, only for full_sft


def log_steps(n_train: int, epochs: int, min_steps: int, effective_batch: int) -> int:
    """max(min_steps, ceil(epochs x n_train / effective_batch)): the same for every arm at a given N."""
    return max(min_steps, math.ceil(epochs * n_train / effective_batch))


def log_seed_of(log_path: Path) -> int:
    """The seed the log was drawn with, from collect_log.py's metrics.json next to it."""
    metrics = log_path.parent / "metrics.json"
    if not metrics.is_file():
        raise SystemExit(f"--log {log_path}: no metrics.json next to it, so its seed and data hash are unknown")
    return int(json.loads(metrics.read_text())["sampling"]["seed"])


def log_run_name(config: dict[str, Any], n: int, noisy: bool, log_seed: int) -> str:
    if config.get("run_name"):
        return config["run_name"]
    return f"v3_{config['size']}_{config['arm']}_n{n}_s{config['seed']}" + ("_noisy" if noisy else "") + (f"_log{log_seed}" if log_seed else "")


def log_examples(interactions: Sequence[Interaction], records: dict[str, dict[str, Any]], jev: JevMark, max_tokens: int, noisy: bool, with_gold: bool) -> list[LogExample]:
    """One example per interaction, the question rebuilt in the record's stored option order, which must be the logged order."""
    examples = []
    for interaction in interactions:
        record = records[interaction.record_id]
        question = record["questions"][interaction.question_id]
        if tuple(question["criteria"]) != interaction.labels:
            raise SystemExit(f"{interaction.record_id}: the data file's option order differs from the logged one; the log was not drawn from this file")
        request = Request.from_dict({"state": record["state"], "questions": {interaction.question_id: question}})
        gold = list(question["criteria"]).index(record["gold"][interaction.question_id]) if with_gold else None
        outcome = interaction.flipped_outcome if noisy else interaction.outcome
        examples.append(LogExample(encode(request, jev.tokenizer, max_tokens), interaction.action, outcome, gold))
    return examples


def log_batch_loss(jev: JevMark, batch: Sequence[LogExample], arm: str) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean loss over the micro-batch, and the mean probability of the logged action and outcome rate."""
    losses, p_action = [], []
    for z, example in zip(jev.slot_logits([e.encoded for e in batch]), batch):
        logp = F.log_softmax(z.float(), dim=-1)
        if arm == "positive_sft":
            losses.append(-logp[example.action])
        elif arm == "full_sft":
            losses.append(-logp[example.gold])
        elif arm == "direct_brier":
            losses.append((example.outcome - logp[example.action].exp()) ** 2)
        else:
            raise ValueError(f"unknown log-mode arm {arm!r}; expected one of {LOG_ARMS}")
        p_action.append(float(logp.detach()[example.action].exp()))
    return torch.stack(losses).mean(), {"p_action": sum(p_action) / len(p_action), "outcome_rate": sum(e.outcome for e in batch) / len(batch)}


def select_score(jev: JevMark, examples: Sequence[LogExample], batch_size: int) -> dict[str, Any]:
    """The selection criterion on the held-out interactions: mean Bernoulli log-likelihood of the outcomes under p(logged action); no gold."""
    was_training = jev.model.training
    jev.model.eval()
    p_action = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            for z, example in zip(jev.slot_logits([e.encoded for e in chunk]), chunk):
                p_action.append(float(F.softmax(z.float(), dim=-1)[example.action]))
    if was_training:
        jev.model.train()
    outcomes = [e.outcome for e in examples]
    return {"n": len(examples), "criterion": bernoulli_log_likelihood(p_action, outcomes), "mean_p_action": sum(p_action) / len(p_action), "outcome_rate": sum(outcomes) / len(outcomes)}


def main_log(args: argparse.Namespace, started: float) -> int:
    """Train one v3 learner from a deployment-feedback log (the module docstring's log mode)."""
    if args.env != "deterministic":
        raise SystemExit("--log and --env noisy do not combine; the log's noisy condition is --noisy")
    if args.n is None or args.n < 1:
        raise SystemExit("--log needs --n, the number of logged interactions the learner sees")
    log_path = Path(args.log)
    log_sha = file_sha256(log_path)
    log_seed = log_seed_of(log_path)
    config = load_config(args.config, args.overrides)
    run_dir = Path(args.runs_dir) / log_run_name(config, args.n, args.noisy, log_seed)
    if args.resume:
        if not (run_dir / "last" / "state.pt").is_file():
            raise SystemExit(f"--resume: no saved state at {run_dir / 'last'}")
        config = load_config(run_dir / "config.yaml")
        if config["v3"]["log_sha256"] != log_sha or config["v3"]["n"] != args.n or config["v3"]["noisy"] != args.noisy:
            raise SystemExit(f"--resume: {run_dir} was trained on another log, N or noisy setting than given")
    else:
        stale = [name for name in RL_STALE_STATE if (run_dir / name).exists()]
        if stale:
            raise SystemExit(f"{run_dir} already holds a training run ({', '.join(stale)}); use --resume, or delete the directory, or another run_name")
        if config["arm"] not in LOG_ARMS:
            raise SystemExit(f"arm {config['arm']!r} does not run in log mode; expected one of {LOG_ARMS}")
        if args.noisy and config["arm"] == "full_sft":
            raise SystemExit("full_sft trains on gold and never reads an outcome, so --noisy would change only its selection; it is refused")
        if float(config["v3"]["beta"]) != 0.0:
            raise SystemExit("log mode has no KL term and no reference model; v3.beta must be 0 (decision 56)")
        init_dir = Path(args.init or config["v3"]["init"])
        if not (init_dir / "adapter" / "adapter_model.safetensors").is_file():
            raise SystemExit(f"--init {init_dir}: no adapter at {init_dir / 'adapter'}")
        config["run_name"] = run_dir.name
        config["v3"].update({"init": str(init_dir), "init_adapter_sha256": adapter_sha256(init_dir / "adapter"), "log": str(log_path), "log_sha256": log_sha, "log_seed": log_seed, "n": args.n, "noisy": args.noisy})
    init_dir = Path(config["v3"]["init"])
    if adapter_sha256(init_dir / "adapter") != config["v3"]["init_adapter_sha256"]:
        raise SystemExit(f"the adapter at {init_dir} is not the one this run started from (sha256 differs)")
    arm, seed, n, noisy = config["arm"], int(config["seed"]), int(config["v3"]["n"]), bool(config["v3"]["noisy"])
    train_cfg = config["training"]
    data_dir = Path(args.data_dir) if args.data_dir else REPO / train_cfg["data_dir"]
    data_path = data_dir / train_cfg["data_file"]
    data_sha = file_sha256(data_path)
    logged_data_sha = json.loads((log_path.parent / "metrics.json").read_text()).get("data_files_sha256", {}).get(train_cfg["data_file"])
    if logged_data_sha and logged_data_sha != data_sha:
        raise SystemExit(f"{data_path} has sha256 {data_sha}, but the log was drawn from {logged_data_sha}")
    config["v3"]["data_sha256"] = data_sha

    split = prefix(read_log(log_path), n, float(config["v3"]["train_share"]))
    micro = int(train_cfg["micro_batch"])
    effective = int(train_cfg["effective_batch"])
    accumulation = max(1, effective // micro)
    total_steps = log_steps(len(split.train), int(config["v3"]["epochs"]), int(config["v3"]["min_steps"]), effective)
    config["v3"].update({"n_train": len(split.train), "n_select": len(split.select), "steps": total_steps})
    warmup = int(train_cfg["warmup_steps"])
    max_tokens = int(train_cfg["max_tokens"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "model_id.txt").write_text(f"jevmark-{config['run_name']}\n")
    log_path_out = run_dir / "training_log.jsonl"
    torch.manual_seed(seed)

    jev = JevMark.load(config, device=args.device)
    records = {r["id"]: r for r in read_jsonl(data_path)}
    train_all = log_examples(split.train, records, jev, max_tokens, noisy, with_gold=arm == "full_sft")
    used = [k for k, e in enumerate(train_all) if arm != "positive_sft" or e.outcome == 1]
    pool = [train_all[k] for k in used]
    select = log_examples(split.select, records, jev, max_tokens, noisy, with_gold=False)
    if not pool:
        raise SystemExit(f"{arm}: no training interactions in the first {len(split.train)} (positive_sft needs outcome-1 interactions)")
    say(f"run {config['run_name']}: arm {arm}, log {log_path} (seed {log_seed}), N {n}: train on {len(split.train)} ({len(pool)} used), select on {len(select)}; {total_steps} steps; {jev.device}, autocast {jev.autocast_dtype}")

    fallback_used = False
    if not args.resume:
        with torch.no_grad():
            finite = all(torch.isfinite(x).all().item() for x in jev.slot_logits([e.encoded for e in pool[:micro]]))
        if not finite:
            if jev.autocast_dtype is None:
                raise RuntimeError("first-batch slot logits contain NaN or inf in fp32; not a precision problem")
            say("WARNING: NaN or inf in first-batch slot logits under fp16; reloading the model in fp32 (decision 28)")
            jev.use_fp32()
            fallback_used = True
        log_line(log_path_out, {"event": "start", "arm": arm, "log": str(log_path), "log_sha256": log_sha, "log_seed": log_seed, "n": n, "n_train": len(split.train), "n_used": len(pool), "n_select": len(select), "noisy": noisy, "init": str(init_dir), "init_adapter_sha256": config["v3"]["init_adapter_sha256"], "fp32_fallback_used": fallback_used, "autocast": str(jev.autocast_dtype), "total_steps": total_steps, "warmup_steps": warmup, "accumulation": accumulation})

    attach_sft_adapter(jev, init_dir / "adapter", config)
    trainable = [p for p in jev.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=jev.device.type == "cuda" and jev.autocast_dtype == torch.float16)

    # Pre-flight on the longest training records with a cross-entropy loss of the same shape (decision 45). Its
    # records carry the first option as a stand-in gold, so no arm but full_sft ever reads a gold label.
    pool_records = []
    for k in used:
        record = records[split.train[k].record_id]
        stand_in = {qid: next(iter(q["criteria"])) for qid, q in record["questions"].items()}
        pool_records.append({**record, "gold": stand_in})
    pool_records = list({r["id"]: r for r in pool_records}.values())
    _, _, lengths = fits(pool_records, jev, max_tokens)

    def preflight_loss(model: JevMark, batch: Sequence[Any]) -> torch.Tensor:
        return torch.stack([-F.log_softmax(z.float(), dim=-1)[e.targets[0]] for z, e in zip(model.slot_logits([e.encoded for e in batch]), batch)]).mean()

    try:
        check = preflight(jev, pool_records, lengths, micro, max_tokens, seed, scaler, preflight_loss)
    except PreflightError as err:
        log_line(log_path_out, {"event": "preflight", "passed": False, "error": str(err)[:500]})
        say(f"PREFLIGHT FAIL: {err}")
        raise SystemExit(f"PREFLIGHT FAIL: {config['run_name']} cannot train at micro-batch {micro}; lower training.micro_batch") from err
    log_line(log_path_out, {"event": "preflight", "passed": True, **check})
    memory = f"peak {check['peak_gib']:.2f} GiB of {check['total_gib']:.2f} GiB" if "peak_gib" in check else "peak memory not measured on CPU"
    say(f"PREFLIGHT PASS: worst-case micro-batch of {check['records']} records ({check['longest_tokens']} tokens): {memory}")

    progress: dict[str, Any] = {"step": 0, "epoch": 0, "micro_done": 0, "best_criterion": None, "best_step": None, "fp32_fallback_used": fallback_used, "init_select": None, "curve": []}
    if args.resume:
        progress = load_state(run_dir, jev, optimizer, scheduler, scaler)
        log_line(log_path_out, {"event": "resume", "step": progress["step"], "epoch": progress["epoch"]})
        say(f"resumed at step {progress['step']}, epoch {progress['epoch']}, micro-batch {progress['micro_done']}")
    else:
        init_select = select_score(jev, select, micro * 2)
        progress["init_select"] = init_select
        progress["curve"].append({"step": 0, **init_select})
        log_line(log_path_out, {"event": "select", "step": 0, **init_select})
        say(f"step 0 (the starting adapter): selection log-likelihood {init_select['criterion']:.4f}, mean p(action) {init_select['mean_p_action']:.4f}")

    limit = min(total_steps, args.limit_steps) if args.limit_steps else total_steps
    deadline = started + args.max_hours * 3600

    def run_selection(step: int) -> None:
        result = select_score(jev, select, micro * 2)
        progress["curve"].append({"step": step, **result})
        log_line(log_path_out, {"event": "select", "step": step, **result})
        say(f"step {step}: selection log-likelihood {result['criterion']:.4f}, mean p(action) {result['mean_p_action']:.4f}")
        if progress["best_criterion"] is None or result["criterion"] > progress["best_criterion"]:
            progress["best_criterion"], progress["best_step"] = result["criterion"], step
            jev.model.save_pretrained(run_dir / "adapter")

    stopped_by_time = False
    last_selected = None
    while progress["step"] < limit:
        order = list(range(len(pool)))
        random.Random(f"{seed}:epoch{progress['epoch']}").shuffle(order)
        batches = [[pool[i] for i in order[j : j + micro]] for j in range(0, len(order), micro)]
        groups = [batches[i : i + accumulation] for i in range(0, len(batches), accumulation)]
        for group in groups[progress["micro_done"] // accumulation :]:
            if progress["step"] >= limit:
                break
            losses, stats = [], []
            for batch in group:
                loss, stat = log_batch_loss(jev, batch, arm)
                scaler.scale(loss / len(group)).backward()
                losses.append(loss.item())
                stats.append(stat)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            progress["step"] += 1
            progress["micro_done"] += len(group)
            mean = {k: sum(x[k] for x in stats) / len(stats) for k in ("p_action", "outcome_rate")}
            log_line(log_path_out, {"step": progress["step"], "epoch": progress["epoch"], "loss": sum(losses) / len(losses), **mean, "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.perf_counter() - started, 1)})
            at_end = progress["step"] == limit
            if progress["step"] % int(train_cfg["eval_every"]) == 0 or at_end:
                run_selection(progress["step"])
                last_selected = progress["step"]
                save_state(run_dir, jev, optimizer, scheduler, scaler, progress)
            if time.perf_counter() > deadline and not at_end:
                stopped_by_time = True
                break
        if stopped_by_time:
            break
        if progress["micro_done"] >= len(batches):
            progress["epoch"] += 1
            progress["micro_done"] = 0

    if last_selected != progress["step"] and progress["step"] > 0 and not stopped_by_time:
        run_selection(progress["step"])
    save_state(run_dir, jev, optimizer, scheduler, scaler, progress)
    if stopped_by_time:
        log_line(log_path_out, {"event": "time_limit", "step": progress["step"], "max_hours": args.max_hours})
        say(f"time limit of {args.max_hours} h reached at step {progress['step']}; saved {run_dir / 'last'}; continue with --resume")
        return 0
    if progress["best_step"] is None:
        raise RuntimeError("no selection ran, so no best adapter exists")
    jev.model.save_pretrained(run_dir / "adapter_last")
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 1.0}) + "\n")
    summary = {
        "run_name": config["run_name"],
        "arm": arm,
        "mode": "log",
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "steps": progress["step"],
        "total_steps": total_steps,
        "limit_steps": args.limit_steps,
        "best_step": progress["best_step"],
        "best_by": "mean Bernoulli log-likelihood of the logged outcomes under p(logged action), last 0.1 N",
        "best_criterion": progress["best_criterion"],
        "log": str(log_path),
        "log_sha256": log_sha,
        "log_seed": log_seed,
        "data_sha256": data_sha,
        "n": n,
        "n_train": len(split.train),
        "n_used": len(pool),
        "n_select": len(select),
        "noisy": noisy,
        "init": str(init_dir),
        "init_adapter_sha256": config["v3"]["init_adapter_sha256"],
        "init_select": progress["init_select"],
        "selection_curve": progress["curve"],
        "fp32_fallback_used": progress["fp32_fallback_used"],
        "autocast": str(jev.autocast_dtype),
        "wall_clock_seconds_this_session": round(time.perf_counter() - started, 1),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    say(f"done at step {progress['step']}: best step {progress['best_step']}, selection log-likelihood {progress['best_criterion']:.4f}")
    return 0


# Main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="RLCD config, for example configs/rlcd_06b.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, for example arm=direct_log seed=1; with --resume, arm= and seed= (and run_name= if it was set) are still required to locate the run directory, and every other setting comes from the run's own config.yaml")
    parser.add_argument("--init", default=None, help="run directory holding the SFT adapter/ to start from; default rlcd.init from the config")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<run_name>/last with the run's own config.yaml")
    parser.add_argument("--env", choices=ENVS, default="deterministic", help="where outcomes come from: the gold option (deterministic, default) or the stochastic-outcome environment (noisy, decision 54); also required with --resume, since it names the run")
    parser.add_argument("--max-hours", type=float, default=8.0, help="save last/ and exit cleanly after this much wall clock (default 8.0)")
    parser.add_argument("--limit-steps", type=int, default=None, help="stop after this many optimizer steps in total, for smoke runs")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:N; default cuda when available")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--data-dir", default=None, help="default: training.data_dir from the config, relative to the repository")
    parser.add_argument("--log", default=None, help="v3 log mode: a deployment-feedback log, runs/v3_log_s<k>/log.jsonl, with configs/v3_06b.yaml; replaces sampling by the logged interactions")
    parser.add_argument("--n", type=int, default=None, help="log mode: the learner sees the first N logged interactions, trains on the first 0.9 N and selects on the last 0.1 N")
    parser.add_argument("--noisy", action="store_true", help="log mode: the outcome is the logged one flipped with probability 0.2 (the noisy condition); refused for full_sft")
    args = parser.parse_args(argv)
    if (args.n is not None or args.noisy) and args.log is None:
        parser.error("--n and --noisy are log-mode options; they need --log")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if args.log is not None:
        return main_log(args, started)
    config = load_config(args.config, args.overrides)
    config["env"] = args.env
    config["run_name"] = resolve_run_name(config)
    run_dir = Path(args.runs_dir) / config["run_name"]
    if args.resume:
        if not (run_dir / "last" / "state.pt").is_file():
            raise SystemExit(f"--resume: no saved state at {run_dir / 'last'}")
        config = load_config(run_dir / "config.yaml")
        if config.get("env", "deterministic") != args.env:
            raise SystemExit(f"--resume: {run_dir} was trained with --env {config.get('env', 'deterministic')}, not {args.env}")
    else:
        if args.env == "noisy":
            apply_noisy_defaults(config, args.overrides)
        stale = [name for name in RL_STALE_STATE if (run_dir / name).exists()]
        if stale:
            raise SystemExit(f"{run_dir} already holds a training run ({', '.join(stale)}); use --resume, or delete the directory, or another run_name")
        init_dir = Path(args.init or config["rlcd"]["init"])
        if not (init_dir / "adapter" / "adapter_model.safetensors").is_file():
            raise SystemExit(f"--init {init_dir}: no SFT adapter at {init_dir / 'adapter'}")
        config["rlcd"]["init"] = str(init_dir)
        config["rlcd"]["init_adapter_sha256"] = adapter_sha256(init_dir / "adapter")
    init_dir = Path(config["rlcd"]["init"])
    if adapter_sha256(init_dir / "adapter") != config["rlcd"]["init_adapter_sha256"]:
        raise SystemExit(f"the SFT adapter at {init_dir} is not the one this run started from (sha256 differs)")
    settings = Settings.from_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "model_id.txt").write_text(f"jevmark-{config['run_name']}\n")
    log_path = run_dir / "training_log.jsonl"
    train_cfg = config["training"]
    seed = int(config["seed"])
    max_tokens = int(train_cfg["max_tokens"])
    micro = int(train_cfg["micro_batch"])
    accumulation = max(1, int(train_cfg["effective_batch"]) // micro)
    total_steps = int(train_cfg["steps"])
    warmup = int(train_cfg["warmup_steps"])
    data_dir = Path(args.data_dir) if args.data_dir else REPO / train_cfg["data_dir"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    noisy = config.get("env", "deterministic") == "noisy"
    env = NoisyEnvironment(seed) if noisy else None
    select = "cross_entropy_theta" if noisy else "nll"  # the validation metric that picks adapter/

    jev = JevMark.load(config, device=args.device)
    say(f"run {config['run_name']}: arm {settings.arm}, env {config['env']}, backbone {config['backbone']['id']} on {jev.device}, autocast {jev.autocast_dtype}, init {init_dir}")
    train_records, dropped_train, train_lengths = fits(read_jsonl(data_dir / "train.jsonl"), jev, max_tokens)
    valid_all, dropped_valid, _ = fits(read_jsonl(data_dir / "valid.jsonl"), jev, max_tokens)
    subset_size = min(int(train_cfg["valid_subset"]), len(valid_all))
    valid_subset = [valid_all[i] for i in sorted(random.Random(f"{seed}:valid_subset").sample(range(len(valid_all)), subset_size))]

    fallback_used = False
    if not args.resume:
        first, _ = epoch_examples(train_records[: micro * 4], jev, max_tokens, seed, 0)
        if not first_batch_finite(jev, first[:micro]):
            if jev.autocast_dtype is None:
                raise RuntimeError("first-batch slot logits contain NaN or inf in fp32; not a precision problem")
            say("WARNING: NaN or inf in first-batch slot logits under fp16; reloading the model in fp32 (decision 28)")
            jev.use_fp32()
            fallback_used = True
            if not first_batch_finite(jev, first[:micro]):
                raise RuntimeError("first-batch slot logits still contain NaN or inf after the fp32 fallback")
        log_line(log_path, {"event": "start", "arm": settings.arm, "env": config["env"], "beta": settings.beta, "init": str(init_dir), "init_adapter_sha256": config["rlcd"]["init_adapter_sha256"], "fp32_fallback_used": fallback_used, "autocast": str(jev.autocast_dtype), "dropped_train": dropped_train, "dropped_valid": dropped_valid, "total_steps": total_steps, "warmup_steps": warmup, "accumulation": accumulation})

    attach_sft_adapter(jev, init_dir / "adapter", config)
    ref = load_reference(config, init_dir, jev.device, half=jev.autocast_dtype is not None)
    trainable = [p for p in jev.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=jev.device.type == "cuda" and jev.autocast_dtype == torch.float16)

    # Pre-flight: the worst-case micro-batch with this arm's loss, reference pass included (decision 45).
    def preflight_loss(model: JevMark, batch: Sequence[Any]) -> torch.Tensor:
        examples = [RLExample(e.encoded, e.targets, tuple(True for _ in e.targets)) for e in batch]
        return batch_loss(model, ref, examples, settings, torch.Generator().manual_seed(seed))[0]

    try:
        check = preflight(jev, train_records, train_lengths, micro, max_tokens, seed, scaler, preflight_loss)
    except PreflightError as err:
        log_line(log_path, {"event": "preflight", "passed": False, "error": str(err)[:500]})
        say(f"PREFLIGHT FAIL: {err}")
        raise SystemExit(f"PREFLIGHT FAIL: {config['run_name']} cannot train at micro-batch {micro}; lower training.micro_batch or turn on training.gradient_checkpointing") from err
    log_line(log_path, {"event": "preflight", "passed": True, **check})
    memory = f"peak {check['peak_gib']:.2f} GiB of {check['total_gib']:.2f} GiB ({check['peak_gib'] / check['total_gib']:.0%})" if "peak_gib" in check else "peak memory not measured on CPU"
    say(f"PREFLIGHT PASS: worst-case micro-batch of {check['records']} records ({check['shortest_tokens']} to {check['longest_tokens']} tokens, up to {check['max_questions']} questions), policy and reference: {memory}")
    if "peak_gib" in check and check["peak_gib"] > PREFLIGHT_WARN_SHARE * check["total_gib"]:
        say(f"WARNING: the pre-flight peak is above {PREFLIGHT_WARN_SHARE:.0%} of GPU memory; fragmentation during training may still run out")

    progress: dict[str, Any] = {"step": 0, "epoch": 0, "micro_done": 0, "best_nll": None, "best_step": None, "fp32_fallback_used": fallback_used, "dropped_train": dropped_train, "dropped_valid": dropped_valid, "init_valid": None}
    if args.resume:
        progress = load_state(run_dir, jev, optimizer, scheduler, scaler)
        generator.set_state(progress["sampler_state"])
        if env is not None:
            env.set_state(progress["env_state"])
        log_line(log_path, {"event": "resume", "step": progress["step"], "epoch": progress["epoch"]})
        say(f"resumed at step {progress['step']}, epoch {progress['epoch']}, micro-batch {progress['micro_done']}")
    else:
        init_valid = validate(jev, ref, valid_subset, micro * 2, max_tokens, settings.arm, noisy)
        progress["init_valid"] = init_valid
        log_line(log_path, {"event": "valid", "step": 0, **init_valid})
        say(f"step 0 (the SFT adapter): valid acc {init_valid['accuracy']:.4f} ece {init_valid['ece']:.4f} nll {init_valid['nll']:.4f}" + (f" ce(theta) {init_valid['cross_entropy_theta']:.4f}" if noisy else ""))
    say(f"{len(train_records)} records, {total_steps} optimizer steps, micro-batch {micro} x {accumulation}, warmup {warmup}, arm {settings.arm}")

    limit = min(total_steps, args.limit_steps) if args.limit_steps else total_steps
    deadline = started + args.max_hours * 3600

    def checkpoint() -> None:
        progress["sampler_state"] = generator.get_state()
        if env is not None:
            progress["env_state"] = env.get_state()
        save_state(run_dir, jev, optimizer, scheduler, scaler, progress)

    def run_validation(step: int) -> None:
        result = validate(jev, ref, valid_subset, micro * 2, max_tokens, settings.arm, noisy)
        log_line(log_path, {"event": "valid", "step": step, **result})
        say(f"step {step}: valid acc {result['accuracy']:.4f} ece {result['ece']:.4f} nll {result['nll']:.4f} kl {result['kl_to_ref']:.4f}" + (f" ce(theta) {result['cross_entropy_theta']:.4f}" if noisy else ""))
        if progress["best_nll"] is None or result[select] < progress["best_nll"]:  # best_nll holds the best value of `select`
            progress["best_nll"], progress["best_step"] = result[select], step
            jev.model.save_pretrained(run_dir / "adapter")

    stopped_by_time = False
    last_validated = None
    while progress["step"] < limit:
        examples, reshuffle_drops = epoch_examples(train_records, jev, max_tokens, seed, progress["epoch"])
        batches = [examples[i : i + micro] for i in range(0, len(examples), micro)]
        groups = [batches[i : i + accumulation] for i in range(0, len(batches), accumulation)]
        for group in groups[progress["micro_done"] // accumulation :]:
            if progress["step"] >= limit:
                break
            losses, stats = [], []
            for batch in group:
                loss, stat = batch_loss(jev, ref, batch, settings, generator, env)
                scaler.scale(loss / len(group)).backward()
                losses.append(loss.item())
                stats.append(stat)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            progress["step"] += 1
            progress["micro_done"] += len(group)
            mean = {k: sum(s[k] for s in stats) / len(stats) for k in ("kl", "reward", "p_chosen")}
            log_line(log_path, {"step": progress["step"], "epoch": progress["epoch"], "loss": sum(losses) / len(losses), **mean, "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.perf_counter() - started, 1)})
            at_end = progress["step"] == limit
            if progress["step"] % int(train_cfg["eval_every"]) == 0 or at_end:
                run_validation(progress["step"])
                last_validated = progress["step"]
                checkpoint()
            if time.perf_counter() > deadline and not at_end:
                stopped_by_time = True
                break
        if stopped_by_time:
            break
        if progress["micro_done"] >= len(batches):
            progress["epoch"] += 1
            progress["micro_done"] = 0
            if reshuffle_drops:
                log_line(log_path, {"event": "reshuffle_drops", "epoch": progress["epoch"] - 1, "dropped": reshuffle_drops})

    if last_validated != progress["step"] and progress["step"] > 0 and not stopped_by_time:
        run_validation(progress["step"])
    checkpoint()
    if stopped_by_time:
        log_line(log_path, {"event": "time_limit", "step": progress["step"], "max_hours": args.max_hours})
        say(f"time limit of {args.max_hours} h reached at step {progress['step']}; saved {run_dir / 'last'}; continue with --resume")
        return 0
    if progress["best_step"] is None:
        raise RuntimeError("no validation ran, so no best adapter exists")

    jev.model.save_pretrained(run_dir / "adapter_last")
    final_last = validate(jev, ref, valid_all, micro * 2, max_tokens, settings.arm, noisy)
    load_adapter_weights(jev, run_dir / "adapter")
    final = validate(jev, ref, valid_all, micro * 2, max_tokens, settings.arm, noisy)
    log_line(log_path, {"event": "final_valid", "best_step": progress["best_step"], **final})
    log_line(log_path, {"event": "final_valid_last", "step": progress["step"], **final_last})
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 1.0}) + "\n")
    summary = {
        "run_name": config["run_name"],
        "arm": settings.arm,
        "env": config["env"],
        "beta": settings.beta,
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "steps": progress["step"],
        "total_steps": total_steps,
        "limit_steps": args.limit_steps,
        "best_step": progress["best_step"],
        "best_by": select,
        f"best_subset_{select}": progress["best_nll"],
        "init": str(init_dir),
        "init_adapter_sha256": config["rlcd"]["init_adapter_sha256"],
        "init_valid": progress["init_valid"],
        "final_valid": final,
        "final_valid_last": final_last,
        "dropped_train": progress["dropped_train"],
        "dropped_valid": progress["dropped_valid"],
        "fp32_fallback_used": progress["fp32_fallback_used"],
        "autocast": str(jev.autocast_dtype),
        "wall_clock_seconds_this_session": round(time.perf_counter() - started, 1),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    say(f"done at step {progress['step']}: best step {progress['best_step']}; full valid (best) acc {final['accuracy']:.4f} ece {final['ece']:.4f} nll {final['nll']:.4f}; (last) nll {final_last['nll']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
