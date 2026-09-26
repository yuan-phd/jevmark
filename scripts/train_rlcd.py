"""RLCD: bandit training of the letter readout from the SFT adapter (docs/TASKS.md 2.2).

    python scripts/train_rlcd.py --config configs/rlcd_06b.yaml --init runs/sft_06b arm=brier seed=0
    python scripts/train_rlcd.py --config configs/rlcd_06b.yaml arm=brier seed=0 --resume

The trainable LoRA starts as the SFT adapter of --init (a run directory holding
adapter/); a second copy of the backbone with that adapter merged is the frozen
reference policy p_ref. The sha256 of the SFT adapter's weights is recorded in the
run's config.yaml and train_summary.json.

Every optimizer step takes an effective batch of training records (the same seeded
order and per-epoch option reshuffles as train_sft.py). For every gold-dependent
question (form nouls are excluded from sampling and from the loss, decision 44):

    p = softmax(letter logits), q = (1 - epsilon) p + epsilon / K
    G actions a ~ q; only r_a = 1[a is gold] is revealed, for sampled actions only

and the arm decides the loss:

- outcome, outcome_minus_p, brier, log (policy gradient): the reward R(a) is
  r_a, r_a - p_a, 1 - (r_a - p_a)^2, or r_a log p_a + (1 - r_a) log(1 - p_a) with p
  clipped to [1e-6, 1 - 1e-6]; rewards use p detached. The advantage is R(a) minus
  the mean over the G samples of that question (divided by their standard deviation
  when rlcd.normalize_std), the importance weight w = p(a) / q(a) clipped at
  rlcd.importance_clip corrects for sampling from q, and the loss is
  mean over samples of -A w log p(a), plus beta KL(p || p_ref) summed over options.
- direct_bandit: no policy gradient; the loss is the mean over the same samples of
  (r_a - p_a)^2 with p_a differentiable (the negative Brier score of the sampled
  action), plus beta KL(p || p_ref).
- sft_cont: the control for extra training: cross-entropy on the gold option of the
  same gold-dependent questions, same records, steps, learning rate and schedule, no
  sampling and no KL term.

The loss is the mean over the micro-batch's gold-dependent questions. Sampling uses
its own torch generator, seeded from the config seed and saved with the resume state.

Validation every training.eval_every steps (and at step 0, the SFT adapter itself,
logged as the reference) on the same seeded 1000-record subset of valid as
train_sft.py, gold-dependent questions only: accuracy, ECE, NLL, Brier, mean
KL(p || p_ref), the mean expected reward under p (sum over a of p_a R(a); the Brier
reward for direct_bandit and sft_cont) and the mean expected probability of the
chosen action (sum of p_a squared). The best adapter by validation NLL among the
trained steps goes to adapter/, the final one to adapter_last/; the final full-valid
pass reports both. Pre-flight memory check, stale-state guard, --max-hours and
--resume behave as in train_sft.py (decision 45); the first-batch NaN check runs on
the backbone before the adapter is attached.

Run directory runs/<run_name>/ (run_name rlcd_<size>_<arm>_s<seed> unless set):
config.yaml, model_id.txt, training_log.jsonl, adapter/, adapter_last/, last/,
train_summary.json, calibration.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import get_linear_schedule_with_warmup

from jevmark.config import load_config
from jevmark.data.form import FORM_KINDS
from jevmark.encode import Encoded, encode
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
PG_ARMS = ("outcome", "outcome_minus_p", "brier", "log")
ARMS = (*PG_ARMS, "direct_bandit", "sft_cont")
RL_STALE_STATE = (*STALE_STATE, "adapter_last")
P_CLIP = 1e-6


# Rewards, advantages, weights


def reward(arm: str, r: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """R(a) for revealed outcomes r (1 when the sampled action is gold) and the policy's probabilities p_a of those actions."""
    if arm == "outcome":
        return r
    if arm == "outcome_minus_p":
        return r - p_a
    if arm in ("brier", "direct_bandit", "sft_cont"):
        return 1.0 - (r - p_a) ** 2
    if arm == "log":
        p = p_a.clamp(P_CLIP, 1.0 - P_CLIP)
        return r * torch.log(p) + (1.0 - r) * torch.log(1.0 - p)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


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
        if arm not in ARMS:
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
    if s.arm == "direct_bandit":
        loss = ((r - p[actions]) ** 2).mean() + s.beta * kl
        rewards = reward("brier", r, p_a)
    else:
        rewards = reward(s.arm, r, p_a)
        weight = importance_weights(p_a, q[actions], s.importance_clip) if s.importance_weight else torch.ones_like(p_a)
        loss = -(advantages(rewards, s.normalize_std) * weight * logp[actions]).mean() + s.beta * kl
    return loss, {"kl": float(kl.detach()), "reward": float(rewards.mean()), "p_chosen": float(p_a.mean())}


def batch_loss(jev: JevMark, ref: JevMark, batch: Sequence[RLExample], s: Settings, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean loss over the micro-batch's gold-dependent questions, and the mean of their statistics."""
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
            loss, stat = question_loss(z, z_ref, target, s, generator)
            losses.append(loss)
            stats.append(stat)
    if not losses:
        raise RuntimeError("a micro-batch without gold-dependent questions")
    mean = {k: sum(x[k] for x in stats) / len(stats) for k in ("kl", "reward", "p_chosen")}
    return torch.stack(losses).mean(), mean


# Validation


def validate(jev: JevMark, ref: JevMark, records: list[dict[str, Any]], batch_size: int, max_tokens: int, arm: str) -> dict[str, Any]:
    """Accuracy, ECE, NLL, Brier, mean KL to p_ref, expected reward and expected p of the chosen action, on gold-dependent questions."""
    was_training = jev.model.training
    jev.model.eval()
    results, kls, rewards, chosen = [], [], [], []
    reward_arm = arm if arm in PG_ARMS else "brier"
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
                    outcomes = (torch.arange(len(p), device=p.device) == target).float()
                    rewards.append(float((p * reward(reward_arm, outcomes, p)).sum()))
                    chosen.append(float((p * p).sum()))
                    probs = tuple(float(x) for x in p.double().cpu())
                    labels = question.labels if isinstance(question, ChoiceQuestion) else tuple(str(i) for i in range(len(probs)))
                    results.append(QuestionResult(record["id"], question.id, question.type, probs, target, labels, kind=question.id if question.type == "noul" else None))
    if was_training:
        jev.model.train()
    overall = split_metrics(results)["overall"]
    n = len(kls)
    return {
        "n": overall["n"],
        "accuracy": overall["accuracy"],
        "ece": overall["ece"],
        "nll": overall["nll"],
        "brier": overall["brier"],
        "kl_to_ref": sum(kls) / n,
        "expected_reward": sum(rewards) / n,
        "expected_p_chosen": sum(chosen) / n,
    }


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
    return config.get("run_name") or f"rlcd_{config['size']}_{config['arm']}_s{config['seed']}"


# Main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="RLCD config, for example configs/rlcd_06b.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, for example arm=log seed=1 (ignored with --resume, which uses the run's config.yaml)")
    parser.add_argument("--init", default=None, help="run directory holding the SFT adapter/ to start from; default rlcd.init from the config")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<run_name>/last with the run's own config.yaml")
    parser.add_argument("--max-hours", type=float, default=8.0, help="save last/ and exit cleanly after this much wall clock (default 8.0)")
    parser.add_argument("--limit-steps", type=int, default=None, help="stop after this many optimizer steps in total, for smoke runs")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:N; default cuda when available")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--data-dir", default=None, help="default: training.data_dir from the config, relative to the repository")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    config = load_config(args.config, args.overrides)
    config["run_name"] = resolve_run_name(config)
    run_dir = Path(args.runs_dir) / config["run_name"]
    if args.resume:
        if not (run_dir / "last" / "state.pt").is_file():
            raise SystemExit(f"--resume: no saved state at {run_dir / 'last'}")
        config = load_config(run_dir / "config.yaml")
    else:
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

    jev = JevMark.load(config, device=args.device)
    say(f"run {config['run_name']}: arm {settings.arm}, backbone {config['backbone']['id']} on {jev.device}, autocast {jev.autocast_dtype}, init {init_dir}")
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
        log_line(log_path, {"event": "start", "arm": settings.arm, "init": str(init_dir), "init_adapter_sha256": config["rlcd"]["init_adapter_sha256"], "fp32_fallback_used": fallback_used, "autocast": str(jev.autocast_dtype), "dropped_train": dropped_train, "dropped_valid": dropped_valid, "total_steps": total_steps, "warmup_steps": warmup, "accumulation": accumulation})

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
        log_line(log_path, {"event": "resume", "step": progress["step"], "epoch": progress["epoch"]})
        say(f"resumed at step {progress['step']}, epoch {progress['epoch']}, micro-batch {progress['micro_done']}")
    else:
        init_valid = validate(jev, ref, valid_subset, micro * 2, max_tokens, settings.arm)
        progress["init_valid"] = init_valid
        log_line(log_path, {"event": "valid", "step": 0, **init_valid})
        say(f"step 0 (the SFT adapter): valid acc {init_valid['accuracy']:.4f} ece {init_valid['ece']:.4f} nll {init_valid['nll']:.4f}")
    say(f"{len(train_records)} records, {total_steps} optimizer steps, micro-batch {micro} x {accumulation}, warmup {warmup}, arm {settings.arm}")

    limit = min(total_steps, args.limit_steps) if args.limit_steps else total_steps
    deadline = started + args.max_hours * 3600

    def checkpoint() -> None:
        progress["sampler_state"] = generator.get_state()
        save_state(run_dir, jev, optimizer, scheduler, scaler, progress)

    def run_validation(step: int) -> None:
        result = validate(jev, ref, valid_subset, micro * 2, max_tokens, settings.arm)
        log_line(log_path, {"event": "valid", "step": step, **result})
        say(f"step {step}: valid acc {result['accuracy']:.4f} ece {result['ece']:.4f} nll {result['nll']:.4f} kl {result['kl_to_ref']:.4f}")
        if progress["best_nll"] is None or result["nll"] < progress["best_nll"]:
            progress["best_nll"], progress["best_step"] = result["nll"], step
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
                loss, stat = batch_loss(jev, ref, batch, settings, generator)
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
    final_last = validate(jev, ref, valid_all, micro * 2, max_tokens, settings.arm)
    load_adapter_weights(jev, run_dir / "adapter")
    final = validate(jev, ref, valid_all, micro * 2, max_tokens, settings.arm)
    log_line(log_path, {"event": "final_valid", "best_step": progress["best_step"], **final})
    log_line(log_path, {"event": "final_valid_last", "step": progress["step"], **final_last})
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 1.0}) + "\n")
    summary = {
        "run_name": config["run_name"],
        "arm": settings.arm,
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "steps": progress["step"],
        "total_steps": total_steps,
        "limit_steps": args.limit_steps,
        "best_step": progress["best_step"],
        "best_subset_nll": progress["best_nll"],
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
