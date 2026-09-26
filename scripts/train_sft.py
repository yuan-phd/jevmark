"""LoRA supervised fine-tuning of the letter readout (docs/TASKS.md 1.7).

    python scripts/train_sft.py --config configs/sft_06b.yaml [key=value ...]
    python scripts/train_sft.py --config configs/sft_06b.yaml --resume

Loss: mean cross-entropy over every answer slot in a micro-batch, on the fp32
letter logits from JevMark.slot_logits. Every epoch reshuffles choice options
(encode.shuffle_request, seeded per epoch) and re-derives each gold index from
its gold label. Records that encode longer than training.max_tokens are dropped
and counted, never truncated.

Run directory runs/<run_name>/:
  config.yaml, model_id.txt   written at the start (the complete merged config)
  training_log.jsonl          one line per optimizer step, per validation and per event
  adapter/                    best adapter by validation NLL (gitignored)
  last/                       latest adapter plus optimizer, scheduler, scaler, RNG, step (gitignored)
  train_summary.json          best step, final full-valid metrics, drops, precision, time
  calibration.json            temperature 1.0 until task 2.1 fits one

Before the first step, a pre-flight check runs one forward and backward pass on
the worst-case micro-batch (the longest training records) and reports peak GPU
memory; a CUDA out-of-memory error there ends the run within a minute with exit
code 1 and "PREFLIGHT FAIL" (decision 45).

The first batch's slot logits are checked for NaN or inf before LoRA is attached
(a fresh LoRA has B = 0 and changes nothing), and use_fp32() reloads the model
in fp32 on failure (decision 28). --max-hours saves last/ and exits cleanly;
--resume continues from last/ in the same run directory. Without --resume the run
refuses to start if the run directory holds train_summary.json,
training_log.jsonl, adapter/ or last/: a summary left by an earlier run, for
example one committed to git, must never pass for this run's result (decision 45).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from jevmark.config import load_config
from jevmark.encode import encode
from jevmark.metrics import QuestionResult, split_metrics
from jevmark.model import JevMark, keep_lora_fp32
from jevmark.schema import ChoiceQuestion, Request
from jevmark.training import (  # noqa: F401 - re-exported for the tests and for continuity
    STALE_STATE,
    Example,
    PREFLIGHT_WARN_SHARE,
    PreflightError,
    epoch_examples,
    fits,
    load_adapter_weights,
    load_state,
    log_line,
    micro_batches,
    prepare,
    read_jsonl,
    save_state,
    say,
    targets_for,
)
from jevmark.training import preflight as _preflight

REPO = Path(__file__).resolve().parents[1]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


# Model steps


def batch_loss(jev: JevMark, batch: Sequence[Example]) -> torch.Tensor:
    """Mean cross-entropy over every slot in the micro-batch, on fp32 letter logits."""
    logits = jev.slot_logits([e.encoded for e in batch])
    targets = [t for e in batch for t in e.targets]
    losses = [F.cross_entropy(x.unsqueeze(0), torch.tensor([t], device=x.device)) for x, t in zip(logits, targets)]
    return torch.stack(losses).mean()


def preflight(jev: JevMark, records: list[dict[str, Any]], lengths: list[int], micro: int, max_tokens: int, seed: int, scaler) -> dict[str, Any]:
    """jevmark.training.preflight with the SFT loss (decision 45)."""
    return _preflight(jev, records, lengths, micro, max_tokens, seed, scaler, batch_loss)


def first_batch_finite(jev: JevMark, batch: Sequence[Example]) -> bool:
    with torch.no_grad():
        return all(torch.isfinite(x).all().item() for x in jev.slot_logits([e.encoded for e in batch]))


def validate(jev: JevMark, records: list[dict[str, Any]], batch_size: int, max_tokens: int) -> dict[str, Any]:
    """Accuracy, ECE and NLL (and Brier) over records in their stored option order."""
    was_training = jev.model.training
    jev.model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            requests = [Request.from_dict({"state": r["state"], "questions": r["questions"]}) for r in chunk]
            flat = iter(jev.forward_distributions([encode(q, jev.tokenizer, max_tokens) for q in requests]))
            for record, request in zip(chunk, requests):
                for question, target in zip(request.questions, targets_for(request, record["gold"])):
                    probs = tuple(float(p) for p in next(flat).double().cpu())
                    labels = question.labels if isinstance(question, ChoiceQuestion) else tuple(str(i) for i in range(len(probs)))
                    results.append(QuestionResult(record["id"], question.id, question.type, probs, target, labels))
    if was_training:
        jev.model.train()
    overall = split_metrics(results)["overall"]
    return {"n": overall["n"], "accuracy": overall["accuracy"], "ece": overall["ece"], "nll": overall["nll"], "brier": overall["brier"]}


# LoRA


def attach_lora(jev: JevMark, config: dict[str, Any]) -> None:
    lora = config["lora"]
    targets = list(lora["target_modules"]) + (MLP_MODULES if lora.get("mlp") else [])
    peft_config = LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]), target_modules=targets, bias="none")
    jev.model = get_peft_model(jev.model, peft_config)
    keep_lora_fp32(jev.model)
    if config["training"].get("gradient_checkpointing"):
        jev.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        jev.model.enable_input_require_grads()
    jev.model.train()


# Main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="training config, for example configs/sft_06b.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, for example run_name=sft_06b_smoke (ignored with --resume, which uses the run's config.yaml)")
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
    run_dir = Path(args.runs_dir) / config["run_name"]
    if args.resume:
        if not (run_dir / "last" / "state.pt").is_file():
            raise SystemExit(f"--resume: no saved state at {run_dir / 'last'}")
        config = load_config(run_dir / "config.yaml")
    else:
        # Any trace of an earlier run, including files committed to git, means this is not a fresh start (decision 45).
        stale = [name for name in STALE_STATE if (run_dir / name).exists()]
        if stale:
            raise SystemExit(f"{run_dir} already holds a training run ({', '.join(stale)}); use --resume, or delete the directory, or another run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "model_id.txt").write_text(f"jevmark-{config['run_name']}\n")
    log_path = run_dir / "training_log.jsonl"
    train_cfg = config["training"]
    seed = int(config["seed"])
    max_tokens = int(train_cfg["max_tokens"])
    micro = int(train_cfg["micro_batch"])
    accumulation = max(1, int(train_cfg["effective_batch"]) // micro)
    data_dir = Path(args.data_dir) if args.data_dir else REPO / train_cfg["data_dir"]
    torch.manual_seed(seed)

    jev = JevMark.load(config, device=args.device)
    say(f"run {config['run_name']}: backbone {config['backbone']['id']} on {jev.device}, autocast {jev.autocast_dtype}")
    train_records, dropped_train, train_lengths = fits(read_jsonl(data_dir / "train.jsonl"), jev, max_tokens)
    valid_all, dropped_valid, _ = fits(read_jsonl(data_dir / "valid.jsonl"), jev, max_tokens)
    say(f"dropped over {max_tokens} tokens: train {dropped_train}, valid {dropped_valid}; kept train {len(train_records)}, valid {len(valid_all)}")
    subset_size = min(int(train_cfg["valid_subset"]), len(valid_all))
    valid_subset = [valid_all[i] for i in sorted(random.Random(f"{seed}:valid_subset").sample(range(len(valid_all)), subset_size))]

    epochs = int(train_cfg["epochs"])
    n_micro = math.ceil(len(train_records) / micro)
    steps_per_epoch = math.ceil(n_micro / accumulation)
    total_steps = epochs * steps_per_epoch
    warmup = math.ceil(float(train_cfg["warmup_ratio"]) * total_steps)

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
        log_line(log_path, {"event": "start", "fp32_fallback_used": fallback_used, "autocast": str(jev.autocast_dtype), "dropped_train": dropped_train, "dropped_valid": dropped_valid, "total_steps": total_steps, "warmup_steps": warmup, "accumulation": accumulation})

    attach_lora(jev, config)
    trainable = [p for p in jev.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    use_scaler = jev.device.type == "cuda" and jev.autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Pre-flight: the worst-case micro-batch, forward and backward, before any step (decision 45).
    try:
        check = preflight(jev, train_records, train_lengths, micro, max_tokens, seed, scaler)
    except PreflightError as err:
        log_line(log_path, {"event": "preflight", "passed": False, "error": str(err)[:500]})
        say(f"PREFLIGHT FAIL: {err}")
        raise SystemExit(f"PREFLIGHT FAIL: {config['run_name']} cannot train at micro-batch {micro}; lower training.micro_batch (the effective batch is kept by accumulation) or turn on training.gradient_checkpointing") from err
    log_line(log_path, {"event": "preflight", "passed": True, **check})
    memory = f"peak {check['peak_gib']:.2f} GiB of {check['total_gib']:.2f} GiB ({check['peak_gib'] / check['total_gib']:.0%})" if "peak_gib" in check else "peak memory not measured on CPU"
    say(f"PREFLIGHT PASS: worst-case micro-batch of {check['records']} records ({check['shortest_tokens']} to {check['longest_tokens']} tokens, up to {check['max_questions']} questions): {memory}")
    if "peak_gib" in check and check["peak_gib"] > PREFLIGHT_WARN_SHARE * check["total_gib"]:
        say(f"WARNING: the pre-flight peak is above {PREFLIGHT_WARN_SHARE:.0%} of GPU memory; fragmentation during training may still run out")

    progress = {"step": 0, "epoch": 0, "micro_done": 0, "best_nll": None, "best_step": None, "fp32_fallback_used": fallback_used, "dropped_train": dropped_train, "dropped_valid": dropped_valid}
    if args.resume:
        progress = load_state(run_dir, jev, optimizer, scheduler, scaler)
        log_line(log_path, {"event": "resume", "step": progress["step"], "epoch": progress["epoch"]})
        say(f"resumed at step {progress['step']}, epoch {progress['epoch']}, micro-batch {progress['micro_done']}")
    say(f"{len(train_records)} records, {total_steps} optimizer steps ({steps_per_epoch} per epoch), micro-batch {micro} x {accumulation}, warmup {warmup}")

    limit = min(total_steps, args.limit_steps) if args.limit_steps else total_steps
    deadline = started + args.max_hours * 3600

    def run_validation(step: int) -> None:
        result = validate(jev, valid_subset, micro * 2, max_tokens)
        log_line(log_path, {"event": "valid", "step": step, **result})
        say(f"step {step}: valid acc {result['accuracy']:.4f} ece {result['ece']:.4f} nll {result['nll']:.4f}")
        if progress["best_nll"] is None or result["nll"] < progress["best_nll"]:
            progress["best_nll"], progress["best_step"] = result["nll"], step
            jev.model.save_pretrained(run_dir / "adapter")

    stopped_by_time = False
    last_validated = None
    while progress["epoch"] < epochs and progress["step"] < limit:
        examples, reshuffle_drops = epoch_examples(train_records, jev, max_tokens, seed, progress["epoch"])
        batches = micro_batches(examples, micro)
        groups = [batches[i : i + accumulation] for i in range(0, len(batches), accumulation)]
        group_index = progress["micro_done"] // accumulation
        for group in groups[group_index:]:
            if progress["step"] >= limit:
                break
            losses = []
            for batch in group:
                loss = batch_loss(jev, batch)
                scaler.scale(loss / len(group)).backward()
                losses.append(loss.item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            progress["step"] += 1
            progress["micro_done"] += len(group)
            log_line(log_path, {"step": progress["step"], "epoch": progress["epoch"], "loss": sum(losses) / len(losses), "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.perf_counter() - started, 1)})
            at_end = progress["step"] == limit or (progress["epoch"] == epochs - 1 and progress["micro_done"] >= len(batches))
            if progress["step"] % int(train_cfg["eval_every"]) == 0 or at_end:
                run_validation(progress["step"])
                last_validated = progress["step"]
                save_state(run_dir, jev, optimizer, scheduler, scaler, progress)
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
    save_state(run_dir, jev, optimizer, scheduler, scaler, progress)
    if stopped_by_time:
        log_line(log_path, {"event": "time_limit", "step": progress["step"], "max_hours": args.max_hours})
        say(f"time limit of {args.max_hours} h reached at step {progress['step']}; saved {run_dir / 'last'}; continue with --resume")
        return 0

    if progress["best_step"] is None:
        raise RuntimeError("no validation ran, so no best adapter exists")
    load_adapter_weights(jev, run_dir / "adapter")
    final = validate(jev, valid_all, micro * 2, max_tokens)
    log_line(log_path, {"event": "final_valid", "best_step": progress["best_step"], **final})
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 1.0}) + "\n")
    summary = {
        "run_name": config["run_name"],
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "steps": progress["step"],
        "total_steps": total_steps,
        "limit_steps": args.limit_steps,
        "best_step": progress["best_step"],
        "best_subset_nll": progress["best_nll"],
        "final_valid": final,
        "dropped_train": progress["dropped_train"],
        "dropped_valid": progress["dropped_valid"],
        "fp32_fallback_used": progress["fp32_fallback_used"],
        "autocast": str(jev.autocast_dtype),
        "wall_clock_seconds_this_session": round(time.perf_counter() - started, 1),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    say(f"done at step {progress['step']}: best step {progress['best_step']}; full valid acc {final['accuracy']:.4f} ece {final['ece']:.4f} nll {final['nll']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
