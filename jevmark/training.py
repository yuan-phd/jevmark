"""Training helpers shared by scripts/train_sft.py and scripts/train_rlcd.py.

Data preparation (per-epoch seeded order and option reshuffles), the record length
filter, the pre-flight memory check (decision 45), and last/ state saving and
loading for --resume. Each training script supplies its own loss.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from jevmark.encode import Encoded, encode, shuffle_request
from jevmark.model import JevMark, keep_lora_fp32
from jevmark.schema import ChoiceQuestion, NoulQuestion, Request

# Files and directories a training run leaves; any of them in the run directory blocks a fresh start.
STALE_STATE = ("train_summary.json", "training_log.jsonl", "adapter", "last")


def log_line(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def say(message: str) -> None:
    print(message, flush=True)


# Data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def targets_for(request: Request, gold: dict[str, Any]) -> list[int]:
    """Gold option index per question, re-derived from the gold label in the request's current option order."""
    targets = []
    for question in request.questions:
        answer = gold[question.id]
        if isinstance(question, ChoiceQuestion):
            targets.append(question.labels.index(answer))
        elif isinstance(question, NoulQuestion):
            targets.append({"true": 0, "false": 1}[answer])
        else:
            targets.append(int(answer))
    return targets


def prepare(record: dict[str, Any], rng: random.Random) -> tuple[Request, list[int]]:
    """The record's request with choice options reshuffled, and its targets."""
    request = Request.from_dict({"state": record["state"], "questions": record["questions"]})
    shuffled, _ = shuffle_request(request, rng)
    return shuffled, targets_for(shuffled, record["gold"])


@dataclass(frozen=True)
class Example:
    encoded: Encoded
    targets: tuple[int, ...]


def fits(records: list[dict[str, Any]], jev: JevMark, max_tokens: int) -> tuple[list[dict[str, Any]], int, list[int]]:
    """Records whose stored encoding fits max_tokens, how many were dropped, and the kept records' encoded lengths."""
    kept, lengths = [], []
    for record in records:
        try:
            encoded = encode(Request.from_dict({"state": record["state"], "questions": record["questions"]}), jev.tokenizer, max_tokens)
        except ValueError:
            continue
        kept.append(record)
        lengths.append(len(encoded.input_ids))
    return kept, len(records) - len(kept), lengths


def epoch_examples(records: list[dict[str, Any]], jev: JevMark, max_tokens: int, seed: int, epoch: int) -> tuple[list[Example], int]:
    """The epoch's examples in a seeded order with seeded option shuffles; the same for a given (seed, epoch)."""
    rng = random.Random(f"{seed}:epoch{epoch}")
    order = list(range(len(records)))
    rng.shuffle(order)
    examples, dropped = [], 0
    for index in order:
        request, targets = prepare(records[index], rng)
        try:
            encoded = encode(request, jev.tokenizer, max_tokens)
        except ValueError:  # a reshuffle made it one token longer than the limit
            dropped += 1
            continue
        examples.append(Example(encoded, tuple(targets)))
    return examples, dropped


def micro_batches(examples: list[Example], size: int) -> list[list[Example]]:
    return [examples[i : i + size] for i in range(0, len(examples), size)]


# Pre-flight (decision 45)

PREFLIGHT_WARN_SHARE = 0.9


def preflight(
    jev: JevMark,
    records: list[dict[str, Any]],
    lengths: list[int],
    micro: int,
    max_tokens: int,
    seed: int,
    scaler,
    loss_fn: Callable[[JevMark, Sequence[Example]], torch.Tensor],
) -> dict[str, Any]:
    """One forward and backward pass on the worst-case micro-batch, before any training step.

    The worst case is the `micro` longest training records by encoded length (ties
    broken by the number of questions), run in training mode with gradients under the
    run's own autocast and GradScaler with the run's own loss (loss_fn), exactly as a training step does, but with no
    optimizer step. Gradients are dropped afterwards and the torch random state is
    restored, so the run that follows is the same as without the check. Returns the
    batch's size and lengths and, on CUDA, the peak memory allocated and the device's
    total. A CUDA out-of-memory error ends the run with a message saying so.
    """
    order = sorted(range(len(records)), key=lambda i: (lengths[i], len(records[i]["questions"])), reverse=True)[:micro]
    rng = random.Random(f"{seed}:preflight")
    batch = []
    for i in order:
        request, targets = prepare(records[i], rng)
        try:
            encoded = encode(request, jev.tokenizer, max_tokens)
        except ValueError:  # the reshuffle added a token past the limit; the stored order fits (fits() checked it)
            request = Request.from_dict({"state": records[i]["state"], "questions": records[i]["questions"]})
            targets = targets_for(request, records[i]["gold"])
            encoded = encode(request, jev.tokenizer, max_tokens)
        batch.append(Example(encoded, tuple(targets)))
    cuda = jev.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(jev.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(jev.device)
    was_training = jev.model.training
    jev.model.train()
    try:
        with torch.random.fork_rng(devices=[jev.device] if cuda else []):
            loss = loss_fn(jev, batch)
            scaler.scale(loss).backward()
    except torch.cuda.OutOfMemoryError as err:
        raise PreflightError(f"CUDA out of memory on the worst-case micro-batch of {len(batch)} records (longest {max(len(e.encoded.input_ids) for e in batch)} tokens): {err}") from err
    finally:
        for param in jev.model.parameters():
            param.grad = None
        if not was_training:
            jev.model.eval()
    report: dict[str, Any] = {
        "records": len(batch),
        "longest_tokens": max(len(e.encoded.input_ids) for e in batch),
        "shortest_tokens": min(len(e.encoded.input_ids) for e in batch),
        "max_questions": max(len(e.targets) for e in batch),
        "worst_case_loss": float(loss.detach()),
        "device": str(jev.device),
    }
    if cuda:
        torch.cuda.synchronize(jev.device)
        report["peak_gib"] = torch.cuda.max_memory_allocated(jev.device) / 2**30
        report["total_gib"] = torch.cuda.get_device_properties(jev.device).total_memory / 2**30
        torch.cuda.empty_cache()
    return report


class PreflightError(RuntimeError):
    pass


# State


def save_state(run_dir: Path, jev: JevMark, optimizer, scheduler, scaler, progress: dict[str, Any]) -> None:
    """Write last/ atomically: adapter, optimizer, scheduler, scaler, torch RNG and progress."""
    tmp = run_dir / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    jev.model.save_pretrained(tmp / "adapter")
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "progress": progress,
    }
    torch.save(state, tmp / "state.pt")
    shutil.rmtree(run_dir / "last", ignore_errors=True)
    tmp.rename(run_dir / "last")


def load_adapter_weights(jev: JevMark, adapter_dir: Path) -> None:
    set_peft_model_state_dict(jev.model, load_file(str(adapter_dir / "adapter_model.safetensors")))
    keep_lora_fp32(jev.model)


def load_state(run_dir: Path, jev: JevMark, optimizer, scheduler, scaler) -> dict[str, Any]:
    load_adapter_weights(jev, run_dir / "last" / "adapter")
    state = torch.load(run_dir / "last" / "state.pt", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]
