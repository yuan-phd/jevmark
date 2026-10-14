"""Collect a v3 deployment-feedback log (task 3.2, decision 56, docs/V3_DESIGN.md section 4).

    python scripts/collect_log.py --ckpt runs/sft_06b --seed 0 --device cuda
    python scripts/collect_log.py --ckpt runs/sft_06b --seed 1 --device cuda   # logging variance only
    python scripts/collect_log.py --ckpt runs/sft_17b --seed 0 --device cuda   # the 1.7B log, runs/v3_log_17b_s0
    python scripts/collect_log.py --ckpt runs/sft_06b --seed 0 --domain emotion --device cuda   # runs/v3_log_emotion_s0

The checkpoint (the logging policy, sft_06b in v3) reads every record of
data/v3_banking77_train.jsonl once, in a fixed order shuffled by
random.Random(f"v3_log_order:{order_seed}") (the same order for every log seed,
since the file is in dataset order, which is grouped by label). For each message it
samples one action from 0.9 p + 0.1 / K and reveals only whether that option is
gold; the noisy condition's outcome is the revealed one flipped with probability 0.2
(jevmark/feedback.py). Distributions are the model's unrounded probabilities with its
own calibration.json temperature, read in one forward pass per batch; the first
batch is checked for NaN or inf under fp16, with a reload in fp32 on failure.

Writes runs/v3_log_s<seed>/log.jsonl for a 0.6B logging policy, or
runs/v3_log_<size>_s<seed>/log.jsonl for another size read from the checkpoint's name
(runs/sft_17b writes v3_log_17b_s<seed>), so logs of different policies never share a
directory (gitignored, keep a copy: every learner trains on it) and metrics.json with the logging policy's accuracy (argmax against gold),
outcome and flipped-outcome rates, predicted-other and chosen-other rates, the data
file's sha256 (refused if it differs from configs/v3_data.yaml), the log's sha256,
the commit and precision. An existing log is never overwritten.

--domain emotion (phase 4) reads data/v3_emotion_train.jsonl, the `emotion` block of
configs/v3_data.yaml, instead, and puts "emotion" after "v3_log" in the run name
(runs/v3_log_emotion_s<seed>); everything else is the same.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from jevmark.config import load_config
from jevmark.data.build import V3_DOMAINS
from jevmark.data.unseen import OTHER_LABEL
from jevmark.encode import Encoded, encode
from jevmark.feedback import EPSILON, FLIP_RATE, file_sha256, interactions, write_log
from jevmark.model import JevMark
from jevmark.provenance import git_state
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(message, flush=True)


def first_batch_check(jev: JevMark, batch: Sequence[Encoded]) -> bool:
    """True if the fp32 fallback was needed (decision 28); raises if the logits are not finite even in fp32. As evaluate.py."""
    with torch.no_grad():
        if all(torch.isfinite(x).all() for x in jev.slot_logits(batch)):
            return False
    if jev.autocast_dtype is None:
        raise RuntimeError("slot logits contain NaN or inf in fp32; this is not a precision problem")
    log("WARNING: NaN or inf in first-batch slot logits under fp16 autocast; switching to fp32 (decision 28)")
    jev.use_fp32()
    with torch.no_grad():
        if not all(torch.isfinite(x).all() for x in jev.slot_logits(batch)):
            raise RuntimeError("slot logits still contain NaN or inf after the fp32 fallback")
    return True


def log_run_name(checkpoint: Path, seed: int, limit: int | None = None, domain: str = "banking77") -> str:
    """v3_log_s<seed> for the 0.6B policy on Banking77 (the phase 3 name), with _<size> for another size named in the checkpoint
    and _emotion before it for the emotion domain: v3_log_17b_s0, v3_log_emotion_s0."""
    size = re.search(r"(06b|17b)", checkpoint.name)
    prefix = "v3_log" + ("_emotion" if domain == "emotion" else "") + ("" if size is None or size.group(1) == "06b" else f"_{size.group(1)}")
    return f"{prefix}_s{seed}" + (f"_limit{limit}" if limit else "")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="the logging policy's run directory (runs/sft_06b in v3)")
    parser.add_argument("--config", default=None, help="default: the run's config.yaml")
    parser.add_argument("--seed", type=int, default=0, help="seeds the actions and the flips; the run is v3_log_s<seed> (v3_log_<size>_s<seed> for a policy other than 0.6B)")
    parser.add_argument("--order-seed", type=int, default=0, help="seeds the message order, the same for every log seed")
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--flip-rate", type=float, default=FLIP_RATE)
    parser.add_argument("--domain", choices=sorted(V3_DOMAINS), default="banking77", help="the log domain: banking77 (phase 3) or emotion")
    parser.add_argument("--limit", type=int, default=None, help="only the first N messages of the order, for smoke runs; writes the log run name plus _limit<N>")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:N; default cuda when available")
    parser.add_argument("--v3-config", default=str(REPO / "configs" / "v3_data.yaml"), help="names the data file and its recorded sha256")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    git = git_state()  # before any output exists (decision 35)
    run_name = log_run_name(Path(args.ckpt), args.seed, args.limit, args.domain)
    out_dir = Path(args.runs_dir) / run_name
    if (out_dir / "log.jsonl").exists():
        log(f"{out_dir / 'log.jsonl'} exists; a log is collected once and never overwritten")
        return 1

    v3 = load_config(args.v3_config)
    name = (v3["files"] if args.domain == "banking77" else v3[args.domain]["files"])["train"]
    data_path = Path(args.data_dir) / name
    digest = file_sha256(data_path)
    expected = (v3.get("sha256") or {}).get(name)
    if expected and digest != expected:
        log(f"{data_path} has sha256 {digest}, but {args.v3_config} records {expected}; rebuild with make data-v3")
        return 1
    records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    order = list(range(len(records)))
    random.Random(f"v3_log_order:{args.order_seed}").shuffle(order)
    if args.limit:
        order = order[: args.limit]
    records = [records[i] for i in order]

    checkpoint = Path(args.ckpt)
    config = load_config(Path(args.config) if args.config else checkpoint / "config.yaml")
    jev = JevMark.load(config, checkpoint=checkpoint, device=args.device)
    merged = jev.merge_lora()
    encoded = [encode(Request.from_dict({"state": r["state"], "questions": r["questions"]}), jev.tokenizer, jev.max_tokens) for r in records]
    fallback = first_batch_check(jev, encoded[: args.batch_size])
    log(f"logging policy {jev.model_id} on {jev.device}, autocast {jev.autocast_dtype}, lora merged {merged}; {len(records)} messages, seed {args.seed}")

    probs: list[list[float]] = []
    for start in range(0, len(encoded), args.batch_size):
        probs += [d.double().cpu().tolist() for d in jev.forward_distributions(encoded[start : start + args.batch_size])]
    rows = []
    for record, p in zip(records, probs):
        (qid, question), = record["questions"].items()
        rows.append((record["id"], qid, list(question["criteria"]), p, record["gold"][qid]))
    interactions_ = interactions(rows, args.seed, jev.model_id, git["commit"] or "unknown", args.epsilon, args.flip_rate)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_sha = write_log(out_dir / "log.jsonl", interactions_)
    n = len(interactions_)
    argmax = [max(range(len(p)), key=p.__getitem__) for p in probs]
    metrics = {
        "run_name": run_name,
        "decision": 56,
        "logging_policy": {"checkpoint": str(checkpoint), "model_id": jev.model_id, "temperature": jev.temperature, "lora_merged": merged,
                           "adapter_sha256": hashlib.sha256((checkpoint / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()},
        "git": git,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "device": str(jev.device),
        "precision": {"autocast": str(jev.autocast_dtype), "fp32_fallback_used": fallback},
        "sampling": {"seed": args.seed, "order_seed": args.order_seed, "epsilon": args.epsilon, "behaviour": "(1 - epsilon) p + epsilon / K",
                     "flip_rate": args.flip_rate, "action_stream": f"v3_log:{args.seed}", "flip_stream": f"v3_flip:{args.seed}", "order_stream": f"v3_log_order:{args.order_seed}"},
        "data_files_sha256": {name: digest},
        "limit": args.limit,
        "n": n,
        "accuracy": sum(rows[i][2][argmax[i]] == rows[i][4] for i in range(n)) / n,
        "outcome_rate": sum(i.outcome for i in interactions_) / n,
        "flipped_outcome_rate": sum(i.flipped_outcome for i in interactions_) / n,
        "flip_share": sum(i.outcome != i.flipped_outcome for i in interactions_) / n,
        "predicted_other_rate": sum(rows[i][2][argmax[i]] == OTHER_LABEL for i in range(n)) / n,
        "chosen_other_rate": sum(i.labels[i.action] == OTHER_LABEL for i in interactions_) / n,
        "mean_propensity": sum(i.propensity for i in interactions_) / n,
        "log_sha256": log_sha,
        "log_file": "log.jsonl",
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"{V3_DOMAINS[args.domain]['train']}: accuracy {metrics['accuracy']:.4f}, outcome rate {metrics['outcome_rate']:.4f}, flipped {metrics['flipped_outcome_rate']:.4f}, "
        f"predicted other {metrics['predicted_other_rate']:.4f}, chosen other {metrics['chosen_other_rate']:.4f}")
    log(f"wrote {out_dir}/log.jsonl (sha256 {log_sha}) and metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
