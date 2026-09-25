"""Evaluate a checkpoint, or the frozen base model, on the JSONL splits (docs/TASKS.md 1.6).

    python scripts/evaluate.py --ckpt base --config configs/base_06b.yaml
    python scripts/evaluate.py --ckpt runs/sft_clinc_v1_06b

Reads the unrounded distributions from JevMark.forward_distributions (through
encode), never the rounded systemone responses. Writes runs/<run_name>/metrics.json,
config.yaml, model_id.txt, plots/<split>.png and results.jsonl.gz (one line per
question, gitignored; scripts/recompute_metrics.py rebuilds metrics.json from it).
--shuffle-questions [SPLIT ...] evaluates each record a second time with its
questions in a seeded different order and reports order_sensitivity per split.
A trained adapter is merged into the backbone
before evaluation unless --no-merge is given; metrics.json records which path ran.
With --ckpt base the run name is
the config's run_name (base_06b or base_17b); a --limit run writes to
runs/<run_name>_limit<N>/ so smoke runs never overwrite real results.

The first batch's slot logits are checked for NaN or inf; on failure autocast is
switched off (JevMark.use_fp32, decision 28), the switch is logged and recorded in
metrics.json, and evaluation continues in fp32.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import random
import subprocess
from collections import defaultdict
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from jevmark.config import load_config  # noqa: E402
from jevmark.data.build import SPLITS  # noqa: E402
from jevmark.data.negation import negate  # noqa: E402
from jevmark.encode import Encoded, encode  # noqa: E402
from jevmark.metrics import QuestionResult, max_abs_difference, split_report, timing_summary, write_results  # noqa: E402
from jevmark.model import JevMark  # noqa: E402
from jevmark.schema import Request  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROBE_REQUESTS = 200
THROUGHPUT_BATCH = 16
# Reference palette slots 1 to 3, fixed per question type; text and surface tokens.
TYPE_COLORS = {"noul": "#2a78d6", "choice": "#eb6834", "score": "#1baf7a"}
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"


def log(message: str) -> None:
    print(message, flush=True)


# Data


def read_split(data_dir: Path, split: str, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    """The split's records (a seeded stratified sample of `limit` of them, when given) and the file's sha256."""
    path = data_dir / f"{split}.jsonl"
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return (sample_records(records, limit, random.Random(f"limit:{split}")) if limit else records), hashlib.sha256(raw).hexdigest()


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split total across groups in proportion to their sizes (largest remainder), at least one per group while total allows."""
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    counts = {k: min(sizes[k], max(1, int(x))) for k, x in exact.items()}
    for k in sorted(sizes, key=lambda k: exact[k] - int(exact[k]), reverse=True):
        if sum(counts.values()) >= total:
            break
        if counts[k] < sizes[k]:
            counts[k] += 1
    while sum(counts.values()) > total:  # the "at least one" floor overshot
        k = max(counts, key=lambda k: counts[k] - exact[k])
        counts[k] -= 1
    return counts


def sample_records(records: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    """A seeded sample of `limit` records, in file order (docs/KAGGLE.md section 8).

    Records are dataset-ordered, so the first N would cover only a few CLINC intents
    and no out-of-scope utterance. Instead the limit is split across sources in
    proportion to their size; within CLINC, out-of-scope utterances get their
    proportional share (at least one) and the rest is dealt round-robin over the
    intents in a seeded order, so the sample holds as many intents as it can;
    other sources are sampled uniformly.
    """
    if limit >= len(records):
        return records
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_source[record["source"]].append(i)
    chosen: list[int] = []
    for source, count in _allocate({s: len(ix) for s, ix in by_source.items()}, limit).items():
        indices = by_source[source]
        intents: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            intent = records[i]["meta"].get("gold_intent")
            if intent is not None:
                intents[intent].append(i)
        if not intents:
            chosen += rng.sample(indices, count)
            continue
        oos = intents.pop("oos", [])
        n_oos = min(len(oos), max(1, round(count * len(oos) / len(indices)))) if oos else 0
        chosen += rng.sample(oos, n_oos)
        pools = {k: rng.sample(v, len(v)) for k, v in sorted(intents.items())}
        order = sorted(pools)
        rng.shuffle(order)
        taken, depth = 0, 0
        while taken < count - n_oos:
            progressed = False
            for intent in order:
                if depth < len(pools[intent]) and taken < count - n_oos:
                    chosen.append(pools[intent][depth])
                    taken += 1
                    progressed = True
            if not progressed:
                break
            depth += 1
    return [records[i] for i in sorted(chosen)]


def request_of(record: dict[str, Any]) -> Request:
    return Request.from_dict({"state": record["state"], "questions": record["questions"]})


def gold_index(question: dict[str, Any], gold: Any) -> int:
    if question["type"] == "noul":
        return {"true": 0, "false": 1}[gold]
    if question["type"] == "choice":
        return list(question["criteria"]).index(gold)
    return int(gold)


def option_labels(question: dict[str, Any]) -> tuple[str, ...]:
    if question["type"] == "noul":
        return ("true", "false")
    if question["type"] == "choice":
        return tuple(question["criteria"])
    return tuple(str(i) for i in range(len(question["criteria"])))


def negated_record(record: dict[str, Any]) -> dict[str, Any]:
    """The same record with every noul instruction in its other phrasing (the question id is the noul kind)."""
    negated = copy.deepcopy(record)
    for qid, question in negated["questions"].items():
        if question["type"] == "noul":
            question["instructions"] = negate(qid, question["instructions"])
    return negated


# Model calls


def distributions(jev: JevMark, encoded: Sequence[Encoded], batch_size: int) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(encoded), batch_size):
        out += [d.double().cpu().tolist() for d in jev.forward_distributions(encoded[start : start + batch_size])]
    return out


def first_batch_check(jev: JevMark, batch: Sequence[Encoded]) -> bool:
    """True if the fp32 fallback was needed. Raises if the logits are not finite even in fp32."""
    with torch.no_grad():
        finite = all(torch.isfinite(x).all() for x in jev.slot_logits(batch))
    if finite:
        return False
    if jev.autocast_dtype is None:
        raise RuntimeError("slot logits contain NaN or inf in fp32; this is not a precision problem")
    log("WARNING: NaN or inf in first-batch slot logits under fp16 autocast; switching to fp32 (decision 28)")
    jev.use_fp32()
    with torch.no_grad():
        if not all(torch.isfinite(x).all() for x in jev.slot_logits(batch)):
            raise RuntimeError("slot logits still contain NaN or inf after the fp32 fallback")
    return True


def _sync(jev: JevMark) -> None:
    if jev.device.type == "cuda":
        torch.cuda.synchronize(jev.device)


def latency(jev: JevMark, requests: Sequence[Request], n: int) -> dict[str, Any]:
    """Batch-1 wall clock per request (encode plus forward), median over n; throughput at batch 16."""
    pool = [requests[i % len(requests)] for i in range(n)]
    for request in pool[:3]:  # warm-up
        jev.forward_distributions([encode(request, jev.tokenizer, jev.max_tokens)])
    times = []
    for request in pool:
        _sync(jev)
        start = time.perf_counter()
        jev.forward_distributions([encode(request, jev.tokenizer, jev.max_tokens)])
        _sync(jev)
        times.append(time.perf_counter() - start)
    _sync(jev)
    start = time.perf_counter()
    for i in range(0, n, THROUGHPUT_BATCH):
        jev.forward_distributions([encode(r, jev.tokenizer, jev.max_tokens) for r in pool[i : i + THROUGHPUT_BATCH]])
    _sync(jev)
    elapsed = time.perf_counter() - start
    return {"batch_1": timing_summary(times), f"batch_{THROUGHPUT_BATCH}_requests_per_second": n / elapsed, "device": str(jev.device)}


def batching_precision(jev: JevMark, encoded: Sequence[Encoded], batch_size: int) -> dict[str, Any]:
    """Max absolute difference between batched and single-call probabilities (decision 30)."""
    batched = distributions(jev, encoded, batch_size)
    single = [d for e in encoded for d in distributions(jev, [e], 1)]
    return {"n_requests": len(encoded), "batch_size": batch_size, "max_abs_difference": max_abs_difference(batched, single)}


# Evaluation of one split


def shuffled_order(record: dict[str, Any]) -> list[str]:
    """A seeded reordering of the record's question ids that differs from the stored order (records with two or more questions)."""
    order = list(record["questions"])
    rng = random.Random(f"shuffle-questions:{record['id']}")
    shuffled = list(order)
    while shuffled == order:
        rng.shuffle(shuffled)
    return shuffled


def reordered_record(record: dict[str, Any], order: Sequence[str]) -> dict[str, Any]:
    return {**record, "questions": {qid: record["questions"][qid] for qid in order}}


def evaluate_split(
    jev: JevMark, split: str, records: list[dict[str, Any]], batch_size: int, shuffle_questions: bool = False
) -> tuple[list[QuestionResult], list[Encoded]]:
    """One QuestionResult per question; noul results carry P(yes) for the negated instruction.

    With shuffle_questions, every record with two or more questions is evaluated a
    second time with its questions in a seeded different order, and each result
    carries that distribution and its position there (order sensitivity).
    """
    encoded = [encode(request_of(r), jev.tokenizer, jev.max_tokens) for r in records]
    flat = iter(distributions(jev, encoded, batch_size))
    results: list[QuestionResult] = []
    for record in records:
        for position, (qid, question) in enumerate(record["questions"].items()):
            results.append(
                QuestionResult(
                    record["id"],
                    qid,
                    question["type"],
                    tuple(next(flat)),
                    gold_index(question, record["gold"][qid]),
                    option_labels(question),
                    split=split,
                    kind=qid if question["type"] == "noul" else None,
                    position=position,
                )
            )

    if shuffle_questions:
        multi = [r for r in records if len(r["questions"]) > 1]
        orders = [shuffled_order(r) for r in multi]
        moved = iter(distributions(jev, [encode(request_of(reordered_record(r, o)), jev.tokenizer, jev.max_tokens) for r, o in zip(multi, orders)], batch_size))
        shuffled = {}
        for record, order in zip(multi, orders):
            for position, qid in enumerate(order):
                shuffled[(record["id"], qid)] = (tuple(next(moved)), position)
        results = [
            replace(r, shuffled_probs=shuffled[key][0], shuffled_position=shuffled[key][1]) if (key := (r.record_id, r.question_id)) in shuffled else r
            for r in results
        ]

    with_noul = [r for r in records if any(q["type"] == "noul" for q in r["questions"].values())]
    if with_noul:
        negated_flat = iter(distributions(jev, [encode(request_of(negated_record(r)), jev.tokenizer, jev.max_tokens) for r in with_noul], batch_size))
        negated_p_yes = {}
        for record in with_noul:
            for qid, question in record["questions"].items():
                probs = next(negated_flat)
                if question["type"] == "noul":
                    negated_p_yes[(record["id"], qid)] = probs[0]
        results = [
            replace(r, negated_p_yes=negated_p_yes[(r.record_id, r.question_id)]) if r.qtype == "noul" else r for r in results
        ]
    return results, encoded


# Output


def plot_reliability(split: str, metrics: dict[str, Any], path: Path) -> None:
    types = [t for t in ("noul", "choice", "score") if t in metrics]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    top.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="perfect calibration")
    for qtype in types:
        bins = [b for b in metrics[qtype]["reliability"] if b["count"]]
        n = metrics[qtype]["n"]
        top.plot(
            [b["mean_confidence"] for b in bins],
            [b["mean_accuracy"] for b in bins],
            color=TYPE_COLORS[qtype],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{qtype} (n={n}, ECE {metrics[qtype]['ece']:.3f})",
        )
        centers = [(b["lower"] + b["upper"]) / 2 for b in metrics[qtype]["reliability"]]
        bottom.plot(centers, [b["count"] for b in metrics[qtype]["reliability"]], color=TYPE_COLORS[qtype], linewidth=2, drawstyle="steps-mid")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_ylabel("accuracy in bin", color=TEXT_PRIMARY)
    top.set_title(f"Reliability, {split} (top-1 probability, 15 bins)", color=TEXT_PRIMARY, fontsize=11)
    top.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper left")
    bottom.set_xlabel("top-1 probability", color=TEXT_PRIMARY)
    bottom.set_ylabel("questions", color=TEXT_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def git_state(repo: Path = REPO) -> dict[str, Any]:
    """Commit and whether tracked code or docs differ from it, ignoring runs/.

    Called before any output is written: a run that overwrites the tracked files
    of an earlier run (for example a B0 re-run) must not report itself dirty.
    """

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)runs")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="a run directory, or 'base' for the frozen backbone of --config")
    parser.add_argument("--config", default=None, help="config file; default configs/base.yaml for base, the run's config.yaml otherwise")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS), help="default: all nine")
    parser.add_argument("--limit", type=int, default=None, help="a seeded, stratified sample of N records per split (sample_records), for smoke runs and the fast cycle")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:N; default cuda when available")
    parser.add_argument("--no-merge", action="store_true", help="keep the LoRA adapter unmerged (default: merge it into the backbone for speed)")
    parser.add_argument(
        "--shuffle-questions",
        nargs="*",
        default=None,
        metavar="SPLIT",
        help="also evaluate each record with its questions reordered, to measure order sensitivity; on the named splits, or on every evaluated split when none is named",
    )
    parser.add_argument("--run-name", default=None, help="write to runs/<run-name>/ instead of the default name (no _limit suffix is added); the fast cycle uses it")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.shuffle_questions:
        unknown = set(args.shuffle_questions) - set(args.splits)
        if unknown:
            parser.error(f"--shuffle-questions names splits that are not evaluated: {sorted(unknown)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ckpt == "base":
        checkpoint = None
        config_path = Path(args.config or REPO / "configs" / "base.yaml")
        config = load_config(config_path)
        run_name = config["run_name"]
    else:
        checkpoint = Path(args.ckpt)
        config_path = Path(args.config) if args.config else checkpoint / "config.yaml"
        config = load_config(config_path)
        run_name = checkpoint.name
    if args.run_name:
        run_name = args.run_name
    elif args.limit:
        run_name = f"{run_name}_limit{args.limit}"
    out_dir = Path(args.runs_dir) / run_name

    log(f"run {run_name}: ckpt {args.ckpt}, config {config_path}, backbone {config['backbone']['id']}")
    jev = JevMark.load(config, checkpoint=checkpoint, device=args.device)
    lora_merged = None if checkpoint is None else (False if args.no_merge else jev.merge_lora())
    log(f"model {jev.model_id} on {jev.device}, autocast {jev.autocast_dtype}, max_tokens {jev.max_tokens}, lora merged {lora_merged}")

    started = time.perf_counter()
    git = git_state()  # before any output exists (decision 35)
    fallback_used = None
    split_results: dict[str, Any] = {}
    all_results: list[QuestionResult] = []
    data_files: dict[str, str] = {}
    probe: list[Encoded] = []
    probe_requests: list[Request] = []
    for split in args.splits:
        records, digest = read_split(Path(args.data_dir), split, args.limit)
        data_files[f"{split}.jsonl"] = digest
        if fallback_used is None:
            fallback_used = first_batch_check(jev, [encode(request_of(r), jev.tokenizer, jev.max_tokens) for r in records[: args.batch_size]])
        split_start = time.perf_counter()
        shuffle = args.shuffle_questions is not None and (not args.shuffle_questions or split in args.shuffle_questions)
        results, encoded = evaluate_split(jev, split, records, args.batch_size, shuffle_questions=shuffle)
        all_results += results
        metrics = split_report(results)
        split_results[split] = metrics
        if len(probe) < PROBE_REQUESTS:
            take = PROBE_REQUESTS - len(probe)
            probe += encoded[:take]
            probe_requests += [request_of(r) for r in records[:take]]
        overall = metrics["overall"]
        log(f"{split:20} {len(records):6} records  acc {overall['accuracy']:.4f}  ece {overall['ece']:.4f}  nll {overall['nll']:.4f}  ({time.perf_counter() - split_start:.1f}s)")
        if "order_sensitivity" in metrics:
            moved = metrics["order_sensitivity"]["overall"]
            log(f"{'':20} questions reordered: acc {moved['accuracy']:.4f} -> {moved['accuracy_shuffled']:.4f}, prediction agreement {moved['prediction_agreement']:.4f}, mean max |dp| {moved['mean_max_abs_difference']:.4f}")
        plot_reliability(split, metrics, out_dir / "plots" / f"{split}.png")

    batching = batching_precision(jev, probe, args.batch_size)
    log(f"batched vs single on {batching['n_requests']} requests: max abs difference {batching['max_abs_difference']:.3g}")
    timing = latency(jev, probe_requests, PROBE_REQUESTS)
    log(f"latency batch 1: median {timing['batch_1']['median_ms']:.2f} ms; batch {THROUGHPUT_BATCH}: {timing[f'batch_{THROUGHPUT_BATCH}_requests_per_second']:.1f} requests/s")

    metrics_json = {
        "run_name": run_name,
        "model_id": jev.model_id,
        "ckpt": args.ckpt,
        "config": str(config_path),
        "git": git,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "limit": args.limit,
        "shuffle_questions": args.shuffle_questions,
        "device": str(jev.device),
        "precision": {"autocast": str(jev.autocast_dtype) if jev.autocast_dtype else None, "fp32_fallback_used": fallback_used},
        "temperature": jev.temperature,
        "lora_merged": lora_merged,
        "data_files_sha256": data_files,
        "confidence_note": "ECE and reliability use the top-1 probability; coverage uses the response confidence field (1 - H/ln K for choice and score, max(p, 1 - p) for noul).",
        "splits": split_results,
        "batching_precision": batching,
        "latency": timing,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2) + "\n")
    write_results(out_dir / "results.jsonl.gz", all_results)
    if checkpoint is None or checkpoint.resolve() != out_dir.resolve():
        (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (out_dir / "model_id.txt").write_text(jev.model_id + "\n")
    size_kb = (out_dir / "results.jsonl.gz").stat().st_size / 1024
    log(f"wrote {out_dir}/metrics.json, results.jsonl.gz ({len(all_results)} questions, {size_kb:.1f} KiB), config.yaml, model_id.txt and {len(args.splits)} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
