"""B1 baseline: an instruct LLM of jevmark's size answers the same questions as generated JSON (task 1.8).

    python scripts/baseline_llm_json.py --device cuda
    python scripts/baseline_llm_json.py --device cuda --limit 5 --latency-requests 10   # smoke run

Qwen/Qwen3-1.7B (the instruct release, revision pinned) reads the shared prompt
(jevmark/baselines/prompt.py) through its chat template with thinking disabled and
answers with greedy decoding, at most --max-new-tokens new tokens, in batches of
--batch-size (left padded, longest prompts first). It runs on the fixed baseline
subset (data/baseline_subset.json) for every split, so the comparison is same size,
generate versus read out.

Writes runs/<run_name>/replies.jsonl (one line per request: reply, input and
generated tokens, whether generation stopped by itself; gitignored, and appended
as it goes so --resume can continue), metrics.json in the evaluate.py layout
(jevmark/baselines/results.py), config.yaml and model_id.txt. Latency: batch-1
wall clock per request (chat template, tokenization and generation), median over
--latency-requests requests, which are the first records of data/train.jsonl, the
same requests evaluate.py times; throughput at --batch-size on the same requests.

The first batch's next-token logits are checked for NaN or inf under fp16 on GPU;
on failure the model is reloaded in fp32, which is logged and recorded.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from jevmark.baselines.prompt import build_prompt
from jevmark.baselines.results import CONFIDENCE_NOTE, answers_from_replies, append_reply, baseline_reports_by_split, read_replies, request_of
from jevmark.baselines.subset import SUBSET_PATH, load_subset, read_jsonl, subset_records
from jevmark.data.build import SPLITS
from jevmark.metrics import timing_summary
from jevmark.provenance import git_state
from jevmark.sampling import sample_records

REPO = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RUN_NAME = "b1_qwen17b_json"


def log(message: str) -> None:
    print(message, flush=True)


# Generation


def chat_text(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    """The prompt as one user turn, with the assistant turn opened and thinking disabled (Qwen3 chat template)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def stop_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    ids = model.generation_config.eos_token_id if model.generation_config is not None else None
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    for token in ("<|im_end|>", tokenizer.eos_token):
        token_id = tokenizer.convert_tokens_to_ids(token) if token else None
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id and token_id not in ids:
            ids.append(token_id)
    return ids


def _batch_inputs(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "left"
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def generate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], max_new_tokens: int, batch_size: int
) -> list[dict[str, Any]]:
    """Greedy replies in prompt order: reply text, input tokens, generated tokens (up to and including the stop token) and whether a stop token ended it."""
    stops = stop_token_ids(model, tokenizer)
    config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
    texts = [chat_text(tokenizer, p) for p in prompts]
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])  # longest first: less padding, and memory peaks early
    out: list[dict[str, Any] | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        inputs = _batch_inputs(tokenizer, [texts[i] for i in idx], model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, generation_config=config)
        new = generated[:, inputs["input_ids"].shape[1] :].tolist()
        for i, tokens in zip(idx, new):
            end = next((k for k, t in enumerate(tokens) if t in stops), None)
            body = tokens if end is None else tokens[:end]
            out[i] = {
                "reply": tokenizer.decode(body, skip_special_tokens=True),
                "input_tokens": lengths[i],
                "output_tokens": len(tokens) if end is None else end + 1,
                "finished": end is not None,
            }
    return out  # type: ignore[return-value]


def first_batch_finite(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]) -> bool:
    inputs = _batch_inputs(tokenizer, [chat_text(tokenizer, p) for p in prompts], model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    return bool(torch.isfinite(logits).all())


def load_model(model_id: str, revision: str | None, device: torch.device, dtype: torch.dtype) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], max_new_tokens: int, batch_size: int) -> dict[str, Any]:
    """Batch-1 wall clock per request, median over the prompts, and requests per second at batch_size on the same prompts."""
    generate(model, tokenizer, prompts[:2], max_new_tokens, 1)  # warm-up
    times, tokens = [], []
    for prompt in prompts:
        _sync(model.device)
        start = time.perf_counter()
        tokens.append(generate(model, tokenizer, [prompt], max_new_tokens, 1)[0]["output_tokens"])
        _sync(model.device)
        times.append(time.perf_counter() - start)
    _sync(model.device)
    start = time.perf_counter()
    generate(model, tokenizer, prompts, max_new_tokens, batch_size)
    _sync(model.device)
    elapsed = time.perf_counter() - start
    return {
        "batch_1": timing_summary(times),
        "batch_1_mean_output_tokens": sum(tokens) / len(tokens),
        f"batch_{batch_size}_requests_per_second": len(prompts) / elapsed,
        "requests": f"the first {len(prompts)} records of train.jsonl, as evaluate.py",
        "device": str(model.device),
    }


def generation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_requests": len(rows),
        "mean_input_tokens": sum(r["input_tokens"] for r in rows) / len(rows),
        "mean_output_tokens": sum(r["output_tokens"] for r in rows) / len(rows),
        "truncated_rate": sum(not r["finished"] for r in rows) / len(rows),
    }


# Run


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--limit", type=int, default=None, help="a seeded stratified sample of N subset records per split, for smoke runs; writes runs/<run_name>_limit<N>/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--latency-requests", type=int, default=200)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:N; default cuda when available")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true", help="continue a run whose replies.jsonl exists; otherwise such a run is never overwritten")
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
    run_name = f"{args.run_name}_limit{args.limit}" if args.limit else args.run_name
    out_dir = Path(args.runs_dir) / run_name
    replies_path = out_dir / "replies.jsonl"
    if replies_path.exists() and not args.resume:
        log(f"{replies_path} exists; pass --resume to continue that run, or delete the directory")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = Path(args.subset)
    subset = load_subset(subset_path)
    data_dir = Path(args.data_dir)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    data_files: dict[str, str] = {}
    for split in args.splits:
        records, digest = subset_records(data_dir, subset, split)
        records_by_split[split] = sample_records(records, args.limit, random.Random(f"limit:{split}")) if args.limit else records
        data_files[f"{split}.jsonl"] = digest

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args.model, args.revision, device, dtype)
    first = next(iter(records_by_split.values()))[: args.batch_size]
    fp32_fallback = False
    if not first_batch_finite(model, tokenizer, [build_prompt(request_of(r)) for r in first]):
        if dtype == torch.float32:
            raise RuntimeError("next-token logits contain NaN or inf in fp32; this is not a precision problem")
        log("WARNING: NaN or inf in first-batch logits under fp16; reloading the model in fp32")
        del model
        model, tokenizer = load_model(args.model, args.revision, device, torch.float32)
        fp32_fallback = True
    log(f"run {run_name}: {args.model}@{args.revision} on {device}, dtype {model.dtype}, batch {args.batch_size}, max_new_tokens {args.max_new_tokens}")

    done = read_replies(replies_path)
    for split, records in records_by_split.items():
        todo = [r for r in records if (split, r["id"]) not in done]
        split_start = time.perf_counter()
        rows = generate(model, tokenizer, [build_prompt(request_of(r)) for r in todo], args.max_new_tokens, args.batch_size)
        for record, row in zip(todo, rows):
            line = {"split": split, "record_id": record["id"], **row}
            append_reply(replies_path, line)
            done[(split, record["id"])] = line
        log(f"{split:20} {len(records):5} requests ({len(todo)} generated) in {time.perf_counter() - split_start:.1f}s")

    replies = {key: row for key, row in done.items() if key[0] in records_by_split}
    splits = baseline_reports_by_split(answers_from_replies(records_by_split, replies))
    for split, report in splits.items():
        overall = report["overall"]
        log(f"{split:20} acc {overall['accuracy'] or 0:.4f} (parsed)  acc_all {overall['accuracy_all']:.4f}  parse failures {overall['parse_failure_rate']:.4f}")

    train_records, _ = read_jsonl(data_dir / "train.jsonl")
    probe = [build_prompt(request_of(r)) for r in train_records[: args.latency_requests]]
    timing = latency(model, tokenizer, probe, args.max_new_tokens, args.batch_size)
    log(f"latency batch 1: median {timing['batch_1']['median_ms']:.1f} ms; batch {args.batch_size}: {timing[f'batch_{args.batch_size}_requests_per_second']:.2f} requests/s")

    rows = list(replies.values())
    metrics = {
        "run_name": run_name,
        "baseline": "B1",
        "model": {"id": args.model, "revision": args.revision},
        "git": git,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "subset": {"path": str(subset_path.relative_to(REPO)) if subset_path.is_relative_to(REPO) else str(subset_path), "per_split": subset["per_split"]},
        "limit": args.limit,
        "device": str(device),
        "precision": {"dtype": str(model.dtype), "fp32_fallback_used": fp32_fallback},
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, "enable_thinking": False},
        "data_files_sha256": data_files,
        "confidence_note": CONFIDENCE_NOTE,
        "splits": splits,
        "generation": {"overall": generation_summary(rows), **{s: generation_summary([r for k, r in replies.items() if k[0] == s]) for s in records_by_split}},
        "latency": timing,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "config.yaml").write_text(yaml.safe_dump({"run_name": run_name, "baseline": "B1", "model": metrics["model"], "decoding": metrics["decoding"], "subset": metrics["subset"]}, sort_keys=False))
    (out_dir / "model_id.txt").write_text(f"{args.model}@{args.revision}\n")
    log(f"wrote {out_dir}/metrics.json, replies.jsonl ({len(rows)} requests), config.yaml, model_id.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
