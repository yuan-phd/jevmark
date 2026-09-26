"""B2 baseline: a commercial API answers the same questions with structured JSON output (task 1.8).

    uv run --extra baselines python scripts/baseline_api.py --dry-run
    uv run --extra baselines python scripts/baseline_api.py --price-input 0.40 --price-output 1.60
    uv run --extra baselines python scripts/baseline_api.py --model <stronger model> --sub --price-input ... --price-output ...

OpenAI chat completions, one request per record: the shared prompt
(jevmark/baselines/prompt.py) as the user message and a strict JSON schema built
for that request as the response format, so every answer is restricted to the
request's own options. The key comes from OPENAI_API_KEY. Prices change, so they
are arguments, in USD per million tokens (decision 31); --price-cached-input
defaults to --price-input. Runs on the fixed baseline subset
(data/baseline_subset.json), 500 records per split, or with --sub on its 200-record
sub-subset, the setting for a stronger and more expensive model.

Every reply is appended to runs/b2_<model>/replies.jsonl as it arrives (reply,
tokens, cost, latency, finish reason; gitignored), so an interrupted run continues
where it stopped when started again with --resume. --max-usd (default 20) is a
hard cap: before each request the spend so far plus a worst case for that request
(prompt characters / 2 input tokens, --max-completion-tokens output tokens) must
stay within it, else the run stops, writes metrics for what it has and exits with
code 2. Transient API errors are retried with backoff; an error that persists, or a
rejected request, stops the run with code 1 and nothing recorded for that request.
Requests are sent one at a time, so latency is the wall clock of one call.

--dry-run prints three of the prompts with their schemas and the size of the run,
and needs neither the key nor the openai package.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from jevmark.baselines.prompt import build_prompt, json_schema
from jevmark.baselines.results import CONFIDENCE_NOTE, answers_from_replies, append_reply, baseline_reports_by_split, read_replies, request_of
from jevmark.baselines.subset import SUBSET_PATH, load_subset, subset_records
from jevmark.data.build import SPLITS
from jevmark.metrics import timing_summary
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
DRY_RUN_SPLITS = ("test_indomain", "test_sst5", "test_emotion")  # between them, every question type
RETRYABLE = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")


class SpendCapReached(Exception):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run_name_for(model: str) -> str:
    return "b2_" + re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def request_body(record: dict[str, Any], model: str, temperature: float | None, max_completion_tokens: int) -> dict[str, Any]:
    request = request_of(record)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(request)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "strict": True, "schema": json_schema(request)}},
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    return body


def worst_case_usd(body: dict[str, Any], price_input: float, price_output: float) -> float:
    """An upper estimate of one request's cost: two characters per input token, every allowed output token."""
    chars = len(body["messages"][0]["content"]) + len(json.dumps(body["response_format"]))
    return (chars / 2 * price_input + body["max_completion_tokens"] * price_output) / 1e6


def cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int, price_input: float, price_cached_input: float, price_output: float) -> float:
    return ((input_tokens - cached_tokens) * price_input + cached_tokens * price_cached_input + output_tokens * price_output) / 1e6


def call_with_retries(create: Callable[..., Any], body: dict[str, Any], retries: int, sleep: Callable[[float], None] = time.sleep) -> tuple[Any, float]:
    """The response and the wall clock of the attempt that succeeded; transient errors are retried with backoff 2, 4, 8 ... seconds."""
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = create(**body)
            return response, time.perf_counter() - start
        except Exception as error:  # noqa: BLE001 - the openai error classes are matched by name so tests need no openai
            if type(error).__name__ not in RETRYABLE or attempt == retries:
                raise
            log(f"  {type(error).__name__}, retry {attempt + 1} of {retries}")
            sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def reply_row(split: str, record_id: str, response: Any, latency_s: float, prices: tuple[float, float, float]) -> dict[str, Any]:
    choice = response.choices[0]
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details is not None else 0
    refusal = getattr(choice.message, "refusal", None)
    return {
        "split": split,
        "record_id": record_id,
        "reply": None if refusal else choice.message.content,
        "refusal": refusal,
        "finish_reason": choice.finish_reason,
        "finished": choice.finish_reason == "stop",
        "model": response.model,
        "input_tokens": usage.prompt_tokens,
        "cached_input_tokens": cached,
        "output_tokens": usage.completion_tokens,
        "cost_usd": cost_usd(usage.prompt_tokens, cached, usage.completion_tokens, *prices),
        "latency_s": latency_s,
    }


# Run


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--price-input", type=float, default=None, help="USD per million input tokens (required unless --dry-run)")
    parser.add_argument("--price-output", type=float, default=None, help="USD per million output tokens (required unless --dry-run)")
    parser.add_argument("--price-cached-input", type=float, default=None, help="USD per million cached input tokens; default --price-input")
    parser.add_argument("--max-usd", type=float, default=20.0, help="hard spend cap for the run, including replies already in replies.jsonl")
    parser.add_argument("--sub", action="store_true", help="the 200-record sub-subset per split instead of the 500-record subset")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--subset", default=str(SUBSET_PATH))
    parser.add_argument("--temperature", default="0", help="sampling temperature, or 'none' to leave it to the API (models that reject it)")
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--run-name", default=None, help="default b2_<model>")
    parser.add_argument("--resume", action="store_true", help="continue a run whose replies.jsonl exists; otherwise such a run is never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="print three prompts and the size of the run; no API call")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = parser.parse_args(argv)
    if not args.dry_run and (args.price_input is None or args.price_output is None):
        parser.error("--price-input and --price-output are required (USD per million tokens at run time)")
    if args.price_cached_input is None:
        args.price_cached_input = args.price_input
    args.temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    return args


def dry_run(args: argparse.Namespace, records_by_split: dict[str, list[dict[str, Any]]]) -> int:
    shown = [s for s in DRY_RUN_SPLITS if s in records_by_split] + [s for s in records_by_split if s not in DRY_RUN_SPLITS]
    for n, split in enumerate(shown[:3], 1):
        record = records_by_split[split][0]
        body = request_body(record, args.model, args.temperature, args.max_completion_tokens)
        log(f"===== prompt {n} of 3: {split} / {record['id']} =====")
        log(body["messages"][0]["content"])
        log(f"----- response_format.json_schema.schema -----\n{json.dumps(body['response_format']['json_schema']['schema'])}")
        log(f"----- gold: {json.dumps(record['gold'])}\n")
    bodies = [request_body(r, args.model, args.temperature, args.max_completion_tokens) for rs in records_by_split.values() for r in rs]
    chars = sum(len(b["messages"][0]["content"]) for b in bodies)
    log(f"model {args.model}, {len(bodies)} requests over {len(records_by_split)} splits ({'sub-subset' if args.sub else 'subset'}), "
        f"{chars} prompt characters (about {chars // 4} input tokens at four characters per token), cap {args.max_usd} USD")
    if args.price_input is not None and args.price_output is not None:
        log(f"worst case at the given prices: {sum(worst_case_usd(b, args.price_input, args.price_output) for b in bodies):.2f} USD")
    return 0


def run(args: argparse.Namespace, records_by_split: dict[str, list[dict[str, Any]]], data_files: dict[str, str], create: Callable[..., Any], sleep: Callable[[float], None] = time.sleep) -> int:
    started = time.perf_counter()
    git = git_state()  # before any output exists (decision 35)
    run_name = args.run_name or run_name_for(args.model)
    out_dir = Path(args.runs_dir) / run_name
    replies_path = out_dir / "replies.jsonl"
    setting = {"run_name": run_name, "baseline": "B2", "model": args.model, "sub": args.sub, "temperature": args.temperature, "max_completion_tokens": args.max_completion_tokens}
    if replies_path.exists():
        if not args.resume:
            log(f"{replies_path} exists; pass --resume to continue that run, or delete the directory")
            return 1
        previous = yaml.safe_load((out_dir / "config.yaml").read_text())
        if previous != setting:
            log(f"--resume with different settings: {previous} against {setting}")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(setting, sort_keys=False))
    (out_dir / "model_id.txt").write_text(args.model + "\n")

    prices = (args.price_input, args.price_cached_input, args.price_output)
    done = read_replies(replies_path)
    spent = sum(row["cost_usd"] for row in done.values())
    stopped_by_cap, exit_code = False, 0
    try:
        for split, records in records_by_split.items():
            todo = [r for r in records if (split, r["id"]) not in done]
            log(f"{split:20} {len(records):5} requests, {len(todo)} to send; spent so far {spent:.4f} USD")
            for record in todo:
                body = request_body(record, args.model, args.temperature, args.max_completion_tokens)
                if spent + worst_case_usd(body, args.price_input, args.price_output) > args.max_usd:
                    raise SpendCapReached
                response, latency_s = call_with_retries(create, body, args.retries, sleep)
                row = reply_row(split, record["id"], response, latency_s, prices)
                append_reply(replies_path, row)
                done[(split, record["id"])] = row
                spent += row["cost_usd"]
    except SpendCapReached:
        stopped_by_cap = True
        log(f"STOPPED: the next request could take the spend past --max-usd {args.max_usd} (spent {spent:.4f} USD)")
        exit_code = 2
    except Exception as error:  # noqa: BLE001 - report, keep what was recorded, and let --resume continue
        log(f"STOPPED on {type(error).__name__}: {error}")
        exit_code = 1

    replies = {key: row for key, row in done.items() if key[0] in records_by_split}
    rows = list(replies.values())
    splits = baseline_reports_by_split(answers_from_replies(records_by_split, replies)) if rows else {}
    for split, report in splits.items():
        overall = report["overall"]
        log(f"{split:20} acc {overall['accuracy'] or 0:.4f} (parsed)  acc_all {overall['accuracy_all']:.4f}  parse failures {overall['parse_failure_rate']:.4f}")
    total_cost = sum(r["cost_usd"] for r in rows)
    expected = sum(len(rs) for rs in records_by_split.values())
    metrics = {
        "run_name": run_name,
        "baseline": "B2",
        "model": {"requested": args.model, "reported": sorted({r["model"] for r in rows})},
        "prices_usd_per_million_tokens": {"input": args.price_input, "cached_input": args.price_cached_input, "output": args.price_output},
        "git": git,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "subset": {"path": "data/baseline_subset.json", "records": "sub_splits (200 per split)" if args.sub else "splits (500 per split)"},
        "request": {"response_format": "json_schema, strict", "temperature": args.temperature, "max_completion_tokens": args.max_completion_tokens},
        "data_files_sha256": data_files,
        "confidence_note": CONFIDENCE_NOTE,
        "complete": len(rows) == expected,
        "n_requests": len(rows),
        "n_requests_expected": expected,
        "splits": splits,
        "usage": {
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "cached_input_tokens": sum(r["cached_input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "cost_usd": total_cost,
            "cost_per_1000_requests_usd": 1000 * total_cost / len(rows) if rows else None,
            "truncated_rate": sum(not r["finished"] for r in rows) / len(rows) if rows else None,
            "refusals": sum(r["refusal"] is not None for r in rows),
        },
        "spend": {"max_usd": args.max_usd, "spent_usd": total_cost, "stopped_by_cap": stopped_by_cap},
        "latency": {"per_request": timing_summary([r["latency_s"] for r in rows]) if rows else None, "note": "one call at a time, wall clock of the successful attempt, from the caller's network"},
        "wall_clock_seconds_this_session": time.perf_counter() - started,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"wrote {out_dir}/metrics.json: {len(rows)} of {expected} requests, {total_cost:.4f} USD ({metrics['usage']['cost_per_1000_requests_usd'] or 0:.4f} per 1000 requests)")
    return exit_code


def main(argv: Sequence[str] | None = None, create: Callable[..., Any] | None = None) -> int:
    args = parse_args(argv)
    subset = load_subset(Path(args.subset))
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    data_files: dict[str, str] = {}
    for split in args.splits:
        records_by_split[split], data_files[f"{split}.jsonl"] = subset_records(Path(args.data_dir), subset, split, sub=args.sub)
    if args.dry_run:
        return dry_run(args, records_by_split)
    if create is None:
        if not os.environ.get("OPENAI_API_KEY"):
            log("OPENAI_API_KEY is not set")
            return 1
        from openai import OpenAI

        create = OpenAI().chat.completions.create
    return run(args, records_by_split, data_files, create)


if __name__ == "__main__":
    sys.exit(main())
