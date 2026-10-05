"""Demo: one customer-support message, five decisions in one forward pass, and a confidence gate.

    uv run python scripts/demo.py [--checkpoint hf://yuanphd/jevmark/sft_06b] [--threshold 0.9] [--device cpu]

Loads the adapter through systemone's default model (JEVMARK_CHECKPOINT, a run
directory or a Hub reference), asks the five questions below about MESSAGE in one
call, prints the response JSON, then applies one confidence threshold: an answer at
or above it is acted on automatically, one below it is escalated to an LLM or a
human. Policy lives here, in code, not in the model (CLAUDE.md design rule 4).

Confidence is the response's confidence field for choice and score questions
(1 - H(p) / ln K), and max(p, 1 - p) for noul questions, which have no confidence
field (API_SPEC section 3; the same rule the coverage metrics use).

Every question and the threshold are defined in this file so they can be read in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_CHECKPOINT = "hf://yuanphd/jevmark/sft_06b"
DEFAULT_THRESHOLD = 0.9

MESSAGE = "Hi, I was charged twice for my March invoice and the export button still crashes every time I click it. I need this sorted today, please."

QUESTIONS: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "Charges, invoices, refunds and payment problems",
            "technical": "Bugs, crashes, outages and integration problems",
            "account": "Login, profile and account settings",
            "other": None,
        },
    },
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "needs_human": {"type": "noul", "instructions": "Does this message need a human agent rather than an automated reply?"},
    "severity": {
        "type": "score",
        "instructions": "How severe is the reported issue?",
        "criteria": [
            "Cosmetic; no impact on functionality",
            "Broken or degraded feature, but a workaround exists",
            "Blocking issue; no workaround exists",
        ],
    },
    "next_tool": {
        "type": "choice",
        "instructions": "Which tool should the support agent call first?",
        "criteria": {
            "lookup_invoice": "Fetch the customer's invoices and payment history",
            "open_bug_ticket": "File a bug report with the engineering team",
            "search_help_center": "Search help articles for a known answer",
        },
    },
}


def confidence(answer: Mapping[str, Any]) -> float:
    """The confidence a caller thresholds: the response field for choice and score, max(p, 1 - p) for noul."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1.0 - answer["noul"])
    return answer["confidence"]


def summary(answer: Mapping[str, Any]) -> str:
    if answer["type"] == "noul":
        return f"{'yes' if answer['noul'] >= 0.5 else 'no'} (P(yes) {answer['noul']:.4f})"
    if answer["type"] == "choice":
        return f"{answer['choice']} (p {answer['probabilities'][answer['choice']]:.4f})"
    return f"level {answer['score']:.2f} of 0 to {len(answer['legend']) - 1}"


def route(response: Mapping[str, Any], threshold: float) -> tuple[list[str], list[str]]:
    """Question ids acted on automatically (confidence >= threshold) and escalated (below it), in request order."""
    automatic, escalated = [], []
    for qid, answer in response["answers"].items():
        (automatic if confidence(answer) >= threshold else escalated).append(qid)
    return automatic, escalated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="a run directory or hf://<owner>/<repo>/<folder>; hf://yuanphd/jevmark/rlcd_banking77_06b is the Banking77 adapter")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="act automatically at or above this confidence, escalate below it")
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")

    os.environ["JEVMARK_CHECKPOINT"] = args.checkpoint  # read once, at the default model's first use
    from jevmark.systemone import default_model, systemone

    started = time.perf_counter()
    model = default_model()
    loaded = time.perf_counter()
    response = systemone(MESSAGE, QUESTIONS)
    answered = time.perf_counter()

    print(f"message: {MESSAGE}")
    print(f"checkpoint {args.checkpoint}, model {response['model']} on {model.device}")
    print(json.dumps(response, indent=2))
    automatic, escalated = route(response, args.threshold)
    print(f"\nconfidence threshold {args.threshold}:")
    for qid, answer in response["answers"].items():
        action = "automatic" if qid in automatic else "escalate "
        print(f"  {action}  {qid:17} {summary(answer):34} confidence {confidence(answer):.4f}")
    print(f"\n{len(automatic)} acted on automatically, {len(escalated)} escalated")
    print(f"model load {loaded - started:.1f} s; the five-question call {1000 * (answered - loaded):.0f} ms on {model.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
