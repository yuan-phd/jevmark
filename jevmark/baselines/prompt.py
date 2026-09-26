"""Prompt and reply parser shared by the generative baselines B1 and B2 (task 1.8).

build_prompt renders the same state and the same questions jevmark reads
(instructions, choice options with their descriptions, noul descriptions, score
levels), in request order, and asks for one JSON object that maps every question
id to {"answer": ..., "confidence": ...}:

- noul: the JSON boolean true or false (the strings "true" and "false" are accepted)
- choice: one option label, exactly as written
- score: the level index, an integer (an integer-valued number or string is accepted)
- confidence: optional, a number from 0 to 1, the model's probability that its
  answer is correct

parse_reply is the one parser for both baselines. Each question gets a status:
"ok", or a parse failure recorded apart from wrong answers:

- "invalid_json": the reply is not one JSON object (after stripping whitespace and
  at most one surrounding Markdown code fence); every question of the request fails
- "missing": the object has no entry for the question id
- "invalid_answer": the entry has no answer, or the answer is outside the allowed set

String answers are compared after stripping surrounding whitespace. A confidence that is missing, not a number, or outside [0, 1] is recorded as None;
the answer still counts. A bare answer in place of the object (for example
{"intent": "billing"}) is accepted as an answer without a confidence. Keys for
unknown question ids are ignored.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from jevmark.schema import ChoiceQuestion, NoulQuestion, Question, Request, ScoreQuestion

STATUSES = ("ok", "invalid_json", "missing", "invalid_answer")
FAILURES = STATUSES[1:]

INTRO = "Read the state below and answer every question about it."
TYPE_LINES = {
    "noul": ("Type: yes or no", "Answer with the JSON boolean true or false."),
    "choice": ("Type: choice", "Answer with exactly one option label, written exactly as listed."),
    "score": ("Type: score", "Answer with the level number as an integer."),
}
OUTRO = (
    "Reply with one JSON object and nothing else. It must map every question id to an object with two fields: "
    '"answer", your answer in the format given for that question, and "confidence", a number from 0 to 1 '
    "giving the probability that your answer is correct."
)
FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n\s*```\Z", re.S)


def _question_block(question: Question) -> str:
    kind = question.type
    type_line, answer_line = TYPE_LINES[kind]
    lines = [f"Question id: {question.id}", type_line, f"Question: {question.instructions}"]
    if isinstance(question, NoulQuestion):
        lines += ["Options:", f"- true: {question.true_description or 'yes'}", f"- false: {question.false_description or 'no'}"]
    elif isinstance(question, ChoiceQuestion):
        lines.append("Options:")
        lines += [f"- {label}" if description is None else f"- {label}: {description}" for label, description in question.options]
    else:
        lines.append("Levels, from low to high:")
        lines += [f"- {index}: {level}" for index, level in enumerate(question.levels)]
    lines.append(answer_line)
    return "\n".join(lines)


def build_prompt(request: Request) -> str:
    """The user message for one request: state, questions in request order, reply format."""
    example = ", ".join(f'"{q.id}": {{"answer": ..., "confidence": ...}}' for q in request.questions)
    parts = [INTRO, f"### State\n{request.state_text}", "### Questions", *(_question_block(q) for q in request.questions), OUTRO, f"Format: {{{example}}}"]
    return "\n\n".join(parts)


def json_schema(request: Request) -> dict[str, Any]:
    """A strict JSON schema for the reply (B2's response format): one entry per question, answers restricted to the allowed set.

    Strict structured outputs require every property, so confidence is required
    here; the parser still treats it as optional.
    """

    def answer(question: Question) -> dict[str, Any]:
        if isinstance(question, NoulQuestion):
            return {"type": "boolean"}
        if isinstance(question, ChoiceQuestion):
            return {"type": "string", "enum": list(question.labels)}
        return {"type": "integer", "enum": list(range(len(question.levels)))}

    properties = {
        q.id: {
            "type": "object",
            "properties": {"answer": answer(q), "confidence": {"type": "number", "description": "probability from 0 to 1 that the answer is correct"}},
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        }
        for q in request.questions
    }
    return {"type": "object", "properties": properties, "required": [q.id for q in request.questions], "additionalProperties": False}


@dataclass(frozen=True)
class ParsedAnswer:
    status: str  # one of STATUSES
    answer: int | None = None  # option index in request order (noul: 0 true, 1 false; score: the level)
    confidence: float | None = None


def _answer_index(question: Question, value: Any) -> int | None:
    if isinstance(question, NoulQuestion):
        if isinstance(value, bool):
            return 0 if value else 1
        if isinstance(value, str) and value.strip() in ("true", "false"):
            return 0 if value.strip() == "true" else 1
        return None
    if isinstance(question, ChoiceQuestion):
        if isinstance(value, str) and value.strip() in question.labels:
            return question.labels.index(value.strip())
        return None
    assert isinstance(question, ScoreQuestion)
    level: Any = value
    if isinstance(level, str):
        try:
            level = float(level.strip())
        except ValueError:
            return None
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not math.isfinite(level) or level != int(level):
        return None
    level = int(level)
    return level if 0 <= level < len(question.levels) else None


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def reply_object(reply: str | None) -> dict[str, Any] | None:
    """The reply's JSON object, or None when the reply is not one."""
    if reply is None:
        return None
    text = reply.strip()
    fenced = FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_reply(reply: str | None, request: Request) -> dict[str, ParsedAnswer]:
    """One ParsedAnswer per question id, in request order. A reply of None (no reply at all) is invalid_json."""
    obj = reply_object(reply)
    if obj is None:
        return {q.id: ParsedAnswer("invalid_json") for q in request.questions}
    parsed = {}
    for question in request.questions:
        if question.id not in obj:
            parsed[question.id] = ParsedAnswer("missing")
            continue
        entry = obj[question.id]
        if isinstance(entry, dict):
            value, confidence = entry.get("answer"), _confidence(entry.get("confidence"))
        else:
            value, confidence = entry, None
        index = _answer_index(question, value)
        parsed[question.id] = ParsedAnswer("invalid_answer") if index is None else ParsedAnswer("ok", index, confidence)
    return parsed
