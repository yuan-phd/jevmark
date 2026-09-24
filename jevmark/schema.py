"""Request and response dataclasses with validation (docs/API_SPEC.md sections 2, 3, 7).

Every validation error is a ValueError whose message starts with the offending
field path, "<question_id>.<field>: <reason>", or "state: ..." / "questions: ..."
for request-level fields. The encoded-length limit needs a tokenizer and is
checked in encode.py.

Dataclasses are frozen and validate in __post_init__, so a question built directly
in Python is held to the same rules as one parsed with from_dict.

Every text that becomes part of a rendered line (instructions, option labels and
descriptions, score levels) must be a single line with no leading or trailing
whitespace (decision 25). The state is free text and may contain newlines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Union

QUESTION_ID_PATTERN = re.compile(r"[a-z0-9_]+")
MAX_STATE_CHARS = 8000
MIN_CHOICE_OPTIONS, MAX_CHOICE_OPTIONS = 2, 26
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
QUESTION_FIELDS = ("type", "instructions", "criteria")
NOUL_CRITERIA_KEYS = ("true", "false")

State = Union[str, dict[str, Any], list[Any]]


def _fail(path: str, reason: str) -> ValueError:
    return ValueError(f"{path}: {reason}")


def render_state(state: State) -> str:
    """State text as it appears in the encoding: strings as is, dicts and lists as pretty JSON."""
    if isinstance(state, str):
        return state
    if not isinstance(state, (dict, list)):
        raise _fail("state", f"must be a string, object or array, got {type(state).__name__}")
    try:
        return json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise _fail("state", f"not JSON-serialisable ({err})") from None


def _check_line(path: str, what: str, text: str) -> None:
    if "\n" in text or "\r" in text:
        raise _fail(path, f"{what} must not contain line breaks, got {text!r}")
    if text != text.strip():
        raise _fail(path, f"{what} must not have leading or trailing whitespace, got {text!r}")


def _check_instructions(qid: str, instructions: Any) -> None:
    path = f"{qid}.instructions"
    if not isinstance(instructions, str) or not instructions.strip():
        raise _fail(path, "must be a non-empty string")
    _check_line(path, "instructions", instructions)


def _check_description(path: str, description: Any) -> None:
    if description is None:
        return
    if not isinstance(description, str):
        raise _fail(path, f"descriptions must be a string or null, got {type(description).__name__}")
    _check_line(path, "descriptions", description)


# Questions


@dataclass(frozen=True)
class NoulQuestion:
    id: str
    instructions: str
    true_description: str | None = None
    false_description: str | None = None
    type: Literal["noul"] = "noul"

    def __post_init__(self) -> None:
        _check_instructions(self.id, self.instructions)
        _check_description(f"{self.id}.criteria", self.true_description)
        _check_description(f"{self.id}.criteria", self.false_description)

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> NoulQuestion:
        criteria = d.get("criteria")
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise _fail(f"{qid}.criteria", "must be an object with optional keys 'true' and 'false'")
        unknown = [k for k in criteria if k not in NOUL_CRITERIA_KEYS]
        if unknown:
            raise _fail(f"{qid}.criteria", f"unknown keys {unknown}; only 'true' and 'false' are allowed")
        return cls(
            id=qid,
            instructions=d.get("instructions"),
            true_description=criteria.get("true"),
            false_description=criteria.get("false"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        criteria = {
            key: value
            for key, value in (("true", self.true_description), ("false", self.false_description))
            if value is not None
        }
        if criteria:
            out["criteria"] = criteria
        return out


@dataclass(frozen=True)
class ChoiceQuestion:
    id: str
    instructions: str
    options: tuple[tuple[str, str | None], ...]
    type: Literal["choice"] = "choice"

    def __post_init__(self) -> None:
        _check_instructions(self.id, self.instructions)
        path = f"{self.id}.criteria"
        options = tuple(tuple(pair) for pair in self.options)
        object.__setattr__(self, "options", options)
        if not MIN_CHOICE_OPTIONS <= len(options) <= MAX_CHOICE_OPTIONS:
            raise _fail(path, f"needs {MIN_CHOICE_OPTIONS} to {MAX_CHOICE_OPTIONS} options, got {len(options)}")
        seen: set[str] = set()
        for label, description in options:
            if not isinstance(label, str) or not label.strip():
                raise _fail(path, f"option labels must be non-empty strings, got {label!r}")
            _check_line(path, "option labels", label)
            _check_description(path, description)
            if label in seen:
                raise _fail(path, f"duplicate option label {label!r}")
            seen.add(label)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.options)

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> ChoiceQuestion:
        criteria = d.get("criteria")
        if not isinstance(criteria, dict):
            raise _fail(f"{qid}.criteria", "must be an object mapping option label to description or null")
        return cls(id=qid, instructions=d.get("instructions"), options=tuple(criteria.items()))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass(frozen=True)
class ScoreQuestion:
    id: str
    instructions: str
    levels: tuple[str, ...]
    type: Literal["score"] = "score"

    def __post_init__(self) -> None:
        _check_instructions(self.id, self.instructions)
        path = f"{self.id}.criteria"
        levels = tuple(self.levels)
        object.__setattr__(self, "levels", levels)
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise _fail(path, f"needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(levels)}")
        for level in levels:
            if not isinstance(level, str):
                raise _fail(path, f"level descriptions must be strings, got {type(level).__name__}")
            _check_line(path, "level descriptions", level)

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> ScoreQuestion:
        criteria = d.get("criteria")
        if not isinstance(criteria, list):
            raise _fail(f"{qid}.criteria", "must be a list of level descriptions, low to high")
        return cls(id=qid, instructions=d.get("instructions"), levels=tuple(criteria))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": list(self.levels)}


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]
QUESTION_TYPES: dict[str, type[NoulQuestion] | type[ChoiceQuestion] | type[ScoreQuestion]] = {
    "noul": NoulQuestion,
    "choice": ChoiceQuestion,
    "score": ScoreQuestion,
}


def question_from_dict(qid: str, d: Any) -> Question:
    if not isinstance(d, dict):
        raise _fail(qid, "question definition must be an object")
    for field in d:
        if field not in QUESTION_FIELDS:
            raise _fail(f"{qid}.{field}", f"unknown field; allowed fields are {list(QUESTION_FIELDS)}")
    if "type" not in d:
        raise _fail(f"{qid}.type", "required")
    qtype = d["type"]
    if not isinstance(qtype, str) or qtype not in QUESTION_TYPES:
        raise _fail(f"{qid}.type", f"unknown type {qtype!r}; expected one of {list(QUESTION_TYPES)}")
    if "instructions" not in d:
        raise _fail(f"{qid}.instructions", "required")
    return QUESTION_TYPES[qtype].from_dict(qid, d)


# Request


@dataclass(frozen=True)
class Request:
    state: State
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        rendered = render_state(self.state)
        if len(rendered) > MAX_STATE_CHARS:
            raise _fail("state", f"{len(rendered)} characters after rendering; the limit is {MAX_STATE_CHARS}")
        questions = tuple(self.questions)
        object.__setattr__(self, "questions", questions)
        if not questions:
            raise _fail("questions", "at least one question is required")
        seen: set[str] = set()
        for question in questions:
            if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
                raise _fail("questions", f"expected question objects, got {type(question).__name__}")
            _check_question_id(question.id)
            if question.id in seen:
                raise _fail("questions", f"duplicate question id {question.id!r}")
            seen.add(question.id)

    @property
    def state_text(self) -> str:
        return render_state(self.state)

    @classmethod
    def from_dict(cls, d: Any) -> Request:
        if not isinstance(d, dict):
            raise _fail("request", "must be an object with 'state' and 'questions'")
        if "state" not in d:
            raise _fail("state", "required")
        # Validate the state before the questions so a bad state is reported first.
        render_state(d["state"])
        questions = d.get("questions")
        if not isinstance(questions, dict):
            raise _fail("questions", "must be an object mapping question id to definition")
        for qid in questions:
            _check_question_id(qid)
        return cls(
            state=d["state"],
            questions=tuple(question_from_dict(qid, definition) for qid, definition in questions.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "questions": {q.id: q.to_dict() for q in self.questions}}


def _check_question_id(qid: Any) -> None:
    if not isinstance(qid, str) or not QUESTION_ID_PATTERN.fullmatch(qid):
        raise _fail("questions", f"invalid question id {qid!r}; ids must match [a-z0-9_]+")


# Answers and response


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise _fail(f"{path}.{key}", "required")
    return d[key]


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: Literal["noul"] = "noul"

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> NoulAnswer:
        return cls(noul=_require(d, "noul", qid))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "noul": self.noul}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: Literal["choice"] = "choice"

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> ChoiceAnswer:
        return cls(
            choice=_require(d, "choice", qid),
            probabilities=dict(_require(d, "probabilities", qid)),
            confidence=_require(d, "confidence", qid),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    type: Literal["score"] = "score"

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> ScoreAnswer:
        return cls(
            score=_require(d, "score", qid),
            legend=dict(_require(d, "legend", qid)),
            probabilities=dict(_require(d, "probabilities", qid)),
            confidence=_require(d, "confidence", qid),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "score": self.score,
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]
ANSWER_TYPES: dict[str, type[NoulAnswer] | type[ChoiceAnswer] | type[ScoreAnswer]] = {
    "noul": NoulAnswer,
    "choice": ChoiceAnswer,
    "score": ScoreAnswer,
}


def answer_from_dict(qid: str, d: Any) -> Answer:
    if not isinstance(d, dict):
        raise _fail(qid, "answer must be an object")
    atype = _require(d, "type", qid)
    if atype not in ANSWER_TYPES:
        raise _fail(f"{qid}.type", f"unknown type {atype!r}; expected one of {list(ANSWER_TYPES)}")
    return ANSWER_TYPES[atype].from_dict(qid, d)


@dataclass(frozen=True)
class Usage:
    input_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"input_tokens": self.input_tokens}


@dataclass(frozen=True)
class Response:
    model: str
    answers: dict[str, Answer]
    usage: Usage

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Response:
        answers = _require(d, "answers", "response")
        usage = _require(d, "usage", "response")
        return cls(
            model=_require(d, "model", "response"),
            answers={qid: answer_from_dict(qid, a) for qid, a in answers.items()},
            usage=Usage(input_tokens=_require(usage, "input_tokens", "usage")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {qid: a.to_dict() for qid, a in self.answers.items()},
            "usage": self.usage.to_dict(),
        }
