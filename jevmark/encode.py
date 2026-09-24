"""Request -> token ids with answer slots (docs/API_SPEC.md section 4).

The text is built from segments: the state block, then for each question a
separator "\\n\\n" and the question block ending in "Answer:". Each segment is
tokenized on its own and the ids are concatenated, so "Answer:" always ends on a
token boundary (decision 17). Training and inference both use `encode`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from jevmark.schema import ChoiceQuestion, NoulQuestion, Question, Request, ScoreQuestion

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEPARATOR = "\n\n"

# Covers the encoding format, digits, JSON punctuation, non-ASCII text and every
# letter token read at a slot. Two tokenizers that agree here agree on encode().
PARITY_PROBE = (
    "### State\n"
    '{\n  "customer": "Zoë Müller",\n  "amount": 1234.50,\n  "note": "東京 café, naïve 🙂"\n}'
    "\n\n### Question: Which team should handle this message?\n"
    "Options:\nA. billing: Charges, invoices, refunds\nB. other\nAnswer:"
    "\n\n### Question: Is it urgent?\nOptions:\nA. true: yes\nB. false: no\nAnswer:"
    " A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
)


class TokenizerError(RuntimeError):
    """A tokenizer assumption the encoding depends on does not hold."""


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Encoded:
    input_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    letter_ids: tuple[tuple[int, ...], ...]
    question_ids: tuple[str, ...]


# Rendering


def option_lines(question: Question) -> list[tuple[str, str | None]]:
    """(label, description) per option in letter order; description None means no colon."""
    if isinstance(question, NoulQuestion):
        return [("true", question.true_description or "yes"), ("false", question.false_description or "no")]
    if isinstance(question, ChoiceQuestion):
        return list(question.options)
    if isinstance(question, ScoreQuestion):
        return [(str(index), level) for index, level in enumerate(question.levels)]
    raise TypeError(f"not a question: {type(question).__name__}")


def render_question(question: Question) -> str:
    lines = [f"### Question: {question.instructions}", "Options:"]
    for letter, (label, description) in zip(LETTERS, option_lines(question)):
        lines.append(f"{letter}. {label}" if description is None else f"{letter}. {label}: {description}")
    lines.append("Answer:")
    return "\n".join(lines)


def render_segments(request: Request) -> list[str]:
    segments = [f"### State\n{request.state_text}"]
    for question in request.questions:
        segments += [SEPARATOR, render_question(question)]
    return segments


def render_text(request: Request) -> str:
    return "".join(render_segments(request))


# Tokenizer checks


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def letter_token_ids(tokenizer: Tokenizer) -> tuple[int, ...]:
    """Ids of " A" to " Z". Raises TokenizerError unless each is exactly one token."""
    ids = []
    for letter in LETTERS:
        pieces = _ids(tokenizer, f" {letter}")
        if len(pieces) != 1:
            raise TokenizerError(f"' {letter}' is {len(pieces)} tokens {pieces}; the readout needs exactly one")
        ids.append(pieces[0])
    if len(set(ids)) != len(ids):
        raise TokenizerError(f"letter tokens are not distinct: {ids}")
    return tuple(ids)


def check_tokenizer_parity(tokenizer: Tokenizer, reference: Tokenizer, probe: str = PARITY_PROBE) -> None:
    """Raises TokenizerError unless both tokenizers give identical ids for the probe."""
    ids, reference_ids = _ids(tokenizer, probe), _ids(reference, probe)
    if ids != reference_ids:
        first = next(
            (i for i, (a, b) in enumerate(zip(ids, reference_ids)) if a != b), min(len(ids), len(reference_ids))
        )
        raise TokenizerError(
            f"tokenizer parity check failed: ids differ from the reference at position {first} "
            f"({len(ids)} vs {len(reference_ids)} ids)"
        )


def load_tokenizer(config: Mapping[str, Any]) -> PreTrainedTokenizerBase:
    """Load the backbone tokenizer and run the load-time checks.

    Checks that " A" to " Z" are single tokens and, when the backbone is not the
    pinned reference, that it tokenizes the parity probe identically.
    """
    backbone = config["backbone"]
    reference = config["tokenizer_reference"]
    tokenizer = AutoTokenizer.from_pretrained(backbone["id"], revision=backbone.get("revision"))
    letter_token_ids(tokenizer)
    if (backbone["id"], backbone.get("revision")) != (reference["id"], reference["revision"]):
        reference_tokenizer = AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])
        check_tokenizer_parity(tokenizer, reference_tokenizer)
    return tokenizer


# Encoding


def encode(request: Request, tokenizer: Tokenizer, max_tokens: int) -> Encoded:
    """Token ids, one slot per question at the last id of its "Answer:", and the letter ids to read there.

    Raises ValueError("max_tokens: ...") when the encoded length exceeds max_tokens,
    and TokenizerError if a slot id does not decode to a string ending in ":".
    """
    letters = letter_token_ids(tokenizer)
    state_block, *question_segments = render_segments(request)
    input_ids = _ids(tokenizer, state_block)
    slot_positions = []
    letter_ids = []
    # question_segments alternates separator, question block.
    for question, block in zip(request.questions, question_segments[1::2]):
        input_ids += _ids(tokenizer, SEPARATOR)
        input_ids += _ids(tokenizer, block)
        slot = len(input_ids) - 1
        slot_text = tokenizer.decode([input_ids[slot]])
        if not slot_text.endswith(":"):
            raise TokenizerError(
                f"{question.id}: slot id {input_ids[slot]} decodes to {slot_text!r}, not a string ending in ':'"
            )
        slot_positions.append(slot)
        letter_ids.append(letters[: len(option_lines(question))])
    if len(input_ids) > max_tokens:
        raise ValueError(f"max_tokens: encoded length {len(input_ids)} exceeds max_tokens {max_tokens}")
    return Encoded(
        input_ids=tuple(input_ids),
        slot_positions=tuple(slot_positions),
        letter_ids=tuple(letter_ids),
        question_ids=tuple(q.id for q in request.questions),
    )


# Training-time option shuffling


def shuffle_options(question: ChoiceQuestion, rng: random.Random) -> tuple[ChoiceQuestion, tuple[int, ...]]:
    """Shuffle a choice question's options.

    Returns the shuffled question and perm, where new position i holds old option
    perm[i]. An old gold index g moves to perm.index(g).
    """
    perm = list(range(len(question.options)))
    rng.shuffle(perm)
    return replace(question, options=tuple(question.options[i] for i in perm)), tuple(perm)


def shuffle_request(request: Request, rng: random.Random) -> tuple[Request, dict[str, tuple[int, ...]]]:
    """Shuffle every choice question; noul order is fixed and score levels are ordered, so both stay."""
    questions = []
    perms = {}
    for question in request.questions:
        if isinstance(question, ChoiceQuestion):
            shuffled, perms[question.id] = shuffle_options(question, rng)
            questions.append(shuffled)
        else:
            questions.append(question)
    return replace(request, questions=tuple(questions)), perms
