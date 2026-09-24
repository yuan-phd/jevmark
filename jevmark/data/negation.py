"""Noul phrasing templates and their negations, per noul kind (decisions 40 and 41).

Every kind has two or three templates. Each template is a pair of phrasings, a
positive one and its negation, around at most one slot (a domain phrase, an intent
description, an emotion). negate() maps any phrasing of any template to its pair
in both directions, so negate(negate(q)) == q, and parse() recovers the template,
the polarity and the slot from a rendered instruction.

The builders render every noul question from these templates, and the symmetry
evaluation negates whichever phrasing a record holds. A consistent model gives
P(yes | q) + P(yes | not q) close to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

ASSISTANT = "a banking, travel, home, work or everyday assistant"

# kind -> list of (positive, negated) templates; "{slot}" marks the slot, if any.
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "about_domain": [
        ("Is this message about {slot}?", "Is this message about something other than {slot}?"),
        ("Does this message concern {slot}?", "Does this message concern something other than {slot}?"),
        ("Is the topic of this message {slot}?", "Is the topic of this message something other than {slot}?"),
    ],
    "out_of_scope": [
        (f"Is this request outside what {ASSISTANT} can help with?", f"Is this request something {ASSISTANT} can help with?"),
        (f"Would {ASSISTANT} be unable to help with this request?", f"Would {ASSISTANT} be able to help with this request?"),
        (f"Is this request out of scope for {ASSISTANT}?", f"Is this request within scope for {ASSISTANT}?"),
    ],
    "about_intent": [
        ("Does this description fit the message: {slot}?", "Does this description fail to fit the message: {slot}?"),
        ("Is this message an example of the following: {slot}?", "Is this message an example of something other than the following: {slot}?"),
    ],
    "is_positive": [
        ("Is the sentiment of this text positive?", "Is the sentiment of this text something other than positive?"),
        ("Does this text express a favourable opinion?", "Does this text express anything other than a favourable opinion?"),
    ],
    "is_negative": [
        ("Is the sentiment of this text negative?", "Is the sentiment of this text something other than negative?"),
        ("Does this text express an unfavourable opinion?", "Does this text express anything other than an unfavourable opinion?"),
    ],
    "expresses_emotion": [
        ("Does this message express {slot}?", "Does this message express something other than {slot}?"),
        ("Is {slot} the main emotion in this message?", "Is something other than {slot} the main emotion in this message?"),
    ],
}
KINDS = tuple(TEMPLATES)


@dataclass(frozen=True)
class Phrasing:
    template: int
    negated: bool
    slot: str | None


def render(kind: str, template: int, negated: bool, slot: str | None = None) -> str:
    """The instruction for a kind, template index and polarity, with the slot filled in."""
    pattern = _templates(kind)[template][1 if negated else 0]
    if ("{slot}" in pattern) != (slot is not None):
        raise ValueError(f"{kind} template {template}: slot {'required' if '{slot}' in pattern else 'not allowed'}")
    return pattern.replace("{slot}", slot) if slot is not None else pattern


def parse(kind: str, instructions: str) -> Phrasing:
    """Template, polarity and slot of a rendered instruction; ValueError if no phrasing of the kind fits.

    When several patterns fit (a negated phrasing often extends its positive one),
    the pattern with the most fixed text wins.
    """
    matches = []
    for index, pair in enumerate(_templates(kind)):
        for negated, pattern in ((False, pair[0]), (True, pair[1])):
            prefix, _, suffix = pattern.partition("{slot}")
            if "{slot}" not in pattern:
                if instructions == pattern:
                    matches.append((len(pattern), Phrasing(index, negated, None)))
            elif instructions.startswith(prefix) and instructions.endswith(suffix) and len(instructions) > len(prefix) + len(suffix):
                slot = instructions[len(prefix) : len(instructions) - len(suffix)]
                matches.append((len(prefix) + len(suffix), Phrasing(index, negated, slot)))
    if not matches:
        raise ValueError(f"{kind}: instruction {instructions!r} fits no phrasing")
    return max(matches, key=lambda m: m[0])[1]


def is_negated(kind: str, instructions: str) -> bool:
    return parse(kind, instructions).negated


def negate(kind: str, instructions: str) -> str:
    """The same template in the other polarity, with the same slot."""
    phrasing = parse(kind, instructions)
    return render(kind, phrasing.template, not phrasing.negated, phrasing.slot)


def _templates(kind: str) -> list[tuple[str, str]]:
    if kind not in TEMPLATES:
        raise ValueError(f"no negation template for noul kind {kind!r}")
    return TEMPLATES[kind]
