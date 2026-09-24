"""Negation templates for noul questions, one pair of phrasings per noul kind (decision 40).

Each kind has a positive and a negated phrasing of the fixed part of its
instruction; the rest (the domain phrase) is kept. negate() maps either phrasing
to the other, so negate(negate(q)) == q:

    "Is this message about travel, such as flights, ...?"
    <-> "Is this message about something other than travel, such as flights, ...?"

Data v1.1 stores half of all noul questions in the negated phrasing with the gold
answer flipped, and the symmetry evaluation negates whichever phrasing a record
holds. A consistent model gives P(yes | q) + P(yes | not q) close to 1.
"""

from __future__ import annotations

from jevmark.data.clinc import DOMAIN_INSTRUCTIONS, OUT_OF_SCOPE_INSTRUCTIONS

# kind -> (positive phrasing of the fixed part, negated phrasing of the fixed part)
NEGATIONS = {
    "about_domain": (DOMAIN_INSTRUCTIONS.split("{phrase}")[0], "Is this message about something other than "),
    "out_of_scope": (OUT_OF_SCOPE_INSTRUCTIONS, "Is this request something a banking, travel, home, work or everyday assistant can help with?"),
}


def _forms(kind: str) -> tuple[str, str]:
    if kind not in NEGATIONS:
        raise ValueError(f"no negation template for noul kind {kind!r}")
    return NEGATIONS[kind]


def is_negated(kind: str, instructions: str) -> bool:
    """True for the negated phrasing, False for the positive one; ValueError if neither fits."""
    positive, negated = _forms(kind)
    # The about_domain positive prefix is itself a prefix of the negated one, so check the negated form first.
    if instructions.startswith(negated):
        return True
    if instructions.startswith(positive):
        return False
    raise ValueError(f"{kind}: instruction {instructions!r} fits neither phrasing")


def negate(kind: str, instructions: str) -> str:
    """The instruction in the other phrasing; ValueError for an unknown kind or an instruction that fits neither."""
    positive, negated = _forms(kind)
    if is_negated(kind, instructions):
        return positive + instructions[len(negated) :]
    return negated + instructions[len(positive) :]
