"""Negation templates for the noul symmetry test, one per noul kind (docs/TASKS.md 1.6).

A template rewrites the fixed part of a noul instruction and keeps the rest, so
"Is this message about travel, such as flights, ...?" becomes "Is this message about
something other than travel, such as flights, ...?". A consistent model gives
P(yes | q) + P(yes | not q) close to 1.
"""

from __future__ import annotations

from jevmark.data.clinc import DOMAIN_INSTRUCTIONS, OUT_OF_SCOPE_INSTRUCTIONS

# kind -> (text the instruction starts with, replacement for that text)
NEGATIONS = {
    "about_domain": (DOMAIN_INSTRUCTIONS.split("{phrase}")[0], "Is this message about something other than "),
    "out_of_scope": (OUT_OF_SCOPE_INSTRUCTIONS, "Is this request something a banking, travel, home, work or everyday assistant can help with?"),
}


def negate(kind: str, instructions: str) -> str:
    """The negated instruction; raises ValueError for an unknown kind or an instruction the template does not fit."""
    if kind not in NEGATIONS:
        raise ValueError(f"no negation template for noul kind {kind!r}")
    prefix, replacement = NEGATIONS[kind]
    if not instructions.startswith(prefix):
        raise ValueError(f"{kind}: instruction {instructions!r} does not start with {prefix!r}")
    return replacement + instructions[len(prefix) :]
