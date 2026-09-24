"""Option descriptions: generated files plus overrides, normalised (docs/DATA.md sections 6 and 7).

The generated JSON files are never edited. Corrections live in overrides.json,
one entry per label with a reason, and replace the generated entry whole. Every
text, generated or overridden, is normalised the same way: one trailing period
is stripped and the curly apostrophe becomes a straight one. The result must
still be a valid option description under API_SPEC section 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevmark.schema import ChoiceQuestion

DESCRIPTIONS_DIR = Path(__file__).resolve().parent / "descriptions"
OVERRIDES_FILE = "overrides.json"
FILES = {
    "clinc": "clinc_intents.json",
    "banking77": "banking77_labels.json",
    "ag_news": "ag_news_labels.json",
    "emotion": "emotion_labels.json",
}
OVERRIDE_FIELDS = ("canonical", "paraphrases", "reason")
CURLY_APOSTROPHE = "’"


@dataclass(frozen=True)
class LabelDescription:
    canonical: str
    paraphrases: tuple[str, ...]
    overridden: bool = False

    @property
    def variants(self) -> tuple[str, ...]:
        return (self.canonical, *self.paraphrases)


def normalise(text: str) -> str:
    """Replace the curly apostrophe with a straight one and strip one trailing period."""
    text = text.replace(CURLY_APOSTROPHE, "'")
    return text[:-1] if text.endswith(".") else text


def _checked(texts: tuple[str, ...], where: str) -> tuple[str, ...]:
    """Normalise, then validate as option descriptions; raises ValueError naming the entry."""
    normalised = tuple(normalise(t) for t in texts)
    try:
        ChoiceQuestion(id="description", instructions="Check.", options=tuple((f"o{i}", t) for i, t in enumerate(normalised)) + (("x", None),))
    except ValueError as err:
        raise ValueError(f"{where}: {err}") from None
    return normalised


def _generated(key: str, directory: Path) -> dict[str, dict[str, Any]]:
    return json.loads((directory / FILES[key]).read_text())["descriptions"]


def load_overrides(directory: Path = DESCRIPTIONS_DIR) -> dict[str, dict[str, dict[str, Any]]]:
    """overrides.json, validated against the generated files; raises ValueError on any bad entry."""
    overrides = json.loads((directory / OVERRIDES_FILE).read_text())
    unknown = sorted(set(overrides) - set(FILES))
    if unknown:
        raise ValueError(f"overrides: unknown dataset keys {unknown}; expected {sorted(FILES)}")
    for key, entries in overrides.items():
        generated = _generated(key, directory)
        for label, entry in entries.items():
            where = f"overrides.{key}.{label}"
            if label not in generated:
                raise ValueError(f"{where}: label {label!r} is not in {FILES[key]}")
            if set(entry) != set(OVERRIDE_FIELDS):
                raise ValueError(f"{where}: fields must be exactly {list(OVERRIDE_FIELDS)}, got {sorted(entry)}")
            if not isinstance(entry["reason"], str) or not entry["reason"].strip():
                raise ValueError(f"{where}: reason must be a non-empty string")
            expected = len(generated[label]["paraphrases"])
            if not isinstance(entry["paraphrases"], list) or len(entry["paraphrases"]) != expected:
                raise ValueError(f"{where}: needs exactly {expected} paraphrases, like the generated entry")
            _checked((entry["canonical"], *entry["paraphrases"]), where)
    return overrides


def load_descriptions(key: str, directory: Path = DESCRIPTIONS_DIR) -> dict[str, LabelDescription]:
    """Label to description for one dataset, in the generated file's label order."""
    overrides = load_overrides(directory).get(key, {})
    descriptions = {}
    for label, entry in _generated(key, directory).items():
        source = overrides.get(label, entry)
        texts = _checked((source["canonical"], *source["paraphrases"]), f"{FILES[key]}.{label}")
        descriptions[label] = LabelDescription(texts[0], texts[1:], overridden=label in overrides)
    return descriptions
