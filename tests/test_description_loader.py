"""Description loader: generated files, overrides and normalisation (docs/DATA.md sections 6 and 7)."""

import json
import shutil
from pathlib import Path

import pytest

from jevmark.data.description_loader import (
    DESCRIPTIONS_DIR,
    FILES,
    load_descriptions,
    load_overrides,
    normalise,
)
from jevmark.schema import ChoiceQuestion

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Messages about lost luggage.", "Messages about lost luggage"),
        ("Ends with two periods..", "Ends with two periods."),
        ("No period here", "No period here"),
        ("A user’s account", "A user's account"),
        ("A user's account", "A user's account"),
        ("The user’s card.", "The user's card"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_normalise_is_idempotent_for_a_single_period():
    assert normalise(normalise("Done.")) == "Done"


@pytest.mark.parametrize("key, count, paraphrases", [("clinc", 150, 2), ("banking77", 77, 0), ("ag_news", 4, 0), ("emotion", 6, 0)])
def test_real_files_load_normalised_and_valid(key, count, paraphrases):
    descriptions = load_descriptions(key)
    assert len(descriptions) == count
    for label, entry in descriptions.items():
        assert len(entry.paraphrases) == paraphrases
        for text in entry.variants:
            assert not text.endswith(".") and "’" not in text, (label, text)
        # Every variant must be a valid option description under API_SPEC section 2.
        ChoiceQuestion(id="q", instructions="Which?", options=tuple((f"o{i}", t) for i, t in enumerate(entry.variants)) + (("x", None),))


def test_normalisation_fixes_known_raw_cases():
    clinc = load_descriptions("clinc")
    assert clinc["account_blocked"].canonical == "Messages about a user's account being blocked or inaccessible"
    assert clinc["lost_luggage"].canonical == "Messages about lost or missing luggage during travel"


def test_committed_overrides_file_has_the_documented_shape():
    overrides = json.loads((DESCRIPTIONS_DIR / "overrides.json").read_text())
    assert set(overrides) == set(FILES)
    load_overrides()  # validates every entry against the generated files


def test_every_override_is_listed_in_data_md():
    data_md = (REPO / "docs" / "DATA.md").read_text()
    section = data_md.split("## 7. Description overrides and normalisation")[1]
    for key, entries in load_overrides().items():
        for label, entry in entries.items():
            assert f"`{key}/{label}`" in section, f"{key}/{label} missing from DATA.md section 7"
            assert entry["reason"] in section


# Overrides in a scratch directory


@pytest.fixture
def scratch(tmp_path):
    for filename in FILES.values():
        shutil.copy(DESCRIPTIONS_DIR / filename, tmp_path / filename)
    return tmp_path


def write_overrides(directory, entries):
    overrides = {key: {} for key in FILES}
    for key, value in entries.items():
        overrides[key] = value
    (directory / "overrides.json").write_text(json.dumps(overrides))


def test_override_replaces_the_entry_and_is_normalised(scratch):
    write_overrides(
        scratch,
        {"clinc": {"meeting_schedule": {
            "canonical": "Asks which meetings are already on the user’s schedule.",
            "paraphrases": ["Wants to know what meetings are planned", "Checks the list of upcoming meetings"],
            "reason": "generated text described arranging a meeting, which is schedule_meeting",
        }}},
    )
    clinc = load_descriptions("clinc", scratch)
    entry = clinc["meeting_schedule"]
    assert entry.canonical == "Asks which meetings are already on the user's schedule"
    assert entry.paraphrases == ("Wants to know what meetings are planned", "Checks the list of upcoming meetings")
    assert entry.overridden
    assert not clinc["schedule_meeting"].overridden
    assert len(clinc) == 150


def entry(canonical="Fine text", paraphrases=("First variant", "Second variant"), reason="a reason"):
    return {"canonical": canonical, "paraphrases": list(paraphrases), "reason": reason}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"clinc": {"no_such_intent": entry()}}, "no_such_intent"),
        ({"clinc": {"transfer": entry(paraphrases=("only one",))}}, "paraphrases"),
        ({"emotion": {"joy": entry()}}, "paraphrases"),
        ({"clinc": {"transfer": entry(reason="")}}, "reason"),
        ({"clinc": {"transfer": {"canonical": "x y z", "paraphrases": ["a b", "c d"]}}}, "reason"),
        ({"clinc": {"transfer": entry(canonical="two\nlines")}}, "line break"),
        ({"clinc": {"transfer": entry(canonical=" edge")}}, "whitespace"),
        ({"clinc": {"transfer": {**entry(), "extra": 1}}}, "extra"),
    ],
)
def test_invalid_overrides_fail_loudly(scratch, overrides, message):
    write_overrides(scratch, overrides)
    with pytest.raises(ValueError, match=message):
        load_descriptions("clinc", scratch)


def test_unknown_dataset_key_in_overrides_fails(scratch):
    (scratch / "overrides.json").write_text(json.dumps({**{k: {} for k in FILES}, "imdb": {}}))
    with pytest.raises(ValueError, match="imdb"):
        load_descriptions("emotion", scratch)


def test_normalisation_that_leaves_bad_text_fails(scratch):
    write_overrides(scratch, {"emotion": {"joy": entry(canonical="Ends with space .", paraphrases=())}})
    with pytest.raises(ValueError, match="whitespace"):
        load_descriptions("emotion", scratch)
