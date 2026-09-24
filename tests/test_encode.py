"""Encoding contract, docs/API_SPEC.md section 4."""

import copy
import json
import random

import pytest

from jevmark.encode import (
    LETTERS,
    PARITY_PROBE,
    SEPARATOR,
    TokenizerError,
    check_tokenizer_parity,
    encode,
    letter_token_ids,
    load_tokenizer,
    render_text,
    shuffle_options,
    shuffle_request,
)
from jevmark.schema import ChoiceQuestion, Request

STATE = "I was charged twice, please refund me."

QUESTIONS = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
        "criteria": {"true": "optional description of yes", "false": "optional description of no"},
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "Charges, invoices, refunds",
            "technical": "Bugs, outages, integration problems",
            "other": None,
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the reported issue?",
        "criteria": [
            "Cosmetic; no impact on functionality",
            "Broken or degraded feature, but a workaround exists",
            "Blocking issue; no workaround exists",
        ],
    },
}

GOLDEN = """### State
I was charged twice, please refund me.

### Question: Does the customer ask for money back?
Options:
A. true: optional description of yes
B. false: optional description of no
Answer:

### Question: Which team should handle this message?
Options:
A. billing: Charges, invoices, refunds
B. technical: Bugs, outages, integration problems
C. other
Answer:

### Question: How severe is the reported issue?
Options:
A. 0: Cosmetic; no impact on functionality
B. 1: Broken or degraded feature, but a workaround exists
C. 2: Blocking issue; no workaround exists
Answer:"""


def spec_request(state=STATE, **overrides) -> Request:
    questions = copy.deepcopy(QUESTIONS)
    questions.update(overrides)
    return Request.from_dict({"state": state, "questions": questions})


@pytest.fixture(scope="module")
def encoded(tokenizer):
    return encode(spec_request(), tokenizer, max_tokens=2048)


class FakeTokenizer:
    """Wraps the real tokenizer and replaces the ids for chosen strings."""

    def __init__(self, inner, overrides):
        self.inner = inner
        self.overrides = overrides

    def encode(self, text, add_special_tokens=False):
        if text in self.overrides:
            return list(self.overrides[text])
        return self.inner.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids):
        return self.inner.decode(ids)


# Rendering and tokenization


def test_rendered_text_matches_golden():
    assert render_text(spec_request()) == GOLDEN


def test_decoding_input_ids_reproduces_golden(tokenizer, encoded):
    assert tokenizer.decode(list(encoded.input_ids)) == GOLDEN


def test_question_ids_in_request_order(encoded):
    assert encoded.question_ids == ("refund_requested", "department", "severity")
    assert len(encoded.slot_positions) == len(encoded.letter_ids) == 3


def test_slot_ids_end_in_colon_and_next_id_is_separator_or_end(tokenizer, encoded):
    separator_ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    assert tokenizer.decode(separator_ids) == "\n\n"
    ids = list(encoded.input_ids)
    for slot in encoded.slot_positions:
        assert tokenizer.decode([ids[slot]]).endswith(":")
        following = ids[slot + 1 : slot + 1 + len(separator_ids)]
        assert following == separator_ids or slot == len(ids) - 1
    assert encoded.slot_positions[-1] == len(ids) - 1
    assert list(encoded.slot_positions) == sorted(encoded.slot_positions)


def test_letter_ids_are_leading_space_letters_one_per_option(tokenizer, encoded):
    expected = [tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in LETTERS]
    assert [len(ids) for ids in encoded.letter_ids] == [2, 3, 3]
    for ids in encoded.letter_ids:
        assert list(ids) == expected[: len(ids)]


def test_whole_text_tokenization_would_move_a_slot(tokenizer, encoded):
    """Why encoding is per segment (decision 17).

    Tokenizing GOLDEN in one call lets Qwen's pre-tokenizer merge the colon of
    "Answer:" with the blank line that follows it into one token, ":\\n\\n". The
    logits at that token predict what follows the blank line, not the answer, so
    every slot except the last would sit on the wrong token. Per-segment
    tokenization keeps ":" as a token of its own.
    """
    whole = tokenizer(GOLDEN, add_special_tokens=False, return_offsets_mapping=True)
    offsets = whole["offset_mapping"]
    colon_ends = []
    start = 0
    while (index := GOLDEN.find("Answer:", start)) != -1:
        colon_ends.append(index + len("Answer:"))
        start = index + 1
    assert len(colon_ends) == len(encoded.slot_positions)

    moved = []
    for end in colon_ends:
        token = next(i for i, (a, b) in enumerate(offsets) if a < end <= b)
        if offsets[token][1] != end:
            moved.append(tokenizer.decode([whole["input_ids"][token]]))
    assert moved, "whole-text tokenization no longer merges ':' with the separator"
    assert all(text == ":\n\n" for text in moved)
    assert len(moved) == len(colon_ends) - 1  # the last slot ends the text, so nothing follows it


# Option order


def test_option_order_changes_letter_assignment_not_letter_ids(tokenizer, encoded):
    reordered_criteria = {
        "other": None,
        "technical": "Bugs, outages, integration problems",
        "billing": "Charges, invoices, refunds",
    }
    reordered_request = spec_request(department={**QUESTIONS["department"], "criteria": reordered_criteria})
    reordered = encode(reordered_request, tokenizer, max_tokens=2048)
    original_text = render_text(spec_request())
    reordered_text = render_text(reordered_request)

    assert "A. billing: Charges, invoices, refunds\n" in original_text
    assert "A. other\n" in reordered_text
    assert "C. billing: Charges, invoices, refunds\n" in reordered_text
    assert reordered.input_ids != encoded.input_ids
    assert set(reordered.letter_ids[1]) == set(encoded.letter_ids[1])
    assert reordered.letter_ids == encoded.letter_ids


# Other rendering rules


def test_noul_without_criteria_renders_yes_and_no():
    request = spec_request(refund_requested={"type": "noul", "instructions": "Is a refund requested?"})
    assert "Options:\nA. true: yes\nB. false: no\nAnswer:" in render_text(request)


def test_json_state_rendered_as_pretty_json(tokenizer):
    state = {"customer": "Zoë", "items": [1, 2]}
    request = spec_request(state=state)
    expected_block = "### State\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n\n### Question:"
    assert render_text(request).startswith(expected_block)
    assert tokenizer.decode(list(encode(request, tokenizer, max_tokens=2048).input_ids)) == render_text(request)


def test_multiline_state_keeps_slots_on_colons(tokenizer):
    request = spec_request(state="Line one:\n\nLine two ends with a colon:")
    encoded = encode(request, tokenizer, max_tokens=2048)
    assert tokenizer.decode(list(encoded.input_ids)) == render_text(request)
    for slot in encoded.slot_positions:
        assert tokenizer.decode([encoded.input_ids[slot]]) == ":"


def test_encoded_length_over_max_tokens(tokenizer, encoded):
    length = len(encoded.input_ids)
    assert encode(spec_request(), tokenizer, max_tokens=length).input_ids == encoded.input_ids
    with pytest.raises(ValueError, match=r"^max_tokens: "):
        encode(spec_request(), tokenizer, max_tokens=length - 1)


# Load-time tokenizer checks


def test_letter_tokens_are_single_and_distinct(tokenizer):
    ids = letter_token_ids(tokenizer)
    assert len(ids) == 26 == len(set(ids))


def test_letter_token_that_splits_fails_loudly(tokenizer):
    broken = FakeTokenizer(tokenizer, {" Q": [220, 48]})
    with pytest.raises(TokenizerError, match="Q"):
        letter_token_ids(broken)


def test_pinned_06b_and_17b_tokenizers_agree(base_config, tokenizer):
    # base.yaml selects the 1.7B backbone, so loading it runs the parity check
    # against the pinned 0.6B reference tokenizer.
    assert base_config["backbone"]["id"] != base_config["tokenizer_reference"]["id"]
    tokenizer_17b = load_tokenizer(base_config)
    probe_ids = tokenizer.encode(PARITY_PROBE, add_special_tokens=False)
    assert tokenizer_17b.encode(PARITY_PROBE, add_special_tokens=False) == probe_ids
    assert letter_token_ids(tokenizer_17b) == letter_token_ids(tokenizer)


def test_parity_mismatch_fails_loudly(tokenizer):
    probe_ids = tokenizer.encode(PARITY_PROBE, add_special_tokens=False)
    drifted = FakeTokenizer(tokenizer, {PARITY_PROBE: probe_ids[:-1] + [probe_ids[-1] + 1]})
    with pytest.raises(TokenizerError, match="parity"):
        check_tokenizer_parity(drifted, tokenizer)


def test_slot_not_ending_in_colon_fails_loudly(tokenizer):
    request = spec_request()
    block = render_text(request).split(SEPARATOR)[1]
    ids = tokenizer.encode(block, add_special_tokens=False)
    broken = FakeTokenizer(tokenizer, {block: ids + tokenizer.encode(" A", add_special_tokens=False)})
    with pytest.raises(TokenizerError, match="slot"):
        encode(request, broken, max_tokens=2048)


# Shuffling for training


def test_shuffle_options_returns_permutation_for_label_remap():
    question = spec_request().questions[1]
    assert isinstance(question, ChoiceQuestion)
    shuffled, perm = shuffle_options(question, random.Random(3))
    assert sorted(perm) == list(range(len(question.options)))
    for new_index, old_index in enumerate(perm):
        assert shuffled.options[new_index] == question.options[old_index]
    gold_old = question.labels.index("technical")
    assert shuffled.labels[perm.index(gold_old)] == "technical"
    assert shuffled.id == question.id and shuffled.instructions == question.instructions


def test_shuffle_is_seeded_and_covers_all_positions():
    question = spec_request().questions[1]
    assert shuffle_options(question, random.Random(7)) == shuffle_options(question, random.Random(7))
    rng = random.Random(0)
    first_labels = {shuffle_options(question, rng)[0].labels[0] for _ in range(200)}
    assert first_labels == set(question.labels)


def test_shuffle_request_touches_choice_questions_only():
    request = spec_request()
    shuffled, perms = shuffle_request(request, random.Random(1))
    assert set(perms) == {"department"}
    assert shuffled.state == request.state
    assert [q.id for q in shuffled.questions] == [q.id for q in request.questions]
    assert shuffled.questions[0] == request.questions[0]
    assert shuffled.questions[2] == request.questions[2]
    assert sorted(shuffled.questions[1].options) == sorted(request.questions[1].options)
