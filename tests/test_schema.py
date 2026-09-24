"""Schema validation against docs/API_SPEC.md sections 2, 3 and 7.

The section 7 case "encoded length over max_tokens" needs a tokenizer and is
tested with encode.py (task 1.2), not here.
"""

import copy

import pytest

from jevmark.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
)

SPEC_QUESTIONS = {
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

SPEC_RESPONSE = {
    "model": "jevmark-sft_clinc_v1",
    "answers": {
        "refund_requested": {"type": "noul", "noul": 0.93},
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.84, "technical": 0.15, "other": 0.01},
            "confidence": 0.61,
        },
        "severity": {
            "type": "score",
            "score": 1.3,
            "legend": {"0": "Cosmetic; no impact on functionality", "1": "...", "2": "..."},
            "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
            "confidence": 0.54,
        },
    },
    "usage": {"input_tokens": 212},
}


def make_request(state="I was charged twice, please refund me.", **overrides):
    questions = copy.deepcopy(SPEC_QUESTIONS)
    questions.update(overrides)
    return {"state": state, "questions": questions}


def one_question(qid, definition, state="some state"):
    return {"state": state, "questions": {qid: definition}}


# Valid requests


def test_valid_noul_with_criteria():
    req = Request.from_dict(one_question("q", SPEC_QUESTIONS["refund_requested"]))
    (q,) = req.questions
    assert isinstance(q, NoulQuestion)
    assert q.id == "q"
    assert q.true_description == "optional description of yes"
    assert q.false_description == "optional description of no"


def test_valid_noul_without_criteria():
    req = Request.from_dict(one_question("q", {"type": "noul", "instructions": "Is it raining?"}))
    (q,) = req.questions
    assert isinstance(q, NoulQuestion)
    assert q.true_description is None and q.false_description is None


def test_valid_noul_with_partial_criteria():
    req = Request.from_dict(
        one_question("q", {"type": "noul", "instructions": "Is it raining?", "criteria": {"true": "rain"}})
    )
    (q,) = req.questions
    assert q.true_description == "rain" and q.false_description is None


def test_valid_choice_preserves_option_order_and_null_description():
    req = Request.from_dict(one_question("department", SPEC_QUESTIONS["department"]))
    (q,) = req.questions
    assert isinstance(q, ChoiceQuestion)
    assert q.labels == ("billing", "technical", "other")
    assert q.options[2] == ("other", None)


def test_valid_score():
    req = Request.from_dict(one_question("severity", SPEC_QUESTIONS["severity"]))
    (q,) = req.questions
    assert isinstance(q, ScoreQuestion)
    assert q.levels[0] == "Cosmetic; no impact on functionality"
    assert len(q.levels) == 3


def test_valid_three_question_request_keeps_order_and_round_trips():
    raw = make_request()
    req = Request.from_dict(raw)
    assert [q.id for q in req.questions] == ["refund_requested", "department", "severity"]
    assert req.to_dict() == raw
    assert Request.from_dict(req.to_dict()) == req


@pytest.mark.parametrize("state", [{"customer": "Ana", "amount": 12.5}, ["a", 1, None], ""])
def test_valid_json_and_empty_states(state):
    req = Request.from_dict(make_request(state=state))
    assert req.state == state


def test_option_counts_at_limits_are_valid():
    choice26 = {"type": "choice", "instructions": "Pick one.", "criteria": {f"opt{i}": None for i in range(26)}}
    choice2 = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": None}}
    score10 = {"type": "score", "instructions": "Rate it.", "criteria": [f"level {i}" for i in range(10)]}
    score2 = {"type": "score", "instructions": "Rate it.", "criteria": ["low", "high"]}
    for definition in (choice26, choice2, score10, score2):
        Request.from_dict(one_question("q", definition))


def test_state_of_exactly_8000_chars_is_valid():
    Request.from_dict(make_request(state="x" * 8000))


def test_dataclasses_are_immutable():
    req = Request.from_dict(make_request())
    with pytest.raises(AttributeError):
        req.questions[0].instructions = "changed"


# Section 7 error cases. Each message must start with "<question_id>.<field>: ".


def expect_error(raw, prefix):
    with pytest.raises(ValueError) as info:
        Request.from_dict(raw)
    message = str(info.value)
    assert message.startswith(prefix + ": "), message
    assert len(message) > len(prefix) + 2, "reason must not be empty"


@pytest.mark.parametrize("bad_type", ["multi", "Choice", "", 3, None])
def test_unknown_type(bad_type):
    expect_error(one_question("q", {"type": bad_type, "instructions": "x"}), "q.type")


def test_missing_type():
    expect_error(one_question("q", {"instructions": "x"}), "q.type")


@pytest.mark.parametrize("qtype", ["noul", "choice", "score"])
def test_missing_instructions(qtype):
    definition = copy.deepcopy(
        {"noul": SPEC_QUESTIONS["refund_requested"], "choice": SPEC_QUESTIONS["department"], "score": SPEC_QUESTIONS["severity"]}[qtype]
    )
    del definition["instructions"]
    expect_error(one_question("q", definition), "q.instructions")


@pytest.mark.parametrize("bad", ["", "   ", 7, None, ["a"]])
def test_empty_or_non_string_instructions(bad):
    expect_error(one_question("q", {"type": "noul", "instructions": bad}), "q.instructions")


@pytest.mark.parametrize(
    "definition",
    [
        # choice: criteria must be a dict of label to string or null
        {"type": "choice", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
        {"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": None}},
        {"type": "choice", "instructions": "x", "criteria": {"a": None, "": None}},
        {"type": "choice", "instructions": "x", "criteria": {"a": None, "  ": None}},
        # score: criteria must be a list of strings
        {"type": "score", "instructions": "x"},
        {"type": "score", "instructions": "x", "criteria": {"0": "low", "1": "high"}},
        {"type": "score", "instructions": "x", "criteria": ["low", 2]},
        {"type": "score", "instructions": "x", "criteria": "low, high"},
        # noul: criteria optional, but if given only "true" and "false" with string or null values
        {"type": "noul", "instructions": "x", "criteria": ["yes", "no"]},
        {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}},
        {"type": "noul", "instructions": "x", "criteria": {"true": 1}},
    ],
)
def test_wrong_criteria_shape(definition):
    expect_error(one_question("q", definition), "q.criteria")


@pytest.mark.parametrize("n", [0, 1, 27, 40])
def test_choice_option_count_out_of_range(n):
    definition = {"type": "choice", "instructions": "x", "criteria": {f"opt{i}": None for i in range(n)}}
    expect_error(one_question("q", definition), "q.criteria")


@pytest.mark.parametrize("n", [0, 1, 11])
def test_score_level_count_out_of_range(n):
    definition = {"type": "score", "instructions": "x", "criteria": [f"level {i}" for i in range(n)]}
    expect_error(one_question("q", definition), "q.criteria")


def test_duplicate_labels_after_whitespace_strip():
    # A JSON object cannot hold the same key twice, but labels that differ only in
    # surrounding whitespace render as the same option line.
    definition = {"type": "choice", "instructions": "x", "criteria": {"billing": None, "billing ": "dup"}}
    expect_error(one_question("q", definition), "q.criteria")


def test_duplicate_labels_on_direct_construction():
    with pytest.raises(ValueError, match=r"^q\.criteria: "):
        ChoiceQuestion(id="q", instructions="x", options=(("billing", None), ("billing", "again")))


def test_state_over_8000_chars():
    expect_error(make_request(state="x" * 8001), "state")


def test_json_state_over_8000_chars_counts_rendered_text():
    # 600 short keys: 7580 characters as compact json.dumps, 8782 once pretty-printed.
    state = {f"k{i}": i for i in range(600)}
    expect_error(make_request(state=state), "state")


@pytest.mark.parametrize("state", [3, 1.5, True, None, {"a": {1, 2}}])
def test_state_wrong_type_or_not_json(state):
    expect_error(make_request(state=state), "state")


# Request-level structure that the spec implies (id pattern, questions mapping).


@pytest.mark.parametrize("qid", ["Bad", "has-dash", "has space", "", 3])
def test_question_id_pattern(qid):
    # An invalid id cannot serve as its own path, so the error is reported on "questions".
    expect_error(one_question(qid, {"type": "noul", "instructions": "x"}), "questions")


def test_questions_must_be_a_non_empty_mapping():
    expect_error({"state": "s", "questions": []}, "questions")
    expect_error({"state": "s", "questions": {}}, "questions")
    expect_error({"state": "s"}, "questions")


def test_missing_state():
    expect_error({"questions": copy.deepcopy(SPEC_QUESTIONS)}, "state")


def test_question_definition_must_be_an_object():
    expect_error(one_question("q", "Is it raining?"), "q")


def test_unknown_question_field_is_rejected():
    # A typo such as "criterion" would otherwise be silently ignored.
    expect_error(one_question("q", {"type": "noul", "instructions": "x", "criterion": {}}), "q.criterion")


def test_request_error_message_uses_question_id_not_index():
    raw = make_request(severity={"type": "score", "instructions": "x", "criteria": ["only one"]})
    expect_error(raw, "severity.criteria")


# Response and answers (section 3)


def test_response_round_trips_spec_example():
    resp = Response.from_dict(SPEC_RESPONSE)
    assert isinstance(resp.answers["refund_requested"], NoulAnswer)
    assert isinstance(resp.answers["department"], ChoiceAnswer)
    assert isinstance(resp.answers["severity"], ScoreAnswer)
    assert resp.usage.input_tokens == 212
    assert resp.to_dict() == SPEC_RESPONSE
    assert list(resp.to_dict()["answers"]) == ["refund_requested", "department", "severity"]


def test_noul_answer_has_no_confidence_field():
    assert NoulAnswer(noul=0.5).to_dict() == {"type": "noul", "noul": 0.5}


def test_response_unknown_answer_type():
    bad = copy.deepcopy(SPEC_RESPONSE)
    bad["answers"]["department"]["type"] = "ranking"
    with pytest.raises(ValueError, match=r"^department\.type: "):
        Response.from_dict(bad)
