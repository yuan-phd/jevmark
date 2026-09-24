"""systemone and systemone_batch (docs/API_SPEC.md sections 1 and 3) and scripts/serve.py."""

import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import jevmark
from jevmark.encode import encode
from jevmark.model import JevMark
from jevmark.schema import ChoiceQuestion, NoulQuestion, Request, Response, ScoreQuestion
from jevmark.systemone import answer_from_distribution, systemone, systemone_batch

SYSTEMONE_MODULE = importlib.import_module("jevmark.systemone")
SERVE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"

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
OTHER_REQUEST = {
    "state": {"ticket": 42, "text": "App crashes on start"},
    "questions": {
        "component": {
            "type": "choice",
            "instructions": "Which component is affected?",
            "criteria": {"ios": None, "android": None, "web": None, "backend": "APIs and jobs", "other": None},
        },
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    },
}


@pytest.fixture(scope="module")
def jev(tiny_model, tokenizer):
    return JevMark(tiny_model, tokenizer, max_tokens=2048, model_id="jevmark-tiny")


@pytest.fixture(scope="module")
def response(jev):
    return systemone(STATE, copy.deepcopy(QUESTIONS), model=jev)


def decimals_at_most_4(x):
    return round(x, 4) == x


# Round trip of the API_SPEC example, field by field


def test_top_level_shape(response, jev, tokenizer):
    assert list(response) == ["model", "answers", "usage"]
    assert response["model"] == "jevmark-tiny"
    assert list(response["answers"]) == ["refund_requested", "department", "severity"]
    encoded = encode(Request.from_dict({"state": STATE, "questions": QUESTIONS}), tokenizer, max_tokens=2048)
    assert response["usage"] == {"input_tokens": len(encoded.input_ids)}
    assert isinstance(response["usage"]["input_tokens"], int)
    assert Response.from_dict(response).to_dict() == response
    json.dumps(response)


def test_noul_answer(response):
    answer = response["answers"]["refund_requested"]
    assert list(answer) == ["type", "noul"]
    assert answer["type"] == "noul"
    assert isinstance(answer["noul"], float) and 0.0 <= answer["noul"] <= 1.0
    assert decimals_at_most_4(answer["noul"])


def test_choice_answer(response):
    answer = response["answers"]["department"]
    assert list(answer) == ["type", "choice", "probabilities", "confidence"]
    assert answer["type"] == "choice"
    assert list(answer["probabilities"]) == ["billing", "technical", "other"]
    assert answer["choice"] in answer["probabilities"]
    assert answer["probabilities"][answer["choice"]] == max(answer["probabilities"].values())
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 and decimals_at_most_4(p) for p in answer["probabilities"].values())
    assert isinstance(answer["confidence"], float) and 0.0 <= answer["confidence"] <= 1.0
    assert decimals_at_most_4(answer["confidence"])


def test_score_answer(response):
    answer = response["answers"]["severity"]
    assert list(answer) == ["type", "score", "legend", "probabilities", "confidence"]
    assert answer["type"] == "score"
    assert answer["legend"] == {str(i): text for i, text in enumerate(QUESTIONS["severity"]["criteria"])}
    assert list(answer["probabilities"]) == ["0", "1", "2"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(isinstance(p, float) and decimals_at_most_4(p) for p in answer["probabilities"].values())
    assert isinstance(answer["score"], float) and 0.0 <= answer["score"] <= 2.0
    assert decimals_at_most_4(answer["score"])
    assert 0.0 <= answer["confidence"] <= 1.0 and decimals_at_most_4(answer["confidence"])


def test_answers_match_model_distributions(response, jev, tokenizer):
    encoded = encode(Request.from_dict({"state": STATE, "questions": QUESTIONS}), tokenizer, max_tokens=2048)
    noul, choice, score = [d.double() for d in jev.forward_distributions([encoded])]
    answers = response["answers"]
    assert answers["refund_requested"]["noul"] == round(noul[0].item(), 4)
    assert list(answers["department"]["probabilities"].values()) == [round(p, 4) for p in choice.tolist()]
    expectation = sum(i * p for i, p in enumerate(score.tolist()))
    assert answers["severity"]["score"] == round(expectation, 4)


# Answer arithmetic on hand-made distributions


def noul_q():
    return NoulQuestion(id="q", instructions="Yes?")


def choice_q(k):
    return ChoiceQuestion(id="q", instructions="Which?", options=tuple((f"o{i}", None) for i in range(k)))


def score_q(k):
    return ScoreQuestion(id="q", instructions="How much?", levels=tuple(f"level {i}" for i in range(k)))


def test_noul_is_probability_of_first_option():
    assert answer_from_distribution(noul_q(), torch.tensor([0.8, 0.2])).to_dict() == {"type": "noul", "noul": 0.8}


def test_choice_argmax_is_computed_before_rounding():
    # Both round to 0.5; only the unrounded values decide the argmax.
    assert answer_from_distribution(choice_q(2), torch.tensor([0.499996, 0.500004], dtype=torch.float64)).choice == "o1"
    assert answer_from_distribution(choice_q(2), torch.tensor([0.500004, 0.499996], dtype=torch.float64)).choice == "o0"


def test_score_expectation_is_computed_before_rounding():
    answer = answer_from_distribution(score_q(3), torch.tensor([0.00004, 0.00004, 0.99992], dtype=torch.float64))
    assert answer.probabilities == {"0": 0.0, "1": 0.0, "2": 0.9999}
    assert answer.score == round(0.00004 + 2 * 0.99992, 4) == 1.9999  # from rounded probabilities it would be 1.9998


@pytest.mark.parametrize(
    "probs, expected",
    [
        ([1 / 3, 1 / 3, 1 / 3], 0.0),
        ([1.0, 0.0, 0.0], 1.0),  # 0 log 0 is 0
        ([0.5, 0.5, 0.0], round(1 - math.log(2) / math.log(3), 4)),
        ([0.7, 0.2, 0.1], round(1 + (0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1)) / math.log(3), 4)),
    ],
)
def test_confidence_is_one_minus_normalised_entropy(probs, expected):
    answer = answer_from_distribution(choice_q(3), torch.tensor(probs, dtype=torch.float64))
    assert answer.confidence == expected


def test_confidence_is_never_negative_zero_or_out_of_range():
    answer = answer_from_distribution(choice_q(4), torch.full((4,), 0.25))
    assert answer.confidence == 0.0 and math.copysign(1.0, answer.confidence) == 1.0


# Batch against single calls, at the response level


def assert_responses_close(x, y, tol=1e-4, path="response"):
    """Same structure, keys and non-float values; floats within tol."""
    if isinstance(x, float) or isinstance(y, float):
        assert isinstance(x, float) and isinstance(y, float), path
        assert abs(x - y) <= tol, f"{path}: {x} vs {y}"
    elif isinstance(x, dict):
        assert isinstance(y, dict) and list(x) == list(y), path
        for key in x:
            assert_responses_close(x[key], y[key], tol, f"{path}.{key}")
    elif isinstance(x, list):
        assert isinstance(y, list) and len(x) == len(y), path
        for i, (a, b) in enumerate(zip(x, y)):
            assert_responses_close(a, b, tol, f"{path}[{i}]")
    else:
        assert x == y, f"{path}: {x!r} vs {y!r}"


def test_batch_responses_match_single_calls(jev):
    # Batching has no semantic effect; numerically, padded batches change the order
    # of floating-point operations, so floats are compared within 1e-4 (decision 30).
    spec = {"state": STATE, "questions": copy.deepcopy(QUESTIONS)}
    requests = [spec, OTHER_REQUEST, {"state": "Short.", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}]
    singles = [systemone(r["state"], r["questions"], model=jev) for r in requests]
    assert_responses_close(systemone_batch(requests, model=jev), singles)
    assert_responses_close(systemone_batch(requests, model=jev, batch_size=2), singles)


def test_batch_of_nothing_is_empty(jev):
    assert systemone_batch([], model=jev) == []


# max_tokens, validation, default model


def test_max_tokens_default_comes_from_model_and_argument_overrides(tiny_model, tokenizer, response):
    short = JevMark(tiny_model, tokenizer, max_tokens=20, model_id="jevmark-tiny")
    with pytest.raises(ValueError, match=r"^max_tokens: "):
        systemone(STATE, QUESTIONS, model=short)
    assert systemone(STATE, QUESTIONS, model=short, max_tokens=2048) == response


def test_invalid_request_raises_value_error_with_path(jev):
    bad = copy.deepcopy(QUESTIONS)
    bad["severity"]["criteria"] = ["only one"]
    with pytest.raises(ValueError, match=r"^severity\.criteria: "):
        systemone(STATE, bad, model=jev)


def test_default_model_loads_lazily_once_with_env_checkpoint(monkeypatch, jev, tmp_path):
    calls = []

    def fake_load(config, checkpoint=None, device=None):
        calls.append((config, checkpoint))
        return jev

    run_dir = tmp_path / "runs" / "my_run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run_name: my_run\nmax_tokens: 1024\n")
    monkeypatch.setattr(JevMark, "load", staticmethod(fake_load))
    monkeypatch.setenv("JEVMARK_CHECKPOINT", str(run_dir))
    SYSTEMONE_MODULE.default_model.cache_clear()
    try:
        assert calls == []
        first = systemone(STATE, QUESTIONS)
        second = systemone(STATE, QUESTIONS)
        assert first == second
        assert len(calls) == 1
        config, checkpoint = calls[0]
        assert checkpoint == str(run_dir)
        assert config["run_name"] == "my_run"  # the run's own config.yaml wins over configs/base.yaml
    finally:
        SYSTEMONE_MODULE.default_model.cache_clear()


def test_default_model_without_env_uses_base_config(monkeypatch, jev):
    calls = []
    monkeypatch.setattr(JevMark, "load", staticmethod(lambda config, checkpoint=None, device=None: calls.append((config, checkpoint)) or jev))
    monkeypatch.delenv("JEVMARK_CHECKPOINT", raising=False)
    SYSTEMONE_MODULE.default_model.cache_clear()
    try:
        systemone(STATE, QUESTIONS)
        (config, checkpoint), = calls
        assert checkpoint is None
        assert config["run_name"] == "base_17b" and config["max_tokens"] == 2048
    finally:
        SYSTEMONE_MODULE.default_model.cache_clear()


def test_package_exports_systemone():
    assert jevmark.systemone is systemone
    assert jevmark.systemone_batch is systemone_batch


# scripts/serve.py through FastAPI's TestClient


@pytest.fixture(scope="module")
def serve():
    spec = importlib.util.spec_from_file_location("jevmark_serve", SERVE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client(serve, jev):
    with TestClient(serve.create_app(model=jev)) as test_client:
        yield test_client


def test_serve_returns_the_same_response(client, response):
    reply = client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS})
    assert reply.status_code == 200
    assert reply.json() == response


def test_serve_rejects_duplicate_keys(client):
    body = (
        '{"state": "s", "questions": {"d": {"type": "choice", "instructions": "Which?", '
        '"criteria": {"billing": null, "billing": "again", "other": null}}}}'
    )
    reply = client.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert reply.status_code == 400
    assert reply.json()["detail"] == "request: duplicate key 'billing'"


def test_serve_rejects_invalid_request_with_path(client):
    reply = client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "multi", "instructions": "x"}}})
    assert reply.status_code == 400
    assert reply.json()["detail"].startswith("q.type: ")


def test_serve_rejects_malformed_json(client):
    reply = client.post("/v1/systemone", content="{not json", headers={"content-type": "application/json"})
    assert reply.status_code == 400
    assert reply.json()["detail"].startswith("request: invalid JSON")


def test_serve_maps_runtime_error_to_500(serve, jev, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("checkpoint missing: no adapter directory")

    monkeypatch.setattr(serve, "systemone_batch", broken)
    with TestClient(serve.create_app(model=jev)) as test_client:
        reply = test_client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS})
    assert reply.status_code == 500
    assert reply.json()["detail"] == "checkpoint missing: no adapter directory"


def test_serve_loads_default_model_once_at_startup(serve, jev, monkeypatch):
    calls = []
    monkeypatch.setattr(serve, "default_model", lambda: calls.append(1) or jev)
    app = serve.create_app()
    assert calls == []
    with TestClient(app) as test_client:
        assert calls == [1]
        for _ in range(2):
            assert test_client.post("/v1/systemone", json={"state": STATE, "questions": QUESTIONS}).status_code == 200
    assert calls == [1]
