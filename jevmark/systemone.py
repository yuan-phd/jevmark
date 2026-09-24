"""Public API: systemone and systemone_batch (docs/API_SPEC.md sections 1 and 3).

Answers are computed in float64 from the model's distributions: noul is P(true),
choice is the argmax label, score is the expected level index, and confidence is
1 - H(p) / ln(K). Every float is rounded to 4 decimals only at the end, so the
argmax and the expectation never see rounded probabilities.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from jevmark.encode import encode
from jevmark.model import JevMark
from jevmark.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
    State,
    Usage,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
CHECKPOINT_ENV = "JEVMARK_CHECKPOINT"
DECIMALS = 4


@functools.lru_cache(maxsize=1)
def default_model() -> JevMark:
    """The model used when systemone gets model=None: loaded on first use, once per process.

    Config is configs/base.yaml. If JEVMARK_CHECKPOINT names a run directory, its
    adapter is loaded too, with the run's own config.yaml when it has one, so the
    adapter meets the backbone it was trained on.
    """
    checkpoint = os.environ.get(CHECKPOINT_ENV) or None
    config_path = DEFAULT_CONFIG_PATH
    if checkpoint is not None and (Path(checkpoint) / "config.yaml").is_file():
        config_path = Path(checkpoint) / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    return JevMark.load(config, checkpoint=checkpoint)


def _round(x: float) -> float:
    # Adding 0.0 turns a rounded -0.0 into 0.0.
    return round(x, DECIMALS) + 0.0


def normalised_confidence(p: Sequence[float]) -> float:
    """1 - H(p) / ln(K) in [0, 1], with 0 log 0 taken as 0."""
    entropy = -sum(x * math.log(x) for x in p if x > 0.0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(p))))


def answer_from_distribution(question: Question, distribution: torch.Tensor) -> Answer:
    """Turn one question's letter distribution into its response answer."""
    total = float(distribution.double().sum())
    p = [float(x) / total for x in distribution.double().cpu()]
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=_round(p[0]))
    confidence = _round(normalised_confidence(p))
    if isinstance(question, ChoiceQuestion):
        best = max(range(len(p)), key=p.__getitem__)
        return ChoiceAnswer(
            choice=question.labels[best],
            probabilities={label: _round(x) for label, x in zip(question.labels, p)},
            confidence=confidence,
        )
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(
            score=_round(sum(index * x for index, x in enumerate(p))),
            legend={str(index): level for index, level in enumerate(question.levels)},
            probabilities={str(index): _round(x) for index, x in enumerate(p)},
            confidence=confidence,
        )
    raise TypeError(f"not a question: {type(question).__name__}")


def systemone_batch(
    requests: Sequence[Mapping[str, Any]],
    model: JevMark | None = None,
    max_tokens: int | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Responses for a list of {"state": ..., "questions": ...} requests, batch_size requests per forward pass.

    Every request is validated and encoded before the model runs, so an invalid
    request raises ValueError without any forward pass.
    """
    jev = model if model is not None else default_model()
    limit = jev.max_tokens if max_tokens is None else max_tokens
    parsed = [Request.from_dict(r) for r in requests]
    encoded = [encode(r, jev.tokenizer, limit) for r in parsed]
    responses = []
    for start in range(0, len(parsed), batch_size):
        chunk = encoded[start : start + batch_size]
        distributions = iter(jev.forward_distributions(chunk))
        for request, e in zip(parsed[start : start + batch_size], chunk):
            answers = {q.id: answer_from_distribution(q, next(distributions)) for q in request.questions}
            response = Response(model=jev.model_id, answers=answers, usage=Usage(input_tokens=len(e.input_ids)))
            responses.append(response.to_dict())
    return responses


def systemone(
    state: State,
    questions: Mapping[str, Any],
    model: JevMark | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Answer every question about the state in one forward pass; see docs/API_SPEC.md."""
    return systemone_batch([{"state": state, "questions": questions}], model=model, max_tokens=max_tokens)[0]
