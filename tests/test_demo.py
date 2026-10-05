"""scripts/demo.py: the questions are a valid request and the confidence gate routes by hand-computed values (no model)."""

import importlib.util
import sys
from pathlib import Path

from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_script", REPO / "scripts" / "demo.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


def test_demo_request_is_valid_with_one_choice_over_departments_two_nouls_a_score_and_a_tool_choice():
    request = Request.from_dict({"state": demo.MESSAGE, "questions": demo.QUESTIONS})
    assert [q["type"] for q in demo.QUESTIONS.values()] == ["choice", "noul", "noul", "score", "choice"]
    assert len(request.questions) == 5 and len(demo.QUESTIONS["next_tool"]["criteria"]) == 3


def test_route_uses_the_confidence_field_and_max_p_for_noul():
    response = {
        "answers": {
            "a": {"type": "choice", "choice": "x", "probabilities": {"x": 0.99, "y": 0.01}, "confidence": 0.95},
            "b": {"type": "noul", "noul": 0.05},
            "c": {"type": "noul", "noul": 0.85},
            "d": {"type": "score", "score": 1.0, "legend": {"0": "l", "1": "m", "2": "h"}, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": 0.9},
        }
    }
    assert demo.confidence(response["answers"]["b"]) == 0.95 and demo.confidence(response["answers"]["c"]) == 0.85
    assert demo.route(response, 0.9) == (["a", "b", "d"], ["c"])
    assert demo.route(response, 0.0) == (["a", "b", "c", "d"], [])
