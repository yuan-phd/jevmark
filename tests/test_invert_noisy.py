"""scripts/invert_noisy.py on synthetic probabilities (decision 57)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from jevmark.data.build import V3_TEST_FULL
from jevmark.metrics import QuestionResult, ece, write_results

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("invert_noisy", REPO / "scripts" / "invert_noisy.py")
invert_noisy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = invert_noisy
spec.loader.exec_module(invert_noisy)

LABELS = ("a", "b", "c", "d")


def test_inversion_by_hand():
    probs, ok = invert_noisy.invert_probs((0.5, 0.3, 0.2), 0.2)
    assert ok and probs == pytest.approx((0.75, 0.25, 0.0))
    probs, ok = invert_noisy.invert_probs((0.8, 0.15, 0.05), 0.2)
    assert ok and probs == pytest.approx((1.0, 0.0, 0.0))
    assert invert_noisy.invert_probs((0.6, 0.3, 0.1), 0.0) == (pytest.approx((0.6, 0.3, 0.1)), True)


def test_all_options_clipped_keeps_the_stored_distribution():
    assert invert_noisy.invert_probs((0.1,) * 10, 0.2) == ((0.1,) * 10, False)
    with pytest.raises(ValueError):
        invert_noisy.invert_probs((0.5, 0.5), 0.5)


def channel_calibrated(n: int, flip: float, seed: int = 0) -> list[QuestionResult]:
    """A model calibrated to the flipped outcome: its top-1 is flip + (1 - 2 flip) c, where c is the chance the top option is gold."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = rng.uniform(0.4, 1.0)
        top = flip + (1 - 2 * flip) * c
        probs = [top] + [(1 - top) / 3] * 3
        gold = 0 if rng.random() < c else int(rng.integers(1, 4))
        out.append(QuestionResult(f"r{i}", "label", "choice", tuple(probs), gold, LABELS, split=V3_TEST_FULL))
    return out


def test_top1_inversion_recovers_calibration_to_correctness(tmp_path):
    results = channel_calibrated(20000, 0.2)
    stored_ece = ece([r.top1 for r in results], [r.correct for r in results])
    top = invert_noisy.top1_only(results, 0.2)
    assert stored_ece > 0.05 and top["ece"] < 0.02
    inverted, unchanged = invert_noisy.invert_results(results, 0.2)
    assert unchanged == 0 and [r.prediction for r in inverted] == [r.prediction for r in results]
    run = tmp_path / "runs" / "noisy"
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", results[:500])
    (run / "metrics.json").write_text(json.dumps({"git": {"commit": "c" * 40, "dirty": False}}))
    assert invert_noisy.main([str(run), "--clean", str(run)]) == 0
    m = json.loads((tmp_path / "runs" / "noisy_inverted" / "metrics.json").read_text())
    assert m["flip"] == 0.2 and m["stored"]["accuracy"] == m["inverted"]["accuracy"] == m["clean_reference"]["accuracy"]
    assert set(m["inverted"]["coverage"]) == {"0.80", "0.90", "0.95"} and m["inverted_top1_only"]["ece"] < m["stored"]["ece"]
