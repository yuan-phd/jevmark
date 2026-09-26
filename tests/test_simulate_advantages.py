"""scripts/simulate_advantages.py on a synthetic results file, CPU (task 2.2, decision 52)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jevmark.metrics import QuestionResult, write_results

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("simulate_advantages", REPO / "scripts" / "simulate_advantages.py")
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


def synthetic_run(tmp_path):
    run = tmp_path / "sft_06b"
    run.mkdir()
    results = []
    for i in range(60):
        results.append(QuestionResult(f"r{i}", "intent", "choice", (0.55, 0.25, 0.15, 0.05), i % 4, ("a", "b", "c", "other"), split="train"))
        results.append(QuestionResult(f"r{i}", "about_domain", "noul", (0.7, 0.3), i % 2, ("true", "false"), split="train", kind="about_domain"))
        results.append(QuestionResult(f"r{i}", "char_count_over", "noul", (0.5, 0.5), 0, ("true", "false"), split="train", kind="char_count_over"))
        results.append(QuestionResult(f"r{i}", "intent", "choice", (0.2, 0.3, 0.5), 0, ("a", "b", "other"), split="valid"))
    write_results(run / "results.jsonl.gz", results)
    return run


def test_simulation_on_a_synthetic_run(tmp_path, capsys):
    run = synthetic_run(tmp_path)
    assert sim.main([str(run)]) == 0
    out = tmp_path / "advantage_simulation_06b" / "metrics.json"
    metrics = json.loads(out.read_text())
    assert metrics["questions"] == {"overall": 120, "noul": 60, "choice": 60, "score": 0}  # train only, form nouls excluded
    rewards = metrics["rewards"]
    # outcome never gives gold a negative advantage or a wrong sample a positive one
    assert rewards["outcome"]["overall"]["gold_negative_advantage"] == 0.0 and rewards["outcome"]["overall"]["wrong_positive_advantage"] == 0.0
    for arm in ("brier", "log"):
        # K = 2: gold and wrong samples always get the same reward, so every advantage is zero
        noul = rewards[arm]["noul"]
        assert noul["mixed_groups"]["all_rewards_equal"] == 1.0 and noul["gold_logit_down"] == noul["gold_logit_up"] == 0.0
        # K > 2: no wrong sample is ever rewarded below gold, and gold's mean advantage is negative in every mixed group
        choice = rewards[arm]["choice"]["mixed_groups"]
        assert choice["n"] > 0 and choice["some_wrong_rewarded_below_gold"] == 0.0 and choice["gold_mean_advantage_negative"] == 1.0
    assert "score" not in rewards["brier"]
    check = metrics["sign_check"]["arms"]
    assert all(arm["code_matches_hand"] for arm in check.values())
    assert check["brier"]["rewards"] == pytest.approx([0.84, 0.91, 0.99])
    assert check["brier"]["gold_logit_lowered"] and check["log"]["gold_logit_lowered"] and not check["outcome"]["gold_logit_lowered"]
    first = out.read_text()
    assert sim.main([str(run)]) == 0
    strip = lambda text: {k: v for k, v in json.loads(text).items() if k != "created"}
    assert strip(out.read_text()) == strip(first)  # seeded: the same actions on a rerun
