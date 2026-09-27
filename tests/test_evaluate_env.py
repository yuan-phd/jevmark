"""scripts/evaluate_env.py and scripts/compare_env.py on synthetic results files, CPU (task 2.5, decision 54)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from jevmark import environment as E
from jevmark.metrics import QuestionResult, write_results

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate_env = load("evaluate_env")
compare_env = load("compare_env")

SPLITS = ("valid", "test_indomain", "test_agnews", "test_emotion", "test_banking77", "test_yelp")


def results(power=1.0, splits=SPLITS, n=30):
    """Every question's distribution is theta raised to `power` and renormalised: power 1 is perfectly calibrated, above 1 sharper."""
    out = []
    for split in splits:
        for i in range(n):
            for k, qid in ((2, "noul"), (4, "choice"), (10, "choice")):
                gold = i % k
                p = E.theta(k, gold) ** power
                probs = tuple(p / p.sum())
                out.append(QuestionResult(f"{split}-{i}", f"q{k}", qid, probs, gold, tuple(str(j) for j in range(k)), split=split, kind="about_domain" if k == 2 else None))
    return out


def test_evaluate_env_recovers_zero_gap_for_a_calibrated_distribution(tmp_path, capsys):
    run = tmp_path / "runs" / "calibrated"
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", results(1.0))
    assert evaluate_env.main([str(run)]) == 0
    out = json.loads((tmp_path / "runs" / "calibrated_env" / "metrics.json").read_text())
    raw = out["variants"]["raw"]["splits"]
    for split in SPLITS:
        m = raw[split]
        assert m["calibration_gap"] == pytest.approx(0.0, abs=1e-12)
        assert m["expected_brier"] == pytest.approx(m["expected_brier_min"]) and m["kl_theta"] == pytest.approx(0.0, abs=1e-12)
        assert m["accuracy"] == 1.0
        assert set(m["by_k"]) == {"2", "4", "10"}
        for k, row in m["by_k"].items():
            assert row["mean_top1"] == pytest.approx(row["expected_top1_hit"]) == pytest.approx(1 - E.eta(int(k)))
        # The per-K oracle of a calibrated distribution is T 1, and theta scores exactly as the raw distribution.
        assert all(t == pytest.approx(1.0, abs=1e-3) for t in out["variants"]["oracle_T_by_K"]["splits"][split]["temperatures"].values())
        assert out["variants"]["theta"]["splits"][split]["cross_entropy"] == pytest.approx(m["cross_entropy"])
        # Every variant scores ECE against the same sampled outcomes.
        assert out["variants"]["theta"]["splits"][split]["accuracy_sampled"] == m["accuracy_sampled"]
    assert out["variants"]["global_T"]["fit"]["split"] == "valid"
    assert "global T" in capsys.readouterr().out


def test_temperatures_soften_a_sharpened_distribution(tmp_path):
    run = tmp_path / "runs" / "sharp"
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", results(3.0))
    assert evaluate_env.main([str(run)]) == 0
    v = json.loads((tmp_path / "runs" / "sharp_env" / "metrics.json").read_text())["variants"]
    assert v["global_T"]["temperature"] > 1.0
    for split in SPLITS:
        raw, oracle = v["raw"]["splits"][split], v["oracle_T_by_K"]["splits"][split]
        assert raw["calibration_gap"] > 0.01 and raw["cross_entropy"] > raw["entropy_theta"]
        # theta ** 3 with a per-K oracle temperature of 3 is theta again.
        assert all(t == pytest.approx(3.0, rel=1e-3) for t in oracle["temperatures"].values())
        assert oracle["calibration_gap"] == pytest.approx(0.0, abs=1e-5)


def test_compare_env_aggregates_seeds_and_pairs_every_column(tmp_path, capsys):
    runs = tmp_path / "runs"
    (runs / "sft_06b").mkdir(parents=True)
    write_results(runs / "sft_06b" / "results.jsonl.gz", results(3.0))
    for arm, seed, power in (("direct_brier", 0, 1.0), ("direct_brier", 1, 1.5), ("outcome", 0, 6.0)):
        run = runs / f"rlcd_06b_noisy_{arm}_s{seed}"
        run.mkdir()
        write_results(run / "results.jsonl.gz", results(power))
    (runs / "rlcd_06b_direct_brier_s0").mkdir()  # a deterministic run: never picked up
    write_results(runs / "rlcd_06b_direct_brier_s0" / "results.jsonl.gz", results(1.0))
    assert compare_env.main(["--runs-dir", str(runs), "--resamples", "50"]) == 0
    out = json.loads((runs / "rlcd_stage3_06b" / "metrics.json").read_text())
    assert out["arms"] == {"direct_brier": [0, 1], "outcome": [0]} and out["seeds"] == [0, 1]
    assert list(out["splits"]) == ["test_indomain", "test_agnews", "test_emotion", "test_banking77", "test_yelp"]  # the stage 3 splits present
    block = out["splits"]["test_emotion"]
    brier = block["arms"]["direct_brier"]
    per_seed = [brier["by_seed"][s]["cross_entropy"] for s in ("0", "1")]
    assert brier["mean"]["cross_entropy"] == pytest.approx(np.mean(per_seed)) and brier["mean"]["cross_entropy_range"] == [min(per_seed), max(per_seed)]
    # Seed 0 is theta itself: zero gap, and lower cross-entropy than the global temperature; the seed mean also carries the sharper seed 1.
    assert brier["by_seed"]["0"]["calibration_gap"] == pytest.approx(0.0, abs=1e-12)
    assert brier["by_seed"]["0"]["cross_entropy"] < block["sft_global_T"]["cross_entropy"] < brier["by_seed"]["1"]["cross_entropy"]
    lo, hi = brier["mean"]["cross_entropy_minus_sft_global_T_ci"]
    assert lo <= brier["mean"]["cross_entropy_minus_sft_global_T"] <= hi
    # The oracle per K undoes theta ** 3 exactly, so it matches theta; the sharper outcome arm is worse than both.
    assert block["sft_oracle_T"]["calibration_gap"] == pytest.approx(0.0, abs=1e-5)
    assert block["arms"]["outcome"]["mean"]["cross_entropy_minus_sft_oracle_T_ci"][0] > 0
    # A column against itself: zero difference in every resample.
    assert block["sft_global_T"]["cross_entropy_minus_sft_global_T_ci"] == [0.0, 0.0]
    assert block["arms"]["direct_brier"]["mean"]["accuracy_minus_sft"] == 0.0
    assert "test_emotion" in capsys.readouterr().out


def test_compare_env_refuses_runs_with_different_questions(tmp_path):
    runs = tmp_path / "runs"
    (runs / "sft_06b").mkdir(parents=True)
    write_results(runs / "sft_06b" / "results.jsonl.gz", results(1.0))
    (runs / "rlcd_06b_noisy_outcome_s0").mkdir()
    write_results(runs / "rlcd_06b_noisy_outcome_s0" / "results.jsonl.gz", results(1.0)[:-3])
    with pytest.raises(SystemExit, match="not comparable"):
        compare_env.main(["--runs-dir", str(runs), "--resamples", "10"])
