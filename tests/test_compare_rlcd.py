"""scripts/compare_rlcd.py on synthetic runs, CPU (task 2.3)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jevmark.calibration import scale_results
from jevmark.metrics import QuestionResult, ece, write_results

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_rlcd", REPO / "scripts" / "compare_rlcd.py")
cmp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cmp
spec.loader.exec_module(cmp)

SPLITS = ("valid", "test_indomain", "test_agnews", "test_emotion", "test_banking77", "test_yelp")


def results(sharpen):
    """Two questions per record (a choice and a noul), 40 records per split; `sharpen` > 1 makes the run more confident."""
    out = []
    for split in SPLITS:
        for i in range(40):
            gold = i % 3
            base = [0.2, 0.2, 0.2]
            base[(gold + (i % 5 == 0)) % 3] = 0.6  # one record in five is wrong
            probs = [p**sharpen for p in base]
            probs = tuple(p / sum(probs) for p in probs)
            out.append(QuestionResult(f"{split}-{i}", "intent", "choice", probs, gold, ("a", "b", "other"), split=split))
            yes = 0.8 if i % 4 else 0.3
            out.append(QuestionResult(f"{split}-{i}", "about_domain", "noul", (yes, 1 - yes), 0, ("true", "false"), split=split, kind="about_domain"))
    return out


def write_rlcd_run(runs, arm, rs, seed=0):
    run = runs / f"rlcd_06b_{arm}_s{seed}"
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", rs)
    valid = lambda step, kl: {"event": "valid", "step": step, "accuracy": 0.9, "ece": 0.02, "nll": 0.2 + kl, "kl_to_ref": kl, "expected_p_chosen": 0.9 + kl, "expected_reward": 0.95}
    log = [valid(0, 0.0), {"step": 1, "loss": 0.1}, valid(5, 0.01), valid(10, 0.02)]
    (run / "training_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    final = {k: 0.5 for k in cmp.VALID_KEYS}
    (run / "train_summary.json").write_text(json.dumps({"best_step": 5, "steps": 10, "final_valid": final, "final_valid_last": final}))
    return run


def setup_runs(tmp_path):
    runs = tmp_path / "runs"
    sft = runs / "sft_06b"
    sft.mkdir(parents=True)
    base = results(1.0)
    write_results(sft / "results.jsonl.gz", base)
    (runs / "sft_06b_temp").mkdir()
    (runs / "sft_06b_temp" / "calibration.json").write_text(json.dumps({"temperature": 1.5}))
    return runs, base


def test_compare_on_synthetic_runs(tmp_path, capsys):
    runs, base = setup_runs(tmp_path)
    write_rlcd_run(runs, "sft_cont", base)  # identical to SFT
    write_rlcd_run(runs, "direct_log", results(2.0))  # sharper
    write_rlcd_run(runs, "brier", results(1.0))
    write_rlcd_run(runs, "outcome", base, seed=1)  # another seed: left out by --seeds 0

    assert cmp.main(["--runs-dir", str(runs), "--seeds", "0", "--resamples", "50"]) == 0
    out = json.loads((runs / "rlcd_stage1_06b" / "metrics.json").read_text())
    assert out["arms"] == {"sft_cont": [0], "direct_log": [0], "brier": [0]}  # ARM_ORDER, this seed only
    assert out["seeds"] == [0] and out["known_broken_arms"] == ["brier"]
    block = out["splits"]["test_emotion"]["overall"]
    cont = block["arms"]["sft_cont"]
    # An arm identical to SFT: zero accuracy difference in every resample; one seed, so the mean is that seed.
    assert cont["mean"]["accuracy_minus_sft"] == 0.0 and cont["mean"]["accuracy_minus_sft_ci"] == [0.0, 0.0]
    assert cont["by_seed"]["0"]["ece"] == cont["mean"]["ece"] and cont["mean"]["ece_range"] == [cont["mean"]["ece"]] * 2
    # Point estimates are jevmark.metrics' own; sft_temp is the SFT probabilities at T 1.5.
    emotion = [r for r in base if r.split == "test_emotion"]
    assert block["sft"]["ece"] == pytest.approx(ece([r.top1 for r in emotion], [r.correct for r in emotion]))
    scaled = scale_results(emotion, 1.5)
    assert block["sft_temp"]["ece"] == pytest.approx(ece([r.top1 for r in scaled], [r.correct for r in scaled]))
    sharp = block["arms"]["direct_log"]["mean"]
    assert sharp["ece_minus_sft_temp_ci"][0] <= sharp["ece_minus_sft_temp"] <= sharp["ece_minus_sft_temp_ci"][1]
    assert set(out["splits"]["test_indomain"]) == {"overall", "noul", "choice"}
    unseen = out["unseen_schemas"]
    assert unseen["splits"] == ["test_agnews", "test_emotion", "test_banking77", "test_yelp"]
    assert unseen["arms"]["sft_cont"]["mean_ece"] == pytest.approx(unseen["mean_ece"]["sft"])
    assert unseen["arms"]["direct_log"]["mean_ece_minus_sft_temp"] == pytest.approx(unseen["arms"]["direct_log"]["mean_ece"] - unseen["mean_ece"]["sft_temp"])
    assert out["temperature_ablation"]["arms"] == {}  # no <run>_temp directories
    v = out["validation"]["direct_log"]["0"]
    assert v["step_0"]["step"] == 0 and v["best"]["step"] == 5 and v["last"]["step"] == 10 and v["last"]["kl_to_ref"] == 0.02
    assert "sft_cont" in capsys.readouterr().out


def test_seeds_are_aggregated_per_arm(tmp_path):
    runs, base = setup_runs(tmp_path)
    write_rlcd_run(runs, "outcome", results(1.0))
    write_rlcd_run(runs, "outcome", results(3.0), seed=1)
    assert cmp.main(["--runs-dir", str(runs), "--stage", "2", "--resamples", "50"]) == 0
    out = json.loads((runs / "rlcd_stage2_06b" / "metrics.json").read_text())
    assert out["seeds"] == [0, 1] and out["arms"] == {"outcome": [0, 1]}
    arm = out["splits"]["test_yelp"]["overall"]["arms"]["outcome"]
    e0, e1 = arm["by_seed"]["0"]["ece"], arm["by_seed"]["1"]["ece"]
    assert e0 != e1 and arm["mean"]["ece"] == pytest.approx((e0 + e1) / 2) and arm["mean"]["ece_range"] == [min(e0, e1), max(e0, e1)]
    assert arm["mean"]["seeds"] == [0, 1]
    unseen = out["unseen_schemas"]["arms"]["outcome"]
    assert unseen["mean_ece"] == pytest.approx(sum(unseen["mean_ece_by_seed"].values()) / 2)
    assert len(out["validation"]["outcome"]) == 2


def test_temperature_ablation_scales_each_arm_by_its_own_temperature(tmp_path):
    runs, base = setup_runs(tmp_path)
    run = write_rlcd_run(runs, "direct_brier", results(2.0))
    (runs / f"{run.name}_temp").mkdir()
    (runs / f"{run.name}_temp" / "calibration.json").write_text(json.dumps({"temperature": 2.0}))
    write_rlcd_run(runs, "brier", results(2.0))  # known broken: never ablated
    (runs / "rlcd_06b_brier_s0_temp").mkdir()
    (runs / "rlcd_06b_brier_s0_temp" / "calibration.json").write_text(json.dumps({"temperature": 2.0}))
    assert cmp.main(["--runs-dir", str(runs), "--resamples", "50"]) == 0
    ablation = json.loads((runs / "rlcd_stage1_06b" / "metrics.json").read_text())["temperature_ablation"]
    assert list(ablation["arms"]) == ["direct_brier"]
    arm = ablation["arms"]["direct_brier"]
    assert arm["temperature_by_seed"] == {"0": 2.0}
    yelp = [r for r in results(2.0) if r.split == "test_yelp"]
    after = scale_results(yelp, 2.0)
    assert arm["splits"]["test_yelp"]["ece_before"] == pytest.approx(ece([r.top1 for r in yelp], [r.correct for r in yelp]))
    assert arm["splits"]["test_yelp"]["ece_after"] == pytest.approx(ece([r.top1 for r in after], [r.correct for r in after]))
    lo, hi = arm["mean_ece_after_minus_sft_temp_ci"]
    assert arm["level_vs_sft_temp"] == ("below" if hi < 0 else "above" if lo > 0 else "level")
    assert ablation["every_arm_back_to_sft_temp_level"] == (arm["level_vs_sft_temp"] in ("level", "below"))


def test_mismatched_questions_are_refused(tmp_path):
    runs = tmp_path / "runs"
    (runs / "sft_06b").mkdir(parents=True)
    write_results(runs / "sft_06b" / "results.jsonl.gz", results(1.0))
    (runs / "sft_06b_temp").mkdir()
    (runs / "sft_06b_temp" / "calibration.json").write_text(json.dumps({"temperature": 1.2}))
    write_rlcd_run(runs, "outcome", results(1.0)[:-2])
    with pytest.raises(SystemExit, match="not comparable"):
        cmp.main(["--runs-dir", str(runs), "--resamples", "10"])


def test_point_estimates_must_match_the_run_metrics(tmp_path):
    runs = tmp_path / "runs"
    (runs / "sft_06b").mkdir(parents=True)
    write_results(runs / "sft_06b" / "results.jsonl.gz", results(1.0))
    (runs / "sft_06b_temp").mkdir()
    (runs / "sft_06b_temp" / "calibration.json").write_text(json.dumps({"temperature": 1.2}))
    run = write_rlcd_run(runs, "outcome", results(1.0))
    (run / "metrics.json").write_text(json.dumps({"splits": {"valid": {"overall": {"accuracy": 0.0, "ece": 0.0, "nll": 0.0}}}}))
    with pytest.raises(SystemExit, match="recomputed"):
        cmp.main(["--runs-dir", str(runs), "--resamples", "10"])
