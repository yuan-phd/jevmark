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


def write_rlcd_run(runs, arm, rs):
    run = runs / f"rlcd_06b_{arm}_s0"
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", rs)
    valid = lambda step, kl: {"event": "valid", "step": step, "accuracy": 0.9, "ece": 0.02, "nll": 0.2 + kl, "kl_to_ref": kl, "expected_p_chosen": 0.9 + kl, "expected_reward": 0.95}
    log = [valid(0, 0.0), {"step": 1, "loss": 0.1}, valid(5, 0.01), valid(10, 0.02)]
    (run / "training_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    final = {k: 0.5 for k in cmp.VALID_KEYS}
    (run / "train_summary.json").write_text(json.dumps({"best_step": 5, "steps": 10, "final_valid": final, "final_valid_last": final}))
    return run


def test_compare_on_synthetic_runs(tmp_path, capsys):
    runs = tmp_path / "runs"
    sft = runs / "sft_06b"
    sft.mkdir(parents=True)
    base = results(1.0)
    write_results(sft / "results.jsonl.gz", base)
    (runs / "sft_06b_temp").mkdir()
    (runs / "sft_06b_temp" / "calibration.json").write_text(json.dumps({"temperature": 1.5}))
    write_rlcd_run(runs, "sft_cont", base)  # identical to SFT
    write_rlcd_run(runs, "direct_log", results(2.0))  # sharper
    write_rlcd_run(runs, "brier", results(1.0))
    write_rlcd_run(runs, "other_arm", base).rename(runs / "rlcd_06b_other_arm_s1")  # another seed: ignored

    assert cmp.main(["--runs-dir", str(runs), "--resamples", "50"]) == 0
    out = json.loads((runs / "rlcd_stage1_06b" / "metrics.json").read_text())
    assert list(out["runs"]) == ["sft", "sft_temp", "sft_cont", "direct_log", "brier"]  # ARM_ORDER, this seed only
    assert out["known_broken_arms"] == ["brier"]
    block = out["splits"]["test_emotion"]["overall"]
    # An arm identical to SFT: zero accuracy difference in every resample.
    assert block["sft_cont"]["accuracy_minus_sft"] == 0.0 and block["sft_cont"]["accuracy_minus_sft_ci"] == [0.0, 0.0]
    # Point estimates are jevmark.metrics' own; sft_temp is the SFT probabilities at T 1.5.
    emotion = [r for r in base if r.split == "test_emotion"]
    assert block["sft"]["ece"] == pytest.approx(ece([r.top1 for r in emotion], [r.correct for r in emotion]))
    scaled = scale_results(emotion, 1.5)
    assert block["sft_temp"]["ece"] == pytest.approx(ece([r.top1 for r in scaled], [r.correct for r in scaled]))
    lo, hi = block["direct_log"]["ece_minus_sft_temp_ci"]
    assert lo <= block["direct_log"]["ece_minus_sft_temp"] <= hi
    assert set(out["splits"]["test_indomain"]) == {"overall", "noul", "choice"}
    unseen = out["unseen_schemas"]
    assert unseen["splits"] == ["test_agnews", "test_emotion", "test_banking77", "test_yelp"]
    assert unseen["mean_ece"]["sft_cont"] == pytest.approx(unseen["mean_ece"]["sft"])
    assert unseen["mean_ece_minus_sft_temp"]["direct_log"] == pytest.approx(unseen["mean_ece"]["direct_log"] - unseen["mean_ece"]["sft_temp"])
    v = out["validation"]["direct_log"]
    assert v["step_0"]["step"] == 0 and v["best"]["step"] == 5 and v["last"]["step"] == 10 and v["last"]["kl_to_ref"] == 0.02
    assert "sft_cont" in capsys.readouterr().out


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
