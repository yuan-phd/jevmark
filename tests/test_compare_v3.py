"""scripts/compare_v3.py on synthetic runs (task 3.4)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from jevmark.data.build import V3_TEST_FULL
from jevmark.metrics import QuestionResult, write_results

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_v3", REPO / "scripts" / "compare_v3.py")
compare_v3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare_v3
spec.loader.exec_module(compare_v3)

LABELS = ("a", "b", "c", "other")
SPLITS = {V3_TEST_FULL: 60, "test_banking77": 20, "test_indomain": 30, "test_unseen_intents": 30}
HASHES = {f"{s}.jsonl": f"{i}" * 64 for i, s in enumerate(SPLITS)}


def results(sharpness: float, seed: int, other_bias: float = 0.0):
    """Synthetic choice results: gold cycles over a, b, c; the model puts `sharpness` on gold with noise."""
    rng = np.random.default_rng(seed)
    out = []
    for split, n in SPLITS.items():
        for i in range(n):
            gold = i % 3
            logits = rng.normal(0, 1, len(LABELS))
            logits[gold] += sharpness
            logits[3] += other_bias
            p = np.exp(logits) / np.exp(logits).sum()
            record = f"banking77-test_banking77-{i:06d}" if split in (V3_TEST_FULL, "test_banking77") and i < 20 else f"{split}-{i}"
            out.append(QuestionResult(record, "label", "choice", tuple(float(x) for x in p), gold, LABELS, split=split))
    return out


def write_run(runs_dir: Path, name: str, rs, hashes=HASHES, **extra):
    run = runs_dir / name
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", rs)
    (run / "metrics.json").write_text(json.dumps({"git": {"commit": "c" * 40, "dirty": False}, "data_files_sha256": hashes, **extra}))
    return run


@pytest.fixture
def runs(tmp_path):
    runs_dir = tmp_path / "runs"
    zero = results(1.0, 0)
    write_run(runs_dir, "v3_06b_zeroshot", zero, adapter_sha256="f" * 64)
    for n in (500, 2000, 5000):
        temp = runs_dir / f"v3_06b_temp_n{n}"
        temp.mkdir(parents=True)
        (temp / "calibration.json").write_text(json.dumps({"temperature": 1.5}))
        for k, arm in enumerate(("positive_sft", "full_sft", "direct_brier")):
            write_run(runs_dir, f"v3_06b_{arm}_n{n}_s0", results(1.0 + n / 2000 + k, 10 + k, other_bias=-2.0 if arm == "direct_brier" else 0.0))
    for s in (1, 2):
        write_run(runs_dir, f"v3_06b_direct_brier_n5000_s{s}", results(4.5, 20 + s))
        write_run(runs_dir, f"v3_06b_full_sft_n5000_s{s}", results(5.0, 50 + s))
    write_run(runs_dir, "v3_06b_direct_brier_n5000_s0_log1", results(4.4, 30))
    write_run(runs_dir, "v3_06b_direct_brier_n5000_s0_noisy", results(3.0, 40))
    write_run(runs_dir, "v3_06b_positive_sft_n5000_s0_noisy", results(1.5, 41))
    for name in [f"v3_06b_direct_brier_n{n}_s0" for n in (500, 2000, 5000)] + ["v3_06b_direct_brier_n5000_s1", "v3_06b_direct_brier_n5000_s2"]:
        (runs_dir / f"{name}_temp").mkdir()
        (runs_dir / f"{name}_temp" / "calibration.json").write_text(json.dumps({"temperature": 1.3}))
    b2 = runs_dir / "b2_gpt-4.1-mini"
    b2.mkdir()
    (b2 / "metrics.json").write_text(json.dumps({"splits": {"test_banking77": {"overall": {"accuracy_all": 0.918, "ece": 0.035, "n": 500}}}}))
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"splits": {"test_banking77": [f"banking77-test_banking77-{i:06d}" for i in range(0, 20, 2)]}, "sub_splits": {}}))
    return runs_dir, subset


def test_compare_writes_every_table(runs):
    runs_dir, subset = runs
    assert compare_v3.main(["--runs-dir", str(runs_dir), "--subset", str(subset), "--resamples", "50"]) == 0
    result = json.loads((runs_dir / "v3_stage_06b" / "metrics.json").read_text())
    assert set(result["n_curve"]) == {"500", "2000", "5000"} and result["missing"] == []
    row = result["n_curve"]["5000"]
    for learner in compare_v3.LEARNERS:
        for metric in compare_v3.METRICS:
            lo, hi = row[learner][f"{metric}_ci"]
            assert lo <= row[learner][metric] <= hi or math.isclose(lo, hi), (learner, metric)
    assert row["zero_shot"] == result["n_curve"]["500"]["zero_shot"]  # the zero-shot learner does not depend on N
    assert row["temperature"]["accuracy"] == row["zero_shot"]["accuracy"] and row["temperature"]["ece"] != row["zero_shot"]["ece"]
    delta = row["direct_brier_minus_positive_sft"]["accuracy"]
    assert delta["delta"] == pytest.approx(row["direct_brier"]["accuracy"] - row["positive_sft"]["accuracy"])
    assert set(result["seeds"]) >= {"seed_0", "seed_1", "seed_2", "log_seed_1", "training_seeds", "minus_seed_0", "training_seed_mean_minus_positive_sft", "training_seed_mean_minus_full_sft"}
    mean_minus = result["seeds"]["training_seed_mean_minus_positive_sft"]["accuracy"]["delta"]
    assert mean_minus == pytest.approx(result["seeds"]["training_seeds"]["accuracy"]["mean"] - row["positive_sft"]["accuracy"])
    assert result["seeds"]["training_seeds"]["accuracy"]["n_seeds"] == 3
    full = result["seeds"]["full_sft"]
    assert full["training_seeds"]["accuracy"]["n_seeds"] == 3 and set(full["minus_seed_0"]) == {"seed_1", "seed_2"}
    both = result["seeds"]["training_seed_mean_minus_full_sft_seed_mean"]["accuracy"]["delta"]
    assert both == pytest.approx(result["seeds"]["training_seeds"]["accuracy"]["mean"] - full["training_seeds"]["accuracy"]["mean"])
    assert result["noisy"]["direct_brier"]["noisy_minus_clean"]["accuracy"]["delta"] < 0 and result["noisy"]["direct_brier_change_minus_positive_sft_change"] is not None
    for split in ("test_indomain", "test_unseen_intents"):
        assert "direct_brier_n5000_s0" in result["forgetting"][split]["minus_zero_shot"] and "temperature_n500" in result["forgetting"][split]["minus_zero_shot"]
    assert result["other"]["direct_brier_n5000_s0"]["predicted_other_rate"] <= result["other"]["zero_shot"]["predicted_other_rate"]
    assert set(result["coverage"]["zero_shot"]) == {"0.80", "0.90", "0.95"}
    assert row["direct_brier_temp"]["accuracy"] == row["direct_brier"]["accuracy"] and row["direct_brier_temp"]["ece"] != row["direct_brier"]["ece"]
    ece = row["direct_brier_temp_minus_direct_brier"]["ece"]
    assert ece["delta"] == pytest.approx(row["direct_brier_temp"]["ece"] - row["direct_brier"]["ece"]) and ece["ci"][0] <= ece["ci"][1]
    assert row["direct_brier_temp_minus_temperature"]["accuracy"]["delta"] == pytest.approx(row["direct_brier"]["accuracy"] - row["temperature"]["accuracy"])
    assert result["direct_brier_temperatures"] == {f"direct_brier_temp_n{n}_s0": 1.3 for n in (500, 2000, 5000)} | {f"direct_brier_temp_n5000_s{s}": 1.3 for s in (1, 2)}
    assert result["seeds"]["direct_brier_temp"]["training_seeds"]["accuracy"]["mean"] == pytest.approx(result["seeds"]["training_seeds"]["accuracy"]["mean"])
    assert result["seeds"]["direct_brier_temp_seed_mean_minus_direct_brier_seed_mean"]["accuracy"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert "direct_brier_temp_n5000_s0" in result["coverage"]
    assert result["b2_reference"]["b2"]["accuracy_all"] == 0.918 and result["b2_reference"]["runs"]["zero_shot"]["n"] == 10


def test_paired_difference_of_a_run_with_itself_is_zero(runs):
    runs_dir, subset = runs
    rs = compare_v3.by_split(compare_v3.read_results(runs_dir / "v3_06b_zeroshot" / "results.jsonl.gz"))[V3_TEST_FULL]
    split = compare_v3.Split(V3_TEST_FULL, {"zero_shot": rs, "copy": list(rs)}, 50)
    for metric, d in split.delta("copy", "zero_shot").items():
        assert d["delta"] == 0 and d["ci"] == [0.0, 0.0], metric


def test_mean_delta_by_hand(runs):
    runs_dir, subset = runs
    read = lambda name: compare_v3.by_split(compare_v3.read_results(runs_dir / name / "results.jsonl.gz"))[V3_TEST_FULL]
    names = ["v3_06b_direct_brier_n5000_s1", "v3_06b_direct_brier_n5000_s2"]
    split = compare_v3.Split(V3_TEST_FULL, {"zero_shot": read("v3_06b_zeroshot"), **{n: read(n) for n in names}}, 50)
    got = split.mean_delta(names, "zero_shot")
    for metric in compare_v3.METRICS:
        draws = (split.draws[names[0]][metric] + split.draws[names[1]][metric]) / 2 - split.draws["zero_shot"][metric]
        assert got[metric]["ci"] == pytest.approx(compare_v3.interval(draws))
        assert got[metric]["delta"] == pytest.approx((split.delta(names[0], "zero_shot")[metric]["delta"] + split.delta(names[1], "zero_shot")[metric]["delta"]) / 2)
    assert split.mean_delta(["zero_shot"], "zero_shot")["accuracy"] == {"delta": 0.0, "ci": [0.0, 0.0]}


def test_mean_minus_mean_by_hand(runs):
    runs_dir, subset = runs
    read = lambda name: compare_v3.by_split(compare_v3.read_results(runs_dir / name / "results.jsonl.gz"))[V3_TEST_FULL]
    a, b = ["v3_06b_direct_brier_n5000_s1", "v3_06b_direct_brier_n5000_s2"], ["v3_06b_full_sft_n5000_s1", "v3_06b_full_sft_n5000_s2", "v3_06b_zeroshot"]
    split = compare_v3.Split(V3_TEST_FULL, {"v3_06b_zeroshot": read("v3_06b_zeroshot"), **{n: read(n) for n in a + b[:2]}}, 50)
    got = split.mean_minus_mean(a, b)
    for metric in compare_v3.METRICS:
        draws = (split.draws[a[0]][metric] + split.draws[a[1]][metric]) / 2 - sum(split.draws[n][metric] for n in b) / 3
        assert got[metric]["ci"] == pytest.approx(compare_v3.interval(draws))
    assert split.mean_minus_mean(a, a)["accuracy"] == {"delta": 0.0, "ci": [0.0, 0.0]}


def test_runs_with_other_questions_or_data_are_refused(runs, tmp_path):
    runs_dir, subset = runs
    write_run(runs_dir, "v3_06b_full_sft_n500_s7", results(2.0, 5)[:-3])
    with pytest.raises(SystemExit, match="questions differ"):
        compare_v3.compare(runs_dir, "06b", 20, subset)
    other = tmp_path / "other"
    other.mkdir()
    for path in runs_dir.iterdir():
        (other / path.name).symlink_to(path)
    (other / "v3_06b_full_sft_n500_s7").unlink()
    write_run(other, "v3_06b_full_sft_n500_s7", results(2.0, 5), hashes={**HASHES, f"{V3_TEST_FULL}.jsonl": "9" * 64})
    with pytest.raises(SystemExit, match="sha256"):
        compare_v3.compare(other, "06b", 20, subset)


def test_missing_runs_are_listed_and_the_zero_shot_run_is_required(runs, tmp_path):
    runs_dir, subset = runs
    import shutil

    shutil.rmtree(runs_dir / "v3_06b_full_sft_n2000_s0")
    result = compare_v3.compare(runs_dir, "06b", 20, subset)
    assert result["missing"] == ["full_sft_n2000_s0"] and result["n_curve"]["2000"]["full_sft"] is None and result["n_curve"]["2000"]["direct_brier_minus_full_sft"] is None
    with pytest.raises(SystemExit, match="zeroshot"):
        compare_v3.compare(tmp_path / "empty", "06b", 20, subset)


def test_run_names_parse():
    m = compare_v3.RUN_PATTERN.match("v3_06b_direct_brier_n5000_s0_noisy_log1")
    assert (m["arm"], m["n"], m["seed"], bool(m["noisy"]), m["log"]) == ("direct_brier", "5000", "0", True, "1")
    assert compare_v3.RUN_PATTERN.match("v3_06b_temp_n500") is None and compare_v3.RUN_PATTERN.match("v3_06b_zeroshot") is None
