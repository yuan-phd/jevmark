"""scripts/compare_v3.py on synthetic runs (task 3.4)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from jevmark.data.build import V3_TEST_FULL
from jevmark.metrics import QuestionResult, aurc, coverage_at_risk, write_results

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


def test_risk_coverage_columns_match_the_metrics_module_and_stay_off_the_forgetting_splits(runs):
    runs_dir, subset = runs
    result = compare_v3.compare(runs_dir, "06b", 50, subset)
    rs = compare_v3.by_split(compare_v3.read_results(runs_dir / "v3_06b_direct_brier_n5000_s0" / "results.jsonl.gz"))[V3_TEST_FULL]
    point = result["n_curve"]["5000"]["direct_brier"]
    assert point["aurc"] == pytest.approx(aurc(rs)) and point["aurc_entropy"] == pytest.approx(aurc(rs, by="entropy"))
    zero = result["n_curve"]["5000"]["zero_shot"]
    assert zero["aurc"] != zero["aurc_entropy"]  # the weak synthetic run ranks differently by top-1 and by entropy
    for by, suffix in (("top1", ""), ("entropy", "_entropy")):
        for eps, block in coverage_at_risk(rs, by=by).items():
            assert point[f"coverage_at_{eps}{suffix}"] == pytest.approx(block["coverage"])
    assert set(result["n_curve"]["5000"]["direct_brier_minus_positive_sft"]) == set(compare_v3.METRICS)
    for split in ("test_indomain", "test_unseen_intents"):
        assert "aurc" not in result["forgetting"][split]["zero_shot"] and "accuracy" in result["forgetting"][split]["zero_shot"]


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


def test_off_policy_arms_are_columns_with_paired_deltas_and_leave_phase_3_values_unchanged(runs):
    """The off-policy arms direct_log, outcome_minus_p_log, ips_log and ips_positive at N 2000 and 5000."""
    runs_dir, subset = runs
    before = compare_v3.compare(runs_dir, "06b", 30, subset)
    assert before["missing_off_policy"] == sorted(f"{a}_n{n}_s0" for a in compare_v3.OFF_POLICY_ARMS for n in (2000, 5000))
    assert before["n_curve"]["5000"]["off_policy"] == {} and before["n_curve"]["5000"]["ips_log"] is None
    for k, arm in enumerate(compare_v3.OFF_POLICY_ARMS):
        for n in (2000, 5000):
            write_run(runs_dir, f"v3_06b_{arm}_n{n}_s0", results(2.0 + k, 60 + k, other_bias=3.0))  # predicts other often
    after = compare_v3.compare(runs_dir, "06b", 30, subset)
    assert after["missing_off_policy"] == [] and after["missing"] == []
    for n in ("2000", "5000"):
        row = after["n_curve"][n]
        assert set(row["off_policy"]) == set(compare_v3.OFF_POLICY_ARMS)
        for arm in compare_v3.OFF_POLICY_ARMS:
            for other, key in (("direct_brier", "minus_direct_brier"), ("positive_sft", "minus_positive_sft")):
                d = row["off_policy"][arm][key]["accuracy"]
                assert d["delta"] == pytest.approx(row[arm]["accuracy"] - row[other]["accuracy"]) and d["ci"][0] <= d["ci"][1]
    assert after["n_curve"]["500"]["off_policy"] == {} and after["n_curve"]["500"]["direct_log"] is None
    assert "ips_positive_n5000_s0" in after["forgetting"]["test_indomain"]["minus_zero_shot"] and "ips_log_n2000_s0" in after["coverage"]
    # The off-policy runs predict other often, yet the phase 3 "no run predicted other" set and its values do not move.
    assert after["other"]["n_questions_where_no_run_predicted_other"] == before["other"]["n_questions_where_no_run_predicted_other"]
    for name in before["other"]:
        assert after["other"][name] == before["other"][name], name
    for learner in compare_v3.LEARNERS:
        assert after["n_curve"]["5000"][learner] == before["n_curve"]["5000"][learner]
    assert compare_v3.RUN_PATTERN.match("v3_06b_outcome_minus_p_log_n5000_s0_noisy")["arm"] == "outcome_minus_p_log"


def test_the_1_7b_comparison_has_only_its_n_5000_runs_and_lists_the_rest_as_missing(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "v3_17b_zeroshot", results(1.5, 0))
    for k, arm in enumerate(("positive_sft", "full_sft", "direct_brier")):
        write_run(runs_dir, f"v3_17b_{arm}_n5000_s0", results(2.5 + k, 60 + k))
    write_run(runs_dir, "v3_06b_direct_brier_n5000_s0", results(9.0, 70))  # another size is never read
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"splits": {"test_banking77": []}, "sub_splits": {}}))
    out = tmp_path / "stage17.json"
    assert compare_v3.main(["--size", "17b", "--runs-dir", str(runs_dir), "--subset", str(subset), "--resamples", "50", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    row = m["n_curve"]["5000"]
    assert m["size"] == "17b" and all(row[arm] for arm in ("zero_shot", "positive_sft", "full_sft", "direct_brier"))
    assert row["direct_brier_minus_positive_sft"] and row["direct_brier_minus_full_sft"] and row["temperature"] is None
    assert m["n_curve"]["500"]["direct_brier"] is None and "direct_brier_n500_s0" in m["missing"]
    assert set(m["sources"]) == {"zero_shot", "positive_sft_n5000_s0", "full_sft_n5000_s0", "direct_brier_n5000_s0"}
    assert all(Path(source["dir"]).name.startswith("v3_17b_") for source in m["sources"].values())
    assert set(m["forgetting"]["test_indomain"]["minus_zero_shot"]) == {"positive_sft_n5000_s0", "full_sft_n5000_s0", "direct_brier_n5000_s0"}



def test_the_1_7b_direct_brier_seeds_get_a_mean_range_and_paired_seed_mean_deltas_against_single_seed_learners(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "v3_17b_zeroshot", results(1.5, 0))
    for k, arm in enumerate(("positive_sft", "full_sft", "direct_brier")):
        write_run(runs_dir, f"v3_17b_{arm}_n5000_s0", results(2.5 + k, 60 + k))
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"splits": {"test_banking77": []}, "sub_splits": {}}))
    before = compare_v3.compare(runs_dir, "17b", 40, subset)
    assert before["seeds"]["training_seeds"]["accuracy"]["n_seeds"] == 1 and before["seeds"]["minus_seed_0"] == {}
    for seed in (1, 2):  # session L
        write_run(runs_dir, f"v3_17b_direct_brier_n5000_s{seed}", results(4.5 + 0.3 * seed, 80 + seed))
    after = compare_v3.compare(runs_dir, "17b", 40, subset)
    seeds = after["seeds"]
    points = [seeds[f"seed_{k}"] for k in (0, 1, 2)]
    for metric in compare_v3.METRICS:
        block = seeds["training_seeds"][metric]
        assert block["n_seeds"] == 3 and block["range"] == [min(p[metric] for p in points), max(p[metric] for p in points)]
        assert math.isclose(block["mean"], sum(p[metric] for p in points) / 3)
        # Paired: each draw averages the three seeds' differences, so the point is the mean of the per-seed differences.
        for other in ("positive_sft", "full_sft"):
            other_point = after["n_curve"]["5000"][other][metric]
            assert math.isclose(seeds[f"training_seed_mean_minus_{other}"][metric]["delta"], block["mean"] - other_point, abs_tol=1e-12)
            lo, hi = seeds[f"training_seed_mean_minus_{other}"][metric]["ci"]
            assert lo <= hi
    # full_sft has one seed at 1.7B: its seed mean is seed 0, so the two forms of the difference agree.
    assert seeds["full_sft"]["training_seeds"]["accuracy"]["n_seeds"] == 1
    assert seeds["training_seed_mean_minus_full_sft_seed_mean"] == seeds["training_seed_mean_minus_full_sft"]
    assert set(seeds["minus_seed_0"]) == {"seed_1", "seed_2"}
    # Adding the seeds leaves every single-seed value as it was.
    assert after["n_curve"] == before["n_curve"] and seeds["seed_0"] == before["seeds"]["seed_0"]
    assert set(after["sources"]) == set(before["sources"]) | {"direct_brier_n5000_s1", "direct_brier_n5000_s2"}

def test_direct_log_seeds_get_a_mean_range_and_a_paired_difference_from_direct_brier_seeds(runs):
    runs_dir, subset = runs
    before = compare_v3.compare(runs_dir, "06b", 30, subset)
    assert before["seeds"]["direct_log"]["seed_0"] is None and "training_seeds" not in before["seeds"]["direct_log"]
    assert before["seeds"]["direct_log_seed_mean_minus_direct_brier_seed_mean"] is None
    for s in (0, 1, 2):
        write_run(runs_dir, f"v3_06b_direct_log_n5000_s{s}", results(4.0 + s / 4, 80 + s))
    after = compare_v3.compare(runs_dir, "06b", 30, subset)
    block = after["seeds"]["direct_log"]
    points = [block[f"seed_{s}"] for s in (0, 1, 2)]
    for metric in compare_v3.METRICS:
        values = [p[metric] for p in points]
        assert block["training_seeds"][metric]["mean"] == pytest.approx(sum(values) / 3)
        assert block["training_seeds"][metric]["range"] == [min(values), max(values)] and block["training_seeds"][metric]["n_seeds"] == 3
    assert block["minus_seed_0"]["seed_2"]["accuracy"]["delta"] == pytest.approx(points[2]["accuracy"] - points[0]["accuracy"])
    brier = after["seeds"]["training_seeds"]["accuracy"]["mean"]
    d = after["seeds"]["direct_log_seed_mean_minus_direct_brier_seed_mean"]
    assert d["accuracy"]["delta"] == pytest.approx(block["training_seeds"]["accuracy"]["mean"] - brier) and d["nll"]["ci"][0] <= d["nll"]["ci"][1]
    # Adding the seed runs changes no earlier entry of the seeds block.
    for key in ("seed_0", "seed_1", "seed_2", "training_seeds", "training_seed_mean_minus_positive_sft", "full_sft", "training_seed_mean_minus_full_sft_seed_mean"):
        assert after["seeds"][key] == before["seeds"][key], key


EMOTIONS = ("anger", "fear", "joy", "love", "sadness", "surprise")
EMOTION_SPLITS = {"v3_emotion_test_full": 60, "test_emotion": 20, "test_indomain": 30, "test_unseen_intents": 30}


def emotion_results(sharpness: float, seed: int):
    """Synthetic emotion results: a six-way choice per record and, on the emotion splits, an expresses_emotion noul after it."""
    rng = np.random.default_rng(seed)
    out = []
    for split, n in EMOTION_SPLITS.items():
        for i in range(n):
            gold = i % 6
            logits = rng.normal(0, 1, 6)
            logits[gold] += sharpness
            p = np.exp(logits) / np.exp(logits).sum()
            record = f"emotion-test_emotion-{i:06d}" if split in ("v3_emotion_test_full", "test_emotion") and i < 20 else f"{split}-{i}"
            out.append(QuestionResult(record, "label", "choice", tuple(float(x) for x in p), gold, EMOTIONS, split=split))
            if split in ("v3_emotion_test_full", "test_emotion"):  # a noul the comparison must leave out: always wrong here
                out.append(QuestionResult(record, "expresses_emotion", "noul", (0.01, 0.99), 0, ("true", "false"), split=split, kind="expresses_emotion"))
    return out


def test_the_emotion_comparison_reads_its_own_runs_and_scores_the_choice_question_only(runs, tmp_path):
    runs_dir, _ = runs  # the Banking77 runs sit in the same directory and must not be read
    emotion_hashes = {f"{s}.jsonl": f"{i}" * 64 for i, s in enumerate(EMOTION_SPLITS)}
    write_run(runs_dir, "v3_emotion_06b_zeroshot", emotion_results(1.0, 0), emotion_hashes)
    for k, arm in enumerate(("positive_sft", "full_sft", "direct_brier")):
        for n in (500, 2000, 5000):
            write_run(runs_dir, f"v3_emotion_06b_{arm}_n{n}_s0", emotion_results(1.5 + n / 2000 + k, 100 + k), emotion_hashes)
    for s in (1, 2):
        write_run(runs_dir, f"v3_emotion_06b_direct_brier_n5000_s{s}", emotion_results(4.6, 110 + s), emotion_hashes)
    write_run(runs_dir, "v3_emotion_06b_direct_log_n5000_s0", emotion_results(4.4, 120), emotion_hashes)
    temp = runs_dir / "v3_emotion_06b_temp_n5000"
    temp.mkdir()
    (temp / "calibration.json").write_text(json.dumps({"temperature": 1.4}))
    (runs_dir / "b2_gpt-4.1-mini" / "metrics.json").write_text(json.dumps({"splits": {
        "test_banking77": {"overall": {"accuracy_all": 0.918, "ece": 0.035, "n": 500}, "choice": {"accuracy_all": 0.918, "ece": 0.035, "n": 500}},
        "test_emotion": {"overall": {"accuracy_all": 0.656, "ece": 0.242, "n": 1000}, "choice": {"accuracy_all": 0.598, "ece": 0.283, "n": 500}}}}))
    subset = tmp_path / "emotion_subset.json"
    subset.write_text(json.dumps({"splits": {"test_emotion": [f"emotion-test_emotion-{i:06d}" for i in range(0, 20, 2)]}, "sub_splits": {}}))
    out = tmp_path / "stage_emotion.json"
    assert compare_v3.main(["--domain", "emotion", "--runs-dir", str(runs_dir), "--subset", str(subset), "--resamples", "50", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["domain"] == "emotion" and m["split"] == "v3_emotion_test_full"
    assert all(Path(s["dir"]).name.startswith("v3_emotion_06b_") for s in m["sources"].values() if "dir" in s)
    assert m["missing"] == [] and m["n_curve"]["5000"]["temperature"] and m["seeds"]["training_seeds"]["accuracy"]["n_seeds"] == 3
    assert m["n_curve"]["5000"]["direct_log"] and "direct_log_n2000_s0" in m["missing_off_policy"]
    # Only the choice question counts: 60 questions on the full test, and the always-wrong noul does not pull accuracy down.
    zero = m["n_curve"]["500"]["zero_shot"]
    choice = [r for r in emotion_results(1.0, 0) if r.split == "v3_emotion_test_full" and r.question_id == "label"]
    assert zero["n"] == 60 and zero["accuracy"] == pytest.approx(sum(r.correct for r in choice) / 60)
    assert m["other"] is None and "no other option" in m["other_note"]
    assert m["b2_reference"]["b2"] == {"accuracy_all": 0.598, "ece": 0.283, "n": 500} and m["b2_reference"]["runs"]["zero_shot"]["n"] == 10
    assert set(m["forgetting"]["test_indomain"]["minus_zero_shot"]) >= {"direct_brier_n5000_s0", "direct_log_n5000_s0"}
    # The Banking77 comparison of the same directory is untouched by the emotion runs.
    banking = compare_v3.compare(runs_dir, "06b", 30, subset_path=runs[1])
    assert banking["domain"] == "banking77" and all(not Path(s["dir"]).name.startswith("v3_emotion") for s in banking["sources"].values() if "dir" in s)
    assert banking["b2_reference"]["b2"]["accuracy_all"] == 0.918
