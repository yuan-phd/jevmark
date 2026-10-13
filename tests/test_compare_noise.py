"""scripts/compare_noise.py, the noise-rate sweep table, on synthetic runs."""

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


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_noise_test", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_noise = load_script("compare_noise")
invert_noisy = load_script("invert_noisy")

LABELS = ("a", "b", "c", "other")
HASHES = {f"{V3_TEST_FULL}.jsonl": "0" * 64}


def results(sharpness: float, seed: int, n: int = 80):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = i % 3
        logits = rng.normal(0, 1, len(LABELS))
        logits[gold] += sharpness
        p = np.exp(logits) / np.exp(logits).sum()
        out.append(QuestionResult(f"r{i}", "label", "choice", tuple(float(x) for x in p), gold, LABELS, split=V3_TEST_FULL))
    return out


def write_run(runs_dir: Path, name: str, rs, curve=None, best=None):
    run = runs_dir / name
    run.mkdir(parents=True)
    write_results(run / "results.jsonl.gz", rs)
    (run / "metrics.json").write_text(json.dumps({"git": {"commit": "c" * 40, "dirty": False}, "data_files_sha256": HASHES}))
    if curve is not None:
        (run / "train_summary.json").write_text(json.dumps({"best_step": best, "selection_curve": [{"step": s, "criterion": c} for s, c in curve]}))
    return run


CURVES = {  # (curve, best step) per (arm, flip)
    ("direct_brier", None): ([(0, -0.30), (50, -0.12), (100, -0.10), (150, -0.16)], 100),
    ("positive_sft", None): ([(0, -0.30), (50, -0.40), (100, -0.50), (150, -0.60)], 50),
}
for f in (0.1, 0.2, 0.3):
    h = -(f * math.log(f) + (1 - f) * math.log(1 - f))
    CURVES[("direct_brier", f)] = ([(0, -h - 0.3), (50, -h - 0.05), (100, -h - 0.08), (150, -h - 0.2)], 50)
    CURVES[("positive_sft", f)] = ([(0, -h - 0.3), (50, -h - 0.4), (100, -h - 0.6), (150, -h - 0.9)], 50)


@pytest.fixture
def runs(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "v3_06b_zeroshot", results(1.0, 0))
    clean = {}
    for k, arm in enumerate(("direct_brier", "positive_sft")):
        curve, best = CURVES[(arm, None)]
        clean[arm] = write_run(runs_dir, compare_noise.run_name("06b", arm, None), results(3.0 - k, 10 + k), curve, best)
    write_run(runs_dir, f"{compare_noise.run_name('06b', 'direct_brier', None)}_last", results(3.1, 12))
    for j, f in enumerate((0.1, 0.2, 0.3)):
        for k, arm in enumerate(("direct_brier", "positive_sft")):
            if (arm, f) == ("positive_sft", 0.3):
                continue  # left missing on purpose
            curve, best = CURVES[(arm, f)]
            run = write_run(runs_dir, compare_noise.run_name("06b", arm, f), results(2.5 - k - j / 2, 20 + 10 * j + k), curve, best)
            invert_noisy.run(run, f, V3_TEST_FULL, clean[arm], runs_dir / f"{run.name}_inverted")
            if arm == "direct_brier":
                last = write_run(runs_dir, f"{run.name}_last", results(1.5 - j / 2, 40 + j))
                invert_noisy.run(last, f, V3_TEST_FULL, clean[arm], runs_dir / f"{last.name}_inverted")
    return runs_dir


def test_run_names_follow_train_rlcd():
    assert compare_noise.run_name("06b", "direct_brier", None) == "v3_06b_direct_brier_n5000_s0"
    assert compare_noise.run_name("06b", "direct_brier", 0.2) == "v3_06b_direct_brier_n5000_s0_noisy"
    assert compare_noise.run_name("06b", "positive_sft", 0.1) == "v3_06b_positive_sft_n5000_s0_noisy_flip0.1"
    spec = importlib.util.spec_from_file_location("train_rlcd_noise_test", REPO / "scripts" / "train_rlcd.py")
    rl = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rl
    spec.loader.exec_module(rl)
    for f in (0.1, 0.2, 0.3):
        assert compare_noise.run_name("06b", "positive_sft", f) == rl.log_run_name({"size": "06b", "arm": "positive_sft", "seed": 0}, 5000, "fixed", 0, f)


def test_the_sweep_table_by_hand(runs):
    out = runs / "sweep.json"
    assert compare_noise.main(["--runs-dir", str(runs), "--resamples", "200", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["rates"] == [0.1, 0.2, 0.3]
    assert len(m["missing"]) == 1 and m["missing"][0].endswith("v3_06b_positive_sft_n5000_s0_noisy_flip0.3")
    assert m["by_rate"]["0.3"]["positive_sft"] is None and m["by_rate"]["0.3"]["direct_brier_loss_minus_positive_sft_loss"] is None

    row = m["by_rate"]["0.1"]
    assert row["floor"] == pytest.approx(0.1 * math.log(0.1) + 0.9 * math.log(0.9)) and row["amplification"] == pytest.approx(1.25)
    db, ps = row["direct_brier"], row["positive_sft"]
    for r, arm, f in ((db, "direct_brier", 0.1), (ps, "positive_sft", 0.1)):
        noisy_acc = r["accuracy"]
        clean_acc = m["clean"][arm]["accuracy"]
        assert r["minus_clean"]["accuracy"]["delta"] == pytest.approx(noisy_acc - clean_acc)
        inverted = json.loads((runs / f"{compare_noise.run_name('06b', arm, f)}_inverted" / "metrics.json").read_text())
        assert r["accuracy"] == pytest.approx(inverted["stored"]["accuracy"])
        ch = r["channel"]["channel_scale"]
        assert ch["channel_ece"] == inverted["stored_channel"]["ece"] and ch["reference_ece"] == inverted["clean_reference"]["mapped_channel"]["ece"]
        assert ch["channel_line"] == pytest.approx(f + (1 - 2 * f) * noisy_acc)
        assert r["channel"]["inverted_top1_ece"] == inverted["inverted_top1_only"]["ece"]
    # selection: floor -H(f) minus the criterion, at the best (50) and the last (150) step
    assert db["selection"]["excess_best"] == pytest.approx(0.05) and db["selection"]["excess_last"] == pytest.approx(0.2)
    assert db["selection"]["growth"] == pytest.approx(0.15)
    assert m["clean"]["direct_brier"]["selection"]["excess_best"] == pytest.approx(0.10) and m["clean"]["direct_brier"]["selection"]["growth"] == pytest.approx(0.06)
    loss = row["direct_brier_loss_minus_positive_sft_loss"]["accuracy"]["delta"]
    assert loss == pytest.approx((db["accuracy"] - m["clean"]["direct_brier"]["accuracy"]) - (ps["accuracy"] - m["clean"]["positive_sft"]["accuracy"]))
    # direct_brier's final adapter, against its selected one
    assert db["last"]["minus_selected"]["accuracy"]["delta"] == pytest.approx(db["last"]["accuracy"] - db["accuracy"])
    assert db["last"]["channel"]["channel_scale"] is not None and "last" not in ps
    assert m["clean"]["direct_brier"]["last"]["accuracy"] > 0


def test_inverted_files_must_match_the_rate_the_results_and_the_clean_run(runs):
    run = runs / compare_noise.run_name("06b", "direct_brier", 0.1)
    inverted = run.parent / f"{run.name}_inverted" / "metrics.json"
    good = inverted.read_text()
    m = json.loads(good)
    m["flip"] = 0.2
    inverted.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="inverted at flip"):
        compare_noise.compare(runs, "06b", 50)
    m = json.loads(good)
    m["source"]["results_sha256"] = "0" * 64
    inverted.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="other results"):
        compare_noise.compare(runs, "06b", 50)
    m = json.loads(good)
    m["clean_reference"]["run"] = "runs/v3_06b_positive_sft_n5000_s0"
    inverted.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="needs --clean"):
        compare_noise.compare(runs, "06b", 50)
    m = json.loads(good)
    del m["stored_channel"]  # written before the channel scale existed
    inverted.write_text(json.dumps(m))
    row = compare_noise.compare(runs, "06b", 50)["by_rate"]["0.1"]["direct_brier"]
    assert row["channel"]["channel_scale"] is None and row["channel"]["inverted_top1_ece"] == m["inverted_top1_only"]["ece"]
