"""jevmark.runcheck: the notebooks' fail-fast training check (decision 45)."""

import datetime
import json

import pytest

from jevmark.runcheck import TrainingFailed, check_training, require_training

STARTED = datetime.datetime(2026, 9, 25, 18, 0, 0, 500000, tzinfo=datetime.timezone.utc)


def write_summary(run_dir, **changes):
    summary = {
        "run_name": run_dir.name,
        "finished": "2026-09-25T19:00:00+00:00",
        "steps": 1378,
        "total_steps": 1378,
        "limit_steps": None,
        "best_step": 1200,
        "final_valid": {"accuracy": 0.95, "ece": 0.01},
        **changes,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_summary.json").write_text(json.dumps(summary))


def test_a_finished_run_passes(tmp_path):
    run_dir = tmp_path / "sft_06b"
    write_summary(run_dir)
    passed, reason, summary = check_training(run_dir, STARTED, 0)
    assert passed and "1378 of 1378 steps" in reason and summary["best_step"] == 1200


def test_a_smoke_run_passes_at_its_step_limit(tmp_path):
    run_dir = tmp_path / "sft_06b_smoke"
    write_summary(run_dir, steps=20, limit_steps=20)
    assert check_training(run_dir, STARTED, 0)[0]


def test_a_summary_finished_in_the_same_second_as_the_start_passes(tmp_path):
    run_dir = tmp_path / "sft_06b"
    write_summary(run_dir, finished="2026-09-25T18:00:00+00:00")
    assert check_training(run_dir, STARTED, 0)[0]


@pytest.mark.parametrize(
    "exit_code, changes, reason",
    [
        (1, {}, "exited with code 1"),
        (None, {}, "exited with code None"),
        (0, {"finished": "2026-09-24T23:17:19+00:00"}, "predates this session"),  # the committed v1.2 summary
        (0, {"steps": 392}, "392 steps of 1378 planned"),
        (0, {"steps": 12, "limit_steps": 20}, "12 steps of 20 planned"),
        (0, {"run_name": "sft_17b"}, "belongs to run 'sft_17b'"),
    ],
)
def test_failures(tmp_path, exit_code, changes, reason):
    run_dir = tmp_path / "sft_06b"
    write_summary(run_dir, **changes)
    passed, why, _ = check_training(run_dir, STARTED, exit_code)
    assert not passed and reason in why


def test_a_missing_summary_fails(tmp_path):
    passed, why, _ = check_training(tmp_path / "sft_06b", STARTED, 0)
    assert not passed and "was not written" in why


def test_require_training_prints_one_line_and_raises_on_fail(tmp_path, capsys):
    run_dir = tmp_path / "sft_06b"
    write_summary(run_dir)
    assert require_training(run_dir, STARTED, 0)["steps"] == 1378
    assert capsys.readouterr().out.startswith("TRAINING PASS: sft_06b: 1378 of 1378 steps")
    write_summary(run_dir, steps=392)
    with pytest.raises(TrainingFailed, match="392 steps of 1378 planned"):
        require_training(run_dir, STARTED, 0)
    assert capsys.readouterr().out.strip() == "TRAINING FAIL: 392 steps of 1378 planned"
