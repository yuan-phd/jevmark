"""Did a training run finish in this session? The notebooks' fail-fast check (decision 45).

A Kaggle cell that runs `!python scripts/train_sft.py ...` does not stop when the
script fails, so the notebook checks the result itself before any evaluation:
the exit code is zero, `train_summary.json` exists and was written after the cell
started (never a file left in the clone), it names this run, and its step count
equals the planned steps (total_steps, or limit_steps for a smoke or fast run).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


class TrainingFailed(RuntimeError):
    pass


def check_training(run_dir: Path, started: datetime.datetime, exit_code: int | None) -> tuple[bool, str, dict[str, Any] | None]:
    """(passed, reason, summary) for the run in run_dir, launched at `started` (timezone-aware)."""
    if exit_code != 0:
        return False, f"train_sft.py exited with code {exit_code}", None
    path = run_dir / "train_summary.json"
    if not path.is_file():
        return False, f"{path} was not written", None
    summary = json.loads(path.read_text())
    finished = datetime.datetime.fromisoformat(summary["finished"])
    if finished < started.replace(microsecond=0):
        return False, f"{path} predates this session (finished {summary['finished']}, cell started {started.isoformat(timespec='seconds')})", summary
    if summary.get("run_name") != run_dir.name:
        return False, f"{path} belongs to run {summary.get('run_name')!r}, not {run_dir.name!r}", summary
    limit = summary.get("limit_steps")
    planned = min(summary["total_steps"], limit) if limit else summary["total_steps"]
    if summary["steps"] != planned:
        return False, f"{summary['steps']} steps of {planned} planned", summary
    valid = summary["final_valid"]
    return True, f"{run_dir.name}: {summary['steps']} of {planned} steps, best step {summary['best_step']}, full valid acc {valid['accuracy']:.4f} ece {valid['ece']:.4f}", summary


def require_training(run_dir: Path, started: datetime.datetime, exit_code: int | None) -> dict[str, Any]:
    """Print TRAINING PASS or TRAINING FAIL with the reason; raise TrainingFailed on FAIL, so no later cell runs."""
    passed, reason, summary = check_training(run_dir, started, exit_code)
    print(f"TRAINING {'PASS' if passed else 'FAIL'}: {reason}")
    if not passed:
        raise TrainingFailed(reason)
    return summary
