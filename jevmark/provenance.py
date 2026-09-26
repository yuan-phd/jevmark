"""Provenance recorded in every metrics.json: the commit and whether tracked files differ from it (decision 35)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def git_state(repo: Path = REPO) -> dict[str, Any]:
    """Commit and whether tracked code or docs differ from it, ignoring runs/.

    Called before any output is written: a run that overwrites the tracked files
    of an earlier run (for example a B0 re-run) must not report itself dirty.
    """

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)runs")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}
