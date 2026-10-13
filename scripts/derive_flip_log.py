"""Derive a sibling of a v3 deployment-feedback log at another flip rate, on CPU.

    python scripts/derive_flip_log.py --log-dir runs/v3_log_s0 --flip-rate 0.1
    python scripts/derive_flip_log.py --log-dir runs/v3_log_s0 --flip-rate 0.3

The flipped outcome is the only part of a log that depends on the flip rate, and the
flips come from their own seeded stream, one draw per interaction (jevmark/feedback.py).
Redrawing them at another rate from the same stream therefore gives the log that
collect_log.py would have written at that rate, without running the logging policy
again; the source log is never modified. Before writing, the script checks that the
source log matches the sha256 its metrics.json records, and that redrawing at the
source's own rate reproduces it byte for byte. The flips are nested across rates, so
runs at different rates are paired.

Writes runs/<source>_flip<rate>/log.jsonl (gitignored, like every log) and metrics.json:
the source's metrics with the flip rate, the flipped-outcome rate, the flip share and
the log's sha256 replaced, plus the source run and its log sha256 under derived_from.
The seed, the data file's sha256, the logging policy, its device and precision and
the accuracy are the source's, so
train_rlcd.py reads the derived log as it reads the source. An existing log is never
overwritten.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jevmark.feedback import FLIP_RATE, file_sha256, log_bytes, read_log, reflip  # noqa: E402
from jevmark.provenance import git_state  # noqa: E402


def derived_name(source: str, rate: float) -> str:
    return f"{source}_flip{rate:g}"


def derive(log_dir: Path, rate: float, runs_dir: Path) -> tuple[Path, dict]:
    started = time.perf_counter()
    git = git_state()
    if not 0 < rate < 0.5:
        raise SystemExit(f"--flip-rate must be in (0, 0.5), got {rate}")
    source_metrics = json.loads((log_dir / "metrics.json").read_text())
    if "derived_from" in source_metrics:
        raise SystemExit(f"{log_dir} is itself derived; derive from the collected log {source_metrics['derived_from']['run']}")
    source_rate = float(source_metrics["sampling"].get("flip_rate", FLIP_RATE))
    if math.isclose(rate, source_rate):
        raise SystemExit(f"{log_dir} was drawn at flip rate {source_rate:g}; a sibling at the same rate would be the same log")
    source_sha = file_sha256(log_dir / "log.jsonl")
    if source_sha != source_metrics["log_sha256"]:
        raise SystemExit(f"{log_dir / 'log.jsonl'} has sha256 {source_sha}, but its metrics.json records {source_metrics['log_sha256']}")
    seed = int(source_metrics["sampling"]["seed"])
    log = read_log(log_dir / "log.jsonl")
    if hashlib.sha256(log_bytes(reflip(log, seed, source_rate))).hexdigest() != source_sha:
        raise SystemExit(f"redrawing the flips of {log_dir} at its own rate {source_rate:g} does not reproduce it; its flips did not come from the seed {seed} stream")
    out_dir = runs_dir / derived_name(log_dir.name, rate)
    if (out_dir / "log.jsonl").exists():
        raise SystemExit(f"{out_dir / 'log.jsonl'} exists; a log is written once and never overwritten")

    derived = reflip(log, seed, rate)
    raw = log_bytes(derived)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log.jsonl").write_bytes(raw)
    n = len(derived)
    metrics = {key: value for key, value in source_metrics.items() if key != "decision"}
    metrics.update({
        "run_name": out_dir.name,
        "derived_from": {"run": log_dir.name, "log_sha256": source_sha, "flip_rate": source_rate,
                         "how": "flipped outcomes redrawn at flip_rate from the same flip stream; every other field unchanged"},
        "git": git,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sampling": {**source_metrics["sampling"], "flip_rate": rate},
        "n": n,
        "flipped_outcome_rate": sum(i.flipped_outcome for i in derived) / n,
        "flip_share": sum(i.outcome != i.flipped_outcome for i in derived) / n,
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "wall_clock_seconds": time.perf_counter() - started,
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return out_dir, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", required=True, help="the collected log's run directory, e.g. runs/v3_log_s0")
    parser.add_argument("--flip-rate", type=float, required=True)
    parser.add_argument("--runs-dir", default=None, help="default: the source's parent directory")
    args = parser.parse_args(argv)
    log_dir = Path(args.log_dir)
    out_dir, m = derive(log_dir, args.flip_rate, Path(args.runs_dir) if args.runs_dir else log_dir.parent)
    print(f"wrote {out_dir}/log.jsonl (sha256 {m['log_sha256']}): {m['n']} interactions, flip rate {args.flip_rate:g}, "
          f"flip share {m['flip_share']:.4f}, flipped outcome rate {m['flipped_outcome_rate']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
