"""Where the negative outcomes of a v3 deployment-feedback log fall, on CPU.

    uv run python scripts/log_stats.py runs/v3_log_s0 runs/v3_log_s1 runs/v3_log_17b_s0 runs/v3_log_emotion_s0

For each log directory, reads log.jsonl (checked against the sha256 its metrics.json
records) and adds one block, "negatives", to its metrics.json; every other field is left
as it was. On the clean outcome column:
- the negatives on the logging policy's top option (argmax of p) and on the other
  options, as counts and shares of all negatives;
- of the negatives on other options, the share attributable to the uniform term of the
  behaviour policy q = (1 - epsilon) p + epsilon / K: the sum over those interactions of
  (epsilon / K) / q(chosen action), divided by their number. The rest comes from sampling p;
- the negatives whose chosen action was the option labelled other (never gold in these
  logs), as a count and share of all negatives, split by whether other was the logging
  policy's top option, so it shows whether all of them are in the on_other count (they
  are only when other was never the top option). A log without an other option has 0.
Before counting, every logged propensity is checked against the behaviour policy
recomputed from the logged p at epsilon (jevmark/feedback.py), so the epsilon used is
the one the log was drawn with.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jevmark.feedback import EPSILON, Interaction, behaviour, file_sha256, read_log  # noqa: E402
from jevmark.provenance import git_state  # noqa: E402

BLOCK = "negatives"
OTHER = "other"
PROPENSITY_TOLERANCE = 1e-6


def negatives(log: Sequence[Interaction], epsilon: float = EPSILON) -> dict[str, Any]:
    worst = 0.0
    on_top = on_other = 0
    other_on_top = other_on_other = 0
    uniform = 0.0
    for i in log:
        if i.propensity is None:
            raise SystemExit(f"interaction {i.index} has no propensity")
        k = len(i.probs)
        worst = max(worst, abs(behaviour(i.probs, epsilon)[i.action] - i.propensity))
        if i.outcome:
            continue
        top = max(range(k), key=i.probs.__getitem__)
        chose_other = i.labels[i.action] == OTHER
        if i.action == top:
            on_top += 1
            other_on_top += chose_other
        else:
            other_on_other += chose_other
            on_other += 1
            uniform += (epsilon / k) / i.propensity
    if worst > PROPENSITY_TOLERANCE:
        raise SystemExit(f"logged propensities differ from the behaviour policy at epsilon {epsilon} by up to {worst:.3g}")
    total = on_top + on_other
    return {
        "outcome_column": "outcome",
        "epsilon": epsilon,
        "n": len(log),
        "negatives": total,
        "on_top": {"count": on_top, "share": on_top / total if total else None},
        "on_other": {"count": on_other, "share": on_other / total if total else None},
        "on_other_uniform_share": uniform / on_other if on_other else None,
        "on_other_uniform_expected_count": uniform,
        "chosen_other": {
            "count": other_on_top + other_on_other,
            "share": (other_on_top + other_on_other) / total if total else None,
            "on_top_count": other_on_top,
            "on_other_count": other_on_other,
            "all_in_on_other": other_on_top == 0,
        },
        "propensity_max_abs_diff": worst,
    }


def run(log_dir: Path) -> dict[str, Any]:
    metrics_path = log_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    log_path = log_dir / metrics.get("log_file", "log.jsonl")
    sha = file_sha256(log_path)
    if sha != metrics["log_sha256"]:
        raise SystemExit(f"{log_path}: sha256 {sha} differs from metrics.json's {metrics['log_sha256']}")
    block = negatives(read_log(log_path))
    block.update({"log_sha256": sha, "script": "scripts/log_stats.py", "script_sha256": file_sha256(Path(__file__)), "git": git_state(),
                  "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    metrics[BLOCK] = block
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return block


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dirs", nargs="+")
    args = parser.parse_args(argv)
    for d in args.log_dirs:
        print(d, json.dumps(run(Path(d)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
