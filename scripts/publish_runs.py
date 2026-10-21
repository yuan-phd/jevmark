"""The run-files dataset: the deployment logs and every run's per-question results on the Hugging Face Hub (review item 9).

    uv run python scripts/publish_runs.py                 # stage: check the files, write the README, upload nothing
    uv run python scripts/publish_runs.py --upload        # the human runs this, logged in with write access

The files are gitignored (decision 37): the four deployment-feedback logs,
runs/<log>/log.jsonl for v3_log_s0, v3_log_s1, v3_log_17b_s0 and v3_log_emotion_s0,
and runs/<run>/results.jsonl.gz of every run that has one, except the fast-cycle and
smoke runs (fast_*, *_smoke, *_limit*), which no report reads. The sibling logs at flip
rates 0.1 and 0.3 (runs/v3_log_s0_flip*) are not published: scripts/derive_flip_log.py
rebuilds them from v3_log_s0 exactly, and they differ from it only in flipped_outcome.

Before anything is written, every log is checked against the log_sha256 its
metrics.json records, and every results file against each results_sha256 that a
committed metrics.json records for it (the comparison files, the inversions, the
review files). A mismatch stops the script: the local file would not be the one the
reported numbers were computed from.

Writes the dataset card to --stage (default data/hf_runs/README.md, gitignored): one
row per file with its path in the dataset, the run it belongs to, its size, its
sha256 and the commit the run's metrics.json records, plus how to read the files. With
--upload it creates the dataset repository yuanphd/jevmark-runs if needed (private
unless --public) and uploads the README and every file in one commit, each file at
runs/<run>/<file>, the path it has in this repository. Nothing is copied locally.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "yuanphd/jevmark-runs"
LOGS = ("v3_log_s0", "v3_log_s1", "v3_log_17b_s0", "v3_log_emotion_s0")
EXCLUDED = ("fast_*", "*_smoke", "*_limit*")  # gitignored cycle and smoke runs, no report reads them


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_files(runs_dir: Path) -> list[tuple[str, Path]]:
    """(run name, file) for the four logs and every results.jsonl.gz, logs first, then by run name."""
    files = [(log, runs_dir / log / "log.jsonl") for log in LOGS]
    for path in sorted(runs_dir.glob("*/results.jsonl.gz")):
        run = path.parent.name
        if not any(fnmatch.fnmatch(run, pattern) for pattern in EXCLUDED):
            files.append((run, path))
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    return files


def recorded_hashes(node: Any, where: str) -> Iterator[tuple[str, str, str]]:
    """(run name, sha256, metrics file) for every results_sha256 in a metrics.json whose run can be told."""
    if isinstance(node, dict):
        value = node.get("results_sha256")
        if isinstance(value, str):
            run = node.get("dir") or node.get("run")
            if isinstance(run, str):
                yield Path(run).name, value, where
        elif isinstance(value, dict):
            runs = node.get("runs") if isinstance(node.get("runs"), dict) else {}
            for name, digest in value.items():
                target = runs.get(name)
                if isinstance(target, dict):
                    target = target.get("run") or target.get("dir")
                if isinstance(target, str):
                    yield Path(target).name, digest, where
                elif isinstance(digest, str):
                    yield name, digest, where
        for child in node.values():
            yield from recorded_hashes(child, where)
    elif isinstance(node, list):
        for child in node:
            yield from recorded_hashes(child, where)


def check(runs_dir: Path, files: list[tuple[str, Path]]) -> tuple[dict[Path, str], dict[str, list[str]]]:
    """sha256 of every file, after checking the logs and every recorded results hash; also where each results file's hash is recorded."""
    digests = {path: sha256(path) for _, path in files}
    for log in LOGS:
        recorded = json.loads((runs_dir / log / "metrics.json").read_text())["log_sha256"]
        if digests[runs_dir / log / "log.jsonl"] != recorded:
            raise SystemExit(f"runs/{log}/log.jsonl: sha256 differs from its metrics.json log_sha256")
    by_run = {run: digests[path] for run, path in files if path.name == "results.jsonl.gz"}
    recorded_in: dict[str, list[str]] = {}
    mismatches = []
    for metrics in sorted(runs_dir.glob("*/metrics.json")):
        where = str(metrics.relative_to(runs_dir.parent)) if metrics.resolve().is_relative_to(runs_dir.parent.resolve()) else str(metrics)
        for run, digest, source in recorded_hashes(json.loads(metrics.read_text()), where):
            if run in by_run:
                if by_run[run] != digest:
                    mismatches.append(f"runs/{run}/results.jsonl.gz differs from the hash in {source}")
                else:
                    recorded_in.setdefault(run, []).append(source)
    if mismatches:
        raise SystemExit("\n".join(sorted(set(mismatches))))
    return digests, {run: sorted(set(v)) for run, v in recorded_in.items()}


def commit_of(runs_dir: Path, run: str) -> str:
    path = runs_dir / run / "metrics.json"
    git = json.loads(path.read_text()).get("git") if path.is_file() else None
    return (git or {}).get("commit") or ""


def card(runs_dir: Path, files: list[tuple[str, Path]], digests: dict[Path, str], recorded_in: dict[str, list[str]], git: dict[str, Any]) -> str:
    rows = []
    for run, path in files:
        rows.append(f"| `runs/{run}/{path.name}` | {run} | {path.stat().st_size} | `{digests[path]}` | `{commit_of(runs_dir, run)[:12]}` |")
    total = sum(p.stat().st_size for _, p in files)
    return "\n".join([
        "---",
        "license: apache-2.0",
        "pretty_name: jevmark run files",
        "---",
        "",
        "# jevmark run files",
        "",
        "Per-question results and deployment-feedback logs behind every number in the jevmark reports and paper. The code, configs and every "
        "metrics.json are in the jevmark repository; these files are gitignored there because of their size (decision 37). Each file sits at the "
        "path it has in the repository, so copying this dataset's `runs/` over a clone's `runs/` lets every comparison script rerun on CPU.",
        "",
        "- `runs/<run>/results.jsonl.gz`: one JSON line per question of every evaluated split, written by `scripts/evaluate.py` (record id, "
        "question id, split, type, kind, labels, the full answer distribution, gold, confidence, prediction). "
        "`jevmark.metrics.read_results` reads it, and `scripts/recompute_metrics.py` rebuilds the run's metrics.json from it.",
        "- `runs/<log>/log.jsonl`: the deployment-feedback logs, one interaction per line (record, options, the logging policy's distribution, "
        "the action drawn from (1 - epsilon) p + epsilon / K at epsilon 0.1, its propensity, the outcome and the outcome flipped at rate 0.2), "
        "read by `jevmark.feedback.read_log`. v3_log_s0 and v3_log_s1 are sft_06b on Banking77 with action seeds 0 and 1, v3_log_17b_s0 is "
        "sft_17b on Banking77, v3_log_emotion_s0 is sft_06b on emotion. The logs at flip rates 0.1 and 0.3 are rebuilt exactly from "
        "v3_log_s0 by `scripts/derive_flip_log.py`.",
        "",
        "The sha256 of every log equals the log_sha256 its metrics.json records, and that of every results file equals each results_sha256 a "
        "committed metrics.json records for it, checked by `scripts/publish_runs.py` before upload. Commit: the git commit in the run's "
        "metrics.json (the code that wrote the file).",
        "",
        f"{len(files)} files, {total} bytes. Built at repository commit `{git.get('commit')}`"
        f"{' (dirty)' if git.get('dirty') else ''} on {datetime.date.today().isoformat()}.",
        "",
        "| file | run | bytes | sha256 | commit |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "Results files whose hash a committed metrics.json also records: "
        + ", ".join(f"{run} ({len(v)})" for run, v in sorted(recorded_in.items())) + ".",
        "",
    ])


def upload(files: list[tuple[str, Path]], readme: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme))]
    operations += [CommitOperationAdd(path_in_repo=f"runs/{run}/{path.name}", path_or_fileobj=str(path)) for run, path in files]
    info = api.create_commit(repo_id, operations, repo_type="dataset", commit_message=f"run files: {len(files)} files and the dataset card")
    print(f"uploaded {len(files)} files and README.md to {repo_id}: {info.commit_url}")


def main(argv: Sequence[str] | None = None) -> int:
    from jevmark.provenance import git_state

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--stage", default=str(REPO / "data" / "hf_runs"), help="where README.md is written")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--upload", action="store_true", help="create the dataset if needed and upload; without it nothing leaves the machine")
    parser.add_argument("--public", action="store_true", help="with --upload, create the dataset public (default private)")
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    files = run_files(runs_dir)
    digests, recorded_in = check(runs_dir, files)
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    readme = stage / "README.md"
    readme.write_text(card(runs_dir, files, digests, recorded_in, git_state()))
    total = sum(p.stat().st_size for _, p in files)
    print(f"{len(files)} files ({len(LOGS)} logs, {len(files) - len(LOGS)} results files), {total / 2**20:.1f} MiB; "
          f"{len(recorded_in)} results files also match a hash recorded in a committed metrics.json; wrote {readme}")
    if args.upload:
        upload(files, readme, args.repo_id, private=not args.public)
    else:
        print(f"nothing uploaded; to publish: uv run python scripts/publish_runs.py --upload (dataset {args.repo_id}, private unless --public)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
