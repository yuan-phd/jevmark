"""A fresh run directory around one stored adapter, so evaluate.py can evaluate it (task 3.6).

    uv run python scripts/prepare_adapter_run.py --adapter <dir with adapter_model.safetensors> \
        --source-run runs/v3_06b_direct_brier_n5000_s0_noisy --adapter-dir adapter_last \
        --name v3_06b_direct_brier_n5000_s0_noisy_last [--sha256 <expected>]

Writes runs/<name>/ with adapter/ (a copy of --adapter), the source run's config.yaml with
run_name set to <name> and an adapter_eval block (source run, adapter directory, adapter
sha256), the source run's calibration.json when it has one, and model_id.txt. The source
run directory supplies only its config and calibration and is never modified. Refuses an
existing run directory, an adapter without adapter_model.safetensors, and, with --sha256,
an adapter whose sha256 differs.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(adapter: Path, source_run: Path, adapter_dir: str, name: str, runs_dir: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"{adapter}: no adapter_model.safetensors")
    digest = file_sha256(weights)
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(f"{weights}: sha256 {digest}, expected {expected_sha256}")
    config_path = source_run / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"{source_run}: no config.yaml")
    out = runs_dir / name
    if out.exists():
        raise SystemExit(f"{out} exists; a run directory is never overwritten")
    config = yaml.safe_load(config_path.read_text())
    config["run_name"] = name
    config["adapter_eval"] = {"source_run": source_run.name, "adapter_dir": adapter_dir, "adapter_sha256": digest}
    out.mkdir(parents=True)
    shutil.copytree(adapter, out / "adapter")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if (source_run / "calibration.json").is_file():
        shutil.copy2(source_run / "calibration.json", out / "calibration.json")
    (out / "model_id.txt").write_text(f"jevmark-{name}\n")
    return config["adapter_eval"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, help="the adapter directory to evaluate")
    parser.add_argument("--source-run", required=True, help="the run directory the adapter came from (its config.yaml is used)")
    parser.add_argument("--adapter-dir", default="adapter_last", help="which of the source run's adapter directories this is, recorded only")
    parser.add_argument("--name", required=True, help="the new run name")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--sha256", default=None, help="refuse an adapter whose adapter_model.safetensors has another sha256")
    args = parser.parse_args(argv)
    record = prepare(Path(args.adapter), Path(args.source_run), args.adapter_dir, args.name, Path(args.runs_dir), args.sha256)
    print(f"prepared {Path(args.runs_dir) / args.name}: {record['adapter_dir']} of {record['source_run']}, adapter sha256 {record['adapter_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
