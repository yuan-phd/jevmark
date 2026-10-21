"""scripts/publish_runs.py on a synthetic runs directory, CPU, no network (review item 9)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_runs", REPO / "scripts" / "publish_runs.py")
publish = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publish
spec.loader.exec_module(publish)


def runs_dir(tmp_path):
    runs = tmp_path / "runs"
    for log in publish.LOGS:
        (runs / log).mkdir(parents=True)
        (runs / log / "log.jsonl").write_text(f'{{"log": "{log}"}}\n')
        (runs / log / "metrics.json").write_text(json.dumps({"log_sha256": hashlib.sha256((runs / log / "log.jsonl").read_bytes()).hexdigest()}))
    for run in ("sft_06b", "v3_06b_zeroshot", "fast_06b", "sft_06b_smoke"):
        (runs / run).mkdir()
        (runs / run / "results.jsonl.gz").write_bytes(run.encode())
        (runs / run / "metrics.json").write_text(json.dumps({"git": {"commit": "abc123", "dirty": False}}))
    digest = hashlib.sha256(b"v3_06b_zeroshot").hexdigest()
    (runs / "v3_stage_06b").mkdir()
    (runs / "v3_stage_06b" / "metrics.json").write_text(json.dumps({"sources": {"zero_shot": {"dir": "runs/v3_06b_zeroshot", "results_sha256": digest}}}))
    return runs


def test_stage_writes_the_card_skips_smoke_runs_and_uploads_nothing(tmp_path, monkeypatch):
    runs = runs_dir(tmp_path)
    monkeypatch.setattr(publish, "upload", lambda *a, **k: pytest.fail("uploaded without --upload"))
    assert publish.main(["--runs-dir", str(runs), "--stage", str(tmp_path / "stage")]) == 0
    card = (tmp_path / "stage" / "README.md").read_text()
    assert "`runs/sft_06b/results.jsonl.gz`" in card and "`runs/v3_log_emotion_s0/log.jsonl`" in card
    assert "fast_06b" not in card and "sft_06b_smoke" not in card
    assert hashlib.sha256(b"sft_06b").hexdigest() in card and "v3_06b_zeroshot (1)" in card


def test_a_results_file_that_differs_from_a_recorded_hash_stops_it(tmp_path):
    runs = runs_dir(tmp_path)
    (runs / "v3_06b_zeroshot" / "results.jsonl.gz").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from the hash in"):
        publish.main(["--runs-dir", str(runs), "--stage", str(tmp_path / "stage")])
