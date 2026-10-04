"""scripts/make_figures.py on the committed metrics files (figures task)."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_figures", REPO / "scripts" / "make_figures.py")
make_figures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = make_figures
spec.loader.exec_module(make_figures)


def test_every_figure_is_drawn_from_the_committed_metrics(tmp_path):
    assert make_figures.main(["--out", str(tmp_path)]) == 0
    for name in make_figures.EXPECTED:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 10_000, name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(make_figures.EXPECTED)


def test_the_script_reads_no_per_question_files():
    source = (REPO / "scripts" / "make_figures.py").read_text()
    code = source.split('"""', 2)[2]  # everything after the module docstring
    assert "results.jsonl" not in code and "replies.jsonl" not in code and "read_results" not in code and "log.jsonl" not in code
