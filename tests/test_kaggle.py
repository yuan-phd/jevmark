"""Kaggle wrapper: requirements file in sync with uv.lock, and the notebook's structure and token handling."""

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_kaggle_requirements", REPO / "scripts" / "export_kaggle_requirements.py")
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)

NOTEBOOK = json.loads((REPO / "notebooks" / "kaggle_eval.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]


def test_requirements_file_matches_uv_lock():
    assert (REPO / "requirements-kaggle.txt").read_text() == export.render()


def test_requirements_pin_everything_but_torch():
    lines = [l for l in (REPO / "requirements-kaggle.txt").read_text().splitlines() if l and not l.startswith("#")]
    names = {re.match(r"^([A-Za-z0-9_.-]+)==", l).group(1).lower() for l in lines}
    assert {"transformers", "peft", "datasets", "accelerate", "tokenizers", "huggingface-hub", "safetensors"} <= names
    assert "torch" not in names
    assert all("==" in l for l in lines)


def test_notebook_parameters_come_first():
    assert re.search(r'^REPO = "', CODE[0], re.M) and re.search(r'^COMMIT = "', CODE[0], re.M)


def test_notebook_token_is_never_printed_or_put_on_a_command_line():
    clone = CODE[1]
    assert 'get_secret("GITHUB_TOKEN")' in clone
    assert "GIT_CONFIG_VALUE_0" in clone and "secrets=(token, header)" in clone
    for cell in CODE:
        for line in cell.splitlines():
            if "print(" in line:
                assert "token" not in line and "header" not in line, line
            if line.lstrip().startswith("!"):
                assert "token" not in line.lower(), line
    assert "https://github.com/{REPO}.git" in clone and "@github.com" not in clone


def test_notebook_runs_smoke_before_both_backbones_and_copies_runs():
    joined = "\n".join(CODE)
    smoke = joined.index("--limit {LIMIT}")
    assert smoke < joined.index("--config configs/base_06b.yaml --device cuda\n") < joined.index("configs/base_17b.yaml")
    assert "requirements-kaggle.txt" in joined and "--no-deps" in joined
    assert "make data PY=python" in joined
    assert '"/kaggle/working/runs"' in joined
