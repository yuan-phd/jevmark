"""Kaggle wrapper: requirements file in sync with uv.lock, and the notebook's structure and token handling."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_kaggle_requirements", REPO / "scripts" / "export_kaggle_requirements.py")
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)

NOTEBOOK = json.loads((REPO / "notebooks" / "kaggle_eval.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]


def test_requirements_file_matches_uv_lock():
    assert (REPO / "requirements-kaggle.txt").read_text() == export.render()


def test_requirements_pin_exactly_the_hugging_face_packages():
    lines = [l for l in (REPO / "requirements-kaggle.txt").read_text().splitlines() if l and not l.startswith("#")]
    names = [re.match(r"^([A-Za-z0-9_.-]+)==", l).group(1).lower() for l in lines]
    assert sorted(names) == sorted(["transformers", "tokenizers", "peft", "datasets", "accelerate", "huggingface-hub", "safetensors"])
    assert "torch" not in names and "numpy" not in names  # Kaggle keeps its own (decision 23)


def test_notebook_parameters_come_first():
    assert re.search(r'^REPO = "', CODE[0], re.M) and re.search(r'^COMMIT = "', CODE[0], re.M)


def code_cells(name):
    notebook = json.loads((REPO / "notebooks" / name).read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("name", ["kaggle_eval.ipynb", "kaggle_train.ipynb"])
def test_notebook_token_is_never_printed_or_put_on_a_command_line(name):
    cells = code_cells(name)
    clone = cells[1]
    assert 'get_secret("GITHUB_TOKEN")' in clone
    assert "GIT_CONFIG_VALUE_0" in clone and "secrets=(token, header)" in clone
    for cell in cells:
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


def test_train_notebook_runs_one_size_smoke_first_then_train_evaluate_and_copy():
    cells = code_cells("kaggle_train.ipynb")
    params = cells[0]
    for name in ("REPO", "COMMIT", "SIZE", "SMOKE_STEPS"):
        assert re.search(rf"^{name} = ", params, re.M), name
    joined = "\n".join(cells)
    smoke = joined.index("run_name=sft_{SIZE}_smoke --limit-steps {SMOKE_STEPS}")
    full = joined.index("train_sft.py --config configs/sft_{SIZE}.yaml --max-hours")
    evaluate_sft = joined.index("evaluate.py --ckpt runs/sft_{SIZE}")
    evaluate_base = joined.index("evaluate.py --ckpt base --config configs/base_{SIZE}.yaml")
    assert smoke < full < evaluate_sft < evaluate_base
    assert joined.rindex('shutil.copytree(WORK / "runs", "/kaggle/working/runs"') > evaluate_base
    assert "make data PY=python" in joined and "requirements-kaggle.txt" in joined


@pytest.mark.parametrize("name", ["kaggle_eval.ipynb", "kaggle_train.ipynb"])
def test_notebooks_uninstall_torchao_before_installing(name):
    install = next(c for c in code_cells(name) if "requirements-kaggle.txt" in c)
    assert install.index("pip uninstall -y -q torchao") < install.index("pip install -q -r requirements-kaggle.txt")
