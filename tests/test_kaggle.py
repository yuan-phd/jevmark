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


@pytest.mark.parametrize("name", ["kaggle_eval.ipynb", "kaggle_train.ipynb", "kaggle_baseline_b1.ipynb", "kaggle_rlcd.ipynb", "kaggle_v3.ipynb"])
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


def test_notebook_runs_smoke_before_every_size_in_sizes_and_copies_runs():
    joined = "\n".join(CODE)
    assert re.search(r'^SIZES = \["06b", "17b"\]', CODE[0], re.M)
    smoke = joined.index("--config configs/base_{SIZES[0]}.yaml --limit {LIMIT}")
    assert smoke < joined.index("for size in (SIZES if ADAPTER_EVAL is None else []):") < joined.index("--ckpt base --config configs/base_{size}.yaml --device cuda")
    assert "configs/base_06b.yaml" not in joined and "configs/base_17b.yaml" not in joined  # sizes come from SIZES only
    assert "requirements-kaggle.txt" in joined and "--no-deps" in joined
    assert "make data-build PY=python" in joined
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
    assert "make data-build PY=python" in joined and "requirements-kaggle.txt" in joined


@pytest.mark.parametrize("name", ["kaggle_eval.ipynb", "kaggle_train.ipynb", "kaggle_baseline_b1.ipynb", "kaggle_rlcd.ipynb", "kaggle_v3.ipynb"])
def test_notebooks_uninstall_torchao_before_installing(name):
    install = next(c for c in code_cells(name) if "requirements-kaggle.txt" in c)
    assert install.index("pip uninstall -y -q torchao") < install.index("pip install -q -r requirements-kaggle.txt")


def test_train_notebook_fast_cycle_trains_300_steps_and_evaluates_300_records_with_every_diagnostic():
    cells = code_cells("kaggle_train.ipynb")
    params = cells[0]
    assert re.search(r"^FAST = False", params, re.M)
    assert re.search(r"^FAST_STEPS = 300\b", params, re.M) and re.search(r"^FAST_RECORDS = 300\b", params, re.M)
    fast = next(c for c in cells if c.lstrip().startswith("# Fast cycle"))
    assert "if FAST:" in fast and 'FAST_RUN = f"fast_{SIZE}"' in fast
    assert "train_sft.py --config configs/sft_{SIZE}.yaml run_name={FAST_RUN} --limit-steps {FAST_STEPS}" in fast
    assert "evaluate.py --ckpt runs/{FAST_RUN} --run-name {FAST_RUN} --limit {FAST_RECORDS} --shuffle-questions {FAST_SHUFFLE_SPLIT}" in fast
    joined = "\n".join(cells)
    assert joined.index("make data-build") < joined.index("# Fast cycle") < joined.index("--limit-steps {SMOKE_STEPS}")
    for cell in cells[cells.index(fast) + 1 :]:
        assert "if not FAST:" in cell  # the full session is skipped in fast mode


def test_fast_runs_are_gitignored():
    import subprocess

    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-q", "runs/fast_06b/metrics.json"], cwd=REPO)
    assert ignored.returncode == 0


@pytest.mark.parametrize("name", ["kaggle_eval.ipynb", "kaggle_train.ipynb", "kaggle_rlcd.ipynb", "kaggle_v3.ipynb"])
def test_notebooks_set_expandable_segments_before_any_training_or_evaluation(name):
    cells = code_cells(name)
    joined = "\n".join(cells)
    first_run = min(joined.index(s) for s in ("train_sft.py", "evaluate.py", "train_rlcd.py", "collect_log.py") if s in joined)
    for var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert joined.index(f'os.environ["{var}"] = "expandable_segments:True"') < first_run


def test_sft_06b_uses_micro_batch_8_with_effective_batch_32():
    from jevmark.config import load_config

    training = load_config(REPO / "configs" / "sft_06b.yaml")["training"]
    assert training["micro_batch"] == 8 and training["effective_batch"] == 32  # accumulation 4


def test_notebooks_delete_committed_training_runs_right_after_the_clone():
    train_cells = code_cells("kaggle_train.ipynb")
    clone = train_cells[1]
    assert 'for stale in (f"sft_{SIZE}", f"sft_{SIZE}_smoke", f"fast_{SIZE}"):' in clone and "shutil.rmtree(WORK / \"runs\" / stale" in clone
    assert clone.index('run(["git", "checkout"') < clone.index("shutil.rmtree")
    eval_clone = code_cells("kaggle_eval.ipynb")[1]
    assert 'WORK.glob("runs/sft_*")' in eval_clone and 'WORK.glob("runs/fast_*")' in eval_clone


def test_train_notebook_fails_fast_and_evaluations_require_a_passed_training():
    cells = code_cells("kaggle_train.ipynb")
    helper = next(c for c in cells if "from jevmark.runcheck import require_training" in c)
    install = next(c for c in cells if "requirements-kaggle.txt" in c)
    assert cells.index(install) < cells.index(helper)
    smoke = next(c for c in cells if c.startswith("# Smoke training"))
    full = next(c for c in cells if c.startswith("# Full training"))
    fast = next(c for c in cells if c.lstrip().startswith("# Fast cycle"))
    for cell, run_dir, flag in ((smoke, 'f"sft_{SIZE}_smoke"', "smoke"), (full, 'f"sft_{SIZE}"', "full"), (fast, "FAST_RUN", "fast")):
        assert "started = datetime.datetime.now(datetime.timezone.utc)" in cell
        assert cell.index("started = ") < cell.index("train_sft.py") < cell.index("require_training(") < cell.index(f'TRAINED["{flag}"] = True')
        assert f'WORK / "runs" / {run_dir}' in cell and "_exit_code" in cell
    assert 'raise RuntimeError("refusing to train: the smoke training cell did not print TRAINING PASS")' in full
    evaluate_sft = next(c for c in cells if c.startswith("# Evaluate the trained adapter"))
    evaluate_base = next(c for c in cells if c.startswith("# Re-evaluate the frozen base"))
    for cell in (evaluate_sft, evaluate_base):
        assert cell.index('evaluation_allowed("full")') < cell.index("evaluate.py") < cell.index('check_exit("evaluate.py", _exit_code)')
    assert "evaluate.py --ckpt runs/sft_{SIZE} --shuffle-questions test_indomain --device cuda" in evaluate_sft
    assert fast.index('evaluation_allowed("fast")') < fast.index("evaluate.py")
    assert "assert (WORK / \"runs\" / f\"sft_{SIZE}\" / \"train_summary.json\").exists()" not in full  # the old check that a stale file passed


def test_train_notebook_skips_the_base_evaluation_when_eval_base_is_false():
    cells = code_cells("kaggle_train.ipynb")
    assert re.search(r"^EVAL_BASE = True", cells[0], re.M)
    base = next(c for c in cells if c.startswith("# Re-evaluate the frozen base"))
    assert "if not FAST and EVAL_BASE:" in base and base.index("if not FAST and EVAL_BASE:") < base.index("evaluate.py")
    copy = next(c for c in cells if c.startswith("# Copy runs/"))
    assert '[f"sft_{SIZE}"] + ([f"base_{SIZE}"] if EVAL_BASE else [])' in copy


def test_b1_notebook_runs_a_smoke_run_then_every_size_on_the_whole_subset_and_copies_runs():
    cells = code_cells("kaggle_baseline_b1.ipynb")
    params = cells[0]
    for name in ("REPO", "COMMIT", "LIMIT", "BATCH_SIZE"):
        assert re.search(rf"^{name} = ", params, re.M), name
    assert re.search(r'^SIZES = \["06b", "17b"\]', params, re.M) and re.search(r"^LATENCY_ONLY = False", params, re.M)
    joined = "\n".join(cells)
    data = joined.index("make data-build PY=python")
    smoke = joined.index("if not LATENCY_ONLY:\n    !python scripts/baseline_llm_json.py --size {SIZES[0]} --device cuda --limit {LIMIT} --latency-requests 10 --batch-size {BATCH_SIZE}")
    loop = joined.index("if not LATENCY_ONLY:\n    for size in SIZES:\n        !python scripts/baseline_llm_json.py --size {size} --device cuda --batch-size {BATCH_SIZE}\n")
    copy = joined.index('shutil.copytree(WORK / "runs", "/kaggle/working/runs"')
    assert data < smoke < loop < copy
    summary = cells[-1]
    assert 'for name in [f"b1_qwen{size}_json" for size in SIZES]:' in summary and 'glob("' not in summary
    assert summary.index("else:") < summary.index('assert metrics["git"]["commit"] == COMMIT, name')
    assert 'WORK.glob("runs/b1_*")' in cells[1] and "requirements-kaggle.txt" in joined and "--no-deps" in joined


def test_b1_notebook_latency_only_probes_every_size_and_copies_only_the_latency_files():
    cells = code_cells("kaggle_baseline_b1.ipynb")
    joined = "\n".join(cells)
    probe = joined.index("if LATENCY_ONLY:\n    for size in SIZES:\n        !python scripts/baseline_llm_json.py --size {size} --device cuda --batch-size {BATCH_SIZE} --latency-only\n")
    assert joined.index("make data-build PY=python") < probe < joined.index("# Copy runs/")
    summary = cells[-1]
    latency_branch = summary[summary.index("if LATENCY_ONLY:") : summary.index("else:")]
    assert 'shutil.copy2(WORK / "runs" / name / "latency.json", target / "latency.json")' in latency_branch
    assert "copytree" not in latency_branch and "metrics.json" not in latency_branch
    assert latency_branch.index('for name in [f"b1_qwen{size}_json" for size in SIZES]:') < latency_branch.index('assert probe["git"]["commit"] == COMMIT, name')
    assert "p95_ms" in latency_branch


def test_b1_sizes_pin_one_instruct_model_per_backbone_size():
    import importlib.util

    spec = importlib.util.spec_from_file_location("b1_sizes", REPO / "scripts" / "baseline_llm_json.py")
    b1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b1)
    assert b1.SIZES["06b"][0] == "Qwen/Qwen3-0.6B" and b1.SIZES["17b"][0] == "Qwen/Qwen3-1.7B"
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _, revision, _ in b1.SIZES.values())
    assert [name for _, _, name in b1.SIZES.values()] == ["b1_qwen06b_json", "b1_qwen17b_json"]
    args = b1.parse_args(["--size", "06b"])
    assert (args.model, args.revision, args.run_name) == b1.SIZES["06b"]
    assert b1.parse_args(["--size", "06b", "--model", "local/tiny"]).model == "local/tiny"


def test_eval_notebook_deletes_the_committed_b0_runs_it_will_write_right_after_the_clone():
    clone = code_cells("kaggle_eval.ipynb")[1]
    deletion = 'for name in (f"base_{size}", f"base_{size}_limit{LIMIT}")'
    assert "for size in SIZES" in clone and deletion in clone and "shutil.rmtree(path" in clone
    assert clone.index('run(["git", "checkout"') < clone.index(deletion)


def test_eval_notebook_summary_checks_only_the_runs_of_this_session():
    """Session B failed here: the summary globbed every runs/*/metrics.json, and the committed base_06b of another commit failed the commit check."""
    summary = code_cells("kaggle_eval.ipynb")[-1]
    assert 'glob("*/metrics.json")' not in summary
    assert 'for name in ([f"base_{size}" for size in SIZES] if ADAPTER_EVAL is None else [ADAPTER_EVAL["name"]]):' in summary
    assert summary.index("for name in") < summary.index('assert metrics["git"]["commit"] == COMMIT')
    assert summary.index('shutil.copytree(WORK / "runs", "/kaggle/working/runs"') < summary.index("for name in")


def test_rlcd_notebook_copies_the_adapter_runs_smoke_then_full_then_evaluates_with_the_v1_protocol():
    cells = code_cells("kaggle_rlcd.ipynb")
    params = cells[0]
    for name in ("REPO", "COMMIT", "SIZE", "ENV", "ARMS", "SEEDS", "ADAPTER_DATASET", "SMOKE_STEPS", "MAX_HOURS"):
        assert re.search(rf"^{name} = ", params, re.M), name
    assert not re.search(r"^ARM = ", params, re.M) and not re.search(r"^SEED = ", params, re.M)
    joined = "\n".join(cells)
    order = [
        joined.index("make data-build PY=python"),
        joined.index('target = WORK / "runs" / f"sft_{SIZE}" / "adapter"'),
        joined.index("train_rlcd.py --config configs/rlcd_{SIZE}.yaml --env {ENV} --init runs/sft_{SIZE} arm={PAIRS[0][0]} seed={PAIRS[0][1]} run_name={SMOKE_RUN} --limit-steps {SMOKE_STEPS}"),
        joined.index("train_rlcd.py --config configs/rlcd_{SIZE}.yaml --env {ENV} --init runs/sft_{SIZE} arm={arm} seed={seed} --max-hours {MAX_HOURS}"),
        joined.index("evaluate.py --ckpt runs/{run_name} --shuffle-questions test_indomain --device cuda"),
    ]
    assert order == sorted(order)
    assert joined.count("require_training(") == 2 and "evaluation_allowed((arm, seed))" in joined
    clone = cells[1]
    assert "PAIRS = [(arm, seed) for seed in SEEDS for arm in ARMS]" in clone  # every arm at one seed before the next seed
    assert 'PREFIX = f"rlcd_{SIZE}_noisy" if ENV == "noisy" else f"rlcd_{SIZE}"' in clone  # the names train_rlcd.py gives
    assert 'RUNS = {(arm, seed): f"{PREFIX}_{arm}_s{seed}" for arm, seed in PAIRS}' in clone and 'SMOKE_RUN = f"{RUNS[PAIRS[0]]}_smoke"' in clone
    assert "for stale in (*RUNS.values(), SMOKE_RUN):" in clone and "shutil.rmtree(WORK / \"runs\" / stale" in clone
    assert clone.index('run(["git", "checkout"') < clone.index("shutil.rmtree")
    assert "all(arm in ARM_CHOICES for arm in ARMS)" in clone and "len(set(ARMS)) == len(ARMS)" in clone and "len(set(SEEDS)) == len(SEEDS)" in clone


def test_rlcd_notebook_runs_each_arm_and_seed_in_order_and_copies_it_before_the_next():
    cells = code_cells("kaggle_rlcd.ipynb")
    loop = next(c for c in cells if "for arm, seed in PAIRS:" in c)
    body = loop[loop.index("for arm, seed in PAIRS:") :]
    assert 'if not TRAINED.get("smoke"):' in loop and loop.index('if not TRAINED.get("smoke"):') < loop.index("for arm, seed in PAIRS:")
    steps = [
        "run_name = RUNS[(arm, seed)]",
        "!python scripts/train_rlcd.py",
        "training_exit = _exit_code",
        "copy_run(run_name)  # keep the state even if the check fails",
        "summary = require_training(WORK / \"runs\" / run_name, run_started, training_exit)",
        "evaluation_allowed((arm, seed))",
        "!python scripts/evaluate.py --ckpt runs/{run_name}",
        'check_exit(f"evaluate.py for {run_name}", _exit_code)',
        "copy_run(run_name)\n",
        "DONE.append(run_summary(arm, seed, run_name, summary, run_started))",
        "print(DONE[-1], flush=True)",
    ]
    positions = [body.index(step) for step in steps]
    assert positions == sorted(positions)
    # Every step is inside the loop body (indented), so each run is trained, evaluated, copied and summarised before the next starts.
    for step in steps:
        line = next(l for l in body.splitlines() if step.strip() in l)
        assert line.startswith("    "), line
    # The copy is of that one run, to the notebook output, never of the whole runs/ directory.
    assert 'shutil.copytree(WORK / "runs" / run_name, OUT / run_name, dirs_exist_ok=True)' in loop and 'OUT = Path("/kaggle/working/runs")' in loop
    assert 'shutil.copytree(WORK / "runs", ' not in "\n".join(cells)
    # One summary line per run: arm, seed, best step, test_indomain accuracy and ECE, test_emotion ECE, wall clock; the commit and SFT adapter are checked.
    summary = loop[loop.index("def run_summary") : loop.index("for arm, seed in PAIRS:")]
    for field in ("{arm} seed {seed}", "summary['best_step']", 'metrics["splits"]["test_indomain"]["overall"]', 'metrics["splits"]["test_emotion"]["overall"]', "indomain['accuracy']", "indomain['ece']", "emotion['ece']", "wall clock {minutes:.1f} min"):
        assert field in summary, field
    assert 'metrics["git"]["commit"] == COMMIT' in summary and 'metrics["init_adapter_sha256"] == summary["init_adapter_sha256"]' in summary
    # MAX_HOURS applies to each run's training command, with the pair's own seed.
    assert "arm={arm} seed={seed} --max-hours {MAX_HOURS}" in body


def test_rlcd_smoke_runs_are_gitignored():
    import subprocess

    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-q", "runs/rlcd_06b_brier_s0_smoke/train_summary.json"], cwd=REPO)
    assert ignored.returncode == 0
    for path in ("runs/rlcd_06b_brier_s0/adapter_last/adapter_config.json", "runs/rlcd_06b_brier_s0/adapter/adapter_config.json", "runs/rlcd_06b_brier_s0/last/state.pt"):
        assert subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=REPO).returncode == 0, path


def test_rlcd_notebook_env_matches_train_rlcd():
    """ENV is passed to every training command, its default is stage 1's, and noisy sessions refuse sft_cont as train_rlcd.py does."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_rlcd_for_notebook", REPO / "scripts" / "train_rlcd.py")
    rl = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rl
    spec.loader.exec_module(rl)
    cells = code_cells("kaggle_rlcd.ipynb")
    assert re.search(r'^ENV = "deterministic"', cells[0], re.M)
    joined = "\n".join(cells)
    assert joined.count("train_rlcd.py --config configs/rlcd_{SIZE}.yaml --env {ENV}") == 2
    noisy_choices = re.search(r'else \((.*)\)\n', cells[1].split("ARM_CHOICES = ")[1]).group(1)
    assert tuple(a.strip().strip('"') for a in noisy_choices.split(",")) == rl.NOISY_ARMS
    assert rl.ENVS == ("deterministic", "noisy") and 'assert ENV in ("deterministic", "noisy")' in cells[1]


# v3 (tasks 3.4 and 3.5, decision 56)


def v3_plan():
    """PLAN and run_name_of from the notebook's clone cell, executed without the clone."""
    clone = code_cells("kaggle_v3.ipynb")[1]
    block = clone[clone.index("PLAN = {") : clone.index("RUNS = [run_name_of")]
    namespace = {"SESSION": None}
    exec("def plan_for(SESSION):\n" + "\n".join("    " + line for line in block.splitlines()) + "\n    return PLAN, run_name_of", namespace)
    return namespace["plan_for"]


def test_v3_notebook_parameters_and_session_plans_match_kaggle_md_and_train_rlcd():
    cells = code_cells("kaggle_v3.ipynb")
    params = cells[0]
    for name in ("REPO", "COMMIT", "SESSION", "ADAPTER_DATASET", "LOG_DATASET", "LOG_SHA256", "SMOKE_STEPS", "MAX_HOURS"):
        assert re.search(rf"^{name} = ", params, re.M), name
    plan_for = v3_plan()
    plans = {session: plan_for(session) for session in "ABCD"}
    names = {session: [run_name_of(*run) for run in plan] for session, (plan, run_name_of) in plans.items()}
    assert names["A"] == [f"v3_06b_{arm}_n{n}_s0" for arm in ("full_sft", "positive_sft") for n in (500, 2000, 5000)]
    assert names["B"] == ["v3_06b_direct_brier_n500_s0", "v3_06b_direct_brier_n2000_s0", "v3_06b_direct_brier_n5000_s0", "v3_06b_direct_brier_n5000_s1"]
    assert names["C"] == ["v3_06b_direct_brier_n5000_s2", "v3_06b_direct_brier_n5000_s0_noisy", "v3_06b_positive_sft_n5000_s0_noisy", "v3_06b_direct_brier_n5000_s0_log1"]
    assert names["D"] == ["v3_06b_full_sft_n5000_s1", "v3_06b_full_sft_n5000_s2"]  # task 3.6
    all_names = [n for session in "ABC" for n in names[session]]
    assert len(all_names) == len(set(all_names)) == 14  # the 14 trained runs of V3_DESIGN section 8
    assert not set(names["D"]) & set(all_names)
    spec = importlib.util.spec_from_file_location("train_rlcd_nb", REPO / "scripts" / "train_rlcd.py")
    rl = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rl
    spec.loader.exec_module(rl)
    for session, (plan, run_name_of) in plans.items():
        for arm, n, seed, noisy, log in plan:
            assert arm in rl.LOG_ARMS and not (noisy and arm == "full_sft")
            assert run_name_of(arm, n, seed, noisy, log) == rl.log_run_name({"size": "06b", "arm": arm, "seed": seed}, n, noisy, log)
    kaggle_md = (REPO / "docs" / "KAGGLE.md").read_text()
    section = kaggle_md[kaggle_md.index("## 12. v3") :]
    for session in "ABCD":
        assert f'SESSION = "{session}"' in section, session
    assert 'assert SESSION in ("A", "B", "C", "D")' in cells[1]
    assert "jevmark-v3-logs" in section and "LOG_SHA256" in section


def test_v3_notebook_session_a_evaluates_zero_shot_and_collects_both_logs_before_training():
    cells = code_cells("kaggle_v3.ipynb")
    joined = "\n".join(cells)
    order = [
        joined.index("make data-build PY=python"),
        joined.index("!python scripts/build_v3_data.py"),
        joined.index('target = WORK / "runs" / "sft_06b" / "adapter"'),
        joined.index("evaluate.py --ckpt runs/sft_06b --run-name v3_06b_zeroshot --device cuda"),
        joined.index("collect_log.py --ckpt runs/sft_06b --seed {k} --device cuda"),
        joined.index("run_name={SMOKE_RUN} {NOISY_FLAG} --limit-steps {SMOKE_STEPS}"),
        joined.index("--log runs/v3_log_s{log}/log.jsonl --n {n} --init runs/sft_06b arm={arm} seed={seed} {noisy_flag} --max-hours {MAX_HOURS}"),
        joined.index("!python scripts/evaluate.py --ckpt runs/{run_name} --device cuda"),
    ]
    assert order == sorted(order)
    logs = next(c for c in cells if "collect_log.py" in c)
    assert 'if SESSION == "A":' in logs and 'LOG_SHA256[k] = m["log_sha256"]' in logs and "LOG DONE seed" in logs
    clone = cells[1]
    assert '"v3_06b_zeroshot", "v3_log_s0", "v3_log_s1"' in clone and clone.index('run(["git", "checkout"') < clone.index("shutil.rmtree")


def test_v3_notebook_sessions_b_and_c_check_the_log_sha256_and_every_run_is_copied_with_a_run_done_line():
    cells = code_cells("kaggle_v3.ipynb")
    clone, logs = cells[1], next(c for c in cells if "collect_log.py" in c)
    assert 're.fullmatch(r"[0-9a-f]{64}", LOG_SHA256.get(k, ""))' in clone
    other = logs[logs.index("else:") :]
    assert "file_sha256(target / \"log.jsonl\")" in other and "digest != recorded or digest != LOG_SHA256[k]" in other and "raise RuntimeError" in other
    loop = next(c for c in cells if "for (arm, n, seed, noisy, log), run_name in zip(PLAN, RUNS):" in c)
    assert 'if not TRAINED.get("smoke"):' in loop and loop.index('if not TRAINED.get("smoke"):') < loop.index("for (arm, n, seed, noisy, log)")
    body = loop[loop.index("for (arm, n, seed, noisy, log)") :]
    steps = [
        "!python scripts/train_rlcd.py",
        "training_exit = _exit_code",
        "copy_run(run_name)  # keep the state even if the check fails",
        'summary = require_training(WORK / "runs" / run_name, run_started, training_exit)',
        "evaluation_allowed(run_name)",
        "!python scripts/evaluate.py --ckpt runs/{run_name}",
        'check_exit(f"evaluate.py for {run_name}", _exit_code)',
        "copy_run(run_name)\n",
        "DONE.append(run_summary(run_name, log, summary, run_started))",
    ]
    positions = [body.index(step) for step in steps]
    assert positions == sorted(positions)
    assert 'metrics["v3"]["log_sha256"] == summary["log_sha256"] == LOG_SHA256[log]' in loop and "RUN DONE" in loop
    assert joined_count(cells, "require_training(") == 2


def joined_count(cells, text):
    return "\n".join(cells).count(text)


# Adapter evaluation (task 3.6)


def test_eval_notebook_adapter_eval_defaults_to_none_and_skips_the_base_evaluation_when_set():
    cells = code_cells("kaggle_eval.ipynb")
    assert re.search(r"^ADAPTER_EVAL = None$", cells[0], re.M)
    clone = cells[1]
    assert '{"dataset", "run", "name", "adapter_dir"} <= set(ADAPTER_EVAL)' in clone
    assert 'stale += [WORK / "runs" / name for name in (ADAPTER_EVAL["name"], f"{ADAPTER_EVAL[\'name\']}_limit{LIMIT}")]' in clone
    smoke = next(c for c in cells if "--ckpt base --config configs/base_{SIZES[0]}.yaml --limit {LIMIT}" in c)
    assert smoke.index("if ADAPTER_EVAL is None:") < smoke.index("!python scripts/evaluate.py")
    base = next(c for c in cells if "--ckpt base --config configs/base_{size}.yaml --device cuda" in c)
    assert "for size in (SIZES if ADAPTER_EVAL is None else []):" in base
    data = next(c for c in cells if "make data-build PY=python" in c)
    assert data.index("if ADAPTER_EVAL is not None:") < data.index("!python scripts/build_v3_data.py")


def test_eval_notebook_adapter_eval_prepares_a_fresh_run_smokes_then_evaluates_and_checks_it():
    cells = code_cells("kaggle_eval.ipynb")
    cell = next(c for c in cells if "prepare_adapter_run.py" in c)
    assert cell.startswith("#") and "if ADAPTER_EVAL is not None:" in cell
    steps = [
        'candidates = [root / "runs" / source / adapter_dir, root / source / adapter_dir, root / adapter_dir, root]',
        "!python scripts/prepare_adapter_run.py --adapter {adapter} --source-run runs/{source} --adapter-dir {adapter_dir} --name {name} {sha_flag}",
        "!python scripts/evaluate.py --ckpt runs/{name} --limit {LIMIT} --device cuda",
        "!python scripts/evaluate.py --ckpt runs/{name} --device cuda",
    ]
    positions = [cell.index(step) for step in steps]
    assert positions == sorted(positions) and cell.count("raise RuntimeError") == 4
    assert "--no-merge" not in cell and "--splits" not in cell  # merged adapter, the v3 run's default splits
    summary = cells[-1]
    assert '[ADAPTER_EVAL["name"]]' in summary and 'metrics["adapter_sha256"] == record["adapter_sha256"]' in summary
    assert '["v3_banking77_test_full", "test_banking77", "test_indomain", "test_unseen_intents"]' in summary and "ADAPTER EVAL DONE" in summary
    assert cells.index(cell) < len(cells) - 1


def test_adapter_eval_parameters_in_kaggle_md_match_the_notebook_and_the_split_order():
    from jevmark.data.build import V3_EVAL_SPLITS

    section = (REPO / "docs" / "KAGGLE.md").read_text().split("## 13.")[1]
    for key in ('"dataset"', '"run"', '"name"', '"adapter_dir"', '"sha256"'):
        assert key in section, key
    assert '"name": "v3_06b_direct_brier_n5000_s0_noisy_last"' in section and '"adapter_dir": "adapter_last"' in section
    summary = code_cells("kaggle_eval.ipynb")[-1]
    assert str(list(V3_EVAL_SPLITS)).replace("'", '"') in summary

