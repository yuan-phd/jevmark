# CLAUDE.md

## What this project is

jevmark is a small System One decision model. A text state and a set of typed questions go in; a probability distribution per question comes out, plus a confidence score for choice and score questions, from one forward pass, with no text generation. It is an independent re-implementation of the concept behind TypeSafe AI's Jev model, built for learning and for a portfolio.

All three phases are complete and tagged: v1, the supervised model (docs/RESULTS_v1.md); v2, RLCD on full labels at 0.6B (docs/RESULTS_v2.md); v3, adaptation from deployment feedback on Banking77 (docs/RESULTS_v3.md). docs/STORY.md summarises them. Three adapters (sft_06b, sft_17b, rlcd_banking77_06b) are published at https://huggingface.co/yuanphd/jevmark with docs/MODEL_CARD.md as the card (decision 61), and the project closed at tag v3-final (decision 62). Stage 2b was not run (decision 58). The only open item is optional and needs the human's go-ahead: a demonstration inside the delta-filing agent (task 3.7, decision 56).

Read docs/STORY.md first for where things stand, then docs/PLAN.md, docs/API_SPEC.md and docs/TASKS.md. Work on exactly one task at a time, in order, unless the human says otherwise.

## Non-negotiable design rules

1. The model never generates text. Every answer is read from logits at an answer slot in a single forward pass.
2. Options are input. Choice options, Score levels and Noul are defined per request. The model must read them, not memorize a fixed label set.
3. Every answer returns a full probability distribution, never only an argmax.
4. Policy lives in code. Thresholds, routing and escalation are the caller's job, not the model's.
5. The public API is the contract in docs/API_SPEC.md. Do not change request or response shape without updating that file first and telling the human.
6. Pure Python. No games, no browser automation, no UI beyond an optional FastAPI wrapper.

## Environment and constraints

- Training runs on Kaggle: 2x T4 (16 GB each), fp16 only (T4 has no bf16), about 30 GPU hours per week. Every training script must fit one run in a single Kaggle session (under 9 hours) and must resume from a checkpoint.
- Local development is CPU only. All unit tests run on CPU in under 3 minutes using a tiny randomly initialised Qwen3 config, never real weights.
- Main backbone: Qwen/Qwen3-1.7B-Base. Development backbone for pipeline debugging and RLCD sweeps: Qwen/Qwen3-0.6B-Base. Both are config switches, never hard-coded. Qwen3-4B-Base is allowed only as an optional 4-bit QLoRA experiment, never as the default.
- Precision: fp16 autocast with GradScaler and fp32 LoRA weights. Every training and evaluation script checks the first batch for NaN or inf in the slot logits and, on failure, restarts in fp32 automatically and logs that it did. fp32 fits both 0.6B and 1.7B on a T4; it does not fit 4B, which is one reason 4B is not the default.
- Where work happens: tests, data building, temperature fitting and every comparison run locally on CPU; training and model evaluation run on Kaggle through the thin notebooks (docs/KAGGLE.md). Run every GPU stage on 0.6B first, then repeat on 1.7B. Exceptions: v2 stage 2b was not run (decision 58) and v3 ran on 0.6B only (decision 56).
- Post-training only, via LoRA (peft). No from-scratch pretraining. No full fine-tune.
- Python 3.11, managed with uv (pyproject plus uv.lock). Core deps: torch (CPU wheels locally; Kaggle's preinstalled torch on Kaggle), transformers>=4.56 (Qwen3 support since 4.51; the `dtype` keyword of from_pretrained since 4.56), peft, datasets, numpy, pyyaml, matplotlib, scikit-learn (the leak probes), huggingface-hub (`hf://` checkpoints). Optional extras: dev (pytest, plus fastapi, uvicorn and httpx to test the endpoint), serve (fastapi, uvicorn), baselines (openai).

## Repository layout

```
jevmark/
  CLAUDE.md
  README.md
  pyproject.toml
  uv.lock
  requirements-kaggle.txt   generated from uv.lock (make kaggle-requirements)
  Makefile
  configs/            YAML configs: data, base, sft, rlcd and v3 per size
  jevmark/
    schema.py         request and response dataclasses, validation
    encode.py         state + questions -> text with answer slots, slot positions
    model.py          backbone + LoRA, readout of letter logits at slots; hf:// checkpoint references
    systemone.py      public function systemone(state, questions)
    config.py         YAML config loading with key=value overrides
    metrics.py        accuracy, ECE, Brier, NLL, coverage, bootstrap intervals
    calibration.py    temperature scaling on stored probabilities, and the outcome-fitted temperature
    training.py       shared training helpers: data, pre-flight, resume state
    runcheck.py       the notebooks' TRAINING PASS or FAIL check
    feedback.py       v3 deployment log: behaviour, propensity, flip, prefix reader
    environment.py    stochastic-outcome environment of the cancelled RLCD stage 3 (history)
    sampling.py       seeded stratified record samples (--limit, baseline subset)
    provenance.py     git commit and dirty flag for metrics.json
    baselines/        B1 and B2: shared prompt and parser, subset, metrics assembly
    data/
      clinc.py, sst5.py, unseen.py   CLINC150, SST-5 and the unseen-schema sets
      form.py, negation.py           form nouls and negated noul phrasings
      dedup.py        train-test and repeated-text duplicate rules
      sources.py      pinned dataset ids and revisions
      description_loader.py  generated descriptions + overrides.json, normalised
      build.py, assemble.py  record helpers, split names, dataset checks, every split in memory
      clinc_domains.json  CLINC intent-to-domain map, original release
      descriptions/   option descriptions (JSON, generated once, checked in)
  scripts/
    check_datasets.py, generate_descriptions.py
    build_data.py, leak_probe.py, check_duplicates.py   v1.3 data and its CPU gates
    build_v3_data.py, collect_log.py                     v3 data and the deployment logs
    train_sft.py, train_rlcd.py      SFT; RLCD arms and the v3 log-mode learners
    evaluate.py, recompute_metrics.py, prepare_adapter_run.py   evaluation; a stored adapter as a new run
    calibrate.py      temperature scaling of a finished run (valid or --fit-log), CPU only
    compare_calibration.py, compare_rlcd.py, compare_v3.py, compare_baselines.py   the report tables
    paired_deltas.py, simulate_advantages.py, invert_noisy.py   supporting analyses
    make_figures.py   the report figures into docs/figures/, from committed metrics files only
    baseline_llm_json.py, baseline_api.py, make_baseline_subset.py, recompute_baseline_metrics.py
    evaluate_env.py, compare_env.py   cancelled stage 3 (history)
    export_kaggle_requirements.py
    build_label_audit.py, summarise_label_audit.py   the Banking77 label-noise audit (docs/audit/)
    demo.py           one support message, five questions, a confidence gate (README)
    serve.py          optional FastAPI wrapper
  notebooks/          thin Kaggle wrappers only: train, eval, baseline B1, rlcd, v3
  tests/
  runs/               adapters, weights, results and logs gitignored; metrics.json, config.yaml,
                      calibration.json, model_id.txt, train_summary.json, training_log.jsonl and the
                      analysis files (metrics_oracle.json, latency.json, ...) are committed so reports
                      can link to them
  docs/
    PLAN.md, API_SPEC.md, TASKS.md, DECISIONS.md, DATA.md, KAGGLE.md, V3_DESIGN.md
    RESULTS_v1.md (frozen, decision 49), RESULTS_v2.md, RESULTS_v3.md, STORY.md, CLOSING_PLAN.md
    MODEL_CARD.md     the Hugging Face model card, uploaded as the Hub repository's README.md
    audit/            the label-noise audit: blind sheet, two model passes, SUMMARY.md
    figures/          PNGs drawn by scripts/make_figures.py, referenced from the reports
```

## How to work

- Before writing code for a task, restate its acceptance criteria in one or two lines and name the interfaces you will touch.
- Write or update tests for schema validation, encode.py and the readout before the implementation. These are the parts that silently break everything else.
- Trace through your own code before declaring a task done. Run `pytest -q` and the task's acceptance command.
- Every experiment produces runs/<run_name>/metrics.json plus the config that produced it. No numbers in any report without a metrics file behind them.
- Record any design decision not already in docs/DECISIONS.md there, with one line of reasoning.
- Ask the human before: changing the backbone, changing datasets or splits, changing the API contract, or spending API credits beyond what a task specifies.
- Do not fake confidence. If a dataset id does not resolve, a tokenizer assumption fails, or a number looks wrong, stop and report instead of working around it silently.
- Never install or remove system packages (brew, apt, global pip), never modify shell startup files, and never touch anything outside the repository without asking first. Project dependencies go through uv only. If a required tool is missing on the machine, stop and ask.
- Human edits to docs are committed separately from task work, with a message starting `docs:`. Do not fold them into a task commit.

## Conventions

- Type hints and dataclasses; no global mutable state.
- Configs are YAML under configs/. Training scripts take `--config` plus optional `key=value` overrides. Evaluation scripts take `--ckpt` (a run directory, or `base`) and `--splits`, plus `--config` to choose the backbone when `--ckpt` is `base`; the run name is then `base_06b` or `base_17b`. Baseline scripts take `--splits` and their model as arguments and run on the committed baseline subset (task 1.8). Every run has a seed and a run_name.
- `.gitignore`: `/data/*` with the one exception `!/data/baseline_subset.json` (the committed baseline subset), `/runs/*/adapter/`, `/runs/*/adapter_last/` (RLCD final adapters), `/runs/*_limit*/` (smoke runs), `/runs/*/results.jsonl.gz` (per-question results), `/runs/*/replies.jsonl` (baseline replies), `/runs/*/log.jsonl` (v3 feedback logs), `/runs/*/last/` (resume state), `/runs/*_smoke/` (smoke training runs), `/runs/fast_*/` (fast cycle runs), `/runs/kaggle_upload/` (adapter staging for Kaggle), weight files (`*.safetensors`, `*.bin`, `*.pt`), `.env`, caches. Never a bare `data/` pattern, which would also match `jevmark/data/`.
- Git: commit on `main` at least once per task, with the task id at the start of the message (`task 1.2: encode.py`). Tag each phase at its end (`v1`, `v2`, `v3` and the closing tag `v3-final` exist); the human pushes tags. No branches for a solo project unless the human asks.
- The tiny test model uses the real Qwen3 tokenizer, fetched once from the HF Hub at a pinned revision and cached. Do not vendor tokenizer files into the repo.
- Data files are JSONL, one record per line. Record schema is documented in docs/DATA.md.
- Prose in docs, comments and commit messages: plain English, no emojis, no em dashes.
- Any demo or integration keeps its question definitions and thresholds in one file so a reviewer can read them in one place.

## Commands

Created in task 0.1; keep this list in sync with the Makefile.

- `make setup` install the package in editable mode with dev deps
- `make test` run pytest on CPU
- `make data` build all JSONL datasets into data/, then run the CPU gates (leak probes, duplicate check)
- `make data-build` build and build checks only (used on Kaggle)
- `make data-v3` build the v3 Banking77 files next to the v1.3 files, then run the leak probes on them
- `make train-sft CONFIG=configs/sft_06b.yaml` (or `configs/sft_17b.yaml`)
- `make eval CKPT=runs/<run_name>` (or `CKPT=base EVAL_CONFIG=configs/base_06b.yaml`)
- `make kaggle-requirements` regenerate requirements-kaggle.txt from uv.lock

## Definition of done for any task

- Acceptance criteria in docs/TASKS.md met and demonstrated by a command or test.
- `pytest -q` passes.
- Docs updated if behaviour or an interface changed.
- Checkbox ticked in docs/TASKS.md with the run name or test name that proves it.
