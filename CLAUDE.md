# CLAUDE.md

## What this project is

jevmark is a small System One decision model. A text state and a set of typed questions go in; a probability distribution per question comes out, plus a confidence score for choice and score questions, from one forward pass, with no text generation. It is an independent re-implementation of the concept behind TypeSafe AI's Jev model, built for learning and for a portfolio. Phase 3 plugs it into an existing LangGraph agent (delta-filing) as the decision backend, replacing LLM calls that only decide and never write.

Read docs/PLAN.md first, then docs/API_SPEC.md, then docs/TASKS.md. Work on exactly one task at a time, in order, unless the human says otherwise.

## Non-negotiable design rules

1. The model never generates text. Every answer is read from logits at an answer slot in a single forward pass.
2. Options are input. Choice options, Score levels and Noul are defined per request. The model must read them, not memorize a fixed label set.
3. Every answer returns a full probability distribution, never only an argmax.
4. Policy lives in code. Thresholds, routing and escalation are the caller's job, not the model's.
5. The public API is the contract in docs/API_SPEC.md. Do not change request or response shape without updating that file first and telling the human.
6. Pure Python. No games, no browser automation, no UI beyond an optional FastAPI wrapper.

## Environment and constraints

- Training runs on Kaggle: 2x T4 (16 GB each), fp16 only (T4 has no bf16), about 30 GPU hours per week. Every training script must fit one run in a single Kaggle session (under 9 hours) and must resume from a checkpoint.
- Local development is CPU only. All unit tests run on CPU in under 2 minutes using a tiny randomly initialised Qwen3 config, never real weights.
- Main backbone: Qwen/Qwen3-1.7B-Base. Development backbone for pipeline debugging and RLCD sweeps: Qwen/Qwen3-0.6B-Base. Both are config switches, never hard-coded. Qwen3-4B-Base is allowed only as an optional 4-bit QLoRA experiment, never as the default.
- Precision: fp16 autocast with GradScaler and fp32 LoRA weights. Every training and evaluation script checks the first batch for NaN or inf in the slot logits and, on failure, restarts in fp32 automatically and logs that it did. fp32 fits both 0.6B and 1.7B on a T4; it does not fit 4B, which is one reason 4B is not the default.
- Where work happens: tasks 0.1 through 1.5 run locally on CPU (tests use the tiny model; data building and description generation need only the API key and dataset downloads). Tasks 1.6 onward run on Kaggle through the thin notebooks. Run every GPU stage on 0.6B first, then repeat on 1.7B.
- Post-training only, via LoRA (peft). No from-scratch pretraining. No full fine-tune.
- Python 3.11, managed with uv (pyproject plus uv.lock). Core deps: torch (CPU wheels locally; Kaggle's preinstalled torch on Kaggle), transformers>=4.56 (Qwen3 support since 4.51; the `dtype` keyword of from_pretrained since 4.56), peft, datasets, numpy, pyyaml, matplotlib, pytest. Optional extras: serve (fastapi, uvicorn), baselines (openai).

## Repository layout

```
jevmark/
  CLAUDE.md
  README.md
  pyproject.toml
  Makefile
  configs/            YAML configs, one per run type
  jevmark/
    schema.py         request and response dataclasses, validation
    encode.py         state + questions -> text with answer slots, slot positions
    model.py          backbone + LoRA, readout of letter logits at slots
    systemone.py      public function systemone(state, questions)
    metrics.py        accuracy, ECE, Brier, NLL, coverage curves
    data/
      clinc.py        CLINC150 -> choice and noul records
      sst5.py         SST-5 -> score records
      sources.py      pinned dataset ids and revisions
      description_loader.py  generated descriptions + overrides.json, normalised
      unseen.py       AG News, emotion, Banking77 -> unseen-schema eval sets
      descriptions/   option descriptions (JSON, generated once, checked in)
  scripts/
    check_datasets.py
    generate_descriptions.py
    build_data.py
    train_sft.py
    train_rlcd.py
    calibrate.py
    evaluate.py
    baseline_llm_json.py
    baseline_api.py
    serve.py          optional FastAPI wrapper
  notebooks/          thin Kaggle wrappers only: clone, install, run a script
  tests/
  runs/               adapters and weights gitignored; metrics.json, config.yaml,
                      calibration.json and model_id.txt are committed so reports can link to them
  docs/
    PLAN.md
    API_SPEC.md
    TASKS.md
    DECISIONS.md
    DATA.md           created in task 1.5
    RESULTS_v1.md     created in task 1.9
    RESULTS_v2.md     created in task 2.3
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
- Configs are YAML under configs/. Training scripts take `--config` plus optional `key=value` overrides. Evaluation and baseline scripts take `--ckpt` (a run directory, or `base`) and `--splits`, plus `--config` to choose the backbone when `--ckpt` is `base`; the run name is then `base_06b` or `base_17b`. Every run has a seed and a run_name.
- `.gitignore`: `/data/`, `/runs/*/adapter/`, weight files (`*.safetensors`, `*.bin`, `*.pt`), `.env`, caches. Never a bare `data/` pattern, which would also match `jevmark/data/`.
- Git: commit on `main` at least once per task, with the task id at the start of the message (`task 1.2: encode.py`). Tag `v1`, `v2`, `v3` at the end of each phase. No branches for a solo project unless the human asks.
- The tiny test model uses the real Qwen3 tokenizer, fetched once from the HF Hub at a pinned revision and cached. Do not vendor tokenizer files into the repo.
- Data files are JSONL, one record per line. Record schema is documented in docs/DATA.md.
- Prose in docs, comments and commit messages: plain English, no emojis, no em dashes.
- Any demo or integration keeps its question definitions and thresholds in one file so a reviewer can read them in one place.

## Commands

Created in task 0.1; keep this list in sync with the Makefile.

- `make setup` install the package in editable mode with dev deps
- `make test` run pytest on CPU
- `make data` build all JSONL datasets into data/
- `make train-sft CONFIG=configs/sft_clinc.yaml`
- `make eval CKPT=runs/<run_name>`

## Definition of done for any task

- Acceptance criteria in docs/TASKS.md met and demonstrated by a command or test.
- `pytest -q` passes.
- Docs updated if behaviour or an interface changed.
- Checkbox ticked in docs/TASKS.md with the run name or test name that proves it.
