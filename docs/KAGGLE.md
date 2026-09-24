# Running jevmark on Kaggle

The notebooks in `notebooks/` are thin wrappers: they clone this repository at one commit, install pinned dependencies and run a script. All logic lives in the repository. This page covers `notebooks/kaggle_eval.ipynb`, which produces the frozen-base baseline B0 (task 1.6) for both backbones.

## 1. One-time setup: a read-only GitHub token

The repository is private, so the notebook needs a token. Use a fine-grained token that can only read this one repository.

1. On GitHub: Settings, Developer settings, Personal access tokens, Fine-grained tokens, Generate new token.
2. Token name: `kaggle-jevmark-read`. Expiration: the shortest that covers your runs (for example 30 days).
3. Resource owner: your account. Repository access: Only select repositories, then pick `jevmark`.
4. Permissions, Repository permissions: Contents set to Read-only. Metadata becomes Read-only automatically. Leave every other permission at No access.
5. Generate the token and copy it once; GitHub will not show it again.

To revoke it later: the same page, the token, Delete.

## 2. One-time setup: the Kaggle secret

1. Open any Kaggle notebook editor, then Add-ons, Secrets, Add a new secret.
2. Label: `GITHUB_TOKEN` (exactly this name; the notebook reads it). Value: the token.
3. Secrets are attached per notebook: after importing the notebook (section 3), open Add-ons, Secrets there and tick `GITHUB_TOKEN`.

The notebook passes the token to git through environment variables only, never on a command line, never in `.git/config`, and replaces it with `***` in any output it prints.

## 3. Import and configure the notebook

1. Kaggle, Create, New Notebook. Then File, Import Notebook, and upload `notebooks/kaggle_eval.ipynb` from your local checkout.
2. Settings panel (right side):
   - Accelerator: GPU T4 x2.
   - Internet: on (needed for GitHub, pip and the Hugging Face Hub; it requires a phone-verified Kaggle account).
   - Persistence: none needed.
3. Attach the `GITHUB_TOKEN` secret (section 2, step 3).
4. In the first code cell set:
   - `REPO` to the repository as `owner/name`.
   - `COMMIT` to the full 40-character sha to evaluate, normally the latest commit on `main` that you have pushed (`git rev-parse HEAD` locally, after `git push`).

## 4. Which cells to run

Run the cells top to bottom:

| Cell | What it does | Time on T4 (estimate; measure and update) |
|---|---|---|
| Parameters | `REPO`, `COMMIT`, `LIMIT` | |
| Clone | fetches exactly `COMMIT` into `/tmp/jevmark` and checks the sha | seconds |
| Install | uninstalls Kaggle's `torchao`, then `pip install -r requirements-kaggle.txt` (the seven Hugging Face packages) and jevmark with `--no-deps`; prints versions and GPU count | 1 to 2 min |
| Data | `make data PY=python`; fails loudly if any data check fails | under 1 min |
| Smoke | 30 records per split on 0.6B, writes `runs/base_06b_limit30/` | a few min, mostly downloads |
| B0 0.6B | all eight splits, writes `runs/base_06b/` | about 0.4 GPU hours (measured 28 min with fp32 weights, 21 min with fp16) |
| B0 1.7B | all eight splits, writes `runs/base_17b/` | about 1.0 GPU hours (measured 56 min) |
| Copy | copies `runs/` to `/kaggle/working/runs` and prints each run's commit, dirty flag, fp32 fallback and wall clock | seconds |

Check the smoke run before the full ones: it should end with `wrote .../metrics.json`. `requirements-kaggle.txt` pins only transformers, tokenizers, peft, datasets, accelerate, huggingface-hub and safetensors, at the versions in `uv.lock`; the image keeps its own torch, numpy, pandas, pyarrow and scipy (decision 23). The install cell first runs `pip uninstall -y torchao`: the Kaggle image ships torchao 0.10, and with it installed peft 0.21 raises an error when it injects LoRA adapters; jevmark does not use torchao. If the install cell reports a dependency conflict that names one of the pinned packages, or torch or numpy, stop there and report it; the pinned versions may need a lock change.

Each evaluation logs a warning and records `"fp32_fallback_used": true` in `metrics.json` if the first batch has NaN or inf slot logits under fp16 (decision 28).

## 5. What to download

From the notebook's Output panel (the `/kaggle/working` directory), download `runs/base_06b/` and `runs/base_17b/`. Each contains:

- `metrics.json`: every metric, per split and per question type, plus the commit sha, data file hashes, precision, batching precision and latency.
- `config.yaml` and `model_id.txt`.
- `plots/<split>.png`: one reliability diagram per split.

The smoke directory `runs/base_06b_limit30/` is for checking only; do not commit it (it is gitignored).

## 6. Committing the results

1. Put the two directories at `runs/base_06b/` and `runs/base_17b/` in the local checkout, replacing anything there.
2. Check that each `metrics.json` has `"git": {"commit": "<COMMIT>", "dirty": false}` with the sha you evaluated.
3. Commit them together, for example `task 1.6: B0 metrics for base_06b and base_17b`, and only then tick the B0 acceptance in `docs/TASKS.md` with those paths.

## 7. Training: `notebooks/kaggle_train.ipynb` (task 1.7)

One session trains one backbone size, then evaluates it and re-evaluates the frozen base of the same size with the same code version (decision 38). Token, secret, import and notebook settings are the same as in sections 1 to 3.

### Two-session plan

| Session | `SIZE` | Training (estimate; measure and update) | Evaluations | Writes |
|---|---|---|---|---|
| 1 | `06b` | about 0.8 GPU hours (measured 47 min) | sft 0.4 h, base 0.4 h | `runs/sft_06b/`, `runs/base_06b/` |
| 2 | `17b` | about 1.5 to 2 GPU hours (gradient checkpointing on) | sft 1.0 h, base 1.0 h | `runs/sft_17b/`, `runs/base_17b/` |

Both sessions use the same `COMMIT`, so the four runs share one code version. Set `REPO`, `COMMIT`, `SIZE`, `SMOKE_STEPS` (default 20) and `MAX_HOURS` (default 6.0, which leaves room for the two evaluations inside a 9 hour session) in the first code cell, then run the cells top to bottom:

1. Clone, install and data: as in the evaluation notebook.
2. Smoke training: `SMOKE_STEPS` steps into `runs/sft_<size>_smoke/`, including one validation, the final valid pass and a `train_summary.json`. Check that it ends with `done at step ...` before going on.
3. Full training into `runs/sft_<size>/`: logs every step to `training_log.jsonl`, validates every 200 steps on 1000 fixed valid records, keeps the best adapter by validation NLL in `adapter/`, and the resumable state in `last/`. The cell copies `runs/` to `/kaggle/working/runs` as soon as training returns, so the state survives a later failure.
4. Evaluate the trained adapter: `evaluate.py --ckpt runs/sft_<size>`.
5. Re-evaluate the frozen base: `evaluate.py --ckpt base --config configs/base_<size>.yaml`, replacing the committed B0 run from commit 79fc74b.
6. Copy `runs/` to `/kaggle/working/runs` and print each run's commit, dirty flag, fp32 fallback and test_indomain accuracy and ECE.

If training logs `WARNING: NaN or inf in first-batch slot logits`, it reloaded the model in fp32 and continued; `train_summary.json` records `"fp32_fallback_used": true`.

### If training stops at MAX_HOURS

The training cell then fails its final assertion on purpose, and `runs/sft_<size>/last/` holds the adapter, optimizer, scheduler, scaler, RNG state and step. To continue in a new session: download `runs/sft_<size>/` from the output, add it to the new session as a Kaggle dataset, copy it to `/tmp/jevmark/runs/sft_<size>/` after the data cell, and run `python scripts/train_sft.py --config configs/sft_<size>.yaml --resume --max-hours <hours> --device cuda` in place of the full training cell. The resumed run continues at the saved step with the same data order and gives the same result as an uninterrupted run (tested on CPU).

### What to download and where it goes

From `/kaggle/working/runs/`:

| Directory | Commit to git | Keep outside git |
|---|---|---|
| `sft_<size>/` | `config.yaml`, `model_id.txt`, `calibration.json`, `training_log.jsonl`, `train_summary.json`, `metrics.json`, `plots/` | `adapter/` (the trained weights; keep a copy, or upload to the HF Hub later), `last/`, `results.jsonl.gz` |
| `base_<size>/` | `config.yaml`, `model_id.txt`, `metrics.json`, `plots/`, replacing the committed B0 files | `results.jsonl.gz` |
| `sft_<size>_smoke/` | nothing | discard |

Put the directories at `runs/sft_<size>/` and `runs/base_<size>/` in the local checkout. `.gitignore` already excludes `adapter/`, `last/`, `results.jsonl.gz` and smoke runs, so `git add runs/sft_<size> runs/base_<size>` picks up exactly the files in the commit column. Check that every `metrics.json` has the session's `COMMIT` and `"dirty": false` before committing.
