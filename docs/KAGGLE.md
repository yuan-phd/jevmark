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
| Install | `pip install -r requirements-kaggle.txt` (the seven Hugging Face packages), then jevmark with `--no-deps`; prints versions and GPU count | 1 to 2 min |
| Data | `make data PY=python`; fails loudly if any data check fails | under 1 min |
| Smoke | 30 records per split on 0.6B, writes `runs/base_06b_limit30/` | a few min, mostly downloads |
| B0 0.6B | all eight splits, writes `runs/base_06b/` | about 0.5 GPU hours (measured 28 min) |
| B0 1.7B | all eight splits, writes `runs/base_17b/` | about 1.0 GPU hours (measured 56 min) |
| Copy | copies `runs/` to `/kaggle/working/runs` and prints each run's commit, dirty flag, fp32 fallback and wall clock | seconds |

Check the smoke run before the full ones: it should end with `wrote .../metrics.json`. `requirements-kaggle.txt` pins only transformers, tokenizers, peft, datasets, accelerate, huggingface-hub and safetensors, at the versions in `uv.lock`; the image keeps its own torch, numpy, pandas, pyarrow and scipy (decision 23). If the install cell reports a dependency conflict that names one of the pinned packages, or torch or numpy, stop there and report it; the pinned versions may need a lock change.

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
