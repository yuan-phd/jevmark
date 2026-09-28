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
   - `SIZES` to the backbone sizes to evaluate, in order: `["06b", "17b"]` (the default) or one of them, for example `["17b"]` for session B of the 1.7B plan (section 7).

## 4. Which cells to run

Run the cells top to bottom:

| Cell | What it does | Time on T4 (estimate; measure and update) |
|---|---|---|
| Parameters | `REPO`, `COMMIT`, `LIMIT`, `SIZES` | |
| Clone | fetches exactly `COMMIT` into `/tmp/jevmark`, checks the sha, and deletes the committed run directories this session could be confused with: every `runs/sft_*` and `runs/fast_*`, and `runs/base_<size>/` and its smoke directory for every size in `SIZES` | seconds |
| Install | uninstalls Kaggle's `torchao`, then `pip install -r requirements-kaggle.txt` (the seven Hugging Face packages) and jevmark with `--no-deps`; prints versions and GPU count | 1 to 2 min |
| Data | `make data-build PY=python`; fails loudly if any build check fails (the leak probes and duplicate check run locally, section 8) | under 1 min |
| Smoke | `LIMIT` records per split (a seeded stratified sample) on the first size in `SIZES`, writes `runs/base_<size>_limit30/`; raises on a non-zero exit | a few min, mostly downloads |
| B0 | every size in `SIZES`, all nine splits in full (decision 46), writes `runs/base_<size>/`; raises on a non-zero exit | 0.6B about 42 min, 1.7B 83.5 min (both measured on v1.3) |
| Copy | copies `runs/` to `/kaggle/working/runs`, then for each size in `SIZES` only prints `base_<size>`'s commit, dirty flag, fp32 fallback and wall clock and asserts its commit is `COMMIT`; other committed runs in the clone come from other commits and are not checked (session B's last cell failed on the committed `base_06b` before this fix, after `base_17b` had been written) | seconds |

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

One session trains one backbone size, then evaluates it and, with `EVAL_BASE = True` (the default), re-evaluates the frozen base of the same size with the same code version (decision 38). Token, secret, import and notebook settings are the same as in sections 1 to 3. Every evaluation runs all nine splits in full, train included (decision 46).

### Session plan and times

Measured for 0.6B on data v1.3 (commit a1dc2bf, one T4): install and data a few minutes, smoke training a few minutes, training 70.5 min for 1378 steps at micro-batch 8 x accumulation 4 (pre-flight peak 7.07 of 14.56 GiB), `sft_06b` evaluation 45 min (with `--shuffle-questions test_indomain`), `base_06b` evaluation 42 min: about 2.7 hours in one session.

Measured for 1.7B on data v1.3 (commit 3a7169c, one T4): training 3.0 hours for 1378 steps at micro-batch 8 x accumulation 4 with gradient checkpointing, which `configs/sft_17b.yaml` turns on (175.3 min to the last step, pre-flight peak 4.02 of 14.56 GiB), `sft_17b` evaluation 98.6 min (with `--shuffle-questions test_indomain`), `base_17b` evaluation 83.5 min. Training plus both evaluations comes to about 6.3 hours before install and smoke runs, too close to the 9 hour session limit, so 1.7B takes two sessions:

| Session | Notebook | Settings | Runs | Writes | Time (1.7B expected) |
|---|---|---|---|---|---|
| 0.6B (done) | `kaggle_train.ipynb` | `SIZE = "06b"`, `EVAL_BASE = True` | smoke, training, sft and base evaluation | `runs/sft_06b/`, `runs/base_06b/` | 2.7 h measured |
| 1.7B A | `kaggle_train.ipynb` | `SIZE = "17b"`, `EVAL_BASE = False` | smoke, training, sft evaluation | `runs/sft_17b/` | 4.7 h measured (training 3.0 h, evaluation 98.6 min), plus install and smoke |
| 1.7B B | `kaggle_eval.ipynb` | `SIZES = ["17b"]` | smoke evaluation, base evaluation | `runs/base_17b/` | 83.5 min evaluation measured, plus install and smoke |

Sessions A and B use the same `COMMIT`, so the two runs share one code version; B can run in parallel with A or after it. The pre-flight line at the start of the smoke training in session A shows the 1.7B peak memory within a minute; if it prints `PREFLIGHT FAIL`, stop and lower `training.micro_batch` in `configs/sft_17b.yaml` in a new commit.

In the training notebook set `REPO`, `COMMIT`, `SIZE`, `EVAL_BASE`, `SMOKE_STEPS` (default 20) and `MAX_HOURS` (default 6.0, a cap on training that leaves room for the evaluations in a 9 hour session) in the first code cell, then run the cells top to bottom:

1. Clone, install and data: as in the evaluation notebook. With `FAST = True` the fast cycle cell runs next and every later cell is skipped (section 8).
2. Smoke training: `SMOKE_STEPS` steps into `runs/sft_<size>_smoke/`, including the pre-flight memory check (`PREFLIGHT PASS` with the peak GPU memory of the worst-case micro-batch), one validation, the final valid pass and a `train_summary.json`. The cell ends with one line, `TRAINING PASS` or `TRAINING FAIL` with the reason, and raises on FAIL (decision 45).
3. Full training into `runs/sft_<size>/`: refuses to start unless the smoke cell passed; runs the pre-flight check again, then logs every step to `training_log.jsonl`, validates every 200 steps on 1000 fixed valid records, keeps the best adapter by validation NLL in `adapter/`, and the resumable state in `last/`. The cell copies `runs/` to `/kaggle/working/runs` as soon as training returns, so the state survives a later failure, then prints `TRAINING PASS` or `TRAINING FAIL`: it passes only if `train_sft.py` exited with code 0 and `train_summary.json` was written after the cell started, names this run and has as many steps as planned (`jevmark/runcheck.py`). On FAIL it raises and no evaluation runs.
4. Evaluate the trained adapter: `evaluate.py --ckpt runs/sft_<size> --shuffle-questions test_indomain`, only after full training passed; a non-zero exit raises.
5. Re-evaluate the frozen base: `evaluate.py --ckpt base --config configs/base_<size>.yaml`, replacing the committed B0 run; it too runs only after full training passed, and only when `EVAL_BASE` is True.
6. Copy `runs/` to `/kaggle/working/runs` and print each run's commit, dirty flag, fp32 fallback and test_indomain accuracy and ECE (the base run only when `EVAL_BASE` is True).

If training logs `WARNING: NaN or inf in first-batch slot logits`, it reloaded the model in fp32 and continued; `train_summary.json` records `"fp32_fallback_used": true`.

Memory: the clone cell sets `PYTORCH_ALLOC_CONF` (and `PYTORCH_CUDA_ALLOC_CONF` for PyTorch before 2.9) to `expandable_segments:True`. 0.6B trains at micro-batch 8 with accumulation 4 (effective batch 32): at micro-batch 16, v1.3 records (up to 392 tokens and four questions) ran out of memory on a T4 at step 392 (decision 45).

Run directories committed to git are history, not state (decision 45). The clone cell deletes `runs/sft_<size>/`, `runs/sft_<size>_smoke/` and `runs/fast_<size>/` (the evaluation notebook deletes every `runs/sft_*` and `runs/fast_*`, and `runs/base_<size>/` for every size in `SIZES`) before anything runs, and `train_sft.py` refuses a fresh start in a run directory that holds `train_summary.json`, `training_log.jsonl`, `adapter/` or `last/`. A crashed run therefore blocks a restart under the same name: resume it with `--resume` if `last/` exists, or delete the directory. The first v1.3 0.6B session evaluated a step-200 adapter because a committed v1.2 `train_summary.json` passed the old completion check.

### If training stops at MAX_HOURS

The training cell then prints `TRAINING FAIL` (fewer steps than planned) and raises on purpose, and `runs/sft_<size>/last/` holds the adapter, optimizer, scheduler, scaler, RNG state and step. To continue in a new session: download `runs/sft_<size>/` from the output, add it to the new session as a Kaggle dataset, copy it to `/tmp/jevmark/runs/sft_<size>/` after the data cell, and run `python scripts/train_sft.py --config configs/sft_<size>.yaml --resume --max-hours <hours> --device cuda` in place of the full training cell, then check it in the notebook with `require_training(WORK / "runs" / f"sft_{SIZE}", started, _exit_code)` (record `started` before the command) and set `TRAINED["full"] = True` only if it prints `TRAINING PASS`. The resumed run continues at the saved step with the same data order and gives the same result as an uninterrupted run (tested on CPU).

### What to download and where it goes

From `/kaggle/working/runs/`:

| Directory | Commit to git | Keep outside git |
|---|---|---|
| `sft_<size>/` | `config.yaml`, `model_id.txt`, `calibration.json`, `training_log.jsonl`, `train_summary.json`, `metrics.json`, `plots/` | `adapter/` (the trained weights; keep a copy, or upload to the HF Hub later), `last/`, `results.jsonl.gz` |
| `base_<size>/` | `config.yaml`, `model_id.txt`, `metrics.json`, `plots/`, replacing the committed B0 files | `results.jsonl.gz` |
| `sft_<size>_smoke/` | nothing | discard |

Put the directories at `runs/sft_<size>/` and `runs/base_<size>/` in the local checkout. `.gitignore` already excludes `adapter/`, `last/`, `results.jsonl.gz` and smoke runs, so `git add runs/sft_<size> runs/base_<size>` picks up exactly the files in the commit column. Check that every `metrics.json` has the session's `COMMIT` and `"dirty": false` before committing.

## 8. Validating a data version: CPU gates, fast cycle, then the full session (decision 42)

A new data version, or any change to the builders, goes through three steps in this order. Each costs far less than the next, and a failure stops the sequence.

1. CPU gates, locally: `make data`. It builds every split with the build checks (balance, phrasing, order rule, positions, held-out leaks), then runs `scripts/leak_probe.py` (state-free logistic regression and gradient-boosted tree probes; fails on any lift above 10 points, or above 3 points and the 99th percentile of 200 shuffled-target runs; lists every probe above 3 points with its p value) and `scripts/check_duplicates.py` (fails on any normalised text shared by train or valid and a test split). About six minutes on a laptop CPU, five of them the tree probes (measured on v1.3). Commit and push only when all three pass.
2. Fast cycle, on Kaggle: `notebooks/kaggle_train.ipynb` with `FAST = True`, `SIZE = "06b"` and `COMMIT` set to the pushed commit. It builds the data (`make data-build`), trains Qwen3-0.6B-Base for `FAST_STEPS` (300) optimizer steps into `runs/fast_06b/`, and evaluates a seeded, stratified sample of 300 records per split into the same directory with every diagnostic: bootstrap intervals, accuracy by question position, noul symmetry, letter bias, batching precision, latency, and `--shuffle-questions` on `FAST_SHUFFLE_SPLIT` (test_indomain). Budget about 25 minutes plus install (measured on v1.3: training 20.0 minutes including its two validations and the final valid pass, evaluation 3.9 minutes). The sample (`evaluate.sample_records`) splits the 300 records across sources in proportion to their size; within CLINC it gives out-of-scope utterances their proportional share, at least one, and deals the rest round-robin over the intents in a seeded order, so it holds every intent of the split (130 seen intents with 102 out-of-scope records in test_indomain, all 20 held-out intents in test_unseen_intents); other sources are sampled uniformly. Records stay in file order. The first fast run used the first 300 records instead, which covered only 3 to 16 CLINC intents and no out-of-scope utterance, because the files are in dataset order. The learning rate schedule is the full run's, so 300 steps are its warmup and early decay: the fast run shows whether training moves in the right direction and whether any diagnostic looks wrong, not final numbers. `runs/fast_*/` is gitignored; download `metrics.json` to look at it, never commit it.
3. Full session: the same notebook with `FAST = False` (section 7), only after the fast cycle looks sound on that commit.

What to check in the fast run's printout and `metrics.json` before a full session: no split with accuracy at or below the frozen base's; noul `by_kind` without a kind near 50 percent where the others have moved; `by_question_position` without a large gap between first and later questions; `order_sensitivity` on test_indomain with a high prediction agreement; symmetry `mean_sum` near 1.

## 9. Baseline B1: `notebooks/kaggle_baseline_b1.ipynb` (task 1.8)

B1 is the instruct release of each jevmark size, Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B (revisions pinned in `scripts/baseline_llm_json.py`), answering the baseline subset (`data/baseline_subset.json`: 500 records per split, all nine splits, committed) as generated JSON (decision 47). Token, secret, import and notebook settings are as in sections 1 to 3; one T4 is used. In the first code cell set `REPO`, `COMMIT`, `LIMIT` (default 5), `BATCH_SIZE` (default 16) and `SIZES` (default `["06b", "17b"]`), then run the cells top to bottom:

1. Clone, install and data, as in the evaluation notebook; the clone cell deletes any committed `runs/b1_*`. `baseline_llm_json.py` checks every split's sha256 against the one stored in the subset file and stops if the build differs.
2. Smoke run on the first size in `SIZES`: `LIMIT` subset records per split and 10 latency requests into `runs/b1_qwen<size>_json_limit<LIMIT>/` (gitignored); raises on a non-zero exit.
3. Full run for every size in `SIZES`: the whole subset, greedy, at most 256 new tokens, into `runs/b1_qwen06b_json/` and `runs/b1_qwen17b_json/`: `replies.jsonl` (one line per request, appended as it goes), `metrics.json`, `config.yaml`, `model_id.txt`. Then batch-1 latency on the first 200 records of `train.jsonl` (the requests `evaluate.py` times) and throughput at `BATCH_SIZE`. Raises on a non-zero exit.
4. Copy `runs/` to `/kaggle/working/runs`; for the runs of this session only (`SIZES`), print model, commit, dirty flag, fp32 fallback, wall clock, truncation rate and latency, and assert the commit.

Expected time, not yet measured: about 1 to 1.5 hours for 1.7B and less for 0.6B (4500 requests of mostly short JSON replies each, then 200 sequential batch-1 generations for latency). If a session ends early, add the partial `runs/b1_qwen<size>_json/` back as a dataset, copy it into place and run the script with `--size <size> --resume`: requests already in `replies.jsonl` are not generated again.

Download `runs/b1_qwen06b_json/` and `runs/b1_qwen17b_json/`. Commit `metrics.json`, `config.yaml` and `model_id.txt`; keep `replies.jsonl` outside git (it is gitignored) but keep a copy, because `scripts/compare_baselines.py` recomputes the table from it. B2 (`scripts/baseline_api.py`) runs locally, not on Kaggle.

## 10. RLCD: `notebooks/kaggle_rlcd.ipynb` (task 2.2)

One session trains one or more RLCD runs at one size from the SFT adapter, every (arm, seed) pair of `ARMS` and `SEEDS`, and evaluates each on all nine splits with the v1 protocol before the next starts. Token, secret, import and notebook settings are as in sections 1 to 3. The SFT adapters are not in git (`runs/*/adapter/` is ignored), so they reach Kaggle as a private dataset.

### The adapter dataset (once)

1. Locally, collect the two adapters in one folder with the repository's layout:
   ```
   mkdir -p ~/jevmark-sft-adapters/runs/sft_06b ~/jevmark-sft-adapters/runs/sft_17b
   cp -R runs/sft_06b/adapter ~/jevmark-sft-adapters/runs/sft_06b/
   cp -R runs/sft_17b/adapter ~/jevmark-sft-adapters/runs/sft_17b/
   shasum -a 256 runs/sft_06b/adapter/adapter_model.safetensors runs/sft_17b/adapter/adapter_model.safetensors
   ```
   Keep the two sha256 values: every RLCD run records the sha256 of the adapter it started from (`rlcd.init_adapter_sha256` in its config.yaml, `init_adapter_sha256` in train_summary.json and metrics.json), and they must match.
2. On Kaggle: Datasets, New Dataset, upload the folder `jevmark-sft-adapters`, visibility Private, slug `jevmark-sft-adapters`. Kaggle may keep the `runs/` level or drop it; the notebook looks for `runs/sft_<size>/adapter` and `sft_<size>/adapter` under `ADAPTER_DATASET`.
3. In the RLCD notebook: Add Input, Your Datasets, `jevmark-sft-adapters`. It appears under `/kaggle/input/jevmark-sft-adapters`, the default `ADAPTER_DATASET`.

### Running a session

Set `REPO`, `COMMIT`, `SIZE`, `ARMS` and `SEEDS` (lists), `ADAPTER_DATASET`, `SMOKE_STEPS` (20) and `MAX_HOURS` (6.0 in the repository, applied to each run's training), then run top to bottom. The runs are the pairs `PAIRS = [(arm, seed) for seed in SEEDS for arm in ARMS]`: every arm at the first seed, then every arm at the next, so a session that stops early leaves whole seeds.

1. Clone. This deletes any committed `runs/rlcd_<size>_<arm>_s<seed>` of every pair and the smoke directory of the first pair.
2. Install, the fail-fast helpers, the data build, and the adapter copy, which prints the adapter's sha256.
3. The smoke run, once, on the first pair (`TRAINING PASS` or `FAIL`).
4. The run loop. For each pair in order:
   - the 500-step run (`TRAINING PASS` or `FAIL`);
   - the evaluation (`evaluate.py --ckpt runs/<run> --shuffle-questions test_indomain`);
   - an immediate copy of that run, including `adapter/`, `adapter_last/` and `last/`, to `/kaggle/working/runs/<run>`;
   - one line, `RUN DONE <arm> seed <seed>: best step ..., test_indomain acc ... ece ..., test_emotion ece ..., wall clock ... min`, after checking the commit and that the evaluated run and its training summary name the same SFT adapter.
5. A recap cell that prints the summary lines again and the headline splits of each finished run.

A `TRAINING FAIL` or an evaluation error raises and stops the session, so no later run starts. Every run finished before it is already in the output, and a failed run's directory is copied too, with its `last/` state. The run name of each pair is `rlcd_<size>_<arm>_s<seed>`.

With several runs the Kaggle session limit binds before `MAX_HOURS`, so set it in the parameter cell per session: at 0.6B a run trains in about 45 minutes, and `MAX_HOURS = 1.5` keeps a stuck run from taking the rest of the session while leaving room for a normal one; at 1.7B use 3.0 (stage 2b).

Download `/kaggle/working/runs/rlcd_<size>_<arm>_s<seed>/` for each finished run. Commit `config.yaml`, `model_id.txt`, `calibration.json`, `training_log.jsonl`, `train_summary.json`, `metrics.json` and `plots/`, one commit per run. Keep `adapter/`, `adapter_last/`, `last/` and `results.jsonl.gz` outside git (all gitignored).

### If an RLCD run stops at MAX_HOURS

The run loop then prints `TRAINING FAIL` (fewer steps than planned) and raises on purpose: that run is not evaluated, no later run starts, and `/kaggle/working/runs/rlcd_<size>_<arm>_s<seed>/last/` holds the adapter, optimizer, scheduler, scaler, torch RNG, the sampler's generator state and the step. To continue in a new session:

1. Download `runs/rlcd_<size>_<arm>_s<seed>/` from the output (it must include `last/` and `config.yaml`) and add it to the new session as a Kaggle dataset, next to the adapter dataset.
2. In the new session, with the same `COMMIT` and `SIZE`, and `ARMS` and `SEEDS` set so that the stopped run is the first pair, run the parameter, clone, install, fail-fast, data and adapter-copy cells. The clone cell deletes `runs/<run>` for every pair, so copy the downloaded run directory to `/tmp/jevmark/runs/<run>/` only after it. The adapter-copy cell puts the SFT adapter back at `runs/sft_<size>/adapter`, the path the run's config.yaml names; `train_rlcd.py --resume` refuses to continue if that adapter's sha256 differs from the one recorded when the run started.
3. Skip the smoke and run-loop cells and run instead, in one cell, for the stopped pair:
   ```
   ARM, SEED = PAIRS[0]
   started = datetime.datetime.now(datetime.timezone.utc)
   !python scripts/train_rlcd.py --config configs/rlcd_{SIZE}.yaml arm={ARM} seed={SEED} --resume --max-hours {MAX_HOURS} --device cuda
   training_exit = _exit_code
   shutil.copytree(WORK / "runs" / RUNS[(ARM, SEED)], Path("/kaggle/working/runs") / RUNS[(ARM, SEED)], dirs_exist_ok=True)
   require_training(WORK / "runs" / RUNS[(ARM, SEED)], started, training_exit)
   !python scripts/evaluate.py --ckpt runs/{RUNS[(ARM, SEED)]} --shuffle-questions test_indomain --device cuda
   check_exit("evaluate.py", _exit_code)
   shutil.copytree(WORK / "runs" / RUNS[(ARM, SEED)], Path("/kaggle/working/runs") / RUNS[(ARM, SEED)], dirs_exist_ok=True)
   ```
   `arm=` and `seed=` are required with `--resume`: they give the run name, and so the run directory; every other setting, including `--init`, comes from the run's own config.yaml. Evaluate only after `require_training` printed `TRAINING PASS`.
4. For any runs that had not started, set `ARMS` and `SEEDS` to them and run the smoke and run-loop cells, or leave them for the next session.

The resumed run continues at the saved step with the same data order and the same sampled actions as an uninterrupted run (`tests/test_train_rlcd.py::test_resume_matches_an_uninterrupted_run`).

### Stage 1 plan: five arms on 0.6B, seed 0

| Arm | Question it answers | Status |
|---|---|---|
| `direct_brier` | a proper scoring rule, bounded, differentiated through p of the sampled action | done, one session: `runs/rlcd_06b_direct_brier_s0` |
| `sft_cont` | the control: is any change just more training on the same records? | the multi-arm session |
| `outcome` | plain REINFORCE with the outcome reward | the multi-arm session |
| `outcome_minus_p` | a reward that penalises confidence on wrong answers | the multi-arm session |
| `direct_log` | a proper scoring rule, unbounded, differentiated through p of the sampled action | the multi-arm session |

Done: the remaining four ran in one session at commit f30800d. `scripts/compare_rlcd.py --size 06b --seeds 0` writes the stage 1 table, `runs/rlcd_stage1_06b/metrics.json` (docs/RESULTS_v2.md section 3).

### Stage 2 plan: seeds and size (decision 53)

Stage 1 already shows the direction (docs/RESULTS_v2.md section 3); stage 2 is completed for rigour, so that the result does not rest on one seed and one size.

**Stage 2a: seeds 1 and 2 for the five usable arms at 0.6B**, ten runs in sessions of at most four runs, `SIZE = "06b"`, `MAX_HOURS = 1.5`, one `COMMIT` for all three:

| Session | `ARMS` | `SEEDS` | Runs | Estimate |
|---|---|---|---|---|
| 2a-1 | `["sft_cont", "outcome"]` | `[1, 2]` | 4 | 6.5 h |
| 2a-2 | `["outcome_minus_p", "direct_brier"]` | `[1, 2]` | 4 | 6.5 h |
| 2a-3 | `["direct_log"]` | `[1, 2]` | 2 | 3.5 h |

**Stage 2b: the five usable arms at seed 0 on 1.7B**, `SIZE = "17b"`, `SEEDS = [0]`, `MAX_HOURS = 3.0`, at most two runs per session (about 3.3 h each):

| Session | `ARMS` | Runs | Estimate |
|---|---|---|---|
| 2b-1 | `["sft_cont", "outcome"]` | 2 | 7 h |
| 2b-2 | `["outcome_minus_p", "direct_brier"]` | 2 | 7 h |
| 2b-3 | `["direct_log"]` | 1 | 3.8 h |

Stage 2 is about 34 GPU hours, more than one week's quota, so it spans two weeks.

After each stage, locally and on CPU:
1. Fit a temperature on every new run with `scripts/calibrate.py runs/<run>`, for the temperature ablation.
2. Write the table:
   - stage 2a: `scripts/compare_rlcd.py --size 06b --seeds 0 1 2 --out runs/rlcd_stage2a_06b`, which covers seeds 0, 1 and 2 and writes `runs/rlcd_stage2a_06b/metrics.json` (done);
   - stage 2b: `scripts/compare_rlcd.py --size 17b --stage 2`, which writes `runs/rlcd_stage2_17b/metrics.json`.

The REINFORCE arms `brier` and `log` are known broken (decision 52): used as a detached reward on the sampled action, a proper score never ranks a wrong action below gold, so the policy gradient lowers gold for K > 2 and is zero for K = 2. Their one run, `runs/rlcd_06b_brier_s0`, is committed as the negative result. The notebook does not offer them; to reproduce that run, set `ARMS = ["brier"]` and `SEEDS = [0]`, drop the notebook's `ARMS` assertion and add `rlcd.reinforce_proper_score=true` to both training commands.

### Time estimates

Measured at 0.6B on the first two sessions (`runs/rlcd_06b_brier_s0`, `runs/rlcd_06b_direct_brier_s0`: `train_summary.json:wall_clock_seconds_this_session` and `metrics.json:wall_clock_seconds`):

- **Training:** 43.1 and 46.3 minutes for 500 steps, including 6 validations and 2 full-valid passes.
- **Evaluation:** 39.3 and 44.0 minutes.

The four-run session (sft_cont, outcome, outcome_minus_p, direct_log) confirmed it: 46.2 to 46.7 minutes of training and 44.0 to 44.1 minutes of evaluation per run. That makes about 1.5 hours per run, against the 1.7 hours per session estimated before the first run.

**A four-run session at 0.6B is estimated at 6.5 hours:** about 5.8 hours for the four runs, plus clone, install, data build, adapter copy and the smoke run. That fits one Kaggle session with room to spare.

1.7B is not measured yet. Scaled from SFT (3.0 hours for 1378 steps, about 7.9 s per step) plus the reference pass, estimate 90 to 110 minutes of training and 99 minutes of evaluation (measured for sft_17b) per run, about 3.3 hours. At most two 1.7B runs fit one session.

The pre-flight line in the smoke run shows peak memory for policy plus reference within a minute: 8.19 of 14.56 GiB at 0.6B (`runs/rlcd_06b_direct_brier_s0/training_log.jsonl`). At 1.7B the reference adds about 3.4 GB to the SFT peak of 4.02 GiB.

## 11. RLCD stage 3: the stochastic-outcome environment (task 2.5)

Stage 3 trains RLCD against outcomes drawn from a known distribution instead of the gold label (decision 54). The data stay frozen. The environment, `jevmark/environment.py`, is a layer on top that gives each question a noise rate η(K) = min(0.40, 0.05 + 0.03 (K − 2)) and a target θ: 1 − η on gold and η / (K − 1) on each other option. On every visit it draws an accepted answer from θ, and the sampled action's outcome is 1 when it equals that answer.

The notebook is the same as in section 10, with `ENV = "noisy"`. Every training command then gets `--env noisy`, runs are named `rlcd_<size>_noisy_<arm>_s<seed>`, and `sft_cont` is not offered.

A noisy run takes `rlcd.beta` 0 and `training.steps` 1000 from the config's `noisy` section. To change them, override `noisy.beta` or `noisy.steps`; `train_rlcd.py` refuses a direct `rlcd.beta` or `training.steps` override in noisy mode. Validation adds expected Brier and cross-entropy against θ, and `adapter/` is the best step by cross-entropy against θ on the valid subset. Resume works as in section 10, with `--env noisy` added to the resume command, since it names the run.

### Sessions at 0.6B

`SIZE = "06b"`, `ENV = "noisy"`, `MAX_HOURS = 2.5`, one `COMMIT` for both sessions:

| Session | `ARMS` | `SEEDS` | Runs | Estimate |
|---|---|---|---|---|
| 3-1 | `["direct_brier"]` | `[0, 1, 2]` | 3 | 6.8 h |
| 3-2 | `["direct_log", "outcome_minus_p", "outcome"]` | `[0]` | 3 | 6.8 h |

The estimate comes from the stage 1 logs (`runs/rlcd_06b_direct_brier_s0/training_log.jsonl`, `runs/rlcd_06b_outcome_s0/training_log.jsonl`):
- **Per step:** about 3.6 s.
- **Per subset validation:** about 60 s, including the checkpoint.
- **The two full-valid passes:** about 9 minutes.
- **Before the first step:** about 2 minutes.

So 1000 steps with 10 validations take about 81 minutes, well inside `MAX_HOURS = 2.5`, and the evaluation adds 44 minutes. That is about 2.1 hours per run, 6.3 hours for three runs, plus clone, install, data build and smoke. Stage 3 is about 14 GPU hours.

### After the sessions, locally on CPU

1. `scripts/evaluate_env.py runs/<run>` for each run, which writes `runs/<run>_env/metrics.json`. `runs/sft_06b_env/metrics.json` is committed with this task.
2. `scripts/compare_env.py --size 06b`, which writes `runs/rlcd_stage3_06b/metrics.json`.

Commit each run as in section 10.
