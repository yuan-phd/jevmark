# Runs still to do: baseline training seeds

Prepared 11 November 2026 for the paper's next revision. **None of these runs has been started.** Each is a GPU run on Kaggle (one T4), launched through `notebooks/kaggle_v3.ipynb` exactly as the earlier v3 sessions (docs/KAGGLE.md, the v3 section), at a commit no earlier than the one that adds this file. The training and evaluation code is unchanged since the runs they extend, so the new seeds are comparable with the stored seed 0.

Why: positive-only SFT has one training seed everywhere, and full-label SFT one at 1.7B and on emotion, so the paper's differences against them are seed means against one baseline seed (paper Section 5, Not run).

## Priority order

| # | Learner | Size, domain | N | Seeds | Config | Log | Run names |
|---|---|---|---|---|---|---|---|
| 1 | positive_sft | 0.6B, Banking77 | 5000 | 1, 2 | `configs/v3_06b.yaml` | `runs/v3_log_s0/log.jsonl` | `v3_06b_positive_sft_n5000_s1`, `_s2` |
| 2 | full_sft | 1.7B, Banking77 | 5000 | 1, 2 | `configs/v3_17b.yaml` | `runs/v3_log_17b_s0/log.jsonl` | `v3_17b_full_sft_n5000_s1`, `_s2` |
| 3 | full_sft | 0.6B, emotion | 5000 | 1, 2 | `configs/v3_emotion_06b.yaml` | `runs/v3_log_emotion_s0/log.jsonl` | `v3_emotion_06b_full_sft_n5000_s1`, `_s2` |
| 4 | positive_sft | 0.6B, emotion | 5000 | 1, 2 | `configs/v3_emotion_06b.yaml` | `runs/v3_log_emotion_s0/log.jsonl` | `v3_emotion_06b_positive_sft_n5000_s1`, `_s2` |

Expected time, from the measured seed-0 runs (`train_summary.json`, `wall_clock_seconds_this_session`): about 1.0 h of training per 0.6B Banking77 run, 0.7 h per emotion run and 2.8 h per 1.7B run, plus about 25 min (1.7B) or 12 min (0.6B) of evaluation each. Rows 1, 3 and 4 fit one session each (`MAX_HOURS = 1.5`); row 2 needs `MAX_HOURS = 3.5` and fits one session with both seeds (about 7 h, like session J's 6.9 h).

## Commands (one per run, in the notebook's run loop)

Each run trains, then evaluates its best adapter on the domain's four v3 splits. The adapter `runs/sft_<size>` must be in place (adapter dataset, docs/KAGGLE.md section 10), and the log must match its recorded sha256 (the notebook's `LOG_SHA256`, `LOG_SHA256_17B` or `LOG_SHA256_EMOTION`).

```sh
# 1. positive_sft, 0.6B, Banking77, seeds 1 and 2
python scripts/train_rlcd.py --config configs/v3_06b.yaml --log runs/v3_log_s0/log.jsonl --n 5000 --init runs/sft_06b arm=positive_sft seed=1 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_06b_positive_sft_n5000_s1 --device cuda
python scripts/train_rlcd.py --config configs/v3_06b.yaml --log runs/v3_log_s0/log.jsonl --n 5000 --init runs/sft_06b arm=positive_sft seed=2 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_06b_positive_sft_n5000_s2 --device cuda

# 2. full_sft, 1.7B, Banking77, seeds 1 and 2
python scripts/train_rlcd.py --config configs/v3_17b.yaml --log runs/v3_log_17b_s0/log.jsonl --n 5000 --init runs/sft_17b arm=full_sft seed=1 --max-hours 3.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_17b_full_sft_n5000_s1 --device cuda
python scripts/train_rlcd.py --config configs/v3_17b.yaml --log runs/v3_log_17b_s0/log.jsonl --n 5000 --init runs/sft_17b arm=full_sft seed=2 --max-hours 3.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_17b_full_sft_n5000_s2 --device cuda

# 3. full_sft, 0.6B, emotion, seeds 1 and 2
python scripts/train_rlcd.py --config configs/v3_emotion_06b.yaml --log runs/v3_log_emotion_s0/log.jsonl --n 5000 --init runs/sft_06b arm=full_sft seed=1 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_emotion_06b_full_sft_n5000_s1 --device cuda
python scripts/train_rlcd.py --config configs/v3_emotion_06b.yaml --log runs/v3_log_emotion_s0/log.jsonl --n 5000 --init runs/sft_06b arm=full_sft seed=2 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_emotion_06b_full_sft_n5000_s2 --device cuda

# 4. positive_sft, 0.6B, emotion, seeds 1 and 2
python scripts/train_rlcd.py --config configs/v3_emotion_06b.yaml --log runs/v3_log_emotion_s0/log.jsonl --n 5000 --init runs/sft_06b arm=positive_sft seed=1 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_emotion_06b_positive_sft_n5000_s1 --device cuda
python scripts/train_rlcd.py --config configs/v3_emotion_06b.yaml --log runs/v3_log_emotion_s0/log.jsonl --n 5000 --init runs/sft_06b arm=positive_sft seed=2 --max-hours 1.5 --device cuda
python scripts/evaluate.py --ckpt runs/v3_emotion_06b_positive_sft_n5000_s2 --device cuda
```

In the notebook these are the `ARM`/`SEED` pairs of the run plan, with `SIZE`, `DOMAIN` and `N = 5000` set in the parameter cell; the commands above are what its run loop executes.

## After the runs (CPU, no code change)

1. Commit each run directory (without `results.jsonl.gz`, adapters or `last/`; `.gitignore` already excludes them) and check every `metrics.json` has the session commit and `"dirty": false`.
2. Regenerate the stage files on a clean tree:
   `uv run python scripts/compare_v3.py --size 06b`, `uv run python scripts/compare_v3.py --size 17b`, `uv run python scripts/compare_v3.py --size 06b --domain emotion`.
   They pick up every seed under `runs/` and write `seeds.positive_sft.training_seeds` and `seeds.full_sft.training_seeds` (mean, range, n_seeds), each seed minus seed 0, and, once a baseline has more than one seed, `seeds.training_seed_mean_minus_positive_sft_seed_mean` beside the existing `seeds.training_seed_mean_minus_full_sft_seed_mean`. Every existing key keeps its value.
3. `uv run python scripts/main_table.py` then shows those baselines as mean (range) without a dagger, and `uv run python scripts/make_figures.py --paper` draws their seed ranges where the figures already draw ranges.
4. `scripts/paired_deltas.py` compares any two run directories, so a new seed is compared by naming it: `uv run python scripts/paired_deltas.py --pair runs/<new seed> runs/<other run> --out runs/<name>/metrics.json`.
5. Update the paper's text, Table 1, the appendix tables, Table C.1 and Section 5 (Not run), and drop the "against one baseline seed" caveats that no longer apply.
