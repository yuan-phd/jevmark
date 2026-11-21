# Reproducing the tables

Moved here in full from the README. The commands run from the repository root.

Every run's `metrics.json`, config and training log are committed under `runs/`. The per-question files behind them (`results.jsonl.gz`, v3 `log.jsonl`) and the adapters are gitignored. The four deployment logs and every run's `results.jsonl.gz` are published as the Hugging Face dataset [yuanphd/jevmark-runs](https://huggingface.co/datasets/yuanphd/jevmark-runs), each at its path in this repository, so copying its `runs/` over a clone's `runs/` puts them in place; the dataset card lists each file's sha256, checked against the hashes the committed metrics files record (`scripts/publish_runs.py`). The sibling logs at flip rates 0.1 and 0.3 are rebuilt by `scripts/derive_flip_log.py`. The baseline `replies.jsonl` files are not published, so the gpt-4.1-mini rows need them; their sha256 is recorded. With the files in place every table recomputes on CPU:

| report | table | command | output |
|---|---|---|---|
| v1 | any run's metrics | `uv run python scripts/recompute_metrics.py runs/<run>` | `runs/<run>/metrics.json` |
| v1 | baselines on the subset | `uv run python scripts/compare_baselines.py` | printed, from `runs/{base,sft}_*`, `runs/b1_*`, `runs/b2_gpt-4.1-mini` |
| v1 | 1.7B against 0.6B | `uv run python scripts/paired_deltas.py --pair runs/sft_17b runs/sft_06b --pair runs/base_17b runs/base_06b --out runs/paired_17b_vs_06b/metrics.json` | `runs/paired_17b_vs_06b/` |
| v2 | temperature scaling | `uv run python scripts/calibrate.py runs/<run>`, then `uv run python scripts/compare_calibration.py` | `runs/<run>_temp/` |
| v2 | the REINFORCE simulation | `uv run python scripts/simulate_advantages.py runs/sft_06b` | `runs/advantage_simulation_06b/` |
| v2 | RLCD arms, three seeds | `uv run python scripts/compare_rlcd.py --size 06b --seeds 0 1 2 --out runs/rlcd_stage2a_06b` | `runs/rlcd_stage2a_06b/` |
| v2 | full-label accuracy with record intervals | `uv run python scripts/v2_accuracy_table.py` | `runs/v2_accuracy_06b/` |
| v3 | temperature learners | `uv run python scripts/calibrate.py runs/v3_06b_zeroshot --fit-log runs/v3_log_s0/log.jsonl --n <N>` | `runs/v3_06b_temp_n<N>/` |
| v3 | temperature on direct_brier | `uv run python scripts/evaluate.py --ckpt runs/<run> --log-prefix runs/v3_log_s0/log.jsonl --n <N> --no-probes --device cpu` (with the adapter and base weights; 2 to 18 minutes per run on a laptop CPU), then `uv run python scripts/calibrate.py runs/<run> --fit-log runs/v3_log_s0/log.jsonl --n <N> --fit-probs runs/<run>_prefix` | `runs/<run>_prefix/`, `runs/<run>_temp/` |
| v3 | the noise inversion | `uv run python scripts/invert_noisy.py runs/v3_06b_<arm>_n5000_s0_noisy --clean runs/v3_06b_<arm>_n5000_s0` | `runs/v3_06b_<arm>_n5000_s0_noisy_inverted/` |
| v3 | where a log's negatives fall (top option or other, and the uniform term's share) | `uv run python scripts/log_stats.py runs/v3_log_s0 runs/v3_log_s1 runs/v3_log_17b_s0 runs/v3_log_emotion_s0` (needs each `log.jsonl`) | the `negatives` block of each `runs/<log>/metrics.json` |
| v3 | every v3 table | `uv run python scripts/compare_v3.py` | `runs/v3_stage_06b/` |
| v3 | off-policy identity and clip counts; forgetting table and selected steps | `uv run python scripts/v3_review.py off-policy`; `uv run python scripts/v3_review.py forgetting` | the `off_policy`, `forgetting_table` and `selected_steps` keys of `runs/v3_stage_06b/metrics.json` |
| v3 | paired intervals against gpt-4.1-mini, Banking77 and emotion | `uv run python scripts/b2_paired.py` (needs `runs/b2_gpt-4.1-mini/replies.jsonl`) | `runs/b2_paired_v3/` |
| phase 4 | the flip-rate sweep: sibling logs, inversions, table | `uv run python scripts/derive_flip_log.py --log-dir runs/v3_log_s0 --flip-rate <f>` for f 0.1 and 0.3 (needs the seed 0 `log.jsonl`); `uv run python scripts/invert_noisy.py runs/v3_06b_<arm>_n5000_s0_noisy_flip<f> --flip <f> --clean runs/v3_06b_<arm>_n5000_s0`; then `uv run python scripts/compare_noise.py` | `runs/v3_log_s0_flip<f>/`, `runs/<run>_inverted/`, `runs/v3_noise_sweep_06b/` |
| phase 4 | the emotion tables | `uv run python scripts/compare_v3.py --domain emotion` | `runs/v3_stage_emotion_06b/` |
| phase 4 | the emotion label-noise audit summary | `uv run python scripts/summarise_label_audit.py --domain emotion` | `docs/audit/emotion_summary.json` |
| phase 4 | the 1.7B tables, with the direct_brier seeds | `uv run python scripts/compare_v3.py --size 17b`; `uv run python scripts/paired_deltas.py --pair runs/v3_17b_<x> runs/v3_06b_<x> ...` for zeroshot and the three learners at N 5000 `--out runs/paired_v3_17b_vs_06b/metrics.json` | `runs/v3_stage_17b/`, `runs/paired_v3_17b_vs_06b/` |
| v3 | the label-noise audit summary | `uv run python scripts/summarise_label_audit.py` | `docs/audit/summary.json` |
| all | every report figure | `uv run python scripts/make_figures.py` (committed metrics files only) | `docs/figures/` |
| all | the paper's figures, with the paper's names | `uv run python scripts/make_figures.py --paper` | `docs/figures/paper/` |
| all | the run-files dataset card and upload | `uv run python scripts/publish_runs.py` (checks and writes the card), `--upload` to publish | `data/hf_runs/README.md`, the dataset `yuanphd/jevmark-runs` |

Training and evaluation themselves (`make train-sft`, `scripts/train_rlcd.py`, `make eval`) run on Kaggle through the thin notebooks in `notebooks/`.
