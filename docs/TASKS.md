# jevmark: Tasks

Work top to bottom. Tick a box only with the test name or run name that proves it. GPU hour estimates are rough and refer to Kaggle T4; measure on the first real run and update.

Datasets are referenced by Hugging Face id. Ids move; the first step of any data task is to confirm the id resolves and to record the exact id and revision in docs/DATA.md.

## Phase 0: bootstrap

### 0.1 Repository skeleton
- [x] `pyproject.toml` with the package `jevmark`, Python 3.11, dev extras (pytest), optional extras (serve, baselines); `uv.lock` committed. Proof: `make setup` (uv sync, Python 3.11.15).
- [x] `.gitignore` exactly as CLAUDE.md conventions specify; verify with `git check-ignore` that `jevmark/data/clinc.py` is not ignored and `runs/x/adapter/a.safetensors` is. Proof: `git check-ignore -v --no-index` exits 1 for `jevmark/data/clinc.py`, 0 for `runs/x/adapter/a.safetensors` (rule `/runs/*/adapter/`).
- [x] `Makefile` with the targets listed in CLAUDE.md. Proof: targets setup, test, data, train-sft, eval; `make setup && make test` pass.
- [x] `configs/base.yaml` with backbone id, max_tokens, LoRA settings, seed, run_name. Proof: `configs/base.yaml`.
- [x] `tests/conftest.py` that builds a tiny random Qwen3 model (2 layers, hidden 64, vocab from the real Qwen3 tokenizer) for CPU tests. Proof: `tests/test_smoke.py::test_tiny_model_forward` passes.
- Acceptance: `make setup && make test` passes with one trivial test.

## Phase 1: v1 supervised decision model

### 1.1 schema.py
- [x] Dataclasses for NoulQuestion, ChoiceQuestion, ScoreQuestion, Request, Answer variants, Response, with `from_dict` and `to_dict`. Proof: `tests/test_schema.py::test_valid_three_question_request_keeps_order_and_round_trips`, `::test_response_round_trips_spec_example`.
- [x] Validation exactly as docs/API_SPEC.md section 2 and 7. Proof: `tests/test_schema.py` (67 tests pass); the encoded-length case is deferred to `tests/test_encode.py` in task 1.2 because it needs the tokenizer.
- Acceptance: `tests/test_schema.py` covers every error case in section 7 and one valid request per type.

### 1.2 encode.py
- [x] `encode(request, tokenizer, max_tokens) -> Encoded(input_ids, slot_positions, letter_ids, question_ids)` producing the exact text in API_SPEC section 4, tokenized per segment as section 4 specifies, with the assertion that every slot id decodes to a string ending in `:`. Proof: `tests/test_encode.py::test_decoding_input_ids_reproduces_golden`, `::test_slot_ids_end_in_colon_and_next_id_is_separator_or_end`, `::test_whole_text_tokenization_would_move_a_slot`, `::test_slot_not_ending_in_colon_fails_loudly`, `::test_encoded_length_over_max_tokens`.
- [x] Assertion at load time that `" A"` to `" Z"` are single tokens. Proof: `tests/test_encode.py::test_letter_tokens_are_single_and_distinct`, `::test_letter_token_that_splits_fails_loudly`.
- [x] Load-time check that the pinned Qwen3-0.6B-Base tokenizer and the Qwen3-1.7B-Base tokenizer produce identical ids for a fixed probe string; fail loudly if not. Proof: `tests/test_encode.py::test_pinned_06b_and_17b_tokenizers_agree`, `::test_parity_mismatch_fails_loudly`.
- [x] Option shuffling helper for training that returns the permutation so labels can be remapped. Proof: `tests/test_encode.py::test_shuffle_options_returns_permutation_for_label_remap`, `::test_shuffle_is_seeded_and_covers_all_positions`, `::test_shuffle_request_touches_choice_questions_only`; option order acceptance: `::test_option_order_changes_letter_assignment_not_letter_ids`.
- Acceptance: `tests/test_encode.py` checks that decoding `input_ids` reproduces a golden string for a three-question request, checks that every slot id decodes to a string ending in `:` and that the next id begins a `\n\n` separator or is the end of the sequence, checks that whole-text tokenization would have moved at least one slot (documenting why per-segment is required), and checks that changing option order changes letter assignment but not the set of letter ids.

### 1.3 model.py
- [x] `JevMark.load(config, checkpoint=None)` loads backbone plus optional LoRA adapter and calibration.json. Proof: `tests/test_model.py::test_load_backbone_adapter_and_calibration`, `::test_load_without_checkpoint_uses_defaults`, `::test_load_missing_checkpoint_raises_runtime_error`.
- [x] `JevMark.forward_distributions(encoded_batch) -> list of per-question probability tensors`, one forward pass per batch, softmax over selected letter columns only, temperature applied. Proof: `tests/test_model.py::test_flat_output_shapes_and_order`, `::test_distributions_sum_to_one`, `::test_hidden_state_letter_logits_equal_full_logit_columns` (tied, untied, LoRA), `::test_temperature_divides_letter_logits`, `::test_slot_logits_carry_gradients_to_lora`, `::test_forward_distributions_has_no_grad`.
- [x] Padding handled on the right; slot positions adjusted accordingly. Proof: `tests/test_model.py::test_collate_pads_on_the_right_with_mask` (right padding leaves slot positions unchanged), `::test_batch_of_two_equals_two_single_calls`.
- Acceptance: `tests/test_model.py` on the tiny model checks output shapes, that each distribution sums to 1, and that a batch of two requests gives the same numbers (within float tolerance) as two single calls. Permutation behaviour is measured in evaluation, not unit-tested on a random model.

### 1.4 systemone.py
- [x] `systemone` and `systemone_batch` per API_SPEC sections 1 and 3, including confidence, score expectation, legend, rounding, usage. Proof: `tests/test_systemone.py::test_top_level_shape`, `::test_noul_answer`, `::test_choice_answer`, `::test_score_answer`, `::test_answers_match_model_distributions`, `::test_choice_argmax_is_computed_before_rounding`, `::test_score_expectation_is_computed_before_rounding`, `::test_confidence_is_one_minus_normalised_entropy`, `::test_batch_responses_match_single_calls`, `::test_max_tokens_default_comes_from_model_and_argument_overrides`, `::test_default_model_loads_lazily_once_with_env_checkpoint`.
- [x] `scripts/serve.py` optional FastAPI wrapper exposing `POST /v1/systemone` with the same JSON. Proof: `tests/test_systemone.py::test_serve_returns_the_same_response`, `::test_serve_rejects_invalid_request_with_path`, `::test_serve_maps_runtime_error_to_500`, `::test_serve_loads_default_model_once_at_startup`.
- [x] `serve.py` parses the request body with a `json` `object_pairs_hook` that raises `ValueError` on any duplicate key, because `json.loads` otherwise keeps only the last duplicate and duplicate option labels would go undetected. Proof: `tests/test_systemone.py::test_serve_rejects_duplicate_keys`, `::test_serve_rejects_malformed_json`.
- Acceptance: `tests/test_systemone.py` round-trips the three-question example from API_SPEC through the tiny model and validates the response shape field by field, with probabilities summing to 1 within 1e-3.

### 1.5 Data
- [x] `jevmark/data/descriptions/clinc_intents.json`: one description per CLINC150 intent, generated once with the LLM API (about 150 short lines), plus two paraphrased variants per intent. Human reviews 20 sampled lines before it is used. Expected spend under a dollar with a mini-tier model; hard cap 5 dollars as a guardrail against loops. Proof: committed in `4fb8b81` as generated by the human's run (gpt-4.1-mini, 0.0699 USD for all four files, DATA.md section 6); human review done, 13 corrections in `overrides.json` (DATA.md section 7); `tests/test_description_loader.py`.
- [x] `jevmark/data/clinc.py`: from `clinc/clinc_oos` (config `plus`, formerly `clinc_oos`) build choice records: for each utterance sample K options (K uniform in 3..10) that include the gold intent with probability 0.8 and always include `other`; when gold is excluded or the utterance is out of scope, the answer is `other`. Descriptions drawn at random from the variants. Build noul records: is this about domain X (10 domains, balanced yes/no); is this out of scope (balanced per DATA.md section 2 and decision 32). Proof: `tests/test_data.py::test_clinc_gold_answers_follow_the_rules`, `::test_clinc_records_have_one_choice_and_one_noul_in_both_orders`, `::test_out_of_scope_utterances_yield_two_records`, `::test_noul_yes_share_within_40_60`, `::test_descriptions_come_from_the_loader_with_overrides`, `::test_domains_file_is_the_pinned_original_and_maps_150_intents_to_10_domains`.
- [x] Hold out 20 intents entirely (all their utterances, spread across domains) as the unseen-intent test. Record the list in docs/DATA.md. Proof: DATA.md section 4; `tests/test_data.py::test_held_out_is_20_intents_two_per_domain_and_seeded`, `::test_no_held_out_intent_in_train_or_valid`, `::test_unseen_intents_split_offers_only_held_out_intents`.
- [x] Every split, including all test splits, gets a seeded random option order fixed at build time (seed recorded in docs/DATA.md), so gold letters never cluster. Training reshuffles on top of that every epoch. Proof: seed 0 in `configs/data.yaml` and DATA.md section 2; `tests/test_data.py::test_gold_position_close_to_uniform_given_k`, `::test_build_is_deterministic`. The per-epoch reshuffle uses `encode.shuffle_request` in the task 1.7 loader.
- [x] `jevmark/data/sst5.py`: from `SetFit/sst5` build score records with five level descriptions (written by hand, checked in). Proof: `jevmark/data/sst5.py` `LEVELS`; `tests/test_data.py::test_sst5_records`.
- [x] `jevmark/data/unseen.py`: eval-only sets from `fancyzhx/ag_news` (4 options), `dair-ai/emotion` (6 options), `legacy-datasets/banking77` (the parquet mirror of `PolyAI/banking77`, verified identical; random 10-option subsets that include the gold label plus `other`). 1000 examples each. Descriptions for these labels written by hand or generated once. Proof: `tests/test_data.py::test_unseen_sets_options`, `::test_split_sizes`.
- [x] `scripts/build_data.py` writes `data/{train,valid,test_indomain,test_unseen_intents,test_sst5,test_agnews,test_emotion,test_banking77}.jsonl` and prints counts, option-count histogram and answer-letter histogram. Proof: `make data` (all checks pass, 8 files written; output in DATA.md section 5); `tests/test_data.py::test_sample_of_200_records_validates_and_encodes_within_1024`.
- [x] docs/DATA.md: record schema, exact dataset ids and revisions, split sizes, held-out intent list, description generation prompt. Proof: DATA.md sections 1 to 7.
- Acceptance: `make data` runs end to end; the gold position histogram is near-uniform conditional on K (the number of options) on every split; no held-out intent appears in train or valid (assert in a test).

### 1.6 metrics.py and evaluate.py
- [x] Metrics: accuracy, macro-F1, ECE (15 equal-width bins on top-1 probability), Brier (multi-class), NLL, reliability diagram data, coverage-vs-accuracy curve at thresholds 0.0 to 1.0 step 0.05 using the response confidence field (max(p, 1-p) for noul). ECE and coverage deliberately use different quantities; `metrics.py` documents this and every report states it once. Proof: `tests/test_metrics.py` (hand-computed values for accuracy, ECE, reliability, Brier, NLL, macro-F1, MAE, coverage); `jevmark/metrics.py` module docstring states the two confidence quantities.
- [x] Letter position bias: accuracy and mean probability by answer letter position. Proof: `tests/test_metrics.py::test_letter_bias_by_hand` (by gold position and by K).
- [x] Symmetry test for noul: for each noul test record, also encode the negated instruction (templates in `jevmark/data/negation.py`), report mean of P(yes|q) + P(yes|not q) and its std. Proof: `jevmark/data/negation.py`; `tests/test_metrics.py::test_symmetry_by_hand`, `tests/test_evaluate.py::test_negation_templates_cover_both_noul_kinds`, `::test_metrics_json_structure`.
- [x] `scripts/evaluate.py --ckpt <run dir or base> --config <cfg> --splits ...` writes `runs/<run_name>/metrics.json` and reliability plots; with `--ckpt base` the backbone comes from `--config` and the run name is `base_06b` or `base_17b`. Proof: `tests/test_evaluate.py::test_metrics_json_structure`, `::test_outputs_written`, `::test_accuracy_matches_a_direct_forward`, `::test_first_batch_nan_switches_to_fp32`, `::test_b0_configs_differ_from_base_only_in_backbone_and_run_name`.
- [x] Latency: batch-1 wall clock per request on GPU, median over 200 requests, recorded in metrics.json. Proof: `runs/base_06b/metrics.json` and `runs/base_17b/metrics.json`, `latency` (T4, commit 79fc74b).
- [x] Batching precision: on 200 requests, record in metrics.json the max absolute difference between batched and single-call probabilities (API_SPEC section 1, decision 30). Proof: `tests/test_evaluate.py::test_metrics_json_structure` (`batching_precision` on 200 requests, CPU); the GPU value arrives with the B0 run.
- [x] Acceptance: `evaluate.py --ckpt base` on Kaggle produces the frozen-base baseline B0 on all test splits for both backbones (`base_06b`, `base_17b`). About 0.4 GPU hours for 0.6B and 1.0 for 1.7B per full evaluation (0.6B measured 0.46 with fp32 weights and 0.36 with fp16 weights; 1.7B measured 0.94 with fp32 weights). Proof: `runs/base_06b/metrics.json`, `runs/base_17b/metrics.json` (commit 79fc74b, not dirty, data hashes match `make data`).

### 1.6b Evaluation follow-up (after B0)
- [x] model.py: on CUDA, frozen backbone weights in fp16, LoRA parameters and the letter readout in fp32; the fp32 fallback reloads the model in fp32. CPU stays fp32. Proof: `tests/test_model.py::test_half_backbone_policy`, `::test_load_on_cpu_keeps_everything_fp32`, `::test_half_load_keeps_lora_and_readout_fp32`, `::test_use_fp32_reloads_the_model_in_fp32`, `::test_use_fp32_without_a_source_casts_in_place`.
- [x] encode.py: cache letter ids per tokenizer object. Proof: `tests/test_encode.py::test_letter_ids_are_computed_once_per_tokenizer`, `::test_letter_id_cache_is_per_tokenizer_object`, `::test_cached_letter_ids_equal_a_fresh_computation`.
- [x] evaluate.py writes `runs/<run_name>/results.jsonl.gz` (one line per question, gitignored); `scripts/recompute_metrics.py` rebuilds `metrics.json` from it. Proof: `tests/test_evaluate.py::test_results_file_has_one_line_per_question`, `::test_recompute_rebuilds_metrics_exactly`, `tests/test_metrics.py::test_result_line_round_trip`; size 29.7 KiB on the tiny `--limit 30` smoke run (decision 37).
- [x] metrics.py: noul by kind with yes rate; choice by gold `other` versus gold named label with the rate of predicting `other`; letter bias by position and K jointly. Proof: `tests/test_metrics.py::test_noul_by_kind_by_hand`, `::test_noul_block_reports_overall_yes_rate`, `::test_choice_by_gold_other_by_hand`, `::test_choice_by_gold_other_absent_without_other_option`, `::test_letter_bias_by_k_and_position_by_hand`; `tests/test_evaluate.py::test_new_breakdowns_reach_metrics_json`.
- [x] API_SPEC section 1 and decision 30 carry the measured batched versus single differences. Proof: API_SPEC section 1; decision 30; values from `runs/base_06b/metrics.json`, `runs/base_17b/metrics.json` (`batching_precision`).
- [x] Decision: B0 is re-evaluated in the task 1.7 Kaggle session with the extended evaluate.py. Proof: decision 38.

### 1.7 train_sft.py
- [x] LoRA SFT: cross-entropy at each slot over the selected letter columns only. Starting hyperparameters: r 16, alpha 32, dropout 0.05, targets q k v o, lr 2e-4, effective batch 32, 2 epochs, max_tokens 1024, fp16 autocast with GradScaler, fp32 LoRA weights, first-batch NaN check with fp32 fallback. Option shuffling on every epoch. Gradient checkpointing on for 1.7B. Proof: `scripts/train_sft.py`, `configs/sft_06b.yaml`, `configs/sft_17b.yaml`; `tests/test_train_sft.py::test_twelve_steps_end_to_end`, `::test_prepare_reshuffles_options_and_remaps_gold`, `::test_epochs_have_different_orders_and_are_reproducible`, `::test_loss_decreases_on_four_records`, `::test_gradient_checkpointing_trains`. The first-batch NaN check runs before LoRA is attached.
- [x] Checkpoint every N steps and on session end; `--resume` flag. Proof: `tests/test_train_sft.py::test_save_at_step_6_and_resume_matches_an_uninterrupted_run`, `::test_load_state_restores_optimizer_and_scheduler`, `::test_max_hours_saves_last_and_exits_cleanly`.
- [ ] Re-evaluate B0 (`base_06b`, `base_17b`) in the same Kaggle session with the extended evaluate.py, replacing the runs from commit 79fc74b (decision 38).
- [x] Training scripts write a complete merged `config.yaml` into the run directory (backbone, tokenizer_reference, run_name, max_tokens, precision, LoRA and training settings), never only the overrides, because `default_model()` loads a run with its own `config.yaml` (API_SPEC section 1). Proof: `tests/test_train_sft.py::test_twelve_steps_end_to_end` (saved config carries backbone, tokenizer_reference, lora and training), `::test_trained_adapter_loads_for_evaluation`.
- [x] If 1.7B training runs out of memory on a T4 with fp32 frozen weights, switch the frozen backbone to fp16 with LoRA parameters in fp32, and make the fp32 fallback reload the model in fp32 instead of only disabling autocast (`JevMark.use_fp32`, decision 28). Done ahead of need in task 1.6b (fp16 frozen backbone on CUDA for inference and training; fallback reloads).
- [x] Cache letter ids per tokenizer instead of recomputing them on every `encode` call, once the data loader exists (decision 26 records the current per-call cost). Done in task 1.6b (decision 36).
- [x] Validation every N steps: accuracy and ECE on `valid`; keep best by validation NLL. Proof: `tests/test_train_sft.py::test_twelve_steps_end_to_end` (valid events at steps 5, 10 and 12, best adapter, final full-valid pass). NLL is also logged, with Brier.
- [x] `notebooks/kaggle_train.ipynb` thin wrapper: clone repo, pip install, run the script, upload `runs/<run_name>` as a Kaggle dataset or to the HF Hub. Proof: `tests/test_kaggle.py::test_train_notebook_runs_one_size_smoke_first_then_train_evaluate_and_copy`, `::test_notebook_token_is_never_printed_or_put_on_a_command_line`; results are downloaded from `/kaggle/working/runs` (docs/KAGGLE.md section 7), not uploaded.
- Acceptance: runs `sft_06b` and `sft_17b` (run names per the task 1.7 spec; previously `sft_clinc_v1_*`) complete on Kaggle, metrics on all test splits written for both. Training about 0.8 GPU hours for 0.6B (measured 47 min for 1378 steps) and 1.5 to 2 GPU hours for 1.7B (not yet measured); each 0.6B evaluation about 0.4.

### 1.8 Baselines
- [ ] B1 `scripts/baseline_llm_json.py`: Qwen3-1.7B (the instruct release, thinking disabled) prompted to answer the same questions as JSON, so the comparison is same size, generate versus read out; report accuracy, parse failure rate, latency. The confidence field is absent, so coverage curves are not produced for B1.
- [ ] B2 `scripts/baseline_api.py`: commercial API chosen by the human, structured output, on a fixed 500-example subset of each test split; record model name, price per million tokens at run time, tokens used, cost, latency. Default model: a mini-tier OpenAI model (GPT-4o-mini or its current successor). Optional: the current flagship on a 200-example subset as the accuracy ceiling. Hard cap 20 dollars as a guardrail; expected spend a few dollars.
- Acceptance: both baselines produce metrics.json in the same format; B2 total spend recorded.

### 1.9 v1 report
- [ ] `docs/RESULTS_v1.md`: table of B0, B1, B2, SFT on every split; reliability diagrams; letter bias; symmetry; latency and cost per 1000 calls; three sentences on what the numbers say and what they do not.
- Acceptance: every number in the report links to a metrics.json path.

## Phase 2: v2 RLCD

### 2.1 calibrate.py
- [ ] Fit a single temperature on `valid` (in-domain only) by minimising NLL; write calibration.json.
- [ ] Evaluate SFT+temperature on all splits, including unseen schemas, without refitting per split.
- Acceptance: run `sft_clinc_v1_temp` metrics written. Under 0.5 GPU hours.

### 2.2 train_rlcd.py
Bandit simulation on the labeled training data, starting from the SFT adapter.

- [ ] For each record, compute the policy distribution p over its K options from the current model.
- [ ] Sample G actions (G 4) from the behaviour distribution q = (1 - epsilon) p + epsilon uniform, epsilon 0.1. Store log pi(a) under p, not q. Multiply the advantage by the importance weight p(a)/q(a), clipped at 5 (config `importance_weight`, default true).
- [ ] Reveal r_a = 1 if a is the gold option else 0, for sampled actions only.
- [ ] Reward function selected by config `reward`:
  - `outcome`: r_a
  - `outcome_minus_p`: r_a - p_a
  - `brier`: 1 - (r_a - p_a)^2
  - `log`: r_a log p_a + (1 - r_a) log(1 - p_a), probabilities clipped to [1e-6, 1 - 1e-6]
- [ ] Advantage: reward minus the group mean over the G samples of the same record; optional division by group std (config).
- [ ] Loss: mean of -advantage * log pi(a), rewards detached, plus beta * KL(pi || pi_ref) with pi_ref the frozen SFT policy, beta starting at 0.02.
- [ ] Contrast arm `direct_bandit`: minimise the negative proper score of p_a differentiably, no policy gradient.
- [ ] Score questions use the same machinery with K levels; ranked probability score reward is a v2.4 option.
- [ ] Logging every step: mean reward, mean p_chosen, KL; every N steps: accuracy and ECE on `valid`. Checkpoint and resume as in 1.7.
- Acceptance: `tests/test_rlcd.py` checks each reward function on hand-computed examples and checks that the advantage has zero group mean; a 20-step smoke run on the tiny model completes on CPU.

### 2.3 Ablation runs and v2 report
- [ ] Runs: `rlcd_outcome`, `rlcd_outcome_minus_p`, `rlcd_brier`, `rlcd_log`, `rlcd_direct`, each 500 steps from the SFT adapter. Sweep all five arms with three seeds on 0.6B first (roughly 0.5 GPU hours per run), then repeat the best two arms and SFT on 1.7B (roughly 1 to 2 GPU hours per run). Measure and update.
- [ ] Evaluate every arm on all splits. Compare against SFT and SFT+temperature.
- [ ] `docs/RESULTS_v2.md`: table of accuracy and ECE per arm per split; the cascade figure (coverage vs accuracy) for SFT, SFT+temperature and the best RLCD arm; reliability diagrams on unseen schemas; a plain statement of whether the core claim held.
- Acceptance: report written; if no arm beats SFT+temperature on unseen-schema ECE, the report says so and lists the hypotheses tested.

### 2.4 Optional improvements, only if 2.3 motivates them
- [ ] Ranked probability score reward for score questions.
- [ ] Permutation averaging at inference for choice (average over R random option orders) if letter bias is material.
- [ ] Marker-token readout head as an alternative to letter logits if bias persists or options exceed 26. Requires an API_SPEC update.

## Phase 3: v3 integration into delta-filing

### 3.1 LocalDecider wrapper
- [ ] In the delta-filing repo, a `LocalDecider` class next to `LocalLLM` that loads jevmark once and exposes `decide(state, questions)`.
- [ ] All question definitions and thresholds for the agent in one file, `decisions.py`.
- Acceptance: the router node runs through LocalDecider on 20 sample queries with identical graph behaviour to the LLM router on the confident cases.

### 3.2 Switchable decision backend
- [ ] A config flag with three values: `llm`, `jevmark`, `cascade`. Cascade sends decisions with confidence below the threshold in `decisions.py` back to the LLM.
- [ ] Nodes covered: router, tool selection, retrieval rerank (one noul per chunk), completion check.
- Acceptance: the same 200-query set runs end to end in all three modes and produces a per-mode log of latency, LLM calls, tokens and final answers.

### 3.3 v3 report
- [ ] Accuracy against a human-labeled or LLM-labeled reference for the 200 queries, per mode.
- [ ] Latency and cost per mode; cost model with the API prices used, stated explicitly.
- [ ] `docs/RESULTS_v3.md` with one table and one paragraph per question in PLAN.md section How we measure.
- Acceptance: the placeholders in PLAN.md success criteria replaced with measured numbers.
