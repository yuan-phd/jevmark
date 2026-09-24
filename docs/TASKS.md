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
- [ ] `JevMark.load(config, checkpoint=None)` loads backbone plus optional LoRA adapter and calibration.json.
- [ ] `JevMark.forward_distributions(encoded_batch) -> list of per-question probability tensors`, one forward pass per batch, softmax over selected letter columns only, temperature applied.
- [ ] Padding handled on the right; slot positions adjusted accordingly.
- Acceptance: `tests/test_model.py` on the tiny model checks output shapes, that each distribution sums to 1, and that a batch of two requests gives the same numbers (within float tolerance) as two single calls. Permutation behaviour is measured in evaluation, not unit-tested on a random model.

### 1.4 systemone.py
- [ ] `systemone` and `systemone_batch` per API_SPEC sections 1 and 3, including confidence, score expectation, legend, rounding, usage.
- [ ] `scripts/serve.py` optional FastAPI wrapper exposing `POST /v1/systemone` with the same JSON.
- [ ] `serve.py` parses the request body with a `json` `object_pairs_hook` that raises `ValueError` on any duplicate key, because `json.loads` otherwise keeps only the last duplicate and duplicate option labels would go undetected.
- Acceptance: `tests/test_systemone.py` round-trips the three-question example from API_SPEC through the tiny model and validates the response shape field by field, with probabilities summing to 1 within 1e-3.

### 1.5 Data
- [ ] `jevmark/data/descriptions/clinc_intents.json`: one description per CLINC150 intent, generated once with the LLM API (about 150 short lines), plus two paraphrased variants per intent. Human reviews 20 sampled lines before it is used. Expected spend under a dollar with a mini-tier model; hard cap 5 dollars as a guardrail against loops.
- [ ] `jevmark/data/clinc.py`: from `clinc_oos` (config `plus`) build choice records: for each utterance sample K options (K uniform in 3..10) that include the gold intent with probability 0.8 and always include `other`; when gold is excluded or the utterance is out of scope, the answer is `other`. Descriptions drawn at random from the variants. Build noul records: is this about domain X (10 domains, balanced yes/no); is this out of scope.
- [ ] Hold out 20 intents entirely (all their utterances, spread across domains) as the unseen-intent test. Record the list in docs/DATA.md.
- [ ] Every split, including all test splits, gets a seeded random option order fixed at build time (seed recorded in docs/DATA.md), so gold letters never cluster. Training reshuffles on top of that every epoch.
- [ ] `jevmark/data/sst5.py`: from `SetFit/sst5` build score records with five level descriptions (written by hand, checked in).
- [ ] `jevmark/data/unseen.py`: eval-only sets from `fancyzhx/ag_news` (4 options), `dair-ai/emotion` (6 options), `PolyAI/banking77` (random 10-option subsets that include the gold label plus `other`). 1000 examples each. Descriptions for these labels written by hand or generated once.
- [ ] `scripts/build_data.py` writes `data/{train,valid,test_indomain,test_unseen_intents,test_sst5,test_agnews,test_emotion,test_banking77}.jsonl` and prints counts, option-count histogram and answer-letter histogram.
- [ ] docs/DATA.md: record schema, exact dataset ids and revisions, split sizes, held-out intent list, description generation prompt.
- Acceptance: `make data` runs end to end; the answer-letter histogram is close to uniform on every split; no held-out intent appears in train or valid (assert in a test).

### 1.6 metrics.py and evaluate.py
- [ ] Metrics: accuracy, macro-F1, ECE (15 equal-width bins on top-1 probability), Brier (multi-class), NLL, reliability diagram data, coverage-vs-accuracy curve at thresholds 0.0 to 1.0 step 0.05 using the response confidence field (max(p, 1-p) for noul). ECE and coverage deliberately use different quantities; `metrics.py` documents this and every report states it once.
- [ ] Letter position bias: accuracy and mean probability by answer letter position.
- [ ] Symmetry test for noul: for each noul test record, also encode the negated instruction (templates in `jevmark/data/negation.py`), report mean of P(yes|q) + P(yes|not q) and its std.
- [ ] `scripts/evaluate.py --ckpt <run dir or base> --config <cfg> --splits ...` writes `runs/<run_name>/metrics.json` and reliability plots; with `--ckpt base` the backbone comes from `--config` and the run name is `base_06b` or `base_17b`.
- [ ] Latency: batch-1 wall clock per request on GPU, median over 200 requests, recorded in metrics.json.
- Acceptance: `evaluate.py --ckpt base` on Kaggle produces the frozen-base baseline B0 on all test splits for both backbones (`base_06b`, `base_17b`). About 0.5 GPU hours each.

### 1.7 train_sft.py
- [ ] LoRA SFT: cross-entropy at each slot over the selected letter columns only. Starting hyperparameters: r 16, alpha 32, dropout 0.05, targets q k v o, lr 2e-4, effective batch 32, 2 epochs, max_tokens 1024, fp16 autocast with GradScaler, fp32 LoRA weights, first-batch NaN check with fp32 fallback. Option shuffling on every epoch. Gradient checkpointing on for 1.7B.
- [ ] Checkpoint every N steps and on session end; `--resume` flag.
- [ ] Cache letter ids per tokenizer instead of recomputing them on every `encode` call, once the data loader exists (decision 26 records the current per-call cost).
- [ ] Validation every N steps: accuracy and ECE on `valid`; keep best by validation NLL.
- [ ] `notebooks/kaggle_train.ipynb` thin wrapper: clone repo, pip install, run the script, upload `runs/<run_name>` as a Kaggle dataset or to the HF Hub.
- Acceptance: runs `sft_clinc_v1_06b` and `sft_clinc_v1_17b` complete on Kaggle, metrics on all test splits written for both. About 0.5 GPU hours for 0.6B and 1.5 to 2 GPU hours for 1.7B; measure and update.

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
