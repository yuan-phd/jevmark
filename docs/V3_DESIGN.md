# v3 design: adaptation from deployment feedback on a real unseen domain

Decision 56. This replaces phase 2 stage 3 (decision 54) and the delta-filing integration as the content of phase 3. Sections 1 to 9 are the specification as given by the human, with the decisions taken on its review (2026-09-30) folded in. Section 10 lists what the code does not yet support. Section 11 records the conflicts and ambiguities the review found and how each was decided, plus the notes the v3 report must carry.

## 1. Claim under test

v2 showed two things. With full deterministic labels, no bandit objective can beat cross-entropy. Off-distribution overconfidence is fixed by a temperature, not by a training objective (docs/RESULTS_v2.md sections 1 and 3). The remaining claim for RLCD is the one it was designed for: after a model is deployed on a new domain, the only feedback is whether the action it chose was right, with no full label. The question v3 answers: with the same deployment feedback, does RLCD learn the new domain faster, more accurately and better calibrated than the methods a practitioner would otherwise use?

## 2. Domain

The Banking77 train split (10,003 real customer messages, 77 intents), never used in training. Its test split is already the unseen-schema evaluation in v1, where sft_06b scores .851 zero-shot on 10-option subsets with gold always present, and gpt-4.1-mini .918 (docs/RESULTS_v1.md sections 2 and 4). Questions are built exactly as test_banking77 records are: one choice question, the gold label plus 8 distractors plus `other`, labels and descriptions from the existing descriptions file, seeded option order, and the same 20 percent share of JSON states (decision 42). No new descriptions are generated. Two duplicate rules apply at the pinned revision (f5412156). A train message is dropped when its normalised text occurs in any source split a test split draws from (the decision 43 set, which contains the whole Banking77 test split): 30 messages, 25 of them Banking77 test messages and 5 CLINC test or held-out-intent utterances such as "I have lost my phone.". A text repeated within train keeps its first copy in dataset order (31 dropped). The log domain is therefore 9942 messages (`data/v3_manifest.json`, task 3.1).

## 3. Feedback channel

The standard supervised-to-bandit conversion. A logging policy answers a message, and only the correctness of its chosen option is revealed. Deterministic feedback is the main condition. A noisy condition flips the revealed outcome with probability 0.2, whether the chosen action was right or wrong, and is a secondary table. The flip is implemented in `jevmark/feedback.py`; `jevmark/environment.py` stays untouched as the history of stage 3.

## 4. Log collection (offline, once)

The logging policy is sft_06b, unchanged, sampling one action per message from q = 0.9 p + 0.1 / K, with seed 0. The recorded propensity q(a) is that mixture's probability of the chosen action. It runs over the Banking77 train split in a fixed seeded order. For every interaction the log records the record id, the question, the option order, the full probability vector, the chosen action, the propensity q(a), the revealed outcome, and, for the noisy condition, the flipped outcome. This log is the simulated deployment log. Learners see the first N interactions of the log, for N in {500, 2000, 5000}. The log is never resampled between learners. A second log is drawn once with seed 1, in the same message order, only to measure logging variance (section 8).

## 5. Learners

All start from the sft_06b adapter and train on the same log prefix.

1. **zero-shot:** sft_06b unchanged. The lower bound.
2. **temperature:** one scalar fitted on the log prefix by maximum likelihood on the Bernoulli outcomes of the chosen actions. CPU.
3. **positive-only SFT:** cross-entropy with the chosen option as the label, on interactions whose outcome was 1. Wrong interactions are discarded. This is the method a practitioner uses without RLCD.
4. **RLCD direct_brier:** the pathwise proper score on the chosen action's probability for every logged interaction, positives and negatives. No importance weights, since the expectation is under the logging policy and that is the objective. Beta 0, trained on the log prefix.
5. **full-label SFT:** cross-entropy on the gold label for every interaction in the prefix. The upper bound; in practice it costs a human label per interaction.

**Training and validation split of the prefix.** The validation slice lies inside N. Learners train on the first 0.9 N interactions and select checkpoints on the last 0.1 N. The criterion, the same for learners 3, 4 and 5, is the mean Bernoulli log-likelihood of the logged outcomes under the model's probability of the logged action. Gold is never used, not even by full-label SFT. The temperature learner is fitted on the first 0.9 N.

**Training budget.** Identical for learners 3 to 5 at each N: steps = max(200, ceil(10 x 0.9 N / 32)), that is 10 epochs over the 0.9 N training prefix at an effective batch of 32 with a floor of 200 steps. Warmup is 20 steps, then linear decay. Positive-only SFT trains on fewer records (only outcome 1) for the same number of steps, so it makes more passes over them; that is the intended equal gradient budget. lr 5e-5, micro-batch 8 x 4, seed 0; RLCD also runs seeds 1 and 2 at N 5000.

| N | training interactions | validation interactions | steps |
|---|---|---|---|
| 500 | 450 | 50 | 200 |
| 2000 | 1800 | 200 | 563 |
| 5000 | 4500 | 500 | 1407 |

## 6. Evaluation

On test_banking77 (the existing 1000-record split) and on the full Banking77 test split, which keeps those 1000 records verbatim and builds the other 2080 from a separate seeded stream with the same construction and JSON-state share: accuracy, ECE, Brier, NLL, coverage curves, bootstrap intervals by record, and paired deltas between learners. A forgetting check evaluates every learner on test_indomain and test_unseen_intents too, to show what adaptation cost in the original domain. Reference rows: gpt-4.1-mini from the B2 subset. An adapted learner's Banking77 numbers never enter the v1 or v2 unseen-schema means; only zero-shot keeps the v1 meaning of test_banking77.

## 7. Predictions, falsifiable

1. RLCD beats positive-only SFT in accuracy at every N, because it uses negative feedback.
2. RLCD's ECE is lower than positive-only SFT's, which only ever sees confirmed answers and should be overconfident.
3. RLCD approaches full-label SFT as N grows; the gap at N 5000 is the price of not having labels.
4. Under noisy feedback, RLCD degrades less than positive-only SFT.

If prediction 1 fails, RLCD has no independent value in the deployment-feedback setting either, and v2 and v3 together are a complete negative result about the method as reconstructed.

## 8. Runs

All on 0.6B only (section 11, item 6).

| runs | training steps |
|---|---|
| two log-collection passes, seeds 0 and 1 (about 10 GPU minutes each) | none |
| 9 learner runs: positive-only, RLCD and full-label SFT at N 500, 2000 and 5000 | 3 x 2170 = 6510 |
| RLCD seeds 1 and 2 at N 5000 | 2 x 1407 = 2814 |
| 2 noisy runs at N 5000: RLCD and positive-only | 2 x 1407 = 2814 |
| RLCD at N 5000 on the seed 1 log, for logging variance | 1407 |
| total, 14 trained runs | 13,545 |

Every trained run is evaluated on the full Banking77 test split (test_banking77 inside it) and on test_indomain and test_unseen_intents. Temperature and zero-shot run on CPU.

**Estimate (measure on the first run).** At the measured 0.6B rates, 3.07 s per step for SFT (`runs/sft_06b`: 70.5 minutes for 1378 steps) and about 3.6 s for RLCD (docs/KAGGLE.md section 11), 13,545 steps are 11.6 to 13.5 hours of training. The three evaluation splits hold about a quarter of the questions of a full nine-split evaluation (45 minutes for sft_06b), so about 11 minutes per run and 2.6 hours for 14 runs. Total: about 15 GPU hours of training and evaluation, about 16.5 with session overhead, in three Kaggle sessions, against the 8 hours first estimated, which assumed a few hundred steps per run. docs/KAGGLE.md section 12 gives the session plan.

## 9. Changes to the existing plan

- Phase 2 stage 3 (the K-dependent noise environment, decision 54) is cancelled. `jevmark/environment.py` and `runs/sft_06b_env` stay as history; the noisy condition is a flip implemented in `jevmark/feedback.py`.
- Phase 2 stage 2b (the 1.7B repeat) is kept but ranked after v3.
- Phase 3 no longer depends on delta-filing. A survey of that repository found only three connected tools, at most one tool call per query, no run logs and no outcome records, so it cannot supply real feedback. Integrating jevmark there becomes an optional demonstration after v3, measured on its track1 evaluation sets (router 160, tool choice 160, review 80), with decision logging added for future feedback.

## 10. Code gaps

What exists today, checked against the code at commit 7525d0f, and what is missing, with the file each gap belongs in.

**Already supported**
- The direct_brier loss on a chosen action's probability, with beta 0 allowed (`scripts/train_rlcd.py`: `question_loss`, `rlcd.beta`, the `noisy` config section).
- Cross-entropy on gold from the SFT adapter with a fixed step count and the RLCD schedule, as the `sft_cont` arm (`scripts/train_rlcd.py`). `training.steps` already fixes the step count per run, which is what the equal-budget rule needs.
- Loading the sft_06b adapter as the starting point and recording its sha256 (`scripts/train_rlcd.py --init`).
- Evaluation of test_banking77, test_indomain and test_unseen_intents with every metric, bootstrap intervals and per-question results (`scripts/evaluate.py`, `jevmark/metrics.py`), and restriction to the baseline subset for the B2 reference (`scripts/recompute_metrics.py --subset`).
- The Bernoulli log-loss temperature objective, but only on samples it draws itself (`jevmark/environment.py: fit_bandit_temperature`).
- Building Banking77 choice questions with gold, 8 distractors and `other` from the existing descriptions, but only from the test split (`jevmark/data/unseen.py: build_split`, which hard-codes `split="test"`).

**Missing**

| # | Gap | Where it belongs |
|---|---|---|
| 1 | Banking77 train records built as test_banking77 records are (9942 after the duplicate rules of section 2), and the full 3080-record test split with the existing 1000 records verbatim and 2080 from a separate stream, both with the 20 percent JSON-state share; files written outside the nine v1.3 files, with their sha256 recorded | `jevmark/data/unseen.py` (a source-split argument), a new `scripts/build_v3_data.py`, `jevmark/data/dedup.py` |
| 2 | Log collection: one action per message from q = 0.9 p + 0.1 / K, its propensity q(a), the revealed and the flipped outcome, the full probability vector, in a fixed seeded order; logs for seeds 0 and 1 | a new `scripts/collect_log.py`, a new `jevmark/feedback.py` (the log record schema, the reader for a prefix of N, the held-out 10 percent slice) |
| 3 | The noisy condition: a flip of the revealed outcome with probability 0.2; `NoisyEnvironment` draws an accepted answer from theta with eta(K) and is not reused | `jevmark/feedback.py` |
| 4 | Training from a fixed log: examples from the log prefix instead of `train.jsonl`, the logged action and outcome instead of G fresh samples from the current policy, no behaviour mixture and no importance weights | `scripts/train_rlcd.py` (a `--log` mode) |
| 5 | Learners 3 and 5 as arms of the log mode: cross-entropy on the logged action for outcome-1 interactions, and on gold for every interaction. `scripts/train_sft.py` cannot start from an existing adapter (`attach_lora` always creates a fresh LoRA) and reads only `train.jsonl` and `valid.jsonl`, so these belong next to `sft_cont` | `scripts/train_rlcd.py` |
| 6 | The step rule max(200, ceil(10 x 0.9 N / 32)) with warmup 20 and linear decay, training on the first 0.9 N interactions | `scripts/train_rlcd.py`, a new `configs/v3_06b.yaml` |
| 7 | Checkpoint selection on the last 0.1 N interactions by the mean Bernoulli log-likelihood of the logged outcomes under p(logged action), for all three trained learners; `validate` today uses gold labels on `valid.jsonl` | `scripts/train_rlcd.py` |
| 8 | The temperature learner: a Bernoulli maximum-likelihood fit on the logged (probability vector, action, outcome) of the first 0.9 N interactions, applied to evaluation results as `calibrate.py` does | `jevmark/calibration.py`, `scripts/calibrate.py` (a `--fit-log` option) |
| 9 | Evaluating new split names: `--splits` accepts only the nine `SPLITS` of `jevmark/data/build.py` | `scripts/evaluate.py`, `jevmark/data/build.py` (a separate v3 split list, so the nine stay as they are) |
| 10 | The N-curve comparison: every learner at every N on test_banking77 and the full test, paired bootstrap deltas between learners, the forgetting deltas on test_indomain and test_unseen_intents against sft_06b, the RLCD seed range at N 5000, the logging-variance run, the B2 subset reference, and the noisy table. `paired_deltas.py` compares two runs at a time only | a new `scripts/compare_v3.py` |
| 11 | A Kaggle notebook for log collection, the learner runs and their evaluations, with the fail-fast checks of decision 45 | a new `notebooks/kaggle_v3.ipynb`, `tests/test_kaggle.py` |
| 12 | Tests for each of the above on the tiny model on CPU: log determinism and propensities, flip rate, the duplicate rules and the 1000-record subset of the full test, log-mode losses by hand, the equal-step rule, feedback-only validation, the Bernoulli fit recovering a known temperature, the compare table | `tests/test_feedback.py`, `tests/test_train_rlcd.py`, `tests/test_evaluate.py` |

## 11. Review: conflicts, ambiguities and decisions (2026-09-30)

Each item was found in the review of this specification against the code and the earlier decisions; the decision taken is recorded with it (decision 56).

**Conflicts with earlier decisions**

1. **The data freeze (decision 46).** v3 adds the Banking77 train split and the remaining 2080 records of its test split. *Decided:* approved on the condition that the nine v1.3 files stay byte-identical; v3 data lives in new files with recorded hashes.
2. **Banking77 stops being an unseen schema for the adapted learners.** *Decided:* an adapted learner's Banking77 numbers never enter the v1 or v2 unseen-schema means.
3. **Duplicates across the train-test boundary** (the decision 43 rule). 25 train messages match a test message, 6 of them in test_banking77, and 31 train texts repeat within train. *Decided:* drop the 25; a repeated train text keeps one copy (the first in dataset order). *Applied in task 3.1:* the build uses the whole decision 43 set, which also drops 5 messages matching CLINC test or held-out-intent utterances, because v3 evaluates forgetting on test_indomain and test_unseen_intents; 30 test matches and 31 repeats leave 9942 messages.
4. **Decisions 11 and 12, the old v3 success criterion and CLAUDE.md** named delta-filing as the phase 3 host. *Decided:* decision 56 supersedes them for phase 3; CLAUDE.md's project description now names the v3 adaptation study, with delta-filing as an optional later demonstration.
5. **The noise model.** Flipping the outcome at 0.2 and `environment.py` with a constant eta differ: the environment reveals a wrong action as 1 with probability eta / (K - 1), .022 at K 10. *Decided:* a symmetric flip of the revealed outcome with probability 0.2, implemented in `jevmark/feedback.py`; `environment.py` is untouched history.
6. **CLAUDE.md repeats every GPU stage on 1.7B after 0.6B.** *Decided:* v3 runs on 0.6B only; CLAUDE.md records this exception.

**Ambiguities**

7. **Epsilon-greedy.** *Decided:* sampling from q = 0.9 p + 0.1 / K; the recorded propensity is q(a).
8. **The validation slice.** *Decided:* inside N. Train on the first 0.9 N, select on the last 0.1 N by the mean Bernoulli log-likelihood of the logged outcomes under p(logged action), the same criterion for learners 3, 4 and 5, gold never used. The temperature is fitted on the first 0.9 N.
9. **The step rule.** *Decided:* 10 epochs over the 0.9 N training prefix with a minimum of 200 steps, warmup 20, linear decay, identical for learners 3 to 5 at each N: 200, 563 and 1407 steps at N 500, 2000 and 5000. Positive-only SFT makes more passes over fewer records under the same step count.
10. **The full Banking77 test split.** *Decided:* it keeps the 1000 test_banking77 records verbatim and builds the other 2080 from a separate seeded stream with the same 20 percent JSON-state share; the log domain records use the same construction.

**Notes the v3 report must carry**

11. **`other` is never correct in this construction**, since gold is always among the options. Negative feedback teaches "never answer other" directly, and sft_06b predicts `other` on .049 of test_banking77 (`runs/sft_06b/metrics.json:splits.test_banking77.choice.by_gold_other.predicted_other_rate`). The report gives each learner's predicted-`other` rate and its accuracy on the questions where no learner predicted `other`.
12. **Positive-only SFT under deterministic feedback is full-label SFT on the interactions the logging policy got right.** Learners 3 and 5 differ only by the logging policy's mistakes, the hard examples.
13. **Beta 0.** Decision 54's reason for beta 0 (a noisy target theta) does not apply here; beta 0 is a choice, and the forgetting check measures its cost.
14. **Seeds and logging variance.** RLCD seeds 1 and 2 at N 5000 vary the training order only. The run on the seed 1 log measures the variance of the logging draw, which the training seeds do not cover.
15. **The B2 reference** exists only on the 500-record subset of test_banking77, so the learners are compared with it on that subset (`recompute_metrics.py --subset`), not on the full test.

**The API contract** (docs/API_SPEC.md) needs no change: every learner is an adapter plus an optional calibration.json, loaded as any run is.
