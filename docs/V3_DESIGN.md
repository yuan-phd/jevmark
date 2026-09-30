# v3 design: adaptation from deployment feedback on a real unseen domain

Decision 56. This replaces phase 2 stage 3 (decision 54) and the delta-filing integration as the content of phase 3. Sections 1 to 9 are the specification as given by the human. Section 10 lists what the code does not yet support. Section 11 lists conflicts with earlier decisions and the points that need a decision before any code is written.

## 1. Claim under test

v2 showed two things. With full deterministic labels, no bandit objective can beat cross-entropy. Off-distribution overconfidence is fixed by a temperature, not by a training objective (docs/RESULTS_v2.md sections 1 and 3). The remaining claim for RLCD is the one it was designed for: after a model is deployed on a new domain, the only feedback is whether the action it chose was right, with no full label. The question v3 answers: with the same deployment feedback, does RLCD learn the new domain faster, more accurately and better calibrated than the methods a practitioner would otherwise use?

## 2. Domain

The Banking77 train split (10,003 real customer messages, 77 intents), never used in training. Its test split is already the unseen-schema evaluation in v1, where sft_06b scores .851 zero-shot on 10-option subsets with gold always present, and gpt-4.1-mini .918 (docs/RESULTS_v1.md sections 2 and 4). Questions are built exactly as test_banking77 records are: one choice question, the gold label plus 8 distractors plus `other`, labels and descriptions from the existing descriptions file, seeded option order. No new descriptions are generated.

## 3. Feedback channel

The standard supervised-to-bandit conversion. A logging policy answers a message, and only the correctness of its chosen option is revealed. Deterministic feedback is the main condition. A noisy condition flips the revealed outcome with probability 0.2 and is a secondary table (section 11, item 5, on how this relates to jevmark/environment.py).

## 4. Log collection (offline, once)

The logging policy is sft_06b, unchanged, with epsilon-greedy sampling at epsilon 0.1 over its own distribution, one action per message, seed 0 (section 11, item 7). It runs over the Banking77 train split in a fixed seeded order. For every interaction the log records the record id, the question, the option order, the full probability vector, the chosen action, the propensity q(a), the revealed outcome, and, for the noisy condition, the flipped outcome. This log is the simulated deployment log. Learners see the first N interactions of the log, for N in {500, 2000, 5000}. The log is never resampled between learners.

## 5. Learners

All start from the sft_06b adapter and train on the same log prefix.

1. **zero-shot:** sft_06b unchanged. The lower bound.
2. **temperature:** one scalar fitted on the log prefix by maximum likelihood on the Bernoulli outcomes of the chosen actions. CPU.
3. **positive-only SFT:** cross-entropy with the chosen option as the label, on interactions whose outcome was 1. Wrong interactions are discarded. This is the method a practitioner uses without RLCD.
4. **RLCD direct_brier:** the pathwise proper score on the chosen action's probability for every logged interaction, positives and negatives. No importance weights, since the expectation is under the logging policy and that is the objective. Beta 0, trained on the log prefix.
5. **full-label SFT:** cross-entropy on the gold label for every interaction in the prefix. The upper bound; in practice it costs a human label per interaction.

Training budget is the same for learners 3 to 5 at each N: the number of epochs over the prefix is chosen so that every learner takes the same number of gradient steps at a given N, with the rule stated in the config. lr 5e-5, micro-batch 8 x 4, seed 0; RLCD also runs seeds 1 and 2 at N 5000. Checkpoint selection uses a held-out slice of the log prefix itself (the last 10 percent of its interactions, feedback only), never the test split and never gold labels for learners 3 and 4.

## 6. Evaluation

On test_banking77 (the existing 1000-record split) and on the full Banking77 test split, of which the 1000-record split is a subset: accuracy, ECE, Brier, NLL, coverage curves, bootstrap intervals by record, and paired deltas between learners. A forgetting check evaluates every learner on test_indomain and test_unseen_intents too, to show what adaptation cost in the original domain. Reference rows: gpt-4.1-mini from the B2 subset.

## 7. Predictions, falsifiable

1. RLCD beats positive-only SFT in accuracy at every N, because it uses negative feedback.
2. RLCD's ECE is lower than positive-only SFT's, which only ever sees confirmed answers and should be overconfident.
3. RLCD approaches full-label SFT as N grows; the gap at N 5000 is the price of not having labels.
4. Under noisy feedback, RLCD degrades less than positive-only SFT.

If prediction 1 fails, RLCD has no independent value in the deployment-feedback setting either, and v2 and v3 together are a complete negative result about the method as reconstructed.

## 8. Runs

One log-collection pass (about 10 GPU minutes); 9 learner runs (3 trained methods x 3 values of N); 2 extra RLCD seeds at N 5000; 2 noisy runs at N 5000 (RLCD and positive-only). Each run is a few hundred steps plus a Banking77 evaluation and the forgetting check: about 8 GPU hours in two Kaggle sessions. Temperature and zero-shot run on CPU.

## 9. Changes to the existing plan

- Phase 2 stage 3 (the K-dependent noise environment, decision 54) is cancelled. Its environment code is reused for the noisy condition, and `runs/sft_06b_env` stays as history.
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
| 1 | Banking77 train records built as test_banking77 records are, and the full 3080-record test split with the existing 1000 records inside it; duplicate removal against the test split (section 11, item 3); files written outside the nine v1.3 files, with their sha256 recorded | `jevmark/data/unseen.py` (a source-split argument), a new `scripts/build_v3_data.py`, `jevmark/data/dedup.py` |
| 2 | Log collection: one action per message from the logging policy, its propensity, the revealed and the flipped outcome, the full probability vector, written once in a fixed seeded order | a new `scripts/collect_log.py`, a new `jevmark/feedback.py` (the log record schema, the reader for a prefix of N, the held-out 10 percent slice) |
| 3 | The noisy condition as a flip of the revealed outcome with a constant rate; `NoisyEnvironment` draws an accepted answer from theta with eta(K) and has no flip | `jevmark/environment.py` (or `jevmark/feedback.py`) |
| 4 | Training from a fixed log: examples from the log prefix instead of `train.jsonl`, the logged action and outcome instead of G fresh samples from the current policy, no behaviour mixture and no importance weights | `scripts/train_rlcd.py` (a `--log` mode) |
| 5 | Learners 3 and 5 as arms of the log mode: cross-entropy on the logged action for outcome-1 interactions, and on gold for every interaction. `scripts/train_sft.py` cannot start from an existing adapter (`attach_lora` always creates a fresh LoRA) and reads only `train.jsonl` and `valid.jsonl`, so these belong next to `sft_cont` | `scripts/train_rlcd.py` |
| 6 | The equal-step rule over prefixes of different sizes (positive-only sees only the outcome-1 interactions), with its number in the config | `scripts/train_rlcd.py`, a new `configs/v3_06b.yaml` |
| 7 | Checkpoint selection on the held-out slice of the log with a feedback-only criterion; `validate` today uses gold labels on `valid.jsonl` | `scripts/train_rlcd.py` |
| 8 | The temperature learner: a Bernoulli maximum-likelihood fit on logged (probability vector, action, outcome) triples, applied to evaluation results as `calibrate.py` does | `jevmark/calibration.py`, `scripts/calibrate.py` (a `--fit-log` option) |
| 9 | Evaluating new split names: `--splits` accepts only the nine `SPLITS` of `jevmark/data/build.py` | `scripts/evaluate.py`, `jevmark/data/build.py` (a separate v3 split list, so the nine stay as they are) |
| 10 | The N-curve comparison: every learner at every N on test_banking77 and the full test, paired bootstrap deltas between learners, the forgetting deltas on test_indomain and test_unseen_intents against sft_06b, the RLCD seed range at N 5000, the B2 subset reference, and the noisy table. `paired_deltas.py` compares two runs at a time only | a new `scripts/compare_v3.py` |
| 11 | A Kaggle notebook for log collection, the learner runs and their evaluations, with the fail-fast checks of decision 45 | a new `notebooks/kaggle_v3.ipynb`, `tests/test_kaggle.py` |
| 12 | Tests for each of the above on the tiny model on CPU: log determinism and propensities, flip rate, log-mode losses by hand, the equal-step rule, feedback-only validation, the Bernoulli fit recovering a known temperature, the compare table | `tests/test_feedback.py`, `tests/test_train_rlcd.py`, `tests/test_evaluate.py` |

## 11. Conflicts and decisions needed before coding

**Conflicts with earlier decisions**

1. **The data freeze (decision 46) and CLAUDE.md.** Data v1.3 is frozen for v1 and v2, and CLAUDE.md requires asking before changing datasets or splits. v3 adds two sources of records, the Banking77 train split and the rest of the Banking77 test split. That is compatible with the freeze only if the new files are built separately and the nine v1.3 files and their hashes stay byte-identical. Decision 56 records the human's approval of the new data on that condition.
2. **Banking77 stops being an unseen schema for the adapted learners.** It is one of the four unseen schemas in the v1 report and in the v2 four-schema mean. A v3 learner's Banking77 numbers must never enter that mean or be compared with it; only zero-shot (sft_06b) keeps the v1 meaning.
3. **Duplicates across the train-test boundary** (decision 43 rule). Checked at the pinned revision (f5412156): 25 Banking77 train messages have the normalised text of a test message, 6 of them in the 1000-record test_banking77, and train has 31 texts that repeat within it (9972 distinct of 10,003). Decision 43 dropped the training-side copy; the same rule here removes the 25 from the log domain. Within-train repeats need a decision too.
4. **Decisions 11 and 12, the v3 success criterion and CLAUDE.md** name delta-filing as the phase 3 host and the cascade as the v3 measure. Decision 56 supersedes them for phase 3. CLAUDE.md's "What this project is" still says phase 3 plugs jevmark into delta-filing; it is the human's file, so it is left for the human to edit.
5. **The noisy condition does not match the environment it reuses.** "Flip the revealed outcome with probability 0.2" and "`environment.py` with a constant eta" are different noise models. `NoisyEnvironment` draws an accepted answer from theta, so a wrong chosen action is revealed as 1 with probability eta / (K - 1), which is .022 at K 10, not .2; only a correct action sees .2. A symmetric flip reveals a wrong action as 1 with probability .2. One of the two has to be chosen.
6. **CLAUDE.md runs every GPU stage on 0.6B first and then repeats it on 1.7B.** The design runs v3 on 0.6B only, and ranks stage 2b (the 1.7B repeat of v2) after v3. Decision 56 records v3 as a 0.6B-only phase; a 1.7B repeat would roughly triple the GPU hours of section 8.

**Ambiguities**

7. **"Epsilon-greedy at 0.1 over its own distribution"** can mean the argmax with probability .9 and a uniform option with probability .1, or sampling from q = .9 p + .1 / K, which is what `train_rlcd.py`'s behaviour policy does. The propensities, and so the log, differ.
8. **The held-out validation slice.** Is it the last 10 percent of the N interactions (so learners train on .9 N), or 10 percent beyond N? What is the feedback-only criterion for each learner: the Bernoulli log loss or the Brier score of the logged action's probability against its outcome, for all three learners alike? Does full-label SFT, the upper bound, select on gold for the slice? Does the temperature learner fit on the whole prefix or on .9 N?
9. **The equal-step rule needs its number.** At micro-batch 8 x 4 an epoch of N 500 is 16 steps (fewer for positive-only). "A few hundred steps" therefore means about 10 to 20 epochs at N 500 and 2 to 4 at N 5000. Choose the steps per N, and whether the RLCD config's 20 warmup steps stay.
10. **The full Banking77 test.** The 1000 records of test_banking77 were drawn with one generator that also drew their options. For them to stay a subset of the full test, the full split must keep those 1000 records verbatim and build the other 2080 from a separate seeded stream. test_banking77 also has 200 JSON states (the 20 percent of decision 42); "exactly as test_banking77" implies the same share in the log domain and in the 2080 new test records.
11. **`other` is never correct in this construction**, since gold is always among the options. Negative feedback on a logged `other` teaches "never answer other", which the positive-only learner only learns indirectly. Part of any RLCD accuracy gain could come from that alone. sft_06b predicts `other` on 4.9 percent of test_banking77 questions (docs/RESULTS_v1.md section 7). The comparison should also report the predicted-`other` rate and accuracy on questions where no learner predicted `other`.
12. **Positive-only under deterministic feedback is full-label SFT on the interactions the logging policy got right**, because an outcome of 1 means the chosen option is gold. Learners 3 and 5 therefore differ only by the logging policy's mistakes, which are the hard examples. That is the intended comparison, but the report should state it.
13. **Beta 0.** Decision 54 set beta 0 because the KL term pulled the policy away from a noisy target theta. Here the target is deterministic, so that argument does not apply. Beta 0 remains a choice, and the forgetting check measures its cost.
14. **Seeds.** RLCD seeds 1 and 2 at N 5000 vary the training order only. The log is collected once with seed 0, so the seed range does not include the variance of the logging draw.
15. **The B2 reference** exists only on the 500-record subset of test_banking77, so the learners are compared with it on that subset (`recompute_metrics.py --subset`), not on the full test.

**The API contract** (docs/API_SPEC.md) needs no change: every learner is an adapter plus an optional calibration.json, loaded as any run is.
