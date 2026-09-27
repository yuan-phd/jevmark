# jevmark v2 results: calibration

Every number cites the file it comes from, as `path:key`. `runs/<run>_temp/` holds temperature scaling of `runs/<run>/`, written by `scripts/calibrate.py` from the source run's `results.jsonl.gz` on CPU; `scripts/compare_calibration.py` prints the tables below from those files. The v1 report (`docs/RESULTS_v1.md`, frozen, decision 49) is the baseline this report starts from.

## 1. Temperature scaling

**Method.** One scalar temperature T per run, fitted on the gold-dependent questions of `valid` (in-domain CLINC and SST-5 only) by minimising the mean NLL of softmax(log p / T). Because log p differs from the letter logits by a constant per question, this equals rerunning the model with softmax(logits / T) (`jevmark/calibration.py`; tested against the model on the tiny backbone in `tests/test_evaluate.py::test_temperature_on_stored_probabilities_equals_the_model_with_that_temperature`). A positive temperature changes no prediction, so accuracy is identical to the v1 runs on every split; only ECE, NLL, Brier and the confidence-based coverage move. Two diagnostics are fitted but never used as results: a temperature per question type on `valid`, and an oracle temperature per split fitted on that split's own questions (`runs/<run>_temp/metrics_oracle.json`), the best any single temperature could do there.

**Fitted temperatures** (`runs/<run>_temp/metrics.json:calibration`):

| run | global T | valid NLL before, after | per type on valid (diagnostic) |
|---|---|---|---|
| sft_06b | 1.266 | .2165, .2094 | noul 1.404, choice 1.176, score 1.266 |
| sft_17b | 1.321 | .2007, .1920 | noul 1.454, choice 1.258, score 1.275 |
| base_06b | 1.114 | 1.0779, 1.0756 | noul 20.0 (the upper bound), choice 0.585, score 2.125 |
| base_17b | 1.075 | .8303, .8290 | noul 4.241, choice 0.934, score 1.155 |

Both SFT models are slightly overconfident on `valid` (T above 1), and on their own training data slightly underconfident (oracle T 0.89 and 0.95 on `train`, `runs/sft_*_temp/metrics_oracle.json:temperatures.train`). The frozen bases barely move with a global T; their per-type temperatures diverge because their noul answers are near chance and a flat distribution minimises NLL there.

**ECE and NLL, SFT against SFT plus the global T and the per-split oracle** (gold-dependent questions; `runs/sft_<size>/metrics.json`, `runs/sft_<size>_temp/metrics.json` and `runs/sft_<size>_temp/metrics_oracle.json`, key `splits.<split>.overall.ece` and `.nll`; oracle T from `metrics_oracle.json:temperatures.<split>`):

| split | sft_06b ECE / NLL | + global T | + oracle T (T) | sft_17b ECE / NLL | + global T | + oracle T (T) |
|---|---|---|---|---|---|---|
| valid | .020 / .216 | .005 / .209 | .005 / .209 (1.27) | .019 / .201 | .006 / .192 | .006 / .192 (1.32) |
| test_indomain | .013 / .123 | .005 / .119 | .005 / .119 (1.20) | .015 / .133 | .003 / .126 | .003 / .126 (1.32) |
| test_sst5 | .030 / .541 | .012 / .533 | .012 / .533 (1.22) | .025 / .508 | .020 / .500 | .015 / .499 (1.24) |
| test_unseen_intents | .068 / .371 | .054 / .320 | .021 / .295 (1.71) | .070 / .365 | .055 / .304 | .023 / .276 (1.83) |
| test_agnews | .164 / 1.080 | .146 / .886 | .029 / .628 (2.70) | .113 / .773 | .097 / .616 | .032 / .469 (2.48) |
| test_emotion | .207 / 1.249 | .166 / 1.087 | .032 / .915 (2.56) | .217 / 1.352 | .175 / 1.122 | .056 / .899 (2.88) |
| test_banking77 | .067 / .545 | .037 / .496 | .032 / .489 (1.43) | .087 / .633 | .058 / .534 | .034 / .502 (1.71) |
| test_yelp | .169 / 1.110 | .137 / 1.080 | .120 / 1.078 (1.36) | .218 / 1.144 | .169 / 1.042 | .101 / 1.006 (1.79) |

On `train`, which the models were fitted to, the global T makes ECE worse (.009 to .026 and .004 to .023), as expected when a model is underconfident on its own training data; `train` is not a target. Per question type the picture is the same; the full table is printed by `scripts/compare_calibration.py`.

**Coverage.** A temperature above 1 lowers every confidence, so a threshold keeps fewer answers, each more accurate (`<run>:splits.<split>.<type>.coverage`, response confidence field). On test_indomain noul at threshold 0.95, sft_06b keeps 81.5 percent at .987 and sft_06b plus T keeps 76.1 percent at .993; on test_unseen_intents choice at 0.9, 78.3 percent at .954 against 69.9 percent at .974; on test_emotion choice at 0.9, 26.8 percent at .791 against 12.4 percent at .863 (1.7B: 29.5 percent at .786 against 14.6 percent at .884). Scaling trades coverage for accuracy at a fixed threshold; it moves the threshold at which a given accuracy is reached, and adds no information about which answers are right.

**What a global temperature fixes, and what it does not.** Fitted on in-domain `valid` alone, one number brings in-domain ECE to .003 to .005 and SST-5 to .012 to .020, equal or close to the per-split oracle there, and it removes 11 to 45 percent of the ECE on unseen intents and unseen schemas (about a fifth on most; AG News .164 to .146, emotion .207 to .166, Yelp .218 to .169 at 1.7B), because the SFT models are overconfident everywhere and a temperature above 1 points the right way. It does not fix the unseen schemas: the temperatures they need range from 1.36 to 2.88, up to 2.2 times the 1.27 and 1.32 fitted on `valid`, and differ per schema, so no single number serves them all; even the oracle leaves Yelp at .10 to .12, a miscalibration in the shape of the distribution over an ordinal scale that no temperature can remove; and on emotion and Yelp at both sizes, and on AG News at 0.6B, the SFT model with the global T (.137 to .175) is still less calibrated than the frozen base it started from (.037 to .090, `runs/base_*/metrics.json`), with AG News at 1.7B level (.097 against .094). The gap RLCD has to close is therefore specific: calibration on schemas the model was not trained on, without knowing the schema's temperature, at in-domain accuracy. Concretely, an RLCD arm has to bring unseen-schema ECE below what SFT plus the global T reaches (.146, .166, .037 and .137 at 0.6B; .097, .175, .058 and .169 at 1.7B), towards the oracle, while keeping in-domain ECE near .005 and accuracy at the SFT level.

## 2. RLCD (in progress)

**Negative result: a proper score as a REINFORCE reward.** The first RLCD run used the Brier score of the sampled action, 1 - (r_a - p_a)^2 with p detached, as a policy-gradient reward with a group-mean baseline (0.6B, seed 0, 500 steps from `runs/sft_06b`). It degraded the model instead of calibrating it (`runs/sft_06b/metrics.json` against `runs/rlcd_06b_brier_s0/metrics.json`, key `splits.<split>.overall.accuracy` and `.ece`, `.score.accuracy` for score; the evaluated adapter is the best by validation NLL, step 100):

| split | sft_06b acc / ECE | rlcd_06b_brier_s0 acc / ECE |
|---|---|---|
| valid | .916 / .020 | .844 / .054 |
| test_indomain | .956 / .013 | .948 / .025 |
| test_sst5 | .773 / .030 | .480 / .178 |
| test_unseen_intents | .892 / .068 | .901 / .062 |
| test_agnews | .788 / .164 | .777 / .078 |
| test_emotion | .628 / .207 | .583 / .150 |
| test_banking77 | .851 / .067 | .864 / .038 |
| test_yelp | .505 / .169 | .044 / .287 |

Score questions collapsed (score accuracy .636 to .111 on test_sst5, .505 to .044 on test_yelp). The lower ECE on AG News, emotion and Banking77 comes with a model that has moved away from its training signal, so it is not evidence for RLCD. Over training, validation NLL rose from .219 at step 0 to .872 at step 500 with KL to the SFT policy at .46, while the mean training reward stayed at about .95 (`runs/rlcd_06b_brier_s0/training_log.jsonl`, `train_summary.json`).

**Why.** A sampled wrong action a has p_a at most 1 - p_gold, so its Brier reward 1 - p_a^2 is never below the gold reward 1 - (1 - p_gold)^2; the log score behaves the same way. The group-mean advantage therefore pushes probability away from gold whenever K > 2 and is exactly zero for K = 2 (every noul). A proper score grades a stated probability; as the reward for an action it measures whether p_a forecast r_a well, which a rarely chosen wrong option always did. Decision 52 records the proof. A CPU simulation of the sampler on the SFT model's train probabilities (G 4, epsilon 0.1, seed 0, 42422 gold-dependent questions) confirms it: under the Brier reward 17.2 percent of gold samples get a negative advantage and 60.7 percent of wrong samples a positive one, and the update lowers the gold logit in 24.0 percent of questions and raises it in 1.8 percent, against 0 and 34.9 percent under the outcome reward; in every group that sampled both gold and a wrong option, gold's mean advantage was negative under brier and log, and every noul group had equal rewards (`runs/advantage_simulation_06b/metrics.json`, keys `rewards.<reward>.overall` and `.mixed_groups`, written by `scripts/simulate_advantages.py`). The same file's `sign_check` runs `question_loss` on a three-option example against a hand computation: the code's signs are correct, the reward is what is wrong.

**Corrected design** (decision 52). The arms are `sft_cont` (control), `outcome` and `outcome_minus_p` (REINFORCE), and `direct_brier` and `direct_log`, which minimise the sampled action's Brier or log score with p_a differentiable, so the update is a gradient of a proper score of p itself. The REINFORCE `brier` and `log` arms remain in the code behind `rlcd.reinforce_proper_score: true`, off by default, only to reproduce this result. Stage 1 runs the five arms on 0.6B with seed 0 (docs/KAGGLE.md section 10); section 3 reports it.

## 3. RLCD stage 1: five arms on 0.6B, seed 0 (interim)

**Status: interim.** This is one seed at one size. Stage 2, seeds 1 and 2 at 0.6B and seed 0 at 1.7B (decision 53, docs/KAGGLE.md section 10), and stage 3 are pending. This section will be revised with them and does not yet state the v2 conclusion.

**Setup.**
- Five arms, each 500 steps from `runs/sft_06b` with the same settings: lr 5e-5, beta 0.02, G 4, epsilon 0.1, and the same records and schedule.
- The arms: `sft_cont` (the control: more cross-entropy on the gold labels), `outcome` and `outcome_minus_p` (REINFORCE), and `direct_brier` and `direct_log` (pathwise proper scores, decision 52).
- Each run is evaluated on its best adapter by validation NLL, on all nine splits.

**Source of the numbers.** Every number below is from `runs/rlcd_stage1_06b/metrics.json`, written by `scripts/compare_rlcd.py --size 06b --seeds 0`, at these keys:
- per split: `splits.<split>.overall.{sft,sft_temp,arms.<arm>.mean}`;
- unseen-schema mean: `unseen_schemas`;
- temperature ablation: `temperature_ablation`;
- validation curves: `validation`.

**How the intervals are computed.** They are 95 percent paired bootstrap intervals over records (1000 resamples, the same draws for every run). Accuracy differences are against `sft_06b`; ECE differences are against `sft_06b_temp`. With one seed, they cover record sampling only; stage 2a adds the variation between seeds.

**Unseen schemas.** ECE on gold-dependent questions; test_indomain is shown for accuracy:

| run | test_indomain acc / ECE | test_agnews | test_emotion | test_banking77 | test_yelp | unseen mean ECE | Δ vs sft_06b_temp [95% CI] |
|---|---|---|---|---|---|---|---|
| sft_06b | .956 / .013 | .164 | .207 | .067 | .169 | .152 | |
| sft_06b_temp | .956 / .005 | .146 | .166 | .037 | .137 | .121 | |
| sft_cont | .958 / .016 | .169 | .244 | .088 | .241 | .185 | +.064 [+.057, +.073] |
| outcome | .958 / .038 | .203 | .333 | .119 | .396 | .262 | +.141 [+.131, +.149] |
| outcome_minus_p | .959 / .014 | .172 | .234 | .088 | .196 | .173 | +.051 [+.043, +.058] |
| direct_brier | .957 / .011 | .172 | .236 | .082 | .201 | .173 | +.051 [+.044, +.059] |
| direct_log | .958 / .016 | .169 | .239 | .085 | .233 | .181 | +.060 [+.053, +.068] |

- **Unseen-schema ECE:** no arm matches `sft_06b_temp` on any unseen schema, and every interval lies above 0. Every arm is also less calibrated there than `sft_06b` itself.
- **In-domain accuracy held:** test_indomain accuracy moved by +.001 to +.003 against `sft_06b`. Only outcome_minus_p's interval, [+.001, +.004], excludes 0.
- **In-domain ECE:** every arm's test_indomain ECE is above `sft_06b_temp`'s .005, from +.005 (direct_brier) to +.032 (outcome), with intervals above 0.
- **Other held-out splits:** accuracy fell on several; the full table is in the metrics file.
- **The broken REINFORCE `brier` arm** (section 2) is in the file for completeness and left out here.

**What training did to confidence.** On the fixed valid subset, from step 0 to step 500 (`validation.<arm>.0`):

| arm | Σp² (expected p of the chosen action) | KL to SFT | train ECE | valid NLL |
|---|---|---|---|---|
| (SFT, step 0) | .912 | 0 | .009 | .219 |
| sft_cont | .929 | .015 | .004 | .232 |
| outcome | .985 | .076 | .057 | .627 |
| outcome_minus_p | .924 | .014 | .005 | .223 |
| direct_brier | .924 | .010 | .005 | .225 |
| direct_log | .930 | .016 | .003 | .234 |

**Temperature ablation.** Each arm was given its own temperature, fitted on valid with `scripts/calibrate.py`, as `runs/rlcd_06b_<arm>_s0_temp/`. The question is whether that temperature brings the arm back to `sft_06b_temp` (T 1.266):

| arm | T | unseen mean ECE before → after | after − sft_06b_temp [95% CI] | level |
|---|---|---|---|---|
| sft_cont | 1.432 | .185 → .132 | +.011 [+.006, +.020] | above |
| outcome | 2.851 | .262 → .156 | +.035 [+.026, +.045] | above |
| outcome_minus_p | 1.384 | .173 → .129 | +.007 [+.001, +.014] | above |
| direct_brier | 1.351 | .173 → .127 | +.006 [+.001, +.014] | above |
| direct_log | 1.394 | .181 → .135 | +.013 [+.007, +.021] | above |

- **Temperature is not enough:** every arm needs a higher temperature than SFT, and even with its own it stays above `sft_06b_temp`, with every interval excluding 0 (`temperature_ablation.every_arm_back_to_sft_temp_level` is false).
- **Where the gap remains:** it is smallest for direct_brier and outcome_minus_p (+.006 and +.007) and sits mostly on emotion, where every arm stays .014 to .040 above. It also remains on Yelp for sft_cont (+.023), outcome (+.079) and direct_log (+.034).
- **What that means:** the extra training did more than scale the SFT logits. It also changed the shape of the distributions off-distribution, in a way one temperature fitted in-domain does not undo.

**Mechanism.**
- **Every arm sharpens, the control included.** Every arm trains on fully labelled in-domain data whose gold answers are deterministic, and every arm sharpens: Σp² rises and train ECE falls. `sft_cont` does so as much as the pathwise arms (Σp² .929 against .924 and .930), so what the pathwise arms do is what more cross-entropy on the same records does.
- **Bandit feedback cannot add anything here.** It reveals only whether the sampled option was gold, which is strictly less information per question than the gold label cross-entropy is given. On deterministic gold, no bandit objective can do what cross-entropy cannot.
- **Outcome-only reward destroys calibration.** With nothing that penalises confidence, REINFORCE on the 0/1 outcome drives the policy towards certainty: Σp² .985, valid NLL .219 to .627, unseen mean ECE .262. It needs T 2.85 to come back even partly.
- **Off-distribution overconfidence needs softening.** The unseen schemas need temperatures of 1.4 to 2.7 (section 1) on top of the model's in-domain fit. Every arm moves the other way, and even an arm's own temperature leaves it above `sft_06b_temp`.

**Pending.**
- **Stage 2a** tests whether these differences hold across seeds; the range across seeds will show whether the ordering between arms is stable.
- **Stage 2b** tests whether they hold at 1.7B.
- **Stage 3** is specified in a separate task.
