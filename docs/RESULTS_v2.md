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

## 2. RLCD: the first run

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

## 3. RLCD stages 1 and 2a: five arms on 0.6B, seeds 0, 1 and 2

**Status: final.** Stage 1 ran the five arms at seed 0; stage 2a added seeds 1 and 2 (decision 53, docs/KAGGLE.md section 10). This section reports all three seeds at 0.6B and states the v2 conclusion. Stage 2b (the five arms at seed 0 on 1.7B) was not run (decision 58: the 0.6B result is consistent across seeds, and v3 made the 1.7B repeat low priority); stage 3 (the stochastic-outcome environment, task 2.5) was cancelled (decision 56).

**Setup.**
- Five arms, each 500 steps from `runs/sft_06b` with the same settings: lr 5e-5, beta 0.02, G 4, epsilon 0.1, and the same records and schedule. The seed sets the data order, the sampled actions and the 1000-record validation subset.
- The arms: `sft_cont` (the control: more cross-entropy on the gold labels), `outcome` and `outcome_minus_p` (REINFORCE), and `direct_brier` and `direct_log` (pathwise proper scores, decision 52).
- Each run is evaluated on its best adapter by NLL on its validation subset, on all nine splits.
- The fifteen runs all record commit f30800d (seed 0, all arms but direct_brier), 9c160d2 (direct_brier, seed 0) or 740371b (seeds 1 and 2), not dirty, the same data hashes as `runs/sft_06b`, 500 of 500 steps, no fp32 fallback, and the SFT adapter sha256 5d2681f3... in config.yaml, train_summary.json and metrics.json (`runs/rlcd_06b_<arm>_s<seed>/`).

**Source of the numbers.** Every number below is from `runs/rlcd_stage2a_06b/metrics.json`, written by `scripts/compare_rlcd.py --size 06b --seeds 0 1 2 --out runs/rlcd_stage2a_06b` on a clean tree, unless another file is named:
- per split: `splits.<split>.overall.{sft,sft_temp,arms.<arm>.mean,arms.<arm>.by_seed}`;
- unseen-schema mean: `unseen_schemas.arms.<arm>` (`mean_ece`, `mean_ece_range`, `mean_ece_by_seed`);
- temperature ablation: `temperature_ablation`, with the per-seed values after temperature from `runs/rlcd_06b_<arm>_s<seed>_temp/metrics.json:splits.<split>.overall.ece`;
- validation curves: `validation.<arm>.<seed>`.

The seed 0 numbers alone are in `runs/rlcd_stage1_06b/metrics.json` (the interim stage 1 table).

**How the intervals are computed.** A cell for an arm is the mean over its three seeds, with the range (min, max) where shown. Its interval is a 95 percent paired bootstrap over records (1000 resamples, the same draws for every run), each draw averaging the three seeds' differences under that record resample. The interval therefore covers record sampling for these three seeds; the range shows the variation between seeds. Accuracy differences are against `sft_06b`; ECE differences are against `sft_06b_temp`.

**Unseen schemas.** ECE on gold-dependent questions, seed means; test_indomain is shown for accuracy:

| run | test_indomain acc / ECE | test_agnews | test_emotion | test_banking77 | test_yelp | unseen mean ECE (seeds 0, 1, 2) | Δ vs sft_06b_temp [95% CI] |
|---|---|---|---|---|---|---|---|
| sft_06b | .956 / .013 | .164 | .207 | .067 | .169 | .152 | |
| sft_06b_temp | .956 / .005 | .146 | .166 | .037 | .137 | .121 | |
| sft_cont | .957 / .017 | .172 | .236 | .087 | .189 | .171 (.185, .172, .156) | +.050 [+.044, +.056] |
| outcome | .953 / .039 | .196 | .326 | .118 | .354 | .248 (.262, .244, .239) | +.127 [+.118, +.135] |
| outcome_minus_p | .957 / .014 | .165 | .211 | .072 | .164 | .153 (.173, .155, .131) | +.031 [+.026, +.038] |
| direct_brier | .957 / .012 | .168 | .222 | .076 | .169 | .159 (.173, .165, .139) | +.037 [+.032, +.043] |
| direct_log | .958 / .018 | .172 | .234 | .083 | .191 | .170 (.181, .172, .156) | +.048 [+.043, +.055] |

- **Unseen-schema ECE:** at every seed, every arm's four-schema mean is above `sft_06b_temp`'s .121; the lowest single run is outcome_minus_p at seed 2 with .131. Every seed-mean interval lies above 0. The stage 1 direction holds at all three seeds.
- **The size is smaller than stage 1 showed:** seed 0 was the least calibrated seed for every arm, so the seed-mean differences (+.031 to +.127) are .012 to .020 below the seed 0 differences (+.051 to +.141, `runs/rlcd_stage1_06b/metrics.json:unseen_schemas`).
- **One run beats `sft_06b_temp` on one schema:** outcome_minus_p at seed 2 on Yelp, .098 against .137, difference -.040 [-.061, -.014] (`splits.test_yelp.overall.arms.outcome_minus_p.by_seed.2`). direct_brier at seed 2 is also below on Yelp (-.017 [-.031, +.003]), with an interval that includes 0. No other run is below `sft_06b_temp` on any unseen schema.
- **Raw SFT:** stage 1 found every arm less calibrated than `sft_06b` itself (.152). Across seeds that holds for sft_cont, outcome and direct_log at every seed, but not for outcome_minus_p (.131 at seed 2) and direct_brier (.139 at seed 2). Their seed means (.153 and .159) are at or above .152.
- **In-domain accuracy:** test_indomain accuracy moved by -.003 to +.001 against `sft_06b`. Only outcome's difference excludes 0: -.003 [-.005, -.001]. Stage 1's small significant gain for outcome_minus_p (+.003 at seed 0) does not survive the seed mean (+.000 [-.001, +.002]).
- **In-domain ECE:** every arm's test_indomain ECE is above `sft_06b_temp`'s .005, from +.007 (direct_brier) to +.034 (outcome), with intervals above 0.
- **Other held-out splits:** accuracy fell against `sft_06b` on test_sst5 for every arm (-.005 to -.009) and on test_unseen_intents for every arm but outcome (-.006 to -.009), all with intervals below 0 (`splits.<split>.overall.arms.<arm>.mean.accuracy_minus_sft_ci`).
- **The broken REINFORCE `brier` arm** (section 2) has seed 0 only; it is in the file for completeness and left out here.

**Seed variation against record intervals.** The seed-mean intervals are 0.011 to 0.017 wide. The range between seeds of the four-schema mean is .023 (outcome) to .041 (outcome_minus_p), 1.4 to 3.5 times the width of its interval, and most of it sits on Yelp: sft_cont's Yelp ECE is .241, .188 and .139 at seeds 0, 1 and 2. A record-level interval from one seed therefore understates how far a rerun can move. Two consequences:
- **The arm ranking is only partly stable.** outcome is separated from the others at every seed (.239 to .262 against at most .185). The ranges of the other four overlap. Compared within the same seed, outcome_minus_p is below sft_cont at every seed (-.013, -.018, -.024) and direct_brier too (-.013, -.008, -.017), while direct_log is level with sft_cont (-.004, .000, +.001). These are point differences with no interval.
- **The seed effect follows the selected step.** At seed 2 all five arms selected step 100, the first trained checkpoint; at seed 0 they selected steps 200 to 400; at seed 1, step 100 (outcome, outcome_minus_p), 400 (sft_cont) or 500 (direct_brier, direct_log) (`validation.<arm>.<seed>.best_step`). Within each arm, the runs selected at step 100 have the lowest Σp² on full valid (`validation.<arm>.<seed>.full_valid_best`) and the lowest unseen-schema ECE. The selection uses NLL on a validation subset drawn per seed (`scripts/train_rlcd.py`, `random.Random(f"{seed}:valid_subset")`), so part of what differs between seeds is which checkpoint the rule picks, not only the training trajectory.

**What training did to confidence.** Validation of the last adapter (step 500) on the full `valid` split, the same records at every seed (`validation.<arm>.<seed>.full_valid_last`), as the range over the three seeds:

| arm | Σp² (expected p of the chosen action) | KL to SFT | valid NLL | valid ECE |
|---|---|---|---|---|
| sft_cont | .924 to .928 | .0096 to .0117 | .228 to .234 | .027 to .033 |
| outcome | .983 to .988 | .070 to .076 | .585 to .677 | .075 to .077 |
| outcome_minus_p | .918 to .922 | .0086 to .0101 | .224 to .226 | .024 to .026 |
| direct_brier | .918 to .922 | .0078 to .0096 | .223 to .226 | .023 to .028 |
| direct_log | .923 to .928 | .0091 to .0125 | .227 to .234 | .026 to .032 |

On the validation subset, from step 0 to step 500 (`validation.<arm>.<seed>.step_0` and `.last`), Σp² rose at every seed for every arm: by .013 to .017 for sft_cont, .067 to .076 for outcome, .007 to .012 for outcome_minus_p, .006 to .011 for direct_brier and .012 to .018 for direct_log, with final KL to SFT of .010 to .015, .066 to .076, .009 to .014, .008 to .010 and .010 to .016. The subset differs per seed (step-0 Σp² .912, .914 and .916), so the full-valid table is the one to compare across seeds. The ordering by sharpening, outcome far above, then sft_cont and direct_log, then outcome_minus_p and direct_brier, is the same at every seed and is the same ordering as unseen-schema ECE.

**Temperature ablation.** Each run was given its own temperature, fitted on valid with `scripts/calibrate.py`, as `runs/rlcd_06b_<arm>_s<seed>_temp/`. The question is whether that temperature brings the arm back to `sft_06b_temp` (T 1.266):

| arm | T (seeds 0, 1, 2) | unseen mean ECE before → after (seeds after) | after − sft_06b_temp [95% CI] | level |
|---|---|---|---|---|
| sft_cont | 1.432, 1.426, 1.315 | .171 → .124 (.132, .124, .116) | +.002 [-.001, +.009] | level |
| outcome | 2.851, 2.528, 2.525 | .248 → .146 (.156, .140, .143) | +.025 [+.018, +.034] | above |
| outcome_minus_p | 1.384, 1.287, 1.165 | .153 → .120 (.129, .119, .112) | -.002 [-.006, +.004] | level |
| direct_brier | 1.351, 1.361, 1.175 | .159 → .122 (.127, .123, .116) | +.001 [-.004, +.007] | level |
| direct_log | 1.394, 1.430, 1.289 | .170 → .126 (.135, .121, .123) | +.005 [+.001, +.011] | above |

- **This changes the stage 1 reading.** At seed 0 every arm stayed above `sft_06b_temp` after its own temperature. Over three seeds, sft_cont, outcome_minus_p and direct_brier come back to its level (intervals containing 0); outcome and direct_log stay above (`temperature_ablation.every_arm_back_to_sft_temp_level` is false). For the three arms that come back, the extra training is mostly a sharpening that one in-domain temperature undoes; the stage 1 claim that it changed the shape of the distributions off-distribution rests on seed 0.
- **Where a gap remains:** on emotion, sft_cont (+.016), direct_brier (+.011) and direct_log (+.015) stay above `sft_06b_temp` after temperature, with intervals above 0; outcome_minus_p does not (+.004 [-.003, +.012]). outcome is above on emotion, Banking77 and Yelp, and below on AG News (-.008 [-.016, -.002]). On Yelp, direct_brier after temperature is below `sft_06b_temp` (-.017 [-.028, -.001]) (`temperature_ablation.arms.<arm>.splits`).
- **Being level is not a win.** An arm with its own temperature is a method of the same kind as `sft_06b_temp`: one T fitted on in-domain valid. The best of them, outcome_minus_p at -.002 [-.006, +.004], is level with it, and the control sft_cont reaches the same level (+.002). Nothing here is attributable to bandit feedback.

**Mechanism.**
- **Every arm sharpens, the control included.** Every arm trains on fully labelled in-domain data whose gold answers are deterministic, and every arm sharpens at every seed: Σp² rises and in-domain valid NLL rises from step 0. `sft_cont` sharpens as much as direct_log and more than the two arms with the lowest unseen ECE, so what the pathwise arms do is at most what more cross-entropy on the same records does.
- **Bandit feedback cannot add anything here.** It reveals only whether the sampled option was gold, which is strictly less information per question than the gold label cross-entropy is given. On deterministic gold, no bandit objective can do what cross-entropy cannot.
- **Outcome-only reward destroys calibration.** With nothing that penalises confidence, REINFORCE on the 0/1 outcome drives the policy towards certainty at every seed: Σp² .983 to .988 on full valid, valid NLL .585 to .677, unseen mean ECE .239 to .262, and in-domain accuracy -.003. It needs T 2.5 to 2.9 and still stays above `sft_06b_temp`.
- **Less sharpening, better off-distribution.** outcome_minus_p and direct_brier sharpen least (Σp² .918 to .922, KL .008 to .010), have the lowest unseen ECE, and need the smallest temperatures (1.16 to 1.38). The unseen schemas need softening (temperatures of 1.4 to 2.7, section 1); any training that sharpens in-domain moves them the wrong way, and how far it moves them tracks how much it sharpens.

**Stage 2a answer.** Across three seeds at 0.6B, no RLCD arm matches `sft_06b_temp` on unseen-schema ECE without a temperature of its own, the sign of every arm's difference is the same at every seed, and the size is about .015 smaller than seed 0 alone showed. With its own temperature, no arm is below `sft_06b_temp`, and the control does as well as the best RLCD arm.

**Not run.**
- **Stage 2b**, the five arms at seed 0 on 1.7B, was not run (decision 58), so the v2 conclusion is for 0.6B only.
- **Stage 3** (task 2.5, decision 54) was cancelled by decision 56; section 4 keeps its method and SFT baseline. The setting in which bandit feedback can carry information that cross-entropy on gold cannot is now tested in v3, on real deployment feedback (docs/V3_DESIGN.md).

## 4. RLCD stage 3: a stochastic-outcome environment (method only; cancelled)

**Status: cancelled by decision 56 before any training run.** Phase 3 (docs/V3_DESIGN.md) tests RLCD on real deployment feedback instead. This section stays as the record of the method and the SFT baseline. Sessions 3-1 and 3-2 (docs/KAGGLE.md section 11) will not run. The SFT numbers below are from `runs/sft_06b_env/metrics.json` (commit b49153b, not dirty, written by `scripts/evaluate_env.py runs/sft_06b` from `runs/sft_06b/results.jsonl.gz` on CPU), key `variants.<variant>.splits.<split>`. Decision 54 records the design.

**Why a new setting.** In stages 1 and 2a the outcome of a sampled action is whether it is gold. That is strictly less information than the gold label cross-entropy already uses, and every arm, the control included, only sharpened (section 3). In that setting RLCD cannot do anything that SFT cannot. Stage 3 changes the outcomes so that the best-calibrated policy is not the one-hot gold answer and is known exactly.

**The environment** (`jevmark/environment.py`). The data stay frozen (v1.3). A question with K options (K = 2 for a noul) gets a noise rate

  eta(K) = min(0.40, 0.05 + 0.03 (K - 2)),

so eta is .05 for a noul, .08 at K 3, .32 at K 11 and .40 from K 12 on. The target distribution theta puts 1 - eta on gold and eta / (K - 1) on each other option. On every visit to a question the environment draws one accepted answer from theta with its own generator, seeded from the run seed and saved with the resume state. All G sampled actions of that visit are scored against the same accepted answer, so an action a is revealed as 1 with probability theta_a. One accepted answer per visit matches a bandit setting with one environment response per decision. It changes the correlation within a group of samples, not any action's outcome rate. theta depends on K, so no single temperature applied to a sharp policy can reach it.

**Training.** `scripts/train_rlcd.py --env noisy`, from `runs/sft_06b`, 1000 steps, G 4, epsilon 0.1, lr 5e-5, with beta 0 and the step count taken from the config's `noisy` section. Direct overrides of `rlcd.beta` or `training.steps` are refused. `adapter/` is the step with the lowest cross-entropy against theta on the validation subset.

**Why beta is 0.** The KL term pulls the policy towards the SFT policy, which is nearly one-hot on gold. Under noise that pull points away from theta: the minimiser of a proper score plus beta KL(p || p_ref) is a blend of theta and p_ref, not theta. The pathwise proper scores are bounded objectives whose minimum is theta itself, so they need no trust region. The runs take 1000 steps rather than 500 because a noisy outcome carries less signal than a gold label.

**Arms and seeds planned** (0.6B, `MAX_HOURS = 2.5`, one commit for both sessions):

| session | arms | seeds | runs |
|---|---|---|---|
| 3-1 | `direct_brier` | 0, 1, 2 | 3 |
| 3-2 | `direct_log`, `outcome_minus_p`, `outcome` | 0 | 3 |

`sft_cont` is refused, because cross-entropy on gold ignores the outcomes. The REINFORCE `brier` and `log` arms are refused as known broken (decision 52). `direct_brier` gets three seeds because it is the arm the prediction is about. The estimate is about 2.1 hours per run (81 minutes of training, 44 of evaluation) and 14 GPU hours in all (docs/KAGGLE.md section 11).

**Post-hoc baselines, computed from SFT.**
- **Global T, fitted from bandit feedback.** One temperature fitted on `valid` from the SFT policy's own bandit outcomes in the environment (G 4, epsilon 0.1, seed 0), minimising the binary log loss of p_T(a) against the revealed outcome. It sees the same kind of feedback RLCD does, so it is the fair post-hoc competitor. On `sft_06b` it is T 2.849 (`variants.global_T.temperature`).
- **Per-K oracle T.** One temperature per K, fitted on each test split's own theta by minimising cross-entropy against theta. It reads the target on the split it is scored on, so it is a bound, not a method. Because it minimises cross-entropy and not the gap, it can leave a larger gap than the global T (Banking77, below).

**Metrics against theta.** Every metric is an expectation over theta rather than over sampled outcomes, on gold-dependent questions (`evaluate_env.py`):
- expected Brier, E over theta of the multi-class Brier score, whose minimum over p is 1 - Σ theta² (`expected_brier_min`);
- cross-entropy against theta, -Σ theta_k log p_k, whose minimum is the entropy of theta, and the KL to theta;
- accuracy against gold;
- ECE against outcomes sampled once per split from theta (`ece_sampled`), the same outcomes for every variant;
- the calibration gap, redefined for this setting: for each K, |mean top-1 probability - mean expected hit rate of the top-1 option|, where the hit rate is theta of the predicted option, then the count-weighted mean over K (`calibration_gap`, per K in `by_k`). The hit rate of the predicted option replaces 1 - eta because a model that is sometimes wrong should report less than 1 - eta on the questions it gets wrong. Mean p(gold) against 1 - eta is kept as `gap_gold`, but it grows when such a model is softened, so it is not the headline.

theta itself scores a gap of 0 and reaches both minima.

**SFT baseline.** Calibration gap, raw / with the global T / with the per-K oracle, and cross-entropy against theta (raw / global T / oracle / theta):

| split | K present | gap raw | + global T | + per-K oracle | cross-entropy raw / global T / oracle / theta |
|---|---|---|---|---|---|
| test_indomain | 2 to 14 | .152 | .027 | .022 | 1.353 / .709 / .706 / .618 |
| test_unseen_intents | 2 to 14 | .192 | .022 | .007 | 1.525 / .785 / .781 / .622 |
| test_sst5 | 2, 3, 5 | .084 | .093 | .025 | .841 / .745 / .721 / .375 |
| test_agnews | 4 | .242 | .069 | .003 | 1.769 / .871 / .851 / .467 |
| test_emotion | 2, 6 | .256 | .029 | .011 | 1.571 / 1.030 / 1.022 / .464 |
| test_banking77 | 10 | .308 | .022 | .032 | 2.653 / 1.479 / 1.479 / 1.239 |
| test_yelp | 5 | .188 | .061 | .027 | 1.415 / 1.298 / 1.267 / .599 |

- **Unseen schemas.** The raw gap is .188 to .308. The global T brings it to .022 to .069 (four-schema mean .045), and the per-K oracle to .003 to .032 (mean .018).
- **The margin left for `direct_brier`.** Between the global T and the oracle there is .066 on AG News, .034 on Yelp and .018 on emotion. On Banking77 there is none: the oracle's gap is .010 above the global T's. Across the four schemas the global T is about .02 to .05 above zero on three and .069 on AG News, which is the most any method can gain there. A stage 3 arm that beats the global T but not the oracle has only these amounts to win.
- **Where a global T cannot fit every K.** The unseen schemas each have one or two values of K, so one temperature can come close on each of them. The in-domain splits mix K from 2 to 14. There the oracle temperatures rise with K, from 2.51 at K 2 to 3.20 at K 13 on test_indomain (`variants.oracle_T_by_K.splits.test_indomain.temperatures`), and the global T of 2.849 leaves a signed gap that runs from -.036 at K 2 (underconfident) to +.034 at K 13 (overconfident) (`variants.global_T.splits.test_indomain.by_k.<K>.gap`). test_sst5 (K 2, 3 and 5) is the extreme case: the global T raises its gap from .084 to .093, while the oracle brings it to .025. The count-weighted gap hides most of this on test_indomain (.027 against .022), because 5900 of its 11800 questions are nouls at K 2.
- **Accuracy and theta.** A temperature changes no prediction, so accuracy against gold is the SFT value in every variant (test_indomain .956). theta's cross-entropy is the floor: on test_indomain the global T is .091 above it and the oracle .088.

**Falsifiable prediction** (decision 54, stated before any stage 3 run):
1. A global temperature cannot fix calibration that depends on K. So `sft_06b` plus the global T keeps a calibration gap that the per-K oracle reduces. The baseline above already shows this on the mixed-K splits (test_sst5 .093 against .025, test_unseen_intents .022 against .007) and on three of the four unseen schemas, with Banking77 as the exception.
2. `direct_brier` should approach theta: lower cross-entropy against theta and a lower calibration gap than `sft_06b` plus the global T, at SFT accuracy.
3. If `direct_brier` does not beat `sft_06b` plus the per-K oracle temperature, RLCD has no value independent of post-hoc scaling in this setting, and the report will say so.

**How the result will be reported.** `scripts/compare_env.py --size 06b` writes `runs/rlcd_stage3_06b/metrics.json`. Its columns are `sft_06b` raw, with the global T and with the per-K oracle, and each noisy arm as evaluated, as the seed mean with the range across seeds. For every column it gives the differences in expected Brier, cross-entropy and calibration gap against the global T and against the oracle, and the accuracy difference against `sft_06b`, with a paired bootstrap by record (1000 resamples, the same draws for every column; for an arm, each draw averages its seeds' differences, as in section 3).
