# Method: Chosen-Action Proper Scoring (CAPS)

This note states the method behind the v3 results: learning from logged feedback on a decision model's own actions. It was motivated by Reinforcement Learning for Calibrated Decisions (RLCD), a community reconstruction of the training idea behind TypeSafe AI's Jev. It is not that method: section 3 shows why the reconstruction's reward fails, and CAPS replaces it. Each empirical statement cites a results section and the metrics file behind it. `S:` is `runs/v3_stage_06b/metrics.json` and `N:` is `runs/v3_noise_sweep_06b/metrics.json`.

## 1. Setting

A decision model maps a state x (a message, plain or JSON) and an option list O(x) = (o_1, ..., o_K) to a distribution p(. | x) over the options. It does this in one forward pass, reading the options as input. It is deployed on a new domain under a logging policy q(. | x). The deployment writes a log of interactions (x, O, a, q(a | x), r). Here a is the action taken, q(a | x) its propensity under the logging policy, and r in {0, 1} whether the action was right. Nothing else is recorded: no gold label, and no outcome for the options not taken.

The learner starts from a general decision model, here the supervised adapter of v1, and sees only the log. In the experiments, q = 0.9 p_0 + 0.1 / K, with p_0 the starting model (RESULTS_v3 section 1, `runs/v3_log_s0/metrics.json`).

## 2. Objective

CAPS minimises the expected proper score of the chosen action's probability under the logging distribution:

  L(theta) = E_x E_{a ~ q(. | x)} [ S(p_theta(a | x), r(x, a)) ]

S is a strictly proper score for a binary event. The default is the Brier score, S(p, r) = (r - p)^2; the alternative is the log score, S(p, r) = -(r log p + (1 - r) log(1 - p)). On a log, the empirical mean over interactions estimates L directly, with no weights. The gradient is pathwise, through p_a: for the Brier score it is -2 (r - p_a) dp_a / dtheta. A right action is pushed up and a wrong one down, in proportion to how badly p_a forecast r.

Accuracy improves as a consequence of this, not as the objective. Pushing a right logged action up and a wrong one down moves the argmax towards the right option, but the loss asks only for calibrated forecasts of r. A negative on an option other than the policy's top choice exists only because the logging policy is random. It samples from p rather than taking its argmax, and in the experiments it also mixes in a uniform draw with weight epsilon 0.1. On the main Banking77 log, 1514 of the 2593 negatives (.584) fall on an option other than the top. About .545 of those are attributable to the uniform term, the sum of (epsilon / K) / q(a) over them, and the rest to sampling p (RESULTS_v3 section 1 describes the log; `runs/v3_log_s0/metrics.json:negatives.on_other`, `:negatives.on_other_uniform_share`). So the information the log carries about those options depends on how much the logging policy explores. Options it never takes carry none (section 6).

**Why no propensity weight.** L is defined under q, so the log's own mean already estimates it, and there is nothing to correct. An importance weight 1 / q(a) would change the target to a uniform average over options, but the aim is not the value of some other policy. The aim is that p(a | x) be a calibrated forecast of r for the actions the deployed policy actually takes. For each (x, a) with q(a | x) > 0, the minimiser of the inner term is p(a | x) = P(r = 1 | x, a) whatever q is. The propensity changes only how the finite model trades errors between actions, and the trade the objective makes is the one under deployment.

**What a weight does in practice** (RESULTS_v3 section 11, `S:n_curve.<N>.off_policy`):
- The log score weighted by min(1 / q, 5) (ips_log) about equals the unweighted log score (direct_log): .906 against .907 accuracy at N 2000 and .930 against .933 at N 5000.
- Weighting positive-only SFT the same way (ips_positive) leaves it equal to positive-only SFT: +.002 [-.001, +.005] and +.000 [-.003, +.004] in accuracy.
- A correct logged action is usually the policy's own top choice. Its weight is then near one constant, which an adaptive optimiser largely removes (`runs/v3_06b_ips_positive_n5000_s0/train_summary.json:ips`).

## 3. Why a proper score is not a REINFORCE reward

**Proposition.** Take a question with K options, gold g and a model p. Use a proper score as a detached reward:
- Brier: R(a) = 1 - (r_a - p_a)^2;
- log score: R(a) = r_a log p_a + (1 - r_a) log(1 - p_a).

Sample a group of actions and form REINFORCE advantages against the group mean.

(i) For K > 2, every wrong action gets a reward at least gold's. Within a group, the lowest-probability wrong action sampled has the highest reward, so its advantage is non-negative, and positive unless every reward is equal. In any group that samples both gold and a wrong action, gold's advantage is non-positive.

(ii) For K = 2, every reward in every group is equal, so every advantage and the gradient are zero.

**Proof.** A wrong action w has r_w = 0, and p_w <= 1 - p_g because the probabilities sum to one.
- *Rewards.* Under Brier, R(w) = 1 - p_w^2 and R(g) = 1 - (1 - p_g)^2. Under the log score, R(w) = log(1 - p_w) and R(g) = log p_g. Both are decreasing in p_w, and at p_w = 1 - p_g they equal R(g). So R(w) >= R(g), with equality only when p_w = 1 - p_g.
- *Ordering in a group.* Among the sampled wrong actions, the one with the smallest p_w has the largest reward. It is then the maximum of the group, so it lies at or above the group mean. In a group holding both gold and a wrong action, gold's reward is the group minimum, so it lies at or below the mean.
- *K = 2.* The single wrong action has p_w = 1 - p_g exactly, so R(w) = R(g). Every sample in a group then has the same reward, and every advantage is zero.

Under (i), the update raises the probability of rarely chosen wrong options and lowers gold's. **Why:** the score grades p_a as a forecast of r_a, and a low probability on a wrong option is always a good forecast. ∎

**Measured** (RESULTS_v2 section 2):
- On full labels this reward degraded the model: test_yelp accuracy fell from .505 to .044, and validation NLL rose from .219 to .872 (`runs/rlcd_06b_brier_s0/metrics.json`, `training_log.jsonl`).
- A simulation of the sampler lowered the gold logit in 24.0 percent of questions and raised it in 1.8 percent (`runs/advantage_simulation_06b/metrics.json:rewards`).

CAPS avoids this by differentiating the score through p_a instead of using it as a reward.

## 4. Selection rule

The learner trains on the first 0.9 N interactions of the log. It keeps the checkpoint with the highest mean Bernoulli log-likelihood of the logged outcomes under p(logged action) on the last 0.1 N, which is the log score of the objective on held-out interactions. Gold is never read (RESULTS_v3 section 1).

**Necessary under noisy feedback** (RESULTS_v3 section 4):
- With outcomes flipped at rate 0.2, the selected checkpoint scores .907. Trained to the end of the budget, the same run falls to .813, below the zero-shot .851, while the clean run keeps .933 (`runs/v3_06b_direct_brier_n5000_s0_noisy{,_last}/metrics.json:splits.v3_banking77_test_full.overall.accuracy`).
- At flip rate 0.3 the final checkpoint reaches .734 (`N:by_rate.0.3.direct_brier.last.accuracy`).
- The learner memorises the single fixed flip draw of each interaction when it passes over the log many times. The feedback-only criterion stopped it before that, every time.

**What it costs:** granularity. Checkpoints are every 50 steps. At N 2000 the log-score arms peaked at the first candidate, so a finer grid could change their ranking. At N 500, 50 interactions select (RESULTS_v3 sections 9 and 11, `runs/<run>/train_summary.json:best_step`).

## 5. Relations

- **Temperature scaling.** A single T fitted post hoc to the logged outcomes is global and leaves every prediction unchanged. It is the cheap baseline any method must beat. It reaches the lowest or joint-lowest ECE at every N, .020 at N 5000, but keeps zero-shot accuracy, .851, while CAPS reaches .929 (RESULTS_v3 section 2, `S:n_curve.5000`).
- **Full-label SFT.** Cross-entropy on gold for every interaction is the upper bound, at the cost of one label per interaction. At N 5000, CAPS's seed mean is 1.5 points below it, -.015 [-.020, -.010], with ECE level (RESULTS_v3 section 3, `S:seeds.training_seed_mean_minus_full_sft_seed_mean`). At 1.7B it is .004 below, with an interval covering 0, on one seed (RESULTS_v3 section 12, `runs/v3_stage_17b/metrics.json:n_curve.5000.direct_brier_minus_full_sft`).
- **Positive-only SFT.** Cross-entropy on the confirmed actions, discarding the failures, is what a practitioner does without the method. CAPS leads it by 2.4 to 2.7 points of accuracy from N 2000 and by .052 to .060 ECE at every N (RESULTS_v3 section 2, `S:n_curve.<N>.direct_brier_minus_positive_sft`). Positive-only SFT sees only confirmed answers, so it grows confident on its mistakes.
- **REINFORCE-style objectives on logged data.** With the logged action as the single sample, reward r - p_a, a batch-mean baseline and a clipped importance weight, the result is level with CAPS on accuracy and slightly worse on NLL: -.001 accuracy at both N, +.027 and +.019 NLL with intervals above 0 (RESULTS_v3 section 11, `S:n_curve.<N>.off_policy.outcome_minus_p_log.minus_direct_brier`). A proper score as the reward fails outright (section 3).

## 6. What the objective leaves open

- **Never-logged options.** The loss touches only p_a of the logged action. Options the policy never takes on a state move only through the normalisation of p, so their probabilities are unconstrained by the data.
- **A noisy channel.** If the revealed outcome is the truth flipped at rate f, the minimiser of the inner term is P(revealed r = 1 | x, a) = f + (1 - 2f) P(correct). The model calibrates to the channel, not to correctness. The selected run at f = 0.2 sits near that line: mean top-1 .743 against .744 (RESULTS_v3 section 4, `runs/v3_06b_direct_brier_n5000_s0_noisy_inverted/metrics.json:stored_channel`).
- **Inverting a known rate.** p_correct = (p - f) / (1 - 2f) undoes the channel, but it amplifies the remaining error by 1 / (1 - 2f), which is 2.5 at f = 0.3 (`N:by_rate.0.3.amplification`).
- **The compression is global.** It is not confined to the adapted domain: in-domain ECE rises by .050 for the fixed-noise run (RESULTS_v3 sections 4 and 5, `S:forgetting.test_indomain.minus_zero_shot`). A caller would have to invert every answer and know f to do it.

## 7. Brier or log score

Both are strictly proper and the gradient has the same form. As measured on Banking77 at 0.6B (RESULTS_v3 section 11):

- **N 5000, three training seeds:** the log score is slightly better on Brier and NLL and level on accuracy. Seed mean minus Brier's: Brier -.005 [-.008, -.001], NLL -.020 [-.029, -.010], accuracy +.002 [-.001, +.005] (`S:seeds.direct_log_seed_mean_minus_direct_brier_seed_mean`).
- **N 2000, one seed:** the log score is worse. Accuracy -.010 [-.016, -.004], Brier +.014, NLL +.020 (`S:n_curve.2000.off_policy.direct_log.minus_direct_brier`).
- **Forgetting:** the log score costs more in-domain accuracy at N 5000, -.038 to -.046 against -.027 to -.043 (`S:forgetting.test_indomain.minus_zero_shot`).
- **Seed spread:** the log score's NLL spreads over .036 across seeds (`S:seeds.direct_log`); the three Brier runs agree to within .0006, .2737, .2742 and .2737 (`runs/v3_06b_direct_brier_n5000_s<k>/metrics.json:splits.v3_banking77_test_full.overall.nll`).

Brier is the default. The log score is the alternative when Brier and NLL on the target domain matter more than in-domain retention and stability across seeds.

## Note: name candidates

1. **Chosen-Action Proper Scoring (CAPS)**, used above: says what is scored (the chosen action's probability) and how (a proper score).
2. **Logged Outcome Calibration (LOC)**: names the data and the goal, not the mechanism.
3. **Pathwise Feedback Scoring (PFS)**: names the gradient that separates it from the REINFORCE reward of section 3.
