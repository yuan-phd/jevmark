# Method: Chosen-Action Proper Scoring (CAPS)

This note states the method behind the v3 results and the paper: learning a decision model from logged bandit feedback on its own actions. Each empirical statement cites a results section and the metrics file behind it. `S:` is `runs/v3_stage_06b/metrics.json` and `N:` is `runs/v3_noise_sweep_06b/metrics.json`. References are keys of `docs/references.bib` in brackets.

## 1. Setting

A decision model maps a state x (a message, plain or JSON) and an option list O(x) = (o_1, ..., o_K) to a distribution p(. | x) over the options. It does this in one forward pass, reading the options as input. It is deployed on a new domain under a logging policy q(. | x). The deployment writes a log of interactions (x, O, a, q(a | x), r). Here a is the action taken, q(a | x) its propensity under the logging policy, and r in {0, 1} whether the action was right. Nothing else is recorded: this is logged bandit feedback, with no gold label and no outcome for the options not taken.

The learner starts from a general decision model, here the supervised adapter of v1, and sees only the log. In the experiments, q = 0.9 p_0 + 0.1 / K, with p_0 the starting model (RESULTS_v3 section 1, `runs/v3_log_s0/metrics.json`).

## 2. Objective

CAPS minimises the expected proper score of the chosen action's probability under the logging distribution:

  L(theta) = E_x E_{a ~ q(. | x)} [ S(p_theta(a | x), r(x, a)) ]

S is a strictly proper score for a binary event. The default is the Brier score, S(p, r) = (r - p)^2; the alternative is the log score, S(p, r) = -(r log p + (1 - r) log(1 - p)). On a log, the empirical mean over interactions estimates L directly, with no weights. The gradient is pathwise, through p_a: for the Brier score it is -2 (r - p_a) dp_a / dtheta. A right action is pushed up and a wrong one down, in proportion to how badly p_a forecast r.

**Identity.** CAPS is the direct method of logged bandit feedback [dudik2011doubly]: a regression of the chosen action's outcome on (x, a), with a proper score as the regression loss and the argmax of the fitted head as the policy. It is the regression-oracle objective of contextual bandits [foster2018regcb; foster2020squarecb]. It is also OpenJev-RLCD's stage 1 [gao2026openjev] specialised to partial feedback: stage 1 scores the whole readout distribution against the one-hot label, while CAPS scores the one component the log reveals, the chosen action's probability against its own outcome.

Accuracy improves as a consequence, not as the objective: moving right actions up and wrong ones down moves the argmax, but the loss asks only for calibrated forecasts of r. A negative on an option other than the policy's top choice exists only because the logging policy is random: it samples from p, and mixes in a uniform draw with weight epsilon 0.1. On the main Banking77 log, 1514 of the 2593 negatives (.584) fall on such options, about .545 of them from the uniform term (`runs/v3_log_s0/metrics.json:negatives.on_other`, `:negatives.on_other_uniform_share`), so what the log says about them depends on exploration; options never taken carry nothing (section 6).

**Why no propensity weight.** For each (x, a) with q(a | x) > 0, the inner term is minimised at p(a | x) = P(r = 1 | x, a) whatever q is, so the unweighted fit is consistent for a well-specified model [shimodaira2000covariate]. The propensity only sets how a finite model trades errors between actions, and L makes that trade as the deployed policy acts; a weight 1 / q(a) would retarget it to a uniform average over options.

**What a weight does in practice** (RESULTS_v3 section 11, `S:n_curve.<N>.off_policy`): the log score weighted by min(1 / q, 5) (ips_log) about equals the unweighted log score (.906 against .907 accuracy at N 2000, .930 against .933 at N 5000), and positive-only SFT with the same weights equals positive-only SFT, +.002 [-.001, +.005] and +.000 [-.003, +.004] (`S:n_curve.<N>.off_policy.ips_positive.minus_positive_sft.accuracy`). A correct logged action is usually the policy's top choice, so its weight is near one constant (`runs/v3_06b_ips_positive_n5000_s0/train_summary.json:ips`).

## 3. Why a proper score is not a REINFORCE reward

A detached, parameter-dependent reward keeps the score-function term of the gradient and drops the pathwise term [schulman2015scg; mohamed2020montecarlo]. The statement below is the corollary for a proper score of the sampled action's own probability.

**Corollary.** Take a question with K options, gold g and a model p. Use a proper score as a detached reward:
- Brier: R(a) = 1 - (r_a - p_a)^2;
- log score: R(a) = r_a log p_a + (1 - r_a) log(1 - p_a).

Sample a group of actions and form REINFORCE advantages against the group mean, not divided by the group standard deviation.

(i) For K > 2, every wrong action gets a reward at least gold's. Within a group, the lowest-probability wrong action sampled has the highest reward, so its advantage is non-negative, and positive unless every reward is equal. In any group that samples both gold and a wrong action, gold's advantage is non-positive.

(ii) For K = 2, every reward in every group is equal, so every advantage and the gradient are zero.

**Proof.** A wrong action w has r_w = 0, and p_w <= 1 - p_g because the probabilities sum to one.
- Under Brier, R(w) = 1 - p_w^2 and R(g) = 1 - (1 - p_g)^2; under the log score, R(w) = log(1 - p_w) and R(g) = log p_g. Both decrease in p_w and equal R(g) at p_w = 1 - p_g, so R(w) >= R(g).
- The sampled wrong action with the smallest p_w therefore has the group's maximum reward, at or above the mean; gold's reward is the group minimum, at or below it.
- At K = 2, p_w = 1 - p_g exactly, so every reward in a group is equal and every advantage is zero.

Under (i), the update raises the probability of rarely chosen wrong options and lowers gold's. **Why:** the score grades p_a as a forecast of r_a, and a low probability on a wrong option is always a good forecast. ∎

**Measured** (RESULTS_v2 section 2). On full labels this reward degraded the model: test_yelp accuracy fell from .505 to .044, and validation NLL rose from .219 to .872 (`runs/sft_06b/metrics.json:splits.test_yelp.overall.accuracy`, `runs/rlcd_06b_brier_s0/metrics.json:splits.test_yelp.overall.accuracy`; `runs/rlcd_06b_brier_s0/train_summary.json:init_valid.nll`, `runs/rlcd_06b_brier_s0/training_log.jsonl`, the step 500 validation row). A simulation of the sampler lowered the gold logit in 24.0 percent of questions and raised it in 1.8 percent (`runs/advantage_simulation_06b/metrics.json:rewards`).

**The baseline is the mean only.** The full-label runs use a group of 4 and subtract its mean, with no standard-deviation normalisation (`rlcd.normalize_std: false` in `configs/rlcd_06b.yaml` and in the `config.yaml` of every full-label run); the logged-feedback REINFORCE arm subtracts the batch mean only. This matters because Bereket and Leskovec [bereket2025uncalibrated] trace GRPO's overconfidence to dividing the advantage by the group standard deviation, and show that with an unbiased mean baseline a proper-score reward on a separately emitted confidence gives calibrated forecasts. That mechanism is absent here, so the failure above comes from scoring the sampling probability itself, not from the estimator.

CAPS avoids this by differentiating the score through p_a instead of using it as a reward.

## 4. Selection rule

The learner trains on the first 0.9 N interactions of the log. It keeps the checkpoint with the highest mean Bernoulli log-likelihood of the logged outcomes under p(logged action) on the last 0.1 N, which is the log score of the objective on held-out interactions. Gold is never read (RESULTS_v3 section 1).

**Necessary under noisy feedback** (RESULTS_v3 section 4). With outcomes flipped at rate 0.2, the selected checkpoint scores .907; trained to the end of the budget the same run falls to .813, below the zero-shot .851, while the clean run keeps .933 (`runs/v3_06b_direct_brier_n5000_s0_noisy/metrics.json`, `runs/v3_06b_direct_brier_n5000_s0_noisy_last/metrics.json`, `runs/v3_06b_zeroshot/metrics.json` and `runs/v3_06b_direct_brier_n5000_s0_last/metrics.json`, each at `splits.v3_banking77_test_full.overall.accuracy`); at 0.3 the final checkpoint reaches .734 (`N:by_rate.0.3.direct_brier.last.accuracy`). Passing over the log many times, the learner memorises the single flip draw of each interaction; the feedback-only criterion stopped it before that every time.

**What it costs:** granularity. Checkpoints are every 50 steps. At N 2000 the log-score arms peaked at the first candidate, so a finer grid could change their ranking. At N 500, 50 interactions select (RESULTS_v3 sections 9 and 11, `runs/<run>/train_summary.json:best_step`).

## 5. Relations

Selective prediction is scored by two metrics for a caller who thresholds the top-1 probability [geifman2017selective]: AURC, the area under the risk-coverage curve (lower is better), and Cov@eps, the largest share of answers a threshold keeps with error at most eps (higher is better).

- **Temperature scaling.** A single T fitted post hoc to the logged outcomes is global and leaves every prediction unchanged. It is the cheap baseline any method must beat. It reaches the lowest or joint-lowest ECE at every N, .020 at N 5000, but keeps zero-shot accuracy, .851, while CAPS reaches .929 (RESULTS_v3 section 2, `S:n_curve.5000`). It changes the level of the probabilities, not their ranking: AURC moves only from .0333 to .0328 and Cov@.05 from .756 to .765 (`S:n_curve.5000.{zero_shot,temperature}`).
- **Full-label SFT.** Cross-entropy on gold for every interaction is the upper bound, at the cost of one label per interaction. At N 5000, CAPS's seed mean is 1.5 points below it, -.015 [-.020, -.010], with ECE level (RESULTS_v3 section 3, `S:seeds.training_seed_mean_minus_full_sft_seed_mean`). At 1.7B its three-seed mean is .6 points below full-label SFT's single seed, -.006 [-.011, -.002], so the gap shrinks with scale (RESULTS_v3 section 12, `runs/v3_stage_17b/metrics.json:seeds.training_seed_mean_minus_full_sft`).
- **Positive-only SFT.** Cross-entropy on the confirmed actions, discarding the failures, and what a practitioner does without the method. It is selection toward the starting model's own successes: the exclusions the negatives carry, which options a message is not, are lost. The positive-unlabeled framing does not apply, because the positives are selected by the logger's confidence, not at random. CAPS leads it by 2.4 to 2.7 points of accuracy from N 2000, by .052 to .060 ECE at every N, and at N 5000 by -.0092 [-.0118, -.0068] in AURC and +.069 [+.046, +.094] in Cov@.05 (RESULTS_v3 section 2, `S:n_curve.<N>.direct_brier_minus_positive_sft`). It grows confident on its mistakes.
- **REINFORCE-style objectives on logged data.** The logged action as the single sample, reward r - p_a, a batch-mean baseline and a clipped importance weight: level with CAPS on accuracy (-.001 at both N), slightly worse on NLL (+.027 and +.019, intervals above 0) (RESULTS_v3 section 11, `S:n_curve.<N>.off_policy.outcome_minus_p_log.minus_direct_brier`). A proper score as the reward fails outright (section 3).

## 6. What the objective leaves open

- **Never-logged options.** The loss touches only p_a of the logged action. Options the policy never takes on a state move only through the normalisation of p, so their probabilities are unconstrained by the data.
- **A noisy channel.** If the revealed outcome is the truth flipped at rate f, the minimiser of the inner term is P(revealed r = 1 | x, a) = f + (1 - 2f) P(correct). The model calibrates to the channel, not to correctness. The selected run at f = 0.2 sits near that line: mean top-1 .743 against .744 (RESULTS_v3 section 4, `runs/v3_06b_direct_brier_n5000_s0_noisy_inverted/metrics.json:stored_channel`).
- **Backward correction.** With f known, p_correct = (p - f) / (1 - 2f) is backward correction with a known transition matrix [natarajan2013noisy; patrini2017losscorrection]. It amplifies the remaining error by 1 / (1 - 2f), the factor of random classification noise [angluin1988noisy]: 1.25, 1.67 and 2.5 at f 0.1, 0.2 and 0.3 (`N:by_rate.<f>.amplification`). The corrected top-1 ECE is .020, .051 and .068, against the clean run's .026 (`N:by_rate.<f>.direct_brier.channel.inverted_top1_ece`, `N:clean.direct_brier.ece`).
- **The compression is global.** It is not confined to the adapted domain: in-domain ECE rises by .050 for the fixed-noise run (RESULTS_v3 sections 4 and 5, `S:forgetting.test_indomain.minus_zero_shot`). A caller would have to invert every answer and know f to do it.

## 7. Brier or log score

Both are strictly proper and the gradient has the same form. On Banking77 at 0.6B (RESULTS_v3 section 11), over three seeds at N 5000 the log score is slightly better on Brier and NLL and level on accuracy: -.005 [-.008, -.001], -.020 [-.029, -.010] and +.002 [-.001, +.005] (`S:seeds.direct_log_seed_mean_minus_direct_brier_seed_mean`). It is worse at N 2000 on one seed, -.010 [-.016, -.004] in accuracy (`S:n_curve.2000.off_policy.direct_log.minus_direct_brier`); it costs more in-domain accuracy at N 5000, -.038 to -.046 against -.027 to -.043 (`S:forgetting.test_indomain.minus_zero_shot`); and its NLL spreads over .036 across seeds, .239 to .275, where the three Brier runs agree to within .0006 (`S:seeds.direct_log.training_seeds.nll.range`, `runs/v3_06b_direct_brier_n5000_s<k>/metrics.json:splits.v3_banking77_test_full.overall.nll`).

Brier is the default. The log score is the alternative when Brier and NLL on the target domain matter more than in-domain retention and stability across seeds.
