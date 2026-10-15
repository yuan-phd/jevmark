# Learning a calibrated decision model from its own logged correctness

*Draft, first pass. Workshop length: main text sections 1 to 7, then the appendix.*

Every number cites the report section and the committed metrics file it comes from. Report sections are RESULTS_v1, RESULTS_v2 and RESULTS_v3 (`docs/RESULTS_v*.md`), and docs/METHOD.md. Metrics files are given as `path:key`, with three shortcuts:
- `S:` for `runs/v3_stage_06b/metrics.json`;
- `N:` for `runs/v3_noise_sweep_06b/metrics.json`;
- `L:` for `runs/v3_stage_17b/metrics.json`.

## Abstract

A decision model reads a state and a list of options and returns a probability for each option in one forward pass. Once deployed, it usually learns only whether the option it chose was right. With a 0.6B Qwen3 decision model we show three things:
1. A proper score used as a detached REINFORCE reward is broken. It rewards wrong actions and cut one score task's accuracy from .636 to .111.
2. With full labels, no bandit objective beat cross-entropy plus one temperature on unseen-schema calibration. The best arm was level, -.002 [-.006, +.004].
3. From logged correctness alone, Chosen-Action Proper Scoring (CAPS), a pathwise Brier loss on the chosen action, beats training on confirmed answers only. On Banking77 it is +2.6 points [1.9, 3.3] more accurate, with ECE .053 lower. It stays 1.5 points [1.0, 2.0] below full-label training at 0.6B and .4 points [-1.0, +.2] below it at 1.7B.

## 1. Introduction

**The decision model.** A decision model answers typed questions about a text state: yes or no, one of K options, or a level on a scale. The options are part of each request, so the same model answers label sets it was never trained on. It returns a full distribution per question in one forward pass, with no text generated. The caller's code turns the probabilities into actions, for example by answering above a confidence threshold and escalating below it. This only works if the probabilities are calibrated. Our model is a LoRA adapter on Qwen3-0.6B-Base and 1.7B-Base. Its in-domain ECE is .013 to .015, but it is overconfident on unfamiliar schemas, with ECE up to .22 (RESULTS_v1 section 2, `runs/sft_{06b,17b}/metrics.json:splits.<split>.overall.ece`).

**Why partial feedback is the realistic case.** After deployment on a new domain, labels are expensive and rare. What a system does record is whether the action it took turned out right: a ticket reopened or not, a routing corrected or not. This is logged bandit feedback. There is one outcome per interaction, only for the chosen action, under a known logging policy. The common practice is to fine-tune on the interactions that went well and discard the rest. That throws away the failures and, as we show, makes the model overconfident.

**This paper.** The study began as a test of a community reconstruction of a reinforcement-learning recipe for calibrated decisions (RLCD), and it makes three claims:
1. **A proper scoring rule must not be used as a detached reward.** With a group-mean baseline, every wrong action earns at least the gold action's reward when K > 2, and nothing is learned at K = 2. Section 3 proves this, and Section 5.2 shows the degradation.
2. **With full labels, bandit objectives add nothing.** Five arms over three seeds: none beats supervised fine-tuning (SFT) plus one temperature on unseen-schema ECE (Section 5.2).
3. **With only logged correctness, CAPS works.** It minimises the expected proper score of the chosen action's probability under the logging distribution, differentiated through that probability. It beats SFT on confirmed answers from 2000 interactions on, and it is better calibrated at every log size. Importance weighting and off-policy REINFORCE add nothing to it. It stays below full-label SFT (Section 5.3).

**What the paper does not show.**
- The feedback is simulated from gold labels, cleanly or with symmetric flips at a known rate.
- One adaptation domain is complete. A second is pending [pending session H].
- 1.7B is one seed [pending session L].
- The gap to full labels does not shrink as the log grows.

## 2. Related work

**Off-policy learning from logged bandit feedback.**
- *Inverse propensity scoring* reweights each logged outcome by 1 / q(a) to evaluate or learn a policy other than the logger [Horvitz and Thompson 1952; Strehl et al. 2010].
- *Doubly robust estimators* combine it with a reward model [Dudík, Langford and Li 2011].
- *Counterfactual risk minimisation* adds a variance penalty [Swaminathan and Joachims 2015a], self-normalisation [Swaminathan and Joachims 2015b], and deep versions [Joachims, Swaminathan and de Rijke 2018].
- *Applications* include news recommendation and advertising [Li et al. 2010; Bottou et al. 2013].

These methods target the value of a new policy. CAPS targets calibration of p on the actions the logger takes, which needs no propensity (Section 3). We test the propensity-weighted forms as baselines (Section 5.3).

**Bandit-to-supervised conversion.** Simulating bandit feedback from a labelled dataset, by revealing only whether a logged action matches the label, is the standard evaluation protocol of this literature [Dudík, Langford and Li 2011; Swaminathan and Joachims 2015a; Beygelzimer and Langford 2009]. We use it with a decision model as the logger.

**Calibration and temperature scaling.** Modern networks are overconfident, and a single temperature fitted on held-out data is a strong post-hoc fix [Guo et al. 2017; Platt 1999]. We measure ECE with equal-width bins [Naeini, Cooper and Hauskrecht 2015]. Temperature changes no prediction, so it is the cheap baseline any training method must beat.

**Proper scoring rules.** The Brier score [Brier 1950] and the log score are strictly proper: their expectation is minimised by the true probability [Gneiting and Raftery 2007]. Recent work uses proper scores as rewards for an LLM's verbalized confidence [Damani et al. 2025]. Our negative result concerns a different use: a proper score as the REINFORCE reward of the chosen action.

**Proper scores of a verbalized confidence against proper scores of the policy.** Damani et al. [2025] train a language model to emit a confidence as part of its output and reward it with a proper score of that confidence against the answer's correctness. There the confidence is itself an action: the policy chooses which number to state, and the policy gradient optimises the expected score of the stated forecast correctly. Our proposition (Section 3) concerns a different object: a proper score of the policy's own sampling probability p_a, detached and used as the reward of the sampled action a. There the forecast and the sampled action are the same object, so a low probability on a wrong action earns a high reward and the gradient raises it. The two results are compatible, and the proposition does not contradict their theorem: it applies only when the scored probability is the policy's probability of the action being rewarded, not a separate stated confidence.

**Policy optimisation for language models.**
- REINFORCE [Williams 1992].
- PPO [Schulman et al. 2017], used in RLHF [Christiano et al. 2017; Ouyang et al. 2022].
- GRPO, with its group-mean baseline [Shao et al. 2024].

The group-mean baseline is the one under which our proposition holds.

**Training on positive outcomes only.** Fine-tuning on a model's own successful outputs is filtered behaviour cloning [Zelikman et al. 2022]. Our positive_sft baseline is that procedure on logged decisions.

**Verbalized confidence.** LLMs can state confidence in words or tokens [Lin, Hilton and Evans 2022; Kadavath et al. 2022; Tian et al. 2023; Xiong et al. 2024]. Our model never generates text; its confidence is the distribution over the options it read.

**Selective prediction and cascades.** Answering only above a confidence threshold trades coverage for accuracy [Chow 1970; El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017]. Cascades route uncertain queries to a larger model [Viola and Jones 2001; Chen, Zaharia and Zou 2023]. We report coverage at fixed thresholds because that is how a caller uses the probabilities.

**Learning with noisy labels.** Symmetric label noise at a known rate can be corrected [Natarajan et al. 2013; Patrini et al. 2017]. Networks memorise noisy labels when trained long enough [Zhang et al. 2017; Arpit et al. 2017], and benchmark test sets contain label errors [Northcutt, Athalye and Mueller 2021]. Our flipped-feedback results show both effects, for feedback instead of labels (Section 5.3).

## 3. Method: Chosen-Action Proper Scoring

**Setting.** A logged interaction is (x, O, a, q(a | x), r):
- x is a state and O its K options;
- a is the action the logging policy q chose, and q(a | x) its propensity;
- r in {0, 1} records whether a was right.

The learner starts from a general decision model p_theta and sees only the log. In our experiments q = 0.9 p_0 + 0.1 / K, with p_0 the starting model (RESULTS_v3 section 1, `runs/v3_log_s0/metrics.json`).

**Objective.** CAPS minimises

  L(theta) = E_x E_{a ~ q(. | x)} [ S(p_theta(a | x), r(x, a)) ]

with S a strictly proper binary score: Brier, (r - p)^2, by default, or the log score. The empirical mean over the log estimates L directly. The gradient flows through p_a; for Brier it is -2 (r - p_a) dp_a / dtheta. A right action is pushed up and a wrong one down.

**Why no propensity weight.** For each (x, a) with q(a | x) > 0, the inner term is minimised at p(a | x) = P(r = 1 | x, a), whatever q is. The propensity only sets how the finite model trades errors between actions, and L makes that trade as the deployed policy acts. A weight 1 / q would retarget it to a uniform average over options.

Accuracy is a consequence, not the objective: moving right actions up and wrong ones down moves the argmax. A negative on an option other than the logger's top choice arises only from the logger's randomness. On our main log, 1514 of 2593 negatives (.584) fall on such options, and about .545 of those come from the uniform exploration term (docs/METHOD.md section 2, `runs/v3_log_s0/metrics.json:negatives`). Options the logger never takes are unconstrained by the objective.

**Proposition.** Take K options, gold g, and a proper score used as a detached reward:
- Brier: R(a) = 1 - (r_a - p_a)^2;
- log score: R(a) = r_a log p_a + (1 - r_a) log(1 - p_a).

Form REINFORCE advantages against the mean of a sampled group.

(i) If K > 2, every wrong action's reward is at least gold's. In each group, the lowest-probability wrong action sampled has the largest reward, so its advantage is non-negative. Whenever the group also samples gold, gold's advantage is non-positive.

(ii) If K = 2, all rewards in every group are equal, and the gradient is zero.

*Proof.* A wrong action w has r_w = 0 and p_w <= 1 - p_g.
- Under Brier, R(w) = 1 - p_w^2 and R(g) = 1 - (1 - p_g)^2. Under the log score, R(w) = log(1 - p_w) and R(g) = log p_g.
- Both rewards decrease in p_w and equal R(g) at p_w = 1 - p_g. Hence R(w) >= R(g), and the smallest sampled p_w gives the group maximum.
- At K = 2, p_w = 1 - p_g exactly, so R(w) = R(g). ∎

The score grades p_a as a forecast of r_a, and a low probability on a wrong option is always a good forecast. CAPS uses the same score as a loss on p_a, not as a reward on a. This does not contradict the proper-score rewards of Damani et al. [2025], which score a confidence the model states as a separate action, not the probability with which it sampled the action being rewarded (Section 2).

**Selection.** Train on the first 0.9 N interactions. Keep the checkpoint, from a 50-step grid, with the highest mean Bernoulli log-likelihood of the logged outcomes under p(logged action) on the last 0.1 N. No gold label is read (RESULTS_v3 section 1). Under noisy feedback this is what prevents collapse (Section 5.3).

## 4. Relation to the baselines

- **Temperature** fitted to the logged outcomes: post-hoc and global, and it changes no prediction.
- **positive_sft**: cross-entropy on confirmed actions. What practitioners do.
- **full_sft**: cross-entropy on gold for every logged interaction. The upper bound, at one label per interaction.
- **Off-policy forms**, all on the same log:
  - REINFORCE with reward r - p_a, a batch-mean baseline and a clipped importance weight (outcome_minus_p_log);
  - CAPS with the log score, weighted by a clipped 1 / q (ips_log);
  - positive_sft with the same weights (ips_positive).

In tables, CAPS with the Brier score is the run name direct_brier, and with the log score direct_log.

## 5. Experiments

### 5.1 Setup

**Model and data.**
- Qwen3-0.6B-Base and 1.7B-Base with LoRA (r 16, alpha 32, attention projections) [Hu et al. 2022; Qwen Team 2025]. Answers are read from the letter logits at each question's answer slot in one forward pass.
- Supervised training uses CLINC150 intents [Larson et al. 2019] and SST-5 sentiment [Socher et al. 2013]. Options are sampled per record, with negated yes-or-no phrasings.
- Evaluation covers 20 held-out intents and four schemas never trained on: AG News, Yelp [Zhang, Zhao and LeCun 2015], emotion [Saravia et al. 2018] and Banking77 [Casanueva et al. 2020] (RESULTS_v1 section 1).

**Leak and duplicate gates.** Before any GPU run, the data must pass two CPU checks (RESULTS_v1 section 3):
- state-free leak probes: classifiers on phrasing, question text and structure, which must not beat chance;
- train-test duplicate rules.

The Banking77 log domain drops 30 messages that match the test split and 31 repeats (RESULTS_v3 section 1, `data/v3_manifest.json`).

**Adaptation protocol** (RESULTS_v3 section 1).
- The 0.6B model logs every message of the Banking77 train split once: 9942 messages, each a 10-option choice with gold always present. It samples from q = 0.9 p + 0.1 / K and records only the correctness of its action. Policy accuracy is .836 and the outcome rate .739 (`runs/v3_log_s0/metrics.json`).
- Learners see the first N in {500, 2000, 5000} interactions. All trained learners take the same number of steps, max(200, ceil(10 x 0.9 N / 32)), at lr 5e-5.
- Evaluation is on the full Banking77 test split (3080 records), plus test_indomain and test_unseen_intents for forgetting.
- Intervals are 95 percent bootstrap over records, 1000 resamples [Efron 1979], paired across runs.
- At N 5000, CAPS-Brier, CAPS-log and full_sft each have three training seeds, and CAPS-Brier also has a second log draw.

### 5.2 Full labels: the negative results (v2)

**The broken reward.** A first run used Brier as a REINFORCE reward on fully labelled in-domain data (RESULTS_v2 section 2, `runs/sft_06b/metrics.json` and `runs/rlcd_06b_brier_s0/metrics.json:splits.<split>.score.accuracy`).
- Score accuracy fell from .636 to .111 on SST-5 and from .505 to .044 on Yelp.
- A simulation of the sampler on the SFT model lowered the gold logit in 24.0 percent of questions and raised it in 1.8 percent (`runs/advantage_simulation_06b/metrics.json:rewards`).

This is the proposition of Section 3 at work.

**No gain from bandit objectives on full labels.** Five arms (RESULTS_v2 section 3, `runs/rlcd_stage2a_06b/metrics.json:unseen_schemas`, `:temperature_ablation`):
- continued cross-entropy (the control);
- two REINFORCE arms, with outcome reward and with r - p reward;
- CAPS with Brier and with the log score.

Each ran three seeds at 0.6B. Against SFT plus one in-domain temperature (unseen-schema mean ECE .121):
- every arm's unseen-schema ECE is higher, by +.031 to +.127;
- given its own temperature, the best arm is level, -.002 [-.006, +.004], and so is the control, +.002 [-.001, +.009].

When gold is known, whether a sampled answer was right carries no more than the label itself. Figure `figures/v2_arm_ece.png` shows the arms.

### 5.3 Logged correctness only (v3, Banking77)

**The N-curve** (RESULTS_v3 section 2, `S:n_curve.<N>`; Figure `figures/v3_n_curve.png`).
- *Against positive_sft:* CAPS-Brier ties at N 500, +.002 [-.006, +.009]. It leads by +.027 [+.019, +.035] at N 2000 and +.024 [+.016, +.031] at N 5000. Its ECE is lower by .052 to .060 at every N.
- *Against full_sft:* it stays 0.7 to 1.8 points below.
- *Gap closed:* it closes 69, 91 and 81 percent of the accuracy gap between zero-shot and full_sft at N 500, 2000 and 5000.
- *The temperature learner* keeps zero-shot accuracy (.851) and has the lowest or joint-lowest ECE (.020 at N 5000). Its Brier and NLL stay far above every trained learner's.

![Accuracy and ECE against N](figures/v3_n_curve.png)

**Seeds** (RESULTS_v3 section 3, `S:seeds`).
- Over three training seeds at N 5000, CAPS-Brier is +.026 [+.019, +.033] over positive_sft and -.015 [-.020, -.010] below full_sft's seed mean, with ECE level, +.003 [-.001, +.008].
- Training seeds and the second log draw move accuracy by at most .6 points.
- A temperature fitted on CAPS-Brier's own training interactions comes out near 1 (.951 to 1.039) and moves ECE by at most .003 (`S:direct_brier_temperatures`).

**Off-policy baselines** (RESULTS_v3 section 11, `S:n_curve.<N>.off_policy`; Figure `figures/v4_offpolicy.png`).
- Every arm that keeps the negatives leads positive_sft, by +.016 to +.028 in accuracy.
- ips_positive equals positive_sft: +.002 [-.001, +.005] at N 2000, +.000 [-.003, +.004] at N 5000. At N 5000, only 25 of the 3342 positives it trains on are clipped (`runs/v3_06b_ips_positive_n5000_s0/train_summary.json:ips`).
- No arm beats CAPS-Brier on accuracy with an interval excluding 0.
- At N 5000, ips_log beats CAPS-Brier on Brier and NLL, but the unweighted CAPS-log beats it by more, so the gain is the log score's, not the weight's.
- Over three seeds, CAPS-log against CAPS-Brier at N 5000 is +.002 [-.001, +.005] in accuracy, -.005 [-.008, -.001] in Brier and -.020 [-.029, -.010] in NLL (`S:seeds.direct_log_seed_mean_minus_direct_brier_seed_mean`).
- CAPS-log forgets more in-domain (-.038 to -.046 against -.027 to -.043) and is behind at N 2000 on one seed (-.010 [-.016, -.004]).

Brier stays the default.

![Off-policy baselines](figures/v4_offpolicy.png)

**Noisy feedback** (RESULTS_v3 section 4, `N:by_rate`; Figure `figures/v4_noise_sweep.png`). Outcomes are flipped at f in {0.1, 0.2, 0.3}, with nested draws.
- *CAPS calibrates to the channel,* p = f + (1 - 2f) P(correct). The selected run at f 0.2 has mean top-1 .743 against a channel line of .744.
- *Its excess over a channel-calibrated copy of the clean run grows with f:* .017, .044 and .060 (`N:by_rate.<f>.direct_brier.channel.channel_scale.channel_ece_minus_reference`).
- *Its accuracy loss grows with f:* -.006, -.022 and -.041.
- *Against positive_sft's loss:* smaller at 0.1 and 0.2 (+.009 [+.000, +.018], +.017 [+.007, +.028]), level at 0.3 (-.003 [-.014, +.007]), one seed per rate.
- *Trained to the end,* CAPS memorises the single flip draw. At 0.3 it falls to .734, below zero-shot's .851, while the feedback-only selection picks steps before the collapse every time.
- *A soft-target diagnostic* removes most of the residual. It needs the clean outcome, so it is a diagnostic, not a learner: channel-scale ECE .019 to .024 against a reference of .014 (RESULTS_v3 section 4).

The residual is therefore single-draw variance, not a limit of the objective.

![Noise-rate sweep](figures/v4_noise_sweep.png)

**Scale** (RESULTS_v3 section 12, `L:n_curve.5000`; Figure `figures/v4_scale.png`). At 1.7B, with its own log:
- CAPS-Brier reaches .949 against full_sft's .953, -.004 [-.010, +.002] on one seed;
- it leads positive_sft by +.036 [+.028, +.045].

The larger model helps the outcome-only learner most. The 1.7B seed ranges are [pending session L].

![Scale](figures/v4_scale.png)

**Forgetting and coverage** (RESULTS_v3 sections 5 and 6, `S:forgetting`, `S:coverage`).
- *In-domain accuracy cost at N 5000:* 2.7 to 4.3 points for CAPS-Brier and 3.6 to 5.2 for full_sft across seeds; positive_sft costs under 1.
- *In-domain ECE:* CAPS-Brier keeps it within .010 of zero-shot.
- *Coverage at a .95 confidence threshold:* positive_sft keeps .916 of answers at .942 accuracy, so it passes on about 6 percent errors as confident. CAPS-Brier keeps .743 at .986.

**Against an API model.** On the 500-record Banking77 subset, every CAPS-Brier run at N 5000 scores .926 to .936, against gpt-4.1-mini's .918. That comes from 5000 correctness bits, with no paired interval (RESULTS_v3 section 7, `S:b2_reference`).

### 5.4 A second domain: emotion

[pending session H] The emotion log domain (dair-ai/emotion train split, 6 options, no `other`) is built and logged, and positive_sft, full_sft and the temperature learner have run at N 500, 2000 and 5000 (`runs/v3_emotion_06b_*`). CAPS-Brier, its seeds and CAPS-log are pending, and the emotion N-curve table and panel will come from `runs/v3_stage_emotion_06b/metrics.json`.

## 6. Limitations

- **Simulated feedback.** Outcomes come from gold labels, revealed cleanly or flipped symmetrically at a known rate. Real feedback is delayed, biased towards some errors, and of unknown noise rate, and the inversion needs that rate. The channel compression is global: in-domain ECE rises by .050 for the noisy run (RESULTS_v3 section 5).
- **Domains and sizes.**
  - One complete adaptation domain; emotion is [pending session H].
  - One logging policy at epsilon 0.1, already .836 accurate.
  - Two sizes, with 1.7B on one seed [pending session L].
  - The noise sweep has one seed per rate.
- **Label noise, judged by two language models, no human.**
  - *Banking77* (docs/audit/SUMMARY.md, `docs/audit/summary.json:groups`): some label problem in .12 and .14 of a uniform sample, and a clearly wrong label in .04. Of 100 questions both full_sft and CAPS-Brier miss, the judges find a problem in .66 and .62. The ceiling is plausibly near .96.
  - *Emotion* (docs/audit/EMOTION_SUMMARY.md, `docs/audit/emotion_summary.json:groups`): a problem in .32 and .40 of a uniform sample, mostly a labelling convention that follows emotion words (.20 for both judges).
  - *Both:* the clean feedback inherits these labels.
- **Selection.** Checkpoints every 50 steps. At N 2000 the log-score arms peaked at the first candidate, so a finer grid could favour them. At N 500, 50 interactions select (RESULTS_v3 sections 9 and 11).
- **Coverage of the claim.** The gap to full labels does not shrink with N (RESULTS_v3 section 8). The claim is that using failures beats discarding them, not that it replaces labels.

## 7. Reproducibility

- **Code and data:** https://github.com/yuan-phd/jevmark, Apache-2.0.
- **Tags:** v1, v2 and v3 for the phases, and v3-final for the closing state.
- **Commits of the runs:**
  - b5913ef: the v3 logs and the phase 3 runs;
  - b6b9e85 and 409c983: the off-policy runs;
  - 36e8ee1: the noise sweep;
  - c9e29c6: the 1.7B runs.
- **Run provenance:** every run's `metrics.json` records its commit, dirty flag, data sha256 and adapter sha256.
- **Adapters on the Hugging Face Hub** (`yuanphd/jevmark`):
  - sft_06b and sft_17b, the general decision model;
  - rlcd_banking77_06b, CAPS-Brier adapted to Banking77.
- **Tables and figures:** every table is regenerated on CPU from committed files by `scripts/compare_v3.py`, `scripts/compare_noise.py`, `scripts/compare_rlcd.py`, `scripts/paired_deltas.py` and `scripts/log_stats.py`. Every figure is regenerated by `scripts/make_figures.py`. The training scripts (`scripts/train_sft.py`, `scripts/train_rlcd.py`) fit one Kaggle T4 session per run (docs/KAGGLE.md).

## References

- Arpit, D., et al. (2017). A closer look at memorization in deep networks. ICML.
- Beygelzimer, A., and Langford, J. (2009). The offset tree for learning with partial labels. KDD.
- Bottou, L., et al. (2013). Counterfactual reasoning and learning systems: the example of computational advertising. JMLR 14.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review 78.
- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., and Vulić, I. (2020). Efficient intent detection with dual sentence encoders. NLP4ConvAI workshop, ACL.
- Chen, L., Zaharia, M., and Zou, J. (2023). FrugalGPT: how to use large language models while reducing cost and improving performance. arXiv:2305.05176.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory 16.
- Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. NeurIPS.
- Damani, M., et al. (2025). Beyond binary rewards: training language models to reason about their uncertainty. ICLR 2026. arXiv:2507.16806.
- Dudík, M., Langford, J., and Li, L. (2011). Doubly robust policy evaluation and learning. ICML.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. Annals of Statistics 7.
- El-Yaniv, R., and Wiener, Y. (2010). On the foundations of noise-free selective classification. JMLR 11.
- Geifman, Y., and El-Yaniv, R. (2017). Selective classification for deep neural networks. NeurIPS.
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. JASA 102.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
- Horvitz, D. G., and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. JASA 47.
- Hu, E. J., et al. (2022). LoRA: low-rank adaptation of large language models. ICLR. arXiv:2106.09685.
- Joachims, T., Swaminathan, A., and de Rijke, M. (2018). Deep learning with logged bandit feedback. ICLR.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Larson, S., et al. (2019). An evaluation dataset for intent classification and out-of-scope prediction. EMNLP-IJCNLP.
- Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. WWW.
- Lin, S., Hilton, J., and Evans, O. (2022). Teaching models to express their uncertainty in words. TMLR. arXiv:2205.14334.
- Naeini, M. P., Cooper, G. F., and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. AAAI.
- Natarajan, N., Dhillon, I. S., Ravikumar, P., and Tewari, A. (2013). Learning with noisy labels. NeurIPS.
- Northcutt, C. G., Athalye, A., and Mueller, J. (2021). Pervasive label errors in test sets destabilize machine learning benchmarks. NeurIPS Datasets and Benchmarks.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.
- Patrini, G., et al. (2017). Making deep neural networks robust to label noise: a loss correction approach. CVPR.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, MIT Press.
- Qwen Team (2025). Qwen3 technical report. arXiv:2505.09388.
- Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., and Chen, Y.-S. (2018). CARER: contextualized affect representations for emotion recognition. EMNLP.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Shao, Z., et al. (2024). DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.
- Socher, R., et al. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. EMNLP.
- Strehl, A., Langford, J., Li, L., and Kakade, S. M. (2010). Learning from logged implicit exploration data. NeurIPS.
- Swaminathan, A., and Joachims, T. (2015a). Counterfactual risk minimization: learning from logged bandit feedback. ICML.
- Swaminathan, A., and Joachims, T. (2015b). The self-normalized estimator for counterfactual learning. NeurIPS.
- Tian, K., et al. (2023). Just ask for calibration: strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. EMNLP. arXiv:2305.14975.
- Viola, P., and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. CVPR.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. ICLR. arXiv:2306.13063.
- Zelikman, E., Wu, Y., Mu, J., and Goodman, N. D. (2022). STaR: bootstrapping reasoning with reasoning. NeurIPS.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. ICLR.
- Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level convolutional networks for text classification. NeurIPS.

## Appendix A. Full tables

### A.1 The v3 N-curve on the Banking77 test split, 0.6B, seed 0

Source: RESULTS_v3 section 2, `S:n_curve.<N>.<learner>`. Brackets are 95 percent record-bootstrap intervals.

| N | metric | zero-shot | temperature | positive_sft | full_sft | CAPS-Brier |
|---|---|---|---|---|---|---|
| 500 | accuracy | .851 [.839, .863] | .851 | .886 [.874, .896] | .904 [.894, .914] | .888 [.877, .898] |
| 2000 | accuracy | .851 | .851 | .890 [.879, .901] | .924 [.915, .933] | .917 [.907, .926] |
| 5000 | accuracy | .851 | .851 | .905 [.896, .915] | .947 [.938, .955] | .929 [.920, .938] |
| 500 | ECE | .063 [.054, .074] | .023 [.021, .036] | .091 [.082, .102] | .043 [.035, .052] | .031 [.026, .044] |
| 2000 | ECE | .063 | .018 [.017, .032] | .082 [.072, .092] | .017 [.013, .027] | .022 [.017, .031] |
| 5000 | ECE | .063 | .020 [.016, .032] | .078 [.069, .087] | .030 [.024, .038] | .026 [.020, .034] |
| 500 | Brier | .229 | .219 | .200 | .149 | .173 |
| 2000 | Brier | .229 | .219 | .185 | .114 | .127 |
| 5000 | Brier | .229 | .219 | .169 | .082 | .111 |
| 500 | NLL | .582 | .518 | .764 | .339 | .392 |
| 2000 | NLL | .582 | .518 | .622 | .236 | .299 |
| 5000 | NLL | .582 | .517 | .663 | .212 | .274 |

Paired differences, CAPS-Brier minus the other learner (`S:n_curve.<N>.direct_brier_minus_{positive_sft,full_sft}`):

| N | vs | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| 500 | positive_sft | +.002 [-.006, +.009] | -.060 [-.066, -.048] | -.027 [-.037, -.018] | -.373 [-.426, -.319] |
| 2000 | positive_sft | +.027 [+.019, +.035] | -.060 [-.067, -.050] | -.058 [-.069, -.048] | -.323 [-.373, -.276] |
| 5000 | positive_sft | +.024 [+.016, +.031] | -.052 [-.060, -.043] | -.058 [-.069, -.046] | -.389 [-.443, -.333] |
| 500 | full_sft | -.017 [-.023, -.011] | -.011 [-.016, -.001] | +.024 [+.018, +.030] | +.052 [+.036, +.070] |
| 2000 | full_sft | -.007 [-.014, +.000] | +.005 [-.002, +.011] | +.013 [+.007, +.019] | +.062 [+.046, +.078] |
| 5000 | full_sft | -.018 [-.025, -.011] | -.004 [-.010, +.003] | +.029 [+.020, +.038] | +.062 [+.037, +.086] |

### A.2 Training seeds at N 5000

Source: RESULTS_v3 sections 3 and 11, `S:seeds`.

| learner | accuracy, mean (range) | ECE, mean (range) | Brier, mean (range) | NLL, mean (range) |
|---|---|---|---|---|
| CAPS-Brier | .931 (.929, .935) | .025 (.019, .031) | .109 (.105, .111) | .274 (.274, .274) |
| CAPS-log | .933 (.931, .934) | .022 (.019, .027) | .105 (.101, .110) | .254 (.239, .275) |
| full_sft | .946 (.943, .949) | .022 (.017, .030) | .082 (.079, .086) | .192 (.175, .212) |

| seed mean minus | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|
| CAPS-Brier minus positive_sft | +.026 [+.019, +.033] | -.053 [-.059, -.044] | -.060 [-.071, -.048] | -.389 [-.443, -.333] |
| CAPS-Brier minus full_sft (seed means) | -.015 [-.020, -.010] | +.003 [-.001, +.008] | +.027 [+.020, +.034] | +.082 [+.061, +.101] |
| CAPS-log minus CAPS-Brier (seed means) | +.002 [-.001, +.005] | -.003 [-.006, +.000] | -.005 [-.008, -.001] | -.020 [-.029, -.010] |

### A.3 Off-policy baselines, seed 0

Source: RESULTS_v3 section 11, `S:n_curve.<N>.<arm>` and `S:n_curve.<N>.off_policy.<arm>.minus_direct_brier`.

| N | learner | accuracy | ECE | Brier | NLL | accuracy minus CAPS-Brier |
|---|---|---|---|---|---|---|
| 2000 | positive_sft | .890 [.879, .901] | .082 | .185 | .622 | |
| 2000 | ips_positive | .892 [.881, .902] | .077 | .183 | .593 | -.025 [-.032, -.018] |
| 2000 | CAPS-log | .907 [.897, .917] | .014 | .140 | .319 | -.010 [-.016, -.004] |
| 2000 | ips_log | .906 [.897, .916] | .012 | .143 | .323 | -.011 [-.016, -.005] |
| 2000 | outcome_minus_p_log | .916 [.906, .925] | .030 | .132 | .325 | -.001 [-.006, +.003] |
| 2000 | CAPS-Brier | .917 [.907, .926] | .022 | .127 | .299 | |
| 2000 | full_sft | .924 [.915, .933] | .017 | .114 | .236 | |
| 5000 | positive_sft | .905 [.896, .915] | .078 | .169 | .663 | |
| 5000 | ips_positive | .905 [.895, .915] | .078 | .169 | .650 | -.024 [-.031, -.016] |
| 5000 | CAPS-log | .933 [.925, .942] | .022 | .103 | .248 | +.005 [-.000, +.010] |
| 5000 | ips_log | .930 [.921, .939] | .026 | .106 | .253 | +.001 [-.004, +.007] |
| 5000 | outcome_minus_p_log | .928 [.918, .937] | .029 | .115 | .293 | -.001 [-.006, +.003] |
| 5000 | CAPS-Brier | .929 [.920, .938] | .026 | .111 | .274 | |
| 5000 | full_sft | .947 [.938, .955] | .030 | .082 | .212 | |

### A.4 Noise-rate sweep, N 5000, seed 0

Source: RESULTS_v3 section 4, `N:by_rate.<f>.<arm>` and `N:clean.<arm>`. "Final step" is step 1407, the end of the budget.

| f | learner | accuracy | minus clean | channel-scale ECE | reference | mean top-1 | channel line | final-step accuracy | criterion growth |
|---|---|---|---|---|---|---|---|---|---|
| 0 | CAPS-Brier | .929 | | | | | | .933 | .059 |
| 0.1 | CAPS-Brier | .923 [.914, .932] | -.006 [-.013, +.001] | .038 [.035, .046] | .021 | .870 | .838 | .892 | .114 |
| 0.2 | CAPS-Brier | .907 [.897, .916] | -.022 [-.030, -.014] | .058 [.052, .063] | .014 | .743 | .744 | .813 | .284 |
| 0.3 | CAPS-Brier | .888 [.877, .898] | -.041 [-.050, -.032] | .069 [.065, .073] | .009 | .674 | .655 | .734 | .757 |
| 0 | positive_sft | .905 | | | | | | | .268 |
| 0.1 | positive_sft | .890 [.880, .901] | -.015 [-.021, -.008] | .124 [.118, .133] | .063 | .935 | .812 | | 1.244 |
| 0.2 | positive_sft | .866 [.855, .878] | -.039 [-.048, -.031] | .196 [.190, .203] | .047 | .915 | .720 | | 2.216 |
| 0.3 | positive_sft | .867 [.856, .879] | -.038 [-.046, -.030] | .251 [.247, .256] | .031 | .896 | .647 | | 2.916 |

### A.5 Scale: 1.7B against 0.6B at N 5000

Source: RESULTS_v3 section 12, `L:n_curve.5000`, `S:n_curve.5000`, `S:seeds`. The 1.7B seed ranges are [pending session L].

| learner | 1.7B accuracy | 1.7B ECE | 1.7B Brier | 1.7B NLL | 0.6B accuracy, seed 0 | 0.6B accuracy, seed mean (range) | 0.6B ECE, seed mean |
|---|---|---|---|---|---|---|---|
| zero-shot | .848 [.836, .860] | .089 [.078, .100] | .242 | .660 | .851 | | .063 |
| temperature | .848 | .031 [.026, .044] | .222 | .519 | .851 | | .020 |
| positive_sft | .913 [.902, .923] | .067 [.058, .077] | .154 | .545 | .905 | (seed 0 only) | .078 |
| CAPS-Brier | .949 [.941, .957] | .021 [.015, .028] | .079 | .188 | .929 | .931 (.929, .935) | .025 |
| full_sft | .953 [.945, .960] | .023 [.018, .030] | .072 | .162 | .947 | .946 (.943, .949) | .022 |

| 1.7B paired difference | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|
| CAPS-Brier minus positive_sft | +.036 [+.028, +.045] | -.046 [-.054, -.037] | -.075 [-.088, -.061] | -.357 [-.411, -.306] |
| CAPS-Brier minus full_sft | -.004 [-.010, +.002] | -.002 [-.008, +.004] | +.007 [+.001, +.013] | +.026 [+.013, +.040] |

### A.6 Full labels: unseen-schema ECE, 0.6B, three seeds

Source: RESULTS_v2 section 3, `runs/rlcd_stage2a_06b/metrics.json:unseen_schemas` and `:temperature_ablation`.

| run | unseen mean ECE (seeds 0, 1, 2) | minus SFT + T [95% CI] | with its own temperature, minus SFT + T |
|---|---|---|---|
| SFT | .152 | | |
| SFT + T | .121 | | |
| continued cross-entropy | .171 (.185, .172, .156) | +.050 [+.044, +.056] | +.002 [-.001, +.009] |
| REINFORCE, outcome | .248 (.262, .244, .239) | +.127 [+.118, +.135] | +.025 [+.018, +.034] |
| REINFORCE, r - p | .153 (.173, .155, .131) | +.031 [+.026, +.038] | -.002 [-.006, +.004] |
| CAPS-Brier | .159 (.173, .165, .139) | +.037 [+.032, +.043] | +.001 [-.004, +.007] |
| CAPS-log | .170 (.181, .172, .156) | +.048 [+.043, +.055] | +.005 [+.001, +.011] |

### A.7 Emotion

[pending session H] The emotion N-curve, seed and forgetting tables, from `runs/v3_stage_emotion_06b/metrics.json`.
