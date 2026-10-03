# jevmark v3 results: adaptation from deployment feedback

Every number cites the file it comes from, as `path:key`. Unless another file is named, numbers are from `runs/v3_stage_06b/metrics.json` (written by `scripts/compare_v3.py` at commit 83454a4, not dirty), shortened to `S:key`. The specification is docs/V3_DESIGN.md and decision 56; the verdict and the inversion diagnostic are decision 57. The v1 report (`docs/RESULTS_v1.md`, frozen, decision 49) and the v2 report (`docs/RESULTS_v2.md`) are the background this starts from.

## 1. Setup

**Question.** After deployment on a new domain, when the only feedback is whether the chosen action was right, does RLCD learn the domain faster, more accurately and better calibrated than what a practitioner would otherwise use?

**Domain.** The Banking77 train split, never trained on: 9942 messages after the duplicate rules (30 dropped as test matches, 31 as repeats; `data/v3_manifest.json`), each one choice question built as test_banking77 records are (gold, 8 distractors, `other`, seeded option order, 20 percent JSON states). Gold is always among the options, so `other` is never correct. Evaluation is on the full Banking77 test split, `v3_banking77_test_full` (3080 records, the 1000 test_banking77 records verbatim plus the other 2080), and, for forgetting, on test_indomain and test_unseen_intents. Data hashes are in `configs/v3_data.yaml`; every run's `metrics.json:data_files_sha256` matches them.

**The log.** sft_06b, unchanged, answered every log-domain message once in a fixed seeded order, sampling its action from q = 0.9 p + 0.1 / K; only the correctness of the chosen option was recorded, plus a copy flipped with probability 0.2 for the noisy condition.

| log | sha256 | policy accuracy | outcome rate | flip share | predicted `other` | source |
|---|---|---|---|---|---|---|
| seed 0 (main) | 847f44c4... | .836 | .739 | .202 | .059 | `runs/v3_log_s0/metrics.json` |
| seed 1 (logging variance) | 7c11c116... | .836 | .736 | .194 | .059 | `runs/v3_log_s1/metrics.json` |

Both logs were written at commit b5913ef, not dirty, by the sft_06b adapter (sha256 5d2681f3...). Policy accuracy is the argmax accuracy, identical for both seeds because p and the order are the same; the outcome rate is that of the sampled action.

**Learners.** All start from sft_06b and see the first N interactions of the seed 0 log, N in {500, 2000, 5000}:

1. **zero-shot:** sft_06b unchanged (`runs/v3_06b_zeroshot`). Its per-question probabilities equal `runs/sft_06b`'s on the three shared splits; test_banking77 accuracy .851 and ECE .067, as in v1 (`runs/v3_06b_zeroshot/metrics.json:splits.test_banking77.overall`).
2. **temperature:** one T fitted on the first 0.9 N logged outcomes by Bernoulli maximum likelihood of p(action): T 1.547, 1.522 and 1.453 at N 500, 2000 and 5000 (`runs/v3_06b_temp_n<N>/calibration.json`, `S:temperatures`).
3. **positive_sft:** cross-entropy on the logged action of the outcome-1 interactions (322, 1332 and 3342 of 450, 1800 and 4500; `runs/v3_06b_positive_sft_n<N>_s0/train_summary.json:n_used`).
4. **direct_brier (RLCD):** (r - p_a)^2 on the logged action's probability for every interaction, beta 0, no importance weights.
5. **full_sft:** cross-entropy on gold for every interaction, the upper bound.

**Budget rule.** Learners 3 to 5 run the same steps at each N: max(200, ceil(10 x 0.9 N / 32)), that is 200, 563 and 1407, warmup 20, linear decay, lr 5e-5, micro-batch 8 x 4 (`configs/v3_06b.yaml`). positive_sft makes more passes over fewer records.

**Selection rule.** Each learner trains on the first 0.9 N interactions and keeps the checkpoint (every 50 steps and the last; step 0 is not a candidate) with the highest mean Bernoulli log-likelihood of the logged outcomes under p(logged action) on the last 0.1 N. Gold is never used to select, not even by full_sft. Best steps: positive_sft 50, 50, 200; full_sft 50, 100, 300; direct_brier 50, 150, 300 at N 500, 2000, 5000 (`runs/<run>/train_summary.json:best_step`). Every learner peaked early and declined after. positive_sft's criterion never improved on step 0 at N 500 and 5000 (-.297 against -.279; -.363 against -.300), so its selected adapters are worse than sft_06b by the selection criterion: on confirmed answers alone it grows confident on its mistakes too.

**Runs.** 14 trained runs in three Kaggle sessions on one T4, all at commit b5913ef, not dirty, fp16 without fallback, every planned step run, each recording its log's sha256 and the sft_06b adapter's (`runs/v3_06b_*/metrics.json:git`, `:v3`, `:init_adapter_sha256`). A run of 1407 steps took 58.6 to 60.9 minutes and an evaluation 11.2 to 12.1 minutes.

**Intervals.** 95 percent bootstrap by record, 1000 resamples, the same record draws for every run on a split, so every difference is paired (`S:bootstrap`).

## 2. The N-curve on v3_banking77_test_full

`S:n_curve.<N>.<learner>`; direct_brier is seed 0.

| N | metric | zero-shot | temperature | positive_sft | full_sft | direct_brier |
|---|---|---|---|---|---|---|
| 500 | accuracy | .851 [.839, .863] | .851 | .886 [.874, .896] | .904 [.894, .914] | .888 [.877, .898] |
| 2000 | accuracy | .851 | .851 | .890 [.879, .901] | .924 [.915, .933] | .917 [.907, .926] |
| 5000 | accuracy | .851 | .851 | .905 [.896, .915] | .947 [.938, .955] | .929 [.920, .938] |
| 500 | ECE | .063 [.054, .074] | .023 [.021, .036] | .091 [.082, .102] | .043 [.035, .052] | .031 [.026, .044] |
| 2000 | ECE | .063 | .018 [.017, .032] | .082 [.072, .092] | .017 [.013, .027] | .022 [.017, .031] |
| 5000 | ECE | .063 | .020 [.016, .032] | .078 [.069, .087] | .030 [.024, .038] | .026 [.020, .034] |
| 500 | Brier | .229 | .219 | .200 | .149 | .173 [.159, .188] |
| 2000 | Brier | .229 | .219 | .185 | .114 | .127 [.113, .140] |
| 5000 | Brier | .229 | .219 | .169 | .082 | .111 [.098, .124] |
| 500 | NLL | .582 | .518 | .764 | .339 | .392 [.354, .430] |
| 2000 | NLL | .582 | .518 | .622 | .236 | .299 [.264, .332] |
| 5000 | NLL | .582 | .517 | .663 | .212 | .274 [.239, .311] |

Paired differences, direct_brier minus the other learner (`S:n_curve.<N>.direct_brier_minus_positive_sft`, `S:n_curve.<N>.direct_brier_minus_full_sft`):

| N | vs | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| 500 | positive_sft | +.002 [-.006, +.009] | -.060 [-.066, -.048] | -.027 [-.037, -.018] | -.373 [-.426, -.319] |
| 2000 | positive_sft | +.027 [+.019, +.035] | -.060 [-.067, -.050] | -.058 [-.069, -.048] | -.323 [-.373, -.276] |
| 5000 | positive_sft | +.024 [+.016, +.031] | -.052 [-.060, -.043] | -.058 [-.069, -.046] | -.389 [-.443, -.333] |
| 500 | full_sft | -.017 [-.023, -.011] | -.011 [-.016, -.001] | +.024 [+.018, +.030] | +.052 [+.036, +.070] |
| 2000 | full_sft | -.007 [-.014, +.000] | +.005 [-.002, +.011] | +.013 [+.007, +.019] | +.062 [+.046, +.078] |
| 5000 | full_sft | -.018 [-.025, -.011] | -.004 [-.010, +.003] | +.029 [+.020, +.038] | +.062 [+.037, +.086] |

- **The temperature** changes no prediction, so its accuracy is zero-shot's at every N. It has the lowest or joint-lowest ECE at every N, but its Brier and NLL stay far above every trained learner's.
- **positive_sft** gains 3.5 to 5.4 points of accuracy and is the worst-calibrated learner at every N, with ECE and NLL above zero-shot's at every N.
- **direct_brier** ties positive_sft at N 500 and leads it by 2.4 to 2.7 points from N 2000, with ECE lower by .052 to .060 at every N. It stays 0.7 to 1.8 points below full_sft, at similar or lower ECE but higher Brier and NLL. It closes 70, 90 and 81 percent of the accuracy gap between zero-shot and full_sft at N 500, 2000 and 5000.

## 3. Training seeds and logging variance at N 5000

direct_brier at N 5000: training seeds 0, 1 and 2 on the seed 0 log, and seed 0 on the seed 1 log (`S:seeds`).

| run | accuracy | ECE | Brier | NLL | minus seed 0, accuracy | minus seed 0, ECE |
|---|---|---|---|---|---|---|
| seed 0 | .929 [.920, .938] | .026 [.020, .034] | .111 | .274 | | |
| seed 1 | .935 [.926, .944] | .031 [.025, .040] | .105 | .274 | +.006 [+.001, +.011] | +.005 [-.001, +.011] |
| seed 2 | .930 [.921, .939] | .019 [.016, .029] | .111 | .274 | +.001 [-.004, +.007] | -.007 [-.012, +.000] |
| seed 1 log | .934 [.925, .943] | .026 [.022, .034] | .103 | .256 | +.005 [-.001, +.011] | +.000 [-.006, +.006] |
| mean of 3 training seeds (range) | .931 (.929, .935) | .025 (.019, .031) | .109 | .274 | | |

The training-seed mean against the other learners, each draw averaging the three seeds' differences under one record resample (decision 53; `S:seeds.training_seed_mean_minus_positive_sft`, `S:seeds.training_seed_mean_minus_full_sft`):

| seed mean minus | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|
| positive_sft | +.026 [+.019, +.033] | -.053 [-.059, -.044] | -.060 [-.071, -.048] | -.389 [-.443, -.333] |
| full_sft | -.016 [-.022, -.010] | -.005 [-.009, +.001] | +.027 [+.018, +.036] | +.062 [+.040, +.084] |

- **Training seeds and the logging draw move accuracy less than record sampling does.** The seed range is .6 points and the seed 1 log moves accuracy by +.5 points, against a record-level half-width of about .9 points for one run. Only seed 1 against seed 0 is significant when paired, +.006 [+.001, +.011].
- **ECE varies across seeds about as much as one run's interval is wide** (.019 to .031 against [.020, .034] for seed 0), so no ECE difference between direct_brier and full_sft at N 5000 is called either way.
- **The conclusions against positive_sft and full_sft hold for the seed mean**, with intervals that exclude 0. Unlike v2, where seed variance exceeded record-level intervals (docs/RESULTS_v2.md section 3), here seed 0 alone was representative.

## 4. Noisy feedback and the inversion diagnostic

The noisy condition trains on outcomes flipped with probability 0.2, for direct_brier and positive_sft at N 5000 (`S:noisy`). Both runs read the flipped column for training and selection: 2886 positives in the first 4500 interactions and a selection outcome rate of .638, against 3342 and .750 clean (`runs/v3_06b_positive_sft_n5000_s0_noisy/train_summary.json:n_used`, `:init_select`).

| learner | clean accuracy | noisy accuracy | noisy minus clean | clean ECE | noisy ECE | noisy minus clean |
|---|---|---|---|---|---|---|
| direct_brier | .929 | .907 [.897, .916] | -.022 [-.030, -.014] | .026 | .164 [.157, .173] | +.138 [+.123, +.151] |
| positive_sft | .905 | .866 [.855, .878] | -.039 [-.048, -.031] | .078 | .050 [.040, .060] | -.029 [-.037, -.021] |

direct_brier's change minus positive_sft's change (`S:noisy.direct_brier_change_minus_positive_sft_change`): accuracy +.017 [+.007, +.028], ECE +.167 [+.151, +.182], Brier +.013 [+.000, +.025], NLL +.277 [+.233, +.320]. In absolute terms noisy direct_brier is still the better of the two: accuracy .907 against .866, Brier .174 against .219, NLL .460 against .573 (`S:noisy.<arm>.noisy`).

**Why RLCD's calibration breaks under noise.** A proper score is minimised by the probability of the variable it is scored against. direct_brier scores p(action) against the revealed outcome, so it learns P(revealed outcome = 1 | action). Under a symmetric flip at rate f that is (1 - f) for a correct action and f for a wrong one, so p = f + (1 - 2f) P(correct). The model is calibrated to the channel, not to correctness, and its confidence is compressed towards .8 at most: noisy direct_brier's mean top-1 probability is .743 against .907 accuracy (`runs/v3_06b_direct_brier_n5000_s0_noisy_inverted/metrics.json:stored.mean_top1`), so it is underconfident, and almost no answer reaches a coverage threshold of .80 (section 6). This is not a failure of the method's logic: it learned exactly what it was shown. A known flip rate is invertible.

**The inversion diagnostic** (decision 57; `scripts/invert_noisy.py`, CPU, on the stored probabilities). p_correct = (p - 0.2) / 0.6 per option, clipped to [0, 1] and renormalised over the options; a question whose every option clips to 0 keeps its stored distribution. The argmax never changes, so accuracy is unchanged. A second view inverts the top-1 probability alone, without clipping the other options or renormalising, which is the direct test of calibration to the channel.

| run on v3_banking77_test_full | ECE | Brier | NLL | mean top-1 | coverage at .80 / .90 / .95 (accuracy) |
|---|---|---|---|---|---|
| direct_brier noisy, stored | .164 [.156, .174] | .174 | .460 | .743 | .032 (1.000) / .000 / .000 |
| direct_brier noisy, inverted | .066 [.057, .076] | .160 | 1.722 | .972 | .931 (.935) / .901 (.942) / .892 (.946) |
| direct_brier noisy, inverted top-1 only | .051 [.042, .059] | | | .861 | |
| direct_brier clean | .026 [.020, .035] | .111 | .274 | .951 | .868 (.974) / .809 (.983) / .743 (.986) |
| positive_sft noisy, stored | .050 [.040, .060] | .219 | .573 | .915 | .794 (.944) / .311 (.971) / .042 (.953) |
| positive_sft noisy, inverted | .121 [.109, .132] | .253 | 3.283 | .987 | .969 (.882) / .956 (.888) / .951 (.889) |
| positive_sft noisy, inverted top-1 only | .105 [.095, .117] | | | .969 | |
| positive_sft clean | .078 [.069, .089] | .169 | .663 | .984 | .958 (.928) / .933 (.938) / .916 (.942) |

Sources: `runs/v3_06b_<arm>_n5000_s0_noisy_inverted/metrics.json:{stored,inverted,inverted_top1_only,clean_reference}`, written at commit a9ea650, not dirty.

- **The inversion recovers most of RLCD's calibration, not all of it.** ECE falls from .164 to .066 (renormalised) or .051 (top-1 only), against .026 for the clean run; neither interval reaches the clean one. Inverted, the top-1 is .861 on average against .907 accuracy, so the noisy run is still underconfident: it was selected at step 250 of 1407, by a criterion on flipped outcomes, and had not converged to the channel's .8 and .2 targets.
- **Renormalising after clipping overshoots.** When every non-top option is below .2 they all clip to 0 and the top option becomes 1, so the inverted mean top-1 is .972. Gold clips to 0 on 184 questions, each scored at the NLL floor, which is why the inverted NLL is 1.722 (`:questions_gold_clipped_to_zero`). The top-1 view is the cleaner test; the per-option inversion is shown because it is the full distribution a caller would get.
- **The same inversion makes positive_sft worse** (.050 to .121), as it should: positive_sft does not model outcome rates, so it has no channel calibration to undo. Its stored ECE improved under noise only because false positives softened it.

## 5. Forgetting

Accuracy and ECE minus zero-shot on test_indomain (zero-shot .956, ECE .013) and test_unseen_intents (zero-shot .892, ECE .068), paired (`S:forgetting.<split>.minus_zero_shot.<run>`).

| run | test_indomain accuracy | test_indomain ECE | test_unseen_intents accuracy | test_unseen_intents ECE |
|---|---|---|---|---|
| temperature n500 / n2000 / n5000 | .000 | +.006 / +.005 / -.000 | .000 | -.033 / -.031 / -.026 |
| positive_sft n500 | -.007 [-.010, -.005] | +.019 | +.016 | +.007 |
| positive_sft n2000 | -.007 [-.009, -.005] | +.017 | +.021 | -.003 |
| positive_sft n5000 | -.008 [-.011, -.006] | +.017 | +.020 | +.001 |
| direct_brier n500 | -.005 [-.008, -.003] | -.002 | +.017 | -.025 |
| direct_brier n2000 | -.017 [-.020, -.013] | -.002 | +.023 | -.036 |
| direct_brier n5000 seed 0 | -.027 [-.031, -.023] | -.001 | +.025 | -.031 |
| direct_brier n5000 seed 1 | -.043 [-.048, -.039] | +.010 | +.020 | -.026 |
| direct_brier n5000 seed 2 | -.029 [-.033, -.025] | +.001 | +.026 | -.032 |
| direct_brier n5000 seed 1 log | -.030 [-.034, -.026] | +.008 | +.025 | -.034 |
| full_sft n500 | -.019 [-.022, -.016] | +.009 | +.030 | -.026 |
| full_sft n2000 | -.025 [-.029, -.021] | +.008 | +.029 | -.029 |
| full_sft n5000 | -.052 [-.057, -.047] | +.043 [+.038, +.047] | +.022 | -.007 |
| direct_brier n5000 noisy | -.014 [-.017, -.011] | +.050 | +.018 | -.018 |
| positive_sft n5000 noisy | -.008 [-.010, -.005] | +.005 | +.004 | -.001 |

- **Every trained learner loses in-domain accuracy and gains on the unseen intents**, which share the CLINC format but not the label set.
- **The in-domain cost grows with N for direct_brier and full_sft**, up to 2.7 to 4.3 points for direct_brier at N 5000 across seeds and 5.2 points for full_sft. positive_sft costs under 1 point at every N. the clean direct_brier runs keep in-domain ECE within .010 of zero-shot; full_sft at N 5000 adds .043.
- **Forgetting varies more across seeds than Banking77 accuracy does:** -.027, -.043 and -.029 for the three training seeds, a range of 1.6 points against .6 on Banking77. Beta 0 (V3_DESIGN note 13) is a choice, and this is its cost.

## 6. Predicted `other` and coverage

**`other` is never correct here** (V3_DESIGN note 11), so negative feedback can teach "never answer other" directly. The predicted-`other` rate on v3_banking77_test_full, and accuracy on the 2899 questions where no run predicted `other` (`S:other`):

| run | predicted `other` | accuracy without `other` |
|---|---|---|
| zero-shot and temperature | .052 | .901 |
| positive_sft n500 / n2000 / n5000 | .012 / .011 / .007 | .914 / .915 / .928 |
| direct_brier n500 / n2000 / n5000 | .006 / .000 / .000 | .910 / .934 / .941 |
| direct_brier n5000 seed 1 / seed 2 / seed 1 log | .000 / .000 / .000 | .944 / .943 / .945 |
| full_sft n500 / n2000 / n5000 | .002 / .001 / .000 | .920 / .934 / .954 |
| direct_brier noisy / positive_sft noisy | .007 / .040 | .926 / .908 |

Every trained learner nearly stops answering `other`, and that alone is worth about 5 points of the zero-shot error. On the questions where nobody answers `other`, direct_brier still leads positive_sft by 1.9 and 1.3 points at N 2000 and 5000 and trails full_sft by 0.0 and 1.3, so its gains are not only the `other` effect. These subset numbers have no interval.

**Coverage** at confidence thresholds .80, .90 and .95, using the response confidence field (1 - H / ln K), not the top-1 probability that ECE uses; accuracy of the kept answers in brackets (`S:coverage.<run>`):

| run | .80 | .90 | .95 |
|---|---|---|---|
| zero-shot | .772 (.948) | .688 (.967) | .607 (.978) |
| temperature n500 / n5000 | .563 (.981) / .607 (.976) | .347 (.990) / .438 (.988) | .080 (.996) / .189 (.998) |
| positive_sft n500 | .940 (.917) | .919 (.928) | .902 (.933) |
| positive_sft n5000 | .958 (.928) | .933 (.938) | .916 (.942) |
| direct_brier n500 | .779 (.968) | .686 (.981) | .594 (.991) |
| direct_brier n2000 | .818 (.976) | .742 (.985) | .649 (.990) |
| direct_brier n5000 seed 0 | .868 (.974) | .809 (.983) | .743 (.986) |
| full_sft n500 | .857 (.958) | .784 (.974) | .719 (.985) |
| full_sft n5000 | .947 (.973) | .918 (.979) | .888 (.982) |

positive_sft keeps over 90 percent of answers at the .95 threshold with 93 to 94 percent accuracy among them, the overconfidence of prediction 2 in a form a cascade would feel: it would pass on about 6 percent errors as confident. direct_brier keeps fewer answers at every threshold than full_sft at the same N, at similar accuracy among those kept. The temperature keeps the fewest.

## 7. The B2 reference

gpt-4.1-mini (B2) exists only on the 500-record subset of test_banking77 (V3_DESIGN note 15): accuracy .918, ECE .035 (`S:b2_reference.b2`). The learners on the same 500 records (`S:b2_reference.runs`):

| run | accuracy | ECE |
|---|---|---|
| zero-shot | .846 | .078 |
| temperature n500 / n5000 | .846 / .846 | .048 / .041 |
| positive_sft n500 / n2000 / n5000 | .884 / .886 / .916 | .093 / .086 / .070 |
| direct_brier n500 / n2000 / n5000 seed 0 | .888 / .910 / .934 | .038 / .028 / .034 |
| direct_brier n5000 seed 1 / seed 2 / seed 1 log | .936 / .926 / .934 | .036 / .029 / .031 |
| full_sft n500 / n2000 / n5000 | .904 / .916 / .948 | .050 / .031 / .032 |

From N 5000, every direct_brier run is above gpt-4.1-mini's .918 on this subset, from 5000 logged correctness bits and no labels; positive_sft reaches .916 and full_sft .948. With 500 records, a 1.6-point difference is within sampling error, and these rows have no paired interval against B2.

## 8. The four predictions

The predictions as stated in docs/V3_DESIGN.md section 7, each with its verdict.

1. **"RLCD beats positive-only SFT in accuracy at every N, because it uses negative feedback."** Not held at every N: a tie at N 500, +.002 [-.006, +.009]; held from N 2000, +.027 [+.019, +.035] at N 2000 and +.024 [+.016, +.031] at N 5000, +.026 [+.019, +.033] for the three-seed mean (section 2, section 3). V3_DESIGN says that if prediction 1 fails, v2 and v3 together are a complete negative result about RLCD as reconstructed. It fails only at the smallest N and by a tie, and holds clearly from N 2000, so v3 is not that complete negative result: with 1800 or more logged interactions, negative feedback adds 2.4 to 2.7 points over discarding it.
2. **"RLCD's ECE is lower than positive-only SFT's, which only ever sees confirmed answers and should be overconfident."** Held at every N: -.060 [-.066, -.048], -.060 [-.067, -.050] and -.052 [-.060, -.043], with Brier and NLL agreeing; positive_sft is the worst-calibrated learner at every N, and its coverage at .95 keeps 90 percent of answers at 93 to 94 percent accuracy (sections 2 and 6).
3. **"RLCD approaches full-label SFT as N grows; the gap at N 5000 is the price of not having labels."** Failed: the accuracy gap is -.017 [-.023, -.011], -.007 [-.014, +.000] and -.018 [-.025, -.011] at N 500, 2000 and 5000; it does not shrink with N. The price of not having labels at N 5000 is 1.6 points of accuracy for the three-seed mean, -.016 [-.022, -.010], with ECE level, -.005 [-.009, +.001], and higher Brier and NLL, +.027 [+.018, +.036] and +.062 [+.040, +.084] (section 3).
4. **"Under noisy feedback, RLCD degrades less than positive-only SFT."** Held for accuracy, failed for calibration. Accuracy: direct_brier loses 2.2 points and positive_sft 3.9, a difference of +.017 [+.007, +.028]. Calibration: direct_brier's ECE rises by .138 while positive_sft's falls by .029, a difference of +.167 [+.151, +.182], with NLL +.277 [+.233, +.320] and Brier +.013 [+.000, +.025]. The calibration failure is the proper score calibrating to the noisy outcome it was given; inverting the known flip rate brings ECE from .164 to .051 to .066, not to the clean .026 (section 4).

## 9. Limitations

- **One domain.** Banking77 only, with 10 options and gold always present, so `other` is never right and "never answer other" is free to learn; other domains, other option counts or a gold-absent construction may differ.
- **One logging policy.** sft_06b at epsilon 0.1, already .836 accurate on the log domain; a weaker or more exploratory policy changes how many negatives there are and which ones.
- **A simulated feedback channel.** Outcomes come from gold labels, revealed perfectly or flipped symmetrically at a known rate; real deployment feedback is delayed, biased towards some errors and of unknown noise rate, which the inversion needs.
- **One size.** 0.6B only (decision 56).
- **full_sft and positive_sft have one seed.** Their sides of every comparison carry no seed variance; direct_brier's three seeds suggest it is small on Banking77 accuracy.
- **Forgetting varies by seed** (-.027 to -.043 in-domain for direct_brier at N 5000), more than the Banking77 result does, and beta 0 leaves it unconstrained.
- **Selection on the log's last 0.1 N** chose early steps for every learner; a different selection slice or rule could change the ranking at N 500, where 50 interactions select.

## 10. What v3 shows and what it does not

v3 shows that when a deployed model sees only the correctness of its own actions on a new domain, training on that signal with a pathwise Brier loss beats discarding the failures, by 2.4 to 2.7 points of accuracy from 2000 logged interactions and by about .05 to .06 ECE at every N, and that it closes 70 to 90 percent of the accuracy gap between zero-shot and full-label SFT. It does not show that RLCD approaches full labels as feedback grows, since the 1.6-point gap at N 5000 is no smaller than at N 500, and it does not show calibration robust to noisy feedback, where a proper score calibrates to the channel and needs a known noise rate to be undone. Together with v2, the claim that survives is narrow: RLCD adds value over positive-only SFT when the only feedback is partial, and nowhere else that was tested.
