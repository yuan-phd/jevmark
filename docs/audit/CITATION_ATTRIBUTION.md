# Citation attribution check

Scope: every sentence in docs/PAPER.md and paper/main.tex (same text) that says what a cited work found, showed, proposed, measured or is. The bibliographic entries were verified earlier against primary sources; this check is about the one-line characterisation only.

Source of support: the summary of each work in the related-work survey compiled for this paper (kept locally, not in the repository), quoted or closely paraphrased in the third column. No new sources were consulted and no claims were added.

Classes:
- **S**, supported by the survey summary;
- **T**, stronger than the survey summary;
- **N**, not covered by the survey.

Every T and N sentence was rewritten to say only what the survey supports, or left as a bare citation with no characterisation. Where a sentence carries several citations, each claim is a row. Section numbers are those of the paper.

## Counts

| class | rows | outcome |
|---|---|---|
| S | 49 | unchanged |
| T | 6 | rewritten (6) |
| N | 2 | rewritten (1), already a bare citation and unchanged (1) |
| total | 57 | 7 sentences changed in each file |

No number was added, removed or changed: the multiset of numbers in docs/PAPER.md is identical before and after, so the claims audit result on PAPER.md is unchanged.

## Rows

| # | where | sentence (as checked) | citation | survey support | class | outcome |
|---|---|---|---|---|---|---|
| 1 | Abstract | "replicating a concurrent finding under a stricter protocol" | Gao and Wang 2026 (by name in the text) | Claim 2 "largely anticipated" by OpenJev-RLCD; "frame yours as a replication under a transfer protocol" | S | unchanged |
| 2 | 1 | The setting is the one TypeSafe AI announced with Jev, trained by an unpublished recipe called RLCD | TypeSafe AI 2026 | "Announces Jev ... trained with reinforcement learning for calibrated decisions (RLCD); no algorithm, reward or reliability data published" | S | unchanged |
| 3 | 1, claim 1 | A detached, parameter-dependent reward drops the pathwise term of the gradient | Schulman et al. 2015; Mohamed et al. 2020 | "Known in general form ... the correct gradient adds the pathwise term through the reward (Schulman et al. 2015; Mohamed et al. 2020)" | S | unchanged |
| 4 | 1, claim 1 | OpenJev-RLCD reaches the same recipe, "pathwise gradient only" and the score-function term harmful, by a gradient-norm argument; its REINFORCE term acts on a sampled rationale | Gao and Wang 2026 | "stage 1 is the pathwise gradient ... stage 2 adds a down-weighted REINFORCE term with a KL anchor; the score-function term has 7 to 12 times the gradient norm ... more of it is monotonically worse"; "reaches the pathwise conclusion by a variance and gradient-norm argument"; "scores each rationale's readout" | T: "only" is stronger, since the survey describes a stage 2 that keeps a down-weighted REINFORCE term | rewritten |
| 5 | 1, claim 1 | REINFORCE with a proper-score reward works when the forecast is a separately emitted probability | Bereket and Leskovec 2025; Damani et al. 2026 | "When the proper score is applied to a separately emitted probability ..., REINFORCE with a proper-score reward and an unbiased baseline does yield calibrated models (Band et al.; Bereket and Leskovec; RLCR)" | S | unchanged |
| 6 | 1, claim 1 | Bereket and Leskovec trace GRPO's overconfidence to dividing the advantage by the group standard deviation | Bereket and Leskovec 2025 | "GRPO's standard-deviation normalisation introduces a policy-dependent bias in the advantage that drives predictions to the extremes" | S | unchanged |
| 7 | 1, claim 2 | OpenJev-RLCD's direct-scorer ablation: proper-score RL never beat cross-entropy on CLINC150 (94.6 against 96.3), RLVR collapsed to the mode | Gao and Wang 2026 | "explicit-probability RLCD on a direct scorer never beat cross-entropy on CLINC150 (94.6 versus 96.3 accuracy) and RLVR collapsed to the mode" | S | unchanged |
| 8 | 1, claim 3 | CAPS is the direct method, the regression-oracle objective | Dudík, Langford and Li 2011; Foster et al. 2018; Foster and Rakhlin 2020 | "Names the direct method (DM: fit a reward regressor on logged context, action, reward)"; "Reduces contextual bandits to online regression on observed rewards"; "the squared loss on the chosen action's reward is the Brier score on the chosen action" | S | unchanged |
| 9 | 1, claim 3 | Positive-only fine-tuning diagnosed as positive-unlabeled learning | Elkan and Noto 2008 | "anticipates the diagnosis of positive-only fine-tuning ... as PU learning with discarded negatives" | S | unchanged |
| 10 | 1, claim 3 | No gain from propensity weights, the expected behaviour of a well-specified regression | Shimodaira 2000; Byrd and Lipton 2019 | "for a well-specified conditional model the unweighted MLE is already consistent"; "Importance weights have vanishing effect on overparameterised networks trained to convergence" | S | unchanged |
| 11 | 1, claim 3 | Backward correction with its 1 / (1 - 2f) factor | Natarajan et al. 2013; Patrini et al. 2017; Angluin and Laird 1988 | "the canonical citation for the inversion"; "Backward correction (multiply the loss vector by T inverse)"; "The 1 / (1 - 2f) amplification goes back to Angluin and Laird 1988" | S | unchanged |
| 12 | 2, para 1 | Jev and OpenJev-RLCD "answer typed questions about a state in one forward pass, with a confidence the caller thresholds" | TypeSafe AI 2026; Gao and Wang 2026 | Jev: "answers typed questions about a state with calibrated probabilities from a parallel sampler"; OpenJev-RLCD: "Reads the softmax over option tokens after an Answer marker; scores each rationale's readout". Neither summary says one forward pass or a thresholding caller | T | rewritten |
| 13 | 2, para 1 | "Their use is the cascade, in which a cheap model answers what it is sure of and defers the rest" | Chen, Zaharia and Zou 2023 | FrugalGPT: "LLM cascades with a learned scorer that decides whether a cheaper answer is acceptable". The survey does not say Jev or OpenJev-RLCD are used in a cascade | N (the attribution of the use to Jev and OpenJev-RLCD) | rewritten |
| 14 | 2, para 1 | "confidence trained into the small model gives the better deferral trade-off" | Chuang et al. 2025 | "Fine-tunes confidence tokens into a small LLM and uses them for routing and rejection". The better trade-off is reported by a different, uncited paper | T | rewritten |
| 15 | 2, para 1 | Thresholding the top posterior is selective classification, evaluated by its risk-coverage curve | Geifman and El-Yaniv 2017 | "Risk-coverage curves and selection by softmax response" | S | unchanged |
| 16 | 2, para 1 | "Option-letter readout has known position and order biases" | Zheng et al. 2024 | "Documents selection bias toward specific option IDs (token bias), proposes PriDe". Option-order sensitivity is Pezeshkpour and Hruschka 2024, not cited | T | rewritten |
| 17 | 2, para 1 | Unlike a confidence stated in words | Xiong et al. 2024 | "Verbalized confidence is overconfident; consistency-based elicitation helps" | S | unchanged |
| 18 | 2, para 2 | CAPS is the direct method and the regression-oracle objective | Dudík, Langford and Li 2011; Foster et al. 2018; Foster and Rakhlin 2020 | as row 8 | S | unchanged |
| 19 | 2, para 2 | A contextual-bandit bake-off found regression competitive without propensity weights | Bietti, Agarwal and Langford 2021 | "a greedy regression-oracle learner (fit a regressor on observed rewards, no importance weighting) is a close second overall" | S | unchanged |
| 20 | 2, para 2 | Counterfactual risk minimisation weights the policy objective by inverse propensities, "with variance penalties, self-normalisation and clipping" | Swaminathan and Joachims 2015a, 2015b; Joachims, Swaminathan and de Rijke 2018 | "propensity-weighted empirical risk, and the variance-regularised CRM principle"; "proposes self-normalised IPS"; "a self-normalised, baseline-shifted IPS objective". Clipped IPS is attributed to Bottou et al. 2013, not cited | T | rewritten |
| 21 | 2, para 2 | Doubly robust estimators combine the two | Dudík, Langford and Li 2011 | "Names the direct method ..., IPS, and the doubly robust estimator" | S | unchanged |
| 22 | 2, para 2 | Simulating bandit feedback from a labelled multiclass dataset is the standard protocol | Beygelzimer and Langford 2009 | "Introduces ... the protocol of converting a supervised multiclass dataset into bandit data by revealing only the chosen action's loss" | S | unchanged |
| 23 | 2, para 2 | Importance weights are unnecessary for a well-specified conditional model | Shimodaira 2000 | "for a well-specified conditional model the unweighted MLE is already consistent" | S | unchanged |
| 24 | 2, para 2 | Importance weights have little effect on deep networks trained to convergence | Byrd and Lipton 2019 | "Importance weights have vanishing effect on overparameterised networks trained to convergence" | S | unchanged |
| 25 | 2, para 2 | Positive-unlabeled learning recovers the posterior only up to a constant | Elkan and Noto 2008 | "estimates c times p(y=1 \| x) for a constant c" | S | unchanged |
| 26 | 2, para 2 | Self-training on filtered successes | Zelikman et al. 2022 | "Positive-only fine-tuning on self-generated correct trajectories, the LLM-era form of your positive-only baseline" | S | unchanged |
| 27 | 2, para 3 | A strictly proper score is minimised in expectation by the true probability | Brier 1950; Gneiting and Raftery 2007 | "a scoring rule is strictly proper iff the forecaster's expected score is uniquely maximised by the true distribution; Brier and log score are strictly proper"; Brier 1950: "Origin of the Brier score" | S | unchanged |
| 28 | 2, para 3 | Binned ECE is biased, so Brier and NLL are reported beside it | Pakdaman Naeini et al. 2015; Nixon et al. 2019 | "Introduces ECE and MCE as binned reliability summaries"; "Binned ECE is biased and bin-sensitive; ... pair with a proper score" | S | unchanged |
| 29 | 2, para 3 | One temperature repairs calibration post hoc | Guo et al. 2017 | "a single post-hoc temperature fitted on held-out NLL repairs ECE without changing accuracy" | S | unchanged |
| 30 | 2, para 3 | As RL rewards on a verbalized confidence, proper scores yield calibrated models | Damani et al. 2026 | "RLCR adds a Brier score of a verbalized confidence to the binary correctness reward ... RLVR degrades calibration while RLCR repairs it", and the passage quoted in row 5 | S | unchanged |
| 31 | 2, para 3 | The estimator, not the score, decides calibration: an unbiased group baseline gives calibrated forecasts, std normalisation drives them to the extremes | Bereket and Leskovec 2025 | "score-function RL with an unbiased group baseline ... yields calibrated predictions ..., while GRPO's standard-deviation normalisation ... drives predictions to the extremes"; "the finding that the estimator, not the score, decides calibration" | S | unchanged |
| 32 | 2, para 4 | The general rule of stochastic computation graphs | Schulman et al. 2015; Mohamed et al. 2020 | as row 3 | S | unchanged |
| 33 | 2, para 4 | REINFORCE, with the group baseline of GRPO | Williams 1992; Shao et al. 2024 | "REINFORCE and the baseline"; "Defines GRPO with the group-mean, group-std normalised advantage" | S | unchanged |
| 34 | 2, para 4 | Outcome-only rewards degrade calibration | Damani et al. 2026 | "shows RLVR degrades calibration" | S | unchanged |
| 35 | 2, para 4 | OpenJev-RLCD optimises the readout distribution by the pathwise gradient (its deterministic policy gradient), and finds the score-function term, which acts on a sampled rationale, at 7 to 12 times the pathwise term's gradient norm and more of it monotonically worse | Gao and Wang 2026 | "stage 1 is the pathwise gradient of the proper score (called the deterministic policy gradient with respect to the reported distribution) ... the score-function term has 7 to 12 times the gradient norm of the pathwise term and more of it is monotonically worse"; "scores each rationale's readout" | S (the rationale detail, inferred from "each rationale's readout", confirmed by the author against the paper) | unchanged |
| 36 | 2, para 4 | Its direct-scorer ablation (repeat of row 7) | Gao and Wang 2026 | as row 7 | S | unchanged |
| 37 | 2, para 5 | Flipped feedback at a known rate is random classification noise | Angluin and Laird 1988 | "Random classification noise model" | S | unchanged |
| 38 | 2, para 5 | Symmetric case of the class-conditional identity, the clean posterior an affine transform of the corrupted one | Natarajan et al. 2013; Menon et al. 2015 | "the noisy posterior is an affine function of the clean one"; "the clean class-probability is a known monotone, affine transform of the corrupted class-probability" | S | unchanged |
| 39 | 2, para 5 | Inverting is backward correction with a known transition matrix, whose error grows by the 1 / (1 - 2f) factor | Patrini et al. 2017; Angluin and Laird 1988 | "backward correction with a known transition matrix is Patrini et al. 2017; the 1 / (1 - 2f) amplification is Angluin and Laird 1988" | S | unchanged |
| 40 | 2, para 5 | Flipped bandit feedback with a known rate for an online linear learner | Agarwal and Manwani 2021 | "Bandit feedback ... flipped with known probability"; "the noisy-channel bandit setting itself (online, linear learner)" | S | unchanged |
| 41 | 2, para 5 | Networks memorise noisy labels given enough passes | Zhang et al. 2017 | "Networks fit random labels perfectly given enough passes"; "supports the memorisation of a fixed noisy log under many passes" | S | unchanged |
| 42 | 2, para 5 | Early stopping is the provable remedy | Li, Soltanolkotabi and Oymak 2020 | "Formal early-stopping robustness result ... supports checkpoint selection as the fix" | S | unchanged |
| 43 | 2, para 6 | A single temperature is strong in-domain and fragile under shift | Ovadia et al. 2019 | "post-hoc temperature scaling fit in-distribution does not protect under shift"; "a single temperature is strong in-domain and fragile under shift (Ovadia et al. 2019)" | S | unchanged |
| 44 | 2, para 6 | LoRA forgets less than full fine-tuning but still forgets | Biderman et al. 2024 | "LoRA forgets less than full fine-tuning but still forgets (Biderman et al. 2024)" | S | unchanged |
| 45 | 2, para 7 | Leakage detection by exploratory models on non-target features, against a permutation null | Kaufman et al. 2012; Ojala and Garriga 2010 | "proposes detection by exploratory models on suspect features"; "Formal permutation (label-shuffling) null for classifier accuracy" | S | unchanged |
| 46 | 2, para 7 | Benchmark test sets contain label errors | Northcutt, Athalye and Mueller 2021 | "Average 3.3 percent test-set label errors across ten benchmarks" | S | unchanged |
| 47 | 2, para 7 | For Banking77, about 1,400 of 13,083 utterances flagged, about 11 percent | Ying and Thomas 2022 | "about 1,400 of 13,083 utterances flagged, 11 percent" | S | unchanged |
| 48 | 2, para 7 | The emotion dataset is distantly supervised from hashtags | Saravia et al. 2018 | "tweets labelled by distant supervision from emotion hashtags, six classes" | S | unchanged |
| 49 | 2, para 7 | "which human validation puts at about 90 percent label accuracy" (the emotion dataset's hashtag labels) | Abdul-Mageed and Ungar 2017 | "Hashtag distant supervision for emotion with human validation at about 90 percent label accuracy"; "the hashtag-label literature itself reports only about 90 percent agreement". The validation is of their own data, not of the emotion dataset | T | rewritten |
| 50 | 3, Identity | CAPS is the direct method; it is OpenJev-RLCD's stage 1 specialised to partial feedback, stage 1 scoring the whole readout distribution against the label | Dudík, Langford and Li 2011; Gao and Wang 2026 | as row 8; "Reads the softmax over option tokens ...; scores each rationale's readout with a strictly proper score ... stage 1 is the pathwise gradient of the proper score" | S | unchanged |
| 51 | 3, propensity | The unweighted fit is consistent for a well-specified model | Shimodaira 2000 | as row 23 | S | unchanged |
| 52 | 3, corollary | Detaching R keeps only the score-function term | Schulman et al. 2015 | "the unbiased gradient is the sum of a score-function term ... and a pathwise term through every deterministic dependence on the parameters, including a reward that depends on the policy" | S | unchanged |
| 53 | 3, after the proof | The standard-deviation normalisation that Bereket and Leskovec identify as GRPO's source of overconfidence; the corollary does not contradict proper-score rewards on a separately emitted confidence | Bereket and Leskovec 2025; Damani et al. 2026 | as rows 5 and 6 | S | unchanged |
| 54 | 4.1 | Model and data sources: LoRA, Qwen3, CLINC150, SST-5, AG News and Yelp, emotion, Banking77 | Hu et al. 2022; Qwen Team 2025; Larson et al. 2019; Socher et al. 2013; Zhang, Zhao and LeCun 2015; Saravia et al. 2018; Casanueva et al. 2020 | LoRA paper; "Qwen3-0.6B-Base and 1.7B-Base"; "CLINC150 (150 intents, 10 domains, out-of-scope class)"; "SST-2 and SST-5"; "AG News, Yelp polarity and Yelp full"; "Source of the dair-ai emotion dataset"; "Introduces Banking77" | S | unchanged |
| 55 | 4.1 | Paired 95 percent record-bootstrap intervals | Efron 1979; Efron and Tibshirani 1993 | not in the survey | N | already a bare method citation with no characterisation; unchanged |
| 56 | 4.1 and 4.2 | Selective-classification metrics; OpenJev-RLCD reports the same direction on CLINC150 with a different training setup | Geifman and El-Yaniv 2017; Gao and Wang 2026 (by name) | "Risk-coverage curves ... AURC-style evaluation"; "say that [OpenJev-RLCD] reports the same for CLINC150 and ChaosNLI with a different training setup" | S | unchanged |
| 57 | 5 and 6 | POEM or BanditNet and a doubly robust arm as the missing arms; CAPS as OpenJev-RLCD's stage 1 restricted to partial feedback; the full-label result consistent with OpenJev-RLCD | Swaminathan and Joachims 2015a; Joachims, Swaminathan and de Rijke 2018; Dudík, Langford and Li 2011; Gao and Wang 2026 (by name) | "the variance-regularised CRM principle with the POEM algorithm"; "BanditNet: CRM for deep networks"; "the doubly robust estimator"; rows 7 and 50 | S | unchanged |

## Rewritten sentences

Row 4 (Section 1, claim 1).
- Before: The concurrent OpenJev-RLCD (Gao and Wang 2026) reaches the same recipe, pathwise gradient only and the score-function term harmful, by a gradient-norm argument for a different object: ...
- After: The concurrent OpenJev-RLCD (Gao and Wang 2026) reaches the same recipe, the pathwise gradient with the score-function term found harmful, by a gradient-norm argument for a different object: ...

Rows 12, 13 and 14 (Section 2, first paragraph).
- Before: Jev (TypeSafe AI 2026) and the other published reconstruction, OpenJev-RLCD (Gao and Wang 2026), answer typed questions about a state in one forward pass, with a confidence the caller thresholds. Their use is the cascade, in which a cheap model answers what it is sure of and defers the rest (Chen, Zaharia and Zou 2023), and confidence trained into the small model gives the better deferral trade-off (Chuang et al. 2025).
- After: Jev (TypeSafe AI 2026) answers typed questions about a state with calibrated probabilities, and the other published reconstruction, OpenJev-RLCD (Gao and Wang 2026), reads the softmax over option tokens after an answer marker. A head of this kind serves a cascade, in which a learned scorer decides whether a cheaper model's answer is acceptable (Chen, Zaharia and Zou 2023), and confidence tokens fine-tuned into a small model can drive routing and rejection (Chuang et al. 2025).

Row 16 (Section 2, first paragraph).
- Before: Option-letter readout has known position and order biases (Zheng et al. 2024), so we randomise option order ...
- After: Option-letter readout is biased towards specific option letters (Zheng et al. 2024), so we randomise option order ...

Row 20 (Section 2, second paragraph).
- Before: Counterfactual risk minimisation instead weights the policy objective by inverse propensities, with variance penalties, self-normalisation and clipping (Swaminathan and Joachims 2015a; ...).
- After: Counterfactual risk minimisation instead weights the policy objective by inverse propensities, with variance penalties and self-normalisation (Swaminathan and Joachims 2015a; ...).

Row 49 (Section 2, last paragraph).
- Before: The emotion dataset is distantly supervised from hashtags (Saravia et al. 2018), which human validation puts at about 90 percent label accuracy (Abdul-Mageed and Ungar 2017); our audit finds a problem in a third of a uniform sample.
- After: The emotion dataset is distantly supervised from hashtags (Saravia et al. 2018); for their own hashtag-labelled emotion data, Abdul-Mageed and Ungar (2017) report human validation at about 90 percent label accuracy, and our audit finds a problem in a third of a uniform sample.

## Confirmed by the author

Row 35 (and the same detail in row 4): that OpenJev-RLCD's score-function term acts on a sampled rationale was inferred from the survey's "scores each rationale's readout" and stage 2's REINFORCE term, not stated in those words. The author checked it against the OpenJev-RLCD paper and confirmed it, so the paper's argument that the two results concern a different object stands as written.
