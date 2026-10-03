# jevmark: the story so far

Every number cites its source: `R1` is docs/RESULTS_v1.md, `R2` docs/RESULTS_v2.md, `R3` docs/RESULTS_v3.md, `D<n>` decision n in docs/DECISIONS.md; each report names the `runs/.../metrics.json` key behind its numbers.

## What it is

jevmark is a decision model: a caller sends a text and typed questions (yes or no, one of the options given in the request, or a level on a scale defined in the request) and gets a probability distribution per question from one forward pass, with no text generated. It is Qwen3-0.6B-Base or Qwen3-1.7B-Base with a LoRA adapter. Each answer is read from the letter logits at the question's answer slot.

## Why a decision model rather than an LLM call

Routing, tool choice, reranking and completion checks in an agent need a choice and a confidence, not text. An LLM call pays for decoding and must be parsed; a forward pass returns a distribution that cannot be malformed, which the caller thresholds to decide what goes to an LLM or a human. On a T4 that is 47.7 to 74.8 ms per request against about 3 s for same-size JSON generation (last question below).

## What was built

- **Engine.** Per-segment encoding that pins each answer slot to a token boundary (D17), a readout over the option letters, and `systemone()` behind a fixed contract (docs/API_SPEC.md).
- **Data with leak gates.** CLINC150 and SST-5 for training, 20 held-out intents, four schemas never trained on (AG News, emotion, Banking77, Yelp). Before any GPU run, `make data` must pass linear and tree leak probes against shuffled-target nulls and a train-test duplicate check (D42, D43).
- **Evaluation protocol.** All nine splits in full for every run; bootstrap intervals by record; ECE, Brier, NLL, coverage, negation symmetry, letter bias, order sensitivity; per-question results kept so metrics recompute without a GPU (D37, D46).
- **Baselines.** The frozen base, the same-size instruct model generating JSON (B1), and gpt-4.1-mini with a strict schema (B2), on one committed subset of 500 records per split (D47).
- **Pipeline safeguards.** fp16 NaN check with fp32 fallback, pre-flight memory check, stale-state guard, fail-fast notebook cells, commit and data hashes in every `metrics.json` (D15, D35, D45).

## v1 results

- In-domain accuracy .956 (0.6B) and .952 (1.7B) at ECE .013 and .015, against .492 and .549 for the frozen bases (R1 section 2).
- On the 20 held-out intents, choice accuracy is .868 and .885, against .975 and .974 in-domain (R1 section 3).
- On the subset, SFT beats same-size JSON generation by 25 to 33 points in-domain and on unseen intents, and beats gpt-4.1-mini in-domain (.944 and .945 against .908). gpt-4.1-mini is ahead on Banking77 (.918 against .846 and .850) and Yelp (.600 against .510 and .518) (R1 section 4).
- Every trained noul kind answers its negated question consistently in at least 90 percent of pairs (R1 section 3).
- Off-distribution, SFT is overconfident: at 0.6B, ECE is .164, .207 and .169 on AG News, emotion and Yelp, against .037, .045 and .070 for the frozen base (R1 section 2).

## v2 results: negative on full labels, and what they taught

Stages 1 and 2a (five arms at 0.6B, three seeds) are complete; stage 2b (1.7B) is kept but has not run, and stage 3 was cancelled in favour of v3 (D56).

- **A proper score used as a REINFORCE reward is broken.** A wrong action a has p_a at most 1 - p_gold, so its Brier or log reward is never below gold's; the group-mean advantage pushes probability away from gold when K > 2 and is zero for nouls. SST-5 score accuracy fell from .636 to .111, and a simulation found the update lowering the gold logit in 24.0 percent of questions (R2 section 2, D52).
- **On deterministic labels, every bandit objective is at most extra SFT.** Once each arm gets its own temperature, the best arm is level with SFT plus T (-.002 [-.006, +.004]), and so is the control, more cross-entropy on the same records (+.002 [-.001, +.009]) (R2 section 3).
- **Outcome-only reward destroys calibration.** Expected probability of the chosen action reaches .983 to .988 and unseen-schema ECE .239 to .262; even with T 2.5 to 2.9 it stays above SFT plus T (R2 section 3).
- **Temperature is the strongest cheap fix.** One scalar fitted on in-domain valid brings the four-schema mean ECE from .152 to .121 and in-domain ECE from .013 to .005. No RLCD arm gets below .121 without a temperature of its own (R2 sections 1 and 3).
- **Seed variance exceeds record-level intervals.** The four-schema mean moves by .023 to .041 between seeds, 1.4 to 3.5 times its bootstrap interval's width; seed 0 alone overstated the gap by .012 to .020 (R2 section 3).

## v3 results: partial feedback on a new domain

sft_06b answered 9942 Banking77 train messages once with epsilon 0.1 exploration and only the correctness of its chosen option was logged; five learners used the first N interactions, N 500, 2000 and 5000, at equal steps, 14 trained runs at 0.6B (R3 section 1, D56).

- **Negative feedback helps once there is enough of it.** RLCD (pathwise Brier on the logged action) ties positive-only SFT at N 500, +.002 [-.006, +.009], and beats it by 2.7 and 2.4 points at N 2000 and 5000, +.026 [+.019, +.033] for the three-seed mean at N 5000 (R3 sections 2 and 3).
- **It is the calibrated way to use the log.** RLCD's ECE is .052 to .060 below positive-only SFT's at every N; positive-only SFT, trained on confirmed answers alone, is the worst-calibrated learner and keeps 90 percent of its answers at the .95 confidence threshold with 93 to 94 percent accuracy (R3 sections 2 and 6).
- **Labels still win.** Full-label SFT stays ahead by 0.7 to 1.8 points and the gap does not shrink with N: 1.6 points [1.0, 2.2] at N 5000 for the seed mean, at level ECE and higher Brier and NLL (R3 section 3).
- **Under noisy feedback a proper score calibrates to the channel.** With outcomes flipped at 0.2, RLCD loses less accuracy than positive-only SFT (+.017 [+.007, +.028] relative), but its ECE goes from .026 to .164, because it learns P(revealed outcome) rather than P(correct). Inverting the known flip rate brings ECE to .051 to .066, not back to .026 (R3 section 4, D57).
- **Adaptation costs the original domain.** At N 5000 RLCD loses 2.7 to 4.3 in-domain points across seeds and full-label SFT 5.2, while positive-only SFT loses under 1; a fitted temperature loses nothing and gives the lowest Banking77 ECE, with no accuracy gain (R3 section 5).

## Pending

Stage 2b (the five RLCD arms on 1.7B) is kept, ranked after v3 (D56). The delta-filing integration is an optional demonstration and needs the human's go-ahead.

## Five lessons about data and pipelines

1. **A model uses the surface form when that is enough.** On v1, sft_06b gave the same answer to a domain question and its negation in 89 percent of pairs. Negating half of all nouls (v1.1) then let the phrasing alone predict `out_of_scope` 85.8 percent of the time in train. Balance must hold within each phrasing and answer; phrasing-only accuracy is now a build check capped at 55 percent (D40, D41).
2. **Question order can leak a label.** In v1.2 a sentiment noul before the score told the model the text was not neutral. Attention is causal, so v1.3 allows one gold-dependent question, after the choice or score. The probes find the v1.2 leaks at +5 to +22 points and nothing above +3.0 on v1.3 (R1 section 3, D42).
3. **A short GPU cycle pays for itself if its sample is right.** The 300-step fast cycle takes about 24 minutes. Its first version evaluated the first 300 records of each file, which covered 3 to 16 intents and no out-of-scope utterance, so the sampler is now stratified (docs/TASKS.md 1.5b).
4. **Check memory on the worst case before the first step.** The first full v1.3 run ran out of memory at step 392 of 1378 on longer records at micro-batch 16. A pre-flight pass on the longest micro-batch now fails in the first minute instead, and 0.6B trains at 8 x 4 (D45).
5. **A file in git is history, not state.** A failing `!python` does not stop a Kaggle cell, and the completion check only tested that `train_summary.json` existed; a committed v1.2 summary passed it, and a step-200 adapter was evaluated as `sft_06b`. Training now refuses to start over old run files, and notebooks check exit code, file time and step count (D45).

## Questions and answers

**How is this different from a fine-tuned classifier?** The options are input, not a fixed output layer: .868 choice accuracy on intents never seen in training, .851 on Banking77's untrained labels (R1 sections 2 and 3).

**From distilling a generative model?** Nothing is generated: the answer is a distribution over the request's options from one forward pass, trained with cross-entropy on the option letters, not imitation of text. It cannot fail to parse; same-size JSON generation fails on up to 14.7 percent of questions (R1 section 4).

**From prompting an LLM?** At 0.6B it is 16 times faster than gpt-4.1-mini's median and costs about 1/76 as much per request, and it is more accurate in-domain (.944 against .908). It is less accurate on all four unseen schemas at 0.6B and on three at 1.7B. The cascade (send low-confidence answers to the LLM) is the design for that gap (R1 section 4, docs/PLAN.md).

**When should RLCD help, and when can it not?** It cannot help when every question has a deterministic gold label: a bandit outcome carries less information than the label, and a temperature fixes off-distribution overconfidence better than any arm (R2 sections 1 and 3). It does help when feedback is partial: on Banking77, from a log of the model's own actions with only their correctness revealed, it beats training on the confirmed answers alone by 2.4 to 2.7 points from 2000 interactions and is far better calibrated, though it ties at 500 and stays 1.6 points behind full labels at 5000 (R3 sections 2 and 3). Under noisy feedback it calibrates to the noise, which is invertible only when the noise rate is known (R3 section 4).

**What are the cost and latency figures, and what do they assume?** Batch-1 median on one T4: 47.7 ms (sft_06b), 74.8 ms (sft_17b), 3231 and 2795 ms (B1 0.6B and 1.7B). B2 has a 777 ms median and 1095 ms p95, which include the network from a laptop. Per 1000 requests: 0.0032 and 0.0072 USD for jevmark and 0.030 USD for B1, assuming a T4 at 0.35 USD per hour (an assumption, not a measured price) at full batch-16 use. B2 cost 0.2431 USD per 1000 at 0.40, 0.10 and 1.60 USD per million input, cached and output tokens (R1 section 4 and addendum).
