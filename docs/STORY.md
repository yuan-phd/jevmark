# jevmark: the story

Every number cites its source: `R1`, `R2` and `R3` are docs/RESULTS_v1.md, RESULTS_v2.md and RESULTS_v3.md, `D<n>` is decision n in docs/DECISIONS.md; each report names the `runs/.../metrics.json` key behind its numbers. Tags `v1`, `v2` and `v3` mark the end of each phase.

## What it is, and why

jevmark is a decision model: a caller sends a text and typed questions (yes or no, one of the options given in the request, or a level on a scale defined in the request) and gets a probability distribution per question from one forward pass, with no text generated. It is Qwen3-0.6B-Base or Qwen3-1.7B-Base with a LoRA adapter, read from the letter logits at each question's answer slot. An agent's routing and tool choices need a choice and a confidence, not text; the caller thresholds the confidence to decide what goes to an LLM or a human.

## What was built

Per-segment encoding that pins each answer slot to a token boundary and a fixed contract (D17, docs/API_SPEC.md); CLINC150 and SST-5 for training with leak probes and a duplicate check before any GPU run (D42, D43); every split evaluated in full with bootstrap intervals and per-question results kept for CPU recomputation (D37, D46); three baselines on one committed subset (D47); LoRA SFT, RLCD arms (D51, D52) and, for v3, learners trained from a deployment log (D56).

## v1: the supervised model works in-domain and reads the options

- In-domain accuracy .956 (0.6B) and .952 (1.7B) at ECE .013 and .015, against .492 and .549 for the frozen bases (R1 section 2).
- On 20 intents never trained on, choice accuracy is .868 and .885, against .975 and .974 in-domain (R1 section 3).
- On the subset it beats same-size JSON generation by 25 to 33 points in-domain and on unseen intents, and gpt-4.1-mini in-domain (.944 and .945 against .908); gpt-4.1-mini leads on Banking77 (.918 against .846 and .850) and Yelp (R1 section 4).
- Off-distribution it is overconfident: at 0.6B, ECE .164, .207 and .169 on AG News, emotion and Yelp (R1 section 2).

## v2: RLCD on full labels is negative

- **A proper score as a REINFORCE reward is broken.** A wrong action's Brier or log reward is never below gold's, so the group-mean advantage pushes probability away from gold when K > 2; SST-5 score accuracy fell from .636 to .111 (R2 section 2, D52).
- **On deterministic labels a bandit objective is at most extra SFT.** With its own temperature, the best arm is level with SFT plus T (-.002 [-.006, +.004]), and so is the control, more cross-entropy (+.002 [-.001, +.009]) (R2 section 3).
- **A temperature is the strongest cheap fix.** One scalar fitted in-domain brings the four-schema mean ECE from .152 to .121; no arm gets below .121 without a temperature of its own, and outcome-only reward pushes it to .239 to .262 (R2 sections 1 and 3).

Stage 2b (the arms on 1.7B) was not run, so this conclusion is for 0.6B (D58); stage 3 was cancelled in favour of v3 (D56).

## v3: partial feedback on a new domain is where RLCD helps

sft_06b answered 9942 Banking77 train messages once with epsilon 0.1 exploration, and only the correctness of its chosen option was logged. Five learners used the first N interactions, N 500, 2000 and 5000, at equal steps: 14 trained runs at 0.6B (R3 section 1).

- **Negative feedback helps once there is enough of it.** RLCD (pathwise Brier on the logged action) ties positive-only SFT at N 500, +.002 [-.006, +.009], and beats it by 2.7 and 2.4 points at N 2000 and 5000, +.026 [+.019, +.033] for the three-seed mean at N 5000 (R3 sections 2 and 3).
- **It is the calibrated way to use the log.** Its ECE is .052 to .060 below positive-only SFT's at every N; positive-only SFT keeps 90 percent of its answers at the .95 confidence threshold with 93 to 94 percent accuracy (R3 sections 2 and 6).
- **Labels still win.** Full-label SFT stays ahead by 0.7 to 1.8 points and the gap does not shrink with N: 1.5 points [1.0, 2.0] at N 5000, seed means of both on three seeds (R3 section 3).
- **Under noisy feedback a proper score calibrates to the channel.** With outcomes flipped at 0.2, RLCD loses less accuracy than positive-only SFT (+.017 [+.007, +.028] relative), but its ECE goes from .026 to .164, because it learns P(revealed outcome), not P(correct); a known flip rate inverts it. What is left (channel-scale ECE .058 against .014 for a channel-calibrated clean model) is the variance of one noisy draw per interaction: training on the expected outcome instead brings it to .019 to .024. Trained too long, the learner memorises those draws and falls to .813 by step 1407, so the feedback-only early stopping that picked step 250 is essential (R3 section 4, D59).
- **Adaptation costs the original domain.** At N 5000 RLCD loses 2.7 to 4.3 in-domain points across seeds and full-label SFT 5.2, positive-only SFT under 1; a fitted temperature loses nothing and has the lowest or joint-lowest Banking77 ECE, with no accuracy gain (R3 sections 2 and 5).

Predictions: 1 is a tie at N 500 and holds from N 2000, 2 holds, 3 fails, 4 holds for accuracy and fails for calibration as stored (R3 section 8, D57, D59). A blind audit by two language-model judges, no human, finds a label problem in 12 to 14 percent of Banking77 test labels and none of 50 CLINC150 ones, so Banking77's ceiling sits near .96 and even the "clean" feedback carried label noise (R3 section 9, docs/audit/SUMMARY.md).

## Five lessons about data and pipelines

1. **A model uses the surface form when that is enough:** sft_06b answered a domain question and its negation the same way in 89 percent of pairs, so phrasing-only accuracy is now a build check (D40, D41).
2. **Question order can leak a label:** a sentiment noul before the score revealed "not neutral", so a record holds one gold-dependent question, after the choice or score (D42).
3. **A fast GPU cycle pays only if its sample is right:** the first 300 records of a file covered 3 to 16 intents, so the sampler is stratified (docs/TASKS.md 1.5b).
4. **Check memory on the worst case first:** a run that ran out of memory at step 392 now fails in its first minute on a pre-flight pass (D45).
5. **A file in git is history, not state:** a committed summary let a step-200 adapter pass as `sft_06b`, so training refuses to start over old run files (D45).

## Questions and answers

**How is this different from a fine-tuned classifier?** The options are input, not a fixed output layer: .868 choice accuracy on intents never seen in training, .851 on Banking77's untrained labels (R1 sections 2 and 3).

**From distilling a generative model?** Nothing is generated: the answer is a distribution over the request's options from one forward pass, trained with cross-entropy on the option letters. It cannot fail to parse; same-size JSON generation fails on up to 14.7 percent of questions (R1 section 4).

**From prompting an LLM?** At 0.6B it is 16 times faster than gpt-4.1-mini's median and about 1/76 of its cost per request, and more accurate in-domain (.944 against .908). Zero-shot it is less accurate on all four unseen schemas at 0.6B; the cascade, low-confidence answers to the LLM, is the design for that gap (R1 section 4). After v3 adaptation from 5000 logged correctness bits and no labels, every clean RLCD run at N 5000 scores .926 to .936 on the Banking77 subset against gpt-4.1-mini's .918, on 500 records and without a paired interval (R3 section 7).

**When should RLCD help, and when can it not?** Not with deterministic gold labels, where a bandit outcome carries less than the label and a temperature fixes overconfidence better (R2). It helps when only the chosen action's correctness is observed: from 2000 logged interactions it beats training on the confirmed answers alone and is far better calibrated, though full labels stay ahead, and under noisy feedback it calibrates to the noise, which can be undone only at a known noise rate, and memorises single noisy draws unless stopped early on the feedback (R3 sections 2 to 4).

**What do cost and latency assume?** Batch-1 median on one T4: 47.7 ms (sft_06b) and 74.8 ms (sft_17b), against about 3 s for B1 and 777 ms for B2; 0.0032 and 0.0072 USD per 1000 requests assuming a T4 at 0.35 USD per hour at full batch-16 use, an assumption rather than a measured price, against 0.2431 USD for B2 (R1 section 4).

## Pending

Only the optional delta-filing demonstration, which needs the human's go-ahead (D56); stage 2b was closed without a run (D58).
