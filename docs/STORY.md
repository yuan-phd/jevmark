# jevmark: the story

Every number cites its source: `R1`, `R2` and `R3` are docs/RESULTS_v1.md, RESULTS_v2.md and RESULTS_v3.md, and `PAPER` is the paper (arXiv, link to follow); each names the `runs/.../metrics.json` key behind its numbers. Tags `v1`, `v2` and `v3` mark the end of each phase, and `v4-paper` the end of phase 4.

## What it is, and why

jevmark is a decision model: a caller sends a text and typed questions (yes or no, one of the options given in the request, or a level on a scale defined in the request) and gets a probability distribution per question from one forward pass, with no text generated. It is Qwen3-0.6B-Base or Qwen3-1.7B-Base with a LoRA adapter, read from the letter logits at each question's answer slot. An agent's routing and tool choices need a choice and a confidence, not text; the caller thresholds the confidence to decide what goes to an LLM or a human.

## What was built

A fixed contract (docs/API_SPEC.md); CLINC150 and SST-5 for training, behind leak probes and a duplicate check; every split evaluated in full with bootstrap intervals; three baselines on one committed subset; LoRA SFT, RLCD arms and learners trained from a deployment log. Three adapters are at https://huggingface.co/yuanphd/jevmark, and `scripts/demo.py` runs a support message through five questions and a confidence gate (README).

## v1: the supervised model works in-domain and reads the options

- In-domain accuracy .956 (0.6B) and .952 (1.7B) at ECE .013 and .015, against .492 and .549 for the frozen bases (R1 section 2).
- On 20 intents never trained on, choice accuracy is .868 and .885, against .975 and .974 in-domain (R1 section 3).
- On the subset it beats same-size JSON generation by 25 to 33 points in-domain and on unseen intents, and gpt-4.1-mini in-domain (.944 and .945 against .908); gpt-4.1-mini leads on Banking77 (.918 against .846 and .850) and Yelp (R1 section 4).
- Off-distribution it is overconfident: at 0.6B, ECE .164, .207 and .169 on AG News, emotion and Yelp (R1 section 2).

## v2: RLCD on full labels is negative

- **A proper score as a REINFORCE reward is broken.** A wrong action's Brier or log reward is never below gold's, so the group-mean advantage pushes probability away from gold when K > 2; SST-5 score accuracy fell from .636 to .111 (R2 section 2).
- **On deterministic labels a bandit objective is at most extra SFT.** With its own temperature, the best arm is level with SFT plus T (-.002 [-.006, +.004]), and so is the control, more cross-entropy (+.002 [-.001, +.009]) (R2 section 3).
- **A temperature is the strongest cheap fix.** One scalar fitted in-domain brings the four-schema mean ECE from .152 to .121; no arm gets below .121 without a temperature of its own, and outcome-only reward pushes it to .239 to .262 (R2 sections 1 and 3).

Stage 2b (the arms on 1.7B) was not run, so this conclusion is for 0.6B; stage 3 was cancelled in favour of v3.

## v3: partial feedback on a new domain is where RLCD helps

sft_06b answered 9942 Banking77 train messages once with epsilon 0.1 exploration, and only the correctness of its chosen option was logged. Five learners used the first N interactions, N 500, 2000 and 5000, at equal steps: 14 trained runs at 0.6B (R3 section 1).

- **Negative feedback helps once there is enough of it.** RLCD (pathwise Brier on the logged action) ties positive-only SFT at N 500, +.002 [-.006, +.009], and beats it by 2.7 and 2.4 points at N 2000 and 5000, +.026 [+.019, +.033] for the three-seed mean at N 5000 (R3 sections 2 and 3).
- **It is the calibrated way to use the negatives.** Its ECE is .052 to .060 below positive-only SFT's at every N; positive-only SFT keeps 90 percent of its answers at the .95 confidence threshold with 93 to 94 percent accuracy (R3 sections 2 and 6).
- **A temperature on top adds nothing.** Fitted on the logged outcomes, T is .95 to 1.04 and ECE moves by at most .003 (R3 section 3).
- **Labels still win.** Full-label SFT stays ahead by 0.7 to 1.8 points and the gap does not shrink with N: 1.5 points [1.0, 2.0] at N 5000, seed means of both on three seeds (R3 section 3).
- **Under noisy feedback a proper score calibrates to the channel.** With outcomes flipped at 0.2, RLCD loses less accuracy than positive-only SFT (+.017 [+.007, +.028] relative), but its ECE goes from .026 to .164, because it learns P(revealed outcome), not P(correct); a known flip rate inverts it. Trained too long, it memorises the single noisy draw and falls to .813, so early stopping on the feedback is essential (R3 section 4).
- **Adaptation costs the original domain.** At N 5000 RLCD loses 2.7 to 4.3 in-domain points across seeds and full-label SFT 3.6 to 5.2, positive-only SFT under 1 (R3 section 5).

Predictions: 1 is a tie at N 500 and holds from N 2000, 2 holds, 3 fails, 4 holds for accuracy and fails for calibration as stored (R3 section 8). A blind audit by two language-model judges, no human, finds a label problem in 12 to 14 percent of Banking77 test labels and none of 50 CLINC150 ones, so Banking77's ceiling sits near .96 and even the "clean" feedback carried label noise (R3 section 9, docs/audit/SUMMARY.md).

## Phase 4: experiments for a paper

Four experiments tested the v3 result, and a draft paper (arXiv, link to follow) and method statement (docs/METHOD.md) report it under the name Chosen-Action Proper Scoring (CAPS) for RLCD's pathwise Brier on the logged action.

- **Off-policy baselines.** Propensity weights add nothing: weighted positive-only SFT is level with the unweighted one (+.000 [-.003, +.004] at N 5000), and no weighted, REINFORCE or log-score arm beats RLCD on accuracy with an interval excluding 0; a log score in place of Brier is a choice within the same family (R3 section 11).
- **A second domain, emotion.** With a logging policy wrong on more than half the log, RLCD leads positive-only SFT at every N, by 21.7 points [19.7, 23.8] at N 5000, and stays 4.3 points [3.1, 5.6] below full labels (three RLCD seeds against one full-label seed; R3 section 13).
- **Scale, with seeds.** At 1.7B, three RLCD seeds against one seed of each baseline: RLCD leads positive-only SFT by 3.4 points [2.6, 4.1]; against full labels it is level on seed 0, -.4 [-1.0, +.2], and .6 points [.2, 1.1] below as a seed mean, against 1.5 points at 0.6B between three-seed means (R3 section 12).
- **A noise-rate sweep.** At flip rates 0.1, 0.2 and 0.3, backward correction brings ECE to .020, .051 and .068 against the clean run's .026, as the 1 / (1 - 2f) amplification predicts; the smaller accuracy loss than positive-only SFT holds at 0.1 and 0.2 and is gone at 0.3 (R3 section 4).
- **A concurrent reconstruction, and the claims repositioned.** OpenJev-RLCD (Gao and Wang 2026) rebuilt the Jev concept independently and reached the same recipe, the pathwise gradient with the score-function term found harmful, and its direct-scorer ablation found that proper-score RL never beat cross-entropy. The paper therefore states the broken REINFORCE reward as a corollary of a known rule, the full-label result as consistent with that ablation (neither confirming nor contradicting OpenJev-RLCD's headline), and the logged-feedback result as its core (PAPER section 1).
- **The direct-method lineage.** CAPS is the direct method of logged bandit feedback (Dudík, Langford and Li 2011) and the regression-oracle objective (Foster et al. 2018), OpenJev-RLCD's full-label stage 1 restricted to the one action partial feedback reveals; positive-only SFT is selection towards the starting model's own successes, which loses the exclusions the negatives carry (the positive-unlabeled framing does not apply, since the logger's confidence selects the positives). What is new is its use for LoRA decision heads, with the noisy channel and label-free checkpoint selection, not the objective (PAPER sections 2 and 3).

## Five lessons about data and pipelines

1. **A model uses the surface form when that is enough:** sft_06b answered a question and its negation alike in 89 percent of pairs, so phrasing-only accuracy is a build check.
2. **Question order can leak a label:** a sentiment noul before the score revealed "not neutral" (+5.0 points in the v1.2 leak probe, docs/DATA.md section 8), so a record holds one gold-dependent question, last.
3. **A fast GPU cycle pays only if its sample is right:** the first 300 records of a file covered 3 to 16 intents, so the sampler is stratified.
4. **Check memory on the worst case first:** a pre-flight pass now catches in a minute what once failed at step 392.
5. **A file in git is history, not state:** a committed summary let a step-200 adapter pass as `sft_06b`, so training refuses to start over old run files.

## Questions and answers

**How is this different from a fine-tuned classifier?** The options are input, not a fixed output layer: .868 choice accuracy on intents never seen in training, .851 on Banking77's untrained labels (R1 sections 2 and 3).

**From distilling a generative model?** Nothing is generated: the answer is a distribution over the request's options from one forward pass, trained with cross-entropy on the option letters. It cannot fail to parse; same-size JSON generation fails on up to 14.7 percent of questions (R1 section 4).

**From prompting an LLM?** At 0.6B it is 16 times faster than gpt-4.1-mini and more accurate in-domain (.944 against .908), less accurate zero-shot on all four unseen schemas, which the cascade covers (R1 section 4). After adaptation from 5000 logged correctness bits, the RLCD runs at N 5000 score .926 to .936 on the Banking77 subset against gpt-4.1-mini's .918, within noise: paired, the three-seed mean is +.014 [-.010, +.037] (R3 section 7).

**When should RLCD help, and when can it not?** Not with deterministic gold labels, where a bandit outcome carries less than the label and a temperature fixes overconfidence better (R2). It helps when only the chosen action's correctness is observed: from 2000 logged interactions on Banking77 and from 500 on emotion it beats training on the confirmed answers alone and is far better calibrated, at 0.6B and 1.7B and without propensity weights. Full labels stay ahead, by 1.5 points at 0.6B, .6 at 1.7B and 4.3 on emotion (at 1.7B and on emotion against one full-label seed). Under noisy feedback it calibrates to the noise, which can be undone only at a known noise rate, and memorises single noisy draws unless stopped early on the feedback (R3 sections 2 to 4 and 11 to 13).

**What do cost and latency assume?** Batch-1 median on one T4: 47.7 ms (sft_06b) and 74.8 ms (sft_17b), against 777 ms for gpt-4.1-mini; the cost assumes a T4 at 0.35 USD per hour at full use, not a measured price (R1 section 4).

## Closed, and what remains

The project closed at tag v3-final, reopened for phase 4, and closed again at tag v4-paper with the paper as a draft. Not done: real deployment feedback (all feedback is simulated from gold labels), a counterfactual-risk policy learner (POEM or BanditNet), a doubly robust arm and an off-policy value estimate, more full-label and positive-only seeds on emotion and at 1.7B, and a second logging policy per domain (PAPER section 5). From the earlier phases, stage 2b was not run, V-c was designed and not run, and the optional improvements (ranked probability score reward, permutation averaging, marker-token head) were not built.
