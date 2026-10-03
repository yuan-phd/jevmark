# jevmark: Plan

## What we are building

A small decision model that does not generate text. Given a piece of text (the state) and a set of typed questions, it returns, in one forward pass, a probability distribution for every question, plus a confidence score for choice and score questions. Three question types:

- Noul: a yes/no question, returns the probability of yes.
- Choice: pick one of N options defined in the request, returns a probability per option.
- Score: a position on an ordered scale whose levels are described in the request, returns the level distribution and its expectation.

The model is Qwen3-1.7B-Base with a LoRA adapter; Qwen3-0.6B-Base is used to debug the pipeline and to run RLCD sweeps cheaply, and the final report includes both sizes. Options are written into the prompt, each question ends with an answer slot, and the answer is read from the language-model logits at that slot restricted to the option letters. No custom head, no decoding.

Deliverable: a Python package exposing `systemone(state, questions)` with the request and response shapes in docs/API_SPEC.md, trained checkpoints, an evaluation report, and (phase 3) a test of adaptation from deployment feedback on a real unseen domain (docs/V3_DESIGN.md, decision 56).

## Where it is used

Inside an agent, at the points where the LLM currently only decides and never writes:

- which route a request should take
- which tool to call next
- which retrieved passages are relevant
- whether the task is complete or needs another loop
- whether a request should be escalated to a human

The LLM keeps the generation work. Decisions go to jevmark. Decisions the model is not confident about go back to the LLM or to a human. That cascade is the product.

## Why it matters

System level: in a typical agent turn, more than half of the LLM calls are decisions. Each one pays for prefill plus decode and takes about a second through an API. Replacing them with a millisecond forward pass on a small model cuts latency and cost by orders of magnitude, while confidence gating keeps the error rate under control.

Learning level: this project covers a training objective the author has not done before, Reinforcement Learning for Calibrated Decisions, and gives a defensible answer to three interview questions:

- how this differs from distilling a small generative model (output type, training objective, system role)
- how this differs from a fine-tuned classifier (options are input, answer space defined at request time)
- when RLCD adds value over plain supervised training and over post-hoc temperature scaling (bandit or delayed feedback, transfer of calibration to unseen schemas)

## Phases

### v1: supervised decision model

- Package, API, encoding, readout, metrics.
- Data from public datasets: CLINC150 for Choice and Noul, SST-5 for Score. Option descriptions generated once with an LLM API and checked in.
- LoRA SFT with cross-entropy over the option letters.
- Baselines: frozen base with the same readout (zero training), the same base as an instruct model generating JSON, and a commercial API with structured output on a test subset.
- Held-out tests: intents never seen in training, and datasets never seen in training (AG News, emotion, Banking77 subsets).

### v2: RLCD

- Bandit simulation on the labeled data: the model samples an option, only that option's correctness is revealed.
- Policy gradient with a proper scoring rule as reward, group-mean baseline, KL to the SFT policy.
- Ablations: outcome-only reward, outcome minus confidence, Brier, log score; SFT plus temperature scaling as the cheap competitor.
- Core claim to test: RLCD keeps calibration on unseen schemas without fitting a temperature per schema.

Status (2026-09-30). Stages 1 and 2a are complete: five arms on 0.6B at seeds 0, 1 and 2. None of them matches SFT plus one in-domain temperature on unseen-schema ECE without a temperature of its own: the four-schema mean is .153 to .248 across arms, against .121 for SFT plus T. With its own temperature, no arm goes below SFT plus T, and the control (more cross-entropy) does as well as the best arm. On deterministic gold labels, bandit feedback is less information than the label, so no bandit objective can do what cross-entropy cannot. Stage 3, the K-dependent noise environment of decision 54, is cancelled by decision 56; its environment code is reused in v3 and `runs/sft_06b_env` stays as history. Stage 2b (the arms on 1.7B) was not run (decision 58). docs/RESULTS_v2.md section 3 gives the evidence, and section 4 the cancelled stage 3 method.

### v3: adaptation from deployment feedback (decision 56)

Specification, code gaps and open decisions: docs/V3_DESIGN.md.

- Claim under test: after deployment on a new domain, when the only feedback is whether the chosen action was right, does RLCD learn the domain faster, more accurately and better calibrated than what a practitioner would otherwise use?
- Domain: the Banking77 train split (10,003 messages, 77 intents, never trained on), questions built as test_banking77 records are. Evaluation on test_banking77, the full Banking77 test split, and a forgetting check on test_indomain and test_unseen_intents.
- Feedback: sft_06b answers the train messages once, with epsilon 0.1 exploration, and only the correctness of its chosen option is logged (deterministic; a noisy condition with outcomes flipped at 0.2 is secondary). Learners see the first N interactions, N in {500, 2000, 5000}.
- Learners, all from sft_06b: zero-shot, a temperature fitted on the outcomes, positive-only SFT, RLCD direct_brier, and full-label SFT as the upper bound, with equal gradient steps at each N.
- Falsifiable: RLCD beats positive-only SFT in accuracy at every N and in ECE, approaches full-label SFT as N grows, and degrades less under noise. If it does not beat positive-only SFT in accuracy, v2 and v3 together are a complete negative result about RLCD as reconstructed.
- This is where partial feedback actually occurs, and the setting RLCD was designed for; v2 tested it only on full deterministic labels.
Status (2026-10-03). Complete: 14 trained runs at 0.6B, the temperature learner and zero-shot, compared in `runs/v3_stage_06b/metrics.json` and reported in docs/RESULTS_v3.md (decision 57). RLCD ties positive-only SFT in accuracy at N 500 and beats it by 2.4 to 2.7 points from N 2000, with ECE lower by .05 to .06 at every N; it stays 0.7 to 1.8 points behind full-label SFT, a gap that does not shrink with N. Under the 0.2 flip it loses less accuracy than positive-only SFT but calibrates to the noisy outcome (ECE .164); inverting the known flip rate brings it to .051 to .066. Prediction 1 is a tie at N 500 and holds from N 2000, prediction 2 holds, prediction 3 fails, prediction 4 holds for accuracy and fails for calibration.

- Optional after v3: a demonstration inside the delta-filing agent, measured on its track1 evaluation sets (router 160, tool choice 160, review 80), with decision logging added for future feedback. That repository has three connected tools, at most one tool call per query and no run logs or outcome records, so it cannot supply the feedback v3 needs. If built, the order rule of API_SPEC section 4 applies to every request the agent sends: a question whose construction depends on another question's answer comes after it, at most one such question per request.

## How we measure

| Question | Metric | Where |
|---|---|---|
| Is it accurate | accuracy, macro-F1 | in-domain test, unseen intents, unseen schemas |
| Does it read the options | accuracy on unseen intents and unseen schemas vs in-domain | v1 report |
| Are the probabilities honest | ECE (15 bins on top-1 probability), Brier, NLL, reliability diagram | every checkpoint |
| Is it symmetric | P(yes) under a question and under its negation sum to about 1 | v1 report |
| Is it usable | coverage vs accuracy at confidence thresholds (the response confidence field; max(p, 1-p) for noul) | v2 report, the cascade figure |
| Is it worth it | latency per call (batch 1, T4), cost per 1000 calls, parse failure rate for the JSON baseline | v1 report |
| Does RLCD add value | ECE on unseen schemas: SFT vs SFT+temperature vs RLCD arms | v2 report |
| Does deployment feedback teach a new domain | accuracy, ECE, Brier, NLL and coverage on Banking77 per learner and per N, paired deltas, forgetting on test_indomain and test_unseen_intents | v3 report |

## Success criteria

- v1: the SFT model beats the frozen base on in-domain accuracy by a clear margin, beats it on unseen intents, and does not collapse on unseen schemas. All baselines measured on the same test sets. Report written.
- v2: at least one RLCD arm matches SFT accuracy and has lower ECE than SFT on unseen schemas without per-schema temperature. If no arm does, the report says so and explains why; a negative result is still a deliverable.
- v3: RLCD beats positive-only SFT in accuracy at every N on the same deployment log (prediction 1 of docs/V3_DESIGN.md). If it does not, the report says so, and v2 and v3 together are the negative result.

## Risks

- Letter position bias: the model prefers A. Mitigation: shuffle option order in training, measure bias explicitly, upgrade to a marker head if it persists.
- Frozen base is already strong: the fine-tune gain may be small on general schemas. Mitigation: report it honestly; the value story rests on calibration and the cascade, not on raw accuracy.
- RLCD does not beat temperature scaling in-domain: expected. The claim is transfer to unseen schemas and behaviour under bandit feedback, so the evaluation is designed around those.
- Kaggle session limits: every script checkpoints and resumes.

## Out of scope

Games, browser agents, multimodal input, more than 26 options per question in v1, from-scratch pretraining, any UI beyond an optional FastAPI wrapper.
