# jevmark: Plan

## What we are building

A small decision model that does not generate text. Given a piece of text (the state) and a set of typed questions, it returns, in one forward pass, a probability distribution for every question, plus a confidence score for choice and score questions. Three question types:

- Noul: a yes/no question, returns the probability of yes.
- Choice: pick one of N options defined in the request, returns a probability per option.
- Score: a position on an ordered scale whose levels are described in the request, returns the level distribution and its expectation.

The model is Qwen3-1.7B-Base with a LoRA adapter; Qwen3-0.6B-Base is used to debug the pipeline and to run RLCD sweeps cheaply, and the final report includes both sizes. Options are written into the prompt, each question ends with an answer slot, and the answer is read from the language-model logits at that slot restricted to the option letters. No custom head, no decoding.

Deliverable: a Python package exposing `systemone(state, questions)` with the request and response shapes in docs/API_SPEC.md, trained checkpoints, an evaluation report, and (phase 3) an integration into the delta-filing LangGraph agent.

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

### v3: integration

- A LocalDecider wrapper in delta-filing next to the existing LocalLLM wrapper.
- Router, tool selection, retrieval rerank and completion check switched to jevmark, with confidence thresholds in one file.
- The agent's own decision questions (its routes, its tools, its completion check) have no public dataset counterpart, so the v1 and v2 checkpoints meet them as unseen schemas. Before measurement, v3 therefore includes a fine-tuning pass on the agent's logged decisions: run the graph with the LLM deciding, log every decision point's state, question and the LLM's answer (checked against the reference where one exists), and fine-tune the adapter on those logs, with the measurement queries held out.
- Design constraint on question order (API_SPEC section 4, usage note): in every request the agent sends, a question whose construction depends on the answer to another question in the same request comes after it, with at most one such dependent question per request; independent questions may be batched freely. Decision points that need a chain of dependent questions are split into separate requests. Violating the order cost about 6 points of noul accuracy and flipped about 9 percent of noul answers on test_indomain for 0.6B.
- The same graph run in three configurations: all decisions by LLM, all by jevmark, cascade. Measure end-to-end latency, cost and accuracy on the same query set.

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
| Does it help the agent | end-to-end latency, cost, accuracy in three configurations | v3 report |

## Success criteria

- v1: the SFT model beats the frozen base on in-domain accuracy by a clear margin, beats it on unseen intents, and does not collapse on unseen schemas. All baselines measured on the same test sets. Report written.
- v2: at least one RLCD arm matches SFT accuracy and has lower ECE than SFT on unseen schemas without per-schema temperature. If no arm does, the report says so and explains why; a negative result is still a deliverable.
- v3: the cascade configuration reaches at least 95 percent of the all-LLM accuracy at under 20 percent of its decision cost and latency. Numbers replace these placeholders once measured.

## Risks

- Letter position bias: the model prefers A. Mitigation: shuffle option order in training, measure bias explicitly, upgrade to a marker head if it persists.
- Frozen base is already strong: the fine-tune gain may be small on general schemas. Mitigation: report it honestly; the value story rests on calibration and the cascade, not on raw accuracy.
- RLCD does not beat temperature scaling in-domain: expected. The claim is transfer to unseen schemas and behaviour under bandit feedback, so the evaluation is designed around those.
- Kaggle session limits: every script checkpoints and resumes.

## Out of scope

Games, browser agents, multimodal input, more than 26 options per question in v1, from-scratch pretraining, any UI beyond an optional FastAPI wrapper.
