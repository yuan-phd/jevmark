# Decision log

One entry per decision. Newest at the bottom. Add an entry whenever a choice is made that a future reader would otherwise have to guess.

| # | Decision | Reason |
|---|---|---|
| 1 | Build our own model instead of calling the Jev API | The goal is learning the training side, in particular RLCD, and an interview-ready story; an API demo would not deliver either. |
| 2 | Post-training with LoRA on a pretrained base, not pretraining from scratch | A from-scratch model on a T4 budget cannot learn enough language to read option descriptions. |
| 3 | Main backbone Qwen3-1.7B-Base; Qwen3-0.6B-Base for pipeline debugging and RLCD sweeps; 4B only as an optional QLoRA experiment | 1.7B reads option descriptions far better than 0.6B while staying clearly smaller than the 4B generator in the target agent; 1.7B still fits fp32 on a T4 if fp16 overflows, 4B does not; 0.6B makes multi-seed ablations cheap; two sizes give a scaling row in the report for free. |
| 4 | Readout via language-model logits at answer slots restricted to option letters, no custom head | Zero new parameters, one forward pass, multi-question in one sequence, and the frozen base is an immediate baseline. Per-candidate sequences (NanoJev style) are too slow to serve; a marker head is kept as a fallback if letter bias or the 26-option cap bites. |
| 5 | Options are always written into the prompt and may vary per request | This is what separates a System One model from a fine-tuned classifier; it is also what makes the unseen-intent and unseen-schema tests meaningful. |
| 6 | v1 data from public datasets (CLINC150, SST-5); LLM API used only to write option descriptions and later to synthesise rare branches; LLM also serves as a baseline | Real human text and trusted labels for a credible test set; synthetic-only evaluation would be circular. |
| 7 | Hold out 20 intents and three whole datasets that never appear in training | Evidence that the model reads options rather than memorising labels. |
| 8 | Confidence for choice and score is 1 - H(p)/ln(K) | TypeSafe has not published its formula; normalised entropy is simple, bounded in [0, 1], and recoverable from the full distribution we already return. |
| 9 | RLCD is defined operationally as policy gradient under bandit feedback with a proper scoring rule as reward, group-mean baseline and a KL term to the SFT policy | TypeSafe has published the objective but not the method; this is the reconstruction used by the public replications (Laya, eve-rlcd) and it maps onto the author's GRPO experience. |
| 10 | RLCD must be compared against SFT plus temperature scaling, and the primary claim is calibration on unseen schemas | With full labels, cross-entropy already optimises a proper score, so in-domain gains are not expected; the honest test is transfer of calibration and behaviour under partial feedback. |
| 11 | Integration host is the delta-filing agent (SEC filings, 12 MCP tools, review loop), not the insurance ReAct agent | delta-filing has many decision points of different kinds per query, so the cascade story is measurable; the insurance loop is nearly deterministic and would be better served by hardcoding. |
| 12 | The engine is built standalone first (phases 1 and 2) and integrated later (phase 3) | The interface is the contract; integration is one wrapper class once the engine exists. |
| 13 | v1 caps choice at 26 options | One letter per option keeps the readout trivial; larger sets are a v2.4 concern via a marker head or two-stage selection. |
| 14 | Project name jevmark; package jevmark; public function stays systemone | A quick search found no existing project with the name. The mark part is read as a measured reproduction: the differentiator of this project is honest evaluation (unseen schemas, RLCD ablations against temperature scaling), so the name fits the product. |
| 15 | fp16 autocast with automatic fp32 fallback on NaN or inf | T4 has no bf16; Qwen models are trained in bf16 and can overflow in fp16. The check is cheap and removes a whole class of silent failures. |
| 16 | LLM API is a mini-tier OpenAI model (GPT-4o-mini or its current successor) for descriptions and the cost baseline; the current flagship on a 200-example subset as the accuracy ceiling | Expected spend is under a dollar for descriptions and a few dollars for baselines; the caps in TASKS.md are guardrails against runaway loops, not budgets. |

