# jevmark API and encoding contract

This file is the contract between data building, training, evaluation and inference. Any change here requires updating all four and telling the human.

## 1. Public function

```python
from jevmark import systemone

response = systemone(state, questions, model=None, max_tokens=None)
```

- `state`: a string, or a JSON-serialisable dict or list. Dicts and lists are rendered as pretty-printed JSON.
- `questions`: dict mapping a question id (str, `[a-z0-9_]+`) to a question definition (section 2).
- `model`: a loaded `JevMark` instance. If None, a default model is loaded lazily on first use and reused for the rest of the process. Its config is `configs/base.yaml`. If the environment variable `JEVMARK_CHECKPOINT` names a run directory, that run's adapter, `calibration.json` and `model_id.txt` are loaded too, and the run's own `config.yaml` replaces `configs/base.yaml` when it exists, so the adapter is loaded onto the backbone it was trained with. The variable is read once, at first use.
- `max_tokens`: encoded length limit. If None, the model's `max_tokens` is used, which `JevMark.load` reads from the config (2048 in `configs/base.yaml`); an explicit argument overrides it. Training configs carry their own `max_tokens` (1024 in v1), which only governs which records are kept for training.
- Returns a response dict (section 3). Raises `ValueError` with the offending field path on invalid input.

A batch variant `systemone_batch(requests, model=None, max_tokens=None, batch_size=16)` takes a list of `{"state": ..., "questions": ...}` dicts, runs them through the model `batch_size` requests per forward pass, and returns a list of responses. Batching has no semantic effect: each request's answers depend only on its own state and questions. Batched and single calls can differ numerically, because padded batches change the order of floating-point operations; the difference is bounded by the arithmetic precision of the run and is measured in every evaluation (`batching_precision` in `metrics.json`). As measured: in fp32 it is about 1e-4 at most (the CPU test tolerance; 6e-8 measured on CPU with the tiny test model); under fp16 on GPU it is up to about 1e-2 (0.0041 for Qwen3-0.6B-Base and 0.0065 for Qwen3-1.7B-Base on a T4, 200 requests at batch 16, B0 run). At the 4 rounded decimals of a response, fp16 batching can therefore change the third decimal and occasionally the second.

## 2. Question definitions

```json
{
  "refund_requested": {
    "type": "noul",
    "instructions": "Does the customer ask for money back?",
    "criteria": {"true": "optional description of yes", "false": "optional description of no"}
  },
  "department": {
    "type": "choice",
    "instructions": "Which team should handle this message?",
    "criteria": {
      "billing": "Charges, invoices, refunds",
      "technical": "Bugs, outages, integration problems",
      "other": null
    }
  },
  "severity": {
    "type": "score",
    "instructions": "How severe is the reported issue?",
    "criteria": [
      "Cosmetic; no impact on functionality",
      "Broken or degraded feature, but a workaround exists",
      "Blocking issue; no workaround exists"
    ]
  }
}
```

Rules:

- `instructions` is required for every type and is the full question. The question id is not shown to the model.
- noul: `criteria` optional. Answer order is fixed: true first, false second.
- choice: `criteria` is a dict of option label to description or null. 2 to 26 options in v1 (one letter each). Option order in the request is preserved at inference; it is shuffled during training only.
- score: `criteria` is a list of 2 to 10 level descriptions, ordered from low to high. Level index starts at 0.
- Every text that is rendered onto a line of the encoding (`instructions`, choice labels, choice descriptions, score level texts, and noul `criteria.true` / `criteria.false` descriptions) must not contain `\n` or `\r` and must not have leading or trailing whitespace. Descriptions are never empty: a choice description or noul `criteria.true` / `criteria.false` is either null or a non-empty string, and a score level text is always a non-empty string. The state is free text and may contain newlines and edge whitespace, and may be empty.

Limits (v1): state at most 8000 characters; state plus all questions at most `max_tokens` tokens after encoding, else `ValueError`.

## 3. Response

```json
{
  "model": "jevmark-sft_clinc_v1_17b",
  "answers": {
    "refund_requested": {"type": "noul", "noul": 0.93},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.84, "technical": 0.15, "other": 0.01},
      "confidence": 0.61
    },
    "severity": {
      "type": "score",
      "score": 1.3,
      "legend": {"0": "Cosmetic; no impact on functionality", "1": "...", "2": "..."},
      "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
      "confidence": 0.54
    }
  },
  "usage": {"input_tokens": 212}
}
```

Definitions:

- `noul`: probability of true. No confidence field in the response, matching Jev. For metrics and coverage curves only, noul confidence is defined as `max(noul, 1 - noul)`; this never appears in the response.
- `choice`: `choice` is the argmax label; `probabilities` sums to 1 over the request's options.
- `score`: `score` is the probability-weighted mean of level indices; `probabilities` keyed by level index as a string.
- `confidence` for choice and score: `1 - H(p) / ln(K)` where H is the entropy in nats and K the number of options. Equals 1 when all mass is on one option, 0 when uniform. Rounded to 4 decimals.
- Probabilities rounded to 4 decimals after normalisation; `choice` and `score` are computed before rounding. After rounding, probabilities sum to 1 within 1e-3; tests use that tolerance.
- `model` is `jevmark-<run_name>`, read from `model_id.txt`; for an untrained backbone it is `jevmark-base_<size>`.

## 4. Encoding

One request becomes one token sequence. Questions are appended in the order given. Every question ends with an answer slot. The model reads the next-token logits at the token that ends `Answer:` and restricts them to the letter tokens of that question's options.

Text format (exact, including blank lines):

```
### State
{state_text}

### Question: {instructions}
Options:
A. {label}: {description}
B. {label}
Answer:

### Question: {instructions}
Options:
A. true: {criteria.true or "yes"}
B. false: {criteria.false or "no"}
Answer:
```

- Choice: one line per option, letter, period, space, label; add colon and description when description is not null.
- Noul: exactly two options, `true` then `false`.
- Score: one line per level in order, label is the level index, description is the level text: `A. 0: Cosmetic; no impact on functionality`.
- Letters are `A` through `Z`. The letter tokens read at the slot are the tokenizer ids of `" A"` through `" Z"` (leading space). `encode.py` must assert at load time that each of these is a single token for the configured tokenizer; if not, fail loudly.
- Tokenization is per segment, never over the whole text. Segments, in order: the state block (`### State\n{state_text}`); then for each question, a separator segment `\n\n` followed by the question block from `### Question:` through the final `Answer:` inclusive. Each segment is tokenized with `add_special_tokens=False` and the ids are concatenated. Reason: Qwen's pre-tokenizer can merge `:` with the newlines that follow when the whole text is tokenized at once, which would move the slot to a token that also contains the blank line. Per-segment tokenization guarantees that `Answer:` ends on a token boundary.
- Slot position: the index of the last id of each question segment. `encode.py` asserts that this id decodes to a string ending in `:`. `encode.py` returns `input_ids`, `slot_positions` (one per question, in request order) and `letter_ids` (list of allowed letter token ids per question). Decoding `input_ids` must reproduce the text format above exactly.
- Option lines are written as `A. label` with no leading space, while the slot reads the tokens `" A"` to `" Z"` with a leading space, because the natural next token after `Answer:` is a space plus a letter. This is the MMLU convention and is kept deliberately; letter bias is measured in evaluation.
- Because attention is causal and the answer is never written into the sequence, later questions see earlier questions but never earlier answers. Answers are therefore independent given the state, matching the Jev contract. Answers are not independent of question order, though: a question's distribution can change when the questions before it change, so reordering questions can change the numbers (measured by `evaluate.py --shuffle-questions`, and the reason training data puts at most one gold-dependent question per record after the choice or score question, docs/DATA.md section 2).
- A JSON state is rendered with `json.dumps(state, indent=2, ensure_ascii=False)`.

Usage note on question order (data v1.3, decisions 42 and 46). A question whose construction depends on the answer to another question in the same request (for example a yes or no question that asks about one of the options of a choice question, or a follow-up that only makes sense for one answer) should come after the question it depends on. The model was trained with the choice or score question first and at most one such dependent question per request; any number of questions that depend on nothing but the state may appear anywhere. Requests outside that pattern are accepted and answered, but cost accuracy: on test_indomain, reordering each record's questions so that the dependent noul can come before the choice lowered noul accuracy from 0.937 to 0.878 (about 6 points) and changed 8.9 percent of noul answers (about 9 percent), and lowered choice accuracy from 0.975 to 0.952 (`runs/sft_06b/metrics.json`, `order_sensitivity`, Qwen3-0.6B-Base). Two or more dependent questions in one request were never trained and are unmeasured.

Training uses exactly this encoding function. The only training-time differences are option reshuffling for choice questions on every epoch and the record filter (records over the training `max_tokens` are dropped, never truncated). Option order in stored data records, for every split, is a seeded random order fixed at build time; the API preserves whatever order the caller sends.

## 5. Model readout

- Backbone: causal LM from config. On CUDA with `precision.autocast: fp16`, the frozen backbone weights are loaded in fp16 and the forward pass runs under fp16 autocast, while LoRA parameters stay fp32; everywhere else, including CPU, every weight is fp32. The first batch's slot logits are checked for NaN or inf, and on failure the model is reloaded in fp32 from the same backbone and checkpoint (not cast up from fp16). LoRA on attention projections (q, k, v, o) and, if enabled, MLP projections.
- Forward once over the sequence. For each question, the letter logits are the logits at its slot position restricted to the columns in `letter_ids`; softmax over those columns only is the answer distribution.
- Full-vocabulary logits are never materialised. The decoder (the inner model, so LoRA layers are active) returns final hidden states after the final norm; the hidden states at `slot_positions` are gathered and multiplied by the `lm_head` weight rows for that question's `letter_ids`. The product runs in fp32 with autocast disabled, whatever the backbone's dtype. Qwen3 has no `lm_head` bias (a bias, if present, is added for the same rows); with tied embeddings the rows are the input embedding rows. The result equals the letter columns of the full logits within float tolerance.
- Two entry points. `slot_logits(encoded_batch)` returns per-question letter logits with gradients enabled, for training. `forward_distributions(encoded_batch)` divides by the temperature and applies softmax under `no_grad`, for inference. Both return a flat list aligned with (request index, question index): request 0's questions in order, then request 1's, and so on, since requests have different numbers of questions and questions have different K.
- Batches are right-padded with an attention mask. Slot positions index real tokens, so padding does not change them.
- Optional calibration: a scalar temperature per model (fit in `calibrate.py`) divides the selected logits before softmax. Stored in the checkpoint's `calibration.json`; default 1.0.

## 6. Checkpoint layout

```
runs/<run_name>/
  config.yaml          the config that produced the run
  adapter/             peft adapter weights
  calibration.json     {"temperature": 1.0}
  metrics.json         evaluation results
  model_id.txt         string returned in response["model"]
```

## 7. Errors

Every invalid input raises `ValueError("<path>: <reason>")`. The path is one of:

| Path | Raised for |
|---|---|
| `request` | the request itself is not an object |
| `state` | state missing, not a string, object or array, not JSON-serialisable, or over 8000 characters after rendering |
| `questions` | questions missing, not an object, empty, a question id not matching `[a-z0-9_]+`, a duplicate question id |
| `<question_id>` | the question definition is not an object |
| `<question_id>.<field>` | `type` missing or unknown; `instructions` missing, empty, containing `\n` or `\r`, or with leading or trailing whitespace; `criteria` of the wrong shape, fewer than 2 or more than 26 choice options, fewer than 2 or more than 10 score levels, duplicate option labels, or an option label, option description or score level text that is empty, contains a line break or has edge whitespace; an unknown field, reported under its own name (for example `q.criterion`) |
| `max_tokens` | encoded length over `max_tokens`; the length belongs to no single question, so the path is the argument name |

Model not loaded or checkpoint missing: `RuntimeError`. A tokenizer that breaks an encoding assumption (letter tokens, tokenizer parity, a slot not ending in `:`) raises `TokenizerError`, a subclass of `RuntimeError`.
