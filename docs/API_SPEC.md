# jevmark API and encoding contract

This file is the contract between data building, training, evaluation and inference. Any change here requires updating all four and telling the human.

## 1. Public function

```python
from jevmark import systemone

response = systemone(state, questions, model=None, max_tokens=None)
```

- `state`: a string, or a JSON-serialisable dict or list. Dicts and lists are rendered as pretty-printed JSON.
- `questions`: dict mapping a question id (str, `[a-z0-9_]+`) to a question definition (section 2).
- `model`: a loaded `JevMark` instance; if None, the default checkpoint from config is used.
- `max_tokens`: encoded length limit. If None, the value from the loaded model's config is used (2048 in `configs/base.yaml`); an explicit argument overrides it. Training configs carry their own `max_tokens` (1024 in v1), which only governs which records are kept for training.
- Returns a response dict (section 3). Raises `ValueError` with the offending field path on invalid input.

A batch variant `systemone_batch(list_of_requests)` returns a list of responses and must produce identical numbers to calling `systemone` one at a time.

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
- Because attention is causal and the answer is never written into the sequence, later questions see earlier questions but never earlier answers. Answers are therefore independent given the state, matching the Jev contract.
- A JSON state is rendered with `json.dumps(state, indent=2, ensure_ascii=False)`.

Training uses exactly this encoding function. The only training-time differences are option reshuffling for choice questions on every epoch and the record filter (records over the training `max_tokens` are dropped, never truncated). Option order in stored data records, for every split, is a seeded random order fixed at build time; the API preserves whatever order the caller sends.

## 5. Model readout

- Backbone: causal LM from config, fp16 autocast on GPU with a NaN/inf check on the first batch and automatic fp32 fallback, fp32 on CPU. LoRA on attention projections (q, k, v, o) and, if enabled, MLP projections.
- Forward once over the sequence. For each question, the letter logits are the logits at its slot position restricted to the columns in `letter_ids`; softmax over those columns only is the answer distribution.
- Full-vocabulary logits are never materialised. The decoder (the inner model, so LoRA layers are active) returns final hidden states after the final norm; the hidden states at `slot_positions` are gathered and multiplied by the `lm_head` weight rows for that question's `letter_ids`. The product runs in fp32 with autocast disabled. Qwen3 has no `lm_head` bias (a bias, if present, is added for the same rows); with tied embeddings the rows are the input embedding rows. The result equals the letter columns of the full logits within float tolerance.
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

- Unknown type, missing instructions, wrong criteria shape, fewer than 2 or more than 26 choice options, fewer than 2 or more than 10 score levels, duplicate option labels, a line break (`\n` or `\r`) or leading or trailing whitespace in instructions, an option label, an option description or a score level text, an empty option description or score level text, state over 8000 characters: `ValueError("<question_id>.<field>: <reason>")`.
- Encoded length over `max_tokens`: `ValueError("max_tokens: <reason>")`. The length belongs to no single question, so the path is the argument name.
- Model not loaded or checkpoint missing: `RuntimeError`.
