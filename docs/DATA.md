# jevmark: Data

Data files are JSONL under `data/` (gitignored, rebuilt by `make data`). Option descriptions are JSON under `jevmark/data/descriptions/` and are checked in. This file is the contract between data building, training and evaluation.

## 1. Record schema

One record is one `systemone` request plus its gold answers. Every record passes `Request.from_dict({"state": ..., "questions": ...})` unchanged.

A real record from `data/train.jsonl` (data v1.3, seed 0): a JSON state, a form noul before the choice, the choice, then the one gold-dependent noul.

```json
{
  "id": "clinc-train-000022",
  "source": "clinc/clinc_oos/plus",
  "split": "train",
  "state": {
    "text": "send over a hundred dollars from huntington into saving",
    "thread_id": "t-d54016",
    "channel": "email"
  },
  "questions": {
    "char_count_over": {
      "type": "noul",
      "instructions": "Is the text at most 38 characters long?"
    },
    "intent": {
      "type": "choice",
      "instructions": "Which intent does this message express?",
      "criteria": {
        "balance": "Inquire about how much money is available in an account",
        "meaning_of_life": "A question regarding the significance or reason for existence",
        "reminder": "Asks to read back or check the reminders that are already set",
        "other": "None of the listed intents",
        "calories": "Inquiries regarding how many calories are in certain foods or dishes",
        "transfer": "A request to move money between accounts or send funds to someone"
      }
    },
    "about_intent": {
      "type": "noul",
      "instructions": "Does this description fit the message: Requesting to send funds or shift money between accounts?"
    }
  },
  "gold": {
    "char_count_over": "false",
    "intent": "transfer",
    "about_intent": "true"
  },
  "meta": {
    "source_split": "train",
    "source_index": 122,
    "gold_intent": "transfer",
    "domain": "banking",
    "gold_in_options": true,
    "nouls": {
      "about_intent": {
        "template": 0,
        "negated": false,
        "slot": "Requesting to send funds or shift money between accounts",
        "asked_intent": "transfer",
        "asked_from": "gold",
        "asked_in_options": true
      },
      "char_count_over": {
        "template": 1,
        "negated": true,
        "slot": "38",
        "form": true,
        "threshold": 38
      }
    },
    "state_format": "json",
    "state_fields": [
      "text",
      "thread_id",
      "channel"
    ]
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `<source short name>-<split>-<6-digit index>`, unique across all files |
| `source` | string | dataset id plus config, as in section 3 (for example `clinc/clinc_oos/plus`) |
| `split` | string | the jevmark split the record belongs to (section 2), equal to the file name stem |
| `state` | string or object | the input text, unchanged from the dataset; in 20 percent of the records of every split, a JSON object holding the text under `text` plus 1 to 3 label-independent fields (section 2, State formats) |
| `questions` | object | question id to question definition, exactly as API_SPEC section 2; key order is the question order in the encoded sequence. A noul question's id is its kind |
| `gold` | object | question id to gold answer: an option label (string) for choice, `"true"` or `"false"` for noul, a level index (integer) for score |
| `meta` | object | provenance and build facts, never read by the model; keys per source in the table below, plus `state_format` and `state_fields` on every record |

`meta` keys:

| Source | Keys |
|---|---|
| CLINC | `source_split`, `source_index`, `gold_intent` (intent name, or `oos`), `domain` (null for `oos`), `gold_in_options` (null for `oos`), `nouls` |
| SST-5 | `source_split`, `source_index`, `label_text`, `scale` (`sst5_5_levels` or `sst5_3_levels`), `nouls` (no sentiment noul on neutral records) |
| emotion | `source_split`, `source_index`, `nouls` |
| AG News, Banking77 | `source_split`, `source_index`, `nouls` (always empty) |
| Yelp | `source_split`, `source_index`, `label_text` (the dataset's label name, `1 star` to `5 stars`), `scale` (`yelp_5_stars`), `nouls` (always empty) |

`meta.nouls` maps each noul question id (its kind) to `template` (index into that kind's templates in `jevmark/data/negation.py`), `negated` (stored in the negated phrasing, gold flipped), `slot` (the text filled into the template, or null) and, where it applies:

- `asked_domain`, `asked_intent` or `asked_emotion`: what the question asks about.
- `asked_from` (`about_domain`, `about_intent`): `gold`, `distractor` (a distractor option of the record's choice question, or that option's domain) or `outside` (an intent outside the options, or a uniformly drawn other domain).
- `asked_in_options` (`about_domain`, `about_intent`): whether the asked intent is one of the choice's named options, or the asked domain is the domain of one of them.
- `form: true` and `threshold` (form nouls): the kind is label-independent; `threshold` is the per-source threshold, or null for a yes or no property.

Every record also has `meta.state_format` (`plain` or `json`) and `meta.state_fields` (the JSON object's keys in order, including `text`; empty for plain states).

Rules:

- Option order inside every `criteria` object of a choice question is a seeded random order fixed at build time, for every split including all test splits (decision 19). Training reshuffles on top of that every epoch (`encode.shuffle_request`). Gold is stored as a label, never as a letter, so reshuffling never needs to touch it. Score levels keep their order.
- Question order inside `questions` follows the order rule (section 2, Question order): among the gold-dependent questions, the choice or score question comes first and at most one noul follows it; form nouls sit at seeded random positions. The build fails on any record that breaks the rule.
- Every text that goes on a rendered line obeys API_SPEC section 2: single line, no edge whitespace, descriptions non-empty or null. The builder validates each record with `Request.from_dict` and fails loudly on any record that does not pass.
- Records whose encoding exceeds the training `max_tokens` are dropped by the training loader, never truncated (API_SPEC section 4); `build_data.py` checks that every record encodes within 1024 tokens, so none is dropped.

## 2. Splits and how each is built

Current data version: **v1.3** (`version` in `configs/data.yaml`, decision 42), frozen for v1 and v2 once it passes the CPU gates (section 8) and the fast GPU cycle (docs/KAGGLE.md). v1.2 (decision 41) added noul and score diversity, v1.1 (decision 40) negated noul questions; v1 had neither. The v1 runs (`runs/sft_06b`, `runs/base_06b` at commit 3fe318d, and the first B0 runs) were evaluated on v1.

### Question order: one gold-dependent question per record

Attention is causal: a question sees every question before it, but no answer, because answers are never written into the sequence (API_SPEC section 4). A question placed before another can therefore still leak information about its gold answer through the way it is built. The review of v1.2 found three such leaks (decision 42): a sentiment noul before the SST-5 score question told the model the text was not neutral, because neutral records had no noul; an `about_intent` question before the choice primed the choice, because it asked the gold intent in half the records; and `about_intent` and `about_domain` in the same record were correlated through the gold.

v1.3 therefore fixes the order and allows exactly one gold-dependent noul per record:

| Source | Gold-dependent questions, in this order |
|---|---|
| CLINC | `intent` (choice), then one of `about_domain`, `about_intent`, `out_of_scope` |
| SST-5 | `sentiment` (score), then `is_positive` or `is_negative` (non-neutral records only) |
| emotion | `label` (choice), then `expresses_emotion` |
| AG News, Banking77, Yelp | the choice or score question alone |

Form nouls, which do not depend on any gold label, are inserted at seeded random positions anywhere in the record, including before the choice or score question; they appear only in the splits built from the training sources. `build.order_violations` enforces the rule on every record.

All randomness comes from one build seed in `configs/data.yaml`: **seed 0**. Each split uses its own generators, `random.Random(f"{seed}:{name}")` with names such as `train`, `train:sst5`, `train:phrasing`, and the held-out intents use `"{seed}:held_out"`, so rebuilding one split never changes another. The other build parameters are in the same file.

### Noul questions: kinds, templates and balance

| Kind | Where | Question | Underlying yes |
|---|---|---|---|
| `about_domain` | CLINC, about 45 percent of records | is the message about a domain | the asked domain is the gold domain |
| `out_of_scope` | CLINC, the option 1 records | is the request outside the assistant's scope | the utterance is out of scope |
| `about_intent` | CLINC, about 45 percent of records | does an intent description fit the message | the asked intent is the gold intent |
| `is_positive`, `is_negative` | non-neutral SST-5 records, one of the two at random | is the sentiment positive (negative) | label 3 or 4 (0 or 1) |
| `expresses_emotion` | every `test_emotion` record (evaluation only) | does the message express an emotion | the asked emotion is the gold emotion |
| form nouls | 0 to 2 per record in train, valid, test_indomain, test_unseen_intents and test_sst5 (section 2, Form nouls) | a property of the text, such as its length | computed from the text |

Every kind has two or three templates, each a positive phrasing and its negation (`jevmark/data/negation.py`, which maps every phrasing to its pair in both directions and parses template, polarity and slot back from a rendered instruction):

| Kind | Templates (positive / negated) |
|---|---|
| `about_domain` | 0: `Is this message about {slot}?` / `Is this message about something other than {slot}?`<br>1: `Does this message concern {slot}?` / `Does this message concern something other than {slot}?`<br>2: `Is the topic of this message {slot}?` / `Is the topic of this message something other than {slot}?` |
| `out_of_scope` | 0: `Is this request outside what a banking, travel, home, work or everyday assistant can help with?` / `Is this request something a banking, travel, home, work or everyday assistant can help with?`<br>1: `Would a banking, travel, home, work or everyday assistant be unable to help with this request?` / `Would a banking, travel, home, work or everyday assistant be able to help with this request?`<br>2: `Is this request out of scope for a banking, travel, home, work or everyday assistant?` / `Is this request within scope for a banking, travel, home, work or everyday assistant?` |
| `about_intent` | 0: `Does this description fit the message: {slot}?` / `Does this description fail to fit the message: {slot}?`<br>1: `Is this message an example of the following: {slot}?` / `Is this message an example of something other than the following: {slot}?` |
| `is_positive` | 0: `Is the sentiment of this text positive?` / `Is the sentiment of this text something other than positive?`<br>1: `Does this text express a favourable opinion?` / `Does this text express anything other than a favourable opinion?` |
| `is_negative` | 0: `Is the sentiment of this text negative?` / `Is the sentiment of this text something other than negative?`<br>1: `Does this text express an unfavourable opinion?` / `Does this text express anything other than an unfavourable opinion?` |
| `expresses_emotion` | 0: `Does this message express {slot}?` / `Does this message express something other than {slot}?`<br>1: `Is {slot} the main emotion in this message?` / `Is something other than {slot} the main emotion in this message?` |
| `word_count_over` | 0: `Does the text have more than {slot} words?` / `Does the text have {slot} words or fewer?`<br>1: `Is the text longer than {slot} words?` / `Is the text at most {slot} words long?` |
| `char_count_over` | 0: `Does the text have more than {slot} characters?` / `Does the text have {slot} characters or fewer?`<br>1: `Is the text longer than {slot} characters?` / `Is the text at most {slot} characters long?` |
| `longest_word_over` | 0: `Does the text contain a word longer than {slot} letters?` / `Are all words in the text at most {slot} letters long?`<br>1: `Does any word in the text have more than {slot} letters?` / `Does every word in the text have {slot} letters or fewer?` |
| `contains_number` | 0: `Does the text contain a digit?` / `Is the text free of digits?`<br>1: `Does the text include a number written in digits?` / `Does the text include no number written in digits?` |
| `contains_comma` | 0: `Does the text contain a comma?` / `Is the text free of commas?`<br>1: `Is there at least one comma in the text?` / `Is there no comma in the text?` |
| `ends_with_question_mark` | 0: `Does the text end with a question mark?` / `Does the text end with something other than a question mark?`<br>1: `Is the last character of the text a question mark?` / `Is the last character of the text something other than a question mark?` |

Balance, in two steps:

1. The builder fixes each noul question's underlying answer with exact counts wherever the data allows: `out_of_scope` asks exactly as many in-scope utterances (no) as there are out-of-scope ones (yes); `about_domain` and `about_intent` each ask the gold in exactly half of their questions, counting the out-of-scope records' questions of that kind, which are always no; `expresses_emotion` asks the gold emotion in exactly half of the records; every form kind is used only where its threshold splits the split's texts within 48 to 52 percent. The SST-5 kinds follow the dataset's labels.
2. `build.assign_phrasings` then gives every question its template and polarity: within each kind and underlying answer, questions take (template, polarity) pairs from consecutive shuffled blocks that hold each pair once. Exactly half of each group is negated (gold flipped), and every phrasing carries about the kind's underlying balance, so no phrasing predicts the answer.

`build_data.py` fails the build if, for any noul kind in any split, the yes share is outside 40 to 60 percent or the phrasing-only accuracy (answering each phrasing with its own majority answer in that split, the best any rule that ignores the message can do) is above 55 percent (`max_phrasing_only_accuracy`). It prints both per kind per split.

### CLINC150 (`clinc/clinc_oos`, config `plus`)

Every CLINC record carries `intent` (choice) first, then exactly one gold-dependent noul, plus 0 to 2 form nouls anywhere, so the model always trains on multi-question sequences.

Choice question `intent`:
- Instructions: `Which intent does this message express?`
- K is uniform in 3..14 and counts every option including `other`.
- For an in-scope utterance, the gold intent is among the options with probability 0.8. The other options are distinct distractor intents drawn uniformly from the intents allowed in that split, then `other` is added.
- If the gold intent is not among the options, or the utterance is out of scope, the gold answer is `other`.
- Each intent's description is drawn at random from its three variants (canonical plus two paraphrases, after overrides, section 7). `other` always has the fixed description `None of the listed intents`.
- Option labels are the CLINC intent names as given by the dataset (for example `freeze_account`).

Noul questions (one per record):
- Every out-of-scope utterance yields two records, each with its own choice options: one with `out_of_scope` (underlying yes), and one with `about_domain` or `about_intent` (underlying no). The choice answer is `other` in both.
- Option 1 rule: exactly N_oos in-scope records, chosen at random, get `out_of_scope` (underlying no), so the kind is balanced before negation (N_oos: train 250, valid 100, test_indomain 1000, test_unseen_intents 0, so that split has no `out_of_scope`).
- Every other record draws `about_domain` or `about_intent` in the ratio of the configured weights (0.45 to 0.45, so one half each). The configured `out_of_scope` weight of 0.10 is not used: the option 1 rule fixes that kind's count exactly, at 3.7 percent of train records and 34 percent of test_indomain records.
- `about_domain` asks the gold domain in exactly half of its questions, chosen among the in-scope records. A "no" question asks, with probability 0.8 (`p_asked_from_options`), the domain of a distractor option (a named option other than the gold intent whose domain differs from the gold domain), and otherwise a uniformly drawn domain other than the gold one. The 10 domain phrases are written by hand in `jevmark/data/clinc.py` (`DOMAIN_PHRASES`); the intent-to-domain map is the original CLINC release's `domains.json` (section 3).
- `about_intent` asks the gold intent in exactly half of its questions, chosen among the in-scope records. A "no" question asks, with probability 0.8, one of the choice's distractor options (its named options other than the gold intent), and otherwise an allowed intent outside the options. The slot is one of the asked intent's description variants, drawn independently of the variant shown among the options. In `test_unseen_intents` the allowed set is the 20 held-out intents, so every asked description is unseen in training; elsewhere it is the 130 seen intents.
- Feature matching: the gold intent is among the options in 80 percent of in-scope records, and a "no" question asks from the options in 80 percent of cases, so whether the asked intent (or domain) appears among the options carries almost no information about the answer. `meta.nouls` records `asked_from` and `asked_in_options`; the leak probes (section 8) measure what is left.

Held-out intents: 20 intents, 2 per domain, chosen with the build seed. None of their utterances appear in `train` or `valid`, none of them appear there as an option, and none is asked by `about_intent` there (`build.held_out_leaks` checks all three). The list is in section 4.

| Split | Built from | Intents allowed as options and asked intents |
|---|---|---|
| `train` | CLINC `train` minus held-out intents; each out-of-scope utterance yields two records | the 130 seen intents |
| `valid` | CLINC `validation` minus held-out intents | the 130 seen intents |
| `test_indomain` | CLINC `test` minus held-out intents | the 130 seen intents |
| `test_unseen_intents` | every utterance of the 20 held-out intents from all three CLINC splits (20 x 150 = 3000); the choice answer is `other` whenever the gold intent is not among the options | the 20 held-out intents only |

### SST-5 (`SetFit/sst5`)

Each record has the score question `sentiment`, `How positive is the sentiment of this text?`, on one of two hand-written scales in `jevmark/data/sst5.py`:

- 5 levels (`LEVELS`), from very negative (0) to very positive (4); gold is the dataset label.
- 3 levels (`LEVELS_3`): negative, neutral or mixed, positive; labels 0 and 1 map to 0, 2 to 1, 3 and 4 to 2. Exactly 30 percent of the records in every split, chosen at random, use this scale (`sst5.p_three_levels`); `meta.scale` records which.

Every non-neutral record (labels 0, 1, 3, 4) also gets one sentiment noul, `is_positive` or `is_negative` chosen at random per record, after the score question; neutral records (label 2) keep only the score question. The scale is chosen independently of the label. Because the noul comes after the score question, the model cannot see from the score slot whether a noul will follow.

| Split | Built from |
|---|---|
| records in `train` | SST-5 `train` |
| records in `valid` | SST-5 `validation` |
| `test_sst5` | SST-5 `test` |

`train` and `valid` therefore mix CLINC and SST-5 records; the `source` field tells them apart.

### Unseen schemas (evaluation only, never in training)

AG News, emotion and Banking77: one choice question per record, 1000 records per set sampled from the dataset's test split with the build seed. Canonical descriptions come from `generate_descriptions.py` (section 6); no paraphrases.

| Split | Built from | Questions |
|---|---|---|
| `test_agnews` | AG News `test` | choice over all 4 topic labels |
| `test_emotion` | emotion `test` | choice over all 6 emotion labels, then an `expresses_emotion` noul (the gold emotion in exactly half of the records, another emotion otherwise) |
| `test_banking77` | Banking77 `test` | choice over the gold label, 8 distinct distractor labels and `other`: 10 options, gold always present |
| `test_yelp` | Yelp `test` | score question `stars`, `How many stars does this review give the business?`, with five hand-written star levels (`YELP_LEVELS` in `jevmark/data/unseen.py`); the unseen score schema |

Instructions: AG News `Which topic is this news article about?`; emotion `Which emotion does this message express most strongly?`; Banking77 `Which banking request does this message make?`. The Banking77 `other` option has the fixed description `None of the listed options` and is never the gold answer.

`test_yelp` is stratified: 200 reviews per star, sampled with the build seed from the reviews of at most 2500 characters (`unseen.yelp_max_chars`), so every record encodes within 1024 tokens without truncation. The cap keeps 97.4 percent of test reviews (96.2 percent of 1-star up to 98.7 percent of 5-star ones), so the sample leans slightly toward shorter reviews.

Labels are used exactly as the datasets give them, never normalised. This includes two irregular Banking77 names, `Refund_not_showing_up` (capital R) and `reverted_card_payment?` (trailing question mark); both are valid option labels under API_SPEC section 2.

### Form nouls

Form nouls are label-independent noul questions answered from the state text alone (`jevmark/data/form.py`). Because they depend on no gold label, they cannot leak one, so they may sit anywhere in a record, before or after the gold-dependent questions. They add question variety, put a question before the choice or score question in about a third of the records that have one, and teach the model to answer from the text rather than from a question's position.

| Kind | Yes when |
|---|---|
| `word_count_over` | the text has more than `threshold` whitespace-separated words |
| `char_count_over` | the text has more than `threshold` characters |
| `longest_word_over` | the longest run of letters A to Z in the text is longer than `threshold` |
| `contains_number` | the text contains a digit |
| `contains_comma` | the text contains a comma |
| `ends_with_question_mark` | the text, without trailing whitespace, ends with `?` |

- Splits: train, valid, test_indomain, test_unseen_intents and test_sst5 (`form.splits`), the splits built from the training sources. The evaluation-only splits (test_agnews, test_emotion, test_banking77, test_yelp) have none (decision 44): their texts are long news articles and reviews, where a form noul measures counting words or characters in long text, a skill that was never trained and that no decision needs. The fast cycle showed it: 0.43 to 0.58 accuracy with ECE up to 0.51 on those splits.
- Which records get form nouls, which kinds and where they go are drawn at random, independently of the text and of every gold answer (decision 43). Each record gets 0, 1 or 2 form nouls (uniformly; 2 is capped at the number of used kinds), of distinct kinds drawn from the kinds used for its split and source.
- Placement: each form noul goes before the record's first gold-dependent question with probability 1/3 (`form.p_before_first`), at a uniform slot among the form nouls already there, and otherwise at a uniform slot anywhere after that question. A fixed probability keeps "a form noul precedes the choice or score question" independent of how many gold-dependent questions the record has; inserting uniformly among the existing questions would put it there with probability 1/2 in a neutral SST-5 record and 1/3 in the others, which would hint at the label.
- Thresholds are set per split and source, on the texts of that split's records from that source: the threshold whose yes share is closest to one half. A kind is used for a (split, source) only if that share (for a yes or no property, its yes share) is within 48 to 52 percent (`form.max_imbalance`); otherwise it is skipped there. Records are never selected by their answer: a selection that balances a text property ties the form noul's presence to the text, and through the text to the labels.
- `build.assign_phrasings` then gives each form noul its template and polarity as for every other noul kind.
- Answers are computed on the text itself; for a JSON state that is the `text` field, never the rendered JSON (whose timestamps and ids contain digits).

`make data` prints the used kinds with their thresholds and yes shares per split and source (section 5).

### State formats

Exactly 20 percent of the records of every split (`state_format.p_json`, seeded per split) store the state as a JSON object instead of plain text: the text under `text`, plus 1 to 3 fields drawn from a fixed list whose values are random and independent of the record: `channel` (chat, email, sms, web, app, phone), `timestamp` (ISO 8601, 2019 to 2025), `message_id`, `user_id`, `thread_id`, `locale` (six English locales). The extra fields come in random order and `text` at a random position among them. The API renders a JSON state with `json.dumps(state, indent=2, ensure_ascii=False)` (API_SPEC section 4), so the model sees both formats in training and at inference. `meta.state_format` and `meta.state_fields` record which.

## 3. Dataset ids and revisions

Verified with `scripts/check_datasets.py` (datasets 5.0.1, huggingface_hub 1.33.0). Every loader passes the revision below, so a moved or edited dataset cannot change a build silently.

| Role | Id originally named in TASKS.md | Id used | Config | Revision (commit sha) | Splits and rows | Labels |
|---|---|---|---|---|---|---|
| CLINC150 | `clinc_oos` | `clinc/clinc_oos` | `plus` | `155b9c710419136e17307b80d0a13e68cd46b4ec` | train 15250, validation 3100, test 5500 | `intent`: 151 classes (150 intents plus `oos`) |
| SST-5 | `SetFit/sst5` | `SetFit/sst5` | none | `e51bdcd8cd3a30da231967c1a249ba59361279a3` | train 8544, validation 1101, test 2210 | `label` integer 0 to 4, `label_text` |
| AG News | `fancyzhx/ag_news` | `fancyzhx/ag_news` | none | `eb185aade064a813bc0b7f42de02595523103ca4` | train 120000, test 7600 | `label`: World, Sports, Business, Sci/Tech |
| emotion | `dair-ai/emotion` | `dair-ai/emotion` | none | `cab853a1dbdf4c42c2b3ef2173804746df8825fe` | train 16000, validation 2000, test 2000 | `label`: sadness, joy, love, anger, fear, surprise |
| Banking77 | `PolyAI/banking77` | `legacy-datasets/banking77` | none | `f54121560de48f2852f90be299010d1d6dc612ec` | train 10003, test 3080 | `label`: 77 classes |
| Yelp | none (added in data v1.2) | `Yelp/yelp_review_full` | none | `c1f9ee939b7d05667af864ee1cb066393154bf85` | train 650000, test 50000 | `label`: `1 star`, `2 star`, `3 stars`, `4 stars`, `5 stars` (names as given, including `2 star`), 10000 per star in test |

`Yelp/yelp_review_full` is the canonical Yelp review stars dataset (Zhang et al., 2015); the old id `yelp_review_full` redirects to it, and it holds parquet files on `main`, so it is the original repository under its current name, not a mirror.

Notes on the two ids that differ from the originally named ones (both substitutions approved by the human; TASKS.md now uses the new ids):

- `clinc_oos`: datasets 5 accepts only `namespace/name` ids, and the Hub redirects `clinc_oos` to `clinc/clinc_oos`, which holds parquet files for `plus` on `main`. Same repository, current name. It has no domain column, so the intent-to-domain map comes from the original CLINC release: `data/domains.json` in github.com/clinc/oos-eval at commit `976178879e5afa9952f60a1f8d3c834f47a25cee` (the commit that added it; the file is identical on `master` at `828f8093`). It is checked in unmodified as `jevmark/data/clinc_domains.json`, sha256 `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`; the builder refuses to run if the hash differs, and asserts that it maps exactly the 150 dataset intents, 15 to each of the 10 domains.
- `PolyAI/banking77`: the repository holds only a loading script (`banking77.py`) that downloads CSVs from github.com/PolyAI-LDN/task-specific-datasets. datasets 5 no longer runs loading scripts, and the repository has no `refs/convert/parquet`. The parquet mirror `legacy-datasets/banking77` was compared row by row with the original `train.csv` and `test.csv` the script downloads: identical text and category for all 10003 train and 3080 test rows, in the same order, and the same 77 label names in the same order as the script's `dataset_infos.json`. `mteb/banking77` was rejected: it has 9993 train and 3076 test rows, so it is not a faithful copy.

## 4. Held-out intents

Unchanged in data v1.1, v1.2 and v1.3 (the held-out stream does not depend on the other build parameters). Drawn with seed 0, two per domain (`random.Random("0:held_out")`, domains and intents in sorted order). None of them appears in `train` or `valid`, as an utterance or as an option; `test_unseen_intents` offers only these intents plus `other`, and its `about_intent` questions ask only these intents.

| Domain | Held-out intents |
|---|---|
| auto_and_commute | current_location, traffic |
| banking | account_blocked, freeze_account |
| credit_cards | international_fees, replacement_card_duration |
| home | order_status, shopping_list_update |
| kitchen_and_dining | food_last, restaurant_reviews |
| meta | change_speed, user_name |
| small_talk | greeting, what_can_i_ask_you |
| travel | carry_on, translate |
| utility | find_phone, spelling |
| work | payday, taxes |

## 5. Split sizes

From `make data` with data v1.3, seed 0 (`scripts/build_data.py`). Max tokens is the longest encoded record with the pinned reference tokenizer; every record must fit within 1024.

| Split | Records | By source | Max tokens |
|---|---|---|---|
| `train` | 22023 | CLINC 13482, SST-5 8541 | 392 |
| `valid` | 3896 | CLINC 2795, SST-5 1101 | 378 |
| `test_indomain` | 5900 | CLINC 5900 | 385 |
| `test_unseen_intents` | 3000 | CLINC 3000 | 381 |
| `test_sst5` | 2210 | SST-5 2210 | 295 |
| `test_agnews` | 1000 | AG News 1000 | 289 |
| `test_emotion` | 1000 | emotion 1000 | 247 |
| `test_banking77` | 1000 | Banking77 1000 | 293 |
| `test_yelp` | 1000 | Yelp 1000 | 742 |

CLINC record counts are in-scope utterances plus twice the out-of-scope utterances, less the utterances dropped as duplicates of test texts (section 8): `train` 13000 + 2 x 250 - 18, `valid` 2600 + 2 x 100 - 5; SST-5 `train` 8544 - 3.

Form noul kinds used, with threshold and yes share on the split's texts (a kind is used within 48 to 52 percent); the evaluation-only splits have no form nouls:

| Split | Source | Used kinds |
|---|---|---|
| `train` | SST-5 | word_count_over >18 49.6%, char_count_over >98 49.8% |
| `train` | CLINC | char_count_over >38 48.9% |
| `valid` | SST-5 | word_count_over >18 50.7%, char_count_over >99 50.4% |
| `valid` | CLINC | char_count_over >37 49.8% |
| `test_indomain` | CLINC | char_count_over >39 49.5% |
| `test_unseen_intents` | CLINC | char_count_over >37 49.4% |
| `test_sst5` | SST-5 | word_count_over >18 50.0%, char_count_over >99 49.9% |

Questions per record, form nouls per record (count: records) and the share of JSON states:

| Split | Questions per record | Form nouls per record | JSON states |
|---|---|---|---|
| `train` | 1: 540, 2: 7520, 3: 11712, 4: 2251 | 0: 7496, 1: 11756, 2: 2771 | 20.0% |
| `valid` | 1: 79, 2: 1263, 3: 2239, 4: 315 | 0: 1261, 1: 2251, 2: 384 | 20.0% |
| `test_indomain` | 2: 1899, 3: 4001 | 0: 1899, 1: 4001 | 20.0% |
| `test_unseen_intents` | 2: 1009, 3: 1991 | 0: 1009, 1: 1991 | 20.0% |
| `test_sst5` | 1: 119, 2: 728, 3: 725, 4: 638 | 0: 715, 1: 719, 2: 776 | 20.0% |
| `test_agnews` | 1: 1000 | 0: 1000 | 20.0% |
| `test_emotion` | 2: 1000 | 0: 1000 | 20.0% |
| `test_banking77` | 1: 1000 | 0: 1000 | 20.0% |
| `test_yelp` | 1: 1000 | 0: 1000 | 20.0% |

Noul balance per kind. Phrasing-only accuracy answers each phrasing (template and polarity) with its own majority answer in the split; the build fails above 55 percent.

| Split | Kind | n | Yes share | Negated share | Phrasing-only accuracy |
|---|---|---|---|---|---|
| `train` | `about_domain` | 6539 | 50.0% | 50.0% | 50.0% |
| `train` | `about_intent` | 6443 | 50.0% | 50.0% | 50.0% |
| `train` | `char_count_over` | 13074 | 50.0% | 50.0% | 51.0% |
| `train` | `is_negative` | 3526 | 50.0% | 50.0% | 52.5% |
| `train` | `is_positive` | 3391 | 50.0% | 50.0% | 51.8% |
| `train` | `out_of_scope` | 500 | 49.8% | 50.2% | 50.2% |
| `train` | `word_count_over` | 4224 | 50.0% | 50.0% | 50.3% |
| `valid` | `about_domain` | 1293 | 50.1% | 50.0% | 50.1% |
| `valid` | `about_intent` | 1302 | 49.9% | 50.0% | 50.1% |
| `valid` | `char_count_over` | 2452 | 50.0% | 50.0% | 50.4% |
| `valid` | `is_negative` | 448 | 50.2% | 50.2% | 50.4% |
| `valid` | `is_positive` | 424 | 50.2% | 50.2% | 52.4% |
| `valid` | `out_of_scope` | 200 | 49.5% | 50.5% | 50.5% |
| `valid` | `word_count_over` | 567 | 49.9% | 50.3% | 52.6% |
| `test_indomain` | `about_domain` | 1949 | 50.0% | 50.0% | 50.1% |
| `test_indomain` | `about_intent` | 1951 | 50.0% | 50.0% | 50.0% |
| `test_indomain` | `char_count_over` | 4001 | 50.0% | 50.0% | 50.2% |
| `test_indomain` | `out_of_scope` | 2000 | 50.0% | 50.0% | 50.1% |
| `test_unseen_intents` | `about_domain` | 1467 | 50.0% | 49.9% | 50.0% |
| `test_unseen_intents` | `about_intent` | 1533 | 49.9% | 50.0% | 50.1% |
| `test_unseen_intents` | `char_count_over` | 1991 | 50.0% | 50.0% | 50.4% |
| `test_sst5` | `char_count_over` | 1156 | 50.0% | 50.0% | 51.6% |
| `test_sst5` | `is_negative` | 932 | 50.0% | 49.9% | 51.2% |
| `test_sst5` | `is_positive` | 889 | 49.9% | 49.9% | 51.1% |
| `test_sst5` | `word_count_over` | 1115 | 50.0% | 50.0% | 51.0% |
| `test_emotion` | `expresses_emotion` | 1000 | 50.0% | 50.0% | 50.0% |

Score gold levels per scale:

| Split | Scale | n | Levels |
|---|---|---|---|
| `train` | `sst5_3_levels` | 2562 | 0: 971 (38%), 1: 491 (19%), 2: 1100 (43%) |
| `train` | `sst5_5_levels` | 5979 | 0: 785 (13%), 1: 1553 (26%), 2: 1133 (19%), 3: 1622 (27%), 4: 886 (15%) |
| `valid` | `sst5_3_levels` | 330 | 0: 135 (41%), 1: 62 (19%), 2: 133 (40%) |
| `valid` | `sst5_5_levels` | 771 | 0: 97 (13%), 1: 196 (25%), 2: 167 (22%), 3: 192 (25%), 4: 119 (15%) |
| `test_sst5` | `sst5_3_levels` | 663 | 0: 288 (43%), 1: 116 (17%), 2: 259 (39%) |
| `test_sst5` | `sst5_5_levels` | 1547 | 0: 185 (12%), 1: 439 (28%), 2: 273 (18%), 3: 371 (24%), 4: 279 (18%) |
| `test_yelp` | `yelp_5_stars` | 1000 | 0: 200 (20%), 1: 200 (20%), 2: 200 (20%), 3: 200 (20%), 4: 200 (20%) |

Gold option position is checked conditional on K, the number of options: for every split and every K with at least 100 choice questions, no position's share may deviate from 1/K by more than 4 binomial standard deviations. The unconditional answer-letter histogram is not uniform and cannot be: every question has options A and B (noul questions have exactly two) while N exists only when K = 14, and score levels follow the datasets' label distributions. `build_data.py` prints it for reference.

## 6. Description generation

`scripts/generate_descriptions.py` writes one JSON file per dataset under `jevmark/data/descriptions/`, generated once with the OpenAI API and checked in. The human runs the real generation and reviews 20 sampled lines before the files are used. The model name and temperature are arguments and are recorded in each file. The run stops before spend passes `--max-usd` (default 5 dollars, the TASKS.md guardrail), computed from the token usage the API reports and the prices passed on the command line. `--dry-run` prints the exact prompts for three labels and makes no call.

Generation run of record: model `gpt-4.1-mini`, temperature 0.3, prompt version 1, 237 calls (one per label, no retries needed), total spend 0.0699 USD across two runs, run on 2026-09-24 by the human. Spend is recorded here only, not in the JSON files. The four files are committed exactly as generated; corrections go through `overrides.json` (section 7), never by editing the generated files.

One API call per label, with JSON output (`response_format` `json_object`). CLINC's out-of-scope class `oos` is not described. Each reply is stripped of edge whitespace and must be a single non-empty line of at most 200 characters; CLINC replies must carry exactly two paraphrases that differ from each other and from the canonical text. An invalid reply is retried at most `--retries` times (default 2), then the run stops. Progress is saved after each label, so a rerun resumes.

| Key | Title | Subject | Paraphrases | File |
|---|---|---|---|---|
| `clinc` | CLINC150 intent classification | a request or message sent to a virtual assistant | 2 | `jevmark/data/descriptions/clinc_intents.json` |
| `banking77` | Banking77 intent classification | a customer message sent to a bank's support chat | 0 | `jevmark/data/descriptions/banking77_labels.json` |
| `ag_news` | AG News topic classification | a news article headline and summary | 0 | `jevmark/data/descriptions/ag_news_labels.json` |
| `emotion` | Emotion classification | a short personal message, such as a tweet | 0 | `jevmark/data/descriptions/emotion_labels.json` |

System prompt (prompt version 1):

```
You write option descriptions for a text classifier. The classifier reads a message and picks one label from a list, using each label's description to decide. A good description is one plain English sentence of 6 to 16 words that states what a message with this label is about or asks for, precise enough to tell the label apart from the other labels in the dataset. Do not use line breaks or quotation marks. Reply with a JSON object only.
```

User prompt template. `{label}` is the dataset's label name as given (for example `freeze_account`, `Sci/Tech`), `{all_labels}` is every label of the dataset joined with `, `, and `{task}` is one of the two task lines below:

```
Dataset: {title}
Each example is {subject}.
Label to describe: {label}
All labels in this dataset, for contrast: {all_labels}

{task}
```

Task line for Banking77, AG News and emotion:

```
Return {"canonical": "<description>"}.
```

Task line for CLINC150:

```
Return {"canonical": "<description>", "paraphrases": ["<paraphrase 1>", "<paraphrase 2>"]}. Each paraphrase says the same thing as the canonical description in different words.
```

Output file format:

```json
{
  "source": "clinc/clinc_oos/plus@155b9c710419136e17307b80d0a13e68cd46b4ec",
  "generator": {"script": "scripts/generate_descriptions.py", "prompt_version": 1, "model": "<model>", "temperature": 0.3},
  "descriptions": {
    "transfer": {"canonical": "<one line>", "paraphrases": ["<one line>", "<one line>"]}
  }
}
```

Real run, by the human (key from `OPENAI_API_KEY`; `.env` is gitignored):

```
uv run --extra baselines --env-file .env python scripts/generate_descriptions.py \
    --model <model> --temperature 0.3 \
    --usd-per-million-input <current price> --usd-per-million-output <current price>
```

## 7. Description overrides and normalisation

Builders read descriptions only through `jevmark/data/description_loader.py`, never the JSON files directly. The loader starts from the generated files (section 6), applies `jevmark/data/descriptions/overrides.json`, then normalises every text.

Overrides:

- Shape: `{"<dataset key>": {"<label>": {"canonical": "...", "paraphrases": ["..."], "reason": "..."}}}`, with the dataset keys `clinc`, `banking77`, `ag_news` and `emotion`.
- An override replaces the generated entry for that label whole. It must name a label that exists in the generated file, carry exactly as many paraphrases as the generated entry (2 for CLINC, 0 otherwise) and a non-empty reason, and have exactly those three fields. Any other entry fails loudly.
- Every override is listed in the table below with its reason; `tests/test_description_loader.py` fails if one is missing.

Normalisation, applied to generated and overridden texts alike, in this order:

1. The curly apostrophe (U+2019) becomes a straight apostrophe.
2. One trailing period is stripped, so option lines never differ only by a final period.

After normalisation every text must still be a valid option description under API_SPEC section 2 (single line, non-empty, no edge whitespace), or loading fails.

| Dataset / label | Reason |
|---|---|
| `clinc/reminder` | Generated text said create or manage; the data is reading existing reminders, and creating belongs to reminder_update |
| `clinc/reminder_update` | Generated text covered only changing; the data is mostly setting new reminders |
| `clinc/todo_list` | Generated text said managing or creating, which overlaps todo_list_update; the data is checking the list |
| `clinc/shopping_list` | Generated text said create or manage, which overlaps shopping_list_update; the data is reading the list |
| `clinc/calendar` | Generated text said managing, which overlaps calendar_update; the data is checking events |
| `clinc/meeting_schedule` | Generated text described arranging a meeting, which duplicates schedule_meeting; the data asks about existing meetings |
| `clinc/schedule_meeting` | The data includes room availability questions the generated text did not cover |
| `clinc/travel_notification` | Generated text described updating itineraries; the data is informing the bank about travel |
| `clinc/user_name` | Generated text included providing the name, which overlaps change_user_name; the data asks what name is on file |
| `clinc/transactions` | Generated paraphrase mentioned money transfers, which overlaps transfer; the data is listing past transactions |
| `clinc/insurance_change` | Generated text covered only modifying existing cover; the data also includes getting new insurance |
| `clinc/new_card` | Generated text drifted toward activation; the data is applying for a card |
| `clinc/time` | Generated text said time-related information, which is broad enough to cover timezone; the data asks the current time |

## 8. CPU gates: leak probes and duplicate check

`make data` runs two hard gates after the build (decision 42). Both read the JSONL files in `data/` and write their full results next to them (`leak_probe.json`, `duplicates.json`, gitignored with the data).

### Leak probes (`scripts/leak_probe.py`)

A leak is anything other than the state that predicts a gold answer. For every split and every question, two probe models predict the gold answer from state-free features, each with 5-fold cross-validation (seed 0): logistic regression on all features, and a histogram gradient-boosted tree ensemble (scikit-learn `HistGradientBoostingClassifier`, one thread) that can represent interactions a linear probe cannot, on the 256 most frequent feature columns of the group (chosen without the targets; this keeps every phrasing and structure column and the most common words). Each model runs four probes:

| Probe | Features |
|---|---|
| phrasing | the noul's template and polarity; the instructions for choice and score |
| question text | bag of words and bigrams of the question block, options included |
| structure | number of questions seen, position, kinds seen before, K, asked-in-options flags (an asked intent among the choice labels, an asked domain among their domains), the asked intent's domain equal to an asked domain, scale, state format and number of JSON fields |
| cross-question | bag of words, kind and K of every question before this one, and the same relational flags |

Rules:

- Attention is causal, so every feature is computed on the question and the questions before it, never on later ones; a noul after the score question cannot leak into it.
- Groups: noul questions by kind; choice questions by question id, with three targets (gold is `other`, when `other` can be gold; gold label, when every question in the group offers the same labels; gold position, without the question-text probe, since a bag of words has no positions); score questions by scale. Groups under 50 questions or with a single answer are skipped.
- Noul targets: the phrasing probe predicts the stored gold; the other three predict the underlying answer (negation undone), because negation flips the stored gold in half of every group and a linear probe cannot undo that.
- Gate, the same for both models: a probe more than 10 points over its group's majority baseline fails outright, with no permutation test. A probe more than 3 points over it is rerun on 200 copies of its targets shuffled at random (shuffling keeps the baseline and breaks every link to the features), and fails if its lift is also above the 99th percentile of those runs. The summary lists every probe above 3 points, passing or not, with n, lift and p value, because with about 250 probe results per build chance alone puts a few past 3 points.

v1.3, from `make data` (seed 0, after decision 44 removed form nouls from the evaluation-only splits): the gate passes. 354 probe results (177 per model); the largest lift is +3.0 points; one probe is above 3 points, logistic and within chance, and no boosting probe is above 3 points:

| Split | Group | Probe | n | Lift | p | 99th percentile of shuffled runs |
|---|---|---|---|---|---|---|
| `valid` | `noul/out_of_scope` | structure (logistic) | 200 | +3.0 | 0.194 | +9.0 |

Before decision 44 the gate also passed, with 418 probe results and a second chance flag, test_agnews `noul/word_count_over` (cross-question, logistic, +5.5 points on 311 questions, p 0.040, 99th percentile +7.7); that group no longer exists.

v1.2, rebuilt from commit 0163a74 (byte-identical to the files the v1.2 `sft_06b` evaluation read, by the sha256 values in its `metrics.json`): the gate fails on 26 probe results, and each known leak is found. Lifts in points over the majority baseline; the single-feature rows refit the probe on only the feature that carries the leak:

| Leak | Split | Group | Probe | n | Lift |
|---|---|---|---|---|---|
| SST-5 order: a sentiment noul before the score means not neutral | `train` | `score/sentiment/sst5_5_levels` | structure | 5981 | +5.0 (p 0.005, 99th percentile +0.5) |
| | `valid` | `score/sentiment/sst5_5_levels` | cross-question | 771 | +7.0 (p 0.005) |
| | `train` | same, kinds seen and position only | structure | 5981 | +5.0 |
| `about_intent` priming: the asked intent among the options means yes | `train` | `noul/about_intent` | structure | 13500 | +22.3 |
| | `train` | same, asked-in-options flag only | structure | 13500 | +18.6 |
| `about_intent` before the choice primes it | `test_indomain` | `choice/intent/gold_other` | structure | 5900 | +18.4 |
| | `test_indomain` | same, asked-in-options flag only | structure | 5900 | +16.7 |
| `about_intent` and `about_domain` correlated through the gold | `train` | `noul/about_domain` | cross-question | 13000 | +17.5 |
| | `train` | `noul/about_domain`, domain agreement only | cross-question | 13000 | +12.6 |
| | `train` | `noul/about_intent`, domain agreement only | cross-question | 13500 | +9.6 |
| emotion (found by the probes): a skewed gold emotion and a uniform asked one | `test_emotion` | `noul/expresses_emotion` | question text | 1000 | +14.9 |
| | `test_emotion` | `choice/label/gold_label` (the noul came first) | cross-question | 1000 | +7.2 (p 0.005) |

Both models side by side on v1.2 (lifts in points; structure and cross-question probes, where every leak shows; the 50 shuffled runs per flagged probe used for this demonstration only, so the smallest possible p is 0.020, while the v1.3 gate uses 200). With both models the v1.2 gate fails on 58 of 290 probe results:

| Split | Group | n | Logistic structure | Logistic cross | Boosting structure | Boosting cross |
|---|---|---|---|---|---|---|
| `train` | `score/sentiment/sst5_5_levels` | 5981 | +5.0 | +4.5 | +5.0 | +4.6 |
| `valid` | `score/sentiment/sst5_5_levels` | 771 | +6.2 | +7.0 | +6.2 | +6.6 |
| `train` | `noul/about_intent` | 13500 | +22.3 | +20.9 | +22.1 | +22.1 |
| `test_indomain` | `choice/intent/gold_other` | 5900 | +18.4 | +17.2 | +18.0 | +18.1 |
| `train` | `noul/about_domain` | 13000 | +19.9 | +17.5 | +19.9 | +19.4 |
| `test_indomain` | `noul/out_of_scope` | 2000 | +7.1 | +5.8 | +7.1 | +6.6 |
| `test_emotion` | `choice/label/gold_label` | 1000 | -0.9 | +7.2 | -0.9 | +7.4 |
| `test_emotion` | `noul/expresses_emotion` (question text) | 1000 | +14.9 | | +14.9 | |

The tree ensemble finds the same leaks at about the same size, and 1 to 2 points more on the cross-question probes of `about_intent` and `about_domain`, where the leak is a relation between two questions; on v1.3 it finds nothing above 3 points, so no interaction leak was hidden from the linear probe.

The first v1.3 build also failed the probes: `expresses_emotion` at +18.7 points (question text), fixed by asking each emotion as often in "no" as in "yes" questions, and a form-noul placement and balancing scheme that tied form nouls to the labels, fixed by drawing them at random (decision 43).

### Duplicate check (`scripts/check_duplicates.py`)

State texts (the `text` field of a JSON state) are normalised with `dedup.normalise` (NFKC, lower case, every run of characters other than letters and digits to one space, edges stripped). The check fails on any normalised text shared by train and a test split, or by valid and a test split, and reports the share of test records that share a word 8-gram with train or valid.

The source datasets contain such duplicates across their own splits (for example `where did you grow up` and `next song please` in CLINC150, `no` and `what s next` in SST-5). train and valid therefore drop every utterance whose normalised text occurs in a source split that a test split draws from (`dedup.test_texts`: CLINC test and every held-out-intent utterance, SST-5 test, the test splits of AG News, emotion, Banking77 and Yelp); the test splits are unchanged (decision 43). This removes 18 CLINC and 3 SST-5 records from train and 5 CLINC records from valid.

v1.3 result: no exact match in any pair. 8-gram overlap, the share of test records with at least one word 8-gram also in the reference split:

| Test split | vs train | vs valid | Test records with an 8-gram |
|---|---|---|---|
| `test_indomain` | 2.46% | 0.64% | 3479 of 5900 |
| `test_unseen_intents` | 0.33% | 0.10% | 1697 of 3000 |
| `test_sst5` | 0.18% | 0.09% | 1919 of 2210 |
| `test_agnews` | 0.00% | 0.00% | 1000 of 1000 |
| `test_emotion` | 0.00% | 0.00% | 873 of 1000 |
| `test_banking77` | 0.60% | 0.10% | 720 of 1000 |
| `test_yelp` | 0.00% | 0.00% | 991 of 1000 |


## 9. Metrics: gold-dependent headline, form nouls apart

In every split's `metrics.json` block, `overall` and the per-type blocks (`noul`, `choice`, `score`), with their symmetry and order sensitivity, cover the gold-dependent questions only: the questions whose answer depends on a gold label, the capability the model is for. Form nouls are reported in their own `form` block, with the same metrics (accuracy and ECE with bootstrap intervals, Brier, NLL, reliability, coverage, yes rate, `by_kind`, `by_question_position`, symmetry, order sensitivity), so they never move a split's headline numbers (decision 44).

## 10. Baseline subset

`data/baseline_subset.json` is the one file under `data/` that is committed (a `.gitignore` exception). It holds, for every split, the ids of 500 records drawn with the fast-cycle sampler and the `--limit` seed (the records `evaluate.py --limit 500` evaluates), the ids of a 200-record sub-subset drawn from those, and the sha256 of each data file they came from (data v1.3, identical to the hashes in `runs/sft_06b/metrics.json`). The generative baselines run on it, and `scripts/compare_baselines.py` restricts every jevmark run to it (task 1.8, decision 47). `scripts/make_baseline_subset.py` wrote it once and refuses to overwrite it without `--force`.

