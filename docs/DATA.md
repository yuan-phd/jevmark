# jevmark: Data

Data files are JSONL under `data/` (gitignored, rebuilt by `make data`). Option descriptions are JSON under `jevmark/data/descriptions/` and are checked in. This file is the contract between data building, training and evaluation.

## 1. Record schema

One record is one `systemone` request plus its gold answers. Every record passes `Request.from_dict({"state": ..., "questions": ...})` unchanged.

A real record from `data/train.jsonl` (data v1.2, seed 0):

```json
{
  "id": "clinc-train-000040",
  "source": "clinc/clinc_oos/plus",
  "split": "train",
  "state": "transfer 200 dollars from paypal to savings",
  "questions": {
    "about_intent": {
      "type": "noul",
      "instructions": "Does this description fail to fit the message: Requesting to send funds or shift money between accounts?"
    },
    "about_domain": {
      "type": "noul",
      "instructions": "Is the topic of this message something other than cars, driving or getting around?"
    },
    "intent": {
      "type": "choice",
      "instructions": "Which intent does this message express?",
      "criteria": {
        "rewards_balance": "Request information on your existing rewards balance",
        "credit_score": "Inquire about the status of your credit score",
        "other": "None of the listed intents",
        "pto_request_status": "Inquire about the progress of a PTO request",
        "insurance": "Inquiries about insurance plans and policy details",
        "transfer": "A request to move money between accounts or send funds to someone",
        "calculator": "Asks to compute math operations or solve arithmetic questions"
      }
    }
  },
  "gold": {
    "about_intent": "false",
    "about_domain": "true",
    "intent": "transfer"
  },
  "meta": {
    "source_split": "train",
    "source_index": 140,
    "gold_intent": "transfer",
    "domain": "banking",
    "gold_in_options": true,
    "nouls": {
      "about_domain": {
        "template": 2,
        "negated": true,
        "slot": "cars, driving or getting around",
        "asked_domain": "auto_and_commute"
      },
      "about_intent": {
        "template": 0,
        "negated": true,
        "slot": "Requesting to send funds or shift money between accounts",
        "asked_intent": "transfer"
      }
    }
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `<source short name>-<split>-<6-digit index>`, unique across all files |
| `source` | string | dataset id plus config, as in section 3 (for example `clinc/clinc_oos/plus`) |
| `split` | string | the jevmark split the record belongs to (section 2), equal to the file name stem |
| `state` | string | the input text, unchanged from the dataset |
| `questions` | object | question id to question definition, exactly as API_SPEC section 2; key order is the question order in the encoded sequence. A noul question's id is its kind |
| `gold` | object | question id to gold answer: an option label (string) for choice, `"true"` or `"false"` for noul, a level index (integer) for score |
| `meta` | object | provenance and build facts, never read by the model; keys per source in the table below |

`meta` keys:

| Source | Keys |
|---|---|
| CLINC | `source_split`, `source_index`, `gold_intent` (intent name, or `oos`), `domain` (null for `oos`), `gold_in_options` (null for `oos`), `nouls` |
| SST-5 | `source_split`, `source_index`, `label_text`, `scale` (`sst5_5_levels` or `sst5_3_levels`), `nouls` (empty for neutral records) |
| emotion | `source_split`, `source_index`, `nouls` |
| AG News, Banking77 | `source_split`, `source_index`, `nouls` (always empty) |
| Yelp | `source_split`, `source_index`, `label_text` (the dataset's label name, `1 star` to `5 stars`), `scale` (`yelp_5_stars`), `nouls` (always empty) |

`meta.nouls` maps each noul question id (its kind) to `template` (index into that kind's templates in `jevmark/data/negation.py`), `negated` (stored in the negated phrasing, gold flipped), `slot` (the text filled into the template, or null) and, where it applies, `asked_domain`, `asked_intent` or `asked_emotion`.

Rules:

- Option order inside every `criteria` object of a choice question is a seeded random order fixed at build time, for every split including all test splits (decision 19). Training reshuffles on top of that every epoch (`encode.shuffle_request`). Gold is stored as a label, never as a letter, so reshuffling never needs to touch it. Score levels keep their order.
- Question order inside `questions` is also seeded and fixed at build time.
- Every text that goes on a rendered line obeys API_SPEC section 2: single line, no edge whitespace, descriptions non-empty or null. The builder validates each record with `Request.from_dict` and fails loudly on any record that does not pass.
- Records whose encoding exceeds the training `max_tokens` are dropped by the training loader, never truncated (API_SPEC section 4); `build_data.py` checks that every record encodes within 1024 tokens, so none is dropped.

## 2. Splits and how each is built

Current data version: **v1.2** (`version` in `configs/data.yaml`, decision 41). v1.1 (decision 40) added negated noul questions; v1 had neither. The v1 runs (`runs/sft_06b`, `runs/base_06b` at commit 3fe318d, and the first B0 runs) were evaluated on v1.

All randomness comes from one build seed in `configs/data.yaml`: **seed 0**. Each split uses its own generators, `random.Random(f"{seed}:{name}")` with names such as `train`, `train:sst5`, `train:phrasing`, and the held-out intents use `"{seed}:held_out"`, so rebuilding one split never changes another. The other build parameters are in the same file.

### Noul questions: kinds, templates and balance

| Kind | Where | Question | Underlying yes |
|---|---|---|---|
| `about_domain` | CLINC records without `out_of_scope` | is the message about a domain | the asked domain is the gold domain |
| `out_of_scope` | CLINC | is the request outside the assistant's scope | the utterance is out of scope |
| `about_intent` | every CLINC record | does an intent description fit the message | the asked intent is the gold intent |
| `is_positive`, `is_negative` | non-neutral SST-5 records, one of the two at random | is the sentiment positive (negative) | label 3 or 4 (0 or 1) |
| `expresses_emotion` | every `test_emotion` record (evaluation only) | does the message express an emotion | the asked emotion is the gold emotion |

Every kind has two or three templates, each a positive phrasing and its negation (`jevmark/data/negation.py`, which maps every phrasing to its pair in both directions and parses template, polarity and slot back from a rendered instruction):

| Kind | Templates (positive / negated) |
|---|---|
| `about_domain` | 0: `Is this message about {slot}?` / `Is this message about something other than {slot}?`<br>1: `Does this message concern {slot}?` / `Does this message concern something other than {slot}?`<br>2: `Is the topic of this message {slot}?` / `Is the topic of this message something other than {slot}?` |
| `out_of_scope` | 0: `Is this request outside what a banking, travel, home, work or everyday assistant can help with?` / `Is this request something a banking, travel, home, work or everyday assistant can help with?`<br>1: `Would a banking, travel, home, work or everyday assistant be unable to help with this request?` / `Would a banking, travel, home, work or everyday assistant be able to help with this request?`<br>2: `Is this request out of scope for a banking, travel, home, work or everyday assistant?` / `Is this request within scope for a banking, travel, home, work or everyday assistant?` |
| `about_intent` | 0: `Does this description fit the message: {slot}?` / `Does this description fail to fit the message: {slot}?`<br>1: `Is this message an example of the following: {slot}?` / `Is this message an example of something other than the following: {slot}?` |
| `is_positive` | 0: `Is the sentiment of this text positive?` / `Is the sentiment of this text something other than positive?`<br>1: `Does this text express a favourable opinion?` / `Does this text express anything other than a favourable opinion?` |
| `is_negative` | 0: `Is the sentiment of this text negative?` / `Is the sentiment of this text something other than negative?`<br>1: `Does this text express an unfavourable opinion?` / `Does this text express anything other than an unfavourable opinion?` |
| `expresses_emotion` | 0: `Does this message express {slot}?` / `Does this message express something other than {slot}?`<br>1: `Is {slot} the main emotion in this message?` / `Is something other than {slot} the main emotion in this message?` |

Balance, in two steps:

1. The builder fixes each noul question's underlying answer with exact counts wherever the data allows: `out_of_scope` asks exactly as many in-scope utterances (no) as there are out-of-scope ones (yes); `about_domain` asks the gold domain in exactly half of its questions, counting the out-of-scope utterances' `about_domain` questions, which are always no; `about_intent` asks the gold intent in exactly half of all CLINC records (only in-scope records can be yes); `expresses_emotion` asks the gold emotion in exactly half of the records. The SST-5 kinds follow the dataset's labels.
2. `build.assign_phrasings` then gives every question its template and polarity: within each kind and underlying answer, questions take (template, polarity) pairs from consecutive shuffled blocks that hold each pair once. Exactly half of each group is negated (gold flipped), and every phrasing carries about the kind's underlying balance, so no phrasing predicts the answer.

`build_data.py` fails the build if, for any noul kind in any split, the yes share is outside 40 to 60 percent or the phrasing-only accuracy (answering each phrasing with its own majority answer in that split, the best any rule that ignores the message can do) is above 55 percent (`max_phrasing_only_accuracy`). It prints both per kind per split.

### CLINC150 (`clinc/clinc_oos`, config `plus`)

Every CLINC record carries three questions in a seeded random order: `intent` (choice), one of `about_domain` or `out_of_scope` (noul), and `about_intent` (noul), so the model always trains on multi-question sequences.

Choice question `intent`:
- Instructions: `Which intent does this message express?`
- K is uniform in 3..10 and counts every option including `other`.
- For an in-scope utterance, the gold intent is among the options with probability 0.8. The other options are distinct distractor intents drawn uniformly from the intents allowed in that split, then `other` is added.
- If the gold intent is not among the options, or the utterance is out of scope, the gold answer is `other`.
- Each intent's description is drawn at random from its three variants (canonical plus two paraphrases, after overrides, section 7). `other` always has the fixed description `None of the listed intents`.
- Option labels are the CLINC intent names as given by the dataset (for example `freeze_account`).

Noul questions:
- Every out-of-scope utterance yields two records: one with `out_of_scope` (underlying yes) and one with `about_domain` for a uniformly drawn domain (underlying no). Each record draws its own choice options; the choice answer is `other` in both.
- Every in-scope utterance yields one record. Exactly N_oos of them, chosen at random, get `out_of_scope` (underlying no), so the kind is balanced before negation (N_oos: train 250, valid 100, test_indomain 1000, test_unseen_intents 0, so that split has no `out_of_scope`). The rest get `about_domain`, asking the gold domain for a random subset sized so that exactly half of all `about_domain` questions are yes, and a uniformly drawn other domain otherwise. The 10 domain phrases are written by hand in `jevmark/data/clinc.py` (`DOMAIN_PHRASES`); the intent-to-domain map is the original CLINC release's `domains.json` (section 3).
- `about_intent` asks, for exactly half of all records in the split, the gold intent, and otherwise a uniformly drawn other intent from the split's allowed set; out-of-scope records always get another intent. The slot is one of the asked intent's description variants, drawn at random. In `test_unseen_intents` the allowed set is the 20 held-out intents, so every asked description is unseen in training; elsewhere it is the 130 seen intents.

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

Every non-neutral record (labels 0, 1, 3, 4) also gets one sentiment noul, `is_positive` or `is_negative` chosen at random per record, in a seeded random question order; neutral records (label 2) keep only the score question.

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
| `test_emotion` | emotion `test` | choice over all 6 emotion labels, plus an `expresses_emotion` noul (the gold emotion in exactly half of the records, another emotion otherwise) in a seeded random order |
| `test_banking77` | Banking77 `test` | choice over the gold label, 8 distinct distractor labels and `other`: 10 options, gold always present |
| `test_yelp` | Yelp `test` | score question `stars`, `How many stars does this review give the business?`, with five hand-written star levels (`YELP_LEVELS` in `jevmark/data/unseen.py`); the unseen score schema |

Instructions: AG News `Which topic is this news article about?`; emotion `Which emotion does this message express most strongly?`; Banking77 `Which banking request does this message make?`. The Banking77 `other` option has the fixed description `None of the listed options` and is never the gold answer.

`test_yelp` is stratified: 200 reviews per star, sampled with the build seed from the reviews of at most 2500 characters (`unseen.yelp_max_chars`), so every record encodes within 1024 tokens without truncation. The cap keeps 97.4 percent of test reviews (96.2 percent of 1-star up to 98.7 percent of 5-star ones), so the sample leans slightly toward shorter reviews.

Labels are used exactly as the datasets give them, never normalised. This includes two irregular Banking77 names, `Refund_not_showing_up` (capital R) and `reverted_card_payment?` (trailing question mark); both are valid option labels under API_SPEC section 2.

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

Unchanged in data v1.1 and v1.2 (the held-out stream does not depend on the other build parameters). Drawn with seed 0, two per domain (`random.Random("0:held_out")`, domains and intents in sorted order). None of them appears in `train` or `valid`, as an utterance or as an option; `test_unseen_intents` offers only these intents plus `other`, and its `about_intent` questions ask only these intents.

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

From `make data` with data v1.2, seed 0 (`scripts/build_data.py`). Max tokens is the longest encoded record with the pinned reference tokenizer; every record must fit within 1024.

| Split | Records | By source | Max tokens |
|---|---|---|---|
| `train` | 22044 | CLINC 13500, SST-5 8544 | 300 |
| `valid` | 3901 | CLINC 2800, SST-5 1101 | 291 |
| `test_indomain` | 5900 | CLINC 5900 | 289 |
| `test_unseen_intents` | 3000 | CLINC 3000 | 288 |
| `test_sst5` | 2210 | SST-5 2210 | 190 |
| `test_agnews` | 1000 | AG News 1000 | 256 |
| `test_emotion` | 1000 | emotion 1000 | 203 |
| `test_banking77` | 1000 | Banking77 1000 | 273 |
| `test_yelp` | 1000 | Yelp 1000 | 710 |

CLINC record counts are in-scope utterances plus twice the out-of-scope utterances (for example `train`: 13000 + 2 x 250).

Noul balance per kind. Phrasing-only accuracy answers each phrasing (template and polarity) with its own majority answer in the split; the build fails above 55 percent.

| Split | Kind | n | Yes share | Negated share | Phrasing-only accuracy |
|---|---|---|---|---|---|
| `train` | `about_domain` | 13000 | 50.0% | 50.0% | 50.0% |
| `train` | `about_intent` | 13500 | 50.0% | 50.0% | 50.0% |
| `train` | `is_negative` | 3465 | 50.0% | 50.0% | 51.5% |
| `train` | `is_positive` | 3455 | 50.0% | 50.0% | 52.8% |
| `train` | `out_of_scope` | 500 | 50.0% | 50.4% | 50.2% |
| `valid` | `about_domain` | 2600 | 50.0% | 50.0% | 50.0% |
| `valid` | `about_intent` | 2800 | 50.0% | 50.0% | 50.0% |
| `valid` | `is_negative` | 436 | 49.8% | 50.0% | 52.1% |
| `valid` | `is_positive` | 436 | 50.0% | 50.2% | 50.2% |
| `valid` | `out_of_scope` | 200 | 49.5% | 50.5% | 50.5% |
| `test_indomain` | `about_domain` | 3900 | 50.0% | 50.0% | 50.0% |
| `test_indomain` | `about_intent` | 5900 | 50.0% | 50.0% | 50.0% |
| `test_indomain` | `out_of_scope` | 2000 | 50.0% | 50.0% | 50.1% |
| `test_unseen_intents` | `about_domain` | 3000 | 50.0% | 50.0% | 50.0% |
| `test_unseen_intents` | `about_intent` | 3000 | 50.0% | 50.0% | 50.0% |
| `test_sst5` | `is_negative` | 941 | 49.9% | 50.1% | 50.2% |
| `test_sst5` | `is_positive` | 880 | 50.1% | 50.0% | 50.3% |
| `test_emotion` | `expresses_emotion` | 1000 | 50.0% | 50.0% | 50.0% |

Score gold levels per scale:

| Split | Scale | n | Levels |
|---|---|---|---|
| `train` | `sst5_3_levels` | 2563 | 0: 1010 (39%), 1: 468 (18%), 2: 1085 (42%) |
| `train` | `sst5_5_levels` | 5981 | 0: 780 (13%), 1: 1520 (25%), 2: 1156 (19%), 3: 1617 (27%), 4: 908 (15%) |
| `valid` | `sst5_3_levels` | 330 | 0: 135 (41%), 1: 62 (19%), 2: 133 (40%) |
| `valid` | `sst5_5_levels` | 771 | 0: 97 (13%), 1: 196 (25%), 2: 167 (22%), 3: 192 (25%), 4: 119 (15%) |
| `test_sst5` | `sst5_3_levels` | 663 | 0: 288 (43%), 1: 116 (17%), 2: 259 (39%) |
| `test_sst5` | `sst5_5_levels` | 1547 | 0: 185 (12%), 1: 439 (28%), 2: 273 (18%), 3: 371 (24%), 4: 279 (18%) |
| `test_yelp` | `yelp_5_stars` | 1000 | 0: 200 (20%), 1: 200 (20%), 2: 200 (20%), 3: 200 (20%), 4: 200 (20%) |

Gold option position is checked conditional on K, the number of options: for every split and every K with at least 100 choice questions, no position's share may deviate from 1/K by more than 4 binomial standard deviations. The unconditional answer-letter histogram is not uniform and cannot be: every question has options A and B (noul questions have exactly two) while J exists only when K = 10, and score levels follow the datasets' label distributions. `build_data.py` prints it for reference.

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
