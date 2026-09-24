# jevmark: Data

Data files are JSONL under `data/` (gitignored, rebuilt by `make data`). Option descriptions are JSON under `jevmark/data/descriptions/` and are checked in. This file is the contract between data building, training and evaluation.

## 1. Record schema

One record is one `systemone` request plus its gold answers. Every record passes `Request.from_dict({"state": ..., "questions": ...})` unchanged.

A real record from `data/train.jsonl` (data v1.1, seed 0):

```json
{
  "id": "clinc-train-000040",
  "source": "clinc/clinc_oos/plus",
  "split": "train",
  "state": "transfer 200 dollars from paypal to savings",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which intent does this message express?",
      "criteria": {
        "oil_change_when": "Ask when the next oil change is needed for a vehicle",
        "transactions": "Asks to show recent transactions or purchases on an account",
        "alarm": "Requests related to creating or modifying alarms",
        "transfer": "Asking to transfer money from one account to another",
        "other": "None of the listed intents"
      }
    },
    "about_domain": {
      "type": "noul",
      "instructions": "Is this message about something other than banking, such as accounts, balances, bills or transfers?"
    }
  },
  "gold": {
    "intent": "transfer",
    "about_domain": "false"
  },
  "meta": {
    "source_split": "train",
    "source_index": 140,
    "gold_intent": "transfer",
    "domain": "banking",
    "gold_in_options": true,
    "noul_kind": "about_domain",
    "negated": true,
    "asked_domain": "banking"
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | `<source short name>-<split>-<6-digit index>`, unique across all files |
| `source` | string | dataset id plus config, as in section 3 (for example `clinc/clinc_oos/plus`) |
| `split` | string | the jevmark split the record belongs to (section 2), equal to the file name stem |
| `state` | string | the input text, unchanged from the dataset |
| `questions` | object | question id to question definition, exactly as API_SPEC section 2; key order is the question order in the encoded sequence |
| `gold` | object | question id to gold answer: an option label (string) for choice, `"true"` or `"false"` for noul, a level index (integer) for score |
| `meta` | object | provenance and build facts, never read by the model; keys per source in the table below |

`meta` keys:

| Source | Keys |
|---|---|
| CLINC | `source_split`, `source_index`, `gold_intent` (intent name, or `oos`), `domain` (null for `oos`), `gold_in_options` (null for `oos`), `noul_kind` (`about_domain` or `out_of_scope`), `negated` (the noul question is stored in its negated phrasing, gold flipped), `asked_domain` (for `about_domain` only) |
| SST-5 | `source_split`, `source_index`, `label_text` |
| AG News, emotion, Banking77 | `source_split`, `source_index` |

Rules:

- Option order inside every `criteria` object is a seeded random order fixed at build time, for every split including all test splits (decision 19). Training reshuffles on top of that every epoch (`encode.shuffle_request`). Gold is stored as a label, never as a letter, so reshuffling never needs to touch it.
- Question order inside `questions` is also seeded and fixed at build time.
- Every text that goes on a rendered line obeys API_SPEC section 2: single line, no edge whitespace, descriptions non-empty or null. The builder validates each record with `Request.from_dict` and fails loudly on any record that does not pass.
- Records whose encoding exceeds the training `max_tokens` are dropped by the training loader, never truncated (API_SPEC section 4); the builder does not drop anything.

## 2. Splits and how each is built

Current data version: **v1.1** (`version` in `configs/data.yaml`, decision 40). v1 had no negated noul questions and an out_of_scope rate tied to each split's out-of-scope count; the v1 runs (`runs/sft_06b`, `runs/base_06b` at commit 3fe318d, and the first B0 runs) were evaluated on v1.

All randomness comes from one build seed in `configs/data.yaml`: **seed 0**. Each split uses its own generator, `random.Random(f"{seed}:{split}")`, and the held-out intents use `random.Random(f"{seed}:held_out")`, so rebuilding one split never changes another. The other build parameters (K range, gold inclusion probability, set sizes, noul balance bounds) are in the same file.

### CLINC150 (`clinc/clinc_oos`, config `plus`)

Every CLINC record carries exactly two questions, one choice and one noul, in a seeded random order, so the model always trains on multi-question sequences.

Choice question `intent`:
- Instructions: `Which intent does this message express?`
- K is uniform in 3..10 and counts every option including `other`.
- For an in-scope utterance, the gold intent is among the options with probability 0.8. The other options are distinct distractor intents drawn uniformly from the intents allowed in that split, then `other` is added.
- If the gold intent is not among the options, or the utterance is out of scope, the gold answer is `other`.
- Each intent's description is drawn at random from its three variants (canonical plus two paraphrases) in `clinc_intents.json`. `other` always has the fixed description `None of the listed intents`.
- Option labels are the CLINC intent names as given by the dataset (for example `freeze_account`).

Noul question, one of two kinds:
- `about_domain`: `Is this message about <domain phrase>?` The 10 domain phrases are written by hand in `jevmark/data/clinc.py` (`DOMAIN_PHRASES`); the intent-to-domain map is the original CLINC release's `domains.json`, checked in as `jevmark/data/clinc_domains.json` (section 3). For an in-scope utterance, the asked domain is the gold domain with probability q and a uniformly drawn other domain otherwise (q is set per split below). For an out-of-scope utterance the asked domain is uniform over the 10 domains and the answer is `false`.
- `out_of_scope`: `Is this request outside what a banking, travel, home, work or everyday assistant can help with?` The answer is `true` for an out-of-scope utterance and `false` for an in-scope one.

Kind assignment (decisions 32 and 40), per split:
- Every out-of-scope utterance yields two records: one with the `out_of_scope` question (answer `true` before negation) and one with the `about_domain` question (answer `false` before negation). Each record draws its own choice options; the choice answer is `other` in both.
- Every in-scope utterance yields one record. It gets the `out_of_scope` question (answer `false` before negation) with probability p, and the `about_domain` question otherwise. p is set once from train so that the out_of_scope kind has about `clinc.out_of_scope_questions_train` = 1750 questions there: p = (1750 - N_oos,train) / N_in,train = (1750 - 250) / 13000 = 0.1154, and the same p is used in every split.
- about_domain asks the gold domain with probability q = (A + N_oos) / (2A), where A = N_in (1 - p) is the expected number of in-scope about_domain questions; this cancels the `false` answers from out-of-scope utterances, so about_domain's answers are balanced before negation. Resulting q: `train` 0.5109, `valid` 0.5217, `test_indomain` 0.6449, `test_unseen_intents` 0.5.
- Negation (data v1.1): every noul question, of either kind and in every split, is stored in its negated phrasing with probability `clinc.p_negated` = 0.5, drawn from its own seeded stream (`"{seed}:{split}:negation"`), and its gold answer is flipped; `meta.negated` records which. The phrasings are in `jevmark/data/negation.py`: `Is this message about <phrase>?` and `Is this message about something other than <phrase>?`; `Is this request outside what a banking, travel, home, work or everyday assistant can help with?` and `Is this request something a banking, travel, home, work or everyday assistant can help with?`. The symmetry evaluation negates whichever phrasing a record holds.
- Negation makes each kind's yes share 50 percent in expectation whatever its underlying balance. It does not balance each phrasing: out_of_scope has about 6 in-scope (`false`) utterances per out-of-scope (`true`) one in train, so its positive phrasing is mostly `false` and its negated phrasing mostly `true`, and the phrasing alone predicts the answer (section 5, phrasing-only accuracy). about_domain has no such shortcut.
- `build_data.py` prints, per kind and split, the yes/no ratio (and fails if either answer is outside 40 to 60 percent), the negated share, the yes/no counts in each phrasing, and the accuracy of answering each phrasing with its train majority. A kind with no records in a split is reported as absent and not checked.

Held-out intents: 20 intents, 2 per domain, chosen with the build seed. None of their utterances appear in `train` or `valid`, and none of them appear as a distractor option in `train` or `valid`. The list is recorded in section 4 when it is drawn.

| Split | Built from | Intents allowed as options |
|---|---|---|
| `train` | CLINC `train` minus held-out intents (in scope and out of scope; each out-of-scope utterance yields two records) | the 130 seen intents |
| `valid` | CLINC `validation` minus held-out intents (out-of-scope utterances yield two records) | the 130 seen intents |
| `test_indomain` | CLINC `test` minus held-out intents (out-of-scope utterances yield two records) | the 130 seen intents |
| `test_unseen_intents` | every utterance of the 20 held-out intents from all three CLINC splits (20 x 150 = 3000); the gold answer is `other` whenever the gold intent is not among the options | the 20 held-out intents only, so every non-`other` option is a label the model never trained on |

### SST-5 (`SetFit/sst5`)

One score question per record, `sentiment`: `How positive is the sentiment of this text?`, with five level descriptions written by hand in `jevmark/data/sst5.py`, from very negative (level 0) to very positive (level 4). Gold is the dataset label as a level index.

| Split | Built from |
|---|---|
| records in `train` | SST-5 `train` |
| records in `valid` | SST-5 `validation` |
| `test_sst5` | SST-5 `test` |

`train` and `valid` therefore mix CLINC and SST-5 records; the `source` field tells them apart.

### Unseen schemas (evaluation only, never in training)

One choice question per record. 1000 records per set, sampled from the dataset's test split with the build seed. Canonical descriptions come from `generate_descriptions.py` (section 6); no paraphrases.

| Split | Built from | Options |
|---|---|---|
| `test_agnews` | AG News `test` | all 4 topic labels |
| `test_emotion` | emotion `test` | all 6 emotion labels |
| `test_banking77` | Banking77 `test` | the gold label, 8 distinct distractor labels and `other`: 10 options, gold always present |

Labels are used exactly as the datasets give them, never normalised. This includes two irregular Banking77 names, `Refund_not_showing_up` (capital R) and `reverted_card_payment?` (trailing question mark); both are valid option labels under API_SPEC section 2.

Instructions: AG News `Which topic is this news article about?`; emotion `Which emotion does this message express most strongly?`; Banking77 `Which banking request does this message make?`. The Banking77 `other` option has the fixed description `None of the listed options` and is never the gold answer.

## 3. Dataset ids and revisions

Verified with `scripts/check_datasets.py` (datasets 5.0.1, huggingface_hub 1.33.0). Every loader passes the revision below, so a moved or edited dataset cannot change a build silently.

| Role | Id originally named in TASKS.md | Id used | Config | Revision (commit sha) | Splits and rows | Labels |
|---|---|---|---|---|---|---|
| CLINC150 | `clinc_oos` | `clinc/clinc_oos` | `plus` | `155b9c710419136e17307b80d0a13e68cd46b4ec` | train 15250, validation 3100, test 5500 | `intent`: 151 classes (150 intents plus `oos`) |
| SST-5 | `SetFit/sst5` | `SetFit/sst5` | none | `e51bdcd8cd3a30da231967c1a249ba59361279a3` | train 8544, validation 1101, test 2210 | `label` integer 0 to 4, `label_text` |
| AG News | `fancyzhx/ag_news` | `fancyzhx/ag_news` | none | `eb185aade064a813bc0b7f42de02595523103ca4` | train 120000, test 7600 | `label`: World, Sports, Business, Sci/Tech |
| emotion | `dair-ai/emotion` | `dair-ai/emotion` | none | `cab853a1dbdf4c42c2b3ef2173804746df8825fe` | train 16000, validation 2000, test 2000 | `label`: sadness, joy, love, anger, fear, surprise |
| Banking77 | `PolyAI/banking77` | `legacy-datasets/banking77` | none | `f54121560de48f2852f90be299010d1d6dc612ec` | train 10003, test 3080 | `label`: 77 classes |

Notes on the two ids that differ from the originally named ones (both substitutions approved by the human; TASKS.md now uses the new ids):

- `clinc_oos`: datasets 5 accepts only `namespace/name` ids, and the Hub redirects `clinc_oos` to `clinc/clinc_oos`, which holds parquet files for `plus` on `main`. Same repository, current name. It has no domain column, so the intent-to-domain map comes from the original CLINC release: `data/domains.json` in github.com/clinc/oos-eval at commit `976178879e5afa9952f60a1f8d3c834f47a25cee` (the commit that added it; the file is identical on `master` at `828f8093`). It is checked in unmodified as `jevmark/data/clinc_domains.json`, sha256 `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a`; the builder refuses to run if the hash differs, and asserts that it maps exactly the 150 dataset intents, 15 to each of the 10 domains.
- `PolyAI/banking77`: the repository holds only a loading script (`banking77.py`) that downloads CSVs from github.com/PolyAI-LDN/task-specific-datasets. datasets 5 no longer runs loading scripts, and the repository has no `refs/convert/parquet`. The parquet mirror `legacy-datasets/banking77` was compared row by row with the original `train.csv` and `test.csv` the script downloads: identical text and category for all 10003 train and 3080 test rows, in the same order, and the same 77 label names in the same order as the script's `dataset_infos.json`. `mteb/banking77` was rejected: it has 9993 train and 3076 test rows, so it is not a faithful copy.

## 4. Held-out intents

Unchanged in data v1.1 (the held-out stream does not depend on the other build parameters). Drawn with seed 0, two per domain (`random.Random("0:held_out")`, domains and intents in sorted order). None of them appears in `train` or `valid`, as an utterance or as an option; `test_unseen_intents` offers only these intents plus `other`.

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

From `make data` with data v1.1, seed 0 (`scripts/build_data.py`). Max tokens is the longest encoded record with the pinned reference tokenizer; every record must fit within 1024.

| Split | Records | By source | Max tokens | about_domain yes / no | out_of_scope yes / no |
|---|---|---|---|---|---|
| `train` | 22044 | CLINC 13500, SST-5 8544 | 257 | 5866 / 5873 (50.0%) | 887 / 874 (50.4%) |
| `valid` | 3901 | CLINC 2800, SST-5 1101 | 246 | 1247 / 1166 (51.7%) | 190 / 197 (49.1%) |
| `test_indomain` | 5900 | CLINC 5900 | 244 | 2213 / 2264 (49.4%) | 696 / 727 (48.9%) |
| `test_unseen_intents` | 3000 | CLINC 3000 | 242 | 1374 / 1294 (51.5%) | 176 / 156 (53.0%) |
| `test_sst5` | 2210 | SST-5 2210 | 159 | | |
| `test_agnews` | 1000 | AG News 1000 | 256 | | |
| `test_emotion` | 1000 | emotion 1000 | 174 | | |
| `test_banking77` | 1000 | Banking77 1000 | 273 | | |

Noul phrasing (data v1.1). "Positive" and "negated" give gold yes / no in each phrasing; phrasing-only accuracy answers each phrasing with its majority answer in train, without reading the message.

| Split | Kind | n | Negated share | Positive yes / no | Negated yes / no | Phrasing-only accuracy |
|---|---|---|---|---|---|---|
| `train` | about_domain | 11739 | 50.0% | 2901 / 2971 | 2965 / 2902 | 50.6% |
| `train` | out_of_scope | 1761 | 49.5% | 133 / 757 | 754 / 117 | 85.8% |
| `valid` | about_domain | 2413 | 51.4% | 604 / 569 | 643 / 597 | 50.2% |
| `valid` | out_of_scope | 387 | 50.6% | 47 / 144 | 143 / 53 | 74.2% |
| `test_indomain` | about_domain | 4477 | 50.1% | 1106 / 1130 | 1107 / 1134 | 50.0% |
| `test_indomain` | out_of_scope | 1423 | 49.8% | 494 / 221 | 202 / 506 | 29.7% |
| `test_unseen_intents` | about_domain | 2668 | 49.9% | 679 / 658 | 695 / 636 | 50.7% |
| `test_unseen_intents` | out_of_scope | 332 | 53.0% | 0 / 156 | 176 / 0 | 100.0% |

CLINC record counts are in-scope utterances plus twice the out-of-scope utterances (for example `train`: 13000 + 2 x 250).

Gold option position is checked conditional on K, the number of options: for every split and every K with at least 100 choice questions, no position's share may deviate from 1/K by more than 4 binomial standard deviations (largest observed in v1.1: 2.85, `test_banking77` K=10). The unconditional answer-letter histogram is not uniform and cannot be: every question has options A and B (noul questions have exactly two) while J exists only when K = 10, and SST-5 levels follow the dataset's label distribution. `build_data.py` prints it for reference.

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
