# jevmark: Data

Data files are JSONL under `data/` (gitignored, rebuilt by `make data`). Option descriptions are JSON under `jevmark/data/descriptions/` and are checked in. This file is the contract between data building, training and evaluation.

## 1. Record schema

One record is one `systemone` request plus its gold answers. Every record passes `Request.from_dict({"state": ..., "questions": ...})` unchanged.

```json
{
  "id": "clinc-train-000123",
  "source": "clinc/clinc_oos/plus",
  "split": "train",
  "state": "how do i move money from checking to savings",
  "questions": {
    "about_domain": {
      "type": "noul",
      "instructions": "Is this message about banking?"
    },
    "intent": {
      "type": "choice",
      "instructions": "Which intent does this message express?",
      "criteria": {
        "freeze_account": "Asks to block or freeze a bank account.",
        "transfer": "Wants to move money between accounts.",
        "other": "None of the listed intents."
      }
    }
  },
  "gold": {"about_domain": "true", "intent": "transfer"},
  "meta": {
    "source_split": "train",
    "source_index": 123,
    "gold_intent": "transfer",
    "domain": "banking",
    "gold_in_options": true,
    "noul_kind": "about_domain"
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
| `meta` | object | provenance and build facts, never read by the model; keys per source below |

Rules:

- Option order inside every `criteria` object is a seeded random order fixed at build time, for every split including all test splits (decision 19). Training reshuffles on top of that every epoch (`encode.shuffle_request`). Gold is stored as a label, never as a letter, so reshuffling never needs to touch it.
- Question order inside `questions` is also seeded and fixed at build time.
- Every text that goes on a rendered line obeys API_SPEC section 2: single line, no edge whitespace, descriptions non-empty or null. The builder validates each record with `Request.from_dict` and fails loudly on any record that does not pass.
- Records whose encoding exceeds the training `max_tokens` are dropped by the training loader, never truncated (API_SPEC section 4); the builder does not drop anything.

## 2. Splits and how each is built

All randomness comes from one build seed in `configs/data.yaml` (created in the second half of task 1.5; the value is recorded here when it is set). Each split uses its own generator derived from the build seed and the split name, so rebuilding one split never changes another.

### CLINC150 (`clinc/clinc_oos`, config `plus`)

Every CLINC record carries exactly two questions, one choice and one noul, in a seeded random order, so the model always trains on multi-question sequences.

Choice question `intent`:
- Instructions: `Which intent does this message express?`
- K is uniform in 3..10 and counts every option including `other`.
- For an in-scope utterance, the gold intent is among the options with probability 0.8. The other options are distinct distractor intents drawn uniformly from the intents allowed in that split, then `other` is added.
- If the gold intent is not among the options, or the utterance is out of scope, the gold answer is `other`.
- Each intent's description is drawn at random from its three variants (canonical plus two paraphrases) in `clinc_intents.json`. `other` always has the fixed description `None of the listed intents.`
- Option labels are the CLINC intent names as given by the dataset (for example `freeze_account`).

Noul question, one of two kinds:
- `about_domain`: `Is this message about <domain phrase>?` The 10 CLINC domains and their phrases are written by hand in `jevmark/data/clinc.py`; the intent-to-domain map comes from the original CLINC release (section 3). For an in-scope utterance, the asked domain is the gold domain with probability q and a uniformly drawn other domain otherwise (q is set per split below). For an out-of-scope utterance the asked domain is uniform over the 10 domains and the answer is `false`.
- `out_of_scope`: `Is this request outside what a banking, travel, home, work or everyday assistant can help with?` The answer is `true` for an out-of-scope utterance and `false` for an in-scope one.

Kind assignment (decision 32), per split:
- Every out-of-scope utterance yields two records: one with the `out_of_scope` question (answer `true`) and one with the `about_domain` question (answer `false`). Each record draws its own choice options; the choice answer is `other` in both.
- Every in-scope utterance yields one record. It gets the `out_of_scope` question (answer `false`) with probability p = (number of out-of-scope utterances in the split) / (number of in-scope utterances in the split), so the expected number of `false` answers equals the number of `true` answers, and the `about_domain` question otherwise.
- Resulting p: `train` 250 / 13000 = 0.0192; `valid` 100 / 2600 = 0.0385; `test_indomain` 1000 / 3900 = 0.2564; `test_unseen_intents` has no out-of-scope utterances, so p = 0 and it has no `out_of_scope` records.
- `about_domain` gets extra `false` answers from the out-of-scope records, so q is raised to cancel them: with N_in in-scope and N_oos out-of-scope utterances, about N_in - N_oos in-scope records ask `about_domain`, and q = N_in / (2 (N_in - N_oos)) makes the expected `true` and `false` counts equal. Resulting q: `train` 13000 / 25500 = 0.5098; `valid` 2600 / 5000 = 0.52; `test_indomain` 3900 / 5800 = 0.6724; `test_unseen_intents` 0.5. With q = 0.5, `test_indomain` would have about 1450 `true` against 2450 `false` (37 percent yes) and fail the check below.
- `build_data.py` prints the yes/no ratio of each noul kind in each split and fails if either answer is outside 40 to 60 percent. A kind with no records in a split is reported as absent and not checked.

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

Instructions: AG News `Which topic is this news article about?`; emotion `Which emotion does this message express most strongly?`; Banking77 `Which banking request does this message make?`.

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

- `clinc_oos`: datasets 5 accepts only `namespace/name` ids, and the Hub redirects `clinc_oos` to `clinc/clinc_oos`, which holds parquet files for `plus` on `main`. Same repository, current name. It has no domain column, so the intent-to-domain mapping for the noul questions comes from the original CLINC release (`domains.json` in github.com/clinc/oos-eval); it is fetched and checked in, with its source commit, in the second half of task 1.5.
- `PolyAI/banking77`: the repository holds only a loading script (`banking77.py`) that downloads CSVs from github.com/PolyAI-LDN/task-specific-datasets. datasets 5 no longer runs loading scripts, and the repository has no `refs/convert/parquet`. The parquet mirror `legacy-datasets/banking77` was compared row by row with the original `train.csv` and `test.csv` the script downloads: identical text and category for all 10003 train and 3080 test rows, in the same order, and the same 77 label names in the same order as the script's `dataset_infos.json`. `mteb/banking77` was rejected: it has 9993 train and 3076 test rows, so it is not a faithful copy.

## 4. Held-out intents

(Drawn in the second half of task 1.5.)

## 5. Split sizes

(Recorded by `build_data.py` in the second half of task 1.5.)

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
