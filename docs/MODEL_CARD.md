---
license: apache-2.0
base_model:
- Qwen/Qwen3-0.6B-Base
- Qwen/Qwen3-1.7B-Base
library_name: peft
datasets:
- clinc/clinc_oos
- SetFit/sst5
- legacy-datasets/banking77
language:
- en
tags:
- decision-model
- calibration
- lora
- qwen3
---

# jevmark adapters

jevmark is a small decision model that never generates text: a text state and a set of typed questions go in, and one forward pass returns a full probability distribution per question. The answer options are part of each request (a yes or no question, one of the options given, or a level on a scale defined in the request), so the model reads them instead of memorising a fixed label set, and the distribution is read from the letter logits at each question's answer slot. It is an independent re-implementation of the concept behind TypeSafe AI's Jev, built as LoRA adapters on Qwen3 base models to measure accuracy, calibration and when Reinforcement Learning for Calibrated Decisions (RLCD) helps.

Code, data builders, every report and the metrics files behind every number: https://github.com/yuan-phd/jevmark

## The adapters

| folder | base model | what it is for |
|---|---|---|
| `sft_06b` | Qwen3-0.6B-Base | The general decision model: LoRA SFT on CLINC150 and SST-5 questions. |
| `sft_17b` | Qwen3-1.7B-Base | The same general decision model at the larger size. |
| `rlcd_banking77_06b` | Qwen3-0.6B-Base | `sft_06b` adapted to Banking77 from a deployment log: 5000 logged interactions where only the correctness of the model's own chosen answer was revealed, no labels, trained with a pathwise Brier loss (RLCD). |

Provenance: each source run is a directory of the GitHub repository with its config and metrics, at the commit that trained it.

| folder | source run | commit |
|---|---|---|
| `sft_06b` | `runs/sft_06b` | a1dc2bf |
| `sft_17b` | `runs/sft_17b` | 3a7169c |
| `rlcd_banking77_06b` | `runs/v3_06b_direct_brier_n5000_s0` | b5913ef |

sha256 of each `adapter_model.safetensors`:

```
sft_06b             5d2681f3ba7a72cd4c0f1413e89c5e21c7bb96cee8645fa2a498516d67568809
sft_17b             566ff13ba9e8400e653e878361ef625e875b43957f80322c046e7e7248003374
rlcd_banking77_06b  3e00b7d467fd2d3955402b2b2fcdb41317a9d6221ae65ae152520de49d8d783a
```

Each folder holds `adapter_config.json` and `adapter_model.safetensors` (LoRA r 16, alpha 32 on q, k, v, o), plus `config.yaml` (the backbone and its pinned revision, the encoding length and the training settings), `calibration.json` (temperature 1.0: the adapters are published as evaluated in the reports, without a fitted temperature) and `model_id.txt`.

## How to call them

Install jevmark from GitHub (Python 3.11):

```bash
pip install git+https://github.com/yuan-phd/jevmark
```

Point `JEVMARK_CHECKPOINT` at a folder of this repository. The folder is downloaded once into the Hugging Face cache, and its `config.yaml` selects the matching backbone.

```python
import os
os.environ["JEVMARK_CHECKPOINT"] = "hf://yuanphd/jevmark/sft_06b"

from jevmark import systemone

response = systemone(
    "Hi, I was charged twice for my March invoice and the export button still crashes.",
    {
        "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs, outages, integration problems", "other": None},
        },
    },
)
# response["answers"]["department"]: the chosen label, a probability per option and a confidence
```

To hold a model object yourself:

```python
import yaml
from jevmark.model import JevMark, resolve_checkpoint

folder = resolve_checkpoint("hf://yuanphd/jevmark/rlcd_banking77_06b")
jev = JevMark.load(yaml.safe_load((folder / "config.yaml").read_text()), checkpoint=folder)
```

The request and response contract, including validation and the exact encoding, is in [docs/API_SPEC.md](https://github.com/yuan-phd/jevmark/blob/main/docs/API_SPEC.md). Thresholds and routing are the caller's job: the model returns distributions and a confidence, 1 - H(p) / ln K for choice and score questions.

## Results

ECE uses 15 equal-width bins on the top-1 probability, on the questions whose answer depends on a gold label. Each number cites a metrics file in the GitHub repository, as `path:key`.

### sft_06b and sft_17b

Full test splits, accuracy / ECE (`runs/<run>/metrics.json:splits.<split>.overall`; [docs/RESULTS_v1.md](https://github.com/yuan-phd/jevmark/blob/main/docs/RESULTS_v1.md) section 2):

| split | what it tests | sft_06b | sft_17b |
|---|---|---|---|
| test_indomain | CLINC150 intents and yes or no questions seen in training | .956 / .013 | .952 / .015 |
| test_unseen_intents | 20 CLINC150 intents never trained on (choice accuracy .868 and .885) | .892 / .068 | .897 / .070 |
| test_sst5 | SST-5 sentiment scale and sentiment questions | .773 / .030 | .790 / .025 |
| test_agnews | unseen schema: news topic | .788 / .164 | .847 / .113 |
| test_emotion | unseen schema: emotion | .628 / .207 | .645 / .217 |
| test_banking77 | unseen schema: 10-option Banking77 subsets | .851 / .067 | .852 / .087 |
| test_yelp | unseen schema: 5-level star scale | .505 / .169 | .527 / .218 |

Batch-1 median latency on one T4: 47.7 ms and 74.8 ms (`runs/<run>/metrics.json:latency.batch_1.median_ms`). On a 500-record subset, sft_06b and sft_17b score .944 and .945 in-domain against .908 for gpt-4.1-mini, while gpt-4.1-mini leads on Banking77 (.918 against .846 and .850) (RESULTS_v1 section 4). One temperature fitted on in-domain validation (T 1.266 and 1.321, `runs/sft_<size>_temp/metrics.json:temperature`) brings in-domain ECE to .005 and .003; to use it, set `temperature` in the folder's `calibration.json` ([docs/RESULTS_v2.md](https://github.com/yuan-phd/jevmark/blob/main/docs/RESULTS_v2.md) section 1).

### rlcd_banking77_06b

On the full Banking77 test split (3080 records), against the sft_06b adapter it started from and the other learners trained on the same 5000 logged interactions (`runs/v3_stage_06b/metrics.json:n_curve.5000`; [docs/RESULTS_v3.md](https://github.com/yuan-phd/jevmark/blob/main/docs/RESULTS_v3.md) section 2):

| learner | accuracy [95% interval] | ECE | Brier | NLL |
|---|---|---|---|---|
| sft_06b, zero-shot | .851 [.839, .863] | .063 | .229 | .582 |
| SFT on the confirmed answers only | .905 [.896, .915] | .078 | .169 | .663 |
| **rlcd_banking77_06b** | **.929 [.920, .938]** | **.026** | **.111** | **.274** |
| SFT on gold labels (needs a label per interaction) | .947 [.938, .955] | .030 | .082 | .212 |

The cost on the original domain (`runs/v3_06b_direct_brier_n5000_s0/metrics.json:splits.<split>.overall`): test_indomain .929 (sft_06b .956) and test_unseen_intents .917 (sft_06b .892), both at ECE no higher than sft_06b's. On the 500-record subset where gpt-4.1-mini scores .918, this adapter scores .934 (`runs/v3_stage_06b/metrics.json:b2_reference.runs.direct_brier_n5000_s0`; 500 records, no paired interval).

## Training data

The adapters are LoRA weights only. No dataset text is redistributed here; the training and test records are rebuilt from the pinned dataset revisions by the code in the GitHub repository. Licences as stated on each dataset's Hugging Face card (read on 2026-10-04):

| dataset | used by | licence on its Hub card |
|---|---|---|
| CLINC150, `clinc/clinc_oos` (config `plus`) | all three (rlcd_banking77_06b starts from sft_06b) | CC BY 3.0 in the card's metadata; its Licensing Information section reads "More Information Needed" |
| SST-5, `SetFit/sst5` | all three | none: the card states no licence |
| Banking77, `legacy-datasets/banking77` (the parquet mirror of `PolyAI/banking77`) | rlcd_banking77_06b (train split, outcomes of the logged actions only) | CC BY 4.0 |

AG News, emotion and Yelp reviews were used for evaluation only. Option descriptions were written once with gpt-4.1-mini and corrected by hand ([docs/DATA.md](https://github.com/yuan-phd/jevmark/blob/main/docs/DATA.md) sections 6 and 7).

## Limitations

- Overconfident on schemas it was not trained on: ECE .113 to .218 on AG News, emotion and Yelp, and one temperature fitted in-domain removes only part of it (RESULTS_v1 section 5, RESULTS_v2 section 1).
- Answers depend on question order: a question that depends on another should come after it, and reordering cost about 6 points of yes or no accuracy in-domain (API_SPEC section 4).
- Banking77's labels are noisy: two language-model judges found a label problem in 12 to 14 percent of a uniform test sample, so its accuracy ceiling is near .96 (RESULTS_v3 section 9).
- The RLCD adapter exists at 0.6B only and was trained from simulated feedback, revealed from gold labels; real deployment feedback is delayed, biased and of unknown noise rate.
- English only: every training and test text is English.
