# jevmark

jevmark is a small decision model that never generates text: a text state and a set of typed questions go in, and a probability distribution per question comes out of one forward pass, read from the letter logits at each question's answer slot. The options are part of the request, so the model answers questions whose labels it never saw in training. It is an independent re-implementation of the concept behind TypeSafe AI's Jev, built on Qwen3-0.6B-Base and Qwen3-1.7B-Base with LoRA, and used to test when Reinforcement Learning for Calibrated Decisions (RLCD) adds anything over supervised training and a temperature.

## Install

Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
make setup            # uv sync --extra dev
make test             # pytest on CPU, a tiny random Qwen3 config, under 3 minutes
```

Optional extras: `uv sync --extra serve` for the FastAPI wrapper (`scripts/serve.py`, `POST /v1/systemone`), `uv sync --extra baselines` for the OpenAI baseline.

## Usage

```python
import os
os.environ["JEVMARK_CHECKPOINT"] = "hf://yuanphd/jevmark/sft_06b"  # the Hub adapter, see below

from jevmark import systemone

response = systemone(
    "Hi, I was charged twice for my March invoice and the export button still crashes "
    "every time I click it. I need this sorted today, please.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Charges, invoices, refunds and payment problems",
                "technical": "Bugs, crashes, outages and integration problems",
                "account": "Login, profile and account settings",
                "other": None,
            },
        },
        "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        "needs_human": {"type": "noul", "instructions": "Does this message need a human agent rather than an automated reply?"},
        "severity": {
            "type": "score",
            "instructions": "How severe is the reported issue?",
            "criteria": [
                "Cosmetic; no impact on functionality",
                "Broken or degraded feature, but a workaround exists",
                "Blocking issue; no workaround exists",
            ],
        },
        "next_tool": {
            "type": "choice",
            "instructions": "Which tool should the support agent call first?",
            "criteria": {
                "lookup_invoice": "Fetch the customer's invoices and payment history",
                "open_bug_ticket": "File a bug report with the engineering team",
                "search_help_center": "Search help articles for a known answer",
            },
        },
    },
)
```

Real output of this call, from `sft_06b` on a laptop CPU (2026-10-05, `scripts/demo.py`; the field definitions are in docs/API_SPEC.md section 3):

```json
{
  "model": "jevmark-sft_06b",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {
        "billing": 0.3526,
        "technical": 0.6462,
        "account": 0.0003,
        "other": 0.0008
      },
      "confidence": 0.5252
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.0216
    },
    "needs_human": {
      "type": "noul",
      "noul": 0.9557
    },
    "severity": {
      "type": "score",
      "score": 1.8884,
      "legend": {
        "0": "Cosmetic; no impact on functionality",
        "1": "Broken or degraded feature, but a workaround exists",
        "2": "Blocking issue; no workaround exists"
      },
      "probabilities": {
        "0": 0.0186,
        "1": 0.0744,
        "2": 0.907
      },
      "confidence": 0.676
    },
    "next_tool": {
      "type": "choice",
      "choice": "open_bug_ticket",
      "probabilities": {
        "lookup_invoice": 0.0081,
        "open_bug_ticket": 0.9813,
        "search_help_center": 0.0106
      },
      "confidence": 0.9036
    }
  },
  "usage": {
    "input_tokens": 265
  }
}
```

The default model is the bare backbone of `configs/base.yaml`. Set `JEVMARK_CHECKPOINT=runs/<run>` to load a trained run's adapter, `calibration.json` and `config.yaml`. Adapters are not in git: they come from the Kaggle training notebooks (docs/KAGGLE.md).

Three trained adapters are published on the Hugging Face Hub at [yuanphd/jevmark](https://huggingface.co/yuanphd/jevmark): `sft_06b` and `sft_17b` (the general decision model at both sizes) and `rlcd_banking77_06b` (sft_06b adapted to Banking77 from logged feedback, docs/RESULTS_v3.md). Load one with a Hub reference, which downloads that folder once into the Hugging Face cache and uses its own `config.yaml`:

```bash
JEVMARK_CHECKPOINT=hf://yuanphd/jevmark/sft_06b python my_script.py
```

`JevMark.load(config, checkpoint=...)` takes the same `hf://<owner>/<repo>/<folder>` form. The model card is docs/MODEL_CARD.md. `systemone_batch` takes a list of requests. The full contract, validation rules and encoding are in docs/API_SPEC.md.

## Demo

```bash
uv run python scripts/demo.py                       # hf://yuanphd/jevmark/sft_06b, threshold 0.9
uv run python scripts/demo.py --checkpoint hf://yuanphd/jevmark/rlcd_banking77_06b --threshold 0.8
```

`scripts/demo.py` sends the request above in one call, prints the response, then applies one confidence threshold: an answer at or above it is acted on automatically, one below it is escalated to an LLM or a human. Confidence is the response field for choice and score and max(p, 1 - p) for noul. The questions and the threshold are defined in that one file. Real output, `sft_06b` on a laptop CPU, 2026-10-05:

```
confidence threshold 0.9:
  escalate   department        technical (p 0.6462)               confidence 0.5252
  automatic  refund_requested  no (P(yes) 0.0216)                 confidence 0.9784
  automatic  needs_human       yes (P(yes) 0.9557)                confidence 0.9557
  escalate   severity          level 1.89 of 0 to 2               confidence 0.6760
  automatic  next_tool         open_bug_ticket (p 0.9813)         confidence 0.9036

3 acted on automatically, 2 escalated
model load 1.5 s; the five-question call 404 ms on cpu
```

The five answers come from one forward pass of 265 tokens: 404 ms for that one call on the laptop CPU in fp32, after a 1.5 s model load with the weights already cached (the first run also downloads the 0.6B backbone and the 18 MB adapter). On a T4 the batch-1 median is 47.7 ms per request (docs/RESULTS_v1.md section 4). The message names both a double charge and a crash, and the model splits department between billing and technical, so that answer is escalated rather than guessed.

## Question types

- **noul:** a yes or no question; returns the probability of yes.
- **choice:** one of 2 to 26 options defined in the request, each with an optional description; returns a probability per option, the argmax and a confidence, 1 - H(p) / ln K.
- **score:** a position on 2 to 10 ordered levels described in the request; returns the level distribution, its expectation and a confidence.

Thresholds and routing are the caller's job: the model returns distributions, and policy lives in code.

## Backbones

Qwen/Qwen3-1.7B-Base is the main backbone and Qwen/Qwen3-0.6B-Base the development backbone; both are config switches (`configs/sft_06b.yaml`, `configs/sft_17b.yaml`). Training is LoRA only (r 16, alpha 32, q k v o), fp16 autocast with fp32 LoRA weights, on Kaggle T4s. v2's RLCD arms and all of v3 ran on 0.6B.

## Data and leak gates

```bash
make data       # build the nine v1.3 splits into data/, then the CPU gates
make data-v3    # the v3 Banking77 files next to them, then the leak probes on them
```

Training data is CLINC150 (choice and noul) and SST-5 (score), with 20 intents held out and four schemas never trained on (AG News, emotion, Banking77, Yelp). `make data` fails unless every build check passes, state-free linear and boosted-tree leak probes find no question answerable from its structure alone (no lift above 10 points, or above 3 points and the 99th percentile of shuffled-target runs), and no test text appears in train or valid. Dataset ids, revisions, record schema and the gates are in docs/DATA.md; data hashes are pinned in `data/baseline_subset.json` and `configs/v3_data.yaml`.

## Reproducing the tables

Every run's `metrics.json`, config and training log are committed under `runs/`. The per-question files behind them (`results.jsonl.gz`, baseline `replies.jsonl`, v3 `log.jsonl`) and the adapters are gitignored; they come from the Kaggle sessions (docs/KAGGLE.md), and with them in place every table recomputes on CPU:

| report | table | command | output |
|---|---|---|---|
| v1 | any run's metrics | `uv run python scripts/recompute_metrics.py runs/<run>` | `runs/<run>/metrics.json` |
| v1 | baselines on the subset | `uv run python scripts/compare_baselines.py` | printed, from `runs/{base,sft}_*`, `runs/b1_*`, `runs/b2_gpt-4.1-mini` |
| v1 | 1.7B against 0.6B | `uv run python scripts/paired_deltas.py --pair runs/sft_17b runs/sft_06b --pair runs/base_17b runs/base_06b --out runs/paired_17b_vs_06b/metrics.json` | `runs/paired_17b_vs_06b/` |
| v2 | temperature scaling | `uv run python scripts/calibrate.py runs/<run>`, then `uv run python scripts/compare_calibration.py` | `runs/<run>_temp/` |
| v2 | the REINFORCE simulation | `uv run python scripts/simulate_advantages.py runs/sft_06b` | `runs/advantage_simulation_06b/` |
| v2 | RLCD arms, three seeds | `uv run python scripts/compare_rlcd.py --size 06b --seeds 0 1 2 --out runs/rlcd_stage2a_06b` | `runs/rlcd_stage2a_06b/` |
| v3 | temperature learners | `uv run python scripts/calibrate.py runs/v3_06b_zeroshot --fit-log runs/v3_log_s0/log.jsonl --n <N>` | `runs/v3_06b_temp_n<N>/` |
| v3 | temperature on direct_brier | `uv run python scripts/evaluate.py --ckpt runs/<run> --log-prefix runs/v3_log_s0/log.jsonl --n <N> --no-probes --device cpu` (with the adapter and base weights; 2 to 18 minutes per run on a laptop CPU), then `uv run python scripts/calibrate.py runs/<run> --fit-log runs/v3_log_s0/log.jsonl --n <N> --fit-probs runs/<run>_prefix` | `runs/<run>_prefix/`, `runs/<run>_temp/` |
| v3 | the noise inversion | `uv run python scripts/invert_noisy.py runs/v3_06b_<arm>_n5000_s0_noisy --clean runs/v3_06b_<arm>_n5000_s0` | `runs/v3_06b_<arm>_n5000_s0_noisy_inverted/` |
| v3 | every v3 table | `uv run python scripts/compare_v3.py` | `runs/v3_stage_06b/` |
| v3 | the label-noise audit summary | `uv run python scripts/summarise_label_audit.py` | `docs/audit/summary.json` |
| all | every report figure | `uv run python scripts/make_figures.py` (committed metrics files only) | `docs/figures/` |

Training and evaluation themselves (`make train-sft`, `scripts/train_rlcd.py`, `make eval`) run on Kaggle through the thin notebooks in `notebooks/`.

## Results

With LoRA SFT, in-domain accuracy is .956 at 0.6B and .952 at 1.7B with ECE .013 and .015, against .492 and .549 for the frozen bases, and .868 and .885 on intents never trained on; on a 500-record subset it beats same-size JSON generation by 25 to 33 points and gpt-4.1-mini in-domain, while gpt-4.1-mini leads on Banking77, emotion and Yelp (docs/RESULTS_v1.md sections 2 to 4). On full deterministic labels, no RLCD arm beats SFT plus one in-domain temperature on unseen-schema ECE (four-schema mean .121 for SFT plus T; the arms, each with its own temperature, are at best level with it), because a bandit outcome carries less information than the label (docs/RESULTS_v2.md section 3). Where only the correctness of the model's own action is observed, on Banking77 from a deployment log, RLCD ties positive-only SFT at 500 interactions and beats it by 2.4 to 2.7 points from 2000, with ECE lower by .05 to .06 at every N, stays 1.5 points [1.0, 2.0] behind full-label SFT at 5000 (seed means of both), and under noisy feedback calibrates to the noise rather than to correctness (docs/RESULTS_v3.md section 8). A temperature fitted on top of it changes ECE by at most .003 (section 3), and a blind audit by two language-model judges finds a label problem in 12 to 14 percent of Banking77 test labels (section 9, docs/audit/SUMMARY.md).

## Documentation

- docs/STORY.md: the whole project in two pages.
- docs/RESULTS_v1.md, docs/RESULTS_v2.md, docs/RESULTS_v3.md: the reports, every number cited to a `metrics.json`.
- docs/PLAN.md, docs/API_SPEC.md, docs/TASKS.md, docs/DECISIONS.md: plan, contract, tasks with their proofs, and every design decision with its reason.
- docs/DATA.md, docs/KAGGLE.md, docs/V3_DESIGN.md: data, how to run on Kaggle, and the v3 specification.
- docs/MODEL_CARD.md: the card of the adapters on the Hugging Face Hub.
- docs/audit/SUMMARY.md: the Banking77 label-noise audit.
- docs/CLOSING_PLAN.md: the closing checklist (in Chinese).
- CLAUDE.md: the rules for working in this repository.
