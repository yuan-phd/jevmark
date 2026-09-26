# jevmark v2 results: calibration

Every number cites the file it comes from, as `path:key`. `runs/<run>_temp/` holds temperature scaling of `runs/<run>/`, written by `scripts/calibrate.py` from the source run's `results.jsonl.gz` on CPU; `scripts/compare_calibration.py` prints the tables below from those files. The v1 report (`docs/RESULTS_v1.md`, frozen, decision 49) is the baseline this report starts from.

## 1. Temperature scaling

**Method.** One scalar temperature T per run, fitted on the gold-dependent questions of `valid` (in-domain CLINC and SST-5 only) by minimising the mean NLL of softmax(log p / T). Because log p differs from the letter logits by a constant per question, this equals rerunning the model with softmax(logits / T) (`jevmark/calibration.py`; tested against the model on the tiny backbone in `tests/test_evaluate.py::test_temperature_on_stored_probabilities_equals_the_model_with_that_temperature`). A positive temperature changes no prediction, so accuracy is identical to the v1 runs on every split; only ECE, NLL, Brier and the confidence-based coverage move. Two diagnostics are fitted but never used as results: a temperature per question type on `valid`, and an oracle temperature per split fitted on that split's own questions (`runs/<run>_temp/metrics_oracle.json`), the best any single temperature could do there.

**Fitted temperatures** (`runs/<run>_temp/metrics.json:calibration`):

| run | global T | valid NLL before, after | per type on valid (diagnostic) |
|---|---|---|---|
| sft_06b | 1.266 | .2165, .2094 | noul 1.404, choice 1.176, score 1.266 |
| sft_17b | 1.321 | .2007, .1920 | noul 1.454, choice 1.258, score 1.275 |
| base_06b | 1.114 | 1.0779, 1.0756 | noul 20.0 (the upper bound), choice 0.585, score 2.125 |
| base_17b | 1.075 | .8303, .8290 | noul 4.241, choice 0.934, score 1.155 |

Both SFT models are slightly overconfident on `valid` (T above 1), and on their own training data slightly underconfident (oracle T 0.89 and 0.95 on `train`, `runs/sft_*_temp/metrics_oracle.json:temperatures.train`). The frozen bases barely move with a global T; their per-type temperatures diverge because their noul answers are near chance and a flat distribution minimises NLL there.

**ECE and NLL, SFT against SFT plus the global T and the per-split oracle** (gold-dependent questions; `runs/sft_<size>/metrics.json`, `runs/sft_<size>_temp/metrics.json` and `runs/sft_<size>_temp/metrics_oracle.json`, key `splits.<split>.overall.ece` and `.nll`; oracle T from `metrics_oracle.json:temperatures.<split>`):

| split | sft_06b ECE / NLL | + global T | + oracle T (T) | sft_17b ECE / NLL | + global T | + oracle T (T) |
|---|---|---|---|---|---|---|
| valid | .020 / .216 | .005 / .209 | .005 / .209 (1.27) | .019 / .201 | .006 / .192 | .006 / .192 (1.32) |
| test_indomain | .013 / .123 | .005 / .119 | .005 / .119 (1.20) | .015 / .133 | .003 / .126 | .003 / .126 (1.32) |
| test_sst5 | .030 / .541 | .012 / .533 | .012 / .533 (1.22) | .025 / .508 | .020 / .500 | .015 / .499 (1.24) |
| test_unseen_intents | .068 / .371 | .054 / .320 | .021 / .295 (1.71) | .070 / .365 | .055 / .304 | .023 / .276 (1.83) |
| test_agnews | .164 / 1.080 | .146 / .886 | .029 / .628 (2.70) | .113 / .773 | .097 / .616 | .032 / .469 (2.48) |
| test_emotion | .207 / 1.249 | .166 / 1.087 | .032 / .915 (2.56) | .217 / 1.352 | .175 / 1.122 | .056 / .899 (2.88) |
| test_banking77 | .067 / .545 | .037 / .496 | .032 / .489 (1.43) | .087 / .633 | .058 / .534 | .034 / .502 (1.71) |
| test_yelp | .169 / 1.110 | .137 / 1.080 | .120 / 1.078 (1.36) | .218 / 1.144 | .169 / 1.042 | .101 / 1.006 (1.79) |

On `train`, which the models were fitted to, the global T makes ECE worse (.009 to .026 and .004 to .023), as expected when a model is underconfident on its own training data; `train` is not a target. Per question type the picture is the same; the full table is printed by `scripts/compare_calibration.py`.

**Coverage.** A temperature above 1 lowers every confidence, so a threshold keeps fewer answers, each more accurate (`<run>:splits.<split>.<type>.coverage`, response confidence field). On test_indomain noul at threshold 0.95, sft_06b keeps 81.5 percent at .987 and sft_06b plus T keeps 76.1 percent at .993; on test_unseen_intents choice at 0.9, 78.3 percent at .954 against 69.9 percent at .974; on test_emotion choice at 0.9, 26.8 percent at .791 against 12.4 percent at .863 (1.7B: 29.5 percent at .786 against 14.6 percent at .884). Scaling trades coverage for accuracy at a fixed threshold; it moves the threshold at which a given accuracy is reached, and adds no information about which answers are right.

**What a global temperature fixes, and what it does not.** Fitted on in-domain `valid` alone, one number brings in-domain ECE to .003 to .005 and SST-5 to .012 to .020, equal or close to the per-split oracle there, and it removes 11 to 45 percent of the ECE on unseen intents and unseen schemas (about a fifth on most; AG News .164 to .146, emotion .207 to .166, Yelp .218 to .169 at 1.7B), because the SFT models are overconfident everywhere and a temperature above 1 points the right way. It does not fix the unseen schemas: the temperatures they need range from 1.36 to 2.88, up to 2.2 times the 1.27 and 1.32 fitted on `valid`, and differ per schema, so no single number serves them all; even the oracle leaves Yelp at .10 to .12, a miscalibration in the shape of the distribution over an ordinal scale that no temperature can remove; and on emotion and Yelp at both sizes, and on AG News at 0.6B, the SFT model with the global T (.137 to .175) is still less calibrated than the frozen base it started from (.037 to .090, `runs/base_*/metrics.json`), with AG News at 1.7B level (.097 against .094). The gap RLCD has to close is therefore specific: calibration on schemas the model was not trained on, without knowing the schema's temperature, at in-domain accuracy. Concretely, an RLCD arm has to bring unseen-schema ECE below what SFT plus the global T reaches (.146, .166, .037 and .137 at 0.6B; .097, .175, .058 and .169 at 1.7B), towards the oracle, while keeping in-domain ECE near .005 and accuracy at the SFT level.
