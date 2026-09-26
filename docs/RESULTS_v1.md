# jevmark v1 results: supervised decision model

Every number in this report is read from a committed file under `runs/`, cited next to it. `metrics.json` paths are given as `run:key`, where `run` is `runs/<run>/metrics.json` unless another file is named and `key` is the path inside it (for example `sft_06b:splits.test_indomain.choice.accuracy`). Reliability diagrams are in `runs/<run>/plots/<split>.png` for the four jevmark runs. Decision numbers refer to `docs/DECISIONS.md`.

## 1. Setup

**Backbones.** Qwen3-0.6B-Base and Qwen3-1.7B-Base, each evaluated frozen (`base_06b`, `base_17b`) and after LoRA SFT (`sft_06b`, `sft_17b`; r 16, alpha 32, q k v o, 2 epochs, 1378 steps at effective batch 32, best step 1378 for both, `runs/sft_06b/train_summary.json`, `runs/sft_17b/train_summary.json`). Answers are read from the letter logits at each question's answer slot in one forward pass (docs/API_SPEC.md sections 4 and 5).

**Data v1.3** (frozen, decision 46; docs/DATA.md). Three question types: noul (yes or no), choice (2 to 26 options defined in the request) and score (an ordered scale defined in the request). Training data comes from CLINC150 (an `intent` choice with K uniform in 3 to 14 options including `other`, followed by one gold-dependent noul of kind `about_domain`, `about_intent` or `out_of_scope`) and SST-5 (a 3-level or 5-level `sentiment` score, followed by `is_positive` or `is_negative` on non-neutral records). Every noul kind has two or three templates, each asked in its positive and its negated phrasing. 0 to 2 form nouls per record (text length questions, label-independent) vary question position. 20 CLINC intents, two per domain, are held out entirely (`test_unseen_intents`, DATA.md section 4). Four unseen schemas never appear in training: AG News (4 topics), emotion (6 emotions plus an `expresses_emotion` noul), Banking77 (10-option subsets with `other`, gold always named) and Yelp (a 5-level star scale).

**Protocol.** jevmark runs are evaluated on all nine splits in full (decision 46). The generative baselines run on a fixed, committed subset of 500 records per split (`data/baseline_subset.json`, decision 47), and our runs are compared with them on exactly those records (`runs/<run>/metrics_subset.json`). Headline metrics cover gold-dependent questions; form nouls are reported apart (decision 44). All four jevmark runs use one code version for the model, encoding and metrics: `sft_06b` and `base_06b` ran at commit a1dc2bf and `sft_17b` and `base_17b` at 3a7169c, and `jevmark/` and `scripts/` are identical between the two commits. Every run read the same data files (identical sha256 in every `metrics.json`) on one T4 GPU with fp16 frozen weights, fp32 LoRA and an fp32 readout, with no fp32 fallback needed (`<run>:precision`). ECE uses 15 equal-width bins on the top-1 probability; coverage uses the response confidence field (1 - H/ln K for choice and score, max(p, 1 - p) for noul). 95 percent intervals are bootstrap intervals by record (1000 resamples).

## 2. Headline results on the full splits

Accuracy [95% interval] / ECE, gold-dependent questions. Source: `<run>:splits.<split>.<block>` for `base_06b`, `sft_06b`, `base_17b`, `sft_17b`.

| split / type | n | base_06b | sft_06b | base_17b | sft_17b |
|---|---|---|---|---|---|
| **train** | 42422 | .524 [.519, .529] / .094 | .932 [.929, .934] / .009 | .590 [.585, .594] / .067 | .931 [.928, .933] / .004 |
| noul | 20399 | .510 [.503, .516] / .191 | .988 [.987, .990] / .003 | .528 [.521, .534] / .127 | .988 [.986, .989] / .002 |
| choice | 13482 | .629 [.621, .637] / .179 | .990 [.989, .992] / .003 | .738 [.730, .745] / .030 | .990 [.988, .992] / .001 |
| score | 8541 | .391 [.380, .401] / .046 | .705 [.696, .715] / .034 | .506 [.495, .516] / .035 | .703 [.693, .712] / .016 |
| **valid** | 7563 | .528 [.517, .538] / .097 | .916 [.909, .923] / .020 | .596 [.586, .607] / .065 | .923 [.917, .930] / .019 |
| noul | 3667 | .504 [.489, .518] / .198 | .966 [.960, .972] / .016 | .525 [.508, .541] / .123 | .967 [.961, .972] / .015 |
| choice | 2795 | .624 [.606, .642] / .176 | .970 [.964, .976] / .014 | .731 [.716, .747] / .027 | .978 [.973, .983] / .011 |
| score | 1101 | .365 [.338, .395] / .062 | .615 [.588, .642] / .059 | .491 [.463, .520] / .050 | .637 [.610, .666] / .054 |
| **test_indomain** | 11800 | .492 [.482, .500] / .099 | .956 [.952, .960] / .013 | .549 [.541, .558] / .079 | .952 [.948, .956] / .015 |
| noul | 5900 | .510 [.497, .522] / .199 | .937 [.931, .943] / .019 | .520 [.507, .533] / .111 | .930 [.923, .936] / .019 |
| choice | 5900 | .473 [.460, .486] / .102 | .975 [.971, .979] / .009 | .579 [.566, .590] / .048 | .974 [.970, .978] / .011 |
| **test_unseen_intents** | 6000 | .525 [.512, .538] / .120 | .892 [.883, .900] / .068 | .614 [.602, .628] / .040 | .897 [.889, .906] / .070 |
| noul | 3000 | .492 [.474, .509] / .209 | .916 [.906, .926] / .059 | .531 [.514, .550] / .096 | .910 [.900, .920] / .067 |
| choice | 3000 | .558 [.540, .576] / .145 | .868 [.855, .879] / .078 | .698 [.681, .714] / .044 | .885 [.874, .896] / .074 |
| **test_sst5** | 4031 | .442 [.427, .456] / .115 | .773 [.760, .786] / .030 | .506 [.491, .520] / .115 | .790 [.777, .803] / .025 |
| noul | 1821 | .511 [.488, .535] / .206 | .938 [.927, .949] / .024 | .507 [.484, .531] / .198 | .951 [.941, .960] / .019 |
| score | 2210 | .385 [.365, .405] / .053 | .636 [.616, .655] / .035 | .504 [.484, .526] / .049 | .657 [.638, .677] / .030 |
| **test_agnews** (choice) | 1000 | .786 [.760, .811] / .037 | .788 [.762, .813] / .164 | .836 [.812, .858] / .094 | .847 [.825, .869] / .113 |
| **test_emotion** | 2000 | .478 [.456, .503] / .045 | .628 [.604, .653] / .207 | .552 [.532, .575] / .090 | .645 [.624, .671] / .217 |
| noul | 1000 | .511 [.481, .544] / .068 | .707 [.679, .735] / .186 | .556 [.526, .585] / .092 | .726 [.699, .754] / .197 |
| choice | 1000 | .445 [.415, .476] / .045 | .549 [.517, .578] / .239 | .549 [.517, .580] / .089 | .565 [.536, .597] / .237 |
| **test_banking77** (choice) | 1000 | .718 [.691, .746] / .333 | .851 [.827, .872] / .067 | .793 [.767, .817] / .094 | .852 [.830, .873] / .087 |
| **test_yelp** (score) | 1000 | .382 [.353, .411] / .070 | .505 [.474, .536] / .169 | .517 [.487, .548] / .055 | .527 [.495, .557] / .218 |

Form nouls, reported apart (`<run>:splits.<split>.form`):

| split | n | base_06b | sft_06b | base_17b | sft_17b |
|---|---|---|---|---|---|
| train | 17298 | .492 [.486, .500] / .209 | .980 [.978, .982] / .003 | .504 [.497, .511] / .141 | .982 [.980, .984] / .003 |
| valid | 3019 | .488 [.471, .505] / .208 | .967 [.961, .973] / .013 | .506 [.488, .524] / .135 | .968 [.961, .974] / .015 |
| test_indomain | 4001 | .495 [.481, .511] / .189 | .969 [.963, .974] / .019 | .501 [.486, .516] / .127 | .974 [.968, .979] / .019 |
| test_unseen_intents | 1991 | .503 [.482, .525] / .183 | .973 [.966, .980] / .017 | .503 [.481, .526] / .119 | .974 [.967, .981] / .020 |
| test_sst5 | 2271 | .494 [.472, .514] / .232 | .956 [.948, .965] / .010 | .509 [.489, .528] / .153 | .959 [.950, .967] / .014 |

The v1 success criterion (docs/PLAN.md) holds at both sizes: SFT beats the frozen base in-domain by a wide margin (test_indomain .956 against .492 at 0.6B, .952 against .549 at 1.7B), beats it on unseen intents (.892 against .525, .897 against .614), and does not collapse on any unseen schema (it matches or beats the base's accuracy on all four).

## 3. Does the model read the questions and options?

**Unseen intents.** On the 20 held-out intents, which never appear in training as an utterance, an option or an asked intent, SFT choice accuracy is .868 (0.6B) and .885 (1.7B) against .975 and .974 in-domain; the frozen bases reach .558 and .698 (`<run>:splits.test_unseen_intents.choice.accuracy`). When the gold intent is among the options, accuracy is .853 and .875 (`<run>:splits.test_unseen_intents.choice.by_gold_other.gold_named.accuracy`). `about_intent`, which asks whether a description of a held-out intent fits the message, reaches .936 and .934 (base .487 and .542; `<run>:splits.test_unseen_intents.noul.by_kind.about_intent.accuracy`). The model generalises to option labels and descriptions it has never seen, with a drop of 9 to 11 points on choice and about 6 on `about_intent`.

**Unseen schemas against the frozen base.** SFT accuracy is at or above the base on all four unseen schemas at both sizes: AG News .788 against .786 (0.6B) and .847 against .836 (1.7B), emotion .628 against .478 and .645 against .552, Banking77 .851 against .718 and .852 against .793, Yelp .505 against .382 and .527 against .517 (`<run>:splits.<split>.overall.accuracy`). The gains are large where the base is weak (0.6B, emotion, Banking77) and within the intervals where the 1.7B base is already strong (AG News, Yelp).

**Negation consistency.** For every noul the evaluation also asks the negated phrasing; a consistent model gives P(yes | q) + P(yes | not q) near 1 and opposite answers (`<run>:splits.<split>.symmetry.by_kind`). The frozen bases say yes to both (mean sum 1.10 to 1.41, consistent answers in 2 to 34 percent of pairs); after SFT:

| split / kind | n | sft_06b sum (sd) / consistent | sft_17b sum (sd) / consistent |
|---|---|---|---|
| test_indomain about_domain | 1949 | .994 (.053) / .993 | .996 (.082) / .986 |
| test_indomain about_intent | 1951 | .999 (.024) / .997 | .998 (.040) / .997 |
| test_indomain out_of_scope | 2000 | .944 (.113) / .929 | .940 (.150) / .907 |
| test_unseen_intents about_domain | 1467 | .978 (.091) / .968 | .950 (.180) / .929 |
| test_unseen_intents about_intent | 1533 | .993 (.045) / .991 | .997 (.080) / .988 |
| test_sst5 is_negative | 932 | .990 (.064) / .983 | .996 (.051) / .989 |
| test_sst5 is_positive | 889 | .996 (.039) / .992 | .992 (.053) / .987 |
| test_emotion expresses_emotion (unseen) | 1000 | .963 (.166) / .917 | 1.057 (.175) / .910 |

The model reads the negation: every trained kind is at least 90 percent consistent, and the phrasing-only baseline for every kind is at 50 percent by construction (DATA.md section 5), so no phrasing predicts the answer. The weakest kinds are `out_of_scope` and, at 1.7B, the unseen `about_domain` and `expresses_emotion`, where 1.7B leans yes on both phrasings (sum 1.057, yes rate .530 against a gold .500, `sft_17b:splits.test_emotion.noul.by_kind.expresses_emotion.yes_rate`).

**Letter position.** Option order is a seeded random order in every stored split, so gold letters do not cluster. After SFT, accuracy by gold position on test_indomain ranges from .939 to .986 (0.6B) and .941 to 1.000 (1.7B) across all 14 positions, against .286 to .590 and .483 to .685 for the bases (`<run>:splits.test_indomain.letter_bias.by_position`). On test_unseen_intents the SFT spread is wider (.737 to .898 and .795 to .925), with the lowest values at positions 13 and 14, which exist only for K 13 and 14, where accuracy is lower at every position (`<run>:splits.test_unseen_intents.letter_bias.by_k_position`, about 14 questions per cell). No position-A preference is visible.

**Why these numbers are credible.** The data passes two CPU gates before any GPU run (docs/DATA.md section 8, decisions 42 and 43): state-free leak probes (logistic regression and gradient-boosted trees on phrasing, question text, structure and earlier questions, 5-fold cross-validation, fail above 10 points or above 3 points and the 99th percentile of 200 shuffled-target runs) find nothing above +3.0 points on v1.3, while the same probes find every known leak in v1.2 at +5 to +22 points; and a duplicate check finds no normalised text shared by train or valid and any test split. At most one gold-dependent noul follows the choice or score question, so no question can see another's answer through its construction.

## 4. Comparison with generative LLM baselines

On the 500-record subset. B1 is the instruct release of each backbone size (Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B, thinking disabled, greedy) generating JSON from the same questions; B2 is gpt-4.1-mini (snapshot gpt-4.1-mini-2025-04-14 for all 4500 replies, temperature 0) with a strict JSON schema per request (`runs/b2_gpt-4.1-mini/metrics.json:model`). jevmark columns are `runs/<run>/metrics_subset.json:splits.<split>.overall` (accuracy / ECE); baseline columns are `runs/<run>/metrics.json:splits.<split>.overall` with accuracy counting parse failures as wrong (`accuracy_all`, the strict headline), the lenient reading in brackets (`lenient.accuracy_all`, decision 48), ECE on the verbalized confidence, and the strict parse failure rate. Test splits only; train and valid are in-distribution for the SFT runs.

| split | n | sft_06b | B1 0.6B | sft_17b | B1 1.7B | B2 gpt-4.1-mini |
|---|---|---|---|---|---|---|
| test_indomain | 1000 | .944 / .019 | .618 (.619) / .358 / .018 | .945 / .024 | .698 (.737) / .253 / .056 | .908 (.908) / .049 / 0 |
| test_unseen_intents | 1000 | .896 / .072 | .567 (.569) / .403 / .018 | .916 / .059 | .625 (.666) / .325 / .066 | .889 (.889) / .074 / 0 |
| test_sst5 | 911 | .767 / .039 | .382 (.382) / .605 / 0 | .774 / .043 | .487 (.490) / .505 / .007 | .733 (.733) / .185 / 0 |
| test_agnews | 500 | .760 / .190 | .696 (.756) / .253 / .066 | .836 / .121 | .786 (.866) / .135 / .088 | .824 (.824) / .122 / 0 |
| test_emotion | 1000 | .630 / .202 | .455 (.455) / .421 / .147 | .633 / .226 | .567 (.568) / .397 / .015 | .656 (.656) / .242 / 0 |
| test_banking77 | 500 | .846 / .078 | .722 (.730) / .252 / .022 | .850 / .092 | .768 (.776) / .208 / .026 | .918 (.918) / .035 / 0 |
| test_yelp | 500 | .510 / .163 | .240 (.240) / .750 / 0 | .518 / .219 | .402 (.402) / .592 / .014 | .600 (.600) / .309 / 0 |

**Accuracy.** Against same-size generation, reading out wins on every test split under the strict reading, by 25 to 33 points in-domain and on unseen intents; under the lenient reading B1 1.7B is ahead on AG News only (.866 against .836). Against gpt-4.1-mini, the SFT models are ahead in-domain (.944 and .945 against .908) and on SST-5 (.767 and .774 against .733) at both sizes and on unseen intents at 1.7B (.916 against .889), level on unseen intents at 0.6B and on AG News at 1.7B, and behind on AG News at 0.6B (.760 against .824), Banking77 (.850 against .918), emotion (.633 against .656) and Yelp (.518 against .600). With 500 records per split, differences of a few points are within sampling error; the intervals are in the cited files (`accuracy_ci`, `accuracy_all_ci`).

**Format failures.** B1 fails to produce an allowed answer on up to 14.7 percent of questions (0.6B emotion) and on 5.6 to 8.8 percent of the CLINC test and AG News questions at 1.7B. Most 1.7B failures echo a whole option line such as `true: yes` (decision 48); the lenient reading recovers them and lifts 1.7B test_indomain from .698 to .737, still 21 points below SFT. B2 has zero parse failures on every split because its decoding is constrained by a JSON schema whose answers are enums or booleans; that is a property of the API's constrained decoding, not of the model, so format reliability is read from B1. jevmark cannot produce a malformed answer: it returns a distribution over the request's options.

**Confidence.** Verbalized confidence carries little information. B1 1.7B states 1.0 on 82 to 96 percent of its answers, and on test_indomain nouls keeps 99 percent of answers at every threshold from 0.5 to 0.95 at an unchanged accuracy of .644 (`runs/b1_qwen17b_json/metrics.json:splits.test_indomain.noul.coverage`). B2 is better calibrated but coarse: on test_indomain nouls, thresholds 0.5, 0.8 and 0.9 keep 100, 100 and 98 percent at .850 to .857, and only 0.95 separates (66 percent kept at .927). jevmark's probabilities rank its answers: sft_06b on the same questions keeps 89, 83 and 79 percent at 0.8, 0.9 and 0.95 with accuracy rising from .922 to .964, .976 and .985 (`runs/sft_06b/metrics_subset.json:splits.test_indomain.noul.coverage`). On unseen-intent choice, sft_17b goes from .906 at full coverage to .980 at 78 percent coverage (threshold 0.95), where B2 stays between .916 and .927. The confidence quantities differ (verbalized for the baselines, the response confidence field for jevmark), but both are what a caller would threshold.

**Latency and cost.** Batch-1 median wall clock per request on a T4 (`<run>:latency.batch_1.median_ms`): sft_06b 47.7 ms, sft_17b 74.8 ms, B1 0.6B 3277 ms, B1 1.7B 2969 ms; B2 first-attempt median 777 ms and p95 1095 ms, sequential calls from the caller's network (`runs/b2_gpt-4.1-mini/metrics.json:latency.first_attempt`). Throughput at batch 16 (`<run>:latency.batch_16_requests_per_second`): 30.5, 13.6, 3.3 and 3.4 requests per second. Cost per 1000 requests, assuming a T4 at 0.35 USD per hour (an assumption, not a measured price) and full utilisation at batch 16: sft_06b 0.0032 USD, sft_17b 0.0072 USD, B1 0.6B 0.030 USD, B1 1.7B 0.029 USD. B2 cost 0.2431 USD per 1000 requests at 0.40, 0.10 and 1.60 USD per million input, cached and output tokens, 1.0941 USD for all 4500 (`runs/b2_gpt-4.1-mini/metrics.json:usage`). On these numbers jevmark is about 40 to 70 times faster than same-size generation at batch 1 and about 34 to 76 times cheaper than the API per request, before any accuracy difference.

## 5. Calibration

In-domain calibration after SFT is good without any temperature: ECE .013 and .015 on test_indomain, .030 and .025 on test_sst5 (`<run>:splits.<split>.overall.ece`; `runs/sft_*/plots/test_indomain.png`). On the unseen schemas SFT is overconfident: ECE .164 and .113 on AG News, .207 and .217 on emotion, .169 and .218 on Yelp, while the frozen base is better calibrated on the same splits (.037 and .094, .045 and .090, .070 and .055), often at similar accuracy (Yelp .527 against .517 at 1.7B; `runs/sft_17b/plots/test_yelp.png`, `runs/base_17b/plots/test_yelp.png`). Banking77 is the exception, where the 0.6B base is badly calibrated (.333) and SFT is not (.067). Fine-tuning taught the model to be sharp, and it carries that sharpness to schemas where its accuracy does not justify it. Whether a single temperature fit in-domain, or an RLCD objective, can keep in-domain calibration while bringing unseen-schema ECE back towards the base's is the v2 question; this report does not claim either will.

## 6. Scale: what 1.7B buys over 0.6B

Paired differences, sft_17b minus sft_06b, on identical questions, with 95 percent bootstrap intervals by record (`runs/paired_17b_vs_06b/metrics.json:pairs.sft_17b_vs_sft_06b.splits.<split>.<block>`):

| split / block | n | delta [95% interval] |
|---|---|---|
| test_indomain overall | 11800 | -.0042 [-.0081, -.0003] |
| test_indomain noul, out_of_scope | 2000 | -.0225 [-.0385, -.0055] |
| test_unseen_intents choice | 3000 | +.0173 [+.0063, +.0283] |
| test_sst5 overall | 4031 | +.0174 [+.0062, +.0291] |
| test_sst5 score | 2210 | +.0213 [+.0050, +.0385] |
| test_agnews | 1000 | +.0590 [+.0390, +.0780] |
| test_emotion | 2000 | +.0175 [-.0025, +.0390] |
| test_banking77 | 1000 | +.0010 [-.0160, +.0180] |
| test_yelp | 1000 | +.0220 [-.0070, +.0480] |
| valid overall | 7563 | +.0069 [+.0019, +.0123] |

1.7B buys accuracy where the task is far from training: unseen-intent choice, AG News, and the SST-5 scale. It buys nothing in-domain, where both sizes are near their ceiling, and is slightly worse there, entirely through `out_of_scope`. The frozen bases differ far more (1.7B base +5.8 points on test_indomain and +9.0 on unseen intents, `pairs.base_17b_vs_base_06b`), so most of the backbone's advantage is absorbed by fine-tuning. 1.7B also costs 1.6 times the batch-1 latency and 2.2 times the per-request cost at batch 16 (section 4), and its negation consistency is lower on unseen `about_domain` (.929 against .968). On this data, 0.6B is the better trade for in-domain decisions; 1.7B is the better choice when schemas will be new.

## 7. Known limitations

- **`out_of_scope` is the weakest kind.** .850 (0.6B) and .828 (1.7B) on test_indomain, against .970 to .994 for the other CLINC kinds, with the lowest negation consistency (.929 and .907), `<run>:splits.test_indomain.noul.by_kind.out_of_scope`. The kind is balanced (phrasing-only baseline 50.1 percent on test_indomain, DATA.md section 5), so this is not a shortcut; it has 500 training questions against about 6500 for the other kinds (`runs/paired_17b_vs_06b/metrics.json:pairs.sft_17b_vs_sft_06b.splits.train.noul_by_kind.<kind>.n`), and it asks about the space of everything an assistant cannot do. test_unseen_intents has no `out_of_scope` questions (DATA.md section 2).
- **Order sensitivity.** Answers depend on the questions before them (causal attention). Reordering each test_indomain record's questions lowers noul accuracy from .937 to .878 (0.6B) and from .930 to .874 (1.7B), and changes 8.9 and 8.2 percent of noul answers; choice drops from .975 to .952 and from .974 to .964 (`<run>:splits.test_indomain.order_sensitivity`). API_SPEC section 4 therefore asks callers to put a dependent question after the question it depends on. Form nouls are almost unaffected (prediction agreement .996 and .997).
- **Prior shift of `other`.** The share of CLINC choice questions whose gold is `other` is 23 percent in train, 25 in valid, 47 in test_indomain (1000 out-of-scope utterances, each in two records) and 20 in test_unseen_intents, and 0 in Banking77 (`<run>:splits.<split>.choice.by_gold_other`). SFT follows the evidence rather than the prior in-domain (predicted-other rate .462 against a gold .468, accuracy on gold `other` .969 and .967), but it predicts `other` for 11.6 to 12.4 percent of held-out intents that are among the options and for 4.9 to 6.7 percent of Banking77 questions, where `other` is never right.
- **Label noise and ordinal ambiguity.** Score accuracy is low for everyone on 5-level scales: SFT .636 to .657 on SST-5 and .505 to .527 on Yelp, gpt-4.1-mini .596 and .600 on the subset. Mean absolute error of the expected level is .435 to .464 on SST-5 and .482 to .526 on Yelp (`<run>:splits.<split>.score.mae`), so most errors fall to a neighbouring level. How much of that is label noise in SST-5 and Yelp is not measured here.
- **Form nouls are not a decision capability.** They measure counting characters and words against a threshold in the question; the bases are at chance (.488 to .509) and SFT at .956 to .982 (section 2 table). They are in the data to vary question position and are kept out of the headline.
- **SST-5 neutral shift, rechecked on the full split.** In the 300-step fast cycle a preceding form noul appeared to double the rate of predicting neutral. On the full test_sst5 after SFT the neutral share is the same with and without a preceding form noul: 11.8 against 12.4 percent on the 5-level scale and 15.2 against 15.3 percent on the 3-level scale at 0.6B, 14.3 against 12.4 and 13.6 against 13.4 at 1.7B, all below the gold neutral shares of 15.4 to 18.6 percent (`<run>:splits.test_sst5.score.levels_by_position`). The fast-cycle effect was noise from an undertrained model; the frozen bases do shift with context (base_17b 5-level: 16.7 against 0.0 percent).
- **B1 batch-1 p95 is missing.** The B1 runs recorded only the median and mean at batch 1; per-request timings, and so p95, are recorded from the next latency probe on (docs/TASKS.md 1.8). The jevmark runs record the median and mean only.
- **1.7B does not batch on a T4.** base_17b takes 68.2 ms per request at batch 1 and 68.1 ms per request at batch 16 (14.7 requests per second), while 0.6B goes from 48.8 to 32.8 ms (`<run>:latency`). The cause is not established. The merged LoRA adapter adds about 10 percent latency at 1.7B (sft_17b 74.8 against base_17b 68.2 ms) and nothing at 0.6B.
- **B2 latency includes the network.** B2 was called one request at a time from a laptop; its latency includes the round trip and the API's queueing, and one call of the first 1800 took 601.8 s (the client's former 600 s timeout plus a silent retry), which is counted as retried and excluded from the first-attempt figures (`runs/b2_gpt-4.1-mini/metrics.json:latency`).
- **Batching precision.** Batched and single calls differ by up to .0068 to .0094 in probability under fp16 on a T4 (`<run>:batching_precision.max_abs_difference`, API_SPEC section 1).

## 8. Reproducibility

| Run | Commit | Directory |
|---|---|---|
| base_06b, sft_06b | a1dc2bf | `runs/base_06b/`, `runs/sft_06b/` |
| base_17b, sft_17b | 3a7169c | `runs/base_17b/`, `runs/sft_17b/` |
| B1 0.6B, B1 1.7B | 972e711 | `runs/b1_qwen06b_json/`, `runs/b1_qwen17b_json/` |
| B2 gpt-4.1-mini | 972e711 (first 1800 replies), 687a90d (the other 2700; request bodies identical) | `runs/b2_gpt-4.1-mini/` (sub run metrics in `metrics_sub200.json`) |
| Paired deltas | this report's commit | `runs/paired_17b_vs_06b/metrics.json` |

Data v1.3 sha256, identical in every run's `data_files_sha256` and in `data/baseline_subset.json`: train `84f218d5`, valid `e35102a5`, test_indomain `c68c7fd0`, test_unseen_intents `3b1a0ff1`, test_sst5 `8cf72b61`, test_agnews `848e6d4a`, test_emotion `12bc79d0`, test_banking77 `5f051453`, test_yelp `9a02eb5b` (first 8 hex digits; full values in any `metrics.json`). `make data` rebuilds them from the pinned dataset revisions (DATA.md section 3).

Every table regenerates without a GPU from the gitignored per-question files (`runs/<run>/results.jsonl.gz` for jevmark runs, `runs/<run>/replies.jsonl` for baselines), which are kept outside git:

- `uv run python scripts/recompute_metrics.py runs/<run>` rebuilds a jevmark run's `metrics.json` (section 2, 3, 5, 7), and `--subset` writes `metrics_subset.json` (section 4).
- `uv run python scripts/recompute_baseline_metrics.py runs/<run>` rebuilds a baseline's `metrics.json`, strict and lenient.
- `uv run python scripts/compare_baselines.py` prints the section 4 comparison for all seven runs, recomputed from the per-question files.
- `uv run python scripts/paired_deltas.py --pair runs/sft_17b runs/sft_06b --pair runs/base_17b runs/base_06b --out runs/paired_17b_vs_06b/metrics.json` rebuilds section 6.

The four jevmark `metrics.json` files were recomputed for this report to add `symmetry.by_kind` and `score.levels_by_position` (decision 49); every other value is unchanged apart from floating-point summation order (at most 1e-14).

## What the numbers say

A small base model with a LoRA adapter, reading answers from letter logits, decides in-domain questions at 95 to 96 percent accuracy with an ECE of .013 to .015, transfers to held-out intents and unseen schemas without collapsing, and does so 40 to 70 times faster than same-size generation and about 34 to 76 times cheaper per request than gpt-4.1-mini, whose accuracy it matches or beats in-domain and on unseen intents. The numbers do not show that its probabilities stay honest off-distribution: on unseen schemas SFT is markedly overconfident and the frozen base is better calibrated, and gpt-4.1-mini is still more accurate on Banking77, emotion and Yelp. Scale from 0.6B to 1.7B helps only far from the training data, so whether calibration can transfer without per-schema fitting, not raw accuracy, is the open question v2 has to answer.
