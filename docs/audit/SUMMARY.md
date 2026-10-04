# Label-noise audit: summary of a two-model pass

**The judges were two language models, no human.** The first pass is Claude's, made from the blind sheet before the model answers were opened (`label_audit_cc_pass.csv`); the second is GPT's, filled into the blind sheet (`label_audit_sheet.csv`). Neither saw the other's verdicts or the model answers, as far as the files show. The review guide (`REVIEW_GUIDE.md`) planned a human pass; until one exists, every number here is a model judgement, and the numbers below that both models agree on are the ones to rely on. All numbers are from `summary.json`, written by `scripts/summarise_label_audit.py`; every row where the passes differ is in `disagreements.csv`.

## Verdict shares per group

Share of rows, with 95 percent Wilson intervals; first pass (Claude) / second pass (GPT).

| group | n | label_correct | ambiguous | label_wrong | convention | any label problem | both agree on a problem |
|---|---|---|---|---|---|---|---|
| a: both trained learners wrong | 100 | .34 / .38 | .39 / .22 | .19 / .30 | .08 / .10 | .66 [.56, .75] / .62 [.52, .71] | 56 (18 both label_wrong) |
| b: uniform over the Banking77 test split | 50 | .88 / .86 | .02 / .06 | .04 / .04 | .06 / .04 | .12 [.06, .24] / .14 [.07, .26] | 5 (1 both label_wrong) |
| c: uniform over test_indomain (CLINC150) | 50 | 1.00 / 1.00 | 0 / 0 | 0 / 0 | 0 / 0 | 0 [0, .07] / 0 [0, .07] | 0 |

Agreement: .82 of rows overall, Cohen's kappa .66 over the four verdicts; group a .69 and kappa .57; group b .90 and kappa .58; group c 1.00 (kappa undefined, every verdict label_correct in both passes). 36 rows differ.

## What is robust and what depends on the judge

- **Robust:** CLINC150's in-domain labels show no problem in 50 of 50 rows for both judges. On Banking77, both judges find a label problem of some kind in about one row in eight of a uniform sample (.12 and .14), and in about six of ten of the questions both trained learners miss (.66 and .62; 56 of those 100 rows by both).
- **Judge-dependent:** what kind of problem it is. On the shared errors the two passes split ambiguous and label_wrong differently (Claude .39 and .19, GPT .22 and .30); 29 of the 31 group-a disagreements involve ambiguous: 12 ambiguous against label_wrong, 14 ambiguous against label_correct in either direction, 3 ambiguous against convention. Where one judge sees a reasonable alternative reading, the other calls the label either right or wrong. The label_wrong share of the uniform sample (.04 in both) rests on two rows each, so its interval is wide, .01 to .13.

## What the group b estimate implies

- **The accuracy ceiling.** If about 4 percent of Banking77 test labels are wrong and a further 8 to 10 percent are ambiguous or follow an annotation convention the option descriptions do not state, a model that read the descriptions perfectly would still score below 1 against these labels, plausibly near .96 with the clearly wrong labels alone and lower when it reads an ambiguous or conventional case differently from the annotator. full_sft at .946 (seed mean, RESULTS_v3 section 3) is already near that region, and direct_brier at .931 not far below it. The intervals are wide (label_wrong .01 to .13), so this is an order of magnitude, not a ceiling to quote.
- **How deterministic the clean feedback was.** The "clean" log revealed whether the chosen option matched the gold label, so it inherits the labels' problems. Assuming the train split resembles the test split, roughly 4 percent of the clean outcomes rest on a wrong label, and another 8 to 10 percent on an ambiguous or conventional one, so the clean condition was not noise-free: it carried a small, label-driven noise channel that is not symmetric or random like the 0.2 flip, since it concentrates on particular intents and phrasings.
- **The shared errors.** The 127 questions both trained learners miss are 4.1 percent of the split. With .56 to .66 of them label problems, about 2.3 to 2.7 points of each learner's error comes from questions whose label a reader would question, and 0.7 to 1.2 points from labels both or one judge call wrong.

## What the audit does not say

Neither group speaks to the gap between direct_brier and full_sft (1.5 points [1.0, 2.0] at N 5000). Group a holds questions both learners miss and group b is uniform; the gap lies in the questions on which the two learners disagree, which neither sample targets.
