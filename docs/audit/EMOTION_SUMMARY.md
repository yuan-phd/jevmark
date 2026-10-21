# Emotion label-noise audit: summary of a two-model pass

**The judges were two language models, no human:** Claude (Fable 5.1 and Opus 5.5, via Claude Code) and GPT-Astra, October 2026. The first pass is Claude's, made from the blind sheet before the answers file was opened (`emotion_audit_cc_pass.csv`). The second is GPT's, filled into the blind sheet (`emotion_audit_sheet.csv`). Neither saw the other's verdicts or the model answers, as far as the files show. Both read the same review guide (`REVIEW_GUIDE.md`, emotion section), and Claude wrote its definition of convention for this dataset, so the two passes share that definition and are not independent in how they read it. Until a human pass exists, every number here is a model judgement, and the numbers both models agree on are the ones to rely on. All numbers are from `emotion_summary.json`, written by `scripts/summarise_label_audit.py --domain emotion`. Every row where the passes differ is in `emotion_disagreements.csv`.

The audit covers the choice question of v3_emotion_test_full: which of six emotions the message expresses most strongly.

## Verdict shares per group

Share of rows, with 95 percent Wilson intervals; first pass (Claude) / second pass (GPT).

| group | n | label_correct | ambiguous | label_wrong | convention | any label problem | both agree on a problem |
|---|---|---|---|---|---|---|---|
| a: full_sft and positive_sft both wrong | 100 | .07 / .10 | .30 / .24 | .11 / .10 | .52 / .56 | .93 [.86, .97] / .90 [.83, .94] | 89 (47 both convention, 5 both label_wrong) |
| b: uniform over v3_emotion_test_full | 50 | .68 / .60 | .12 / .20 | 0 / 0 | .20 / .20 | .32 [.21, .46] / .40 [.28, .54] | 16 (7 both convention, 0 both label_wrong) |

Intervals for the single verdicts in group b: convention .20 [.11, .33] in both passes, label_wrong 0 [0, .07] in both, ambiguous .12 [.06, .24] / .20 [.11, .33].

Agreement: .79 of rows overall, Cohen's kappa .69 over the four verdicts; group a .76 and kappa .61; group b .86 and kappa .73. 31 rows differ (24 in a, 7 in b).

## What both passes agree on and what depends on the judge

- **Agreed:**
  - About a third of a uniform sample of emotion test labels has a label problem of some kind (.32 and .40), against .12 and .14 on Banking77 and none on CLINC150 (SUMMARY.md).
  - Most of those problems are convention, where the gold label follows an emotion word in the message whatever that word means there: .20 in both passes, with 7 of the 10 rows the same in each.
  - Neither judge calls any label in the uniform sample plainly wrong.
  - Of the questions both learners miss, nine in ten carry a label problem (.93 and .90; 89 of 100 rows by both), and about half are convention (.52 and .56; 47 rows by both).
- **Judge-dependent:** where convention ends and ambiguous begins, and where ambiguous ends and label_wrong begins. Of the 31 disagreements:
  - 11 are ambiguous against convention;
  - 6 are ambiguous against label_correct;
  - 6 are convention against label_wrong;
  - 5 are ambiguous against label_wrong;
  - 3 are convention against label_correct.

  The share of convention in particular rests on a definition both judges took from the same guide.

## What the estimates imply

- **The accuracy ceiling.** full_sft (N 5000, seed 0) scores .871 on this question (.8705, `runs/v3_emotion_06b_full_sft_n5000_s0/metrics.json`, `splits.v3_emotion_test_full.choice.accuracy`). There is no single ceiling; it depends on whether the word convention counts as learnable.
  - **Wrong labels alone:** no plainly wrong label turned up in 50 uniform rows. The interval's upper end, .07, puts that ceiling at .93 or higher.
  - **A reader who goes only by the option descriptions:** that reader would also miss the convention rows, about one in five (.20 [.11, .33]). This puts their ceiling near .80, and lower once they read some ambiguous rows differently from the annotator.
  - **What full_sft's .871 says:** it is above .80, so full_sft has learned part of the convention from the log domain's labels, which follow the same rule. The remaining gap to a ceiling between .93 and 1 is largely the convention cases it has not learned.

  These intervals are wide. The figures are orders of magnitude, not ceilings to quote.
- **The shared errors.**
  - The 197 questions that full_sft and positive_sft both miss are 9.85 percent of the 2000 test questions.
  - Both judges find a label problem in 89 of the 100 sampled, and one or the other in up to 93. So about 8.8 to 9.2 points of full_sft's 12.95 points of error lie on questions whose label a reader would question.
  - Most of that is convention, not labels that are plainly wrong. Both judges call a label wrong in only 5 of the 100, about half a point.
  - On emotion, most of what the trained learners get wrong is the dataset's labelling rule rather than the message.
- **The clean feedback.** As on Banking77, the clean log reveals whether the chosen option matched the gold label, so it inherits these problems. Assume the train split follows the same rule, which is what full_sft's accuracy above .80 suggests. Then about one clean outcome in five rewards the word convention rather than the description, a larger label-driven channel than on Banking77.

## What the audit does not say

- **Nothing about direct_brier.** It has not run on emotion yet. Group a is defined by full_sft and positive_sft, so the audit can neither explain nor predict the direct_brier results still to come. Once they exist, the questions direct_brier also misses can be picked out of group a, but no conclusion here depends on them.
- **Nothing about the gap between learners.** Group a holds questions two learners both miss, and group b is uniform. The gap between any two learners lies in the questions on which they disagree, which neither sample targets.
