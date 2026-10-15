# Label-noise audit: review guide

The audit asks one question per row: does the gold label describe the message, given the options the model was shown? It is about the label, never about the model.

## Files

- `label_audit_sheet.csv`: the blind sheet, 200 rows in a shuffled order. Each row has the record id, question id, message, state format (plain or json; a json state is shown by its text field), the gold label with its description, and the full option list as the model saw it (label: description, separated by ` | `). Fill in `verdict` and, where useful, `note`.
- `label_audit_answers.csv`: the sample group of every row and the model answers, keyed by record id. Do not open it until every verdict on the sheet is filled in.
- `label_audit_cc_pass.csv`: Claude's first-pass verdicts, made from the sheet alone, before the answers were opened. Do not open it before your own pass either, so the two passes stay independent.
- `scripts/build_label_audit.py` rebuilds the sheet and the answers file from the seed.

## Verdicts

Judge from the message, the gold label and its description, and the other options only.

- **label_correct**: the gold label describes the message, and no other offered option describes it better.
- **ambiguous**: the gold label is a reasonable reading, but at least one other offered option is an equally reasonable one, or the message is too short or vague to decide between them.
- **label_wrong**: another offered option describes the message clearly better, or, if none does, the gold label does not describe the message at all.
- **convention**: the gold label follows a Banking77 annotation convention that a reader would not pick from the option descriptions alone, for example a question about a card PIN labelled get_physical_card, a SWIFT transfer labelled top_up_by_bank_transfer_charge, or a new card sent abroad labelled card_about_to_expire. Use it when the label is consistent with how the dataset labels such messages but not with the descriptions the model reads.

For a test_indomain row whose gold label is `other`, judge whether none of the offered intents fits the message.

## The samples (in the answers file)

- **a**: 100 questions from v3_banking77_test_full that full_sft and direct_brier (N 5000, seed 0) both answer wrongly, drawn from the 127 such questions.
- **b**: 50 questions drawn uniformly from v3_banking77_test_full.
- **c**: 50 intent questions drawn uniformly from test_indomain (CLINC150), for contrast.

## How the summary will be computed

After the human pass is complete, and only then, the answers file is joined to the sheet by record id, and per group:

1. The count and share of each verdict, with a 95 percent Wilson interval per share.
2. Group b estimates the share of label problems in the full Banking77 test split: label_wrong alone, and label_wrong, convention and ambiguous together.
3. Group a estimates the share of the shared errors that are label problems. Scaled by the 127 questions in 3080 (4.1 percent of the split) that both learners miss, it gives how much of each learner's error rate could be label noise rather than model error, that is, how far below a perfect score the ceiling on this split lies. It says nothing about the gap between direct_brier and full_sft, which lies in the questions where they disagree.
4. Group c gives the same shares for the in-domain CLINC150 test split, the contrast for a dataset trained on.
5. The agreement between the human and the first pass: the share of rows with the same verdict and Cohen's kappa over the four values, with the rows where they differ listed. The human verdict is the one reported.

The results go into the limitations of docs/RESULTS_v3.md and an addendum to docs/RESULTS_v1.md.

## Emotion

The same audit for the emotion log domain (dair-ai/emotion), on the choice question of v3_emotion_test_full: which of six emotions the message expresses most strongly. There is no `other` option, and every row shows the same six options in a seeded order.

### Files

- `emotion_audit_sheet.csv`: the blind sheet, 147 rows in a shuffled order, with the same columns as the Banking77 sheet. Fill in `verdict` and, where useful, `note`.
- `emotion_audit_answers.csv`: the sample group of every row and the zero-shot, full_sft and positive_sft answers, keyed by record id. Do not open it until every verdict on the sheet is filled in.
- `emotion_audit_cc_pass.csv`: Claude's first-pass verdicts, made from the sheet alone, before the answers were opened. Do not open it before your own pass.
- `scripts/build_label_audit.py --domain emotion` rebuilds the sheet and the answers file from the seed.

### Verdicts

Use the same four verdicts, defined as above. Read convention for this dataset as follows: the gold label tracks an emotion word in the message, whatever that word means in context. That covers a word used in another sense, a negated word, or a word that is not about the writer. For example, a message about cold weather or a cold object labelled anger, because the message contains "cold". Use convention when the label can be traced to such a word but a reader would not pick it from the option descriptions. Use label_wrong when the message itself names or clearly shows another emotion and no such word explains the gold label.

### The samples (in the answers file)

- **a**: 100 questions that full_sft and positive_sft (N 5000, seed 0) both answer wrongly, drawn from the 197 such questions. direct_brier had not run on emotion when the sheet was built, so unlike the Banking77 audit, group a is defined by full_sft and positive_sft. Questions that direct_brier also misses can be picked out of group a afterwards.
- **b**: 50 questions drawn uniformly from v3_emotion_test_full (2000 questions), independently of a. Three questions are in both samples, which is why the sheet has 147 rows.

There is no contrast group. CLINC150 already has one in the Banking77 audit.

### How the summary will be computed

As for Banking77, after the human pass is complete. Group b estimates the share of label problems in the emotion test split. Group a estimates the share of the shared errors that are label problems, scaled by the 197 questions in 2000 that both learners miss. Agreement with the first pass is reported as the same-verdict share and Cohen's kappa. `scripts/summarise_label_audit.py` reads the Banking77 files only and needs an emotion mode before this summary runs.
