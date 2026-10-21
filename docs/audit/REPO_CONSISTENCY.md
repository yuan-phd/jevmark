# Repository consistency before submission

The check of this repository against the paper, "Learning a calibrated decision head from logged bandit feedback", before the arXiv submission. The paper's text was read from the private paper repository's docs/PAPER.md on 21 October 2026. Each finding names the file, what disagreed with the paper and the fix; where nothing disagreed, it says so.

## 1. README.md

| place | finding | fix |
|---|---|---|
| Results, first paragraph | "The paper is on arXiv, link to follow": the paper is not yet submitted. | Now says a paper with the paper's title is being prepared for arXiv, link to follow, and that it names the v3 learner CAPS-Brier. |
| v3 table | The column "RLCD" is the paper's CAPS-Brier. | Column renamed "RLCD (CAPS-Brier)"; every value checked against `runs/v3_stage_06b/metrics.json:n_curve` and unchanged. |
| v3, gpt-4.1-mini | "no paired interval": the paired comparison now exists and shows the difference is within noise, as the paper says. | Gives the paired seed mean +0.014 [-0.010, +0.037] and full-label SFT's +0.029 [+0.003, +0.053] (`runs/b2_paired_v3/metrics.json`). |
| Phase 4, emotion | The 4.3-point gap to full labels did not say it is a three-seed mean against one full-label seed. | States seed 0 for the 21.7-point lead and one full-label seed for the gap. |
| Phase 4, 1.7B | "the gap to full labels shrinks from 1.5 points at 0.6B to 0.6": the paper says CAPS-Brier is level with full_sft on seed 0 and .6 below as a seed mean, against one full-label seed. | Rewritten in the paper's words: level on seed 0, -0.4 [-1.0, +0.2]; 0.6 [0.2, 1.1] below as a seed mean; 1.5 [1.0, 2.0] at 0.6B between three-seed means. |
| Phase 4 and Limits | RLCD used for the logged-feedback learner. | CAPS-Brier, the paper's name. |
| Usage, Hub adapters | The adapter name rlcd_banking77_06b was not tied to the paper's name. | Says the paper calls the learner CAPS-Brier and the adapter keeps its original name. |
| Backbones | "v2's RLCD arms and all of v3 ran on 0.6B": the paper reports a 1.7B repeat at N 5000. | Corrected. |
| Reproducing the tables | The per-question files were said to come only from the Kaggle sessions; the run-files dataset was not mentioned; four scripts the paper relies on had no row. | Points to the dataset `yuanphd/jevmark-runs`, notes that the gpt-4.1-mini replies are not published, and adds rows for `scripts/v2_accuracy_table.py`, `scripts/v3_review.py` (both subcommands), `scripts/b2_paired.py`, `scripts/make_figures.py --paper` and `scripts/publish_runs.py`. |
| Documentation | "Paper: arXiv, link to follow" kept as the step 8 placeholder, now with the title. The references.bib line said one entry is marked NOT FOUND; no entry is (`grep NOT FOUND docs/references.bib` finds nothing). | Fixed the references.bib line to the paper's wording; added docs/figures/paper/, this file and both Hugging Face links. |
| AI assistance | Missing. | Added the paper's section 8 paragraph, word for word apart from "this paper" becoming "the paper". |

The Hub links resolve to what they describe: `yuanphd/jevmark` holds the folders rlcd_banking77_06b, sft_06b and sft_17b, and `yuanphd/jevmark-runs` holds 81 results files and the 4 logs plus its card (listed with `huggingface_hub.HfApi.list_repo_files` on 21 October 2026).

## 2. docs/MODEL_CARD.md and the Hub README

Every number in the card was checked against the metrics files: the adapter table (`runs/v3_stage_06b/metrics.json:n_curve.5000`, accuracy, interval, ECE, Brier, NLL for zero-shot, positive_sft, direct_brier and full_sft), the forgetting line (.929 and .917, `S:forgetting_table`) and the v1 table. None disagreed with the paper.

| place | finding | fix |
|---|---|---|
| Adapter table, rlcd_banking77_06b | Named the method RLCD. | Says the paper calls it CAPS-Brier (Chosen-Action Proper Scoring) and the folder keeps its original name. |
| Links | No link to the run-files dataset or the paper. | Added both (the paper as "arXiv, link to follow"). |
| gpt-4.1-mini | "no paired interval". | Paired: +.016 [-.008, +.040] for this adapter's run and +.014 [-.010, +.037] for the three-seed mean, within noise (`runs/b2_paired_v3/metrics.json`). |
| Limitations | "The RLCD adapter exists at 0.6B only" could be read as no 1.7B runs existing. | Says the released Banking77 adapter is 0.6B only and the paper's 1.7B runs are not released. |

The Hub README (`yuanphd/jevmark`, README.md, downloaded on 21 October 2026) differs from docs/MODEL_CARD.md as committed before this pass only by the missing non-affiliation paragraph ("jevmark is not affiliated with or endorsed by TypeSafe AI ..."). After this commit it also lacks the fixes above. Not uploaded from this session: the human uploads docs/MODEL_CARD.md as the Hub repository's README.md (the plan's step 8 pushes it again with the arXiv id). The emotion adapter was not added; the plan adds it only if the human asks.

## 3. docs/RESULTS_v3.md, docs/METHOD.md, docs/STORY.md

Checked against the paper for the 1.7B wording, the abstract's numbers, the citation years, the gpt-4.1-mini comparison and the dropped positive-unlabeled framing.

**Abstract numbers.** +.026 accuracy and -.053 ECE against positive_sft (CAPS-Brier's three-seed mean at N 5000, RESULTS_v3 section 3, `S:seeds.training_seed_mean_minus_positive_sft`), 21.7 points on emotion (section 13), 1.5 points below full labels at 0.6B (section 3), .6 at 1.7B against one full-label seed (section 12) and .636 to .111 (RESULTS_v2 section 2, METHOD and STORY): each is stored as the paper states it, and after the fixes below no sentence in the three files gives them without the paper's qualifiers. No number was changed.

**Citation years.** Every bracketed key in docs/METHOD.md exists in docs/references.bib (12 keys), and every author-year in the three files matches the paper: Gao and Wang 2026, Dudík, Langford and Li 2011, Foster et al. 2018, Bereket and Leskovec 2025. Damani et al. appears in none of them; the bib key `damani2025rlcr` carries the year 2026, as the paper's file does.

| file | finding | fix |
|---|---|---|
| RESULTS_v3 section 7 | "these rows have no paired interval against B2". | Adds the paired differences from `runs/b2_paired_v3/metrics.json:domains.banking77`: within noise for CAPS-Brier (seed mean +.014 [-.010, +.037], every seed's interval covering 0), +.029 [+.003, +.053] for full_sft. |
| RESULTS_v3 section 13 | The emotion B2 rows said no paired interval. | Adds the emotion paired differences: seed mean +.251 [+.209, +.295], full_sft's one seed +.290 [+.248, +.332]. |
| RESULTS_v3 section 12 | "The gap to full_sft shrinks from 1.5 points to .6" did not say the two gaps are not like for like. | Gives both gaps with their seed structure and the seed 0 level result. |
| METHOD section 3 | The corollary had parts (i) and (ii); the paper adds (iii), the expected gold-logit gradient, and the full split of the simulation. | Added (iii) with its one-line proof and metrics key, and the 74.2 percent unchanged share with its two parts. |
| METHOD section 5, full-label SFT | "so the gap shrinks with scale": the 1.7B gap is a seed mean against one full-label seed. | The paper's wording: level on seed 0, .6 below as a seed mean, with the seed structure stated. |
| METHOD section 5, REINFORCE | Did not state that outcome_minus_p_log is the IPS-weighted CAPS-Brier estimator in expectation, as the paper's section 3 does. | Added the identity, the tiny-model check (4.5 x 10^-7) and the clip shares (`S:off_policy`). |
| METHOD section 5, positive-only SFT | Already rewritten to drop the positive-unlabeled framing (commit 834ddf1). | None. |
| STORY, phase 4 | "the gap to full labels shrinks from 1.5 points at 0.6B to .6". | The paper's 1.7B wording. |
| STORY, phase 4 | "reached the same recipe and the same full-label result ... the full-label result as a replication under a stricter protocol": the paper now says the full-label result is consistent with OpenJev-RLCD's direct-scorer ablation and neither confirms nor contradicts its headline. | Rewritten to match. |
| STORY, phase 4 | "positive-only SFT is positive-unlabeled learning": the paper dropped that framing. | Replaced by the paper's description. |
| STORY, questions | gpt-4.1-mini comparison "without a paired interval". | Within noise, with the paired seed mean. |
| STORY, questions | ".6 at 1.7B and 4.3 on emotion" without the one-seed qualifier. | Qualifier added. |
| STORY, last section | Cited "PAPER section 6" for the work not done; the paper's limitations are section 5 (section 6 is the conclusion). | Section 5. |

Not changed, outside this pass: docs/RESULTS_v1.md, docs/RESULTS_v2.md and docs/RESULTS_v3.md cite decision and task numbers (11, 16 and 12 occurrences). The README states that these refer to the author's private working log and that no number depends on it. README.md, docs/MODEL_CARD.md, docs/STORY.md and docs/METHOD.md contain none.

## 4. Scripts named in the paper

Every script the paper names (Appendix B and section 7) exists in `scripts/`, and `uv run python scripts/<name>.py --help` exits 0 for each:

| script | named in | `--help` |
|---|---|---|
| compare_v3.py | Appendix B, tables | exit 0 |
| compare_noise.py | Appendix B, tables | exit 0 |
| compare_rlcd.py | Appendix B, tables | exit 0 |
| paired_deltas.py | Appendix B, tables | exit 0 |
| log_stats.py | Appendix B, tables | exit 0 |
| v2_accuracy_table.py | Appendix B, tables | exit 0 |
| simulate_advantages.py | Appendix B, tables | exit 0 |
| b2_paired.py | Appendix B, tables | exit 0 |
| make_figures.py | Appendix B, figures | exit 0 |
| publish_runs.py | Appendix B and section 7 | exit 0 |
| train_sft.py | Appendix B, training | exit 0 |
| train_rlcd.py | Appendix B and section 3 | exit 0 |
| derive_flip_log.py | section 7 | exit 0 |

For the paper session, not fixed here: the paper cites `S:off_policy`, `S:forgetting_table` and `S:selected_steps`, which `scripts/v3_review.py` writes (its `--help` also exits 0), but Appendix B lists only `scripts/compare_v3.py` for that file. The paper's figures are drawn by `scripts/make_figures.py --paper` into docs/figures/paper/, which is the source to copy into the paper repository.
