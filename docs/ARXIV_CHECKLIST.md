# arXiv submission checklist

The arXiv version of docs/PAPER.md is built from `paper/main.tex`. This file lists what the arXiv submission form asks for and the steps to upload the source.

## Before upload

- [ ] Fill in the author block in `paper/main.tex`: the `\author{...}` placeholders (NAME, AFFILIATION, EMAIL) and `pdfauthor` in `\hypersetup`. For more than one author, separate the authors with `\and`:

  ```latex
  \author{FIRST AUTHOR\\
    AFFILIATION\\
    \texttt{EMAIL}
    \and
    SECOND AUTHOR\\
    AFFILIATION\\
    \texttt{EMAIL}}
  ```

- [ ] `cd paper && make check`: fails on an undefined reference or citation, an overfull box wider than 10pt, or a missing figure file.
- [ ] `make arxiv`: runs the check and writes `paper/arxiv.tar.gz` with `main.tex`, `main.bbl` and the six figures under `figures/`. `references.bib` is left out because arXiv reads `main.bbl` and does not run BibTeX when the .bib is missing.
- [ ] If pdflatex is installed, `make clean && make` builds with arXiv's engine (pdflatex, bibtex, pdflatex, pdflatex). Otherwise the Makefile falls back to tectonic, which uses XeTeX: line breaks and fonts can then differ slightly from arXiv's build, so read arXiv's PDF preview closely.

## Form fields

- **Primary category:** cs.LG (Machine Learning).
- **Cross-list:** cs.CL (Computation and Language).
- **Licence:** CC BY 4.0 (Creative Commons Attribution 4.0 International).
- **Title:** Learning a calibrated decision head from logged bandit feedback
- **Authors:** as in the author block, in the same order, comma separated.
- **Comments (optional):** for example "21 pages, 6 figures. Code and adapters: https://github.com/yuan-phd/jevmark".
- **Abstract:** below. arXiv allows at most 1920 characters, and the paper's abstract is about 2,220 characters as plain text. The version below is 1905 characters, ASCII only, with no markup. It is the paper's abstract with these deletions and nothing new added:
  - "the lowest-probability wrong action sampled gets the largest advantage,"
  - "; the best arm was level, -.002 [-.006, +.004]"
  - the list after "a stricter protocol" (transfer to unseen label schemas, the continued cross-entropy control, four bandit arms under one temperature)
  - ", which is positive-unlabeled learning with the negatives discarded"
  - ", .0095 in AURC" (with ".053 in ECE" joined by "and")

  The abstract in the PDF stays the full one. Approve or shorten the form version before pasting it in.

```text
A decision head reads a state and a list of options and returns a probability for each option in one forward pass; a caller answers when the top probability clears a threshold and escalates otherwise. Once deployed, it usually observes only whether the option it chose was right: logged bandit feedback. We adapt such a head, a LoRA adapter on Qwen3-0.6B-Base and 1.7B-Base, from that feedback, and report three results.

First, a proper score of the sampled action's own probability is not a usable REINFORCE reward. That a detached, parameter-dependent reward drops the pathwise term of the gradient is known; we give the corollary for this reward: against a group-mean baseline every wrong action earns at least the gold action's reward, and at K = 2 the signal is exactly zero. In practice it cut one score task's accuracy from .636 to .111.

Second, with full labels, no bandit objective beat cross-entropy plus one temperature on calibration under label-space shift. This replicates a concurrent finding under a stricter protocol.

Third, from logged bandit feedback alone, Chosen-Action Proper Scoring (CAPS), the direct method with a proper score that fits the chosen action's probability to its logged outcome by the pathwise gradient, beats fine-tuning on confirmed answers only: on Banking77 by +2.6 points [1.9, 3.3] in accuracy and .053 in ECE. It stays 1.5 points [1.0, 2.0] below full-label training at 0.6B and .6 points [.2, 1.1] below it at 1.7B, so the gap shrinks with scale (three CAPS seeds, one full-label seed at 1.7B), and propensity weights add nothing, as expected of a well-specified regression. Under feedback flipped at a known rate, CAPS calibrates to the noisy channel without any correction, backward correction recovers most of the clean calibration, and early stopping on the held-out log-likelihood of logged outcomes prevents the memorisation of the single noisy draw.
```

## Upload

1. Log in at https://arxiv.org/user and choose "Start a new submission".
2. Accept the licence page with CC BY 4.0 and confirm you are an author with the right to submit.
3. Choose cs.LG as the primary category.
4. On the file page, upload `paper/arxiv.tar.gz`. arXiv unpacks it; the listing should show exactly `main.tex`, `main.bbl` and `figures/` with six PNG files.
5. Choose "Process". arXiv detects PNG figures and builds with pdflatex.
6. Open the build log from the process page and check that:
   - it ran pdflatex (not latex with dvips);
   - it reports no "Citation ... undefined" or "Reference ... undefined" warnings and no "There were undefined references";
   - it reports no missing file (`File ... not found`) and no font substitution warnings;
   - it reports no "Overfull \hbox" wider than 10pt;
   - the last line gives "Output written on main.pdf (N pages, ...)", with N close to the local build's 21.
7. Open arXiv's PDF preview and compare it with the local `paper/main.pdf`: title block, the keywords line, Corollary 1 and its proof, the six figures, tables with captions above, the references, and appendices A, B and C with the provenance table breaking across pages.
8. Fill in the metadata page (title, authors, abstract above, comments), add cs.CL as the cross-list, and submit.
9. After the announcement, add the arXiv id to README.md and docs/MODEL_CARD.md in a separate commit.
