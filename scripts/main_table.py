"""The paper's main-text summary table (Table 1), from the committed stage files only.

    uv run python scripts/main_table.py [--runs-dir runs] [--out docs/tables/paper/main_table.tex]
                                        [--check-tex ../jevmark-paper/paper/main.tex]

Reads runs/v3_stage_06b/metrics.json, runs/v3_stage_17b/metrics.json and
runs/v3_stage_emotion_06b/metrics.json and writes a booktabs table at N 5000 with
three blocks (Banking77 0.6B, Banking77 1.7B, emotion 0.6B) and, in each, the rows
zero-shot, temperature, positive-only SFT, CAPS-Brier and full-label SFT; the
columns are accuracy, ECE and AURC.

A trained learner with more than one training seed in its stage file is shown as the
seed mean with the range in parentheses: CAPS-Brier from seeds.training_seeds, and
positive-only and full-label SFT from seeds.positive_sft.training_seeds and
seeds.full_sft.training_seeds, so further baseline seeds appear here, with no code
change, once compare_v3.py has written them. Otherwise the learner is shown at seed 0,
n_curve.5000.<learner>, with a dagger (one training seed). Zero-shot and the
temperature learner have no training seed and carry no dagger.

Every cell is asserted equal to the stored key it is read from, and the same keys
feed the appendix tables (A.1.1, A.2.1, A.5.1 and the text after A.5.2, A.7.1 and
A.7.3). With --check-tex, every cell is also compared with the cell the paper's
LaTeX prints in those tables, and the script stops on any difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "ece", "aurc")
LEARNERS = (("zero_shot", "zero-shot"), ("temperature", "temperature"), ("positive_sft", "positive-only SFT"), ("direct_brier", "CAPS-Brier"),
            ("full_sft", "full-label SFT"))
TRAINED = ("positive_sft", "direct_brier", "full_sft")
# block key, stage directory, heading, decimals per metric (the appendix tables' precision: AURC to four decimals on Banking77, three on emotion)
BLOCKS = (("banking77_06b", "v3_stage_06b", "Banking77, 0.6B", {"accuracy": 3, "ece": 3, "aurc": 4}),
          ("banking77_17b", "v3_stage_17b", "Banking77, 1.7B", {"accuracy": 3, "ece": 3, "aurc": 4}),
          ("emotion_06b", "v3_stage_emotion_06b", "emotion, 0.6B", {"accuracy": 3, "ece": 3, "aurc": 3}))
CAPTION = ("Summary at N 5000. Three-seed mean (range) where three seeds exist, otherwise seed 0 ($\\dagger$). Intervals and paired differences are in "
           "Appendix~\\ref{sec:A}; every interval there is a record bootstrap with no training-seed variance.")


def num(x: float, decimals: int) -> str:
    """The paper's style: no leading zero, a true minus sign."""
    text = f"{x:.{decimals}f}"
    neg = text.startswith("-")
    text = text.lstrip("-")
    if text.startswith("0."):
        text = text[1:]
    return ("$-$" if neg else "") + text


def seed_block(stage: dict[str, Any], learner: str) -> dict[str, Any] | None:
    """The stage file's training-seed block of a trained learner, or None if it has none."""
    seeds = stage["seeds"]
    if learner == "direct_brier":
        return seeds.get("training_seeds")
    return (seeds.get(learner) or {}).get("training_seeds")


def cell(stage: dict[str, Any], learner: str, metric: str, decimals: int) -> tuple[str, str, float]:
    """(text, stored key, value) for one cell."""
    block = seed_block(stage, learner) if learner in TRAINED else None
    if block and block[metric]["n_seeds"] > 1:
        m = block[metric]
        key = ("seeds.training_seeds" if learner == "direct_brier" else f"seeds.{learner}.training_seeds") + f".{metric}"
        lo, hi = m["range"]
        assert lo <= m["mean"] <= hi, f"{key}: mean outside its range"
        return f"{num(m['mean'], decimals)} ({num(lo, decimals)}, {num(hi, decimals)})", key, m["mean"]
    key = f"n_curve.5000.{learner}.{metric}"
    value = stage["n_curve"]["5000"][learner][metric]
    dagger = "$^{\\dagger}$" if learner in TRAINED else ""
    return num(value, decimals) + dagger, key, value


def build(runs: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[str] = []
    cells: list[dict[str, Any]] = []
    for b, (block_key, stage_dir, heading, decimals) in enumerate(BLOCKS):
        stage = json.loads((runs / stage_dir / "metrics.json").read_text())
        if b:
            rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{heading}}}}} \\\\")
        for learner, name in LEARNERS:
            texts = []
            for metric in METRICS:
                text, key, value = cell(stage, learner, metric, decimals[metric])
                texts.append(text)
                cells.append({"block": block_key, "learner": learner, "metric": metric, "text": text, "file": f"runs/{stage_dir}/metrics.json", "key": key, "value": value})
            rows.append(f"\\quad {name} & " + " & ".join(texts) + " \\\\")
    tex = "\n".join([
        "% Written by scripts/main_table.py from runs/v3_stage_{06b,17b,emotion_06b}/metrics.json; do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{CAPTION}}}",
        "\\label{tab:main}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "learner & accuracy & ECE & AURC \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip{\\footnotesize $^{\\dagger}$One training seed.\\par}",
        "\\end{table}",
    ])
    return tex + "\n", cells


# --check-tex: the appendix cells that print the same values


def table_rows(tex: str, label: str) -> list[list[str]]:
    """The body rows of the table labelled `label`, split into cells."""
    start = tex.index(f"\\label{{{label}}}")
    body = tex[tex.index("\\midrule", start) + len("\\midrule"):tex.index("\\bottomrule", start)]
    return [[c.strip() for c in row.split("&")] for row in body.split("\\\\") if row.strip()]


def lead(text: str) -> str:
    """A cell's value without its bracketed interval: `.851 [.839, .863]` -> `.851`; a `mean (range)` cell is kept whole."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", text.strip())


def expected_cells(tex: str) -> dict[tuple[str, str, str], str]:
    """(block, learner, metric) -> the text the appendix prints for it."""
    out: dict[tuple[str, str, str], str] = {}
    a1 = {(r[0], r[1]): r for r in table_rows(tex, "tab:A1")}
    for metric, label in (("accuracy", "accuracy"), ("ece", "ECE"), ("aurc", "AURC")):
        row = a1[("5000", label)]
        for col, learner in ((2, "zero_shot"), (3, "temperature"), (4, "positive_sft")):
            out[("banking77_06b", learner, metric)] = lead(row[col])
    a2 = {r[0]: r for r in table_rows(tex, "tab:A2")}
    for metric, col in (("accuracy", 1), ("ece", 2), ("aurc", 5)):
        out[("banking77_06b", "direct_brier", metric)] = a2["CAPS-Brier"][col]
        out[("banking77_06b", "full_sft", metric)] = a2["full-label SFT"][col]
    a5 = {r[0]: r for r in table_rows(tex, "tab:A5")}
    for metric, col in (("accuracy", 1), ("ece", 2), ("aurc", 5)):
        for learner, name in (("zero_shot", "zero-shot"), ("temperature", "temperature"), ("positive_sft", "positive-only SFT"), ("full_sft", "full-label SFT")):
            out[("banking77_17b", learner, metric)] = lead(a5[name][col])
    after = tex[tex.index("\\label{tab:A5b}"):tex.index("\\label{sec:A6}")]
    for metric, word in (("accuracy", "accuracy"), ("ece", "ECE"), ("aurc", "AURC is")):
        m = re.search(re.escape(word) + r" (\.\d+ \(\.\d+, \.\d+\))", after)
        assert m, f"the text after Table A.5.2 has no {word} value"
        out[("banking77_17b", "direct_brier", metric)] = m.group(1)
    a7 = {(r[0], r[1]): r for r in table_rows(tex, "tab:A7")}
    for metric, label in (("accuracy", "accuracy"), ("ece", "ECE"), ("aurc", "AURC")):
        row = a7[("5000", label)]
        for col, learner in ((2, "zero_shot"), (3, "temperature"), (4, "positive_sft"), (5, "full_sft")):
            out[("emotion_06b", learner, metric)] = lead(row[col])
    a7c = {r[0]: r for r in table_rows(tex, "tab:A7c")}
    for metric, col in (("accuracy", 1), ("ece", 2), ("aurc", 5)):
        out[("emotion_06b", "direct_brier", metric)] = a7c["mean of 3 training seeds (range)"][col]
    return out


def check_tex(cells: Sequence[dict[str, Any]], tex_path: Path) -> int:
    expected = expected_cells(tex_path.read_text())
    bad = 0
    for c in cells:
        want = expected[(c["block"], c["learner"], c["metric"])]
        got = c["text"].replace("$^{\\dagger}$", "")
        if got != want:
            print(f"MISMATCH {c['block']} {c['learner']} {c['metric']}: Table 1 {got!r}, appendix {want!r} ({c['file']}: {c['key']})")
            bad += 1
    print(f"checked {len(cells)} cells against {tex_path}: {len(cells) - bad} equal, {bad} different")
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=str(REPO / "docs" / "tables" / "paper" / "main_table.tex"))
    parser.add_argument("--check-tex", default=None, help="the paper's main.tex: compare every cell with the appendix table that prints it")
    args = parser.parse_args(argv)
    tex, cells = build(Path(args.runs_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")
    for c in cells:
        print(f"{c['block']:14} {c['learner']:13} {c['metric']:9} {c['text']:28} {c['file']}: {c['key']}")
    if args.check_tex and check_tex(cells, Path(args.check_tex)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
