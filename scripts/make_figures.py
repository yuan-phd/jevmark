"""The report figures, drawn on CPU from committed metrics.json files only.

    uv run python scripts/make_figures.py [--runs-dir runs] [--out docs/figures]

Reads `runs/<run>/metrics.json`, `metrics_subset.json`, the inversion
summaries and the comparison files (`runs/rlcd_stage2a_06b`, `runs/v3_stage_06b`),
never results.jsonl.gz or replies.jsonl, so every figure rebuilds from git. Writes:

- v1_full_splits.png: accuracy and ECE per test split for base_06b, sft_06b, base_17b, sft_17b.
- v1_subset.png: the 500-record subset: accuracy, ECE and batch-1 latency for sft_06b,
  sft_17b, B1 at both sizes and B2.
- v2_cascade.png: coverage against accuracy on the four unseen schemas for sft_06b,
  sft_06b_temp and outcome_minus_p (seed mean).
- v2_reliability.png: reliability on the four unseen schemas for sft_06b and sft_06b_temp.
- v2_arm_ece.png: unseen-schema mean ECE per RLCD arm before and after its own temperature,
  with the range over seeds 0, 1 and 2.
- v3_n_curve.png: accuracy and ECE against N on v3_banking77_test_full per learner, with the
  seed ranges at N 5000.
- v3_coverage.png: coverage against accuracy at N 5000 on v3_banking77_test_full.
- v3_noise.png: the noisy diagnostics on the channel scale against the mapped-clean reference,
  and their accuracy.

Each figure carries a caption line naming its metrics files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The reference palette of the data-viz method, light mode, in its validated order (adjacent pairs pass the CVD and
# normal-vision checks); text wears ink tokens, never series colours; references are neutral grey.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

TEST_SPLITS = ["test_indomain", "test_unseen_intents", "test_sst5", "test_agnews", "test_emotion", "test_banking77", "test_yelp"]
UNSEEN = ["test_agnews", "test_emotion", "test_banking77", "test_yelp"]
UNSEEN_TYPE = {"test_agnews": "choice", "test_emotion": "choice", "test_banking77": "choice", "test_yelp": "score"}
SHORT = {"test_indomain": "in-domain", "test_unseen_intents": "unseen\nintents", "test_sst5": "SST-5", "test_agnews": "AG News",
         "test_emotion": "emotion", "test_banking77": "Banking77", "test_yelp": "Yelp"}
ARMS = ["sft_cont", "outcome", "outcome_minus_p", "direct_brier", "direct_log"]
NS = [500, 2000, 5000]
V3 = "v3_banking77_test_full"
FLIP_FLOOR = 0.5004024235381879  # H(0.2), the best achievable selection criterion under the 0.2 flip


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    })


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)


def save(fig: plt.Figure, out: Path, name: str, written: list[Path]) -> None:
    path = out / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)


def grouped_bars(ax: plt.Axes, groups: Sequence[str], series: dict[str, Sequence[float]], colors: Sequence[str]) -> None:
    width = 0.8 / len(series)
    for k, (label, values) in enumerate(series.items()):
        xs = [i + (k - (len(series) - 1) / 2) * width for i in range(len(groups))]
        ax.bar(xs, values, width * 0.88, color=colors[k], label=label, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def coverage_points(block: dict[str, Any]) -> tuple[list[float], list[float]]:
    rows = [r for r in block["coverage"] if r["n"] > 0 and r["accuracy"] is not None]
    return [r["coverage"] for r in rows], [r["accuracy"] for r in rows]


def mean_coverage(blocks: Sequence[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Seed mean of coverage curves at the same thresholds: coverage averaged, accuracy pooled over the answers kept."""
    xs, ys = [], []
    for rows in zip(*(b["coverage"] for b in blocks)):
        kept = sum(r["n"] for r in rows)
        if kept == 0:
            continue
        xs.append(sum(r["coverage"] for r in rows) / len(rows))
        ys.append(sum(r["n"] * r["accuracy"] for r in rows if r["n"]) / kept)
    return xs, ys


def unseen_mean_ece(metrics: dict[str, Any]) -> float:
    return sum(metrics["splits"][s]["overall"]["ece"] for s in UNSEEN) / len(UNSEEN)


# v1


def v1_full_splits(runs: Path, out: Path, written: list[Path]) -> None:
    names = ["base_06b", "sft_06b", "base_17b", "sft_17b"]
    metrics = {n: load(runs / n / "metrics.json")["splits"] for n in names}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, metric, label in ((axes[0], "accuracy", "accuracy"), (axes[1], "ece", "ECE (top-1, 15 bins)")):
        grouped_bars(ax, [SHORT[s] for s in TEST_SPLITS], {n: [metrics[n][s]["overall"][metric] for s in TEST_SPLITS] for n in names}, SERIES[:4])
        ax.set_ylabel(label)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("v1: frozen base and SFT on every test split (gold-dependent questions)", loc="left")
    axes[0].legend(ncol=4, loc="upper right")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    caption(fig, "Source: runs/{base,sft}_{06b,17b}/metrics.json, splits.<split>.overall.{accuracy,ece}; full splits.")
    save(fig, out, "v1_full_splits.png", written)


def v1_subset(runs: Path, out: Path, written: list[Path]) -> None:
    jev = {"sft_06b": "sft_06b", "sft_17b": "sft_17b"}
    base = {"B1 0.6B": "b1_qwen06b_json", "B1 1.7B": "b1_qwen17b_json", "B2 gpt-4.1-mini": "b2_gpt-4.1-mini"}
    labels = list(jev) + list(base)
    acc, ece, latency = {}, {}, {}
    for label, run in jev.items():
        sub = load(runs / run / "metrics_subset.json")
        acc[label] = [sub["splits"][s]["overall"]["accuracy"] for s in TEST_SPLITS]
        ece[label] = [sub["splits"][s]["overall"]["ece"] for s in TEST_SPLITS]
        latency[label] = load(runs / run / "metrics.json")["latency"]["batch_1"]["median_ms"]
    for label, run in base.items():
        m = load(runs / run / "metrics.json")
        acc[label] = [m["splits"][s]["overall"]["accuracy_all"] for s in TEST_SPLITS]
        ece[label] = [m["splits"][s]["overall"]["ece"] for s in TEST_SPLITS]
        lat = m["latency"]
        latency[label] = lat["batch_1"]["median_ms"] if "batch_1" in lat else lat["first_attempt"]["median_ms"]
    colors = [SERIES[1], SERIES[3], SERIES[4], SERIES[5], SERIES[6]]  # sft_06b and sft_17b keep their v1_full_splits colours
    fig = plt.figure(figsize=(9, 8.2))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8])
    ax_acc, ax_ece, ax_lat = fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[2])
    grouped_bars(ax_acc, [SHORT[s] for s in TEST_SPLITS], acc, colors)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("accuracy (failures count as wrong)")
    fig.suptitle("v1: jevmark against JSON generation (B1) and gpt-4.1-mini (B2) on the 500-record subset", x=0.01, ha="left", fontsize=10)
    handles, names = ax_acc.get_legend_handles_labels()
    fig.legend(handles, names, ncol=5, loc="upper left", bbox_to_anchor=(0.01, 0.965))
    grouped_bars(ax_ece, [SHORT[s] for s in TEST_SPLITS], ece, colors)
    ax_ece.set_ylabel("ECE (jevmark: top-1;\nbaselines: verbalized)")
    ys = range(len(labels))
    ax_lat.barh(list(ys), [latency[l] for l in labels], color=colors, height=0.6, edgecolor=SURFACE)
    ax_lat.set_yticks(list(ys), labels)
    ax_lat.invert_yaxis()
    ax_lat.set_xscale("log")
    ax_lat.set_xticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1000", "3000"])
    ax_lat.minorticks_off()
    ax_lat.set_xlabel("median latency per request, batch 1, ms (log scale; B2 includes the network)")
    ax_lat.grid(axis="y", visible=False)
    for y, l in zip(ys, labels):
        ax_lat.text(latency[l] * 1.08, y, f"{latency[l]:.0f} ms", va="center", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    caption(fig, "Source: runs/sft_{06b,17b}/metrics_subset.json and metrics.json:latency.batch_1; runs/b1_qwen{06b,17b}_json/metrics.json; "
                 "runs/b2_gpt-4.1-mini/metrics.json:latency.first_attempt (accuracy_all, ece, median latency).")
    save(fig, out, "v1_subset.png", written)


# v2


def v2_cascade(runs: Path, out: Path, written: list[Path]) -> None:
    sft = load(runs / "sft_06b" / "metrics.json")["splits"]
    temp = load(runs / "sft_06b_temp" / "metrics.json")["splits"]
    arm = [load(runs / f"rlcd_06b_outcome_minus_p_s{s}" / "metrics.json")["splits"] for s in (0, 1, 2)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=False)
    for ax, split in zip(axes, UNSEEN):
        kind = UNSEEN_TYPE[split]
        for k, (label, xy) in enumerate((("sft_06b", coverage_points(sft[split][kind])),
                                         ("sft_06b + T", coverage_points(temp[split][kind])),
                                         ("outcome_minus_p (seed mean)", mean_coverage([a[split][kind] for a in arm])))):
            ax.plot(*xy, color=SERIES[k], marker=MARKERS[k], markersize=3.5, label=label)
        ax.set_title(f"{SHORT[split]} ({kind})", loc="left")
        ax.set_xlabel("coverage (share answered)")
        ax.set_xlim(0, 1.02)
    axes[0].set_ylabel("accuracy of the answers kept")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("v2 cascade: answer only above a confidence threshold (0 to 1, step .05), unseen schemas, 0.6B", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    caption(fig, "Source: runs/sft_06b/metrics.json, runs/sft_06b_temp/metrics.json, runs/rlcd_06b_outcome_minus_p_s{0,1,2}/metrics.json: "
                 "splits.<split>.<choice|score>.coverage (confidence 1 - H/ln K). Seed mean: coverage averaged, accuracy pooled.")
    save(fig, out, "v2_cascade.png", written)


def v2_reliability(runs: Path, out: Path, written: list[Path]) -> None:
    sft = load(runs / "sft_06b" / "metrics.json")["splits"]
    temp = load(runs / "sft_06b_temp" / "metrics.json")["splits"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, split in zip(axes, UNSEEN):
        ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="perfect calibration")
        for k, (label, block) in enumerate((("sft_06b", sft[split]["overall"]), ("sft_06b + T", temp[split]["overall"]))):
            bins = [b for b in block["reliability"] if b["count"]]
            ax.plot([b["mean_confidence"] for b in bins], [b["mean_accuracy"] for b in bins], color=SERIES[k], marker=MARKERS[k], markersize=4,
                    label=f"{label} (ECE {block['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(SHORT[split], loc="left")
        ax.set_xlabel("mean top-1 probability")
        ax.legend(loc="upper left", fontsize=6.5)
    axes[0].set_ylabel("accuracy in the bin")
    fig.suptitle("v2: reliability on the unseen schemas, sft_06b before and after one in-domain temperature", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    caption(fig, "Source: runs/sft_06b/metrics.json and runs/sft_06b_temp/metrics.json, splits.<split>.overall.reliability (15 equal-width bins; empty bins left out).")
    save(fig, out, "v2_reliability.png", written)


def v2_arm_ece(runs: Path, out: Path, written: list[Path]) -> None:
    stage = load(runs / "rlcd_stage2a_06b" / "metrics.json")
    unseen = stage["unseen_schemas"]
    before, before_rng, after, after_rng = [], [], [], []
    for arm in ARMS:
        a = unseen["arms"][arm]
        before.append(a["mean_ece"])
        before_rng.append(a["mean_ece_range"])
        seeds = [unseen_mean_ece(load(runs / f"rlcd_06b_{arm}_s{s}_temp" / "metrics.json")) for s in (0, 1, 2)]
        mean_after = stage["temperature_ablation"]["arms"][arm]["mean_ece_after"]
        if abs(sum(seeds) / 3 - mean_after) > 1e-9:
            raise SystemExit(f"{arm}: the _temp runs' mean {sum(seeds) / 3} differs from runs/rlcd_stage2a_06b's {mean_after}")
        after.append(mean_after)
        after_rng.append([min(seeds), max(seeds)])
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    xs = range(len(ARMS))
    for k, (label, values, ranges) in enumerate((("raw", before, before_rng), ("with its own temperature", after, after_rng))):
        pos = [x + (k - 0.5) * 0.36 for x in xs]
        err = [[v - r[0] for v, r in zip(values, ranges)], [r[1] - v for v, r in zip(values, ranges)]]
        ax.bar(pos, values, 0.32, color=SERIES[k], label=label, edgecolor=SURFACE, yerr=err, ecolor=INK_2, capsize=3, error_kw={"linewidth": 1})
    for value, label, style_ in ((unseen["mean_ece"]["sft"], "sft_06b", ":"), (unseen["mean_ece"]["sft_temp"], "sft_06b + T", "--")):
        ax.axhline(value, color=REFERENCE, linestyle=style_, linewidth=1.2, label=f"{label}, {value:.3f}")
    ax.set_xticks(list(xs), ARMS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("unseen-schema mean ECE")
    ax.set_title("v2: RLCD arms on 0.6B, mean of seeds 0, 1, 2 (bars) and their range (whiskers)", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    caption(fig, "Source: runs/rlcd_stage2a_06b/metrics.json (unseen_schemas, temperature_ablation); "
                 "runs/rlcd_06b_<arm>_s<k>_temp/metrics.json for the per-seed values after temperature. Mean over AG News, emotion, Banking77, Yelp.")
    save(fig, out, "v2_arm_ece.png", written)


# v3


V3_LEARNERS = [("zero_shot", "zero-shot"), ("temperature", "temperature"), ("positive_sft", "positive_sft"), ("full_sft", "full_sft"), ("direct_brier", "direct_brier (RLCD)")]


def v3_colors() -> dict[str, str]:
    return {"zero_shot": REFERENCE, "temperature": SERIES[3], "positive_sft": SERIES[2], "full_sft": SERIES[0], "direct_brier": SERIES[1]}


def v3_n_curve(runs: Path, out: Path, written: list[Path]) -> None:
    stage = load(runs / "v3_stage_06b" / "metrics.json")
    curve, seeds = stage["n_curve"], stage["seeds"]
    colors = v3_colors()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in ((axes[0], "accuracy", "accuracy"), (axes[1], "ece", "ECE (top-1, 15 bins)")):
        for k, (key, name) in enumerate(V3_LEARNERS):
            points = [curve[str(n)][key] for n in NS]
            ys = [p[metric] for p in points]
            err = [[p[metric] - p[f"{metric}_ci"][0] for p in points], [p[f"{metric}_ci"][1] - p[metric] for p in points]]
            line = "--" if key == "zero_shot" else "-"
            ax.errorbar(NS, ys, yerr=err, color=colors[key], marker=MARKERS[k], linestyle=line, capsize=2, elinewidth=0.8, label=name)
        for key, block, shift in (("direct_brier", seeds["training_seeds"], 1.09), ("full_sft", seeds["full_sft"]["training_seeds"], 1.17)):
            lo, hi = block[metric]["range"]
            ax.vlines(5000 * shift, lo, hi, color=colors[key], linewidth=4, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlabel("N, logged interactions the learner sees")
        ax.set_ylabel(label)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=5, loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("v3: learning Banking77 from the deployment log, v3_banking77_test_full (3080 records)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.87))
    caption(fig, "Source: runs/v3_stage_06b/metrics.json: n_curve.<N>.<learner> (seed 0; whiskers 95% record bootstrap), "
                 "seeds.training_seeds and seeds.full_sft.training_seeds (thick bars right of N 5000: range over training seeds 0, 1, 2, direct_brier then full_sft). The temperature changes no prediction, so its accuracy line covers zero-shot's.")
    save(fig, out, "v3_n_curve.png", written)


def v3_coverage(runs: Path, out: Path, written: list[Path]) -> None:
    sources = {"zero_shot": "v3_06b_zeroshot", "temperature": "v3_06b_temp_n5000", "positive_sft": "v3_06b_positive_sft_n5000_s0",
               "full_sft": "v3_06b_full_sft_n5000_s0", "direct_brier": "v3_06b_direct_brier_n5000_s0"}
    colors = v3_colors()
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for k, (key, name) in enumerate(V3_LEARNERS):
        block = load(runs / sources[key] / "metrics.json")["splits"][V3]["choice"]
        ax.plot(*coverage_points(block), color=colors[key], marker=MARKERS[k], markersize=3.5, linestyle="--" if key == "zero_shot" else "-", label=name)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("coverage (share answered)")
    ax.set_ylabel("accuracy of the answers kept")
    ax.set_title("v3 cascade at N 5000: confidence thresholds 0 to 1, step .05", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    caption(fig, "Source: runs/<run>/metrics.json, splits.v3_banking77_test_full.choice.coverage for v3_06b_zeroshot, v3_06b_temp_n5000, "
                 "v3_06b_{positive_sft,full_sft,direct_brier}_n5000_s0 (confidence 1 - H/ln K). The temperature curve covers zero-shot's: "
                 "a temperature changes no prediction and nearly preserves the confidence ranking (exactly for top-1; the entropy-based "
                 "confidence can reorder a few questions), so it only extends the curve to lower coverage.")
    save(fig, out, "v3_coverage.png", written)


def v3_noise(runs: Path, out: Path, written: list[Path]) -> None:
    noisy = [("fixed noise\nstep 250 (selected)", "v3_06b_direct_brier_n5000_s0_noisy"), ("fixed noise\nstep 1407", "v3_06b_direct_brier_n5000_s0_noisy_last"),
             ("soft target\nstep 350 (selected)", "v3_06b_direct_brier_n5000_s0_noisy_soft"), ("soft target\nstep 1407", "v3_06b_direct_brier_n5000_s0_noisy_soft_last")]
    inverted = {run: load(runs / f"{run}_inverted" / "metrics.json") for _, run in noisy}
    reference = inverted[noisy[0][1]]["clean_reference"]["mapped_channel"]
    accuracy_runs = [("clean\nstep 300", "v3_06b_direct_brier_n5000_s0"), ("clean\nstep 1407", "v3_06b_direct_brier_n5000_s0_last"), *noisy]
    zero = load(runs / "v3_06b_zeroshot" / "metrics.json")["splits"][V3]["overall"]["accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [4, 6]})
    ax = axes[0]
    colors = [SERIES[1], SERIES[1], SERIES[2], SERIES[2]]
    values = [inverted[run]["stored_channel"]["ece"] for _, run in noisy]
    cis = [inverted[run]["stored_channel"]["ece_ci"] for _, run in noisy]
    ax.axhspan(*reference["ece_ci"], color=GRID, zorder=0)
    ax.axhline(reference["ece"], color=REFERENCE, linestyle="--", linewidth=1.2, label=f"mapped-clean reference {reference['ece']:.3f}")
    ax.legend(loc="upper right")
    ax.bar(range(len(noisy)), values, 0.6, color=colors, edgecolor=SURFACE, hatch=["", "//", "", "//"],
           yerr=[[v - c[0] for v, c in zip(values, cis)], [c[1] - v for v, c in zip(values, cis)]], ecolor=INK_2, capsize=3)
    ax.set_xticks(range(len(noisy)), [label for label, _ in noisy], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("channel-scale ECE\n(top-1 against .2 + .6 x correct)")
    ax.set_title("calibration to the 0.2-flip channel", loc="left")
    ax = axes[1]
    acc_colors = [SERIES[0], SERIES[0], *colors]
    acc = [load(runs / run / "metrics.json")["splits"][V3]["overall"] for _, run in accuracy_runs]
    ys = [a["accuracy"] for a in acc]
    ax.bar(range(len(acc)), ys, 0.6, color=acc_colors, edgecolor=SURFACE, hatch=["", "//", "", "//", "", "//"],
           yerr=[[y - a["accuracy_ci"][0] for y, a in zip(ys, acc)], [a["accuracy_ci"][1] - y for y, a in zip(ys, acc)]], ecolor=INK_2, capsize=3)
    ax.axhline(zero, color=REFERENCE, linestyle="--", linewidth=1.2, label=f"zero-shot {zero:.3f}")
    ax.legend(loc="upper right")
    ax.set_ylim(0.75, 0.97)
    ax.set_xticks(range(len(acc)), [label for label, _ in accuracy_runs], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy: selected step (plain) and final step 1407 (hatched)", loc="left")
    fig.suptitle("v3 noisy feedback, direct_brier at N 5000: the residual is single-draw variance, the collapse is memorisation", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    caption(fig, "Source: runs/<run>_inverted/metrics.json:stored_channel and clean_reference.mapped_channel (whiskers and band: 95% record bootstrap); "
                 "runs/<run>/metrics.json:splits.v3_banking77_test_full.overall.accuracy for the clean, fixed-noise and soft-target runs and v3_06b_zeroshot.")
    save(fig, out, "v3_noise.png", written)


FIGURES = (v1_full_splits, v1_subset, v2_cascade, v2_reliability, v2_arm_ece, v3_n_curve, v3_coverage, v3_noise)
EXPECTED = ("v1_full_splits.png", "v1_subset.png", "v2_cascade.png", "v2_reliability.png", "v2_arm_ece.png", "v3_n_curve.png", "v3_coverage.png", "v3_noise.png")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--out", default=str(REPO / "docs" / "figures"))
    args = parser.parse_args(argv)
    runs, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    written: list[Path] = []
    for figure in FIGURES:
        figure(runs, out, written)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
