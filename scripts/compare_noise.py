"""The v3 noise-rate sweep: direct_brier and positive_sft at flip rates 0.1, 0.2 and 0.3, on CPU.

    uv run python scripts/compare_noise.py [--size 06b] [--runs-dir runs] [--resamples 1000]

Reads, on v3_banking77_test_full, N 5000, training seed 0, the seed 0 log:
- the clean runs v3_<size>_{direct_brier,positive_sft}_n5000_s0 and the zero-shot run
  (the alignment reference and the collapse floor);
- the noisy runs at each rate f: v3_<size>_<arm>_n5000_s0_noisy at 0.2 and
  ..._noisy_flip<f> at the other rates (trained on a sibling log, derive_flip_log.py);
- direct_brier's final adapter at each rate, <run>_last, and the clean run's;
- each noisy run's <run>_inverted/metrics.json from invert_noisy.py --flip f --clean
  <clean counterpart> (and <run>_last_inverted for direct_brier), checked against the
  rate, the source run's results and the clean counterpart. A file written before
  invert_noisy.py reported the channel scale gives channel_scale null; rerunning
  invert_noisy.py on its run fills it.

Writes runs/v3_noise_sweep_<size>/metrics.json with, per rate and learner: accuracy,
ECE, Brier and NLL with bootstrap intervals by record; the difference from the clean
run, paired; stored channel-scale ECE and the mapped-clean reference; the inverted
top-1 ECE; mean top-1 against the channel line f + (1 - 2f) x accuracy; the selection
criterion at the selected and the last step as its excess over the best achievable
value (-H(f) for flipped outcomes, 0 for clean ones) and the growth between them; for
direct_brier, the final adapter's metrics and its difference from the selected one.
Per rate, direct_brier's loss from clean minus positive_sft's, paired. The review
block `uncorrected` repeats, per rate including the clean runs (f 0), each learner's
uncorrected ECE (the stored top-1 against clean, gold correctness, before any
inversion) with its interval, and that paired difference of accuracy changes with its
interval, so the table has them side by side. Missing runs
are listed, and their entries are null. Every number comes from the runs' committed
or regenerable files; nothing is retrained.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jevmark.data.build import V3_TEST_FULL  # noqa: E402
from jevmark.feedback import FLIP_RATE  # noqa: E402
from jevmark.metrics import read_results  # noqa: E402
from jevmark.provenance import git_state  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_v3_for_noise", REPO / "scripts" / "compare_v3.py")
cv3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cv3
_spec.loader.exec_module(cv3)

RATES = (0.1, 0.2, 0.3)
ARMS = ("direct_brier", "positive_sft")
LAST_ARMS = ("direct_brier",)
N = 5000


def entropy(f: float) -> float:
    """H(f) in nats: minus the best achievable mean Bernoulli log-likelihood of outcomes flipped at rate f."""
    return 0.0 if f in (0.0, 1.0) else -(f * math.log(f) + (1 - f) * math.log(1 - f))


def run_name(size: str, arm: str, flip: float | None) -> str:
    """The clean run for flip None, else the noisy run at that rate (the sibling-log suffix away from 0.2)."""
    base = f"v3_{size}_{arm}_n{N}_s0"
    if flip is None:
        return base
    return base + "_noisy" + ("" if math.isclose(flip, FLIP_RATE) else f"_flip{flip:g}")


def selection(run_dir: Path, flip: float | None) -> dict[str, Any]:
    """The selection criterion at the selected and the last step, as excess over its floor (-H(f), or 0 for clean outcomes)."""
    summary = json.loads((run_dir / "train_summary.json").read_text())
    curve = {int(c["step"]): float(c["criterion"]) for c in summary["selection_curve"]}
    floor = -entropy(flip or 0.0)
    best, last = int(summary["best_step"]), max(curve)
    excess_best, excess_last = floor - curve[best], floor - curve[last]
    return {"best_step": best, "last_step": last, "floor": floor, "criterion_best": curve[best], "criterion_last": curve[last],
            "excess_best": excess_best, "excess_last": excess_last, "growth": excess_last - excess_best}


def inverted(run_dir: Path, flip: float, clean_dir: Path) -> dict[str, Any] | None:
    """invert_noisy.py's channel-scale and top-1 numbers, after checking the rate, the source results and the clean reference."""
    path = run_dir.parent / f"{run_dir.name}_inverted" / "metrics.json"
    if not path.is_file():
        return None
    m = json.loads(path.read_text())
    if not math.isclose(float(m["flip"]), flip):
        raise SystemExit(f"{path}: inverted at flip {m['flip']}, the run's rate is {flip:g}")
    if m["source"]["results_sha256"] != cv3.sha256(run_dir / "results.jsonl.gz"):
        raise SystemExit(f"{path}: written from other results than {run_dir}/results.jsonl.gz; rerun invert_noisy.py")
    if m["split"] != V3_TEST_FULL:
        raise SystemExit(f"{path}: split {m['split']}, expected {V3_TEST_FULL}")
    reference = m.get("clean_reference")
    if reference is None or Path(reference["run"]).name != clean_dir.name:
        raise SystemExit(f"{path}: needs --clean {cv3.shown(clean_dir)}")
    out: dict[str, Any] = {"source": cv3.shown(path), "inverted_top1_ece": m["inverted_top1_only"]["ece"], "inverted_top1_ece_ci": m["inverted_top1_only"]["ece_ci"]}
    channel, mapped = m.get("stored_channel"), reference.get("mapped_channel")
    if channel is None or mapped is None:  # written before invert_noisy.py reported the channel scale; rerun it to fill these
        out["channel_scale"] = None
        return out
    out["channel_scale"] = {
        "channel_ece": channel["ece"], "channel_ece_ci": channel["ece_ci"],
        "reference_ece": mapped["ece"], "reference_ece_ci": mapped["ece_ci"],
        "channel_ece_minus_reference": channel["ece"] - mapped["ece"],
        "mean_top1": channel["mean_confidence"], "channel_line": channel["mean_channel_outcome"],
        "mean_top1_minus_channel_line": channel["mean_confidence"] - channel["mean_channel_outcome"],
    }
    return out


def compare(runs_dir: Path, size: str, resamples: int) -> dict[str, Any]:
    zero_dir = runs_dir / f"v3_{size}_zeroshot"
    if not (zero_dir / "results.jsonl.gz").is_file():
        raise SystemExit(f"{zero_dir}: no results.jsonl.gz")
    zero = cv3.by_split(read_results(zero_dir / "results.jsonl.gz"))[V3_TEST_FULL]
    data_hashes = json.loads((zero_dir / "metrics.json").read_text()).get("data_files_sha256", {})

    wanted: dict[str, Path] = {}  # name in the table -> run directory
    for arm in ARMS:
        wanted[f"{arm}_clean"] = runs_dir / run_name(size, arm, None)
        for f in RATES:
            wanted[f"{arm}_f{f:g}"] = runs_dir / run_name(size, arm, f)
    for arm in LAST_ARMS:
        wanted[f"{arm}_clean_last"] = runs_dir / f"{run_name(size, arm, None)}_last"
        for f in RATES:
            wanted[f"{arm}_f{f:g}_last"] = runs_dir / f"{run_name(size, arm, f)}_last"
    present = {name: path for name, path in wanted.items() if (path / "results.jsonl.gz").is_file()}
    missing = sorted(cv3.shown(path) for name, path in wanted.items() if name not in present)

    members = {"zero_shot": zero}
    sources: dict[str, Any] = {"zero_shot": {"dir": cv3.shown(zero_dir)}}
    for name, path in present.items():
        metrics = json.loads((path / "metrics.json").read_text())
        for file, digest in metrics.get("data_files_sha256", {}).items():
            if file in data_hashes and data_hashes[file] != digest:
                raise SystemExit(f"{path}: {file} has sha256 {digest}, the zero-shot run read {data_hashes[file]}")
        results = cv3.by_split(read_results(path / "results.jsonl.gz")).get(V3_TEST_FULL)
        if results is None:
            raise SystemExit(f"{path}: no {V3_TEST_FULL} results")
        members[name] = cv3.aligned(zero, results, name)
        sources[name] = {"dir": cv3.shown(path), "git": metrics.get("git"), "results_sha256": cv3.sha256(path / "results.jsonl.gz")}
    split = cv3.Split(V3_TEST_FULL, members, resamples)

    def learner(arm: str, flip: float | None) -> dict[str, Any] | None:
        key = f"{arm}_clean" if flip is None else f"{arm}_f{flip:g}"
        if key not in present:
            return None
        clean_key, clean_dir = f"{arm}_clean", wanted[f"{arm}_clean"]
        out: dict[str, Any] = {"run": cv3.shown(present[key]), **split.point(key)}
        out["selection"] = selection(present[key], flip)
        if flip is not None:
            out["minus_clean"] = split.delta(key, clean_key) if clean_key in present else None
            out["channel"] = inverted(present[key], flip, clean_dir)
        if arm in LAST_ARMS:
            last = f"{key}_last"
            if last in present:
                out["last"] = {"run": cv3.shown(present[last]), **split.point(last), "minus_selected": split.delta(last, key)}
                if flip is not None:
                    out["last"]["channel"] = inverted(present[last], flip, clean_dir)
            else:
                out["last"] = None
        return out

    clean = {arm: learner(arm, None) for arm in ARMS}
    by_rate: dict[str, Any] = {}
    for f in RATES:
        row: dict[str, Any] = {"flip": f, "floor": -entropy(f), "amplification": 1 / (1 - 2 * f)}
        for arm in ARMS:
            row[arm] = learner(arm, f)
        keys = [f"direct_brier_f{f:g}", "direct_brier_clean", f"positive_sft_f{f:g}", "positive_sft_clean"]
        row["direct_brier_loss_minus_positive_sft_loss"] = split.delta_of_deltas(*keys) if all(k in present for k in keys) else None
        by_rate[f"{f:g}"] = row
    uncorrected: dict[str, Any] = {
        "note": ("ece is the uncorrected ECE: the stored top-1 probability against clean correctness (the gold label), before any inversion. "
                 "accuracy_change is noisy minus clean accuracy, paired; direct_brier_change_minus_positive_sft_change is direct_brier's change minus "
                 "positive_sft's, paired, positive when direct_brier loses less."),
        "rates": {},
    }
    for f, row in [("0", {arm: clean[arm] for arm in ARMS}), *by_rate.items()]:
        entry: dict[str, Any] = {}
        for arm in ARMS:
            r = row[arm]
            entry[arm] = None if r is None else {"accuracy": r["accuracy"], "accuracy_ci": r["accuracy_ci"], "ece": r["ece"], "ece_ci": r["ece_ci"],
                                                 "accuracy_change": (r.get("minus_clean") or {}).get("accuracy")}
        d = row.get("direct_brier_loss_minus_positive_sft_loss")
        entry["direct_brier_change_minus_positive_sft_change"] = d["accuracy"] if d else None
        uncorrected["rates"][f] = entry
    return {
        "size": size,
        "split": V3_TEST_FULL,
        "n": N,
        "git": git_state(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": cv3.sha256(Path(__file__)),
        "bootstrap": {"resamples": resamples, "by": "record", "paired": True},
        "note": ("Noisy runs train on outcomes flipped at rate f and select on the stored flipped outcomes. The flips are nested "
                 "across rates (one draw per interaction from the same stream), so the rates are paired. Channel-scale ECE scores "
                 "the stored top-1 against f + (1 - 2f) x correct; the reference is the clean run's top-1 mapped through "
                 "f + (1 - 2f) p. The selection criterion's excess is its floor minus its value, the floor being -H(f) for "
                 "flipped outcomes and 0 for clean ones."),
        "rates": list(RATES),
        "zero_shot": split.point("zero_shot"),
        "clean": clean,
        "by_rate": by_rate,
        "uncorrected": uncorrected,
        "sources": sources,
        "missing": missing,
    }


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="06b")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", default=None, help="default: runs/v3_noise_sweep_<size>/metrics.json")
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    result = compare(runs_dir, args.size, args.resamples)
    out = Path(args.out) if args.out else runs_dir / f"v3_noise_sweep_{args.size}" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"== noise sweep on {V3_TEST_FULL}, N {N}, seed 0")
    print(f"{'f':5s} {'learner':13s} {'acc':>6s} {'ece':>6s} {'acc-clean':>10s} {'ch.ece':>7s} {'ref':>6s} {'inv.top1':>8s} {'top1':>6s} {'line':>6s} {'last acc':>8s} {'growth':>7s}")
    for arm, c in result["clean"].items():
        if c:
            last = c.get("last") or {}
            print(f"{'0':5s} {arm:13s} {fmt(c['accuracy']):>6s} {fmt(c['ece']):>6s} {'':>10s} {'':>7s} {'':>6s} {'':>8s} {'':>6s} {'':>6s} {fmt(last.get('accuracy')):>8s} {fmt(c['selection']['growth'], True):>7s}")
    for f, row in result["by_rate"].items():
        for arm in ARMS:
            r = row[arm]
            if r is None:
                print(f"{f:5s} {arm:13s} missing")
                continue
            inv = r.get("channel") or {}
            ch = {**(inv.get("channel_scale") or {}), "inverted_top1_ece": inv.get("inverted_top1_ece")}
            last = r.get("last") or {}
            delta = (r.get("minus_clean") or {}).get("accuracy", {}).get("delta")
            print(f"{f:5s} {arm:13s} {fmt(r['accuracy']):>6s} {fmt(r['ece']):>6s} {fmt(delta, True):>10s} {fmt(ch.get('channel_ece')):>7s} {fmt(ch.get('reference_ece')):>6s} "
                  f"{fmt(ch.get('inverted_top1_ece')):>8s} {fmt(ch.get('mean_top1')):>6s} {fmt(ch.get('channel_line')):>6s} {fmt(last.get('accuracy')):>8s} {fmt(r['selection']['growth'], True):>7s}")
        d = row["direct_brier_loss_minus_positive_sft_loss"]
        if d:
            a = d["accuracy"]
            print(f"{f:5s} direct_brier loss minus positive_sft loss (accuracy): {a['delta']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]")
    print("\n== uncorrected ECE (stored top-1 against clean correctness) and the paired accuracy changes [95 percent intervals]")
    print("f | CAPS-Brier ECE | positive-only ECE | CAPS-Brier change | positive-only change | CAPS minus positive-only change")
    ci = lambda v: f"{v['ece']:.3f} [{v['ece_ci'][0]:.3f}, {v['ece_ci'][1]:.3f}]"
    dc = lambda d: "-" if d is None else f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
    for f, e in result["uncorrected"]["rates"].items():
        db, ps = e["direct_brier"], e["positive_sft"]
        if db is None or ps is None:
            print(f"{f} | missing")
            continue
        print(f"{f} | {ci(db)} | {ci(ps)} | {dc(db['accuracy_change'])} | {dc(ps['accuracy_change'])} | {dc(e['direct_brier_change_minus_positive_sft_change'])}")
    if result["missing"]:
        print("missing:", ", ".join(result["missing"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
