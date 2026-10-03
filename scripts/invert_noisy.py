"""Known-flip inversion of a noisy v3 run's probabilities (decision 57), on CPU from its results.jsonl.gz.

    uv run python scripts/invert_noisy.py runs/v3_06b_direct_brier_n5000_s0_noisy [--flip 0.2] [--clean runs/<clean run>]

A run trained on outcomes flipped with probability f learns, for a proper score, the
probability that the flipped outcome is 1: (1 - f) for a correct action and f for a
wrong one, so p = f + (1 - 2f) p_correct. The inversion p_correct = (p - f) / (1 - 2f),
clipped to [0, 1] and renormalised over the question's options, maps it back to the
probability of being correct. A question whose every option clips to 0 keeps its
stored distribution, and the count is recorded. The argmax never changes otherwise,
since the map is increasing in p.

Writes runs/<run>_inverted/metrics.json with the stored and the inverted metrics on
the split (accuracy, ECE, Brier, NLL with bootstrap intervals by record, coverage at
0.80, 0.90 and 0.95), the ECE of the inverted top-1 probability alone (no clipping of
the other options, no renormalisation: the direct test of calibration to the channel,
since renormalising after clipping moves the mass of clipped options onto the rest),
the number of questions whose gold probability clipped to 0
(their NLL is the floor, -log 1e-12), and, with --clean, the clean run's stored
metrics on the same split for reference. The source run is never modified.

Channel scale (decision 59): the stored run's top-1 probability against the channel
outcome of its top option, f + (1 - 2f) x correct, with the same 15 bins as ECE, and
with --clean the mapped-clean reference: the clean run's top-1 passed through
f + (1 - 2f) p and scored against the same channel outcome, which is what a
channel-calibrated copy of the clean run would score. Both carry bootstrap intervals
by record. No clipping or renormalisation is involved, so the two are on one scale.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jevmark.data.build import V3_TEST_FULL  # noqa: E402
from jevmark.metrics import QuestionResult, bootstrap_intervals, coverage_curve, ece, read_results, summarize  # noqa: E402
from jevmark.provenance import git_state  # noqa: E402

COVERAGE_AT = (0.80, 0.90, 0.95)


def invert_probs(probs: Sequence[float], flip: float) -> tuple[tuple[float, ...], bool]:
    """(p - flip) / (1 - 2 flip) per option, clipped to [0, 1] and renormalised; the stored distribution and False if all clip to 0."""
    if not 0 <= flip < 0.5:
        raise ValueError(f"flip must be in [0, 0.5), got {flip}")
    q = [min(1.0, max(0.0, (p - flip) / (1 - 2 * flip))) for p in probs]
    total = sum(q)
    if total <= 0:
        return tuple(probs), False
    return tuple(x / total for x in q), True


def invert_results(results: Sequence[QuestionResult], flip: float) -> tuple[list[QuestionResult], int]:
    out, kept = [], 0
    for r in results:
        probs, ok = invert_probs(r.probs, flip)
        kept += not ok
        out.append(replace(r, probs=probs, shuffled_probs=None))
    return out, kept


def channel_targets(results: Sequence[QuestionResult], flip: float) -> list[float]:
    """The expected flipped outcome of each question's top option: f + (1 - 2f) x correct."""
    return [flip + (1 - 2 * flip) * float(r.correct) for r in results]


def channel_ece(confidences: Sequence[float], results: Sequence[QuestionResult], flip: float) -> dict[str, Any]:
    """ECE of the confidences against the channel outcome, same bins as ECE, with its bootstrap interval by record."""
    targets = channel_targets(results, flip)
    ci = bootstrap_intervals([r.record_id for r in results], targets, list(confidences))
    return {"n": len(results), "ece": ece(confidences, targets), "ece_ci": ci["ece_ci"],
            "mean_confidence": sum(confidences) / len(confidences), "mean_channel_outcome": sum(targets) / len(targets)}


def stored_channel(results: Sequence[QuestionResult], flip: float) -> dict[str, Any]:
    """The run's own top-1 on the channel scale."""
    return channel_ece([r.top1 for r in results], results, flip)


def mapped_channel(results: Sequence[QuestionResult], flip: float) -> dict[str, Any]:
    """A clean run's top-1 mapped onto the channel scale, f + (1 - 2f) p: the reference for a channel-calibrated learner."""
    return channel_ece([flip + (1 - 2 * flip) * r.top1 for r in results], results, flip)


def top1_only(results: Sequence[QuestionResult], flip: float) -> dict[str, Any]:
    """ECE of the inverted top-1 probability alone, (top1 - flip) / (1 - 2 flip) clipped to [0, 1], without renormalising."""
    conf = [min(1.0, max(0.0, (r.top1 - flip) / (1 - 2 * flip))) for r in results]
    correct = [r.correct for r in results]
    ci = bootstrap_intervals([r.record_id for r in results], correct, conf)
    return {"n": len(results), "ece": ece(conf, correct), "ece_ci": ci["ece_ci"], "mean_confidence": sum(conf) / len(conf)}


def block(results: Sequence[QuestionResult]) -> dict[str, Any]:
    s = summarize(results, with_coverage=False)
    curve = {f"{row['threshold']:.2f}": {k: row[k] for k in ("coverage", "n", "accuracy")} for row in coverage_curve(results, COVERAGE_AT)}
    return {k: s[k] for k in ("n", "accuracy", "accuracy_ci", "ece", "ece_ci", "brier", "nll")} | {
        "mean_top1": sum(r.top1 for r in results) / len(results),
        "coverage": curve,
    }


def run(source: Path, flip: float, split: str, clean: Path | None, out: Path) -> dict[str, Any]:
    stored = [r for r in read_results(source / "results.jsonl.gz") if r.split == split]
    if not stored:
        raise SystemExit(f"{source}: no {split} results")
    inverted, unchanged = invert_results(stored, flip)
    result: dict[str, Any] = {
        "run_name": out.name,
        "decision": 57,
        "source": {"run": str(source.relative_to(REPO) if source.is_relative_to(REPO) else source), "git": json.loads((source / "metrics.json").read_text()).get("git"),
                   "results_sha256": hashlib.sha256((source / "results.jsonl.gz").read_bytes()).hexdigest()},
        "git": git_state(),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "flip": flip,
        "inversion": "p_correct = (p - flip) / (1 - 2 flip), clipped to [0, 1], renormalised over the options",
        "questions_left_unchanged_all_options_clipped": unchanged,
        "questions_gold_clipped_to_zero": sum(r.probs[r.gold] == 0 for r in inverted),
        "stored": block(stored),
        "inverted": block(inverted),
        "inverted_top1_only": top1_only(stored, flip),
        "stored_channel": stored_channel(stored, flip),
    }
    if clean is not None:
        clean_results = [r for r in read_results(clean / "results.jsonl.gz") if r.split == split]
        result["clean_reference"] = {"run": str(clean.relative_to(REPO) if clean.is_relative_to(REPO) else clean), **block(clean_results),
                                     "mapped_channel": mapped_channel(clean_results, flip)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def fmt_accuracy(accuracy: float | None) -> str:
    return "-" if accuracy is None else f"{accuracy:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--flip", type=float, default=0.2)
    parser.add_argument("--split", default=V3_TEST_FULL)
    parser.add_argument("--clean", default=None, help="the clean counterpart, reported on the same split for reference")
    parser.add_argument("--out", default=None, help="default runs/<run>_inverted next to the source")
    args = parser.parse_args(argv)
    source = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else source.parent / f"{source.name}_inverted"
    r = run(source, args.flip, args.split, Path(args.clean).resolve() if args.clean else None, out)
    s, i = r["stored"], r["inverted"]
    print(f"{out.name}: flip {args.flip}, {s['n']} questions; unchanged {r['questions_left_unchanged_all_options_clipped']}, gold clipped to 0 {r['questions_gold_clipped_to_zero']}")
    for name, b in (("stored", s), ("inverted", i), *((("clean", r["clean_reference"]),) if "clean_reference" in r else ())):
        cov = "  ".join(f"{t} {c['coverage']:.3f} ({fmt_accuracy(c['accuracy'])})" for t, c in b["coverage"].items())
        print(f"  {name:8s} acc {b['accuracy']:.3f} ece {b['ece']:.3f} [{b['ece_ci'][0]:.3f}, {b['ece_ci'][1]:.3f}] brier {b['brier']:.3f} nll {b['nll']:.3f} top1 {b['mean_top1']:.3f}  {cov}")
    top = r["inverted_top1_only"]
    print(f"  top-1 only: ece {top['ece']:.3f} [{top['ece_ci'][0]:.3f}, {top['ece_ci'][1]:.3f}] mean confidence {top['mean_confidence']:.3f}")
    ch = r["stored_channel"]
    print(f"  channel scale: stored ece {ch['ece']:.3f} [{ch['ece_ci'][0]:.3f}, {ch['ece_ci'][1]:.3f}], mean top-1 {ch['mean_confidence']:.3f} against channel line {ch['mean_channel_outcome']:.3f}")
    if "clean_reference" in r:
        ref = r["clean_reference"]["mapped_channel"]
        print(f"  channel scale: mapped-clean reference ece {ref['ece']:.3f} [{ref['ece_ci'][0]:.3f}, {ref['ece_ci'][1]:.3f}]")
    print(f"wrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
