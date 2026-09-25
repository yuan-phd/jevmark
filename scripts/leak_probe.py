"""Leak probes: can a model that never reads the state predict the gold answers? (docs/DATA.md section 8)

    uv run python scripts/leak_probe.py [--config configs/data.yaml] [--data-dir data] [--splits ...]

For every split and every question, two probe models predict the gold answer from
state-free features only, each with 5-fold cross-validation: logistic regression,
and a histogram gradient-boosted tree ensemble (HistGradientBoostingClassifier) that
can represent interactions between features, such as a flag that matters only for
some K, which a linear probe cannot. Four feature sets (probes) per model:

- phrasing: the question's template and polarity (noul), or its instructions;
- question text: bag of words of the question block, options included;
- structure: number of questions seen, position, kinds seen, K, asked-in-options
  flags, scale, state format;
- cross-question: the texts and structure of the other questions seen, and
  relations between them (an asked intent's domain equals an asked domain).

Attention is causal, so a question only sees the questions before it: every
feature is computed on the question and the questions that precede it, never on
later ones. For noul questions the phrasing probe predicts the stored gold, and
the other three predict the underlying answer (the stored gold with negation
undone), because negation flips the stored gold in half of every group, which a
linear probe cannot undo.

Targets per group: noul questions by kind; choice questions by question id as
gold-is-other (when other can be gold), gold label (when every question of the
group offers the same labels) and gold position (without the question text probe:
a bag of words holds no position information); score questions by scale.

The gate (decision 42) applies to each model's probes alike: a probe that beats its group's majority baseline by more
than hard_lift (10 points) fails unconditionally; every real leak found so far was
above 20 points. A probe above max_lift (3 points) is rerun on `permutations` (200)
copies of its targets shuffled at random, and fails if its lift is also above the
99th percentile of those runs. With about 250 probe results per build, chance alone
lifts a few small groups past 3 points and past a 95th percentile, so the summary
lists every probe above 3 points, passing or not, with n, lift and p value (none
above the hard limit, where no permutation test is run).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from threadpoolctl import threadpool_limits

from jevmark.config import load_config
from jevmark.data.build import SPLITS
from jevmark.data.clinc import load_domains
from jevmark.data.negation import KINDS, parse

REPO = Path(__file__).resolve().parents[1]
PROBES = ("phrasing", "question_text", "structure", "cross_question")
OTHER = "other"


@dataclass
class Row:
    group: str
    stored: Any  # the stored gold answer
    underlying: Any  # noul: the answer with negation undone; otherwise the same as stored
    features: dict[str, dict[str, float]] = field(default_factory=dict)  # probe -> sparse features


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def question_text(question: Mapping[str, Any]) -> str:
    """The question block as the model reads it, without letters: instructions and one line per option."""
    lines = [question["instructions"]]
    if question["type"] == "choice":
        lines += [f"{label}: {desc}" if desc else label for label, desc in question["criteria"].items()]
    elif question["type"] == "score":
        lines += [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
    return "\n".join(lines)


def kind_of(qid: str, question: Mapping[str, Any]) -> str:
    return qid if question["type"] == "noul" else question["type"]


def phrasing(qid: str, question: Mapping[str, Any]) -> str:
    if question["type"] == "noul" and qid in KINDS:
        p = parse(qid, question["instructions"])
        return f"t{p.template}{'-neg' if p.negated else ''}"
    return question["instructions"]


def negated(qid: str, question: Mapping[str, Any]) -> bool:
    return question["type"] == "noul" and qid in KINDS and parse(qid, question["instructions"]).negated


class Features:
    """State-free features for every question of a record, from the question and the ones before it."""

    def __init__(self) -> None:
        self.domain_of = {intent: domain for domain, members in load_domains().items() for intent in members}

    def asked(self, record: Mapping[str, Any], qid: str) -> dict[str, str]:
        """What a noul asks about (intent, domain, emotion), as its text states it; meta holds the parsed values."""
        info = record["meta"].get("nouls", {}).get(qid, {})
        return {k: info[k] for k in ("asked_intent", "asked_domain", "asked_emotion") if info.get(k)}

    def flags(self, record: Mapping[str, Any], visible: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, float]:
        """Relations among the visible questions: asked items among the choice options, and intent-domain agreement."""
        labels = [label for _, q in visible if q["type"] == "choice" for label in q["criteria"] if label != OTHER]
        option_domains = {self.domain_of.get(label) for label in labels} - {None}
        out: dict[str, float] = {}
        asked_intents, asked_domains = [], []
        for qid, q in visible:
            asked = self.asked(record, qid)
            if labels and "asked_intent" in asked:
                out[f"{qid}:asked_in_options={asked['asked_intent'] in labels}"] = 1
            if labels and "asked_domain" in asked:
                out[f"{qid}:asked_domain_in_options={asked['asked_domain'] in option_domains}"] = 1
            if labels and "asked_emotion" in asked:
                out[f"{qid}:asked_in_options={asked['asked_emotion'] in labels}"] = 1
            if "asked_intent" in asked:
                asked_intents.append(asked["asked_intent"])
            if "asked_domain" in asked:
                asked_domains.append(asked["asked_domain"])
        if asked_intents and asked_domains:
            out[f"intent_domain_match={self.domain_of.get(asked_intents[0]) in asked_domains}"] = 1
        return out

    def rows(self, record: Mapping[str, Any]) -> list[Row]:
        items = list(record["questions"].items())
        meta = record["meta"]
        state_format = meta.get("state_format", "plain")
        n_fields = len(record["state"]) - 1 if isinstance(record["state"], dict) else 0
        rows = []
        for i, (qid, question) in enumerate(items):
            before, visible = items[:i], items[: i + 1]
            gold = record["gold"][qid]
            for group, stored in self.targets(qid, question, gold, meta):
                underlying = stored
                if question["type"] == "noul" and negated(qid, question):
                    underlying = {"true": "false", "false": "true"}[stored]
                k = len(question["criteria"]) if question["type"] != "noul" else 2
                structure = {
                    "bias": 1.0,
                    f"position={i}": 1.0,
                    f"seen_questions={i + 1}": 1.0,
                    f"K={k}": 1.0,
                    f"format={state_format}": 1.0,
                    f"fields={n_fields}": 1.0,
                    **{f"seen_kind={kind_of(j, q)}": 1.0 for j, q in before},
                    **self.flags(record, visible),
                }
                if question["type"] == "score":
                    structure[f"scale={meta.get('scale', k)}"] = 1.0
                cross = {"bias": 1.0, **self.flags(record, visible)}
                for j, q in before:
                    cross[f"other_kind={kind_of(j, q)}"] = 1.0
                    cross[f"other_K={len(q['criteria']) if q['type'] != 'noul' else 2}"] = 1.0
                    for t in tokens(question_text(q)):
                        cross[f"other:{t}"] = 1.0
                rows.append(
                    Row(
                        group,
                        stored,
                        underlying,
                        {
                            "phrasing": {"bias": 1.0, f"phrasing={phrasing(qid, question)}": 1.0},
                            "question_text": {"bias": 1.0, **{f"w:{t}": 1.0 for t in tokens(question_text(question))}},
                            "structure": structure,
                            "cross_question": cross,
                        },
                    )
                )
        return rows

    @staticmethod
    def targets(qid: str, question: Mapping[str, Any], gold: Any, meta: Mapping[str, Any]) -> list[tuple[str, Any]]:
        if question["type"] == "noul":
            return [(f"noul/{qid}", gold)]
        if question["type"] == "score":
            return [(f"score/{qid}/{meta.get('scale', len(question['criteria']))}", gold)]
        labels = list(question["criteria"])
        out = [(f"choice/{qid}/gold_position", labels.index(gold)), (f"choice/{qid}/gold_label", gold)]
        if OTHER in labels:
            out.append((f"choice/{qid}/gold_other", gold == OTHER))
        return out


def majority(targets: Sequence[Any]) -> float:
    return Counter(targets).most_common(1)[0][1] / len(targets)


MODELS = ("logistic", "boosting")
BOOSTING_MAX_FEATURES = 256


def probe_accuracy(features: Sequence[Mapping[str, float]], targets: Sequence[Any], folds: int, seed: int = 0, model: str = "logistic") -> float:
    """Mean cross-validated accuracy of a probe model on the given features.

    logistic: logistic regression on all features (sparse). boosting: a histogram
    gradient-boosted tree ensemble, which can represent interactions between
    features that a linear probe cannot; it needs dense input, so it sees the
    BOOSTING_MAX_FEATURES most frequent columns of the group (chosen without the
    targets), which keeps every column of the phrasing and structure probes and the
    most common words of the bag-of-words probes.
    """
    x = DictVectorizer().fit_transform(features)
    y = [str(t) for t in targets]
    if model == "boosting":
        frequency = np.asarray((x != 0).sum(axis=0)).ravel()
        keep = np.sort(np.argsort(-frequency, kind="stable")[:BOOSTING_MAX_FEATURES])
        x = x[:, keep].toarray().astype(np.float32)
        estimator = HistGradientBoostingClassifier(random_state=seed)
    else:
        estimator = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)  # classes with fewer members than folds
        # One thread: on these small dense matrices the default thread pool oversubscribes the CPU and
        # runs about three times slower (measured on the 14-class train gold_position group); results are identical.
        with threadpool_limits(limits=1):
            predicted = cross_val_predict(estimator, x, y, cv=KFold(folds, shuffle=True, random_state=seed))
    return sum(p == t for p, t in zip(predicted, y)) / len(y)


def usable_groups(rows: Sequence[Row], min_n: int) -> dict[str, list[Row]]:
    """Groups with at least min_n questions and two target values; gold_label only where every question offers the same labels."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.group].append(row)
    usable = {}
    for group, members in sorted(groups.items()):
        if len(members) < min_n or len({r.underlying for r in members}) < 2:
            continue
        usable[group] = members
    return usable


def permutation_null(features: Sequence[Mapping[str, float]], targets: Sequence[Any], folds: int, permutations: int, model: str = "logistic") -> list[float]:
    """The probe's lifts over the majority baseline on shuffled targets (seeds 0 to permutations - 1), sorted.

    Shuffling keeps the class counts, so the baseline is unchanged, and breaks any link
    between features and targets: the result is how large a lift chance alone gives
    this probe on this group.
    """
    baseline = majority(targets)
    lifts = []
    for seed in range(permutations):
        shuffled = list(targets)
        random.Random(seed).shuffle(shuffled)
        lifts.append(probe_accuracy(features, shuffled, folds, model=model) - baseline)
    return sorted(lifts)


def gate(lift: float, null: Sequence[float], max_lift: float, hard_lift: float, percentile: float) -> dict[str, Any]:
    """The gate for one probe with lift above max_lift (decision 42).

    p_value is (1 + number of shuffled lifts at least as large) / (1 + number of shuffles).
    Above hard_lift the probe fails unconditionally; otherwise it fails when its lift is
    above the given percentile of the shuffled lifts.
    """
    threshold = null[min(len(null) - 1, math.ceil(percentile * len(null)) - 1)]
    p_value = (1 + sum(n >= lift for n in null)) / (1 + len(null))
    failed = lift > hard_lift or lift > threshold
    reason = "above the hard limit" if lift > hard_lift else ("above chance" if failed else "chance")
    return {"null_percentile": threshold, "p_value": p_value, "failed": failed, "reason": reason}


def probe_split(
    records: Iterable[Mapping[str, Any]],
    folds: int,
    min_n: int,
    max_lift: float = 0.03,
    permutations: int = 200,
    hard_lift: float = 0.10,
    percentile: float = 0.99,
) -> dict[str, dict[str, Any]]:
    """Group -> n, majority baselines and each probe's accuracy and lift over its baseline.

    A probe whose lift is above max_lift also gets a "gate" entry: above hard_lift it
    fails without a permutation test (p_value None), otherwise see gate().
    """
    features = Features()
    records = list(records)
    rows = [row for record in records for row in features.rows(record)]
    # gold_label is meaningful only when the option set is the same for every question of the group.
    label_sets: dict[str, set] = defaultdict(set)
    for record in records:
        for qid, q in record["questions"].items():
            if q["type"] == "choice":
                label_sets[f"choice/{qid}/gold_label"].add(tuple(sorted(q["criteria"])))
    rows = [r for r in rows if not r.group.endswith("/gold_label") or len(label_sets[r.group]) == 1]
    report = {}
    for group, members in usable_groups(rows, min_n).items():
        stored = [r.stored for r in members]
        underlying = [r.underlying for r in members]
        entry: dict[str, Any] = {"n": len(members), "majority_stored": majority(stored), "majority_underlying": majority(underlying)}
        for model in MODELS:
            results: dict[str, Any] = {}
            for probe in PROBES:
                if probe == "question_text" and group.endswith("/gold_position"):
                    # A bag of words has no position information, so it cannot predict a position; skipping it saves most of the run time.
                    results[probe] = {"accuracy": None, "lift": None}
                    continue
                targets, baseline = (stored, entry["majority_stored"]) if probe == "phrasing" else (underlying, entry["majority_underlying"])
                accuracy = probe_accuracy([r.features[probe] for r in members], targets, folds, model=model)
                lift = accuracy - baseline
                results[probe] = {"accuracy": accuracy, "lift": lift}
                if lift > hard_lift:
                    results[probe]["gate"] = {"null_percentile": None, "p_value": None, "failed": True, "reason": "above the hard limit"}
                elif lift > max_lift:
                    null = permutation_null([r.features[probe] for r in members], targets, folds, permutations, model)
                    results[probe]["gate"] = gate(lift, null, max_lift, hard_lift, percentile)
            entry[model] = results
        entry["max_lift"] = max(entry[m][p]["lift"] for m in MODELS for p in PROBES if entry[m][p]["lift"] is not None)
        report[group] = entry
    return report


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_table(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    heads = " ".join(f"{p[:9]:>9}" for p in ("phrasing", "q_text", "structure", "cross"))
    print(f"{'split':20} {'group':34} {'n':>6} {'maj':>6}  logistic: {heads}  |  boosting: {heads}")
    for split, groups in results.items():
        for group, e in groups.items():
            cells = {m: " ".join(f"{100 * e[m][p]['lift']:+9.1f}" if e[m][p]["lift"] is not None else f"{'n/a':>9}" for p in PROBES) for m in MODELS}
            print(f"{split:20} {group:34} {e['n']:6} {100 * e['majority_underlying']:5.1f}%            {cells['logistic']}  |            {cells['boosting']}")
    flagged = [(split, group, m, p, e) for split, groups in results.items() for group, e in groups.items() for m in MODELS for p in PROBES if "gate" in e[m][p]]
    tests = sum(1 for groups in results.values() for e in groups.values() for m in MODELS for p in PROBES if e[m][p]["lift"] is not None)
    print(f"\n== every probe more than 3 points over its baseline ({len(flagged)} of {tests} probe results); chance alone is expected to give a few")
    failures = []
    for split, group, m, p, e in flagged:
        g = e[m][p]["gate"]
        chance = "p   n/a  (no permutation test above the hard limit)" if g["p_value"] is None else f"p {g['p_value']:.3f}  null 99th pct {100 * g['null_percentile']:+5.1f}"
        print(f"{split:20} {group:34} {m:8} {p:15} n {e['n']:6}  lift {100 * e[m][p]['lift']:+5.1f}  {chance}  {'FAIL' if g['failed'] else 'pass'} ({g['reason']})")
        if g["failed"]:
            failures.append(f"{split}/{group}: {m} {p} probe +{100 * e[m][p]['lift']:.1f} points ({g['reason']})")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "data.yaml"))
    parser.add_argument("--data-dir", default=None, help="default: out_dir from the config")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--permutations", type=int, default=None, help="shuffled-target runs per flagged probe; default from the config (200, the gate). Lower only for demonstrations")
    parser.add_argument("--out", default=None, help="write the full results as JSON here (default: <data-dir>/leak_probe.json)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    params = config["leak_probe"]
    data_dir = Path(args.data_dir) if args.data_dir else REPO / config["out_dir"]
    max_lift = float(params["max_lift"])
    print(
        f"== leak probes: logistic regression and gradient-boosted trees, {params['folds']}-fold cross-validation, state-free features; "
        f"fail above +{100 * float(params['hard_lift']):.0f} points, or above +{100 * max_lift:.0f} points and the "
        f"{100 * float(params['null_percentile']):.0f}th percentile of {args.permutations or params['permutations']} shuffled-target runs"
    )
    results = {}
    for split in args.splits:
        results[split] = probe_split(
            read_split(data_dir, split),
            int(params["folds"]),
            int(params["min_n"]),
            max_lift,
            args.permutations or int(params["permutations"]),
            float(params["hard_lift"]),
            float(params["null_percentile"]),
        )
    failures = print_table(results)
    out = Path(args.out) if args.out else data_dir / "leak_probe.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    worst = max(((s, g, e["max_lift"]) for s, gs in results.items() for g, e in gs.items()), key=lambda t: t[2])
    print(f"\nlargest lift: {100 * worst[2]:+.1f} points ({worst[0]}/{worst[1]}); wrote {out}")
    if failures:
        print("\nFAILED: leak probes\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("leak probes passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
