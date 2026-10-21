"""Review additions to runs/v3_stage_06b/metrics.json, CPU only, from stored files (review items 3 and 5).

    uv run python scripts/v3_review.py off-policy
    uv run python scripts/v3_review.py forgetting

Each subcommand writes one top-level key (or two) into the existing
runs/v3_stage_06b/metrics.json, which scripts/compare_v3.py writes and keeps these
keys of when it rewrites the file. The rest of the file is not touched.

off-policy writes `off_policy`:

- identity: outcome_minus_p_log's loss is -(R_k - b) w_k log p(a_k) on a micro-batch
  of B logged interactions, with R = r - p(a) and w = p(a) / q(a), both detached, and b
  the micro-batch mean of R. For actions logged from q and outcomes r_a fixed given the
  action, while no weight is clipped, its expected gradient is
  -(1 - 1/B) sum_a (r_a - p_a) grad p_a per question: E_q[w R grad log p] =
  sum_a p_a R(a) grad log p_a = sum_a (r_a - p_a) grad p_a, the baseline's other
  samples drop out since sum_a grad p_a = 0, and its own sample removes 1/B. The
  IPS-weighted CAPS-Brier loss (1 / q(a)) (r - p(a))^2 has expected gradient
  sum_a grad (r_a - p_a)^2 = -2 sum_a (r_a - p_a) grad p_a, the gradient of the
  full-information Brier score. So the two expected gradients are equal up to the
  constant (1 - 1/B) / 2 (7/16 at the training micro-batch of 8). The check runs
  train_rlcd.log_batch_loss on a tiny randomly initialised Qwen3 with a LoRA adapter
  (B 2, K 3 and 4, q = (1 - epsilon) p + epsilon / K at the current p), enumerates
  every logged action pair with its probability under q, and compares the LoRA
  gradients of the two expectations.
- clipping: the clip at 5 cannot bind at the logging policy (p_log / q <= 1 / (1 -
  epsilon), checked on the first 5000 interactions of runs/v3_log_s0/log.jsonl), but
  outcome_minus_p_log's weight uses the learner's current p, which moves away from
  p_log during training, so its clipped share is read from train_summary.json (every
  training example of every step) and from the step lines of training_log.jsonl up to
  the selected step. For ips_log and ips_positive (weight min(1 / q, 5), normalised)
  the number and share of clipped weights over the 0.9 N training prefix and over the
  interactions each arm trains on, from train_summary.json.

forgetting writes `forgetting_table` and `selected_steps`:

- forgetting_table: on test_indomain and test_unseen_intents (every gold-dependent
  question), every learner compare_v3.py reads (the zero-shot run, the temperature
  learners, every trained run at every N, seed and log, and direct_brier plus its
  temperature), with accuracy and ECE, each with its 95 percent record bootstrap
  interval, and the paired difference from the zero-shot run. The draws are
  compare_v3.Split's (seed text "v3:<split>"), so the zero-shot values and the
  differences equal the file's `forgetting` block, which is checked.
- selected_steps: every Banking77 0.6B log-mode run with a train_summary.json (clean,
  noisy at every flip rate, soft, the seed 1 log): the selected step (best selection
  criterion), the last step, the interactions it trained on (n_used: every training
  interaction, or the outcome-1 ones for positive_sft and ips_positive), the steps an
  epoch takes (ceil(ceil(n_used / micro_batch) / accumulation)) and the epochs the
  selected and the last step imply.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from jevmark.feedback import EPSILON, read_log
from jevmark.provenance import git_state

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "runs" / "v3_stage_06b" / "metrics.json"
LOG = REPO / "runs" / "v3_log_s0" / "log.jsonl"
OFF_POLICY_NS = (2000, 5000)
MICRO_BATCH = 8  # configs/v3_06b.yaml training.micro_batch
IDENTITY_TOL = 1e-5  # relative, fp32 gradients


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


# Off-policy: the identity on a tiny model


def tiny_jevmark() -> Any:
    """A randomly initialised two-layer Qwen3 (the test configuration) with a random LoRA adapter, fp32 on CPU."""
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from jevmark.model import JevMark

    reference = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())["tokenizer_reference"]
    tokenizer = AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True, pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=False)
    model = get_peft_model(model, lora)
    model.eval()  # no dropout; gradients still flow
    return JevMark(model, tokenizer, max_tokens=512)


def identity_check(rl: Any) -> dict[str, Any]:
    from jevmark.encode import encode
    from jevmark.schema import Request

    jev = tiny_jevmark()
    questions = [
        ("I am still waiting on my card?", {"card_arrival": "When a new card will arrive", "pin_blocked": "The card PIN is blocked", "other": "None of the listed options"}, "card_arrival"),
        ("Why was my transfer declined?", {"declined_transfer": "A transfer was declined", "card_arrival": "When a new card will arrive", "exchange_rate": "The exchange rate applied", "other": "None of the listed options"}, "declined_transfer"),
    ]
    encoded = [encode(Request.from_dict({"state": s, "questions": {"label": {"type": "choice", "instructions": "Which banking request does this message make?", "criteria": c}}}), jev.tokenizer, jev.max_tokens) for s, c, _ in questions]
    gold = [list(c).index(g) for _, c, g in questions]
    trainable = [p for p in jev.model.parameters() if p.requires_grad]
    with torch.no_grad():
        p = [torch.softmax(z, -1).double() for z in jev.slot_logits(encoded)]
    q = [(1.0 - EPSILON) * pk + EPSILON / pk.shape[0] for pk in p]
    batch_size = len(encoded)

    def gradient(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, trainable)
        return torch.cat([g.reshape(-1) for g in grads]).double()

    # E_q of outcome_minus_p_log's micro-batch loss gradient, every logged action pair enumerated.
    expected_reinforce = torch.zeros(sum(t.numel() for t in trainable), dtype=torch.float64)
    max_ratio = 0.0
    for actions in itertools.product(*[range(pk.shape[0]) for pk in p]):
        batch = [rl.LogExample(e, a, float(a == g), None, float(qk[a])) for e, a, g, qk in zip(encoded, actions, gold, q)]
        loss, stats = rl.log_batch_loss(jev, batch, "outcome_minus_p_log")
        if stats["weight_clipped_share"] > 0:
            raise SystemExit("a weight was clipped in the identity check; the identity holds only unclipped")
        max_ratio = max(max_ratio, *(float(pk[a] / qk[a]) for pk, qk, a in zip(p, q, actions)))
        probability = float(torch.prod(torch.tensor([float(qk[a]) for qk, a in zip(q, actions)])))
        expected_reinforce += probability * gradient(loss)
    # E_q of the IPS-weighted CAPS-Brier loss (1 / q(a)) (r - p(a))^2, from train_rlcd's direct_brier loss on single examples.
    expected_ips_brier = torch.zeros_like(expected_reinforce)
    for e, g, qk in zip(encoded, gold, q):
        for a in range(qk.shape[0]):
            loss, _ = rl.log_batch_loss(jev, [rl.LogExample(e, a, float(a == g), None, float(qk[a]))], "direct_brier")
            expected_ips_brier += float(qk[a]) * gradient(loss / float(qk[a])) / batch_size
    # The full-information Brier score sum_a (r_a - p_a)^2, averaged over the questions.
    logits = jev.slot_logits(encoded)
    full = sum(((torch.softmax(z, -1) - torch.nn.functional.one_hot(torch.tensor(g), z.shape[0]).float()) ** 2).sum() for z, g in zip(logits, gold)) / batch_size
    full_brier = gradient(full)
    constant = (1.0 - 1.0 / batch_size) / 2.0
    scale = float(expected_ips_brier.abs().max())
    return {
        "setting": {"model": "tiny random Qwen3 (2 layers, hidden 64) with a random LoRA adapter, fp32, CPU", "micro_batch": batch_size, "k": [pk.shape[0] for pk in p],
                    "behaviour": "q = (1 - epsilon) p + epsilon / K at the current p", "epsilon": EPSILON, "action_pairs_enumerated": int(torch.prod(torch.tensor([pk.shape[0] for pk in p]))),
                    "gradient_entries": int(expected_reinforce.numel())},
        "max_weight_p_over_q": max_ratio,
        "constant": constant,
        "constant_formula": "(1 - 1/B) / 2",
        "max_abs_ips_brier_gradient": scale,
        "max_rel_difference_reinforce_vs_constant_times_ips_brier": float((expected_reinforce - constant * expected_ips_brier).abs().max()) / scale,
        "max_rel_difference_ips_brier_vs_full_information_brier": float((expected_ips_brier - full_brier).abs().max()) / scale,
        "cosine_reinforce_ips_brier": float(torch.nn.functional.cosine_similarity(expected_reinforce, expected_ips_brier, dim=0)),
        "tolerance": IDENTITY_TOL,
        "matches": float((expected_reinforce - constant * expected_ips_brier).abs().max()) / scale < IDENTITY_TOL and float((expected_ips_brier - full_brier).abs().max()) / scale < IDENTITY_TOL,
    }


# Off-policy: clipping


def logging_policy_ratio(n: int) -> dict[str, Any]:
    """max p_log(a) / q(a) over the first n logged interactions: the weight at the start of training, when p is p_log."""
    log = read_log(LOG)[:n]
    ratios = [i.probs[i.action] / sum(i.probs) / i.propensity for i in log]
    return {"n": len(ratios), "max_p_log_over_q": max(ratios), "bound_1_over_1_minus_epsilon": 1.0 / (1.0 - EPSILON), "clipped": sum(r > 5.0 for r in ratios)}


def step_lines(run: Path) -> list[dict[str, Any]]:
    return [e for e in (json.loads(line) for line in (run / "training_log.jsonl").read_text().splitlines() if line.strip()) if "event" not in e]


def clipping(runs_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"logging_policy": logging_policy_ratio(max(OFF_POLICY_NS)), "outcome_minus_p_log": {}, "ips_log": {}, "ips_positive": {}}
    for n in OFF_POLICY_NS:
        run = runs_dir / f"v3_06b_outcome_minus_p_log_n{n}_s0"
        summary = json.loads((run / "train_summary.json").read_text())
        w = summary["weights"]
        steps = step_lines(run)
        best = summary["best_step"]
        upto = [s for s in steps if s["step"] <= best]
        # Step lines hold the mean over the step's micro-batches; times the step's examples it is the count when the micro-batches are full.
        per_step = 4 * MICRO_BATCH
        out["outcome_minus_p_log"][str(n)] = {
            "weight": "min(p(a) / q(a), 5), p the learner's current probability",
            "all_steps": {"steps": summary["steps"], "weights": w["n"], "clipped": round(w["clipped_share"] * w["n"]), "clipped_share": w["clipped_share"], "mean_weight": w["mean"]},
            "to_selected_step": {"selected_step": best, "weights": per_step * len(upto), "clipped": round(sum(s["weight_clipped_share"] for s in upto) * per_step),
                                 "clipped_share": sum(s["weight_clipped_share"] for s in upto) / len(upto),
                                 "first_step_with_a_clip": next((s["step"] for s in steps if s["weight_clipped_share"] > 0), None),
                                 "note": "from the step lines (mean micro-batch share times 32 examples a step), exact when every micro-batch is full"},
            "source": str(run.relative_to(REPO) if run.resolve().is_relative_to(REPO) else run),
        }
        for arm in ("ips_log", "ips_positive"):
            run = runs_dir / f"v3_06b_{arm}_n{n}_s0"
            ips = json.loads((run / "train_summary.json").read_text())["ips"]
            out[arm][str(n)] = {
                "weight": "min(1 / q(a), 5) divided by its mean over the training prefix",
                "prefix": {"weights": ips["n"], "clipped": ips["clipped"], "clipped_share": ips["clipped_share"], "normaliser": ips["normaliser"]},
                "trained_on": {"weights": ips["n_used"], "clipped": ips["clipped_used"], "clipped_share": ips["clipped_used"] / ips["n_used"], "mean_normalised_weight": ips["mean_weight_used"]},
                "source": str(run.relative_to(REPO) if run.resolve().is_relative_to(REPO) else run),
            }
    return out


def off_policy(runs_dir: Path) -> dict[str, Any]:
    rl = load_script("train_rlcd")
    return {
        "identity": {
            "claim": "E_q[grad outcome_minus_p_log loss] = (1 - 1/B) / 2 * E_q[grad (1 / q(a)) (r - p(a))^2] = (1 - 1/B) / 2 * grad sum_a (r_a - p_a)^2, per question, while no weight is clipped",
            "derivation": "E_q[w R grad log p] = sum_a p_a (r_a - p_a) grad log p_a = sum_a (r_a - p_a) grad p_a; the micro-batch baseline's other samples give b sum_a grad p_a = 0 and its own sample removes 1/B; E_q[(1 / q) grad (r - p_a)^2] = -2 sum_a (r_a - p_a) grad p_a",
            "tiny_model_check": identity_check(rl),
        },
        "clipping": clipping(runs_dir),
    }


# Forgetting and selected steps


FORGETTING_METRICS = ("accuracy", "ece")


def forgetting(runs_dir: Path, resamples: int) -> dict[str, Any]:
    from jevmark.calibration import scale_results
    from jevmark.metrics import read_results

    cv3 = load_script("compare_v3")
    zero_dir = runs_dir / "v3_06b_zeroshot"
    zero_all = read_results(zero_dir / "results.jsonl.gz")
    per_run = {"zero_shot": cv3.by_split(zero_all)}
    sources = {"zero_shot": cv3.shown(zero_dir)}
    for n in cv3.NS:
        temp_dir = runs_dir / f"v3_06b_temp_n{n}"
        t = float(json.loads((temp_dir / "calibration.json").read_text())["temperature"])
        per_run[f"temperature_n{n}"] = cv3.by_split(scale_results(zero_all, t))
        sources[f"temperature_n{n}"] = cv3.shown(temp_dir)
    runs = cv3.discover(runs_dir, "06b")
    for (arm, n, seed, noisy, log), path in runs.items():
        name = cv3.learner_name(arm, n, seed, noisy, log)
        per_run[name] = cv3.by_split(read_results(path / "results.jsonl.gz"))
        sources[name] = cv3.shown(path)
        temp_dir = path.parent / f"{path.name}_temp"
        if arm == "direct_brier" and not noisy and not log and (temp_dir / "calibration.json").is_file():
            t = float(json.loads((temp_dir / "calibration.json").read_text())["temperature"])
            temp_name = cv3.learner_name(cv3.TEMP_ON_RLCD, n, seed)
            per_run[temp_name] = cv3.by_split(scale_results(read_results(path / "results.jsonl.gz"), t))
            sources[temp_name] = cv3.shown(temp_dir)
    stored = json.loads(STAGE.read_text())["forgetting"]
    table: dict[str, Any] = {
        "note": ("accuracy and ECE on every gold-dependent question of the split, 95 percent record bootstrap intervals (compare_v3.Split draws, "
                 "seed text v3:<split>, the same draws for every learner), and the paired difference from the zero-shot run (sft_06b, the logging policy)"),
        "resamples": resamples,
        "sources": sources,
    }
    for split in cv3.FORGETTING_SPLITS:
        reference = per_run["zero_shot"][split]
        members = {name: reference if name == "zero_shot" else cv3.aligned(reference, results[split], name) for name, results in per_run.items()}
        s = cv3.Split(split, members, resamples, FORGETTING_METRICS)
        block: dict[str, Any] = {"n": len(reference), "n_records": len({r.record_id for r in reference}), "learners": {}}
        for name in members:
            point = s.point(name)
            entry = {m: point[m] for m in ("accuracy", "accuracy_ci", "ece", "ece_ci")}
            if name != "zero_shot":
                entry["minus_zero_shot"] = s.delta(name, "zero_shot")
                if stored[split]["minus_zero_shot"].get(name) != entry["minus_zero_shot"]:
                    raise SystemExit(f"{split} {name}: the difference from zero-shot differs from the file's forgetting block")
            elif any(stored[split]["zero_shot"][k] != entry[k] for k in entry):
                raise SystemExit(f"{split}: the zero-shot values differ from the file's forgetting block")
            block["learners"][name] = entry
        table[split] = block
    return table


def selected_steps(runs_dir: Path) -> dict[str, Any]:
    import yaml

    out: dict[str, Any] = {
        "note": ("selected_step is the step with the best selection criterion (mean Bernoulli log-likelihood of the logged outcomes on the held-out 0.1 N); "
                 "n_used is the interactions trained on (outcome-1 ones only for positive_sft and ips_positive); an epoch is ceil(ceil(n_used / micro_batch) / accumulation) steps"),
        "runs": {},
    }
    for path in sorted(runs_dir.glob("v3_06b_*")):
        summary_path = path / "train_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("mode") != "log" or summary.get("run_name") != path.name:
            continue
        training = yaml.safe_load((path / "config.yaml").read_text())["training"]
        micro, effective = int(training["micro_batch"]), int(training["effective_batch"])
        per_epoch = math.ceil(math.ceil(summary["n_used"] / micro) / (effective // micro))
        out["runs"][path.name] = {
            "arm": summary["arm"], "n": summary["n"], "n_train": summary["n_train"], "n_used": summary["n_used"], "noisy": summary.get("noisy"),
            "log": summary["log"], "selected_step": summary["best_step"], "last_step": summary["steps"], "steps_per_epoch": per_epoch,
            "selected_epochs": summary["best_step"] / per_epoch, "last_epochs": summary["steps"] / per_epoch,
            "selection_every": int(training["eval_every"]),
        }
    return out


# Writing


def write_keys(values: dict[str, Any], script_key: str) -> None:
    stage = json.loads(STAGE.read_text())
    stage.update(values)
    stage.setdefault("review", {})[script_key] = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "script_sha256": sha256(Path(__file__)),
        "keys": sorted(values),
    }
    STAGE.write_text(json.dumps(stage, indent=2) + "\n")


def print_off_policy(o: dict[str, Any]) -> None:
    c = o["identity"]["tiny_model_check"]
    print(f"identity on the tiny model (B {c['setting']['micro_batch']}, K {c['setting']['k']}, {c['setting']['action_pairs_enumerated']} action pairs, max p/q {c['max_weight_p_over_q']:.4f}):")
    print(f"  E grad outcome_minus_p_log vs {c['constant']:.4f} x E grad IPS CAPS-Brier: max rel difference {c['max_rel_difference_reinforce_vs_constant_times_ips_brier']:.2e}, cosine {c['cosine_reinforce_ips_brier']:.8f}")
    print(f"  E grad IPS CAPS-Brier vs grad full-information Brier: max rel difference {c['max_rel_difference_ips_brier_vs_full_information_brier']:.2e}")
    lp = o["clipping"]["logging_policy"]
    print(f"\nlogging policy on the first {lp['n']} interactions: max p_log/q {lp['max_p_log_over_q']:.4f} (bound {lp['bound_1_over_1_minus_epsilon']:.4f}), clipped {lp['clipped']}")
    print("\narm | N | weights | clipped | share")
    for n, v in o["clipping"]["outcome_minus_p_log"].items():
        a, s = v["all_steps"], v["to_selected_step"]
        print(f"outcome_minus_p_log | {n} | {a['weights']} (all {a['steps']} steps) | {a['clipped']} | {a['clipped_share']:.4f}")
        print(f"outcome_minus_p_log | {n} | {s['weights']} (to selected step {s['selected_step']}) | {s['clipped']} | {s['clipped_share']:.4f} (first clip at step {s['first_step_with_a_clip']})")
    for arm in ("ips_log", "ips_positive"):
        for n, v in o["clipping"][arm].items():
            p, t = v["prefix"], v["trained_on"]
            print(f"{arm} | {n} | {p['weights']} prefix / {t['weights']} trained on | {p['clipped']} / {t['clipped']} | {p['clipped_share']:.4f} / {t['clipped_share']:.4f}")


def print_forgetting(table: dict[str, Any], steps: dict[str, Any]) -> None:
    fmt = lambda v, k: f"{v[k]:.3f} [{v[k + '_ci'][0]:.3f}, {v[k + '_ci'][1]:.3f}]"
    for split in ("test_indomain", "test_unseen_intents"):
        print(f"\n== {split}: accuracy and ECE [95 percent record intervals], accuracy minus zero-shot [paired interval]")
        for name, v in table[split]["learners"].items():
            d = v.get("minus_zero_shot", {}).get("accuracy")
            delta = f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d else ""
            print(f"{name:32s} {fmt(v, 'accuracy')}  {fmt(v, 'ece')}  {delta}")
    print("\n== selected steps: run | n_used | selected step (epochs) | last step (epochs)")
    for name, v in steps["runs"].items():
        print(f"{name:52s} {v['n_used']:5d} | {v['selected_step']:5d} ({v['selected_epochs']:.2f}) | {v['last_step']:5d} ({v['last_epochs']:.2f})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("off-policy", "forgetting"))
    parser.add_argument("--runs-dir", default=str(REPO / "runs"))
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if args.what == "off-policy":
        result = off_policy(runs_dir)
        write_keys({"off_policy": result}, "off_policy")
        print_off_policy(result)
    else:
        table, steps = forgetting(runs_dir, args.resamples), selected_steps(runs_dir)
        write_keys({"forgetting_table": table, "selected_steps": steps}, "forgetting")
        print_forgetting(table, steps)
    print(f"\nwrote {STAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
