"""scripts/evaluate.py end to end on the tiny model, plus negation templates and the B0 configs."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from jevmark.config import load_config
from jevmark.data.assemble import build_all
from jevmark.data.build import SPLITS
from jevmark.data.clinc import DOMAIN_PHRASES
from jevmark.data.form import FORM_KINDS
from jevmark.data.negation import KINDS, TEMPLATES, negate, parse, render

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_script", REPO / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate
spec.loader.exec_module(evaluate)

PER_SPLIT = 30


# Negation templates


SLOTS = {
    "about_domain": DOMAIN_PHRASES["travel"],
    "about_intent": "Wants to move money between accounts",
    "expresses_emotion": "joy",
    "word_count_over": "12",
    "char_count_over": "80",
    "longest_word_over": "9",
}
GOLD_KINDS = {"about_domain", "out_of_scope", "about_intent", "is_positive", "is_negative", "expresses_emotion"}


def test_every_kind_has_two_or_three_templates():
    assert set(KINDS) == GOLD_KINDS | set(FORM_KINDS)
    assert all(2 <= len(TEMPLATES[kind]) <= 3 for kind in KINDS)


@pytest.mark.parametrize("kind", sorted(GOLD_KINDS | set(FORM_KINDS)))
def test_every_template_maps_to_its_pair_in_both_directions(kind):
    rendered = set()
    for template in range(len(TEMPLATES[kind])):
        positive = render(kind, template, False, SLOTS.get(kind))
        negated = render(kind, template, True, SLOTS.get(kind))
        assert negate(kind, positive) == negated and negate(kind, negated) == positive
        assert parse(kind, positive).template == parse(kind, negated).template == template
        assert not parse(kind, positive).negated and parse(kind, negated).negated
        assert parse(kind, negated).slot == SLOTS.get(kind)
        rendered |= {positive, negated}
    assert len(rendered) == 2 * len(TEMPLATES[kind])


def test_negation_examples():
    phrase = DOMAIN_PHRASES["travel"]
    assert negate("about_domain", f"Is this message about {phrase}?") == f"Is this message about something other than {phrase}?"
    assert negate("expresses_emotion", "Is joy the main emotion in this message?") == "Is something other than joy the main emotion in this message?"


def test_negation_rejects_unknown_kind_or_unfit_instruction():
    with pytest.raises(ValueError, match="no negation template"):
        negate("sentiment", "How positive?")
    with pytest.raises(ValueError, match="fits no phrasing"):
        negate("about_domain", "Is this about travel?")


def test_negated_record_works_for_either_stored_phrasing():
    phrase = DOMAIN_PHRASES["work"]
    record = {
        "questions": {
            "about_domain": {"type": "noul", "instructions": render("about_domain", 1, True, phrase)},
            "about_intent": {"type": "noul", "instructions": render("about_intent", 0, False, "Asks about pay")},
        }
    }
    flipped = evaluate.negated_record(record)
    assert flipped["questions"]["about_domain"]["instructions"] == render("about_domain", 1, False, phrase)
    assert flipped["questions"]["about_intent"]["instructions"] == render("about_intent", 0, True, "Asks about pay")
    assert evaluate.negated_record(flipped)["questions"] == record["questions"]


# B0 configs


@pytest.mark.parametrize("name, backbone", [("base_06b", "Qwen/Qwen3-0.6B-Base"), ("base_17b", "Qwen/Qwen3-1.7B-Base")])
def test_b0_configs_differ_from_base_only_in_backbone_and_run_name(name, backbone):
    base = load_config(REPO / "configs" / "base.yaml")
    config = load_config(REPO / "configs" / f"{name}.yaml")
    assert config["run_name"] == name and config["backbone"]["id"] == backbone
    assert {k: v for k, v in config.items() if k not in ("run_name", "backbone")} == {k: v for k, v in base.items() if k not in ("run_name", "backbone")}


# End to end


@pytest.fixture(scope="module")
def eval_inputs(tmp_path_factory, tiny_model, tokenizer, base_config):
    root = tmp_path_factory.mktemp("evaluate")
    backbone = root / "backbone"
    tiny_model.save_pretrained(backbone)
    tokenizer.save_pretrained(backbone)
    config = copy.deepcopy(base_config)
    config["backbone"] = {"id": str(backbone), "revision": None}
    config["run_name"] = "tiny_eval"
    config_path = root / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))

    data_dir = root / "data"
    data_dir.mkdir()
    built = build_all(load_config(REPO / "configs" / "data.yaml"))
    for split, records in built.splits.items():
        with (data_dir / f"{split}.jsonl").open("w") as f:
            for record in records[:PER_SPLIT]:
                f.write(json.dumps(record) + "\n")
    return root, config_path, data_dir


@pytest.fixture(scope="module")
def run(eval_inputs):
    root, config_path, data_dir = eval_inputs
    runs_dir = root / "runs"
    code = evaluate.main(["--ckpt", "base", "--config", str(config_path), "--limit", str(PER_SPLIT), "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", "--batch-size", "8"])
    assert code == 0
    return runs_dir / f"tiny_eval_limit{PER_SPLIT}"


def test_outputs_written(run):
    assert (run / "config.yaml").is_file()
    assert (run / "model_id.txt").read_text().strip() == "jevmark-tiny_eval"
    assert sorted(p.name for p in (run / "plots").iterdir()) == sorted(f"{s}.png" for s in SPLITS)


def test_metrics_json_structure(run, eval_inputs):
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["run_name"] == f"tiny_eval_limit{PER_SPLIT}"
    assert metrics["limit"] == PER_SPLIT and metrics["device"] == "cpu"
    assert len(metrics["git"]["commit"]) == 40
    assert metrics["precision"] == {"autocast": None, "fp32_fallback_used": False}
    assert set(metrics["data_files_sha256"]) == {f"{s}.jsonl" for s in SPLITS}
    assert set(metrics["splits"]) == set(SPLITS)

    train = metrics["splits"]["train"]
    assert train["n_records"] == PER_SPLIT
    assert {"overall", "noul", "choice", "letter_bias", "symmetry"} <= set(train)
    for block in ("overall", "noul", "choice"):
        assert {"n", "accuracy", "ece", "brier", "nll", "reliability"} <= set(train[block])
        assert len(train[block]["reliability"]) == 15
    train_records = [json.loads(line) for line in (eval_inputs[2] / "train.jsonl").read_text().splitlines()]
    assert train["overall"]["n"] == sum(len(r["questions"]) for r in train_records)  # choice, one gold noul, 0 to 2 form nouls
    assert len(train["noul"]["coverage"]) == 21 and "coverage" not in train["overall"]
    assert "macro_f1" in train["choice"]
    assert train["symmetry"]["n"] == sum(q["type"] == "noul" for r in train_records for q in r["questions"].values())
    assert 0.0 <= train["symmetry"]["argmax_consistent"] <= 1.0

    sst5 = metrics["splits"]["test_sst5"]
    assert set(sst5) == {"overall", "score", "noul", "symmetry", "n_records"} and "mae" in sst5["score"]  # v1.2: sentiment nouls
    assert set(metrics["splits"]["test_agnews"]) == {"overall", "choice", "noul", "symmetry", "letter_bias", "n_records"}  # v1.3: form nouls

    assert metrics["batching_precision"]["n_requests"] == 200
    assert metrics["batching_precision"]["max_abs_difference"] < 1e-4
    assert metrics["latency"]["batch_1"]["n"] == 200 and metrics["latency"]["batch_16_requests_per_second"] > 0


def test_accuracy_matches_a_direct_forward(run, eval_inputs, tiny_model, tokenizer):
    """The recorded choice accuracy on test_agnews equals an independent recomputation."""
    from jevmark.encode import encode
    from jevmark.model import JevMark
    from jevmark.schema import Request

    _, _, data_dir = eval_inputs
    records = [json.loads(line) for line in (data_dir / "test_agnews.jsonl").read_text().splitlines()]
    jev = JevMark(tiny_model, tokenizer, max_tokens=2048)
    encoded = [encode(Request.from_dict({"state": r["state"], "questions": r["questions"]}), tokenizer, 2048) for r in records]
    flat = iter(jev.forward_distributions(encoded))
    per_record = [{qid: next(flat) for qid in r["questions"]} for r in records]  # form nouls share the sequence (data v1.3)
    correct = [list(r["questions"]["label"]["criteria"])[int(p["label"].argmax())] == r["gold"]["label"] for r, p in zip(records, per_record)]
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["splits"]["test_agnews"]["choice"]["accuracy"] == pytest.approx(sum(correct) / len(correct))


def test_first_batch_nan_switches_to_fp32(monkeypatch, tiny_model, tokenizer):
    import torch

    from jevmark.model import JevMark

    jev = JevMark(tiny_model, tokenizer, max_tokens=2048, autocast_dtype=torch.float16)
    calls = []

    def fake_slot_logits(batch):
        calls.append(jev.autocast_dtype)
        return [torch.tensor([float("nan"), 0.0])] if jev.autocast_dtype is not None else [torch.tensor([0.0, 0.0])]

    monkeypatch.setattr(jev, "slot_logits", fake_slot_logits)
    assert evaluate.first_batch_check(jev, []) is True
    assert jev.autocast_dtype is None and calls == [torch.float16, None]


def test_nan_in_fp32_is_an_error(monkeypatch, tiny_model, tokenizer):
    import torch

    from jevmark.model import JevMark

    jev = JevMark(tiny_model, tokenizer, max_tokens=2048)
    monkeypatch.setattr(jev, "slot_logits", lambda batch: [torch.tensor([float("inf"), 0.0])])
    with pytest.raises(RuntimeError, match="fp32"):
        evaluate.first_batch_check(jev, [])


# results.jsonl.gz and recompute_metrics.py (task 1.6b)

spec_r = importlib.util.spec_from_file_location("recompute_metrics", REPO / "scripts" / "recompute_metrics.py")
recompute_script = importlib.util.module_from_spec(spec_r)
sys.modules[spec_r.name] = recompute_script
spec_r.loader.exec_module(recompute_script)


def test_results_file_has_one_line_per_question(run):
    import gzip

    lines = [json.loads(line) for line in gzip.open(run / "results.jsonl.gz", "rt")]
    metrics = json.loads((run / "metrics.json").read_text())
    assert len(lines) == sum(split["overall"]["n"] for split in metrics["splits"].values())
    assert set(lines[0]) == {
        "record_id", "question_id", "split", "type", "kind", "labels", "probs", "gold", "confidence", "prediction", "negated_p_yes",
        "position", "shuffled_probs", "shuffled_position",
    }  # fmt: skip
    assert all(l["shuffled_probs"] is None for l in lines)  # no --shuffle-questions in this run
    nouls = [l for l in lines if l["type"] == "noul"]
    assert nouls and all(l["kind"] in KINDS and l["kind"] == l["question_id"] and l["negated_p_yes"] is not None for l in nouls)
    assert all(l["kind"] is None and l["negated_p_yes"] is None for l in lines if l["type"] != "noul")


def test_recompute_rebuilds_metrics_exactly(run, tmp_path):
    out = tmp_path / "recomputed.json"
    assert recompute_script.main([str(run), "--out", str(out)]) == 0
    original = json.loads((run / "metrics.json").read_text())
    rebuilt = json.loads(out.read_text())
    assert rebuilt["splits"] == original["splits"]
    for key in ("git", "precision", "latency", "batching_precision", "data_files_sha256"):
        assert rebuilt[key] == original[key]
    assert rebuilt["recomputed"]["questions"] == sum(s["overall"]["n"] for s in original["splits"].values())


def test_new_breakdowns_reach_metrics_json(run):
    metrics = json.loads((run / "metrics.json").read_text())
    indomain = metrics["splits"]["test_indomain"]
    assert set(indomain["noul"]["by_kind"]) <= {"about_domain", "out_of_scope", "about_intent"} | set(FORM_KINDS) and "about_intent" in indomain["noul"]["by_kind"]
    assert "yes_rate" in indomain["noul"]
    assert set(metrics["splits"]["test_emotion"]["noul"]["by_kind"]) - set(FORM_KINDS) == {"expresses_emotion"}
    assert "score" in metrics["splits"]["test_yelp"] and set(metrics["splits"]["test_yelp"]["noul"]["by_kind"]) <= set(FORM_KINDS)
    assert {"n_offering_other", "predicted_other_rate"} <= set(indomain["choice"]["by_gold_other"])
    assert "by_k_position" in indomain["letter_bias"]
    assert "by_gold_other" not in metrics["splits"]["test_agnews"]["choice"]  # AG News offers no "other"


# Git state (fix after the base_06b re-run)


def test_git_state_ignores_runs_and_sees_code(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "runs" / "old").mkdir(parents=True)
    (tmp_path / "runs" / "old" / "metrics.json").write_text("{}\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert evaluate.git_state(tmp_path)["dirty"] is False
    (tmp_path / "runs" / "old" / "metrics.json").write_text('{"rerun": true}\n')
    state = evaluate.git_state(tmp_path)
    assert state["dirty"] is False and len(state["commit"]) == 40
    (tmp_path / "code.py").write_text("x = 2\n")
    assert evaluate.git_state(tmp_path)["dirty"] is True


def test_git_state_is_recorded_before_outputs(eval_inputs, monkeypatch, tmp_path):
    root, config_path, data_dir = eval_inputs
    runs_dir = tmp_path / "runs"
    seen = []
    real = evaluate.git_state
    monkeypatch.setattr(evaluate, "git_state", lambda: seen.append(runs_dir.exists()) or real())
    assert evaluate.main(["--ckpt", "base", "--config", str(config_path), "--limit", "2", "--splits", "test_agnews", "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), "--device", "cpu"]) == 0
    assert seen == [False]


# Merged LoRA path recorded in metrics.json (task 1.7)


@pytest.fixture(scope="module")
def lora_checkpoint(eval_inputs, tiny_model):
    import torch
    from peft import LoraConfig, get_peft_model

    root, config_path, _ = eval_inputs
    run_dir = root / "checkpoints" / "tiny_lora"
    model = get_peft_model(copy.deepcopy(tiny_model), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
    model.save_pretrained(run_dir / "adapter")
    config = yaml.safe_load(config_path.read_text())
    config["run_name"] = "tiny_lora"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "model_id.txt").write_text("jevmark-tiny_lora\n")
    return run_dir


@pytest.mark.parametrize("extra, merged", [([], True), (["--no-merge"], False)])
def test_metrics_record_whether_lora_was_merged(eval_inputs, lora_checkpoint, tmp_path, extra, merged):
    _, _, data_dir = eval_inputs
    runs_dir = tmp_path / "runs"
    args = ["--ckpt", str(lora_checkpoint), "--limit", "3", "--splits", "test_agnews", "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", *extra]
    assert evaluate.main(args) == 0
    metrics = json.loads((runs_dir / "tiny_lora_limit3" / "metrics.json").read_text())
    assert metrics["lora_merged"] is merged and metrics["model_id"] == "jevmark-tiny_lora"


def test_base_run_records_no_lora(run):
    assert json.loads((run / "metrics.json").read_text())["lora_merged"] is None


# Question position and order sensitivity (decision 42)


def test_positions_follow_the_stored_question_order(run, eval_inputs):
    import gzip

    lines = [json.loads(line) for line in gzip.open(run / "results.jsonl.gz", "rt")]
    records = {json.loads(l)["id"]: json.loads(l) for l in (eval_inputs[2] / "test_indomain.jsonl").read_text().splitlines()}
    for line in lines:
        if line["split"] == "test_indomain":
            assert list(records[line["record_id"]]["questions"]).index(line["question_id"]) == line["position"]
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["splits"]["test_indomain"]["noul"]["by_question_position"]) <= {"first", "later"}
    assert len(metrics["splits"]["test_indomain"]["overall"]["accuracy_ci"]) == 2


def test_shuffled_order_is_seeded_and_differs():
    record = {"id": "r1", "questions": {"a": {}, "b": {}, "c": {}}}
    order = evaluate.shuffled_order(record)
    assert order != ["a", "b", "c"] and sorted(order) == ["a", "b", "c"]
    assert evaluate.shuffled_order(record) == order
    assert evaluate.shuffled_order({"id": "r2", "questions": {"a": {}, "b": {}}}) == ["b", "a"]


def test_shuffle_questions_measures_order_sensitivity(eval_inputs, tmp_path):
    import gzip

    _, config_path, data_dir = eval_inputs
    runs_dir = tmp_path / "runs"
    args = ["--ckpt", "base", "--config", str(config_path), "--limit", "12", "--splits", "test_indomain", "test_agnews",
            "--shuffle-questions", "test_indomain", "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), "--device", "cpu"]  # fmt: skip
    assert evaluate.main(args) == 0
    run_dir = runs_dir / "tiny_eval_limit12"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["shuffle_questions"] == ["test_indomain"]
    sensitivity = metrics["splits"]["test_indomain"]["order_sensitivity"]
    import random

    all_records = [json.loads(l) for l in (data_dir / "test_indomain.jsonl").read_text().splitlines()]
    records = evaluate.sample_records(all_records, 12, random.Random("limit:test_indomain"))
    assert sensitivity["overall"]["n"] == sum(len(r["questions"]) for r in records if len(r["questions"]) > 1)
    assert 0.0 <= sensitivity["overall"]["prediction_agreement"] <= 1.0 and sensitivity["overall"]["max_abs_difference"] >= 0.0
    assert "order_sensitivity" not in metrics["splits"]["test_agnews"]
    lines = [json.loads(l) for l in gzip.open(run_dir / "results.jsonl.gz", "rt")]
    moved = [l for l in lines if l["shuffled_probs"] is not None]
    assert moved and all(l["split"] == "test_indomain" for l in moved)
    assert any(l["shuffled_position"] != l["position"] for l in moved)


def test_shuffle_questions_rejects_a_split_that_is_not_evaluated():
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--ckpt", "base", "--splits", "test_agnews", "--shuffle-questions", "test_indomain"])


# --limit sampling (fast cycle)


def clinc_like(n_intents, per_intent, n_oos, extra_source=0):
    records = []
    for i in range(n_intents):
        for j in range(per_intent):
            records.append({"id": f"c{i}-{j}", "source": "clinc", "meta": {"gold_intent": f"intent{i}"}})
    records += [{"id": f"o{j}", "source": "clinc", "meta": {"gold_intent": "oos"}} for j in range(n_oos)]
    records += [{"id": f"s{j}", "source": "sst", "meta": {}} for j in range(extra_source)]
    return records


def test_limit_sample_covers_intents_and_out_of_scope_in_file_order():
    import random

    records = clinc_like(n_intents=50, per_intent=30, n_oos=500, extra_source=1000)
    sample = evaluate.sample_records(records, 300, random.Random("limit:x"))
    assert len(sample) == 300 and len({r["id"] for r in sample}) == 300
    assert sample == evaluate.sample_records(records, 300, random.Random("limit:x"))  # seeded
    positions = [records.index(r) for r in sample]
    assert positions == sorted(positions)  # file order kept
    clinc = [r for r in sample if r["source"] == "clinc"]
    assert len(clinc) == round(300 * 2000 / 3000)  # sources in proportion
    intents = {r["meta"]["gold_intent"] for r in clinc}
    assert len(intents - {"oos"}) == 50  # every intent
    assert sum(r["meta"]["gold_intent"] == "oos" for r in clinc) == round(200 * 500 / 2000)  # out-of-scope share


def test_limit_sample_is_the_whole_split_when_the_limit_is_larger():
    import random

    records = clinc_like(3, 2, 1)
    assert evaluate.sample_records(records, 50, random.Random(0)) == records


def test_limit_sample_keeps_one_out_of_scope_when_its_share_rounds_to_zero():
    import random

    records = clinc_like(n_intents=20, per_intent=50, n_oos=3)
    sample = evaluate.sample_records(records, 30, random.Random(0))
    assert sum(r["meta"]["gold_intent"] == "oos" for r in sample) == 1 and len(sample) == 30
    assert len({r["meta"]["gold_intent"] for r in sample}) == 21
