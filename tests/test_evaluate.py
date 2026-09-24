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
from jevmark.data.clinc import DOMAIN_INSTRUCTIONS, DOMAIN_PHRASES, OUT_OF_SCOPE_INSTRUCTIONS
from jevmark.data.negation import NEGATIONS, negate

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_script", REPO / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate
spec.loader.exec_module(evaluate)

PER_SPLIT = 30


# Negation templates


def test_negation_templates_cover_both_noul_kinds():
    assert set(NEGATIONS) == {"about_domain", "out_of_scope"}
    phrase = DOMAIN_PHRASES["travel"]
    assert negate("about_domain", DOMAIN_INSTRUCTIONS.format(phrase=phrase)) == f"Is this message about something other than {phrase}?"
    assert negate("out_of_scope", OUT_OF_SCOPE_INSTRUCTIONS) == "Is this request something a banking, travel, home, work or everyday assistant can help with?"


def test_negation_rejects_unknown_kind_or_unfit_instruction():
    with pytest.raises(ValueError, match="no negation template"):
        negate("sentiment", "How positive?")
    with pytest.raises(ValueError, match="does not start with"):
        negate("about_domain", "Is this about travel?")


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


def test_metrics_json_structure(run):
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
    assert train["overall"]["n"] == 2 * PER_SPLIT  # every CLINC record has two questions
    assert len(train["noul"]["coverage"]) == 21 and "coverage" not in train["overall"]
    assert "macro_f1" in train["choice"]
    assert train["symmetry"]["n"] == PER_SPLIT and 0.0 <= train["symmetry"]["argmax_consistent"] <= 1.0

    sst5 = metrics["splits"]["test_sst5"]
    assert set(sst5) == {"overall", "score", "n_records"} and "mae" in sst5["score"]
    assert set(metrics["splits"]["test_agnews"]) == {"overall", "choice", "letter_bias", "n_records"}

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
    probs = jev.forward_distributions(encoded)
    correct = [list(r["questions"]["label"]["criteria"])[int(p.argmax())] == r["gold"]["label"] for r, p in zip(records, probs)]
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
    assert set(lines[0]) == {"record_id", "question_id", "split", "type", "kind", "labels", "probs", "gold", "confidence", "prediction", "negated_p_yes"}
    nouls = [l for l in lines if l["type"] == "noul"]
    assert nouls and all(l["kind"] in ("about_domain", "out_of_scope") and l["negated_p_yes"] is not None for l in nouls)
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
    assert set(indomain["noul"]["by_kind"]) <= {"about_domain", "out_of_scope"} and "yes_rate" in indomain["noul"]
    assert {"n_offering_other", "predicted_other_rate"} <= set(indomain["choice"]["by_gold_other"])
    assert "by_k_position" in indomain["letter_bias"]
    assert "by_gold_other" not in metrics["splits"]["test_agnews"]["choice"]  # AG News offers no "other"
