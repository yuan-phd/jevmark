"""Task 1.8: the shared baseline prompt and parser, metrics assembly, the subset, B1 on the tiny model, B2 with a fake client, and the comparison."""

import copy
import gzip
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from jevmark.baselines.prompt import ParsedAnswer, build_prompt, json_schema, parse_reply
from jevmark.baselines.results import BaselineAnswer, answers_for_record, baseline_split_metrics, read_replies
from jevmark.baselines.subset import build_subset, load_subset, subset_records
from jevmark.metrics import QuestionResult, bootstrap_ci, bootstrap_intervals, ece, write_results
from jevmark.sampling import sample_records
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


b1 = load_script("baseline_llm_json")
b2 = load_script("baseline_api")
recompute_script = load_script("recompute_metrics")
compare = load_script("compare_baselines")


QUESTIONS = {
    "char_count_over": {"type": "noul", "instructions": "Is the text at most 38 characters long?"},
    "intent": {
        "type": "choice",
        "instructions": "Which intent does this message express?",
        "criteria": {"balance": "How much money is in an account", "transfer": "Move money between accounts", "other": None},
    },
    "sentiment": {"type": "score", "instructions": "How positive is the text?", "criteria": ["Negative", "Neutral", "Positive"]},
    "about_intent": {"type": "noul", "instructions": "Does this fit: sending funds?", "criteria": {"true": "it fits", "false": "it does not"}},
}


SHOWN = {"char_count_over": "q1", "intent": "q2", "sentiment": "q3", "about_intent": "q4"}  # the anonymous ids of QUESTIONS


def reply_of(answers):
    """A reply as the model sees the questions: real ids replaced by their anonymous ids (other keys kept)."""
    return json.dumps({SHOWN.get(k, k): v for k, v in answers.items()})


def record(rid="r1", state="send a hundred dollars to savings", questions=QUESTIONS, gold=None, source="clinc", split="test_indomain"):
    gold = gold or {"char_count_over": "true", "intent": "transfer", "sentiment": 1, "about_intent": "true"}
    return {"id": rid, "source": source, "split": split, "state": state, "questions": questions, "gold": gold, "meta": {}}


def request():
    return Request.from_dict({"state": record()["state"], "questions": QUESTIONS})


# Prompt


def test_prompt_renders_state_and_every_question_in_order_with_options_and_levels():
    prompt = build_prompt(request())
    assert "### State\nsend a hundred dollars to savings" in prompt
    order = [prompt.index(f"Question id: q{i}\n") for i in range(1, len(QUESTIONS) + 1)]
    assert order == sorted(order)
    assert prompt.index("Question id: q2\nType: choice\nQuestion: Which intent") > 0
    for qid in QUESTIONS:
        assert f"Question id: {qid}" not in prompt and f'"{qid}"' not in prompt  # real ids name the question kind; the model never sees them
    for line in (
        "Question: Which intent does this message express?",
        "- balance: How much money is in an account",
        "- transfer: Move money between accounts",
        "- other\n",  # a null description renders the label alone
        "- 0: Negative\n- 1: Neutral\n- 2: Positive",
        "- true: it fits\n- false: it does not",
        "- true: yes\n- false: no",  # the default noul descriptions, as in the encoding
    ):
        assert line in prompt, line
    assert prompt.rstrip().endswith('"q4": {"answer": ..., "confidence": ...}}')


def test_prompt_renders_a_json_state_as_the_encoding_does():
    state = {"text": "hello", "channel": "email"}
    prompt = build_prompt(Request.from_dict({"state": state, "questions": {"q": QUESTIONS["intent"]}}))
    assert "### State\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n\n### Questions" in prompt


def test_json_schema_restricts_every_answer_and_requires_every_question():
    schema = json_schema(request())
    assert schema["required"] == ["q1", "q2", "q3", "q4"] and schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["q1"]["properties"]["answer"] == {"type": "boolean"}
    assert props["q2"]["properties"]["answer"] == {"type": "string", "enum": ["balance", "transfer", "other"]}
    assert props["q3"]["properties"]["answer"] == {"type": "integer", "enum": [0, 1, 2]}
    assert props["q4"]["properties"]["answer"] == {"type": "boolean"}
    for entry in props.values():
        assert entry["required"] == ["answer", "confidence"] and entry["additionalProperties"] is False


# Parser


def test_parser_reads_a_valid_reply():
    reply = reply_of({
        "char_count_over": {"answer": True, "confidence": 0.9},
        "intent": {"answer": "transfer", "confidence": 0.75},
        "sentiment": {"answer": 2, "confidence": 1},
        "about_intent": {"answer": False, "confidence": 0},
    })
    assert parse_reply(reply, request()) == {
        "char_count_over": ParsedAnswer("ok", 0, 0.9),
        "intent": ParsedAnswer("ok", 1, 0.75),
        "sentiment": ParsedAnswer("ok", 2, 1.0),
        "about_intent": ParsedAnswer("ok", 1, 0.0),
    }


@pytest.mark.parametrize("reply", [None, "", "not json", "[1, 2]", '{"intent": ', 'Sure! {"intent": {"answer": "other"}}', "```json\n{}\n``` trailing"])
def test_a_reply_that_is_not_one_json_object_fails_every_question(reply):
    assert {p.status for p in parse_reply(reply, request()).values()} == {"invalid_json"}


def test_a_single_code_fence_is_stripped():
    reply = '```json\n{"q2": {"answer": "other", "confidence": 0.5}}\n```'
    assert parse_reply(reply, request())["intent"] == ParsedAnswer("ok", 2, 0.5)


def test_missing_and_out_of_set_answers_are_failures_apart_from_wrong_answers():
    reply = reply_of({
        "intent": {"answer": "billing", "confidence": 0.9},  # not an option
        "sentiment": {"answer": 3},  # out of range
        "about_intent": {"answer": "yes"},  # not true or false
        "unknown_question": {"answer": 1},  # ignored
    })
    parsed = parse_reply(reply, request())
    assert parsed["char_count_over"].status == "missing"
    assert parsed["intent"].status == parsed["sentiment"].status == parsed["about_intent"].status == "invalid_answer"


@pytest.mark.parametrize(
    "qid, value, expected",
    [
        ("about_intent", "true", 0),
        ("about_intent", " false ", 1),
        ("about_intent", 1, None),
        ("intent", " other ", 2),
        ("intent", "Other", None),
        ("sentiment", "1", 1),
        ("sentiment", 2.0, 2),
        ("sentiment", 1.5, None),
        ("sentiment", True, None),
        ("sentiment", -1, None),
    ],
)
def test_answer_formats(qid, value, expected):
    parsed = parse_reply(reply_of({qid: {"answer": value}}), request())[qid]
    assert parsed.answer == expected and parsed.status == ("ok" if expected is not None else "invalid_answer")


@pytest.mark.parametrize("confidence, expected", [(0.3, 0.3), (1, 1.0), (1.2, None), (-0.1, None), ("0.8", None), (True, None), (None, None)])
def test_an_invalid_confidence_is_dropped_but_the_answer_counts(confidence, expected):
    parsed = parse_reply(reply_of({"intent": {"answer": "balance", "confidence": confidence}}), request())["intent"]
    assert parsed == ParsedAnswer("ok", 0, expected)


def test_a_bare_answer_counts_without_confidence_and_an_empty_entry_fails():
    parsed = parse_reply(reply_of({"intent": "balance", "sentiment": {"confidence": 0.5}}), request())
    assert parsed["intent"] == ParsedAnswer("ok", 0, None)
    assert parsed["sentiment"].status == "invalid_answer"


# Metrics assembly


def answer(rid, qid, qtype, status, prediction, gold, confidence, kind=None, labels=("true", "false")):
    return BaselineAnswer(rid, qid, "s", qtype, kind, labels, gold, status, prediction, confidence, 0)


def test_metrics_by_hand():
    labels = ("a", "b", "c")
    answers = [
        answer("r1", "q", "choice", "ok", 0, 0, 0.9, labels=labels),  # right
        answer("r2", "q", "choice", "ok", 1, 0, 0.6, labels=labels),  # wrong
        answer("r3", "q", "choice", "ok", 2, 2, None, labels=labels),  # right, no confidence
        answer("r4", "q", "choice", "invalid_answer", None, 1, None, labels=labels),
        answer("r5", "q", "choice", "invalid_json", None, 1, None, labels=labels),
        answer("r1", "about_intent", "noul", "ok", 0, 0, 0.8, kind="about_intent"),
        answer("r2", "about_intent", "noul", "missing", None, 1, None, kind="about_intent"),
        answer("r1", "char_count_over", "noul", "ok", 1, 0, 0.7, kind="char_count_over"),  # a form noul
    ]
    metrics = baseline_split_metrics(answers)
    choice = metrics["choice"]
    assert choice["n"] == 5 and choice["n_parsed"] == 3
    assert choice["parse_failure_rate"] == pytest.approx(2 / 5)
    assert choice["parse_failures"] == {"invalid_json": 1, "missing": 0, "invalid_answer": 1}
    assert choice["accuracy"] == pytest.approx(2 / 3) and choice["accuracy_all"] == pytest.approx(2 / 5)
    assert choice["n_with_confidence"] == 2
    assert choice["ece"] == pytest.approx(ece([0.9, 0.6], [True, False]))  # (0.1 + 0.6) / 2
    assert choice["ece"] == pytest.approx(0.35)
    assert [row["n"] for row in choice["coverage"] if row["threshold"] in (0.0, 0.7, 0.95)] == [2, 1, 0]
    noul = metrics["noul"]
    assert noul["n"] == 2 and noul["accuracy"] == 1.0 and noul["accuracy_all"] == 0.5 and noul["yes_rate"] == 1.0
    assert set(noul["by_kind"]) == {"about_intent"}
    assert metrics["overall"]["n"] == 7  # the form noul is not in the headline
    assert metrics["form"]["n"] == 1 and metrics["form"]["accuracy"] == 0.0
    assert metrics["n_records"] == 5
    assert choice["accuracy_all_ci"][0] <= choice["accuracy_all"] <= choice["accuracy_all_ci"][1]


def test_metrics_without_any_confidence_or_parse():
    answers = [answer("r1", "q", "noul", "invalid_json", None, 0, None, kind="about_domain")]
    block = baseline_split_metrics(answers)["noul"]
    assert block["accuracy"] is None and block["accuracy_all"] == 0.0 and block["ece"] is None and block["coverage"] is None


def test_answers_for_record_maps_gold_and_positions():
    reply = reply_of({"char_count_over": {"answer": True, "confidence": 0.9}, "intent": {"answer": "transfer", "confidence": 0.8}, "sentiment": {"answer": 0}})
    answers = answers_for_record(record(), "test_indomain", reply)
    assert [(a.question_id, a.position, a.gold, a.status, a.correct) for a in answers] == [
        ("char_count_over", 0, 0, "ok", True),
        ("intent", 1, 1, "ok", True),
        ("sentiment", 2, 1, "ok", False),
        ("about_intent", 3, 0, "missing", False),
    ]
    assert answers[0].is_form and not answers[3].is_form


def test_bootstrap_intervals_equal_bootstrap_ci_on_question_results():
    rng = random.Random(0)
    results = [QuestionResult(f"r{i // 2}", f"q{i}", "noul", (p, 1 - p), rng.randrange(2), ("true", "false")) for i in range(60) for p in [rng.random()]]
    assert bootstrap_intervals([r.record_id for r in results], [r.correct for r in results], [r.top1 for r in results]) == bootstrap_ci(results)


# Subset


def write_split(data_dir, split, records):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def clinc_records(split, n):
    out = []
    for i in range(n):
        rec = record(rid=f"{split}-{i:03d}", state=f"message number {i}", split=split)
        rec["meta"] = {"gold_intent": "oos" if i % 10 == 0 else f"intent{i % 7}"}
        out.append(rec)
    return out


def test_subset_is_the_limit_sample_and_the_sub_subset_is_inside_it(tmp_path):
    records = clinc_records("test_indomain", 120)
    write_split(tmp_path, "test_indomain", records)
    subset = build_subset(tmp_path, ["test_indomain"], per_split=40, sub_per_split=10)
    expected = sample_records(records, 40, random.Random("limit:test_indomain"))
    assert subset["splits"]["test_indomain"] == [r["id"] for r in expected]  # what evaluate.py --limit 40 evaluates
    assert set(subset["sub_splits"]["test_indomain"]) <= set(subset["splits"]["test_indomain"])
    assert len(subset["sub_splits"]["test_indomain"]) == 10
    assert build_subset(tmp_path, ["test_indomain"], 40, 10) == subset  # seeded
    chosen, _ = subset_records(tmp_path, subset, "test_indomain")
    assert [r["id"] for r in chosen] == subset["splits"]["test_indomain"]


def test_subset_records_refuse_a_changed_data_file(tmp_path):
    write_split(tmp_path, "test_indomain", clinc_records("test_indomain", 20))
    subset = build_subset(tmp_path, ["test_indomain"], 5, 2)
    write_split(tmp_path, "test_indomain", clinc_records("test_indomain", 21))
    with pytest.raises(RuntimeError, match="sha256"):
        subset_records(tmp_path, subset, "test_indomain")


def test_committed_subset_has_500_records_per_split_and_matches_its_data_hashes():
    from jevmark.data.build import SPLITS

    subset = load_subset()
    assert list(subset["splits"]) == list(SPLITS)
    assert all(len(ids) == 500 and len(set(ids)) == 500 for ids in subset["splits"].values())
    assert all(len(subset["sub_splits"][s]) == 200 and set(subset["sub_splits"][s]) <= set(subset["splits"][s]) for s in SPLITS)
    metrics = json.loads((REPO / "runs" / "sft_06b" / "metrics.json").read_text())
    assert subset["data_files_sha256"] == metrics["data_files_sha256"]  # drawn from the frozen v1.3 build


def test_the_subset_file_is_committed_and_the_data_files_are_not():
    import subprocess

    def ignored(path):
        return subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=REPO).returncode == 0

    assert not ignored("data/baseline_subset.json") and not ignored("jevmark/data/clinc.py")
    assert ignored("data/train.jsonl") and ignored("runs/b1_qwen17b_json/replies.jsonl")


# recompute_metrics.py --subset


def test_recompute_restricts_to_the_subset(tmp_path):
    results = [
        QuestionResult(rid, "q", "noul", (0.9, 0.1), 0, ("true", "false"), split=split, kind="about_domain", position=1)
        for split in ("test_indomain", "valid")
        for rid in ("a", "b", "c")
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_results(run_dir / "results.jsonl.gz", results)
    subset = {"per_split": 2, "sub_per_split": 1, "splits": {"test_indomain": ["a", "c"]}, "sub_splits": {"test_indomain": ["c"]}}
    (tmp_path / "subset.json").write_text(json.dumps(subset))
    assert recompute_script.main([str(run_dir), "--subset", str(tmp_path / "subset.json")]) == 0
    metrics = json.loads((run_dir / "metrics_subset.json").read_text())
    assert list(metrics["splits"]) == ["test_indomain"] and metrics["splits"]["test_indomain"]["n_records"] == 2
    assert recompute_script.main([str(run_dir), "--subset", str(tmp_path / "subset.json"), "--sub"]) == 0
    assert json.loads((run_dir / "metrics_subset_sub.json").read_text())["splits"]["test_indomain"]["n_records"] == 1
    assert not (run_dir / "metrics.json").exists()


# B1 on the tiny model


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tiny_model, tokenizer):
    path = tmp_path_factory.mktemp("b1") / "tiny"
    tiny_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_chat_text_disables_thinking(tokenizer):
    text = b1.chat_text(tokenizer, "hello")
    assert text.startswith("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
    assert text.endswith("<think>\n\n</think>\n\n")  # the Qwen3 template closes an empty thinking block


def test_tiny_generation_is_greedy_batched_and_counts_tokens(tiny_dir):
    model, tokenizer = b1.load_model(str(tiny_dir), None, torch.device("cpu"), torch.float32)
    prompts = [build_prompt(request()), build_prompt(Request.from_dict({"state": "x", "questions": {"q": QUESTIONS["intent"]}}))]
    batched = b1.generate(model, tokenizer, prompts, max_new_tokens=5, batch_size=2)
    single = [b1.generate(model, tokenizer, [p], max_new_tokens=5, batch_size=1)[0] for p in prompts]
    assert [r["reply"] for r in batched] == [r["reply"] for r in single]  # left padding does not change greedy output here
    for row, prompt in zip(batched, prompts):
        assert 1 <= row["output_tokens"] <= 5 and isinstance(row["reply"], str)
        assert row["input_tokens"] == len(tokenizer(b1.chat_text(tokenizer, prompt), add_special_tokens=False)["input_ids"])
        assert row["finished"] or row["output_tokens"] == 5  # unfinished only when the token budget ran out
    assert b1.first_batch_finite(model, tokenizer, prompts)


@pytest.fixture(scope="module")
def baseline_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("baseline_data")
    data_dir = root / "data"
    sst_questions = {"sentiment": QUESTIONS["sentiment"], "is_positive": {"type": "noul", "instructions": "Is this favourable?"}}
    for split in ("train", "test_indomain", "test_sst5"):
        if split == "test_sst5":
            records = [record(rid=f"sst-{i}", state=f"review {i}", questions=sst_questions, gold={"sentiment": i % 3, "is_positive": "false"}, source="sst5", split=split) for i in range(6)]
        else:
            records = clinc_records(split, 8)
        write_split(data_dir, split, records)
    subset = build_subset(data_dir, ["train", "test_indomain", "test_sst5"], per_split=4, sub_per_split=2)
    (root / "subset.json").write_text(json.dumps(subset))
    return root, data_dir


def test_b1_end_to_end_on_the_tiny_model(tiny_dir, baseline_data, tmp_path):
    root, data_dir = baseline_data
    runs_dir = tmp_path / "runs"
    argv = ["--model", str(tiny_dir), "--revision", "main", "--splits", "test_indomain", "test_sst5", "--subset", str(root / "subset.json"),
            "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), "--device", "cpu", "--batch-size", "3", "--max-new-tokens", "4", "--latency-requests", "3"]
    assert b1.main(argv) == 0
    run = runs_dir / "b1_qwen17b_json"
    replies = read_replies(run / "replies.jsonl")
    assert len(replies) == 8 and {k[0] for k in replies} == {"test_indomain", "test_sst5"}
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["baseline"] == "B1" and metrics["size"] == "17b" and metrics["decoding"] == {"greedy": True, "max_new_tokens": 4, "batch_size": 3, "enable_thinking": False}
    assert set(metrics["splits"]) == {"test_indomain", "test_sst5"}
    indomain = metrics["splits"]["test_indomain"]
    assert indomain["choice"]["n"] == 4 and indomain["choice"]["parse_failure_rate"] == 1.0  # four random tokens are never valid JSON
    assert indomain["form"]["n"] == 4 and indomain["overall"]["n"] == 12
    assert metrics["splits"]["test_sst5"]["score"]["n"] == 4
    assert metrics["latency"]["batch_1"]["n"] == 3 and "batch_3_requests_per_second" in metrics["latency"]
    per_request = metrics["latency"]["batch_1_per_request"]
    assert len(per_request) == 3 and all(r["ms"] > 0 and 1 <= r["output_tokens"] <= 4 for r in per_request)
    assert metrics["latency"]["batch_1"]["p95_ms"] == pytest.approx(float(np.percentile([r["ms"] for r in per_request], 95)))
    assert metrics["generation"]["overall"]["n_requests"] == 8 and metrics["precision"]["fp32_fallback_used"] is False
    assert set(metrics["git"]) == {"commit", "dirty"}
    assert b1.main(argv) == 1  # never overwrites a run without --resume
    assert b1.main([*argv, "--resume"]) == 0
    assert len((run / "replies.jsonl").read_text().splitlines()) == 8  # nothing generated twice


# B2 with a fake client


def fake_response(body, content=None, prompt_tokens=100, completion_tokens=20, cached=0, finish="stop", refusal=None):
    if content is None:
        schema = body["response_format"]["json_schema"]["schema"]
        content = json.dumps({qid: {"answer": _first(p["properties"]["answer"]), "confidence": 0.7} for qid, p in schema["properties"].items()})
    message = SimpleNamespace(content=content, refusal=refusal)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage, model="fake-model-2026")


def _first(answer_schema):
    return answer_schema["enum"][0] if "enum" in answer_schema else True


class FakeClient:
    def __init__(self, fail_first=0, error_name="RateLimitError", prompt_tokens=100):
        self.bodies, self.fail_first, self.error_name, self.prompt_tokens = [], fail_first, error_name, prompt_tokens

    def create(self, **body):
        if self.fail_first:
            self.fail_first -= 1
            raise type(self.error_name, (Exception,), {})("transient")
        self.bodies.append(body)
        return fake_response(body, prompt_tokens=self.prompt_tokens)


def b2_argv(baseline_data, runs_dir, *extra):
    root, data_dir = baseline_data
    return ["--price-input", "0.4", "--price-output", "1.6", "--splits", "test_indomain", "test_sst5", "--subset", str(root / "subset.json"),
            "--data-dir", str(data_dir), "--runs-dir", str(runs_dir), *extra]


def test_b2_dry_run_prints_three_prompts_without_a_key(baseline_data, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root, data_dir = baseline_data
    argv = ["--dry-run", "--subset", str(root / "subset.json"), "--data-dir", str(data_dir), "--splits", "train", "test_indomain", "test_sst5"]
    assert b2.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("===== prompt ") == 3 and "prompt 1 of 3: test_indomain" in out and "prompt 2 of 3: test_sst5" in out
    assert "12 requests over 3 splits" in out
    assert not (Path(tmp_path) / "runs").exists()


def test_b2_requires_prices_unless_dry_run(baseline_data):
    with pytest.raises(SystemExit):
        b2.parse_args(["--model", "x"])


def test_b2_request_body_uses_the_strict_schema_and_optional_temperature():
    body = b2.request_body(record(), "m", None, 64)
    assert "temperature" not in body and body["max_completion_tokens"] == 64
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == json_schema(request())
    assert body["messages"] == [{"role": "user", "content": build_prompt(request())}]
    assert b2.request_body(record(), "m", 0.0, 64)["temperature"] == 0.0


def test_b2_cost_counts_cached_tokens_at_their_price():
    assert b2.cost_usd(1000, 400, 100, 0.4, 0.1, 1.6) == pytest.approx((600 * 0.4 + 400 * 0.1 + 100 * 1.6) / 1e6)


def test_b2_run_records_tokens_cost_latency_and_metrics(baseline_data, tmp_path):
    client = FakeClient()
    assert b2.main(b2_argv(baseline_data, tmp_path), create=client.create) == 0
    run = tmp_path / "b2_gpt-4.1-mini"
    rows = read_replies(run / "replies.jsonl")
    assert len(rows) == len(client.bodies) == 8
    row = next(iter(rows.values()))
    assert row["cost_usd"] == pytest.approx((100 * 0.4 + 20 * 1.6) / 1e6) and row["latency_s"] >= 0 and row["model"] == "fake-model-2026"
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["complete"] and metrics["usage"]["cost_usd"] == pytest.approx(8 * row["cost_usd"])
    assert metrics["usage"]["cost_per_1000_requests_usd"] == pytest.approx(1000 * row["cost_usd"])
    assert metrics["model"] == {"requested": "gpt-4.1-mini", "reported": ["fake-model-2026"], "reported_counts": {"fake-model-2026": 8}}
    assert metrics["request"]["temperature"] == 0.0  # the default
    assert all(len(r["requested_at"]) == 25 and r["requested_at"].endswith("+00:00") for r in rows.values())
    assert metrics["request_dates_utc"] == {"first": min(r["requested_at"] for r in rows.values()), "last": max(r["requested_at"] for r in rows.values())}
    assert row["prices_usd_per_million_tokens"] == {"input": 0.4, "cached_input": 0.4, "output": 1.6}
    assert metrics["prices_usd_per_million_tokens"] == {"input": 0.4, "cached_input": 0.4, "output": 1.6}
    assert metrics["prices_used_by_recorded_replies"] == [{"input": 0.4, "cached_input": 0.4, "output": 1.6}]
    assert all(b["temperature"] == 0.0 for b in client.bodies)
    assert metrics["splits"]["test_indomain"]["choice"]["parse_failure_rate"] == 0.0
    assert metrics["splits"]["test_indomain"]["choice"]["n_with_confidence"] == 4
    assert row["retries"] == 0 and row["retry_errors"] == []
    latency = metrics["latency"]
    assert latency["first_attempt"]["n"] == 8 and latency["retried_requests"] == 0 and latency["timeout_s"] == 60.0
    assert latency["first_attempt"]["median_ms"] <= latency["first_attempt"]["p95_ms"]


def test_b2_stops_at_the_spend_cap_and_resumes(baseline_data, tmp_path):
    first = FakeClient(prompt_tokens=20000)  # each reply costs more than the worst-case estimate of the next request
    each = b2.cost_usd(20000, 0, 20, 0.4, 0.4, 1.6)
    body = b2.request_body(subset_records(baseline_data[1], load_subset(baseline_data[0] / "subset.json"), "test_sst5")[0][0], "gpt-4.1-mini", 0.0, 1024)
    assert b2.worst_case_usd(body, 0.4, 1.6) < each
    cap = 2 * each  # two requests fit; before a third, the spend plus its worst case is over the cap
    assert b2.main(b2_argv(baseline_data, tmp_path, "--max-usd", str(cap)), create=first.create) == 2
    run = tmp_path / "b2_gpt-4.1-mini"
    metrics = json.loads((run / "metrics_partial.json").read_text())
    assert len(first.bodies) == 2 and metrics["complete"] is False and metrics["spend"]["stopped_by_cap"] is True
    assert not (run / "metrics.json").exists()  # metrics.json only for a complete run
    assert b2.main(b2_argv(baseline_data, tmp_path), create=FakeClient().create) == 1  # no overwrite without --resume
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume", "--sub"), create=FakeClient().create) == 1  # other settings
    second = FakeClient()
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume"), create=second.create) == 0
    assert len(second.bodies) == 6 and len(read_replies(run / "replies.jsonl")) == 8
    assert json.loads((run / "metrics.json").read_text())["complete"] is True


def test_b2_retries_transient_errors_records_them_and_stops_on_others(baseline_data, tmp_path):
    flaky = FakeClient(fail_first=2, error_name="APITimeoutError")
    waits = []
    assert b2.main(b2_argv(baseline_data, tmp_path / "a"), create=flaky.create, sleep=waits.append) == 0
    assert len(flaky.bodies) == 8 and waits == [2, 4]
    run = tmp_path / "a" / "b2_gpt-4.1-mini"
    rows = list(read_replies(run / "replies.jsonl").values())
    assert [r["retries"] for r in rows] == [2] + [0] * 7 and rows[0]["retry_errors"] == ["APITimeoutError", "APITimeoutError"]
    latency = json.loads((run / "metrics.json").read_text())["latency"]
    assert latency["first_attempt"]["n"] == 7 and latency["retried_requests"] == 1 and latency["retries_total"] == 2
    assert latency["retry_errors"] == {"APITimeoutError": 2}
    hopeless = FakeClient(fail_first=100)
    assert b2.main(b2_argv(baseline_data, tmp_path / "c"), create=hopeless.create, sleep=lambda s: None) == 1  # three retries, then stop
    assert hopeless.fail_first == 100 - 4
    broken = FakeClient(fail_first=100, error_name="BadRequestError")
    assert b2.main(b2_argv(baseline_data, tmp_path / "b"), create=broken.create) == 1
    assert not read_replies(tmp_path / "b" / "b2_gpt-4.1-mini" / "replies.jsonl")


def test_b2_refusal_is_a_parse_failure(baseline_data):
    body = b2.request_body(record(), "m", 0.0, 64)
    row = b2.reply_row("test_indomain", "r1", fake_response(body, refusal="I cannot help"), 0.1, (0.4, 0.4, 1.6), b2.utc_now())
    assert row["reply"] is None and row["refusal"] == "I cannot help"
    assert {a.status for a in answers_for_record(record(), "test_indomain", row["reply"])} == {"invalid_json"}


# Comparison


def test_compare_prints_one_row_per_split_and_type_for_every_run(baseline_data, tmp_path, capsys):
    root, data_dir = baseline_data
    subset = load_subset(root / "subset.json")
    jev_dir = tmp_path / "sft_x"
    jev_dir.mkdir()
    results = []
    for split in ("test_indomain", "test_sst5"):
        for rec in subset_records(data_dir, subset, split)[0]:
            for pos, (qid, q) in enumerate(rec["questions"].items()):
                k = 2 if q["type"] == "noul" else len(q["criteria"])
                gold = b2_gold(q, rec["gold"][qid])
                probs = tuple(0.9 if i == gold else 0.1 / (k - 1) for i in range(k))
                results.append(QuestionResult(rec["id"], qid, q["type"], probs, gold, tuple(str(i) for i in range(k)), split=split, kind=qid if q["type"] == "noul" else None, position=pos))
    write_results(jev_dir / "results.jsonl.gz", results)
    (jev_dir / "metrics.json").write_text(json.dumps({"latency": {"batch_1": {"median_ms": 50.0}, "batch_16_requests_per_second": 30.0}}))
    client = FakeClient()
    b2.main(b2_argv(baseline_data, tmp_path), create=client.create)
    capsys.readouterr()
    assert compare.main([str(jev_dir), str(tmp_path / "b2_gpt-4.1-mini"), str(tmp_path / "nothing"), "--subset", str(root / "subset.json"), "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    indomain = next(l for l in lines if l.startswith("test_indomain"))
    assert "1.000 / 0.100" in indomain  # jevmark: every gold at 0.9
    assert "  choice" in out and "  score" in out and "  noul" in out
    assert "missing" in out and "30.0 requests/s" in out and "USD per 1000 requests" in out
    assert "B2 parse failures are near zero by construction" in out and "format reliability is read from B1" in out


def test_compare_defaults_include_both_b1_sizes_next_to_their_jevmark_size():
    runs = list(compare.DEFAULT_RUNS)
    assert runs.index("runs/sft_06b") < runs.index("runs/b1_qwen06b_json") < runs.index("runs/sft_17b") < runs.index("runs/b1_qwen17b_json")
    assert "runs/b2_gpt-4.1-mini" in runs


def b2_gold(question, gold):
    if question["type"] == "noul":
        return {"true": 0, "false": 1}[gold]
    if question["type"] == "choice":
        return list(question["criteria"]).index(gold)
    return int(gold)


def test_a_reply_keyed_by_the_real_ids_counts_as_missing():
    reply = json.dumps({"intent": {"answer": "transfer", "confidence": 0.9}})
    assert {p.status for p in parse_reply(reply, request()).values()} == {"missing"}


def test_every_noul_of_every_subset_record_renders_both_options():
    """Over the committed subset: each noul block lists true then false, and every question shows only its anonymous id."""
    from jevmark.baselines.results import request_of
    from jevmark.data.build import SPLITS

    subset = load_subset()
    data_dir = REPO / "data"
    if not all((data_dir / f"{split}.jsonl").is_file() for split in SPLITS):
        pytest.skip("data/ not built (make data)")
    nouls = 0
    for split in SPLITS:
        for rec in subset_records(data_dir, subset, split)[0]:
            req = request_of(rec)
            prompt = build_prompt(req)
            blocks = prompt.split("\n\nQuestion id: ")[1:]
            assert len(blocks) == len(req.questions)
            for i, (question, block) in enumerate(zip(req.questions, blocks), 1):
                assert block.startswith(f"q{i}\n")
                if question.type == "noul":
                    nouls += 1
                    options = [line for line in block.splitlines() if line.startswith("- ")]
                    assert options == [f"- true: {question.true_description or 'yes'}", f"- false: {question.false_description or 'no'}"], (rec["id"], question.id)
    assert nouls > 4000


def test_b2_latency_counts_old_rows_slower_than_the_timeout_as_retried():
    rows = [{"latency_s": 0.5}, {"latency_s": 0.7}, {"latency_s": 601.8}, {"latency_s": 0.6, "retries": 0}, {"latency_s": 9.0, "retries": 1, "retry_errors": ["RateLimitError"]}]
    latency = b2.latency_summary(rows, 60.0)
    assert latency["first_attempt"]["n"] == 3 and latency["first_attempt"]["median_ms"] == pytest.approx(600.0)
    assert latency["retried_requests"] == 2 and latency["retries_total"] == 1
    assert latency["rows_without_retry_count"] == 3 and latency["rows_without_retry_count_over_timeout"] == 1


def test_b2_client_has_a_60_second_timeout_and_no_retries_of_its_own(baseline_data, tmp_path, monkeypatch):
    made = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            made.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=FakeClient().create))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert b2.main(b2_argv(baseline_data, tmp_path)) == 0
    assert made == [{"timeout": 60.0, "max_retries": 0}]


def test_b2_resume_widens_a_sub_run_and_keeps_its_metrics(baseline_data, tmp_path):
    sub = FakeClient()
    assert b2.main(b2_argv(baseline_data, tmp_path, "--sub"), create=sub.create) == 0
    run = tmp_path / "b2_gpt-4.1-mini"
    sub_metrics = (run / "metrics.json").read_text()
    assert len(sub.bodies) == 4 and json.loads(sub_metrics)["subset"]["records"].startswith("sub_splits")
    full = FakeClient()
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume"), create=full.create) == 0
    subset = load_subset(baseline_data[0] / "subset.json")
    sent = {b["messages"][0]["content"] for b in full.bodies}
    sub_prompts = {b["messages"][0]["content"] for b in sub.bodies}
    assert len(full.bodies) == 4 and not sent & sub_prompts  # only the records the sub run did not have
    assert (run / "metrics_sub200.json").read_text() == sub_metrics
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["complete"] and metrics["n_requests"] == 8 and metrics["subset"]["records"].startswith("splits")
    assert yaml.safe_load((run / "config.yaml").read_text())["sub"] is False
    assert len(subset["sub_splits"]["test_indomain"]) == 2


def test_b2_widening_that_stops_early_leaves_the_sub_metrics_in_place(baseline_data, tmp_path):
    assert b2.main(b2_argv(baseline_data, tmp_path, "--sub"), create=FakeClient().create) == 0
    run = tmp_path / "b2_gpt-4.1-mini"
    sub_metrics = (run / "metrics.json").read_text()
    broken = FakeClient(fail_first=100, error_name="BadRequestError")
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume"), create=broken.create) == 1
    assert (run / "metrics.json").read_text() == sub_metrics  # the full run's metrics.json waits for every reply
    assert json.loads((run / "metrics_partial.json").read_text())["n_requests"] == 4
    assert (run / "metrics_sub200.json").read_text() == sub_metrics
    finish = FakeClient()
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume"), create=finish.create) == 0  # now an ordinary full resume
    assert len(finish.bodies) == 4 and json.loads((run / "metrics.json").read_text())["complete"]
    assert (run / "metrics_sub200.json").read_text() == sub_metrics  # never overwritten


@pytest.mark.parametrize("extra", [["--temperature", "0.5"], ["--max-completion-tokens", "512"], ["--model", "gpt-other", "--run-name", "b2_gpt-4.1-mini"]])
def test_b2_resume_refuses_any_other_config_difference(baseline_data, tmp_path, extra):
    assert b2.main(b2_argv(baseline_data, tmp_path, "--sub"), create=FakeClient().create) == 0
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume", *extra), create=FakeClient().create) == 1


def test_b2_resume_never_narrows_a_full_run_to_the_sub_subset(baseline_data, tmp_path):
    assert b2.main(b2_argv(baseline_data, tmp_path), create=FakeClient().create) == 0
    assert b2.main(b2_argv(baseline_data, tmp_path, "--resume", "--sub"), create=FakeClient().create) == 1


# Lenient reading (decision 48)


def test_lenient_reading_takes_the_label_before_the_first_colon():
    reply = reply_of({
        "char_count_over": {"answer": "true: yes", "confidence": 1.0},
        "intent": {"answer": "transfer: Move money between accounts", "confidence": 0.9},
        "sentiment": {"answer": "1: Neutral", "confidence": 0.8},
        "about_intent": {"answer": "maybe: not sure"},
    })
    strict = parse_reply(reply, request())
    lenient = parse_reply(reply, request(), lenient=True)
    assert {p.status for p in strict.values()} == {"invalid_answer"}
    assert lenient["char_count_over"] == ParsedAnswer("ok", 0, 1.0)
    assert lenient["intent"] == ParsedAnswer("ok", 1, 0.9)
    assert lenient["sentiment"] == ParsedAnswer("ok", 1, 0.8)
    assert lenient["about_intent"].status == "invalid_answer"  # the text before the colon is still not an option


def test_lenient_reading_never_changes_an_answer_the_strict_reading_accepts():
    labels_with_colon = {"q": {"type": "choice", "instructions": "Pick one", "criteria": {"a: b": None, "a": None}}}
    req = Request.from_dict({"state": "x", "questions": labels_with_colon})
    reply = json.dumps({"q1": {"answer": "a: b"}})
    assert parse_reply(reply, req)["q"] == parse_reply(reply, req, lenient=True)["q"] == ParsedAnswer("ok", 0, None)


def test_metrics_report_the_lenient_reading_next_to_the_strict_headline():
    reply = reply_of({"char_count_over": {"answer": "true: yes"}, "intent": {"answer": "transfer"}, "sentiment": {"answer": 1}, "about_intent": {"answer": "false: no"}})
    metrics = baseline_split_metrics(answers_for_record(record(), "test_indomain", reply))
    noul = metrics["noul"]  # about_intent, gold true, answered "false: no"
    assert noul["parse_failure_rate"] == 1.0 and noul["accuracy_all"] == 0.0
    assert noul["lenient"] == {"accuracy_all": 0.0, "parse_failure_rate": 0.0, "recovered": 1}
    form = metrics["form"]  # char_count_over, gold true, answered "true: yes"
    assert form["accuracy_all"] == 0.0 and form["lenient"] == {"accuracy_all": 1.0, "parse_failure_rate": 0.0, "recovered": 1}
    assert metrics["overall"]["accuracy_all"] == pytest.approx(2 / 3) and metrics["overall"]["lenient"]["accuracy_all"] == pytest.approx(2 / 3)


def test_recompute_baseline_metrics_rebuilds_the_splits_and_keeps_run_fields(baseline_data, tmp_path):
    recompute_b = load_script("recompute_baseline_metrics")
    root, data_dir = baseline_data
    assert b2.main(b2_argv(baseline_data, tmp_path, "--sub"), create=FakeClient().create) == 0
    run = tmp_path / "b2_gpt-4.1-mini"
    before = json.loads((run / "metrics.json").read_text())
    assert recompute_b.main([str(run), "--subset", str(root / "subset.json"), "--data-dir", str(data_dir), "--out", str(tmp_path / "re.json")]) == 0
    after = json.loads((tmp_path / "re.json").read_text())
    assert after["splits"] == before["splits"] and after["usage"] == before["usage"] and after["recomputed"]["requests"] == 4
