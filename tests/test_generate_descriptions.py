"""scripts/generate_descriptions.py with fake label sets and a fake client: no network, no API."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_descriptions", REPO / "scripts" / "generate_descriptions.py")
gen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gen  # dataclasses look up their module while the file executes
spec.loader.exec_module(gen)

CLINC = gen.LabelSet("clinc", "CLINC150 intent classification", "a request", ("transfer", "freeze_account", "balance"), 2, "clinc_intents.json", "clinc/clinc_oos/plus@sha")
EMOTION = gen.LabelSet("emotion", "Emotion classification", "a tweet", ("joy", "anger"), 0, "emotion_labels.json", "dair-ai/emotion@sha")

REAL_ARGS = ["--model", "test-model", "--usd-per-million-input", "0.15", "--usd-per-million-output", "0.6"]


class FakeClient:
    """Mimics client.chat.completions.create; replies come from a function of the messages."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.reply(kwargs["messages"])
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def good_reply(messages):
    label = messages[1]["content"].split("Label to describe: ")[1].split("\n")[0]
    reply = {"canonical": f"A message about {label}."}
    if '"paraphrases"' in messages[1]["content"]:
        reply["paraphrases"] = [f"Something concerning {label}.", f"The topic is {label}."]
    return json.dumps(reply)


def test_dry_run_prints_three_prompts_and_makes_no_call(tmp_path, capsys):
    args = gen.parse_args(["--dry-run"])
    gen.dry_run([CLINC, EMOTION], args)
    out = capsys.readouterr().out
    assert out.count("===== ") == 3
    assert "===== clinc / transfer =====" in out and "===== emotion / joy =====" in out
    assert "===== clinc / freeze_account =====" in out
    assert "no API call made" in out
    assert "Label to describe: transfer" in out
    assert "All labels in this dataset, for contrast: transfer, freeze_account, balance" in out
    assert list(tmp_path.iterdir()) == []


def test_dry_run_needs_no_model_key_or_prices(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = gen.parse_args(["--dry-run", "--limit", "2"])
    assert args.dry_run and args.model is None and args.limit == 2


def test_real_run_requires_model_and_prices():
    with pytest.raises(SystemExit):
        gen.parse_args([])
    with pytest.raises(SystemExit):
        gen.parse_args(["--model", "m"])


def test_real_run_without_key_stops(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        gen.make_client()


def test_prompts_ask_for_paraphrases_only_for_clinc():
    clinc_user = gen.build_messages(CLINC, "transfer")[1]["content"]
    emotion_user = gen.build_messages(EMOTION, "joy")[1]["content"]
    assert clinc_user.endswith(gen.TASK_WITH_PARAPHRASES)
    assert emotion_user.endswith(gen.TASK_CANONICAL)


def test_prompt_text_is_documented_in_data_md():
    data_md = (REPO / "docs" / "DATA.md").read_text()
    for text in (gen.SYSTEM_PROMPT, gen.TASK_CANONICAL, gen.TASK_WITH_PARAPHRASES, *gen.USER_TEMPLATE.split("\n")):
        assert text in data_md, text


def test_generate_writes_json_limit_and_resume(tmp_path):
    client = FakeClient(good_reply)
    args = gen.parse_args([*REAL_ARGS, "--limit", "2"])
    gen.generate([CLINC, EMOTION], client, args, tmp_path, log=lambda _: None)
    assert len(client.calls) == 4
    clinc = json.loads((tmp_path / "clinc_intents.json").read_text())
    assert clinc["source"] == CLINC.source
    assert clinc["generator"]["model"] == "test-model"
    assert list(clinc["descriptions"]) == ["transfer", "freeze_account"]
    assert len(clinc["descriptions"]["transfer"]["paraphrases"]) == 2
    emotion = json.loads((tmp_path / "emotion_labels.json").read_text())
    assert emotion["descriptions"]["joy"] == {"canonical": "A message about joy.", "paraphrases": []}

    # A rerun without --limit only asks for the one missing CLINC label.
    gen.generate([CLINC, EMOTION], client, gen.parse_args(REAL_ARGS), tmp_path, log=lambda _: None)
    assert len(client.calls) == 5
    assert list(json.loads((tmp_path / "clinc_intents.json").read_text())["descriptions"]) == ["transfer", "freeze_account", "balance"]


def test_resume_with_different_model_is_refused(tmp_path):
    gen.generate([EMOTION], FakeClient(good_reply), gen.parse_args(REAL_ARGS), tmp_path, log=lambda _: None)
    other = gen.parse_args(["--model", "other-model", *REAL_ARGS[2:]])
    with pytest.raises(SystemExit, match="--overwrite"):
        gen.generate([EMOTION], FakeClient(good_reply), other, tmp_path, log=lambda _: None)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"canonical": "two\nlines", "paraphrases": ["a b", "c d"]}),
        json.dumps({"canonical": "   ", "paraphrases": ["a b", "c d"]}),
        json.dumps({"canonical": "Fine.", "paraphrases": ["only one"]}),
        json.dumps({"canonical": "Fine.", "paraphrases": ["fine.", "Other."]}),
        json.dumps({"canonical": "x" * 201, "paraphrases": ["a b", "c d"]}),
    ],
)
def test_invalid_replies_are_rejected(content):
    with pytest.raises(ValueError):
        gen.parse_reply(content, paraphrases=2)


def test_edge_whitespace_is_stripped():
    assert gen.parse_reply(json.dumps({"canonical": "  Joyful message. "}), paraphrases=0)["canonical"] == "Joyful message."


def test_retries_then_fails_and_keeps_progress(tmp_path):
    def reply(messages):
        return "not json" if "Label to describe: anger" in messages[1]["content"] else good_reply(messages)

    client = FakeClient(reply)
    with pytest.raises(RuntimeError, match="anger"):
        gen.generate([EMOTION], client, gen.parse_args([*REAL_ARGS, "--retries", "2"]), tmp_path, log=lambda _: None)
    assert len(client.calls) == 1 + 3
    assert list(json.loads((tmp_path / "emotion_labels.json").read_text())["descriptions"]) == ["joy"]


def test_spend_cap_stops_the_run(tmp_path):
    # Each fake call costs (1000 * 0.15 + 100 * 0.6) / 1e6 = $0.00021; a $0.0005 cap allows 3 calls.
    client = FakeClient(good_reply)
    args = gen.parse_args([*REAL_ARGS, "--max-usd", "0.0005"])
    with pytest.raises(gen.BudgetExceeded):
        gen.generate([CLINC, EMOTION], client, args, tmp_path, log=lambda _: None)
    assert len(client.calls) == 3
