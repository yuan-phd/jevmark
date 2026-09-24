"""Generate one-line option descriptions for dataset labels with the OpenAI API (docs/DATA.md section 6).

Writes one JSON file per dataset under jevmark/data/descriptions/: a canonical
description for every label, plus two paraphrases per label for CLINC150. The
files are generated once and checked in; the human reviews 20 sampled lines
before they are used.

Real run (the human runs this; the key comes from OPENAI_API_KEY, .env is never committed):

    uv run --extra baselines --env-file .env python scripts/generate_descriptions.py \\
        --model <model> --temperature 0.3 \\
        --usd-per-million-input <price> --usd-per-million-output <price>

Preview without any API call: add --dry-run (no key, prices or openai package needed).

Every output is checked against the API_SPEC line rules and retried at most
--retries times. Progress is saved after each label, so a rerun resumes where the
last one stopped; --overwrite starts over. Spend is computed from the token usage
the API reports and the run stops before it passes --max-usd.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "jevmark" / "data" / "descriptions"
PROMPT_VERSION = 1
MAX_DESCRIPTION_CHARS = 200

SYSTEM_PROMPT = (
    "You write option descriptions for a text classifier. The classifier reads a message and picks one "
    "label from a list, using each label's description to decide. A good description is one plain English "
    "sentence of 6 to 16 words that states what a message with this label is about or asks for, precise "
    "enough to tell the label apart from the other labels in the dataset. Do not use line breaks or "
    "quotation marks. Reply with a JSON object only."
)

USER_TEMPLATE = (
    "Dataset: {title}\n"
    "Each example is {subject}.\n"
    "Label to describe: {label}\n"
    "All labels in this dataset, for contrast: {all_labels}\n"
    "\n"
    "{task}"
)

TASK_CANONICAL = 'Return {"canonical": "<description>"}.'
TASK_WITH_PARAPHRASES = (
    'Return {"canonical": "<description>", "paraphrases": ["<paraphrase 1>", "<paraphrase 2>"]}. '
    "Each paraphrase says the same thing as the canonical description in different words."
)


@dataclass(frozen=True)
class LabelSet:
    key: str
    title: str
    subject: str
    labels: tuple[str, ...]
    paraphrases: int
    filename: str
    source: str


# key -> (title, subject, paraphrases per label, output file)
DATASETS: dict[str, tuple[str, str, int, str]] = {
    "clinc": ("CLINC150 intent classification", "a request or message sent to a virtual assistant", 2, "clinc_intents.json"),
    "banking77": ("Banking77 intent classification", "a customer message sent to a bank's support chat", 0, "banking77_labels.json"),
    "ag_news": ("AG News topic classification", "a news article headline and summary", 0, "ag_news_labels.json"),
    "emotion": ("Emotion classification", "a short personal message, such as a tweet", 0, "emotion_labels.json"),
}


def load_label_sets(keys: Sequence[str]) -> list[LabelSet]:
    """Label names from the pinned datasets (metadata only). CLINC's out-of-scope class is not a label to describe."""
    from jevmark.data.sources import CLINC_OUT_OF_SCOPE, SOURCES, label_names

    label_sets = []
    for key in keys:
        title, subject, paraphrases, filename = DATASETS[key]
        source = SOURCES[key]
        labels = tuple(name for name in label_names(source) if not (key == "clinc" and name == CLINC_OUT_OF_SCOPE))
        label_sets.append(LabelSet(key, title, subject, labels, paraphrases, filename, source.pinned))
    return label_sets


def build_messages(label_set: LabelSet, label: str) -> list[dict[str, str]]:
    task = TASK_WITH_PARAPHRASES if label_set.paraphrases else TASK_CANONICAL
    user = USER_TEMPLATE.format(
        title=label_set.title,
        subject=label_set.subject,
        label=label,
        all_labels=", ".join(label_set.labels),
        task=task,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _clean_line(text: Any, what: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{what} is not a string")
    text = text.strip()
    if not text:
        raise ValueError(f"{what} is empty")
    if "\n" in text or "\r" in text:
        raise ValueError(f"{what} contains a line break")
    if len(text) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"{what} is longer than {MAX_DESCRIPTION_CHARS} characters")
    return text


def parse_reply(content: str, paraphrases: int) -> dict[str, Any]:
    """Validated {"canonical": str, "paraphrases": [str, ...]}; raises ValueError on anything else."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as err:
        raise ValueError(f"reply is not JSON ({err})") from None
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    canonical = _clean_line(data.get("canonical"), "canonical")
    variants: list[str] = []
    if paraphrases:
        raw = data.get("paraphrases")
        if not isinstance(raw, list) or len(raw) != paraphrases:
            raise ValueError(f"expected {paraphrases} paraphrases")
        variants = [_clean_line(p, f"paraphrase {i + 1}") for i, p in enumerate(raw)]
        if len({v.lower() for v in [canonical, *variants]}) != paraphrases + 1:
            raise ValueError("paraphrases repeat each other or the canonical description")
    return {"canonical": canonical, "paraphrases": variants}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Spend:
    usd_per_million_input: float
    usd_per_million_output: float
    max_usd: float
    usd: float = 0.0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.usd += (input_tokens * self.usd_per_million_input + output_tokens * self.usd_per_million_output) / 1e6

    def check(self) -> None:
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.usd:.4f} of the ${self.max_usd:.2f} cap after {self.calls} calls; stopping")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_existing(path: Path, label_set: LabelSet, args: argparse.Namespace) -> dict[str, Any]:
    fresh = {
        "source": label_set.source,
        "generator": {
            "script": "scripts/generate_descriptions.py",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "temperature": args.temperature,
        },
        "descriptions": {},
    }
    if args.overwrite or not path.exists():
        return fresh
    existing = json.loads(path.read_text())
    if existing.get("generator") != fresh["generator"] or existing.get("source") != fresh["source"]:
        raise SystemExit(
            f"{path} was generated with different settings {existing.get('generator')}; "
            "pass --overwrite to regenerate it"
        )
    return existing


def generate(
    label_sets: Sequence[LabelSet],
    client: Any,
    args: argparse.Namespace,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> Spend:
    spend = Spend(args.usd_per_million_input, args.usd_per_million_output, args.max_usd)
    for label_set in label_sets:
        path = out_dir / label_set.filename
        data = _load_existing(path, label_set, args)
        todo = [label for label in label_set.labels[: args.limit] if label not in data["descriptions"]]
        log(f"{label_set.key}: {len(todo)} labels to generate, {len(data['descriptions'])} already in {path.name}")
        for label in todo:
            messages = build_messages(label_set, label)
            for attempt in range(args.retries + 1):
                spend.check()
                reply = client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                spend.add(reply.usage.prompt_tokens, reply.usage.completion_tokens)
                try:
                    data["descriptions"][label] = parse_reply(reply.choices[0].message.content, label_set.paraphrases)
                    break
                except ValueError as err:
                    log(f"  {label}: attempt {attempt + 1} rejected ({err})")
            else:
                raise RuntimeError(f"{label_set.key}/{label}: no valid reply after {args.retries + 1} attempts")
            _write_json(path, data)
        log(f"{label_set.key}: done, spent ${spend.usd:.4f} over {spend.calls} calls so far")
    return spend


def dry_run(label_sets: Sequence[LabelSet], args: argparse.Namespace, log: Callable[[str], None] = print) -> None:
    """Print the exact messages for three labels (round robin over the datasets) and the planned call count."""
    picks: list[tuple[LabelSet, str]] = []
    for index in range(max(len(s.labels) for s in label_sets)):
        for label_set in label_sets:
            if index < len(label_set.labels[: args.limit]) and len(picks) < 3:
                picks.append((label_set, label_set.labels[index]))
    for label_set, label in picks:
        log(f"===== {label_set.key} / {label} =====")
        for message in build_messages(label_set, label):
            log(f"--- {message['role']} ---")
            log(message["content"])
        log("")
    for label_set in label_sets:
        n = len(label_set.labels[: args.limit])
        log(f"planned: {label_set.key}: {n} labels -> {label_set.filename} ({label_set.paraphrases} paraphrases each)")
    log(f"planned: {sum(len(s.labels[: args.limit]) for s in label_sets)} calls before retries; no API call made")


def make_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; export it or pass --env-file .env to uv run")
    from openai import OpenAI  # only needed for a real run: uv run --extra baselines

    return OpenAI()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="OpenAI model name, for example a mini-tier model")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N labels of each dataset")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-usd", type=float, default=5.0, help="hard spend cap (TASKS.md 1.5)")
    parser.add_argument("--usd-per-million-input", type=float)
    parser.add_argument("--usd-per-million-output", type=float)
    parser.add_argument("--overwrite", action="store_true", help="regenerate files instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="print prompts for three labels; no API call")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.dry_run:
        missing = [
            flag
            for flag, value in (
                ("--model", args.model),
                ("--usd-per-million-input", args.usd_per_million_input),
                ("--usd-per-million-output", args.usd_per_million_output),
            )
            if value is None
        ]
        if missing:
            parser.error(f"a real run needs {', '.join(missing)} (current prices, so the spend cap is real)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    label_sets = load_label_sets(args.datasets)
    if args.dry_run:
        dry_run(label_sets, args)
        return 0
    spend = generate(label_sets, make_client(), args, OUT_DIR)
    print(f"total: ${spend.usd:.4f} over {spend.calls} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
