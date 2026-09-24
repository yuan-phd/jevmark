"""CLINC150 -> records with one choice and one noul question each (docs/DATA.md section 2).

The intent-to-domain map is the original CLINC release's domains.json, checked in
unmodified as clinc_domains.json. The domain phrases below are written by hand and
live only here.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from jevmark.data.build import choice_question, make_record, split_rng
from jevmark.data.description_loader import LabelDescription, load_descriptions
from jevmark.data.sources import CLINC, CLINC_OUT_OF_SCOPE

DOMAINS_FILE = Path(__file__).resolve().parent / "clinc_domains.json"
DOMAINS_SOURCE = "https://github.com/clinc/oos-eval/blob/976178879e5afa9952f60a1f8d3c834f47a25cee/data/domains.json"
DOMAINS_SHA256 = "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"

# Hand-written; "Is this message about <phrase>?"
DOMAIN_PHRASES = {
    "banking": "banking, such as accounts, balances, bills or transfers",
    "credit_cards": "credit cards, such as card limits, rewards or credit scores",
    "kitchen_and_dining": "cooking, food or restaurants",
    "home": "home organisation, such as lists, reminders, calendars, music or orders",
    "auto_and_commute": "cars, driving or getting around",
    "travel": "travel, such as flights, hotels or trip preparation",
    "utility": "everyday utilities, such as the time, weather, timers or calls",
    "work": "work, such as meetings, time off, pay or benefits",
    "small_talk": "small talk with the assistant about itself",
    "meta": "the conversation or the assistant's settings, such as yes or no replies, volume or language",
}

INTENT_INSTRUCTIONS = "Which intent does this message express?"
DOMAIN_INSTRUCTIONS = "Is this message about {phrase}?"
OUT_OF_SCOPE_INSTRUCTIONS = "Is this request outside what a banking, travel, home, work or everyday assistant can help with?"
OTHER_LABEL = "other"
OTHER_DESCRIPTION = "None of the listed intents"
SOURCE = f"{CLINC.id}/{CLINC.config}"
HF_SPLITS = {"train": "train", "valid": "validation", "test_indomain": "test"}


def load_domains() -> dict[str, list[str]]:
    """Domain -> intents, from the checked-in domains.json; fails if the file changed."""
    raw = DOMAINS_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DOMAINS_SHA256:
        raise RuntimeError(f"{DOMAINS_FILE} does not match the file at {DOMAINS_SOURCE}")
    domains = json.loads(raw)
    if set(domains) != set(DOMAIN_PHRASES):
        raise RuntimeError(f"domains.json domains {sorted(domains)} differ from DOMAIN_PHRASES")
    return domains


def intent_domains(intents: Sequence[str]) -> dict[str, str]:
    """Intent -> domain; fails unless domains.json maps exactly these intents, each to one domain."""
    mapping = {intent: domain for domain, members in load_domains().items() for intent in members}
    count = sum(len(members) for members in load_domains().values())
    if count != len(mapping) or set(mapping) != set(intents):
        raise RuntimeError("domains.json does not map exactly the dataset's intents to one domain each")
    return mapping


def choose_held_out(domains: Mapping[str, Sequence[str]], per_domain: int, seed: int) -> list[str]:
    rng = split_rng(seed, "held_out")
    return sorted(i for domain in sorted(domains) for i in rng.sample(sorted(domains[domain]), per_domain))


@dataclass(frozen=True)
class Utterance:
    text: str
    intent: str  # CLINC_OUT_OF_SCOPE for out-of-scope
    source_split: str
    source_index: int


def load_utterances() -> dict[str, list[Utterance]]:
    """HF split name -> utterances at the pinned revision, in dataset order."""
    dataset = load_dataset(CLINC.id, CLINC.config, revision=CLINC.revision)
    names = dataset["train"].features[CLINC.label_column].names
    return {
        hf_split: [Utterance(row["text"], names[row[CLINC.label_column]], hf_split, i) for i, row in enumerate(part)]
        for hf_split, part in dataset.items()
    }


class ClincBuilder:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.seed = int(config["seed"])
        self.params = config["clinc"]
        self.utterances = load_utterances()
        self.intents = sorted({u.intent for part in self.utterances.values() for u in part} - {CLINC_OUT_OF_SCOPE})
        self.domain_of = intent_domains(self.intents)
        self.held_out = choose_held_out(load_domains(), int(self.params["held_out_per_domain"]), self.seed)
        self.seen = [i for i in self.intents if i not in set(self.held_out)]
        self.descriptions: dict[str, LabelDescription] = load_descriptions("clinc")
        self.p_negated = float(self.params["p_negated"])
        self.p_out_of_scope = self._out_of_scope_rate(int(self.params["out_of_scope_questions_train"]))

    def _pool(self, split: str) -> tuple[list[Utterance], list[str]]:
        """Utterances of a split and the intents allowed as options there."""
        held = set(self.held_out)
        if split == "test_unseen_intents":
            pool = [u for hf in ("train", "validation", "test") for u in self.utterances[hf] if u.intent in held]
            return pool, self.held_out
        return [u for u in self.utterances[HF_SPLITS[split]] if u.intent not in held], self.seen

    def _out_of_scope_rate(self, target_train: int) -> float:
        """In-scope rate p of out_of_scope questions, set so train has about target_train of them (decision 40).

        Each out-of-scope utterance contributes one out_of_scope question, so p =
        (target - N_oos) / N_in on train; the same p is used in every split.
        """
        pool, _ = self._pool("train")
        n_oos = sum(u.intent == CLINC_OUT_OF_SCOPE for u in pool)
        n_in = len(pool) - n_oos
        p = (target_train - n_oos) / n_in
        if not 0.0 <= p < 1.0:
            raise RuntimeError(f"out_of_scope target {target_train} needs in-scope rate {p:.4f}, outside [0, 1)")
        return p

    def _choice(self, rng: random.Random, gold_intent: str | None, allowed: Sequence[str]) -> tuple[dict[str, Any], str, bool]:
        k = rng.randint(int(self.params["k_min"]), int(self.params["k_max"]))
        include_gold = gold_intent is not None and rng.random() < float(self.params["p_gold_in_options"])
        pool = [i for i in allowed if i != gold_intent]
        intents = rng.sample(pool, k - 1 - int(include_gold)) + ([gold_intent] if include_gold else [])
        options = [(i, rng.choice(self.descriptions[i].variants)) for i in intents] + [(OTHER_LABEL, OTHER_DESCRIPTION)]
        gold = gold_intent if include_gold else OTHER_LABEL
        return choice_question(INTENT_INSTRUCTIONS, options, rng), gold, include_gold

    def _record(
        self,
        split: str,
        index: int,
        u: Utterance,
        choice: tuple[dict[str, Any], str, bool],
        noul: tuple[str, dict[str, Any], str],
        noul_first: bool,
        negated: bool,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        from jevmark.data.negation import negate  # negation.py imports this module's instruction texts

        question, choice_gold, gold_in_options = choice
        noul_id, noul_question, noul_gold = noul
        if negated:
            noul_question = {**noul_question, "instructions": negate(noul_id, noul_question["instructions"])}
            noul_gold = {"true": "false", "false": "true"}[noul_gold]
        pairs = [("intent", question, choice_gold), (noul_id, noul_question, noul_gold)]
        if noul_first:
            pairs.reverse()
        return make_record(
            f"clinc-{split}-{index:06d}",
            SOURCE,
            split,
            u.text,
            {qid: q for qid, q, _ in pairs},
            {qid: g for qid, _, g in pairs},
            {
                "source_split": u.source_split,
                "source_index": u.source_index,
                "gold_intent": u.intent,
                "domain": self.domain_of.get(u.intent),
                "gold_in_options": gold_in_options if u.intent != CLINC_OUT_OF_SCOPE else None,
                "noul_kind": noul_id,
                "negated": negated,
                **meta,
            },
        )

    def build_split(self, split: str) -> list[dict[str, Any]]:
        rng = split_rng(self.seed, split)
        negation_rng = split_rng(self.seed, f"{split}:negation")  # its own stream, so negation shifts no other draw
        pool, allowed = self._pool(split)
        n_oos = sum(u.intent == CLINC_OUT_OF_SCOPE for u in pool)
        n_in = len(pool) - n_oos
        p = self.p_out_of_scope
        # q balances about_domain's underlying answers: about A = N_in (1 - p) in-scope records ask it,
        # and every out-of-scope utterance adds one "false"; q A = (1 - q) A + N_oos.
        in_scope_about_domain = n_in * (1 - p)
        if in_scope_about_domain < n_oos:
            raise RuntimeError(f"{split}: {n_oos} out-of-scope utterances exceed {in_scope_about_domain:.0f} in-scope about_domain questions")
        q = (in_scope_about_domain + n_oos) / (2 * in_scope_about_domain)
        domains = sorted(DOMAIN_PHRASES)

        def negated() -> bool:
            return negation_rng.random() < self.p_negated

        records: list[dict[str, Any]] = []
        for u in pool:
            if u.intent == CLINC_OUT_OF_SCOPE:
                # Two records: out_of_scope (true), then about_domain for a uniform domain (false).
                choice = self._choice(rng, None, allowed)
                records.append(self._record(split, len(records), u, choice, out_of_scope_question("true"), rng.random() < 0.5, negated(), {}))
                asked = rng.choice(domains)
                choice = self._choice(rng, None, allowed)
                noul = domain_question(asked, "false")
                records.append(self._record(split, len(records), u, choice, noul, rng.random() < 0.5, negated(), {"asked_domain": asked}))
                continue
            choice = self._choice(rng, u.intent, allowed)
            if rng.random() < p:
                noul, meta = out_of_scope_question("false"), {}
            else:
                gold_domain = self.domain_of[u.intent]
                asked = gold_domain if rng.random() < q else rng.choice([d for d in domains if d != gold_domain])
                noul, meta = domain_question(asked, "true" if asked == gold_domain else "false"), {"asked_domain": asked}
            records.append(self._record(split, len(records), u, choice, noul, rng.random() < 0.5, negated(), meta))
        return records


def out_of_scope_question(gold: str) -> tuple[str, dict[str, Any], str]:
    return "out_of_scope", {"type": "noul", "instructions": OUT_OF_SCOPE_INSTRUCTIONS}, gold


def domain_question(domain: str, gold: str) -> tuple[str, dict[str, Any], str]:
    return "about_domain", {"type": "noul", "instructions": DOMAIN_INSTRUCTIONS.format(phrase=DOMAIN_PHRASES[domain])}, gold
