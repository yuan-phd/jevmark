"""CLINC150 -> records with one choice and one gold-dependent noul each (docs/DATA.md section 2, data v1.3).

Every record asks the intent (choice) first, then exactly one of about_domain,
about_intent or out_of_scope (noul). Attention is causal, so a question placed
before another can prime it; with one gold-dependent noul after the choice, no
question sees another question that depends on the same gold (decision 42).
The builder fixes each noul question's underlying answer with exact counts so every
kind is balanced before negation, then build.assign_phrasings picks the final
template and polarity. A "no" question asks, with probability p_asked_from_options,
a distractor option (about_intent) or a distractor option's domain (about_domain),
so whether the asked item appears among the options does not predict the answer.

The intent-to-domain map is the original CLINC release's domains.json, checked in
unmodified as clinc_domains.json. The domain phrases below are written by hand and
live only here; the noul phrasings live in negation.py.
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

from jevmark.data.build import assign_phrasings, choice_question, make_record, noul_question, split_rng
from jevmark.data.description_loader import LabelDescription, load_descriptions
from jevmark.data.sources import CLINC, CLINC_OUT_OF_SCOPE

DOMAINS_FILE = Path(__file__).resolve().parent / "clinc_domains.json"
DOMAINS_SOURCE = "https://github.com/clinc/oos-eval/blob/976178879e5afa9952f60a1f8d3c834f47a25cee/data/domains.json"
DOMAINS_SHA256 = "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"

# Hand-written; the slot of the about_domain templates in negation.py.
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

    def _pool(self, split: str) -> tuple[list[Utterance], list[str]]:
        """Utterances of a split and the intents allowed as options and asked intents there."""
        held = set(self.held_out)
        if split == "test_unseen_intents":
            pool = [u for hf in ("train", "validation", "test") for u in self.utterances[hf] if u.intent in held]
            return pool, self.held_out
        return [u for u in self.utterances[HF_SPLITS[split]] if u.intent not in held], self.seen

    def _choice(self, rng: random.Random, gold_intent: str | None, allowed: Sequence[str]) -> tuple[dict[str, Any], str, bool, list[str]]:
        """The intent question, its gold answer, whether the gold intent is offered, and the named options."""
        k = rng.randint(int(self.params["k_min"]), int(self.params["k_max"]))
        include_gold = gold_intent is not None and rng.random() < float(self.params["p_gold_in_options"])
        pool = [i for i in allowed if i != gold_intent]
        intents = rng.sample(pool, k - 1 - int(include_gold)) + ([gold_intent] if include_gold else [])
        options = [(i, rng.choice(self.descriptions[i].variants)) for i in intents] + [(OTHER_LABEL, OTHER_DESCRIPTION)]
        gold = gold_intent if include_gold else OTHER_LABEL
        return choice_question(INTENT_INSTRUCTIONS, options, rng), gold, include_gold, intents

    def _intent_noul(self, rng: random.Random, gold_intent: str, asks_gold: bool, named: Sequence[str], allowed: Sequence[str]) -> tuple:
        """about_intent: the gold intent, else a distractor option (p_asked_from_options) or an intent outside the options."""
        distractors = [i for i in named if i != gold_intent]
        outside = [i for i in allowed if i != gold_intent and i not in named]
        if asks_gold:
            asked, source = gold_intent, "gold"
        elif distractors and (rng.random() < float(self.params["p_asked_from_options"]) or not outside):
            asked, source = rng.choice(distractors), "distractor"
        else:
            asked, source = rng.choice(outside), "outside"
        description = rng.choice(self.descriptions[asked].variants)
        return noul_question("about_intent", asks_gold, description, asked_intent=asked, asked_from=source, asked_in_options=asked in named)

    def _domain_noul(self, rng: random.Random, gold_domain: str | None, asks_gold: bool, named: Sequence[str]) -> tuple:
        """about_domain: the gold domain, else a distractor option's domain (p_asked_from_options) or another domain."""
        option_domains = {self.domain_of[i] for i in named}
        distractor_domains = [self.domain_of[i] for i in named if self.domain_of[i] != gold_domain]
        if asks_gold:
            asked, source = gold_domain, "gold"
        elif distractor_domains and rng.random() < float(self.params["p_asked_from_options"]):
            asked, source = rng.choice(distractor_domains), "distractor"
        else:
            asked, source = rng.choice([d for d in sorted(DOMAIN_PHRASES) if d != gold_domain]), "outside"
        return noul_question(
            "about_domain", asks_gold, DOMAIN_PHRASES[asked], asked_domain=asked, asked_from=source, asked_in_options=asked in option_domains
        )

    def build_split(self, split: str) -> list[dict[str, Any]]:
        """Records with the intent question first and exactly one gold-dependent noul after it (decision 42).

        out_of_scope: one of the two records of every out-of-scope utterance (yes), and
        exactly as many in-scope records (no). Every other record draws about_domain or
        about_intent with the configured weights; each of the two asks the gold for exactly
        half of its questions, choosing from the in-scope records.
        """
        rng = split_rng(self.seed, split)
        pool, allowed = self._pool(split)
        in_scope = [i for i, u in enumerate(pool) if u.intent != CLINC_OUT_OF_SCOPE]
        n_oos = len(pool) - len(in_scope)
        if 2 * n_oos > len(in_scope):
            raise RuntimeError(f"{split}: {n_oos} out-of-scope utterances are too many to balance against {len(in_scope)} in-scope ones")
        asks_out_of_scope = set(rng.sample(in_scope, n_oos))
        weights = self.params["kind_weights"]
        p_domain = float(weights["about_domain"]) / (float(weights["about_domain"]) + float(weights["about_intent"]))

        # Draft records: (utterance, choice, kind); the noul question is filled in below.
        drafts: list[tuple[Utterance, tuple, str]] = []
        for index, u in enumerate(pool):
            if u.intent == CLINC_OUT_OF_SCOPE:
                drafts.append((u, self._choice(rng, None, allowed), "out_of_scope"))
                drafts.append((u, self._choice(rng, None, allowed), "about_domain" if rng.random() < p_domain else "about_intent"))
            elif index in asks_out_of_scope:
                drafts.append((u, self._choice(rng, u.intent, allowed), "out_of_scope"))
            else:
                drafts.append((u, self._choice(rng, u.intent, allowed), "about_domain" if rng.random() < p_domain else "about_intent"))

        asks_gold: set[int] = set()
        for kind in ("about_domain", "about_intent"):
            members = [j for j, (_, _, k) in enumerate(drafts) if k == kind]
            candidates = [j for j in members if drafts[j][0].intent != CLINC_OUT_OF_SCOPE]
            if len(candidates) < len(members) // 2:
                raise RuntimeError(f"{split}: too few in-scope {kind} records to ask the gold in half of them")
            asks_gold |= set(rng.sample(candidates, len(members) // 2))

        records: list[dict[str, Any]] = []
        for j, (u, choice, kind) in enumerate(drafts):
            question, choice_gold, gold_in_options, named = choice
            if kind == "out_of_scope":
                noul = noul_question("out_of_scope", u.intent == CLINC_OUT_OF_SCOPE)
            elif kind == "about_intent":
                noul = self._intent_noul(rng, u.intent, j in asks_gold, named, allowed)
            else:
                noul = self._domain_noul(rng, self.domain_of.get(u.intent), j in asks_gold, named)
            noul_q, noul_gold, noul_info = noul
            records.append(
                {
                    "id": f"clinc-{split}-{len(records):06d}",
                    "source": SOURCE,
                    "split": split,
                    "state": u.text,
                    "questions": {"intent": question, kind: noul_q},
                    "gold": {"intent": choice_gold, kind: noul_gold},
                    "meta": {
                        "source_split": u.source_split,
                        "source_index": u.source_index,
                        "gold_intent": u.intent,
                        "domain": self.domain_of.get(u.intent),
                        "gold_in_options": gold_in_options if u.intent != CLINC_OUT_OF_SCOPE else None,
                        "nouls": {kind: noul_info},
                    },
                }
            )

        assign_phrasings(records, split_rng(self.seed, f"{split}:phrasing"))
        return [make_record(r["id"], r["source"], r["split"], r["state"], r["questions"], r["gold"], r["meta"]) for r in records]
