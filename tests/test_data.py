"""Data builders against docs/DATA.md section 2, on an in-memory build with configs/data.yaml."""

import hashlib
import random
from collections import Counter
from pathlib import Path

import pytest

from jevmark.config import load_config
from jevmark.data.assemble import build_all
from jevmark.data.build import SPLITS, gold_positions, held_out_leaks, noul_balance, position_deviations
from jevmark.data.clinc import DOMAIN_PHRASES, DOMAINS_FILE, DOMAINS_SHA256, choose_held_out, intent_domains, load_domains
from jevmark.data.description_loader import load_descriptions, load_overrides
from jevmark.data.sources import CLINC, CLINC_OUT_OF_SCOPE, label_names
from jevmark.data.sst5 import LEVELS
from jevmark.encode import encode
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "configs" / "data.yaml")
EXPECTED_SIZES = {
    "train": 13000 + 2 * 250 + 8544,
    "valid": 2600 + 2 * 100 + 1101,
    "test_indomain": 3900 + 2 * 1000,
    "test_unseen_intents": 20 * 150,
    "test_sst5": 2210,
    "test_agnews": 1000,
    "test_emotion": 1000,
    "test_banking77": 1000,
}


@pytest.fixture(scope="module")
def built():
    return build_all(CONFIG)


def clinc_records(records):
    return [r for r in records if r["source"] == f"{CLINC.id}/{CLINC.config}"]


# domains.json and held-out intents


def test_domains_file_is_the_pinned_original_and_maps_150_intents_to_10_domains():
    assert hashlib.sha256(DOMAINS_FILE.read_bytes()).hexdigest() == DOMAINS_SHA256
    domains = load_domains()
    assert len(domains) == 10 and set(domains) == set(DOMAIN_PHRASES)
    assert all(len(members) == 15 for members in domains.values())
    intents = sorted(set(label_names(CLINC)) - {CLINC_OUT_OF_SCOPE})
    mapping = intent_domains(intents)
    assert len(mapping) == 150 and set(mapping.values()) == set(domains)


def test_held_out_is_20_intents_two_per_domain_and_seeded(built):
    domain_of = intent_domains(sorted(set(label_names(CLINC)) - {CLINC_OUT_OF_SCOPE}))
    assert len(built.held_out) == 20
    assert Counter(domain_of[i] for i in built.held_out) == Counter({d: 2 for d in DOMAIN_PHRASES})
    assert list(built.held_out) == choose_held_out(load_domains(), 2, CONFIG["seed"])
    assert choose_held_out(load_domains(), 2, CONFIG["seed"] + 1) != list(built.held_out)


def test_no_held_out_intent_in_train_or_valid(built):
    assert held_out_leaks(built.splits["train"] + built.splits["valid"], built.held_out) == []
    assert not set(built.held_out) & set(built.seen)


def test_unseen_intents_split_offers_only_held_out_intents(built):
    held = set(built.held_out)
    for record in built.splits["test_unseen_intents"]:
        assert record["meta"]["gold_intent"] in held
        assert set(record["questions"]["intent"]["criteria"]) <= held | {"other"}


# Sizes, validity, encoding


def test_split_sizes(built):
    assert {split: len(records) for split, records in built.splits.items()} == EXPECTED_SIZES


def test_record_ids_are_unique(built):
    ids = [r["id"] for records in built.splits.values() for r in records]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("split", SPLITS)
def test_sample_of_200_records_validates_and_encodes_within_1024(built, tokenizer, split):
    records = built.splits[split]
    for record in random.Random(0).sample(records, min(200, len(records))):
        assert record["split"] == split
        assert set(record) == {"id", "source", "split", "state", "questions", "gold", "meta"}
        request = Request.from_dict({"state": record["state"], "questions": record["questions"]})
        assert len(encode(request, tokenizer, max_tokens=1024).input_ids) <= 1024
        for qid, question in record["questions"].items():
            gold = record["gold"][qid]
            if question["type"] == "choice":
                assert gold in question["criteria"]
            elif question["type"] == "noul":
                assert gold in ("true", "false")
            else:
                assert 0 <= gold < len(question["criteria"])


# Balance and positions


@pytest.mark.parametrize("split", SPLITS)
def test_noul_yes_share_within_40_60(built, split):
    for kind, counts in noul_balance(built.splits[split]).items():
        share = counts["true"] / (counts["true"] + counts["false"])
        assert 0.4 <= share <= 0.6, (split, kind, counts)


def test_noul_kinds_present_where_expected(built):
    assert set(noul_balance(built.splits["train"])) == {"about_domain", "out_of_scope"}
    assert set(noul_balance(built.splits["test_indomain"])) == {"about_domain", "out_of_scope"}
    assert set(noul_balance(built.splits["test_unseen_intents"])) == {"about_domain"}


@pytest.mark.parametrize("split", SPLITS)
def test_gold_position_close_to_uniform_given_k(built, split):
    deviations = position_deviations(gold_positions(built.splits[split]))
    assert all(sd <= 4.0 for sd in deviations.values()), deviations


# CLINC record structure


def test_clinc_records_have_one_choice_and_one_noul_in_both_orders(built):
    orders = Counter()
    for record in clinc_records(built.splits["train"]):
        types = [q["type"] for q in record["questions"].values()]
        assert sorted(types) == ["choice", "noul"]
        orders[types[0]] += 1
    assert 0.45 < orders["choice"] / sum(orders.values()) < 0.55


def test_clinc_gold_answers_follow_the_rules(built):
    for split in ("train", "valid", "test_indomain", "test_unseen_intents"):
        for record in clinc_records(built.splits[split]):
            meta, gold, questions = record["meta"], record["gold"], record["questions"]
            k = len(questions["intent"]["criteria"])
            assert CONFIG["clinc"]["k_min"] <= k <= CONFIG["clinc"]["k_max"] and "other" in questions["intent"]["criteria"]
            if meta["gold_intent"] == CLINC_OUT_OF_SCOPE:
                assert gold["intent"] == "other"
            else:
                assert gold["intent"] == (meta["gold_intent"] if meta["gold_in_options"] else "other")
            if meta["noul_kind"] == "out_of_scope":
                assert gold["out_of_scope"] == ("true" if meta["gold_intent"] == CLINC_OUT_OF_SCOPE else "false")
            else:
                assert gold["about_domain"] == ("true" if meta["asked_domain"] == meta["domain"] else "false")
                assert DOMAIN_PHRASES[meta["asked_domain"]] in questions["about_domain"]["instructions"]


def test_out_of_scope_utterances_yield_two_records(built):
    oos = [r for r in clinc_records(built.splits["train"]) if r["meta"]["gold_intent"] == CLINC_OUT_OF_SCOPE]
    per_utterance = Counter((r["meta"]["source_split"], r["meta"]["source_index"]) for r in oos)
    assert len(per_utterance) == 250 and set(per_utterance.values()) == {2}
    assert Counter(r["meta"]["noul_kind"] for r in oos) == {"out_of_scope": 250, "about_domain": 250}


def test_descriptions_come_from_the_loader_with_overrides(built):
    clinc = load_descriptions("clinc")
    overridden = set(load_overrides()["clinc"])
    seen_override = False
    for record in clinc_records(built.splits["train"]):
        for label, description in record["questions"]["intent"]["criteria"].items():
            if label == "other":
                continue
            assert description in clinc[label].variants
            seen_override |= label in overridden
    assert seen_override


# SST-5 and unseen sets


def test_sst5_records(built):
    records = [r for r in built.splits["train"] if r["source"] == "SetFit/sst5"]
    assert len(records) == 8544
    for record in records[:200]:
        assert record["questions"] == {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of this text?", "criteria": list(LEVELS)}}
        assert record["gold"]["sentiment"] in range(5)


def test_unseen_sets_options(built):
    for record in built.splits["test_agnews"]:
        assert sorted(record["questions"]["label"]["criteria"]) == sorted(["World", "Sports", "Business", "Sci/Tech"])
    for record in built.splits["test_emotion"]:
        assert len(record["questions"]["label"]["criteria"]) == 6
    for record in built.splits["test_banking77"]:
        criteria = record["questions"]["label"]["criteria"]
        assert len(criteria) == 10 and "other" in criteria and record["gold"]["label"] in criteria
        assert record["gold"]["label"] != "other"


def test_build_is_deterministic():
    first = build_all(CONFIG, ["test_agnews", "test_unseen_intents"])
    second = build_all(CONFIG, ["test_agnews", "test_unseen_intents"])
    assert first.splits == second.splits
    other_seed = build_all({**CONFIG, "seed": 1}, ["test_agnews"])
    assert other_seed.splits["test_agnews"] != first.splits["test_agnews"]


# Config overrides


def test_config_overrides():
    config = load_config(REPO / "configs" / "data.yaml", ["seed=3", "clinc.k_max=8", "noul_balance=[0.3, 0.7]"])
    assert config["seed"] == 3 and config["clinc"]["k_max"] == 8 and config["noul_balance"] == [0.3, 0.7]
    for bad in (["seed"], ["nope=1"], ["seed.x=1"]):
        with pytest.raises(ValueError):
            load_config(REPO / "configs" / "data.yaml", bad)
