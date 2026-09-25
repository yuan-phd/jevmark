"""Data builders against docs/DATA.md section 2, on an in-memory build with configs/data.yaml."""

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from jevmark.config import load_config
from jevmark.data.assemble import build_all
from jevmark.data.build import SPLITS, gold_positions, held_out_leaks, is_form, noul_balance, order_violations, position_deviations, state_text
from jevmark.data.form import FORM_KINDS, STATE_FIELDS, answer
from jevmark.data.clinc import DOMAIN_PHRASES, DOMAINS_FILE, DOMAINS_SHA256, choose_held_out, intent_domains, load_domains
from jevmark.data.description_loader import load_descriptions, load_overrides
from jevmark.data.negation import parse
from jevmark.data.build import noul_phrasing
from jevmark.data.sources import CLINC, CLINC_OUT_OF_SCOPE, label_names
from jevmark.data.sst5 import LABEL_TEXTS, LEVELS, LEVELS_3, TO_3_LEVELS
from jevmark.encode import encode
from jevmark.schema import Request

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "configs" / "data.yaml")
# train and valid drop the texts that also occur where a test split is drawn from (decision 43):
# 18 CLINC records and 3 SST-5 records in train, 5 CLINC records in valid.
EXPECTED_SIZES = {
    "train": 13000 + 2 * 250 + 8544 - 18 - 3,
    "valid": 2600 + 2 * 100 + 1101 - 5,
    "test_indomain": 3900 + 2 * 1000,
    "test_unseen_intents": 20 * 150,
    "test_sst5": 2210,
    "test_agnews": 1000,
    "test_emotion": 1000,
    "test_banking77": 1000,
    "test_yelp": 1000,
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


GOLD_NOUL_KINDS = {
    "train": {"about_domain", "out_of_scope", "about_intent", "is_positive", "is_negative"},
    "valid": {"about_domain", "out_of_scope", "about_intent", "is_positive", "is_negative"},
    "test_indomain": {"about_domain", "out_of_scope", "about_intent"},
    "test_unseen_intents": {"about_domain", "about_intent"},
    "test_sst5": {"is_positive", "is_negative"},
    "test_emotion": {"expresses_emotion"},
    "test_agnews": set(),
    "test_banking77": set(),
    "test_yelp": set(),
}


def gold_nouls(record):
    return {qid: info for qid, info in record["meta"]["nouls"].items() if not info.get("form")}


def test_noul_kinds_present_where_expected(built):
    for split, records in built.splits.items():
        kinds = set(noul_balance(records))
        assert kinds - set(FORM_KINDS) == GOLD_NOUL_KINDS[split]
        used = {k for r in records for k, s in built.form_settings[split][r["source"]].items() if s["used"]}
        assert kinds & set(FORM_KINDS) == used, split


@pytest.mark.parametrize("split", SPLITS)
def test_no_noul_kind_is_answerable_from_its_phrasing(built, split):
    for kind, row in noul_phrasing(built.splits[split]).items():
        assert row["phrasing_only_accuracy"] <= CONFIG["max_phrasing_only_accuracy"], (split, kind, row)
        assert 0.45 <= row["negated_share"] <= 0.55, (split, kind, row)


@pytest.mark.parametrize("split", SPLITS)
def test_every_noul_instruction_parses_to_its_recorded_phrasing(built, split):
    for record in built.splits[split]:
        nouls = record["meta"]["nouls"]
        assert set(nouls) == {qid for qid, q in record["questions"].items() if q["type"] == "noul"}
        for qid, info in nouls.items():
            phrasing = parse(qid, record["questions"][qid]["instructions"])
            assert (phrasing.template, phrasing.negated, phrasing.slot) == (info["template"], info["negated"], info["slot"])


@pytest.mark.parametrize("split", SPLITS)
def test_gold_position_close_to_uniform_given_k(built, split):
    deviations = position_deviations(gold_positions(built.splits[split]))
    assert all(sd <= 4.0 for sd in deviations.values()), deviations


# CLINC record structure


@pytest.mark.parametrize("split", SPLITS)
def test_one_gold_dependent_noul_after_the_choice_or_score(built, split):
    # Decision 42: attention is causal, so no gold-dependent question may precede another one.
    assert order_violations(built.splits[split]) == []
    for record in built.splits[split]:
        dependent = [qid for qid in record["questions"] if not is_form(record, qid)]
        assert record["questions"][dependent[0]]["type"] in ("choice", "score")
        assert len(gold_nouls(record)) == len(dependent) - 1 <= 1


def test_order_violations_catches_a_noul_before_the_choice():
    record = {"id": "x", "questions": {"about_intent": {"type": "noul"}, "intent": {"type": "choice"}}, "meta": {"nouls": {"about_intent": {}}}}
    two = {"id": "y", "questions": {"intent": {"type": "choice"}, "a": {"type": "noul"}, "b": {"type": "noul"}}, "meta": {"nouls": {"a": {}, "b": {}}}}
    form_first = {"id": "z", "questions": {"w": {"type": "noul"}, "intent": {"type": "choice"}}, "meta": {"nouls": {"w": {"form": True}}}}
    assert order_violations([record, two, form_first]) == ["x", "y"]


def test_clinc_noul_kind_mix(built):
    records = clinc_records(built.splits["train"])
    kinds = Counter(next(iter(gold_nouls(r))) for r in records)
    assert kinds["out_of_scope"] == 2 * 250
    assert abs(kinds["about_domain"] - kinds["about_intent"]) < 0.04 * len(records)
    ks = Counter(len(r["questions"]["intent"]["criteria"]) for r in records)
    assert set(ks) == set(range(3, 15)) and min(ks.values()) > 0.8 * len(records) / 12


def underlying_answer(record, kind):
    meta, info = record["meta"], record["meta"]["nouls"][kind]
    if kind == "out_of_scope":
        return meta["gold_intent"] == CLINC_OUT_OF_SCOPE
    if kind == "about_domain":
        return info["asked_domain"] == meta["domain"]
    return info["asked_intent"] == meta["gold_intent"]


def test_clinc_gold_answers_follow_the_rules(built):
    clinc = load_descriptions("clinc")
    for split in ("train", "valid", "test_indomain", "test_unseen_intents"):
        allowed = set(built.held_out) if split == "test_unseen_intents" else set(built.seen)
        for record in clinc_records(built.splits[split]):
            meta, gold, questions = record["meta"], record["gold"], record["questions"]
            k = len(questions["intent"]["criteria"])
            assert CONFIG["clinc"]["k_min"] <= k <= CONFIG["clinc"]["k_max"] and "other" in questions["intent"]["criteria"]
            named = [label for label in questions["intent"]["criteria"] if label != "other"]
            if meta["gold_intent"] == CLINC_OUT_OF_SCOPE:
                assert gold["intent"] == "other"
            else:
                assert gold["intent"] == (meta["gold_intent"] if meta["gold_in_options"] else "other")
            for kind, info in gold_nouls(record).items():
                answer = underlying_answer(record, kind) != info["negated"]
                assert gold[kind] == ("true" if answer else "false")
                if kind == "about_domain":
                    assert info["slot"] == DOMAIN_PHRASES[info["asked_domain"]]
                    domains = {built_domain_of(built)[i] for i in named}
                    assert info["asked_in_options"] == (info["asked_domain"] in domains)
                    assert (info["asked_from"] == "gold") == (info["asked_domain"] == meta["domain"])
                    if info["asked_from"] == "distractor":
                        assert any(built_domain_of(built)[i] == info["asked_domain"] and i != meta["gold_intent"] for i in named)
                if kind == "about_intent":
                    assert info["asked_intent"] in allowed and info["slot"] in clinc[info["asked_intent"]].variants
                    assert info["asked_in_options"] == (info["asked_intent"] in named)
                    assert info["asked_from"] == {True: "gold", False: "distractor" if info["asked_intent"] in named else "outside"}[info["asked_intent"] == meta["gold_intent"]]


def built_domain_of(built):
    return intent_domains(sorted(set(built.seen) | set(built.held_out)))


def test_out_of_scope_utterances_yield_two_records(built):
    oos = [r for r in clinc_records(built.splits["train"]) if r["meta"]["gold_intent"] == CLINC_OUT_OF_SCOPE]
    per_utterance = defaultdict(list)
    for r in oos:
        (kind,) = gold_nouls(r)
        per_utterance[(r["meta"]["source_split"], r["meta"]["source_index"])].append(kind)
    assert len(per_utterance) == 250
    for kinds in per_utterance.values():
        assert len(kinds) == 2 and kinds.count("out_of_scope") == 1 and set(kinds) - {"out_of_scope"} <= {"about_domain", "about_intent"}


@pytest.mark.parametrize("kind", ["about_intent", "about_domain"])
def test_no_answers_are_feature_matched_to_the_options(built, kind):
    # Decision 42: a "no" asks a distractor option (or its domain) with probability 0.8, so
    # whether the asked item is among the options does not predict the answer.
    infos = [(underlying_answer(r, kind), gold_nouls(r)[kind]) for r in clinc_records(built.splits["train"]) if kind in r["meta"]["nouls"]]
    no = [info for yes, info in infos if not yes]
    assert abs(sum(i["asked_from"] == "distractor" for i in no) / len(no) - 0.8) < 0.03
    in_options = {yes: sum(i["asked_in_options"] for y, i in infos if y == yes) / sum(1 for y, _ in infos if y == yes) for yes in (True, False)}
    assert abs(in_options[True] - in_options[False]) < 0.05, in_options


@pytest.mark.parametrize("split, n_oos", [("train", 250), ("valid", 100), ("test_indomain", 1000)])
def test_out_of_scope_underlying_answers_are_balanced(built, split, n_oos):
    # Option 1 of the v1.1 review: exactly as many in-scope out_of_scope questions as out-of-scope utterances.
    records = [r for r in clinc_records(built.splits[split]) if "out_of_scope" in r["meta"]["nouls"]]
    assert Counter(underlying_answer(r, "out_of_scope") for r in records) == {True: n_oos, False: n_oos}


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


@pytest.mark.parametrize("split", ["train", "valid", "test_sst5"])
def test_sst5_scales_and_sentiment_nouls(built, split):
    records = [r for r in built.splits[split] if r["source"] == "SetFit/sst5"]
    three = [r for r in records if r["meta"]["scale"] == "sst5_3_levels"]
    assert len(three) == round(0.3 * len(records))
    for record in records:
        label = LABEL_TEXTS.index(record["meta"]["label_text"])
        score = record["questions"]["sentiment"]
        if record["meta"]["scale"] == "sst5_3_levels":
            assert score["criteria"] == list(LEVELS_3) and record["gold"]["sentiment"] == TO_3_LEVELS[label]
        else:
            assert score["criteria"] == list(LEVELS) and record["gold"]["sentiment"] == label
        nouls = gold_nouls(record)
        dependent = [qid for qid in record["questions"] if not is_form(record, qid)]
        if label == 2:
            assert nouls == {} and dependent == ["sentiment"]
            continue
        (kind,) = nouls
        assert dependent == ["sentiment", kind]
        answer = label > 2 if kind == "is_positive" else label < 2
        assert record["gold"][kind] == ("true" if answer != nouls[kind]["negated"] else "false")


def test_emotion_nouls(built):
    names = {"sadness", "joy", "love", "anger", "fear", "surprise"}
    asks_gold = 0
    for record in built.splits["test_emotion"]:
        info = record["meta"]["nouls"]["expresses_emotion"]
        assert info["asked_emotion"] in names and info["slot"] == info["asked_emotion"]
        answer = info["asked_emotion"] == record["gold"]["label"]
        asks_gold += answer
        assert record["gold"]["expresses_emotion"] == ("true" if answer != info["negated"] else "false")
        assert [qid for qid in record["questions"] if not is_form(record, qid)] == ["label", "expresses_emotion"]
    assert asks_gold == 500


def test_yelp_is_stratified_short_and_scored_on_five_levels(built):
    from jevmark.data.unseen import YELP_LEVELS

    records = built.splits["test_yelp"]
    assert Counter(r["gold"]["stars"] for r in records) == {star: 200 for star in range(5)}
    for record in records:
        assert len(state_text(record)) <= CONFIG["unseen"]["yelp_max_chars"]
        assert record["questions"]["stars"] == {"type": "score", "instructions": "How many stars does this review give the business?", "criteria": list(YELP_LEVELS)}
        assert [qid for qid in record["questions"] if not is_form(record, qid)] == ["stars"]
        assert record["meta"]["scale"] == "yelp_5_stars" and gold_nouls(record) == {}
    assert len({r["meta"]["source_index"] for r in records}) == 1000


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


# Form nouls and state formats (data v1.3)


@pytest.mark.parametrize("split", SPLITS)
def test_form_nouls_are_computed_from_the_text_and_nearly_balanced(built, split):
    underlying = defaultdict(Counter)
    for record in built.splits[split]:
        forms = [qid for qid in record["questions"] if is_form(record, qid)]
        assert len(forms) <= 2
        for kind in forms:
            info = record["meta"]["nouls"][kind]
            setting = built.form_settings[split][record["source"]][kind]
            assert setting["used"] and info["threshold"] == setting["threshold"]
            yes = answer(kind, state_text(record), info["threshold"])
            assert record["gold"][kind] == ("true" if yes != info["negated"] else "false")
            underlying[kind][yes] += 1
    assert underlying
    for kind, c in underlying.items():
        n = c[True] + c[False]
        # Within 48-52 percent on the split's texts; sampling the records that get the kind adds noise.
        assert abs(c[True] / n - 0.5) <= 0.02 + 3 * (0.25 / n) ** 0.5, (split, kind, c)


def test_form_nouls_are_placed_independently_of_the_gold_questions(built):
    records = built.splits["train"]
    counts = Counter(sum(is_form(r, q) for q in r["questions"]) for r in records)
    assert set(counts) == {0, 1, 2} and abs(counts[0] / len(records) - 1 / 3) < 0.03  # 2 is capped at the number of used kinds
    # A form noul precedes the first gold-dependent question about a third of the time, for
    # records with one and with two gold-dependent questions alike (neutral and other SST-5 records).
    before = defaultdict(list)
    for r in records:
        dependent = [q for q in r["questions"] if not is_form(r, q)]
        first = list(r["questions"]).index(dependent[0])
        for i, q in enumerate(r["questions"]):
            if is_form(r, q):
                before[len(dependent)].append(i < first)
    for n_dependent, flags in before.items():
        assert abs(sum(flags) / len(flags) - 1 / 3) < 0.03, (n_dependent, sum(flags) / len(flags))


def test_unbalanceable_kinds_are_skipped(built):
    for split, sources in built.form_settings.items():
        for kinds in sources.values():
            assert not kinds["ends_with_question_mark"]["used"] or split == "test_banking77"
            for s in kinds.values():
                assert s["used"] == (abs(s["yes_share"] - 0.5) <= CONFIG["form"]["max_imbalance"])


@pytest.mark.parametrize("split", SPLITS)
def test_twenty_percent_of_states_are_json(built, split):
    records = built.splits[split]
    wrapped = [r for r in records if r["meta"]["state_format"] == "json"]
    assert len(wrapped) == round(0.2 * len(records))
    for record in records:
        if record["meta"]["state_format"] == "plain":
            assert isinstance(record["state"], str) and record["meta"]["state_fields"] == []
            continue
        state = record["state"]
        assert list(state) == record["meta"]["state_fields"] and "text" in state
        extra = [k for k in state if k != "text"]
        assert 1 <= len(extra) <= 3 and set(extra) <= set(STATE_FIELDS)
        assert all(isinstance(v, str) and v for v in state.values())


# Config overrides


def test_config_overrides():
    config = load_config(REPO / "configs" / "data.yaml", ["seed=3", "clinc.k_max=8", "noul_balance=[0.3, 0.7]"])
    assert config["seed"] == 3 and config["clinc"]["k_max"] == 8 and config["noul_balance"] == [0.3, 0.7]
    for bad in (["seed"], ["nope=1"], ["seed.x=1"]):
        with pytest.raises(ValueError):
            load_config(REPO / "configs" / "data.yaml", bad)


# Duplicates (decision 43)


def test_train_and_valid_share_no_normalised_text_with_any_test_split(built):
    from jevmark.data.dedup import normalise

    test_texts = {normalise(state_text(r)) for split, records in built.splits.items() if split.startswith("test_") for r in records}
    for split in ("train", "valid"):
        assert not {normalise(state_text(r)) for r in built.splits[split]} & test_texts, split
