"""scripts/log_stats.py: where a log's negatives fall, against a hand computation."""

from __future__ import annotations

import json

import pytest

from jevmark.feedback import Interaction, behaviour, file_sha256, write_log
from scripts import log_stats


def interaction(index: int, probs: tuple[float, ...], action: int, outcome: int) -> Interaction:
    return Interaction(index, f"r{index}", "label", tuple(f"o{k}" for k in range(len(probs))), probs, action,
                       behaviour(probs)[action], outcome, outcome, "m", "c")


def test_negatives_split_by_top_option_and_uniform_share_by_hand(tmp_path):
    log = [
        interaction(0, (0.7, 0.2, 0.1), 0, 1),  # positive: not counted
        interaction(1, (0.7, 0.2, 0.1), 0, 0),  # negative on top
        interaction(2, (0.5, 0.5, 0.0), 2, 0),  # negative on other, q = .1 / 3: all uniform
        interaction(3, (0.6, 0.4), 1, 0),  # negative on other, q = .9 x .4 + .05 = .41
    ]
    block = log_stats.negatives(log)
    assert block["negatives"] == 3
    assert block["on_top"] == {"count": 1, "share": pytest.approx(1 / 3)}
    assert block["on_other"] == {"count": 2, "share": pytest.approx(2 / 3)}
    assert block["on_other_uniform_share"] == pytest.approx((1.0 + 0.05 / 0.41) / 2)
    assert block["chosen_other"] == {"count": 0, "share": 0.0, "on_top_count": 0, "on_other_count": 0, "all_in_on_other": True}

    d = tmp_path / "v3_log_x"
    d.mkdir()
    sha = write_log(d / "log.jsonl", log)
    (d / "metrics.json").write_text(json.dumps({"run_name": "v3_log_x", "log_sha256": sha, "log_file": "log.jsonl"}, indent=2) + "\n")
    log_stats.run(d)
    written = json.loads((d / "metrics.json").read_text())
    assert written["run_name"] == "v3_log_x" and written["negatives"]["log_sha256"] == sha == file_sha256(d / "log.jsonl")


def test_a_propensity_off_the_behaviour_policy_is_refused():
    bad = interaction(0, (0.7, 0.3), 1, 0)
    bad = Interaction(**{**bad.__dict__, "propensity": 0.5})
    with pytest.raises(SystemExit, match="behaviour policy"):
        log_stats.negatives([bad])


def test_negatives_on_the_other_option_split_by_whether_other_was_top():
    def labelled(index, probs, action, outcome):
        i = interaction(index, probs, action, outcome)
        return Interaction(**{**i.__dict__, "labels": ("o0", "o1", "other")})

    log = [
        labelled(0, (0.2, 0.1, 0.7), 2, 0),  # chose other, other on top
        labelled(1, (0.7, 0.2, 0.1), 2, 0),  # chose other, not top
        labelled(2, (0.7, 0.2, 0.1), 1, 0),  # another negative on other options
        labelled(3, (0.7, 0.2, 0.1), 0, 0),  # negative on top
    ]
    block = log_stats.negatives(log)
    assert block["chosen_other"] == {"count": 2, "share": 0.5, "on_top_count": 1, "on_other_count": 1, "all_in_on_other": False}
