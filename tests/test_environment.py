"""jevmark/environment.py: the stochastic-outcome environment and its metrics (task 2.5, decision 54)."""

import math

import numpy as np
import pytest
import torch

from jevmark import environment as E
from jevmark.metrics import QuestionResult


def test_eta_by_hand():
    assert E.eta(2) == pytest.approx(0.05)
    assert E.eta(3) == pytest.approx(0.08)
    assert E.eta(10) == pytest.approx(0.29)
    assert E.eta(13) == pytest.approx(0.38)
    assert E.eta(14) == E.eta(26) == 0.40  # capped
    with pytest.raises(ValueError):
        E.eta(1)


@pytest.mark.parametrize("k", [2, 3, 5, 10, 14, 26])
def test_theta_sums_to_one_and_its_argmax_is_gold(k):
    for gold in range(k):
        t = E.theta(k, gold)
        assert t.sum() == pytest.approx(1.0) and int(t.argmax()) == gold
        assert t[gold] == pytest.approx(1 - E.eta(k))
        assert all(t[j] == pytest.approx(E.eta(k) / (k - 1)) for j in range(k) if j != gold)


@pytest.mark.parametrize("k,gold", [(2, 1), (4, 0), (10, 7)])
def test_accepted_answers_follow_theta(k, gold):
    env = E.NoisyEnvironment(seed=0)
    n = 40000
    counts = np.bincount([env.accepted(gold, k) for _ in range(n)], minlength=k)
    t = E.theta(k, gold)
    sd = np.sqrt(t * (1 - t) / n)
    assert np.all(np.abs(counts / n - t) < 4 * sd), (counts / n, t)


def test_outcome_rate_of_an_action_equals_its_theta():
    """The revealed outcome of action a is 1[a == accepted], so its rate is theta_a."""
    env = E.NoisyEnvironment(seed=3)
    k, gold, n = 5, 2, 30000
    hits = np.zeros(k)
    for _ in range(n):
        accepted = env.accepted(gold, k)
        hits += np.arange(k) == accepted
    t = E.theta(k, gold)
    assert np.all(np.abs(hits / n - t) < 4 * np.sqrt(t * (1 - t) / n))


def test_environment_is_seeded_and_its_state_round_trips():
    a, b = E.NoisyEnvironment(1), E.NoisyEnvironment(1)
    first = [a.accepted(3, 8) for _ in range(50)]
    assert first == [b.accepted(3, 8) for _ in range(50)]
    assert first != [E.NoisyEnvironment(2).accepted(3, 8) for _ in range(50)]
    state = a.get_state()
    after = [a.accepted(0, 4) for _ in range(20)]
    a.set_state(state)
    assert [a.accepted(0, 4) for _ in range(20)] == after
    # Its own generator: the global torch RNG is untouched.
    torch.manual_seed(5)
    x = torch.rand(1)
    torch.manual_seed(5)
    E.NoisyEnvironment(0).accepted(0, 3)
    assert torch.equal(torch.rand(1), x)


def test_scores_against_theta_are_minimised_at_theta():
    t = E.theta(4, 1)
    assert E.expected_brier(t, t) == pytest.approx(1 - (t * t).sum())
    assert E.cross_entropy(t, t) == pytest.approx(-(t * np.log(t)).sum())
    sharp = np.array([0.0, 1.0, 0.0, 0.0])
    assert E.expected_brier(sharp, t) > E.expected_brier(t, t) and E.cross_entropy(sharp, t) > E.cross_entropy(t, t)
    # Brier by hand: E over y of sum (p - e_y)^2 with p one-hot on gold equals 2 * eta.
    assert E.expected_brier(sharp, t) == pytest.approx(2 * E.eta(4))


def calibrated(ks=(2, 3, 5, 10, 14), per_k=30):
    out = []
    for k in ks:
        for i in range(per_k):
            gold = i % k
            out.append(QuestionResult(f"r{k}-{i}", "q", "choice" if k > 2 else "noul", tuple(E.theta(k, gold)), gold, tuple(str(j) for j in range(k)), split="test"))
    return out


def test_a_calibrated_distribution_has_zero_gap_and_minimal_scores():
    results = calibrated()
    m = E.env_metrics(results, "test")
    assert m["calibration_gap"] == pytest.approx(0.0, abs=1e-12)
    assert m["expected_brier"] == pytest.approx(m["expected_brier_min"]) and m["kl_theta"] == pytest.approx(0.0, abs=1e-12)
    assert m["accuracy"] == 1.0
    for k, row in m["by_k"].items():
        assert row["mean_p_gold"] == pytest.approx(1 - E.eta(int(k))) and row["count"] == 30


def test_oracle_temperature_of_a_calibrated_distribution_is_one():
    results = calibrated()
    temps = E.oracle_temperatures_by_k(results)
    assert set(temps) == {2, 3, 5, 10, 14}
    assert all(t == pytest.approx(1.0, abs=1e-3) for t in temps.values())


def test_oracle_temperature_undoes_a_known_sharpening():
    """p = softmax(log theta * 2): the oracle temperature is 2 and scaling by it restores theta."""
    sharp = []
    for r in calibrated(ks=(6,)):
        z = np.log(np.asarray(r.probs)) * 2
        p = np.exp(z - z.max())
        sharp.append(QuestionResult(r.record_id, r.question_id, r.qtype, tuple(p / p.sum()), r.gold, r.labels, split=r.split))
    temps = E.oracle_temperatures_by_k(sharp)
    assert temps[6] == pytest.approx(2.0, rel=1e-3)
    assert E.env_metrics(E.scale_by_k(sharp, temps), "t")["calibration_gap"] == pytest.approx(0.0, abs=1e-5)


def test_bandit_temperature_is_seeded_and_softens_a_sharp_policy():
    sharp = [QuestionResult(f"r{i}", "q", "choice", (0.97, 0.01, 0.01, 0.01), 0, ("a", "b", "c", "d"), split="valid") for i in range(400)]
    t = E.fit_bandit_temperature(sharp, 4, 0.1, 0)
    assert t == E.fit_bandit_temperature(sharp, 4, 0.1, 0)
    assert t > 1.0  # theta puts 0.86 on gold at K 4, less than 0.97
    samples = E.bandit_samples(sharp, 4, 0.1, 0)
    assert len(samples) == 1600 and {s[2] for s in samples} <= {0.0, 1.0}
    assert math.isfinite(t)


def test_calibration_gap_compares_top1_confidence_with_its_expected_hit_rate():
    """K 4, gold 1, p (0.7, 0.1, 0.1, 0.1): the top-1 option is wrong, so it is accepted with probability eta / 3."""
    r = QuestionResult("r", "q", "choice", (0.7, 0.1, 0.1, 0.1), 1, ("a", "b", "c", "d"), split="t")
    row = E.by_k([r])["4"]
    assert row["expected_top1_hit"] == pytest.approx(E.eta(4) / 3)
    assert row["gap"] == pytest.approx(0.7 - E.eta(4) / 3)
    assert row["gap_gold"] == pytest.approx(0.1 - (1 - E.eta(4)))
    assert E.calibration_gap(E.by_k([r])) == pytest.approx(abs(row["gap"]))
