"""scripts/v3_review.py: the off-policy identity on the tiny model, CPU (review item 3)."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v3_review", REPO / "scripts" / "v3_review.py")
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
spec.loader.exec_module(review)


def test_outcome_minus_p_log_expected_gradient_is_a_constant_times_ips_caps_brier(tokenizer):
    check = review.identity_check(review.load_script("train_rlcd"))
    assert check["max_weight_p_over_q"] < 1 / (1 - 0.1) < 5.0  # no weight clipped
    assert check["constant"] == 0.25  # (1 - 1/2) / 2
    assert check["max_rel_difference_reinforce_vs_constant_times_ips_brier"] < review.IDENTITY_TOL
    assert check["max_rel_difference_ips_brier_vs_full_information_brier"] < review.IDENTITY_TOL
