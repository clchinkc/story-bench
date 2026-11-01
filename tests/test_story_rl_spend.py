"""The spend bound and its gate: a ceiling that can be exceeded is not a ceiling.

Two behaviours are load-bearing — the worst case must be strictly above the expected figure (otherwise
the "bound" is a prediction), and the gate must refuse both an unset and an insufficient approval,
printing the number it refused on.
"""

import pytest

from story_rl.spend import (
    MEASURED_COMPLETION_TOKENS,
    MEASURED_PROMPT_TOKENS,
    bound,
    load_rates,
    per_rollout,
    require_approval,
)

CONFIG = (30, 8, 16)


def test_the_pinned_rates_load_and_are_positive():
    rates = load_rates()

    assert rates.sample > rates.train > rates.prefill > rates.cached_prefill > 0


def test_the_worst_case_exceeds_the_expected_case():
    report = bound(*CONFIG, load_rates(), max_tokens=200)

    assert report["worst_case_usd"] > report["expected_usd"]


def test_the_bound_scales_with_the_number_of_rollouts():
    rates = load_rates()
    one = bound(10, 8, 16, rates, max_tokens=200)
    three = bound(30, 8, 16, rates, max_tokens=200)

    assert three["rollouts"] == 3 * one["rollouts"]
    assert three["worst_case_usd"] == pytest.approx(3 * one["worst_case_usd"], rel=0.02)


def test_a_longer_completion_costs_more():
    rates = load_rates()

    assert per_rollout(200, rates) > per_rollout(50, rates)


def test_the_measured_token_counts_are_recorded_with_the_bound():
    report = bound(*CONFIG, load_rates(), max_tokens=200)

    assert report["measured"]["prompt_tokens"] == pytest.approx(MEASURED_PROMPT_TOKENS, abs=0.1)
    assert report["measured"]["completion_tokens"] == pytest.approx(MEASURED_COMPLETION_TOKENS, abs=0.1)


def test_an_unset_approval_is_refused_and_names_the_figure():
    report = bound(*CONFIG, load_rates(), max_tokens=200)

    with pytest.raises(SystemExit, match="no approval figure"):
        require_approval(report, None)


def test_an_approval_below_the_worst_case_is_refused_and_names_both_numbers():
    report = bound(*CONFIG, load_rates(), max_tokens=200)

    with pytest.raises(SystemExit, match="exceeds the approved"):
        require_approval(report, report["worst_case_usd"] - 0.01)


def test_an_approval_at_or_above_the_worst_case_passes():
    report = bound(*CONFIG, load_rates(), max_tokens=200)

    require_approval(report, report["worst_case_usd"])
    require_approval(report, report["worst_case_usd"] + 1.0)
