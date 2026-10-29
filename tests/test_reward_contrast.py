"""P5 reward contrast at both levels: one group, and the per-run share W9 records.

GRPO's advantage is group-relative, so a group whose samples all score the same
contributes no gradient at all. W9 therefore records the share of groups that carry
contrast and resamples the rest. These tests pin the two levels apart, and pin the
distinction that matters most: a group with **nothing scorable** is not a uniform
group — it is an unmeasured one, and it must not be reported as 0.
"""

import pytest

from reward_programmatic import group_has_contrast, run_contrast_share


def test_a_uniform_passing_group_has_no_contrast():
    assert group_has_contrast([1, 1, 1, 1]) is False


def test_a_uniform_failing_group_has_no_contrast_either():
    assert group_has_contrast([0, 0, 0]) is False


def test_a_mixed_group_has_contrast():
    assert group_has_contrast([1, 0, 1, 1]) is True


def test_an_unmeasured_group_is_not_reported_as_no_contrast():
    """No scorable sample means no measurement, not a measured zero."""
    assert group_has_contrast([]) is None
    assert group_has_contrast([None, None]) is None


def test_unscorable_samples_are_excluded_rather_than_scored_as_zero():
    assert group_has_contrast([1, None]) is False
    assert group_has_contrast([1, 0, None]) is True


def test_the_run_share_counts_only_scorable_groups():
    groups = [[1, 0], [1, 1], [None, None]]

    assert run_contrast_share(groups) == pytest.approx(1 / 2)


def test_the_run_share_is_none_when_nothing_was_scorable():
    assert run_contrast_share([]) is None
    assert run_contrast_share([[None, None], []]) is None


def test_a_fully_contrasted_run_scores_one():
    assert run_contrast_share([[1, 0], [0, 1]]) == pytest.approx(1.0)


def test_a_fully_uniform_run_scores_zero():
    assert run_contrast_share([[1, 1], [0, 0]]) == pytest.approx(0.0)
