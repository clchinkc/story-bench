"""P5 programmatic reward: three outcomes, no judge, and the contrast statistic."""

import pytest

from prompt_set import CanonFact, build_task
from reward_programmatic import (
    RewardError,
    RewardVerdict,
    reward_contrast,
    score_assertions,
    terminal_reward,
)

PASSAGE = "Mara had grey eyes. The lantern was lit."
EYES = CanonFact("Mara", "eye_colour", "grey")
LANTERN = CanonFact("the lantern", "state", "lit")
TASK = build_task("t1", PASSAGE, (EYES, LANTERN), [(EYES, "brown")])


def test_preserving_canon_resolves():
    verdict = score_assertions([EYES, LANTERN], TASK)

    assert verdict.resolved is True
    assert verdict.violations == ()
    assert terminal_reward(verdict) == 1


def test_contradicting_canon_is_unresolved_and_names_the_violation():
    verdict = score_assertions([CanonFact("Mara", "eye_colour", "brown")], TASK)

    assert verdict.resolved is False
    assert verdict.unassessable is None
    assert verdict.violations == (CanonFact("Mara", "eye_colour", "brown"),)
    assert terminal_reward(verdict) == 0


def test_silence_is_a_wrong_answer_not_a_missing_measurement():
    """A completion that says nothing about canon is unresolved, never unassessable."""
    verdict = score_assertions([], TASK)

    assert verdict.resolved is False
    assert verdict.unassessable is None
    assert terminal_reward(verdict) == 0


def test_asserting_entities_outside_canon_is_allowed():
    """Invention is free; omission is not. Canon must still be covered."""
    verdict = score_assertions([EYES, LANTERN, CanonFact("the inn", "state", "crowded")], TASK)

    assert verdict.resolved is True


def test_partial_coverage_is_unresolved_and_names_what_was_missing():
    """Coverage, not mere non-contradiction: one affirmed fact is not a pass."""
    verdict = score_assertions([EYES], TASK)

    assert verdict.resolved is False
    assert verdict.violations == ()
    assert verdict.missing == (LANTERN,)
    assert terminal_reward(verdict) == 0


def test_self_contradiction_is_unassessable_and_has_no_scalar():
    verdict = score_assertions(
        [CanonFact("Mara", "eye_colour", "grey"), CanonFact("Mara", "eye_colour", "green")],
        TASK,
    )

    assert verdict.unassessable is not None
    assert verdict.resolved is False
    assert terminal_reward(verdict) is None


def test_an_unassessable_verdict_cannot_also_be_resolved():
    with pytest.raises(RewardError, match="cannot also be resolved"):
        RewardVerdict(task_id="t1", resolved=True, unassessable="conflict")


def test_an_unassessable_verdict_cannot_carry_violations():
    with pytest.raises(RewardError, match="cannot carry violations"):
        RewardVerdict(task_id="t1", resolved=False, violations=(EYES,), unassessable="x")


def test_a_uniform_group_has_no_contrast():
    assert reward_contrast([1, 1, 1, 1]) == 0.0
    assert reward_contrast([0, 0, 0]) == 0.0


def test_a_mixed_group_has_contrast():
    assert reward_contrast([1, 0, 1, 1]) == 1.0


def test_unassessable_samples_are_excluded_rather_than_scored_as_zero():
    assert reward_contrast([None, None]) == 0.0
    assert reward_contrast([1, None]) == 0.0
    assert reward_contrast([1, 0, None]) == 1.0
