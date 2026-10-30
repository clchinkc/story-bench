"""Repair-reward tests. The load-bearing case is the one that kills the Stage A exploit."""

import pytest

from prompt_set import CanonFact, build_task
from reward_repair import RepairError, mentions, score_repair, terminal_reward

EYES = CanonFact("Mara", "eye_colour", "grey")
LANTERN = CanonFact("the lantern", "state", "lit")

# The passage carries the injected error, exactly as the repair task needs it.
TASK = build_task("t1", "Mara had grey eyes. The lantern was lit.", (EYES, LANTERN), [(EYES, "brown")])


def test_carrying_the_correction_into_prose_resolves():
    verdict = score_repair("Mara turned, her grey eyes catching the lamplight.", TASK)

    assert verdict.resolved is True
    assert verdict.correct_present == ("grey",)
    assert terminal_reward(verdict) == 1


def test_copying_the_passage_fails_because_it_repeats_the_error():
    """The whole point: Stage A's exploit (transcribe what is visible) now loses."""
    verdict = score_repair("Mara had brown eyes, and she waited.", TASK)

    assert verdict.resolved is False
    assert verdict.leaked_wrong == ("brown",)
    assert terminal_reward(verdict) == 0


def test_ignoring_the_attribute_fails():
    verdict = score_repair("Mara turned toward the door.", TASK)

    assert verdict.resolved is False
    assert verdict.correct_missing == ("grey",)
    assert terminal_reward(verdict) == 0


def test_restating_the_error_in_order_to_deny_it_still_fails():
    """'grey, not brown' mentions both. The prompt tells the model not to mention the
    wrong value at all, so this is a failure rather than a grey area."""
    verdict = score_repair("Her eyes were grey, not brown.", TASK)

    assert verdict.resolved is False
    assert verdict.leaked_wrong == ("brown",)


def test_a_blank_continuation_is_a_wrong_answer_not_an_unmeasured_one():
    verdict = score_repair("   ", TASK)

    assert verdict.resolved is False
    assert terminal_reward(verdict) == 0


@pytest.mark.parametrize(
    ("text", "value", "expected"),
    [
        ("the lantern was lit", "lit", True),
        ("a little light", "lit", False),
        ("her eyes were greying", "grey", False),
        ("Grey eyes", "grey", True),
        ("a succession of rain followed", "succession of rain", True),
        ("rain succeeded", "succession of rain", False),
    ],
)
def test_word_boundary_matching_does_not_lie(text, value, expected):
    assert mentions(text, value) is expected


def test_a_task_with_no_injected_break_is_refused():
    clean = build_task("t2", "Mara had grey eyes.", (EYES,), [])

    with pytest.raises(RepairError, match="no contradiction"):
        score_repair("grey eyes", clean)


def test_a_resolved_verdict_cannot_carry_a_leak():
    from reward_repair import RepairVerdict

    with pytest.raises(RepairError, match="cannot coexist"):
        RepairVerdict(task_id="t1", resolved=True, leaked_wrong=("brown",))
