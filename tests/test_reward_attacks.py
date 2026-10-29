"""P5 reward-attack battery: each named attack must fail to buy reward.

W9 requires the adversarial classes to be tested *before* any shaping is enabled.
Every test here is an attack that would pay off under a laxer reward — padding,
constraint deletion, N/A exploitation, evidence-id invention, cherry-picking, and
break-then-repair. The reward must refuse each one.
"""

from prompt_set import CanonFact, build_task
from reward_programmatic import group_has_contrast, score_assertions, terminal_reward

EYES = CanonFact("Mara", "eye_colour", "grey")
LANTERN = CanonFact("the lantern", "state", "lit")
TASK = build_task(
    "t1",
    "Mara had grey eyes. The lantern was lit.",
    (EYES, LANTERN),
    [(EYES, "brown")],
)


def test_padding_does_not_substitute_for_satisfying_canon():
    """Volume is not evidence: twenty invented facts earn nothing without the canon."""
    padding = [CanonFact("the inn", f"detail_{n}", "noted") for n in range(20)]
    verdict = score_assertions(padding, TASK)

    assert verdict.resolved is False
    assert terminal_reward(verdict) == 0


def test_constraint_deletion_earns_no_reward():
    """Saying nothing is the cheapest attack; it must score as a wrong answer."""
    verdict = score_assertions([], TASK)

    assert terminal_reward(verdict) == 0


def test_an_unscored_episode_cannot_be_minted_by_silence():
    """N/A exploitation: only a self-contradiction is unassessable, not emptiness."""
    silence = score_assertions([], TASK)
    conflict = score_assertions(
        [CanonFact("Mara", "eye_colour", "grey"), CanonFact("Mara", "eye_colour", "brown")],
        TASK,
    )

    assert terminal_reward(silence) == 0
    assert terminal_reward(conflict) is None


def test_inventing_an_evidence_id_does_not_pay():
    """Only canon membership matters; an authoritative-sounding entity buys nothing."""
    invented = [
        CanonFact("evidence", "id", "EV-0001"),
        CanonFact("citation", "status", "verified"),
    ]
    verdict = score_assertions(invented, TASK)

    assert terminal_reward(verdict) == 0


def test_cherry_picking_one_canon_fact_cannot_pass():
    """Satisfying the easy fact while violating the hard one is still a failure."""
    verdict = score_assertions([LANTERN, CanonFact("Mara", "eye_colour", "brown")], TASK)

    assert verdict.resolved is False
    assert verdict.violations == (CanonFact("Mara", "eye_colour", "brown"),)
    assert terminal_reward(verdict) == 0


def test_break_then_repair_becomes_unmeasurable_not_successful():
    """Asserting both the broken and the true value must never read as resolved.

    It lands on `None` rather than 0 because the two assertions are mutually
    contradictory. That is the contract's grader-invalid bucket, which is excluded
    from task-negative labels — so the attack buys neither reward nor a free negative,
    and the run records a contrast gap instead of silently learning from it.
    """
    verdict = score_assertions(
        [CanonFact("Mara", "eye_colour", "brown"), EYES], TASK
    )

    assert verdict.resolved is False
    assert terminal_reward(verdict) is None
    assert group_has_contrast([terminal_reward(verdict)]) is None
