"""Binary terminal reward over canon assertions — no judge anywhere in the loop.

W9's reward is *programmatic*: a completion either preserves the task's canon or it
does not. A frontier-LLM judge in this position would be both the largest
reward-hacking surface and a recurring per-epoch cost against the USD 1,000 ceiling,
so this module accepts **structured assertions**, never prose, and never calls a model.

The boundary is deliberate and load-bearing:

    prose -> assertions   OUT OF SCOPE. That is a separate parser, unbuilt, and it is
                          where a judge would sneak back in if this module took text.
    assertions -> reward  THIS MODULE. Pure set logic over (entity, attribute, value).

Three outcomes, not two, because a wrong answer and an unmeasurable one are different
objects (the contract excludes grader-invalid results from task-negative labels):

    resolved        every canon fact affirmed, none contradicted   -> reward 1
    unresolved      a canon fact is contradicted or never affirmed -> reward 0
    unassessable    the assertions contradict EACH OTHER           -> reward None

`terminal_reward` returns None for the third case so a caller cannot turn an
unmeasurable episode into a negative label by truthiness.

Purity: imports only `collections.abc` and `dataclasses`; no file, network,
environment, clock or random access.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from prompt_set import CanonFact, PromptTask


class RewardError(ValueError):
    """The verdict was constructed or consumed in a way the contract forbids."""


@dataclass(frozen=True)
class RewardVerdict:
    """One immutable reward decision over one completion's assertions."""

    task_id: str
    resolved: bool
    violations: tuple[CanonFact, ...] = ()
    missing: tuple[CanonFact, ...] = ()
    unassessable: str | None = None

    def __post_init__(self) -> None:
        if self.unassessable is not None:
            if self.resolved:
                raise RewardError(
                    f"{self.task_id}: an unassessable episode cannot also be resolved"
                )
            if self.violations or self.missing:
                raise RewardError(
                    f"{self.task_id}: an unassessable episode cannot carry violations or "
                    "missing facts; that detail would imply a decision that was not made"
                )
            if not self.unassessable.strip():
                raise RewardError(f"{self.task_id}: unassessable needs a non-blank reason")


def score_assertions(assertions: Iterable[CanonFact], task: PromptTask) -> RewardVerdict:
    """Decide `task` from the canon assertions a completion made.

    Satisfaction is *coverage*, not merely the absence of contradiction: a completion
    must affirm every canon fact with its true value. A violation-only reward pays 1 for
    asserting nothing at all — the cheapest attack there is — and the attack battery
    caught exactly that before this clause existed. Assertions about entities outside
    canon remain allowed: a completion may invent freely, it just cannot omit what the
    task stated. A completion that affirms nothing is unresolved, not unassessable —
    silence is a wrong answer, not a missing measurement.
    """
    materialised = tuple(assertions)
    canon_by_key = {fact.key(): fact for fact in task.canon}

    claimed: dict[tuple[str, str], str] = {}
    conflicting: list[tuple[str, str]] = []
    for assertion in materialised:
        key = assertion.key()
        if key in claimed and claimed[key] != assertion.value:
            conflicting.append(key)
        claimed[key] = assertion.value

    if conflicting:
        rendered = ", ".join(f"{entity}.{attribute}" for entity, attribute in conflicting)
        return RewardVerdict(
            task_id=task.task_id,
            resolved=False,
            unassessable=f"completion asserts conflicting values for {rendered}",
        )

    violations = tuple(
        CanonFact(entity, attribute, value)
        for (entity, attribute), value in claimed.items()
        if (entity, attribute) in canon_by_key
        and value != canon_by_key[(entity, attribute)].value
    )
    violation_keys = {(fact.entity, fact.attribute) for fact in violations}
    missing = tuple(
        fact
        for fact in task.canon
        if claimed.get(fact.key()) != fact.value and fact.key() not in violation_keys
    )
    return RewardVerdict(
        task_id=task.task_id,
        resolved=not violations and not missing,
        violations=violations,
        missing=missing,
    )


def terminal_reward(verdict: RewardVerdict) -> int | None:
    """Project a verdict to the training scalar: 1, 0, or None when unassessable."""
    if verdict.unassessable is not None:
        return None
    return 1 if verdict.resolved else 0


def reward_contrast(rewards: Iterable[int | None]) -> float:
    """Share of a group carrying a non-zero advantage; None entries are excluded.

    A group where every sample scores the same teaches nothing — GRPO's advantage is
    group-relative, so a uniform group contributes zero gradient. W9 records this
    statistic per run and resamples uniform groups rather than training on them.
    """
    scored = [reward for reward in rewards if reward is not None]
    if not scored:
        return 0.0
    if len(set(scored)) > 1:
        return 1.0
    return 0.0
