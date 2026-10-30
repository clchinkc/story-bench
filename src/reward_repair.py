"""Repair reward: does the continuation carry a correction forward, or echo the error?

Stage A showed why this module exists. The coverage reward in `reward_programmatic`
pays for restating a list the prompt prints, and a prompt instruction ("exactly four
entries, in that order") drove `e` to 0.000 — not by improving continuity, but by
turning restatement into positional transcription. At `e = 0` every GRPO group is
uniform, so the RL stage had nothing to learn from and would have taught copying.

The inversion: `score_assertions` rewards copying the **right** value. This rewards
carrying the **correction** into new prose, which copying cannot satisfy, because the
visible text contains the wrong value.

    prompt     states the true value and the injected error, and asks the model to
               continue without mentioning the error at all
    reward     TRUE value present  AND  WRONG value absent, both word-boundary

Two deterministic checks over the continuation. No judge, no criteria list, and the
whole "which key did it use" channel that dominated Stage A disappears — there are no
keys, only prose.

Scope boundary: this scores *mention and non-mention*, not coherence. A model can pass
by writing one flat sentence. That is acceptable for a training signal and is a reason
not to reuse this as a final benchmark.

Purity: imports `re`, `collections.abc` and `dataclasses`; no file, network, clock or
random access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_set import CanonFact, PromptTask


class RepairError(ValueError):
    """A repair verdict was built in a way the contract forbids."""


@dataclass(frozen=True)
class RepairVerdict:
    """One immutable decision about one continuation."""

    task_id: str
    resolved: bool
    correct_present: tuple[str, ...] = ()
    correct_missing: tuple[str, ...] = ()
    leaked_wrong: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.resolved and (self.correct_missing or self.leaked_wrong):
            raise RepairError(
                f"{self.task_id}: resolved cannot coexist with a missing or leaked value"
            )


def mentions(text: str, value: str) -> bool:
    """Word-boundary, case-insensitive containment.

    `\\b` is not enough for phrases that start or end with punctuation, so the guard
    is a non-word character lookaround. This is what stops `lit` matching `little` and
    `grey` matching `greying` — the false positive that would make the reward lie in
    the model's favour.
    """
    if not value.strip():
        return False
    pattern = r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _true_value(task: PromptTask, broken: CanonFact) -> str | None:
    for fact in task.canon:
        if fact.key() == broken.key():
            return fact.value
    return None


def score_repair(continuation: str, task: PromptTask) -> RepairVerdict:
    """Score one continuation against every injected break the task carries.

    A task with no break is unscoreable here — there is nothing to repair — and that is
    refused rather than silently resolved.
    """
    if not task.contradictions:
        raise RepairError(
            f"{task.task_id}: task carries no contradiction; the repair reward needs one"
        )
    if not isinstance(continuation, str) or not continuation.strip():
        return RepairVerdict(task_id=task.task_id, resolved=False)

    present: list[str] = []
    missing: list[str] = []
    leaked: list[str] = []
    for broken in task.contradictions:
        true_value = _true_value(task, broken)
        if true_value is None:
            raise RepairError(f"{task.task_id}: break {broken.key()} has no canon fact")
        if mentions(continuation, true_value):
            present.append(true_value)
        else:
            missing.append(true_value)
        if mentions(continuation, broken.value):
            leaked.append(broken.value)

    return RepairVerdict(
        task_id=task.task_id,
        resolved=not missing and not leaked,
        correct_present=tuple(present),
        correct_missing=tuple(missing),
        leaked_wrong=tuple(leaked),
    )


def terminal_reward(verdict: RepairVerdict) -> int:
    """Project to the training scalar. Unlike the coverage reward there is no None:
    a blank continuation is a wrong answer, not an unmeasured one."""
    return 1 if verdict.resolved else 0
