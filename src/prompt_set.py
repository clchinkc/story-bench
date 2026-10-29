"""Deterministic construction of P5 prompt tasks from a passage and a canon statement.

W9 (P5) trains a sequence-level policy against a *binary* terminal reward, so the
task population has to state something a completion can get wrong. Public-domain
classics do not: presented unmodified they offer no canon fact to contradict, every
well-formed continuation passes, and every GRPO group is uniform (D019 measures that
as exactly 0.0 % contrast).

This module supplies the missing half — a passage plus the canon facts it must
preserve, with the contradictions the caller declares *injected* rather than sampled.
Injection is declarative on purpose: the same input always yields byte-identical
tasks, so a task set can be audited, diffed and replayed without a seed.

Scope boundary, stated so callers do not over-read this module:

    it builds TASKS      passage + canon + the wrong values present in the passage
    it does NOT score    that is `reward_programmatic`
    it does NOT parse    turning free prose into canon assertions is a separate,
                         unbuilt component — see `reward_programmatic`'s docstring

Purity: this module imports only `collections.abc` and `dataclasses`. It performs no
file, network, environment, clock or random access.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class PromptSetError(ValueError):
    """A task set is malformed, ambiguous or inconsistent with its own canon."""


@dataclass(frozen=True)
class CanonFact:
    """One assertion a completion must not contradict."""

    entity: str
    attribute: str
    value: str

    def __post_init__(self) -> None:
        for field_name in ("entity", "attribute", "value"):
            raw = getattr(self, field_name)
            if not isinstance(raw, str) or not raw.strip():
                raise PromptSetError(
                    f"CanonFact.{field_name} must be a non-blank str, got {raw!r}"
                )

    def key(self) -> tuple[str, str]:
        """Identity of the assertion, independent of the value it asserts."""
        return (self.entity, self.attribute)


@dataclass(frozen=True)
class PromptTask:
    """One training task: a passage, the canon it must preserve, and its breaks.

    `contradictions` are the canon facts the passage states FALSELY. They are kept as
    facts rather than as free text so a reward can compare them to what a completion
    asserts without re-reading the prose.
    """

    task_id: str
    passage: str
    canon: tuple[CanonFact, ...]
    contradictions: tuple[CanonFact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise PromptSetError(f"task_id must be a non-blank str, got {self.task_id!r}")
        if not isinstance(self.passage, str) or not self.passage.strip():
            raise PromptSetError(f"{self.task_id}: passage must be a non-blank str")
        if not self.canon:
            raise PromptSetError(
                f"{self.task_id}: a task with no canon fact cannot produce contrast (D019)"
            )
        canon_keys = [fact.key() for fact in self.canon]
        if len(set(canon_keys)) != len(canon_keys):
            raise PromptSetError(
                f"{self.task_id}: canon states the same (entity, attribute) twice; "
                "the completion could satisfy one and violate the other"
            )
        for broken in self.contradictions:
            if broken.key() not in set(canon_keys):
                raise PromptSetError(
                    f"{self.task_id}: contradiction {broken.key()} is not a canon fact, "
                    "so a completion cannot resolve it"
                )
            true_value = next(f.value for f in self.canon if f.key() == broken.key())
            if broken.value == true_value:
                raise PromptSetError(
                    f"{self.task_id}: contradiction {broken.key()} asserts the true value "
                    f"{true_value!r}; it is not a break"
                )


def inject_break(passage: str, true_fact: CanonFact, wrong_value: str) -> tuple[str, CanonFact]:
    """Replace one stated value in `passage` with `wrong_value`; return text and the break.

    Fails closed when the true value is absent, appears more than once, or the
    replacement would be a no-op — each of those makes the break unverifiable by
    reading the result.
    """
    if not isinstance(passage, str) or not passage.strip():
        raise PromptSetError("passage must be a non-blank str")
    if not isinstance(wrong_value, str) or not wrong_value.strip():
        raise PromptSetError(f"wrong_value must be a non-blank str, got {wrong_value!r}")
    if wrong_value == true_fact.value:
        raise PromptSetError(
            f"wrong_value {wrong_value!r} equals the canonical value; that is not a break"
        )
    occurrences = passage.count(true_fact.value)
    if occurrences == 0:
        raise PromptSetError(
            f"passage never states {true_fact.key()} as {true_fact.value!r}; "
            "the break would be invisible"
        )
    if occurrences > 1:
        raise PromptSetError(
            f"passage states {true_fact.value!r} {occurrences} times; "
            "replacement would be ambiguous"
        )
    broken = CanonFact(true_fact.entity, true_fact.attribute, wrong_value)
    return passage.replace(true_fact.value, wrong_value, 1), broken


def build_task(
    task_id: str,
    passage: str,
    canon: Sequence[CanonFact],
    breaks: Sequence[tuple[CanonFact, str]],
) -> PromptTask:
    """Inject every declared break into `passage` and return the validated task."""
    text = passage
    contradictions: list[CanonFact] = []
    for true_fact, wrong_value in breaks:
        text, broken = inject_break(text, true_fact, wrong_value)
        contradictions.append(broken)
    return PromptTask(
        task_id=task_id,
        passage=text,
        canon=tuple(canon),
        contradictions=tuple(contradictions),
    )


def validate_task_set(tasks: Iterable[PromptTask]) -> tuple[PromptTask, ...]:
    """Return the tasks as a tuple, refusing an empty set or duplicate task ids."""
    materialised = tuple(tasks)
    if not materialised:
        raise PromptSetError("a task set must contain at least one task")
    ids = [task.task_id for task in materialised]
    if len(set(ids)) != len(ids):
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        raise PromptSetError(f"duplicate task ids: {duplicates}")
    return materialised
