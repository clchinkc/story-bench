"""The repair task: one passage, one injected break, one prompt — frozen so a run can be replayed.

`reprobe.py` measured `e = 0.708` by embedding this prompt inside the probe. Stage B has to send
the *same bytes* to a different model on a different service, and the whole comparison dies if the
template drifts by a comma. So the template lives here, once, and `tests/test_repair_task.py` pins
it to a literal.

The prompt states the truth and names the error, then asks for a continuation that uses the correct
value and does not mention the incorrect one. Stating the truth is what makes the task scoreable;
naming the error is what makes omission a *choice* the model has to make rather than an accident.

Purity: imports `hashlib` and `dataclasses`; no file, network, clock or random access.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuation_parser import extract
from holdout_guard import prompt_digest
from prompt_set import CanonFact, PromptTask
from reward_repair import correction_reward, score_repair, terminal_reward

REPAIR_PROMPT = (
    "Story so far: {passage}\n"
    "\n"
    "The passage contains one error: {entity}'s {attribute} is {true_value}, not {wrong_value}.\n"
    "\n"
    "Continue the story for exactly two sentences. Use the correct value. "
    "Do not mention the incorrect value at all."
)


class RepairTaskError(ValueError):
    """A repair-task row is malformed, or its break is inconsistent with its own canon."""


@dataclass(frozen=True)
class RepairTask:
    """One scoreable repair task. Mirrors `tasks_repair.json` exactly."""

    task_id: str
    passage: str
    canon: tuple[CanonFact, ...]
    broken_entity: str
    broken_attribute: str
    true_value: str
    wrong_value: str

    def __post_init__(self) -> None:
        for name in ("task_id", "passage", "broken_entity", "broken_attribute"):
            raw = getattr(self, name)
            if not isinstance(raw, str) or not raw.strip():
                raise RepairTaskError(f"RepairTask.{name} must be a non-blank str, got {raw!r}")
        if self.true_value == self.wrong_value:
            raise RepairTaskError(f"{self.task_id}: a break must change the value")
        if not self.canon:
            raise RepairTaskError(f"{self.task_id}: a task with no canon cannot be scored")

    def prompt(self) -> str:
        """The exact text the model sees."""
        return REPAIR_PROMPT.format(
            passage=self.passage,
            entity=self.broken_entity,
            attribute=self.broken_attribute,
            true_value=self.true_value,
            wrong_value=self.wrong_value,
        )

    def as_prompt_task(self) -> PromptTask:
        """The shape `reward_repair.score_repair` consumes: canon + the one injected break."""
        broken = CanonFact(self.broken_entity, self.broken_attribute, self.wrong_value)
        return PromptTask(
            task_id=self.task_id,
            passage=self.passage,
            canon=self.canon,
            contradictions=(broken,),
        )

    def content_digest(self) -> str:
        """Identity of the passage, for checking a split stays disjoint.

        Delegates to `holdout_guard.prompt_digest`: the guard already owns this identity, and two
        implementations of one digest is how a split check silently stops matching what it guards.
        """
        return prompt_digest(self.passage)


def from_row(row: dict) -> RepairTask:
    """Build from one entry of `tasks_repair.json`, refusing a row that disagrees with itself."""
    canon = tuple(CanonFact(f["entity"], f["attribute"], f["value"]) for f in row["canon"])
    task = RepairTask(
        task_id=row["task_id"],
        passage=row["passage"],
        canon=canon,
        broken_entity=row["broken_entity"],
        broken_attribute=row["broken_attribute"],
        true_value=row["true_value"],
        wrong_value=row["wrong_value"],
    )
    truth = [f for f in canon if f.key() == (task.broken_entity, task.broken_attribute)]
    if not truth:
        raise RepairTaskError(f"{task.task_id}: the broken fact is not in the task canon")
    if truth[0].value != task.true_value:
        raise RepairTaskError(
            f"{task.task_id}: true_value {task.true_value!r} is not the canon value "
            f"{truth[0].value!r}; the reward would score a value the passage does not carry"
        )
    return task


def from_rows(rows: list[dict]) -> tuple[RepairTask, ...]:
    """Build a task set, refusing duplicate ids rather than silently shadowing one."""
    tasks = tuple(from_row(row) for row in rows)
    ids = [task.task_id for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise RepairTaskError(f"duplicate task ids: {duplicates}")
    return tasks


@dataclass(frozen=True)
class RolloutScore:
    """One raw completion scored under both projections.

    `strict` is the training reward (correction present, error absent). `correction` drops the
    non-mention clause and is reported alongside, so a run can separate "learned to repair" from
    "learned to avoid a string". Both come from one read of the text, so they cannot disagree about
    what was parsed.
    """

    task_id: str
    parsed: bool
    how: str
    strict: int
    correction: int
    missing: tuple[str, ...] = ()
    leaked: tuple[str, ...] = ()
    text: str = ""


def score_rollout(raw: str, task: RepairTask, policy: str = "prose") -> RolloutScore:
    """Parse a completion under `policy` and score it both ways. The single scoring entry point."""
    extraction = extract(raw, policy)
    if extraction.unparsed:
        return RolloutScore(
            task_id=task.task_id, parsed=False, how=extraction.how, strict=0, correction=0
        )
    continuation = extraction.text or ""
    verdict = score_repair(continuation, task.as_prompt_task())
    return RolloutScore(
        task_id=task.task_id,
        parsed=True,
        how=extraction.how,
        strict=terminal_reward(verdict),
        correction=correction_reward(verdict),
        missing=verdict.correct_missing,
        leaked=verdict.leaked_wrong,
        text=continuation,
    )
