"""What the training env actually needs from a task — the seam a new task shape plugs into.

`story_rl.env` used to name `RepairTask` in every annotation, which made "add a shape" mean "edit the
env". Nothing in the env depends on repair specifics: it asks a task for its prompt text, for the
`PromptTask` the reward reads, for the reference answer, and for a digest of its passage. So that is
the whole contract, stated once, and a shape is a class that satisfies it plus a corpus file.

This is deliberately structural, not an ABC: `RepairTask` was written first and must not change to
adopt a base class, and a test can hand the env a two-line stub to prove the seam is real rather than
nominal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime is the thing this file avoids
    from prompt_set import PromptTask


@runtime_checkable
class ScorableTask(Protocol):
    """One episode's task. Anything satisfying this can be trained on or evaluated."""

    task_id: str
    true_value: str

    def prompt(self) -> str:
        """The exact bytes the model sees. Frozen per task, never rebuilt per rollout."""

    def as_prompt_task(self) -> PromptTask:
        """The canon-plus-break shape `reward_repair.score_repair` consumes."""

    def content_digest(self) -> str:
        """Identity of the passage, for proving a held-out split stays disjoint."""
