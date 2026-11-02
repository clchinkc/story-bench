"""Shape A: the answer is a computed weekday, and the truth is never printed.

The repair task states the correction in the prompt, so its reward can be satisfied by copying. This
shape removes that: the passage gives an anchor weekday and a duration, the derived day is replaced with
a wrong one, and the model has to compute the right day from the two premises before it can use it. The
prompt is byte-identical to the one `bindings/w2-corpus/probe.py` measured, so a trained rate and a
prompted rate are answers to the same question.

What makes this a *shape* rather than another env: it supplies `prompt()`, `as_prompt_task()` and
`true_value`, and everything downstream — the reward, the parser policy, the GRPO loop, the metrics — is
`story_rl.env`'s, unchanged. There is no shape-specific reward here, which is the only way a shape's
gain can be compared to another's.

Verified on load, every refusal fatal:

    derived == anchor + days (mod 7)          the arithmetic the model has to redo
    the true day appears exactly once         the passage is meant to carry the wrong one
    the replacement day appears               the break is visible, so the error is unambiguous
    no other weekday appears                  a stray day makes the arithmetic ambiguous
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from holdout_guard import HoldoutSet, assert_disjoint, prompt_digest
from prompt_set import CanonFact, PromptTask
from scorable import ScorableTask
from story_rl.env import DEFAULT_MODEL, DEFAULT_POLICY, DEFAULT_RENDERER, TaskDataset

DATA = Path(__file__).resolve().parent.parent.parent / "dataset" / "story_rl"
TASKS = DATA / "shape_a.json"
SPLIT = DATA / "shape_a_split.json"

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
CHANCE_FLOOR = 1 / 6

# Salt and fraction are part of the split's definition, not tunables: change either and the shipped
# manifest stops being reproducible, which the shape-A test catches.
SPLIT_SALT = "story-rl-shape-a-v1"
HOLDOUT_COUNT = 16


class ShapeATaskError(ValueError):
    """A shape-A row is malformed, or its break disagrees with its own arithmetic."""


def expected_day(anchor: str, days: int) -> str:
    """The day `days` after `anchor`. Raises rather than returning a guess."""
    if anchor not in DAYS:
        raise ShapeATaskError(f"{anchor!r} is not a weekday")
    return DAYS[(DAYS.index(anchor) + days) % 7]


def stray_days(passage: str, anchor: str, derived: str) -> list[str]:
    return [day for day in DAYS if day not in (anchor, derived) and day in passage]


@dataclass(frozen=True)
class ShapeATask:
    """One temporal task, exactly as `shape_a.json` carries it."""

    task_id: str
    passage: str
    anchor: str
    duration_days: int
    true_value: str
    wrong_value: str
    event: str
    prompt_text: str

    def __post_init__(self) -> None:
        self._check_fields()
        self._check_arithmetic()
        self._check_passage()

    def _check_fields(self) -> None:
        for name in ("task_id", "passage", "anchor", "true_value", "wrong_value", "event"):
            raw = getattr(self, name)
            if not isinstance(raw, str) or not raw.strip():
                raise ShapeATaskError(f"ShapeATask.{name} must be a non-blank str, got {raw!r}")
        if self.true_value == self.wrong_value:
            raise ShapeATaskError(f"{self.task_id}: a break must change the value")
        if self.true_value not in DAYS or self.wrong_value not in DAYS:
            raise ShapeATaskError(f"{self.task_id}: both values must be weekdays")

    def _check_arithmetic(self) -> None:
        computed = expected_day(self.anchor, self.duration_days)
        if self.true_value != computed:
            raise ShapeATaskError(
                f"{self.task_id}: {self.anchor}+{self.duration_days} is {computed}, "
                f"not {self.true_value}"
            )

    def _check_passage(self) -> None:
        if self.true_value in self.passage:
            raise ShapeATaskError(
                f"{self.task_id}: the true day {self.true_value!r} is printed in the passage; the "
                "task would be answerable by copying, which is exactly what this shape removes"
            )
        if self.passage.count(self.wrong_value) != 1:
            raise ShapeATaskError(
                f"{self.task_id}: the replacement day appears "
                f"{self.passage.count(self.wrong_value)}x, not once"
            )
        strays = stray_days(self.passage, self.anchor, self.wrong_value)
        if strays:
            raise ShapeATaskError(f"{self.task_id}: stray weekday(s) {strays} make the sum ambiguous")
        if self.passage not in self.prompt_text:
            raise ShapeATaskError(f"{self.task_id}: the prompt does not carry the passage verbatim")

    def prompt(self) -> str:
        """The exact text the model sees — frozen at build time, never rebuilt per rollout."""
        return self.prompt_text

    def as_prompt_task(self) -> PromptTask:
        """The canon-plus-break the repair reward consumes: truth in canon, error as the break.

        The prompt does not print the truth, so `score_repair`'s truth check is answered from the canon
        and cannot be satisfied by copying — which is the entire point of the shape.
        """
        truth = CanonFact(self.event, "day", self.true_value)
        broken = CanonFact(self.event, "day", self.wrong_value)
        return PromptTask(
            task_id=self.task_id,
            passage=self.passage,
            canon=(truth,),
            contradictions=(broken,),
        )

    def content_digest(self) -> str:
        """Identity of the passage, for proving the split stays disjoint. Delegates, as repair does."""
        return prompt_digest(self.passage)


def from_row(row: dict) -> ShapeATask:
    return ShapeATask(
        task_id=row["task_id"],
        passage=row["passage"],
        anchor=row["anchor"],
        duration_days=row["duration_days"],
        true_value=row["true_value"],
        wrong_value=row["wrong_value"],
        event=row["event"],
        prompt_text=row["prompt"],
    )


def load_tasks(path: Path = TASKS) -> tuple[ShapeATask, ...]:
    rows = json.loads(path.read_text())["tasks"]
    tasks = tuple(from_row(row) for row in rows)
    ids = [task.task_id for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise ShapeATaskError(f"duplicate task ids: {duplicates}")
    return tasks


def rank(task_id: str, salt: str = SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def split_manifest(
    tasks: tuple[ShapeATask, ...], salt: str = SPLIT_SALT, holdout: int = HOLDOUT_COUNT
) -> dict:
    """Exactly `holdout` passages, allocated across anchor days in proportion to how many exist.

    Stratifying by anchor is what stops the held-out half becoming "can it add days to a Monday": with
    one third held out and every anchor present in both halves, a gain cannot be an artefact of which
    weekdays landed where. Largest-remainder allocation keeps the count exact rather than approximately
    proportional, so `holdout` is a fact about the manifest and not an average.
    """
    if holdout >= len(tasks):
        raise ShapeATaskError(f"holding out {holdout} of {len(tasks)} leaves nothing to train on")

    by_anchor: dict[str, list[ShapeATask]] = {}
    for task in tasks:
        by_anchor.setdefault(task.anchor, []).append(task)

    take = _anchor_quotas(by_anchor, len(tasks), holdout)
    held_out = [
        task.task_id
        for anchor in sorted(by_anchor)
        for task in sorted(by_anchor[anchor], key=lambda task: rank(task.task_id, salt))[: take[anchor]]
    ]

    chosen = set(held_out)
    return {
        "salt": salt,
        "holdout_count": holdout,
        "train": [task.task_id for task in tasks if task.task_id not in chosen],
        "held_out": sorted(held_out),
    }


def _anchor_quotas(by_anchor: dict[str, list[ShapeATask]], total: int, holdout: int) -> dict[str, int]:
    """Proportional hold-out per anchor, integer, summing to `holdout` exactly (largest remainder)."""
    quotas = {anchor: len(group) * holdout / total for anchor, group in by_anchor.items()}
    take = {anchor: int(quota) for anchor, quota in quotas.items()}
    remaining = holdout - sum(take.values())
    # Largest remainder first, anchor name as the tie-break so the manifest is a pure function.
    for anchor in sorted(quotas, key=lambda a: (-(quotas[a] - take[a]), a))[:remaining]:
        take[anchor] += 1
    for anchor, group in by_anchor.items():
        if take[anchor] < 1 or take[anchor] >= len(group):
            raise ShapeATaskError(
                f"anchor {anchor}: holding out {take[anchor]} of {len(group)} leaves a half empty"
            )
    return take


@dataclass(frozen=True)
class ShapeATaskSet:
    """The corpus, split into the passages a run trains on and the passages it is judged on."""

    train: tuple[ShapeATask, ...]
    held_out: tuple[ShapeATask, ...]

    def __post_init__(self) -> None:
        if not self.train:
            raise ShapeATaskError("a task set with no training passages cannot source a batch")
        if not self.held_out:
            raise ShapeATaskError("a task set with no held-out passages cannot judge anything")
        shared = {task.task_id for task in self.train} & {task.task_id for task in self.held_out}
        if shared:
            raise ShapeATaskError(f"task ids in both halves: {sorted(shared)}")

    def by_id(self) -> dict[str, ShapeATask]:
        return {task.task_id: task for task in self.train + self.held_out}


def load_split(tasks: tuple[ShapeATask, ...] | None = None, path: Path = SPLIT) -> ShapeATaskSet:
    """Read the frozen manifest and refuse one that disagrees with the corpus on disk."""
    corpus = tasks if tasks is not None else load_tasks()
    manifest = json.loads(path.read_text())
    known = {task.task_id for task in corpus}

    named = manifest["train"] + manifest["held_out"]
    missing = [task_id for task_id in named if task_id not in known]
    if missing:
        raise ShapeATaskError(f"{path.name} names task ids the corpus does not carry: {missing}")

    train = tuple(task for task in corpus if task.task_id in set(manifest["train"]))
    held_out = tuple(task for task in corpus if task.task_id in set(manifest["held_out"]))
    assert_disjoint(
        [task.task_id for task in train],
        [task.passage for task in train],
        HoldoutSet(
            task_ids=frozenset(task.task_id for task in held_out),
            digests=frozenset(task.content_digest() for task in held_out),
        ),
    )
    return ShapeATaskSet(train=train, held_out=held_out)


@chz.chz
class ShapeADatasetBuilder(RLDatasetBuilder):
    """What `tinker_cookbook.rl.train.Config` takes for shape A. The training half only."""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    n_batches: int = 200
    policy: str = DEFAULT_POLICY
    dataset_name: str = "shape-a"

    async def __call__(self) -> tuple[TaskDataset, None]:
        corpus = load_split()
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer: renderers.Renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=tokenizer
        )
        return (
            TaskDataset(
                tasks=corpus.train,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                n_batches=self.n_batches,
                policy=self.policy,
                salt=SPLIT_SALT,
                dataset_name=self.dataset_name,
            ),
            None,
        )


__all__ = [
    "CHANCE_FLOOR",
    "DAYS",
    "HOLDOUT_COUNT",
    "SPLIT",
    "SPLIT_SALT",
    "TASKS",
    "ScorableTask",
    "ShapeADatasetBuilder",
    "ShapeATask",
    "ShapeATaskError",
    "ShapeATaskSet",
    "expected_day",
    "load_split",
    "load_tasks",
    "split_manifest",
    "stray_days",
]
