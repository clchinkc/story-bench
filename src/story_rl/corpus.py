"""The repair corpus: twenty passages over six books, and the 12/8 split that was frozen before measuring.

The split is a **shipped artefact**, not a runtime choice. `split_manifest` recomputes it as a pure
function so a test can prove the shipped file still matches its own rule; nothing regenerates it during a
run. That is what makes "the held-out half was chosen before any measurement" checkable rather than
claimed.

The first corpus's twelve passages are a subset of the twenty — the second corpus added eight — so one
task file and one split serve both the historical numbers and any future run. Splitting them was the
old arrangement and it made two datasets where one was meant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from holdout_guard import HoldoutSet, assert_disjoint
from repair_task import RepairTask, from_rows

DATA = Path(__file__).resolve().parent.parent.parent / "dataset" / "story_rl"
TASKS = DATA / "tasks_repair.json"
SPLIT = DATA / "split.json"

# Salt and fraction are part of the split's definition, not tunables: change either and the shipped
# manifest stops being reproducible, which the corpus test catches.
SPLIT_SALT = "story-rl-stageb-v2"
HOLDOUT_FRACTION = 0.4


@dataclass(frozen=True)
class TaskSet:
    """The corpus, split into the passages a run trains on and the passages it is judged on."""

    train: tuple[RepairTask, ...]
    held_out: tuple[RepairTask, ...]

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("a task set with no training passages cannot source a batch")
        if not self.held_out:
            raise ValueError("a task set with no held-out passages cannot judge anything")
        shared = {task.task_id for task in self.train} & {task.task_id for task in self.held_out}
        if shared:
            raise ValueError(f"task ids in both halves: {sorted(shared)}")

    def by_id(self) -> dict[str, RepairTask]:
        return {task.task_id: task for task in self.train + self.held_out}


def book_of(task_id: str) -> str:
    """The source book, from the `book-index` id convention the corpus uses."""
    return task_id.split("-")[0]


def rank(task_id: str, salt: str = SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def split_manifest(tasks: tuple[RepairTask, ...], salt: str = SPLIT_SALT) -> dict:
    """40 % of each book's passages held out, minimum one, chosen by hash order.

    Stratifying by book is what stops the held-out half becoming "can it repair Melville". The manifest
    is a pure function of the task ids so the shipped file can be re-derived and compared.
    """
    by_book: dict[str, list[RepairTask]] = {}
    for task in tasks:
        by_book.setdefault(book_of(task.task_id), []).append(task)

    held_out: list[str] = []
    for book in sorted(by_book):
        ranked = sorted(by_book[book], key=lambda task: rank(task.task_id, salt))
        held_out.extend(task.task_id for task in ranked[: max(1, round(len(ranked) * HOLDOUT_FRACTION))])

    chosen = set(held_out)
    return {
        "salt": salt,
        "holdout_fraction": HOLDOUT_FRACTION,
        "train": [task.task_id for task in tasks if task.task_id not in chosen],
        "held_out": sorted(held_out),
    }


def load_tasks(path: Path = TASKS) -> tuple[RepairTask, ...]:
    return from_rows(json.loads(path.read_text())["tasks"])


def load_split(tasks: tuple[RepairTask, ...] | None = None, path: Path = SPLIT) -> TaskSet:
    """Read the frozen manifest and refuse one that disagrees with the corpus on disk."""
    corpus = tasks if tasks is not None else load_tasks()
    manifest = json.loads(path.read_text())
    by_id = {task.task_id: task for task in corpus}

    named = manifest["train"] + manifest["held_out"]
    missing = [task_id for task_id in named if task_id not in by_id]
    if missing:
        raise ValueError(f"{path.name} names task ids the corpus does not carry: {missing}")

    train = tuple(by_id[task_id] for task_id in manifest["train"])
    held_out = tuple(by_id[task_id] for task_id in manifest["held_out"])
    assert_disjoint(
        [task.task_id for task in train],
        [task.passage for task in train],
        HoldoutSet(
            task_ids=frozenset(task.task_id for task in held_out),
            digests=frozenset(task.content_digest() for task in held_out),
        ),
    )
    return TaskSet(train=train, held_out=held_out)
