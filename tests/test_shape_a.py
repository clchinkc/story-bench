"""Shape A, exercised offline: the corpus cannot print its own answer, and the reward notices.

The loader already refuses a malformed row, so a test that only feeds it good rows proves little. What
these tests establish is that the *shipped* corpus satisfies the invariants when recomputed here,
independently, and that the reward pays for the computed day and refuses every shortcut the shape was
built to remove — copying the visible wrong day, and copying the passage.

No network, no spend: scoring is a pure function of text.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

from repair_task import RepairTask, score_rollout  # noqa: E402
from scorable import ScorableTask  # noqa: E402
from story_rl import shape_a  # noqa: E402
from story_rl.corpus import load_split as load_repair_split  # noqa: E402
from story_rl.env import TaskEnv  # noqa: E402
from story_rl.shape_a import ShapeATask, ShapeATaskError, load_split, load_tasks  # noqa: E402

WEEK = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def day_after(anchor: str, days: int) -> str:
    """The arithmetic again, written here rather than imported — a check, not a restatement."""
    index = WEEK.index(anchor)
    for _ in range(days):
        index = (index + 1) % len(WEEK)
    return WEEK[index]


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


@pytest.fixture(scope="module")
def corpus():
    return load_split()


def test_the_corpus_is_forty_eight_passages_over_every_anchor(tasks):
    assert len(tasks) == 48
    assert len({task.task_id for task in tasks}) == 48
    assert {task.anchor for task in tasks} == set(WEEK)
    assert {task.duration_days for task in tasks} == {2, 3, 4, 5, 6}


def test_every_passage_refuses_to_print_the_day_it_asks_for(tasks):
    for task in tasks:
        assert task.true_value == day_after(task.anchor, task.duration_days), task.task_id
        assert task.true_value not in task.passage, task.task_id
        assert task.true_value not in task.prompt_text, task.task_id
        assert task.passage.count(task.wrong_value) == 1, task.task_id
        strays = [day for day in WEEK if day not in (task.anchor, task.wrong_value)
                  and day in task.passage]
        assert strays == [], (task.task_id, strays)


def test_the_split_is_sixteen_held_out_and_recomputes_from_its_own_rule(tasks, corpus):
    assert len(corpus.train) == 32
    assert len(corpus.held_out) == 16
    assert not ({task.task_id for task in corpus.train}
                & {task.task_id for task in corpus.held_out})

    shipped = json.loads(Path(shape_a.SPLIT).read_text())
    assert shipped == shape_a.split_manifest(tasks)
    assert len(shipped["held_out"]) == shape_a.HOLDOUT_COUNT


def test_every_anchor_appears_in_both_halves(corpus):
    """Otherwise a gain can be an artefact of which weekdays landed where, not of the arithmetic."""
    train_anchors = {task.anchor for task in corpus.train}
    held_anchors = {task.anchor for task in corpus.held_out}
    assert train_anchors == set(WEEK)
    assert held_anchors == set(WEEK)


def test_a_continuation_that_names_the_computed_day_passes(corpus):
    task = corpus.held_out[0]
    score = score_rollout(
        f"She said nothing for a while. The {task.event} had been on the {task.true_value}, "
        "she was certain of it.",
        task,
    )
    assert score.parsed is True
    assert score.strict == 1


def test_naming_the_visible_wrong_day_fails(tasks):
    task = tasks[0]
    score = score_rollout(
        f"She said nothing for a while. The {task.event} had been on the {task.wrong_value}, "
        "she was certain of it.",
        task,
    )
    assert score.strict == 0
    assert score.leaked == (task.wrong_value,)


def test_naming_no_day_at_all_fails(tasks):
    """The measured majority failure on the prompt arm: fluent prose that never commits to a day."""
    task = tasks[0]
    score = score_rollout(
        f"She said nothing for a while, and then she wrote it down. The {task.event} was settled.",
        task,
    )
    assert score.parsed is True
    assert score.strict == 0
    assert score.missing == (task.true_value,)


def test_copying_the_passage_is_not_a_way_to_pass(tasks):
    """The shortcut this shape exists to remove. Echoing the prompt must never pay."""
    for task in tasks:
        assert score_rollout(task.passage, task).strict == 0, task.task_id


def test_the_env_serves_a_shape_a_task_through_the_same_seam(tasks):
    task = tasks[0]
    assert isinstance(task, ScorableTask)
    assert isinstance(load_repair_split().train[0], RepairTask)
    assert isinstance(load_repair_split().train[0], ScorableTask)
    assert task.prompt() == task.prompt_text
    assert task.as_prompt_task().canon[0].value == task.true_value
    assert task.as_prompt_task().contradictions[0].value == task.wrong_value


def test_the_loader_refuses_a_passage_that_prints_its_own_answer():
    with pytest.raises(ShapeATaskError, match="printed in the passage"):
        ShapeATask(
            task_id="t99",
            passage="He came on a Monday. It rained 2 days later, so that was the Wednesday.",
            anchor="Monday",
            duration_days=2,
            true_value="Wednesday",
            wrong_value="Friday",
            event="the rain",
            prompt_text="Story so far: He came on a Monday. It rained 2 days later, so that was the "
            "Wednesday.",
        )


def test_the_loader_refuses_a_stray_weekday():
    with pytest.raises(ShapeATaskError, match="stray weekday"):
        ShapeATask(
            task_id="t98",
            passage="He came on a Monday and left on the Friday. The rain came 2 days after the "
            "Monday, so it was the Sunday.",
            anchor="Monday",
            duration_days=2,
            true_value="Wednesday",
            wrong_value="Sunday",
            event="the rain",
            prompt_text="Story so far: He came on a Monday and left on the Friday. The rain came 2 "
            "days after the Monday, so it was the Sunday.",
        )
