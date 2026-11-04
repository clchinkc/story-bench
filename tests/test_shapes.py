"""Every shape stays alive: its split is disjoint, and its builder produces a batch.

Three shapes exist — `repair`, `shape-a` and `prose` — and only `prose` is in daily use. The other two were
one experiment each, and an experiment that stops being exercised stops working quietly: nothing fails until
someone reaches for it. This file is what fails instead.

It asserts only what a shape cannot survive without: a train half and a held-out half that do not overlap, and
a builder that returns group builders. It does not assert the reward, the task difficulty or whether a run
converges — those need a run, and claiming them here would be a test that proves its own fixture.
"""

from __future__ import annotations

import asyncio
import importlib
import os

# The suite forbids network; the tokenizer comes from the local cache, so pin the hub to offline before the
# tokenizer libraries are imported — the same rule `test_story_rl_pipeline.py` follows.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

SHAPES = ("repair", "shape-a", "prose")


def split_for(shape: str):
    """The split a shape trains on, loaded from the shape's own module — the same path the CLI takes."""
    module = importlib.import_module({
        "repair": "story_rl.corpus",
        "shape-a": "story_rl.shape_a",
        "prose": "story_rl.prose",
    }[shape])
    return module.load_split()


def builder_for(shape: str):
    from story_rl.train import builder_for as make_builder

    return make_builder(shape, steps=1, batch=2, group=2)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_shape_has_both_halves_and_they_do_not_overlap(shape):
    split = split_for(shape)
    train = {task.task_id for task in split.train}
    held = {task.task_id for task in split.held_out}

    assert train and held, f"{shape} has an empty half"
    assert not (train & held), f"{shape} trains on a task it is judged on"


@pytest.mark.parametrize("shape", SHAPES)
def test_a_shape_builds_a_batch(shape):
    dataset, _ = asyncio.run(builder_for(shape)())
    batch = dataset.get_batch(0)

    assert len(batch) == 2
    assert all(hasattr(group, "env_thunk") for group in batch)


def test_the_shapes_do_not_share_a_corpus():
    """Three shapes reading one corpus would make a "different shape" a label rather than an experiment."""
    firsts = {shape: split_for(shape).train[0].task_id for shape in SHAPES}
    assert len(set(firsts.values())) == len(SHAPES), firsts
