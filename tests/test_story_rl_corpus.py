"""The corpus and its split: shipped as data, reproducible as a rule, disjoint by two identities.

The split was frozen before any measurement, so the property that matters is not "it is balanced" but
"the file on disk is what its own rule produces" — otherwise a later edit could move a passage into the
training half and no test would notice.
"""

import json

import pytest

from story_rl.corpus import SPLIT, TASKS, TaskSet, book_of, load_split, load_tasks, split_manifest


def test_the_shipped_split_is_exactly_what_its_rule_produces():
    tasks = load_tasks()
    shipped = json.loads(SPLIT.read_text())

    assert split_manifest(tasks)["train"] == shipped["train"]
    assert split_manifest(tasks)["held_out"] == shipped["held_out"]


def test_every_passage_is_in_exactly_one_half():
    corpus = load_split()
    ids = [task.task_id for task in corpus.train + corpus.held_out]

    assert len(ids) == len(set(ids))
    assert len(ids) == len(load_tasks())


def test_both_halves_cover_every_book():
    """Otherwise the held-out half becomes 'can it repair one author'."""
    corpus = load_split()

    assert {book_of(task.task_id) for task in corpus.train} == {
        book_of(task.task_id) for task in corpus.held_out
    }


def test_the_held_out_half_is_large_enough_to_falsify_something():
    """Eight paired passages is the smallest size whose exact sign test can reach p < 0.05."""
    assert len(load_split().held_out) >= 8


def test_a_task_set_with_a_passage_in_both_halves_is_refused():
    corpus = load_split()

    with pytest.raises(ValueError, match="both halves"):
        TaskSet(train=corpus.train, held_out=(corpus.held_out[0], corpus.train[0]))


def test_a_task_set_with_an_empty_half_is_refused():
    corpus = load_split()

    with pytest.raises(ValueError, match="no held-out"):
        TaskSet(train=corpus.train, held_out=())
    with pytest.raises(ValueError, match="no training"):
        TaskSet(train=(), held_out=corpus.held_out)


def test_a_manifest_naming_a_task_outside_the_corpus_is_refused(tmp_path):
    shipped = json.loads(SPLIT.read_text())
    broken = {**shipped, "train": [*shipped["train"], "not-a-task"]}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(broken))

    with pytest.raises(ValueError, match="does not carry"):
        load_split(load_tasks(), path)


def test_the_corpus_file_is_the_one_the_package_ships():
    assert TASKS.exists() and TASKS.parent.name == "story_rl"
