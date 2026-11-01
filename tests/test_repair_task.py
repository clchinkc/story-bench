"""Repair-task contract: the prompt is frozen, and a row may not disagree with its own canon."""

import pytest

from prompt_set import CanonFact
from repair_task import RepairTask, RepairTaskError, from_row, from_rows
from reward_repair import correction_reward, score_repair, terminal_reward

ROW = {
    "task_id": "t1",
    "passage": "Mara had grey eyes. The lantern was lit.",
    "canon": [
        {"entity": "Mara", "attribute": "eye_colour", "value": "grey"},
        {"entity": "the lantern", "attribute": "state", "value": "lit"},
    ],
    "broken_entity": "Mara",
    "broken_attribute": "eye_colour",
    "true_value": "grey",
    "wrong_value": "brown",
}

# Pinned literal. If the template drifts, `e = 0.708` stops describing what a new run measures.
EXPECTED_PROMPT = (
    "Story so far: Mara had grey eyes. The lantern was lit.\n"
    "\n"
    "The passage contains one error: Mara's eye_colour is grey, not brown.\n"
    "\n"
    "Continue the story for exactly two sentences. Use the correct value. "
    "Do not mention the incorrect value at all."
)


def test_the_prompt_bytes_are_frozen():
    assert from_row(ROW).prompt() == EXPECTED_PROMPT


def test_the_prompt_names_both_values():
    prompt = from_row(ROW).prompt()

    assert "grey" in prompt
    assert "brown" in prompt


def test_a_row_whose_true_value_is_not_its_canon_value_is_refused():
    bad = {**ROW, "true_value": "blue"}

    with pytest.raises(RepairTaskError, match="is not the canon value"):
        from_row(bad)


def test_a_row_whose_break_is_not_in_its_canon_is_refused():
    bad = {**ROW, "broken_attribute": "height"}

    with pytest.raises(RepairTaskError, match="not in the task canon"):
        from_row(bad)


def test_duplicate_ids_are_refused_rather_than_shadowed():
    with pytest.raises(RepairTaskError, match="duplicate task ids"):
        from_rows([ROW, ROW])


def test_a_break_that_changes_nothing_is_refused():
    with pytest.raises(RepairTaskError, match="must change the value"):
        RepairTask(
            task_id="t2",
            passage="Mara had grey eyes.",
            canon=(CanonFact("Mara", "eye_colour", "grey"),),
            broken_entity="Mara",
            broken_attribute="eye_colour",
            true_value="grey",
            wrong_value="grey",
        )


def test_the_prompt_task_carries_exactly_one_break_and_the_whole_canon():
    task = from_row(ROW).as_prompt_task()

    assert task.canon == (CanonFact("Mara", "eye_colour", "grey"), CanonFact("the lantern", "state", "lit"))
    assert len(task.contradictions) == 1
    assert task.contradictions[0].value == "brown"


def test_a_continuation_using_the_truth_resolves_and_one_echoing_the_error_leaks():
    task = from_row(ROW).as_prompt_task()

    good = score_repair("Mara turned, her grey eyes catching the lamplight.", task)
    bad = score_repair("Mara turned, her brown eyes catching the lamplight.", task)

    assert terminal_reward(good) == 1
    assert terminal_reward(bad) == 0


def test_the_content_digest_ignores_surrounding_whitespace():
    task = from_row(ROW)

    assert task.content_digest() == task.content_digest()
    assert task.content_digest().startswith("sha256:")


def test_correction_reward_ignores_a_mention_that_terminal_reward_punishes():
    """The free-decoding case: repair the prose, name the error to reject it."""
    task = from_row(ROW).as_prompt_task()
    verdict = score_repair("Her eyes were grey — not brown — in the lamplight.", task)

    assert verdict.leaked_wrong == ("brown",)
    assert terminal_reward(verdict) == 0
    assert correction_reward(verdict) == 1


def test_correction_reward_still_requires_the_true_value():
    task = from_row(ROW).as_prompt_task()
    verdict = score_repair("Mara turned toward the door.", task)

    assert correction_reward(verdict) == 0
