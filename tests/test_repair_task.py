"""The repair task: its prompt is frozen, its rows cannot disagree with their own canon, and one
function turns a raw completion into the two rewards a run reports.

Behaviour only — every assertion is about what a caller sees (a rendered prompt, a raised error, a
reward), never about how the module stores it.
"""

import pytest

from prompt_set import CanonFact
from repair_task import RepairTask, RepairTaskError, from_row, from_rows, score_rollout

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

# Pinned literal. If the template drifts, the error rate it produced stops describing a new run.
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


def test_the_prompt_states_both_values_so_the_error_is_visible():
    prompt = from_row(ROW).prompt()

    assert "grey" in prompt and "brown" in prompt


def test_a_row_whose_declared_truth_disagrees_with_its_canon_is_refused():
    with pytest.raises(RepairTaskError, match="is not the canon value"):
        from_row({**ROW, "true_value": "blue"})


def test_a_row_whose_broken_fact_is_absent_from_its_canon_is_refused():
    with pytest.raises(RepairTaskError, match="not in the task canon"):
        from_row({**ROW, "broken_attribute": "height"})


def test_duplicate_ids_are_refused_rather_than_one_shadowing_the_other():
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


def test_the_prompt_task_carries_the_whole_canon_and_exactly_one_break():
    task = from_row(ROW).as_prompt_task()

    assert len(task.canon) == 2
    assert len(task.contradictions) == 1
    assert task.contradictions[0].value == "brown"


def test_the_content_digest_ignores_whitespace_so_the_split_guard_is_not_fooled():
    task = from_row(ROW)
    reflowed = from_row({**ROW, "passage": "  " + ROW["passage"].replace(" ", "  ") + "  "})

    assert task.content_digest() == reflowed.content_digest()
    assert task.content_digest().startswith("sha256:")


def test_a_continuation_using_the_truth_passes_and_one_echoing_the_error_does_not():
    task = from_row(ROW)

    truthful = score_rollout("Mara turned, her grey eyes catching the lamplight.", task)
    echoing = score_rollout("Mara turned, her brown eyes catching the lamplight.", task)

    assert (truthful.strict, truthful.correction) == (1, 1)
    assert (echoing.strict, echoing.correction) == (0, 0)
    assert echoing.leaked == ("brown",)


def test_naming_the_error_in_order_to_reject_it_separates_the_two_projections():
    """What the diagnostic projection exists to expose: repaired, but the error was written anyway."""
    score = score_rollout("Her eyes were grey — not brown — in the lamplight.", from_row(ROW))

    assert score.strict == 0
    assert score.correction == 1


def test_omitting_the_truth_fails_both_projections():
    score = score_rollout("Mara turned toward the door.", from_row(ROW))

    assert (score.strict, score.correction) == (0, 0)
    assert score.missing == ("grey",)


def test_a_blank_completion_is_a_wrong_answer_not_an_unmeasured_one():
    score = score_rollout("   ", from_row(ROW))

    assert score.parsed is True
    assert (score.strict, score.correction) == (0, 0)


def test_a_json_shaped_parser_reports_unparsed_free_prose_rather_than_scoring_it():
    """The measured fact behind the policy choice: strict parsing sees nothing without a schema."""
    score = score_rollout("Mara turned, her grey eyes catching the lamplight.", from_row(ROW), "strict")

    assert score.parsed is False
    assert (score.strict, score.correction) == (0, 0)
