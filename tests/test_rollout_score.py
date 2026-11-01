"""`score_rollout` is the one place a completion becomes a reward, so both projections are pinned."""

from repair_task import from_row, score_rollout

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


def test_prose_policy_scores_free_text():
    """Tinker cannot constrain decoding, so this is the configuration a real run uses."""
    score = score_rollout("Mara turned, her grey eyes catching the lamplight.", from_row(ROW))

    assert score.parsed is True
    assert score.how.startswith("prose/")
    assert score.strict == 1
    assert score.correction == 1


def test_a_negated_mention_separates_the_two_projections():
    score = score_rollout("Her eyes were grey — not brown — in the lamplight.", from_row(ROW))

    assert score.leaked == ("brown",)
    assert score.strict == 0
    assert score.correction == 1


def test_omitting_the_truth_fails_both():
    score = score_rollout("Mara turned toward the door.", from_row(ROW))

    assert score.missing == ("grey",)
    assert score.strict == 0
    assert score.correction == 0


def test_a_blank_completion_is_parsed_and_scores_zero():
    score = score_rollout("   ", from_row(ROW))

    assert score.parsed is True
    assert score.how == "prose/raw"
    assert score.strict == 0 and score.correction == 0


def test_the_strict_policy_reports_unparsed_free_prose_rather_than_scoring_it():
    """The measured fact behind the policy choice: strict parses 0 % of free decoding."""
    score = score_rollout("Mara turned, her grey eyes catching the lamplight.", from_row(ROW), "strict")

    assert score.parsed is False
    assert score.strict == 0 and score.correction == 0
