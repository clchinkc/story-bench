"""P5 holdout isolation: neither a task id nor a passage may cross into training."""

import pytest

from holdout_guard import (
    DIGEST_PREFIX,
    HoldoutLeakError,
    HoldoutSet,
    assert_disjoint,
    prompt_digest,
)

SEALED = "The tide came in over the harbour wall."
HOLDOUT = HoldoutSet(
    task_ids=frozenset({"sealed-1"}),
    digests=frozenset({prompt_digest(SEALED)}),
)


def test_a_disjoint_training_set_passes():
    assert_disjoint(["train-1"], ["A different passage entirely."], HOLDOUT)


def test_reusing_a_holdout_task_id_is_refused():
    with pytest.raises(HoldoutLeakError, match="sealed-1"):
        assert_disjoint(["train-1", "sealed-1"], ["A different passage entirely."], HOLDOUT)


def test_retitling_the_same_passage_is_refused_by_content_digest():
    """The reason both identities are checked: a rename defeats an id-only guard."""
    retitled = "  The tide came in\n over the harbour wall.  "

    with pytest.raises(HoldoutLeakError, match=DIGEST_PREFIX):
        assert_disjoint(["brand-new-id"], [retitled], HOLDOUT)


def test_every_overlap_is_reported_not_just_the_first():
    with pytest.raises(HoldoutLeakError) as excinfo:
        assert_disjoint(["sealed-1"], [SEALED], HOLDOUT)

    assert "sealed-1" in str(excinfo.value)
    assert DIGEST_PREFIX in str(excinfo.value)


def test_an_empty_holdout_is_refused_because_it_proves_nothing():
    with pytest.raises(HoldoutLeakError, match="proves nothing"):
        HoldoutSet(task_ids=frozenset(), digests=frozenset({prompt_digest(SEALED)}))

    with pytest.raises(HoldoutLeakError, match="cannot catch"):
        HoldoutSet(task_ids=frozenset({"sealed-1"}), digests=frozenset())


def test_a_malformed_digest_is_refused():
    with pytest.raises(HoldoutLeakError, match="not 'sha256"):
        HoldoutSet(task_ids=frozenset({"sealed-1"}), digests=frozenset({"abc123"}))


def test_prompt_digest_normalises_surrounding_whitespace():
    assert prompt_digest("a b") == prompt_digest("  a\n b  ")
