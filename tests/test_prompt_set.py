"""P5 prompt-set construction: canon, declared breaks, fail-closed validation."""

import pytest

from prompt_set import (
    CanonFact,
    PromptSetError,
    PromptTask,
    build_task,
    inject_break,
    validate_task_set,
)

PASSAGE = "Mara had grey eyes. The lantern was lit."
EYES = CanonFact("Mara", "eye_colour", "grey")
LANTERN = CanonFact("the lantern", "state", "lit")


def test_a_task_without_canon_is_refused_because_it_cannot_produce_contrast():
    """D019 measures 0.0 % contrast when nothing can be contradicted."""
    with pytest.raises(PromptSetError, match="no canon fact"):
        PromptTask(task_id="t1", passage=PASSAGE, canon=(), contradictions=())


def test_canon_may_not_state_the_same_attribute_twice():
    with pytest.raises(PromptSetError, match="same"):
        PromptTask(
            task_id="t1",
            passage=PASSAGE,
            canon=(EYES, CanonFact("Mara", "eye_colour", "brown")),
            contradictions=(),
        )


def test_a_contradiction_must_be_a_canon_fact():
    with pytest.raises(PromptSetError, match="not a canon fact"):
        PromptTask(
            task_id="t1",
            passage=PASSAGE,
            canon=(EYES,),
            contradictions=(LANTERN,),
        )


def test_a_contradiction_asserting_the_true_value_is_not_a_break():
    with pytest.raises(PromptSetError, match="not a break"):
        PromptTask(task_id="t1", passage=PASSAGE, canon=(EYES,), contradictions=(EYES,))


def test_inject_break_replaces_the_value_exactly_once_and_returns_the_break():
    text, broken = inject_break(PASSAGE, EYES, "brown")

    assert "grey" not in text
    assert "brown eyes" in text
    assert broken == CanonFact("Mara", "eye_colour", "brown")


def test_inject_break_refuses_a_value_the_passage_never_states():
    with pytest.raises(PromptSetError, match="never states"):
        inject_break(PASSAGE, CanonFact("Mara", "build", "tall"), "short")


def test_inject_break_refuses_an_ambiguous_value():
    with pytest.raises(PromptSetError, match="ambiguous"):
        inject_break("grey eyes and grey sky", EYES, "brown")


def test_inject_break_refuses_a_no_op():
    with pytest.raises(PromptSetError, match="not a break"):
        inject_break(PASSAGE, EYES, "grey")


def test_build_task_applies_every_declared_break():
    task = build_task("t1", PASSAGE, (EYES, LANTERN), [(EYES, "brown"), (LANTERN, "dark")])

    assert task.passage == "Mara had brown eyes. The lantern was dark."
    assert {fact.value for fact in task.contradictions} == {"brown", "dark"}


def test_a_task_set_refuses_duplicates_and_emptiness():
    task = build_task("t1", PASSAGE, (EYES,), [(EYES, "brown")])

    assert validate_task_set([task]) == (task,)
    with pytest.raises(PromptSetError, match="duplicate task ids"):
        validate_task_set([task, task])
    with pytest.raises(PromptSetError, match="at least one task"):
        validate_task_set([])


def test_canon_fact_refuses_blank_fields():
    with pytest.raises(PromptSetError, match="non-blank"):
        CanonFact("Mara", "eye_colour", "   ")
