import pytest
from agentic_evaluator import create_constraint_discovery_oracle
from agentic_evaluator import AgenticEvaluator, _keyword_fallback
from agentic_generator import AgenticGenerator, AgenticPromptBuilder


class TopicMatcher:
    def call(self, **kwargs):
        return type("Response", (), {"success": True, "content": "1"})()


def test_R06_unqualified_oracle_must_not_return_topic_polarity():
    task = {"hidden_constraints": [{"id": "one", "constraint": "The door is locked", "answer": "NO"}]}
    with pytest.raises(ValueError, match="unqualified"):
        create_constraint_discovery_oracle(task, llm_client=TopicMatcher())


@pytest.mark.parametrize("question", ["Is the door locked?", "Is it unlocked?", "Is it not locked?", "Can anyone enter?", "Is it locked and blue?", "What is the moon made of?", "Locked and unlocked?"])
def test_disabled_oracle_never_invents_polarity_for_any_question(question):
    with pytest.raises(ValueError, match="unqualified"):
        _keyword_fallback(question, [{"answer": "YES", "question_patterns": [question]}])
    with pytest.raises(ValueError, match="unqualified"):
        AgenticPromptBuilder.build_oracle_prompt(question, {}, [])


def test_disabled_oracle_all_measured_entry_points():
    gen = object.__new__(AgenticGenerator)
    with pytest.raises(ValueError, match="unqualified"):
        gen.run_constraint_discovery({}, "fake", 0, lambda question: "YES")
    evaluator = object.__new__(AgenticEvaluator)
    with pytest.raises(ValueError, match="unqualified"):
        evaluator.evaluate_agentic_result({"agentic_type": "constraint_discovery"}, {})
