from agentic_generator import AgenticPromptBuilder
from agentic_evaluator import AgenticEvalPromptBuilder
import json
import pytest
from types import SimpleNamespace
from measurement_contract import context_packet, CoverageError, digest, project_report
from agentic_generator import AgenticGenerator, AgenticConfig
from agentic_evaluator import AgenticEvaluator
from evaluator import EvalPromptBuilder
from llm_client import LLMResponse


def test_R05_revision_ending_and_quote_are_preserved():
    story = "EARLY CAUSE\n" + "x" * 2200 + '\nENDING: “Door remains locked.”'
    prompt = AgenticPromptBuilder.build_iterative_revision_feedback(story, ["Fix ending"])
    assert story in prompt


def test_R05_all_versions_and_critiques_are_preserved():
    turns = []
    for n in range(6):
        turns += [{"turn_type": "generation", "metadata": {"revision": n}, "content": f"CAUSE-{n}" + "x" * 2100 + f"END-{n}"},
                  {"turn_type": "critique", "metadata": {"round": n}, "content": f"CRIT-{n}" + "y" * 500 + f"FIX-{n}"}]
    prompt = AgenticEvalPromptBuilder.build_critique_improvement_eval({}, {"turns": turns, "output": "final"})
    for turn in turns:
        assert turn["content"] in prompt


def test_standard_judge_early_cause_ending_and_exact_quotes():
    cause = '“Promise.”\n' + "x" * 2000 + "CAUSE AT END"
    after = "y" * 2000 + "ENDING CONTRADICTION"
    task = dict(task_type="beat_interpolation", beat_before={"name": "before", "content": cause}, beat_after={"name": "after", "content": after}, missing_beat={"name": "middle"}, requirements={"must_include": [], "word_count": [1, 200]})
    prompt = EvalPromptBuilder.build_eval_prompt(task, "generated", 1)
    assert cause in prompt and after in prompt


def test_explicit_budget_failure_has_auditable_hash_and_no_dispatch():
    calls = []
    client = SimpleNamespace(call=lambda **kwargs: calls.append(kwargs))
    generator = AgenticGenerator(config=AgenticConfig(context_budget_bytes=5), llm_client=client)
    with pytest.raises(CoverageError) as exc:
        generator._call_model("fake", [{"role": "user", "content": "EARLY CAUSE; ENDING"}])
    assert calls == []
    assert exc.value.receipt["coverage"] == "failed"
    assert exc.value.receipt["stop_reason"] == "context_budget"
    assert exc.value.receipt["bytes"] > 5
    assert len(exc.value.receipt["packet_hash"]) == 64


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_agentic_truncated_output_never_valid(finish):
    client = SimpleNamespace(call=lambda **kw: LLMResponse("cut off", 10, 10, 0, 2., True, finish_reason=finish))
    gen = AgenticGenerator(config=AgenticConfig(context_budget_bytes=10000), llm_client=client)
    with pytest.raises(ValueError, match="truncated"):
        gen._call_model("fake", [{"role": "user", "content": "whole prompt"}])


def test_fake_judge_receives_full_trajectory_and_records_receipt():
    calls = []
    def call(**kw):
        calls.append(kw)
        return LLMResponse(json.dumps(dict.fromkeys(["improvement_trajectory", "feedback_responsiveness", "preservation", "constraint_satisfaction", "beat_execution", "narrative_quality"], .5)), 10, 10, 0, 0., True, finish_reason="stop")
    evaluator = AgenticEvaluator(llm_client=SimpleNamespace(call=call))
    task = dict(task_id="fixture", agentic_type="iterative_revision", context_budget_bytes=100000)
    turns = [{"turn_type": "generation", "content": f"CAUSE-{n}" + "x" * 2200 + f"END-{n}"} for n in range(6)]
    result = evaluator.evaluate_agentic_result(task, {"turns": turns, "output": "final"})
    assert result.success and result.final_score is None
    assert result.context_receipt["coverage"] == "complete"
    assert all(t["content"] in calls[0]["messages"][1]["content"] for t in turns)


@pytest.mark.parametrize("field", ["finish_reason", "context_parts", "context"])
def test_repaired_report_rejects_context_and_completion_tampering(fixture_records, field):
    manifest, data = fixture_records()
    gen = data["generations"][0]
    if field == "finish_reason": gen[field] = "length"
    elif field == "context_parts": gen[field]["task"] = "missing ending"
    else: gen[field]["coverage"] = "partial"
    with pytest.raises(ValueError):
        project_report(data, manifest)
