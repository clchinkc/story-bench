"""Offline-only suite: never read credentials or contact a provider."""
import socket
import dotenv
import json
import pytest
from measurement_contract import PROTOCOL, digest, context_packet

dotenv.load_dotenv = lambda *args, **kwargs: False


def pytest_sessionstart(session):
    def deny_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in offline regression tests")
    socket.socket.connect = deny_network


@pytest.fixture
def fixture_records():
    """Named independent outcomes; no network, native outcomes or human labels."""
    def make(models=("A", "B"), tasks=("easy", "hard"), samples=(0,)):
        assignments, generations, evaluations = [], [], []
        for model in models:
            for task_id in tasks:
                for sample in samples:
                    task = {"task_id": task_id, "task_type": "theory_conversion", "subtype": "default", "source": "EARLY CAUSE\nEND ‘quote’"}
                    assigned = dict(model=model, task_id=task_id, sample=sample, condition="plain", story_id="one-story", task=task, task_hash=digest(task), model_snapshot=model + "@frozen", judge_snapshot="fake-judge@1", prompt_version="offline-fixture-v1")
                    assignments.append(assigned)
                    ident = f"{model}-{task_id}-{sample}"
                    parts = {"task": json.dumps(task, sort_keys=True, ensure_ascii=False)}
                    prompt, context = context_packet(parts, max_bytes=10000)
                    generation = dict(**{k: assigned[k] for k in ("model", "task_id", "sample", "condition", "task_hash", "model_snapshot")}, protocol=PROTOCOL, record_id="gen-" + ident, status="completed", timestamp="2026-09-21T00:00:00Z", output="Complete end.", output_hash=digest("Complete end."), prompt=prompt, prompt_hash=digest(prompt), context_parts=parts, context=context, finish_reason="stop", attempts=[dict(attempt_id="g-" + ident, role="generation", status="completed", cost_usd=2., usage={"prompt_tokens": 20, "completion_tokens": 5, "reasoning_tokens": 2})])
                    generations.append(generation)
                    generation["prompt_version"] = assigned["prompt_version"]
                    parts = {**parts, "output": generation["output"]}
                    prompt, context = context_packet(parts, max_bytes=10000)
                    # Independently set A .75/.25 and B .5/.5 => both means .5.
                    score = (.75 if task_id == "easy" else .25) if model == "A" else .5
                    evaluation = dict(**{k: assigned[k] for k in ("model", "task_id", "sample", "condition", "task_hash", "model_snapshot", "judge_snapshot")}, protocol=PROTOCOL, record_id="eval-" + ident, status="completed", timestamp="2026-09-21T00:00:00Z", generation_hash=digest(generation), prompt=prompt, prompt_hash=digest(prompt), context_parts=parts, context=context, finish_reason="stop", llm_results={k: score for k in ["beats_score", "preservation_score", "structural_accuracy_score", "tone_score"]}, attempts=[dict(attempt_id="e-" + ident, role="research_evaluation", status="completed", cost_usd=.25, usage=None)])
                    evaluations.append(evaluation)
                    evaluation["prompt_version"] = assigned["prompt_version"]
        manifest = {"protocol": PROTOCOL, "assignments": assignments}
        data = {"protocol": PROTOCOL, "assignment_hash": digest(manifest), "generations": generations, "evaluations": evaluations}
        return manifest, data
    return make
