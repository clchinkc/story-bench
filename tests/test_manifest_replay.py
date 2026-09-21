import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from measurement_contract import digest, project_report
from results_db import ResultsDatabase
from llm_client import LLMClient
from evaluator import BenchmarkEvaluator


@pytest.mark.parametrize("mutation", ["task", "model", "judge", "prompt", "output", "protocol", "assignment", "duplicate", "conflict", "attempt", "sample", "condition", "prompt_version"])
def test_stale_mixed_collision_unassigned_rejected(fixture_records, mutation):
    manifest, data = fixture_records()
    gen = data["generations"][0]; ev = data["evaluations"][0]
    if mutation == "task": gen["task_hash"] = "old"
    elif mutation == "model": gen["model_snapshot"] = "changed"
    elif mutation == "judge": ev["judge_snapshot"] = "changed"
    elif mutation == "prompt": gen["prompt"] += " changed"
    elif mutation == "output": gen["output"] += " changed"
    elif mutation == "protocol": gen["protocol"] = "legacy"
    elif mutation == "assignment": data["assignment_hash"] = "old"
    elif mutation in {"duplicate", "conflict"}:
        data["evaluations"].append(copy.deepcopy(ev))
        if mutation == "conflict": data["evaluations"][-1]["llm_results"]["beats_score"] = 0
    elif mutation == "attempt": data["generations"][1]["attempts"][0]["attempt_id"] = gen["attempts"][0]["attempt_id"]
    elif mutation == "sample": gen["sample"] = 99
    elif mutation == "condition": gen["condition"] = "other"
    elif mutation == "prompt_version": gen["prompt_version"] = "stale"
    with pytest.raises(ValueError):
        project_report(data, manifest)


def test_concurrent_append_lock_integrity_idempotence_and_conflicts(fixture_records, tmp_path):
    manifest, data = fixture_records(samples=tuple(range(4)))
    path = tmp_path / "db.json"
    def add(row):
        ResultsDatabase(path, assignment_manifest=manifest).add_generation(row)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, data["generations"]))
    db = ResultsDatabase(path, assignment_manifest=manifest)
    assert {x["record_id"] for x in db._data["generations"]} == {x["record_id"] for x in data["generations"]}
    for ev in data["evaluations"]:
        db.add_evaluation(ev)
    original = path.read_bytes()
    db.add_generation(data["generations"][0])
    assert path.read_bytes() == original
    conflicting = copy.deepcopy(data["generations"][0]); conflicting["timestamp"] = "changed"
    with pytest.raises(ValueError, match="Conflicting"):
        db.add_generation(conflicting)
    assert path.read_bytes() == original
    for collection in ["generations", "evaluations"]:
        stored = json.loads(path.read_text())[collection]
        assert len(stored) == len({x["record_id"] for x in stored}) == len(data[collection])


def test_stale_cache_lookup_rejected_and_sample_condition_preserved(fixture_records, tmp_path):
    manifest, data = fixture_records(samples=(0, 1))
    path = tmp_path / "db.json"; path.write_text(json.dumps(data))
    db = ResultsDatabase(path)
    result = db.get_generation("easy", "A", 1, condition="plain", assignment_manifest=manifest)
    assert result["sample"] == 1
    manifest["assignments"][0]["model_snapshot"] = "new-model"
    with pytest.raises(ValueError, match="stale"):
        db.get_generation("easy", "A", 1, condition="plain", assignment_manifest=manifest)


def test_actual_dispatch_boundaries_fail_before_transport_even_retries():
    called = []
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: called.append(k))))
    client = object.__new__(LLMClient); client.client = fake
    with pytest.raises(ValueError, match="W3"):
        client.call("model", [], retry_attempts=100, retry_delay=0)
    evaluator = object.__new__(BenchmarkEvaluator); evaluator.client = fake
    with pytest.raises(ValueError, match="W3"):
        evaluator.evaluate_generation({}, {})
    assert called == []


@pytest.mark.parametrize("reason", ["length", "content_filter", None, "error"])
def test_provider_response_truncation_invalid_and_cost_unknown(reason):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="unfinished"), finish_reason=reason)], usage=None)
    parsed = object.__new__(LLMClient)._parse_response(response)
    assert not parsed.success and parsed.cost is None
    assert parsed.prompt_tokens is parsed.completion_tokens is parsed.reasoning_tokens is None


def test_unknown_cost_is_not_a_free_call():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="end"), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2))
    parsed = object.__new__(LLMClient)._parse_response(response)
    assert parsed.success and parsed.cost is None
