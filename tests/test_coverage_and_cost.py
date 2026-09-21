import pytest
from results_db import ResultsDatabase
from measurement_contract import STATUSES, digest, project_report


def test_R01_R09_legacy_success_only_report_cannot_rank(tmp_path):
    db = ResultsDatabase(tmp_path / "records.json")
    db._data["evaluations"] = [dict(model="partial", task_id="easy", evaluator_model="judge", success=True, final_score=1.)]
    with pytest.raises(ValueError, match="assignment"):
        db.get_results_summary()


def test_R02_failed_attempt_spend_is_retained(tmp_path):
    db = ResultsDatabase(tmp_path / "records.json")
    db._data["generations"] = [dict(model="m", task_id="easy", success=True, generation_cost=2.), dict(model="m", task_id="hard", success=False, generation_cost=7.)]
    db._data["evaluations"] = [dict(model="m", task_id="easy", evaluator_model="judge", success=True, final_score=1., task_type="beat_revision")]
    # Frozen report drops the failed $7 call. The repaired report requires assignments.
    try:
        report = db.get_results_summary()
    except ValueError as exc:
        assert "assignment" in str(exc)
    else:
        assert report["models"]["m"]["generation_cost"] == 9.


def test_partial_easy_only_never_becomes_complete_or_better(fixture_records):
    manifest, data = fixture_records()
    data["generations"] = [g for g in data["generations"] if not (g["model"] == "A" and g["task_id"] == "hard")]
    data["evaluations"] = [e for e in data["evaluations"] if not (e["model"] == "A" and e["task_id"] == "hard")]
    report = project_report(data, manifest)
    a = report["models"][0]
    assert a["assigned"] == 2 and a["valid_diagnostics"] == 1
    assert a["statuses"] == {"completed": 1, "absent": 1}
    assert a["complete_assigned_diagnostic_mean"] is None
    assert a["deployment_cost"]["known_usd"] == 2
    assert a["deployment_cost"]["total_usd"] is None
    assert report["matched_valid_task_samples"] == [["easy", 0]]


@pytest.mark.parametrize("status", sorted(STATUSES - {"completed"}))
def test_failure_taxonomy_preserves_denominator_and_spend(fixture_records, status):
    manifest, data = fixture_records(models=("A",), tasks=("hard",))
    gen = data["generations"][0]
    gen["status"] = status
    gen["attempts"][0].update(status=status, cost_usd=7)
    if status in {"not_run", "absent"}:
        gen["attempts"] = []
    data["evaluations"] = []
    row = project_report(data, manifest)["models"][0]
    assert row["statuses"] == {status: 1}
    assert row["assigned"] == 1 and row["valid_diagnostics"] == 0
    assert row["complete_assigned_diagnostic_mean"] is None
    assert row["deployment_cost"]["total_usd"] == (None if status == "absent" else 0 if status == "not_run" else 7)


def test_all_attempt_generation_critic_retry_selection_research_costs(fixture_records):
    manifest, data = fixture_records(models=("A",), tasks=("easy",))
    gen = data["generations"][0]
    for n, role in enumerate(["generation", "critic", "selection", "retrieval"]):
        gen["attempts"].append(dict(attempt_id=f"extra-{n}", role=role, status="provider_failure" if n == 0 else "completed", cost_usd=n + 1, usage=None))
    data["evaluations"][0]["generation_hash"] = digest(gen)
    result = project_report(data, manifest)["models"][0]
    assert result["deployment_cost"]["total_usd"] == 12
    assert result["deployment_cost"]["attempts"] == 5
    assert result["research_evaluation_cost"]["total_usd"] == .25
    gen["attempts"][1]["cost_usd"] = None
    data["evaluations"][0]["generation_hash"] = digest(gen)
    result = project_report(data, manifest)["models"][0]
    assert result["deployment_cost"]["total_usd"] is None
    assert result["deployment_cost"]["known_usd"] == 11


def test_free_call_zero_success_unknown_usage_are_distinct(fixture_records):
    manifest, data = fixture_records(models=("A",), tasks=("hard",))
    gen = data["generations"][0]
    gen["status"] = "agent_failure"; gen["attempts"][0].update(cost_usd=0, usage=None)
    data["evaluations"] = []
    row = project_report(data, manifest)["models"][0]
    assert row["deployment_cost"]["total_usd"] == 0
    assert row["deployment_cost"]["unknown_usage_attempts"] == 1
    assert row["cost_per_resolved"] is None


def test_repeat_samples_are_not_independent_stories_and_order_is_invariant(fixture_records):
    manifest, data = fixture_records(samples=(0, 1, 2))
    before = project_report(data, manifest)
    data["generations"].reverse(); data["evaluations"].reverse()
    after = project_report(data, manifest)
    assert before["rows"] == after["rows"] and before["models"] == after["models"]
    assert all(m["story_clusters"] == 1 and m["assigned"] == 6 for m in before["models"])


def test_failed_only_judge_attempt_never_becomes_valid(fixture_records):
    manifest, data = fixture_records()
    data["evaluations"][0]["attempts"][0]["status"] = "provider_failure"
    with pytest.raises(ValueError, match="successful research"):
        project_report(data, manifest)


def test_same_task_name_different_story_is_not_matched(fixture_records):
    manifest, data = fixture_records()
    manifest["assignments"][0]["story_id"] = "different-story"
    data["assignment_hash"] = digest(manifest)
    report = project_report(data, manifest)
    assert report["comparability"] == "incomparable_assignment_matrix"
    assert report["matched_valid_task_samples"] == []
    assert all(row["complete_assigned_diagnostic_mean"] is None for row in report["models"])
