import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

import pytest
from measurement_contract import digest, project_report
from results_db import ResultsDatabase


def test_enabled_complete_comparable_report_and_cli(fixture_records, tmp_path):
    manifest, data = fixture_records()
    mp, rp = tmp_path / "manifest.json", tmp_path / "records.json"
    mp.write_text(json.dumps(manifest)); rp.write_text(json.dumps(data))
    expected = project_report(data, manifest)
    run = subprocess.run([sys.executable, "run.py", "--offline-report", str(mp), "--records", str(rp)], capture_output=True, text=True, env={**os.environ, "OPENROUTER_API_KEY": ""})
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == expected
    assert len(expected["assigned_ids"]) == 4
    assert len({tuple(x) for x in expected["assigned_ids"]}) == 4
    assert expected["comparability"] == "matched_assigned_matrix"
    for model in expected["models"]:
        assert model["assigned"] == model["valid_diagnostics"] == 2
        assert model["story_clusters"] == 1
        assert model["complete_assigned_diagnostic_mean"] == pytest.approx(.5)
        assert model["deployment_cost"]["total_usd"] == 4
        assert model["research_evaluation_cost"]["total_usd"] == .5
        assert model["resolved"] is model["cost_per_resolved"] is None
    assert set(map(tuple, expected["matched_valid_task_samples"])) == {("easy", 0), ("hard", 0)}


def test_report_negative_controls_change_values_or_reject(fixture_records):
    manifest, data = fixture_records()
    data["evaluations"][0]["llm_results"]["beats_score"] = 0
    report = project_report(data, manifest)
    assert report["models"][0]["complete_assigned_diagnostic_mean"] == pytest.approx(.36875)
    del data["evaluations"][0]["llm_results"]["tone_score"]
    with pytest.raises(ValueError, match="Missing"):
        project_report(data, manifest)


def _run_offline_cli_preserving_inputs(tmp_path, manifest, data):
    mp, rp = tmp_path / "manifest.json", tmp_path / "records.json"
    mp.write_text(json.dumps(manifest)); rp.write_text(json.dumps(data))
    before = {path: path.read_bytes() for path in (mp, rp)}
    result = subprocess.run(
        [sys.executable, "run.py", "--offline-report", str(mp), "--records", str(rp)],
        capture_output=True, text=True,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
    )
    assert {path: path.read_bytes() for path in before} == before
    return result


@pytest.mark.parametrize("attempt_status", ["completed", "provider_failure"])
@pytest.mark.parametrize("cost_usd", [2, 0, None], ids=["billed", "free", "unknown"])
def test_not_run_evaluation_rejects_any_attempt_cli(fixture_records, tmp_path, attempt_status, cost_usd):
    manifest, data = fixture_records(models=("A",), tasks=("easy",))
    evaluation = data["evaluations"][0]
    evaluation["status"] = "not_run"
    evaluation["attempts"][0].update(status=attempt_status, cost_usd=cost_usd)
    result = _run_offline_cli_preserving_inputs(tmp_path, manifest, data)
    assert result.returncode == 2, result.stdout
    assert result.stdout == ""
    assert "Not-run evaluation has attempts" in result.stderr


def test_not_run_evaluation_empty_ledger_cli(fixture_records, tmp_path):
    manifest, data = fixture_records(models=("A",), tasks=("easy",))
    data["evaluations"][0].update(status="not_run", attempts=[])
    result = _run_offline_cli_preserving_inputs(tmp_path, manifest, data)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["assigned_ids"] == [["A", "easy", 0, "plain"]]
    assert report["rows"][0]["status"] == "not_run"
    assert report["rows"][0]["diagnostic"] is None
    assert report["matched_valid_task_samples"] == []
    model = report["models"][0]
    assert model["assigned"] == 1 and model["statuses"] == {"not_run": 1}
    assert model["valid_diagnostics"] == 0 and model["diagnostic_coverage"] == 0
    assert model["complete_assigned_diagnostic_mean"] is None
    assert model["deployment_cost"]["total_usd"] == 2
    assert model["research_evaluation_cost"] == {
        "known_usd": 0, "unknown_attempts": 0, "total_usd": 0,
        "attempts": 0, "roles": {}, "unknown_usage_attempts": 0,
    }


@pytest.mark.parametrize("status", [
    "agent_failure", "provider_failure", "infrastructure_failure",
    "grader_failure", "coverage_failure", "truncated",
])
def test_failed_evaluation_preserves_billed_attempt_cli(fixture_records, tmp_path, status):
    manifest, data = fixture_records(models=("A",), tasks=("easy",))
    evaluation = data["evaluations"][0]
    evaluation["status"] = status
    evaluation["attempts"][0].update(status=status, cost_usd=7)
    result = _run_offline_cli_preserving_inputs(tmp_path, manifest, data)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["rows"][0]["status"] == status
    model = report["models"][0]
    assert model["assigned"] == 1 and model["statuses"] == {status: 1}
    assert model["valid_diagnostics"] == 0
    assert model["complete_assigned_diagnostic_mean"] is None
    assert model["deployment_cost"]["total_usd"] == 2
    assert model["research_evaluation_cost"]["total_usd"] == 7
    assert model["research_evaluation_cost"]["attempts"] == 1


def test_no_legacy_cli_mutation_or_rank(tmp_path):
    for option in ["--leaderboard", "--rebuild-db", "--clean-failed", "--task-analysis", "--gen-model"]:
        args = [sys.executable, "run.py", option] + (["fake"] if option == "--gen-model" else [])
        result = subprocess.run(args, capture_output=True, text=True)
        assert result.returncode != 0


def test_native_reference_does_not_invent_resolved_success(fixture_records):
    manifest, data = fixture_records()
    for row in data["evaluations"]:
        row.update(resolved=True, native_evaluation_id="unverified", native_evaluation_hash="fake")
    report = project_report(data, manifest)
    assert all(row["resolved"] is None for row in report["rows"])


def test_historical_disposition_exact_row_ids_hashes_dates_and_conflicts():
    root = Path(__file__).resolve().parents[1]
    disposition = json.loads((root / "results/analysis/measurement-repair-v1/disposition.json").read_text())
    original = json.loads((root / "results/benchmark_results.json").read_text())
    assert len(disposition["raw_files"]) == 3468
    for raw in disposition["raw_files"]:
        assert hashlib.sha256((root / raw["path"]).read_bytes()).hexdigest() == raw["sha256"]
    expected = {(kind, index) for kind in ["generations", "evaluations"] for index in range(len(original[kind]))}
    actual = [(row["kind"], row["row_index"]) for row in disposition["rows"]]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    for row in disposition["rows"]:
        source = original[row["kind"]][row["row_index"]]
        assert row["raw_row_sha256"] == digest(source)
        assert row["original_date"] == source.get("timestamp")
        assert row["original_success_field"] == source.get("success")
        assert row["resolved"] is None
    conflicts = disposition["duplicate_groups"]["evaluations"]
    assert len(conflicts) == 3
    assert sum(len(group["row_indexes"]) for group in conflicts) == 6
    for group in conflicts:
        assert len(set(group["row_hashes"])) == 2
        assert all(next(r for r in disposition["rows"] if r["kind"] == "evaluations" and r["row_index"] == index)["primary_disposition"] == "quarantine" for index in group["row_indexes"])
