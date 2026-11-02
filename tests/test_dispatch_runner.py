"""The runner must refuse an undeclared design and plan a declared one exactly.

The dangerous failure here is not a crash — it is a manifest that looks authoritative over a design the
study never froze. So the controls are about what is *refused*, and about the plan being reproducible
enough that a regenerated manifest can be compared rather than trusted.
"""
import importlib.util
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Loaded by path, NOT by inserting scripts/ on sys.path: `scripts/story_rl.py` would then shadow the
# `src/story_rl/` package for every test collected after this module, which is exactly what happened on
# the first attempt — five unrelated modules failed to collect while this one passed.
_spec = importlib.util.spec_from_file_location("dispatch_runner", ROOT / "scripts" / "dispatch_runner.py")
dr = importlib.util.module_from_spec(_spec)
# Register before executing: `dataclasses` resolves a decorated class's module through `sys.modules`, so
# a module loaded by path but never registered fails its own @dataclass with a NoneType AttributeError.
sys.modules["dispatch_runner"] = dr
_spec.loader.exec_module(dr)


def _tasks(repo, count=3):
    directory = repo / "dataset" / "tasks" / "beat_interpolation"
    directory.mkdir(parents=True)
    for index in range(count):
        (directory / f"task_{index:03d}.yaml").write_text(
            f"task_id: beat_interpolation_{index:03d}\ntask_type: beat_interpolation\n")


def _records(base, routes=("vendor/one", "vendor/two")):
    (base / "records").mkdir(parents=True)
    for index, model in enumerate(routes):
        (base / "records" / f"r{index}.json").write_text(
            '{"route_identity": {"model_id": "%s"}}' % model)


def _repo(tmp_path, main_run):
    repo = tmp_path / "repo"
    (repo / "studies").mkdir(parents=True)
    study = {"status": "frozen", "task_mix": {"families": {"beat_interpolation": 3}}}
    if main_run is not None:
        study["main_run"] = main_run
    (repo / "studies" / "revision-v1.yaml").write_text(yaml.safe_dump(study))
    _tasks(repo)
    return repo


def _base(tmp_path):
    base = tmp_path / "base"
    _records(base)
    return base


DECLARED = dict(status="frozen", units_per_model=6, arms=2,
                 units_per_cluster=3, clusters_per_locale=2, locales=["en"])


def test_refuses_when_the_study_declares_no_main_run(tmp_path):
    repo = _repo(tmp_path, None)
    spec, refusals = dr.declared_spec(dr._study(repo))
    assert spec is None
    assert any("main_run declares no status" in reason for reason in refusals)


def test_refuses_a_design_that_is_only_proposed(tmp_path):
    repo = _repo(tmp_path, {**DECLARED, "status": "proposed"})
    spec, refusals = dr.declared_spec(dr._study(repo))
    assert spec is not None
    assert any("not 'frozen'" in reason for reason in refusals)


def test_refuses_when_the_study_itself_is_unfrozen(tmp_path):
    repo = _repo(tmp_path, DECLARED)
    study = dr._study(repo)
    study["status"] = "unfrozen"
    _, refusals = dr.declared_spec(study)
    assert any("the study itself is 'unfrozen'" in reason for reason in refusals)


def test_refuses_when_the_composition_contradicts_units_per_model(tmp_path):
    repo = _repo(tmp_path, {**DECLARED, "units_per_model": 99})
    _, refusals = dr.declared_spec(dr._study(repo))
    assert any("contradict" in reason for reason in refusals)


def test_plans_every_arm_run(tmp_path):
    repo, base = _repo(tmp_path, DECLARED), _base(tmp_path)
    spec, refusals = dr.declared_spec(dr._study(repo))
    assert refusals == []
    units = dr.assign(spec, dr.task_set(repo))
    attempts = dr.plan(spec, units, dr.route_names(base))
    assert len(units) == spec.units == 6
    assert len(attempts) == 6 * 2 * 2  # units x arms x routes


def test_the_plan_is_deterministic(tmp_path):
    repo, base = _repo(tmp_path, DECLARED), _base(tmp_path)
    spec, _ = dr.declared_spec(dr._study(repo))
    first = dr.plan(spec, dr.assign(spec, dr.task_set(repo)), dr.route_names(base))
    second = dr.plan(spec, dr.assign(spec, dr.task_set(repo)), dr.route_names(base))
    assert [row["attempt_id"] for row in first] == [row["attempt_id"] for row in second]


def test_every_attempt_id_is_unique(tmp_path):
    repo, base = _repo(tmp_path, DECLARED), _base(tmp_path)
    spec, _ = dr.declared_spec(dr._study(repo))
    attempts = dr.plan(spec, dr.assign(spec, dr.task_set(repo)), dr.route_names(base))
    ids = [row["attempt_id"] for row in attempts]
    assert len(ids) == len(set(ids))


def test_the_design_selector_reads_the_named_block(tmp_path):
    """The pilot and the main run share one planner; only the declared block differs."""
    repo = _repo(tmp_path, None)
    study = dr._study(repo)
    study["pilot_run"] = {**DECLARED, "units_per_model": 6}
    spec, refusals = dr.declared_spec(study, "pilot_run")
    assert refusals == []
    assert spec.units == 6
    assert "main_run" in dr.declared_spec(study, "main_run")[1][0]  # the other block still refuses


def test_dispatch_refuses_without_a_token():
    result = dr.dispatch([{"attempt_id": "a"}], authorise=None,
                         build=lambda attempt: attempt, execute=lambda plan: None)
    assert result["refused"] and result["executed"] == 0


def test_dispatch_refuses_a_token_of_the_wrong_shape():
    result = dr.dispatch([{"attempt_id": "a"}], authorise="please",
                         build=lambda attempt: attempt, execute=lambda plan: None)
    assert result["refused"] and result["executed"] == 0


def test_dispatch_executes_every_attempt_it_is_given():
    seen = []
    attempts = [{"attempt_id": f"a{i}"} for i in range(3)]
    result = dr.dispatch(attempts, authorise="PAID-PILOT-2026-11-02",
                         build=lambda attempt: attempt["attempt_id"], execute=seen.append)
    assert seen == ["a0", "a1", "a2"]
    assert result == dict(refused=None, executed=3, skipped=0)


def test_dispatch_skips_what_the_ledger_already_settled():
    """Resumption is the ledger's fact, not a re-run: a reconciled attempt must never be resent."""
    seen = []
    attempts = [{"attempt_id": f"a{i}"} for i in range(4)]
    result = dr.dispatch(attempts, authorise="PAID-PILOT-2026-11-02",
                         build=lambda attempt: attempt["attempt_id"], execute=seen.append,
                         completed={"a1", "a3"})
    assert seen == ["a0", "a2"]
    assert result["executed"] == 2 and result["skipped"] == 2


def test_dispatch_stops_at_the_first_failure():
    seen = []

    def execute(attempt_id):
        if attempt_id == "a1":
            raise RuntimeError("transport died")
        seen.append(attempt_id)

    attempts = [{"attempt_id": f"a{i}"} for i in range(3)]
    with pytest.raises(RuntimeError, match="transport died"):
        dr.dispatch(attempts, authorise="PAID-PILOT-2026-11-02",
                    build=lambda attempt: attempt["attempt_id"], execute=execute)
    assert seen == ["a0"]


def test_units_cover_every_locale_and_cluster_once(tmp_path):
    repo = _repo(tmp_path, {**DECLARED, "locales": ["en", "zh-Hant"], "units_per_model": 12})
    spec, _ = dr.declared_spec(dr._study(repo))
    units = dr.assign(spec, dr.task_set(repo))
    assert len(units) == 12
    assert {row["locale"] for row in units} == {"en", "zh-Hant"}
    assert len({row["unit_id"] for row in units}) == 12
