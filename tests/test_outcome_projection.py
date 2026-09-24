"""Focused acceptance evidence for the pure Story-bench outcome projection.

Unit W2C-SCOPE-01. Each test name carries its acceptance-criterion tag so the
focused id list in the producer evidence can be audited one line at a time.
"""
import ast
import builtins
import dataclasses
import json
import pathlib
import random
import subprocess
import sys
import time

import pytest

import outcome_projection as op

IDENTITY = ("evaluationId", "profileRef", "criterionSpecDigest", "projectionVersion")
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
PROFILE_ID = "sb.story.v1"
PROFILE_REVISION = 3
#: The exact native ProfileRef wire (narrative-craft ProfileRef.to_dict()).
PROFILE_REF = {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": DIGEST_C}
#: Its deterministic canonical form: the frozen (profileId, revision, digest) tuple.
PROFILE_REF_CANON = (PROFILE_ID, PROFILE_REVISION, DIGEST_C)
ALLOWED_IMPORTS = {"__future__", "collections.abc", "dataclasses"}
FORBIDDEN_CALLS = {"open", "input", "eval", "exec", "compile", "__import__"}


def _record(status="resolved", execution="valid", grading="valid", reasons=(), **extra):
    record = {
        "evaluationId": "eval-1",
        "profileRef": dict(PROFILE_REF),
        "criterionSpecDigest": DIGEST_A,
        "projectionVersion": op.PROJECTION_VERSION,
        "status": status,
        "validity": {"execution": execution, "grading": grading},
        "reasons": list(reasons),
    }
    record.update(extra)
    return record


def _law(status, execution, grading):
    invalid = [token for token in (execution, grading) if token != "valid"]
    if invalid:
        return (False, None, "+".join(invalid))
    return {
        "observation": (False, None, "operational-mode"),
        "resolved": (True, 1, None),
        "unresolved": (True, 0, None),
        "unassessable": (False, None, "unassessable"),
    }[status]


_CELLS = []
for _status in sorted(op.STATUSES):
    _CELLS.append((_status, "valid", "valid"))
    for _execution in sorted(op.EXECUTION_VALIDITY - {"valid"}):
        _CELLS.append((_status, _execution, "valid"))
    _CELLS.append((_status, "valid", "grader-invalid"))
    _CELLS.append((_status, "provider-invalid", "grader-invalid"))
_CELL_IDS = [f"{status}-{execution}-{grading}" for status, execution, grading in _CELLS]


# --------------------------------------------------------------------------
# AC-1 purity
# --------------------------------------------------------------------------
def test_ac1_projection_is_pure_under_poisoned_open_clock_and_rng(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("pure projection must not perform I/O, clock or random access")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(time, "time", boom)
    monkeypatch.setattr(random, "random", boom)
    try:
        record = _record(status="unresolved", reasons=[{"code": "missing_submission"}])
        first = op.project_outcome(record)
        second = op.project_outcome(dict(record))
    finally:
        monkeypatch.undo()
    assert first == second
    assert op.projection_key(first) == op.projection_key(second)
    op.assert_projection_parity(first, second)


def test_ac1_module_source_has_no_io_or_nondeterministic_imports():
    source = pathlib.Path(op.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS, sorted(imported - ALLOWED_IMPORTS)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & FORBIDDEN_CALLS), sorted(called & FORBIDDEN_CALLS)


# --------------------------------------------------------------------------
# AC-2 one call returns report status/diagnostics AND reward eligibility
# --------------------------------------------------------------------------
def test_ac2_one_call_returns_report_status_and_reward_eligibility():
    projection = op.project_outcome(
        _record(status="resolved", reasons=[{"code": "ok", "detail": "proved"}, {"code": "note"}])
    )
    # report-facing half of the same object
    assert projection.status == "resolved"
    assert projection.executionValidity == "valid"
    assert projection.gradingValidity == "valid"
    assert projection.reasons == ((("code", "ok"), ("detail", "proved")), (("code", "note"),))
    # reward-facing half of the SAME object (no second call, no caller-set flag)
    assert projection.rewardEligible is True
    assert projection.trainingLabel == 1
    assert projection.exclusionReason is None
    assert projection.inDenominator is True


def test_ac2_diagnostics_are_never_dropped():
    reasons = [{"code": "a"}, "plain text", ["nested", 3], {"nested": {"x": [1, 2]}}]
    projection = op.project_outcome(_record(reasons=reasons))
    assert len(projection.reasons) == len(reasons)
    assert projection.reasons[0] == (("code", "a"),)
    assert projection.reasons[1] == "plain text"
    assert projection.reasons[2] == ("nested", 3)
    assert projection.reasons[3] == (("nested", (("x", (1, 2)),)),)


@pytest.mark.parametrize("reasons", ["a bare string", None, 7, {"code": "not-a-sequence"}])
def test_ac2_reasons_must_be_a_sequence(reasons):
    record = _record()
    record["reasons"] = reasons
    with pytest.raises(op.OutcomeRecordError):
        op.project_outcome(record)


def test_ac2_missing_reasons_member_fails_closed():
    record = _record()
    del record["reasons"]
    with pytest.raises(op.OutcomeRecordError):
        op.project_outcome(record)


# --------------------------------------------------------------------------
# AC-3 identity is exactly the four names; missing/extra fails closed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("member", IDENTITY)
def test_ac3_missing_identity_member_fails_closed(member):
    record = _record()
    record.pop(member)
    with pytest.raises(op.OutcomeRecordError, match=rf"missing identity member\(s\): \['{member}'\]"):
        op.project_outcome(record)


@pytest.mark.parametrize(
    "extra",
    [
        {"resolved": True},
        {"reward": 1},
        {"trainingLabel": 0},
        {"inDenominator": False},
        {"langfuseScore": 0.5},
    ],
)
def test_ac3_extra_member_fails_closed(extra):
    key = next(iter(extra))
    with pytest.raises(op.OutcomeRecordError, match=rf"unknown record member\(s\).*{key}"):
        op.project_outcome(_record(**extra))


def test_ac3_identity_names_are_exactly_the_four():
    assert op.IDENTITY_FIELDS == IDENTITY
    projection = op.project_outcome(_record())
    assert [field.name for field in dataclasses.fields(projection)][:4] == list(IDENTITY)
    assert op.projection_key(projection) == {
        "evaluationId": "eval-1",
        "profileRef": PROFILE_REF_CANON,
        "criterionSpecDigest": DIGEST_A,
        "projectionVersion": op.PROJECTION_VERSION,
        "status": "resolved",
    }


def test_ac3_blank_identity_member_fails_closed():
    with pytest.raises(op.OutcomeRecordError, match=r"missing or invalid evaluationId"):
        op.project_outcome(_record(evaluationId="   "))


def test_ac3_foreign_projection_version_fails_closed():
    with pytest.raises(op.OutcomeRecordError, match=r"foreign projectionVersion"):
        op.project_outcome(_record(projectionVersion="sb-outcome-projection/2"))


@pytest.mark.parametrize("value", [None, ["not", "a", "mapping"], "record", 3, object()])
def test_ac3_non_mapping_record_fails_closed(value):
    with pytest.raises(op.OutcomeRecordError, match=r"record must be a Mapping"):
        op.project_outcome(value)


# --------------------------------------------------------------------------
# AC-R1..AC-R4 native ProfileRef closed mapping (repair 1 / F-01)
# --------------------------------------------------------------------------
def test_acr1_native_profile_ref_closed_mapping_is_accepted():
    projection = op.project_outcome(_record())
    assert projection.profileRef == PROFILE_REF_CANON
    assert op.projection_key(projection)["profileRef"] == PROFILE_REF_CANON


_PROFILE_REF_REJECTIONS = [
    ("bare-profileId", "sb.story.v1", r"profileRef must be the native closed mapping"),
    ("bare-content-hash", DIGEST_C, r"profileRef must be the native closed mapping"),
    ("sequence", ["sb.story.v1", 3, DIGEST_C], r"profileRef must be the native closed mapping"),
    ("none", None, r"profileRef must be the native closed mapping"),
    ("missing-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION}, r"profileRef must carry exactly"),
    ("missing-revision", {"profileId": PROFILE_ID, "digest": DIGEST_C}, r"profileRef must carry exactly"),
    ("missing-profileId", {"revision": PROFILE_REVISION, "digest": DIGEST_C}, r"profileRef must carry exactly"),
    (
        "extra-key",
        {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": DIGEST_C, "extra": 1},
        r"profileRef must carry exactly",
    ),
    ("blank-profileId", {"profileId": "   ", "revision": PROFILE_REVISION, "digest": DIGEST_C}, r"invalid profileRef\.profileId"),
    ("empty-profileId", {"profileId": "", "revision": PROFILE_REVISION, "digest": DIGEST_C}, r"invalid profileRef\.profileId"),
    ("non-string-profileId", {"profileId": 7, "revision": PROFILE_REVISION, "digest": DIGEST_C}, r"invalid profileRef\.profileId"),
    ("zero-revision", {"profileId": PROFILE_ID, "revision": 0, "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("negative-revision", {"profileId": PROFILE_ID, "revision": -1, "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("bool-true-revision", {"profileId": PROFILE_ID, "revision": True, "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("bool-false-revision", {"profileId": PROFILE_ID, "revision": False, "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("float-revision", {"profileId": PROFILE_ID, "revision": 3.0, "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("string-revision", {"profileId": PROFILE_ID, "revision": "3", "digest": DIGEST_C}, r"invalid profileRef\.revision"),
    ("uppercase-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": DIGEST_C.upper()}, r"invalid profileRef\.digest"),
    ("short-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": "sha256:" + "a" * 63}, r"invalid profileRef\.digest"),
    ("no-prefix-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": "a" * 64}, r"invalid profileRef\.digest"),
    ("non-hex-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": "sha256:" + "g" * 64}, r"invalid profileRef\.digest"),
    ("non-string-digest", {"profileId": PROFILE_ID, "revision": PROFILE_REVISION, "digest": None}, r"invalid profileRef\.digest"),
]


@pytest.mark.parametrize(
    "profile_ref,match",
    [(profile_ref, match) for _, profile_ref, match in _PROFILE_REF_REJECTIONS],
    ids=[label for label, _, _ in _PROFILE_REF_REJECTIONS],
)
def test_acr1_non_native_profile_ref_fails_closed(profile_ref, match):
    with pytest.raises(op.OutcomeRecordError, match=match):
        op.project_outcome(_record(profileRef=profile_ref))


def test_acr2_distinct_native_profiles_do_not_collide():
    base = op.project_outcome(_record())
    other_revision = op.project_outcome(
        _record(profileRef={**PROFILE_REF, "revision": PROFILE_REVISION + 1})
    )
    other_digest = op.project_outcome(_record(profileRef={**PROFILE_REF, "digest": DIGEST_B}))
    # Same profileId in all three: a flattened profileId string would have collided.
    assert base.profileRef[0] == other_revision.profileRef[0] == other_digest.profileRef[0] == PROFILE_ID
    assert base.profileRef == PROFILE_REF_CANON
    assert other_revision.profileRef == (PROFILE_ID, PROFILE_REVISION + 1, DIGEST_C)
    assert other_digest.profileRef == (PROFILE_ID, PROFILE_REVISION, DIGEST_B)
    assert op.projection_key(base) != op.projection_key(other_revision)
    assert op.projection_key(base) != op.projection_key(other_digest)
    with pytest.raises(op.ProjectionParityError):
        op.assert_projection_parity(base, other_revision)
    with pytest.raises(op.ProjectionParityError):
        op.assert_projection_parity(base, other_digest)


def test_acr3_native_profile_ref_key_order_is_irrelevant():
    reordered = {
        "digest": PROFILE_REF["digest"],
        "revision": PROFILE_REF["revision"],
        "profileId": PROFILE_REF["profileId"],
    }
    assert list(reordered) == ["digest", "revision", "profileId"]
    first = op.project_outcome(_record(profileRef=dict(PROFILE_REF)))
    second = op.project_outcome(_record(profileRef=reordered))
    assert first == second
    assert op.projection_key(first) == op.projection_key(second)
    op.assert_projection_parity(first, second)


def test_acr4_realistic_native_record_projects_end_to_end():
    # The frozen native wire: evaluationId and criterionSpecDigest are plain
    # 'sha256:<64hex>' strings (ContentHash.to_dict); profileRef is the native closed
    # mapping (ProfileRef.to_dict).
    evaluation_id = "sha256:" + "e" * 64
    profile_digest = "sha256:" + "a" * 64
    record = {
        "evaluationId": evaluation_id,
        "profileRef": {"profileId": "sb.story.v1", "revision": 3, "digest": profile_digest},
        "criterionSpecDigest": "sha256:" + "c" * 64,
        "projectionVersion": op.PROJECTION_VERSION,
        "status": "resolved",
        "validity": {"execution": "valid", "grading": "valid"},
        "reasons": [{"code": "resolution", "detail": "decided natively elsewhere"}],
    }
    projection = op.project_outcome(record)
    assert projection.evaluationId == evaluation_id
    assert projection.profileRef == ("sb.story.v1", 3, profile_digest)
    assert projection.criterionSpecDigest == "sha256:" + "c" * 64
    assert projection.status == "resolved"
    assert projection.rewardEligible is True
    assert projection.trainingLabel == 1
    assert projection.exclusionReason is None
    assert projection.inDenominator is True


# --------------------------------------------------------------------------
# AC-4 all statuses x validity cells exact; unknown tokens fail closed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("status,execution,grading", _CELLS, ids=_CELL_IDS)
def test_ac4_status_validity_cells_exact(status, execution, grading):
    projection = op.project_outcome(_record(status=status, execution=execution, grading=grading))
    eligible, label, reason = _law(status, execution, grading)
    assert projection.status == status
    assert projection.executionValidity == execution
    assert projection.gradingValidity == grading
    assert projection.rewardEligible == eligible
    assert projection.trainingLabel == label
    assert projection.exclusionReason == reason


@pytest.mark.parametrize("token", ["RESOLVED", "resolved ", " excluded", "", "valid"])
def test_ac4_unknown_status_token_fails_closed(token):
    with pytest.raises(op.UnknownOutcomeTokenError, match=r"unknown outcome status token"):
        op.project_outcome(_record(status=token))


@pytest.mark.parametrize(
    "token", ["VALID", "invalid", "provider_invalid", "not_run", "infrastructure-invalid "]
)
def test_ac4_unknown_execution_validity_token_fails_closed(token):
    with pytest.raises(op.UnknownOutcomeTokenError, match=r"unknown execution validity token"):
        op.project_outcome(_record(execution=token))


@pytest.mark.parametrize("token", ["VALID", "invalid", "grader_invalid", "grader-invalid ", ""])
def test_ac4_unknown_grading_validity_token_fails_closed(token):
    with pytest.raises(op.UnknownOutcomeTokenError, match=r"unknown grading validity token"):
        op.project_outcome(_record(grading=token))


@pytest.mark.parametrize(
    "validity",
    [
        {},
        {"execution": "valid"},
        {"grading": "valid"},
        {"execution": "valid", "grading": "valid", "extra": "x"},
        "valid",
        None,
    ],
)
def test_ac4_malformed_validity_fails_closed(validity):
    record = _record()
    record["validity"] = validity
    with pytest.raises(op.OutcomeRecordError):
        op.project_outcome(record)


# --------------------------------------------------------------------------
# AC-5 every assigned case stays in the denominator and visible
# --------------------------------------------------------------------------
@pytest.mark.parametrize("status,execution,grading", _CELLS, ids=_CELL_IDS)
def test_ac5_every_assigned_case_stays_in_denominator(status, execution, grading):
    reasons = [{"code": "diagnostic", "status": status, "execution": execution}]
    projection = op.project_outcome(
        _record(status=status, execution=execution, grading=grading, reasons=reasons)
    )
    assert projection.inDenominator is True
    assert projection.status == status
    assert len(projection.reasons) == 1
    if projection.trainingLabel is None:
        assert projection.exclusionReason is not None
        assert projection.trainingLabel not in (0, 1)
    else:
        assert projection.trainingLabel in (0, 1)


# --------------------------------------------------------------------------
# AC-6 quality-pair outcomes never change the projection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["win", "tie", "loss", "cannot-assess", None])
def test_ac6_quality_pair_does_not_alter_projection(pair):
    reasons = [{"code": "missing_submission"}]
    base = op.project_outcome(_record(status="unresolved", reasons=reasons))
    camel = op.project_outcome(
        _record(status="unresolved", reasons=reasons, qualityPair={"outcome": pair})
    )
    snake = op.project_outcome(
        _record(status="unresolved", reasons=reasons, quality_pair={"outcome": pair, "pairId": "qp-1"})
    )
    assert base == camel == snake
    assert op.projection_key(base) == op.projection_key(camel) == op.projection_key(snake)


@pytest.mark.parametrize("status", ["resolved", "unresolved", "unassessable", "observation"])
@pytest.mark.parametrize("pair", ["tie", "cannot-assess"])
def test_ac6_quality_pair_never_changes_any_mapping_cell(status, pair):
    without = op.project_outcome(_record(status=status))
    with_pair = op.project_outcome(_record(status=status, qualityPair={"outcome": pair}))
    assert without == with_pair
    assert op.projection_key(without) == op.projection_key(with_pair)


# --------------------------------------------------------------------------
# AC-7 parity: identical keys for the same record; a forked cell is caught
# --------------------------------------------------------------------------
def test_ac7_same_record_produces_identical_projection_key():
    record = _record(status="unresolved", reasons=[{"code": "missing_submission"}, {"code": "coverage"}])
    first = op.project_outcome(record)
    second = op.project_outcome(dict(record))
    third = op.project_outcome({**record, "validity": {"execution": "valid", "grading": "valid"}})
    assert first == second == third
    assert op.projection_key(first) == op.projection_key(second) == op.projection_key(third)
    op.assert_projection_parity(first, second)


def test_ac7_forked_mapping_cell_is_caught_by_parity():
    original = op.project_outcome(_record(status="resolved"))
    forked = dataclasses.replace(original, trainingLabel=0)
    assert op.projection_key(original) == op.projection_key(forked)
    with pytest.raises(op.ProjectionParityError):
        op.assert_projection_parity(original, forked)


def test_ac7_forked_identity_is_caught_by_parity():
    original = op.project_outcome(_record(status="resolved"))
    forked = dataclasses.replace(original, criterionSpecDigest=DIGEST_B)
    with pytest.raises(op.ProjectionParityError):
        op.assert_projection_parity(original, forked)


def test_ac7_parity_rejects_non_projections():
    original = op.project_outcome(_record())
    with pytest.raises(op.OutcomeProjectionError):
        op.assert_projection_parity(original, op.projection_key(original))


# --------------------------------------------------------------------------
# AC-8 stdlib only, no repo import, importable standalone
# --------------------------------------------------------------------------
def _repo_module_names(src_dir, under_test):
    """Every repo module name a stray import could surface in sys.modules.

    Recursive over src/** so a repo SUBPACKAGE (src/pkg/__init__.py) and every module
    inside it are detected, not only top-level src/*.py. For each file both the dotted
    module name and its top-level package name are recorded, because importing a
    submodule also loads its parent package into sys.modules.
    """
    names = set()
    for path in src_dir.rglob("*.py"):
        relative = path.relative_to(src_dir)
        if relative.name == "__init__.py":
            parts = relative.parts[:-1]
        else:
            parts = relative.parts[:-1] + (relative.stem,)
        if not parts or parts[0] in {"__pycache__", under_test}:
            continue
        names.add(".".join(parts))
        names.add(parts[0])
    return sorted(names)


def test_ac8_module_imports_standalone_without_repo_modules():
    src_dir = pathlib.Path(__file__).resolve().parents[1] / "src"
    under_test = pathlib.Path(op.__file__).stem
    repo_modules = _repo_module_names(src_dir, under_test)
    assert "arm_b_package" in repo_modules, repo_modules  # subpackage must be covered
    probe = (
        "import json, sys;"
        f"sys.path.insert(0, {str(src_dir)!r});"
        "import outcome_projection as module;"
        "print(json.dumps({'version': module.PROJECTION_VERSION, 'loaded': sorted(sys.modules)}))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["version"] == op.PROJECTION_VERSION
    overlap = sorted(set(payload["loaded"]) & set(repo_modules))
    assert not overlap, overlap


# --------------------------------------------------------------------------
# AC-9 named wrong builds are killed by the shipped mapping oracle / parity
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status,forged_label",
    [("unassessable", 0), ("observation", 1)],
    ids=["unassessable-as-zero", "observation-as-one"],
)
def test_ac9_named_wrong_builds_are_killed(status, forged_label):
    correct = op.project_outcome(_record(status=status))
    assert correct.trainingLabel is None
    assert correct.rewardEligible is False
    wrong = dataclasses.replace(correct, trainingLabel=forged_label, rewardEligible=True)
    with pytest.raises(op.ProjectionParityError):
        op.assert_projection_parity(correct, wrong)
    assert _law(status, "valid", "valid")[1] is None
