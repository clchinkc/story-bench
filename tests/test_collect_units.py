"""`collect_units` must turn a ledger + evidence into parseable units without inventing anything.

The three controls the task names are here — a paired happy path, a unit that lost its treatment arm,
and a cost disagreement — plus one refusal control: a row with no cluster is refused instead of
receiving a made-up id. Each control asserts the thing that would be wrong if the rule were relaxed,
not merely that the program ran: the emitted JSONL is parsed back through `study_analysis.parse_units`
and its decimal cost text is compared with what the provider wrote.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from measurement_contract import digest
import spend_ledger as sl
import study_analysis as sa

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Loaded by path and registered before execution, the same way `tests/test_dispatch_runner.py`
# loads a script: inserting scripts/ on sys.path would shadow `src/story_rl/` for later modules.
_spec = importlib.util.spec_from_file_location("collect_units",
                                               ROOT / "scripts" / "collect_units.py")
cu = importlib.util.module_from_spec(_spec)
sys.modules["collect_units"] = cu
_spec.loader.exec_module(cu)

MODEL = "vendor/model-one"
SNAPSHOT = "vendor/model-one@frozen-2026-11-01"
TASK = "beat_interpolation_001"
STORY = "beat_interpolation:fantasy"
TASK_DOCUMENT = {"task_id": TASK, "task_type": "beat_interpolation", "subtype": "default"}
PROGRAM = "story-bench"
STORE = "revision-v1"


def _row(arm, sample=0, **overrides):
    """One W1 row in the shape `assignment_builder.rows_for` emits, plus the design coordinates.

    `locale`/`cluster` are what the pilot design knows and `rows_for` currently drops; the row is
    written here the way a complete W1 manifest would have to carry them.
    """
    row = {
        "model": MODEL,
        "model_snapshot": SNAPSHOT,
        "task_id": TASK,
        "task": TASK_DOCUMENT,
        "task_hash": digest(TASK_DOCUMENT),
        "sample": sample,
        "condition": arm,
        "story_id": STORY,
        "judge_snapshot": "judge@1",
        "prompt_version": "w3b2-single-call-v1",
        "locale": "en",
        "cluster": "en-c00",
        "unit_id": "en-c00-u0",
    }
    row.update(overrides)
    return row


def _assignment(tmp_path, rows):
    path = tmp_path / "w1.json"
    path.write_text(json.dumps({
        "protocol": "measurement-repair-v1",
        "design": "pilot_run",
        "judge_designated": True,
        "judge_snapshot": "judge@1",
        "prompt_version": "w3b2-single-call-v1",
        "assignments": rows,
    }), encoding="utf-8")
    return path


def _attempt_id(arm, sample=0):
    return f"vendor_model-one:pilot-{TASK}-s{sample}:{arm}:deadbeef"


def _spec(attempt_id, arm, sample, assignment_hash, overrides):
    """The `AttemptSpec` fields `manifest.json` retains, in the ledger's own vocabulary."""
    spec = {
        "attempt_id": attempt_id,
        "run_id": f"pilot-{TASK}-s{sample}",
        "episode_id": f"pilot-{TASK}-s{sample}",
        "assignment_hash": assignment_hash,
        "task_id": TASK,
        "task_sha256": digest(TASK_DOCUMENT),
        "story_id": STORY,
        "condition": arm,
        "sample": sample,
        "assigned_model_snapshot": SNAPSHOT,
        "harness_sha256": "0" * 64,
        "protocol_sha256": "0" * 64,
        "settings_sha256": "0" * 64,
        "effort": "max",
        "parent_attempt_id": None,
        "retry_index": 0,
        "max_attempts": 1,
        "envelope": "pilot",
        "cost_role": "deployment",
        "activity_role": "generation",
        "program_id": PROGRAM,
        "store_id": STORE,
        "native_package_sha256": "0" * 64,
    }
    spec.update(overrides)
    return spec


def _write_manifest(directory, spec):
    (directory / "manifest.json").write_text(json.dumps({
        "version": "w3b1-offline-v1",
        "spec": spec,
        "route": {"model_id": MODEL, "canonical_model_slug": SNAPSHOT},
        "snapshot": {"observed_at": "2026-11-01T00:00:00+00:00"},
        "method": "POST",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "body_sha256": "0" * 64,
        "deadline_seconds": 600,
    }), encoding="utf-8")


def _write_transport(directory):
    """Exactly the fields `execute_attempt` writes — and it writes no timestamp, which is the point.

    The absence is what the collector must report as "latency underived" instead of guessing.
    """
    (directory / "transport.json").write_text(json.dumps({
        "method": "POST",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "body_hex": "",
        "body_sha256": "0" * 64,
        "marker": {"state": "SENT"},
        "calls": 1,
    }), encoding="utf-8")


def _write_response(directory, arm, cost_literal):
    """`response.bin` with the charge as RAW JSON TEXT, so no binary float can ever touch it."""
    body = (
        '{"id": "gen-%s", "model": "%s", "provider": "Vendor", "service_tier": null, '
        '"choices": [{"finish_reason": "stop", '
        '"message": {"role": "assistant", "content": "ok"}}], '
        '"usage": {"cost": %s, "is_byok": false, "prompt_tokens": 100, '
        '"completion_tokens": 50, "total_tokens": 150}}' % (arm, MODEL, cost_literal)
    ).encode()
    (directory / "response.bin").write_bytes(body)
    (directory / "response.json").write_text(json.dumps({
        "status": 200,
        "sha256": hashlib.sha256(body).hexdigest(),
        "request_sha256": "0" * 64,
    }), encoding="utf-8")
    return body


def _evidence(root, attempt_id, arm, sample, cost_literal, assignment_hash, **spec_overrides):
    directory = root / "attempts" / attempt_id
    directory.mkdir(parents=True)
    _write_manifest(directory, _spec(attempt_id, arm, sample, assignment_hash, spec_overrides))
    _write_transport(directory)
    return _write_response(directory, arm, cost_literal)


def _write_ledger(path, entries):
    """A real ledger, settled exactly the way `reconcile_attempt` settles one."""
    authorization = sl.Authorization(
        "offline-test", "a" * 64, "Disposable offline fixture only")
    ledger = sl.Ledger.create(path, program_id=PROGRAM, store_id=STORE,
                              authorization=authorization)
    # A fresh ledger holds the unresolved `hosted_session` charge, and an unresolved external
    # charge blocks every reservation. Settle it at a synthetic zero, exactly as the ledger's own
    # suite does, so the fixtures can reserve.
    ledger.reconcile_external(
        "offline-hosted-zero", charge_id="hosted_session", receipt_id="offline-zero-receipt",
        actual_usd="0", evidence_sha256="e" * 64, authorization=authorization)
    now = datetime.now(timezone.utc)
    for entry in entries:
        quote = sl.Quote(
            provider="openrouter", model=SNAPSHOT, snapshot_sha256="b" * 64,
            observed_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
            request_sha256="c" * 64, currency="USD",
            lines=tuple(sl.UsageBound(kind, 1, "0.1") for kind in sl.USAGE_KINDS))
        attempt_id = entry["attempt_id"]
        ledger.reserve(f"w3b:{attempt_id}:reserve", attempt_id=attempt_id, envelope="pilot",
                       role="deployment", quote=quote)
        ledger.mark_sent(f"w3b:{attempt_id}:send", attempt_id=attempt_id)
        # Every usage class must be a real integer: the ledger turns `None` into an
        # `unreconciled_usage` incident, and an incident permanently blocks dispatch.
        usage = tuple((kind, 1 if kind in ("input_tokens", "output_tokens") else 0)
                      for kind in sl.USAGE_KINDS)
        ledger.reconcile(
            f"w3b:{attempt_id}:reconcile", attempt_id=attempt_id,
            receipt=sl.Receipt(
                receipt_id=f"openrouter:{attempt_id}", provider="openrouter", model=SNAPSHOT,
                snapshot_sha256="b" * 64, request_sha256="c" * 64, currency="USD",
                actual_usd=entry["actual_usd"],
                usage=usage,
                outcome=entry.get("outcome", "completed"),
                evidence_sha256=entry["evidence_sha256"],
                recorded_at=datetime.now(timezone.utc).isoformat()))
    return ledger


def _run(tmp_path, rows, arms, costs, actual_usd):
    """Build evidence + ledger for the named arms and run the collector. Returns (code, out)."""
    assignment = _assignment(tmp_path, rows)
    assignment_hash = digest(json.loads(assignment.read_text(encoding="utf-8")))
    evidence = tmp_path / "evidence"
    entries = []
    for arm in arms:
        attempt_id = _attempt_id(arm)
        body = _evidence(evidence, attempt_id, arm, 0, costs[arm], assignment_hash)
        entries.append({"attempt_id": attempt_id, "actual_usd": actual_usd[arm],
                        "evidence_sha256": hashlib.sha256(body).hexdigest()})
    _write_ledger(tmp_path / "pilot.sqlite3", entries)
    out = tmp_path / "units.jsonl"
    code = cu.main(["--evidence", str(evidence), "--ledger", str(tmp_path / "pilot.sqlite3"),
                    "--assignment", str(assignment), "--out", str(out)])
    return code, out


def test_two_reconciled_attempts_produce_one_unit_with_both_arms(tmp_path, capsys):
    code, out = _run(
        tmp_path,
        rows=[_row("A"), _row("D")],
        arms=("A", "D"),
        costs={"A": "0.012345", "D": "0.012345"},
        actual_usd={"A": "0.012345", "D": "0.012345"},
    )
    assert code == 0, capsys.readouterr().err

    units = sa.load_records(out)
    assert len(units) == 1
    unit = units[0]
    assert (unit.cluster_id, unit.locale, unit.task_family, unit.sample_id) == \
        ("en-c00", "en", "beat_interpolation", "0")
    assert sorted(unit.arms) == ["A", "D"]
    for arm in ("A", "D"):
        observation = unit.arms[arm]
        # Unscored, not scored-zero: the judge has not run, so no claim is made either way.
        assert observation.resolved is None
        assert observation.invalid_kind is None
        assert observation.quality is None
        # No workspace observation exists, so a breach cannot be claimed from a response alone.
        assert observation.authority_breach is False
        assert observation.cost_usd == Decimal("0.012345")
        assert observation.latency_ms == 0

    raw = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "quality" not in raw["arms"]["A"]
    assert raw["arms"]["A"]["cost_usd"] == "0.012345"
    assert raw["arms"]["D"]["cost_usd"] == "0.012345"

    printed = capsys.readouterr().out
    assert "emitted units    : 1" in printed
    assert "latency underived: 2 of 2" in printed
    assert f"{MODEL}: reconciled 2" in printed


def test_a_unit_missing_its_treatment_arm_is_reported_and_not_emitted(tmp_path, capsys):
    code, out = _run(
        tmp_path,
        rows=[_row("A"), _row("D")],
        arms=("A",),
        costs={"A": "0.012345"},
        actual_usd={"A": "0.012345"},
    )
    assert code == 1
    assert not out.exists(), "an incomplete unit must not reach the analysis file"
    err = capsys.readouterr().err
    assert "incomplete unit" in err
    assert "missing arm(s) ['D']" in err
    # The refusal names the consequence instead of writing an empty file the analyser would reject.
    assert "no complete units" in err


def test_a_cost_disagreement_is_reported_and_the_ledger_value_is_used(tmp_path, capsys):
    code, out = _run(
        tmp_path,
        rows=[_row("A"), _row("D")],
        arms=("A", "D"),
        # The provider's own text says one cent; the ledger settled twelve thousand three hundred
        # and forty-five micros. The ledger is canonical, so its value is what the unit carries.
        costs={"A": "0.01", "D": "0.01"},
        actual_usd={"A": "0.012345", "D": "0.012345"},
    )
    assert code == 0, capsys.readouterr().err
    err = capsys.readouterr().err
    assert err.count("cost disagreement") == 2
    assert "provider usage.cost=0.01" in err
    assert "ledger actual_micros=12345" in err
    assert "using the ledger value 0.012345" in err

    raw = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert raw["arms"]["A"]["cost_usd"] == "0.012345"
    assert raw["arms"]["D"]["cost_usd"] == "0.012345"


def test_a_row_without_a_cluster_is_refused_rather_than_invented(tmp_path, capsys):
    row = _row("A")
    for key in ("locale", "cluster", "unit_id"):
        row.pop(key)
    assignment = _assignment(tmp_path, [row])
    code = cu.main(["--evidence", str(tmp_path / "evidence"),
                    "--ledger", str(tmp_path / "absent.sqlite3"),
                    "--assignment", str(assignment),
                    "--out", str(tmp_path / "units.jsonl")])
    assert code == 1
    err = capsys.readouterr().err
    assert "carries no" in err
    assert "cluster_id" in err and "locale" in err
    # The refusal names the concrete thing that cannot be derived, not just "invalid manifest".
    assert "fabricate the strata" in err
