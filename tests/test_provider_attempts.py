"""Offline behavior through the actual adapter; synthetic authority only."""
import asyncio
import copy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

import httpx
import pytest

from measurement_contract import PROTOCOL, context_packet, digest
import provider_attempts as pa
from provider_prices import ArtifactRef, Route, json_bytes, sha, validate_snapshot
from provider_reporting import project_attempts, project_cost_report
from spend_ledger import Authorization, Ledger, LedgerError, USAGE_KINDS, ValidationError


def make_case(root, *, attempt_id="attempt-1", role="generation", hosted=True, cost_rate="0.001",
              snapshot=None, manifest=None, prompt=None, ledger=None, retry_index=0, parent=None,
              max_attempts=1, profile_changes=None):
    root = Path(root).resolve()
    route = Route("openrouter", "https://openrouter.ai/api/v1", "/chat/completions", "fixture/model",
                  "fixture/model-20260923", "fixture/local", "default", "USD")
    now = datetime.now(timezone.utc)
    start, end = (now - timedelta(seconds=1)).isoformat(), (now + timedelta(hours=1)).isoformat()
    catalog = json_bytes({"data": [{"id": route.model_id, "canonical_slug": route.canonical_model_slug}]})
    endpoints = json_bytes({"data": {"id": route.model_id, "endpoints": [dict(
        tag=route.endpoint_tag, provider_name="Fixture", model_id=route.model_id,
        context_length=10000, max_prompt_tokens=9000, max_completion_tokens=1000,
        supported_parameters=["max_tokens", "reasoning", "temperature", "seed"],
        pricing=dict(prompt=cost_rate, completion=cost_rate, input_cache_read=cost_rate,
                     input_cache_write=cost_rate, input_cache_write_1h=cost_rate, request="0"))]}})
    profile = dict(kind="SYNTHETIC OFFLINE ONLY", catalog_sha256=sha(catalog), endpoints_sha256=sha(endpoints),
                   model_id=route.model_id, canonical_model_slug=route.canonical_model_slug,
                   endpoint_tag=route.endpoint_tag, input_bound=1000, output_bound=100, reasoning_bound=100,
                   cache_read_bound=1000, cache_write_bound=1000, structural_zero=[],
                   no_incremental_infrastructure=True, no_metered_agent=True)
    profile.update(profile_changes or {})
    snapshot = snapshot or validate_snapshot((catalog, endpoints, json_bytes(profile)), route, start, end)
    task = dict(task_id="task", task_type="theory_conversion", subtype="default", source="EARLY CAUSE and END")
    assignment = dict(model="fixture-model", task_id="task", sample=0, condition="A", story_id="story",
                      task=task, task_hash=digest(task), model_snapshot=route.canonical_model_slug,
                      judge_snapshot=route.canonical_model_slug, prompt_version="fixture-v1")
    manifest = manifest or dict(protocol=PROTOCOL, assignments=[assignment])
    task = manifest["assignments"][0]["task"]
    prompt = prompt or context_packet({"task": json.dumps(task, sort_keys=True, ensure_ascii=False)}, max_bytes=10000)[0]
    fields = pa.RequestFields((("user", prompt),), 100, reasoning_effort="low")
    artifacts = dict(assignment=json_bytes(manifest), task=json_bytes(task), prompt=prompt.encode(),
                     harness=b"synthetic-harness-v1", protocol=b"synthetic-protocol-v1",
                     settings=json_bytes(asdict(fields)), native_package=b"SYNTHETIC ONLY native38428703 version0.8.0")
    route_hash = sha(json_bytes(asdict(route)))
    for gate in sorted(pa.GATES):
        artifacts["gate_" + gate] = json_bytes(dict(kind=gate, authority="SYNTHETIC OFFLINE ONLY",
            assignment_hash=digest(manifest), route_sha256=route_hash, valid_until=end))
    refs = tuple(ArtifactRef.of(k, v) for k, v in artifacts.items())
    spec = pa.AttemptSpec(attempt_id, "run", "episode", digest(manifest), "task", sha(artifacts["task"]),
        "story", "A", 0, route.canonical_model_slug, sha(artifacts["harness"]), sha(artifacts["protocol"]),
        sha(artifacts["settings"]), "low", parent, retry_index, max_attempts, "repair",
        "research" if role == "research_evaluation" else "deployment", role, "synthetic-program", "synthetic-store",
        sha(artifacts["native_package"]), refs)
    plan = pa.prepare_attempt(spec, snapshot, tuple(artifacts.items()), fields, root / "evidence")
    admission = pa.Admission("offline_fixture", tuple(r for r in refs if r.name.startswith("gate_")),
                             spec.assignment_hash, route_hash, end)
    if ledger is None:
        ledger = Ledger.create(root / "ledger.sqlite", program_id=spec.program_id, store_id=spec.store_id,
                               authorization=Authorization("synthetic-create", "a" * 64, "SYNTHETIC ONLY"))
        if hosted:
            ledger.reconcile_external("synthetic-host", charge_id="hosted_session", receipt_id="synthetic-host",
                actual_usd="0", evidence_sha256="b" * 64, authorization=Authorization("synthetic-host", "c" * 64, "SYNTHETIC ONLY"))
    return plan, ledger, admission, manifest


def response(plan, *, cost="0.03", finish="stop", content="Complete end.", generation_id=None):
    # Inject the literal JSON monetary number without a float round trip.
    data = dict(id=generation_id or "gen-" + plan.spec.attempt_id, model=plan.route.canonical_model_slug,
                provider="Fixture", service_tier="default",
                choices=[dict(finish_reason=finish, message=dict(content=content))],
                usage=dict(cost="MONEY", is_byok=False, prompt_tokens=20, completion_tokens=5,
                    total_tokens=25, prompt_tokens_details=dict(cached_tokens=0, cache_write_tokens=0),
                    completion_tokens_details=dict(reasoning_tokens=2)))
    return json_bytes(data).replace(b'"MONEY"', cost.encode())


def fixture(raw, status=200, header_delay=0, chunks=None):
    return pa.OfflineFixture(status, header_delay, chunks or ((0, raw),))


def execute(case, raw=None, **kwargs):
    p, ledger, admission, _ = case
    return asyncio.run(pa.execute_attempt(p, ledger, admission, fixture(raw or response(p), **kwargs)))


def trace(plan):
    return json.loads((plan.evidence_root / "attempts" / plan.spec.attempt_id / "transport.json").read_text())


def test_A01_success_actual_entry_and_exact_wire_receipt(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    p, ledger, _, _ = case
    calls = []
    original = httpx.MockTransport.handle_async_request
    async def observed(self, request):
        calls.append(request.content)
        assert ledger.attempt(p.spec.attempt_id)["state"] == "SENT"
        return await original(self, request)
    monkeypatch.setattr(httpx.MockTransport, "handle_async_request", observed)
    result = execute(case, response(p, cost="0.030000000000000001"))
    assert result.ledger_state == "RECONCILED" and result.transport_status == "completed"
    assert calls == [p.body_bytes]
    actual = ledger.attempt(p.spec.attempt_id)
    assert actual["receipt"]["actual_usd"] == "0.030000000000000001"
    assert actual["actual_micros"] == 30001
    assert trace(p)["body_hex"] == p.body_bytes.hex()
    body = json.loads(p.body_bytes)
    assert body["provider"] == dict(only=["fixture/local"], order=["fixture/local"], allow_fallbacks=False, require_parameters=True)
    assert body["stream"] is False and body["max_tokens"] == 100
    assert [e["kind"] for e in ledger.events()][-3:] == ["reserved", "sent", "reconciled"]
    assert set(dict(actual["receipt"]["usage"])) == set(USAGE_KINDS)
    assert ledger.status()["incident_ids"] == []


@pytest.mark.parametrize("finish,content,status", [("length", "", "truncated"), ("content_filter", "", "refused"), ("stop", "", "absent_output")])
def test_A02_finish_failure_billing(tmp_path, finish, content, status):
    case = make_case(tmp_path)
    result = execute(case, response(case[0], finish=finish, content=content))
    assert result.transport_status == status
    assert case[1].attempt("attempt-1")["actual_micros"] == 30000


def test_A02_http_failed_receipt(tmp_path):
    case = make_case(tmp_path)
    assert execute(case, status=503).transport_status == "provider_failure"
    assert case[1].status()["committed_micros"] == 30000


@pytest.mark.parametrize("role,kind", [("critic", "critic_calls"), ("retrieval", "retrieval_calls"), ("selection", "selection_calls"), ("research_evaluation", "grading_calls")])
def test_A03_usage_census_activity(tmp_path, role, kind):
    case = make_case(tmp_path, role=role)
    execute(case)
    counts = dict(case[1].attempt("attempt-1")["receipt"]["usage"])
    assert counts[kind] == 1 and counts["reasoning_tokens"] == 2
    assert counts["agent_units"] == counts["training_units"] == counts["infrastructure_units"] == 0


def test_A04_response_saved_reconciliation_restart_no_resend(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    real = pa.reconcile_attempt
    def crash(*args):
        raise SystemExit("crash after response persistence")
    monkeypatch.setattr(pa, "reconcile_attempt", crash)
    with pytest.raises(SystemExit):execute(case)
    assert case[1].attempt("attempt-1")["state"] == "SENT"
    monkeypatch.setattr(pa, "reconcile_attempt", real)
    assert execute(case).replayed is True
    raw = (case[0].evidence_root / "attempts/attempt-1/response.bin").read_bytes()
    result = real(case[0], case[1], raw, 200)
    again = real(case[0], case[1], raw, 200)
    assert result.ledger_state == "RECONCILED" and again.replayed is True
    assert trace(case[0])["calls"] == 1 and case[1].status()["committed_micros"] == 30000


@pytest.mark.parametrize("field,value", [("condition", "B"), ("assigned_model_snapshot", "other"), ("task_id", "wrong"), ("harness_sha256", "f"*64), ("settings_sha256", "f"*64), ("program_id", "wrong"), ("effort", "high")])
def test_D06_identity_drift_zero_send(tmp_path, field, value):
    case = make_case(tmp_path)
    p = replace(case[0], spec=replace(case[0].spec, **{field: value}))
    with pytest.raises(ValueError):asyncio.run(pa.execute_attempt(p, case[1], case[2], fixture(response(p))))
    assert not (case[0].evidence_root / "attempts/attempt-1/transport.json").exists()
    assert case[1].status()["attempt_ids"] == []


@pytest.mark.parametrize("change", ["paid", "missing_rights", "wrong_assignment", "expired", "missing_billing"])
def test_D04_admission_zero_send(tmp_path, change):
    p, ledger, admission, _ = make_case(tmp_path)
    if change == "paid":admission = replace(admission, mode="paid")
    elif change == "wrong_assignment":admission = replace(admission, assignment_hash="0"*64)
    elif change == "expired":admission = replace(admission, valid_until="2000-01-01T00:00:00+00:00")
    else:admission = replace(admission, authority_refs=tuple(r for r in admission.authority_refs if r.name != "gate_"+change.removeprefix("missing_")))
    with pytest.raises(ValueError):asyncio.run(pa.execute_attempt(p, ledger, admission, fixture(response(p))))
    assert ledger.status()["attempt_ids"] == []


def test_D05_hosted_unknown_and_exhaustion_zero_send(tmp_path):
    case = make_case(tmp_path, hosted=False)
    with pytest.raises(LedgerError):execute(case)
    assert case[1].status()["attempt_ids"] == []
    case2 = make_case(tmp_path / "other", cost_rate="1")
    with pytest.raises(LedgerError):execute(case2)
    assert case2[1].status()["attempt_ids"] == []


def test_D01_expired_snapshot_zero_send(tmp_path):
    case = make_case(tmp_path)
    p = replace(case[0], snapshot=replace(case[0].snapshot, observed_at="2000-01-01T00:00:00+00:00", expires_at="2000-01-02T00:00:00+00:00"))
    with pytest.raises(ValueError):asyncio.run(pa.execute_attempt(p, case[1], case[2], fixture(response(p))))
    assert case[1].status()["attempt_ids"] == []


def test_D02_D06_changed_body_artifact_symlink_zero_send(tmp_path):
    case = make_case(tmp_path)
    p = replace(case[0], body_bytes=b'{"model":"other"}')
    with pytest.raises(ValueError):asyncio.run(pa.execute_attempt(p, case[1], case[2], fixture(response(p))))
    prompt_path = case[0].evidence_root / "blobs" / case[0].prompt_ref.sha256
    prompt_path.write_bytes(b"changed")
    with pytest.raises(ValueError):execute(case)
    prompt_path.unlink(); prompt_path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):execute(case)
    assert case[1].status()["attempt_ids"] == []


def test_D07_replay_marker_does_not_send(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    real = case[1].mark_sent
    def replay(*args, **kwargs):
        row = real(*args, **kwargs)
        return dict(row, replayed=True)
    monkeypatch.setattr(case[1], "mark_sent", replay)
    result = execute(case)
    assert result.replayed is True
    assert not (case[0].evidence_root / "attempts/attempt-1/transport.json").exists()
    assert case[1].attempt("attempt-1")["state"] == "SENT"


@pytest.mark.parametrize("raw_kind", ["duplicate_key", "nan", "bool_cost", "negative_cost", "wrong_model", "wrong_tier", "byok", "invalid_token", "bad_finish", "conflicting_cost"])
def test_D08_R04_invalid_response_retains_liability(tmp_path, raw_kind):
    case = make_case(tmp_path)
    data = json.loads(response(case[0]))
    if raw_kind == "duplicate_key":raw = response(case[0]).replace(b'"id":', b'"id":"duplicate","id":')
    elif raw_kind == "nan":raw = response(case[0], cost="NaN")
    else:
        if raw_kind == "bool_cost":data["usage"]["cost"] = True
        if raw_kind == "negative_cost":data["usage"]["cost"] = -1
        if raw_kind == "wrong_model":data["model"] = "wrong"
        if raw_kind == "wrong_tier":data["service_tier"] = "priority"
        if raw_kind == "byok":data["usage"]["is_byok"] = True
        if raw_kind == "invalid_token":data["usage"]["prompt_tokens"] = True
        if raw_kind == "bad_finish":data["choices"][0]["finish_reason"] = "tool_calls"
        if raw_kind == "conflicting_cost":data["total_cost"] = 2
        raw = json_bytes(data)
    execute(case, raw)
    row = case[1].attempt("attempt-1")
    assert row["state"] == "UNKNOWN" and row["actual_micros"] is None
    assert case[1].status()["reserved_micros"] > 0


@pytest.mark.parametrize("missing", ["cost", "prompt_tokens", "completion_tokens_details", "prompt_tokens_details"])
def test_R02_unknown_never_zero(tmp_path, missing):
    case = make_case(tmp_path)
    data = json.loads(response(case[0])); del data["usage"][missing]
    execute(case, json_bytes(data))
    row = case[1].attempt("attempt-1")
    if missing == "cost":
        assert row["state"] == "UNKNOWN" and row["actual_micros"] is None and case[1].status()["reserved_micros"] > 0
    else:
        assert row["state"] == "RECONCILED" and row["actual_micros"] == 30000
        assert any(v is None for _, v in row["receipt"]["usage"])
        assert case[1].status()["incident_ids"]


def test_R01_cumulative_cost_not_sum_or_upstream(tmp_path):
    case = make_case(tmp_path)
    data = json.loads(response(case[0], cost="0.125")); data["total_cost"] = .125
    data["usage"]["cost_details"] = {"upstream_inference_cost": 9}
    execute(case, json_bytes(data))
    assert case[1].status()["committed_micros"] == 125000


def test_R03_overrun_retains_truth_blocks_next(tmp_path):
    case = make_case(tmp_path)
    execute(case, response(case[0], cost="9"))
    assert case[1].status()["committed_micros"] == 9000000
    assert case[1].status()["dispatch_blocked"]
    next_case = make_case(tmp_path, attempt_id="next", ledger=case[1])
    with pytest.raises(LedgerError):execute(next_case)


def test_R05_zero_receipt_not404_absence(tmp_path):
    case = make_case(tmp_path)
    execute(case, response(case[0], cost="0"))
    assert case[1].attempt("attempt-1")["state"] == "RECONCILED"
    assert case[1].status()["reserved_micros"] == 0
    other = make_case(tmp_path / "other")
    execute(other, b'{}', status=404)
    assert other[1].attempt("attempt-1")["state"] == "UNKNOWN"


@pytest.mark.parametrize("phase", ["header", "body"])
def test_C05_absolute_deadline_retains_liability(tmp_path, phase, monkeypatch):
    case = make_case(tmp_path)
    raw = response(case[0]); times = {}
    original_send, original_unknown = httpx.MockTransport.handle_async_request, pa._unknown
    async def observed(self, request):
        times["start"] = time.monotonic()
        return await original_send(self, request)
    def unknown(*args):
        times["end"] = time.monotonic()
        return original_unknown(*args)
    monkeypatch.setattr(httpx.MockTransport,"handle_async_request",observed)
    monkeypatch.setattr(pa,"_unknown",unknown)
    result = execute(case, raw, header_delay=1500 if phase == "header" else 0,
                     chunks=((600, raw[:2]), (600, raw[2:])) if phase == "body" else None)
    assert result.transport_status == "timeout"
    # Measure the cooperative transport interval, excluding SQLite and host fsync work.
    assert times["end"] - times["start"] < 1.5
    assert case[1].attempt("attempt-1")["state"] == "UNKNOWN"
    assert case[1].status()["reserved_micros"] > 0 and trace(case[0])["calls"] == 1


@pytest.mark.parametrize("phase", ["before_marker", "header", "body"])
def test_C05_cancellation_propagates_without_refund(tmp_path, phase):
    case = make_case(tmp_path)
    async def run():
        raw = response(case[0])
        task = asyncio.create_task(pa.execute_attempt(case[0], case[1], case[2], fixture(raw,
            header_delay=100 if phase == "header" else 0, chunks=((100, raw),) if phase == "body" else None)))
        await asyncio.sleep(0 if phase == "before_marker" else .03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    asyncio.run(run())
    row = case[1].attempt("attempt-1")
    assert row["state"] == ("RESERVED" if phase == "before_marker" else "UNKNOWN")
    assert case[1].status()["reserved_micros"] > 0


@pytest.mark.parametrize("wrong", [lambda r: None, httpx.MockTransport(lambda r: httpx.Response(200)), pa.OfflineFixture(True, 0, ()), pa.OfflineFixture(200, 0, [(0, b'{}')])])
def test_C07_constructor_rejects_callbacks_and_mutable_inputs(tmp_path, wrong):
    case = make_case(tmp_path)
    with pytest.raises(ValueError):asyncio.run(pa.execute_attempt(case[0], case[1], case[2], wrong))
    assert case[1].status()["attempt_ids"] == []


def test_C06_marker_failure_zero_send(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    def failed(*args, **kwargs):raise LedgerError("contention")
    monkeypatch.setattr(case[1], "mark_sent", failed)
    with pytest.raises(LedgerError):execute(case)
    assert case[1].attempt("attempt-1")["state"] == "RESERVED"
    assert not (case[0].evidence_root / "attempts/attempt-1/transport.json").exists()


def test_C08_finite_ledger_method_calls(tmp_path, monkeypatch):
    case = make_case(tmp_path); calls = []
    for name in ("reserve", "attempt", "mark_sent", "mark_unknown", "reconcile", "status"):
        original = getattr(case[1], name)
        def observed(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(case[1], name, observed)
    execute(case)
    assert calls == ["reserve", "attempt", "mark_sent", "reconcile"]


def _child_execute(plan, path, admission, raw, queue):
    ledger = Ledger.open(path, program_id=plan.spec.program_id, store_id=plan.spec.store_id)
    try:result = asyncio.run(pa.execute_attempt(plan, ledger, admission, fixture(raw))); queue.put(result.ledger_state)
    except (LedgerError, ValueError):queue.put("denied")


def test_C01_concurrent_processes_same_attempt_one_send(tmp_path):
    case = make_case(tmp_path)
    ctx = multiprocessing.get_context("spawn"); queue = ctx.Queue()
    args = (case[0], case[1].path, case[2], response(case[0]), queue)
    processes = [ctx.Process(target=_child_execute, args=args) for _ in range(2)]
    for process in processes:process.start()
    for process in processes:process.join(10); assert process.exitcode == 0
    assert [queue.get(timeout=1) for _ in processes].count("RECONCILED") >= 1
    assert trace(case[0])["calls"] == 1
    assert sum(e["kind"] == "sent" for e in case[1].events()) == 1
    assert case[1].status()["committed_micros"] == 30000


def test_retry_lineage_counts_fresh_request(tmp_path):
    first = make_case(tmp_path, max_attempts=2)
    execute(first, response(first[0], finish="length"))
    second = make_case(tmp_path, attempt_id="retry-1", retry_index=1, parent="attempt-1", max_attempts=2, ledger=first[1])
    execute(second)
    assert first[1].status()["committed_micros"] == 60000
    assert dict(first[1].attempt("retry-1")["receipt"]["usage"])["retry_calls"] == 1
    assert trace(first[0])["calls"] == trace(second[0])["calls"] == 1


def test_R04_duplicate_receipt_other_attempt_retains_liability(tmp_path):
    first = make_case(tmp_path); execute(first)
    second = make_case(tmp_path, attempt_id="other", ledger=first[1])
    execute(second, response(second[0], generation_id="gen-attempt-1"))
    assert first[1].attempt("other")["state"] == "UNKNOWN"
    assert first[1].status()["committed_micros"] == 30000


def test_R02_structurally_inapplicable_cache_only_with_proof(tmp_path):
    case = make_case(tmp_path, profile_changes=dict(cache_write_bound=0, structural_zero=["cache_write_tokens"]))
    data = json.loads(response(case[0])); del data["usage"]["prompt_tokens_details"]["cache_write_tokens"]
    execute(case, json_bytes(data))
    assert case[1].status()["incident_ids"] == []
    assert dict(case[1].attempt("attempt-1")["receipt"]["usage"])["cache_write_tokens"] == 0


def test_reporting_actual_adapter_all_attempts_and_immutable_provenance(tmp_path):
    generation = make_case(tmp_path)
    p, ledger, admission, manifest = generation
    execute(generation)
    assigned = manifest["assignments"][0]
    parts = {"task": json.dumps(assigned["task"], sort_keys=True, ensure_ascii=False)}
    prompt, context = context_packet(parts, max_bytes=10000)
    common = {k: assigned[k] for k in ("model", "task_id", "sample", "condition", "task_hash", "model_snapshot", "prompt_version")}
    gen = dict(**common, protocol=PROTOCOL, record_id="generation", status="completed", timestamp="2026-09-23T00:00:00Z",
        output="Complete end.", output_hash=digest("Complete end."), prompt=prompt, prompt_hash=digest(prompt), context_parts=parts,
        context=context, finish_reason="stop", attempts=[dict(attempt_id=p.spec.attempt_id, role="generation", status="completed", cost_usd=7, usage=None)])
    research_prompt, research_context = context_packet({**parts, "output": gen["output"]}, max_bytes=10000)
    research = make_case(tmp_path, attempt_id="judge", role="research_evaluation", ledger=ledger, manifest=manifest, prompt=research_prompt)
    execute(research, response(research[0], cost="0.07"))
    evaluation = dict(**common, protocol=PROTOCOL, record_id="evaluation", status="completed", timestamp="2026-09-23T00:00:00Z",
        judge_snapshot=assigned["judge_snapshot"], generation_hash=digest(gen), prompt=research_prompt, prompt_hash=digest(research_prompt),
        context_parts={**parts,"output":gen["output"]}, context=research_context, finish_reason="stop",
        llm_results=dict(beats_score=.5,preservation_score=.5,structural_accuracy_score=.5,tone_score=.5),
        attempts=[dict(attempt_id="judge",role="research_evaluation",status="completed",cost_usd=9,usage=None)])
    records = dict(protocol=PROTOCOL, assignment_hash=digest(manifest), generations=[gen], evaluations=[evaluation])
    original = copy.deepcopy(records)
    result = project_cost_report(manifest, records, (p,research[0]), ledger)
    assert records == original
    model = result["report"]["models"][0]
    assert model["deployment_cost"]["total_usd"] == .03
    assert model["research_evaluation_cost"]["total_usd"] == .07
    assert model["resolved"] is None and model["assigned"] == 1
    assert result["exact_cost"]["program"]["committed_micros"] == 100000
    assert result["provenance"]["original_records_hash"] == digest(original)
    assert result["provenance"]["projected_records_hash"] != digest(original)
    stale = copy.deepcopy(records); stale["evaluations"][0]["generation_hash"] = "0"*64
    before = copy.deepcopy(stale)
    with pytest.raises(ValueError, match="Stale judge/generation"):project_cost_report(manifest, stale, (p,research[0]), ledger)
    assert stale == before
    with pytest.raises(ValueError, match="census"):project_cost_report(manifest, records, (p,), ledger)
    swapped = copy.deepcopy(records)
    swapped["generations"][0]["attempts"][0]["attempt_id"] = "equal-count-different-id"
    swapped["evaluations"][0]["generation_hash"] = digest(swapped["generations"][0])
    with pytest.raises(ValueError, match="census"):project_cost_report(manifest, swapped, (p,research[0]), ledger)
    changed_output = copy.deepcopy(records)
    changed_output["generations"][0].update(output="different",output_hash=digest("different"))
    changed_output["evaluations"][0]["generation_hash"] = digest(changed_output["generations"][0])
    with pytest.raises(ValueError):project_cost_report(manifest, changed_output, (p,research[0]), ledger)
    extra = make_case(tmp_path,attempt_id="orphan",ledger=ledger); execute(extra)
    with pytest.raises(ValueError,match="orphan"):project_cost_report(manifest, records, (p,research[0]), ledger)


def test_C02_different_send_event_cannot_gain_permission(tmp_path):
    case = make_case(tmp_path)
    execute(case)
    with pytest.raises(LedgerError):case[1].mark_sent("different-send-id", attempt_id="attempt-1")
    assert execute(case).replayed is True
    assert trace(case[0])["calls"] == 1


def test_C03_processes_contend_last_envelope_capacity(tmp_path):
    # Each quote reserves $62 against the unchanged $100 repair envelope.
    first = make_case(tmp_path, cost_rate="0.02")
    second = make_case(tmp_path, attempt_id="other", cost_rate="0.02", ledger=first[1])
    ctx = multiprocessing.get_context("spawn"); queue = ctx.Queue()
    processes = [ctx.Process(target=_child_execute, args=(c[0], c[1].path, c[2], b'{}', queue)) for c in (first, second)]
    for process in processes:process.start()
    for process in processes:process.join(10); assert process.exitcode == 0
    assert sorted(queue.get(timeout=1) for _ in processes) == ["UNKNOWN", "denied"]
    assert first[1].status()["reserved_micros"] == 62000000
    assert sum((c[0].evidence_root / "attempts" / c[0].spec.attempt_id / "transport.json").exists() for c in (first, second)) == 1


def _crash_child(plan, path, admission, phase):
    ledger = Ledger.open(path, program_id=plan.spec.program_id, store_id=plan.spec.store_id)
    if phase in {"before_reserve", "after_reserve", "after_marker"}:
        name = "mark_sent" if phase == "after_marker" else "reserve"
        original = getattr(ledger, name)
        def crash(*args, **kwargs):
            if phase != "before_reserve":original(*args, **kwargs)
            os._exit(73)
        setattr(ledger, name, crash)
    elif phase == "after_wire":
        original = pa._persist
        def persist(path, raw):
            if path.name == "response.bin":os._exit(73)
            return original(path, raw)
        pa._persist = persist
    else:
        def crash(*args):os._exit(73)
        pa.reconcile_attempt = crash
    asyncio.run(pa.execute_attempt(plan, ledger, admission, fixture(response(plan))))


@pytest.mark.parametrize("phase,state,calls", [("before_reserve",None,0), ("after_reserve","RESERVED",0),
    ("after_marker","SENT",0), ("after_wire","SENT",1), ("after_response","SENT",1)])
def test_C04_real_process_crash_matrix(tmp_path, phase, state, calls):
    case = make_case(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_crash_child, args=(case[0],case[1].path,case[2],phase))
    process.start(); process.join(10)
    assert process.exitcode == 73
    assert int((case[0].evidence_root / "attempts/attempt-1/transport.json").exists()) == calls
    if state is None:
        assert case[1].status()["attempt_ids"] == []
    else:
        assert case[1].attempt("attempt-1")["state"] == state
        assert case[1].status()["reserved_micros"] == 3100000
    if state == "SENT":
        assert execute(case).replayed is True
        assert int((case[0].evidence_root / "attempts/attempt-1/transport.json").exists()) == calls
    if phase == "after_response":
        raw = (case[0].evidence_root / "attempts/attempt-1/response.bin").read_bytes()
        assert pa.reconcile_attempt(case[0],case[1],raw,200).ledger_state == "RECONCILED"


def test_C05_C08_failed_persistence_and_failed_unknown_keep_sent(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    original = pa._persist
    def fail_response(path, raw):
        if path.name == "response.bin":raise OSError("synthetic fsync failure")
        return original(path, raw)
    def fail_unknown(*args, **kwargs):raise LedgerError("synthetic contention")
    monkeypatch.setattr(pa, "_persist", fail_response)
    monkeypatch.setattr(case[1], "mark_unknown", fail_unknown)
    assert execute(case).ledger_state == "SENT"
    assert case[1].attempt("attempt-1")["state"] == "SENT"
    assert case[1].status()["reserved_micros"] == 3100000
    assert trace(case[0])["calls"] == 1


def test_R03_usage_overrun_records_full_counter(tmp_path):
    case = make_case(tmp_path)
    data = json.loads(response(case[0])); data["usage"].update(prompt_tokens=2000,total_tokens=2005)
    execute(case,json_bytes(data))
    assert dict(case[1].attempt("attempt-1")["receipt"]["usage"])["input_tokens"] == 2000
    assert case[1].status()["dispatch_blocked"]


def test_R04_saved_receipt_tamper_is_not_reconciled(tmp_path):
    case = make_case(tmp_path); execute(case)
    directory = case[0].evidence_root / "attempts/attempt-1"
    receipt = json.loads((directory / "receipt.json").read_bytes()); receipt["actual_usd"] = "0"
    (directory / "receipt.json").write_bytes(json_bytes(receipt))
    with pytest.raises(ValueError):pa.reconcile_attempt(case[0],case[1],response(case[0]),200)
    assert case[1].status()["committed_micros"] == 30000


def test_P02_failed_retry_and_all_activity_roles_survive_projection(tmp_path):
    first = make_case(tmp_path,max_attempts=2); execute(first,status=503)
    retry = make_case(tmp_path,attempt_id="retry",retry_index=1,parent="attempt-1",max_attempts=2,ledger=first[1])
    execute(retry,response(retry[0],cost="0.07"))
    cases = [first,retry]
    for role in ("critic","retrieval","selection","research_evaluation"):
        case = make_case(tmp_path,attempt_id=role,role=role,ledger=first[1]); execute(case); cases.append(case)
    unknown = make_case(tmp_path,attempt_id="unknown",ledger=first[1]); execute(unknown,b'{}'); cases.append(unknown)
    projected = project_attempts(tuple(c[0] for c in cases),first[1])
    assert [r["attempt_id"] for r in projected] == ["attempt-1","retry","critic","retrieval","selection","research_evaluation","unknown"]
    assert projected[0]["status"] == "provider_failure" and projected[0]["cost_usd"] == .03
    assert projected[-1]["cost_usd"] is None
    assert first[1].status()["committed_micros"] == 220000


def test_P04_unknown_usage_projection_retains_original_detail(tmp_path):
    case = make_case(tmp_path)
    data = json.loads(response(case[0])); del data["usage"]["completion_tokens_details"]
    execute(case,json_bytes(data))
    assert project_attempts((case[0],),case[1])[0]["usage"] is None
    receipt = case[1].attempt("attempt-1")["receipt"]
    assert dict(receipt["usage"])["input_tokens"] == 20 and dict(receipt["usage"])["reasoning_tokens"] is None


def test_P01_duplicate_plan_same_count_swapped_id_denied(tmp_path):
    case = make_case(tmp_path); execute(case)
    with pytest.raises(ValueError):project_attempts((case[0],case[0]),case[1])
    swapped = replace(case[0],spec=replace(case[0].spec,attempt_id="different"))
    with pytest.raises(ValueError):project_attempts((swapped,),case[1])


@pytest.mark.parametrize("field,value", [("max_tokens",101),("reasoning_effort","unknown"),("messages",(("tool","x"),))])
def test_D03_unsupported_request_fields_deny(tmp_path,field,value):
    case = make_case(tmp_path)
    changed = replace(case[0]._fields,**{field:value})
    artifacts = tuple((r.name,pa.artifact_bytes(case[0],r)) for r in case[0].spec.artifacts)
    with pytest.raises(ValueError):pa.prepare_attempt(case[0].spec,case[0].snapshot,artifacts,changed,case[0].evidence_root)
    assert case[1].status()["attempt_ids"] == []


def test_D03_real_snapshot_cannot_receive_synthetic_admission(tmp_path):
    case = make_case(tmp_path)
    snapshot = validate_snapshot(case[0].snapshot._sources[:2],case[0].route,case[0].snapshot.observed_at,case[0].snapshot.expires_at)
    with pytest.raises(ValueError,match="unqualified"):make_case(tmp_path / "other",snapshot=snapshot)


def test_P05_fractional_quote_rounds_each_line_upward(tmp_path):
    case = make_case(tmp_path,cost_rate="0.000000001")
    execute(case,response(case[0],cost="0"))
    row = case[1].attempt("attempt-1")
    assert row["reserved_micros"] == 4 and row["actual_micros"] == 0


def test_P05_nonfinite_display_rejected_without_changing_ledger(tmp_path,monkeypatch):
    case = make_case(tmp_path); execute(case)
    original = case[1].attempt
    def hostile_display_row(attempt_id):
        row = original(attempt_id); row["receipt"]["actual_usd"] = "1e999"
        return row
    # Beyond accepted ledger precision; exercise the explicit defensive display boundary.
    monkeypatch.setattr(case[1],"attempt",hostile_display_row)
    with pytest.raises(ValueError,match="displayed"):project_attempts((case[0],),case[1])
    assert original("attempt-1")["actual_micros"] == 30000


def test_D07_same_body_distinct_arm_distinct_identity(tmp_path):
    first = make_case(tmp_path)
    manifest = copy.deepcopy(first[3]); manifest["assignments"][0]["condition"] = "B"
    # Build a fully valid alternate arm instead of mutating an already frozen plan.
    p = first[0]; artifacts = {r.name:pa.artifact_bytes(p,r) for r in p.spec.artifacts}
    artifacts["assignment"] = json_bytes(manifest)
    for key in pa.GATES:
        gate = json.loads(artifacts['gate_'+key]); gate['assignment_hash'] = digest(manifest)
        artifacts['gate_'+key] = json_bytes(gate)
    spec = replace(p.spec,attempt_id="arm-B",condition="B",assignment_hash=digest(manifest),
        artifacts=tuple(ArtifactRef.of(k,v) for k,v in artifacts.items()))
    second = pa.prepare_attempt(spec,p.snapshot,tuple(artifacts.items()),p._fields,p.evidence_root)
    assert second.body_bytes == p.body_bytes
    assert pa.request_hash(second) != pa.request_hash(p)
    admission = replace(first[2],assignment_hash=digest(manifest),
        authority_refs=tuple(r for r in spec.artifacts if r.name.startswith('gate_')))
    execute(first)
    asyncio.run(pa.execute_attempt(second,first[1],admission,fixture(response(second))))
    assert first[1].status()["attempt_ids"] == ["arm-B","attempt-1"]
    assert first[1].status()["committed_micros"] == 60000


@pytest.mark.parametrize("mutation",["missing_request","missing_cache","missing_limit","missing_reasoning"])
def test_D03_missing_required_support_never_sends(tmp_path,mutation):
    case = make_case(tmp_path)
    catalog,endpoints,profile = [json.loads(raw) for raw in case[0].snapshot._sources]
    endpoint = endpoints['data']['endpoints'][0]
    if mutation == 'missing_request':del endpoint['pricing']['request']
    if mutation == 'missing_cache':del endpoint['pricing']['input_cache_read']
    if mutation == 'missing_limit':endpoint['supported_parameters'].remove('max_tokens')
    if mutation == 'missing_reasoning':endpoint['supported_parameters'].remove('reasoning')
    profile['endpoints_sha256'] = sha(json_bytes(endpoints))
    with pytest.raises(ValueError):
        snapshot = validate_snapshot(tuple(json_bytes(v) for v in (catalog,endpoints,profile)),case[0].route,
            case[0].snapshot.observed_at,case[0].snapshot.expires_at)
        make_case(tmp_path/'other',snapshot=snapshot)
    assert case[1].status()['attempt_ids'] == []


def test_W1_nonmonetary_fractional_assignment_metadata(tmp_path):
    base = make_case(tmp_path/'base')
    manifest = copy.deepcopy(base[3]); task = manifest['assignments'][0]['task']
    task['synthetic_difficulty_metadata'] = .25
    manifest['assignments'][0]['task_hash'] = digest(task)
    case = make_case(tmp_path/'fraction',manifest=manifest)
    assert execute(case).ledger_state == 'RECONCILED'
    assert pa.artifact_bytes(case[0],next(r for r in case[0].spec.artifacts if r.name=='assignment')) == json_bytes(manifest)


def _retry_case(root,parent,attempt_id,index=1):
    return make_case(root,attempt_id=attempt_id,retry_index=index,parent=parent[0].spec.attempt_id,
                     max_attempts=3,ledger=parent[1])


def _observed_ledger_calls(monkeypatch,ledger):
    calls=[]
    for name in ('events','reserve','attempt','mark_sent','mark_unknown','reconcile','status'):
        original=getattr(ledger,name)
        def wrapped(*args,_name=name,_original=original,**kwargs):
            calls.append(_name); return _original(*args,**kwargs)
        monkeypatch.setattr(ledger,name,wrapped)
    return calls


@pytest.mark.parametrize('depth',[1,2])
def test_R1_retry_chain_slot_occupancy_and_replay(tmp_path,depth,monkeypatch):
    root=make_case(tmp_path,max_attempts=3);execute(root)
    first=_retry_case(tmp_path,root,'child');execute(first)
    parent=root if depth==1 else first
    if depth==2:
        last=_retry_case(tmp_path,first,'grandchild',2);execute(last)
    else:last=first
    other=_retry_case(tmp_path,parent,'sibling',depth)
    calls=_observed_ledger_calls(monkeypatch,root[1])
    with pytest.raises(LedgerError):execute(other)
    assert len(calls)<=6
    calls.clear();assert execute(last).replayed is True;assert len(calls)<=6
    assert sum(e['kind']=='sent' for e in root[1].events())==depth+1


def test_R1_concurrent_retry_siblings_one_slot(tmp_path):
    root=make_case(tmp_path,max_attempts=3);execute(root)
    children=[_retry_case(tmp_path,root,'child-a'),_retry_case(tmp_path,root,'child-b')]
    ctx=multiprocessing.get_context('spawn');queue=ctx.Queue()
    processes=[ctx.Process(target=_child_execute,args=(c[0],c[1].path,c[2],response(c[0]),queue)) for c in children]
    for p in processes:p.start()
    for p in processes:p.join(10);assert p.exitcode==0
    assert sorted(queue.get(timeout=1) for _ in processes)==['RECONCILED','denied']
    assert sum(e['kind']=='sent' for e in root[1].events())==2
    assert root[1].status()['committed_micros']==60000


def test_R1_cancelled_retry_slot_remains_occupied(tmp_path,monkeypatch):
    root=make_case(tmp_path,max_attempts=3);execute(root)
    child=_retry_case(tmp_path,root,'cancelled-child')
    original=child[1].mark_sent
    def stop(*args,**kwargs):raise LedgerError('before marker')
    monkeypatch.setattr(child[1],'mark_sent',stop)
    with pytest.raises(LedgerError):execute(child)
    monkeypatch.setattr(child[1],'mark_sent',original)
    child[1].cancel_unsent('synthetic-cancel',attempt_id='cancelled-child',reason='SYNTHETIC ONLY')
    sibling=_retry_case(tmp_path,root,'other-child')
    with pytest.raises(LedgerError):execute(sibling)
    assert root[1].status()['committed_micros']==30000


@pytest.mark.parametrize('poison',['root-label','ancestor-index','ancestor-policy','substituted-parent'])
def test_R1_ancestor_manifest_must_match_actual_ledger(tmp_path,poison):
    root=make_case(tmp_path,max_attempts=3);execute(root)
    child=_retry_case(tmp_path,root,'child');execute(child)
    grandchild=_retry_case(tmp_path,child,'grandchild',2)
    target=root[0] if poison=='root-label' else child[0]
    path=target.evidence_root/'attempts'/target.spec.attempt_id/'manifest.json'
    manifest=json.loads(path.read_bytes())
    if poison=='root-label':manifest['spec']['attempt_id']='invented-root'
    if poison=='ancestor-index':manifest['spec']['retry_index']=0;manifest['spec']['parent_attempt_id']=None
    if poison=='ancestor-policy':manifest['spec']['max_attempts']=2
    if poison=='substituted-parent':
        other=make_case(tmp_path,attempt_id='other-root',max_attempts=3,ledger=root[1]);execute(other)
        manifest['spec']['parent_attempt_id']='other-root'
    path.write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):execute(grandchild)
    assert not (grandchild[0].evidence_root/'attempts/grandchild/transport.json').exists()


@pytest.mark.parametrize('field,value',[(field,value) for field in ('model','provider','service_tier','id') for value in ([],{},True)])
def test_R1_malformed_identity_is_typed_failure(tmp_path,field,value):
    case=make_case(tmp_path);data=json.loads(response(case[0]));data[field]=value
    raw=json_bytes(data);escaped=None;result=None
    try:result=execute(case,raw)
    except Exception as exc:escaped=exc
    assert escaped is None, f"untyped response failure: {escaped}"
    assert result.transport_status=='infrastructure_failure' and result.ledger_state=='UNKNOWN'
    assert case[1].status()['reserved_micros']==3100000
    assert (case[0].evidence_root/'attempts/attempt-1/response.bin').read_bytes()==raw


@pytest.mark.parametrize('location',['choices','choice','message','finish','refusal','role','cost_details','prompt_details','completion_details'])
def test_R1_malformed_neighbor_containers_are_typed(tmp_path,location):
    case=make_case(tmp_path);data=json.loads(response(case[0]))
    if location=='choices':data['choices']={}
    if location=='choice':data['choices']=[[]]
    if location=='message':data['choices'][0]['message']=[]
    if location=='finish':data['choices'][0]['finish_reason']={}
    if location in {'refusal','role'}:data['choices'][0]['message'][location]=[]
    if location=='cost_details':data['usage']['cost_details']=[]
    if location=='prompt_details':data['usage']['prompt_tokens_details']=[]
    if location=='completion_details':data['usage']['completion_tokens_details']=[]
    result=execute(case,json_bytes(data))
    assert result.transport_status=='infrastructure_failure' and result.ledger_state=='UNKNOWN'


@pytest.mark.parametrize('part',['blobs','attempts'])
def test_R1_intermediate_symlink_before_permission_and_recovery(tmp_path,part):
    case=make_case(tmp_path);root=case[0].evidence_root
    target=root/part;outside=tmp_path/'outside';target.rename(outside);target.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):execute(case)
    assert case[1].status()['attempt_ids']==[]
    target.unlink();outside.rename(target)
    execute(case)
    target.rename(outside);target.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):pa.reconcile_attempt(case[0],case[1],response(case[0]),200)
    assert case[1].status()['committed_micros']==30000


@pytest.mark.parametrize('kind',['directory','fifo','final-symlink','parent-symlink','escape'])
def test_R1_reader_rejects_unsafe_objects_and_paths(tmp_path,kind):
    base=tmp_path.resolve();valid=base/'valid';valid.write_bytes(b'confined')
    assert pa._read(valid)==b'confined'
    target=base/'target'
    if kind=='directory':target.mkdir()
    if kind=='fifo':os.mkfifo(target)
    if kind=='final-symlink':target.symlink_to(valid)
    if kind=='parent-symlink':
        target.symlink_to(base,target_is_directory=True);target=target/'valid'
    if kind=='escape':target=base/'..'/base.name/'valid'
    with pytest.raises(ValueError):pa._read(target)


@pytest.mark.parametrize('depth',[0,1,2])
@pytest.mark.parametrize('failure',['result-read','commit-ack','unknown-cleanup','authority-unavailable'])
def test_R1_financial_truth_and_six_call_bound(tmp_path,monkeypatch,depth,failure):
    case=make_case(tmp_path,max_attempts=3)
    for index in range(1,depth+1):
        execute(case);case=_retry_case(tmp_path,case,'child-'+str(index),index)
    ledger=case[1];original_reconcile=ledger.reconcile;original_read=pa._read
    committed=False
    if failure in {'commit-ack','authority-unavailable'}:
        def lost_ack(*args,**kwargs):
            nonlocal committed
            original_reconcile(*args,**kwargs);committed=True;raise LedgerError('commit acknowledgment lost')
        monkeypatch.setattr(ledger,'reconcile',lost_ack)
    if failure=='result-read':
        def bad_read(path):
            if committed and path.name=='receipt.json':raise OSError('postcommit read')
            return original_read(path)
        def reconcile(*args,**kwargs):
            nonlocal committed
            row=original_reconcile(*args,**kwargs);committed=True;return row
        monkeypatch.setattr(ledger,'reconcile',reconcile);monkeypatch.setattr(pa,'_read',bad_read)
    if failure=='unknown-cleanup':
        def broken(*args,**kwargs):raise LedgerError('unknown cleanup failed')
        monkeypatch.setattr(ledger,'mark_unknown',broken)
    original_attempt=ledger.attempt
    if failure=='authority-unavailable':
        def unavailable(*args):
            if committed:raise LedgerError('authority unavailable')
            return original_attempt(*args)
        monkeypatch.setattr(ledger,'attempt',unavailable)
    calls=_observed_ledger_calls(monkeypatch,ledger)
    if failure=='authority-unavailable':
        with pytest.raises(LedgerError,match='unavailable'):execute(case)
    else:
        result=execute(case,b'{}' if failure=='unknown-cleanup' else None)
        assert result.ledger_state==('SENT' if failure=='unknown-cleanup' else 'RECONCILED')
        if failure!='unknown-cleanup':assert result.transport_status=='completed'
    assert len(calls)<=6,calls
    row=original_attempt(case[0].spec.attempt_id)
    assert row['state']==('SENT' if failure=='unknown-cleanup' else 'RECONCILED')
    assert trace(case[0])['calls']==1
