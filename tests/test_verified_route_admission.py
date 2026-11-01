"""A verified route must be admissible WITHOUT admitting a fixture, and refuse in both directions.

Every negative control here has a positive partner, so a gate that refuses everything cannot pass for
a gate that refuses correctly. Three of these controls were added after an adversarial review found
the earlier versions passing for the wrong reason: an assertion that could not see the bound it named,
a tautology that blessed the very hole it was meant to catch, and a credential test that never wrote
the file it inspected.
"""
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import httpx
import pytest

from measurement_contract import PROTOCOL, context_packet, digest
import provider_attempts as pa
import provider_prices as pp
from provider_prices import ArtifactRef, Route, json_bytes, sha, validate_snapshot
from spend_ledger import Authorization, Ledger, LedgerError, USAGE_KINDS, ValidationError

BOUND = 4000
OUTPUT_BOUND = 100


def _blobs(*, cost_rate="0.001", route=None, verified_minutes_ago=30, verdict="PASS",
           route_independent=True, subject=None):
    route = route or Route("openrouter", "https://openrouter.ai/api/v1", "/chat/completions",
                           "fixture/model", "fixture/model-20260923", "fixture/local", "default", "USD")
    now = datetime.now(timezone.utc)
    window = ((now - timedelta(seconds=1)).isoformat(), (now + timedelta(hours=1)).isoformat())
    catalog = json_bytes({"data": [{"id": route.model_id, "canonical_slug": route.canonical_model_slug}]})
    endpoints = json_bytes({"data": {"id": route.model_id, "endpoints": [dict(
        tag=route.endpoint_tag, provider_name="Fixture", model_id=route.model_id,
        context_length=10000, max_prompt_tokens=9000, max_completion_tokens=1000,
        supported_parameters=["max_tokens", "reasoning", "temperature", "seed"],
        pricing=dict(prompt=cost_rate, completion=cost_rate, input_cache_read=cost_rate,
                     input_cache_write=cost_rate, input_cache_write_1h=cost_rate, request="0"))]}})
    # The retained C8 artifacts write `+08:00` and an uppercase verdict. A parser that demanded `Z`
    # and lowercase would have refused every real verification and accepted every forgery that
    # copied the profile's own format.
    verified = (now - timedelta(minutes=verified_minutes_ago)).astimezone(timezone(timedelta(hours=8)))
    verification = json_bytes({"phase": "w3b2-c8", "subject": subject or route.model_id,
                               "verifier": "route-independent", "route_independent": route_independent,
                               "verified_at": verified.isoformat(), "verdict": verdict,
                               "checks": [{"id": "C8.1", "status": "pass"}]})
    profile = dict(kind="C8 VERIFIED ROUTE", catalog_sha256=sha(catalog), endpoints_sha256=sha(endpoints),
                   model_id=route.model_id, canonical_model_slug=route.canonical_model_slug,
                   endpoint_tag=route.endpoint_tag, input_bound=BOUND, output_bound=OUTPUT_BOUND,
                   reasoning_bound=OUTPUT_BOUND, cache_read_bound=BOUND, cache_write_bound=BOUND,
                   structural_zero=[], no_incremental_infrastructure=True, no_metered_agent=True,
                   verification_sha256=sha(verification), verifier="route-independent",
                   verdict=verdict, verified_at=verified.astimezone(timezone.utc).isoformat())
    return route, window, catalog, endpoints, verification, profile


def make_verified_case(root, *, profile_changes=None, gate_authority="C8 VERIFIED ROUTE",
                       mode="c8_verified", max_tokens=40, reasoning="max", **blob_kwargs):
    root = Path(root).resolve()
    route, (start, end), catalog, endpoints, verification, profile = _blobs(**blob_kwargs)
    profile.update(profile_changes or {})
    snapshot = validate_snapshot((catalog, endpoints, json_bytes(profile), verification),
                                 route, start, end)
    task = dict(task_id="task", task_type="theory_conversion", subtype="default", source="EARLY CAUSE and END")
    assignment = dict(model="fixture-model", task_id="task", sample=0, condition="A", story_id="story",
                      task=task, task_hash=digest(task), model_snapshot=route.canonical_model_slug,
                      judge_snapshot=route.canonical_model_slug, prompt_version="fixture-v1")
    manifest = dict(protocol=PROTOCOL, assignments=[assignment])
    prompt = context_packet({"task": json.dumps(task, sort_keys=True, ensure_ascii=False)}, max_bytes=10000)[0]
    if isinstance(reasoning, int):
        fields = pa.RequestFields((("user", prompt),), max_tokens, reasoning_max_tokens=reasoning)
        effort = f"tokens:{reasoning}"
    else:
        fields = pa.RequestFields((("user", prompt),), max_tokens, reasoning_effort=reasoning)
        effort = reasoning or "unspecified"
    artifacts = dict(assignment=json_bytes(manifest), task=json_bytes(task), prompt=prompt.encode(),
                     harness=b"verified-harness-v1", protocol=b"verified-protocol-v1",
                     settings=json_bytes(asdict(fields)), native_package=b"VERIFIED native38428703 version0.8.0")
    route_hash = sha(json_bytes(asdict(route)))
    for gate in sorted(pa.GATES):
        artifacts["gate_" + gate] = json_bytes(dict(kind=gate, authority=gate_authority,
            assignment_hash=digest(manifest), route_sha256=route_hash, valid_until=end))
    refs = tuple(ArtifactRef.of(k, v) for k, v in artifacts.items())
    spec = pa.AttemptSpec("attempt-v", "run", "episode", digest(manifest), "task", sha(artifacts["task"]),
        "story", "A", 0, route.canonical_model_slug, sha(artifacts["harness"]), sha(artifacts["protocol"]),
        sha(artifacts["settings"]), effort, None, 0, 1, "repair", "deployment", "generation",
        "verified-program", "verified-store", sha(artifacts["native_package"]), refs)
    plan = pa.prepare_attempt(spec, snapshot, tuple(artifacts.items()), fields, root / "evidence")
    admission = pa.Admission(mode, tuple(r for r in refs if r.name.startswith("gate_")),
                             spec.assignment_hash, route_hash, end)
    ledger = Ledger.create(root / "ledger.sqlite", program_id=spec.program_id, store_id=spec.store_id,
                           authorization=Authorization("create", "a" * 64, "C8 VERIFIED ROUTE"))
    # Without this the seed charge leaves the ledger dispatch-blocked, and every assertion below
    # about "no reservation happened" would be satisfied by a ledger that could never reserve.
    ledger.reconcile_external("verify-host", charge_id="hosted_session", receipt_id="verify-host",
        actual_usd="0", evidence_sha256="b" * 64,
        authorization=Authorization("hosted", "c" * 64, "C8 VERIFIED ROUTE"))
    return plan, ledger, admission, end


def _wire_response(plan, cost="0.002", finish="stop", content="Complete end."):
    # A monetary JSON literal, injected without a float round trip.
    data = dict(id="gen-" + plan.spec.attempt_id, model=plan.route.model_id, provider="Fixture",
                service_tier="default",
                choices=[dict(finish_reason=finish, message=dict(content=content, role="assistant"))],
                usage=dict(cost="MONEY", is_byok=False, prompt_tokens=20, completion_tokens=5,
                           total_tokens=25, completion_tokens_details=dict(reasoning_tokens=2)))
    return json_bytes(data).replace(b'"MONEY"', cost.encode())


# --- the positive direction: a verified route is admissible -------------------------------

def test_verified_profile_qualifies():
    route, (start, end), catalog, endpoints, verification, profile = _blobs()
    snapshot = validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)
    assert snapshot.qualification == "c8_verified"
    assert snapshot.qualification in pp.QUALIFIED_KINDS
    assert snapshot.source_refs[3].sha256 == sha(verification)


def test_verified_route_reserves_the_request_not_the_profile_worst_case(tmp_path):
    """The profile is the ceiling; the request is the number.

    Reserving the profile verbatim charged every 6,000-token task for a 10,000-token one and every
    short prompt for the longest in the set, which is what left no room to retry anything. The
    leaner figure is still a PROVEN ceiling, so it cannot under-reserve.
    """
    plan, _, _, _ = make_verified_case(tmp_path, max_tokens=40)
    census = {c.kind: c for c in plan.census}
    assert census["output_tokens"].max_units == 40 < OUTPUT_BOUND
    assert census["input_tokens"].max_units == len(plan.body_bytes) < BOUND
    assert census["cache_read_tokens"].max_units == 0
    assert plan.deadline_seconds == pa.MAX_DEADLINE_SECONDS


def test_verified_request_stays_pinned_and_carries_its_effort(tmp_path):
    plan, _, _, _ = make_verified_case(tmp_path)
    body = json.loads(plan.body_bytes)
    assert body["provider"] == {"only": ["fixture/local"], "order": ["fixture/local"],
                               "allow_fallbacks": False, "require_parameters": True}
    assert body["reasoning"] == {"effort": "max"}


def test_real_admission_selects_the_network_seam_not_a_mock(tmp_path):
    _, _, admission, _ = make_verified_case(tmp_path)
    transport = pa._transport(admission, None)
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert not isinstance(transport, httpx.MockTransport)


def test_verified_route_gets_a_deadline_a_real_call_can_finish_in(tmp_path):
    """A one-second deadline was the fixture default for the whole history of this adapter. The
    end-to-end control below cannot catch it, because the mock answers instantly."""
    plan, _, _, _ = make_verified_case(tmp_path)
    assert plan.deadline_seconds == pa.MAX_DEADLINE_SECONDS == 600
    assert plan.deadline_seconds > 120  # the range the offline fixture was frozen at


def test_verified_route_reserves_sends_and_reconciles(tmp_path, monkeypatch):
    """The claim under test: a verified route actually reserves, sends, and settles against a ledger.

    Nothing else in this file runs `execute_attempt`, so without this control the central change
    would be untested — which is exactly what the review found in the first version.
    """
    plan, ledger, admission, _ = make_verified_case(tmp_path)
    monkeypatch.setenv(pa.CREDENTIAL_ENV, "test-key-not-a-secret")
    seen = {}

    def fake_transport(*args, **kwargs):
        async def handler(request):
            seen["authorization"] = request.headers.get("authorization")
            seen["body_sha256"] = sha(request.content)
            return httpx.Response(200, content=_wire_response(plan))
        return httpx.MockTransport(handler)

    monkeypatch.setattr(pa.httpx, "AsyncHTTPTransport", fake_transport)
    result = asyncio.run(pa.execute_attempt(plan, ledger, admission, None))
    assert result.ledger_state == "RECONCILED"
    assert seen["authorization"] == "Bearer test-key-not-a-secret"
    assert seen["body_sha256"] == sha(plan.body_bytes)
    kinds = [event["kind"] for event in ledger.events()]
    assert "reserved" in kinds and "sent" in kinds and "reconciled" in kinds
    recorded = (tmp_path / "evidence" / "attempts" / plan.spec.attempt_id / "transport.json").read_text()
    assert "test-key-not-a-secret" not in recorded


# --- the negative direction: every refusal fails closed ------------------------------------

def test_verified_kind_without_its_artifact_is_refused():
    route, (start, end), catalog, endpoints, verification, profile = _blobs()
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile)), route, start, end)


def test_synthetic_kind_with_a_fourth_source_is_refused():
    route, (start, end), catalog, endpoints, verification, profile = _blobs()
    profile["kind"] = "SYNTHETIC OFFLINE ONLY"
    for key in ("verification_sha256", "verifier", "verdict", "verified_at"):
        profile.pop(key)
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_non_passing_verdict_does_not_qualify():
    route, (start, end), catalog, endpoints, verification, profile = _blobs(verdict="FAIL")
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_the_artifact_must_be_route_independent():
    route, (start, end), catalog, endpoints, verification, profile = _blobs(route_independent=False)
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_the_artifact_must_name_the_same_subject():
    route, (start, end), catalog, endpoints, verification, profile = _blobs(subject="some/other-model")
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_a_stale_verification_does_not_qualify():
    route, (start, end), catalog, endpoints, verification, profile = _blobs(verified_minutes_ago=60 * 30)
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_the_artifact_bytes_are_read_not_merely_hashed():
    """The profile used to grade its own homework: the artifact was hashed and never inspected, so
    the catalog bytes themselves served as a "passing" verification."""
    route, (start, end), catalog, endpoints, verification, profile = _blobs()
    profile["verification_sha256"] = sha(catalog)
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), catalog), route, start, end)


def test_verification_digest_must_match_the_artifact_bytes():
    route, (start, end), catalog, endpoints, verification, profile = _blobs()
    profile["verification_sha256"] = "e" * 64
    with pytest.raises(ValidationError):
        validate_snapshot((catalog, endpoints, json_bytes(profile), verification), route, start, end)


def test_offline_admission_cannot_admit_a_verified_route(tmp_path):
    # The forged admission carries gates that name the SYNTHETIC authority, so the only thing left
    # to refuse it is the mode/qualification equality — otherwise this control would pass on the
    # gate check alone and the equality would stay untested.
    plan, _, admission, _ = make_verified_case(tmp_path, gate_authority="SYNTHETIC OFFLINE ONLY")
    forged = pa.Admission("offline_fixture", admission.authority_refs,
                          admission.assignment_hash, admission.route_sha256, admission.valid_until)
    with pytest.raises(ValidationError, match="authority does not match"):
        pa._admit(plan, forged)


def test_synthetic_gate_authority_cannot_vouch_for_a_verified_route(tmp_path):
    plan, _, admission, _ = make_verified_case(tmp_path, gate_authority="SYNTHETIC OFFLINE ONLY")
    with pytest.raises(ValidationError, match="gate binding"):
        pa._admit(plan, admission)


def test_verified_admission_refuses_fixture_bytes(tmp_path):
    _, _, admission, _ = make_verified_case(tmp_path)
    fixture = pa.OfflineFixture(200, 0, ((0, b"{}"),))
    with pytest.raises(ValidationError):
        pa._transport(admission, fixture)


def test_absent_credential_refuses_before_permission_is_consumed(tmp_path, monkeypatch):
    plan, ledger, admission, _ = make_verified_case(tmp_path)
    monkeypatch.delenv(pa.CREDENTIAL_ENV, raising=False)
    with pytest.raises(ValidationError):
        asyncio.run(pa.execute_attempt(plan, ledger, admission, None))
    assert not [e for e in ledger.events() if e["kind"] == "reserved"]


# --- the reservation must cover what is actually sent --------------------------------------

def test_prompt_over_the_declared_input_bound_is_refused(tmp_path):
    # The cache bounds are narrowed with it, or the snapshot would be refused for the WRONG reason
    # and this control would pass while testing nothing.
    with pytest.raises(ValidationError, match="input bound"):
        make_verified_case(tmp_path, profile_changes=dict(
            input_bound=8, cache_read_bound=8, cache_write_bound=8))


def test_reasoning_bound_must_cover_what_effort_permits(tmp_path):
    """`effort` lets the provider spend up to `max_tokens` completion tokens as reasoning. A profile
    that reserves reasoning 0 could under-reserve by the whole completion budget."""
    with pytest.raises(ValidationError, match="reasoning bound"):
        make_verified_case(tmp_path, profile_changes=dict(reasoning_bound=0))


def test_request_output_above_the_profile_bound_is_refused(tmp_path):
    # An explicit reasoning ceiling AT the bound, so the reasoning check cannot fire first and this
    # control is actually testing the output check (the first version let the other one shadow it).
    with pytest.raises(ValidationError, match="output bound"):
        make_verified_case(tmp_path, max_tokens=OUTPUT_BOUND + 1, reasoning=OUTPUT_BOUND)
