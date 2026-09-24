"""One immutable, ledger-bound OFFLINE attempt; no credential or live transport path."""
import asyncio
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import stat

import httpx

from measurement_contract import digest, identity, project_report
from provider_prices import (ArtifactRef, MAX_BYTES, PriceSnapshot, Route, digest_text,
                             exact_fields, fresh, integer, json_bytes, money, parse_json,
                             sha, snapshot_identity, text, utc, verify_snapshot)
from spend_ledger import (Ledger, LedgerError, Quote, Receipt, USAGE_KINDS, UsageBound,
                         ValidationError)

GATES = frozenset({"rights", "provider_permission", "phase", "protocol", "roster", "effort", "budget", "billing"})
ROLES = {"generation": "deployment", "critic": "deployment", "retrieval": "deployment",
         "selection": "deployment", "operational": "deployment", "research_evaluation": "research"}


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: str
    run_id: str
    episode_id: str
    assignment_hash: str
    task_id: str
    task_sha256: str
    story_id: str
    condition: str
    sample: int
    assigned_model_snapshot: str
    harness_sha256: str
    protocol_sha256: str
    settings_sha256: str
    effort: str
    parent_attempt_id: str | None
    retry_index: int
    max_attempts: int
    envelope: str
    cost_role: str
    activity_role: str
    program_id: str
    store_id: str
    native_package_sha256: str
    artifacts: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class RequestFields:
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    temperature: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class UsageClassPlan:
    kind: str
    max_units: int
    usd_per_unit: str
    evidence_ref: ArtifactRef
    inactive_reason: str | None


@dataclass(frozen=True)
class RequestPlan:
    spec: AttemptSpec
    route: Route
    snapshot: PriceSnapshot
    body_bytes: bytes
    prompt_ref: ArtifactRef
    census: tuple[UsageClassPlan, ...]
    deadline_seconds: int
    evidence_root: Path
    _fields: RequestFields = field(repr=False)


@dataclass(frozen=True)
class Admission:
    mode: str
    authority_refs: tuple[ArtifactRef, ...]
    assignment_hash: str
    route_sha256: str
    valid_until: str


@dataclass(frozen=True)
class OfflineFixture:
    status_code: int
    header_delay_ms: int
    chunks: tuple[tuple[int, bytes], ...]


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    manifest_sha256: str
    ledger_state: str
    response_ref: ArtifactRef | None
    receipt_ref: ArtifactRef | None
    output_ref: ArtifactRef | None
    transport_status: str
    replayed: bool


def _name(value):
    text(value)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value) or value in {".", ".."}:
        raise ValidationError("unsafe evidence name")
    return value


def _root(path):
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValidationError("absolute trusted evidence root required")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValidationError("symlink evidence root")
    if path.resolve() != path:
        raise ValidationError("noncanonical evidence root")
    return path


def _read(path):
    # Walk every component through pinned directory descriptors, never symlinks.
    # O_NONBLOCK prevents opening a substituted FIFO from waiting for a writer.
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValidationError("absolute confined evidence path required")
    directory = None
    try:
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValidationError("regular evidence file required")
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise ValidationError("missing or unsafe evidence") from exc
    finally:
        if directory is not None:
            os.close(directory)
    if len(raw) > MAX_BYTES:
        raise ValidationError("evidence too large")
    return raw


def _persist(path, raw):
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValidationError("bounded evidence bytes required")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValidationError("symlink evidence path")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        if _read(path) != raw:
            raise ValidationError("immutable evidence conflict")
        return
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def artifact_bytes(plan, ref):
    if type(ref) is not ArtifactRef:
        raise ValidationError("immutable artifact ref required")
    _name(ref.name)
    digest_text(ref.sha256)
    integer(ref.size_bytes, maximum=MAX_BYTES)
    raw = _read(_root(plan.evidence_root) / "blobs" / ref.sha256)
    if sha(raw) != ref.sha256 or len(raw) != ref.size_bytes:
        raise ValidationError("artifact byte preimage mismatch")
    return raw


def _refs(spec):
    if type(spec.artifacts) is not tuple or any(type(r) is not ArtifactRef for r in spec.artifacts):
        raise ValidationError("immutable artifact census required")
    refs = {r.name: r for r in spec.artifacts}
    if len(refs) != len(spec.artifacts):
        raise ValidationError("duplicate artifact name")
    required = {"assignment", "task", "prompt", "harness", "protocol", "settings", "native_package"}
    if not required <= refs.keys():
        raise ValidationError("required artifact preimages missing")
    return refs


def _endpoint(snapshot):
    return next(e for e in parse_json(snapshot._sources[1])["data"]["endpoints"] if e["tag"] == snapshot.route.endpoint_tag)


def _body(fields, snapshot):
    if type(fields) is not RequestFields or type(fields.messages) is not tuple or not fields.messages:
        raise ValidationError("finite immutable request fields required")
    messages = []
    for pair in fields.messages:
        if type(pair) is not tuple or len(pair) != 2 or pair[0] not in {"system", "user", "assistant"} or type(pair[1]) is not str or not pair[1]:
            raise ValidationError("text message required")
        messages.append(dict(role=pair[0], content=pair[1]))
    integer(fields.max_tokens, 1)
    supported = _endpoint(snapshot)["supported_parameters"]
    limit = "max_tokens" if "max_tokens" in supported else "max_completion_tokens" if "max_completion_tokens" in supported else None
    if limit is None:
        raise ValidationError("unsupported output limit")
    route = snapshot.route
    body = dict(model=route.canonical_model_slug, messages=messages, stream=False,
                provider=dict(only=[route.endpoint_tag], order=[route.endpoint_tag], allow_fallbacks=False,
                              require_parameters=True), **{limit: fields.max_tokens})
    body["service_tier"] = route.service_tier
    if fields.reasoning_effort is not None and fields.reasoning_max_tokens is not None:
        raise ValidationError("one reasoning setting allowed")
    if fields.reasoning_effort is not None or fields.reasoning_max_tokens is not None:
        if "reasoning" not in supported:
            raise ValidationError("unsupported reasoning")
        if fields.reasoning_effort is not None:
            if fields.reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise ValidationError("invalid reasoning effort")
            body["reasoning"] = dict(effort=fields.reasoning_effort)
        else:
            integer(fields.reasoning_max_tokens, 1, fields.max_tokens - 1)
            body["reasoning"] = dict(max_tokens=fields.reasoning_max_tokens)
    if fields.temperature is not None:
        if "temperature" not in supported or not 0 <= money(fields.temperature) <= 2:
            raise ValidationError("unsupported temperature")
        body["temperature"] = float(fields.temperature)  # model setting, never money
    if fields.seed is not None:
        if "seed" not in supported:
            raise ValidationError("unsupported seed")
        body["seed"] = integer(fields.seed)
    return json_bytes(body)


def _census(spec, snapshot, fields):
    if snapshot.qualification != "offline_fixture":
        raise ValidationError("real route remains unqualified")
    profile = parse_json(snapshot._sources[2])
    rates = {r.name: r.usd_per_unit for r in snapshot.rate_components}
    if fields.max_tokens > profile["output_bound"]:
        raise ValidationError("request exceeds supported synthetic output bound")
    units = dict.fromkeys(USAGE_KINDS, 0)
    units.update(input_tokens=profile["input_bound"], output_tokens=profile["output_bound"],
                 reasoning_tokens=profile["reasoning_bound"], cache_read_tokens=profile["cache_read_bound"],
                 cache_write_tokens=profile["cache_write_bound"])
    activity = {"critic": "critic_calls", "retrieval": "retrieval_calls", "selection": "selection_calls", "research_evaluation": "grading_calls"}.get(spec.activity_role)
    if activity:
        units[activity] = 1
    units["retry_calls"] = int(spec.retry_index > 0)
    prices = dict.fromkeys(USAGE_KINDS, "0")
    prices.update(input_tokens=rates["prompt"], output_tokens=rates["completion"],
                  cache_read_tokens=rates.get("input_cache_read", "0"),
                  cache_write_tokens=str(max(Decimal(rates.get("input_cache_write", "0")), Decimal(rates.get("input_cache_write_1h", "0")))))
    return tuple(UsageClassPlan(k, units[k], prices[k], snapshot.source_refs[-1],
                               "SYNTHETIC ONLY: inactive in this local single text request" if units[k] == 0 else None)
                 for k in USAGE_KINDS)


def manifest_bytes(plan):
    return json_bytes(dict(version="w3b1-offline-v1", spec=asdict(plan.spec), route=asdict(plan.route),
                           snapshot=snapshot_identity(plan.snapshot), method="POST", url=plan.route.base_url + plan.route.path,
                           headers={"content-type": "application/json"}, body_sha256=sha(plan.body_bytes),
                           prompt=asdict(plan.prompt_ref), census=[asdict(c) for c in plan.census],
                           deadline_seconds=plan.deadline_seconds))


def request_hash(plan):
    return sha(manifest_bytes(plan))


def _directory(plan):
    return _root(plan.evidence_root) / "attempts" / _name(plan.spec.attempt_id)


def quote_for(plan):
    return Quote(plan.route.provider + "/" + plan.route.endpoint_tag, plan.route.canonical_model_slug,
                 sha(json_bytes(snapshot_identity(plan.snapshot))), plan.snapshot.observed_at, plan.snapshot.expires_at,
                 request_hash(plan), "USD", tuple(UsageBound(c.kind, c.max_units, c.usd_per_unit) for c in plan.census))


def _validate_spec(spec, snapshot, fields, artifacts):
    if type(spec) is not AttemptSpec:
        raise ValidationError("immutable attempt specification required")
    for k, value in asdict(spec).items():
        if k not in {"sample", "retry_index", "max_attempts", "artifacts", "parent_attempt_id"}:
            text(value)
    _name(spec.attempt_id)
    integer(spec.sample)
    integer(spec.max_attempts, 1, 3)
    integer(spec.retry_index, 0, spec.max_attempts - 1)
    if (spec.retry_index == 0) != (spec.parent_attempt_id is None):
        raise ValidationError("retry lineage required")
    if spec.parent_attempt_id is not None:
        _name(spec.parent_attempt_id)
        if spec.parent_attempt_id == spec.attempt_id:
            raise ValidationError("self retry")
    if spec.envelope not in {"repair", "data_judge", "pilot", "main"} or ROLES.get(spec.activity_role) != spec.cost_role:
        raise ValidationError("invalid envelope/activity/cost role")
    refs = _refs(spec)
    if set(artifacts) != set(refs):
        raise ValidationError("artifact census mismatch")
    for name, ref in refs.items():
        _name(name)
        if ArtifactRef.of(name, artifacts[name]) != ref:
            raise ValidationError("artifact preimage mismatch")
    for name, bound in (("task", spec.task_sha256), ("harness", spec.harness_sha256), ("protocol", spec.protocol_sha256), ("settings", spec.settings_sha256), ("native_package", spec.native_package_sha256)):
        if digest_text(bound) != refs[name].sha256:
            raise ValidationError("artifact identity drift")
    parse_json(artifacts["assignment"])  # Strict duplicate/nonfinite/UTF-8 validation.
    # W1's immutable, nonmonetary assignment digest uses ordinary JSON numbers.
    # Monetary provider response parsing remains Decimal-only below.
    assignment = json.loads(artifacts["assignment"])
    # Existing W1 owns fixed assignment validity, even before there are records.
    project_report(dict(protocol=assignment.get("protocol"), assignment_hash=digest(assignment), generations=[], evaluations=[]), assignment)
    if digest(assignment) != spec.assignment_hash:
        raise ValidationError("assignment hash drift")
    matches = [a for a in assignment["assignments"] if (a["task_id"], a["sample"], a["condition"], a["model_snapshot"]) == (spec.task_id, spec.sample, spec.condition, spec.assigned_model_snapshot)]
    if len(matches) != 1 or matches[0]["story_id"] != spec.story_id or json_bytes(matches[0]["task"]) != artifacts["task"]:
        raise ValidationError("assigned task/condition/model drift")
    route_snapshot = matches[0]["judge_snapshot"] if spec.cost_role == "research" else spec.assigned_model_snapshot
    if route_snapshot != snapshot.route.canonical_model_slug:
        raise ValidationError("assigned model differs from route")
    if artifacts["settings"] != json_bytes(asdict(fields)) or artifacts["prompt"] != "\n\n".join(p[1] for p in fields.messages).encode():
        raise ValidationError("prompt/settings drift")
    expected_effort = fields.reasoning_effort or (f"tokens:{fields.reasoning_max_tokens}" if fields.reasoning_max_tokens is not None else "unspecified")
    if spec.effort != expected_effort:
        raise ValidationError("effort drift")


def prepare_attempt(spec, snapshot, artifact_bytes, request_fields, evidence_root):
    verify_snapshot(snapshot)
    if type(artifact_bytes) is not tuple or any(type(p) is not tuple or len(p) != 2 for p in artifact_bytes):
        raise ValidationError("immutable artifact bytes tuple required")
    artifacts = dict(artifact_bytes)
    if len(artifacts) != len(artifact_bytes):
        raise ValidationError("duplicate artifact bytes")
    body = _body(request_fields, snapshot)
    _validate_spec(spec, snapshot, request_fields, artifacts)
    plan = RequestPlan(spec, snapshot.route, snapshot, body, _refs(spec)["prompt"],
                       _census(spec, snapshot, request_fields), 1, _root(evidence_root), request_fields)
    for raw in (*artifacts.values(), *snapshot._sources, body):
        _persist(plan.evidence_root / "blobs" / sha(raw), raw)
    _persist(_directory(plan) / "manifest.json", manifest_bytes(plan))
    return plan


def verify_plan(plan, *, current=True):
    if type(plan) is not RequestPlan or type(plan.census) is not tuple:
        raise ValidationError("immutable request plan required")
    integer(plan.deadline_seconds, 1, 120)
    verify_snapshot(plan.snapshot, require_current=current)
    if plan.route != plan.snapshot.route or _body(plan._fields, plan.snapshot) != plan.body_bytes:
        raise ValidationError("route/request drift")
    if plan.census != _census(plan.spec, plan.snapshot, plan._fields):
        raise ValidationError("usage census drift")
    artifacts = {r.name: artifact_bytes(plan, r) for r in plan.spec.artifacts}
    _validate_spec(plan.spec, plan.snapshot, plan._fields, artifacts)
    for ref, raw in zip(plan.snapshot.source_refs, plan.snapshot._sources):
        if artifact_bytes(plan, ref) != raw:
            raise ValidationError("source bytes drift")
    if artifact_bytes(plan, ArtifactRef.of("request_body", plan.body_bytes)) != plan.body_bytes or _read(_directory(plan) / "manifest.json") != manifest_bytes(plan):
        raise ValidationError("immutable request manifest drift")
    return plan


def _admit(plan, admission):
    if type(admission) is not Admission or admission.mode != "offline_fixture":
        raise ValidationError("paid dispatch disabled; synthetic local admission only")
    if admission.assignment_hash != plan.spec.assignment_hash or admission.route_sha256 != sha(json_bytes(asdict(plan.route))):
        raise ValidationError("admission identity mismatch")
    if utc(admission.valid_until) <= datetime.now(timezone.utc):
        raise ValidationError("admission expired")
    if type(admission.authority_refs) is not tuple:
        raise ValidationError("gate tuple required")
    kinds = []
    for ref in admission.authority_refs:
        if ref not in plan.spec.artifacts:
            raise ValidationError("unbound gate evidence")
        gate = parse_json(artifact_bytes(plan, ref))
        exact_fields(gate, {"kind", "authority", "assignment_hash", "route_sha256", "valid_until"})
        if gate["authority"] != "SYNTHETIC OFFLINE ONLY" or gate["assignment_hash"] != admission.assignment_hash or gate["route_sha256"] != admission.route_sha256 or utc(gate["valid_until"]) < utc(admission.valid_until):
            raise ValidationError("invalid synthetic gate binding")
        kinds.append(gate["kind"])
    if len(kinds) != len(GATES) or set(kinds) != GATES:
        raise ValidationError("missing or duplicate rights/phase/billing gate")


def _fixture(fixture):
    if type(fixture) is not OfflineFixture:
        raise ValidationError("only finite OfflineFixture data; no callback or transport")
    integer(fixture.status_code, 100, 599)
    integer(fixture.header_delay_ms, 0, 120000)
    if type(fixture.chunks) is not tuple or len(fixture.chunks) > 256:
        raise ValidationError("finite chunks required")
    size = 0
    for pair in fixture.chunks:
        if type(pair) is not tuple or len(pair) != 2 or type(pair[1]) is not bytes:
            raise ValidationError("immutable byte chunk required")
        integer(pair[0], 0, 120000)
        size += len(pair[1])
    integer(size, maximum=MAX_BYTES)


def _ledger_identity(plan, ledger):
    if type(ledger) is not Ledger or ledger.program_id != plan.spec.program_id or ledger.store_id != plan.spec.store_id:
        raise ValidationError("program/store identity mismatch")


def _result(plan, state, status, replayed=False, response_ref=None, receipt_ref=None, output_ref=None):
    return AttemptResult(plan.spec.attempt_id, request_hash(plan), state, response_ref, receipt_ref, output_ref, status, replayed)


class LedgerStateUnavailable(LedgerError):
    """Permission remains consumed, but final accounting cannot currently be read."""


def _unknown(plan, ledger, reason):
    try:
        row = ledger.mark_unknown(f"w3b:{plan.spec.attempt_id}:unknown", attempt_id=plan.spec.attempt_id, reason=reason)
        if row["replayed"] is False:
            return _result(plan, row["state"], reason)
    except LedgerError:
        pass
    # Failed/replayed cleanup is not authority to invent a lifecycle state.
    try:
        current = ledger.attempt(plan.spec.attempt_id)
    except LedgerError as exc:
        raise LedgerStateUnavailable("final ledger state unavailable; never resend or refund") from exc
    status = current["receipt"]["outcome"] if current["state"] == "RECONCILED" else reason
    return _result(plan, current["state"], status)


def _reservation_event(plan, ledger):
    if plan.spec.retry_index == 0:
        return f"w3b:{plan.spec.attempt_id}:reserve"
    # One transactional audit snapshot validates both ancestors (maximum depth 2).
    # These are actual ledger records, never caller-supplied lineage labels.
    rows = {}
    for event in ledger.events():
        if event["kind"] in {"reserved", "sent", "unknown", "reconciled", "cancelled_unsent"}:
            row = event["result"]
            rows[row["attempt_id"]] = row
    parent_id = plan.spec.parent_attempt_id
    fixed = ("assignment_hash", "task_id", "task_sha256", "sample", "condition", "assigned_model_snapshot",
             "story_id", "episode_id", "run_id", "max_attempts", "cost_role", "activity_role",
             "program_id", "store_id", "envelope", "harness_sha256", "protocol_sha256", "native_package_sha256")
    for index in range(plan.spec.retry_index - 1, -1, -1):
        _name(parent_id)
        raw = _read(plan.evidence_root / "attempts" / parent_id / "manifest.json")
        manifest = parse_json(raw)
        try:
            previous, row = manifest["spec"], rows[parent_id]
            if row["state"] not in {"SENT", "UNKNOWN", "RECONCILED"} or row["quote"]["request_sha256"] != sha(raw):
                raise ValidationError("unbound retry ancestor")
            route = manifest["route"]
            quote = dict(provider=route["provider"] + "/" + route["endpoint_tag"], model=route["canonical_model_slug"],
                snapshot_sha256=sha(json_bytes(manifest["snapshot"])), observed_at=manifest["snapshot"]["observed_at"],
                expires_at=manifest["snapshot"]["expires_at"], request_sha256=sha(raw), currency="USD",
                lines=[dict(kind=c["kind"],max_units=c["max_units"],usd_per_unit=c["usd_per_unit"]) for c in manifest["census"]])
            if row["quote"] != quote or row["role"] != previous["cost_role"] or row["envelope"] != previous["envelope"]:
                raise ValidationError("retry ancestor quote drift")
            if previous["attempt_id"] != parent_id or previous["retry_index"] != index or any(previous[k] != getattr(plan.spec,k) for k in fixed):
                raise ValidationError("retry ancestor identity drift")
            parent_id = previous["parent_attempt_id"]
            if (index == 0) != (parent_id is None):
                raise ValidationError("invalid retry root")
        except (KeyError, TypeError) as exc:
            raise ValidationError("invalid retry ancestor manifest") from exc
    # The global unique event and immutable payload atomically occupy this slot,
    # including after completion/cancellation. Conflicting siblings cannot reserve.
    return "w3b:retry-slot:" + sha(json_bytes(dict(root_request_sha256=sha(raw), retry_index=plan.spec.retry_index)))


async def execute_attempt(plan, ledger, admission, fixture):
    _fixture(fixture)
    verify_plan(plan)
    _ledger_identity(plan, ledger)
    _admit(plan, admission)
    reservation_event = _reservation_event(plan, ledger)
    quote = quote_for(plan)
    reserved = ledger.reserve(reservation_event, attempt_id=plan.spec.attempt_id,
                   envelope=plan.spec.envelope, role=plan.spec.cost_role, quote=quote)
    # Retry ancestry already consumed one ledger call. The atomic reserve result
    # binds its quote; a fresh mark_sent still exclusively grants permission.
    current = reserved if plan.spec.retry_index else ledger.attempt(plan.spec.attempt_id)
    if current["quote"] != parse_json(json_bytes(asdict(quote))) or current["role"] != plan.spec.cost_role or current["envelope"] != plan.spec.envelope:
        raise ValidationError("current ledger identity mismatch")
    if current["state"] != "RESERVED":
        return _result(plan, current["state"], "replay_no_send", True)
    # Cancellation before permission may leave RESERVED but can never send.
    await asyncio.sleep(0)
    verify_plan(plan)
    _admit(plan, admission)
    marker = ledger.mark_sent(f"w3b:{plan.spec.attempt_id}:send", attempt_id=plan.spec.attempt_id)
    if marker["replayed"] is not False or marker["state"] != "SENT":
        current = ledger.attempt(plan.spec.attempt_id)
        return _result(plan, current["state"], "replay_no_send", True)
    deadline = asyncio.get_running_loop().time() + plan.deadline_seconds
    response = None

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for delay, chunk in fixture.chunks:
                await asyncio.sleep(delay / 1000)
                yield chunk

        async def aclose(self):
            return None

    async def handler(request):
        _persist(_directory(plan) / "transport.json", json_bytes(dict(
            method=request.method, url=str(request.url), body_hex=request.content.hex(),
            body_sha256=sha(request.content), marker=marker, calls=1)))
        await asyncio.sleep(fixture.header_delay_ms / 1000)
        return httpx.Response(fixture.status_code, stream=Stream())

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False) as client:
            try:
                async with asyncio.timeout_at(deadline):
                    request = client.build_request("POST", plan.route.base_url + plan.route.path,
                                                   content=plan.body_bytes, headers={"content-type": "application/json"})
                    response = await client.send(request, stream=True)
                    raw = await response.aread()
            finally:
                if response is not None:
                    await response.aclose()
        _persist(_directory(plan) / "response.bin", raw)
        _persist(_directory(plan) / "response.json", json_bytes(dict(status=response.status_code, sha256=sha(raw), request_sha256=request_hash(plan))))
        return reconcile_attempt(plan, ledger, raw, response.status_code)
    except asyncio.CancelledError:
        try:
            _unknown(plan, ledger, "cancelled")
        except LedgerStateUnavailable:
            pass  # Cancellation propagates; no state or refund is manufactured.
        raise
    except TimeoutError:
        return _unknown(plan, ledger, "timeout")
    except LedgerStateUnavailable:
        raise
    except (ValueError, OSError, httpx.HTTPError, LedgerError):
        return _unknown(plan, ledger, "infrastructure_failure")


def reconcile_attempt(plan, ledger, response_bytes, response_status):
    verify_plan(plan, current=False)
    _ledger_identity(plan, ledger)
    raw = _read(_directory(plan) / "response.bin")
    metadata = parse_json(_read(_directory(plan) / "response.json"))
    if raw != response_bytes or metadata != dict(status=response_status, sha256=sha(raw), request_sha256=request_hash(plan)):
        raise ValidationError("untrusted response provenance")
    data = parse_json(raw)
    if type(data) is not dict:
        raise ValidationError("invalid provider response")
    allowed = {"id", "model", "provider", "service_tier", "choices", "usage", "error", "total_cost"}
    if set(data) - allowed:
        raise ValidationError("unsupported response field")
    for key in ("id", "model", "provider", "service_tier"):
        text(data.get(key))
    if data.get("model") not in {plan.route.model_id, plan.route.canonical_model_slug} or data.get("provider") != _endpoint(plan.snapshot)["provider_name"] or data.get("service_tier") != plan.route.service_tier:
        raise ValidationError("provider/model/tier mismatch")
    generation_id = text(data.get("id"))
    usage = data.get("usage")
    if type(usage) is not dict:
        return _unknown(plan, ledger, "missing_cost")
    if set(usage) - {"cost", "is_byok", "cost_details", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}:
        raise ValidationError("unsupported usage component")
    if usage.get("is_byok") is not False:
        raise ValidationError("BYOK or unknown account mode unsupported")
    if usage.get("cost_details") is not None and type(usage["cost_details"]) is not dict:
        raise ValidationError("invalid cost details")
    amount = usage.get("cost")
    if amount is None:
        return _unknown(plan, ledger, "missing_cost")
    if type(amount) not in (int, Decimal):
        raise ValidationError("original monetary JSON number required")
    actual = str(amount)
    money(actual)
    if "total_cost" in data and (type(data["total_cost"]) not in (int, Decimal) or data["total_cost"] != amount):
        raise ValidationError("conflicting cumulative receipt")
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    for details, keys in ((prompt_details, {"cached_tokens", "cache_write_tokens"}), (completion_details, {"reasoning_tokens"})):
        if details is not None and (type(details) is not dict or set(details) - keys):
            raise ValidationError("unsupported token detail")
    counters = dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=(completion_details or {}).get("reasoning_tokens"),
                    cache_read_tokens=(prompt_details or {}).get("cached_tokens"),
                    cache_write_tokens=(prompt_details or {}).get("cache_write_tokens"))
    zeros = parse_json(plan.snapshot._sources[2])["structural_zero"]
    for key in counters:
        if counters[key] is None and key in zeros:
            counters[key] = 0
        if counters[key] is not None:
            integer(counters[key])
    if counters["input_tokens"] is not None and counters["output_tokens"] is not None and "total_tokens" in usage:
        integer(usage["total_tokens"])
        if usage["total_tokens"] != counters["input_tokens"] + counters["output_tokens"]:
            raise ValidationError("inconsistent total tokens")
    for subset, whole in (("reasoning_tokens", "output_tokens"), ("cache_read_tokens", "input_tokens"), ("cache_write_tokens", "input_tokens")):
        if counters[subset] is not None and counters[whole] is not None and counters[subset] > counters[whole]:
            raise ValidationError("impossible token subset")
    for line in plan.census:
        if line.kind not in counters:
            counters[line.kind] = line.max_units
    content = ""
    if response_status >= 400 or data.get("error") is not None:
        outcome = "provider_failure"
    else:
        choices = data.get("choices")
        if type(choices) is not list or len(choices) != 1:
            raise ValidationError("one complete choice required")
        exact_fields(choices[0], {"finish_reason", "message"})
        message = choices[0]["message"]
        if type(message) is not dict or set(message) - {"content", "refusal", "role"}:
            raise ValidationError("invalid response message")
        for key in ("role", "refusal"):
            if message.get(key) is not None and type(message[key]) is not str:
                raise ValidationError("invalid response message scalar")
        content = message.get("content")
        if content is None:
            content = ""
        if type(content) is not str:
            raise ValidationError("text output required")
        finish = choices[0]["finish_reason"]
        text(finish)
        if finish == "length":
            outcome = "truncated"
        elif finish == "content_filter" or message.get("refusal"):
            outcome = "refused"
        elif finish == "stop":
            outcome = "completed" if content.strip() else "absent_output"
        else:
            raise ValidationError("unknown or incomplete finish reason")
    quote = quote_for(plan)
    response_ref = ArtifactRef.of("response", raw)
    output_ref = ArtifactRef.of("output", content.encode())
    _persist(plan.evidence_root / "blobs" / output_ref.sha256, content.encode())
    receipt = Receipt("openrouter:" + generation_id, quote.provider, quote.model, quote.snapshot_sha256,
                      quote.request_sha256, "USD", actual, tuple((k, counters[k]) for k in USAGE_KINDS), outcome,
                      sha(raw), datetime.now(timezone.utc).isoformat())
    receipt_path = _directory(plan) / "receipt.json"
    if receipt_path.exists():
        prior = parse_json(_read(receipt_path))
        expected = parse_json(json_bytes(asdict(receipt)))
        if {k: v for k, v in prior.items() if k != "recorded_at"} != {k: v for k, v in expected.items() if k != "recorded_at"}:
            raise ValidationError("stored receipt differs from raw response")
        receipt = Receipt(**{**prior, "usage": tuple(tuple(p) for p in prior["usage"])})
        if receipt.evidence_sha256 != sha(raw):
            raise ValidationError("conflicting response receipt")
    else:
        _persist(receipt_path, json_bytes(asdict(receipt)))
    # All filesystem access and result construction precede the financial commit.
    prepared = _result(plan, "RECONCILED", outcome, False, response_ref,
                       ArtifactRef.of("receipt", _read(receipt_path)), output_ref)
    row = ledger.reconcile(f"w3b:{plan.spec.attempt_id}:reconcile", attempt_id=plan.spec.attempt_id, receipt=receipt)
    return replace(prepared, ledger_state=row["state"], replayed=row["replayed"])
