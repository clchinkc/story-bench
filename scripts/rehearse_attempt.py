#!/usr/bin/env python3
"""Rehearse ONE attempt: build it with the native functions, reserve nothing, send nothing.

WHY THIS FILE EXISTS. The plan called for "wiring an adapter". Reading `prepare_attempt` showed the
adapter was never a layer -- it was four inputs that did not exist, the assignment manifest being the
largest. With the manifest produced (`assignment_builder.py`), the remaining work is not an
abstraction but a call: build the seven artifact preimages, the AttemptSpec, the RequestFields and a
Snapshot from the retained base, then hand them to `prepare_attempt`.

A rehearsal is the only honest pre-flight before money: `prepare_attempt` persists the evidence
blobs and the manifest and validates every binding, while `execute_attempt` -- which reserves the
ledger and opens the transport -- is never called. If this passes, the next failure would be a
transport failure, not a wiring failure.

It refuses to rehearse a manifest whose judge is unassigned: the schema allows the absence, the run
should not pretend it is a designation.

    uv run python scripts/rehearse_attempt.py --base <route-evidence-base> --assignment /tmp/a.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import provider_attempts as pa  # noqa: E402
import provider_prices as pp  # noqa: E402
from measurement_contract import digest  # noqa: E402

import dispatch_runner as dr  # noqa: E402

PROGRAM_ID = "story-bench"
STORE_ID = "revision-v1"


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _siblings(tag: str, endpoints: bytes) -> tuple[str, ...]:
    live = [e["tag"] for e in json.loads(endpoints)["data"]["endpoints"]]
    return tuple(sorted(t for t in live if t.startswith(tag + "/")))


def _route(identity: dict, endpoints: bytes) -> pp.Route:
    return pp.Route(identity["provider"], identity["base_url"], identity["path"],
                    identity["model_id"], identity["canonical_model_slug"],
                    identity["endpoint_tag"], identity["service_tier"], identity["currency"],
                    _siblings(identity["endpoint_tag"], endpoints))


def load_route(base: pathlib.Path, model_id: str):
    """The record, its endpoints payload, its profile and its verification artifact, as bytes."""
    for path in sorted((base / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record["route_identity"]["model_id"] != model_id:
            continue
        slug = model_id.replace("/", "_")
        endpoints = (base / f"endpoints-{slug}.json").read_bytes()
        profile = (base / "verified" / f"{slug}.profile.json").read_bytes()
        artifact = next((p.read_bytes() for p in sorted((base / "verification").glob("*.json"))
                         if json.loads(p.read_text()).get("subject") == model_id), None)
        if artifact is None:
            raise SystemExit(f"{model_id}: no verification artifact")
        return record, endpoints, profile, artifact
    raise SystemExit(f"{model_id}: no record in {base}")


def request_fields(task: dict, effort: str, max_tokens: int) -> pa.RequestFields:
    from generator import PromptBuilder  # noqa: PLC0415

    system = PromptBuilder.build_system_prompt(task["task_type"])
    user = PromptBuilder.build_user_prompt(task)
    return pa.RequestFields(messages=(("system", system), ("user", user)),
                            max_tokens=max_tokens, reasoning_effort=effort)


def artifacts_for(row: dict, fields, assignment: bytes, base: pathlib.Path) -> dict[str, bytes]:
    """The seven preimages. Three of them are run-level files, hashed exactly as the manifest says."""
    task = json_bytes(row["task"])
    prompt = "\n\n".join(text for _, text in fields.messages).encode()
    return {
        "task": task,
        "prompt": prompt,
        "settings": json_bytes(pa.asdict(fields)),
        "assignment": assignment,
        "harness": (REPO / "scripts" / "dispatch_runner.py").read_bytes(),
        "protocol": (REPO / "studies" / "revision-v1.yaml").read_bytes(),
        "native_package": b"".join(p.read_bytes() for p in
                                   sorted((REPO / "dataset" / "tasks").rglob("*.yaml"))),
    }


def spec_for(row: dict, attempt: dict, effort: str, artifacts: dict[str, bytes],
             assignment_hash: str) -> pa.AttemptSpec:
    refs = tuple(pa.ArtifactRef.of(name, raw) for name, raw in sorted(artifacts.items()))
    by = dict(artifacts)
    return pa.AttemptSpec(
        attempt_id=attempt["attempt_id"], run_id=f"pilot-{attempt['unit_id']}",
        episode_id=attempt["unit_id"], assignment_hash=assignment_hash,
        task_id=row["task_id"], task_sha256=digest(row["task"]), story_id=row["story_id"],
        condition=row["condition"], sample=row["sample"],
        assigned_model_snapshot=row["model_snapshot"],
        harness_sha256=pp.sha(by["harness"]), protocol_sha256=pp.sha(by["protocol"]),
        settings_sha256=pp.sha(by["settings"]), effort=effort,
        parent_attempt_id=None, retry_index=0, max_attempts=1,
        envelope="pilot", cost_role="deployment", activity_role="generation",
        program_id=PROGRAM_ID, store_id=STORE_ID,
        native_package_sha256=pp.sha(by["native_package"]), artifacts=refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--assignment", type=pathlib.Path, required=True)
    ap.add_argument("--evidence", type=pathlib.Path, default=pathlib.Path("/tmp/w3b2-rehearsal"))
    args = ap.parse_args()

    raw = args.assignment.read_bytes()
    manifest = json.loads(raw)
    if not manifest.get("judge_designated"):
        raise SystemExit("manifest records no judge designation; refusing to rehearse it as one")
    row = manifest["assignments"][0]
    record, endpoints, profile, artifact = load_route(args.base, row["model"])
    identity = record["route_identity"]

    snapshot = pp.validate_snapshot(
        (( args.base / "openrouter-models.json").read_bytes(), endpoints, profile, artifact),
        _route(identity, endpoints), identity["observed_at"], identity["expires_at"])
    if snapshot.qualification != "c8_verified":
        raise SystemExit(f"{row['model']}: qualification {snapshot.qualification!r} is not c8_verified")

    effort = "max"
    max_tokens = 6000
    fields = request_fields(row["task"], effort, max_tokens)
    artifacts = artifacts_for(row, fields, raw, args.base)
    attempt = dict(attempt_id=dr.attempt_id(row["model"], "rehearsal-u0", row["condition"]),
                   unit_id="rehearsal-u0")
    spec = spec_for(row, attempt, effort, artifacts, digest(manifest))
    plan = pa.prepare_attempt(spec, snapshot, tuple(sorted(artifacts.items())), fields, args.evidence)

    print(f"  route        : {row['model']} ({snapshot.route.endpoint_tag})")
    print(f"  task         : {row['task_id']} cond={row['condition']} sample={row['sample']}")
    print(f"  body bytes   : {len(plan.body_bytes)}")
    print(f"  deadline     : {plan.deadline_seconds}s")
    print(f"  census       : {[(c.kind, c.max_units) for c in plan.census if c.max_units]}")
    print(f"  evidence     : {plan.evidence_root}")
    print("  RESERVED     : nothing; execute_attempt was not called")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
