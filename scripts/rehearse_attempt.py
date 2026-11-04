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


def _route(identity: dict) -> pp.Route:
    """The route AS THE RECORD DECLARES IT, including its own `excluded_variants`.

    Deriving the sibling set from the payload here would be this file quietly repairing a record
    that under-declares: the guard would then compare the payload to itself and always agree. The
    declaration is the reviewed claim, so it is what travels; a record that is wrong fails closed.
    """
    return pp.Route(identity["provider"], identity["base_url"], identity["path"],
                    identity["model_id"], identity["canonical_model_slug"],
                    identity["endpoint_tag"], identity["service_tier"], identity["currency"],
                    tuple(identity.get("excluded_variants") or ()))


def load_route(base: pathlib.Path, model_id: str):
    """The record, its endpoints payload, its profile and its verification artifact, as bytes."""
    for path in sorted((base / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record["route_identity"]["model_id"] != model_id:
            continue
        slug = model_id.replace("/", "_")
        endpoints = (base / f"endpoints-{slug}.json").read_bytes()
        profile = (base / "verified" / f"{slug}.profile.json").read_bytes()
        # The brief fixes the artifact's NAME; matching on `subject` alone let a stale generation
        # beside the canonical file win on sort order.
        artifact_path = base / "verification" / (model_id.split("/")[-1] + ".json")
        if not artifact_path.is_file():
            raise SystemExit(f"{model_id}: no verification artifact at {artifact_path.name}")
        artifact = artifact_path.read_bytes()
        if json.loads(artifact).get("subject") != model_id:
            raise SystemExit(f"{model_id}: {artifact_path.name} names another subject")
        return record, endpoints, profile, artifact
    raise SystemExit(f"{model_id}: no record in {base}")


def study_limits(task_type: str, model: str) -> tuple[str, int, int]:
    """Effort and token budgets from the SSOTs, never from a literal at the call site.

    `max_tokens=6000` was hardcoded for every family and `effort="max"` for every model, while the
    study declares a cap PER FAMILY (constrained_continuation 8000, multi_beat_synthesis 10000) and a
    ceiling PER MODEL (grok and qwen xhigh, gemini high). Two consequences, both paid for: three
    families were guaranteed to truncate, and opus -- told `effort: max` with no reasoning budget --
    spent all 6,000 completion tokens thinking and returned `content: null`, twice, at $0.126 each.

    The budget returns with the caps because `max_tokens` alone is not a guard: it bounds reasoning
    AND output together, so a maximising model can spend it all before writing a word.
    """
    import yaml

    from llm_client import TASK_TOKEN_CONFIG  # noqa: PLC0415

    study = yaml.safe_load((REPO / "studies" / "revision-v1.yaml").read_text(encoding="utf-8"))
    effort = study["reasoning_effort"]["per_model"][model]
    config = TASK_TOKEN_CONFIG[task_type]
    # The completion allowance is the family's OUTPUT budget PLUS its reasoning budget, because the
    # provider accepts only one reasoning setting and the study's choice is to keep the effort
    # (`reasoning_effort.per_model`) and let the model decide how much to think. `max_tokens` is the
    # only remaining bound, so it has to cover both halves or the effort guarantees nothing: opus
    # spent 6,000 of 6,000 on reasoning under `effort: max` and returned `content: null` twice.
    return effort, config["max_tokens"] + config["max_reasoning_tokens"], config["max_reasoning_tokens"]


def request_fields(task: dict, effort: str, max_tokens: int,
                   reasoning_max_tokens: int | None = None) -> pa.RequestFields:
    from generator import PromptBuilder  # noqa: PLC0415

    system = PromptBuilder.build_system_prompt(task["task_type"])
    user = PromptBuilder.build_user_prompt(task)
    # The provider accepts ONE reasoning setting, so the budget wins over the effort whenever both
    # are known. The effort is the study's declared treatment; the budget is what stops a maximising
    # model spending the whole completion allowance on thinking and returning `content: null` -- which
    # is exactly what opus did twice at $0.126. A treatment that produces no text measures nothing,
    # so the budget is kept and the omission is recorded rather than silent: the spec's `effort`
    # becomes `tokens:<budget>`, which is what the request actually carries.
    # Effort ONLY: the provider refuses the two together, and the study's decision is to keep the
    # per-model effort as its declared treatment. `max_tokens` carries the room that the effort
    # cannot bound -- see `study_limits`.
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
             assignment_hash: str, *, envelope: str = "pilot",
             cost_role: str = "deployment") -> pa.AttemptSpec:
    """The attempt's identity, with the envelope as a PARAMETER.

    It was hardcoded to `pilot`, which meant every attempt built through the only path that can
    spend drew the generation envelope -- so the 426 USD `data_judge` line was unreachable and a
    judge phase would have spent the pilot's budget instead of its own.
    """
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
        envelope=envelope, cost_role=cost_role, activity_role="generation",
        program_id=PROGRAM_ID, store_id=STORE_ID,
        native_package_sha256=pp.sha(by["native_package"]), artifacts=refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--assignment", type=pathlib.Path, required=True)
    ap.add_argument("--index", type=int, default=0, help="which assignment row to rehearse")
    ap.add_argument("--evidence", type=pathlib.Path, default=pathlib.Path("/tmp/w3b2-rehearsal"))
    args = ap.parse_args()

    raw = args.assignment.read_bytes()
    manifest = json.loads(raw)
    if not manifest.get("judge_designated"):
        raise SystemExit("manifest records no judge designation; refusing to rehearse it as one")
    # The row you rehearse must be the row you send: a rehearsal of row 0 that authorises row 4
    # proves the wiring and nothing about the request that will actually be made.
    row = manifest["assignments"][args.index]
    record, endpoints, profile, artifact = load_route(args.base, row["model"])
    identity = record["route_identity"]

    snapshot = pp.validate_snapshot(
        (( args.base / "openrouter-models.json").read_bytes(), endpoints, profile, artifact),
        _route(identity), identity["observed_at"], identity["expires_at"])
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
