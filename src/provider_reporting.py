"""Cost/attempt projection into the unchanged W1 authority, with exact-money sidecar."""
import copy
from dataclasses import asdict
import math

from measurement_contract import digest, identity, project_report
from provider_attempts import (_directory, _read, artifact_bytes, quote_for, request_hash,
                               verify_plan)
from provider_prices import json_bytes, parse_json, sha
from spend_ledger import ValidationError

STATUS = {"completed": "completed", "truncated": "truncated", "absent_output": "absent",
          "refused": "agent_failure", "provider_failure": "provider_failure", "timeout": "provider_failure",
          "infrastructure_failure": "infrastructure_failure", "cancelled": "infrastructure_failure",
          "task_failure": "agent_failure", "grader_failure": "grader_failure"}


def _assignment(plan):
    ref = next(r for r in plan.spec.artifacts if r.name == "assignment")
    manifest = parse_json(artifact_bytes(plan, ref))
    return next(a for a in manifest["assignments"] if (a["task_id"], a["condition"], a["sample"], a["model_snapshot"]) == (
        plan.spec.task_id, plan.spec.condition, plan.spec.sample, plan.spec.assigned_model_snapshot))


def project_attempts(plans, ledger):
    if type(plans) is not tuple or len({p.spec.attempt_id for p in plans}) != len(plans):
        raise ValidationError("unique immutable plan tuple required")
    result = []
    for plan in plans:
        verify_plan(plan, current=False)
        if (ledger.program_id, ledger.store_id) != (plan.spec.program_id, plan.spec.store_id):
            raise ValidationError("report ledger identity mismatch")
        row = ledger.attempt(plan.spec.attempt_id)
        if row["quote"] != parse_json(json_bytes(asdict(quote_for(plan)))) or row["role"] != plan.spec.cost_role or row["envelope"] != plan.spec.envelope:
            raise ValidationError("report attempt identity mismatch")
        if row["state"] in {"RESERVED", "CANCELLED"}:
            continue
        receipt = row["receipt"]
        if receipt is None:
            amount, usage, status = None, None, "infrastructure_failure"
        else:
            amount = float(receipt["actual_usd"])
            if not math.isfinite(amount):
                raise ValidationError("exact amount cannot be displayed by W1")
            counts = dict(receipt["usage"])
            keys = ("input_tokens", "output_tokens", "reasoning_tokens")
            complete = (*keys, "cache_read_tokens", "cache_write_tokens")
            usage = None if any(counts[k] is None for k in complete) else dict(zip(
                ("prompt_tokens", "completion_tokens", "reasoning_tokens"), (counts[k] for k in keys)))
            status = STATUS[receipt["outcome"]]
        result.append(dict(attempt_id=plan.spec.attempt_id, role=plan.spec.activity_role,
                           status=status, cost_usd=amount, usage=usage))
    return tuple(result)


def project_cost_report(manifest, records, plans, ledger):
    # Validate original grade/generation links BEFORE any copy, replacement or rehash.
    project_report(records, manifest)
    old_records_hash, manifest_hash = digest(records), digest(manifest)
    copied = copy.deepcopy(records)
    projected = {r["attempt_id"]: r for r in project_attempts(plans, ledger)}
    planned = {p.spec.attempt_id: p for p in plans}
    original_ids = [a["attempt_id"] for kind in ("generations", "evaluations") for r in records[kind] for a in r["attempts"]]
    if len(original_ids) != len(set(original_ids)) or set(original_ids) != set(projected):
        raise ValidationError("exact report/attempt census mismatch")
    # Do not hide a sibling qualified attempt merely by omitting its plan/record.
    for attempt_id in ledger.status()["attempt_ids"]:
        row = ledger.attempt(attempt_id)
        if row["quote"]["provider"].startswith("openrouter/") and row["state"] in {"SENT", "UNKNOWN", "RECONCILED"}:
            roots = {p.evidence_root for p in plans}
            bound = []
            for root in roots:
                path = root / "attempts" / attempt_id / "manifest.json"
                if path.exists():
                    bound.append(parse_json(_read(path)))
            if not bound or any(b["spec"]["assignment_hash"] == manifest_hash for b in bound) and attempt_id not in projected:
                raise ValidationError("orphan qualified ledger attempt")
    for kind in ("generations", "evaluations"):
        for original, changed in zip(records[kind], copied[kind]):
            for attempt in original["attempts"]:
                plan = planned[attempt["attempt_id"]]
                if plan.spec.assignment_hash != manifest_hash or identity(original) != identity(_assignment(plan)):
                    raise ValidationError("assigned report identity mismatch")
                if (kind == "evaluations") != (plan.spec.cost_role == "research"):
                    raise ValidationError("deployment/research mixing")
                if plan.spec.activity_role in {"generation", "research_evaluation"}:
                    if artifact_bytes(plan, plan.prompt_ref).decode() != original["prompt"]:
                        raise ValidationError("record prompt differs from sent bytes")
            changed["attempts"] = [copy.deepcopy(projected[a["attempt_id"]]) for a in original["attempts"]]
            if {k: v for k, v in changed.items() if k != "attempts"} != {k: v for k, v in original.items() if k != "attempts"}:
                raise ValidationError("projection changed immutable record fields")
            if kind == "generations" and original["status"] == "completed":
                matches = []
                for a in changed["attempts"]:
                    if a["role"] == "generation" and a["status"] == "completed":
                        plan = planned[a["attempt_id"]]
                        raw = _read(_directory(plan) / "response.bin")
                        receipt = ledger.attempt(a["attempt_id"])["receipt"]
                        if receipt["evidence_sha256"] != sha(raw):
                            raise ValidationError("response evidence changed")
                        matches.append(parse_json(raw)["choices"][0]["message"]["content"] == original["output"])
                if not any(matches):
                    raise ValidationError("completed output differs from immutable response")
    original_generations = {identity(g): g for g in records["generations"]}
    copied_generations = {identity(g): g for g in copied["generations"]}
    provenance = []
    for evaluation in copied["evaluations"]:
        key = identity(evaluation)
        old, new = original_generations[key], copied_generations[key]
        # Original project_report already checked this; retain the binding explicitly.
        if evaluation["generation_hash"] != digest(old):
            raise ValidationError("unvalidated original generation binding")
        provenance.append(dict(identity=list(key), old_generation_hash=digest(old), new_generation_hash=digest(new)))
        evaluation["generation_hash"] = digest(new)
    report = project_report(copied, manifest)
    if digest(records) != old_records_hash or digest(manifest) != manifest_hash:
        raise ValidationError("input mutation during projection")
    return dict(report=report, records=copied, exact_cost=dict(
        program=ledger.status(), attempts=[dict(attempt_id=p.spec.attempt_id,
            actual_usd=(ledger.attempt(p.spec.attempt_id)["receipt"] or {}).get("actual_usd"),
            state=ledger.attempt(p.spec.attempt_id)["state"], request_sha256=request_hash(p)) for p in plans]),
        provenance=dict(original_records_hash=old_records_hash, projected_records_hash=digest(copied),
                        manifest_hash=manifest_hash, generations=provenance,
                        ledger_events_sha256=sha(json_bytes(ledger.events()))))
