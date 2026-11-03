#!/usr/bin/env python3
"""Send ONE authorised paid attempt, and refuse to be anything other than that.

WHY THIS FILE EXISTS. `prepare_attempt` builds and validates an attempt; `execute_attempt` reserves
the ledger and opens the transport. `rehearse_attempt.py` proved the first half with no money. This is
the second half, and it is a separate program rather than an `--execute` flag on the rehearsal because
a dry-run tool that can spend is a tool where a typo becomes an invoice.

Four refusals stand between here and a charge, each one placed where something upstream could not see:
  * the route must still carry `c8_verified` on the RETAINED bytes -- a stale profile and a stale
    verdict both read as a pass;
  * all eight admission gates must be `satisfied` in the RECORD, so the gate artifacts travel as
    derived evidence rather than as eight more things this file asserts;
  * the credential is resolved BEFORE the reservation is taken, so a missing key costs nothing --
    a reservation is the one step that cannot be undone;
  * the ledger is OPENED, never created: a program's budget identity is made once, deliberately.

    export OPENROUTER_API_KEY=...      # never on a command line, never in a file
    uv run python scripts/paid_attempt.py --base <base> --assignment <a.json> \\
        --ledger <path.db> --evidence <dir> --authorise PAID-PILOT-2026-11-03
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json

import httpx
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import provider_attempts as pa  # noqa: E402
import provider_prices as pp  # noqa: E402
import spend_ledger as sl  # noqa: E402
from measurement_contract import digest  # noqa: E402

import dispatch_runner as dr  # noqa: E402
import rehearse_attempt as reh  # noqa: E402


def _token(value: str | None) -> str:
    """The dispatch gate's rule, applied again at the only place in the programme that can spend."""
    if not value:
        raise SystemExit("AUTHORISE no token supplied — this is a procedure, not a permission")
    if not value.startswith("PAID-PILOT-"):
        raise SystemExit(f"AUTHORISE token {value!r} does not match PAID-PILOT-<date>")
    try:
        stamp = datetime.date.fromisoformat(value.removeprefix("PAID-PILOT-"))
    except ValueError:
        raise SystemExit(f"AUTHORISE token {value!r} has no ISO date") from None
    if stamp > datetime.date.today():
        raise SystemExit(f"AUTHORISE token is dated {stamp}, in the future")
    return value


def gates(record: dict, assignment_hash: str, route_sha256: str, valid_until: str) -> dict[str, bytes]:
    """Eight gate artifacts, derived from the record's own statuses — never written here.

    The gate schema admits five fields, so a gate cannot carry its evidence text. The evidence stays
    in `record.admission_gates`; this function refuses to emit anything unless all eight say
    `satisfied` there. Otherwise a gate is an assertion with no referent.
    """
    stated = record.get("admission_gates") or {}
    bad = sorted(kind for kind in pa.GATES if (stated.get(kind) or {}).get("status") != "satisfied")
    if bad:
        raise SystemExit(f"admission gates not satisfied in the record: {bad}")
    authority = pp.AUTHORITY_FOR_QUALIFICATION[pp.QUALIFICATION_FOR_KIND[pp.VERIFIED_KIND]]
    return {f"gate-{kind}": reh.json_bytes(dict(
        kind=kind, authority=authority, assignment_hash=assignment_hash,
        route_sha256=route_sha256, valid_until=valid_until)) for kind in sorted(pa.GATES)}


def _prepared(args, token: str):
    """Everything that can be refused for free, named and validated before a ledger is touched."""
    raw = args.assignment.read_bytes()
    manifest = json.loads(raw)
    if not manifest.get("judge_designated"):
        raise SystemExit("manifest records no judge designation; refusing to spend on it")
    row = manifest["assignments"][args.index]
    assignment_hash = digest(manifest)
    record, endpoints, profile, artifact = reh.load_route(args.base, row["model"])
    identity = record["route_identity"]
    snapshot = pp.validate_snapshot(
        ((args.base / "openrouter-models.json").read_bytes(), endpoints, profile, artifact),
        reh._route(identity), identity["observed_at"], identity["expires_at"])
    if snapshot.qualification != pp.QUALIFICATION_FOR_KIND[pp.VERIFIED_KIND]:
        raise SystemExit(f"{row['model']}: qualification {snapshot.qualification!r} is not payable")
    fields = reh.request_fields(row["task"], "max", 6000)
    # The eight gate artifacts must be part of the PLAN's artifact set: `_admit` requires every
    # authority ref to be an artifact the request already carries ("unbound gate evidence"
    # otherwise). They are derived from the record's own statuses, and the route hash they bind to
    # is the one the plan will carry, so they are computed before the plan rather than after it.
    route_sha256 = pp.sha(reh.json_bytes(dataclasses.asdict(reh._route(identity))))
    gate_blobs = gates(record, assignment_hash, route_sha256, identity["expires_at"])
    artifacts = dict(reh.artifacts_for(row, fields, raw, args.base)) | gate_blobs
    # The unit tag is explicit because an attempt id is ONE ledger slot, forever. The first smoke
    # attempt now sits in UNKNOWN and can never be resent, so a second call has to be a different
    # unit -- and the id must say which one it is rather than silently colliding.
    unit_id = f"{args.unit}-u{args.index}"
    plan = pa.prepare_attempt(
        reh.spec_for(row, dict(attempt_id=dr.attempt_id(row["model"], unit_id, row["condition"]),
                               unit_id=unit_id),
                     "max", artifacts, assignment_hash),
        snapshot, tuple(sorted(artifacts.items())), fields, args.evidence)
    admission = pa.Admission(
        mode=snapshot.qualification,
        authority_refs=tuple(pa.ArtifactRef.of(name, blob) for name, blob in sorted(gate_blobs.items())),
        assignment_hash=assignment_hash,
        route_sha256=route_sha256,
        valid_until=identity["expires_at"])
    return token, row, plan, admission, identity


def preflight(plan, identity: dict) -> dict | None:
    """What the provider says about the pinned endpoint AT SEND TIME. Free, unauthenticated, advisory.

    The first paid attempt failed at the transport layer and the evidence could not say whether the
    endpoint was healthy when we hit it: the retained snapshot is 18 hours old by then, and its
    `status` field had already moved from -5 to 0 in between. This reads the live endpoint list
    first and records the pinned tag's numbers beside the attempt, so a failure arrives with its
    context rather than a single word.

    It never blocks and never raises. A preflight that refused on `status` would quietly re-pin the
    endpoint the owner explicitly decided to validate with a real call, and one that could throw
    would be a new way to lose an attempt -- which is why the ENTIRE body is inside the guard.
    """
    try:
        url = f"https://openrouter.ai/api/v1/models/{identity['model_id']}/endpoints"
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
            data = client.get(url).json()["data"]["endpoints"]
        live = next((e for e in data if e.get("tag") == identity["endpoint_tag"]), None)
        if live is None:
            return None
        observed = dict(
            url=url, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            tag=identity["endpoint_tag"], status=live.get("status"),
            uptime_last_5m=live.get("uptime_last_5m"),
            uptime_last_30m=live.get("uptime_last_30m"),
            uptime_last_1d=live.get("uptime_last_1d"))
        pa._persist(pa._directory(plan) / "preflight.json", reh.json_bytes(observed))
        return observed
    except Exception:  # advisory only: bookkeeping must never cost an attempt
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--assignment", type=pathlib.Path, required=True)
    ap.add_argument("--ledger", type=pathlib.Path, required=True)
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--index", type=int, default=0, help="which assignment row to send")
    ap.add_argument("--unit", default="smoke",
                    help="unit tag; a new tag is a new ledger slot, an existing one is a replay")
    ap.add_argument("--authorise", default=None)
    args = ap.parse_args()

    token, row, plan, admission, identity = _prepared(args, _token(args.authorise))
    print(f"  token        : {token}")
    print(f"  route        : {row['model']} ({admission.mode})")
    print(f"  attempt      : {plan.spec.attempt_id}")
    print(f"  body bytes   : {len(plan.body_bytes)}  deadline {plan.deadline_seconds}s")
    print(f"  gates        : {len(admission.authority_refs)} satisfied")
    observed = preflight(plan, identity)
    if observed:
        print(f"  live status  : {observed['status']!r}  5m={observed['uptime_last_5m']} "
              f"30m={observed['uptime_last_30m']} 1d={observed['uptime_last_1d']}")
    ledger = sl.Ledger.open(args.ledger, program_id=reh.PROGRAM_ID, store_id=reh.STORE_ID)
    result = asyncio.run(pa.execute_attempt(plan, ledger, admission))
    print(f"  state        : {result.ledger_state}  transport {result.transport_status}")
    after = ledger.attempt(plan.spec.attempt_id)
    print(f"  reserved usd : {after['reserved_micros'] / 1_000_000:.4f}")
    if after.get("actual_micros") is not None:
        print(f"  actual usd   : {after['actual_micros'] / 1_000_000:.4f}")
    print(f"  evidence     : {plan.evidence_root / 'attempts' / plan.spec.attempt_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
