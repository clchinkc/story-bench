#!/usr/bin/env python3
"""Run the pilot: 432 arm-runs, sequentially, resumably, and stopping when something is wrong.

WHY THIS IS NOT A LOOP AROUND `paid_attempt`. The pilot is the same act four hundred times, so the
preparation must be the SAME code — `paid_attempt.prepared` — or the two will drift and only one of
them will be the one that was rehearsed. What differs is everything around it: an attempt id is one
ledger slot forever, so a resume must skip what the ledger already holds; a systematic failure must
stop the run rather than buy 400 more copies of it; and the envelope must be watched, because the
ledger refuses overruns but only after the reservation is attempted.

Three stops, and each one is a mistake I would otherwise pay for 400 times:

  * `--stop-after-failures` consecutive UNKNOWNs end the run. The 2026-11-03 smoke units failed
    twice in a row for the same reason; without this, 432 arms would have been billed for nothing.
  * an unreconciled attempt is never resent: the ledger holds it and this skips it.
  * `--limit` exists so a tranche can be proved before the rest is spent.

    export OPENROUTER_API_KEY=...
    uv run python scripts/pilot_run.py --base <base> --assignment <a.json> \\
        --ledger <pilot.sqlite3> --evidence <dir> --authorise PAID-PILOT-2026-11-04 --limit 36

Progress lands in `<evidence>/pilot-progress.json` after every attempt, so a run that dies mid-way
can be read without archaeology.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import provider_attempts as pa  # noqa: E402
import spend_ledger as sl  # noqa: E402

import dispatch_runner as dr  # noqa: E402
import paid_attempt as paid  # noqa: E402
import rehearse_attempt as reh  # noqa: E402


def unit_for(row: dict) -> str:
    """The STUDY's unit, not the row's position.

    A unit is one task occurrence, and its two arms (A and D) must share the id or the pairing the
    whole design rests on is lost. `--unit` prefixes it so a re-run is a new ledger slot rather
    than a collision.
    """
    return f"{row['task_id']}-s{row['sample']}"


def attempt_id_for(row: dict, unit: str) -> str:
    return dr.attempt_id(row["model"], f"{unit}-{unit_for(row)}", row["condition"])


def one(args, raw: bytes, ledger, index: int, row: dict) -> dict:
    """One attempt, from prepared row to ledger entry. Raises SystemExit if it cannot prepare."""
    unit = f"{args.unit}-{unit_for(row)}"
    _, plan, admission, identity = paid.prepared(args.base, raw, index, unit, args.evidence)
    head = ledger.status()["envelopes"][plan.spec.envelope]["remaining_micros"]
    if args.dry_run:
        print(f"  [dry] {index:3} {plan.spec.attempt_id}  envelope left {head/1e6:.2f}", flush=True)
        return dict(attempt_id=plan.spec.attempt_id, state="DRY", status="not_sent",
                    reserved_usd=0.0, actual_usd=0.0, live_status=None)
    observed = paid.preflight(plan, identity)
    result = asyncio.run(pa.execute_attempt(plan, ledger, admission))
    after = ledger.attempt(plan.spec.attempt_id)
    entry = dict(attempt_id=plan.spec.attempt_id, state=result.ledger_state,
                 status=result.transport_status,
                 reserved_usd=after["reserved_micros"] / 1e6,
                 actual_usd=(after["actual_micros"] or 0) / 1e6,
                 live_status=(observed or {}).get("status"))
    print(f"  {index:3} {plan.spec.attempt_id[:52]:52} {entry['state']:11} "
          f"${entry['actual_usd']:.4f}  (envelope {head/1e6:.2f})", flush=True)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--assignment", type=pathlib.Path, required=True)
    ap.add_argument("--ledger", type=pathlib.Path, required=True)
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--authorise", default=None)
    ap.add_argument("--unit", default="pilot")
    ap.add_argument("--limit", type=int, default=None, help="stop after N attempts this run")
    ap.add_argument("--stop-after-failures", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-unsupported", action="store_true",
                    help="skip rows whose task type the harness cannot build, and say which")
    args = ap.parse_args()
    paid._token(args.authorise)                     # refuse before anything is built
    # A 432-attempt run must not discover a missing credential at attempt 1, behind a traceback.
    # `execute_attempt` already refuses before reserving, so nothing is ever spent on this -- the
    # check exists to fail in one line instead of four hundred.
    if not args.dry_run and not os.environ.get(pa.CREDENTIAL_ENV):
        raise SystemExit(
            f"{pa.CREDENTIAL_ENV} absent; refusing to start a pilot that cannot send.\n"
            f"  Export it in THIS shell, in the same block as the run: "
            f"export {pa.CREDENTIAL_ENV}='...'  -- a background job inherits only what its "
            f"own shell had.")

    raw = args.assignment.read_bytes()
    rows = json.loads(raw)["assignments"]
    ledger = sl.Ledger.open(args.ledger, program_id=reh.PROGRAM_ID, store_id=reh.STORE_ID)
    held = set(ledger.status()["attempt_ids"])
    progress = args.evidence / "pilot-progress.json"
    state = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 done=[], failed=[], already_held=len(held))
    sent = failures = 0
    for index, row in enumerate(rows):
        if attempt_id_for(row, args.unit) in held:
            continue
        if args.limit is not None and sent >= args.limit:
            break
        try:
            entry = one(args, raw, ledger, index, row)
        except (SystemExit, ValueError) as exc:      # a route that will not prepare is a finding
            reason = str(exc)
            # Exactly one class is skippable, and only on request: a task family the harness has no
            # prompt builder for. Anything else -- a withdrawn route, a stale profile -- is a real
            # failure and stops the run, because those are the ones worth being loud about.
            if args.skip_unsupported and "Unknown task type" in reason:
                state.setdefault("skipped", []).append(dict(index=index, reason=reason))
                print(f"  {index:3} SKIP  {reason}", flush=True)
                progress.write_text(json.dumps(state | dict(sent=sent), indent=1))
                continue
            state["failed"].append(dict(index=index, reason=reason))
            failures += 1
            continue
        except Exception as exc:                     # a setup error is not a per-row failure
            print(f"\n  ABORTED at row {index}: {type(exc).__name__}: {exc}")
            print("  This is not an attempt outcome; nothing further was sent.")
            progress.write_text(json.dumps(state | dict(aborted=str(exc)), indent=1))
            return 3
        state["done"].append(entry)
        sent += 1
        failures = failures + 1 if entry["state"] == "UNKNOWN" else 0
        progress.write_text(json.dumps(state | dict(sent=sent, failures=failures), indent=1))
        if failures >= args.stop_after_failures:
            print(f"\n  STOPPED after {failures} consecutive failures; nothing further was bought.")
            break
    status = ledger.status()
    print(f"\n  attempts this run {sent} · committed ${status['committed_micros']/1e6:.4f} · "
          f"reserved ${status['reserved_micros']/1e6:.4f}")
    print(f"  progress: {progress}")
    return 0 if not state["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
