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
import shutil
import sys
import tempfile
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


def row_family(row: dict) -> str:
    """The row's task family, from whichever field the manifest generation recorded it in.

    `assignment_builder` now writes `task_family` onto the row, but a manifest built before that
    carries it only inside the embedded task document, so both are read rather than assuming the
    newer shape.
    """
    return row.get("task_family") or (row.get("task") or {}).get("task_type") or "unknown"


def one(args, raw: bytes, ledger, index: int, row: dict) -> dict:
    """One attempt, from prepared row to ledger entry. Raises SystemExit if it cannot prepare."""
    unit = f"{args.unit}-{unit_for(row)}"
    # A DRY RUN MUST NOT LEAVE EVIDENCE. `prepared` writes the attempt's immutable manifest, and a
    # dry run that wrote it into the real evidence root poisoned the unit it was inspecting: the next
    # real run hit "immutable evidence conflict" on a directory no call had ever used. Preparing into
    # a throwaway root keeps the dry run's promise -- it validates the wiring, then leaves.
    root = args.evidence
    if args.dry_run:
        # `.resolve()` is load-bearing: on macOS `mkdtemp` returns a path under `/var`, which is a
        # symlink to `/private/var`, and the evidence root refuses a symlinked path.
        root = pathlib.Path(tempfile.mkdtemp(prefix="pilot-dry-run-")).resolve()
    try:
        _, plan, admission, identity = paid.prepared(args.base, raw, index, unit, root)
        head = ledger.status()["envelopes"][plan.spec.envelope]["remaining_micros"]
        if args.dry_run:
            print(f"  [dry] {index:3} {plan.spec.attempt_id}  envelope left {head/1e6:.2f}", flush=True)
            return dict(attempt_id=plan.spec.attempt_id, state="DRY", status="not_sent",
                        reserved_usd=0.0, actual_usd=0.0, live_status=None)
    finally:
        if args.dry_run:
            shutil.rmtree(root, ignore_errors=True)
    observed = paid.preflight(plan, identity)
    result = asyncio.run(pa.execute_attempt(plan, ledger, admission))
    after = ledger.attempt(plan.spec.attempt_id)
    attempt_dir = plan.evidence_root / "attempts" / plan.spec.attempt_id
    detail = ""
    failure = attempt_dir / "failure.json"
    if failure.is_file():
        detail = json.loads(failure.read_text()).get("detail", "")
    # What the PROVIDER said about the generation itself, which the ledger cannot see: a call that
    # exhausted its budget on reasoning reconciles and is charged exactly like a finished one. The
    # first paid attempt was `outcome: truncated`, 9,000 of 9,000 tokens reasoning, zero characters,
    # $0.186 -- a success to the ledger and an empty string to the study.
    receipt = attempt_dir / "receipt.json"
    outcome = json.loads(receipt.read_text()).get("outcome") if receipt.is_file() else None
    entry = dict(attempt_id=plan.spec.attempt_id, state=result.ledger_state,
                 status=result.transport_status, detail=detail, outcome=outcome,
                 reserved_usd=after["reserved_micros"] / 1e6,
                 actual_usd=(after["actual_micros"] or 0) / 1e6,
                 live_status=(observed or {}).get("status"))
    print(f"  {index:3} {plan.spec.attempt_id[:52]:52} {entry['state']:11} "
          f"{str(outcome or ''):9s} ${entry['actual_usd']:.4f}  (envelope {head/1e6:.2f})", flush=True)
    return entry


SENT_MARKERS = ("transport.json", "response.bin", "receipt.json")


def discard_unsent(evidence_root, attempt_id):
    """Remove an attempt's directory when it provably never contacted the provider.

    `prepare_attempt` writes the immutable manifest BEFORE the reservation, so any row that fails
    between preparation and send leaves a directory behind, and evidence immutability then makes the
    whole unit unresumable. Three runs died on directories no call had ever used, one holding 172.

    This does not weaken immutability: an attempt carrying `transport.json`, `response.bin` or
    `receipt.json` HAS been sent and is never touched. What is removed is a preparation artifact.
    """
    root = evidence_root / "attempts" / attempt_id
    if not root.is_dir():
        return False
    if any((root / marker).exists() for marker in SENT_MARKERS):
        return False
    purge(root)
    return True


def purge(path):
    """Delegate the removal of a directory tree."""
    import shutil  # noqa: PLC0415

    shutil.rmtree(path, ignore_errors=True)


class StopRun(Exception):
    """Ends the loop from inside a handler without unwinding the process."""


class RunState:
    """One pilot run: counters, block list, progress file, and the loop over rows.

    Extracted from `main` because that function held the whole policy inline and every new stop
    condition grew it. Each stop is a named method now, and every row writes a trace line -- the
    failure path used to write nothing, so a run that prepared 172 attempts and sent 8 left no
    record of why the other 172 stopped.
    """

    def __init__(self, args, raw, ledger, held):
        self.args, self.raw, self.ledger, self.held = args, raw, ledger, held
        # The run's own directory, created before anything is written into it. A dry run persists no
        # ATTEMPT evidence, but it still reports progress, and that report needs somewhere to land.
        args.evidence.mkdir(parents=True, exist_ok=True)
        self.progress = args.evidence / "pilot-progress.json"
        self.state = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          done=[], failed=[], already_held=len(held))
        self.sent = 0
        self.failures = 0
        self.blocked = {}   # model -> why; later rows for it are skipped
        self.deferred = {}
        self.exit_code = 0

    # --- writing ---------------------------------------------------------------------
    def write(self, **extra):
        self.progress.write_text(json.dumps(
            self.state | dict(sent=self.sent, failures=self.failures) | extra, indent=1))

    def trace(self, **record):
        """One line per row, appended. A run must not be un-auditable after the fact."""
        with (self.args.evidence / "pilot-trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # --- one row ---------------------------------------------------------------------
    def skip_reason(self, row):
        if self.args.model and row["model"] not in self.args.model:
            return "deferred-model"
        family = row_family(row)
        if self.args.family and family not in self.args.family:
            self.deferred[family] = self.deferred.get(family, 0) + 1
            return "deferred-family"
        if attempt_id_for(row, self.args.unit) in self.held:
            return "already-held"
        if row["model"] in self.blocked:
            self.state.setdefault("blocked", []).append(
                dict(model=row["model"], reason=self.blocked[row["model"]]))
            return "blocked-model"
        return None

    def on_error(self, exc, index, row):
        reason = str(exc)
        self.trace(index=index, attempt=attempt_id_for(row, self.args.unit), error=reason[:400])
        if "immutable evidence conflict" not in reason:
            # RETURN, do not re-raise. Re-raising here killed the whole run on the first row that
            # could not be prepared -- which is how a 432-row pilot died at row 180 with `failed 0`
            # recorded and nothing on screen. A row that will not prepare is a FINDING to be counted
            # and skipped, not a reason to abandon the other 250 rows.
            return
        # A unit PREPARED but never sent leaves an immutable directory behind, so the same --unit
        # cannot pass it. The guard is right; the message is useless without a remedy.
        self.write(blocked_by_stale_evidence=str(self.args.unit))
        print("\n  STOPPED: --unit %s already has evidence for row %d that was prepared but never "
              "sent. Evidence is immutable by design.\n  Resume under a NEW unit tag: --unit %s-b"
              % (self.args.unit, index, self.args.unit))
        self.exit_code = 4
        raise StopRun

    def on_entry(self, entry, row):
        self.state["done"].append(entry)
        if entry["state"] == "UNKNOWN" and "HTTP 404" in entry.get("detail", ""):
            self.blocked[row["model"]] = entry["detail"][:200]
            print("  BLOCKED %s: %s" % (row["model"], entry["detail"][:110]), flush=True)
        elif entry.get("outcome") == "truncated":
            # Deterministic for a given (model, budget): it would spend the same budget thinking next
            # time. Blocking after ONE is what stops a roster being billed for empty strings.
            self.blocked[row["model"]] = "outcome=truncated: no text returned"
            print("  BLOCKED %s: truncated, zero characters" % row["model"], flush=True)
        self.sent += 1
        failed = entry["state"] == "UNKNOWN" or entry.get("outcome") == "truncated"
        self.failures = self.failures + 1 if failed else 0
        self.trace(attempt=entry["attempt_id"], state=entry["state"],
                   outcome=entry.get("outcome"), usd=entry.get("actual_usd"))
        self.write()

    # --- errors ----------------------------------------------------------------------
    def on_row_error(self, exc, index, row):
        """A row that would not prepare. False = keep going, True = end the run."""
        if self.args.skip_unsupported and "Unknown task type" in str(exc):
            self.state.setdefault("skipped", []).append(dict(index=index, reason=str(exc)))
            print("  %3d SKIP  %s" % (index, exc), flush=True)
            self.write()
            return False
        self.on_error(exc, index, row)      # raises unless it can name a remedy
        discard_unsent(self.args.evidence, attempt_id_for(row, self.args.unit))
        self.state["failed"].append(dict(index=index, reason=str(exc)))
        self.failures += 1
        self.write()
        # The threshold used to be checked ONLY after a successful attempt, so a ledger that was
        # already blocked produced 56 consecutive failures and then ran out of rows instead of
        # stopping at three. A blocked ledger answers every row the same way.
        if self.failures >= self.args.stop_after_failures:
            print("\n  STOPPED after %d consecutive failures; nothing further was bought."
                  % self.failures)
            return True
        return False

    def on_abort(self, exc, index):
        """A setup error is not an attempt outcome: nothing further is sent."""
        print("\n  ABORTED at row %d: %s: %s" % (index, type(exc).__name__, exc))
        print("  This is not an attempt outcome; nothing further was sent.")
        self.write(aborted=str(exc))
        self.exit_code = 3
        return True

    # --- the loop --------------------------------------------------------------------
    def step(self, index, row):
        """Attempt one row. True means the run should end."""
        if row_family(row) == "critique_improvement":
            if self.args.dry_run:
                return self.step_episode_dry(index, row)
            return self.step_episode(index, row)
        try:
            entry = one(self.args, self.raw, self.ledger, index, row)
        except StopRun:
            raise
        except (SystemExit, ValueError) as exc:
            return self.on_row_error(exc, index, row)
        except Exception as exc:
            return self.on_abort(exc, index)
        self.on_entry(entry, row)
        if self.failures >= self.args.stop_after_failures:
            print("\n  STOPPED after %d consecutive failures; nothing further was bought."
                  % self.failures)
            return True
        return False

    def step_episode_dry(self, index, row):
        """A dry run of an episode builds all seven requests and sends none.

        `one()` cannot do this: the critique family has no single-call prompt builder, which is the
        "Unknown task type" that kept these rows out of every run. Building the seven specs is the
        check that matters, and it costs nothing.
        """
        import critique_run

        unit = f"{self.args.unit}-{row['task_id']}-s{row['sample']}"
        study = reh._study()
        rounds = int((study.get("critique_episode") or {}).get("rounds", 3))
        from agentic_generator import AgenticPromptBuilder  # noqa: PLC0415

        specs = critique_run.build_specs(self.args.base, self.raw, index, unit, study=study,
                                         rounds=rounds, builder=AgenticPromptBuilder)
        print(f"  [dry] {index:3} {unit[:46]:46s} {len(specs)} turns", flush=True)
        self.sent += 1
        self.write()
        return False

    def step_episode(self, index, row):
        """One critique_improvement row = seven calls, recorded as ONE row of the run.

        The episode refuses to return unless all seven turns reconciled, so a partial episode can
        never be filed as a finished one; a failure raises out of the sender and lands in the same
        per-row failure handling a single-call row uses.
        """
        import critique_run

        unit = f"{self.args.unit}-{row['task_id']}-s{row['sample']}"
        study = reh._study()
        rounds = int((study.get("critique_episode") or {}).get("rounds", 3))
        from agentic_generator import AgenticPromptBuilder  # noqa: PLC0415

        summary = critique_run.run_row(
            base=self.args.base, assignment=self.args.assignment, index=index, unit=unit,
            evidence=self.args.evidence, ledger=self.ledger, authorise=self.args.authorise,
            study=study, builder=AgenticPromptBuilder, rounds=rounds)
        entry = dict(attempt_id=f"{unit}-episode", state="RECONCILED", status="completed",
                     detail="", outcome="completed", reserved_usd=0.0,
                     actual_usd=summary["cost_usd"], live_status=None,
                     turns=summary["turns"], turn_states=summary["states"])
        print(f"  {index:3} {unit[:52]:52} {'EPISODE':11} {summary['turns']} turns  "
              f"${summary['cost_usd']:.4f}", flush=True)
        self.on_entry(entry, row)
        if self.failures >= self.args.stop_after_failures:
            print("\n  STOPPED after %d consecutive failures; nothing further was bought."
                  % self.failures)
            return True
        return False

    def loop(self, rows):
        try:
            for index, row in enumerate(rows):
                if self.skip_reason(row):
                    continue
                if self.args.limit is not None and self.sent >= self.args.limit:
                    break
                if self.step(index, row):
                    break
        except StopRun:
            pass

    def finish(self):
        status = self.ledger.status()
        if self.deferred:
            self.state["deferred"] = self.deferred
            print("  DEFERRED by --family, not attempted: %s"
                  % ", ".join("%s x%d" % item for item in sorted(self.deferred.items())))
        print("\n  attempts this run %d · committed $%.4f · reserved $%.4f"
              % (self.sent, status["committed_micros"] / 1e6, status["reserved_micros"] / 1e6))
        print("  progress: %s" % self.progress)
        self.write()
        return self.exit_code or (2 if self.state["failed"] else 0)


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
    ap.add_argument("--model", action="append", default=None, metavar="MODEL",
                    help="only attempt rows for these model ids (repeatable). A route whose reasoning "
                         "encoding was wrong is re-run ON ITS OWN this way, instead of buying the "
                         "other eight routes a second time")
    ap.add_argument("--family", action="append", default=None, metavar="FAMILY",
                    help="only attempt rows of these task families (repeatable). A family that is "
                         "deliberately deferred -- because its episode runner does not exist yet -- "
                         "is skipped OUT LOUD: the count and the names are recorded, so a partial "
                         "run cannot be mistaken for the whole pilot")
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
    run = RunState(args, raw, ledger, held)
    run.loop(rows)
    return run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
