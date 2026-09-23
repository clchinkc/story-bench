# Atomic program spend ledger (W3a)

`src/spend_ledger.py` implements offline accounting for one USD 1,000 program.
It uses only Python's standard library and the existing locked environment.
This unit does not fetch prices, implement a provider adapter, qualify any real
provider, resolve actual hosted-session charges, or permit a paid experiment.
The existing whole legacy live provider/evaluator routes remain quarantined.

## Authority and storage

The trusted host initializes one canonical local SQLite file with
`Ledger.create(path, program_id=..., store_id=..., authorization=Authorization(...))`.
Create is exclusive and never replaces a file. Workers receive the same identities
and use `Ledger.open`; a missing file, different identity/schema, or copy at a
different canonical path rejects. There is no budget reset or cap-change method.

The cap is 1,000,000,000 integer micro-USD (1 USD = 1,000,000 micro-USD). Initial
allocations are repair 100, data_judge 200, pilot 150, main 400 and reserve 150 USD.
Reserve cannot fund a request directly. The parent must transfer available
allocation out of it before use, with an immutable authorization event.

Every new store starts with an unresolved `hosted_session` external charge in
repair/research. It blocks both reservation and initial send marking. Only a
trusted, evidence-bound final monetary receipt can reconcile it, including zero
when actually established. Disposable tests use explicitly synthetic receipts;
they say nothing about the program's current hosted billing, which is unknown.

`Authorization(authorization_id, evidence_sha256, reason)` records a parent
assertion. It is **not authentication or a signature**. The host must control the
canonical path, administrative methods, filesystem and executable code; workers
must not have arbitrary database, replacement, rollback or create access. Copies
are not tamper-proof. A malicious host can forge or replace local state. Use a
local durable filesystem with working SQLite locks; NFS and distributed use are
not supported. A crash during initial creation can leave an unusable file; host
recovery must inspect it, never automatically delete/recreate a program ledger.

## Quotes and receipts

Inputs are frozen dataclasses. `UsageBound(kind, max_units, usd_per_unit)` carries a
strict nonnegative integer bound and a known decimal-text unit price.
`Quote(provider, model, snapshot_sha256, observed_at, expires_at, request_sha256,
currency, lines)` carries a tuple of bounds. USD is the only currency. SHA256
identities are lowercase hex; timestamps include UTC offset zero. A snapshot's
validity must be positive and at most 24 hours, and it must be current at reserve
and initial send marking. Zero usage is explicit and still needs a known price.

Both quote and receipt require exactly these usage classes, without duplicates:

```
input_tokens output_tokens reasoning_tokens cache_read_tokens cache_write_tokens
critic_calls retrieval_calls selection_calls retry_calls grading_calls
training_units infrastructure_units agent_units
```

The future provider adapter owns official price provenance, applicability of rates,
complete request bounds and transport-enforced limits for all billable dimensions.
Structurally valid synthetic prices cannot qualify a real provider. The ledger
validates the input contract and arithmetic; it does not verify invoices with a
provider or authenticate externally supplied facts.

`Receipt(receipt_id, provider, model, snapshot_sha256, request_sha256, currency,
actual_usd, usage, outcome, evidence_sha256, recorded_at)` is a final cumulative
charge for one attempt. `usage` is a tuple of `(kind, strict_integer_or_None)`
pairs. Actual money remains the original exact decimal text; usage `None` remains
unknown. Missing usage classes reject, but explicit unknown usage with known money
records the money and a permanent `unreconciled_usage` incident. It never invents
usage zero or discards a known receipt. Unknown money is not a zero receipt: retain
the reservation with `mark_unknown` until the final monetary receipt arrives.

Outcomes: completed, task_failure, absent_output, provider_failure,
infrastructure_failure, grader_failure, refused, truncated, timeout, cancelled.
Not-run means unsent cancellation, not a billed receipt outcome. Failed billed
attempts remain in the ledger. Deployment and research are distinct cost roles.

`usd_to_micros(str | Decimal)` uses exact integer ratios and rounds liabilities
upward, independently of Decimal's ambient precision. Each quote line is rounded
up separately before summing. Floats, booleans, negative/nonfinite amounts and
unsupported decimal precision reject. Usage/transfer integers are bounded by
10^15; money supports at most 200 decimal digits and exponent magnitude 100.

## Operations and recovery

Every mutation uses `BEGIN IMMEDIATE`, a five-second contention timeout and SQLite
`synchronous=FULL`. State changes, receipt claims, incidents and the audit event
commit together. Contention/error fails closed with `LedgerError`; it never grants
permission. Rejected operations leave accounting state and events unchanged.

| API | Accepted behavior |
| --- | --- |
| `reserve(event_id, attempt_id=..., envelope=..., role=..., quote=...)` | Creates RESERVED only if known committed spend plus all open reservations fits both the envelope and cap, and there is no unknown external charge or incident. |
| `mark_sent(event_id, attempt_id=...)` | RESERVED → SENT; rechecks dispatch blocks and snapshot freshness, then durably commits permission before transport. |
| `mark_unknown(event_id, attempt_id=..., reason=...)` | SENT → UNKNOWN for timeout/local cancellation/uncertain billing; retains the full reservation. |
| `cancel_unsent(event_id, attempt_id=..., reason=...)` | RESERVED → CANCELLED; releases only when no send permission has committed. The attempt ID remains used. |
| `reconcile(event_id, attempt_id=..., receipt=...)` | SENT/UNKNOWN → RECONCILED; records final actual truth and releases the open reservation. Identity mismatch rejects. |
| `record_external_unknown(event_id, charge_id=..., envelope=..., role=..., reason=...)` | Creates a new unknown external liability and blocks dispatch globally. |
| `reconcile_external(event_id, charge_id=..., receipt_id=..., actual_usd=..., evidence_sha256=..., authorization=...)` | Records final known external charge without deleting its original unknown event. |
| `transfer(event_id, source=..., target=..., amount_micros=..., authorization=...)` | Moves a positive integer amount of uncommitted allocation between distinct envelopes; total allocation remains fixed. |

Event IDs are global across operations. An exact retry returns the original
JSON-compatible snapshot with `replayed=True` and appends no event; a fresh accepted
operation returns `replayed=False`. Conflicting reuse rejects. A retried historical
reservation can return its original RESERVED result even after finalization: it is
an audit replay, not current state or new permission. `attempt(id)` reads current
state. Receipt IDs are global across attempts and external reconciliation.

The transport owner **must send only after a fresh `mark_sent` result and must
never send on `replayed=True`**. Concurrent send IDs for one attempt cannot both
grant permission. Crashes between reservation and marking retain RESERVED;
cancellation is allowed only when the adapter has obeyed the marker-before-send
protocol. Crashes after marking retain liability even if the transport was never
reached. There is no automatic expiration or refund. An evidenced zero receipt can
release a SENT/UNKNOWN reservation. A timeout, process exit, or local cancellation
alone cannot establish zero billing.

Actual money above the quoted micro-USD liability records a `money_overrun` incident.
Usage above any bound records `usage_overrun`; known money with unknown usage records
`unreconciled_usage`. External charges exceeding remaining envelope/cap record
`external_overrun`. Truth is never clipped to fit the budget; known spend and nominal
remaining can show an overrun. These incidents are permanent in v1 and block new
reservations/send marking, while reconciliation remains possible. There is no
incident-clearing service, usage correction, partial invoice or credit-adjustment
API. Such recovery needs a separately specified and reviewed future change.

## Reports and audit

`status()` returns identity, currency/quantum/cap, known committed and open reserved
micro-USD, nominal remaining, usable amount, unknown external IDs, incident IDs,
sorted attempt IDs, uncertain SENT/UNKNOWN attempt IDs, envelope summaries and
deployment/research totals. Known committed zero does not imply the actual bill is
zero when unknown IDs remain. Usable amount is zero when blocked; untransferred
reserve allocation is excluded. `attempt(id)` and `events()` return detached
JSON-compatible snapshots. Mutating a returned object does not change storage.

Schema version 1 contains program, envelopes, attempts, external, receipts,
incidents and events. Ordered events include sequence, unique event_id, kind,
canonical payload, original result and UTC recorded_at. Kinds are program_created,
reserved, sent, unknown, cancelled_unsent, reconciled, external_unknown,
external_reconciled and transferred. Creation includes the initial unknown charge.
SQL triggers prohibit updating/deleting program metadata or audit events; those
triggers are an integrity aid within trusted storage, not a hostile-host defense.

Errors derive from `LedgerError(ValueError)`: ValidationError, ConflictError,
StateError, BudgetError and DispatchBlocked. Callers must treat every exception as
no new dispatch permission. Retry a transient store failure using the same event ID;
after an uncertain return, a replayed send means do not resend.

## Offline verification

Run `uv run --no-sync pytest -q -p no:cacheprovider`. Ledger tests use disposable
stores and synthetic inputs, including independent spawned processes contending
for the last allowance and abrupt exits around the send marker/transport boundary.
They exercise successful reserve → mark_sent → reconcile as well as atomic denial,
failure billing, exact replay, permanent incidents and unknown-cost blocks.
Mutation evidence belongs in the capsule's W3a producer artifacts. Passing this
suite establishes local accounting behavior only; independent code/security/QA
review and later provider, data, human and phase gates remain required.
