# Offline provider attempts

`provider_prices`, `provider_attempts`, and `provider_reporting` qualify a finite local
provider simulation against the existing spend ledger and W1 reporting contract.
They cannot dispatch to a real provider. No credentials are read and no CLI or legacy
client is enabled. A successful fixture is evidence of adapter behavior, not financial
authority, author rights, a model roster, human labels, or native story outcomes.

## API and immutable inputs

`validate_snapshot(source_bytes, route, observed_at, expires_at)` accepts exact catalog
and endpoint response bytes, an immutable `Route`, and offset-zero UTC timestamps.
The maximum lifetime is 24 hours; observation must not be in the future and expiration
must be strictly later than now. Exact model ID, canonical slug, provider tag, default
tier, USD, and the OpenRouter chat endpoint are bound. Dynamic aliases, ambiguous
base-provider tags, unsupported price components, nonzero request fees, and unsupported
time tiers fail closed. Context pricing uses the largest applicable component rate;
discounts never reduce a reservation. All original source bytes and hashes survive.

Two official source responses alone yield `qualification="unqualified"`. A third,
strict `SYNTHETIC OFFLINE ONLY` profile may describe bounded test input/output/cache
units, inclusive reasoning, and inactive services. Its hashes must bind those exact
two sources. It authorizes only the internal fixture simulator. Cache structural zero
requires an explicit profile assertion; missing telemetry otherwise stays unknown.
Real route support would need separate route-specific evidence for enforceable context
or input/output bounds, inclusive reasoning and cache charges, tier/routing behavior,
terminal cumulative billing, and funding fees. Generic documentation or absent fields
cannot supply that proof. This implementation does not invent endpoint echo or finality
flags for real providers, and makes no real route admission decision.

`prepare_attempt(spec, snapshot, artifact_bytes, request_fields, evidence_root)` returns
a frozen `RequestPlan`. `AttemptSpec` binds program/store, assignment/task/story/arm/sample,
assigned model, harness/protocol/settings/native-package hashes, effort, role, envelope,
and retry lineage. Artifact tuples contain actual bytes for assignment, task, prompt,
harness, protocol, settings, native package and eight admission gates. Prompt, request
body, full messages, raw sources, and manifest are persisted by content hash with
exclusive creation and fsync. Existing different bytes, symlinks, or changed preimages
are rejected. W1 validates assignments. No normalization replaces the submitted bytes.

`RequestFields` accepts a tuple of text messages, output limit, one exact reasoning
setting, optional temperature and seed. The endpoint must advertise each used control.
The body pins the canonical model and provider tag, disables fallbacks and streaming,
requires parameters, and specifies the default tier. The synthetic profile bounds all
13 existing ledger usage classes, including logical critic/retrieval/selection/retry
and grading activity. A reasoning effort label alone is never a real cost bound.

## Awaited transport and liability

```python
result = await execute_attempt(plan, ledger, admission, OfflineFixture(
    status_code=200,
    header_delay_ms=0,
    chunks=((0, complete_response_bytes),),
))
```

`Admission` must have mode `offline_fixture`, matching assignment/route, current validity,
and all eight strict synthetic gate artifacts: rights, provider permission, phase,
protocol, roster, effort, budget, billing. Synthetic gates never represent actual rights
or invoices. The existing ledger independently checks hosted billing, envelope/cap,
incidents and quote freshness before reservation. Its source is unchanged.

Only exact `OfflineFixture` data is accepted: status, finite millisecond delays, and up to
256 immutable byte chunks totaling at most 16 MiB. Callbacks, sync/blocking handlers,
caller transports and custom streams are rejected. The adapter constructs its own
cooperative async handler, body stream and no-op close using installed httpx MockTransport.
The client disables redirects and environment proxy use and has no socket transport.

One attempt performs reserve, checks current state, yields a pre-marker cancellation
checkpoint, and obtains a fresh durable `mark_sent` result with `replayed is False`.
Only that result permits one transport call. A historical event result cannot permit
another send. Re-execution of SENT, UNKNOWN or RECONCILED returns `replay_no_send`.
Retries require a new attempt ID, adjacent retry index, same episode/assignment/role,
independent reservation and explicit invocation. The finite maximum is three; there
are no automatic retries or background jobs. One immutable attempt identity occupies each
root-lineage/retry-index slot permanently, including after completion or cancellation.
The existing ledger's unique transactional reserve event claims that slot atomically;
there is no count-then-reserve check. A single ledger audit snapshot verifies the actual
ancestor quote/request hashes, fixed assignment/policy, adjacent indices and root identity
against their immutable manifest bytes. Callers cannot supply a separate root label.
For retries, the reserve result supplies quote identity while the fresh marker remains
the only permission; a replayed marker is followed by an authoritative current-state read.

The plan uses one absolute **one-second transport deadline**, covering awaited headers
and the entire body read. Slow chunks do not reset it. Cancellation propagates; timeout
or persistence failure retains SENT/UNKNOWN liability. Unknown cleanup failure cannot
refund a durable marker. SQLite's existing five-second contention timeout is separate;
execute uses at most six ledger methods including retry validation/error cleanup.
Host scheduling and fsync remain trusted assumptions. There is no universal whole-method
return-time promise and no process/worker timeout platform. Failed or replayed UNKNOWN
cleanup reads the actual ledger state once; it preserves a committed receipt's state and
outcome. If authority itself cannot be read, `LedgerStateUnavailable` explicitly surfaces
uncertainty without inventing SENT, refunding, or sending again. Cancellation still propagates.

## Recovery and reporting

Response bytes and HTTP status/request-hash metadata are saved before parsing/reconcile.
The synthetic response includes matching model/provider/tier, generation ID, terminal
choice, and original JSON numeric `usage.cost`; optional `total_cost` must equal it.
This is a cumulative receipt: neither upstream cost detail nor a second total is added.
Money parses directly to Decimal and remains decimal text in the existing ledger;
reservation and accounting use its upward integer-microdollar rule. Refusal, truncation,
empty output and billed HTTP failures retain cost. Missing money preserves liability.
Missing applicable usage records known money plus the ledger's permanent usage incident.
All 13 counters survive, including unknowns, and overrun facts are never clipped.
Scalar response identities are type-checked before membership operations; malformed
neighboring containers yield retained infrastructure failure, not an untyped parser escape.

After a process exits with a saved complete response, explicitly call
`reconcile_attempt(plan, ledger, response_bytes, response_status)`. It checks immutable
manifest/source/artifact bytes and response provenance, permits expired quote recovery,
claims one global generation receipt, and is idempotent. Receipt references and the return
object are prepared before committing, so result construction has no post-commit filesystem
read. Every evidence read walks pinned directory descriptors using no-follow flags for all
components; nonregular files (including FIFOs) reject before reading. Send admission and
recovery share this reader. This is local path confinement under a trusted host, not hostile
host isolation or a filesystem wall-time promise. A crash after marker but before
response save has no automatic resend or refund. A RESERVED attempt has not received
send permission; any cancellation/release remains the existing ledger owner's action.

`project_attempts(tuple_of_plans, ledger)` emits W1 attempts, retaining failed and unknown
cost entries and mapping rich usage to the existing input/output/reasoning triple only
when all five token dimensions, including cache read/write, are known. Applicable missing
cache counters keep W1 usage unknown; explicit proven structural zeros remain valid.
`project_cost_report(manifest, original_records, plans, ledger)` first executes the
unchanged W1 validator against the **original** records. Stale evaluation-to-generation
hashes reject before any copy or rehash. It then deep-copies, requires exact attempt and
assignment identities, replaces only attempt arrays, binds completed output to saved
response bytes, rebinds previously valid evaluation hashes, and revalidates through W1.
The return contains report, projected records, exact ledger money/state sidecar and
old/new provenance. Original objects remain unchanged on success and failure.
Unattempted fixed assignments and story clusters stay in the denominator; W1 `resolved`
remains `None`. Provider completion never certifies a native outcome.

## Verification

Run `uv run --no-sync pytest -p no:cacheprovider` in the repository. Existing tests trap
network access. Provider tests exercise actual awaited entry points, local process crash
and contention boundaries, exact MockTransport calls, money/unknown behavior, original
W1 validation, and finite retry/cancellation behavior. The capsule implementation evidence
contains exact collected IDs, JUnit results, ten copied-source adversarial mutants, source
preservation hashes and synthetic behavioral receipts. Independent code/security review
and parent acceptance remain required before any broader integration.
