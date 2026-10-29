# Offline provider attempts

`provider_prices`, `provider_attempts`, and `provider_reporting` qualify a finite local
provider simulation against the existing spend ledger and W1 reporting contract.
They cannot dispatch to a real provider. No credentials are read and no CLI or legacy
client is enabled. A successful fixture is evidence of adapter behavior, not financial
authority, author rights, a model roster, human labels, or native story outcomes.

**Amendment 2026-10-07 (W3b2 scope revision, independently reviewed).** Real OpenRouter
endpoints declare two kinds of field the original allowlist refused, which made a route
carrying them unpriceable rather than unsafe. Both are now admitted, and the guard still
fails closed on anything genuinely unknown:

- **`internal_reasoning`** — a reasoning charge priced *separately from* `completion`. It is
  admitted **and mapped into the reservation census**, because admitting a price component
  without reserving it would be an under-reservation hole, not a feature: when a route prices
  reasoning separately, `reasoning_tokens` is reserved at that rate; when it does not, the
  contract's `reasoning_inclusive` holds and the reasoning charge stays inside `completion`
  (census price 0). This matters directly because the roster pins reasoning effort to Max.
- **`input_audio_cache`** — a media-scoped cache charge. It is admitted so a route carrying it
  can be priced, and it cannot enter a reservation because `provider_attempts._census` prices
  only the usage slots it explicitly maps — input, output, reasoning, cache read and cache
  write — and leaves every other usage class at 0, so a request profile
  declaring the text-only, no-media, no-tools protocol is never charged for it. That is a
  property of the census whitelist, **not** of a declared media-key set: an earlier draft of
  this amendment named a `MEDIA_KEYS` constant as the mechanism, but no code read it, so it was
  deleted rather than left as decoration (independent review, F3).
- **Override selectors** are no longer restricted to `min_prompt_tokens`. Real routes also
  narrow a rate by time: `utc_days` (day of week) and `utc_start`/`utc_end` (an HHMM window).
  Any override must carry at least one selector and at least one rate, and the applied rate
  remains the **maximum** over the base rate and every override — so a discount or a cheaper
  time window can still never reduce a reservation, which is the rule that made
  `deepseek-v4.1-flash`'s weekend discount priceable without weakening the guard.

**One observability limit of this amendment, stated plainly.** Its changes are **uncommitted**,
so a verifier who reads only `git diff HEAD` cannot see every hunk: the deleted `MEDIA_KEYS`
constant (F3) never existed in `HEAD` at all, so its removal is invisible to a diff-based check —
only the file as it now stands proves its absence. The same applies to any of these hunks until
they are committed. That is a property of reviewing uncommitted work, not a defect in the change,
but it means **the next verifier must read the working files, not only the diff**. (The stale
`.pyc` that briefly contained `MEDIA_KEYS` — cited by the delta review as unexplained evidence —
was deleted deliberately as bytecode cleanup, not by any concurrent process.)

**Amendment file list and consequences (disclosed, not implied).** The amendment's own diff is
the five in-scope files plus two record documents whose assertions it invalidates:
`src/provider_prices.py`, `src/provider_attempts.py`, `tests/test_provider_prices.py`,
`tests/test_provider_attempts.py`, `docs/provider-attempts.md` (this block), and — because they
assert a now-superseded count — `docs/repair-register.md` and `docs/experiment-contract.md`.
The working tree also carries an **unrelated** comment-only change to `config/models.yaml`
(+12/−0, the historical banner pointing at `config/study-roster-candidates.yaml`, which is
untracked). It belongs to the roster work, not to this amendment, and is declared here so the
diff is fully accounted for rather than left as an unexplained sixth file (review, F1).

**Test-ID drift this amendment causes.** The W3b1 acceptance is ID-exact: 397 exact test IDs at
checkpoint `ac6747af1f81ca77b33d63bbe366f1c85643c25b`, with
`canonical_set_sha256 6cc007461541386d6e4b44e5e1490d6422c4b7b9a2c7ef79daff6a56df2dc062`. **That binding is vault-side, not in this repo** — it is `🧭-GROUND-TRUTH/2026-09-21-story-evaluation-plan-consolidation/bindings/execution/w3b/acceptance.json`, which also records a per-file `source_file_sha256` for each of the five in-scope files (that is how this amendment's pre-image stays verifiable from the vault side). A reader of this repository cannot resolve that path, so it is named in full rather than cited as if local. The
amendment **removes exactly two IDs and adds 19**. Both removals are in the same control:
`test_D03_unknown_or_unsupported_prices_deny[reasoning_price]` was **deleted** because it asserted
denial for `internal_reasoning`, which is now deliberately supported, and its sibling
`[time_override]` was **renamed** to `[time_selector_bad_type]` so its name matches what it now
asserts — a rename that removes one ID and adds another, which is why 19 were added against 18
net-new tests. (Counts reproduced by collecting both in-scope files at the accepted commit in a
clean worktree and diffing the sorted ID lists: base 165, now 182, 2 removed, 19 added.) The
deletion is justified (its replacement
`test_a_genuinely_unknown_pricing_component_still_fails_closed` covers three genuinely-unknown
keys, including the near-miss `internal_reasoning_v2`), but **that binding is superseded** and
the two documents above carried the stale "397 exact test IDs" claim. Any later verifier must
re-collect IDs from a clean tree rather than reuse the 397 set.

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
