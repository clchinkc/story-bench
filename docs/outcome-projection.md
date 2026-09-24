# Outcome projection (Story-bench)

Unit: W2C-SCOPE-01. Module: src/outcome_projection.py. Tests: tests/test_outcome_projection.py.

## What this is, and what it is not

This module is the deferred single pure Story-bench report/reward outcome projection
required by the evaluation contract (reviewed-contract.md line 65; contract-probe.md
lines 93 and 130). ONE call, project_outcome(record), returns ONE immutable object
carrying the report-facing status/diagnostics AND the reward-facing eligibility for
the same already-decided native assessment result.

It is not a resolver and not an authority. It does not decide whether a story
resolved, does not read evidence, does not score prose, and does not establish native
truth. Another component decides the native outcome; this module only projects it,
and fails closed on anything it does not recognize.

## Frozen interface

    PROJECTION_VERSION = 'sb-outcome-projection/1'
    IDENTITY_FIELDS = ('evaluationId', 'profileRef', 'criterionSpecDigest', 'projectionVersion')

    project_outcome(record: Mapping) -> OutcomeProjection
    projection_key(p: OutcomeProjection) -> dict
    assert_projection_parity(a: OutcomeProjection, b: OutcomeProjection) -> None

project_outcome is pure and total over the closed vocabulary. Identity member names
are exactly the four above, in that canonical order; the projection dataclass keeps
them as its first four fields. A missing identity member, an unknown extra member, a
foreign projectionVersion, a non-Mapping record, a malformed validity mapping, a
non-sequence reasons member or an unknown status/validity token all raise a typed
error from the OutcomeProjectionError family.

## Input record

A canonical Mapping with:

    evaluationId            non-empty string (identity)
    profileRef              native closed mapping (identity), exactly:
                              profileId  non-blank string
                              revision   positive int (bool rejected)
                              digest     'sha256:' + 64 lowercase hex digits
                            canonicalized to the frozen tuple
                            (profileId, revision, digest) in both the dataclass field and
                            projection_key, so two native profiles sharing a profileId but
                            differing in revision or digest stay distinct. A bare string, a
                            missing or extra key, a blank profileId, a bool/non-int/
                            non-positive revision or a non-canonical digest fails closed.
    criterionSpecDigest     non-empty string (identity)
    projectionVersion       must equal 'sb-outcome-projection/1' (identity)
    status                  one of observation | resolved | unresolved | unassessable
    validity                mapping with exactly execution and grading
    reasons                 sequence of diagnostic entries (preserved, never dropped)
    qualityPair / quality_pair   optional; recognized and deliberately ignored

execution is one of valid | provider-invalid | infrastructure-invalid | not-run.
grading is one of valid | grader-invalid.

The record is closed: any top-level member outside that set fails closed, so a
caller-set resolved flag, reward scalar, training label or in-denominator override
cannot enter the projection.

## Closed mapping law

Validity is evaluated first, then status (see Precedence below). Cells:

    fully valid (execution=valid, grading=valid)
      observation   -> rewardEligible False, trainingLabel None, exclusionReason 'operational-mode'
      resolved      -> rewardEligible True,  trainingLabel 1,    exclusionReason None
      unresolved    -> rewardEligible True,  trainingLabel 0,    exclusionReason None
      unassessable  -> rewardEligible False, trainingLabel None, exclusionReason 'unassessable'

    any invalid validity (for every status)
      execution=provider-invalid|infrastructure-invalid|not-run
        -> rewardEligible False, trainingLabel None, exclusionReason = that exact token
      grading=grader-invalid
        -> rewardEligible False, trainingLabel None, exclusionReason 'grader-invalid'
      both invalid
        -> exclusionReason = the two exact tokens joined by '+', execution token first,
           e.g. 'provider-invalid+grader-invalid'

Every row has inDenominator True and stays visible. An unassessable, invalid or
not-run case is never converted to 0 or 1 by truthiness; trainingLabel is None and
exclusionReason names the cause.

## Precedence: why validity beats status

The activation's mapping law states the validity rule first, and validity-first
preserves information that status-first would destroy: with status-first, both a
valid-but-unassessable case and a provider-invalid-unassessable case would collapse
to the single reason 'unassessable', losing the disposition distinction that
contract-probe.md line 127 requires ('provider-invalid, infrastructure-invalid,
grader-invalid, not-run and valid-but-unassessable remain different native
dispositions'). The native status itself is never lost: it is always carried
verbatim in the projection's status field and in projection_key.

## Reasons (diagnostics)

Diagnostics are carried in the report half as an immutable canonical form. Sequences
keep their order; mappings become key-sorted (key, value) tuples; scalars are kept
verbatim. No entry is ever dropped. Non-finite floats, non-string mapping keys and
non-JSON-shaped values (sets, bytes, arbitrary objects) fail closed with
OutcomeRecordError rather than being silently stringified.

## Quality pairs

Quality-pair outcomes (win / tie / loss / cannot-assess) may be supplied alongside
the record as qualityPair or quality_pair and are deliberately ignored: they are an
assessment consumer, never a terminal resolution input (contract-probe.md line 81;
reviewed-contract.md line 65). The projection fields and projection_key are identical
with or without them. The projection does not carry quality-pair data at all, so a
pair judgment cannot leak into a training label.

## Parity

projection_key(p) returns a canonical dict over the four identity members plus
status. profileRef in that key is the canonical frozen tuple (profileId, revision,
digest), never a flattened string, so two distinct native ProfileRefs cannot collide
onto one key: assert_projection_parity raises between them.
assert_projection_parity(a, b) compares both the identity/status key and the
canonical mapping body (status, validity, diagnostics, rewardEligible,
trainingLabel, exclusionReason, inDenominator) and raises ProjectionParityError on
any divergence. A deliberately forked projection that flips one mapping cell (for
example trainingLabel 1 -> 0) keeps the same identity key and is still caught, so
the report and reward halves cannot drift apart silently.

## Purity

The module imports only collections.abc and dataclasses. It performs no file,
network, environment, clock or random access. tests/test_outcome_projection.py proves
this three ways: it poisons builtins.open, time.time and random.random while calling
the projection and asserts repeat-call equality; it audits the module AST for
forbidden imports and calls; and it imports the module in a fresh interpreter and
asserts that no repo module — top-level or subpackage under `src/**` — is loaded. A
static import whitelist test also pins the allowed import set.

## Errors

    OutcomeProjectionError(ValueError)   base for every typed failure
    OutcomeRecordError                   not a canonical, closed outcome record
    UnknownOutcomeTokenError             unknown status or validity token
    ProjectionParityError                two projections that must agree diverge

## Run contract

    cd /Users/clchinkc/Documents/GitHub/story-bench
    export PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=$HOME/.cache/uv
    uv run --no-sync --offline python -m pytest -q -p no:cacheprovider tests/test_outcome_projection.py

Offline only. No network, no provider, no dependency change, no background jobs.

## Non-claims

This module does not establish native outcome truth, participant isolation, treatment
separability, provider routes or empirical benefit. It does not compute resolution;
it projects an already-decided native result. A passing suite here does not prove the
native record was correctly decided elsewhere.
