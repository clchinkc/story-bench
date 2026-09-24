"""Pure Story-bench projection of one already-decided native assessment result.

This module performs the single deterministic projection required by the Story-bench
evaluation contract: ONE call returns the report-facing status/diagnostics together
with reward eligibility for the same immutable native outcome record.

It is deliberately NOT a resolver and NOT an authority. It does not decide whether a
story resolved, does not read evidence, does not score prose and does not establish
native truth. Callers pass a canonical outcome record that another component already
decided; this module projects it and fails closed on anything it does not recognize.

Purity: the module imports only 'collections.abc' and 'dataclasses'. It performs no
file, network, environment, clock or random access, so the same input always produces
the same output. 'tests/test_outcome_projection.py' proves this with a poisoned
'builtins.open', a poisoned clock/RNG and a source-level import audit.

Frozen interface (W2C-SCOPE-01):

    project_outcome(record: Mapping) -> OutcomeProjection
    projection_key(p: OutcomeProjection) -> dict
    assert_projection_parity(a, b) -> None

Identity note: 'profileRef' is the native CLOSED mapping
{'profileId': non-blank str, 'revision': positive int, 'digest': 'sha256:<64 lowercase
hex>'} produced by the native ProfileRef.to_dict(), NOT a bare string. It is
canonicalized to the frozen tuple (profileId, revision, digest) in both the dataclass
field and projection_key, so two native profiles that share a profileId yet differ in
revision or digest never collide and assert_projection_parity raises between them. A
bare string, a missing or extra key, a blank profileId, a bool/non-int/non-positive
revision or a non-canonical digest fails closed.

Closed mapping law (validity first, then status; see the docs for the precedence
rationale):

    execution != 'valid' or grading != 'valid'
        -> rewardEligible False, trainingLabel None, exclusionReason = the exact
           invalid validity token(s) joined by '+' in execution-then-grading order
    status 'observation'   -> not eligible, label None, exclusionReason 'operational-mode'
    status 'resolved'      -> eligible, label 1   (requires fully valid execution+grading)
    status 'unresolved'    -> eligible, label 0   (requires fully valid execution+grading)
    status 'unassessable'  -> not eligible, label None, exclusionReason 'unassessable'

Every assigned case keeps 'inDenominator is True' and stays visible; an
unassessable/invalid/not-run case is never converted to 0 or 1 by truthiness.
Quality-pair outcomes supplied alongside (win/tie/loss/cannot-assess) are recognized
and deliberately ignored: a pair judgment is an assessment consumer, never a terminal
resolution input.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PROJECTION_VERSION = "sb-outcome-projection/1"

#: The exactly-four identity members, in canonical order.
IDENTITY_FIELDS = ("evaluationId", "profileRef", "criterionSpecDigest", "projectionVersion")

#: Closed native vocabularies. Anything else fails closed with a typed error.
STATUSES = frozenset({"observation", "resolved", "unresolved", "unassessable"})
EXECUTION_VALIDITY = frozenset({"valid", "provider-invalid", "infrastructure-invalid", "not-run"})
GRADING_VALIDITY = frozenset({"valid", "grader-invalid"})

#: The native ProfileRef is a closed mapping carrying exactly these three keys.
_PROFILE_REF_FIELDS = frozenset({"profileId", "revision", "digest"})
#: A native content address is 'sha256:' + 64 lowercase hex digits. Validated by hand
#: so the module keeps its stdlib-only import whitelist (no 're' import).
_CONTENT_HASH_PREFIX = "sha256:"
_HEX_DIGITS = frozenset("0123456789abcdef")

_QUALITY_PAIR_FIELDS = ("qualityPair", "quality_pair")
_NON_IDENTITY_FIELDS = frozenset({"status", "validity", "reasons"}) | frozenset(_QUALITY_PAIR_FIELDS)

#: Every top-level field a canonical outcome record may carry. Anything else is an
#: extra member and fails closed, so no caller-set resolved flag or reward scalar can
#: enter the projection.
ALLOWED_RECORD_FIELDS = frozenset(IDENTITY_FIELDS) | _NON_IDENTITY_FIELDS

__all__ = [
    "PROJECTION_VERSION",
    "IDENTITY_FIELDS",
    "STATUSES",
    "EXECUTION_VALIDITY",
    "GRADING_VALIDITY",
    "ALLOWED_RECORD_FIELDS",
    "OutcomeProjectionError",
    "OutcomeRecordError",
    "UnknownOutcomeTokenError",
    "ProjectionParityError",
    "OutcomeProjection",
    "project_outcome",
    "projection_key",
    "assert_projection_parity",
]


class OutcomeProjectionError(ValueError):
    """Base class for every typed failure raised by this module."""


class OutcomeRecordError(OutcomeProjectionError):
    """The input is not a canonical, closed outcome record."""


class UnknownOutcomeTokenError(OutcomeRecordError):
    """An outcome status or validity token is outside the closed native vocabulary."""


class ProjectionParityError(OutcomeProjectionError):
    """Two projections that must agree diverge (identity/status or mapping cells)."""


@dataclass(frozen=True, slots=True)
class OutcomeProjection:
    """One immutable object carrying report status/diagnostics AND reward eligibility.

    The first four fields are the exactly-four identity members. 'status' plus
    'executionValidity'/'gradingValidity' and the preserved 'reasons' are the
    report-facing half; 'rewardEligible'/'trainingLabel'/'exclusionReason'/
    'inDenominator' are the reward-facing half. 'reasons' is a canonical, immutable
    projection of the input diagnostics: sequences keep their order and mappings are
    stored as key-sorted (key, value) tuples. No entry is dropped.
    """

    evaluationId: str
    profileRef: tuple
    criterionSpecDigest: str
    projectionVersion: str
    status: str
    executionValidity: str
    gradingValidity: str
    reasons: tuple
    rewardEligible: bool
    trainingLabel: int | None
    exclusionReason: str | None
    inDenominator: bool


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise OutcomeRecordError(f"missing or invalid {name}: {value!r}")
    return value


def _content_hash(value):
    if type(value) is not str or not value.startswith(_CONTENT_HASH_PREFIX):
        return False
    hexdigest = value[len(_CONTENT_HASH_PREFIX):]
    return len(hexdigest) == 64 and all(digit in _HEX_DIGITS for digit in hexdigest)


def _profile_ref(value):
    """Canonicalize the native closed ProfileRef mapping into a frozen tuple.

    Accepts EXACTLY {'profileId': non-blank str, 'revision': positive int (bool
    rejected), 'digest': 'sha256:<64 lowercase hex>'} - the shape emitted by the
    native ProfileRef.to_dict(). Extraction is by key, so key order is irrelevant.
    Anything else fails closed. Because revision and digest stay in the canonical
    tuple, two profiles that share a profileId cannot collide in projection_key.
    """
    if not isinstance(value, Mapping):
        raise OutcomeRecordError(
            f"profileRef must be the native closed mapping, got {type(value).__name__}"
        )
    keys = set(value)
    if keys != _PROFILE_REF_FIELDS:
        raise OutcomeRecordError(
            "profileRef must carry exactly profileId, revision and digest, got "
            f"{sorted(map(repr, keys))}"
        )
    profile_id = value["profileId"]
    if type(profile_id) is not str or not profile_id.strip():
        raise OutcomeRecordError(f"invalid profileRef.profileId: {profile_id!r}")
    revision = value["revision"]
    if type(revision) is not int or revision < 1:
        raise OutcomeRecordError(f"invalid profileRef.revision: {revision!r}")
    digest = value["digest"]
    if not _content_hash(digest):
        raise OutcomeRecordError(f"invalid profileRef.digest: {digest!r}")
    return (profile_id, revision, digest)


def _freeze(value, path):
    """Canonicalize one diagnostic entry into an immutable, deterministic value."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise OutcomeRecordError(f"non-finite number in {path}: {value!r}")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise OutcomeRecordError(f"non-string diagnostic key in {path}: {key!r}")
            pairs.append((key, _freeze(item, f"{path}.{key}")))
        return tuple(sorted(pairs))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise OutcomeRecordError(f"unsupported diagnostic value in {path}: {type(value).__name__}")


def _freeze_reasons(value):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OutcomeRecordError("reasons must be a sequence of diagnostic entries")
    return tuple(_freeze(entry, f"reasons[{index}]") for index, entry in enumerate(value))


def _token(value, name, vocabulary):
    if not isinstance(value, str) or value not in vocabulary:
        raise UnknownOutcomeTokenError(
            f"unknown {name} token {value!r}; expected one of {sorted(vocabulary)}"
        )
    return value


def _project(identity, status, execution, grading, reasons):
    invalid = [token for token in (execution, grading) if token != "valid"]
    if invalid:
        eligible, label, reason = False, None, "+".join(invalid)
    elif status == "observation":
        eligible, label, reason = False, None, "operational-mode"
    elif status == "resolved":
        eligible, label, reason = True, 1, None
    elif status == "unresolved":
        eligible, label, reason = True, 0, None
    elif status == "unassessable":
        eligible, label, reason = False, None, "unassessable"
    else:  # unreachable while STATUSES is the validated vocabulary
        raise UnknownOutcomeTokenError(f"unknown outcome status token {status!r}")
    return OutcomeProjection(
        evaluationId=identity[0],
        profileRef=identity[1],
        criterionSpecDigest=identity[2],
        projectionVersion=identity[3],
        status=status,
        executionValidity=execution,
        gradingValidity=grading,
        reasons=reasons,
        rewardEligible=eligible,
        trainingLabel=label,
        exclusionReason=reason,
        inDenominator=True,
    )


def project_outcome(record):
    """Project one already-decided native outcome record into one OutcomeProjection.

    Pure and total over the closed vocabulary; fails closed with a typed
    OutcomeRecordError (or its UnknownOutcomeTokenError subclass) on anything else.
    """
    if not isinstance(record, Mapping):
        raise OutcomeRecordError(f"record must be a Mapping, got {type(record).__name__}")

    unknown = [key for key in record if key not in ALLOWED_RECORD_FIELDS]
    if unknown:
        raise OutcomeRecordError(f"unknown record member(s): {sorted(map(repr, unknown))}")

    missing = [name for name in IDENTITY_FIELDS if name not in record]
    if missing:
        raise OutcomeRecordError(f"missing identity member(s): {missing}")

    identity = (
        _text(record["evaluationId"], "evaluationId"),
        _profile_ref(record["profileRef"]),
        _text(record["criterionSpecDigest"], "criterionSpecDigest"),
        _text(record["projectionVersion"], "projectionVersion"),
    )
    if identity[3] != PROJECTION_VERSION:
        raise OutcomeRecordError(
            f"foreign projectionVersion {identity[3]!r}; this projection is {PROJECTION_VERSION!r}"
        )

    if "status" not in record:
        raise OutcomeRecordError("missing status member")
    status = _token(record["status"], "outcome status", STATUSES)

    if "validity" not in record:
        raise OutcomeRecordError("missing validity member")
    validity = record["validity"]
    if not isinstance(validity, Mapping):
        raise OutcomeRecordError(f"validity must be a Mapping, got {type(validity).__name__}")
    validity_keys = set(validity)
    if validity_keys != {"execution", "grading"}:
        raise OutcomeRecordError(
            f"validity must carry exactly execution and grading, got {sorted(map(repr, validity_keys))}"
        )
    execution = _token(validity["execution"], "execution validity", EXECUTION_VALIDITY)
    grading = _token(validity["grading"], "grading validity", GRADING_VALIDITY)

    if "reasons" not in record:
        raise OutcomeRecordError("missing reasons member (diagnostics are never dropped)")
    reasons = _freeze_reasons(record["reasons"])

    return _project(identity, status, execution, grading, reasons)


def _require_projection(value, name):
    if not isinstance(value, OutcomeProjection):
        raise OutcomeProjectionError(
            f"{name} must be an OutcomeProjection, got {type(value).__name__}"
        )
    return value


def projection_key(projection):
    """Canonical parity key over the four identity members plus status."""
    p = _require_projection(projection, "projection")
    return {
        "evaluationId": p.evaluationId,
        "profileRef": p.profileRef,
        "criterionSpecDigest": p.criterionSpecDigest,
        "projectionVersion": p.projectionVersion,
        "status": p.status,
    }


def _mapping_body(projection):
    return {
        "status": projection.status,
        "executionValidity": projection.executionValidity,
        "gradingValidity": projection.gradingValidity,
        "reasons": projection.reasons,
        "rewardEligible": projection.rewardEligible,
        "trainingLabel": projection.trainingLabel,
        "exclusionReason": projection.exclusionReason,
        "inDenominator": projection.inDenominator,
    }


def _divergence(left, right):
    keys = sorted(set(left) | set(right))
    return "; ".join(
        f"{key}: {left.get(key)!r} != {right.get(key)!r}"
        for key in keys
        if left.get(key) != right.get(key)
    )


def assert_projection_parity(a, b):
    """Raise ProjectionParityError if two projections diverge on any cell.

    Identity/status is compared through projection_key; the closed mapping cells
    (validity, diagnostics, reward eligibility, label, exclusion, denominator) are
    compared through their canonical body, so a deliberately forked projection that
    flips one mapping cell is caught rather than silently accepted.
    """
    left = _require_projection(a, "a")
    right = _require_projection(b, "b")
    left_key, right_key = projection_key(left), projection_key(right)
    if left_key != right_key:
        raise ProjectionParityError(f"identity/status divergence: {_divergence(left_key, right_key)}")
    left_body, right_body = _mapping_body(left), _mapping_body(right)
    if left_body != right_body:
        raise ProjectionParityError(f"mapping divergence: {_divergence(left_body, right_body)}")
