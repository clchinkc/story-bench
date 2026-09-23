"""Synthetic, offline accounting behavior; no real provider or program store."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from spend_ledger import (
    Authorization, BudgetError, CAP_MICROS, ConflictError, DispatchBlocked,
    INITIAL_ENVELOPES, Ledger, LedgerError, OUTCOMES, Quote, Receipt, StateError,
    USAGE_KINDS, UsageBound, ValidationError, usd_to_micros,
)


AUTH = Authorization("synthetic-parent", "a" * 64, "Disposable offline test only")
IDENTITY = dict(program_id="synthetic-program", store_id="synthetic-store")


def quote(dollars="1", **changes):
    now = datetime.now(timezone.utc)
    value = Quote(
        "synthetic-provider", "synthetic-model", "b" * 64,
        (now - timedelta(minutes=1)).isoformat(),
        (now + timedelta(hours=1)).isoformat(), "c" * 64, "USD",
        tuple(UsageBound(kind, 1 if kind == "input_tokens" else 0,
                         dollars if kind == "input_tokens" else "0")
              for kind in USAGE_KINDS),
    )
    return replace(value, **changes)


def receipt(q, dollars="0.50", receipt_id="receipt", **changes):
    value = Receipt(
        receipt_id, q.provider, q.model, q.snapshot_sha256, q.request_sha256,
        "USD", dollars,
        tuple((line.kind, line.max_units) for line in q.lines),
        "completed", "d" * 64, datetime.now(timezone.utc).isoformat(),
    )
    return replace(value, **changes)


def create(path, *, clear_synthetic_unknown=True):
    ledger = Ledger.create(path, **IDENTITY, authorization=AUTH)
    if clear_synthetic_unknown:
        # This is an explicitly synthetic zero-charge receipt, never real billing.
        ledger.reconcile_external(
            "synthetic-hosted-zero", charge_id="hosted_session",
            receipt_id="synthetic-zero-receipt", actual_usd="0",
            evidence_sha256="e" * 64, authorization=AUTH,
        )
    return ledger


@pytest.fixture
def ledger(tmp_path):
    return create(tmp_path / "synthetic.sqlite")


def reserve(ledger, attempt="a", q=None, envelope="repair", role="deployment"):
    return ledger.reserve("reserve-" + attempt, attempt_id=attempt,
                          envelope=envelope, role=role, quote=q or quote())


def sent(ledger, attempt="a", q=None, **kwargs):
    q = q or quote()
    reserve(ledger, attempt, q, **kwargs)
    ledger.mark_sent("send-" + attempt, attempt_id=attempt)
    return q


def snapshot(ledger):
    return ledger.status(), ledger.events(), [ledger.attempt(a) for a in ledger.status()["attempt_ids"]]


def rejects_unchanged(ledger, error, call):
    before = snapshot(ledger)
    with pytest.raises(error):
        call()
    assert snapshot(ledger) == before


@pytest.mark.parametrize("value,expected", [
    ("0", 0), ("0.0000001", 1), ("0.000001", 1),
    ("0.00000100000000000000001", 2), ("99.99999999999999999", 100_000_000),
    (Decimal("1.0000000001"), 1_000_001), ("1e100", 10**106),
])
def test_exact_upward_money_rounding(value, expected):
    with localcontext() as context:
        context.prec = 2
        assert usd_to_micros(value) == expected


@pytest.mark.parametrize("value", [True, False, 1, 0.1, None, "-0.1", "NaN", "Infinity", "bad", "1e101"])
def test_invalid_money_rejected(value):
    with pytest.raises(ValidationError):
        usd_to_micros(value)


def test_individual_usage_line_rounding(ledger):
    q = quote(lines=tuple(UsageBound(kind, 3, "0.0000001") for kind in USAGE_KINDS))
    assert reserve(ledger, q=q)["reserved_micros"] == 13
    assert ledger.status()["reserved_micros"] == 13


def test_initial_unknown_blocks_realistic_dispatch_until_evidenced_resolution(tmp_path):
    ledger = create(tmp_path / "initial.sqlite", clear_synthetic_unknown=False)
    status = ledger.status()
    assert status["cap_micros"] == 1_000_000_000
    assert status["unknown_charge_ids"] == ["hosted_session"]
    assert status["committed_micros"] == 0  # known-only subtotal, not a total bill
    assert status["usable_micros"] == 0
    rejects_unchanged(ledger, DispatchBlocked, lambda: reserve(ledger))
    ledger.reconcile_external("known-hosted", charge_id="hosted_session", receipt_id="hosted-actual",
                              actual_usd="2.10", evidence_sha256="f" * 64, authorization=AUTH)
    assert ledger.status()["committed_micros"] == 2_100_000
    assert ledger.status()["roles"]["research"]["committed_micros"] == 2_100_000
    reserve(ledger)


def test_successful_dispatch_receipt_and_exact_event_attempt_census(ledger):
    q = quote()
    transport_calls = []
    reserve(ledger, q=q)
    permit = ledger.mark_sent("send-a", attempt_id="a")
    if not permit["replayed"]:
        transport_calls.append("a")
    retry = ledger.mark_sent("send-a", attempt_id="a")
    if not retry["replayed"]:
        transport_calls.append("a")
    assert transport_calls == ["a"]
    result = ledger.reconcile("reconcile-a", attempt_id="a", receipt=receipt(q))
    assert result["state"] == "RECONCILED"
    assert result["receipt"]["actual_usd"] == "0.50"
    assert result["reserved_micros"] == 1_000_000  # retained original quote
    status = ledger.status()
    assert (status["committed_micros"], status["reserved_micros"]) == (500_000, 0)
    assert status["attempt_ids"] == ["a"]
    assert status["uncertain_attempt_ids"] == []
    events = ledger.events()
    assert [e["event_id"] for e in events] == [
        "program-created", "synthetic-hosted-zero", "reserve-a", "send-a", "reconcile-a",
    ]
    assert [e["sequence"] for e in events] == list(range(1, 6))
    assert [e["kind"] for e in events] == [
        "program_created", "external_reconciled", "reserved", "sent", "reconciled",
    ]
    result["receipt"]["actual_usd"] = "900"
    events[0]["payload"]["metadata"]["cap_micros"] = 0
    assert ledger.attempt("a")["receipt"]["actual_usd"] == "0.50"
    assert ledger.events()[0]["payload"]["metadata"]["cap_micros"] == CAP_MICROS


def test_envelope_equality_exhaustion_and_unsent_release(ledger):
    q = quote("100")
    reserve(ledger, q=q)
    rejects_unchanged(ledger, BudgetError, lambda: reserve(ledger, "over", quote("0.0000001")))
    result = ledger.cancel_unsent("cancel", attempt_id="a", reason="Never sent")
    assert result["state"] == "CANCELLED"
    assert ledger.status()["reserved_micros"] == 0
    reserve(ledger, "replacement", q)
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reserve(
        "new-event-old-attempt", attempt_id="a", envelope="repair", role="deployment", quote=q))


def test_total_cap_exhaustion_and_authorized_reserve_transfer(ledger):
    rejects_unchanged(ledger, BudgetError,
                      lambda: reserve(ledger, "direct-reserve", envelope="reserve"))
    transfer = dict(source="reserve", target="repair", amount_micros=150_000_000, authorization=AUTH)
    assert ledger.transfer("transfer", **transfer)["replayed"] is False
    assert ledger.transfer("transfer", **transfer)["replayed"] is True
    for envelope, dollars in [("repair", "250"), ("data_judge", "200"), ("pilot", "150"), ("main", "400")]:
        reserve(ledger, envelope, quote(dollars), envelope)
    status = ledger.status()
    assert status["reserved_micros"] == CAP_MICROS
    assert status["nominal_remaining_micros"] == status["usable_micros"] == 0
    rejects_unchanged(ledger, BudgetError, lambda: reserve(ledger, "over", quote("0.0000001")))
    rejects_unchanged(ledger, BudgetError, lambda: ledger.transfer(
        "spent-transfer", source="main", target="repair", amount_micros=1, authorization=AUTH))
    assert sum(e["allocation_micros"] for e in status["envelopes"].values()) == CAP_MICROS


@pytest.mark.parametrize("amount", [0, -1, True, 1.0, "1"])
def test_invalid_transfer_preserves_allocation(ledger, amount):
    rejects_unchanged(ledger, ValidationError, lambda: ledger.transfer(
        "invalid-transfer", source="reserve", target="main", amount_micros=amount, authorization=AUTH))


def test_exact_retries_and_conflicting_event_attempt_cancel_ids(ledger):
    q = quote()
    first = reserve(ledger, q=q)
    again = reserve(ledger, q=q)
    assert again == {**first, "replayed": True}
    rejects_unchanged(ledger, ConflictError, lambda: reserve(ledger, q=replace(q, request_sha256="9" * 64)))
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reserve(
        "different-reserve-event", attempt_id="a", envelope="repair", role="deployment", quote=q))
    cancel = dict(attempt_id="a", reason="Unsent")
    first = ledger.cancel_unsent("cancel", **cancel)
    assert ledger.cancel_unsent("cancel", **cancel) == {**first, "replayed": True}
    rejects_unchanged(ledger, ConflictError, lambda: ledger.cancel_unsent("cancel", attempt_id="a", reason="Changed"))
    rejects_unchanged(ledger, StateError, lambda: ledger.cancel_unsent("other-cancel", **cancel))
    rejects_unchanged(ledger, ConflictError, lambda: ledger.mark_sent("cancel", attempt_id="a"))


def test_receipt_replay_and_global_receipt_uniqueness(ledger):
    q = sent(ledger)
    r = receipt(q)
    first = ledger.reconcile("reconcile", attempt_id="a", receipt=r)
    assert ledger.reconcile("reconcile", attempt_id="a", receipt=r) == {**first, "replayed": True}
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reconcile(
        "reconcile", attempt_id="a", receipt=replace(r, actual_usd="0.60")))
    q2 = sent(ledger, "b")
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reconcile("reconcile-b", attempt_id="b", receipt=receipt(q2)))
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reconcile(
        "external-receipt-reuse", attempt_id="b", receipt=receipt(q2, receipt_id="synthetic-zero-receipt")))
    ledger.record_external_unknown("external", charge_id="infra", envelope="repair", role="research", reason="Invoice pending")
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reconcile_external(
        "external-duplicate", charge_id="infra", receipt_id="receipt", actual_usd="1", evidence_sha256="e" * 64, authorization=AUTH))
    assert ledger.status()["reserved_micros"] == 1_000_000


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_all_receipt_outcomes_preserve_billed_attempt_and_role(ledger, outcome):
    q = sent(ledger, role="research")
    ledger.reconcile("final", attempt_id="a", receipt=receipt(q, outcome=outcome))
    assert ledger.attempt("a")["receipt"]["outcome"] == outcome
    assert ledger.status()["roles"]["research"]["committed_micros"] == 500_000
    assert ledger.status()["roles"]["deployment"]["committed_micros"] == 0
    assert ledger.status()["attempt_ids"] == ["a"]


def test_timeout_retains_liability_until_known_receipt(ledger):
    q = sent(ledger, q=quote("100"))
    first = ledger.mark_unknown("timeout", attempt_id="a", reason="Transport timed out; billing unknown")
    assert ledger.mark_unknown("timeout", attempt_id="a", reason="Transport timed out; billing unknown") == {**first, "replayed": True}
    assert ledger.status()["reserved_micros"] == 100_000_000
    rejects_unchanged(ledger, BudgetError, lambda: reserve(ledger, "after-timeout"))
    rejects_unchanged(ledger, StateError, lambda: ledger.cancel_unsent("cancel-sent", attempt_id="a", reason="Local cancellation"))
    rejects_unchanged(ledger, StateError, lambda: ledger.mark_unknown("again", attempt_id="a", reason="Still unknown"))
    assert ledger.status()["uncertain_attempt_ids"] == ["a"]
    ledger.reconcile("billed-timeout", attempt_id="a", receipt=receipt(q, "1.23456789", outcome="timeout"))
    assert ledger.status()["reserved_micros"] == 0
    assert ledger.status()["committed_micros"] == 1_234_568
    assert ledger.attempt("a")["receipt"]["actual_usd"] == "1.23456789"


def test_sent_cancellation_retains_liability_and_zero_receipt_can_release(ledger):
    q = sent(ledger)
    rejects_unchanged(ledger, StateError, lambda: ledger.cancel_unsent("cancel", attempt_id="a", reason="Cancelled locally"))
    assert ledger.status()["reserved_micros"] == 1_000_000
    ledger.mark_unknown("cancelled-locally", attempt_id="a", reason="Cancelled transport; charge unavailable")
    ledger.reconcile("unbilled", attempt_id="a", receipt=receipt(q, "0", outcome="cancelled"))
    assert ledger.status()["committed_micros"] == ledger.status()["reserved_micros"] == 0


@pytest.mark.parametrize("mode,expected", [
    ("money", "money_overrun"), ("usage", "usage_overrun"), ("unknown", "unreconciled_usage"),
])
def test_truthful_receipt_incident_permanently_blocks_reserve_and_send(ledger, mode, expected):
    q = sent(ledger)
    reserve(ledger, "waiting")
    usage = dict((line.kind, line.max_units) for line in q.lines)
    if mode == "usage":
        usage["reasoning_tokens"] = 1
    if mode == "unknown":
        usage["input_tokens"] = None
    dollars = "1001.0000001" if mode == "money" else "0.123456789"
    result = ledger.reconcile("truth", attempt_id="a", receipt=receipt(q, dollars, usage=tuple(usage.items())))
    assert result["receipt"]["actual_usd"] == dollars
    assert dict(result["receipt"]["usage"]) == usage
    assert result["actual_micros"] == usd_to_micros(dollars)
    assert result["incident_ids"] == ["truth"]
    # The named reason is persisted alongside the same-transaction receipt/event.
    with sqlite3.connect(ledger.path) as db:
        assert expected in db.execute("SELECT payload FROM incidents WHERE id='truth'").fetchone()[0]
    assert ledger.status()["committed_micros"] == usd_to_micros(dollars)
    assert ledger.status()["reserved_micros"] == 1_000_000
    ledger = Ledger.open(ledger.path, **IDENTITY)
    rejects_unchanged(ledger, DispatchBlocked, lambda: reserve(ledger, "blocked"))
    rejects_unchanged(ledger, DispatchBlocked, lambda: ledger.mark_sent("blocked-send", attempt_id="waiting"))
    assert ledger.status()["usable_micros"] == 0


def test_unknown_external_gate_blocks_both_reservation_and_send(ledger):
    reserve(ledger)
    pending = dict(charge_id="unknown-infra", envelope="repair", role="research", reason="No invoice")
    first = ledger.record_external_unknown("external", **pending)
    assert ledger.record_external_unknown("external", **pending) == {**first, "replayed": True}
    rejects_unchanged(ledger, ConflictError, lambda: ledger.record_external_unknown("external-2", **pending))
    rejects_unchanged(ledger, DispatchBlocked, lambda: reserve(ledger, "b"))
    rejects_unchanged(ledger, DispatchBlocked, lambda: ledger.mark_sent("send-a", attempt_id="a"))
    actual = dict(charge_id="unknown-infra", receipt_id="external-receipt", actual_usd="0.1",
                  evidence_sha256="e" * 64, authorization=AUTH)
    first = ledger.reconcile_external("known", **actual)
    assert ledger.reconcile_external("known", **actual) == {**first, "replayed": True}
    assert ledger.status()["unknown_charge_ids"] == []
    assert ledger.status()["roles"]["research"]["committed_micros"] == 100_000
    ledger.mark_sent("send-a", attempt_id="a")


def test_external_overrun_preserves_invoice_and_existing_liability(ledger):
    reserve(ledger, q=quote("99"))
    ledger.record_external_unknown("external", charge_id="infra", envelope="repair", role="research", reason="Pending")
    result = ledger.reconcile_external("external-final", charge_id="infra", receipt_id="infra-receipt",
                                       actual_usd="1001", evidence_sha256="e" * 64, authorization=AUTH)
    assert result["incident_ids"] == ["external-final"]
    assert result["receipt"]["actual_usd"] == "1001"
    status = ledger.status()
    assert (status["committed_micros"], status["reserved_micros"], status["nominal_remaining_micros"]) == (1_001_000_000, 99_000_000, -100_000_000)
    assert status["dispatch_blocked"] is True
    rejects_unchanged(ledger, DispatchBlocked, lambda: reserve(ledger, "b"))


@pytest.mark.parametrize("field,value", [
    ("provider", "other"), ("model", "other"), ("snapshot_sha256", "1" * 64), ("request_sha256", "2" * 64),
])
def test_mismatched_receipt_identity_cannot_release_reservation(ledger, field, value):
    q = sent(ledger)
    rejects_unchanged(ledger, ConflictError, lambda: ledger.reconcile(
        "wrong", attempt_id="a", receipt=replace(receipt(q), **{field: value})))
    assert ledger.status()["reserved_micros"] == 1_000_000


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "extra", "unknown-price", "float-price", "negative", "bool", "unbounded", "mutable", "bad-hash", "currency", "future", "expired", "long-lived", "naive",
])
def test_invalid_quotes_are_atomic_denials(ledger, mutation):
    q = quote()
    lines = q.lines
    changes = {
        "missing": {"lines": lines[:-1]}, "duplicate": {"lines": lines + (lines[0],)},
        "extra": {"lines": lines + (UsageBound("unqualified", 1, "1"),)},
        "unknown-price": {"lines": (replace(lines[0], usd_per_unit=None),) + lines[1:]},
        "float-price": {"lines": (replace(lines[0], usd_per_unit=0.1),) + lines[1:]},
        "negative": {"lines": (replace(lines[0], max_units=-1),) + lines[1:]},
        "bool": {"lines": (replace(lines[0], max_units=True),) + lines[1:]},
        "unbounded": {"lines": (replace(lines[0], max_units=None),) + lines[1:]},
        "mutable": {"lines": list(lines)}, "bad-hash": {"snapshot_sha256": "missing"},
        "currency": {"currency": "EUR"},
        "future": {"observed_at": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()},
        "expired": {"expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()},
        "long-lived": {"expires_at": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()},
        "naive": {"observed_at": "2026-09-23T00:00:00"},
    }[mutation]
    rejects_unchanged(ledger, LedgerError, lambda: reserve(ledger, q=replace(q, **changes)))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "bool", "negative", "mutable", "unknown-money", "float-money", "future", "outcome", "currency"])
def test_invalid_receipts_cannot_discard_liability(ledger, mutation):
    q = sent(ledger)
    r = receipt(q)
    changes = {
        "missing": {"usage": r.usage[:-1]}, "duplicate": {"usage": r.usage + (r.usage[0],)},
        "extra": {"usage": r.usage + (("unknown", 0),)},
        "bool": {"usage": (("input_tokens", True),) + r.usage[1:]},
        "negative": {"usage": (("input_tokens", -1),) + r.usage[1:]},
        "mutable": {"usage": list(r.usage)}, "unknown-money": {"actual_usd": None},
        "float-money": {"actual_usd": 0.5}, "future": {"recorded_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
        "outcome": {"outcome": "not-run"}, "currency": {"currency": "EUR"},
    }[mutation]
    rejects_unchanged(ledger, ValidationError, lambda: ledger.reconcile("invalid", attempt_id="a", receipt=replace(r, **changes)))


def test_price_expiry_rechecked_before_initial_send_but_replay_does_not_resend(ledger, monkeypatch):
    import spend_ledger
    q = quote()
    reserve(ledger, q=q)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(spend_ledger, "_now", lambda: now + timedelta(hours=2))
    rejects_unchanged(ledger, DispatchBlocked, lambda: ledger.mark_sent("send-a", attempt_id="a"))
    monkeypatch.setattr(spend_ledger, "_now", lambda: now)
    ledger.mark_sent("send-a", attempt_id="a")
    monkeypatch.setattr(spend_ledger, "_now", lambda: now + timedelta(days=1))
    assert ledger.mark_sent("send-a", attempt_id="a")["replayed"] is True
    assert reserve(ledger, q=q)["replayed"] is True
    assert ledger.status()["reserved_micros"] == 1_000_000


def _all_tables(ledger):
    with sqlite3.connect(ledger.path) as db:
        return list(db.iterdump())


@pytest.mark.parametrize("observed,expires", [
    ("9999-12-31T00:00:00+00:00", "9999-12-31T01:00:00+00:00"),
    ("9999-12-31T22:00:00+00:00", "9999-12-31T23:00:00+00:00"),
], ids=["code-review-repro", "security-review-repro"])
def test_extreme_future_quote_rejects_through_ledger_error_without_writes(ledger, observed, expires):
    before = _all_tables(ledger)
    try:
        with pytest.raises(DispatchBlocked):
            reserve(ledger, q=quote(observed_at=observed, expires_at=expires))
    finally:
        assert _all_tables(ledger) == before


@pytest.mark.parametrize("duration,accepted", [
    (timedelta(hours=24), True),
    (timedelta(hours=24, microseconds=1), False),
    (timedelta(0), False),
    (timedelta(microseconds=-1), False),
], ids=["exact-24h", "24h-plus-microsecond", "zero", "reversed"])
def test_snapshot_duration_boundaries(ledger, monkeypatch, duration, accepted):
    import spend_ledger
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(spend_ledger, "_now", lambda: now)
    q = quote(observed_at=now.isoformat(), expires_at=(now + duration).isoformat())
    before = _all_tables(ledger)
    if accepted:
        reserve(ledger, q=q)
        assert ledger.mark_sent("send-a", attempt_id="a")["replayed"] is False
    else:
        try:
            with pytest.raises(ValidationError):
                reserve(ledger, q=q)
        finally:
            assert _all_tables(ledger) == before


@pytest.mark.parametrize("observed,expires,clock,accepted", [
    ("0001-01-01T00:00:00+00:00", "0001-01-02T00:00:00+00:00", "0001-01-01T12:00:00+00:00", True),
    ("9999-12-31T22:00:00+00:00", "9999-12-31T23:59:59.999999+00:00", "9999-12-31T23:00:00+00:00", True),
    ("0001-01-01T00:00:00+00:00", "9999-12-31T23:59:59.999999+00:00", "2026-09-23T00:00:00+00:00", False),
], ids=["minimum-year", "maximum-year", "whole-supported-range-too-long"])
def test_snapshot_duration_across_supported_datetime_range(ledger, monkeypatch, observed, expires, clock, accepted):
    import spend_ledger
    monkeypatch.setattr(spend_ledger, "_now", lambda: datetime.fromisoformat(clock))
    q = quote(observed_at=observed, expires_at=expires)
    before = _all_tables(ledger)
    if accepted:
        reserve(ledger, q=q)
        assert ledger.mark_sent("send-a", attempt_id="a")["state"] == "SENT"
    else:
        try:
            with pytest.raises(ValidationError):
                reserve(ledger, q=q)
        finally:
            assert _all_tables(ledger) == before


def test_exact_snapshot_expiry_blocks_reserve_and_send_without_writes(ledger, monkeypatch):
    import spend_ledger
    observed = datetime(2026, 9, 23, tzinfo=timezone.utc)
    expires = observed + timedelta(hours=24)
    q = quote(observed_at=observed.isoformat(), expires_at=expires.isoformat())
    monkeypatch.setattr(spend_ledger, "_now", lambda: observed)
    reserve(ledger, "before", q)
    reserve(ledger, "at", q)
    monkeypatch.setattr(spend_ledger, "_now", lambda: expires - timedelta(microseconds=1))
    assert ledger.mark_sent("send-before", attempt_id="before")["state"] == "SENT"
    monkeypatch.setattr(spend_ledger, "_now", lambda: expires)
    before = _all_tables(ledger)
    with pytest.raises(DispatchBlocked):
        ledger.mark_sent("send-at", attempt_id="at")
    assert _all_tables(ledger) == before
    with pytest.raises(DispatchBlocked):
        reserve(ledger, "new", q)
    assert _all_tables(ledger) == before


@pytest.mark.parametrize("state,operation", [
    ("RESERVED", "reconcile"), ("RESERVED", "unknown"), ("SENT", "sent"),
    ("SENT", "cancel"), ("CANCELLED", "sent"), ("CANCELLED", "reconcile"),
    ("RECONCILED", "cancel"), ("RECONCILED", "reconcile"),
])
def test_impossible_transitions_do_not_append_or_partially_write(ledger, state, operation):
    q = quote()
    reserve(ledger, q=q)
    if state in {"SENT", "RECONCILED"}:
        ledger.mark_sent("sent", attempt_id="a")
    if state == "CANCELLED":
        ledger.cancel_unsent("cancelled", attempt_id="a", reason="Unsent")
    if state == "RECONCILED":
        ledger.reconcile("reconciled", attempt_id="a", receipt=receipt(q))
    calls = {
        "sent": lambda: ledger.mark_sent("invalid", attempt_id="a"),
        "cancel": lambda: ledger.cancel_unsent("invalid", attempt_id="a", reason="Unsent"),
        "unknown": lambda: ledger.mark_unknown("invalid", attempt_id="a", reason="No receipt"),
        "reconcile": lambda: ledger.reconcile("invalid", attempt_id="a", receipt=receipt(q, receipt_id="new")),
    }
    rejects_unchanged(ledger, StateError, calls[operation])


def test_creation_and_open_never_reset_or_accept_copied_identity(ledger, tmp_path):
    reserve(ledger)
    before = snapshot(ledger)
    with pytest.raises(ConflictError):
        Ledger.create(ledger.path, **IDENTITY, authorization=AUTH)
    for key in IDENTITY:
        with pytest.raises(ConflictError):
            Ledger.open(ledger.path, **{**IDENTITY, key: "other"})
    copied = tmp_path / "copied.sqlite"
    shutil.copyfile(ledger.path, copied)
    with pytest.raises(ConflictError):
        Ledger.open(copied, **IDENTITY)
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(LedgerError):
        Ledger.open(missing, **IDENTITY)
    assert not missing.exists()
    assert snapshot(Ledger.open(ledger.path, **IDENTITY)) == before


def test_frozen_inputs_and_sql_audit_immutability(ledger):
    with pytest.raises(FrozenInstanceError):
        quote().model = "changed"
    for table in ("events", "program"):
        with sqlite3.connect(ledger.path) as db:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"DELETE FROM {table}")
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"UPDATE {table} SET payload='{{}}'")
    assert ledger.status()["cap_micros"] == CAP_MICROS


def test_transaction_failure_rolls_back_receipt_claim_charge_and_event(ledger, monkeypatch):
    q = sent(ledger)
    before = snapshot(ledger)
    original = Ledger._append

    def fail_append(*args):
        raise sqlite3.OperationalError("synthetic crash before event commit")

    monkeypatch.setattr(Ledger, "_append", staticmethod(fail_append))
    with pytest.raises(LedgerError):
        ledger.reconcile("final", attempt_id="a", receipt=receipt(q))
    assert snapshot(ledger) == before
    monkeypatch.setattr(Ledger, "_append", staticmethod(original))
    ledger.reconcile("final", attempt_id="a", receipt=receipt(q))
    assert ledger.status()["committed_micros"] == 500_000


def test_lock_contention_fails_closed_without_any_permission(ledger):
    before = snapshot(ledger)
    with sqlite3.connect(ledger.path, isolation_level=None) as owner:
        owner.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(LedgerError, match="locked"):
                reserve(ledger)
        finally:
            owner.rollback()
    assert snapshot(ledger) == before


def test_blocked_ledger_still_reconciles_other_known_charges(ledger):
    qa, qb = sent(ledger), sent(ledger, "b", role="research")
    ledger.reconcile("incident", attempt_id="a", receipt=receipt(qa, "2"))
    ledger.reconcile("other-truth", attempt_id="b", receipt=receipt(qb, "0.75", receipt_id="receipt-b"))
    status = ledger.status()
    assert status["committed_micros"] == 2_750_000
    assert status["reserved_micros"] == 0
    assert status["dispatch_blocked"] is True
    assert status["roles"]["deployment"]["committed_micros"] == 2_000_000
    assert status["roles"]["research"]["committed_micros"] == 750_000


@pytest.mark.parametrize("field,value", [("envelope", []), ("role", {}), ("envelope", "absent"), ("role", "grader")])
def test_invalid_classification_fails_closed(ledger, field, value):
    values = dict(envelope="repair", role="research")
    values[field] = value
    rejects_unchanged(ledger, ValidationError, lambda: ledger.reserve(
        "invalid", attempt_id="invalid", quote=quote(), **values))


def test_parent_authorization_is_required_for_administration(ledger):
    rejects_unchanged(ledger, ValidationError, lambda: ledger.transfer(
        "invalid", source="reserve", target="repair", amount_micros=1, authorization=None))
    ledger.record_external_unknown("external", charge_id="pending", envelope="repair", role="research", reason="Pending invoice")
    rejects_unchanged(ledger, ValidationError, lambda: ledger.reconcile_external(
        "no-authorization", charge_id="pending", receipt_id="zero", actual_usd="0", evidence_sha256="e" * 64,
        authorization=replace(AUTH, evidence_sha256="unverified")))


def _race_worker(path, ready, start, output, number):
    try:
        ledger = Ledger.open(path, **IDENTITY)
        ready.put(number)
        if not start.wait(10):
            raise RuntimeError("race start timed out")
        attempt = f"worker-{number}"
        reserve(ledger, attempt, quote("100"))
        permit = ledger.mark_sent("send-" + attempt, attempt_id=attempt)
        output.put(("dispatch", attempt, permit["replayed"]))
    except BudgetError:
        output.put(("denied", f"worker-{number}", False))
    except Exception as exc:
        output.put(("error", type(exc).__name__, str(exc)))


def test_independent_processes_cannot_dispatch_same_last_allowance(ledger):
    context = multiprocessing.get_context("spawn")
    ready, output, start = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_race_worker, args=(str(ledger.path), ready, start, output, n)) for n in range(2)]
    try:
        for worker in workers:
            worker.start()
        assert sorted(ready.get(timeout=15) for _ in workers) == [0, 1]
        start.set()
        results = [output.get(timeout=15) for _ in workers]
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
        assert sorted(result[0] for result in results) == ["denied", "dispatch"], results
        dispatched = [result[1] for result in results if result[0] == "dispatch"]
        assert ledger.status()["attempt_ids"] == dispatched
        assert ledger.status()["reserved_micros"] == 100_000_000
        assert ledger.status()["envelopes"]["repair"]["remaining_micros"] == 0
        assert len(ledger.events()) == 4
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
        for queue in (ready, output):
            queue.close()
            queue.join_thread()


def _crash_worker(path, stage):
    ledger = Ledger.open(path, **IDENTITY)
    reserve(ledger, "crashed", quote("100"))
    if stage != "before-marker":
        ledger.mark_sent("send-crashed", attempt_id="crashed")
    if stage == "after-transport":
        Path(path + ".synthetic-transport").write_text("crashed")
    os._exit(23)


@pytest.mark.parametrize("stage", ["before-marker", "after-marker", "after-transport"])
def test_process_crash_retains_durable_liability(ledger, stage):
    context = multiprocessing.get_context("spawn")
    worker = context.Process(target=_crash_worker, args=(str(ledger.path), stage))
    worker.start()
    try:
        worker.join(15)
        assert worker.exitcode == 23
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
    ledger = Ledger.open(ledger.path, **IDENTITY)
    assert ledger.status()["reserved_micros"] == 100_000_000
    rejects_unchanged(ledger, BudgetError, lambda: reserve(ledger, "after-crash"))
    if stage == "before-marker":
        ledger.cancel_unsent("recover-unsent", attempt_id="crashed", reason="No committed send marker")
        assert ledger.status()["reserved_micros"] == 0
    else:
        rejects_unchanged(ledger, StateError, lambda: ledger.cancel_unsent(
            "unsafe-refund", attempt_id="crashed", reason="Process exited"))
        assert ledger.mark_sent("send-crashed", attempt_id="crashed")["replayed"] is True
        assert ledger.status()["reserved_micros"] == 100_000_000
    assert Path(str(ledger.path) + ".synthetic-transport").exists() == (stage == "after-transport")
