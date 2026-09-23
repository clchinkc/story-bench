"""Offline program accounting. This module does not authorize live provider use.

The trusted host owns the canonical local SQLite file and administrative APIs.
Mark send permission durably before transport; never resend a replayed event.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sqlite3


MICROS_PER_USD = 1_000_000
CAP_MICROS = 1_000 * MICROS_PER_USD
INITIAL_ENVELOPES = dict(repair=100_000_000, data_judge=200_000_000,
                         pilot=150_000_000, main=400_000_000, reserve=150_000_000)
USAGE_KINDS = (
    "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
    "cache_write_tokens", "critic_calls", "retrieval_calls", "selection_calls",
    "retry_calls", "grading_calls", "training_units", "infrastructure_units",
    "agent_units",
)
OUTCOMES = frozenset({"completed", "task_failure", "absent_output", "provider_failure",
                      "infrastructure_failure", "grader_failure", "refused",
                      "truncated", "timeout", "cancelled"})
OPEN_STATES = frozenset({"RESERVED", "SENT", "UNKNOWN"})


class LedgerError(ValueError):
    """An accounting operation failed closed."""


class ValidationError(LedgerError):
    pass


class ConflictError(LedgerError):
    pass


class StateError(LedgerError):
    pass


class BudgetError(LedgerError):
    pass


class DispatchBlocked(LedgerError):
    pass


@dataclass(frozen=True)
class UsageBound:
    kind: str
    max_units: int
    usd_per_unit: str


@dataclass(frozen=True)
class Quote:
    provider: str
    model: str
    snapshot_sha256: str
    observed_at: str
    expires_at: str
    request_sha256: str
    currency: str
    lines: tuple[UsageBound, ...]


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    provider: str
    model: str
    snapshot_sha256: str
    request_sha256: str
    currency: str
    actual_usd: str
    usage: tuple[tuple[str, int | None], ...]
    outcome: str
    evidence_sha256: str
    recorded_at: str


@dataclass(frozen=True)
class Authorization:
    authorization_id: str
    evidence_sha256: str
    reason: str


def _text(value, name):
    if type(value) is not str or not value.strip() or len(value) > 1024:
        raise ValidationError(f"{name} must be nonempty text of at most 1024 characters")
    if any(ord(c) < 32 for c in value):
        raise ValidationError(f"{name} contains control characters")
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError("expected lowercase SHA256")


def _integer(value):
    if type(value) is not int or not 0 <= value <= 10**15:
        raise ValidationError("usage/transfer must be a bounded nonnegative integer")
    return value


def _decimal(value):
    if type(value) not in (str, Decimal):
        raise ValidationError("money requires exact decimal text or Decimal, never float")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValidationError("invalid money") from exc
    if not number.is_finite() or number < 0:
        raise ValidationError("money must be finite and nonnegative")
    # Bound parser resource use, not spend: even 1e100 USD remains recordable truth.
    if len(number.as_tuple().digits) > 200 or abs(number.as_tuple().exponent) > 100:
        raise ValidationError("decimal representation exceeds supported precision")
    return number


def _cost(value, units=1):
    numerator, denominator = _decimal(value).as_integer_ratio()
    numerator *= units * MICROS_PER_USD
    return (numerator + denominator - 1) // denominator


def usd_to_micros(value: str | Decimal) -> int:
    """Round liability upward without depending on the global Decimal context."""
    return _cost(value)


def _timestamp(value):
    _text(value, "timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError("invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValidationError("timestamp must have explicit UTC offset")
    return result


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _authorization(value):
    if type(value) is not Authorization:
        raise ValidationError("parent Authorization required")
    _text(value.authorization_id, "authorization_id")
    _text(value.reason, "reason")
    _sha(value.evidence_sha256)
    return asdict(value)


def _identity(value):
    for name in ("provider", "model"):
        _text(getattr(value, name), name)
    for name in ("snapshot_sha256", "request_sha256"):
        _sha(getattr(value, name))
    if value.currency != "USD":
        raise ValidationError("USD required")


def _quote(value):
    if type(value) is not Quote or type(value.lines) is not tuple:
        raise ValidationError("immutable Quote required")
    _identity(value)
    observed, expires = _timestamp(value.observed_at), _timestamp(value.expires_at)
    if not timedelta(0) < expires - observed <= timedelta(hours=24):
        raise ValidationError("snapshot validity must be positive and at most 24 hours")
    kinds, total = [], 0
    for line in value.lines:
        if type(line) is not UsageBound or type(line.usd_per_unit) is not str:
            raise ValidationError("immutable priced UsageBound required")
        _text(line.kind, "usage kind")
        kinds.append(line.kind)
        total += _cost(line.usd_per_unit, _integer(line.max_units))
    if sorted(kinds) != sorted(USAGE_KINDS):
        raise ValidationError("quote requires exact usage-class census")
    return asdict(value), total


def _fresh(quote):
    if not _timestamp(quote["observed_at"]) <= _now() < _timestamp(quote["expires_at"]):
        raise DispatchBlocked("price snapshot is stale or from the future")


def _receipt(value):
    if type(value) is not Receipt or type(value.usage) is not tuple:
        raise ValidationError("immutable Receipt required")
    _identity(value)
    _text(value.receipt_id, "receipt_id")
    _sha(value.evidence_sha256)
    if _timestamp(value.recorded_at) > _now():
        raise ValidationError("receipt timestamp is in the future")
    if type(value.actual_usd) is not str:
        raise ValidationError("receipt actual_usd requires exact decimal text")
    actual = usd_to_micros(value.actual_usd)
    _text(value.outcome, "outcome")
    if value.outcome not in OUTCOMES:
        raise ValidationError("unsupported outcome")
    kinds = []
    for pair in value.usage:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValidationError("usage requires immutable pairs")
        kind, units = pair
        _text(kind, "usage kind")
        kinds.append(kind)
        if units is not None:
            _integer(units)
    if sorted(kinds) != sorted(USAGE_KINDS):
        raise ValidationError("receipt requires exact usage-class census")
    return asdict(value), actual


def _classification(envelope, role):
    _text(envelope, "envelope")
    _text(role, "role")
    if envelope not in INITIAL_ENVELOPES or role not in ("deployment", "research"):
        raise ValidationError("unknown envelope or cost role")


class Ledger:
    """One canonical program ledger; instances do not keep SQLite connections."""

    def __init__(self, path, *, program_id, store_id):
        self.path = Path(path).resolve()
        self.program_id = _text(program_id, "program_id")
        self.store_id = _text(store_id, "store_id")

    @classmethod
    def create(cls, path, *, program_id, store_id, authorization):
        authorization = _authorization(authorization)
        ledger = cls(path, program_id=program_id, store_id=store_id)
        try:
            descriptor = os.open(ledger.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ConflictError("store already exists; never reset it") from exc
        os.close(descriptor)
        with ledger._transaction(verify=False) as db:
            db.execute("CREATE TABLE program (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
            for table in ("attempts", "external", "incidents"):
                db.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE envelopes (id TEXT PRIMARY KEY, allocation INTEGER NOT NULL)")
            db.execute("CREATE TABLE receipts (id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE events (sequence INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL, recorded_at TEXT NOT NULL)")
            for table in ("events", "program"):
                for operation in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable audit/identity'); END")
            metadata = dict(schema_version=1, program_id=program_id, store_id=store_id,
                            canonical_path=str(ledger.path), cap_micros=CAP_MICROS,
                            currency="USD", micros_per_usd=MICROS_PER_USD)
            db.execute("INSERT INTO program VALUES (1, ?)", (_json(metadata),))
            db.executemany("INSERT INTO envelopes VALUES (?, ?)", INITIAL_ENVELOPES.items())
            unknown = dict(charge_id="hosted_session", envelope="repair", role="research",
                           reason="Hosted-session incremental charge is unresolved",
                           actual_micros=None, receipt=None)
            ledger._put(db, "external", "hosted_session", unknown)
            ledger._append(db, "program-created", "program_created",
                           dict(metadata=metadata, authorization=authorization,
                                envelopes=INITIAL_ENVELOPES, initial_unknown=unknown), metadata)
        return ledger

    @classmethod
    def open(cls, path, *, program_id, store_id):
        ledger = cls(path, program_id=program_id, store_id=store_id)
        with ledger._transaction(write=False):
            pass
        return ledger

    @contextmanager
    def _transaction(self, *, write=True, verify=True):
        db = None
        try:
            db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True,
                                 timeout=5, isolation_level=None)
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if verify:
                row = db.execute("SELECT payload FROM program WHERE id=1").fetchone()
                if row is None:
                    raise StateError("uninitialized store")
                meta = json.loads(row[0])
                expected = dict(schema_version=1, program_id=self.program_id,
                                store_id=self.store_id, canonical_path=str(self.path),
                                cap_micros=CAP_MICROS, currency="USD", micros_per_usd=MICROS_PER_USD)
                if meta != expected:
                    raise ConflictError("program/store/path/schema identity mismatch")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            raise LedgerError(f"SQLite failed closed: {exc}") from exc
        finally:
            if db is not None:
                db.close()  # rolls back an uncommitted transaction, including failures

    @staticmethod
    def _put(db, table, key, value):
        # Table names are internal constants, never caller input.
        db.execute(f"INSERT INTO {table} VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                   (key, _json(value)))

    @staticmethod
    def _get(db, table, key):
        row = db.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()
        if row is None:
            raise StateError(f"unknown {table} ID")
        return json.loads(row[0])

    @staticmethod
    def _rows(db, table):
        return [json.loads(r[0]) for r in db.execute(f"SELECT payload FROM {table} ORDER BY id")]

    @staticmethod
    def _append(db, event_id, kind, payload, result):
        db.execute("INSERT INTO events(event_id, kind, payload, result, recorded_at) VALUES (?, ?, ?, ?, ?)",
                   (event_id, kind, _json(payload), _json(result), _now().isoformat()))

    def _mutate(self, event_id, kind, payload, operation):
        _text(event_id, "event_id")
        encoded = _json(payload)
        with self._transaction() as db:
            prior = db.execute("SELECT kind, payload, result FROM events WHERE event_id=?", (event_id,)).fetchone()
            if prior:
                if prior[:2] != (kind, encoded):
                    raise ConflictError("event_id reused with conflicting input")
                return {**json.loads(prior[2]), "replayed": True}
            result = operation(db)
            self._append(db, event_id, kind, payload, result)
            # Return the same detached JSON shape on the first call and replay.
            return {**json.loads(_json(result)), "replayed": False}

    def _status(self, db):
        envelopes = {key: dict(allocation_micros=value, committed_micros=0, reserved_micros=0)
                     for key, value in db.execute("SELECT id, allocation FROM envelopes ORDER BY id")}
        roles = {role: dict(committed_micros=0, reserved_micros=0, unknown_charge_ids=[])
                 for role in ("deployment", "research")}
        attempts, externals = self._rows(db, "attempts"), self._rows(db, "external")
        unknowns = []
        for row in attempts + externals:
            committed = row["actual_micros"] if row["actual_micros"] is not None else 0
            reserved = row["reserved_micros"] if row.get("state") in OPEN_STATES else 0
            for destination in (envelopes[row["envelope"]], roles[row["role"]]):
                destination["committed_micros"] += committed
                destination["reserved_micros"] += reserved
            if "charge_id" in row and row["actual_micros"] is None:
                unknowns.append(row["charge_id"])
                roles[row["role"]]["unknown_charge_ids"].append(row["charge_id"])
        committed = sum(e["committed_micros"] for e in envelopes.values())
        reserved = sum(e["reserved_micros"] for e in envelopes.values())
        incidents = [r[0] for r in db.execute("SELECT id FROM incidents ORDER BY id")]
        blocked = bool(unknowns or incidents)
        for key, item in envelopes.items():
            item["remaining_micros"] = item["allocation_micros"] - item["committed_micros"] - item["reserved_micros"]
            item["usable_micros"] = 0 if blocked or key == "reserve" else max(0, item["remaining_micros"])
        return dict(program_id=self.program_id, store_id=self.store_id, currency="USD",
                    micros_per_usd=MICROS_PER_USD, cap_micros=CAP_MICROS,
                    committed_micros=committed, reserved_micros=reserved,
                    unknown_charge_ids=sorted(unknowns), incident_ids=incidents,
                    nominal_remaining_micros=CAP_MICROS-committed-reserved,
                    usable_micros=0 if blocked else sum(e["usable_micros"] for e in envelopes.values()),
                    dispatch_blocked=blocked, envelopes=envelopes, roles=roles,
                    attempt_ids=sorted(r["attempt_id"] for r in attempts),
                    uncertain_attempt_ids=sorted(r["attempt_id"] for r in attempts if r["state"] in {"SENT", "UNKNOWN"}))

    def _guard(self, db):
        status = self._status(db)
        if status["dispatch_blocked"]:
            raise DispatchBlocked("unreconciled external charges or permanent incidents")
        return status

    def reserve(self, event_id, *, attempt_id, envelope, role, quote):
        _text(attempt_id, "attempt_id")
        _classification(envelope, role)
        quotation, amount = _quote(quote)
        payload = dict(attempt_id=attempt_id, envelope=envelope, role=role, quote=quotation)

        def operation(db):
            if db.execute("SELECT 1 FROM attempts WHERE id=?", (attempt_id,)).fetchone():
                raise ConflictError("attempt_id already exists")
            status = self._guard(db)
            _fresh(quotation)
            if envelope == "reserve":
                raise BudgetError("reserve requires explicit parent transfer before use")
            if amount > status["nominal_remaining_micros"] or amount > status["envelopes"][envelope]["remaining_micros"]:
                raise BudgetError("reservation exceeds program/envelope allowance")
            row = dict(**payload, state="RESERVED", reserved_micros=amount, actual_micros=None, receipt=None)
            self._put(db, "attempts", attempt_id, row)
            return row
        return self._mutate(event_id, "reserved", payload, operation)

    def _transition(self, event_id, attempt_id, kind, allowed, target, reason=None):
        _text(attempt_id, "attempt_id")
        payload = dict(attempt_id=attempt_id)
        if reason is not None:
            payload["reason"] = _text(reason, "reason")

        def operation(db):
            row = self._get(db, "attempts", attempt_id)
            if row["state"] not in allowed:
                raise StateError(f"cannot {kind} from {row['state']}")
            if target == "SENT":
                self._guard(db)
                _fresh(row["quote"])
            row["state"] = target
            self._put(db, "attempts", attempt_id, row)
            return row
        return self._mutate(event_id, kind, payload, operation)

    def mark_sent(self, event_id, *, attempt_id):
        return self._transition(event_id, attempt_id, "sent", {"RESERVED"}, "SENT")

    def mark_unknown(self, event_id, *, attempt_id, reason):
        _text(reason, "reason")
        return self._transition(event_id, attempt_id, "unknown", {"SENT"}, "UNKNOWN", reason)

    def cancel_unsent(self, event_id, *, attempt_id, reason):
        _text(reason, "reason")
        return self._transition(event_id, attempt_id, "cancelled_unsent", {"RESERVED"}, "CANCELLED", reason)

    @staticmethod
    def _claim_receipt(db, receipt_id):
        if db.execute("SELECT 1 FROM receipts WHERE id=?", (receipt_id,)).fetchone():
            raise ConflictError("receipt_id already reconciled")
        db.execute("INSERT INTO receipts VALUES (?)", (receipt_id,))

    def _incident(self, db, event_id, reasons):
        if reasons:
            self._put(db, "incidents", event_id, dict(incident_id=event_id, reasons=reasons))
            return [event_id]
        return []

    def reconcile(self, event_id, *, attempt_id, receipt):
        _text(attempt_id, "attempt_id")
        actual_receipt, actual = _receipt(receipt)
        payload = dict(attempt_id=attempt_id, receipt=actual_receipt)

        def operation(db):
            row = self._get(db, "attempts", attempt_id)
            if row["state"] not in {"SENT", "UNKNOWN"}:
                raise StateError("reconciliation requires send permission")
            quote = row["quote"]
            for key in ("provider", "model", "snapshot_sha256", "request_sha256", "currency"):
                if actual_receipt[key] != quote[key]:
                    raise ConflictError(f"receipt {key} mismatch")
            self._claim_receipt(db, receipt.receipt_id)
            bounds = {line["kind"]: line["max_units"] for line in quote["lines"]}
            reasons = []
            if actual > row["reserved_micros"]:
                reasons.append("money_overrun")
            if any(units is None for _, units in receipt.usage):
                reasons.append("unreconciled_usage")
            if any(units is not None and units > bounds[kind] for kind, units in receipt.usage):
                reasons.append("usage_overrun")
            row.update(state="RECONCILED", actual_micros=actual, receipt=actual_receipt)
            self._put(db, "attempts", attempt_id, row)
            return dict(**row, incident_ids=self._incident(db, event_id, reasons))
        return self._mutate(event_id, "reconciled", payload, operation)

    def record_external_unknown(self, event_id, *, charge_id, envelope, role, reason):
        _text(charge_id, "charge_id")
        _text(reason, "reason")
        _classification(envelope, role)
        payload = dict(charge_id=charge_id, envelope=envelope, role=role, reason=reason)

        def operation(db):
            if db.execute("SELECT 1 FROM external WHERE id=?", (charge_id,)).fetchone():
                raise ConflictError("charge_id already exists")
            row = dict(**payload, actual_micros=None, receipt=None)
            self._put(db, "external", charge_id, row)
            return row
        return self._mutate(event_id, "external_unknown", payload, operation)

    def reconcile_external(self, event_id, *, charge_id, receipt_id, actual_usd, evidence_sha256, authorization):
        for value, name in ((charge_id, "charge_id"), (receipt_id, "receipt_id")):
            _text(value, name)
        _sha(evidence_sha256)
        if type(actual_usd) is not str:
            raise ValidationError("external actual_usd requires exact decimal text")
        actual = usd_to_micros(actual_usd)
        payload = dict(charge_id=charge_id, receipt_id=receipt_id, actual_usd=actual_usd,
                       evidence_sha256=evidence_sha256, authorization=_authorization(authorization))

        def operation(db):
            row = self._get(db, "external", charge_id)
            if row["actual_micros"] is not None:
                raise StateError("external charge is already final")
            self._claim_receipt(db, receipt_id)
            row.update(actual_micros=actual, receipt=payload)
            self._put(db, "external", charge_id, row)
            status = self._status(db)
            over = status["nominal_remaining_micros"] < 0 or status["envelopes"][row["envelope"]]["remaining_micros"] < 0
            return dict(**row, incident_ids=self._incident(db, event_id, ["external_overrun"] if over else []))
        return self._mutate(event_id, "external_reconciled", payload, operation)

    def transfer(self, event_id, *, source, target, amount_micros, authorization):
        _classification(source, "research")
        _classification(target, "research")
        if source == target or _integer(amount_micros) == 0:
            raise ValidationError("transfer requires distinct envelopes and positive amount")
        payload = dict(source=source, target=target, amount_micros=amount_micros,
                       authorization=_authorization(authorization))

        def operation(db):
            if amount_micros > self._status(db)["envelopes"][source]["remaining_micros"]:
                raise BudgetError("transfer exceeds uncommitted source allocation")
            db.execute("UPDATE envelopes SET allocation=allocation-? WHERE id=?", (amount_micros, source))
            db.execute("UPDATE envelopes SET allocation=allocation+? WHERE id=?", (amount_micros, target))
            return payload
        return self._mutate(event_id, "transferred", payload, operation)

    def status(self):
        with self._transaction(write=False) as db:
            return self._status(db)

    def attempt(self, attempt_id):
        _text(attempt_id, "attempt_id")
        with self._transaction(write=False) as db:
            return self._get(db, "attempts", attempt_id)

    def events(self):
        with self._transaction(write=False) as db:
            return [dict(sequence=seq, event_id=eid, kind=kind, payload=json.loads(payload),
                         result=json.loads(result), recorded_at=when)
                    for seq, eid, kind, payload, result, when in db.execute("SELECT * FROM events ORDER BY sequence")]
