#!/usr/bin/env python3
"""Collect a pilot's evidence + canonical ledger into the W6 unit JSONL that `analyze_study` reads.

    uv run python scripts/collect_units.py --evidence <dir> --ledger <pilot.sqlite3> \\
        --assignment <w1.json> --out results/revision-v1/units.jsonl

WHAT A COLLECTED UNIT IS. It is UNSCORED, NOT SCORED-ZERO. `resolved` is emitted as null and
`quality` is omitted entirely, because a judge model has not read these generations yet. Writing
`resolved: false` would be a claim the run never made: `study_analysis._resolved_value` maps null to
0.0, so every collected arm would enter the paired difference as a task failure and the D-minus-A
lift would tend to zero for a reason that is an artifact of collection. The same rule is why
`invalid_kind` is carried but `quality` is left out rather than set to `cannot_assess` — the judge,
not this collector, decides whether a generation is assessable.

WHY EACH RULE BELOW EXISTS, in the order the concrete failure it prevents:

  * `cost_usd` is the provider's own `usage.cost`, copied as EXACT DECIMAL TEXT and parsed with
    `parse_float=Decimal` so the text on the wire is never round-tripped through a binary float.
    Recomputing the charge from tokens x prices would be a second opinion about an invoice, and when
    the two disagree nothing in the JSONL would say which number was actually billed. The ledger's
    `actual_micros` is the cross-check; when it disagrees the ledger wins, because the ledger is the
    canonical accounting record, and the disagreement is a finding printed on stderr.

  * `cluster_id` is REFUSED, never invented. The primary interval is a hierarchical PAIRED CLUSTER
    bootstrap: it resamples clusters. A fabricated cluster id fabricates the pairing the analysis
    rests on, and it would do so invisibly — every unit would still parse and the interval would
    still print. When the W1 row does not retain the design's cluster, this program stops and names
    the missing key.

  * `authority_breach` is always false, and that is a limitation rather than a measurement. A
    breach is a write the harness made outside its entitlement; it can only be seen by observing the
    workspace (filesystem, native state) during the episode. The provider's response contains the
    generation text and nothing about what the harness did with it, so deriving the flag from a
    response would be an assertion with no referent.

  * `latency_ms` is 0 when it cannot be derived from a retained timestamp, and the run summary says
    how many arms that happened to. A plausible-looking latency is worse than a stated zero: the
    zero is visibly missing, while an estimate is indistinguishable from a measurement.

Read-only on every input. The only path this program writes is `--out`, and it validates the exact
bytes it is about to write by running them back through `study_analysis.parse_units` first, so a
JSONL that the analyser would reject is never created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import spend_ledger as sl
import study_analysis as sa


PROGRAM_ID = "story-bench"
STORE_ID = "revision-v1"
REQUIRED_ARMS = ("A", "D")

# Receipt outcome -> `invalid_kind`. The mapping is the contract's §7 rule, not this program's
# opinion: a provider/infrastructure/grader failure is not a task-negative label, an absent
# submission by the agent is unresolved rather than dropped, and `truncated`/`refused`/`timeout`
# are properties of the generation the judge will score, so they carry no invalid kind.
INVALID_KIND_BY_OUTCOME = {
    "completed": None,
    "task_failure": None,
    "truncated": None,
    "refused": None,
    "timeout": None,
    "cancelled": None,
    "absent_output": "absent",
    "provider_failure": "provider",
    "infrastructure_failure": "infrastructure",
    "grader_failure": "grader",
}

# Top-level keys only. `manifest.json` nests `snapshot.observed_at`, which is when the PRICE CATALOG
# was observed — subtracting that from anything would produce a latency of days that looks like a
# measurement. Nested keys are therefore never inspected.
_START_KEYS = ("started_at", "sent_at", "request_at")
_END_KEYS = ("finished_at", "completed_at", "response_at", "received_at")
_DURATION_KEYS = ("duration_ms", "latency_ms", "elapsed_ms")


class Refusal(Exception):
    """A condition that makes the collection wrong, reported instead of worked around."""


@dataclass(frozen=True)
class Row:
    """One W1 assignment row, with the design coordinates the analysis needs."""
    index: int
    model: str
    model_snapshot: str
    task_id: str
    task_family: str
    story_id: str
    sample: int
    condition: str
    locale: str
    cluster_id: str

    @property
    def identity(self):
        return (self.model_snapshot, self.task_id, self.sample, self.condition)

    @property
    def unit_key(self):
        return (self.model, self.task_id, self.sample)


@dataclass(frozen=True)
class Arm:
    """One reconciled arm-run: the unit's coordinate plus its observation payload."""
    attempt_id: str
    row: Row
    cost_usd: str
    latency_ms: int
    invalid_kind: str | None
    outcome: str
    retry_index: int
    latency_source: str | None


@dataclass
class Collected:
    units: list
    arms: list
    incomplete: list
    disagreements: list
    skipped: Counter
    outcomes: Counter
    per_model: dict
    latency_underived: int
    superseded: int


# --------------------------------------------------------------------------------- assignment

def _text_field(document, index, key):
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Refusal(f"assignment row {index}: `{key}` must be non-empty text, got {value!r}")
    return value


def _parse_unit_id(value):
    """`en-c00-u3` -> ("en", "en-c00"). The form `dispatch_runner.assign` emits."""
    parts = value.rsplit("-", 2)
    if len(parts) != 3 or not parts[2].startswith("u") or not parts[1].startswith("c"):
        return None
    locale, cluster, _unit = parts
    if not locale or not cluster[1:].isdigit():
        return None
    return locale, f"{locale}-{cluster}"


def _design_from_unit_id(document, locale, cluster):
    """Fill whichever coordinate is absent from the row's own `unit_id`, never overwriting one."""
    unit_id = document.get("unit_id")
    parsed = _parse_unit_id(unit_id) if isinstance(unit_id, str) else None
    if parsed is None:
        return locale, cluster
    return (locale if locale is not None else parsed[0],
            cluster if cluster is not None else parsed[1])


def _design_coordinates(document, index):
    """The (locale, cluster_id) pair, or a refusal that names what the row is missing.

    These two are DESIGN strata, not properties of a task: `pilot_run` declares
    `clusters_per_locale: 2` and `locales: [en, zh-Hant, zh-Hans]`, and the interval resamples
    clusters while the estimand weights locales equally. Neither can be recovered from a task id or
    a model response, so a row that does not carry them cannot produce a usable unit.
    """
    locale = document.get("locale")
    cluster = document.get("cluster_id") or document.get("cluster")
    locale, cluster = _design_from_unit_id(document, locale, cluster)
    problems = []
    if not isinstance(locale, str) or not locale.strip():
        problems.append("`locale` (searched `locale`, `unit_id`)")
    if not isinstance(cluster, str) or not cluster.strip():
        problems.append("`cluster_id` (searched `cluster_id`, `cluster`, `unit_id`)")
    if problems:
        raise Refusal(
            f"assignment row {index} carries no " + " and no ".join(problems) + ". "
            "The pilot design declares locales and clusters, and the analysis resamples clusters "
            "and weights locales equally; inventing either would fabricate the strata the interval "
            "is computed over. Add the missing key(s) to the W1 manifest that the run was bound to.")
    return locale, cluster


def _task_family(document, index, story_id):
    """`task_family` from the retained task document, cross-checked against every other witness.

    Three fields can claim the family: the frozen task's own `task_type`, an explicit
    `task_family`, and the prefix of `story_id` (`beat_interpolation:fantasy`). They are required to
    agree rather than ranked, because a disagreement means the row is internally inconsistent and
    picking a winner would silently choose a stratum.
    """
    task = document.get("task")
    claimed = {}
    if isinstance(task, dict) and isinstance(task.get("task_type"), str):
        claimed["task.task_type"] = task["task_type"]
    if isinstance(document.get("task_family"), str):
        claimed["task_family"] = document["task_family"]
    if ":" in story_id:
        claimed["story_id prefix"] = story_id.split(":", 1)[0]
    if not claimed:
        raise Refusal(
            f"assignment row {index} carries no task family (searched `task.task_type`, "
            f"`task_family`, and the `story_id` prefix of {story_id!r})")
    distinct = set(claimed.values())
    if len(distinct) != 1:
        detail = ", ".join(f"{key}={value!r}" for key, value in claimed.items())
        raise Refusal(f"assignment row {index} disagrees about its task family: {detail}")
    return distinct.pop()


def _row_from_entry(position, entry):
    """One W1 row, with every field the unit object needs and nothing invented."""
    if not isinstance(entry, dict):
        raise Refusal(f"assignment row {position}: object required")
    model = _text_field(entry, position, "model")
    snapshot = _text_field(entry, position, "model_snapshot")
    task_id = _text_field(entry, position, "task_id")
    story_id = _text_field(entry, position, "story_id")
    condition = _text_field(entry, position, "condition")
    if condition not in sa.ARMS:
        raise Refusal(f"assignment row {position}: condition {condition!r} is not one of {sa.ARMS}")
    sample = entry.get("sample")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise Refusal(f"assignment row {position}: `sample` must be a non-negative integer, "
                      f"got {sample!r}")
    locale, cluster = _design_coordinates(entry, position)
    if locale not in sa.LOCALES:
        raise Refusal(f"assignment row {position}: locale {locale!r} is not one of {sa.LOCALES}; a "
                      f"new locale adds a stratum the equal-weight estimand never declared")
    return Row(position, model, snapshot, task_id, _task_family(entry, position, story_id),
               story_id, sample, condition, locale, cluster)


def _register(index, units, row):
    """Index the row, refusing a shadowed identity or a unit whose design coordinates disagree."""
    if row.identity in index:
        raise Refusal(f"assignment rows {index[row.identity].index} and {row.index} share the "
                      f"assignment identity {row.identity}; one row would shadow the other")
    index[row.identity] = row
    existing = units.get(row.unit_key)
    if existing is None:
        units[row.unit_key] = row
        return
    if (existing.locale, existing.cluster_id, existing.task_family,
            existing.story_id) != (row.locale, row.cluster_id, row.task_family, row.story_id):
        raise Refusal(
            f"assignment rows {existing.index} and {row.index} are the same unit {row.unit_key} "
            f"but disagree about its design coordinates "
            f"({(existing.locale, existing.cluster_id)} vs {(row.locale, row.cluster_id)}); "
            f"the A and D arms would be paired into different clusters")


def _read_assignment(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Refusal(f"cannot read assignment manifest {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Refusal(f"assignment manifest {path} is not JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("assignments"), list):
        raise Refusal(f"assignment manifest {path}: an `assignments` array is required")
    if not document["assignments"]:
        raise Refusal(f"assignment manifest {path}: no assignments")
    return document, document["assignments"]


def load_assignment(path):
    """Parse the W1 manifest into rows indexed by assignment identity.

    The index is keyed by (model snapshot, task, sample, arm) because that is the identity
    `AttemptSpec` retains: `sample` is the OCCURRENCE index, so `(task, sample)` names one unit and
    the arm completes the row. Matching on the task id alone would collapse the ~5.76 reuses of each
    frozen task into one row and pair the wrong A with the wrong D.
    """
    document, entries = _read_assignment(path)
    index, units = {}, {}
    for position, entry in enumerate(entries):
        _register(index, units, _row_from_entry(position, entry))
    return document, index


# ------------------------------------------------------------------------------------ evidence

def _read_json(path, what):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Refusal(f"{what} {path} is missing or unreadable: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Refusal(f"{what} {path} is not JSON: {exc}") from exc


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _provider_cost(path):
    """The provider's `usage.cost` as exact Decimal, lifted from the response bytes.

    `parse_float=Decimal` is the point of this function: `json.loads` would put the charge through a
    binary float, and `repr` of that float is not guaranteed to be the decimal the provider wrote.
    The invoice is decimal text; it is kept as decimal text.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Refusal(f"response bytes {path} are unreadable: {exc}") from exc
    try:
        document = json.loads(raw, parse_float=Decimal, parse_int=int)
    except json.JSONDecodeError as exc:
        raise Refusal(f"response bytes {path} are not JSON: {exc}") from exc
    usage = document.get("usage") if isinstance(document, dict) else None
    if not isinstance(usage, dict) or "cost" not in usage:
        raise Refusal(
            f"{path}: the response carries no usage.cost. A reconciled attempt's charge cannot be "
            f"reconstructed from tokens x prices without inventing it, so this arm is not emitted.")
    amount = usage["cost"]
    if isinstance(amount, bool) or not isinstance(amount, (int, Decimal)):
        raise Refusal(f"{path}: usage.cost must be a JSON number, got {type(amount).__name__}")
    cost = Decimal(amount)
    if not cost.is_finite() or cost < 0:
        raise Refusal(f"{path}: usage.cost must be finite and non-negative, got {cost}")
    return cost


def _duration_of(label, document):
    """A recorded duration field, or None. Preferred over a timestamp pair: no clock agreement."""
    for key in _DURATION_KEYS:
        if key not in document:
            continue
        value = document[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Refusal(f"{label}.{key}: a non-negative integer is required, got {value!r}")
        return value, f"{label}.{key}"
    return None


def _interval_of(label, document):
    """A start/end pair ON THE SAME DOCUMENT, or None.

    Never mixed across documents: two files written by different code paths have no shared clock.
    """
    start = next(((key, document[key]) for key in _START_KEYS if key in document), None)
    end = next(((key, document[key]) for key in _END_KEYS if key in document), None)
    if start is None or end is None:
        return None
    try:
        began = datetime.fromisoformat(start[1])
        finished = datetime.fromisoformat(end[1])
    except (TypeError, ValueError) as exc:
        raise Refusal(f"{label}: unparseable timestamp ({exc})") from exc
    milliseconds = (finished - began).total_seconds() * 1000.0
    if milliseconds < 0:
        raise Refusal(f"{label}: {end[0]} precedes {start[0]}; latency cannot be negative")
    return int(round(milliseconds)), f"{label} {start[0]}->{end[0]}"


def _latency(documents):
    """(milliseconds, source) from the first retained document that can answer, else (0, None)."""
    for label, document in documents:
        if not isinstance(document, dict):
            continue
        found = _duration_of(label, document) or _interval_of(label, document)
        if found:
            return found
    return 0, None


def _receipt_of(state):
    receipt = state.get("receipt")
    if not isinstance(receipt, dict):
        raise Refusal(
            f"attempt {state.get('attempt_id')!r}: state {state.get('state')!r} carries no receipt. "
            f"A settlement without a receipt cannot be costed, and the ledger is the only record.")
    return receipt


# ----------------------------------------------------------------------------------- collection

def _ledger_attempt_ids(ledger):
    """Every attempt the ledger has ever touched, in first-event order.

    The attempt ids are taken from event payloads rather than from `status()`, because `status()`
    reports envelopes and totals but not the per-attempt state this program needs, and a settled
    attempt's events are the canonical trace of how it settled.
    """
    seen = {}
    for event in ledger.events():
        payload = event.get("payload") or {}
        attempt_id = payload.get("attempt_id")
        if isinstance(attempt_id, str) and attempt_id:
            seen.setdefault(attempt_id, event["sequence"])
    return [attempt_id for attempt_id, _ in sorted(seen.items(), key=lambda pair: pair[1])]


def _lookup_row(index, spec, attempt_id):
    row = index.get((spec.get("assigned_model_snapshot"), spec.get("task_id"),
                     spec.get("sample"), spec.get("condition")))
    if row is None:
        raise Refusal(
            f"attempt {attempt_id}: no assignment row matches "
            f"(model_snapshot={spec.get('assigned_model_snapshot')!r}, "
            f"task_id={spec.get('task_id')!r}, sample={spec.get('sample')!r}, "
            f"condition={spec.get('condition')!r}). The evidence and the W1 manifest name different "
            f"runs, so the arm cannot be placed in a unit.")
    if spec.get("story_id") != row.story_id:
        raise Refusal(f"attempt {attempt_id}: evidence story_id {spec.get('story_id')!r} disagrees "
                      f"with assignment row {row.index} {row.story_id!r}")
    return row


def _attempt_manifest(directory, attempt_id):
    manifest = _read_json(directory / "manifest.json", "attempt manifest")
    spec = manifest.get("spec") if isinstance(manifest, dict) else None
    if not isinstance(spec, dict):
        raise Refusal(f"attempt {attempt_id}: manifest.json carries no `spec` object")
    if spec.get("attempt_id") != attempt_id:
        raise Refusal(f"attempt {attempt_id}: the evidence directory and manifest.json disagree "
                      f"about the attempt id ({spec.get('attempt_id')!r}); one of them is not this "
                      f"attempt's evidence")
    return manifest, spec


def _provenance(directory, attempt_id, state):
    """The billed bytes, after checking that the metadata and the receipt cover the same ones."""
    response_bin = directory / "response.bin"
    if not response_bin.is_file():
        raise Refusal(f"attempt {attempt_id} is RECONCILED but {response_bin} is absent; the "
                      f"provider's own usage.cost cannot be recovered")
    raw = response_bin.read_bytes()
    metadata = _read_json(directory / "response.json", "response metadata")
    if metadata.get("sha256") != _sha256(raw):
        raise Refusal(f"attempt {attempt_id}: response.json sha256 does not cover response.bin; the "
                      f"retained bytes are not the ones the metadata describes")
    if _receipt_of(state).get("evidence_sha256") != _sha256(raw):
        raise Refusal(f"attempt {attempt_id}: the ledger receipt covers other bytes than "
                      f"response.bin; the response on disk is not the one that was billed")
    return raw


def _cost_text(state, response_bin, attempt_id):
    """(decimal text for the unit, disagreement tuple or None). Ledger wins on disagreement."""
    provider_cost = _provider_cost(response_bin)
    provider_micros = sl.usd_to_micros(provider_cost)
    actual_micros = state.get("actual_micros")
    if isinstance(actual_micros, bool) or not isinstance(actual_micros, int):
        raise Refusal(f"attempt {attempt_id}: the ledger carries no integer actual_micros")
    if provider_micros == actual_micros:
        return str(provider_cost), None
    ledger_text = str(Decimal(actual_micros) / Decimal(sl.MICROS_PER_USD))
    return ledger_text, (attempt_id, str(provider_cost), provider_micros, actual_micros, ledger_text)


def _resolve_attempt(evidence_root, attempt_id, state, index):
    """Read one reconciled attempt into (Arm, disagreement, assignment_hash), or refuse."""
    directory = Path(evidence_root) / "attempts" / attempt_id
    if not directory.is_dir():
        raise Refusal(f"attempt {attempt_id} is RECONCILED in the ledger but its evidence directory "
                      f"{directory} does not exist; a paid arm cannot be collected without it")
    manifest, spec = _attempt_manifest(directory, attempt_id)
    row = _lookup_row(index, spec, attempt_id)
    route = manifest.get("route")
    if isinstance(route, dict) and route.get("model_id") not in (None, row.model):
        raise Refusal(f"attempt {attempt_id}: the evidence route {route.get('model_id')!r} is not "
                      f"the assignment row's model {row.model!r}")

    _provenance(directory, attempt_id, state)
    cost_text, disagreement = _cost_text(state, directory / "response.bin", attempt_id)
    transport = _read_json(directory / "transport.json", "transport record")
    latency_ms, latency_source = _latency([("transport.json", transport),
                                           ("manifest.json", manifest)])
    outcome = _receipt_of(state).get("outcome")
    if outcome not in INVALID_KIND_BY_OUTCOME:
        raise Refusal(f"attempt {attempt_id}: receipt outcome {outcome!r} has no contract mapping; "
                      f"guessing an invalid_kind would put an unlabelled arm into the analysis")
    retry_index = spec.get("retry_index", 0)
    if isinstance(retry_index, bool) or not isinstance(retry_index, int):
        raise Refusal(f"attempt {attempt_id}: retry_index must be an integer")
    arm = Arm(attempt_id, row, cost_text, latency_ms, INVALID_KIND_BY_OUTCOME[outcome], outcome,
              retry_index, latency_source)
    return arm, disagreement, spec.get("assignment_hash")


def _label_for(attempt_id, directory):
    """Best available model label for the skip summary, without inventing one for a broken dir."""
    if directory.is_dir():
        try:
            spec = _read_json(directory / "manifest.json", "attempt manifest").get("spec") or {}
            if spec.get("assigned_model_snapshot"):
                return spec["assigned_model_snapshot"]
        except Refusal:
            pass
    return attempt_id.split(":", 1)[0]


def _walk_ledger(evidence_root, ledger, index):
    """Every ledger attempt, split into resolved arms and counted skips. This is the only loop
    that decides whether an attempt is collected, so the skip reasons are all in one place.
    """
    arm_records, disagreements = [], []
    skipped, outcomes = Counter(), Counter()
    per_model = defaultdict(lambda: dict(reconciled=0, skipped=Counter(), incomplete=0))
    bound_hash = None
    for attempt_id in _ledger_attempt_ids(ledger):
        try:
            state = ledger.attempt(attempt_id)
        except sl.LedgerError as exc:
            raise Refusal(f"attempt {attempt_id}: the ledger cannot describe it: {exc}") from exc
        directory = Path(evidence_root) / "attempts" / attempt_id
        if state.get("state") != "RECONCILED":
            reason = f"state_{str(state.get('state')).lower()}"
            skipped[reason] += 1
            per_model[_label_for(attempt_id, directory)]["skipped"][reason] += 1
            continue
        arm, disagreement, observed = _resolve_attempt(evidence_root, attempt_id, state, index)
        if disagreement:
            disagreements.append(disagreement)
            sys.stderr.write(
                f"cost disagreement: {disagreement[0]} provider usage.cost={disagreement[1]} "
                f"({disagreement[2]} micros) vs ledger actual_micros={disagreement[3]}; "
                f"using the ledger value {disagreement[4]}\n")
        if bound_hash is None:
            bound_hash = observed
        elif observed != bound_hash:
            raise Refusal(f"attempt {attempt_id}: assignment_hash {observed!r} differs from the "
                          f"run's other attempts ({bound_hash!r}); these arms were not bound to one "
                          f"manifest and must not be pooled")
        arm_records.append(arm)
        outcomes[arm.outcome] += 1
        per_model[arm.row.model]["reconciled"] += 1
    return arm_records, disagreements, skipped, outcomes, per_model, bound_hash


def _select_and_group(arm_records):
    """Keep the latest retry per (unit, arm); return chosen arms and the superseded count.

    Retries create a NEW attempt id per attempt, so one (unit, arm) can legitimately hold several
    settlements. The latest retry is the arm; the earlier ones are counted and reported.
    """
    chosen, superseded = {}, 0
    for arm in arm_records:
        key = (arm.row.unit_key, arm.row.condition)
        current = chosen.get(key)
        if current is None:
            chosen[key] = arm
            continue
        superseded += 1
        if (arm.retry_index, arm.attempt_id) > (current.retry_index, current.attempt_id):
            chosen[key] = arm
    return chosen, superseded


def _unit_payload(arms):
    representative = arms[REQUIRED_ARMS[0]].row
    return {
        "cluster_id": representative.cluster_id,
        "model": representative.model,
        "locale": representative.locale,
        "task_family": representative.task_family,
        "sample_id": str(representative.sample),
        "arms": {
            condition: {
                "resolved": None,
                "invalid_kind": arms[condition].invalid_kind,
                "authority_breach": False,
                "cost_usd": arms[condition].cost_usd,
                "latency_ms": arms[condition].latency_ms,
            }
            for condition in sorted(arms)
        },
    }


def _assemble(chosen, per_model):
    """Group chosen arms into units; report units that lost an arm instead of emitting them."""
    grouped = defaultdict(dict)
    for (_unit_key, condition), arm in chosen.items():
        grouped[arm.row.unit_key][condition] = arm
    units, incomplete = [], []
    for unit_key in sorted(grouped):
        arms = grouped[unit_key]
        missing = [name for name in REQUIRED_ARMS if name not in arms]
        first = next(iter(arms.values()))
        if missing:
            incomplete.append((unit_key, missing, first.row))
            per_model[first.row.model]["incomplete"] += 1
            sys.stderr.write(
                f"incomplete unit: model={unit_key[0]} task={unit_key[1]} sample={unit_key[2]} "
                f"locale={first.row.locale} cluster={first.row.cluster_id} "
                f"missing arm(s) {missing}; not emitted\n")
            continue
        units.append(_unit_payload(arms))
    return units, incomplete


def _verify_assignment_hash(assignment_document, observed):
    """Refuse when the evidence was bound to another manifest than the one supplied."""
    from measurement_contract import digest  # local: one call site, keeps import surface small
    supplied = digest(assignment_document)
    if supplied != observed:
        raise Refusal(
            f"the supplied --assignment manifest hashes to {supplied}, but every attempt was bound "
            f"to {observed}. These arm-runs were made against a different manifest, so matching "
            f"rows by (task, sample, arm) would pair evidence with the wrong row.")


def collect(evidence_root, ledger, assignment_document, index):
    """Walk the ledger, resolve every settled attempt, and assemble complete units."""
    arm_records, disagreements, skipped, outcomes, per_model, bound_hash = _walk_ledger(
        evidence_root, ledger, index)
    if arm_records and bound_hash is not None:
        _verify_assignment_hash(assignment_document, bound_hash)
    chosen, superseded = _select_and_group(arm_records)
    units, incomplete = _assemble(chosen, per_model)
    latency_underived = sum(1 for arm in chosen.values() if arm.latency_source is None)
    return Collected(units, list(chosen.values()), incomplete, disagreements, skipped, outcomes,
                     per_model, latency_underived, superseded)


# ---------------------------------------------------------------------------------------- output

def _summary(collected, ledger_total):
    lines = [f"  ledger attempts  : {ledger_total}",
             f"  emitted arms     : {len(collected.arms)}",
             f"  emitted units    : {len(collected.units)}",
             f"  superseded arms  : {collected.superseded} (latest retry kept)",
             f"  incomplete units : {len(collected.incomplete)} (not emitted)",
             f"  cost disagreement: {len(collected.disagreements)} (ledger value used, see stderr)"]
    if collected.skipped:
        detail = ", ".join(f"{reason} {count}" for reason, count in sorted(collected.skipped.items()))
        lines.append(f"  skipped attempts : {sum(collected.skipped.values())} ({detail})")
    else:
        lines.append("  skipped attempts : 0")
    if collected.outcomes:
        detail = ", ".join(f"{name} {count}" for name, count in sorted(collected.outcomes.items()))
        lines.append(f"  receipt outcomes : {detail}")
    if collected.arms:
        lines.append(f"  latency underived: {collected.latency_underived} of "
                     f"{len(collected.arms)} arms recorded 0 — transport.json and manifest.json "
                     f"carry no top-level timestamps, so there is nothing to subtract")
    lines.append("  per-model:")
    for model in sorted(collected.per_model):
        counts = collected.per_model[model]
        skipped = sum(counts["skipped"].values())
        detail = ", ".join(f"{reason} {count}" for reason, count in sorted(counts["skipped"].items()))
        lines.append(f"    {model}: reconciled {counts['reconciled']}, failed/skipped {skipped}"
                     + (f" ({detail})" if detail else "")
                     + f", complete units {counts['reconciled'] // 2}"
                     f", incomplete units {counts['incomplete']}")
    return "\n".join(lines)


def _validate(text):
    """Parse the exact bytes about to be written, through the analyser's own contract."""
    raw_units = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not raw_units:
        raise Refusal("no complete units to write; nothing was emitted")
    try:
        return sa.parse_units(raw_units)
    except sa.ManifestError as exc:
        raise Refusal(f"the collected JSONL would fail study_analysis.parse_units: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", required=True,
                        help="evidence root holding attempts/<attempt_id>/")
    parser.add_argument("--ledger", required=True, help="canonical spend ledger (SQLite)")
    parser.add_argument("--assignment", required=True, help="the W1 assignment manifest JSON")
    parser.add_argument("--out", required=True, help="JSONL path to write; nothing else is written")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing --out; without it an existing file is a refusal")
    args = parser.parse_args(argv)

    try:
        assignment_document, index = load_assignment(args.assignment)
        if not Path(args.evidence).is_dir():
            raise Refusal(f"evidence root {args.evidence} is not a directory")
        try:
            ledger = sl.Ledger.open(args.ledger, program_id=PROGRAM_ID, store_id=STORE_ID)
        except sl.LedgerError as exc:
            raise Refusal(f"cannot open ledger {args.ledger} as {PROGRAM_ID}/{STORE_ID}: "
                          f"{exc}") from exc
        ledger_total = len(_ledger_attempt_ids(ledger))
        collected = collect(args.evidence, ledger, assignment_document, index)
        text = "".join(json.dumps(unit, ensure_ascii=False) + "\n" for unit in collected.units)
        if collected.units:
            _validate(text)
        out = Path(args.out)
        if out.exists() and not args.overwrite:
            raise Refusal(f"{out} already exists; refusing to overwrite a collected dataset "
                          f"(pass --overwrite to replace it deliberately)")
        if not collected.units:
            raise Refusal("no complete units; refusing to write an empty JSONL that the analysis "
                          "would reject as 'no records found'")
        try:
            out.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise Refusal(f"cannot write {out}: {exc}") from exc
        print(_summary(collected, ledger_total))
        print(f"  wrote            : {out} ({len(collected.units)} units)")
        return 0
    except Refusal as exc:
        sys.stderr.write(f"refused: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
