#!/usr/bin/env python3
"""Score collected outputs with the judge model — plan it, rehearse it, and only then spend.

WHY THIS FILE EXISTS, AND WHY IT IS THREE MODES. Generation already has `paid_attempt.prepared`
(build and refuse for free) and `provider_attempts.execute_attempt` (reserve, mark sent, transport,
reconcile). A judge phase is the same kind of purchase, so it gets the same shape and one extra
gate: `paid_attempt.py` can spend on its first line of output, while this file's default mode
prints what it would do. A tool whose dry run is a later flag on the spending path is a tool where
a typo becomes an invoice.

WHAT A JUDGEMENT IS. One paired call: both arms of one (model, task, sample) unit, scored against
the task's own `evaluation_criteria`. It is paired because the analysis's quality field is
`Q=(wins+0.5*ties)/valid_assessed_pairs` — a *preference between two submissions*. Asking a judge
to emit `win`/`loss` for a single submission would force it to invent a comparator, which is the
same sin as inventing the rubric. One slot in the ledger, one evidence directory, one receipt.

THE FOUR REFUSALS, AND THE FAILURE EACH ONE PREVENTS

  * No `evaluation_criteria` — refuse that task, by id. A judge given an invented rubric is a
    different instrument, and its numbers cannot be pooled with the study's declared estimand.
  * An unparseable verdict — the judgement FAILED (`invalid_kind: grader`), never a default score.
    `src/story_rl/assessor.py` records the same rule from the same scare: a judge that cannot
    answer in the required JSON is a broken instrument, and scoring it 0 makes a bad judge look
    like bad prose.
  * A consumed ledger slot — skipped, not resent. `Ledger.reserve` refuses a duplicate attempt_id
    and an UNKNOWN attempt can never be re-sent; a retry is a new id (`--unit`), never a replay.
  * A reservation the `data_judge` envelope cannot cover — the plan refuses before any send.

THE CONFOUND, RECORDED RATHER THAN REFUSED. In this study the judge is drawn from the same nine
routes that generate the outputs, because those are the only qualified routes. That is a declared
design fact, not an accident: the judge's snapshot is one of the generation subjects, so a judge
that prefers its own prose is indistinguishable from a model that writes better prose. The script
therefore does not refuse on the overlap; it REQUIRES the overlap to travel inside every
judgement's evidence, together with a citation to where the study discloses it (see
`confound_record`). A refusal here would block the run the owner already froze; silence would put
an unlabelled confound into a paid measurement.

WHAT IS PRICED. `prepare_attempt` derives the usage census and the price from the judge route's own
retained record (`_census` over the snapshot's rate components) — the same mechanism the generation
path uses. This file never carries a price of its own; the plan's total is the sum of the census
lines the ledger will actually reserve.

    # what would be judged, how many calls, the reservation — sends nothing
    uv run python scripts/judge_units.py --plan \\
        --manifest /tmp/assignment.json --base <route-evidence> \\
        --ledger <pilot.sqlite3> --evidence <dir>

    # every request body built and validated; still sends nothing
    uv run python scripts/judge_units.py --dry-run ... (same flags)

    # the real thing; the token is deliberately NOT the generation token
    uv run python scripts/judge_units.py --execute \\
        --authorise PAID-JUDGE-2026-11-04 ... (same flags)
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import datetime
from decimal import Decimal
import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import paid_attempt as paid  # noqa: E402
import provider_attempts as pa  # noqa: E402
import provider_prices as pp  # noqa: E402
import rehearse_attempt as reh  # noqa: E402
import spend_ledger as sl  # noqa: E402
from measurement_contract import PROTOCOL, digest  # noqa: E402

PROGRAM_ID = reh.PROGRAM_ID
STORE_ID = reh.STORE_ID

# The judge draws on its own envelope and its own cost role. `provider_attempts.ROLES` maps
# `research_evaluation` to `research`, which is what `_validate_spec` re-derives; a judge call
# booked as `deployment/generation` would be invisible to the critic/grading line of the analysis.
JUDGE_ENVELOPE = "data_judge"
JUDGE_ROLE = "research"
JUDGE_ACTIVITY = "research_evaluation"

# Only these can produce an output that is a *subject* of judgement.
GENERATION_ENVELOPES = ("pilot", "main")
GENERATION_ROLE = "deployment"

# A verdict has to fit: two submissions x up to 13 criteria x a short basis, plus the object.
#
# 3000 was too small, and the failure was measured rather than suspected. On the 2026-11-09 clean1
# run, 12 of 15 judge failures were `finish_reason == "length"` -- the reply was cut off mid-object,
# so a paid call produced no verdict at all. 8000 fixed 19 of 45 such judgements and still truncated
# 16, which is the evidence for this value: on this route the reply plus its reasoning does not
# reliably fit below ~10000. The route publishes `max_completion_tokens` 65536, so 16000 is inside the
# bound the provider actually declares and inside the reservation's output ceiling.
JUDGE_MAX_TOKENS = 16000
# Sampling is pinned because a judge that samples is a different judge on every call, and the
# study's quality gate compares arms across hundreds of calls. A route whose endpoint does not
# declare `temperature` cannot be pinned, and `_body` refuses it loudly rather than letting the
# provider choose a seed.
JUDGE_TEMPERATURE = "0"

VERDICT_SCHEMA_ID = "judge-verdict-v1"
CRITERION_VERDICTS = ("met", "not_met", "cannot_assess")
BETTER_OUTCOMES = ("1", "2", "tie", "cannot_assess")
SLOTS = ("1", "2")

# The schema is DECLARED, not implied by the parser, and `parse_verdict` asserts that the two
# agree. A judge prompt and a validator that drift apart produce a run where every judgement is
# "unparseable" and the cost is already committed.
SUBMISSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["criteria", "resolved"],
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criterion", "verdict", "basis"],
                "properties": {
                    "criterion": {"type": "string"},
                    "verdict": {"enum": list(CRITERION_VERDICTS)},
                    "basis": {"type": "string", "minLength": 1, "maxLength": 400},
                },
            },
        },
        "resolved": {"type": ["boolean", "null"]},
    },
}

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "submissions", "better"],
    "properties": {
        "schema": {"const": VERDICT_SCHEMA_ID},
        "submissions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["1", "2"],
            "properties": {"1": SUBMISSION_SCHEMA, "2": SUBMISSION_SCHEMA},
        },
        "better": {"enum": list(BETTER_OUTCOMES)},
    },
}

CRITERIA_HEADER = "CRITERIA (verbatim from the task; score exactly these and nothing else)"
SECTION = "=== {} ==="
CRITERIA_SECTION = SECTION.format(CRITERIA_HEADER)
CRITERION_LINE = re.compile(r"^(\d+)\. (.+)$")

JUDGE_SYSTEM = (
    "You are the judge for a paid writing study. You do not write, improve or explain the "
    "submissions.\n"
    "You score two submissions against the task's own evaluation criteria, quoted verbatim below. "
    "You must not introduce a criterion of your own: a criterion the task does not state is not "
    "part of this measurement.\n\n"
    "Reply with exactly one JSON object matching this schema, and nothing else — no prose, no code "
    "fence:\n")

JUDGE_RULES = (
    "\n\nFor each criterion, `verdict` is one of met / not_met / cannot_assess, and `basis` names "
    "the evidence in that submission (a short quote or a precise reason).\n"
    "`resolved` is true only when every criterion is met, false when at least one is not_met, and "
    "null when at least one is cannot_assess and none is not_met.\n"
    "`better` names the submission you would keep. If exactly one submission is resolved true, "
    "`better` must be that one. If neither is resolved true, `better` must be \"tie\" or "
    "\"cannot_assess\" — do not assert a preference the criteria do not license.")


class JudgeError(ValueError):
    """The judge phase cannot proceed, and saying so is the only safe action."""


class RubricError(JudgeError):
    """A task cannot be judged without inventing criteria."""


class VerdictError(JudgeError):
    """The judge did not answer in the declared schema — an instrument failure, not a score."""


class ResolutionConflict(VerdictError):
    """The reply is well formed but contradicts itself: `better` and `resolved` disagree.

    Separate from every other `VerdictError` because it is the one failure a re-ask can fix. A
    malformed reply, an invented criterion or an out-of-schema value all say the instrument was not
    driven correctly; this one says the judge's two summary fields disagree about a comparison it has
    already scored, which is a mis-read of its own work rather than a failure to do it.
    """


# --------------------------------------------------------------------------- the rubric


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    weight: str | None


def _one_line(value):
    return " ".join(value.split())


def _weight(task_id, name, weight):
    if weight is None:
        return None
    if (isinstance(weight, bool) or not isinstance(weight, (int, float, Decimal))
            or not 0 <= float(weight) <= 1):
        raise RubricError(f"{task_id}: criterion {name!r} has an invalid weight {weight!r}")
    return str(weight)


def _criterion(task_id, index, item):
    """One criterion, validated. Refusals name the task so the reader knows what to fix."""
    if not isinstance(item, dict):
        raise RubricError(f"{task_id}: evaluation_criteria[{index}] is not an object")
    name, description = item.get("criterion"), item.get("description")
    if not isinstance(name, str) or not name.strip():
        raise RubricError(f"{task_id}: evaluation_criteria[{index}] has no criterion id")
    if not isinstance(description, str) or not description.strip():
        raise RubricError(
            f"{task_id}: criterion {name!r} has no description; a criterion that does not say what "
            f"is being checked cannot be scored")
    name = _one_line(name)
    return Criterion(name, _one_line(description), _weight(task_id, name, item.get("weight")))


def _criteria_kind(task_id, raw):
    if isinstance(raw, dict):
        raise RubricError(
            f"{task_id}: evaluation_criteria is a mapping ({sorted(raw)}), not a list of "
            f"{{criterion, description}} items; refusing to judge it — flattening it would invent "
            f"criterion ids and score against them")
    if not isinstance(raw, list) or not raw:
        raise RubricError(
            f"{task_id}: task declares no evaluation_criteria; refusing to judge it — a judge given "
            f"an invented rubric is a different measurement")


def criteria_of(task):
    """The task's OWN criteria, or a refusal that names the task.

    The contract's criteria live in `evaluation_criteria`. Three of the 25 frozen tasks declare
    that key as a mapping (`beat_specific` bullet lists under `cross_beat` criteria) rather than a
    list of `{criterion, description}` items; those are refused BY NAME instead of being flattened,
    because flattening invents criterion ids and then scores against them.

    Weights are carried when the task states them and are NOT used to aggregate: the primary
    outcome is the task's binary `resolved`, and the quality field is a pairwise preference. A
    weighted scalar here would be a second, undeclared outcome.
    """
    if not isinstance(task, dict):
        raise RubricError("task document is not an object; refusing to judge it")
    task_id = task.get("task_id")
    raw = task.get("evaluation_criteria")
    _criteria_kind(task_id, raw)
    criteria = tuple(_criterion(task_id, index, item) for index, item in enumerate(raw))
    names = [criterion.name for criterion in criteria]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RubricError(
            f"{task_id}: criteria {duplicates} are declared twice; a verdict could not say which "
            f"one it answered")
    return criteria


def render_criteria(criteria):
    """The block the judge sees and the block the rehearsal parses back.

    One renderer for both directions on purpose: if the rehearsal re-rendered the rubric its own
    way, `--dry-run` would prove that the rehearsal is self-consistent, not that the judge saw the
    task's criteria.
    """
    lines = []
    for index, criterion in enumerate(criteria, start=1):
        lines.append(f"{index}. {criterion.name}")
        lines.append(f"   {criterion.description}")
    return "\n".join(lines)


def parse_rendered_criteria(prompt):
    """Read the criteria block back out of a user prompt. Raises when it is not there."""
    lines = prompt.splitlines()
    if CRITERIA_SECTION not in lines:
        raise RubricError("prompt carries no criteria block")
    found, index = [], lines.index(CRITERIA_SECTION) + 1
    while index < len(lines) and lines[index].strip() and not lines[index].startswith(("===", "SUBMISSION")):
        match = CRITERION_LINE.match(lines[index])
        if match is None:
            raise RubricError(f"malformed criteria line: {lines[index]!r}")
        if index + 1 >= len(lines) or not lines[index + 1].startswith("   "):
            raise RubricError(f"criterion {match.group(2)!r} has no description line")
        found.append((match.group(2), lines[index + 1].strip()))
        index += 2
    return tuple(found)


def assert_prompt_carries_rubric(prompt, criteria):
    """Every criterion the task declares, and no other criterion line in the block.

    This is the executable form of "nothing invented": the dry run compares the block the judge
    will read against the task's own list, so an edit that slips a house style rule ("literary
    quality") into the rubric fails before the call, not after it.
    """
    found = parse_rendered_criteria(prompt)
    expected = tuple((criterion.name, criterion.description) for criterion in criteria)
    if found != expected:
        raise RubricError(f"prompt rubric differs from the task's criteria: {found} != {expected}")
    return found


def task_brief(task):
    """The brief the writer was given — never the task's answer key.

    The criteria refer to the task's own requirements ("all four must_include elements"), so a
    judge without the brief cannot check them. The brief is `PromptBuilder.build_user_prompt`,
    which is exactly what the subject saw; `ground_truth` and any other field the builder does not
    emit stay out of the judge's context.
    """
    from generator import PromptBuilder  # noqa: PLC0415

    try:
        return PromptBuilder.build_user_prompt(task)
    except (KeyError, TypeError, ValueError) as exc:
        raise RubricError(
            f"{task.get('task_id')}: cannot render the task brief ({type(exc).__name__}: {exc}); "
            f"refusing to judge a task the writer's own prompt builder cannot express") from exc


def judge_prompt(task, criteria, submissions, correction=None):
    """(system, user) for one paired judgement. `submissions` maps slot -> text.

    `correction` is set only on a re-ask, and only for `ResolutionConflict`. It quotes the judge's
    own two summary fields back at it and names the rule they break; it never says which submission
    is the better one, because that is the measurement. It sits after both submissions and before the
    reply instruction, so the last thing read is still the instruction to answer.
    """
    if set(submissions) != set(SLOTS):
        raise JudgeError(f"a paired judgement needs both slots {SLOTS}, got {sorted(submissions)}")
    for slot, text in submissions.items():
        if not isinstance(text, str) or not text.strip():
            raise JudgeError(f"submission {slot} is empty; nothing to judge")
    system = JUDGE_SYSTEM + json.dumps(VERDICT_SCHEMA, sort_keys=True, separators=(",", ":")) + (
        JUDGE_RULES)
    parts = [
        f"TASK: {task.get('task_id')} ({task.get('task_type')})",
        "",
        SECTION.format("TASK BRIEF (what the writer was given)"),
        task_brief(task),
        "",
        CRITERIA_SECTION,
        render_criteria(criteria),
        "",
        SECTION.format("SUBMISSION 1"),
        submissions["1"],
        "",
        SECTION.format("SUBMISSION 2"),
        submissions["2"],
        "",
    ]
    if correction is not None:
        parts += [SECTION.format("CORRECTION REQUIRED"), correction, ""]
    parts += [
        SECTION.format("REPLY"),
        "Return the JSON object now.",
    ]
    return system, "\n".join(parts)


# --------------------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class Verdict:
    """A parsed judgement: per-slot criterion verdicts and the pairwise preference."""

    criteria: tuple  # slot 1, (criterion name, verdict) pairs
    criteria_2: tuple  # slot 2, same
    resolved: dict  # slot -> True / False / None
    better: str

    def quality_for(self, slot):
        """The pairwise outcome as seen from `slot` (win / tie / loss / cannot_assess)."""
        if self.better == "cannot_assess":
            return "cannot_assess"
        if self.better == "tie":
            return "tie"
        return "win" if self.better == slot else "loss"


def _derive_resolved(entries):
    if any(verdict == "not_met" for _, verdict in entries):
        return False
    if any(verdict == "cannot_assess" for _, verdict in entries):
        return None
    return True


def _strict_keys(value, expected, label):
    if not isinstance(value, dict):
        raise VerdictError(f"{label} must be an object")
    if set(value) != set(expected):
        raise VerdictError(f"{label} fields {sorted(value)} != {sorted(expected)}")
    return value


def _json_text(content):
    """One JSON object, or a failure. A fenced block is tolerated; prose around it is not."""
    if not isinstance(content, str) or not content.strip():
        raise VerdictError("judge returned no content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if not lines[-1].strip().startswith("```"):
            raise VerdictError("judge output opens a code fence it never closes")
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise VerdictError(f"judge output is not one JSON object: {text[:160]!r}") from exc


def _entry(item, slot, expected):
    _strict_keys(item, SUBMISSION_SCHEMA["properties"]["criteria"]["items"]["properties"],
                 f"submission {slot} criterion")
    name, verdict, basis = item["criterion"], item["verdict"], item["basis"]
    if name not in expected:
        raise VerdictError(f"submission {slot} scored undeclared criterion {name!r}")
    if verdict not in CRITERION_VERDICTS:
        raise VerdictError(f"criterion {name!r} verdict {verdict!r} is not declared")
    if not isinstance(basis, str) or not basis.strip() or len(basis) > 400:
        raise VerdictError(f"criterion {name!r} has no usable basis")
    return (name, verdict)


def _submission(raw, slot, expected):
    """One submission's block: exact criterion set, no duplicates, a self-consistent summary."""
    _strict_keys(raw, SUBMISSION_SCHEMA["properties"], f"submission {slot}")
    items = raw["criteria"]
    if not isinstance(items, list):
        raise VerdictError(f"submission {slot}.criteria must be an array")
    entries = tuple(_entry(item, slot, expected) for item in items)
    names = [name for name, _ in entries]
    if len(set(names)) != len(names):
        raise VerdictError(f"submission {slot} scored a criterion twice")
    # Set equality is checked LAST so an undeclared name is reported as undeclared by `_entry`
    # rather than as a count mismatch, which sends the reader looking at the wrong thing.
    if set(names) != expected:
        raise VerdictError(f"submission {slot} scored {sorted(set(names))}, the task declares "
                           f"{sorted(expected)}")
    declared = raw["resolved"]
    if declared is not None and type(declared) is not bool:
        raise VerdictError(f"submission {slot}.resolved must be boolean or null")
    derived = _derive_resolved(entries)
    if declared != derived:
        raise VerdictError(f"submission {slot}.resolved {declared!r} contradicts its criterion "
                           f"verdicts ({derived!r})")
    return entries, derived


def _preference(raw, resolved):
    """The pairwise claim, refused when it contradicts the rubric the judge was given.

    The rubric names two cases and both are about EXACTLY ONE or NEITHER resolved: "If exactly one
    submission is resolved true, `better` must be that one. If neither is resolved true, `better` must
    be tie or cannot_assess." When BOTH are resolved true the rubric asserts nothing, so any of the
    four outcomes is licensed -- both passed, the judge may still prefer one.

    This was implemented as a chain of ternaries instead of a count, and the chain is wrong at the one
    point that matters: `"1" if first is True and second is not True else "2" if second is True` makes
    `only == "2"` when BOTH are true, so a legitimate `better: "1"` was rejected as a contradiction --
    and the corrective re-ask then told the judge "better must name submission 2", which was FALSE
    guidance derived from a comparison the rubric never ordered.
    """
    better = raw["better"]
    if better not in BETTER_OUTCOMES:
        raise VerdictError(f"better {better!r} is not one of {BETTER_OUTCOMES}")
    resolved_text = f"resolved={{{', '.join(f'{k}: {v!r}' for k, v in sorted(resolved.items()))}}}"
    trues = [slot for slot in sorted(resolved) if resolved[slot] is True]
    if len(trues) == 1:
        if better != trues[0]:
            raise ResolutionConflict(
                f"better must name submission {trues[0]}, the only resolved submission "
                f"({resolved_text}, better={better!r})")
        return better
    if not trues and better not in ("tie", "cannot_assess"):
        raise ResolutionConflict(
            f"neither submission is resolved true, yet better={better!r}: a preference between them "
            f"is taste the task's criteria do not license ({resolved_text})")
    return better


def parse_verdict(content, criteria):
    """Strictly parse one judge reply, or fail the judgement.

    Strict on purpose, in both directions. The reply must be exactly one JSON object (a fenced
    block is accepted because models habitually add one; prose around it is not), every declared
    criterion must appear exactly once and no undeclared one may appear, and the judge's own
    `resolved` must equal what its criterion verdicts imply. A judge that cannot keep its summary
    consistent with its own lines is not a measurable instrument, and defaulting the disagreement
    to any value would launder it into the study.
    """
    data = _json_text(content)
    _strict_keys(data, VERDICT_SCHEMA["properties"], "verdict")
    if data["schema"] != VERDICT_SCHEMA_ID:
        raise VerdictError(f"verdict schema {data['schema']!r} != {VERDICT_SCHEMA_ID!r}")
    _strict_keys(data["submissions"], SLOTS, "submissions")
    expected = {criterion.name for criterion in criteria}
    parsed, resolved = {}, {}
    for slot in SLOTS:
        parsed[slot], resolved[slot] = _submission(data["submissions"][slot], slot, expected)
    return Verdict(parsed["1"], parsed["2"], resolved, _preference(data, resolved))


def analysis_fields(verdict, slot_by_arm):
    """Map a verdict onto the analysis's per-arm fields, without inventing a second observation.

    `quality` is written only for the treatment arm: it is a PAIRWISE outcome, and copying it to
    both arms would present one judgement as two independent readings — the same error the
    contract forbids by refusing to count judges/repeats as stories. The control arm keeps
    `resolved`, which is per-submission and genuinely two readings.
    """
    slot_of = {arm: slot for slot, arm in slot_by_arm.items()}
    fields = {}
    for arm in ("A", "D"):
        slot = slot_of[arm]
        fields[arm] = dict(resolved=verdict.resolved[slot], invalid_kind=None,
                           quality=verdict.quality_for(slot) if arm == "D" else None)
    return fields


def failed_analysis(reason):
    """A failed judgement is grader-invalid, never a low score.

    `resolved=None` with `invalid_kind="grader"` is what `study_analysis` excludes from the point
    estimate and bounds in the missingness analysis; `resolved=False` would enter the estimate as a
    task failure and make a broken judge look like a bad writer.
    """
    return {"A": dict(resolved=None, invalid_kind="grader", quality="cannot_assess", reason=reason),
            "D": dict(resolved=None, invalid_kind="grader", quality="cannot_assess", reason=reason)}


# --------------------------------------------------------------------------- collected outputs


@dataclass(frozen=True)
class Collected:
    """One retained, paid, completed generation output."""

    attempt_id: str
    model_snapshot: str
    task_id: str
    sample: int
    condition: str
    story_id: str
    task: dict
    text: str
    cost_usd: str


def _skip(skipped, reason):
    skipped[reason] = skipped.get(reason, 0) + 1


def _retained_output(root, record, attempt_id):
    """Read one attempt's manifest, response and task bytes, checking every hash on the way.

    Absence is a REASON, not an exception. One ledger outlives several evidence roots -- the smoke
    units live under `spend/evidence`, the pilot under `spend/pilot-evidence` -- so an attempt the
    ledger holds is not necessarily an attempt this root contains. Crashing on that made `--plan`
    unusable against the real ledger, and a traceback is not a finding.
    """
    directory = root / "attempts" / attempt_id
    if not (directory / "manifest.json").is_file():
        raise JudgeError(f"{attempt_id}: no retained evidence under {root.name}")
    manifest_raw = (directory / "manifest.json").read_bytes()
    if pp.sha(manifest_raw) != record["quote"]["request_sha256"]:
        raise JudgeError(f"{attempt_id}: retained manifest is not the request the ledger holds")
    spec = json.loads(manifest_raw)["spec"]
    response_raw = (directory / "response.bin").read_bytes()
    if pp.sha(response_raw) != record["receipt"]["evidence_sha256"]:
        raise JudgeError(f"{attempt_id}: retained response is not the receipt's response")
    content = pp.parse_json(response_raw)["choices"][0]["message"].get("content") or ""
    task_raw = (root / "blobs" / spec["task_sha256"]).read_bytes()
    if pp.sha(task_raw) != spec["task_sha256"]:
        raise JudgeError(f"{attempt_id}: retained task bytes do not match the attempt's digest")
    return spec, content, json.loads(task_raw)


def _output_or_reason(root, record, attempt_id):
    """(Collected, None) for a judgeable output, or (None, skip-reason)."""
    if record["envelope"] not in GENERATION_ENVELOPES or record["role"] != GENERATION_ROLE:
        return None, "not a generation attempt"
    if record["state"] != "RECONCILED":
        return None, f"state {record['state']} is not RECONCILED"
    receipt = record["receipt"]
    if receipt["outcome"] != "completed":
        return None, f"outcome {receipt['outcome']}"
    spec, content, task = _retained_output(root, record, attempt_id)
    if not isinstance(content, str) or not content.strip():
        return None, "empty output"
    return Collected(attempt_id=attempt_id, model_snapshot=spec["assigned_model_snapshot"],
                     task_id=spec["task_id"], sample=spec["sample"],
                     condition=spec["condition"], story_id=spec["story_id"], task=task,
                     text=content, cost_usd=receipt["actual_usd"]), None


def collected(ledger, evidence_root, scope=()):
    """Read what the ledger says was paid for AND the evidence still holds. Nothing is inferred.

    The ledger is the source of truth for which attempts exist and what they cost; the retained
    manifest is the source for what the attempt was; the retained response is the source for the
    text. Each link is checked (request hash against the ledger's quote, response hash against the
    receipt), because a judge that scores text nobody can trace back to a paid attempt produces
    numbers that cannot be reconciled with the spend.
    """
    root = pathlib.Path(evidence_root)
    outputs, skipped = [], {}
    for attempt_id in ledger.status()["attempt_ids"]:
        # A ledger outlives a run. Attempts from an earlier run are bound to another manifest (and
        # their evidence lives under another root), so reading them here would bury the real skips
        # under hundreds of "no retained evidence" lines. `scope` selects ONE run by its unit tag.
        if scope and not any(marker in attempt_id for marker in scope):
            continue
        try:
            output, reason = _output_or_reason(root, ledger.attempt(attempt_id), attempt_id)
        except JudgeError as exc:
            # A ledger entry this root does not hold, or a hash that disagrees with it, is a SKIP
            # with a reason. One traceback per foreign attempt is not a diagnostic.
            output, reason = None, str(exc)
        if output is None:
            _skip(skipped, reason)
        else:
            outputs.append(output)
    return tuple(outputs), skipped


# --------------------------------------------------------------------------- pairing


@dataclass(frozen=True)
class Judgement:
    """One paired judgement: the D row carries the ledger slot, both arms carry text."""

    judgement_id: str
    attempt_id: str
    row: dict
    task: dict
    criteria: tuple
    control: Collected
    treatment: Collected


def judgement_id_for(output):
    return f"{output.model_snapshot.replace('/', '_')}__{output.task_id}__s{output.sample}"


def pair_units(outputs, manifest):
    """Join retained outputs to the pre-registered matrix, one pair per (model, task, sample).

    An output that is not in the manifest is refused rather than judged: the matrix is what the
    study pre-registered, and a judged unit outside it cannot enter the paired estimate. An
    incomplete pair is reported as unpaired rather than judged alone, because a single-submission
    judgement cannot produce the pairwise `quality` field the analysis reads.
    """
    rows = {(row["model_snapshot"], row["task_id"], row["sample"], row["condition"]): row
            for row in manifest["assignments"]}
    if len(rows) != len(manifest["assignments"]):
        raise JudgeError("manifest holds two rows for one identity; refusing an ambiguous pair")
    grouped, unassigned = {}, []
    for output in outputs:
        key = (output.model_snapshot, output.task_id, output.sample)
        if key + (output.condition,) not in rows:
            unassigned.append(output.attempt_id)
            continue
        grouped.setdefault(key, {})[output.condition] = output
    pairs, unpaired = [], []
    for key, arms in sorted(grouped.items()):
        if set(arms) != {"A", "D"}:
            unpaired.append(dict(model_snapshot=key[0], task_id=key[1], sample=key[2],
                                 conditions=sorted(arms)))
        else:
            pairs.append((arms["D"], arms["A"]))
    return pairs, unpaired, unassigned


def build_judgements(pairs, manifest, *, unit):
    """Turn pairs into judgements, refusing per task where the rubric is unusable."""
    rows = {(row["model_snapshot"], row["task_id"], row["sample"], row["condition"]): row
            for row in manifest["assignments"]}
    judgements, refused = [], {}
    for treatment, control in pairs:
        row = rows[(treatment.model_snapshot, treatment.task_id, treatment.sample, "D")]
        try:
            criteria = criteria_of(row["task"])
        except RubricError as exc:
            refused.setdefault(row["task_id"], str(exc))
            continue
        judgement_id = judgement_id_for(treatment)
        judgements.append(Judgement(judgement_id=judgement_id,
                                    attempt_id=f"{unit}-{judgement_id}-judge",
                                    row=row, task=row["task"], criteria=criteria,
                                    control=control, treatment=treatment))
    return judgements, refused


# --------------------------------------------------------------------------- the confound


def _structured_disclosure(doc, path, raw):
    for parent in ("judge", "judge_designation"):
        block = doc.get(parent) if isinstance(doc, dict) else None
        if not isinstance(block, dict):
            continue
        for field in ("confound_disclosure", "confound", "overlap"):
            value = block.get(field)
            if isinstance(value, str) and value.strip():
                return dict(path=str(path), key=f"{parent}.{field}", found=True,
                            quote=_one_line(value), sha256=pp.sha(raw.encode()))
    return None


def _line_disclosure(path, raw):
    """A disclosure may live in a comment, which YAML drops; scan the file's lines too."""
    judge = re.compile(r"(?i)judge")
    confound = re.compile(r"(?i)confound|overlap|also a subject|same model|self-judg")
    for number, line in enumerate(raw.splitlines(), start=1):
        if judge.search(line) and confound.search(line):
            return dict(path=str(path), line=number, found=True, quote=_one_line(line),
                        sha256=pp.sha(raw.encode()))
    return None


def disclosure_citation(path, raw):
    """Locate the study's own statement of the judge confound, or say it was not found."""
    try:
        import yaml  # noqa: PLC0415

        doc = yaml.safe_load(raw) or {}
    except Exception:  # noqa: BLE001 - a study that will not parse is reported, not guessed at
        doc = {}
    found = _structured_disclosure(doc, path, raw) or _line_disclosure(path, raw)
    if found is not None:
        return found
    return dict(path=str(path), found=False, sha256=pp.sha(raw.encode()),
                searched=["judge.confound_disclosure", "judge.confound",
                          "judge_designation.confound",
                          "a line mentioning judge + confound/overlap"])


_DATED_SNAPSHOT = re.compile(r"^(?P<stem>.+?)-(?P<date>\d{8})$")


def same_route(a, b):
    """Are these two strings the same route, given one may be the dated snapshot?

    `judge_snapshot` is the CANONICAL SLUG (`google/gemini-4-argon`) while a row's `model_snapshot`
    is the dated form (`google/gemini-4-argon-20261013`), so a plain `==` reports no overlap even
    when the judge IS a subject — which is exactly this study's case. That comparison is not a
    detail: it decides whether the paid evidence admits a confound or quietly omits it.
    """
    def stem(value):
        match = _DATED_SNAPSHOT.match(value)
        return match.group("stem") if match else value

    return stem(str(a)) == stem(str(b))


def confound_record(manifest, study_path):
    """The judge-subject overlap, with a citation to where the study discloses it.

    This study's only qualified routes ARE its generation subjects, so the judge overlaps by
    construction. Refusing on that fact would block a frozen design; dropping it would hide a
    confound inside a paid measurement. So the overlap is computed from the manifest (the judge's
    snapshot against every row's `model_snapshot`), and the citation is looked up in the study file
    — a structured `judge.confound_disclosure`, or a line that mentions the judge and a confound.
    When neither exists the record says `found: false` and names what was searched: an evidence
    file that cites nothing is worse than one that admits the study has not said it yet.
    """
    judge = manifest.get("judge_snapshot")
    if not isinstance(judge, str) or not judge.strip():
        raise JudgeError("manifest records no judge_snapshot; there is no judge to identify")
    subjects = sorted({row["model_snapshot"] for row in manifest["assignments"]})
    path = pathlib.Path(study_path)
    raw = path.read_text(encoding="utf-8")
    overlapping = [subject for subject in subjects if same_route(judge, subject)]
    return dict(judge_snapshot=judge, overlapping=bool(overlapping), subjects=len(subjects),
                overlapping_subjects=overlapping,
                affected_arm_runs=sum(1 for row in manifest["assignments"]
                                      if same_route(row["model_snapshot"], judge)),
                disclosure=disclosure_citation(path, raw),
                consequence=("The judge is one of the study's generation subjects; its preference "
                             "for its own prose cannot be separated from its prose being better. "
                             "Every judgement carries that confound as evidence."))


# --------------------------------------------------------------------------- preparation


def _route_sources(base, judge_snapshot):
    """The judge route's retained record, found by the slug the manifest names.

    `rehearse_attempt.load_route` matches `route_identity.model_id`, but an assignment row records
    the CANONICAL SLUG as `judge_snapshot` (that is what `_validate_spec` compares the route to), so
    the lookup has to accept either. Matching only one of them refuses every real manifest while
    looking like a missing route.
    """
    matches = []
    for path in sorted((base / "records").glob("*.json")):
        record = json.loads(path.read_text())
        identity = record["route_identity"]
        if judge_snapshot in (identity["model_id"], identity["canonical_model_slug"]):
            matches.append(record)
    if len(matches) != 1:
        raise JudgeError(f"{judge_snapshot}: {len(matches)} route records match; refusing")
    identity = matches[0]["route_identity"]
    slug = identity["model_id"].replace("/", "_")
    catalog = (base / "openrouter-models.json").read_bytes()
    endpoints = (base / f"endpoints-{slug}.json").read_bytes()
    profile = (base / "verified" / f"{slug}.profile.json").read_bytes()
    verification = base / "verification" / (identity["model_id"].split("/")[-1] + ".json")
    if pp.parse_json(profile).get("kind") != pp.VERIFIED_KIND:
        return matches[0], (catalog, endpoints, profile)
    if not verification.is_file():
        raise JudgeError(f"{judge_snapshot}: a verified profile without {verification.name}")
    return matches[0], (catalog, endpoints, profile, verification.read_bytes())


def _gate_blobs(record, assignment_hash, route_sha256, valid_until, mode):
    """`paid.gates` hardcodes the C8 authority, which an offline rehearsal cannot claim.

    A synthetic admission needs gates vouching for the synthetic authority; reusing the C8 ones
    would produce an admission `_admit` refuses. The verified branch IS `paid.gates`, so the paid
    path's gate derivation is not reimplemented.
    """
    if mode == pp.QUALIFICATION_FOR_KIND[pp.VERIFIED_KIND]:
        return paid.gates(record, assignment_hash, route_sha256, valid_until)
    if mode != pp.QUALIFICATION_FOR_KIND[pp.SYNTHETIC_KIND]:
        raise JudgeError(f"unknown route qualification {mode!r}")
    authority = pp.AUTHORITY_FOR_QUALIFICATION[mode]
    stated = record.get("admission_gates") or {}
    bad = sorted(kind for kind in pa.GATES if (stated.get(kind) or {}).get("status") != "satisfied")
    if bad:
        raise JudgeError(f"admission gates not satisfied in the record: {bad}")
    return {f"gate-{kind}": reh.json_bytes(dict(
        kind=kind, authority=authority, assignment_hash=assignment_hash,
        route_sha256=route_sha256, valid_until=valid_until)) for kind in sorted(pa.GATES)}


def judge_spec(row, judgement, artifacts, assignment_hash, tag=None):
    """The AttemptSpec for a judge call: `data_judge`, `research`, `research_evaluation`.

    This is the one place this file departs from `paid_attempt.prepared`, and it must. `prepared`
    calls `rehearse_attempt.spec_for`, which hardcodes `envelope="pilot"`,
    `cost_role="deployment"` and `activity_role="generation"`; using it would (a) take the
    reservation from the pilot envelope the study priced for generation, and (b) make
    `_validate_spec` bind the route to `row["model_snapshot"]` — a generation subject — instead of
    `row["judge_snapshot"]`, so the ledger would record the judge call as a generation call.
    Everything else `prepared` does (route load, snapshot validation, gate derivation, artifact
    assembly, `prepare_attempt`) is reused through the same functions.

    `tag` distinguishes a corrective re-ask from the first call. It changes the attempt id and
    nothing else: the correction is not a retry (`retry_index` stays 0) because the first call
    completed and was paid for — what failed was its answer's internal consistency, not the transport.
    """
    refs = tuple(pa.ArtifactRef.of(name, raw) for name, raw in sorted(artifacts.items()))
    by = dict(artifacts)
    return pa.AttemptSpec(
        attempt_id=f"{judgement.attempt_id}{tag or ''}",
        run_id=f"judge-{judgement.judgement_id}",
        episode_id=judgement.judgement_id, assignment_hash=assignment_hash,
        task_id=row["task_id"], task_sha256=digest(row["task"]),
        story_id=row["story_id"], condition=row["condition"], sample=row["sample"],
        assigned_model_snapshot=row["model_snapshot"],
        harness_sha256=pp.sha(by["harness"]), protocol_sha256=pp.sha(by["protocol"]),
        settings_sha256=pp.sha(by["settings"]), effort="unspecified",
        parent_attempt_id=None, retry_index=0, max_attempts=1,
        envelope=JUDGE_ENVELOPE, cost_role=JUDGE_ROLE, activity_role=JUDGE_ACTIVITY,
        program_id=PROGRAM_ID, store_id=STORE_ID,
        native_package_sha256=pp.sha(by["native_package"]), artifacts=refs)


def blind_slots(judgement_id):
    """Which arm sits in slot 1, fixed by the judgement id so it is reproducible and recorded.

    Without blinding, the judge knows which text is the treatment and can prefer it; with a random
    per-call order that is not recorded, the pairing cannot be reconstructed. The digest fixes the
    order without a clock or a random source, and the mapping travels in the evidence.
    """
    even = int(pp.sha(judgement_id.encode())[:2], 16) % 2 == 0
    return {"1": "A" if even else "D", "2": "D" if even else "A"}


def _submissions(judgement, slot_by_arm):
    return {slot: (judgement.treatment if arm == "D" else judgement.control).text
            for slot, arm in slot_by_arm.items()}


SENT_MARKERS = ("transport.json", "response.bin", "receipt.json")


def _purge(path):
    """Delegate removal of a directory tree."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _dir_digest(root):
    """A stable name for a directory's evidence: every file name and its bytes, sorted."""
    parts = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            parts.append(path.name.encode() + b"\0" + pp.sha(path.read_bytes()).encode())
    return pp.sha(b"\n".join(parts))[:16]


def _set_aside(root):
    """Rename a spent directory out of the way, keeping every file inside it, IDEMPOTENTLY.

    Used only when the LEDGER ALREADY HOLDS the attempt: the call was paid for and reconciled, so
    the next run must not re-send it, but `prepare_attempt` cannot write the name it wants while the
    old directory sits there. Renaming preserves the receipt, response and transport for audit and
    lets the plan be built; the walk then skips the judgement as held, so no money is spent twice.

    The target name is the directory's CONTENT digest, not a counter. A counter made every run add a
    copy: three runs of the 2026-11-09 judge left `.superseded-1`, `-2` and `-3` of the same attempts
    -- 177 directories and 2396 changed paths in git, all describing one call each. With the digest
    the same evidence always lands on the same name, so the second run finds it already set aside and
    what is left at the canonical path is only the marker-less manifest it just wrote.
    """
    candidate = root.with_name(f"{root.name}.superseded-{_dir_digest(root)}")
    if candidate.exists():
        if any((root / marker).exists() for marker in SENT_MARKERS):
            raise JudgeError(f"two sent copies of {root.name}; refusing to discard evidence")
        # Same content already preserved, and this one never sent: a plan-rebuild leftover.
        _purge(root)
        return candidate.name
    root.rename(candidate)
    return candidate.name


def _discard_unsent(evidence_root, attempt_id, held=()):
    """Make the evidence name writable without ever losing a sent attempt.

    Three cases:
      * no directory            -> nothing to do;
      * directory, nothing sent -> a `--plan` leftover; drop it, it holds no evidence;
      * directory, something sent -> the ledger holds it (a crash spent the call before the verdict
        was written): set it aside so the plan can be rebuilt, and never resend it.
    """
    root = evidence_root / "attempts" / attempt_id
    if not root.is_dir():
        return False
    if not any((root / marker).exists() for marker in SENT_MARKERS):
        _purge(root)
        return True
    if attempt_id in held:
        _set_aside(root)
        return True
    return False


def prepare_judgement(judgement, *, base, manifest_raw, evidence_root,
                      judge_snapshot, discard_unsent=False, held=(), correction=None, tag=None):
    """Build and validate one judge request under the judge route's own record. Sends nothing.

    `correction`/`tag` build the corrective re-ask: same submissions, same blinding, same rubric, one
    added section that quotes the judge's own contradiction back at it.
    """
    record, sources = _route_sources(pathlib.Path(base), judge_snapshot)
    identity = record["route_identity"]
    route = reh._route(identity)
    snapshot = pp.validate_snapshot(sources, route, identity["observed_at"],
                                    identity["expires_at"])
    if snapshot.qualification not in pp.QUALIFIED_KINDS:
        raise JudgeError(f"{judge_snapshot}: route is {snapshot.qualification!r}, not payable")
    slot_by_arm = blind_slots(judgement.judgement_id)
    system, user = judge_prompt(judgement.task, judgement.criteria,
                                _submissions(judgement, slot_by_arm), correction=correction)
    fields = pa.RequestFields(messages=(("system", system), ("user", user)),
                              max_tokens=JUDGE_MAX_TOKENS, temperature=JUDGE_TEMPERATURE)
    artifacts = reh.artifacts_for(judgement.row, fields, manifest_raw, base)
    # The harness digest must name the program that assembled THIS request. Keeping
    # `dispatch_runner` there would make the evidence cite a script that never ran.
    artifacts["harness"] = pathlib.Path(__file__).read_bytes()
    assignment_hash = digest(json.loads(manifest_raw))
    route_sha256 = pp.sha(reh.json_bytes(asdict(route)))
    gate_blobs = _gate_blobs(record, assignment_hash, route_sha256, identity["expires_at"],
                             snapshot.qualification)
    artifacts |= gate_blobs
    if discard_unsent:
        _discard_unsent(pathlib.Path(evidence_root), judge_spec(
            judgement.row, judgement, artifacts, assignment_hash, tag=tag).attempt_id, held=held)
    plan = pa.prepare_attempt(
        judge_spec(judgement.row, judgement, artifacts, assignment_hash, tag=tag),
        snapshot, tuple(sorted(artifacts.items())), fields,
        pathlib.Path(evidence_root))
    admission = pa.Admission(
        mode=snapshot.qualification,
        authority_refs=tuple(pa.ArtifactRef.of(name, blob)
                             for name, blob in sorted(gate_blobs.items())),
        assignment_hash=assignment_hash, route_sha256=route_sha256,
        valid_until=identity["expires_at"])
    return plan, admission, identity, slot_by_arm


def reservation_micros(plan):
    """What `Ledger.reserve` will hold for this plan, computed the way `_quote` computes it.

    Per line, rounded up: `spend_ledger._quote` sums per-line ceilings, so summing the exact
    decimals and rounding once would let the plan advertise a figure the ledger then exceeds by a
    micro or two. A plan that disagrees with the reservation it predicts is a plan that lies.
    """
    return sum(sl.usd_to_micros(Decimal(line.usd_per_unit) * line.max_units)
               for line in plan.census)


def _usd(micros):
    return str((Decimal(micros) / Decimal(1_000_000)).quantize(Decimal("0.000001")))


def _validate_body(plan, criteria):
    """Read the built body back and re-check that the rubric the judge sees is the task's."""
    body = pp.parse_json(plan.body_bytes)
    if body.get("stream") is not False:
        raise JudgeError("judge request must be non-streaming")
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise JudgeError("judge request must carry exactly the system and user messages")
    user = next((message["content"] for message in messages if message.get("role") == "user"), None)
    if not isinstance(user, str):
        raise JudgeError("judge request carries no user message")
    assert_prompt_carries_rubric(user, criteria)
    return body


# --------------------------------------------------------------------------- the three modes


def _affordability(status, reservations):
    envelope = status["envelopes"][JUDGE_ENVELOPE]
    total = sum(reservations)
    reasons = []
    if status["dispatch_blocked"]:
        reasons.append(
            "ledger dispatch is blocked by unreconciled external charges "
            f"{status['unknown_charge_ids']} or incidents {status['incident_ids']}")
    if total > envelope["usable_micros"]:
        reasons.append(
            f"the judgement phase needs {_usd(total)} USD but {JUDGE_ENVELOPE} can release "
            f"{_usd(envelope['usable_micros'])} USD of its {_usd(envelope['allocation_micros'])} "
            f"USD allocation")
    return dict(total_micros=total, total_usd=_usd(total),
                allocation_usd=_usd(envelope["allocation_micros"]),
                remaining_usd=_usd(envelope["remaining_micros"]),
                usable_usd=_usd(envelope["usable_micros"]), affordable=not reasons,
                reasons=reasons)


def _prepare_all(judgements, prepare):
    plans, failed = [], []
    for judgement in judgements:
        try:
            plans.append((judgement, prepare(judgement)))
        except (JudgeError, sl.LedgerError, ValueError, SystemExit) as exc:
            failed.append(dict(judgement_id=judgement.judgement_id, reason=str(exc)))
    return plans, failed


def _refusals(*, refused, unassigned, judgements, outputs, partial_ok):
    """The reasons a plan must not execute. Partial work needs `--partial-ok`, explicitly."""
    reasons = []
    if refused and not partial_ok:
        reasons.append(f"{len(refused)} task(s) declare no usable evaluation_criteria: "
                       f"{sorted(refused)} — fix the rubric or pass --partial-ok")
    if unassigned:
        reasons.append(f"{len(unassigned)} retained output(s) are not in the assignment matrix, "
                       f"e.g. {unassigned[:3]}; an unassigned record cannot be judged")
    if not judgements:
        reasons.append(f"no judgeable pair was found among {len(outputs)} output(s); nothing "
                       f"would be sent")
    return reasons


def _envelope_block(reservations):
    return dict(per_call_max_usd=_usd(max(reservations, default=0)),
                total_usd=_usd(sum(reservations)), total_micros=sum(reservations))


def _route_block(plans):
    """The route whose record priced and bounded the call — printed, not assumed."""
    if not plans:
        return None
    plan = plans[0][1][0]
    profile = pp.parse_json(plan.snapshot._sources[2])
    return dict(model_id=plan.route.model_id,
                canonical_model_slug=plan.route.canonical_model_slug,
                endpoint_tag=plan.route.endpoint_tag, service_tier=plan.route.service_tier,
                qualification=plan.snapshot.qualification,
                price_observed_at=plan.snapshot.observed_at,
                price_expires_at=plan.snapshot.expires_at,
                input_bound_bytes=profile["input_bound"],
                output_bound_tokens=profile["output_bound"])


def run_plan(judgements, *, ledger, prepare, manifest, evidence_root, study, outputs,
             partial_ok=False, refused=None, unpaired=None, unassigned=None):
    """Print what would be judged, how many calls, and the reservation — else refuse.

    Every plan is PREPARED here, not estimated: a reservation the ledger cannot take is a plan that
    cannot execute, and discovering that after 200 cheap-looking rows is the mistake this mode
    exists to make impossible.
    """
    refused, unpaired, unassigned = refused or {}, unpaired or [], unassigned or []
    plans, failed = _prepare_all(judgements, prepare)
    reservations = [reservation_micros(plan) for _, (plan, _, _, _) in plans]
    affordability = _affordability(ledger.status(), reservations)
    reasons = list(affordability["reasons"])
    if failed and not partial_ok:
        reasons.append(f"{len(failed)} judgement(s) cannot be prepared: {failed[:3]}")
    reasons += _refusals(refused=refused, unassigned=unassigned, judgements=judgements,
                         outputs=outputs, partial_ok=partial_ok)
    return dict(mode="plan", manifest_sha256=digest(manifest),
                judge_snapshot=manifest.get("judge_snapshot"),
                judge_route=_route_block(plans), judgements=len(judgements),
                calls=len(plans), reservation=_envelope_block(reservations),
                envelope=affordability, prepared=[j.attempt_id for j, _ in plans],
                prepare_failures=failed, refused_tasks=refused, unpaired=unpaired,
                unassigned_outputs=unassigned, confound=confound_record(manifest, study),
                refused_reasons=reasons)


def run_dry_run(judgements, *, ledger, prepare, manifest, evidence_root, study, outputs,
                partial_ok=False, refused=None, unpaired=None, unassigned=None):
    """Build and validate EVERY request body, and send nothing.

    "Validate" is not "does not throw": each body is parsed back, the rubric block is compared
    against the task's own criteria, and the census that the ledger would reserve is printed. The
    only network call in `paid_attempt` is `preflight`, and this mode does not call it.
    """
    refused, unpaired, unassigned = refused or {}, unpaired or [], unassigned or []
    plans, failed = _prepare_all(judgements, prepare)
    bodies, reservations, invalid = [], [], []
    for judgement, (plan, _, _, slot_by_arm) in plans:
        try:
            _validate_body(plan, judgement.criteria)
        except (JudgeError, ValueError) as exc:
            invalid.append(dict(judgement_id=judgement.judgement_id, reason=str(exc)))
            continue
        reservations.append(reservation_micros(plan))
        bodies.append(dict(judgement_id=judgement.judgement_id, attempt_id=plan.spec.attempt_id,
                           body_bytes=len(plan.body_bytes), body_sha256=pp.sha(plan.body_bytes),
                           deadline_seconds=plan.deadline_seconds, slots=dict(slot_by_arm),
                           reservation_usd=_usd(reservations[-1])))
    affordability = _affordability(ledger.status(), reservations)
    reasons = list(affordability["reasons"])
    if failed and not partial_ok:
        reasons.append(f"{len(failed)} judgement(s) cannot be prepared: {failed[:3]}")
    if invalid:
        reasons.append(f"{len(invalid)} built body(s) failed validation: {invalid[:3]}")
    reasons += _refusals(refused=refused, unassigned=unassigned, judgements=judgements,
                         outputs=outputs, partial_ok=partial_ok)
    return dict(mode="dry-run", manifest_sha256=digest(manifest),
                judge_snapshot=manifest.get("judge_snapshot"),
                judge_route=_route_block(plans), judgements=len(judgements),
                calls=len(bodies), bodies=bodies, reservation=_envelope_block(reservations),
                envelope=affordability, prepare_failures=failed, invalid_bodies=invalid,
                refused_tasks=refused, unpaired=unpaired, unassigned_outputs=unassigned,
                confound=confound_record(manifest, study), refused_reasons=reasons)


def _messages(plan):
    return tuple(message["content"] for message in pp.parse_json(plan.body_bytes)["messages"])


def _verdict_doc(judgement, plan, slot_by_arm, confound, *, content, parsed, failure, call,
                 analysis):
    """The per-judgement evidence artifact. The confound is a required field, not an addendum.

    The check below fails the record if the block does not name the route that was called, so a
    judgement cannot be written with another run's confound pasted in — the one way this required
    field could become decoration.
    """
    if confound.get("judge_snapshot") != plan.route.canonical_model_slug:
        raise JudgeError("evidence would record a confound for a different judge route")
    system, user = _messages(plan)
    return dict(
        schema="judge-evidence-v1", judgement_id=judgement.judgement_id,
        judge_snapshot=plan.route.canonical_model_slug,
        judge_attempt_id=plan.spec.attempt_id, judge_envelope=plan.spec.envelope,
        judge_role=plan.spec.cost_role, judge_activity=plan.spec.activity_role,
        subject_model_snapshot=judgement.row["model_snapshot"],
        task_id=judgement.row["task_id"], sample=judgement.row["sample"],
        control_attempt_id=judgement.control.attempt_id,
        treatment_attempt_id=judgement.treatment.attempt_id,
        criteria=[asdict(criterion) for criterion in judgement.criteria],
        criteria_sha256=pp.sha(pp.json_bytes([asdict(c) for c in judgement.criteria])),
        slots=dict(slot_by_arm),
        prompt=dict(system_sha256=pp.sha(system.encode()), user_sha256=pp.sha(user.encode()),
                    body_sha256=pp.sha(plan.body_bytes)),
        response=dict(sha256=pp.sha(content.encode()) if isinstance(content, str) else None,
                      parse="failed" if failure else "ok", failure=failure, verdict=parsed),
        call=call, analysis=analysis, confound=confound)


def _parsed_block(parsed, slot_by_arm):
    def entries(values):
        return [dict(criterion=name, verdict=verdict) for name, verdict in values]

    return dict(slots=dict(slot_by_arm), resolved=parsed.resolved, better=parsed.better,
                criteria={"1": entries(parsed.criteria), "2": entries(parsed.criteria_2)})


def _call_record(ledger, attempt_id, latency_ms, observed):
    try:
        row = ledger.attempt(attempt_id)
    except sl.LedgerError as exc:
        raise JudgeError(f"{attempt_id}: ledger state unreadable after the call: {exc}") from exc
    receipt = row.get("receipt") or {}
    return dict(attempt_id=attempt_id, state=row["state"],
                reserved_usd=_usd(row["reserved_micros"]), cost_usd=receipt.get("actual_usd"),
                outcome=receipt.get("outcome"), latency_ms=latency_ms,
                live_status=(observed or {}).get("status"))


def _judge_outcome(plan, ledger, judgement, slot_by_arm, result, latency_ms, observed, confound,
                   evidence_root, stem="verdict"):
    """Turn one executed call into (evidence document, result line, verdict, error).

    `error` is the `VerdictError` instance when the reply was rejected, so the caller can tell a
    correctable contradiction from an instrument failure. `stem` names the evidence file: the retry
    writes over `verdict.json` only after the rejected reading has been moved aside.
    """
    call = _call_record(ledger, plan.spec.attempt_id, latency_ms, observed)
    content, failure, parsed, error = None, None, None, None
    if result.transport_status != "completed" or result.output_ref is None:
        failure = f"judge call did not complete: {result.transport_status}"
        analysis = failed_analysis(failure)
    else:
        content = pa.artifact_bytes(plan, result.output_ref).decode("utf-8")
        try:
            parsed = parse_verdict(content, judgement.criteria)
            analysis = analysis_fields(parsed, slot_by_arm)
        except VerdictError as exc:
            error = exc
            failure = str(exc)
            analysis = failed_analysis(failure)
    document = _verdict_doc(judgement, plan, slot_by_arm, confound, content=content,
                            parsed=None if parsed is None else _parsed_block(parsed, slot_by_arm),
                            failure=failure, call=call, analysis=analysis)
    directory = pathlib.Path(evidence_root) / "judgements" / judgement.judgement_id
    pa._persist(directory / f"{stem}.json", pp.json_bytes(document))
    return dict(judgement_id=judgement.judgement_id, attempt_id=plan.spec.attempt_id,
                state=call["state"], cost_usd=call["cost_usd"], latency_ms=latency_ms,
                failure=failure, analysis=analysis,
                evidence=str(directory / f"{stem}.json")), failure, parsed, error


async def _execute_one(plan, ledger, admission, identity):
    observed = paid.preflight(plan, identity)
    started = time.monotonic()
    result = await pa.execute_attempt(plan, ledger, admission)
    return result, int((time.monotonic() - started) * 1000), observed


CORRECTIVE_TAG = "-r1"


def _corrective_message(conflict):
    """What a re-asked judge is told: its own two fields, the rule, and nothing about the answer.

    Deliberately not "name submission N". That field is derivable from the judge's own `resolved`,
    so telling it the answer would turn a measurement into a compliance check: it could satisfy the
    re-ask by flipping one field without re-reading either text. It is asked to make the two agree,
    in whichever direction its reading actually supports.
    """
    return "\n".join((
        "Your previous reply was rejected for an internal contradiction, not for the verdicts you "
        "gave on the criteria.",
        f"The contradiction: {conflict}",
        "",
        "The rule it breaks, quoted from the instructions above: \"If exactly one submission is "
        "resolved true, `better` must be that one. If neither is resolved true, `better` must be "
        "\"tie\" or \"cannot_assess\" — do not assert a preference the criteria do not license.\"",
        "",
        "Re-read both submissions against the same criteria and return ONE corrected JSON object in "
        "which `resolved` and `better` agree. Either correct `better` to match the criterion verdicts "
        "you already gave, or correct a criterion verdict that you now believe you misread so that "
        "`resolved` supports the preference you intended. Do not change a criterion verdict you "
        "still believe is right merely to remove the disagreement.",
    ))


def _corrective_reattempt(*, judgement, prepare, ledger, confound, evidence_root, first_line,
                          conflict):
    """Re-ask once. Returns (line, failure, parsed, error), or None when the re-ask could not run.

    The rejected reading is moved to `verdict.rejected.json`, never deleted: `verdict.json` is the
    verdict of record, and the rejected one is the evidence that the first answer contradicted
    itself. A re-ask that cannot be prepared puts the original record back and leaves the original
    failure standing, so a failed retry never costs the run a paid call's evidence.
    """
    directory = pathlib.Path(evidence_root) / "judgements" / judgement.judgement_id
    original, rejected = directory / "verdict.json", directory / "verdict.rejected.json"
    moved = original.exists()
    if moved:
        os.replace(original, rejected)
    try:
        plan, admission, identity, slot_by_arm = prepare(
            judgement, correction=_corrective_message(conflict), tag=CORRECTIVE_TAG)
        result, latency_ms, observed = asyncio.run(
            _execute_one(plan, ledger, admission, identity))
    except (JudgeError, sl.LedgerError, ValueError, SystemExit) as exc:
        if moved and rejected.exists():
            os.replace(rejected, original)
        print(f"  {judgement.judgement_id[:58]:58} corrective re-ask could not be prepared: "
              f"{type(exc).__name__}: {str(exc)[:60]}", flush=True)
        return None
    line, failure, parsed, error = _judge_outcome(
        plan, ledger, judgement, slot_by_arm, result, latency_ms, observed, confound, evidence_root)
    line["corrective_reask"] = dict(
        superseded_attempt_id=first_line["attempt_id"],
        superseded_cost_usd=first_line["cost_usd"],
        conflict=str(conflict),
        rejected_evidence=str(rejected) if moved else None,
        also_conflicted=isinstance(error, ResolutionConflict))
    return line, failure, parsed, error



def _verdict_path(evidence_root, judgement):
    return pathlib.Path(evidence_root) / "judgements" / judgement.judgement_id / "verdict.json"


def recorded_failure(document):
    """The failure a verdict artifact recorded, or None.

    `_verdict_doc` writes it at `response.failure` -- there is no top-level `failure` key. Reading the
    wrong one is silent and expensive: the first repair pass read the top level, so it saw only the 2
    judgements with no file at all and missed the 45 whose replies had been truncated, every one of
    which looked like a clean "cannot_assess" downstream. The top-level read survives only as a
    fallback so a hand-written fixture keeps working.
    """
    failure = (document.get("response") or {}).get("failure")
    return failure if failure else document.get("failure")


def _needs_repair(evidence_root, judgement):
    """(True, why) when this judgement carries no usable verdict: absent, unreadable, or failed.

    A judgement whose call settled but whose reply was rejected -- truncated, unparseable, or
    contradicting itself twice -- is `held`: its attempt id is spent, so a normal run skips it and the
    unit loses both arms. Its evidence is still readable, which is what lets a repair pass tell "this
    one already has a verdict" from "this one has a failure and needs another call".
    """
    path = _verdict_path(evidence_root, judgement)
    if not path.exists():
        return True, "no verdict recorded"
    try:
        document = pp.parse_json(path.read_bytes())
    except (ValueError, OSError):
        return True, "verdict unreadable"
    failure = recorded_failure(document)
    if failure:
        return True, str(failure)[:120]
    return False, None


def _preserve(path):
    """Move an existing evidence file aside rather than overwriting it."""
    if not path.exists():
        return None
    for index in range(1, 100):
        candidate = path.with_name(f"{path.stem}.prior{index}{path.suffix}")
        if not candidate.exists():
            os.replace(path, candidate)
            return candidate
    raise JudgeError(f"cannot preserve {path}: too many prior readings")


def _free_repair_tag(judgement, taken):
    """The first repair attempt id this judgement has never used.

    A repair that fails leaves its attempt id spent, and the guard stops the run; the next pass has
    to use a NEW id for the same judgement. A fixed `-r2` would collide with the failed attempt and
    the ledger would refuse the reservation, so the tag escalates: `-r2`, `-r3`, ... The alternative
    -- reusing the id -- is exactly what the ledger exists to prevent.
    """
    for index in range(2, 100):
        tag = f"-r{index}"
        if f"{judgement.attempt_id}{tag}" not in taken:
            return tag
    raise JudgeError(f"{judgement.judgement_id}: no unused repair attempt id remains")


def _prepare_repair(judgement, prepare, evidence_root, taken):
    """Re-judge a held judgement under a new attempt id. None when it needs no repair."""
    needed, reason = _needs_repair(evidence_root, judgement)
    if not needed:
        return None
    plan, admission, identity, slot_by_arm = prepare(
        judgement, tag=_free_repair_tag(judgement, taken))
    _preserve(_verdict_path(evidence_root, judgement))
    print(f"  {judgement.judgement_id[:58]:58} repairing: {reason}", flush=True)
    return plan, admission, identity, slot_by_arm


def _ledger_hop(judgement, packed, *, held, prepare, repair_invalid, evidence_root):
    """The plan to send, or None plus the `held_back` note when this slot is spent and stays spent.

    `repair_invalid` is the single exception to "a consumed judgement is never resent", and it is
    narrow on purpose: a held attempt is re-judged only when its stored verdict carries a `failure`,
    under a new `-r2` attempt id, so the spent slot is never reused and a verdict that already parsed
    is never re-bought.
    """
    plan = packed[0]
    if plan.spec.attempt_id not in held:
        return packed, None
    if repair_invalid and prepare is not None:
        repair = _prepare_repair(judgement, prepare, evidence_root, held)
        if repair is not None:
            return repair, None
    return None, dict(attempt_id=plan.spec.attempt_id, state="CONSUMED",
                      reason="already held; a consumed judgement is never resent")


def _record_outcome(judgement, plan, slot_by_arm, outcome, *, ledger, confound, evidence_root,
                    prepare):
    """Verdict for one executed call, with the single corrective re-ask when it applies."""
    result, latency_ms, observed = outcome
    line, failure, _, error = _judge_outcome(
        plan, ledger, judgement, slot_by_arm, result, latency_ms, observed, confound, evidence_root)
    if not isinstance(error, ResolutionConflict) or prepare is None:
        return line, failure, False
    again = _corrective_reattempt(
        judgement=judgement, prepare=prepare, ledger=ledger, confound=confound,
        evidence_root=evidence_root, first_line=line, conflict=error)
    if again is None:
        return line, failure, False
    line, failure, _, _ = again
    return line, failure, True


def _execute_plans(plans, *, ledger, confound, evidence_root, held, stop_after_failures,
                   prepare=None, repair_invalid=False):
    """Walk the prepared plans. Returns (lines, held_back, stop_reason).

    One judgement whose reply contradicts itself is re-asked once, through `prepare`, with the rule
    quoted and no hint at the answer. Every other rejection is final: a malformed reply or an
    invented criterion says the instrument was driven wrong, and buying the same failure again is how
    a run spends its envelope learning nothing.
    """
    results, held_back, consecutive = [], [], 0
    for judgement, packed in plans:
        packed, note = _ledger_hop(judgement, packed, held=held, prepare=prepare,
                                   repair_invalid=repair_invalid, evidence_root=evidence_root)
        if note is not None:
            held_back.append(note)
            continue
        plan, admission, identity, slot_by_arm = packed
        try:
            outcome = asyncio.run(_execute_one(plan, ledger, admission, identity))
        except (sl.LedgerError, ValueError) as exc:
            # An accounting refusal (a budget that shrank between plan and send, an incident) is
            # a stop, not a per-row failure: continuing would buy calls the ledger just refused.
            return results, held_back, f"stopped: {type(exc).__name__}: {exc}"
        line, failure, retried = _record_outcome(
            judgement, plan, slot_by_arm, outcome, ledger=ledger, confound=confound,
            evidence_root=evidence_root, prepare=prepare)
        results.append(line)
        print(f"  {plan.spec.attempt_id[:58]:58} {line['state']:11} "
              f"${Decimal(line['cost_usd'] or '0'):.6f}  "
              f"{'FAILED: ' + failure[:60] if failure else 'ok'}"
              f"{'  (re-asked)' if retried else ''}", flush=True)
        consecutive = consecutive + 1 if failure else 0
        if consecutive >= stop_after_failures:
            return results, held_back, (
                f"stopped after {consecutive} consecutive judge failures; the run stopped buying "
                f"the same failure")
    return results, held_back, None


def run_execute(judgements, *, ledger, prepare, manifest, evidence_root, study, outputs,
                partial_ok=False, refused=None, unpaired=None, unassigned=None,
                stop_after_failures=3, repair_invalid=False):
    """Reserve, send, and record one judgement at a time — never resend a consumed slot.

    The envelope arithmetic is only reviewable if the calls are sequential: two concurrent
    reservations would both pass the same headroom check, and the second would either overrun the
    envelope or be refused after the first was already paid for.
    """
    refused, unpaired, unassigned = refused or {}, unpaired or [], unassigned or []
    confound = confound_record(manifest, study)
    plans, failed = _prepare_all(judgements, prepare)
    # The manifest names the judge the way the STUDY names it ("google/gemini-4-argon"); the route
    # actually called carries the dated slug ("google/gemini-4-argon-20261013"). The evidence guard
    # compares the dated one, so recording only the study name failed every judgement with
    # "evidence would record a confound for a different judge route" -- after the call was paid for.
    # Both names are kept: the study's, so the citation stays traceable, and the route's, so the
    # guard compares like with like.
    if plans:
        confound = dict(confound, judge_study_name=manifest.get("judge_snapshot"),
                        judge_snapshot=plans[0][1][0].route.canonical_model_slug)
    reservations = [reservation_micros(plan) for _, (plan, _, _, _) in plans]
    status = ledger.status()
    affordability = _affordability(status, reservations)
    reasons = list(affordability["reasons"])
    if failed and not partial_ok:
        reasons.append(f"{len(failed)} judgement(s) cannot be prepared: {failed[:3]}")
    reasons += _refusals(refused=refused, unassigned=unassigned, judgements=judgements,
                         outputs=outputs, partial_ok=partial_ok)
    results, held_back, stopped = [], [], None
    if not reasons:
        results, held_back, stopped = _execute_plans(
            plans, ledger=ledger, confound=confound, evidence_root=evidence_root,
            held=set(status["attempt_ids"]), stop_after_failures=stop_after_failures,
            prepare=prepare, repair_invalid=repair_invalid)
    if stopped:
        reasons.append(stopped)
    return dict(mode="execute", manifest_sha256=digest(manifest),
                judge_snapshot=manifest.get("judge_snapshot"), judgements=len(judgements),
                calls=len(results), results=results, held_back=held_back,
                prepare_failures=failed, refused_tasks=refused, unpaired=unpaired,
                unassigned_outputs=unassigned, envelope=affordability, confound=confound,
                refused_reasons=reasons)


# --------------------------------------------------------------------------- CLI


def authorise_token(value):
    """The judge's own token. `PAID-PILOT-...` is not authority to judge, and vice versa.

    Two distinct tokens because the two phases spend on different questions: a pilot token that
    also authorised judge calls would let a generation run spend against a rubric nobody reviewed.
    """
    if not value:
        raise SystemExit("AUTHORISE no token supplied — the judge phase is a procedure, not a "
                         "permission")
    if value.startswith("PAID-PILOT-"):
        raise SystemExit(f"AUTHORISE token {value!r} is the GENERATION token; the judge phase "
                         f"requires PAID-JUDGE-<date>")
    if not value.startswith("PAID-JUDGE-"):
        raise SystemExit(f"AUTHORISE token {value!r} does not match PAID-JUDGE-<date>")
    try:
        stamp = datetime.date.fromisoformat(value.removeprefix("PAID-JUDGE-"))
    except ValueError:
        raise SystemExit(f"AUTHORISE token {value!r} has no ISO date") from None
    if stamp > datetime.date.today():
        raise SystemExit(f"AUTHORISE token is dated {stamp}, in the future")
    return value


def _load_manifest(raw):
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise JudgeError(f"manifest is not JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL:
        raise JudgeError(f"manifest protocol must be {PROTOCOL!r}")
    if manifest.get("judge_designated") is not True:
        raise JudgeError("manifest records no judge designation; refusing to spend on it")
    if not isinstance(manifest.get("assignments"), list) or not manifest["assignments"]:
        raise JudgeError("manifest holds no assignments")
    return manifest


def _mode(args):
    chosen = [name for name, flag in (("plan", args.plan), ("dry-run", args.dry_run),
                                      ("execute", args.execute)) if flag]
    if len(chosen) != 1:
        raise SystemExit("choose exactly one of --plan, --dry-run, --execute")
    return chosen[0]


@dataclass(frozen=True)
class Phase:
    """Everything read from disk before a mode runs, so `main` stays a dispatch."""

    manifest: dict
    raw: bytes
    judge_snapshot: str
    ledger: object
    outputs: tuple
    skipped: dict
    pairs: list
    unpaired: list
    unassigned: list
    judgements: list
    refused: dict


def _phase(args):
    raw = args.manifest.read_bytes()
    manifest = _load_manifest(raw)
    judge_snapshot = manifest["judge_snapshot"]
    if not isinstance(judge_snapshot, str) or not judge_snapshot.strip():
        raise SystemExit("manifest records no judge_snapshot")
    ledger = sl.Ledger.open(args.ledger, program_id=PROGRAM_ID, store_id=STORE_ID)
    outputs, skipped = collected(ledger, args.evidence, tuple(args.scope or ()))
    pairs, unpaired, unassigned = pair_units(outputs, manifest)
    judgements, refused = build_judgements(pairs, manifest, unit=args.unit)
    return Phase(manifest, raw, judge_snapshot, ledger, outputs, skipped, pairs, unpaired,
                 unassigned, judgements, refused)


def _authority(mode, authorise):
    """A token is authority for `--execute` only; a plan needs no permission."""
    if mode != "execute":
        if authorise:
            raise SystemExit("--authorise belongs to --execute; a token is not a plan")
        return
    authorise_token(authorise)
    if not os.environ.get(pa.CREDENTIAL_ENV):
        raise SystemExit(f"{pa.CREDENTIAL_ENV} absent; refusing to start a judge run that cannot "
                         f"send. Export it in THIS shell.")


def _dispatch(mode, phase, args):
    def prepare(judgement, correction=None, tag=None):
        """Build one request. `correction`/`tag` are set only by the corrective re-ask."""
        return prepare_judgement(judgement, base=args.base, manifest_raw=phase.raw,
                                 evidence_root=args.evidence, judge_snapshot=phase.judge_snapshot,
                                 discard_unsent=args.discard_unsent,
                                 held=tuple(phase.ledger.status()["attempt_ids"]),
                                 correction=correction, tag=tag)

    common = dict(ledger=phase.ledger, prepare=prepare, manifest=phase.manifest,
                  evidence_root=args.evidence, study=args.study, outputs=phase.outputs,
                  partial_ok=args.partial_ok, refused=phase.refused, unpaired=phase.unpaired,
                  unassigned=phase.unassigned)
    if mode == "plan":
        return run_plan(phase.judgements, **common)
    if mode == "dry-run":
        return run_dry_run(phase.judgements, **common)
    return run_execute(phase.judgements, **common,
                       stop_after_failures=args.stop_after_failures,
                       repair_invalid=args.repair_invalid)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True,
                    help="the assignment manifest the generation run was bound to")
    ap.add_argument("--study", type=pathlib.Path, default=REPO / "studies" / "revision-v1.yaml")
    ap.add_argument("--base", type=pathlib.Path, required=True,
                    help="retained route evidence (records/, verified/, verification/)")
    ap.add_argument("--ledger", type=pathlib.Path, required=True)
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--plan", action="store_true", help="print the plan and refuse if unaffordable")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every request body and validate it; send nothing")
    ap.add_argument("--execute", action="store_true", help="reserve, send, record")
    ap.add_argument("--authorise", default=None, help="PAID-JUDGE-YYYY-MM-DD")
    ap.add_argument("--unit", default="judge",
                    help="unit tag; a new tag is a new ledger slot, an existing one is a replay")
    # NOT `--unit`: that name is already the tag for the judge's OWN attempt ids, and a duplicate
    # option silently replaced it -- `tuple(args.unit)` became the characters of a string, so every
    # single-letter "marker" matched and the scope selected everything.
    ap.add_argument("--scope", action="append", default=None, metavar="TAG",
                    help="only judge attempts whose id contains this tag (repeatable). A ledger "
                         "outlives a run, and a previous run's attempts are bound to another manifest")
    ap.add_argument("--discard-unsent", action="store_true",
                    help="before each judgement, drop an evidence directory that carries no send "
                         "marker. A --plan run writes each attempt's manifest, so a later --execute "
                         "in the same root collides with its own leftovers and stops after ONE "
                         "judgement; a directory holding transport/response/receipt is never touched")
    ap.add_argument("--partial-ok", action="store_true",
                    help="proceed knowing some tasks are refused or some plans fail")
    ap.add_argument("--stop-after-failures", type=int, default=3,
                    help="consecutive failed judgements end the run")
    ap.add_argument("--repair-invalid", action="store_true",
                    help="re-judge a HELD judgement whose stored verdict carries a failure, under a "
                         "new -r2 attempt id. The spent slot is never reused; a verdict that already "
                         "parsed is never re-bought. For repairing a run whose replies were truncated")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    mode = _mode(args)
    _authority(mode, args.authorise)
    phase = _phase(args)
    print(f"  judge        : {phase.judge_snapshot}")
    print(f"  manifest     : {digest(phase.manifest)[:16]}  rows={len(phase.manifest['assignments'])}")
    print(f"  collected    : {len(phase.outputs)} retained outputs  {phase.skipped or ''}")
    print(f"  pairs        : {len(phase.pairs)}  unpaired={len(phase.unpaired)}  "
          f"unassigned={len(phase.unassigned)}")
    print(f"  judgements   : {len(phase.judgements)}  refused tasks={sorted(phase.refused)}")
    report = _dispatch(mode, phase, args)
    report["skipped_outputs"] = phase.skipped
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["refused_reasons"]:
        print("\n  REFUSED:", flush=True)
        for reason in report["refused_reasons"]:
            print(f"    - {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
