"""W1 validators and report projections, not native story/outcome authorities.

These diagnostics are uncalibrated. Schema/byte binding does not establish
editorial truth, author acceptance, or treatment benefit. No provider is enabled.
"""
import hashlib
import json
import math
from collections import Counter
from typing import Any

PROTOCOL = "measurement-repair-v1"
STATUSES = {"completed", "agent_failure", "absent", "provider_failure",
            "infrastructure_failure", "grader_failure", "not_run", "coverage_failure", "truncated"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def number(value, name="number", maximum=1):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid finite {name}: {value!r}")
    return float(value)


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}")
    return value


def require_paid_dispatch():
    raise ValueError("Paid dispatch disabled until W3 atomic program ledger is qualified")


def require_qualified_oracle():
    raise ValueError("Semantic oracle is unqualified and disabled; W5 truth calibration required")


SCORE_WEIGHTS = {
    ("beat_interpolation", "default"): dict(zip(
        ["beat_elements_score", "beat_execution_score", "must_not_score", "character_score", "bridge_score", "continuity_score"], [.25, .25, .15, .1, .15, .1])),
    ("beat_revision", "flawed"): dict(zip(
        ["diagnosis_score", "flaw_correction_score", "beat_satisfaction_score", "preservation_score", "required_preserved_score", "minimal_change_score", "quality_score"], [.2, .2, .2, .1, .1, .1, .1])),
    ("beat_revision", "no_flaw"): dict(zip(
        ["correct_diagnosis_score", "false_positive_avoided_score", "beat_understanding_score", "reasoning_quality_score"], [.4, .3, .15, .15])),
    ("constrained_continuation", "default"): dict(zip(
        ["beats_score", "must_include_score", "must_not_score", "tone_score", "ending_score"], [.2, .3, .25, .15, .1])),
    ("theory_conversion", "default"): dict(zip(
        ["beats_score", "preservation_score", "structural_accuracy_score", "tone_score"], [.35, .3, .2, .15])),
    ("multi_beat_synthesis", "default"): dict(zip(
        ["beat_requirements_score", "cross_beat_score", "context_score", "coherence_score"], [.4, .35, .15, .1])),
}
AGENT_PROCESS = {
    "planning_execution": ["plan_completeness", "plan_specificity", "plan_adherence"],
    "iterative_revision": ["improvement_trajectory", "feedback_responsiveness", "preservation"],
    "critique_improvement": ["critique_responsiveness", "improvement_trajectory", "preservation"],
}
AGENT_OUTPUT = ["constraint_satisfaction", "beat_execution", "narrative_quality"]
for _kind, _fields in AGENT_PROCESS.items():
    _p = {"planning_execution": .35, "iterative_revision": .3, "critique_improvement": .35}[_kind]
    SCORE_WEIGHTS[(_kind, "default")] = {**dict.fromkeys(_fields, _p / 3), **dict.fromkeys(AGENT_OUTPUT, (1 - _p) / 3)}

AUXILIARY_FIELDS = {
    ("beat_interpolation", "default"): {"elements_found", "elements_total", "violations_found", "word_count_valid", "evidence"},
    ("beat_revision", "flawed"): {"preserved_count", "preserved_total", "required_preserved_count", "required_preserved_total", "word_count_valid", "evidence"},
    ("beat_revision", "no_flaw"): {"model_said_no_revision", "evidence"},
    ("constrained_continuation", "default"): {"beats_present", "beats_total", "must_include_present", "must_include_total", "must_not_avoided", "must_not_total", "word_count_valid", "failed_constraints"},
    ("theory_conversion", "default"): {"beats_present", "beats_total", "preserved_count", "preserved_total", "word_count_valid", "evidence"},
    ("multi_beat_synthesis", "default"): {"beat_reqs_satisfied", "beat_reqs_total", "cross_beat_satisfied", "cross_beat_total", "word_count_valid", "failed_items"},
    ("planning_execution", "default"): {"constraints_satisfied", "constraints_total", "beats_satisfied", "beats_total", "evidence"},
    ("iterative_revision", "default"): {"constraints_satisfied", "constraints_total", "evidence"},
    ("critique_improvement", "default"): {"constraints_satisfied", "constraints_total", "evidence"},
}
COUNT_PAIRS = [("elements_found", "elements_total"), ("preserved_count", "preserved_total"),
               ("required_preserved_count", "required_preserved_total"), ("beats_present", "beats_total"),
               ("must_include_present", "must_include_total"), ("must_not_avoided", "must_not_total"),
               ("beat_reqs_satisfied", "beat_reqs_total"), ("cross_beat_satisfied", "cross_beat_total"),
               ("constraints_satisfied", "constraints_total"), ("beats_satisfied", "beats_total")]


def schema_weights(task_type, subtype=None):
    if task_type in ("constraint_discovery", "agentic_constraint_discovery"):
        require_qualified_oracle()
    if subtype is None and task_type != "beat_revision":
        subtype = "default"
    try:
        return SCORE_WEIGHTS[(task_type, subtype)]
    except (KeyError, TypeError):
        raise ValueError(f"Explicit known task/subtype required: {task_type}/{subtype}") from None


def validate_verdict(value):
    """Numeric legacy diagnostic or explicit four-state criterion observation."""
    if not isinstance(value, dict):
        return number(value, "criterion")
    verdict = value.get("verdict")
    if verdict not in {"satisfied", "violated", "unknown", "not_applicable"}:
        raise ValueError("Unknown criterion verdict")
    text(value.get("rationale"), "criterion rationale")
    if set(value) != {"verdict", "rationale", "evidence"}:
        raise ValueError("Malformed criterion fields")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or any(not isinstance(v, str) or not v for v in evidence):
        raise ValueError("Malformed evidence collection")
    if verdict in {"satisfied", "violated"} and not evidence:
        raise ValueError("Decisive verdict requires evidence")
    return {"satisfied": 1., "violated": 0., "unknown": None, "not_applicable": "N/A"}[verdict]


def diagnostic_score(results, task_type, subtype=None):
    weights = schema_weights(task_type, subtype)
    if not isinstance(results, dict) or not weights.keys() <= results.keys():
        raise ValueError("Missing required judge fields")
    auxiliary = AUXILIARY_FIELDS[(task_type, subtype or "default")]
    if results.keys() - (weights.keys() | auxiliary):
        raise ValueError("Unknown judge field for selected task/subtype")
    # Extra fields are permitted only from this task's existing prompt contract.
    for key, value in results.items():
        if key in weights:
            continue
        if key == "evidence":
            text(value, "judge evidence")
        elif key in {"word_count_valid", "model_said_no_revision"}:
            if type(value) is not bool:
                raise ValueError("word_count_valid must be boolean")
        elif key in {"failed_constraints", "failed_items"}:
            if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
                raise ValueError("Malformed finding collection")
        elif key in {"elements_found", "elements_total", "violations_found", "constraints_satisfied", "constraints_total", "beats_satisfied", "beats_total", "preserved_count", "preserved_total", "required_preserved_count", "required_preserved_total", "beats_present", "must_include_present", "must_include_total", "must_not_avoided", "must_not_total", "beat_reqs_satisfied", "beat_reqs_total", "cross_beat_satisfied", "cross_beat_total"}:
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid count {key}")
        else:
            raise ValueError(f"Unknown judge field: {key}")
    for count, total in COUNT_PAIRS:
        if count in auxiliary and (count in results or total in results):
            if count not in results or total not in results or results[count] > results[total]:
                raise ValueError(f"Inconsistent or incomplete count pair: {count}/{total}")
    values = {key: validate_verdict(results[key]) for key in weights}
    if any(v is None for v in values.values()):
        return None
    applicable = {k: v for k, v in values.items() if v != "N/A"}
    if not applicable:
        return None
    return sum(weights[k] * v for k, v in applicable.items()) / sum(weights[k] for k in applicable)


def context_packet(parts, *, max_bytes):
    """Bind complete entitled source strings; refuse overflow before dispatch.

    Byte budget is a conservative harness bound, not a provider tokenizer or
    proof that a model attended to text. Output finish reason is checked too.
    """
    if not isinstance(parts, dict) or not parts or any(not isinstance(v, str) for v in parts.values()):
        raise ValueError("Context requires named complete source strings")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("Explicit context byte budget required")
    packet = "\n\n".join(f"=== {k} ===\n{v}" for k, v in parts.items())
    receipt = {"protocol": PROTOCOL, "parts": {k: digest(v) for k, v in parts.items()},
               "packet_hash": digest(packet), "bytes": len(packet.encode()), "budget_bytes": max_bytes,
               "coverage": "complete", "stop_reason": None}
    if receipt["bytes"] > max_bytes:
        receipt.update(coverage="failed", stop_reason="context_budget")
        raise CoverageError(receipt)
    return packet, receipt


class CoverageError(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("Required complete context exceeds explicit budget")


def complete_response(response):
    if not response.success or response.finish_reason != "stop" or not response.content.strip():
        raise ValueError("Incomplete, truncated or failed provider response")


def identity(row):
    for field in ("model", "task_id", "condition"):
        text(row.get(field), field)
    if type(row.get("sample")) is not int or row["sample"] < 0:
        raise ValueError("Explicit nonnegative sample required")
    return row["model"], row["task_id"], row["sample"], row["condition"]


def unique_rows(rows, key):
    if not isinstance(rows, list):
        raise ValueError("Expected record collection")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed record collection")
        k = key(row)
        if k in indexed:
            reason = "conflicting" if digest(indexed[k]) != digest(row) else "duplicate"
            raise ValueError(f"{reason} record identity: {k}")
        indexed[k] = row
    return indexed


def cost_summary(attempts, *, absent=False):
    known = 0.
    unknown = int(absent)
    roles = {}
    for attempt in attempts:
        amount = attempt.get("cost_usd")
        if amount is None:
            unknown += 1
        else:
            known += number(amount, "cost", maximum=None)
        role = attempt["role"]
        roles[role] = roles.get(role, 0) + 1
    return {"known_usd": known, "unknown_attempts": unknown,
            "total_usd": None if unknown else known, "attempts": len(attempts), "roles": roles,
            "unknown_usage_attempts": sum(a.get("usage") is None for a in attempts)}


def validate_attempts(rows, allowed_roles, used_ids):
    if not isinstance(rows, list):
        raise ValueError("Explicit all-attempt ledger required")
    for row in rows:
        attempt_id = text(row.get("attempt_id"), "attempt_id")
        if attempt_id in used_ids:
            raise ValueError("Duplicate attempt ID; cannot double count spend")
        used_ids.add(attempt_id)
        if row.get("role") not in allowed_roles:
            raise ValueError("Invalid cost role")
        if row.get("status") not in STATUSES:
            raise ValueError("Explicit attempt status required")
        if "cost_usd" not in row or "usage" not in row:
            raise ValueError("Unknown cost/usage must be explicit null")
        if row["cost_usd"] is not None:
            number(row["cost_usd"], "cost", maximum=None)
        usage = row["usage"]
        if usage is not None:
            if not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens", "reasoning_tokens"}:
                raise ValueError("Malformed token usage")
            if any(type(v) is not int or v < 0 for v in usage.values()):
                raise ValueError("Malformed token count")
    return rows


def validate_binding(record, assignment, kind):
    if record.get("protocol") != PROTOCOL or record.get("task_hash") != assignment["task_hash"]:
        raise ValueError("Historical, mixed or stale task/protocol binding")
    if record.get("model_snapshot") != assignment["model_snapshot"]:
        raise ValueError("Stale model snapshot")
    if record.get("prompt_version") != assignment["prompt_version"]:
        raise ValueError("Stale prompt protocol")
    if record.get("status") not in STATUSES:
        raise ValueError("Explicit record status required")
    text(record.get("timestamp"), "timestamp")
    text(record.get("record_id"), "record_id")
    if kind == "generation" and record["status"] == "completed":
        text(record.get("output"), "output")
        if record.get("output_hash") != digest(record["output"]):
            raise ValueError("Stale output hash")
    if record["status"] not in {"not_run", "absent"}:
        if record.get("prompt_hash") != digest(record.get("prompt")):
            raise ValueError("Stale prompt hash")
        if not isinstance(record.get("prompt"), str) or not record["prompt"]:
            raise ValueError("Complete prompt required")
    if record["status"] == "completed":
        coverage = record.get("context")
        if not isinstance(coverage, dict) or coverage.get("coverage") != "complete" or record.get("finish_reason") != "stop":
            raise ValueError("Incomplete context or truncated output cannot be valid")
        parts = record.get("context_parts")
        packet, expected = context_packet(parts, max_bytes=coverage.get("budget_bytes"))
        if coverage != expected or record["prompt"] != packet:
            raise ValueError("Stale context receipt")
        if parts.get("task") != json.dumps(assignment["task"], sort_keys=True, ensure_ascii=False):
            raise ValueError("Required complete task context missing")


def project_report(data, manifest):
    """Project a fixed assigned matrix; never infer assignment from successes.

    Resolved outcomes are deliberately unavailable: W1 has no qualified native
    semantic outcome adapter. A supplied native ID is not author acceptance.
    """
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Explicit repaired assignment manifest required")
    assignments = unique_rows(manifest.get("assignments"), identity)
    if not assignments:
        raise ValueError("Nonempty assignment matrix required")
    for row in assignments.values():
        if not isinstance(row.get("task"), dict) or row.get("task_hash") != digest(row["task"]):
            raise ValueError("Stale assigned task hash")
        if row["task"].get("task_id") != row["task_id"]:
            raise ValueError("Task identity mismatch")
        text(row.get("story_id"), "story cluster identity")
        text(row.get("model_snapshot"), "model snapshot")
        text(row.get("judge_snapshot"), "judge snapshot")
        text(row.get("prompt_version"), "prompt version")
        schema_weights(row["task"].get("task_type"), row["task"].get("subtype"))
    if data.get("protocol") != PROTOCOL or data.get("assignment_hash") != digest(manifest):
        raise ValueError("Historical/mixed protocol or stale assignment manifest")
    generations = unique_rows(data.get("generations"), identity)
    evaluations = unique_rows(data.get("evaluations"), identity)
    if generations.keys() - assignments.keys() or evaluations.keys() - assignments.keys():
        raise ValueError("Unexpected unassigned record")
    unique_rows(list(generations.values()) + list(evaluations.values()), lambda r: text(r.get("record_id"), "record_id"))
    groups = {}
    seen_attempts = set()
    rows = []
    for key, assigned in assignments.items():
        generation, evaluation = generations.get(key), evaluations.get(key)
        deploy, research = [], []
        status, diagnostic = "absent", None
        if generation is not None:
            validate_binding(generation, assigned, "generation")
            status = generation["status"]
            deploy = validate_attempts(generation.get("attempts"), {"generation", "critic", "retrieval", "selection", "operational"}, seen_attempts)
            if status == "not_run" and deploy:
                raise ValueError("Not-run record has billed attempts")
            if status not in {"not_run", "absent"} and not deploy:
                raise ValueError("Attempted generation requires ledger")
            if status == "completed" and not any(a["role"] == "generation" and a["status"] == "completed" for a in deploy):
                raise ValueError("Completed generation lacks successful generation attempt")
        if evaluation is not None:
            if generation is None:
                raise ValueError("Evaluation has no generation")
            validate_binding(evaluation, assigned, "evaluation")
            if evaluation.get("judge_snapshot") != assigned["judge_snapshot"] or evaluation.get("generation_hash") != digest(generation):
                raise ValueError("Stale judge/generation binding")
            research = validate_attempts(evaluation.get("attempts"), {"research_evaluation"}, seen_attempts)
            if evaluation["status"] == "not_run" and research:
                raise ValueError("Not-run evaluation has attempts")
            if evaluation["status"] not in {"not_run", "absent"} and not research:
                raise ValueError("Attempted evaluation requires ledger")
            if evaluation["status"] == "completed":
                if status != "completed":
                    raise ValueError("Invalid generation cannot acquire a valid grade")
                if not any(a["status"] == "completed" for a in research):
                    raise ValueError("Completed evaluation lacks successful research attempt")
                if evaluation["context_parts"].get("output") != generation["output"]:
                    raise ValueError("Judge did not receive complete bound output")
                diagnostic = diagnostic_score(evaluation.get("llm_results"), assigned["task"]["task_type"], assigned["task"].get("subtype"))
            elif status == "completed":
                status = evaluation["status"]
        elif status == "completed":
            status = "grader_failure"
        group = (assigned["model"], assigned["condition"])
        groups.setdefault(group, []).append({"assignment": assigned, "status": status, "diagnostic": diagnostic,
                                            "deploy": deploy, "research": research, "absent": generation is None or generation["status"] == "absent"})
        rows.append({"identity": list(key), "story_id": assigned["story_id"], "status": status,
                     "diagnostic": diagnostic, "resolved": None})
    signatures = [{(r["assignment"]["task_id"], r["assignment"]["sample"], r["assignment"]["task_hash"], r["assignment"]["story_id"]) for r in values} for values in groups.values()]
    comparable = all(s == signatures[0] for s in signatures)
    matched = set.intersection(*[{(r["assignment"]["task_id"], r["assignment"]["sample"]) for r in values if r["diagnostic"] is not None and r["status"] == "completed"} for values in groups.values()])
    if not comparable:
        matched = set()
    summaries = []
    for (model, condition), values in sorted(groups.items()):
        valid = [r for r in values if r["status"] == "completed" and r["diagnostic"] is not None]
        complete = comparable and len(valid) == len(values)
        summaries.append({"model": model, "condition": condition, "assigned": len(values),
                          "story_clusters": len({r["assignment"]["story_id"] for r in values}),
                          "statuses": dict(Counter(r["status"] for r in values)), "valid_diagnostics": len(valid),
                          "diagnostic_coverage": len(valid) / len(values),
                          "complete_assigned_diagnostic_mean": sum(r["diagnostic"] for r in valid) / len(values) if complete else None,
                          "deployment_cost": cost_summary([a for r in values for a in r["deploy"]], absent=sum(r["absent"] for r in values)),
                          "research_evaluation_cost": cost_summary([a for r in values for a in r["research"]], absent=sum((r["status"] == "grader_failure" or r["absent"]) and not r["research"] for r in values)),
                          "resolved": None, "cost_per_resolved": None})
    return {"protocol": PROTOCOL, "assignment_hash": digest(manifest), "report_input_hash": digest(data),
            "comparability": "matched_assigned_matrix" if comparable else "incomparable_assignment_matrix",
            "matched_valid_task_samples": [list(k) for k in sorted(matched)],
            "assigned_ids": [list(k) for k in sorted(assignments)], "rows": sorted(rows, key=lambda r: r["identity"]),
            "models": summaries, "diagnostic_status": "unvalidated; no quality/value ranking",
            "resolved_status": "unavailable: native semantic outcome adapter not qualified in W1"}
