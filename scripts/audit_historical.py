"""Read-only causal inventory; writes NEW versioned disposition artifacts only.

No raw grade is promoted to a repaired measurement. Source reconstruction is
explicitly distinguished from recorded prompt provenance, which is absent.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from measurement_contract import PROTOCOL, digest, diagnostic_score


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_id(row, kind):
    fields = ["task_id", "model", "sample"] + (["evaluator_model"] if kind == "evaluations" else [])
    return tuple(row.get(k) for k in fields)


def audit(repo, inventory):
    hashes = inventory["raw_results_file_sha256"]
    for name, expected in hashes.items():
        if file_hash(repo / name) != expected:
            raise ValueError(f"Raw historical input changed: {name}")
    data = json.loads((repo / "results/benchmark_results.json").read_text())
    sources = defaultdict(list)
    raw_files = []
    for name, sha in sorted(hashes.items()):
        entry = {"path": name, "sha256": sha, "disposition": "retain_immutable_historical", "report_protocol": PROTOCOL}
        if name.endswith(".yaml"):
            raw = yaml.safe_load((repo / name).read_text())
            if isinstance(raw, dict) and raw.get("task_id") and raw.get("model"):
                kind = "evaluations" if "evaluator_model" in raw else "generations"
                ident = (raw["task_id"], raw["model"], raw.get("sample_index", 0))
                if kind == "evaluations":
                    ident += (raw["evaluator_model"],)
                sources[(kind, ident)].append((name, raw))
                entry.update(original_id=raw.get("evaluation_id", raw.get("generation_id")), identity=list(ident), kind=kind,
                             original_date=raw.get("timestamp", raw.get("metadata", {}).get("timestamp")),
                             original_protocol=raw.get("protocol", "unrecorded"))
        raw_files.append(entry)
    tasks = {}
    task_sources = {}
    for p in (repo / "dataset/tasks").rglob("*.yaml"):
        task = yaml.safe_load(p.read_text())
        if isinstance(task, dict) and "task_id" in task:
            tasks[task["task_id"]] = task
            task_sources[task["task_id"]] = {"path": str(p.relative_to(repo)), "sha256": file_hash(p), "binding": "current frozen-source reconstruction; historical version unrecorded"}
    rows = []
    duplicate_groups = {}
    for kind in ("generations", "evaluations"):
        grouped = defaultdict(list)
        for index, row in enumerate(data[kind]):
            grouped[row_id(row, kind)].append(index)
        duplicate_groups[kind] = [{"identity": list(k), "row_indexes": v,
                                   "row_hashes": [digest(data[kind][i]) for i in v]}
                                  for k, v in grouped.items() if len(v) > 1]
        for index, row in enumerate(data[kind]):
            ident = row_id(row, kind)
            source_records = sources.get((kind, ident), [])
            gen_records = sources.get(("generations", ident[:3]), [])
            task = tasks.get(row["task_id"], {})
            task_type = row.get("task_type", task.get("task_type", "unknown"))
            reasons = ["R10: original task/prompt/model/judge/protocol content bindings absent; no repaired-regime pooling",
                       "R02: full failed/retry/operational/price ledger not recorded; original amount is not full-program cost"]
            actions = {"quarantine_provenance"}
            primary = "quarantine"
            details = {}
            if len(grouped[ident]) > 1:
                reasons.append("R10: conflicting duplicate identity; all versions retained, no winner selected")
                details["conflicting_indexes"] = grouped[ident]
            elif "constraint_discovery" in task_type:
                primary = "regenerate"
                actions.add("regenerate")
                reasons.append("R06: measured topic-polarity oracle is unqualified; dependent generation and grades cannot be repaired by reaggregation")
            elif kind == "evaluations":
                if "iterative_revision" in task_type or "critique_improvement" in task_type:
                    limit = 400 if "critique_improvement" in task_type else 500
                    clipped = [i for _, g in gen_records for i, turn in enumerate(g.get("turns", []))
                               if turn.get("turn_type") == "generation" and len(turn.get("content", "")) > limit]
                    history_cap = any(sum(t.get("turn_type") == "generation" for t in g.get("turns", [])) > 4 for _, g in gen_records) if "critique_improvement" in task_type else False
                    if clipped or history_cap:
                        primary = "rejudge"; actions.add("rejudge")
                        details["source_reconstructed_clipped_turn_indexes"] = clipped
                        reasons.append("R05: frozen judge builder drops recorded version content/history; rejudge with complete context")
                elif "planning_execution" in task_type:
                    plans = [t.get("content", "") for _, g in gen_records for t in g.get("turns", []) if t.get("turn_type") == "plan"]
                    if any(len(plan) > 2000 for plan in plans):
                        primary = "rejudge"; actions.add("rejudge")
                        reasons.append("R05: recorded plan exceeds frozen 2000-character judge window")
                else:
                    no_flaw = task.get("ground_truth", {}).get("has_flaw") is False
                    limits = {"beat_interpolation": [("beat_before", "content", 800), ("beat_after", "content", 800)],
                              "beat_revision": [("flawed_segment", "content", 1000 if no_flaw else 800), ("", "beat_definition", 600)] + ([("ground_truth", "reasoning", 500)] if no_flaw else []),
                              "constrained_continuation": [("story_opening", "content", 800)],
                              "theory_conversion": [("original_segment", "content", 1000)],
                              "multi_beat_synthesis": [("story_context", "protagonist", 300), ("story_context", "setting", 200), ("story_context", "central_conflict", 200)]}
                    clipped = [f"{parent}.{field}" for parent, field, limit in limits.get(task_type, [])
                               if len(str((task.get(parent, {}) if parent else task).get(field, "")).strip()) > limit]
                    if clipped:
                        primary = "rejudge"; actions.add("rejudge")
                        details["source_reconstructed_clipped_task_fields"] = clipped
                        reasons.append("R05: frozen judge source reconstructs clipped required context; historical prompt hash unavailable")
                subtype = "no_flaw" if task.get("ground_truth", {}).get("has_flaw") is False else "flawed" if task_type == "beat_revision" else "default"
                score_type = task_type.removeprefix("agentic_")
                try:
                    diagnostic_score(row.get("llm_results"), score_type, subtype)
                except ValueError as exc:
                    primary = "rejudge"; actions.add("rejudge")
                    reasons.append("R04: original judge response fails explicit schema: " + str(exc))
                breakdown = row.get("score_breakdown", {}).get("components", {}).get("programmatic", {}).get("breakdown", {})
                if "element_count_score" in breakdown:
                    values = [breakdown.get(k) for k in ["word_count_score", "repetition_score", "slop_score", "element_count_score"]]
                    if all(type(v) in (int, float) and 0 <= v <= 1 for v in values):
                        details["R03_corrected_unvalidated_optional_diagnostic"] = sum(w * v for w, v in zip([.27, .27, .27, .10], values)) / .91
                        actions.add("aggregation_correction_only")
                reasons.append("R07/R08: legacy scalar is not calibrated literary quality, resolved success or paired benefit")
            elif "iterative_revision" in task_type:
                reasons.append("R05: feedback quote is clipped but full assistant versions remain in conversation; clipping alone does not prove generation deprivation")
            # Conflicting IDs always remain quarantined even if another causal action applies.
            if len(grouped[ident]) > 1:
                primary = "quarantine"
            rows.append({"kind": kind, "row_index": index, "identity": list(ident), "raw_row_sha256": digest(row),
                         "raw_file": "results/benchmark_results.json", "raw_file_sha256": hashes["results/benchmark_results.json"],
                         "original_protocol": data.get("benchmark_version", "unrecorded"), "original_row_protocol": row.get("protocol", "unrecorded"),
                         "original_date": row.get("timestamp"), "original_success_field": row.get("success"),
                         "original_amount_usd": row.get("generation_cost" if kind == "generations" else "evaluation_cost"),
                         "original_source_ids": [{"path": name, "id": raw.get("evaluation_id", raw.get("generation_id")), "sha256": hashes[name]} for name, raw in source_records],
                         "primary_disposition": primary, "required_actions": sorted(actions), "reasons": reasons, "details": details,
                         "report_protocol": PROTOCOL, "resolved": None})
    costs = {}
    for kind, field in [("generations", "generation_cost"), ("evaluations", "evaluation_cost")]:
        duplicate_indexes = {i for group in duplicate_groups[kind] for i in group["row_indexes"]}
        amounts = [r.get(field) for i, r in enumerate(data[kind]) if i not in duplicate_indexes]
        costs[kind] = {"nonconflicting_recorded_amount_sum_usd": sum(v for v in amounts if type(v) in (int, float) and v >= 0),
                       "excluded_conflicting_rows": sorted(duplicate_indexes), "all_attempt_total_usd": None,
                       "scope": "sum of recorded amounts including failed rows; unknown ledger completeness/prices; no comparative value claim"}
    return {"protocol": PROTOCOL, "inventory_hash": digest(inventory), "raw_files": raw_files, "task_reconstruction_sources": task_sources,
            "rows": rows, "counts": dict(Counter((r["kind"] + ":" + r["primary_disposition"]) for r in rows)),
            "duplicate_groups": duplicate_groups, "recorded_cost_correction": costs,
            "no_repaired_regime_pooling": True, "direct_paid_usd": 0, "host_billing_usd": None,
            "note": "No labels, prices, resolved successes or original dates are invented. Original files remain immutable. Synthetic software checks do not calibrate literary judgments."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = audit(args.repo, json.loads(args.inventory.read_text()))
    if not args.verify_only:
        output = args.repo / "results/analysis/measurement-repair-v1"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "disposition.json").open("x") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        with (output / "report.md").open("x") as f:
            f.write("# Historical measurement disposition, v1\n\nRaw files and dates are unchanged. No repaired-regime comparison or resolved outcome is claimed.\n\n" + json.dumps(report["counts"], indent=2) + "\n\nRecorded amounts are arithmetic corrections only; full program spend remains unknown. All row IDs, hashes, original source IDs/dates, causal actions, and three conflicting identity groups are in disposition.json.\n\nOracle episodes require regeneration. Incomplete grading requires rejudging. Ambiguous provenance and duplicate identities remain quarantined. Iterative feedback clipping alone does not prove missing generation context because earlier assistant versions remain in the conversation.\n")
    print(json.dumps({"raw_files": len(report["raw_files"]), "rows": len(report["rows"]), "counts": report["counts"], "conflicting_eval_ids": len(report["duplicate_groups"]["evaluations"])}))


if __name__ == "__main__":
    main()
