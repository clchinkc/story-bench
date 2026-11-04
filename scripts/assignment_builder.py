#!/usr/bin/env python3
"""Emit the W1 assignment manifest that a dispatch attempt must be bound to.

WHY THIS EXISTS. `prepare_attempt` refuses without an `assignment` artifact, and `_validate_spec`
re-derives the assigned row from it: task bytes, sample, condition and model snapshot must match the
attempt exactly, or the call is not the call the study pre-registered. No producer for that manifest
existed in this repo -- every grep for one found only its consumers. So the plan's "wire an adapter"
was a euphemism for a missing input, not a missing layer, and this file is that input.

WHAT IS DECLARED, NOT GUESSED. Three of the digests have no definition anywhere in the contract, so
they are DEFINED here, in one place, and each one names what it fingerprints:

    task           the frozen task document, canonical JSON
    prompt         the bytes the harness will send, built by the same builder the run uses
    settings       the request fields, canonical JSON
    harness        the script that assembles an attempt (this runner)
    protocol       the frozen study config the run reads
    native_package the frozen task tree the mix was counted from
    assignment     this manifest itself

That is a declaration of provenance, not a measurement of it: each digest is computed from the bytes
the run actually ships, so the check `digest_text(bound) == refs[name].sha256` is honest even though
the choice of object is ours.

    uv run python scripts/assignment_builder.py --rehearsal --judge unassigned --out /tmp/a.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from measurement_contract import PROTOCOL, digest, project_report  # noqa: E402

UNASSIGNED = "unassigned:no-judge-designated"
PROMPT_VERSION = "w3b2-single-call-v1"


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def subtype_of(task: dict) -> str | None:
    """The subtype `schema_weights` needs, derived from the document rather than assumed.

    `beat_revision` is the one family the contract refuses to default: it scores a `flawed` task
    differently from a `no_flaw` one, and all five frozen files carry `flawed_segment`. Reading that
    key is what makes the subtype a property of the task instead of a guess at it.
    """
    if task.get("subtype"):
        return task["subtype"]
    if task.get("task_type") == "beat_revision":
        return "flawed" if task.get("flawed_segment") else "no_flaw"
    return None


def task_documents(repo: pathlib.Path, families: dict[str, int]) -> dict[str, dict]:
    """The frozen tasks the mix counts, keyed by task_id. Counts are checked, not trusted.

    The document recorded in a row is the frozen task plus its DERIVED subtype (`subtype_of`), and
    `task_hash` covers that same document, so the binding is self-consistent. The derivation is
    stated here because a reader comparing these bytes against the yaml on disk would otherwise see
    one added key and have to guess where it came from.
    """
    import yaml  # noqa: PLC0415

    docs, counted = {}, {}
    for path in sorted((repo / "dataset" / "tasks").rglob("*.yaml")):
        task = yaml.safe_load(path.read_text())
        if not (isinstance(task, dict) and "task_id" in task and task.get("task_type") in families):
            continue
        recorded = dict(task)
        subtype = subtype_of(recorded)
        if subtype:
            recorded["subtype"] = subtype
        docs[task["task_id"]] = recorded
        counted[task["task_type"]] = counted.get(task["task_type"], 0) + 1
    wrong = {family: (counted.get(family, 0), count) for family, count in families.items()
             if counted.get(family, 0) != count}
    if wrong:
        raise SystemExit(f"task counts disagree with the declared mix (found, declared): {wrong}")
    return docs


def snapshots(base: pathlib.Path) -> dict[str, str]:
    """model_id -> canonical_model_slug, read from the retention records (never from a catalogue)."""
    out = {}
    for path in sorted((base / "records").glob("*.json")):
        ident = json.loads(path.read_text())["route_identity"]
        out[ident["model_id"]] = ident["canonical_model_slug"]
    if not out:
        raise SystemExit(f"{base}: no records; a manifest cannot name a model snapshot")
    return out


def story_of(task: dict) -> str:
    for key in ("story_id", "cluster"):
        if task.get(key):
            return str(task[key])
    meta = task.get("metadata") or {}
    return str(meta.get("story_id") or f"{task['task_type']}:{meta.get('genre', 'none')}")


def rows_for(units: list[dict], routes: list[str], snaps: dict[str, str],
             tasks: dict[str, dict], judge: str) -> list[dict]:
    """One row per (unit, model, arm). `sample` is the OCCURRENCE index, not a constant.

    The design reuses each frozen task about 5.76 times per model, and `identity()` is
    (model, task_id, sample, condition) -- so a fixed sample of 0 made every reuse a duplicate and
    `project_report` refused the whole matrix. The counter is what makes a repeat a repeat rather
    than a collision.
    """
    rows, seen = [], {}
    for unit in units:
        task = tasks[unit["task_id"]]
        for route in routes:
            for arm in ("A", "D"):
                key = (route, unit["task_id"], arm)
                sample = seen.get(key, 0)
                seen[key] = sample + 1
                rows.append({
                    "model": route, "model_snapshot": snaps[route],
                    "task_id": unit["task_id"], "task": task, "task_hash": digest(task),
                    "sample": sample, "condition": arm, "story_id": story_of(task),
                    "judge_snapshot": judge, "prompt_version": PROMPT_VERSION,
                })
    return rows


def artifacts_of(repo: pathlib.Path) -> dict[str, str]:
    """The three run-level preimages, each named by what it fingerprints."""
    harness = (REPO / "scripts" / "dispatch_runner.py").read_bytes()
    protocol = (repo / "studies" / "revision-v1.yaml").read_bytes()
    tree = b"".join(path.read_bytes() for path in sorted((repo / "dataset" / "tasks").rglob("*.yaml")))
    return {"harness": _sha(harness), "protocol": _sha(protocol), "native_package": _sha(tree)}


def build(repo: pathlib.Path, base: pathlib.Path, design: str, judge: str,
          rehearsal: bool) -> dict:
    import dispatch_runner as dr  # noqa: PLC0415

    if judge == UNASSIGNED and not rehearsal:
        raise SystemExit("refusing a dispatchable manifest with no judge designated: "
                         "pass --judge <snapshot>, or --rehearsal to record the absence explicitly")
    study = dr._study(repo)
    families = ((study.get("task_mix") or {}).get("families") or {})
    if not families:
        raise SystemExit("study declares no task_mix.families; the mix is the price basis")
    spec, refusals = dr.declared_spec(study, design)
    if refusals:
        raise SystemExit(f"{design}: " + "; ".join(refusals))
    tasks = task_documents(repo, families)
    units = dr.assign(spec, dr.task_set(repo, set(families)))
    manifest = {
        "protocol": PROTOCOL,
        "design": design,
        "judge_designated": judge != UNASSIGNED,
        "judge_snapshot": judge,
        "prompt_version": PROMPT_VERSION,
        "run_artifacts": artifacts_of(repo),
        "assignments": rows_for(units, sorted(snapshots(base)), snapshots(base), tasks, judge),
    }
    project_report(dict(protocol=PROTOCOL, assignment_hash=digest(manifest), generations=[],
                        evaluations=[]), manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=pathlib.Path, default=REPO)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--design", default="pilot_run", choices=("pilot_run", "main_run"))
    ap.add_argument("--judge", required=True,
                    help=f"the judge model snapshot, or {UNASSIGNED} with --rehearsal")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    manifest = build(args.repo, args.base, args.design, args.judge, args.rehearsal)
    raw = _json_bytes(manifest)
    args.out.write_bytes(raw)
    print(f"  rows        : {len(manifest['assignments'])}")
    print(f"  families    : {len({r['task']['task_type'] for r in manifest['assignments']})}")
    print(f"  judge       : {'DESIGNATED' if manifest['judge_designated'] else 'UNASSIGNED (rehearsal)'}")
    print(f"  digest      : {_sha(raw)[:16]}")
    print(f"  wrote       : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
