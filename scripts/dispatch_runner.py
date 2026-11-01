#!/usr/bin/env python3
"""Plan a run, and refuse to plan one the study has not declared.

Run from the repo root so the project environment resolves:

    uv run python scripts/dispatch_runner.py --plan

WHY THIS REFUSES TODAY. `studies/revision-v1.yaml` is `status: unfrozen` with `sample_size: null`, and
its own header says why: *"sample_size is null because the contract freezes it only from pilot evidence"*.
Planning 2,592 arm-runs against an undeclared design would produce a manifest that looks authoritative
and is not — the same failure this unit keeps finding, in a third guise. So the runner reads a
`main_run` block, and until that block exists and is `frozen` it refuses and names exactly what is
missing. That refusal IS the remaining work, expressed as a check rather than a note.

The plan is deterministic: the same config and the same task set produce the same attempt ids, so a
manifest can be regenerated and compared rather than trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass

DEFAULT_REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASE = (pathlib.Path.home() / "Documents/GitHub/Obsidian-note/🧭-GROUND-TRUTH"
                / "2026-09-21-story-evaluation-plan-consolidation/bindings/execution/w3b2"
                / "route-evidence-2026-11-01")
REQUIRED = ("status", "units_per_model", "arms", "units_per_cluster", "clusters_per_locale", "locales")
FROZEN = "frozen"


@dataclass(frozen=True)
class RunSpec:
    """The declared design, in the only shape this runner will accept."""
    status: str
    units_per_model: int
    arms: int
    units_per_cluster: int
    clusters_per_locale: int
    locales: tuple[str, ...]

    @property
    def units(self) -> int:
        return self.clusters_per_locale * len(self.locales) * self.units_per_cluster

    @property
    def arm_runs(self) -> int:
        return self.units * self.arms


def _study(repo: pathlib.Path) -> dict:
    import yaml  # noqa: PLC0415
    path = repo / "studies" / "revision-v1.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def declared_spec(study: dict) -> tuple[RunSpec | None, list[str]]:
    """Read `main_run`, or say precisely which of its fields is missing."""
    block = study.get("main_run") or {}
    missing = [key for key in REQUIRED if key not in block]
    if missing:
        return None, [f"main_run declares no {key}" for key in missing]
    spec = RunSpec(
        status=str(block["status"]),
        units_per_model=int(block["units_per_model"]),
        arms=int(block["arms"]),
        units_per_cluster=int(block["units_per_cluster"]),
        clusters_per_locale=int(block["clusters_per_locale"]),
        locales=tuple(block["locales"]),
    )
    refusals = []
    if spec.status != FROZEN:
        refusals.append(f"main_run is {spec.status!r}, not {FROZEN!r}")
    if spec.units != spec.units_per_model:
        refusals.append(f"declared units {spec.units} contradict units_per_model {spec.units_per_model}")
    if study.get("status") != FROZEN:
        refusals.append(f"the study itself is {study.get('status')!r}, not {FROZEN!r}")
    return spec, refusals


def task_set(repo: pathlib.Path) -> list[tuple[str, str]]:
    """(family, task_id) for every frozen task, sorted so the plan is reproducible."""
    import yaml  # noqa: PLC0415
    out = []
    for path in sorted((repo / "dataset" / "tasks").rglob("*.yaml")):
        task = yaml.safe_load(path.read_text())
        if isinstance(task, dict) and "task_id" in task and "task_type" in task:
            out.append((task["task_type"], task["task_id"]))
    return out


def assign(spec: RunSpec, tasks: list[tuple[str, str]]) -> list[dict]:
    """One row per unit. Task choice is a deterministic cycle, so a unit's identity never drifts."""
    if not tasks:
        return []
    rows = []
    for locale in spec.locales:
        for cluster in range(spec.clusters_per_locale):
            for unit in range(spec.units_per_cluster):
                index = (cluster * spec.units_per_cluster + unit) % len(tasks)
                family, task_id = tasks[index]
                rows.append(dict(locale=locale, cluster=f"{locale}-c{cluster:02d}", unit=unit,
                                 task_family=family, task_id=task_id,
                                 unit_id=f"{locale}-c{cluster:02d}-u{unit}"))
    return rows


def attempt_id(route: str, unit_id: str, arm: str) -> str:
    """Stable, human-readable, and collision-free: the digest pins the triple."""
    return f"{route.replace('/', '_')}:{unit_id}:{arm}:{hashlib.sha256(f'{route}|{unit_id}|{arm}'.encode()).hexdigest()[:8]}"


def plan(spec: RunSpec, units: list[dict], routes: list[str]) -> list[dict]:
    arms = ("A", "D")[:spec.arms]
    return [dict(attempt_id=attempt_id(route, row["unit_id"], arm), route=route, arm=arm, **row)
            for route in routes for row in units for arm in arms]


def route_names(base: pathlib.Path) -> list[str]:
    import json as _json  # noqa: PLC0415
    return sorted(_json.loads(p.read_text())["route_identity"]["model_id"]
                  for p in (base / "records").glob("*.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path, default=DEFAULT_REPO)
    ap.add_argument("--base", type=pathlib.Path, default=DEFAULT_BASE)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args(argv)

    study = _study(args.repo)
    spec, refusals = declared_spec(study)
    if refusals:
        print("  REFUSED:")
        for reason in refusals:
            print(f"    - {reason}")
        return 2
    units = assign(spec, task_set(args.repo))
    routes = route_names(args.base)
    attempts = plan(spec, units, routes)
    print(f"  units/model : {len(units)}")
    print(f"  routes      : {len(routes)}")
    print(f"  arm-runs    : {len(attempts)}")
    if args.plan:
        print(json.dumps(attempts[:2], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
