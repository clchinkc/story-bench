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


def declared_spec(study: dict, design: str = "main_run") -> tuple[RunSpec | None, list[str]]:
    """Read the named design block, or say precisely which of its fields is missing."""
    block = study.get(design) or {}
    missing = [key for key in REQUIRED if key not in block]
    if missing:
        return None, [f"{design} declares no {key}" for key in missing]
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
        refusals.append(f"{design} is {spec.status!r}, not {FROZEN!r}")
    if spec.units != spec.units_per_model:
        refusals.append(f"declared units {spec.units} contradict units_per_model {spec.units_per_model}")
    if study.get("status") != FROZEN:
        refusals.append(f"the study itself is {study.get('status')!r}, not {FROZEN!r}")
    return spec, refusals


def _declared_families(repo: pathlib.Path) -> set[str] | None:
    """The families the study priced against, or None when it declares no mix (the test repos)."""
    declared = ((_study(repo).get("task_mix") or {}).get("families") or {})
    return set(declared) or None


def _task_row(path: pathlib.Path) -> tuple[str, str] | None:
    import yaml  # noqa: PLC0415
    task = yaml.safe_load(path.read_text())
    if isinstance(task, dict) and "task_id" in task and "task_type" in task:
        return (task["task_type"], task["task_id"])
    return None


def task_set(repo: pathlib.Path, families: set[str] | None = None) -> list[tuple[str, str]]:
    """(family, task_id) for every frozen task in the DECLARED mix, sorted for reproducibility.

    The mix filter is not a convenience. `task_mix.families` is what the reservation ceiling was
    priced against, and cycling over every yaml under `dataset/tasks` silently planned units from
    families the study EXCLUDES: the four agentic families sort first, so an 18-unit pilot would
    have been built entirely from them -- costs and claims for a mix the study never declared. When
    no mix is declared (the synthetic repos the tests build), every task is admissible.
    """
    if families is None:
        families = _declared_families(repo)
    rows = [_task_row(path) for path in sorted((repo / "dataset" / "tasks").rglob("*.yaml"))]
    return [row for row in rows if row and (families is None or row[0] in families)]


def assign(spec: RunSpec, tasks: list[tuple[str, str]]) -> list[dict]:
    """One row per unit. Task choice is a deterministic cycle, so a unit's identity never drifts."""
    if not tasks:
        return []
    rows, index = [], 0
    for locale in spec.locales:
        for cluster in range(spec.clusters_per_locale):
            for unit in range(spec.units_per_cluster):
                # ONE running counter, not a per-cluster offset. The old formula restarted the
                # cycle inside every cluster, so an 18-unit pilot drew only indices 0..5 and landed
                # in 2 of the 6 declared families -- while the study's own note promises "each of
                # the 25 frozen tasks used about 5.76 times per model per arm". The counter is what
                # makes that promise true at any unit count.
                family, task_id = tasks[index % len(tasks)]
                index += 1
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


def dispatch(attempts, *, authorise, build, execute, completed=()) -> dict:
    """Walk the plan. Refuse without a token, skip what is already settled, stop on the first failure.

    `build` and `execute` are injected rather than imported: the control flow is what this function
    owns, and keeping the provider modules out of it means the loop can be exercised without a
    transport, a ledger or a base — which is the only way its refusal paths get tested at all.
    """
    if not authorise or not str(authorise).startswith("PAID-PILOT-"):
        return dict(refused="no PAID-PILOT token", executed=0, skipped=0)
    done = set(completed)
    executed = skipped = 0
    for attempt in attempts:
        if attempt["attempt_id"] in done:
            skipped += 1
            continue
        execute(build(attempt))
        executed += 1
    return dict(refused=None, executed=executed, skipped=skipped)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=pathlib.Path, default=DEFAULT_REPO)
    ap.add_argument("--base", type=pathlib.Path, default=DEFAULT_BASE)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--design", default="main_run", choices=("main_run", "pilot_run"))
    ap.add_argument("--authorise", default=None)
    ap.add_argument("--dispatch", action="store_true")
    args = ap.parse_args(argv)

    study = _study(args.repo)
    spec, refusals = declared_spec(study, args.design)
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
    if args.dispatch:
        # The loop is ready; the two adapters it takes are not. Refusing here rather than reaching
        # for an import that does not exist yet is the same rule the gate follows everywhere else.
        print("  REFUSED: the descriptor-to-attempt adapter is not wired; "
              "`provider_attempts.prepare_attempt` is the intended build step and "
              "`provider_attempts.execute_attempt` the execute step")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
