#!/usr/bin/env python3
"""The mix-based reservation ceiling: what the hold costs once the task families are declared.

Run from the repo root:

    uv run python scripts/mix_ceiling.py

The mix itself lives in `studies/revision-v1.yaml` under `task_mix`, not in this file, because the
published study and the published ceiling must not be able to disagree while both look right. The
dispatch gate reads the same block, and `tests/test_task_mix.py` fails if either the weights or the
token caps drift from the frozen task set or from `TASK_TOKEN_CONFIG`.

WHY THE AGENTIC FAMILIES ARE NOT PRICED HERE. They are multi-call by construction —
`agentic_iterative_revision` loops `max_revisions` and planning execution is plan plus execute — so
pricing them at one `max_tokens` would UNDERSTATE the hold, which is the one direction this arithmetic
must never err in. They are listed in the config's `excluded_multicall` and named in the output rather
than folded in at a convenient value.
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import sys
from decimal import Decimal

sys.path.insert(0, "src")

import yaml  # noqa: E402

from generator import PromptBuilder  # noqa: E402
from measurement_contract import context_packet  # noqa: E402

STUDY = pathlib.Path("studies/revision-v1.yaml")
RECORDS = pathlib.Path.home() / "Documents/GitHub/Obsidian-note/🧭-GROUND-TRUTH" \
    / "2026-09-21-story-evaluation-plan-consolidation/bindings/execution/w3b2" \
    / "route-evidence-2026-11-01/records"
UNITS_PER_MODEL = 144
ARMS = 2
PROFILE_INPUT_CEILING = 8000
RETRY_MARGIN = Decimal("1.1")


def declared_mix():
    mix = (yaml.safe_load(STUDY.read_text()) or {}).get("task_mix") or {}
    families, caps = mix.get("families") or {}, mix.get("max_tokens_by_family") or {}
    if not families or set(families) - set(caps):
        raise SystemExit(f"{STUDY} declares no usable task_mix")
    return families, caps, list(mix.get("excluded_multicall") or [])


def rates_by_model() -> dict[str, dict[str, Decimal]]:
    out = {}
    for path in sorted(RECORDS.glob("*.json")):
        record = json.loads(path.read_text())
        out[record["route_identity"]["model_id"]] = {
            c["name"]: Decimal(c["usd_per_unit"]) for c in record["price_components"]}
    return out


def measured_bodies(families) -> dict[str, int]:
    """Largest serialized body per family, measured from the frozen task set, plus JSON framing."""
    body: dict[str, int] = {}
    for path in sorted(glob.glob("dataset/tasks/**/*.yaml", recursive=True)):
        task = yaml.safe_load(pathlib.Path(path).read_text())
        if task["task_type"] not in families:
            continue
        try:
            system = PromptBuilder.build_system_prompt(task["task_type"])
            user = PromptBuilder.build_user_prompt(task)
        except ValueError:
            continue
        packet, _ = context_packet({"system": system, "entitled_input": user}, max_bytes=10 ** 7)
        body[task["task_type"]] = max(body.get(task["task_type"], 0), len(packet.encode()) + 220)
    return body


def ceiling(rates, families, weights, caps, bodies=None):
    total_weight = sum(weights[f] for f in families)
    mean_out = sum(caps[f] * weights[f] for f in families) / total_weight
    # An unmeasured family falls back to the profile ceiling, never to zero: an unmeasured family is
    # not a cheap one, and a defaultdict here once understated the hold by ten dollars.
    mean_in = (sum(bodies.get(f, PROFILE_INPUT_CEILING) * weights[f] for f in families) / total_weight
               if bodies is not None else PROFILE_INPUT_CEILING)
    hold = sum(ARMS * UNITS_PER_MODEL * (Decimal(mean_in) * rate["prompt"]
                                         + Decimal(mean_out) * (rate["completion"]
                                                                + rate.get("internal_reasoning", Decimal(0))))
               for rate in rates.values())
    return mean_out, mean_in, hold


def main() -> int:
    families, caps, excluded = declared_mix()
    rates, bodies = rates_by_model(), measured_bodies(families)
    uniform = {f: 1 for f in families}
    weighted = {f: len(glob.glob(f"dataset/tasks/{f}/*.yaml")) for f in families}
    print(f"  declared families: {len(families)}; excluded as multi-call: {excluded}")
    print(f"  measured body    : { {k: v for k, v in bodies.items() if k in families} }")
    rows = [
        ("all at the longest family (the old assumption)", uniform, None, True),
        ("uniform over the declared families", uniform, None, False),
        ("dataset-weighted (the declared mix)", weighted, None, False),
        ("dataset-weighted, measured input", weighted, bodies, False),
    ]
    print(f"\n  {'mix':44} {'mean_out':>8} {'mean_in':>8} {'ceiling':>9} {'+10% retry':>11}")
    last = Decimal(0)
    for name, weights, body_arg, longest in rows:
        out, inp, hold = ceiling(rates, families, weights, caps, body_arg)
        if longest:
            out = float(max(caps.values()))
            hold = ceiling(rates, families, {f: (1 if caps[f] == max(caps.values()) else 0) or 0
                                             for f in families}, caps, body_arg)[2]
        last = hold
        print(f"  {name:44} {out:8.0f} {inp:8.0f} {hold:9.2f} {hold * RETRY_MARGIN:11.2f}")
    print(f"\n  the gate funds the conservative row; {last * RETRY_MARGIN:.2f} buys a 10% retry margin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
