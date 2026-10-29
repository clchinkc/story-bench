#!/usr/bin/env python3
"""Generate the SYNTHETIC example results used to smoke-test the W6 toolchain.

This produces simulated units only. They are mechanics evidence that
validate_study.py / analyze_study.py run end to end offline. They are NOT study
results, and the example manifest is marked SYNTHETIC EXAMPLE so that
analyze_study.py stamps that warning into every report it prints from them.

    uv run --no-sync --offline python studies/examples/generate_synthetic_units.py

Writes studies/examples/synthetic-v1-units.jsonl (deterministic for seed 20261007).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from study_analysis import ARMS, simulate_units  # noqa: E402

SEED = 20261007
OUTPUT = Path(__file__).resolve().parent / "synthetic-v1-units.jsonl"


def main():
    units = simulate_units(
        baseline_rate=0.55, true_lift=0.12, cluster_sd=0.12, unassessable_rate=0.06,
        clusters_per_locale=14, units_per_cluster=3, seed=SEED,
        quality_win_rate=0.62, quality_tie_rate=0.20,
        cost_per_unit="0.0042", latency_ms=850, label="synthetic/example-model")
    lines = []
    for unit in units:
        lines.append(json.dumps({
            "cluster_id": unit.cluster_id, "model": unit.model, "locale": unit.locale,
            "task_family": unit.task_family, "sample_id": unit.sample_id,
            "arms": {arm: {
                "resolved": unit.observation(arm).resolved,
                "invalid_kind": unit.observation(arm).invalid_kind,
                "quality": unit.observation(arm).quality,
                "authority_breach": unit.observation(arm).authority_breach,
                "cost_usd": str(unit.observation(arm).cost_usd),
                "latency_ms": unit.observation(arm).latency_ms,
            } for arm in ARMS},
        }, sort_keys=True))
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} synthetic units to {OUTPUT}")
    assert set(ARMS) == set(json.loads(lines[0])["arms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
