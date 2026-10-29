#!/usr/bin/env python3
"""Run the preregistered W6 paired cluster analysis over recorded episode units.

Offline only: reads the manifest and a results file, contacts nothing, reads no
credential, and never writes into the results directory.

    uv run --no-sync --offline python scripts/analyze_study.py \
        --manifest studies/revision-v1.yaml \
        --results results/revision-v1/units.jsonl --offline

Exit 0 = analysis produced. Exit 1 = refused (incomplete manifest or no records).
The printed benefit_status is a verdict about this frozen dataset only; a
"not_met" or "inconclusive*" status still completes the study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_analysis import (  # noqa: E402
    ManifestError, analyse, assert_manifest, load_manifest, load_records,
    power_curve, render_power_curve)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True,
                        help="JSONL/JSON file of matched four-arm units")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resamples", type=int, default=None,
                        help="override only for smoke tests; the manifest value governs")
    parser.add_argument("--out", default=None, help="optional report path")
    parser.add_argument("--power-grid", action="store_true")
    parser.add_argument("--power-trials", type=int, default=100)
    parser.add_argument("--power-resamples", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.offline:
        print("refused: --offline is required; this tool never contacts a provider")
        return 2

    try:
        manifest = assert_manifest(load_manifest(args.manifest))
    except (ManifestError, OSError) as exc:
        print(f"refused: manifest is not a complete preregistration: {exc}")
        return 1
    try:
        units = load_records(args.results)
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"refused: {exc}")
        return 1

    try:
        report = analyse(units, manifest, resamples=args.resamples)
    except ManifestError as exc:
        print(f"refused: {exc}")
        return 1

    if manifest.get("kind") == "SYNTHETIC EXAMPLE":
        report["WARNING"] = ("SYNTHETIC EXAMPLE — these numbers are simulated "
                            "mechanics checks and are not study results")

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    if args.power_grid:
        grid = manifest.get("power_simulation", {}).get("grid") or [[20, 3], [30, 4]]
        points = power_curve(manifest, [tuple(pair) for pair in grid],
                             trials=args.power_trials, resamples=args.power_resamples)
        print()
        print(render_power_curve(points))
    return 0


if __name__ == "__main__":
    sys.exit(main())
