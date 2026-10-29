#!/usr/bin/env python3
"""Validate a W6 study manifest against the experiment contract.

Offline only: reads one YAML file, contacts nothing, reads no credential.

    uv run --no-sync --offline python scripts/validate_study.py \
        --manifest studies/revision-v1.yaml --offline

Exit 0 = the manifest is a complete preregistration.
Exit 1 = it is not, with every missing or non-frozen field named.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from study_analysis import (  # noqa: E402
    ManifestError, load_manifest, power_curve, render_power_curve, validate_manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--offline", action="store_true",
                        help="required; this tool never goes online")
    parser.add_argument("--power-grid", action="store_true",
                        help="also render the preregistered power/precision curve")
    parser.add_argument("--power-trials", type=int, default=100)
    parser.add_argument("--power-resamples", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.offline:
        print("refused: --offline is required; this tool never contacts a provider")
        return 2

    try:
        manifest = load_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(f"unreadable manifest: {exc}")
        return 2

    problems = validate_manifest(manifest)
    report = {
        "manifest": args.manifest,
        "study_id": manifest.get("study_id"),
        "status": manifest.get("status"),
        "roster_status": manifest.get("roster_status"),
        "roster_size": len(manifest.get("roster") or []),
        "preregistration_complete": not problems,
        "problems": problems,
        "note": ("A complete preregistration is not a frozen study. Freezing also needs "
                 "an owner-approved roster and a sample size chosen from pilot evidence."),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"manifest : {args.manifest}")
        print(f"study_id : {report['study_id']}")
        print(f"status   : {report['status']}  roster_status: {report['roster_status']}  "
              f"roster_size: {report['roster_size']}")
        if problems:
            print(f"\npreregistration NOT complete ({len(problems)} problem(s)):")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\npreregistration complete.")

    if args.power_grid and not problems:
        grid = manifest.get("power_simulation", {}).get("grid") or [[20, 3], [30, 4]]
        points = power_curve(manifest, [tuple(pair) for pair in grid],
                             trials=args.power_trials, resamples=args.power_resamples)
        print()
        print(render_power_curve(points))

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
