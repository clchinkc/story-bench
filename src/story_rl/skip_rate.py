"""The skip rate the training metrics cannot report, and the step at which a run should have stopped.

`by_group/frac_mixed` reads 1.0 in almost every step of a run where most groups were being discarded. It
is computed **after** constant-reward filtering, so it describes the data that survived, not the data that
was bought. The two numbers that matter live in different places:

    sampled   `iteration_*/train_rollout_summaries.jsonl`, one row per trajectory
    trained   `env/all/total_episodes` in `metrics.jsonl`, trajectories that entered the gradient

A run whose skip rate passes 25 % for three consecutive steps is paying for rollouts that cannot teach
anything. On the one run this was built against, that rule fires at step 11 of 30, by which time the
reward had already reached 0.906.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

STOP_SKIP = 0.25
STOP_CONSECUTIVE = 3


def sampled_per_step(run: Path) -> dict[int, int]:
    """Trajectories actually generated per iteration — what was paid for."""
    counts: dict[int, int] = {}
    for path in sorted(glob.glob(str(run / "iteration_*" / "train_rollout_summaries.jsonl"))):
        step = int(Path(path).parent.name.split("_")[1])
        counts[step] = sum(1 for line in open(path) if line.strip())
    return counts


def trained_per_step(run: Path) -> dict[int, int]:
    """Trajectories that entered the gradient — `total_episodes`, post-filter."""
    counts: dict[int, int] = {}
    for line in open(run / "metrics.jsonl"):
        row = json.loads(line)
        counts[int(row["step"])] = int(row["env/all/total_episodes"])
    return counts


def build(run: Path) -> list[dict]:
    """One row per step, joining what was sampled against what was trained."""
    sampled, trained = sampled_per_step(run), trained_per_step(run)
    rows = []
    for step in sorted(set(sampled) | set(trained)):
        bought, used = sampled.get(step, 0), trained.get(step, 0)
        rows.append({
            "step": step,
            "sampled": bought,
            "trained": used,
            "skip": round(1 - used / bought, 4) if bought else None,
        })
    return rows


def first_stop_step(rows: list[dict], threshold: float = STOP_SKIP, consecutive: int = STOP_CONSECUTIVE) -> int | None:
    """The step at which the run should have stopped, or None if it never should have."""
    streak = 0
    for row in rows:
        streak = streak + 1 if (row["skip"] or 0.0) > threshold else 0
        if streak >= consecutive:
            return row["step"] - consecutive + 1
    return None


def summarise(run: Path, threshold: float = STOP_SKIP) -> dict:
    if not (run / "metrics.jsonl").exists() and not list(run.glob("iteration_*/train_rollout_summaries.jsonl")):
        raise ValueError(f"{run}: no iteration summaries and no metrics found")
    rows = build(run)
    if not rows:
        raise ValueError(f"{run}: the run directory carries no steps")
    sampled = sum(row["sampled"] for row in rows)
    trained = sum(row["trained"] for row in rows)
    return {
        "steps": len(rows),
        "sampled": sampled,
        "trained": trained,
        "skip": round(1 - trained / sampled, 4) if sampled else None,
        "threshold": threshold,
        "should_have_stopped_at": first_stop_step(rows, threshold),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="a tinker-cookbook run directory")
    parser.add_argument("--threshold", type=float, default=STOP_SKIP)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = summarise(Path(args.run).expanduser(), args.threshold)
    if args.json:
        print(json.dumps(result, indent=1))
        return 0

    print(f"{'step':>4} {'sampled':>8} {'trained':>8} {'skip':>7}")
    for row in result["rows"]:
        bar = "#" * int(round((row["skip"] or 0) * 30))
        print(f"{row['step']:>4} {row['sampled']:>8} {row['trained']:>8} {row['skip'] or 0:>7.2f} {bar}")
    print(f"\nsampled {result['sampled']}  trained {result['trained']}  skip {result['skip']:.1%}")
    stop = result["should_have_stopped_at"]
    print(
        f"should have stopped at step {stop}" if stop is not None
        else f"never exceeded {args.threshold:.0%} skip for {STOP_CONSECUTIVE} consecutive steps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
