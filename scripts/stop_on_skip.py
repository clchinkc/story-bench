"""Stop a run when it stops teaching: the skip-rate rule AC-4 asks for, as a watchdog.

The problem it fixes is measured, not hypothetical. The shape-A run went 30 steps, passed 50 % skip at step 16,
and the last 14 steps re-sampled groups whose reward was constant — no gradient, full price. Nothing acted on
the number the run itself was printing.

The rule: when `skip = 1 − episodes_kept / (batch × group)` exceeds the threshold for N consecutive steps, send
SIGINT to the training process, which lets the cookbook save its checkpoint and exit cleanly rather than being
killed. The reason is written to `stop_reason.json` in the run directory so the log and the rule agree.

    python scripts/stop_on_skip.py --run ~/tinker-runs/story_prose_grpo_2026-11-02 --threshold 0.6

It exits 0 whether or not it fired — "the run finished on its own" is a valid outcome, and the file says which
happened.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

SKIP_THRESHOLD = 0.60
CONSECUTIVE = 3
POLL_SECONDS = 10


def steps_seen(metrics: Path) -> list[dict]:
    """Every parsed metrics row, ignoring a half-written last line."""
    if not metrics.exists():
        return []
    rows = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def skip_of(row: dict, batch: int, group: int) -> float:
    kept = float(row.get("env/all/total_episodes", 0))
    return 1.0 - (kept / float(batch * group))


def advance(skip: float, streak: int, threshold: float) -> int:
    """The rule as a pure function: one step's skip in, the run's streak out."""
    return streak + 1 if skip > threshold else 0


def find_pid(pattern: str) -> int | None:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split() if p.strip()]
    return max(pids) if pids else None


def fire(run: Path, payload: dict, pattern: str) -> None:
    pid = find_pid(pattern)
    payload |= {"pid": pid, "acted": pid is not None}
    if pid is not None:
        os.kill(pid, signal.SIGINT)
    (run / "stop_reason.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"RULE FIRED at step {payload['step']}: skip {payload['skip']:.3f} > {payload['threshold']} for "
          f"{payload['consecutive']} steps — SIGINT {'sent' if pid else 'not sent (no pid)'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pattern", default="story_rl.py train --shape prose")
    parser.add_argument("--threshold", type=float, default=SKIP_THRESHOLD)
    parser.add_argument("--consecutive", type=int, default=CONSECUTIVE)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--poll", type=float, default=POLL_SECONDS)
    args = parser.parse_args()

    run = Path(args.run).expanduser()
    metrics = run / "metrics.jsonl"
    streak, last_step = 0, -1

    while True:
        rows = steps_seen(metrics)
        if not rows:
            time.sleep(args.poll)
            continue
        step = int(rows[-1].get("step", -1))
        if step == last_step:
            time.sleep(args.poll)
            continue

        last_step = step
        skip = skip_of(rows[-1], args.batch, args.group)
        streak = advance(skip, streak, args.threshold)
        print(f"step {step:3d}  skip {skip:.3f}  streak {streak}/{args.consecutive}", flush=True)

        if streak >= args.consecutive:
            fire(run, {"step": step, "skip": round(skip, 4), "threshold": args.threshold,
                       "consecutive": args.consecutive}, args.pattern)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
