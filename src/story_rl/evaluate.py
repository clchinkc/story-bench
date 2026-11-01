"""Score generations against the repair reward, at the level of the task rather than the rollout.

Four samples of one passage are not four independent observations: they share a passage, a truth and an
error. A rollout-level interval is therefore too narrow. This reports both, labels which is which, and
reports the per-task rates because with eight passages an interval is arithmetic without inference.

The input is a plain JSONL of `{"task_id", "raw"}` rather than a live client, so the same instrument
scores a local probe, a service sample and a hand-written file — and can be tested before any of them
exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repair_task import RepairTask, score_rollout
from story_rl.corpus import load_split
from story_rl.stats import cluster_bootstrap, sign_test, wilson

DEFAULT_POLICY = "prose"


def read_generations(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_task_rewards(rows: list[dict], tasks: dict[str, RepairTask], policy: str) -> dict[str, list[float]]:
    """task_id -> the reward of each of its generations."""
    by_task: dict[str, list[float]] = {}
    for row in rows:
        task = tasks.get(row["task_id"])
        if task is None:
            raise ValueError(f"generation names an unknown task: {row['task_id']!r}")
        by_task.setdefault(row["task_id"], []).append(float(score_rollout(row["raw"], task, policy).strict))
    return by_task


def summarise(by_task: dict[str, list[float]], policy: str = DEFAULT_POLICY) -> dict:
    rewards = [value for values in by_task.values() for value in values]
    passes = int(sum(rewards))
    low, high = wilson(len(rewards) - passes, len(rewards))
    cluster_low, cluster_high = cluster_bootstrap(by_task)
    return {
        "policy": policy,
        "tasks": len(by_task),
        "generations": len(rewards),
        "pass_rate": round(passes / len(rewards), 4) if rewards else None,
        "pass_rate_ci": [round(1 - high, 4), round(1 - low, 4)],
        "pass_rate_cluster_ci": [cluster_low, cluster_high],
        "per_task": {
            task_id: round(sum(values) / len(values), 3) for task_id, values in sorted(by_task.items())
        },
    }


def compare(baseline: dict[str, list[float]], treatment: dict[str, list[float]]) -> dict:
    """Paired per-task comparison. Ties are excluded, because a passage already solved cannot improve."""
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        raise ValueError("the two arms share no tasks; a paired comparison is impossible")
    improved = sum(
        1 for task_id in shared
        if sum(treatment[task_id]) / len(treatment[task_id]) > sum(baseline[task_id]) / len(baseline[task_id])
    )
    worsened = sum(
        1 for task_id in shared
        if sum(treatment[task_id]) / len(treatment[task_id]) < sum(baseline[task_id]) / len(baseline[task_id])
    )
    return {
        "tasks": len(shared),
        "improved": improved,
        "unchanged": len(shared) - improved - worsened,
        "worsened": worsened,
        "p": sign_test(improved, worsened),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--against", default=None, help="a baseline JSONL for a paired comparison")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    corpus = load_split()
    tasks = corpus.by_id()
    by_task = per_task_rewards(read_generations(Path(args.generations)), tasks, args.policy)
    result: dict = {"treatment": summarise(by_task, args.policy)}

    if args.against:
        baseline = per_task_rewards(read_generations(Path(args.against)), tasks, args.policy)
        result["baseline"] = summarise(baseline, args.policy)
        result["paired"] = compare(baseline, by_task)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    treatment = result["treatment"]
    print(f"{treatment['tasks']} tasks, {treatment['generations']} generations, "
          f"pass/task {treatment['pass_rate']:.3f} {treatment['pass_rate_ci']} "
          f"cluster {treatment['pass_rate_cluster_ci']}")
    if "paired" in result:
        paired = result["paired"]
        print(f"paired: {paired['improved']} improved, {paired['unchanged']} unchanged, "
              f"{paired['worsened']} worse, p = {paired['p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
