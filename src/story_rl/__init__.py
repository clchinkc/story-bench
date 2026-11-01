"""Story continuity repair: a GRPO training harness for the repair reward.

The benchmark in `src/` measures models. This package *changes* one: it wraps the repair reward in a
Tinker environment, runs GRPO against it, and scores the result with the same instrument that scored
the baseline. It exists because a reward that has only ever been measured has not been tested — the
question "can a policy learn this?" is answered by training, not by more probing.

Two design decisions are load-bearing and are documented where they are made, not here:

    `env.RepairEnv`     zeroes `ProblemEnv`'s format term, because a formatting reward is optimisable
                        instead of the task
    `train`             pins the renderer to the thinking-disabled variant, because the recommended one
                        prefills `<think>` and would score a reasoning trace as a continuation

Requires the `train` dependency group (`tinker`, `tinker-cookbook`); the benchmark's own modules do not.
"""

from __future__ import annotations

from story_rl.corpus import TaskSet, load_split, load_tasks, split_manifest
from story_rl.spend import Rates, bound, load_rates, per_rollout

__all__ = [
    "Rates",
    "TaskSet",
    "bound",
    "load_rates",
    "load_split",
    "load_tasks",
    "per_rollout",
    "split_manifest",
]
