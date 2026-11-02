"""The Tinker environment: one scoreable task per episode, GRPO over the task's own reward.

Tinker's cookbook already has the right shape — `ProblemEnv` for a single-turn scored episode,
`ProblemGroupBuilder` for the GRPO group, `RLDataset` for the batch source — so this is a mapping from
the benchmark's reward onto it, not a new training loop. The less of the loop that is ours, the less of
it can be wrong.

Nothing here knows what a task *is*. It asks for a prompt, a canon-plus-break to score against, and a
reference answer — `scorable.ScorableTask`. That is what makes a shape a class plus a corpus file
rather than an edit to the env: `story_rl.shape_a` and `story_rl.corpus` are two instances of the same
seam, and neither can grow its own reward because both are scored by `repair_task.score_rollout`.

Two settings are deliberate and neither is a default:

    format_coef = 0.0          `ProblemEnv` otherwise adds a formatting term to the reward, which is
                               the channel the prompt study showed can be optimised instead of the task
    policy = "prose"           a stack without constrained decoding emits no JSON, so a JSON-shaped
                               parser sees nothing; measured 0 of 96 and 0 of 80 free generations

The metrics exist so a stop condition can be *read from the run* rather than inferred afterwards:
`correct` (the reward), `format` (not truncated), `correction` (diagnostic projection), `parsed` and
`route` (the parser's verdict, as an integer because metrics carry floats).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from collections.abc import Callable

from repair_task import RolloutScore, score_rollout
from scorable import ScorableTask
from story_rl.corpus import SPLIT_SALT, TaskSet, load_split, load_tasks

DEFAULT_POLICY = "prose"

# A scorer is anything shaped like `score_rollout`: text and task in, a scored result out. Prose supplies its
# own (lint veto + frozen assessor); repair supplies the canonical one. The env does not care which.
ScoreFn = Callable[[str, ScorableTask, str], RolloutScore]
DEFAULT_RENDERER = "qwen3_5_disable_thinking"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# The parser's route as a metric value: 0 strict shape, 1 object, 2 bare string, 3 raw prose,
# -1 unparsed. Negative means the parser saw nothing, which is a rendering failure, not a model failure.
ROUTE_CODES = {
    "prose/strict-shape": 0,
    "prose/lenient/strict-shape": 0,
    "prose/object": 1,
    "prose/lenient/object": 1,
    "prose/bare-string": 2,
    "prose/lenient/bare-string": 2,
    "prose/raw": 3,
}
UNPARSED_ROUTE = -1


def route_code(route: str) -> int:
    if route in ROUTE_CODES:
        return ROUTE_CODES[route]
    return UNPARSED_ROUTE


def permuted(tasks: Sequence[ScorableTask], epoch: int, salt: str = SPLIT_SALT) -> list[ScorableTask]:
    """A deterministic permutation per epoch — new order each pass, identical order on replay."""
    def key(task: ScorableTask) -> str:
        return hashlib.sha256(f"{salt}:{epoch}:{task.task_id}".encode()).hexdigest()

    return sorted(tasks, key=key)


class TaskEnv(ProblemEnv):
    """One task. The reward is the task's own verdict, and nothing else."""

    def __init__(
        self,
        task: ScorableTask,
        renderer: renderers.Renderer,
        policy: str = DEFAULT_POLICY,
        score_fn: ScoreFn = score_rollout,
    ):
        super().__init__(
            renderer,
            convo_prefix=None,
            format_coef=0.0,
            require_stop_sequence_for_format=False,
        )
        self.task = task
        self.policy = policy
        self.score_fn = score_fn
        # `ProblemEnv.step` scores through `check_answer` and this class reports the diagnostic
        # projection in the same step. The reward is a pure function of the text, so caching by text
        # means one parse per rollout instead of three.
        self._scored: tuple[str, RolloutScore] | None = None

    def get_question(self) -> str:
        return self.task.prompt()

    def score(self, sample_str: str) -> RolloutScore:
        if self._scored is not None and self._scored[0] == sample_str:
            return self._scored[1]
        result = self.score_fn(sample_str, self.task, self.policy)
        self._scored = (sample_str, result)
        return result

    def check_answer(self, sample_str: str) -> bool:
        return bool(self.score(sample_str).strict)

    def check_format(self, sample_str: str) -> bool:
        """Always true: `format_coef` is zero, so this cannot pay anything, and leaving it true keeps
        truncation the only formatting quantity in the log."""
        return True

    def get_reference_answer(self) -> str:
        return self.task.true_value

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """The parent scores the episode; this adds the parser and projection metrics to it."""
        result = await super().step(action, extra=extra)
        message, _termination = self.renderer.parse_response(action)
        score = self.score(renderers.get_text_content(message))
        metrics = dict(result.metrics)
        metrics["correction"] = float(score.correction)
        metrics["parsed"] = float(score.parsed)
        metrics["route"] = float(route_code(score.how))
        return replace(result, metrics=metrics)


class TaskDataset(RLDataset):
    """Batches of training passages. One batch is `batch_size` groups of `group_size` rollouts.

    `dataset_name` is the shape's name in the harness log. It defaults to `repair` so every run
    recorded before shapes existed keeps its own label rather than being retro-named.
    """

    def __init__(
        self,
        tasks: Sequence[ScorableTask],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        n_batches: int = 200,
        policy: str = DEFAULT_POLICY,
        salt: str = SPLIT_SALT,
        dataset_name: str = "repair",
        score_fn: ScoreFn = score_rollout,
    ):
        if not tasks:
            raise ValueError("an empty task set cannot source a batch")
        if batch_size > len(tasks):
            raise ValueError(
                f"batch_size {batch_size} exceeds the {len(tasks)} training passages; GRPO would "
                "replay a passage inside one batch and centre its reward twice"
            )
        self.tasks = list(tasks)
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.n_batches = n_batches
        self.policy = policy
        self.salt = salt
        self.dataset_name = dataset_name
        self.score_fn = score_fn

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """`batch_size` distinct passages, reshuffled every pass through the training set."""
        epoch, offset = divmod(index * self.batch_size, len(self.tasks))
        order = permuted(self.tasks, epoch, self.salt)
        chosen = [order[(offset + step) % len(order)] for step in range(self.batch_size)]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    TaskEnv, task=task, renderer=self.renderer, policy=self.policy,
                    score_fn=self.score_fn,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for task in chosen
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class TaskDatasetBuilder(RLDatasetBuilder):
    """What `tinker_cookbook.rl.train.Config` takes, for the repair corpus.

    The training half only, never the held-out one. A second shape subclasses or mirrors this with its
    own corpus loader; the class is chz-serialisable, so the run definition stays reconstructible.
    """

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str = DEFAULT_RENDERER
    n_batches: int = 200
    policy: str = DEFAULT_POLICY
    tasks_path: str = ""
    split_path: str = ""
    dataset_name: str = "repair"

    async def __call__(self) -> tuple[TaskDataset, None]:
        from story_rl.corpus import SPLIT, TASKS

        tasks = load_tasks(Path(self.tasks_path) if self.tasks_path else TASKS)
        corpus: TaskSet = load_split(tasks, Path(self.split_path) if self.split_path else SPLIT)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            TaskDataset(
                tasks=corpus.train,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                n_batches=self.n_batches,
                policy=self.policy,
                dataset_name=self.dataset_name,
            ),
            None,
        )


# The names the repair recipe shipped under. Kept as aliases rather than renamed everywhere in one
# commit, because a run's log path and its tests both spell them, and a rename is not an improvement
# to what the run measures.
RepairEnv = TaskEnv
RepairDataset = TaskDataset
RepairDatasetBuilder = TaskDatasetBuilder
