"""The Tinker environment: one repair task per episode, GRPO over the repair reward.

Tinker's cookbook already has the right shape — `ProblemEnv` for a single-turn scored episode,
`ProblemGroupBuilder` for the GRPO group, `RLDataset` for the batch source — so this is a mapping from
the benchmark's reward onto it, not a new training loop. The less of the loop that is ours, the less of
it can be wrong.

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

from repair_task import RepairTask, RolloutScore, score_rollout
from story_rl.corpus import SPLIT_SALT, TaskSet, load_split, load_tasks

DEFAULT_POLICY = "prose"
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


def permuted(tasks: Sequence[RepairTask], epoch: int, salt: str = SPLIT_SALT) -> list[RepairTask]:
    """A deterministic permutation per epoch — new order each pass, identical order on replay."""
    def key(task: RepairTask) -> str:
        return hashlib.sha256(f"{salt}:{epoch}:{task.task_id}".encode()).hexdigest()

    return sorted(tasks, key=key)


class RepairEnv(ProblemEnv):
    """One repair task. The reward is the repair verdict, and nothing else."""

    def __init__(self, task: RepairTask, renderer: renderers.Renderer, policy: str = DEFAULT_POLICY):
        super().__init__(
            renderer,
            convo_prefix=None,
            format_coef=0.0,
            require_stop_sequence_for_format=False,
        )
        self.task = task
        self.policy = policy
        # `ProblemEnv.step` scores through `check_answer` and this class reports the diagnostic
        # projection in the same step. The reward is a pure function of the text, so caching by text
        # means one parse per rollout instead of three.
        self._scored: tuple[str, RolloutScore] | None = None

    def get_question(self) -> str:
        return self.task.prompt()

    def score(self, sample_str: str) -> RolloutScore:
        if self._scored is not None and self._scored[0] == sample_str:
            return self._scored[1]
        result = score_rollout(sample_str, self.task, self.policy)
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


class RepairDataset(RLDataset):
    """Batches of training passages. One batch is `batch_size` groups of `group_size` rollouts."""

    def __init__(
        self,
        tasks: Sequence[RepairTask],
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        n_batches: int = 200,
        policy: str = DEFAULT_POLICY,
        salt: str = SPLIT_SALT,
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

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """`batch_size` distinct passages, reshuffled every pass through the training set."""
        epoch, offset = divmod(index * self.batch_size, len(self.tasks))
        order = permuted(self.tasks, epoch, self.salt)
        chosen = [order[(offset + step) % len(order)] for step in range(self.batch_size)]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(RepairEnv, task=task, renderer=self.renderer, policy=self.policy),
                num_envs=self.group_size,
                dataset_name="repair",
            )
            for task in chosen
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class RepairDatasetBuilder(RLDatasetBuilder):
    """What `tinker_cookbook.rl.train.Config` takes. The training half only, never the held-out one."""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str = DEFAULT_RENDERER
    n_batches: int = 200
    policy: str = DEFAULT_POLICY
    tasks_path: str = ""
    split_path: str = ""

    async def __call__(self) -> tuple[RepairDataset, None]:
        from story_rl.corpus import SPLIT, TASKS

        tasks = load_tasks(Path(self.tasks_path) if self.tasks_path else TASKS)
        corpus: TaskSet = load_split(tasks, Path(self.split_path) if self.split_path else SPLIT)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            RepairDataset(
                tasks=corpus.train,
                renderer=renderer,
                batch_size=self.batch_size,
                group_size=self.group_size,
                n_batches=self.n_batches,
                policy=self.policy,
            ),
            None,
        )
