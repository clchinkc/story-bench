"""The training path, exercised offline: prompt → rollout → reward → advantage.

This is the test that makes a paid run a formality rather than an experiment. Every step runs against the
real cookbook code — `ProblemEnv`, `ProblemGroupBuilder`, `do_group_rollout`, `compute_advantages` — with
the only fake being the `TokenCompleter` that would otherwise call a GPU. No API key, no network, no spend.

It does not establish that the model can pass the task, that a group will be mixed at the service's
temperature, or that the loss decreases. Those need the run.
"""

import asyncio
import os

# The repo forbids network in tests, and the tokenizer is cached locally — so pin the hub to offline
# mode *before* the tokenizer libraries are imported. This reads the cache, not the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

from tinker_cookbook import renderers
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

from repair_task import score_rollout
from story_rl.corpus import load_split
from story_rl.env import DEFAULT_MODEL, DEFAULT_RENDERER, RepairDataset, RepairEnv

GROUP = 4


class ScriptedPolicy:
    """A `TokenCompleter` that answers with a fixed script instead of sampling."""

    def __init__(self, texts: list[str], tokenizer):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.calls = 0

    async def __call__(self, model_input, stop, *, max_tokens=None) -> TokensWithLogprobs:
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        tokens = self.tokenizer.encode(text)
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens), stop_reason="stop")


@pytest.fixture(scope="module")
def renderer():
    return renderers.get_renderer(DEFAULT_RENDERER, tokenizer=get_tokenizer(DEFAULT_MODEL))


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(DEFAULT_MODEL)


def builder_for(task, renderer, group: int = GROUP):
    return RepairDataset([task], renderer, batch_size=1, group_size=group).get_batch(0)[0]


def passing(task) -> str:
    return f"She thought of it once more, and the {task.broken_attribute} was {task.true_value}."


def failing(task) -> str:
    return f"She thought of it once more, and the {task.broken_attribute} was {task.wrong_value}."


def test_the_prompt_reaches_the_model_intact_and_without_an_open_thinking_block(renderer, tokenizer):
    task = load_split().train[0]
    env = RepairEnv(task=task, renderer=renderer)
    observation, stop = asyncio.run(env.initial_observation())
    decoded = tokenizer.decode(observation.to_ints())

    assert task.passage in decoded
    assert task.true_value in decoded and task.wrong_value in decoded
    assert bool(stop)
    # The disabled renderer prefills `<think></think>`, closed; the recommended one leaves it open, which
    # is the defect that would make the reward score a reasoning trace instead of prose.
    assert "<think>" not in decoded or decoded.rfind("</think>") > decoded.rfind("<think>")


def test_the_environment_reward_is_the_reward_the_benchmark_scores(renderer, tokenizer):
    task = load_split().train[0]
    texts = [passing(task), failing(task)]
    group = asyncio.run(do_group_rollout(builder_for(task, renderer, 2), ScriptedPolicy(texts, tokenizer)))

    assert group.get_total_rewards() == [
        float(score_rollout(text, task, "prose").strict) for text in texts
    ]


def test_a_mixed_group_produces_a_centred_non_zero_advantage(renderer, tokenizer):
    task = load_split().train[0]
    texts = [passing(task), failing(task)]
    group = asyncio.run(do_group_rollout(builder_for(task, renderer, 2), ScriptedPolicy(texts, tokenizer)))
    advantages = compute_advantages([group])[0]

    assert float(advantages.abs().sum()) > 0.0
    assert abs(float(advantages.mean())) < 1e-6


def test_a_uniform_group_carries_no_gradient(renderer, tokenizer):
    task = load_split().train[0]
    group = asyncio.run(do_group_rollout(builder_for(task, renderer), ScriptedPolicy([passing(task)], tokenizer)))
    rewards = group.get_total_rewards()

    assert len(set(rewards)) == 1
    assert float(compute_advantages([group])[0].abs().sum()) == 0.0


def test_the_dataset_draws_distinct_passages_and_replays_deterministically(renderer):
    corpus = load_split()
    dataset = RepairDataset(corpus.train, renderer, batch_size=8, group_size=GROUP, n_batches=2)
    first = [b.env_thunk.keywords["task"].task_id for b in dataset.get_batch(0)]
    second = [b.env_thunk.keywords["task"].task_id for b in dataset.get_batch(1)]
    replay = [b.env_thunk.keywords["task"].task_id for b in
              RepairDataset(corpus.train, renderer, batch_size=8, group_size=GROUP).get_batch(0)]

    assert len(set(first)) == len(first) == 8
    assert first != second
    assert replay == first
    assert len(dataset) == 2


def test_a_batch_wider_than_the_training_set_is_refused(renderer):
    corpus = load_split()

    with pytest.raises(ValueError, match="exceeds"):
        RepairDataset(corpus.train, renderer, batch_size=len(corpus.train) + 1, group_size=GROUP)


def test_the_training_set_is_the_half_the_held_out_passages_are_not_in(renderer):
    corpus = load_split()
    dataset = RepairDataset(corpus.train, renderer, batch_size=4, group_size=GROUP)
    drawn = {b.env_thunk.keywords["task"].task_id for b in dataset.get_batch(0)}

    assert drawn <= {task.task_id for task in corpus.train}
    assert not drawn & {task.task_id for task in corpus.held_out}
