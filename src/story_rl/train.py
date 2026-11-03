"""Run GRPO against one shape's reward. The paid entry point, and the only one.

    python scripts/story_rl.py estimate                             the spend bound, no side effect
    STORY_RL_APPROVED_USD=3.00 python scripts/story_rl.py train     the paid run
    python scripts/story_rl.py train --shape shape-a                the same run on shape A

The spend gate is an environment variable rather than a flag: a flag gets typed into a shell loop by
accident, an exported figure is a decision someone made. With no figure the script refuses; below the
worst case it refuses too, and prints both numbers.

Everything the run depends on is fixed here rather than passed at the command line — the reward, the
parser policy, the renderer, the hyper-parameters, and each shape's corpus and split (which live in that
shape's own module). `--shape` selects a *recipe*, not a data knob: two shapes are two frozen runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train as cookbook_train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.utils.session_naming import default_cookbook_session_name

from story_rl.env import DEFAULT_MODEL, DEFAULT_RENDERER
from story_rl.env import TaskDatasetBuilder as RepairDatasetBuilder
from story_rl.prose import ProseDatasetBuilder
from story_rl.shape_a import ShapeADatasetBuilder
from story_rl.spend import bound, load_rates, require_approval

APPROVAL_ENV = "STORY_RL_APPROVED_USD"

DEFAULT_SHAPE = "repair"
RECIPES = {
    "repair": "story_repair_grpo",
    "shape-a": "story_shape_a_grpo",
    "prose": "story_prose_grpo",
}
SHAPE_BUILDERS: dict[str, type[RLDatasetBuilder]] = {
    "repair": RepairDatasetBuilder,
    "shape-a": ShapeADatasetBuilder,
    "prose": ProseDatasetBuilder,
}

DEFAULT_STEPS = 30
DEFAULT_BATCH = 8
DEFAULT_GROUP = 16
DEFAULT_LEARNING_RATE = 4e-5
DEFAULT_MAX_TOKENS = 200


def builder_for(shape: str, steps: int, batch: int, group: int) -> RLDatasetBuilder:
    """The training half for one shape. A shape is a builder, a corpus and a name — nothing more.

    Each shape's corpus and split are fixed inside its own module; this function selects one, it does
    not parameterise the data. Adding a shape is therefore an edit to `SHAPE_BUILDERS`, which is the
    doctrine the original comment stated, kept while the second shape arrived.
    """
    if shape not in SHAPE_BUILDERS:
        raise SystemExit(f"unknown shape {shape!r}; known: {sorted(SHAPE_BUILDERS)}")
    return SHAPE_BUILDERS[shape](
        batch_size=batch,
        group_size=group,
        model_name_for_tokenizer=DEFAULT_MODEL,
        renderer_name=DEFAULT_RENDERER,
        n_batches=steps,
    )


def build_config(
    steps: int = DEFAULT_STEPS,
    batch: int = DEFAULT_BATCH,
    group: int = DEFAULT_GROUP,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    log_path: str | None = None,
    save_every: int = 5,
    shape: str = DEFAULT_SHAPE,
) -> cookbook_train.Config:
    """The whole run definition, in one place, with no default inherited by accident.

    Group width 16 rather than 8: at G=8 the contrast falls under 60 % once the interval accounts for
    clustering, and halving the step count pays for the wider group, so the two cost the same.
    """
    builder = builder_for(shape, steps, batch, group)
    recipe = RECIPES[shape]
    return cookbook_train.Config(
        model_name=DEFAULT_MODEL,
        recipe_name=recipe,
        renderer_name=DEFAULT_RENDERER,
        dataset_builder=builder,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        log_path=log_path or cli_utils.runs_path(recipe),
        eval_every=0,
        # Every 5 steps, not 20: a run that dies at step 12 has produced nothing otherwise. A checkpoint costs
        # seconds and the alternative is losing the whole run to one transient judge error.
        save_every=save_every,
        lora_rank=32,
        temperature=1.0,
        loss_fn="importance_sampling",
        # Uniform groups carry no gradient; skipping them saves the forward/backward pass on data that
        # cannot teach anything. The rollouts are paid for either way, so this saves train tokens only —
        # and it is why `skip_rate` must be read beside the reward.
        remove_constant_reward_groups=True,
    )


def spend_report(config: cookbook_train.Config) -> dict:
    builder = config.dataset_builder
    return bound(
        builder.n_batches,
        builder.batch_size,
        builder.group_size,
        load_rates(),
        max_tokens=config.max_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true", help="print the spend bound and exit")
    parser.add_argument("--shape", default=DEFAULT_SHAPE, choices=sorted(SHAPE_BUILDERS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--group", type=int, default=DEFAULT_GROUP)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args(argv)

    config = build_config(
        steps=args.steps,
        batch=args.batch,
        group=args.group,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        shape=args.shape,
    )
    report = spend_report(config)
    print(f"{args.shape}: {report['steps']} steps x {report['batch']} groups x "
          f"{report['group']} rollouts = {report['rollouts']} rollouts")
    print(f"worst case ${report['worst_case_usd']:.2f}   expected ${report['expected_usd']:.2f}")

    if args.estimate:
        import json

        print(json.dumps(report, indent=1))
        return 0

    approved = os.environ.get(APPROVAL_ENV)
    require_approval(report, float(approved) if approved else None)

    if not config.session_name:
        config = chz.replace(
            config,
            session_name=default_cookbook_session_name(
                f"story-{args.shape}", wandb_name=config.wandb_name
            ),
        )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(cookbook_train.main(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
