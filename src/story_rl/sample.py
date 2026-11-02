"""Sample the base model and a trained checkpoint from the **same** stack, for a fair verdict.

The baseline is produced through the same client as the trained arm, so a comparison isolates the
weights and not a serving change. One client, one temperature, one prompt set, one number of samples.

Two things here are load-bearing and were each wrong once:

    renderer        pinned to the thinking-disabled variant. The recommended one prefills `<think>`,
                    which is not the configuration any number was measured in.
    scored text     `get_text_content(parse_response(tokens))`, never `tokenizer.decode(tokens)`.
                    Raw decoding keeps the reasoning block, and a trace can carry the value the reward
                    looks for with no prose written.

Shape A adds a third, and it is why `--shape` exists at all: the prompt of that shape does **not**
print the day it asks for, so a sampled pass cannot be a copy — but only if the scored text is the
message content. The shape selects the corpus; the split rule and the reward stay the same.

`--plan` prints what would be sent and what it would cost without constructing a client.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from scorable import ScorableTask
from story_rl.env import DEFAULT_MODEL, DEFAULT_RENDERER
from story_rl.spend import load_rates, per_rollout

SHAPES = ("repair", "shape-a")


def load_passages(half: str = "held_out", shape: str = "repair") -> tuple[ScorableTask, ...]:
    """The passages of one half for one shape. The shape picks the corpus, never the split rule."""
    if shape not in SHAPES:
        raise SystemExit(f"unknown shape {shape!r}; known: {SHAPES}")
    if shape == "repair":
        from story_rl.corpus import load_split
    else:
        from story_rl.shape_a import load_split

    corpus = load_split()
    return tuple(corpus.held_out if half == "held_out" else corpus.train)


def model_inputs(tasks, renderer):
    """One (task, model_input) pair per task — the prompt is identical across its samples."""
    prepared = []
    for task in tasks:
        prepared.append((task, renderer.build_generation_prompt(
            [{"role": "user", "content": task.prompt()}]
        )))
    return prepared


def plan(
    half: str, samples: int, max_tokens: int, temperature: float, renderer, shape: str = "repair"
) -> dict:
    prepared = model_inputs(load_passages(half, shape), renderer)
    requests = len(prepared) * samples
    return {
        "model": DEFAULT_MODEL,
        "renderer": DEFAULT_RENDERER,
        "shape": shape,
        "half": half,
        "tasks": len(prepared),
        "samples_per_task": samples,
        "requests": requests,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "worst_case_usd": round(per_rollout(max_tokens, load_rates()) * requests, 4),
    }


def scored_text(renderer, tokens) -> str:
    """The continuation the reward is allowed to see — message content, never raw tokens."""
    message, _termination = renderer.parse_response(tokens)
    return renderers.get_text_content(message)


def sample(
    *,
    checkpoint: str | None,
    half: str = "held_out",
    samples: int = 8,
    max_tokens: int = 200,
    temperature: float = 1.0,
    shape: str = "repair",
) -> list[dict]:
    """Sample one arm. `checkpoint=None` means the base model."""
    import tinker

    tokenizer = get_tokenizer(DEFAULT_MODEL)
    renderer = renderers.get_renderer(DEFAULT_RENDERER, tokenizer=tokenizer)
    service = tinker.ServiceClient()
    client = (
        service.create_sampling_client(model_path=checkpoint)
        if checkpoint
        else service.create_sampling_client(base_model=DEFAULT_MODEL)
    )
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95)

    rows = []
    for task, model_input in model_inputs(load_passages(half, shape), renderer):
        response = client.sample(prompt=model_input, num_samples=samples, sampling_params=params).result()
        rows.extend(
            {
                "task_id": task.task_id,
                "raw": scored_text(renderer, sequence.tokens),
                "stop_reason": sequence.stop_reason,
            }
            for sequence in response.sequences
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print the plan and exit, no client")
    parser.add_argument("--checkpoint", default=None, help="omit for the base model")
    parser.add_argument("--shape", default="repair", choices=SHAPES)
    parser.add_argument("--half", default="held_out", choices=("train", "held_out"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.plan:
        tokenizer = get_tokenizer(DEFAULT_MODEL)
        renderer = renderers.get_renderer(DEFAULT_RENDERER, tokenizer=tokenizer)
        print(json.dumps(
            plan(args.half, args.samples, args.max_tokens, args.temperature, renderer, args.shape),
            indent=1,
        ))
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --plan is given")

    rows = sample(
        checkpoint=args.checkpoint,
        half=args.half,
        samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        shape=args.shape,
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} generations -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
