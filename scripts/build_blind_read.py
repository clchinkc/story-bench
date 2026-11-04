"""Build the blind-read sheet: twenty unlabelled continuations, half from each arm, shuffled.

The owner marks each `keep` or `drop`; `score_blind_read.py` turns those marks into a two-sided exact test.
This script is the only producer of the pair, so the sheet and its key can never be rebuilt into a different
shuffle by hand.

Both arms are served from GGUF on the Mac — the stack the writing tool actually loads — so the sheet measures
the product, not a training checkpoint that will never be quantised.

    python scripts/build_blind_read.py --sheet "📤-OUTPUT/blind-read-v10-2026-11-04.md" \
        --key ".../bindings/blind-read/key-v10.json" --seed 20261104
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path


from story_rl.prose import load_split  # noqa: E402

PROMPT = (
    "續寫這個場景。\n\n{opening}\n\n只寫兩句。保持同一個聲音、同一個時刻；不要總結，不要另起場景，"
    "不要解釋這段文字的意思。"
)
GENERATION = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 160}

HEADER = """# Blind read — 20 continuations

Ten are the base model, ten are the model trained on your own stories. Shuffled, unlabelled.
For each: mark **keep** if you would read on, **drop** if you would rewrite it.
Do not try to guess which is which — that is the point.
"""


def continue_from(url: str, model: str, opening: str, seed: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(opening=opening)}],
        **GENERATION,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"].strip()


def render(items: list[dict]) -> str:
    blocks = [HEADER]
    for number, item in enumerate(items, start=1):
        blocks.append(f"## #{number}\n\n*opening:* {item['opening']}\n\n{item['continuation']}\n\nverdict:\n")
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--trained-url", default="http://127.0.0.1:8771")
    parser.add_argument("--trained-model", default="voice-zh-v10-Q4_K_M.gguf")
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument("--base-model", default="Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--per-arm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    openings = sorted(task.opening for task in load_split().held_out)
    rng = random.Random(args.seed)
    chosen = rng.sample(openings, min(args.per_arm, len(openings)) * 1)

    items: list[dict] = []
    for index, opening in enumerate(chosen):
        for arm, url, model in (
            ("trained", args.trained_url, args.trained_model),
            ("base", args.base_url, args.base_model),
        ):
            items.append({
                "arm": arm,
                "opening": opening,
                "continuation": continue_from(url, model, opening, seed=index),
            })

    rng.shuffle(items)
    sheet, key = Path(args.sheet), Path(args.key)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(render(items), encoding="utf-8")
    key.write_text(json.dumps(
        {f"#{number}": {"arm": item["arm"]} for number, item in enumerate(items, start=1)},
        ensure_ascii=False, indent=1,
    ) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items ({args.per_arm} per arm) -> {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
