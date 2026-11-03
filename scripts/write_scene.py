"""Draft scene continuations from the local model — the writer, inside the Narrative Craft flow.

This is the delivery surface. The skill's flow is *read one accepted view → prepare one complete recommended
result → prove, seal and present*; the drafting happens at the second step, and until now that step was the
agent writing by hand. This script puts a trained, quantised, on-device model there:

    python scripts/write_scene.py --opening scene.md --samples 3

It prints N candidate continuations, each on the same prompt the model was trained on, and nothing else. The
skill's own review then chooses and seals — the model drafts, the flow judges. No judge runs here; the
artifact carries no assessor (03-CONTEXT § Exclusions).

The endpoint defaults to the prose artifact on `:8770`; override with `--url` for the base model on `:8768`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8770"
DEFAULT_MODEL = "prose-final-Q4_K_M.gguf"

PROMPT_ZH = (
    "續寫這個場景。\n"
    "\n"
    "{opening}\n"
    "\n"
    "只寫兩句。保持同一個聲音、同一個時刻；不要總結，不要另起場景，不要解釋這段文字的意思。"
)

PROMPT = (
    "Continue this scene.\n"
    "\n"
    "{opening}\n"
    "\n"
    "Write exactly two sentences. Stay in the same voice and the same moment; do not summarise, do not "
    "introduce a new scene, and do not explain what the prose means."
)


def prompt_for(lang: str) -> str:
    """The template for one language. Frozen per corpus: an English prompt on Chinese prose asks a model to
    write in the wrong language, and the run then measures the prompt rather than the training."""
    return PROMPT_ZH if lang == "zh" else PROMPT


def healthy(url: str) -> bool:
    """A dead server should read as a missing tool, not as a traceback."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:
            return json.loads(response.read().decode()).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def draft(url: str, model: str, opening: str, *, seed: int, tokens: int, temperature: float,
          lang: str = "en") -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(lang).format(opening=opening.strip())}],
        "n_predict": tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    return payload["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening", help="file holding the scene opening; omit to read stdin")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lang", default="en", choices=("en", "zh"))
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if not healthy(args.url):
        raise SystemExit(f"no model at {args.url} — start it with: llama-server -m {args.model} --port "
                         f"{args.url.rsplit(':', 1)[-1]}")

    opening = Path(args.opening).read_text(encoding="utf-8") if args.opening else sys.stdin.read()
    if not opening.strip():
        raise SystemExit("an empty opening has nothing to continue")

    for sample in range(args.samples):
        print(f"--- draft {sample + 1} ---")
        print(draft(args.url, args.model, opening, seed=sample, tokens=args.tokens,
                    temperature=args.temperature, lang=args.lang))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
