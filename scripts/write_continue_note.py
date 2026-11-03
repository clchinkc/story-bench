"""Draft the next lines of a note you are writing, and append them as a callout you can keep or delete.

This is the writing-time surface, not the distribution one. It reads the prose already in a markdown note,
asks the local voice model for three continuations, and appends them under a `> [!draft]` callout with
provenance — which model, which seed, when, and a hash of the prompt — so a line you keep can still be traced
to the model that wrote it.

Design choices that matter:

    append, never overwrite  the note is the author's; the tool only adds a block at the end
    frontmatter stripped     YAML is not prose, and a model that sees `---` writes `---`
    refuse a dead server     a missing model reads as a missing tool, not a traceback
    provenance per draft     model file, seed, prompt hash, timestamp

    python scripts/write_continue_note.py --note ~/vault/場景.md --samples 3 --lang zh

`--dry-run` prints what it would append without touching the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8771"
DEFAULT_MODEL = "voice-zh-Q4_K_M.gguf"
CONTEXT_CHARS = 600

PROMPT_ZH = (
    "續寫這個場景。\n\n{opening}\n\n只寫兩句。保持同一個聲音、同一個時刻；不要總結，不要另起場景，"
    "不要解釋這段文字的意思。"
)
PROMPT_EN = (
    "Continue this scene.\n\n{opening}\n\nWrite exactly two sentences. Stay in the same voice and the same "
    "moment; do not summarise, do not introduce a new scene, and do not explain what the prose means."
)


def prompt_for(lang: str) -> str:
    return PROMPT_ZH if lang == "zh" else PROMPT_EN


def body_of(note: str) -> str:
    """The note's prose: frontmatter, existing draft callouts and headings removed."""
    text = note
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("> [!draft]"):
            continue
        if stripped.startswith(">"):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:
            return json.loads(response.read().decode()).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def draft(url: str, model: str, context: str, *, seed: int, lang: str, tokens: int,
          temperature: float) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(lang).format(opening=context)}],
        "n_predict": tokens, "temperature": temperature, "top_p": 0.95, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    return payload["choices"][0]["message"]["content"].strip()


def render(drafts: list[tuple[int, str]], *, model: str, prompt_hash: str, url: str) -> str:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [f"> [!draft] {model} · {len(drafts)} drafts · {stamp}", ">"]
    for seed, text in drafts:
        lines.append(f"> **seed {seed}** — {text}")
        lines.append(">")
    lines.append(f"> provenance: model `{model}` at `{url}` · prompt `{prompt_hash}` · take or delete")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    note_path = Path(args.note).expanduser()
    note = note_path.read_text(encoding="utf-8")
    context = body_of(note)[-CONTEXT_CHARS:]
    if len(context) < 20:
        raise SystemExit("not enough prose in the note to continue from")

    if not healthy(args.url):
        raise SystemExit(f"no model at {args.url} — start it with: llama-server -m {args.model} --port "
                         f"{args.url.rsplit(':', 1)[-1]}")

    drafts = []
    for seed in range(args.samples):
        drafts.append((seed, draft(args.url, args.model, context, seed=seed, lang=args.lang,
                                   tokens=args.tokens, temperature=args.temperature)))

    prompt_hash = hashlib.sha256(prompt_for(args.lang).format(opening=context).encode()).hexdigest()[:12]
    block = render(drafts, model=args.model, prompt_hash=prompt_hash, url=args.url)

    if args.dry_run:
        print(block)
        return 0

    with note_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + block)
    print(f"appended {len(drafts)} drafts to {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
