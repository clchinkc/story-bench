"""Turn a flat blind-read sheet into paired A/B choices.

The 2026-11-04 sheet holds twenty unlabelled continuations, but its openings repeat: each of the ten
openings was continued once by the base model and once by the trained one. Judging them as twenty
independent items throws that away — it pays twice for the same reading and lets a dull opening drag
both arms down together.

Pairing them costs the author ten choices instead of twenty marks, and it removes item difficulty from
the comparison: a hard opening stops being a hard opening and becomes a hard *pair*. The sheet still
never names an arm; which side of a pair is which is decided by a seeded coin flip and recorded in a
key written somewhere else.

    python scripts/build_blind_pairs.py \
      --sheet "../../Obsidian-note/📤-OUTPUT/blind-read-v10-2026-11-04.md" \
      --key   "../../Obsidian-note/🧭-GROUND-TRUTH/2026-11-04-writing-tool-context-gap/bindings/blind-read/key-v10.json" \
      --out-sheet pairs.md --out-key pairs-key.json

Refuses rather than guesses: an opening with three continuations, an opening whose two continuations
came from the same arm, or a pair whose two texts are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ITEM = re.compile(r"^## #(\d+)\s*$", re.MULTILINE)
OPENING = re.compile(r"^\*opening:\*\s*(.*)$", re.MULTILINE)
VERDICT = re.compile(r"^verdict:.*$", re.IGNORECASE | re.MULTILINE)


class SheetError(Exception):
    """The sheet cannot be paired without deciding something the author did not."""


def parse_sheet(text: str) -> dict[int, tuple[str, str]]:
    """item number -> (opening, continuation)."""
    items: dict[int, tuple[str, str]] = {}
    heads = list(ITEM.finditer(text))
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        body = text[head.end():end]
        verdict = VERDICT.search(body)
        if verdict:
            body = body[: verdict.start()]
        opening = OPENING.search(body)
        if not opening:
            raise SheetError(f"item #{head.group(1)} has no *opening:* line")
        continuation = body[opening.end():].strip()
        if not continuation:
            raise SheetError(f"item #{head.group(1)} has no continuation")
        items[int(head.group(1))] = (opening.group(1).strip(), continuation)
    if not items:
        raise SheetError("no items found; expected '## #N' headings")
    return items


def group_pairs(items: dict[int, tuple[str, str]], key: dict[str, dict[str, str]]) -> list[tuple[str, list[int]]]:
    """opening -> its two item numbers, in a stable order."""
    by_opening: dict[str, list[int]] = {}
    for number, (opening, _) in items.items():
        by_opening.setdefault(opening, []).append(number)

    pairs: list[tuple[str, list[int]]] = []
    for opening, numbers in by_opening.items():
        numbers.sort()
        if len(numbers) != 2:
            raise SheetError(
                f"opening with {len(numbers)} continuations (items {numbers}); pairing needs exactly 2 — "
                "an unpaired item would silently become its own comparison"
            )
        arms = [key.get(f"#{n}", {}).get("arm") for n in numbers]
        if None in arms:
            raise SheetError(f"key has no arm for items {numbers}")
        if arms[0] == arms[1]:
            raise SheetError(f"items {numbers} are both '{arms[0]}'; there is nothing to compare")
        if items[numbers[0]][1] == items[numbers[1]][1]:
            raise SheetError(f"items {numbers} have identical continuations")
        pairs.append((opening, numbers))
    pairs.sort(key=lambda pair: pair[1][0])
    return pairs


SHEET_HEADER = (
    "# Blind read — ten paired choices",
    "",
    "Each pair is one opening continued twice, by two different models. Which side is which was",
    "decided by a seed and is not recorded here.",
    "",
    "For each pair write **A**, **B**, or **tie** after `choice:`. Pick the one you would read on —",
    "the one that sounds like it was written rather than produced. If both fail, write **neither**.",
    "A blank is read as unrated, not as a vote.",
    "",
    "Do not try to work out which is which. That is the measurement.",
    "",
    "*Character forms are quoted as generated. Where a continuation carries a Simplified character, that is",
    "the generation's orthography and not a transcription slip — converting it would erase the evidence.*",
    "",
)


def build(items, key, seed: str) -> tuple[str, dict]:
    pairs = group_pairs(items, key)
    rng = random.Random(seed)
    lines = list(SHEET_HEADER)
    record: dict[str, dict[str, str]] = {}
    for index, (opening, numbers) in enumerate(pairs, start=1):
        first, second = (numbers[0], numbers[1]) if rng.random() < 0.5 else (numbers[1], numbers[0])
        pair_id = f"P{index}"
        record[pair_id] = {
            "A": f"#{first}",
            "B": f"#{second}",
            "A_arm": key[f"#{first}"]["arm"],
            "B_arm": key[f"#{second}"]["arm"],
            "opening_sha256": hashlib.sha256(opening.encode()).hexdigest()[:16],
        }
        lines += [
            f"## {pair_id}",
            "",
            f"*opening:* {opening}",
            "",
            "**A**",
            "",
            items[first][1],
            "",
            "**B**",
            "",
            items[second][1],
            "",
            "choice:",
            "",
        ]

    document = "\n".join(lines)
    for word in ("trained", "base", "v10", "arm"):
        if word in document.lower():
            raise SheetError(f"the sheet would leak the word {word!r}; the blind is the instrument")
    return document, {"seed": seed, "pairs": record}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", required=True, type=Path, help="flat sheet with '## #N' items")
    parser.add_argument("--key", required=True, type=Path, help="item -> arm mapping")
    parser.add_argument("--out-sheet", required=True, type=Path)
    parser.add_argument("--out-key", required=True, type=Path)
    parser.add_argument("--seed", default="zh-vignette-2026-11-15")
    args = parser.parse_args(argv)

    try:
        items = parse_sheet(args.sheet.read_text(encoding="utf-8"))
        key = json.loads(args.key.read_text(encoding="utf-8"))
        document, record = build(items, key, args.seed)
    except (SheetError, json.JSONDecodeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    args.out_sheet.write_text(document, encoding="utf-8")
    args.out_key.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(record['pairs'])} pairs -> {args.out_sheet} (key: {args.out_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
