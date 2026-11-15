"""Score the paired blind read: did the trained model win more pairs than the base one?

Ten pairs, one choice each, key held apart. Because the pairs are matched on opening, the comparison
is a sign test over the pairs that expressed a preference — not the unpaired table the flat sheet
produced. A tie or "neither" is reported and excluded from the test, because "both are bad" is not
evidence about which arm is better.

    python scripts/score_blind_pairs.py --sheet pairs.md --key pairs-key.json

A blank is unrated, never a vote. Anything the sheet does not say — an unrecognised word — is reported
as invalid rather than read as a preference, because a scorer that guesses is a scorer that decides
the result.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

PAIR = re.compile(r"^## (P\d+)\s*$", re.MULTILINE)
CHOICE = re.compile(r"^choice:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

PREFERS_A = {"a", "aa"}
PREFERS_B = {"b", "bb"}
NO_PREFERENCE = {"tie", "draw", "same", "either", "平", "一樣"}
BOTH_BAD = {"neither", "both", "none", "x", "都不", "皆否"}


class SheetError(Exception):
    """The sheet or the key cannot be read without guessing."""


def parse_choices(text: str) -> dict[str, str]:
    """pair id -> the word written after `choice:`, lowercased; '' means marked but blank."""
    choices: dict[str, str] = {}
    heads = list(PAIR.finditer(text))
    if not heads:
        raise SheetError("no pairs found; expected '## Pn' headings")
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        match = CHOICE.search(text[head.end():end])
        if not match:
            raise SheetError(f"{head.group(1)} has no 'choice:' line")
        choices[head.group(1)] = match.group(1).strip().lower()
    return choices


def classify(word: str) -> str:
    """'A' | 'B' | 'tie' | 'neither' | 'unrated' | 'invalid'."""
    if not word:
        return "unrated"
    if word in PREFERS_A:
        return "A"
    if word in PREFERS_B:
        return "B"
    if word in NO_PREFERENCE:
        return "tie"
    if word in BOTH_BAD:
        return "neither"
    return "invalid"


def sign_test(trained: int, base: int) -> float:
    """Two-sided exact binomial on the pairs that expressed a preference."""
    n = trained + base
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(trained, base) + 1))
    return min(1.0, 2 * tail / 2**n)


def score(choices: dict[str, str], key: dict) -> dict:
    pairs = key.get("pairs", {})
    counts = {"trained": 0, "base": 0, "tie": 0, "neither": 0, "unrated": 0, "invalid": 0}
    counted = 0
    for pair_id, entry in pairs.items():
        if pair_id not in choices:
            raise SheetError(f"the key names {pair_id} but the sheet does not have it")
        verdict = classify(choices[pair_id])
        counted += 1
        if verdict in {"A", "B"}:
            arm = entry[f"{verdict}_arm"]
            if arm not in {"trained", "base"}:
                raise SheetError(f"{pair_id}: unknown arm {arm!r} in the key")
            counts[arm] += 1
        else:
            counts[verdict] += 1

    extra = set(choices) - set(pairs)
    if extra:
        raise SheetError(f"the sheet has pairs the key does not: {sorted(extra)}")

    p = sign_test(counts["trained"], counts["base"])
    decided = counts["trained"] + counts["base"]
    return {**counts, "pairs": counted, "decided": decided, "p": p}


def render(result: dict) -> str:
    lines = [
        f"pairs      {result['pairs']}",
        f"trained    {result['trained']}",
        f"base       {result['base']}",
        f"tie        {result['tie']}",
        f"neither    {result['neither']}",
        f"unrated    {result['unrated']}",
        f"invalid    {result['invalid']}",
    ]
    if result["invalid"]:
        lines.append("")
        lines.append("invalid marks are not read as preferences; fix them and re-run.")
    if result["decided"] == 0:
        lines.append("")
        lines.append("no pair expressed a preference — there is no result here, only an unmarked sheet.")
        return "\n".join(lines)
    lines += [
        "",
        f"sign test  n={result['decided']} p={result['p']:.4f} (two-sided, ties and 'neither' excluded)",
    ]
    if result["p"] < 0.05:
        winner = "trained" if result["trained"] > result["base"] else "base"
        lines.append(f"verdict    {winner} preferred at p<0.05")
    else:
        lines.append("verdict    no detectable preference at this n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        choices = parse_choices(args.sheet.read_text(encoding="utf-8"))
        key = json.loads(args.key.read_text(encoding="utf-8"))
        result = score(choices, key)
    except (SheetError, json.JSONDecodeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
