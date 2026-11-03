"""Score the blind read: did you keep more of the trained model's continuations than the base's?

The sheet holds twenty unlabelled drafts and a `verdict:` line under each; the key says which arm each came
from. This reads both and reports, per arm, how many you kept — plus a two-sided exact test, because with ten
per arm a difference of two is nothing.

    python scripts/score_blind_read.py \
      --sheet "📤-OUTPUT/blind-read-prose-2026-11-03.md" \
      --key   "🧭-GROUND-TRUTH/…/bindings/blind-read/key.json"

Unrated items are reported as unrated rather than counted as drops — a blank is not a verdict, and treating it
as one would silently bias the result toward whichever arm you got bored of first.
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

ITEM = re.compile(r"^## #(\d+)\s*$", re.MULTILINE)
VERDICT = re.compile(r"^verdict:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_sheet(text: str) -> dict[int, str]:
    """item number -> verdict word, lowercased. Empty string means marked but blank."""
    verdicts: dict[int, str] = {}
    heads = list(ITEM.finditer(text))
    for index, head in enumerate(heads):
        body = text[head.end(): heads[index + 1].start() if index + 1 < len(heads) else len(text)]
        match = VERDICT.search(body)
        verdicts[int(head.group(1))] = (match.group(1).strip().lower() if match else "")
    return verdicts


def keep_word(verdict: str) -> bool | None:
    """None means unrated. Anything else is read as keep only when it actually says keep."""
    if not verdict:
        return None
    if verdict.startswith(("keep", "kept", "yes", "k", "留")):
        return True
    if verdict.startswith(("drop", "dropped", "no", "d", "rewrite", "改")):
        return False
    return None


def exact_two_sided(a_keep: int, a_n: int, b_keep: int, b_n: int) -> float:
    """Fisher's exact test on the 2x2 keep/drop table; two-sided by summing tables no more likely."""
    total_keep, total_n = a_keep + b_keep, a_n + b_n
    if total_n == 0:
        return 1.0
    observed = comb(a_n, a_keep) * comb(b_n, b_keep)
    hits = 0
    for k in range(0, min(a_n, total_keep) + 1):
        b = total_keep - k
        if b < 0 or b > b_n:
            continue
        if comb(a_n, k) * comb(b_n, b) <= observed:
            hits += comb(a_n, k) * comb(b_n, b)
    return hits / (comb(total_n, total_keep) or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    verdicts = parse_sheet(Path(args.sheet).expanduser().read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).expanduser().read_text(encoding="utf-8"))

    tally = {"base": [0, 0], "v1": [0, 0]}  # [kept, rated]
    unrated = 0
    for label, entry in key.items():
        number = int(label.lstrip("#"))
        kept = keep_word(verdicts.get(number, ""))
        if kept is None:
            unrated += 1
            continue
        arm = entry["arm"]
        tally.setdefault(arm, [0, 0])
        tally[arm][1] += 1
        tally[arm][0] += 1 if kept else 0

    print(f"{'arm':8s} {'kept':>5s} {'rated':>6s} {'rate':>7s}")
    for arm, (kept, rated) in tally.items():
        rate = f"{kept / rated:.0%}" if rated else "—"
        print(f"{arm:8s} {kept:5d} {rated:6d} {rate:>7s}")

    if unrated:
        print(f"\n{unrated} unrated — a blank is not a drop, so they are excluded.")
    if len(tally) >= 2 and all(v[1] for v in tally.values()):
        (a_keep, a_n), (b_keep, b_n) = tally["base"], tally["v1"]
        p = exact_two_sided(a_keep, a_n, b_keep, b_n)
        print(f"\nexact two-sided p = {p:.4f} — with ten per arm, treat anything above 0.05 as 'no signal'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
