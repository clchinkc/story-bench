"""Traditional-character audit and repair, in the author's own orthography.

Why this exists: the shipped model wrote an unambiguous simplified character in **52 %** of its generations
(measured on 99 samples), the untrained base model in 26 %, and the reward could not see any of it — `guard()`
rejects Latin, Greek and Cyrillic, not simplified Chinese. The author's vault already refuses such text with
its own `format:chinese-lint` gate, so the tool was producing notes the rest of the system would reject.

Three decisions shape this module, and each came from a measurement rather than a preference:

    the author's corpus is the authority   `裏` 311 / `裡` 1, `後` 210 / `后` 0, `台` 28 / `臺` 0. Where a
                                           character has two traditional forms, the one she writes is the
                                           correct one, so the map is her usage rather than a generic table.
    one-to-one only in the blanket map     `发`, `干`, `面`, `只`, `里`, `后`, `台`, `卷`, `托`, `冲` have more
                                           than one traditional target. A blanket rewrite of those is how a
                                           clean draft acquires a wrong-but-traditional character that the
                                           author will not notice — worse than the simplified one she would.
    characters she already uses are safe   excluded from the map outright. Her usage is correct by
                                           definition, so a rule can never "correct" her.

`audit()` reports three kinds, and only `simplified` is a defect:

    simplified   an unambiguous mapping, repaired deterministically
    context      a one-to-many character whose form depends on the word — repaired by rule, reported
    ambiguous    a one-to-many character the rules do not recognise — left alone and flagged

The reward vetoes `simplified` and `context` only. `ambiguous` is a person's call, so it pays nothing and
costs nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "story_rl" / "traditional_map.json"

# One-to-many characters: the target depends on the word, so a blanket map would be wrong. Each rule is a
# (pattern, replacement) pair applied in order; the first match wins and the text is consumed left to right.
#
#   the word decides      松開 -> 鬆開, but 松樹 stays 松樹
#   the author decides    里 -> 裏 (her form), 着 -> 著 (her form), 台 stays 台 (her form)
#   the compound decides  衝動 vs 沖洗, 捲起 vs 卷宗, 託付 vs 托住
CONTEXT_RULES: tuple[tuple[str, str], ...] = (
    (r"松(?=[開弛散軟懈])", "鬆"),
    (r"(頭|鬢|鬚|髮)发", r"\1髮"),
    (r"发(?=[型絲梢尾際膚])", "髮"),
    (r"冲动|冲突|冲击|冲进|冲出|冲上|冲过", lambda m: m.group(0).replace("冲", "衝")),
    (r"冲洗|冲茶|冲淡|冲走", lambda m: m.group(0).replace("冲", "沖")),
    (r"卷(?=[起入曲縮進動拢])", "捲"),
    (r"托(?=[付管辭詞])", "託"),
    (r"并(?![非列])", "並"),
)

# Applied when no rule matched: the author's form for a character she uses in exactly one way. Each is a
# single deterministic target, so it is a repair rather than a judgement.
AUTHOR_FORMS: dict[str, str] = {
    "里": "裏",
    "后": "後",
    "着": "著",
    "么": "麼",
    "湿": "濕",   # she writes 濕 (19) and never 溼 (0); the generic map prefers 溼
}

# Characters that are valid Traditional in their own right. The author may simply never have needed them
# (岩), or she writes the unaccented form consistently (台: 28 uses, never 臺). Correcting these would be a
# regression, so they are excluded from the map and left alone.
KEEP: frozenset[str] = frozenset("台岩托卷")

# One-to-many characters the map deliberately excludes. If one survives the context rules it is reported as
# ambiguous and left exactly as written: a wrong-but-Traditional character is harder to notice than a visible
# simplified one, so guessing here costs more than flagging.
AMBIGUOUS: frozenset[str] = frozenset("发干面只里后台卷托冲并么着于余谷斗云松采系表制划板蒙征")


@dataclass(frozen=True)
class Finding:
    """One character that is not the author's, with enough context to judge it."""

    char: str
    replacement: str | None
    kind: str          # simplified | context | ambiguous
    position: int
    excerpt: str


class TraditionalError(ValueError):
    """The map is missing or malformed — a broken instrument, never an empty result."""


def load_map(path: Path = MAP_PATH) -> dict[str, str]:
    if not path.exists():
        raise TraditionalError(f"traditional map missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("map")
    if not isinstance(table, dict) or len(table) < 1000:
        raise TraditionalError(f"{path.name}: expected a mapping table, got {type(table).__name__}")
    return table


def _excerpt(text: str, position: int, width: int = 6) -> str:
    start, end = max(0, position - width), min(len(text), position + width + 1)
    return text[start:end]


def _context_pass(text: str) -> tuple[str, list[Finding]]:
    """Apply the word-level rules; report every hit so a repair is never silent."""
    findings: list[Finding] = []
    for pattern, replacement in CONTEXT_RULES:
        for match in re.finditer(pattern, text):
            fixed = replacement(match) if callable(replacement) else re.sub(pattern, replacement, match.group(0))
            if fixed == match.group(0):
                continue
            findings.append(Finding(
                char=match.group(0), replacement=fixed, kind="context",
                position=match.start(), excerpt=_excerpt(text, match.start()),
            ))
        text = re.sub(pattern, replacement, text)
    return text, findings


def _blanket_pass(text: str, table: dict[str, str]) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    out = []
    for index, char in enumerate(text):
        fixed = table.get(char)
        if fixed is None:
            out.append(char)
            continue
        findings.append(Finding(char=char, replacement=fixed, kind="simplified",
                                position=index, excerpt=_excerpt(text, index)))
        out.append(fixed)
    return "".join(out), findings


def audit(text: str, table: dict[str, str] | None = None) -> list[Finding]:
    """Every character in `text` that is not the author's, classified.

    Order matters: the context rules run first, because several exist precisely to stop the blanket map from
    touching a character that belongs to a word (`松樹` must not become `鬆樹`).
    """
    table = load_map() if table is None else table
    remaining, context_findings = _context_pass(text)

    for char, author_form in AUTHOR_FORMS.items():
        if author_form == char:
            continue
        for match in re.finditer(re.escape(char), remaining):
            context_findings.append(Finding(
                char=char, replacement=author_form, kind="context",
                position=match.start(), excerpt=_excerpt(remaining, match.start()),
            ))
        remaining = remaining.replace(char, author_form)

    repaired, blanket_findings = _blanket_pass(remaining, table)
    findings = context_findings + blanket_findings
    findings.sort(key=lambda f: f.position)
    return findings


def repair(text: str, table: dict[str, str] | None = None) -> tuple[str, list[Finding]]:
    """Return the cleaned text and the findings.

    The rewrite runs the same three passes in the same order as `audit` — word rules, the author's forms,
    then the blanket map — rather than patching characters by index. Index patching was the first version and
    it was wrong: a word rule replaces *several* characters at one position (`冲动` -> `衝動`), so writing it
    into a single slot lengthened the string and produced text like `衝动動`.
    """
    table = load_map() if table is None else table
    findings = audit(text, table)

    cleaned, _ = _context_pass(text)
    for char, author_form in AUTHOR_FORMS.items():
        cleaned = cleaned.replace(char, author_form)
    cleaned, _ = _blanket_pass(cleaned, table)
    return cleaned, findings


def defects(text: str, table: dict[str, str] | None = None) -> list[Finding]:
    """The subset the reward may veto: unambiguous, and repaired by rule. Ambiguity is a person's call."""
    return [f for f in audit(text, table) if f.kind in {"simplified", "context"}]
