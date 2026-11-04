"""Traditional-character audit and repair, in the author's own orthography.

The shipped model wrote an unambiguous simplified character in 52 % of its generations before this existed
(v9: 25 %, v10: 2 %). The reward could not see it, and the author's own `format:chinese-lint` gate would
reject the same text, so the tool was producing notes the rest of the system refuses.

Two measurements shape the rules:

    the author's corpus is the authority   裏 311 / 裡 1, 後 210 / 后 0, 台 28 / 臺 0. Where a character has
                                           two traditional forms, the one she writes is the correct one.
    one-to-one only in the blanket map     `發 干 面 只 裏 後 台 卷 托 衝 並 鬆` map one-to-many. A blanket
                                           rewrite there is how a clean draft acquires a wrong-but-
                                           Traditional character the author will not notice.

`rewrite()` returns the cleaned text and what it changed; `audit()` returns only the findings. Both are the
same pass, so a repair can never disagree with its own report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "story_rl" / "traditional_map.json"


class Kind(str, Enum):
    """Why a character was rewritten. Both kinds are deterministic, so both may veto a rollout."""

    SIMPLIFIED = "simplified"
    CONTEXT = "context"


@dataclass(frozen=True)
class Finding:
    """One rewrite, with enough context for a person to audit it."""

    matched: str
    repaired: str
    kind: Kind
    position: int
    excerpt: str


# The author's form where she uses a character in exactly one way. Single deterministic targets, so these are
# repairs rather than judgements. `湿` is here because the generic map prefers 溼 and she writes 濕 (19 / 0).
AUTHOR_FORMS: dict[str, str] = {
    "里": "裏",
    "后": "後",
    "着": "著",
    "么": "麼",
    "湿": "濕",
}

# Characters that are valid Traditional in their own right but map one-to-many, so the word decides. Applied
# before the blanket map, because `松樹` must not become `鬆樹`.
CONTEXT_RULES: tuple[tuple[str, str], ...] = (
    (r"松(?=[開弛散軟懈])", "鬆"),
    (r"(頭|鬢|鬚|髮)发", r"\1髮"),
    (r"发(?=[型絲梢尾際膚])", "髮"),
    (r"(?:冲)(?=[动突击进出上过])", "衝"),
    (r"(?:冲)(?=[洗茶淡走])", "沖"),
    (r"卷(?=[起入曲縮進動拢])", "捲"),
    (r"托(?=[付管辭詞])", "託"),
    (r"并(?![非列])", "並"),
)

# Valid Traditional characters the map must never touch: `岩` is correct as written, and the author writes `台`
# (28 uses, never 臺). Excluded from the generated map; asserted against it at load.
KEEP: frozenset[str] = frozenset("台岩托卷")

# One-to-many characters the generated map must exclude, because a blanket rewrite would pick one form and the
# word decides. `scripts/build_traditional_map.py` reads this set, so the rule and the data cannot drift apart.
ONE_TO_MANY: frozenset[str] = frozenset("发干面只冲并么于余谷斗云松采系表制划板蒙征")


class TraditionalError(ValueError):
    """The map is missing, malformed, or disagrees with this module's own rules."""


@lru_cache(maxsize=1)
def load_map(path: Path = MAP_PATH) -> dict[str, str]:
    """The generated one-to-one table, validated against the rules that must take precedence over it."""
    if not path.exists():
        raise TraditionalError(f"traditional map missing: {path}")
    table = json.loads(path.read_text(encoding="utf-8")).get("map")
    if not isinstance(table, dict) or len(table) < 1000:
        raise TraditionalError(f"{path.name}: expected a mapping table, got {type(table).__name__}")

    shadowed = set(table) & (set(AUTHOR_FORMS) | KEEP | ONE_TO_MANY)
    if shadowed:
        raise TraditionalError(f"{path.name}: map shadows a rule or a kept character: {sorted(shadowed)}")
    return table


def _excerpt(text: str, position: int, width: int = 6) -> str:
    start, end = max(0, position - width), min(len(text), position + width + 1)
    return text[start:end]


def _record(text: str, match: re.Match[str], fixed: str, kind: Kind) -> Finding:
    return Finding(
        matched=match.group(0),
        repaired=fixed,
        kind=kind,
        position=match.start(),
        excerpt=_excerpt(text, match.start()),
    )


def _apply_pattern(text: str, pattern: str, replacement: str) -> tuple[str, list[Finding]]:
    """Apply one word rule, reporting each hit. Length is preserved by every rule, so positions stay true."""
    findings: list[Finding] = []

    def substitute(match: re.Match[str]) -> str:
        fixed = match.expand(replacement)
        if fixed != match.group(0):
            findings.append(_record(text, match, fixed, Kind.CONTEXT))
        return fixed

    return re.sub(pattern, substitute, text), findings


def _apply_pairs(text: str, mapping: dict[str, str], kind: Kind) -> tuple[str, list[Finding]]:
    """Character-for-character substitution. A dict lookup per character beats a 2,579-way alternation."""
    findings: list[Finding] = []
    if not mapping:
        return text, findings
    out: list[str] = []
    for position, character in enumerate(text):
        fixed = mapping.get(character)
        if fixed is None:
            out.append(character)
            continue
        findings.append(Finding(character, fixed, kind, position, _excerpt(text, position)))
        out.append(fixed)
    return "".join(out), findings


def rewrite(text: str, table: dict[str, str] | None = None) -> tuple[str, list[Finding]]:
    """Rewrite to the author's orthography, returning the cleaned text and every change made.

    Order is load-bearing: word rules, then her forms, then the blanket map. The blank map runs last so a
    character that belongs to a word has already been claimed.
    """
    table = load_map() if table is None else table
    findings: list[Finding] = []

    cleaned = text
    for pattern, replacement in CONTEXT_RULES:
        cleaned, hits = _apply_pattern(cleaned, pattern, replacement)
        findings.extend(hits)
    cleaned, hits = _apply_pairs(cleaned, AUTHOR_FORMS, Kind.CONTEXT)
    findings.extend(hits)
    cleaned, hits = _apply_pairs(cleaned, table, Kind.SIMPLIFIED)
    findings.extend(hits)

    findings.sort(key=lambda finding: finding.position)
    return cleaned, findings


def audit(text: str, table: dict[str, str] | None = None) -> list[Finding]:
    """The findings alone. Every one is deterministic and repairable, so every one may veto."""
    return rewrite(text, table)[1]
