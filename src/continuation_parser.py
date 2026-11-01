"""Getting the continuation out of a raw completion — three policies, nested by construction.

This module exists because the reward is only as honest as the text it is handed. W0 attempt 1
measured `e = 0.986` on a model whose prose was fine and whose JSON was not: the reward scored
formatting and called it continuity. Stage B runs on Tinker, where `SamplingParams` has no
grammar and no `json_schema`, so the same cliff is approached deliberately. Which parser the
training loop uses is therefore a measurement decision, not an implementation detail, and it
has to be chosen from numbers — see
`🧭-GROUND-TRUTH/2026-10-30-story-rl-end-to-end/bindings/w1-parser/`.

Three policies, each a strict superset of the one before:

    strict   a JSON object whose `continuation` is a string. Anything else is unparsed.
    lenient  the same, plus fenced JSON, key aliases, single-string objects and bare strings.
    prose    never unparsed. A recognised key wins; otherwise the whole text is the continuation.

`prose` is the only policy that cannot report "unparsed", which is exactly why it is the one
that cannot silently convert a formatting failure into a wrong answer. That is a property to
be measured, not a preference to be asserted.

Purity: imports `json`, `re` and `dataclasses`; no file, network, clock or random access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Order matters: each policy accepts everything the one to its left accepts.
POLICIES = ("strict", "lenient", "prose")

# A JSON object that names its own continuation in one of these is readable by every policy.
CONTINUATION_KEYS = ("continuation", "text", "prose", "story", "output")

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass(frozen=True)
class Extraction:
    """What a policy made of one completion. `how` is for the log, not for control flow."""

    policy: str
    text: str | None
    how: str

    @property
    def unparsed(self) -> bool:
        return self.text is None


def strip_fences(text: str) -> str:
    """Remove a leading ```json and a trailing ``` without touching anything else."""
    return _FENCE.sub("", text).strip()


def _mask_strings(text: str) -> str:
    """Replace every JSON string literal with spaces, keeping the length and the indices.

    Masking rather than state-tracking is what keeps the brace count flat: once no string can
    contain a brace, counting braces is the whole algorithm. Length is preserved so the caller
    can still slice the original text at the index this scan returns.
    """
    return _STRING.sub(lambda match: " " * len(match.group()), text)


def _closing_brace(masked: str, start: int) -> int | None:
    """Index of the brace that closes the one at `start` in a string-masked text."""
    depth = 0
    for index in range(start, len(masked)):
        if masked[index] == "{":
            depth += 1
            continue
        if masked[index] != "}":
            continue
        depth -= 1
        if depth == 0:
            return index
    return None


def first_object(text: str) -> str | None:
    """The first brace-balanced object in `text`, or None.

    Deliberately brace-counting rather than `first {` … `last }`: a model that emits two
    objects, or an object followed by chatter containing a brace, would otherwise hand the
    loader a string that is not valid JSON and lose a good continuation.
    """
    masked = _mask_strings(text)
    start = masked.find("{")
    if start == -1:
        return None
    end = _closing_brace(masked, start)
    return None if end is None else text[start : end + 1]


def _continuation_from_object(payload: object) -> str | None:
    """Pull the continuation out of a decoded JSON value, or None if it carries none."""
    if isinstance(payload, str):
        return payload or None
    if not isinstance(payload, dict):
        return None
    for key in CONTINUATION_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # A single-string object is unambiguous however the model named the key.
    strings = [value for value in payload.values() if isinstance(value, str) and value.strip()]
    return strings[0] if len(strings) == 1 else None


def extract_strict(text: str) -> Extraction:
    """Only the constrained shape: one object, key `continuation`."""
    candidate = first_object(text)
    if candidate is None:
        return Extraction("strict", None, "no-object")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return Extraction("strict", None, "bad-json")
    if not isinstance(payload, dict):
        return Extraction("strict", None, "not-object")
    value = payload.get("continuation")
    if not isinstance(value, str) or not value.strip():
        return Extraction("strict", None, "no-continuation-key")
    return Extraction("strict", value, "strict")


def _object_value(text: str) -> str | None:
    """The continuation carried by the first object in `text`, or None."""
    candidate = first_object(text)
    if candidate is None:
        return None
    try:
        return _continuation_from_object(json.loads(candidate))
    except json.JSONDecodeError:
        return None


def _bare_string(text: str) -> str | None:
    """A quoted JSON string standing alone, with or without surrounding whitespace."""
    for probe in (strip_fences(text), text.strip()):
        if len(probe) >= 2 and probe[0] == '"' and probe[-1] == '"':
            try:
                value = json.loads(probe)
            except json.JSONDecodeError:
                continue
            if isinstance(value, str) and value.strip():
                return value
    return None


def extract_lenient(text: str) -> Extraction:
    """Fenced JSON, key aliases, single-string objects, bare strings."""
    strict = extract_strict(text)
    if not strict.unparsed:
        return Extraction("lenient", strict.text, "lenient/strict-shape")
    value = _object_value(strip_fences(text))
    if value is not None:
        return Extraction("lenient", value, "lenient/object")
    value = _bare_string(text)
    if value is not None:
        return Extraction("lenient", value, "lenient/bare-string")
    return Extraction("lenient", None, "no-json")


def extract_prose(text: str) -> Extraction:
    """Never unparsed: a recognised key wins, else the text itself is the continuation."""
    lenient = extract_lenient(text)
    if not lenient.unparsed:
        return Extraction("prose", lenient.text, f"prose/{lenient.how}")
    return Extraction("prose", strip_fences(text), "prose/raw")


_POLICY_FUNCS = {
    "strict": extract_strict,
    "lenient": extract_lenient,
    "prose": extract_prose,
}


def extract(text: str, policy: str) -> Extraction:
    """Apply a named policy. Unknown names are refused rather than defaulted."""
    try:
        function = _POLICY_FUNCS[policy]
    except KeyError:
        raise ValueError(f"unknown parser policy {policy!r}; expected one of {POLICIES}") from None
    return function(text)
