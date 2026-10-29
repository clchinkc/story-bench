"""Refuse a prompt set whose train/dev half touches the sealed holdout.

W9 trains on a public-domain corpus that is deliberately *outside* story-bench's own
evaluated set. The failure this guards against is not malice but drift: a passage
picked up for training that happens to be one the sealed holdout also uses, which
silently converts a held-out evaluation into a training example and inflates every
number that follows.

Two identities are checked, because either alone is bypassable:

    task id       catches the same task re-used under a familiar name
    content hash  catches the same passage re-titled, re-split or re-punctuated

The guard is pure and fails closed: an empty holdout is refused (it proves nothing),
and every overlap is reported rather than the first.

Purity: imports `hashlib` (deterministic, no I/O), `collections.abc` and `dataclasses`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

DIGEST_PREFIX = "sha256:"


class HoldoutLeakError(ValueError):
    """A training identity or a training passage also appears in the sealed holdout."""


@dataclass(frozen=True)
class HoldoutSet:
    """The identities that must never appear in training."""

    task_ids: frozenset[str]
    digests: frozenset[str]

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise HoldoutLeakError(
                "a holdout with no task id proves nothing; refusing an empty set"
            )
        if not self.digests:
            raise HoldoutLeakError(
                "a holdout with no content digest cannot catch a re-titled passage"
            )
        for digest in self.digests:
            if not digest.startswith(DIGEST_PREFIX) or len(digest) != len(DIGEST_PREFIX) + 64:
                raise HoldoutLeakError(
                    f"digest {digest!r} is not '{DIGEST_PREFIX}<64 lowercase hex>'"
                )


def prompt_digest(text: str) -> str:
    """Stable content identity for a passage, normalised for surrounding whitespace."""
    if not isinstance(text, str):
        raise HoldoutLeakError(f"passage must be a str, got {type(text).__name__}")
    normalised = " ".join(text.split())
    return DIGEST_PREFIX + hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def assert_disjoint(
    train_ids: Iterable[str],
    train_passages: Iterable[str],
    holdout: HoldoutSet,
) -> None:
    """Raise `HoldoutLeakError` listing every overlap; return None when clean."""
    ids = set(train_ids)
    digests = {prompt_digest(passage) for passage in train_passages}

    shared_ids = sorted(ids & set(holdout.task_ids))
    shared_digests = sorted(digests & set(holdout.digests))
    if shared_ids or shared_digests:
        raise HoldoutLeakError(
            "training set leaks the sealed holdout — "
            f"task ids: {shared_ids or 'none'}; content: {shared_digests or 'none'}"
        )
