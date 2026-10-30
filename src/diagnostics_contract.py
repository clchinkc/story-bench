"""Checker for the diagnostic instrument contract (docs/diagnostics-instrument-contract.md).

A diagnostic that is quietly wrong is worse than no diagnostic: it looks like evidence. This module
turns each contract requirement into a test that a diagnostics report must pass BEFORE it is
reportable.

Division of labour with the contract: the contract says WHAT must hold, this module decides whether a
given report holds it, and `tests/test_diagnostics_contract.py` carries the controls — a report that
satisfies every requirement must pass, and five separately-broken reports must each be refused for the
requirement they break. A checker without those controls would be the same class of claim as the
contract without a checker.

Nothing here decides whether a diagnostic is *good*. It decides whether it is ADMISSIBLE.
"""
from __future__ import annotations

from datetime import date
from typing import Any

# D-1: a style diagnostic must separate from its own controls by these margins.
MIN_POSITIVE_SEPARATION = 3.0
MAX_NEGATIVE_RATIO = 1.0

# D-5: below this fraction of its prior separation, a diagnostic is retired.
MIN_SEPARATION_RETENTION = 0.8

KINDS = ("style", "homogenization", "punctuation")


def _require(cond: bool, out: list[str], message: str) -> None:
    if not cond:
        out.append(message)


def _check_ratio(diag: dict, out: list[str], name: str) -> None:
    """D-1: numerator and denominator, both present, denominator non-zero."""
    numerator, denominator = diag.get("numerator_rate"), diag.get("denominator_rate")
    _require(isinstance(numerator, (int, float)), out, f"{name}: numerator_rate missing — D-1")
    _require(isinstance(denominator, (int, float)) and denominator > 0, out,
             f"{name}: denominator_rate must be a positive number — D-1 (a ratio needs its base rate)")


def _check_controls(diag: dict, out: list[str], name: str) -> None:
    """D-1: both controls, and both must hold — one passing is a failed diagnostic."""
    controls = diag.get("controls") or {}
    positive, negative = controls.get("positive_ratio"), controls.get("negative_ratio")
    _require(isinstance(positive, (int, float)) and positive > MIN_POSITIVE_SEPARATION, out,
             f"{name}: positive control must exceed {MIN_POSITIVE_SEPARATION}x — D-1")
    _require(isinstance(negative, (int, float)) and negative <= MAX_NEGATIVE_RATIO, out,
             f"{name}: negative control (the author's own text) must be <= {MAX_NEGATIVE_RATIO}x "
             f"— D-1, else the diagnostic fires on human writing")


def _check_anchor_and_invariance(diag: dict, out: list[str], name: str) -> None:
    """D-2: a fixed external anchor, and a byte-identical roster-invariance test."""
    baseline = diag.get("baseline") or {}
    _require(baseline.get("fixed_anchor") is True, out,
             f"{name}: baseline must be a FIXED external anchor — D-2 (the roster is not a scale)")
    _require(bool(baseline.get("name")), out, f"{name}: baseline must be named — D-2")
    invariance = diag.get("invariance") or {}
    sizes = invariance.get("roster_sizes_tested") or []
    _require(len(sizes) >= 2, out,
             f"{name}: invariance must be tested at >=2 roster sizes — D-2")
    _require(invariance.get("byte_identical") is True, out,
             f"{name}: scores must be byte-identical across roster sizes — D-2 "
             f"(a score that moves with the roster is roster-relative)")


def _check_homogenization(diag: dict, out: list[str], name: str) -> None:
    """D-3: within one prompt across samples, never across models."""
    _require(isinstance(diag.get("samples_n"), int) and diag["samples_n"] >= 2, out,
             f"{name}: homogenization needs a declared samples_n >= 2 — D-3")
    _require(diag.get("scope") == "same_prompt_same_model", out,
             f"{name}: scope must be 'same_prompt_same_model' — D-3 (cross-model comparison is "
             f"meaningless for this quantity)")


def _check_punctuation(diag: dict, out: list[str], name: str) -> None:
    """D-4: locale-checked first, or explicitly withdrawn for the affected locales."""
    locale = diag.get("locale_invariance") or {}
    if diag.get("locale_confounded") is True:
        _require(diag.get("locales_withdrawn") not in (None, [], ()), out,
                 f"{name}: a locale-confounded diagnostic must list the locales it is withdrawn "
                 f"from — D-4")
        return
    _require(locale.get("tested") is True, out,
             f"{name}: punctuation diagnostics need a locale-invariance test — D-4 "
             f"(the full-width dash is a different code point)")
    _require(locale.get("within_declared_band") is True, out,
             f"{name}: locale values must fall inside the pre-declared band — D-4")


def _check_expiry(diag: dict, out: list[str], name: str, as_of: date) -> None:
    """D-5: bound to a probe set and an expiry; a lapsed diagnostic must be retired, and that kept."""
    _require(bool(diag.get("probe_set")), out, f"{name}: probe_set missing — D-5")
    separation = diag.get("separation")
    _require(isinstance(separation, (int, float)), out, f"{name}: separation not measured — D-5")
    prior = diag.get("separation_prior")
    if isinstance(separation, (int, float)) and isinstance(prior, (int, float)) and prior > 0:
        _require(separation >= prior * MIN_SEPARATION_RETENTION, out,
                 f"{name}: separation {separation} fell below "
                 f"{MIN_SEPARATION_RETENTION:.0%} of {prior} — D-5, retire it")
    expires_on = diag.get("expires_on")
    _require(bool(expires_on), out, f"{name}: expires_on missing — D-5")
    if not expires_on:
        return
    lapsed = str(expires_on) < as_of.isoformat()
    if lapsed:
        _require(diag.get("retired") is True, out,
                 f"{name}: expired on {expires_on} and not retired — D-5")


def _check_one(diag: dict, out: list[str], as_of: date) -> None:
    name = diag.get("name") or "<unnamed>"
    kind = diag.get("kind")
    _require(kind in KINDS, out, f"{name}: kind must be one of {KINDS} — got {kind!r}")
    _check_expiry(diag, out, name, as_of)
    if kind == "style":
        _check_ratio(diag, out, name)
        _check_controls(diag, out, name)
        _check_anchor_and_invariance(diag, out, name)
    elif kind == "homogenization":
        _check_homogenization(diag, out, name)
    elif kind == "punctuation":
        _check_punctuation(diag, out, name)


def _check_reward_separation(report: dict, out: list[str]) -> None:
    """D-0: no diagnostic may appear inside the reward, and never in the same table as one."""
    names = [d.get("name") for d in report.get("diagnostics", []) if d.get("name")]
    reward_blob = repr(report.get("reward", {}))
    for name in names:
        _require(name not in reward_blob, out,
                 f"{name}: appears inside the reward section — D-0 (a diagnostic never gates, "
                 f"weights or scales the terminal reward)")
    _require(report.get("reward_and_diagnostics_share_table") is not True, out,
             "diagnostics share a table with the reward — D-0")


def check_report(report: dict, as_of: date | None = None) -> list[str]:
    """Return every contract violation in `report`; an empty list means admissible."""
    as_of = as_of or date.today()
    out: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    _check_reward_separation(report, out)
    diagnostics: list[Any] = report.get("diagnostics") or []
    _require(bool(diagnostics), out, "report declares no diagnostics")
    for diag in diagnostics:
        if not isinstance(diag, dict):
            out.append("a diagnostic entry is not an object")
            continue
        _check_one(diag, out, as_of)
    return out
