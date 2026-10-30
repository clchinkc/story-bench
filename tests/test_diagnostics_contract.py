"""Controls for the diagnostic instrument contract checker.

The positive control proves the checker can PASS a compliant report. Each negative control breaks
exactly ONE requirement and must be refused for that requirement — a checker that refuses everything
is as useless as one that refuses nothing, and a negative control that fails for the wrong reason
would let the named rule rot.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from diagnostics_contract import check_report

AS_OF = date(2026, 10, 30)


def compliant_report():
    """A report that satisfies every requirement. Deliberately minimal, not elaborate."""
    return {
        "reward": {"terminal": "binary resolved success"},
        "diagnostics": [
            {
                "name": "slop-ratio-en",
                "kind": "style",
                "numerator_rate": 0.012,
                "denominator_rate": 0.004,
                "baseline": {"name": "human-en-1990s", "fixed_anchor": True},
                "controls": {"positive_ratio": 4.2, "negative_ratio": 0.9},
                "invariance": {"roster_sizes_tested": [17, 3], "byte_identical": True},
                "probe_set": "probes-v1",
                "separation": 0.31,
                "separation_prior": 0.33,
                "expires_on": "2026-12-31",
            },
            {
                "name": "opener-reuse",
                "kind": "homogenization",
                "samples_n": 5,
                "scope": "same_prompt_same_model",
                "probe_set": "probes-v1",
                "separation": 0.22,
                "separation_prior": 0.22,
                "expires_on": "2026-12-31",
            },
            {
                "name": "dash-glyph-ratio",
                "kind": "punctuation",
                "locale_invariance": {"tested": True, "within_declared_band": True},
                "probe_set": "probes-v1",
                "separation": 0.18,
                "separation_prior": 0.20,
                "expires_on": "2026-12-31",
            },
        ],
    }


def test_compliant_report_is_admissible():
    """POSITIVE CONTROL: nothing to report means the checker is not simply refusing everything."""
    assert check_report(compliant_report(), AS_OF) == []


def test_fixed_lexicon_count_without_a_base_rate_is_refused():
    """D-1: a wordlist count measures the list, not the text."""
    report = compliant_report()
    del report["diagnostics"][0]["denominator_rate"]
    violations = check_report(report, AS_OF)
    assert any("denominator_rate" in v and "D-1" in v for v in violations), violations


def test_a_diagnostic_that_fires_on_human_writing_is_refused():
    """D-1: the author's own text is the negative control, and one control passing is a failure."""
    report = compliant_report()
    report["diagnostics"][0]["controls"]["negative_ratio"] = 1.4
    violations = check_report(report, AS_OF)
    assert any("negative control" in v and "D-1" in v for v in violations), violations


def test_roster_relative_anchor_is_refused():
    """D-2: the roster is not a scale."""
    report = compliant_report()
    report["diagnostics"][0]["baseline"] = {"name": "roster-min-max", "fixed_anchor": False}
    violations = check_report(report, AS_OF)
    assert any("FIXED external anchor" in v and "D-2" in v for v in violations), violations


def test_score_that_moves_with_the_roster_is_refused():
    """D-2: the invariance test is the acceptance condition, not a formality."""
    report = compliant_report()
    report["diagnostics"][0]["invariance"]["byte_identical"] = False
    violations = check_report(report, AS_OF)
    assert any("byte-identical" in v and "D-2" in v for v in violations), violations


def test_cross_model_homogenization_is_refused():
    """D-3: this quantity is only meaningful within one prompt, across samples."""
    report = compliant_report()
    report["diagnostics"][1]["scope"] = "across_models"
    violations = check_report(report, AS_OF)
    assert any("same_prompt_same_model" in v and "D-3" in v for v in violations), violations


def test_punctuation_diagnostic_without_a_locale_test_is_refused():
    """D-4: the full-width dash is a different code point and a different keystroke path."""
    report = compliant_report()
    del report["diagnostics"][2]["locale_invariance"]
    violations = check_report(report, AS_OF)
    assert any("locale-invariance" in v and "D-4" in v for v in violations), violations


def test_expired_diagnostic_that_was_not_retired_is_refused():
    """D-5: a tell list is a perishable instrument; an instrument with no expiry is an assumption."""
    report = compliant_report()
    report["diagnostics"][0]["expires_on"] = "2026-01-01"
    violations = check_report(report, AS_OF)
    assert any("not retired" in v and "D-5" in v for v in violations), violations


def test_collapsed_separation_forces_retirement():
    """D-5: below 80% of prior separation the diagnostic must go, not be annotated."""
    report = compliant_report()
    report["diagnostics"][2]["separation"] = 0.05
    report["diagnostics"][2]["separation_prior"] = 0.20
    violations = check_report(report, AS_OF)
    assert any("fell below" in v and "D-5" in v for v in violations), violations


def test_a_diagnostic_inside_the_reward_is_refused():
    """D-0: the one thing nothing else may compensate for."""
    report = compliant_report()
    report["reward"]["style_multiplier"] = {"slop-ratio-en": 0.25}
    violations = check_report(report, AS_OF)
    assert any("inside the reward" in v and "D-0" in v for v in violations), violations


def test_shared_table_between_reward_and_diagnostics_is_refused():
    """D-0: a reader must never mistake a diagnostic for a reward."""
    report = compliant_report()
    report["reward_and_diagnostics_share_table"] = True
    violations = check_report(report, AS_OF)
    assert any("share a table" in v and "D-0" in v for v in violations), violations


@pytest.mark.parametrize("field", ["numerator_rate", "controls", "baseline", "invariance",
                                   "probe_set", "expires_on"])
def test_every_required_style_field_is_load_bearing(field):
    """Mutation control: removing any single required field must be caught, so no rule is decorative."""
    report = deepcopy(compliant_report())
    del report["diagnostics"][0][field]
    assert check_report(report, AS_OF), f"removing {field!r} produced no violation"
