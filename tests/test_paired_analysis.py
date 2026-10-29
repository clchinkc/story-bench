"""W6 paired-analysis contract: estimand, locale weighting, guardrails, gates.

Every behavioural test here carries a deliberately wrong alternative that turns it
red (the two-strike assertions below), so a passing suite means the guardrail was
exercised rather than merely present. Review 2026-10-07 found three of these tests
were NOT load-bearing (direction, the quality lower bound, and the interval's
locale structure); the cases marked REVIEW-FIX below are the replacements, each
verified by mutating the source and confirming the test goes red.
"""
from decimal import Decimal

import pytest

from study_analysis import (
    ManifestError, REQUIRED_MANIFEST_PATHS, analyse, assert_manifest, parse_units,
    validate_manifest)
from study_fixtures import manifest, unit


def units_from(spec, locale="en", quality=None, cost="0"):
    """spec: iterable of (a_resolved, d_resolved) within distinct clusters."""
    return parse_units([unit(f"c{i}", locale=locale, a=a, d=d, quality=quality,
                             cost=cost) for i, (a, d) in enumerate(spec)])


# ------------------------------------------------------------------- preregistration

def test_complete_manifest_preregisters():
    problems = validate_manifest(manifest())
    assert problems == [], problems


@pytest.mark.parametrize("path", REQUIRED_MANIFEST_PATHS,
                         ids=[".".join(p) for p in REQUIRED_MANIFEST_PATHS])
def test_each_required_field_is_load_bearing(path):
    """Remove one required field at a time; the validator must name it.

    This is the deliberately-wrong control: a validator that returned [] for any
    of these would be schema-decoration, not a preregistration gate.
    """
    doc = manifest()
    node = doc
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    problems = validate_manifest(doc)
    assert problems, f"removing {'.'.join(path)} did not fail the gate"
    assert any(".".join(path) in problem for problem in problems), problems


def test_frozen_policy_cannot_be_reweighted():
    problems = validate_manifest(manifest(**{"estimand.locale_weights": "proportional"}))
    assert any("locale_weights" in problem for problem in problems)


def test_frozen_study_needs_sample_size_and_owner_roster():
    doc = manifest(status="frozen", roster_status="candidates_pending_owner_approval")
    problems = validate_manifest(doc)
    assert any("sample_size" in problem for problem in problems)
    assert any("roster_status" in problem for problem in problems)
    doc["sample_size"] = {"clusters_per_locale": 30, "units_per_cluster": 4}
    doc["roster_status"] = "owner_approved"
    assert validate_manifest(doc) == []


def test_synthetic_example_can_never_be_frozen():
    problems = validate_manifest(manifest(kind="SYNTHETIC EXAMPLE", status="frozen",
                                          roster_status="owner_approved",
                                          sample_size={"clusters_per_locale": 2}))
    assert any("synthetic example" in problem for problem in problems)


def test_unassessable_is_not_a_tie_is_frozen():
    problems = validate_manifest(manifest(**{"valid_assessment.unassessable_is_not_a_tie": False}))
    assert any("unassessable_is_not_a_tie" in problem for problem in problems)


def test_assert_manifest_raises_with_the_reason():
    with pytest.raises(ManifestError) as excinfo:
        assert_manifest(manifest(**{"interval.seed": None}))
    assert "interval.seed" in str(excinfo.value)


# ------------------------------------------------------------------------- estimand

def test_paired_difference_direction_and_null():
    report = analyse(units_from([(True, False), (False, True), (True, True)]), manifest())
    assert report["observed_lift"] == pytest.approx(0.0)          # +1, -1, 0
    assert report["units_assessed"] == 3
    assert report["units_assigned"] == 3


def test_direction_is_treatment_minus_control():
    """REVIEW-FIX: the previous version averaged a symmetric set, so it stayed green
    when `_pair_difference` was reversed. These two single-unit cases are directional."""
    assert analyse(units_from([(False, True)]), manifest())["observed_lift"] == \
        pytest.approx(1.0)
    assert analyse(units_from([(True, False)]), manifest())["observed_lift"] == \
        pytest.approx(-1.0)


def test_treatment_only_success_is_positive_lift():
    report = analyse(units_from([(False, True)] * 4), manifest())
    assert report["observed_lift"] == pytest.approx(1.0)


def test_agent_caused_absence_counts_unresolved_not_dropped():
    """A missing agent submission is unresolved; silently dropping it would
    inflate the denominator-omission failure mode the contract forbids."""
    doc = parse_units([unit("c1", a=True, d=None, d_invalid="absent")])
    report = analyse(doc, manifest())
    assert report["units_assessed"] == 1
    assert report["observed_lift"] == pytest.approx(-1.0)


def test_infrastructure_and_grader_failures_leave_the_point_estimate():
    doc = parse_units([unit("c1", a=True, d=None, d_invalid="infrastructure"),
                       unit("c2", a=None, d=True, a_invalid="grader"),
                       unit("c3", a=False, d=True)])
    report = analyse(doc, manifest())
    assert report["units_assessed"] == 1
    assert report["units_assigned"] == 3
    assert report["observed_lift"] == pytest.approx(1.0)
    assert report["missingness"]["worst_case"] < report["missingness"]["best_case"]


def test_equal_locale_weights_stop_a_big_locale_dominating():
    """10 English losses and 1 Chinese win must not average to a negative lift:
    each locale is weighted equally, not each unit."""
    doc = parse_units([unit(f"en{i}", locale="en", a=True, d=False) for i in range(10)]
                      + [unit("zh1", locale="zh-Hant", a=False, d=True)])
    report = analyse(doc, manifest())
    assert report["observed_lift"] == pytest.approx(0.0)         # mean(-1, +1)
    unweighted = (10 * -1 + 1 * 1) / 11
    assert report["observed_lift"] != pytest.approx(unweighted)


def test_interval_is_deterministic_for_a_preregistered_seed():
    doc = units_from([(False, True)] * 6 + [(True, False)] * 4)
    first = analyse(doc, manifest())
    second = analyse(doc, manifest())
    assert (first["lower_bound"], first["upper_bound"]) == \
           (second["lower_bound"], second["upper_bound"])


# -------------------------------------------------------------------------- gates

def test_benefit_met_when_all_three_conditions_hold():
    doc = units_from([(False, True)] * 20, quality="win")
    report = analyse(doc, manifest())
    assert report["benefit_status"] == "met"
    assert report["quality"]["Q"] == pytest.approx(1.0)
    assert report["coverage"] == pytest.approx(1.0)


def test_benefit_not_met_below_the_lift_gate():
    doc = units_from([(False, True)] * 2 + [(True, False)] * 10, quality="win")
    report = analyse(doc, manifest())
    assert report["observed_lift"] < 0.10
    assert report["benefit_status"] == "not_met"


def test_benefit_not_met_when_quality_below_its_gate():
    doc = units_from([(False, True)] * 20, quality="loss")
    report = analyse(doc, manifest())
    assert report["quality"]["Q"] == pytest.approx(0.0)
    assert report["benefit_status"] == "not_met"


def test_benefit_withheld_when_missingness_straddles_the_gate():
    """A large excluded block must suppress the verdict rather than let the
    assessed subset speak for the whole assignment."""
    rows = [unit(f"ok{i}", a=False, d=True) for i in range(20)]
    rows += [unit(f"bad{i}", a=True, d=None, d_invalid="infrastructure")
             for i in range(40)]
    report = analyse(parse_units(rows), manifest())
    assert report["missingness"]["worst_case"] < 0.10 <= report["missingness"]["best_case"]
    assert report["benefit_status"] == "inconclusive_missingness"


def test_coverage_below_threshold_blocks_the_verdict():
    rows = [unit(f"ok{i}", a=False, d=True) for i in range(8)]
    rows += [unit(f"bad{i}", a=True, d=None, d_invalid="provider") for i in range(4)]
    report = analyse(parse_units(rows), manifest())
    assert report["coverage"] < 0.90
    assert report["benefit_status"] != "met"


def test_quality_cannot_assess_is_excluded_not_counted_as_a_tie():
    rows = [unit(f"w{i}", a=False, d=True, quality="win") for i in range(10)]
    rows += [unit(f"c{i}", a=False, d=True, quality="cannot_assess") for i in range(10)]
    report = analyse(parse_units(rows), manifest())
    assert report["quality"]["valid_assessed_pairs"] == 10
    assert report["quality"]["cannot_assess"] == 10
    assert report["quality"]["Q"] == pytest.approx(1.0)


# ------------------------------------------------------------------------ reporting

def test_all_attempt_cost_and_breaches_are_reported_across_every_arm():
    rows = [unit("c1", a=True, d=False, cost="1.50", breach=True)]
    report = analyse(parse_units(rows), manifest())
    assert Decimal(report["all_attempt_cost_usd"]) == Decimal("6.00")   # 4 arms
    assert report["authority_breaches"] == 4


def test_infrastructure_failure_is_never_a_task_negative_label():
    rows = [unit("c1", a=True, d=None, d_invalid="infrastructure", quality="loss")]
    report = analyse(parse_units(rows), manifest())
    assert report["units_assessed"] == 0
    assert report["observed_lift"] is None
    assert report["benefit_status"] == "inconclusive"


def test_report_carries_its_own_limits():
    report = analyse(units_from([(False, True)]), manifest())
    assert any("not proof" in limit for limit in report["limits"])
    assert any("completes the study" in limit for limit in report["limits"])


# ---------------------------------------------- REVIEW-FIX cases (see module docstring)

def test_interval_respects_locale_structure():
    """REVIEW-FIX. Ten English wins and one Chinese loss: each locale is weighted
    equally, and with one cluster in the Chinese locale the interval is degenerate at
    zero. Flattening the bootstrap into one pooled pseudo-locale would return a
    narrow interval around +0.8 and this test would go red."""
    rows = [unit(f"en{i}", locale="en", a=False, d=True) for i in range(10)]
    rows.append(unit("zh0", locale="zh-Hant", a=True, d=False))
    report = analyse(parse_units(rows), manifest())
    assert report["observed_lift"] == pytest.approx(0.0)
    assert report["lower_bound"] == pytest.approx(0.0)
    assert report["upper_bound"] == pytest.approx(0.0)


def test_quality_interval_is_a_bound_for_the_reported_Q():
    """REVIEW-FIX. Q is pooled by contract, so its interval must be pooled too. The
    earlier locale-weighted interval returned a lower bound of 0.50 for a
    one-win-of-101 sample, i.e. a bound above the point estimate."""
    rows = [unit("w0", locale="en", a=False, d=True, quality="win")]
    rows += [unit(f"l{i}", locale="zh-Hant", a=False, d=True, quality="loss")
             for i in range(100)]
    quality = analyse(parse_units(rows), manifest())["quality"]
    assert quality["Q"] == pytest.approx(1 / 101)
    assert quality["Q_lower_bound"] <= quality["Q"] + 1e-12


def test_quality_safeguard_does_not_block_a_genuine_high_win_study():
    """REVIEW-FIX. The previous estimator blocked a 98%-win study because the Q
    interval was computed with a different weighting from the Q point estimate."""
    rows = [unit(f"w{i}", locale="en", a=False, d=True, quality="win")
            for i in range(100)]
    rows.append(unit("z0", locale="zh-Hant", a=False, d=True, quality="loss"))
    rows.append(unit("z1", locale="zh-Hans", a=False, d=True, quality="loss"))
    report = analyse(parse_units(rows), manifest())
    assert report["quality"]["Q"] == pytest.approx(100 / 102)
    assert report["quality"]["Q_lower_bound"] > 0.45
    assert report["benefit_status"] == "met"


def test_q_lower_bound_alone_can_block_the_verdict():
    """REVIEW-FIX. Q clears its observed gate but its lower bound does not, so the
    verdict must be withheld. Dropping `q_lower > q_lower_gate` from the gate turns
    this test red."""
    rows = [unit(f"w{i}", a=False, d=True, quality="win") for i in range(20)]
    rows += [unit(f"l{i}", a=False, d=True, quality="loss") for i in range(20)]
    report = analyse(parse_units(rows), manifest())
    assert report["quality"]["Q"] == pytest.approx(0.5)
    assert report["quality"]["Q_lower_bound"] < 0.45
    assert report["benefit_status"] == "not_met"


def test_analyse_refuses_a_manifest_that_is_not_a_preregistration():
    """REVIEW-FIX. `analyse()` previously trusted its caller's manifest."""
    with pytest.raises(ManifestError) as excinfo:
        analyse(units_from([(False, True)]),
                manifest(**{"estimand.locale_weights": "proportional"}))
    assert "locale_weights" in str(excinfo.value)


def test_invalid_kind_with_a_resolved_outcome_is_rejected():
    """REVIEW-FIX. `invalid_kind="absent"` with `resolved=True` scored as a success."""
    with pytest.raises(ManifestError) as excinfo:
        parse_units([unit("c1", a=True, d=True, d_invalid="absent")])
    assert "invalid_kind" in str(excinfo.value)


def test_report_includes_resolved_rates_status_breakdown_and_per_model():
    """Contract § 8 requires resolved counts/rates, per-model effects and the
    invalid-kind breakdown, not only the aggregate lift."""
    rows = [unit(f"a{i}", a=True, d=True, model="vendor/model-one") for i in range(4)]
    rows += [unit(f"b{i}", a=False, d=True, model="vendor/model-two") for i in range(4)]
    rows.append(unit("bad", a=True, d=None, d_invalid="provider",
                     model="vendor/model-two"))
    report = analyse(parse_units(rows), manifest())
    assert report["resolved"]["control"]["resolved"] == 5
    assert report["resolved"]["control"]["rate"] == pytest.approx(5 / 9)
    assert report["resolved"]["treatment"]["resolved"] == 8
    assert report["status_breakdown"]["D"]["provider"] == 1
    assert set(report["per_model"]) == {"vendor/model-one", "vendor/model-two"}
    assert report["per_model"]["vendor/model-one"]["observed_lift"] == pytest.approx(0.0)
    assert report["per_model"]["vendor/model-two"]["observed_lift"] == pytest.approx(1.0)
