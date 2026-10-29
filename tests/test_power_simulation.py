"""W6 power/precision simulation contract.

Thresholds below are derived from the recorded seeded run, not guessed: power
0.18 -> 0.37 -> 0.78 -> 1.00 and mean 95% width 0.485 -> 0.308 -> 0.200 -> 0.141
across (6,2) (12,3) (24,4) (40,5) at baseline 0.55, true lift 0.15, cluster sd
0.10, 60 trials, seed 11.
"""
from decimal import Decimal

import pytest

from study_analysis import (LOCALES, PowerPoint, analyse, power_curve,
                            power_simulation, render_power_curve, simulate_units)
from study_fixtures import manifest, unit

SETTINGS = dict(baseline_rate=0.55, true_lift=0.15, cluster_sd=0.10,
                unassessable_rate=0.05, seed=11, trials=60, resamples=200)


def test_simulation_is_deterministic_for_a_preregistered_seed():
    first = power_simulation(clusters_per_locale=12, units_per_cluster=3, **SETTINGS)
    second = power_simulation(clusters_per_locale=12, units_per_cluster=3, **SETTINGS)
    assert first == second


def test_a_different_seed_moves_the_estimate():
    other = dict(SETTINGS, seed=99)
    assert power_simulation(clusters_per_locale=12, units_per_cluster=3, **SETTINGS) != \
           power_simulation(clusters_per_locale=12, units_per_cluster=3, **other)


def test_power_rises_with_sample_size():
    powers = [power_simulation(clusters_per_locale=c, units_per_cluster=u,
                               **SETTINGS)["power"]
              for c, u in ((6, 2), (12, 3), (24, 4), (40, 5))]
    assert powers[0] < powers[1] < powers[2] < powers[3]
    assert powers[3] >= 0.80


def test_precision_improves_with_sample_size():
    widths = [power_simulation(clusters_per_locale=c, units_per_cluster=u,
                               **SETTINGS)["mean_interval_width"]
              for c, u in ((6, 2), (12, 3), (24, 4), (40, 5))]
    assert widths[0] > widths[1] > widths[2] > widths[3]


def test_no_true_effect_yields_negligible_power():
    weak = dict(SETTINGS, true_lift=0.0)
    assert power_simulation(clusters_per_locale=12, units_per_cluster=3, **weak)["power"] <= 0.10
    assert power_simulation(clusters_per_locale=40, units_per_cluster=5, **weak)["power"] <= 0.10


def test_simulation_declares_every_gate_it_does_not_cover():
    """REVIEW-FIX: the omissions were previously stated only in a docstring."""
    result = power_simulation(clusters_per_locale=6, units_per_cluster=2, **SETTINGS)
    assert result["covers"] == {"resolved_gate": True, "quality_safeguard": False,
                               "coverage_gate": False, "missingness_suppression": False}
    assert result["trials"] == 60


def test_missingness_rate_actually_affects_the_simulation():
    """REVIEW-FIX. Making `unassessable_rate` a no-op used to leave the suite green."""
    low = power_simulation(clusters_per_locale=12, units_per_cluster=3,
                           **dict(SETTINGS, unassessable_rate=0.0))
    high = power_simulation(clusters_per_locale=12, units_per_cluster=3,
                            **dict(SETTINGS, unassessable_rate=0.60))
    assert low["power"] > high["power"]


def test_simulated_units_never_invent_quality_outcomes():
    units = simulate_units(baseline_rate=0.55, true_lift=0.10, cluster_sd=0.10,
                           unassessable_rate=0.0, clusters_per_locale=2,
                           units_per_cluster=3, seed=1)
    assert units
    assert {u.observation(arm).quality for u in units
            for arm in ("A", "B", "C", "D")} == {None}


def test_quality_is_populated_only_when_requested():
    units = simulate_units(baseline_rate=0.55, true_lift=0.10, cluster_sd=0.10,
                           unassessable_rate=0.0, clusters_per_locale=1,
                           units_per_cluster=8, seed=1, quality_win_rate=0.7)
    qualities = {u.observation("D").quality for u in units}
    assert qualities <= {"win", "tie", "loss"}
    assert "win" in qualities


def test_simulated_units_cover_every_locale_and_are_analysable():
    units = simulate_units(baseline_rate=0.55, true_lift=0.20, cluster_sd=0.05,
                           unassessable_rate=0.0, clusters_per_locale=4,
                           units_per_cluster=2, seed=3, quality_win_rate=0.8,
                           cost_per_unit="0.001", latency_ms=5)
    assert {u.locale for u in units} == set(LOCALES)
    report = analyse(units, manifest())
    assert report["units_assigned"] == len(units)
    assert Decimal(report["all_attempt_cost_usd"]) > 0


def test_power_curve_covers_the_preregistered_grid():
    grid_manifest = manifest(**{"interval.seed": 11})
    grid_manifest["power_simulation"] = {"baseline_rate": 0.55, "true_lift": 0.15,
                                         "cluster_sd": 0.10, "unassessable_rate": 0.05}
    points = power_curve(grid_manifest, [(6, 2), (12, 3)], trials=30, resamples=100)
    assert [type(p) for p in points] == [PowerPoint, PowerPoint]
    assert [(p.clusters_per_locale, p.units_per_cluster) for p in points] == [(6, 2), (12, 3)]
    assert points[0].power < points[1].power
    assert power_curve(grid_manifest, [(6, 2)], trials=30, resamples=100) == \
           power_curve(grid_manifest, [(6, 2)], trials=30, resamples=100)


def test_rendered_curve_carries_the_quality_caveat():
    points = [PowerPoint(6, 2, 0.18, 0.485, 0.127)]
    text = render_power_curve(points)
    assert "does NOT cover the quality" in text
    assert "6" in text and "0.18" in text
