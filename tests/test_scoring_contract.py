import math
import pytest
import scoring
from measurement_contract import SCORE_WEIGHTS, diagnostic_score, schema_weights


def test_R03_optional_perfect_components_are_normalized(monkeypatch):
    monkeypatch.setattr(scoring, "word_count_score", lambda *a, **k: 1.)
    monkeypatch.setattr(scoring, "repetition_score", lambda *a: 1.)
    monkeypatch.setattr(scoring, "slop_score", lambda *a: 1.)
    result = scoring.calculate_programmatic_scores("fixture", 1, 1, 1, gt_element_count=1, pd_element_count=1)
    assert result.overall == pytest.approx(1.)


@pytest.mark.parametrize("value", [None, True, "NaN", float("nan"), float("inf"), -1, 100, []])
def test_R04_invalid_judge_numbers_rejected(value):
    fields = dict.fromkeys(["beat_elements_score", "beat_execution_score", "must_not_score", "character_score", "bridge_score", "continuity_score"], .5)
    fields["beat_elements_score"] = value
    with pytest.raises(ValueError):
        scoring.normalize_llm_results_to_score(fields, "beat_interpolation")


def test_R04_no_default_credit():
    with pytest.raises(ValueError):
        scoring.normalize_llm_results_to_score({"unrelated": True}, "beat_interpolation")


@pytest.mark.parametrize("schema", sorted(SCORE_WEIGHTS))
def test_each_task_schema_requires_every_field_and_bounds(schema):
    fields = SCORE_WEIGHTS[schema]
    for key in fields:
        missing = dict.fromkeys(fields, 1.)
        del missing[key]
        with pytest.raises(ValueError):
            diagnostic_score(missing, *schema)
    assert diagnostic_score(dict.fromkeys(fields, 1.), *schema) == pytest.approx(1.)
    assert diagnostic_score(dict.fromkeys(fields, 0.), *schema) == 0


@pytest.mark.parametrize("task, subtype", [("bogus", "default"), ("beat_revision", None), ("beat_revision", "bogus"), ("theory_conversion", "flawed")])
def test_no_content_based_subtype_guess(task, subtype):
    with pytest.raises(ValueError):
        diagnostic_score(dict.fromkeys(SCORE_WEIGHTS[("beat_revision", "no_flaw")], 1), task, subtype)


def test_unknown_and_not_applicable_are_not_default_credit():
    fields = dict.fromkeys(SCORE_WEIGHTS[("theory_conversion", "default")], 1.)
    fields["beats_score"] = dict(verdict="unknown", rationale="insufficient source", evidence=[])
    assert diagnostic_score(fields, "theory_conversion") is None
    fields["beats_score"] = dict(verdict="not_applicable", rationale="No beat instruction in brief", evidence=[])
    assert diagnostic_score(fields, "theory_conversion") == pytest.approx(1.)
    fields["beats_score"] = dict(verdict="violated", rationale="ending contradicts source", evidence=[])
    with pytest.raises(ValueError, match="evidence"):
        diagnostic_score(fields, "theory_conversion")


@pytest.mark.parametrize("value", [None, True, "errors", {}, [1], [""]])
def test_malformed_finding_collection_rejected(value):
    fields = dict.fromkeys(SCORE_WEIGHTS[("constrained_continuation", "default")], 1.)
    fields["failed_constraints"] = value
    with pytest.raises(ValueError):
        diagnostic_score(fields, "constrained_continuation")


def test_unvalidated_composite_not_a_default_headline():
    fields = dict.fromkeys(SCORE_WEIGHTS[("theory_conversion", "default")], 1.)
    result = scoring.calculate_final_score("short", 1, (1, 3), fields, "theory_conversion")
    assert result.final_score is None
    assert "unvalidated" in result.to_dict()["diagnostic_status"]


@pytest.mark.parametrize("values", [(-1, 2), (float("nan"), .5), (True, 0), (.4, .59)])
def test_invalid_diagnostic_weights_rejected(values):
    with pytest.raises(ValueError):
        scoring.ScoringWeights(*values)


@pytest.mark.parametrize("extras", [{"elements_found": 1, "elements_total": 2}, {"beats_present": 3, "beats_total": 2}, {"beats_present": 1}, {"beats_total": 3}])
def test_auxiliary_fields_are_subtype_selected_and_count_consistent(extras):
    fields = dict.fromkeys(SCORE_WEIGHTS[("theory_conversion", "default")], .5)
    with pytest.raises(ValueError):
        diagnostic_score({**fields, **extras}, "theory_conversion")
