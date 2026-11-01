"""The scoring instrument: same input, same number, and a paired comparison that ignores ties.

Ties matter here more than they look. A passage already solved at baseline cannot improve, so counting it
as a success would inflate every result; the sign test is over *discordant* pairs only, and its floor is
what "with eight passages you can only detect a near-uniform result" means numerically.
"""

import json
from pathlib import Path

import pytest

from story_rl.evaluate import compare, per_task_rewards, read_generations, summarise
from story_rl.stats import cluster_bootstrap, contrast, sign_test, wilson


def test_re_scoring_the_same_text_gives_the_same_number():
    """The instrument must be deterministic or no two runs are comparable."""
    from repair_task import from_row

    task = from_row({
        "task_id": "t1",
        "passage": "Mara had grey eyes. The lantern was lit.",
        "canon": [{"entity": "Mara", "attribute": "eye_colour", "value": "grey"}],
        "broken_entity": "Mara",
        "broken_attribute": "eye_colour",
        "true_value": "grey",
        "wrong_value": "brown",
    })
    text = "Mara turned, her grey eyes catching the lamplight."

    first = per_task_rewards([{"task_id": "t1", "raw": text}], {"t1": task}, "prose")
    again = per_task_rewards([{"task_id": "t1", "raw": text}], {"t1": task}, "prose")

    assert first == again == {"t1": [1.0]}


def test_generations_are_grouped_by_task(tmp_path):
    path = tmp_path / "gens.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"task_id": "a", "raw": "x"}, {"task_id": "b", "raw": "y"}, {"task_id": "a", "raw": "z"},
    ]))

    assert len(read_generations(path)) == 3


def test_a_generation_naming_an_unknown_task_is_refused():
    from repair_task import from_row

    task = from_row({
        "task_id": "t1",
        "passage": "p p.",
        "canon": [{"entity": "e", "attribute": "a", "value": "v"}],
        "broken_entity": "e",
        "broken_attribute": "a",
        "true_value": "v",
        "wrong_value": "w",
    })

    with pytest.raises(ValueError, match="unknown task"):
        per_task_rewards([{"task_id": "nope", "raw": "v"}], {"t1": task}, "prose")


def test_the_summary_reports_rollout_and_cluster_intervals_separately():
    by_task = {"a": [1.0, 0.0], "b": [1.0, 1.0], "c": [0.0, 0.0], "d": [1.0, 0.0]}
    result = summarise(by_task)

    assert result["tasks"] == 4 and result["generations"] == 8
    assert result["pass_rate"] == pytest.approx(0.5, abs=1e-4)
    assert result["pass_rate_ci"] != result["pass_rate_cluster_ci"]


def test_a_paired_comparison_excludes_ties_from_the_test():
    baseline = {"a": [0.0], "b": [1.0], "c": [0.0]}
    treatment = {"a": [1.0], "b": [1.0], "c": [0.5]}

    result = compare(baseline, treatment)

    assert (result["improved"], result["unchanged"], result["worsened"]) == (2, 1, 0)
    assert result["p"] == sign_test(2, 0) == 0.5


def test_a_paired_comparison_with_no_shared_task_is_refused():
    with pytest.raises(ValueError, match="share no tasks"):
        compare({"a": [1.0]}, {"b": [1.0]})


def test_the_sign_test_reaches_significance_only_when_nearly_everything_moves():
    assert sign_test(6, 0) == pytest.approx(0.0312, abs=1e-4)
    assert sign_test(7, 0) == pytest.approx(0.0156, abs=1e-4)
    assert sign_test(5, 1) == pytest.approx(0.2188, abs=1e-4)
    assert sign_test(0, 0) == 1.0


def test_the_wilson_interval_covers_the_previously_measured_corpus_rate():
    low, high = wilson(53, 80)

    assert low < 0.662 < high
    assert 0.0 <= low < high <= 1.0


def test_the_cluster_interval_is_wider_than_the_rollout_interval_on_uneven_tasks():
    """The reason the cluster interval is the honest one: four draws of one passage are not four facts."""
    by_task = {"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0, 0.0], "c": [1.0, 0.0, 1.0, 0.0]}
    low, high = cluster_bootstrap(by_task)

    assert low <= 0.5 <= high
    assert high - low > 0.2


def test_contrast_falls_as_a_policy_solves_the_task():
    """`pass = (1-e)**breaks` multiplies the error rate, so a solved task cannot be rescued by breaks."""
    assert contrast(0.662, 1, 16) > 0.6
    assert contrast(0.0, 1, 16) < 0.01
    assert contrast(0.0, 6, 16) < 0.02


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_contrast_refuses_an_impossible_error_rate(rate):
    assert contrast(rate, 1, 16) is None
