"""The skip instrument: a run that has learned should be allowed to stop.

The training metrics cannot answer this — `frac_mixed` is computed after filtering, so it describes what
survived. This reads the two places where the real counts live and applies the stop rule, and the tests
build a synthetic run so the behaviour is pinned without depending on anyone's run directory.
"""

import json
from pathlib import Path

import pytest

from story_rl.skip_rate import build, first_stop_step, summarise


def make_run(root: Path, steps: list[tuple[int, int]]) -> Path:
    """A run directory with `(sampled, trained)` per step, in the two files the instrument reads."""
    (root).mkdir(parents=True, exist_ok=True)
    metrics = []
    for step, (sampled, trained) in enumerate(steps):
        iteration = root / f"iteration_{step:06d}"
        iteration.mkdir()
        (iteration / "train_rollout_summaries.jsonl").write_text(
            "".join(json.dumps({"traj_idx": index}) + "\n" for index in range(sampled))
        )
        metrics.append(json.dumps({"step": step, "env/all/total_episodes": trained}))
    (root / "metrics.jsonl").write_text("\n".join(metrics) + "\n")
    return root


def test_the_curve_joins_what_was_sampled_against_what_was_trained(tmp_path):
    rows = build(make_run(tmp_path / "run", [(128, 128), (128, 64)]))

    assert [row["sampled"] for row in rows] == [128, 128]
    assert [row["trained"] for row in rows] == [128, 64]
    assert [row["skip"] for row in rows] == [0.0, 0.5]


def test_a_healthy_run_never_trips_the_stop_rule(tmp_path):
    rows = build(make_run(tmp_path / "run", [(128, 120)] * 5))

    assert first_stop_step(rows) is None


def test_three_consecutive_saturated_steps_trip_the_rule_at_the_first_of_them(tmp_path):
    rows = build(make_run(tmp_path / "run", [(128, 128), (128, 128), (128, 16), (128, 16), (128, 16)]))

    assert first_stop_step(rows) == 2


def test_two_saturated_steps_do_not_trip_it(tmp_path):
    rows = build(make_run(tmp_path / "run", [(128, 16), (128, 16), (128, 128)]))

    assert first_stop_step(rows) is None


def test_the_summary_reports_the_overall_skip_and_the_stop_step(tmp_path):
    run = make_run(tmp_path / "run", [(128, 128), (128, 16), (128, 16), (128, 16)])
    result = summarise(run)

    assert result["sampled"] == 512
    assert result["trained"] == 176
    assert result["skip"] == pytest.approx(1 - 176 / 512, abs=1e-3)
    assert result["should_have_stopped_at"] == 1


def test_a_directory_with_no_run_files_is_refused_rather_than_reported_as_empty(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="no iteration summaries"):
        summarise(tmp_path / "empty")
