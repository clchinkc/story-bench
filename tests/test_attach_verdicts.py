"""The join that decides whether D and A are the arms the judge actually scored.

This is the highest-consequence step in the pipeline and the easiest to get silently wrong: a
mis-joined verdict still produces a number, and the interval still prints. The controls below pin
the four ways it can go wrong -- wrong task, swapped arms, a failed reading counted as a score, and
a rejected reading used as if it were the verdict of record.

The task-identity control is a regression test for a real defect: with only `task_family` in the
unit record, the clean1 run's 176 units collapsed to 45 keys, because five `beat_interpolation_*`
tasks share one family and one sample index. Those units now differ only in their attempt ids, so a
join that used the family would have attached task 003's verdict to tasks 001, 002, 004 and 005.
"""
import importlib.util
import json
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "attach_verdicts", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "attach_verdicts.py")
av = importlib.util.module_from_spec(_spec)
sys.modules["attach_verdicts"] = av
_spec.loader.exec_module(av)


def unit(task_id, a_attempt, d_attempt, family="beat_interpolation"):
    return {
        "cluster_id": "en-c00", "model": "anthropic/claude-opus-5.5",
        "model_snapshot": "anthropic/claude-opus-5.5-20260921", "task_id": task_id,
        "locale": "en", "task_family": family, "sample_id": "0",
        "arms": {
            "A": {"attempt_id": a_attempt, "resolved": None, "invalid_kind": None,
                  "authority_breach": False, "cost_usd": "0.01", "latency_ms": 3},
            "D": {"attempt_id": d_attempt, "resolved": None, "invalid_kind": None,
                  "authority_breach": False, "cost_usd": "0.01", "latency_ms": 3},
        },
    }


def verdict(root, judgement_id, control, treatment, a_resolved, d_resolved,
            failure=None, reask=None):
    directory = pathlib.Path(root) / judgement_id
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "judgement_id": judgement_id, "control_attempt_id": control,
        "treatment_attempt_id": treatment, "failure": failure,
        "analysis": {
            "A": {"resolved": a_resolved, "quality": None, "invalid_kind": None},
            "D": {"resolved": d_resolved, "quality": "win" if d_resolved else "loss",
                  "invalid_kind": None},
        },
    }
    if reask is not None:
        document["corrective_reask"] = reask
    (directory / "verdict.json").write_text(json.dumps(document), encoding="utf-8")
    return directory


def test_a_verdict_reaches_the_arm_its_attempt_id_names(tmp_path):
    """control is A and treatment is D; a swap here would flip the sign of the whole study."""
    verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", False, True)
    units = [unit("t1", "m:t1:A:aaa", "m:t1:D:ddd")]
    index, failed, refused = av._verdict_index(tmp_path)
    stats, unmatched = av._attach(units, index)

    assert units[0]["arms"]["A"]["resolved"] is False
    assert units[0]["arms"]["D"]["resolved"] is True
    assert units[0]["arms"]["D"]["quality"] == "win"
    assert not failed and not refused and not unmatched
    assert stats["unit_fully_scored"] == 1


def test_five_tasks_in_one_family_do_not_share_a_verdict(tmp_path):
    """The regression control: same family, same sample, different tasks, different attempt ids."""
    units = [unit(f"beat_interpolation_00{n}", f"m:bi{n}:A:a{n}", f"m:bi{n}:D:d{n}")
             for n in range(1, 6)]
    for n in range(1, 6):
        # task 003 alone has D resolved true; a family-keyed join would smear it across all five
        verdict(tmp_path, f"j{n}", f"m:bi{n}:A:a{n}", f"m:bi{n}:D:d{n}",
                False, n == 3)
    index, _, _ = av._verdict_index(tmp_path)
    av._attach(units, index)

    resolved_d = {u["task_id"]: u["arms"]["D"]["resolved"] for u in units}
    assert resolved_d == {"beat_interpolation_001": False, "beat_interpolation_002": False,
                          "beat_interpolation_003": True, "beat_interpolation_004": False,
                          "beat_interpolation_005": False}


def test_a_failed_verdict_scores_nothing_and_is_excluded_from_the_estimate(tmp_path):
    """A reply that did not parse is a grader failure, never `resolved: false` for the writer."""
    verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", None, None,
            failure="better must name submission 2, the only resolved submission")
    units = [unit("t1", "m:t1:A:aaa", "m:t1:D:ddd")]
    index, failed, refused = av._verdict_index(tmp_path)
    stats, _ = av._attach(units, index)

    assert len(failed) == 1 and not refused and not index
    for arm in ("A", "D"):
        assert units[0]["arms"][arm]["resolved"] is None
        assert units[0]["arms"][arm]["invalid_kind"] == "grader"
    assert stats["unit_unscored"] == 1


def test_an_arm_with_no_verdict_is_unscored_not_a_failure(tmp_path):
    verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", True, False)
    units = [unit("t1", "m:t1:A:aaa", "m:t1:D:ddd"),
             unit("t2", "m:t2:A:unnjudged", "m:t2:D:unnjudged")]
    index, _, _ = av._verdict_index(tmp_path)
    stats, _ = av._attach(units, index)

    assert units[1]["arms"]["A"]["invalid_kind"] == "grader"
    assert units[1]["arms"]["A"]["resolved"] is None
    assert stats["unit_partially_scored"] + stats["unit_unscored"] == 1


def test_the_rejected_reading_is_never_attached(tmp_path):
    """`verdict.rejected.json` is the answer the judge was told to redo; only `verdict.json` counts."""
    directory = verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", True, True,
                        reask={"superseded_attempt_id": "old", "also_conflicted": False})
    rejected = dict(json.loads((directory / "verdict.json").read_text(encoding="utf-8")))
    rejected["analysis"] = {"A": {"resolved": False, "quality": None, "invalid_kind": None},
                            "D": {"resolved": False, "quality": None, "invalid_kind": None}}
    rejected["failure"] = "contradicted itself"
    (directory / "verdict.rejected.json").write_text(json.dumps(rejected), encoding="utf-8")

    index, failed, _ = av._verdict_index(tmp_path)
    assert len(index) == 2, "only the verdict of record is read"
    assert not failed
    assert index["m:t1:D:ddd"]["resolved"] is True, "the accepted reading wins, not the rejected one"
    assert index["m:t1:D:ddd"]["corrective_reask"]["superseded_attempt_id"] == "old"


def test_a_second_claim_on_one_attempt_id_is_refused(tmp_path):
    """Two verdicts for one generation means a replay against another rubric; never pick one."""
    verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", True, False)
    verdict(tmp_path, "j2", "m:t1:A:aaa", "m:t1:D:ddd", False, True)
    index, _, refused = av._verdict_index(tmp_path)

    assert len(refused) == 2, "the second file's two claims are both refused"
    assert index["m:t1:A:aaa"]["judgement_id"] == "j1"
    assert index["m:t1:D:ddd"]["judgement_id"] == "j1"


def test_a_verdict_with_identical_arms_is_refused(tmp_path):
    """One generation cannot be both the control and the treatment."""
    verdict(tmp_path, "j1", "m:t1:A:same", "m:t1:A:same", True, False)
    index, failed, refused = av._verdict_index(tmp_path)

    assert not index and len(refused) == 1
    assert "unusable" in refused[0]["reason"]


def test_cli_writes_every_unit_and_refuses_an_unmatched_verdict(tmp_path, capsys):
    units = tmp_path / "units.jsonl"
    units.write_text(json.dumps(unit("t1", "m:t1:A:aaa", "m:t1:D:ddd")) + "\n", encoding="utf-8")
    judgements = tmp_path / "judgements"
    verdict(judgements, "j1", "m:t1:A:aaa", "m:t1:D:ddd", True, False)
    verdict(judgements, "j2", "m:zzz:A:xxx", "m:zzz:D:yyy", True, True)  # matches no unit

    out = tmp_path / "out" / "units-scored.jsonl"
    code = av.main(["--units", str(units), "--judgements", str(judgements), "--out", str(out)])
    assert code == 1, "an unmatched verdict must refuse by default"
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1

    code = av.main(["--units", str(units), "--judgements", str(judgements), "--out", str(out),
                    "--allow-unmatched"])
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8").strip())
    assert written["arms"]["A"]["resolved"] is True and written["arms"]["D"]["resolved"] is False
    assert "attempt ids matching nothing: 2" in capsys.readouterr().out


def test_an_unresolved_verdict_is_a_measurement_not_a_grader_failure(tmp_path):
    """`resolved: None` from a judge that answered is "did not fully resolve", NOT "never measured".

    Marking it `grader` would exclude the pair from the point estimate, so a task nobody resolved
    would vanish from the denominator and the lift would read higher than the judge actually said.
    Measured consequence: 47 of 176 units on the 2026-11-09 run are `resolved: None` on both arms.
    """
    verdict(tmp_path, "j1", "m:t1:A:aaa", "m:t1:D:ddd", None, None)  # answered, nothing resolved
    units = [unit("t1", "m:t1:A:aaa", "m:t1:D:ddd")]
    index, failed, refused = av._verdict_index(tmp_path)
    stats, unmatched = av._attach(units, index)

    assert not failed and not refused and not unmatched
    for arm in ("A", "D"):
        assert units[0]["arms"][arm]["resolved"] is None
        assert units[0]["arms"][arm]["invalid_kind"] is None, "not a grader failure — a verdict"
    assert stats["unit_fully_scored"] == 1, "both arms were measured, so the unit is complete"
    assert stats["arm_no_verdict"] == 0
