"""The judge phase must refuse, not improvise: no rubric, no score; no parse, no score.

The four controls the brief names are here, plus the two refusals that would otherwise be prose:

  * an unparseable judgement is a FAILED judgement, mapped to `invalid_kind: grader` — never a
    default `resolved=False`, which would enter the point estimate as a task failure;
  * a task with no `evaluation_criteria` is refused BY ID rather than given a house rubric;
  * the prompt carries the task's criteria and no other criterion line;
  * `--dry-run` builds every body and touches no network and no ledger slot.

Everything below runs offline: `conftest` already makes `socket.connect` raise, and the route
fixture is a SYNTHETIC profile, which can never admit a real request.
"""
import importlib.util
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

import provider_attempts as pa
import provider_prices as pp
import spend_ledger as sl
from measurement_contract import PROTOCOL, digest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Loaded by path, NOT by putting scripts/ on sys.path: a module named like the package would shadow
# `src/story_rl/` for every test collected afterwards (the reason `test_dispatch_runner` does this).
_spec = importlib.util.spec_from_file_location("judge_units", ROOT / "scripts" / "judge_units.py")
ju = importlib.util.module_from_spec(_spec)
sys.modules["judge_units"] = ju
_spec.loader.exec_module(ju)

JUDGE = "fixture/judge-20261104"
STUDY = ROOT / "studies" / "revision-v1.yaml"


def task(**overrides):
    """A theory_conversion task with the fields the writer's own prompt builder needs."""
    doc = dict(
        task_id="theory_conversion_900", task_type="theory_conversion", subtype="default",
        from_theory="three-act", to_theory="kishotenketsu",
        original_segment=dict(content="A short original.", beats=["setup", "turn"]),
        target_requirements=dict(word_count=[100, 200], beats=["intro", "twist", "close"],
                                 preserve=["the sister"], tone="dry"),
        evaluation_criteria=[
            {"criterion": "beats_present", "description": "all three target beats are present"},
            {"criterion": "preservation", "description": "the sister survives the rewrite"},
        ])
    doc.update(overrides)
    return doc


def manifest(task_doc=None, *, judge_snapshot=JUDGE, model_snapshot=None):
    doc = task_doc or task()
    model = model_snapshot or judge_snapshot
    rows = [dict(model="fixture/judge", model_snapshot=model, task_id=doc["task_id"], task=doc,
                 task_hash=digest(doc), sample=0, condition=condition, story_id="story-1",
                 judge_snapshot=judge_snapshot, prompt_version="fixture-v1")
            for condition in ("A", "D")]
    return dict(protocol=PROTOCOL, design="pilot_run", judge_designated=True,
                judge_snapshot=judge_snapshot, prompt_version="fixture-v1", assignments=rows)


def verdict(content, criteria, *, better="1", resolved=(True, False), extra=None):
    """A well-formed reply, with whatever the caller wants to break."""
    by_slot = {slot: [dict(criterion=c.name, verdict="met", basis="it is there")
                      for c in criteria] for slot in ("1", "2")}
    declared = {slot: resolved[index] for index, slot in enumerate(("1", "2"))}
    for slot, value in declared.items():
        if value is False:
            by_slot[slot][0]["verdict"] = "not_met"
    body = dict(schema=ju.VERDICT_SCHEMA_ID, submissions={
        slot: dict(criteria=by_slot[slot], resolved=declared[slot]) for slot in ("1", "2")},
        better=better)
    if extra:
        body.update(extra)
    return content if isinstance(content, str) else json.dumps(body)


def synthetic_base(root, *, input_bound=200000):
    """A retained base the offline rehearsal can prepare against; never admits a real send."""
    model_id, canonical, tag = "fixture/judge", JUDGE, "fixture"
    route = pp.Route("openrouter", "https://openrouter.ai/api/v1", "/chat/completions",
                     model_id, canonical, tag, "default", "USD")
    now = datetime.now(timezone.utc)
    start = (now - timedelta(seconds=1)).isoformat()
    end = (now + timedelta(hours=1)).isoformat()
    catalog = pp.json_bytes({"data": [{"id": model_id, "canonical_slug": canonical}]})
    endpoints = pp.json_bytes({"data": {"id": model_id, "endpoints": [dict(
        tag=tag, provider_name="Fixture", model_id=model_id, context_length=300000,
        max_prompt_tokens=290000, max_completion_tokens=8000,
        supported_parameters=["max_tokens", "temperature"],
        pricing=dict(prompt="0.000001", completion="0.000002"))]}})
    profile = dict(kind=pp.SYNTHETIC_KIND, catalog_sha256=pp.sha(catalog),
                   endpoints_sha256=pp.sha(endpoints), model_id=model_id,
                   canonical_model_slug=canonical, endpoint_tag=tag, input_bound=input_bound,
                   output_bound=8000, reasoning_bound=0, cache_read_bound=0, cache_write_bound=0,
                   structural_zero=["reasoning_tokens", "cache_read_tokens", "cache_write_tokens",
                                    "request"],
                   no_incremental_infrastructure=True, no_metered_agent=True)
    slug = model_id.replace("/", "_")
    (root / "records").mkdir(parents=True)
    (root / "verified").mkdir()
    (root / "openrouter-models.json").write_bytes(catalog)
    (root / f"endpoints-{slug}.json").write_bytes(endpoints)
    (root / "verified" / f"{slug}.profile.json").write_bytes(pp.json_bytes(profile))
    identity = {key: value for key, value in (("provider", route.provider),
                                              ("base_url", route.base_url), ("path", route.path),
                                              ("model_id", model_id),
                                              ("canonical_model_slug", canonical),
                                              ("endpoint_tag", tag),
                                              ("service_tier", route.service_tier),
                                              ("currency", route.currency),
                                              ("excluded_variants", []))}
    record = dict(route_identity=dict(identity, observed_at=start, expires_at=end),
                  admission_gates={kind: {"status": "satisfied"} for kind in pa.GATES})
    (root / "records" / f"{slug}.json").write_text(json.dumps(record))
    return root


def ledger_at(root):
    ledger = sl.Ledger.create(root / "ledger.sqlite3", program_id="judge-fixture",
                              store_id="judge-fixture",
                              authorization=sl.Authorization("fixture-create", "a" * 64,
                                                             "OFFLINE FIXTURE"))
    ledger.reconcile_external("fixture-host", charge_id="hosted_session",
                              receipt_id="fixture-host", actual_usd="0",
                              evidence_sha256="b" * 64,
                              authorization=sl.Authorization("fixture-host", "c" * 64,
                                                             "OFFLINE FIXTURE"))
    return ledger


def collected(condition, doc):
    return ju.Collected(attempt_id=f"gen-{condition}", model_snapshot=JUDGE,
                        task_id=doc["task_id"], sample=0, condition=condition,
                        story_id="story-1", task=doc, text=f"submission {condition} prose",
                        cost_usd="0.010000")


def judgement(doc=None):
    doc = doc or task()
    row = next(r for r in manifest(doc)["assignments"] if r["condition"] == "D")
    return ju.Judgement(judgement_id=ju.judgement_id_for(collected("D", doc)),
                        attempt_id="judge-fixture-1", row=row, task=doc,
                        criteria=ju.criteria_of(doc), control=collected("A", doc),
                        treatment=collected("D", doc))


def prepare_for(tmp_path, raw, base):
    def prepare(unit):
        return ju.prepare_judgement(unit, base=base, manifest_raw=raw,
                                    evidence_root=tmp_path / "evidence", judge_snapshot=JUDGE)
    return prepare


# ── 1. an unparseable judgement is a failure, never a default score ─────────────────────────────


def test_unparseable_judgement_fails_instead_of_defaulting():
    """A judge that answers in prose is a broken instrument, not a low-scoring writer."""
    criteria = ju.criteria_of(task())
    for broken in ("Submission 1 is clearly stronger.", "{}", "",
                   json.dumps({"schema": ju.VERDICT_SCHEMA_ID})):
        with pytest.raises(ju.VerdictError):
            ju.parse_verdict(broken, criteria)
    failed = ju.failed_analysis("not one JSON object")
    for arm in ("A", "D"):
        assert failed[arm]["resolved"] is None, "a failed judgement must not become resolved=False"
        assert failed[arm]["invalid_kind"] == "grader"
        assert failed[arm]["quality"] == "cannot_assess"


def test_a_self_contradicting_judgement_is_a_failure():
    """`resolved: true` with a `not_met` criterion is incoherent, not a rounding decision."""
    criteria = ju.criteria_of(task())
    body = json.loads(verdict(None, criteria, resolved=(True, False)))
    body["submissions"]["2"]["resolved"] = True
    with pytest.raises(ju.VerdictError, match="contradicts"):
        ju.parse_verdict(json.dumps(body), criteria)


def test_an_undeclared_criterion_is_refused():
    """Scoring a criterion the task never stated would import a rubric we did not review."""
    criteria = ju.criteria_of(task())
    body = json.loads(verdict(None, criteria, resolved=(True, True), better="tie"))
    body["submissions"]["1"]["criteria"].append(
        dict(criterion="literary_quality", verdict="met", basis="it reads well"))
    with pytest.raises(ju.VerdictError, match="undeclared criterion"):
        ju.parse_verdict(json.dumps(body), criteria)


def test_a_valid_verdict_maps_onto_resolved_and_quality():
    criteria = ju.criteria_of(task())
    parsed = ju.parse_verdict(verdict(None, criteria), criteria)
    fields = ju.analysis_fields(parsed, {"1": "D", "2": "A"})
    assert fields["D"]["resolved"] is True and fields["D"]["quality"] == "win"
    assert fields["A"]["resolved"] is False and fields["A"]["quality"] is None


# ── 2. a task with no evaluation criteria is refused, and named ────────────────────────────────


def test_a_task_without_criteria_is_refused_by_id():
    with pytest.raises(ju.RubricError) as excinfo:
        ju.criteria_of(task(evaluation_criteria=[]))
    assert "theory_conversion_900" in str(excinfo.value)
    assert "no evaluation_criteria" in str(excinfo.value)


def test_a_mapping_shaped_rubric_is_refused_rather_than_flattened():
    doc = task(task_id="multi_beat_synthesis_001",
               evaluation_criteria={"beat_specific": {"beat_1": ["a", "b"]},
                                    "cross_beat": [{"criterion": "x", "description": "y"}]})
    with pytest.raises(ju.RubricError, match="mapping"):
        ju.criteria_of(doc)


def test_build_judgements_refuses_only_the_task_without_a_rubric():
    """One unusable task must not silently disappear; it is named, and the rest still judge."""
    good, bad = task(), task(task_id="theory_conversion_901", evaluation_criteria=[])
    unit = judgement(good)
    pairs = [(unit.treatment, unit.control), (collected("D", bad), collected("A", bad))]
    docs = manifest(good)
    docs["assignments"] += [dict(row, task=bad, task_id=bad["task_id"], task_hash=digest(bad),
                                 condition=condition) for row, condition in
                            ((docs["assignments"][0], "A"), (docs["assignments"][1], "D"))]
    judgements, refused = ju.build_judgements(pairs, docs, unit="judge")
    assert len(judgements) == 1
    assert "theory_conversion_901" in refused


# ── 3. the prompt carries the task's criteria and nothing invented ─────────────────────────────


def test_prompt_carries_exactly_the_tasks_criteria():
    doc, criteria = task(), ju.criteria_of(task())
    _, user = ju.judge_prompt(doc, criteria, {"1": "first prose", "2": "second prose"})
    assert ju.parse_rendered_criteria(user) == tuple(
        (criterion.name, criterion.description) for criterion in criteria)
    assert ju.assert_prompt_carries_rubric(user, criteria)
    for invented in ("literary_quality", "narrative_craft", "taste"):
        assert invented not in user


def test_a_prompt_with_an_invented_criterion_is_refused():
    criteria = ju.criteria_of(task())
    prompt = "\n".join((ju.SECTION.format(ju.CRITERIA_HEADER), "1. literary_quality",
                        "   prose that reads well", ju.SECTION.format("SUBMISSION 1"), "text"))
    with pytest.raises(ju.RubricError):
        ju.assert_prompt_carries_rubric(prompt, criteria)


# ── 4. the dry run builds every body and sends nothing ────────────────────────────────────────


def test_dry_run_builds_every_body_and_sends_nothing(tmp_path, monkeypatch):
    base = synthetic_base(tmp_path / "base")
    ledger = ledger_at(tmp_path)
    unit = judgement()
    raw = pp.json_bytes(manifest(unit.task))
    monkeypatch.setattr(ju.pa, "execute_attempt", lambda *a, **k: pytest.fail("dry run sent"))

    report = ju.run_dry_run([unit], ledger=ledger, prepare=prepare_for(tmp_path, raw, base),
                            manifest=manifest(unit.task), evidence_root=tmp_path / "evidence",
                            study=STUDY, outputs=(unit.control, unit.treatment))

    assert report["refused_reasons"] == []
    assert report["calls"] == 1 and report["bodies"][0]["body_bytes"] > 0
    assert report["reservation"]["total_micros"] > 0
    assert ledger.status()["attempt_ids"] == [], "a dry run must not reserve a ledger slot"
    assert (tmp_path / "evidence" / "attempts" / unit.attempt_id / "manifest.json").is_file()


def test_plan_refuses_when_the_data_judge_envelope_cannot_cover_it(tmp_path):
    base = synthetic_base(tmp_path / "base")
    ledger = ledger_at(tmp_path)
    unit = judgement()
    raw = pp.json_bytes(manifest(unit.task))
    status = ledger.status()
    ledger.transfer("fixture-transfer", source="data_judge", target="main",
                    amount_micros=status["envelopes"]["data_judge"]["remaining_micros"] - 1,
                    authorization=sl.Authorization("fixture-transfer", "d" * 64, "OFFLINE FIXTURE"))

    report = ju.run_plan([unit], ledger=ledger, prepare=prepare_for(tmp_path, raw, base),
                         manifest=manifest(unit.task), evidence_root=tmp_path / "evidence",
                         study=STUDY, outputs=(unit.control, unit.treatment))

    assert report["refused_reasons"] and "data_judge" in report["refused_reasons"][0]


def test_plan_prepares_and_prices_an_affordable_judgement(tmp_path):
    base = synthetic_base(tmp_path / "base")
    ledger = ledger_at(tmp_path)
    unit = judgement()
    raw = pp.json_bytes(manifest(unit.task))
    report = ju.run_plan([unit], ledger=ledger, prepare=prepare_for(tmp_path, raw, base),
                         manifest=manifest(unit.task), evidence_root=tmp_path / "evidence",
                         study=STUDY, outputs=(unit.control, unit.treatment))
    assert report["refused_reasons"] == []
    assert report["calls"] == 1 and report["reservation"]["total_micros"] > 0
    assert report["envelope"]["affordable"] is True


# ── the confound, and the token ───────────────────────────────────────────────────────────────


def test_judge_subject_confound_is_recorded_with_its_citation(tmp_path):
    study = tmp_path / "study.yaml"
    study.write_text("study_id: fixture\n"
                     "judge:\n"
                     "  confound_disclosure: 'the judge is one of the generation subjects'\n")
    record = ju.confound_record(manifest(), study)
    assert record["overlapping"] is True and record["overlapping_subjects"] == [JUDGE]
    assert record["disclosure"]["found"] is True
    assert record["disclosure"]["key"] == "judge.confound_disclosure"


def test_an_undisclosed_confound_is_recorded_as_not_found_not_as_a_quote():
    """No citation beats a fabricated one: the field says what was searched."""
    record = ju.confound_record(manifest(), STUDY)
    assert record["overlapping"] is True
    assert record["disclosure"]["found"] is False
    assert record["disclosure"]["searched"]


def test_evidence_refuses_a_confound_block_from_another_judge(tmp_path):
    base = synthetic_base(tmp_path / "base")
    unit = judgement()
    plan, _, _, slots = ju.prepare_judgement(
        unit, base=base, manifest_raw=pp.json_bytes(manifest(unit.task)),
        evidence_root=tmp_path / "evidence", judge_snapshot=JUDGE)
    wrong = dict(ju.confound_record(manifest(), STUDY), judge_snapshot="another/judge-20260101")
    with pytest.raises(ju.JudgeError, match="different judge"):
        ju._verdict_doc(unit, plan, slots, wrong, content=None, parsed=None, failure="x",
                        call={}, analysis={})


@pytest.mark.parametrize("token", ["", "PAID-PILOT-2026-01-01", "PAID-JUDGE-2026-13-01", "nope"])
def test_only_a_judge_token_authorises_a_judge_run(token):
    with pytest.raises(SystemExit):
        ju.authorise_token(token)


def test_a_judge_token_from_the_past_is_accepted():
    assert ju.authorise_token("PAID-JUDGE-2026-01-01") == "PAID-JUDGE-2026-01-01"


def test_a_body_over_the_route_profile_bound_is_refused(tmp_path):
    """The judge prompt is bigger than a generation prompt; a profile cannot underwrite it.

    The retained generation profiles declare an 8,000-byte input bound, and two submissions plus
    the brief exceed that at any realistic length. Preparing against them must refuse, not send a
    body the profile never vouched for.
    """
    base = synthetic_base(tmp_path / "base", input_bound=500)
    ledger = ledger_at(tmp_path)
    unit = judgement()
    raw = pp.json_bytes(manifest(unit.task))
    report = ju.run_plan([unit], ledger=ledger, prepare=prepare_for(tmp_path, raw, base),
                         manifest=manifest(unit.task), evidence_root=tmp_path / "evidence",
                         study=STUDY, outputs=(unit.control, unit.treatment))
    assert report["calls"] == 0
    assert any("input bound" in reason for reason in report["refused_reasons"])


def test_execute_without_a_judge_token_refuses_before_reading_anything(tmp_path):
    """The token is checked before the manifest, the ledger or the route record are touched."""
    with pytest.raises(SystemExit, match="AUTHORISE"):
        ju.main(["--execute", "--manifest", str(tmp_path / "missing.json"),
                 "--base", str(tmp_path / "base"), "--ledger", str(tmp_path / "l.sqlite3"),
                 "--evidence", str(tmp_path / "evidence")])


def test_a_consumed_judgement_is_skipped_never_resent(tmp_path, monkeypatch):
    """An attempt id is one ledger slot forever; a held slot is reported, not re-sent."""
    import types

    unit = judgement()
    plan = types.SimpleNamespace(spec=types.SimpleNamespace(attempt_id=unit.attempt_id))
    monkeypatch.setattr(ju, "_execute_one", lambda *a, **k: pytest.fail("a consumed slot was resent"))

    lines, held_back, stopped = ju._execute_plans(
        [(unit, (plan, object(), {}, {"1": "A", "2": "D"}))], ledger=None, confound={},
        evidence_root=tmp_path, held={unit.attempt_id}, stop_after_failures=3)

    assert lines == [] and stopped is None
    assert held_back == [dict(attempt_id=unit.attempt_id, state="CONSUMED",
                              reason="already held; a consumed judgement is never resent")]
