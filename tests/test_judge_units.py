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


def test_an_undisclosed_confound_is_recorded_as_not_found_not_as_a_quote(tmp_path):
    """No citation beats a fabricated one: the field says what was searched."""
    silent = tmp_path / "study.yaml"
    silent.write_text("study_id: fixture\njudge: {}\n")
    record = ju.confound_record(manifest(), silent)
    assert record["overlapping"] is True
    assert record["disclosure"]["found"] is False
    assert record["disclosure"]["searched"]


def test_the_real_study_discloses_the_confound_it_actually_has():
    """The overlap is a frozen-design fact, so the study must state it in its own words."""
    record = ju.confound_record(manifest(), STUDY)
    assert record["overlapping"] is True
    assert record["disclosure"]["found"] is True, record["disclosure"]


def test_the_judge_overlaps_a_dated_snapshot_of_the_same_route():
    """The judge is a CANONICAL SLUG; a row carries a DATED snapshot. Comparing them with `==`
    says 'no overlap', and that false negative is the one error that would hide a confound
    inside paid evidence."""
    judge = "google/gemini-4-argon"
    assert ju.same_route(judge, "google/gemini-4-argon-20261013")
    assert ju.same_route("google/gemini-4-argon-20261013", judge)
    assert not ju.same_route(judge, "google/gemini-4-argon-pro-20261013")
    record = ju.confound_record(
        manifest(judge_snapshot=judge, model_snapshot="google/gemini-4-argon-20261013"), STUDY)
    assert record["overlapping"] is True
    assert record["affected_arm_runs"] == 2
    assert record["overlapping_subjects"] == ["google/gemini-4-argon-20261013"]


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


# ── 6. a self-contradicting reply is re-asked ONCE, and the rejected reading is kept ─────────────
#
# The failure this covers was observed on a paid run: 4 of 11 replies set `resolved` so that exactly
# one submission was true, then named the OTHER as `better`. The rule is in JUDGE_RULES and the judge
# still broke it. Discarding those judgements wastes a paid call; deriving the answer from `resolved`
# would let the harness write a preference the judge never asserted. One re-ask, quoting the judge's
# own two fields and the rule, is the middle path -- and it never names the answer.


def test_a_contradiction_is_its_own_error_class_and_still_a_verdict_error():
    """Positive control: the one re-askable failure. Negative controls: every consistent reply.

    It must stay a `VerdictError` subclass so every existing handler keeps treating it as an
    instrument failure rather than crashing on an uncaught type.
    """
    assert issubclass(ju.ResolutionConflict, ju.VerdictError)
    with pytest.raises(ju.ResolutionConflict, match="only resolved submission"):
        ju._preference({"better": "2"}, {"1": True, "2": False})
    with pytest.raises(ju.ResolutionConflict):
        ju._preference({"better": "1"}, {"1": False, "2": True})

    # consistent replies raise nothing, including the two legal non-preference outcomes
    assert ju._preference({"better": "1"}, {"1": True, "2": False}) == "1"
    assert ju._preference({"better": "2"}, {"1": False, "2": True}) == "2"
    assert ju._preference({"better": "tie"}, {"1": False, "2": False}) == "tie"
    assert ju._preference({"better": "cannot_assess"}, {"1": None, "2": None}) == "cannot_assess"


def test_a_correction_quotes_the_rule_and_leaves_the_reply_instruction_last():
    """The correction must not become the last thing read, and must not break the rubric check."""
    criteria = ju.criteria_of(task())
    conflict = ju.ResolutionConflict("better must name submission 2, the only resolved submission")
    message = ju._corrective_message(conflict)
    assert "better must name submission 2" in message, "the judge's own contradiction is quoted back"
    assert "resolved" in message and "better" in message, "it is told which two fields disagree"
    for directive in ("must be that one", "do not assert a preference"):
        assert directive in message, "the rule it broke is quoted, not paraphrased from memory"

    _, user = ju.judge_prompt(task(), criteria, {"1": "first prose", "2": "second prose"},
                              correction=message)
    assert "CORRECTION REQUIRED" in user
    assert user.index("CORRECTION REQUIRED") < user.index("Return the JSON object now.")
    assert user.rstrip().endswith("Return the JSON object now.")
    ju.assert_prompt_carries_rubric(user, criteria)  # the rubric check survives a correction

    # without a correction the prompt is unchanged: the first call reads exactly as it always did
    without = ju.judge_prompt(task(), criteria, {"1": "first prose", "2": "second prose"})[1]
    assert "CORRECTION REQUIRED" not in without


def _loop_harness(monkeypatch, unit, errors):
    """Drive `_execute_plans` offline: record what was sent, and reply from a queue of outcomes."""
    sent, prepared = [], []

    async def fake_execute(plan, *args, **kwargs):
        sent.append(plan.spec.attempt_id)
        return object(), 5, {}

    def fake_outcome(plan, ledger, judgement, slot_by_arm, result, latency_ms, observed, confound,
                     evidence_root, stem="verdict"):
        error = errors.pop(0) if errors else None
        failure = None if error is None else str(error)
        line = dict(judgement_id=judgement.judgement_id, attempt_id=plan.spec.attempt_id,
                    state="RECONCILED", cost_usd="0.010000", latency_ms=5, failure=failure,
                    analysis={}, evidence="e")
        return line, failure, None, error

    monkeypatch.setattr(ju, "_execute_one", fake_execute)
    monkeypatch.setattr(ju, "_judge_outcome", fake_outcome)

    def prepare(judgement, correction=None, tag=None):
        prepared.append(dict(correction=correction, tag=tag))
        plan = type("P", (), {"spec": type("S", (), {"attempt_id": unit.attempt_id + "-r1"})()})
        return plan, object(), {}, {"1": "A", "2": "D"}

    first = type("P", (), {"spec": type("S", (), {"attempt_id": unit.attempt_id})()})
    return sent, prepared, prepare, first


def test_a_contradiction_is_reasked_once_and_the_rejected_reading_is_kept(tmp_path, monkeypatch):
    unit = judgement()
    sent, prepared, prepare, first = _loop_harness(
        monkeypatch, unit, [ju.ResolutionConflict("better must name submission 2"), None])
    directory = tmp_path / "judgements" / unit.judgement_id
    directory.mkdir(parents=True)
    (directory / "verdict.json").write_text("{\"first\": true}", encoding="utf-8")

    lines, held_back, stopped = ju._execute_plans(
        [(unit, (first, object(), {}, {"1": "A", "2": "D"}))], ledger=None, confound={},
        evidence_root=tmp_path, held=set(), stop_after_failures=3, prepare=prepare)

    assert sent == [unit.attempt_id, unit.attempt_id + "-r1"], "exactly one re-ask, new attempt id"
    assert len(prepared) == 1 and prepared[0]["tag"] == ju.CORRECTIVE_TAG
    assert "better must name submission 2" in prepared[0]["correction"]
    assert len(lines) == 1, "one row per judgement: a re-ask is not a second judgement"
    assert lines[0]["failure"] is None
    assert lines[0]["corrective_reask"]["superseded_attempt_id"] == unit.attempt_id
    assert lines[0]["corrective_reask"]["also_conflicted"] is False
    assert (directory / "verdict.rejected.json").read_text(encoding="utf-8") == "{\"first\": true}"
    assert stopped is None


def test_a_second_contradiction_is_recorded_not_reasked_a_third_time(tmp_path, monkeypatch):
    """One re-ask is the budget. A judge that contradicts itself twice is grader-invalid, and the
    repetition guard still sees a failure so a systematic fault cannot buy 176 calls."""
    unit = judgement()
    sent, _, prepare, first = _loop_harness(
        monkeypatch, unit, [ju.ResolutionConflict("again"), ju.ResolutionConflict("again")])

    lines, _, _ = ju._execute_plans(
        [(unit, (first, object(), {}, {"1": "A", "2": "D"}))], ledger=None, confound={},
        evidence_root=tmp_path, held=set(), stop_after_failures=3, prepare=prepare)

    assert sent == [unit.attempt_id, unit.attempt_id + "-r1"], "no third call"
    assert lines[0]["corrective_reask"]["also_conflicted"] is True
    assert lines[0]["failure"] is not None


def test_a_malformed_reply_is_never_reasked(tmp_path, monkeypatch):
    """Only a contradiction is re-askable. A reply that is not the schema says the instrument was
    driven wrong; buying it again is how a run spends its envelope learning nothing."""
    unit = judgement()
    sent, prepared, prepare, first = _loop_harness(
        monkeypatch, unit, [ju.VerdictError("better 'maybe' is not one of ('1', '2')")])

    lines, _, _ = ju._execute_plans(
        [(unit, (first, object(), {}, {"1": "A", "2": "D"}))], ledger=None, confound={},
        evidence_root=tmp_path, held=set(), stop_after_failures=3, prepare=prepare)

    assert sent == [unit.attempt_id]
    assert prepared == []
    assert lines[0]["failure"].startswith("better 'maybe'")


# ── 7. `--repair-invalid`: a spent slot is reused only when its verdict is unusable ─────────────
#
# Measured need: on the 2026-11-09 clean1 run 47 of 176 judgements settled with a rejected reply
# (mostly `finish_reason == "length"`). Their attempt ids were spent, so every later run skipped them
# and those units lost both arms. Repair is the narrow exception: a held judgement is re-judged only
# when its STORED verdict carries a failure, under a new `-r2` id, after the old reading is preserved.


def _held_pair(unit, verdict_document=None):
    """(judgement, prepared-plan tuple) shaped exactly as `_prepare_all` returns it."""
    plan = type("P", (), {"spec": type("S", (), {"attempt_id": unit.attempt_id})()})
    return unit, (plan, object(), {}, {"1": "A", "2": "D"})


def _repair_harness(monkeypatch, unit, sent, tags):
    async def fake_execute(plan, *args, **kwargs):
        sent.append(plan.spec.attempt_id)
        return object(), 1, {}

    def fake_outcome(plan, ledger, judgement, slot_by_arm, result, latency_ms, observed, confound,
                     evidence_root, stem="verdict"):
        line = dict(judgement_id=judgement.judgement_id, attempt_id=plan.spec.attempt_id,
                    state="RECONCILED", cost_usd="0.010000", failure=None, analysis={}, evidence="e")
        return line, None, None, None

    monkeypatch.setattr(ju, "_execute_one", fake_execute)
    monkeypatch.setattr(ju, "_judge_outcome", fake_outcome)

    def prepare(judgement, correction=None, tag=None):
        tags.append(tag)
        plan = type("P", (), {"spec": type("S", (), {"attempt_id": unit.attempt_id + (tag or "")})()})
        return plan, object(), {}, {"1": "A", "2": "D"}

    return prepare


def test_repair_invalid_rejudges_only_a_held_judgement_whose_verdict_failed(tmp_path, monkeypatch):
    unit = judgement()
    directory = tmp_path / "judgements" / unit.judgement_id
    directory.mkdir(parents=True)
    (directory / "verdict.json").write_text('{"failure": "judge call did not complete: truncated"}',
                                            encoding="utf-8")
    sent, tags = [], []
    prepare = _repair_harness(monkeypatch, unit, sent, tags)

    lines, held_back, stopped = ju._execute_plans(
        [_held_pair(unit, None)], ledger=None, confound={}, evidence_root=tmp_path,
        held={unit.attempt_id}, stop_after_failures=3, prepare=prepare, repair_invalid=True)

    assert tags == [ju.REPAIR_TAG], "a repair is tagged, so its attempt id is new"
    assert sent == [unit.attempt_id + ju.REPAIR_TAG], "the spent slot is never reused"
    assert held_back == [], "a repaired judgement is not reported as held back"
    assert lines[0]["failure"] is None
    preserved = directory / "verdict.prior1.json"
    assert preserved.exists(), "the failed reading is kept, not overwritten"
    assert "truncated" in preserved.read_text(encoding="utf-8")
    assert stopped is None


def test_repair_invalid_never_rebuys_a_verdict_that_already_parsed(tmp_path, monkeypatch):
    """The guard that keeps a repair pass from paying twice for good judgements."""
    unit = judgement()
    directory = tmp_path / "judgements" / unit.judgement_id
    directory.mkdir(parents=True)
    (directory / "verdict.json").write_text('{"failure": null, "analysis": {}}', encoding="utf-8")
    sent, tags = [], []
    prepare = _repair_harness(monkeypatch, unit, sent, tags)

    lines, held_back, _ = ju._execute_plans(
        [_held_pair(unit, None)], ledger=None, confound={}, evidence_root=tmp_path,
        held={unit.attempt_id}, stop_after_failures=3, prepare=prepare, repair_invalid=True)

    assert sent == [] and tags == []
    assert lines == []
    assert held_back == [dict(attempt_id=unit.attempt_id, state="CONSUMED",
                              reason="already held; a consumed judgement is never resent")]


def test_without_the_flag_a_held_judgement_is_still_never_resent(tmp_path, monkeypatch):
    """The default must not change: a failed held verdict stays failed unless repair is asked for."""
    unit = judgement()
    directory = tmp_path / "judgements" / unit.judgement_id
    directory.mkdir(parents=True)
    (directory / "verdict.json").write_text('{"failure": "truncated"}', encoding="utf-8")
    sent, tags = [], []
    prepare = _repair_harness(monkeypatch, unit, sent, tags)

    lines, held_back, _ = ju._execute_plans(
        [_held_pair(unit, None)], ledger=None, confound={}, evidence_root=tmp_path,
        held={unit.attempt_id}, stop_after_failures=3, prepare=prepare)

    assert sent == [] and tags == []
    assert len(held_back) == 1 and lines == []
