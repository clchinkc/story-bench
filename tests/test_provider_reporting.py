"""W1 original-input validation is executable, including before failed projection."""
import copy
import json
import pytest

from measurement_contract import PROTOCOL, digest, context_packet
from provider_reporting import project_cost_report
from spend_ledger import Authorization, Ledger
from provider_prices import json_bytes
from test_provider_attempts import make_case, execute, response


@pytest.mark.parametrize("invalid", ["stale_generation_hash", "missing_diagnostic", "wrong_output_hash", "duplicate_attempt"])
def test_P07_original_invalid_input_rejects_before_projection(fixture_records, invalid):
    manifest, records = fixture_records(models=("A",), tasks=("easy",))
    if invalid == "stale_generation_hash":records["evaluations"][0]["generation_hash"] = "0"*64
    if invalid == "missing_diagnostic":del records["evaluations"][0]["llm_results"]["beats_score"]
    if invalid == "wrong_output_hash":records["generations"][0]["output_hash"] = "0"*64
    if invalid == "duplicate_attempt":records["generations"][0]["attempts"] *= 2
    before = copy.deepcopy(records)
    # No ledger is supplied: original validation must reject before touching it.
    with pytest.raises(ValueError):project_cost_report(manifest, records, (), None)
    assert records == before


def test_P03_P06_unattempted_assignments_and_repeats_stay_in_denominator(fixture_records, tmp_path):
    manifest, _ = fixture_records(samples=(0,1,2))
    records = dict(protocol=PROTOCOL, assignment_hash=digest(manifest), generations=[], evaluations=[])
    ledger = Ledger.create(tmp_path/'ledger', program_id="synthetic", store_id="synthetic",
        authorization=Authorization("synthetic", "a"*64, "SYNTHETIC ONLY"))
    original = copy.deepcopy(records)
    result = project_cost_report(manifest, records, (), ledger)
    assert records == original
    assert len(result["report"]["assigned_ids"]) == 12
    for model in result["report"]["models"]:
        assert model["story_clusters"] == 1 and model["assigned"] == 6
        assert model["resolved"] is None and model["deployment_cost"]["total_usd"] is None
    assert result["exact_cost"]["program"]["unknown_charge_ids"] == ["hosted_session"]
    assert result["exact_cost"]["program"]["usable_micros"] == 0


@pytest.mark.parametrize('counter',['cached_tokens','cache_write_tokens'])
@pytest.mark.parametrize('structural_zero',[False,True])
def test_R1_full_report_cache_completeness(tmp_path,counter,structural_zero):
    kind='cache_read_tokens' if counter=='cached_tokens' else 'cache_write_tokens'
    bound='cache_read_bound' if counter=='cached_tokens' else 'cache_write_bound'
    changes={bound:0,'structural_zero':[kind]} if structural_zero else None
    case=make_case(tmp_path,profile_changes=changes)
    p,ledger,_,manifest=case;data=json.loads(response(p));del data['usage']['prompt_tokens_details'][counter]
    execute(case,json_bytes(data))
    assignment=manifest['assignments'][0]
    parts={'task':json.dumps(assignment['task'],sort_keys=True,ensure_ascii=False)}
    prompt,context=context_packet(parts,max_bytes=10000)
    gen=dict(**{k:assignment[k] for k in ('model','task_id','sample','condition','task_hash','model_snapshot','prompt_version')},
        protocol=PROTOCOL,record_id='cache-projection',status='completed',timestamp='2026-09-23T00:00:00Z',
        output='Complete end.',output_hash=digest('Complete end.'),prompt=prompt,prompt_hash=digest(prompt),
        context_parts=parts,context=context,finish_reason='stop',
        attempts=[dict(attempt_id=p.spec.attempt_id,role='generation',status='completed',cost_usd=7,usage=None)])
    records=dict(protocol=PROTOCOL,assignment_hash=digest(manifest),generations=[gen],evaluations=[])
    before=copy.deepcopy(records)
    result=project_cost_report(manifest,records,(p,),ledger)
    projected=result['records']['generations'][0]['attempts'][0]
    summary=result['report']['models'][0]['deployment_cost']
    assert (projected['usage'] is None) is (not structural_zero)
    assert summary['unknown_usage_attempts']==int(not structural_zero)
    assert ledger.status()['committed_micros']==30000 and records==before
    assert bool(ledger.status()['incident_ids']) is (not structural_zero)
