"""Finite offline source/route/price qualification; no live requests."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

import pytest

from provider_prices import (Route, json_bytes, money, parse_json, sha, validate_snapshot,
                             verify_snapshot)


def source_pack():
    route = Route("openrouter", "https://openrouter.ai/api/v1", "/chat/completions", "fixture/model", "fixture/model-date", "fixture/local", "default", "USD")
    catalog = {"data": [{"id": route.model_id, "canonical_slug": route.canonical_model_slug}]}
    endpoints = {"data": {"id": route.model_id, "endpoints": [dict(tag=route.endpoint_tag, model_id=route.model_id,
        provider_name="Fixture", pricing=dict(prompt="0.000001", completion="0.000002", input_cache_read="0.0000001",
            input_cache_write="0.00000125", request="0", discount=0), supported_parameters=["max_tokens"]) ]}}
    now = datetime.now(timezone.utc)
    return route, catalog, endpoints, (now-timedelta(seconds=1)).isoformat(), (now+timedelta(hours=1)).isoformat()


def validate(pack):
    route, catalog, endpoints, start, end = pack
    return validate_snapshot((json_bytes(catalog), json_bytes(endpoints)), route, start, end)


def test_current_official_shape_metadata_remains_unqualified():
    snapshot = validate(source_pack())
    assert snapshot.qualification == "unqualified"
    assert [r.sha256 for r in snapshot.source_refs] == [sha(raw) for raw in snapshot._sources]
    assert dict((c.name,c.usd_per_unit) for c in snapshot.rate_components)["prompt"] == "0.000001"


def test_context_override_highest_rate_discount_not_subtracted():
    pack = source_pack()
    pricing = pack[2]["data"]["endpoints"][0]["pricing"]
    pricing.update(discount=.5, overrides=[dict(min_prompt_tokens=200, prompt="0.000005", completion="0.000004")])
    rates = {r.name:r.usd_per_unit for r in validate(pack).rate_components}
    assert Decimal(rates["prompt"]) == Decimal("0.000005")
    assert Decimal(rates["completion"]) == Decimal("0.000004")


@pytest.mark.parametrize("change", ["unknown_component", "reasoning_price", "fixed_request", "time_override", "negative", "boolean", "invalid_discount", "missing_prompt"])
def test_D03_unknown_or_unsupported_prices_deny(change):
    pack = source_pack(); pricing = pack[2]["data"]["endpoints"][0]["pricing"]
    if change == "unknown_component":pricing["surprise"] = "1"
    if change == "reasoning_price":pricing["internal_reasoning"] = "1"
    if change == "fixed_request":pricing["request"] = "0.1"
    if change == "time_override":pricing["overrides"] = [dict(utc_start="00:00", prompt="1")]
    if change == "negative":pricing["prompt"] = "-1"
    if change == "boolean":pricing["prompt"] = True
    if change == "invalid_discount":pricing["discount"] = True
    if change == "missing_prompt":del pricing["prompt"]
    with pytest.raises(ValueError):validate(pack)


@pytest.mark.parametrize("change", ["base_url", "model_alias", "tag", "tier", "currency", "ambiguous_base"])
def test_D02_route_identity_and_matching_deny(change):
    pack = list(source_pack())
    if change == "base_url":pack[0] = replace(pack[0], base_url="https://attacker.invalid")
    if change == "model_alias":pack[0] = replace(pack[0], model_id="openrouter/auto")
    if change == "tag":pack[0] = replace(pack[0], endpoint_tag="other")
    if change == "tier":pack[0] = replace(pack[0], service_tier="priority")
    if change == "currency":pack[0] = replace(pack[0], currency="EUR")
    if change == "ambiguous_base":
        pack[0] = replace(pack[0], endpoint_tag="fixture")
        ep = dict(pack[2]["data"]["endpoints"][0]); ep["tag"] = "fixture"
        pack[2]["data"]["endpoints"].append(ep)
    with pytest.raises(ValueError):validate(pack)


@pytest.mark.parametrize("mode", ["expired", "future", "too_long", "zero", "naive"])
def test_D01_freshness_boundaries(mode):
    pack = list(source_pack()); now = datetime.now(timezone.utc)
    if mode == "expired":pack[3:5] = [(now-timedelta(days=1)).isoformat(),now.isoformat()]
    if mode == "future":pack[3:5] = [(now+timedelta(days=1)).isoformat(),(now+timedelta(days=1,hours=1)).isoformat()]
    if mode == "too_long":pack[4] = (now+timedelta(days=2)).isoformat()
    if mode == "zero":pack[4] = pack[3]
    if mode == "naive":pack[3] = "2026-09-23T00:00:00"
    with pytest.raises(ValueError):validate(pack)


def test_snapshot_dataclass_tamper_detected():
    snapshot = validate(source_pack())
    changed = replace(snapshot, rate_components=(replace(snapshot.rate_components[0], usd_per_unit="0"), *snapshot.rate_components[1:]))
    with pytest.raises(ValueError):verify_snapshot(changed)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff', b'{', b'['*2000])
def test_D08_strict_raw_json(raw):
    with pytest.raises(ValueError):parse_json(raw)


@pytest.mark.parametrize("amount", [True, 1.2, "NaN", "Infinity", "-0.1", "1e999"])
def test_exact_money_validation(amount):
    with pytest.raises(ValueError):money(amount)
