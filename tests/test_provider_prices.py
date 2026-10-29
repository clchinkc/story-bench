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


@pytest.mark.parametrize("change", ["unknown_component", "fixed_request", "time_selector_bad_type", "negative", "boolean", "invalid_discount", "missing_prompt"])
def test_D03_unknown_or_unsupported_prices_deny(change):
    pack = source_pack(); pricing = pack[2]["data"]["endpoints"][0]["pricing"]
    if change == "unknown_component":pricing["surprise"] = "1"
    # The former `reasoning_price` case used `internal_reasoning` as its example of an unknown
    # component. The 2026-10-07 W3b2 amendment ADMITS that component deliberately, so the case
    # was asserting denial for something now supported — the "a control decays when its check
    # gains a condition" failure. It is deleted rather than re-pointed, because
    # `test_a_genuinely_unknown_pricing_component_still_fails_closed` covers the same ground
    # with three distinct genuinely-unknown keys.
    if change == "fixed_request":pricing["request"] = "0.1"
    if change == "time_selector_bad_type":pricing["overrides"] = [dict(utc_start="00:00", utc_end=1400, prompt="1")]
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


# --- W3b2 amendment 2026-10-07: components and override shapes real routes declare -----
# Each admission carries its own control, in both directions: the newly admitted shape must
# be ACCEPTED, and a genuinely unknown key or selector must still FAIL CLOSED. Deleting the
# admission turns the accept-cases red; widening it to a blanket pass turns the refuse-cases red.

def _with_pricing(**extra):
    pack = source_pack()
    pricing = pack[2]["data"]["endpoints"][0]["pricing"]
    pricing.update(extra)
    return pack


def test_internal_reasoning_component_is_admitted():
    snapshot = validate(_with_pricing(internal_reasoning="0.000003"))
    rates = {r.name: r.usd_per_unit for r in snapshot.rate_components}
    assert rates["internal_reasoning"] == "0.000003"


def test_input_audio_cache_component_is_admitted():
    snapshot = validate(_with_pricing(input_audio_cache="0.000002"))
    rates = {r.name: r.usd_per_unit for r in snapshot.rate_components}
    assert rates["input_audio_cache"] == "0.000002"


@pytest.mark.parametrize("key", ["brand_new_charge", "internal_reasoning_v2", "utc_days"])
def test_a_genuinely_unknown_pricing_component_still_fails_closed(key):
    with pytest.raises(ValueError):
        validate(_with_pricing(**{key: "0.000001"}))


def _with_overrides(*overrides):
    pack = source_pack()
    pack[2]["data"]["endpoints"][0]["pricing"]["overrides"] = list(overrides)
    return pack


def test_time_conditional_override_is_admitted_and_never_discounts():
    # The base rate stays the reservation basis: a cheaper time window cannot reduce it.
    snapshot = validate(_with_overrides(
        {"utc_days": ["saturday", "sunday"], "prompt": "0.0000005", "completion": "0.000001"}))
    rates = {r.name: r.usd_per_unit for r in snapshot.rate_components}
    assert rates["prompt"] == "0.000001"
    assert rates["completion"] == "0.000002"


def test_time_window_override_is_admitted_and_a_higher_rate_raises_the_basis():
    snapshot = validate(_with_overrides(
        {"utc_start": 0, "utc_end": 1400, "prompt": "0.000004"}))
    rates = {r.name: r.usd_per_unit for r in snapshot.rate_components}
    assert rates["prompt"] == "0.000004"


@pytest.mark.parametrize("override", [
    {"unknown_selector": 1, "prompt": "0.000001"},          # unknown selector
    {"utc_days": ["funday"], "prompt": "0.000001"},         # invalid day
    {"utc_days": [], "prompt": "0.000001"},                 # empty day list
    {"utc_start": 0, "utc_end": 9999, "prompt": "0.000001"},  # out-of-range clock
    {"utc_days": ["monday"]},                               # selector but no rate
    {"prompt": "0.000001"},                                 # rate but no selector
    {"utc_start": 0, "prompt": "0.000001"},                 # half-declared window (review F4)
    {"utc_end": 1400, "prompt": "0.000001"},                # half-declared window (review F4)
    {"utc_start": "00:00", "utc_end": 1400, "prompt": "0.000001"},  # non-int clock (review F5)
])
def test_malformed_override_still_fails_closed(override):
    with pytest.raises(ValueError):
        validate(_with_overrides(override))


# --- W3b2 amendment 2026-10-29: bare base tags need an EXPLICIT sibling declaration ----------
# The old rule used a hardcoded suffix allowlist (`/fast`, `/flex` were tolerated while `/priority`
# and `/zdr` were not), which made a DEFAULT tier unpinnable for any provider offering such a
# variant — and was itself a copy of an external rule that drifted from it. The rule is now: a bare
# tag is acceptable only when the caller declares the exact sibling set, and the guard fails closed
# when the live set differs, including when a new sibling appears later.

def _bare_tag_pack(siblings):
    pack = source_pack()
    route = pack[0]
    endpoints = pack[2]["data"]["endpoints"]
    base = endpoints[0]
    base["tag"] = "acme"
    for name in siblings:
        extra = dict(base)
        extra["tag"] = name
        endpoints.append(extra)
    return [replace(route, endpoint_tag="acme"), *pack[1:]]


def test_bare_tag_with_no_siblings_is_accepted():
    pack = _bare_tag_pack([])
    assert validate(pack).rate_components


def test_bare_tag_with_declared_siblings_is_accepted():
    siblings = ["acme/priority", "acme/zdr"]
    pack = _bare_tag_pack(siblings)
    pack[0] = replace(pack[0], excluded_variants=tuple(siblings))
    assert validate(pack).rate_components


def test_bare_tag_with_undeclared_siblings_denies():
    with pytest.raises(ValueError):
        validate(_bare_tag_pack(["acme/priority"]))


@pytest.mark.parametrize("declared", [
    ("acme/priority",),                      # partially declared
    ("acme/priority", "acme/zdr", "acme/x"),  # over-declared (a variant that does not exist)
])
def test_partial_or_over_declaration_denies(declared):
    pack = _bare_tag_pack(["acme/priority", "acme/zdr"])
    pack[0] = replace(pack[0], excluded_variants=declared)
    with pytest.raises(ValueError):
        validate(pack)


def test_a_new_sibling_appearing_later_fails_closed():
    """The declaration is a claim about the live set, so a route that gains a sibling refuses."""
    pack = _bare_tag_pack(["acme/priority"])
    pack[0] = replace(pack[0], excluded_variants=("acme/priority",))
    assert validate(pack).rate_components
    pack = _bare_tag_pack(["acme/priority", "acme/priority/eu"])
    pack[0] = replace(pack[0], excluded_variants=("acme/priority",))
    with pytest.raises(ValueError):
        validate(pack)
