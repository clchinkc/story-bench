"""Finite price snapshots under two named authorities; this module never opens a connection.

A snapshot is `unqualified` from catalog+endpoint bytes alone. A synthetic profile yields
`offline_fixture`; a profile carrying a passing, route-independent C8 verification artifact
yields `c8_verified`. The difference is who vouches, and it is read from the fourth source
rather than asserted by the profile that benefits from it.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import re

from spend_ledger import ValidationError, usd_to_micros

MAX_BYTES = 16 * 1024 * 1024


def sha(data):
    if type(data) is not bytes:
        raise ValidationError("exact bytes required")
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_json(raw):
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValidationError("bounded JSON bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValidationError("nonfinite JSON number")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_float=Decimal, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("invalid JSON bytes") from exc


def exact_fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValidationError("unexpected or missing fields")


def integer(value, minimum=0, maximum=10**15):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError("bounded integer required")
    return value


def text(value):
    if type(value) is not str or not value.strip() or len(value) > 1024 or any(ord(c) < 32 for c in value):
        raise ValidationError("bounded nonempty text required")
    return value


def digest_text(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError("lowercase SHA256 required")
    return value


def money(value):
    if type(value) is not str:
        raise ValidationError("decimal text required")
    usd_to_micros(value)  # accepted ledger owns exact precision/range validation
    return Decimal(value)


def utc(value):
    try:
        parsed = datetime.fromisoformat(text(value))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("UTC required")
        return parsed
    except (ValueError, OverflowError) as exc:
        raise ValidationError("offset-zero UTC timestamp required") from exc


def fresh(observed, expires):
    start, end = utc(observed), utc(expires)
    if not timedelta(0) < end - start <= timedelta(hours=24):
        raise ValidationError("price lifetime must be positive and at most24h")
    if not start <= datetime.now(timezone.utc) < end:
        raise ValidationError("stale or future snapshot")


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    sha256: str
    size_bytes: int

    @classmethod
    def of(cls, name, raw):
        text(name)
        integer(len(raw), maximum=MAX_BYTES)
        return cls(name, sha(raw), len(raw))


@dataclass(frozen=True)
class Route:
    provider: str
    base_url: str
    path: str
    model_id: str
    canonical_model_slug: str
    endpoint_tag: str
    service_tier: str
    currency: str
    # Bare base tags are ambiguous whenever the route exposes `<tag>/...` siblings. Instead of a
    # hardcoded suffix allowlist (which drifts from reality — `/fast` and `/flex` were excluded
    # while `/priority` and `/zdr` were not), the caller DECLARES the exact sibling set it is not
    # selecting. The guard then fails closed if the live set differs, including when a NEW sibling
    # appears later. The default `()` means "this tag must have no siblings".
    excluded_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class RateComponent:
    name: str
    usd_per_unit: str
    unit: str
    applicability: str
    source_ref: ArtifactRef


@dataclass(frozen=True)
class PriceSnapshot:
    route: Route
    source_refs: tuple[ArtifactRef, ...]
    observed_at: str
    expires_at: str
    rate_components: tuple[RateComponent, ...]
    qualification: str
    qualification_evidence: tuple[ArtifactRef, ...]
    _sources: tuple[bytes, ...] = field(repr=False)


PRICE_KEYS = {"prompt", "completion", "input_cache_read", "input_cache_write",
              "input_cache_write_1h", "request", "web_search", "image", "audio",
              "audio_output", "image_output", "image_token",
              # W3b2 amendment, 2026-10-07: components real OpenRouter endpoints declare.
              # `internal_reasoning` is a separately-priced reasoning charge; `input_audio_cache`
              # is a media-scoped cache charge. Both were previously refused as unknown, which
              # kept a route carrying them unpriceable rather than making it unsafe.
              "internal_reasoning", "input_audio_cache"}
# Override selectors real routes use. A selector narrows WHEN a rate applies; the applied
# rate is always the MAXIMUM over the base rate and every override, so a discount or a
# cheaper time window can never reduce a reservation. `utc_start` and `utc_end` describe one
# window and must be declared together.
OVERRIDE_SELECTORS = {"min_prompt_tokens", "utc_days", "utc_start", "utc_end"}
UTC_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
PROFILE_FIELDS = {"kind", "catalog_sha256", "endpoints_sha256", "model_id",
                  "canonical_model_slug", "endpoint_tag", "input_bound", "output_bound",
                  "reasoning_bound", "cache_read_bound", "cache_write_bound",
                  "structural_zero", "no_incremental_infrastructure", "no_metered_agent"}
# Two authorities may vouch for a route's execution profile, and they are NAMED rather than assumed.
# A synthetic profile exists so MockTransport can be exercised without claiming anything about the
# world. A verified profile claims something about the world, so it must carry the C8 verification
# artifact that made the claim, and that artifact's bytes are hashed into the snapshot beside the
# prices. The difference between the two is who vouches, recorded explicitly — not a code path.
SYNTHETIC_KIND = "SYNTHETIC OFFLINE ONLY"
VERIFIED_KIND = "C8 VERIFIED ROUTE"
REAL_PROFILE_FIELDS = PROFILE_FIELDS | {"verification_sha256", "verifier", "verdict",
                                        "verified_at"}
_STRUCTURAL_ZERO_KEY = {"reasoning_tokens": "reasoning_bound",
                        "cache_read_tokens": "cache_read_bound",
                        "cache_write_tokens": "cache_write_bound"}
# Non-token components that may be asserted zero by the profile. `request` has no token bound to
# contradict, so it needs no entry in the map above — but asserting it is a CLAIM, which is why it
# must be declared explicitly rather than inferred from the provider omitting a price key.
_STRUCTURAL_ZERO_PLAIN = frozenset({"request"})
# The qualifications that permit a reservation. `unqualified` never appears here: a bare
# catalog+endpoint fetch is evidence of PRICES, not of a route anyone has agreed to pay for.
QUALIFICATION_FOR_KIND = {SYNTHETIC_KIND: "offline_fixture", VERIFIED_KIND: "c8_verified"}
AUTHORITY_FOR_QUALIFICATION = {v: k for k, v in QUALIFICATION_FOR_KIND.items()}
QUALIFIED_KINDS = frozenset(QUALIFICATION_FOR_KIND.values())


def _profile_identity(profile, route, refs, label):
    """The profile must describe THESE bytes and THIS route, or it describes nothing."""
    if profile["catalog_sha256"] != refs[0].sha256 or profile["endpoints_sha256"] != refs[1].sha256:
        raise ValidationError(f"{label} source binding")
    for key in ("model_id", "canonical_model_slug", "endpoint_tag"):
        if profile[key] != getattr(route, key):
            raise ValidationError(f"{label} route binding")


def _token_bounds(profile):
    """The three token ceilings a reservation is built from, checked before anything is priced."""
    for key in ("input_bound", "output_bound", "reasoning_bound", "cache_read_bound", "cache_write_bound"):
        integer(profile[key])
    if profile["output_bound"] < 1 or profile["reasoning_bound"] > profile["output_bound"]:
        raise ValidationError("invalid combined completion bound")
    if max(profile["cache_read_bound"], profile["cache_write_bound"]) > profile["input_bound"]:
        raise ValidationError("invalid cache subset bound")


def _zero_proof(profile):
    """Which counters are structurally zero. A claim of zero that the bounds contradict is refused."""
    zeros = profile["structural_zero"]
    allowed = set(_STRUCTURAL_ZERO_KEY) | _STRUCTURAL_ZERO_PLAIN
    if type(zeros) is not list or len(zeros) != len(set(zeros)) or set(zeros) - allowed:
        raise ValidationError("invalid structural-zero proof")
    for kind in zeros:
        if kind in _STRUCTURAL_ZERO_KEY and profile[_STRUCTURAL_ZERO_KEY[kind]] != 0:
            raise ValidationError("structural-zero contradicts bound")
    return frozenset(zeros)


def _cost_completeness(profile, rates, zeros, label):
    """An unknown cost is not zero, so it blocks rather than underwrites a reservation."""
    if profile["no_incremental_infrastructure"] is not True or profile["no_metered_agent"] is not True:
        raise ValidationError(f"unknown {label} execution cost")
    for kind, key in (("cache_read_tokens", "input_cache_read"), ("cache_write_tokens", "input_cache_write")):
        if key not in rates and kind not in zeros:
            raise ValidationError("unknown applicable cache price")
    if "request" not in rates and "request" not in zeros:
        # Absence is UNKNOWN unless the profile explicitly asserts the component cannot be charged.
        # The retained records record exactly this component as unknown, so a route only becomes
        # priceable when that assertion is made deliberately and auditable.
        raise ValidationError("unknown request price")


def _profile_bounds(profile, rates, label):
    """Bounds are what the ledger reserves against, so they are checked identically whatever
    vouches for them; only the authority differs between a fixture and a verified route."""
    _token_bounds(profile)
    _cost_completeness(profile, rates, _zero_proof(profile), label)



VERIFICATION_MAX_AGE = timedelta(hours=24)


def _instant(value):
    """A tz-aware instant in UTC. The retained artifacts write `+08:00`, so demanding a `Z` suffix
    would have refused every real verification while accepting every forged one that copied the
    format — the same shape of bug as comparing `pass` to `PASS`."""
    try:
        parsed = datetime.fromisoformat(text(value))
        if parsed.tzinfo is None:
            raise ValueError("offset required")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError("offset-aware timestamp required") from exc


def _verification(profile, route, artifact, expires):
    """Read the C8 artifact and let IT be the authority, rather than the profile vouching for itself.

    Without this the fourth source was any JSON file whatsoever: the profile declared a verdict and
    the code compared only a hash the profile itself wrote, so a passing claim needed no passing
    evidence. A self-signed qualification is not a qualification; it is a restatement.
    """
    if type(artifact) is not dict:
        raise ValidationError("verification artifact must be an object")
    if str(artifact.get("verdict", "")).lower() != "pass":
        raise ValidationError("verified qualification requires a passing verdict")
    if artifact.get("subject") != route.model_id:
        raise ValidationError("verification artifact names another subject")
    if artifact.get("route_independent") is not True:
        raise ValidationError("route-dependent verification does not qualify")
    if artifact.get("verifier") != profile["verifier"]:
        raise ValidationError("verification verifier mismatch")
    verified = _instant(artifact.get("verified_at"))
    if verified != _instant(profile["verified_at"]):
        raise ValidationError("verification timestamp mismatch")
    # Anchored on the WINDOW, not on `observed_at`: the C8 reviews for this base legitimately ran
    # minutes AFTER the price snapshot was taken, so "must not postdate the snapshot" refused
    # every real artifact while accepting synthetic ones written earlier. What matters is that
    # the review is not from the future of its own evidence and has not gone stale.
    expires = _instant(expires)
    if verified > expires:
        raise ValidationError("verification postdates the evidence window")
    if expires - verified > VERIFICATION_MAX_AGE:
        raise ValidationError("verification is stale for these bytes")


def _qualify(sources, route, refs, rates, expires_at):
    """Decide WHOSE authority backs this profile, and refuse anything that does not name itself.

    A synthetic profile may describe any route; a verified one must carry the C8 artifact that
    verified those exact bytes, so the claim and the evidence travel together.
    """
    profile = sources[2]
    kind = profile.get("kind") if type(profile) is dict else None
    if len(sources) == 3 and kind == SYNTHETIC_KIND:
        exact_fields(profile, PROFILE_FIELDS)
        _profile_identity(profile, route, refs, "synthetic")
        _profile_bounds(profile, rates, "fixture")
        return QUALIFICATION_FOR_KIND[kind]
    if len(sources) == 4 and kind == VERIFIED_KIND:
        exact_fields(profile, REAL_PROFILE_FIELDS)
        _profile_identity(profile, route, refs, "verified")
        _profile_bounds(profile, rates, "verified route")
        if profile["verification_sha256"] != refs[3].sha256:
            raise ValidationError("verification artifact binding")
        _verification(profile, route, sources[3], expires_at)
        return QUALIFICATION_FOR_KIND[kind]
    raise ValidationError("unknown profile authority or source count")


def validate_snapshot(raw_sources, route, observed_at, expires_at):
    """Parse real catalog+endpoint bytes, optionally with a profile that names its own authority.

    Two profiles are admissible and they claim different things. A SYNTHETIC profile is enforcement
    evidence for MockTransport only: it says nothing about the world. A C8 VERIFIED profile says the
    prices and identity in this snapshot were independently verified, and it must therefore carry the
    verification artifact whose sha256 it declares, as a fourth source, so the claim and its evidence
    are hashed together. With neither profile the snapshot is `unqualified` — the default, and the
    state that keeps a bare catalog+endpoint fetch from being mistaken for a qualified route.
    """
    if type(route) is not Route or type(raw_sources) is not tuple or len(raw_sources) not in (2, 3, 4):
        raise ValidationError("finite route and source tuple required")
    if (route.provider, route.base_url, route.path, route.currency) != (
            "openrouter", "https://openrouter.ai/api/v1", "/chat/completions", "USD"):
        raise ValidationError("unsupported provider URL/currency")
    for item in (route.model_id, route.canonical_model_slug, route.endpoint_tag, route.service_tier):
        text(item)
    if ":" in route.model_id or ":" in route.canonical_model_slug or route.model_id.startswith("openrouter/"):
        raise ValidationError("dynamic model/variant forbidden")
    if route.service_tier not in {"default", "flex", "priority"}:
        raise ValidationError("unsupported service tier")
    fresh(observed_at, expires_at)
    sources = tuple(parse_json(raw) for raw in raw_sources)
    refs = tuple(ArtifactRef.of(f"price_source_{n}", raw) for n, raw in enumerate(raw_sources))
    try:
        models = [m for m in sources[0]["data"] if m["id"] == route.model_id and m["canonical_slug"] == route.canonical_model_slug]
        endpoint_data = sources[1]["data"]
        endpoints = endpoint_data["endpoints"]
        matches = [e for e in endpoints if e["tag"] == route.endpoint_tag]
        if len(models) != 1 or endpoint_data["id"] != route.model_id or len(matches) != 1:
            raise ValidationError("catalog/model/endpoint identity mismatch")
        if "/" not in route.endpoint_tag:
            siblings = sorted(e["tag"] for e in endpoints
                              if e["tag"].startswith(route.endpoint_tag + "/"))
            if siblings != sorted(route.excluded_variants):
                raise ValidationError("ambiguous base provider slug")
        endpoint = matches[0]
        expected_tier = "priority" if route.endpoint_tag.endswith("/fast") else "flex" if route.endpoint_tag.endswith("/flex") else "default"
        if route.service_tier != expected_tier or endpoint["model_id"] != route.model_id:
            raise ValidationError("endpoint tier/model mismatch")
        pricing = endpoint["pricing"]
        if set(pricing) - PRICE_KEYS - {"discount", "overrides"} or not {"prompt", "completion"} <= pricing.keys():
            raise ValidationError("unsupported or missing pricing component")
        discount = pricing.get("discount", 0)
        if type(discount) not in (int, Decimal) or not 0 <= discount <= 1:
            raise ValidationError("invalid discount")
        rates = {k: money(v) for k, v in pricing.items() if k in PRICE_KEYS}
        overrides = pricing.get("overrides", [])
        if type(overrides) is not list:
            raise ValidationError("invalid overrides")
        for override in overrides:
            if type(override) is not dict or not set(override) & OVERRIDE_SELECTORS \
                    or set(override) - PRICE_KEYS - OVERRIDE_SELECTORS:
                raise ValidationError("unsupported price override")
            if not set(override) & PRICE_KEYS:
                raise ValidationError("price override declares no rate")
            if "min_prompt_tokens" in override:
                integer(override["min_prompt_tokens"])
            if "utc_days" in override:
                days = override["utc_days"]
                if type(days) is not list or not days or set(days) - UTC_DAYS:
                    raise ValidationError("invalid utc_days override")
            for selector in ("utc_start", "utc_end"):
                if selector in override:
                    integer(override[selector], maximum=2359)
            if ("utc_start" in override) != ("utc_end" in override):
                raise ValidationError("utc_start and utc_end must be declared together")
            for key in override.keys() & PRICE_KEYS:
                rates[key] = max(rates.get(key, Decimal(0)), money(override[key]))
        if rates.get("request", Decimal(0)) != 0:
            raise ValidationError("fixed request charge unsupported")
        components = tuple(RateComponent(k, str(v),
                                         "request" if k in {"request", "web_search"} else "token",
                                         "maximum over the base rate and every declared override, "
                                         "token-threshold or time-conditional; a discount is never "
                                         "subtracted", refs[1])
                           for k, v in sorted(rates.items()))
        qualification = _qualify(sources, route, refs, rates, expires_at) if len(sources) > 2 else "unqualified"
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValidationError("malformed catalog or endpoint metadata") from exc
    return PriceSnapshot(route, refs, observed_at, expires_at, components, qualification,
                         refs[2:] if qualification in QUALIFIED_KINDS else (), raw_sources)


def verify_snapshot(snapshot, *, require_current=True):
    if type(snapshot) is not PriceSnapshot:
        raise ValidationError("immutable snapshot required")
    # Recovery validates original bytes with the original interval; expiry only blocks send.
    if require_current:
        rebuilt = validate_snapshot(snapshot._sources, snapshot.route, snapshot.observed_at, snapshot.expires_at)
        if rebuilt != snapshot:
            raise ValidationError("snapshot drift")
    else:
        if tuple(ArtifactRef.of(f"price_source_{n}", raw) for n, raw in enumerate(snapshot._sources)) != snapshot.source_refs:
            raise ValidationError("snapshot bytes changed")
        # Bytes matching is not enough: a hand-built snapshot could keep genuine source bytes and
        # relabel itself `c8_verified`. The authority is re-derived from the profile's own kind, so
        # recovery cannot launder an unverified route into a qualified one.
        sources = tuple(parse_json(raw) for raw in snapshot._sources)
        kind = sources[2].get("kind") if len(sources) > 2 and type(sources[2]) is dict else None
        if snapshot.qualification != QUALIFICATION_FOR_KIND.get(kind, "unqualified"):
            raise ValidationError("snapshot qualification drift")
    return snapshot


def snapshot_identity(snapshot):
    value = asdict(snapshot)
    del value["_sources"]
    return value
