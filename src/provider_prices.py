"""Finite offline price snapshots. No real route is qualified or contacted."""
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
              "audio_output", "image_output", "image_token"}
PROFILE_FIELDS = {"kind", "catalog_sha256", "endpoints_sha256", "model_id",
                  "canonical_model_slug", "endpoint_tag", "input_bound", "output_bound",
                  "reasoning_bound", "cache_read_bound", "cache_write_bound",
                  "structural_zero", "no_incremental_infrastructure", "no_metered_agent"}


def validate_snapshot(raw_sources, route, observed_at, expires_at):
    """Parse real catalog+endpoint bytes, optionally with an explicitly fake profile.

    An offline profile supplies synthetic enforcement evidence ONLY for MockTransport.
    It cannot qualify live prices, invoices or transport. Real snapshots stay unqualified.
    """
    if type(route) is not Route or type(raw_sources) is not tuple or len(raw_sources) not in (2, 3):
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
            variants = [e for e in endpoints if e["tag"].startswith(route.endpoint_tag + "/") and not e["tag"].endswith(("/fast", "/flex"))]
            if variants:
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
            if type(override) is not dict or "min_prompt_tokens" not in override or set(override) - PRICE_KEYS - {"min_prompt_tokens"}:
                raise ValidationError("unsupported price override")
            integer(override["min_prompt_tokens"])
            for key in override.keys() & PRICE_KEYS:
                rates[key] = max(rates.get(key, Decimal(0)), money(override[key]))
        if rates.get("request", Decimal(0)) != 0:
            raise ValidationError("fixed request charge unsupported")
        components = tuple(RateComponent(k, str(v), "request" if k in {"request", "web_search"} else "token",
                                         "maximum over all declared context tiers; discount not subtracted", refs[1])
                           for k, v in sorted(rates.items()))
        qualification = "unqualified"
        if len(sources) == 3:
            profile = sources[2]
            exact_fields(profile, PROFILE_FIELDS)
            if profile["kind"] != "SYNTHETIC OFFLINE ONLY" or profile["catalog_sha256"] != refs[0].sha256 or profile["endpoints_sha256"] != refs[1].sha256:
                raise ValidationError("synthetic source binding")
            for key in ("model_id", "canonical_model_slug", "endpoint_tag"):
                if profile[key] != getattr(route, key):
                    raise ValidationError("synthetic route binding")
            for key in ("input_bound", "output_bound", "reasoning_bound", "cache_read_bound", "cache_write_bound"):
                integer(profile[key])
            if profile["output_bound"] < 1 or profile["reasoning_bound"] > profile["output_bound"]:
                raise ValidationError("invalid combined completion bound")
            if max(profile["cache_read_bound"], profile["cache_write_bound"]) > profile["input_bound"]:
                raise ValidationError("invalid cache subset bound")
            if profile["no_incremental_infrastructure"] is not True or profile["no_metered_agent"] is not True:
                raise ValidationError("unknown fixture execution cost")
            zeros = profile["structural_zero"]
            if type(zeros) is not list or len(zeros) != len(set(zeros)) or set(zeros) - {"reasoning_tokens", "cache_read_tokens", "cache_write_tokens"}:
                raise ValidationError("invalid structural-zero proof")
            for kind in zeros:
                key = {"reasoning_tokens": "reasoning_bound", "cache_read_tokens": "cache_read_bound", "cache_write_tokens": "cache_write_bound"}[kind]
                if profile[key] != 0:
                    raise ValidationError("structural-zero contradicts bound")
            for kind, key in (("cache_read_tokens", "input_cache_read"), ("cache_write_tokens", "input_cache_write")):
                if key not in rates and kind not in zeros:
                    raise ValidationError("unknown applicable cache price")
            if "request" not in rates:
                raise ValidationError("unknown request price")
            qualification = "offline_fixture"
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValidationError("malformed catalog or endpoint metadata") from exc
    return PriceSnapshot(route, refs, observed_at, expires_at, components, qualification,
                         refs[2:] if qualification == "offline_fixture" else (), raw_sources)


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
    return snapshot


def snapshot_identity(snapshot):
    value = asdict(snapshot)
    del value["_sources"]
    return value
