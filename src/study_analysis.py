"""W6 offline pilot analysis: preregistration validation, paired cluster analysis,
quality safeguard, missingness bounds and power simulation.

No provider is contacted and no credential is read. Every number this module
produces is either derived from records on disk or from a seeded simulation.

Contract references (story-bench/docs/experiment-contract.md):
  § 7  binary terminal reward; infrastructure/grader-invalid is not a task-negative label
  § 8  primary estimand, hierarchical paired cluster bootstrap, benefit policy,
       quality safeguard Q=(wins+0.5 ties)/valid assessed pairs, missingness handling
  § 9  all-attempt cost accounting
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:  # pyyaml is an existing declared dependency (config/arms.yaml)
    import yaml
except ImportError:  # pragma: no cover - declared dependency must be present
    yaml = None

ARMS = ("A", "B", "C", "D")
LOCALES = ("en", "zh-Hant", "zh-Hans")
QUALITY_OUTCOMES = ("win", "tie", "loss", "cannot_assess")
INVALID_KINDS = (None, "absent", "provider", "infrastructure", "grader")

# Provider/infrastructure/grader failures are not task-negative labels and are
# handled by missingness bounds rather than the point estimate (contract § 7/§ 8).
# An agent-caused absent submission is different: parse_units forces it to
# resolved=None, so it counts as unresolved rather than being silently dropped.
EXCLUDED_FROM_POINT_ESTIMATE = ("provider", "infrastructure", "grader")


class ManifestError(ValueError):
    """Raised when a study manifest or record set cannot be used."""


# --------------------------------------------------------------------------- manifest

REQUIRED_MANIFEST_PATHS = (
    ("schema_version",), ("study_id",), ("kind",), ("status",),
    ("estimand", "primary"), ("estimand", "locale_weights"),
    ("estimand", "arm_control"), ("estimand", "arm_treatment"),
    ("interval", "method"), ("interval", "seed"), ("interval", "resamples"),
    ("interval", "confidence"), ("roster",),
    ("benefit_gate", "observed_lift_gate"), ("benefit_gate", "lower_bound_gate"),
    ("quality_safeguard", "observed_gate"), ("quality_safeguard", "lower_bound_gate"),
    ("missingness", "policy"), ("missingness", "report_full_denominator"),
    ("missingness", "worst_best_bounds"),
    ("valid_assessment", "definition"), ("valid_assessment", "unassessable_is_not_a_tie"),
    ("coverage_threshold",),
    ("retry_limits", "max_retries"), ("retry_limits", "new_attempt_id_per_retry"),
    ("retry_limits", "fixed_on_all_arms"),
    ("stop_criteria",), ("multiplicity", "policy"),
)

FROZEN_VALUES = {
    ("estimand", "primary"): "weighted mean within-cluster paired D-minus-A resolved difference",
    ("estimand", "locale_weights"): "equal",
    ("estimand", "arm_control"): "A",
    ("estimand", "arm_treatment"): "D",
    ("interval", "method"): "hierarchical_paired_cluster_bootstrap",
    ("valid_assessment", "unassessable_is_not_a_tie"): True,
    ("retry_limits", "new_attempt_id_per_retry"): True,
    ("retry_limits", "fixed_on_all_arms"): True,
}


def _dig(doc, path, default=None):
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_manifest(path):
    if yaml is None:  # pragma: no cover
        raise ManifestError("pyyaml is required to read a study manifest")
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ManifestError("manifest must be a mapping")
    return doc


def validate_manifest(doc):
    """Return a list of problems. Empty list == preregistration is complete."""
    if not isinstance(doc, dict):
        return ["manifest must be a mapping"]
    problems = []
    for path in REQUIRED_MANIFEST_PATHS:
        value = _dig(doc, path)
        if value is None or value == "" or value == []:
            problems.append(f"{'.'.join(path)}: required preregistration field is missing")
    if problems:
        return problems

    for path, expected in FROZEN_VALUES.items():
        actual = _dig(doc, path)
        if actual != expected:
            problems.append(
                f"{'.'.join(path)}: frozen policy requires {expected!r}, got {actual!r}")

    if doc["status"] not in ("unfrozen", "frozen"):
        problems.append("status: must be 'unfrozen' or 'frozen'")
    if doc["kind"] not in ("PREREGISTRATION", "SYNTHETIC EXAMPLE"):
        problems.append("kind: must be 'PREREGISTRATION' or 'SYNTHETIC EXAMPLE'")

    roster = doc["roster"]
    if not isinstance(roster, list) or len(roster) < 2:
        problems.append("roster: at least two named models are required")
    elif any(not isinstance(m, str) or not m.strip() for m in roster):
        problems.append("roster: every entry must be a non-empty model id")
    elif len(set(roster)) != len(roster):
        problems.append("roster: duplicate model ids are not allowed")

    seed = _dig(doc, ("interval", "seed"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append("interval.seed: an integer seed must be preregistered")
    resamples = _dig(doc, ("interval", "resamples"))
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1000:
        problems.append("interval.resamples: preregister at least 1000 resamples")
    confidence = _dig(doc, ("interval", "confidence"))
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or not 0.5 < confidence < 1.0:
        problems.append("interval.confidence: must lie strictly between 0.5 and 1.0")

    if doc["status"] == "frozen" and _dig(doc, ("sample_size",)) is None:
        problems.append("sample_size: a frozen study must carry a frozen sample size")
    roster_status = _dig(doc, ("roster_status",))
    if roster_status not in ("pending_research", "candidates_pending_owner_approval",
                             "owner_approved"):
        problems.append("roster_status: one of pending_research, "
                        "candidates_pending_owner_approval, owner_approved")
    elif doc["status"] == "frozen" and roster_status != "owner_approved":
        problems.append(f"roster_status: a frozen study needs 'owner_approved', "
                        f"got {roster_status!r}")
    if doc["kind"] == "SYNTHETIC EXAMPLE" and doc["status"] == "frozen":
        problems.append("kind: a synthetic example may never be frozen as a study")
    return problems


def assert_manifest(doc):
    problems = validate_manifest(doc)
    if problems:
        raise ManifestError("; ".join(problems))
    return doc


# ----------------------------------------------------------------------------- records

@dataclass(frozen=True)
class Observation:
    arm: str
    resolved: bool | None
    invalid_kind: str | None
    quality: str | None
    authority_breach: bool
    cost_usd: Decimal
    latency_ms: int


@dataclass(frozen=True)
class Unit:
    """One matched (cluster, model, locale, sample) unit holding all four arms."""
    cluster_id: str
    model: str
    locale: str
    task_family: str
    sample_id: str
    arms: dict

    def observation(self, arm):
        return self.arms[arm]


def load_records(path):
    """Read JSONL (one unit object per line) or a JSON array of unit objects."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ManifestError(f"no records found in {path}")
    raw = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    return parse_units(raw)


def parse_units(raw):
    """Validate and convert raw unit mappings into immutable Unit objects."""
    units = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ManifestError(f"record {index}: object required")
        for key in ("cluster_id", "model", "locale", "task_family", "sample_id"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ManifestError(f"record {index}.{key}: non-empty text required")
        arms_raw = item.get("arms")
        if not isinstance(arms_raw, dict) or set(arms_raw) != set(ARMS):
            raise ManifestError(f"record {index}: exactly arms {list(ARMS)} required")
        observations = {}
        for arm in ARMS:
            payload = arms_raw[arm]
            if not isinstance(payload, dict):
                raise ManifestError(f"record {index}.{arm}: object required")
            resolved = payload.get("resolved")
            if resolved is not None and not isinstance(resolved, bool):
                raise ManifestError(f"record {index}.{arm}.resolved: bool or null")
            invalid_kind = payload.get("invalid_kind")
            if invalid_kind not in INVALID_KINDS:
                raise ManifestError(
                    f"record {index}.{arm}.invalid_kind: one of {INVALID_KINDS}")
            if invalid_kind is not None and resolved is not None:
                raise ManifestError(
                    f"record {index}.{arm}: invalid_kind and resolved cannot both be "
                    f"set; an invalid submission is unresolved, never a success")
            quality = payload.get("quality")
            if quality is not None and quality not in QUALITY_OUTCOMES:
                raise ManifestError(
                    f"record {index}.{arm}.quality: one of {QUALITY_OUTCOMES}")
            try:
                cost = Decimal(payload.get("cost_usd", "0"))
            except (InvalidOperation, TypeError):
                raise ManifestError(
                    f"record {index}.{arm}.cost_usd: decimal text required")
            if cost < 0 or not cost.is_finite():
                raise ManifestError(f"record {index}.{arm}.cost_usd: must be finite and >= 0")
            latency = payload.get("latency_ms", 0)
            if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
                raise ManifestError(f"record {index}.{arm}.latency_ms: non-negative integer")
            observations[arm] = Observation(
                arm=arm, resolved=resolved, invalid_kind=invalid_kind, quality=quality,
                authority_breach=bool(payload.get("authority_breach", False)),
                cost_usd=cost, latency_ms=latency)
        units.append(Unit(item["cluster_id"], item["model"], item["locale"],
                          item["task_family"], item["sample_id"], observations))
    return units


# ------------------------------------------------------------------ paired statistics

def _excluded(observation):
    return observation.invalid_kind in EXCLUDED_FROM_POINT_ESTIMATE


def _resolved_value(observation):
    """Agent-caused absence is unresolved; an explicit None with no invalid kind
    is treated the same way rather than silently dropped."""
    if observation.resolved is None:
        return 0.0
    return 1.0 if observation.resolved else 0.0


def _pair_difference(unit, control, treatment):
    """D-minus-A resolved difference, or None when a side is not task-assessable."""
    left, right = unit.observation(control), unit.observation(treatment)
    if _excluded(left) or _excluded(right):
        return None
    return _resolved_value(right) - _resolved_value(left)


def _quality_value(unit, control, treatment):
    """Quality is recorded D-relative: win/tie/loss against the control."""
    quality = unit.observation(treatment).quality
    if quality in ("win", "tie", "loss"):
        return {"win": 1.0, "tie": 0.5, "loss": 0.0}[quality]
    return None


def _pairs(units, function):
    values = []
    for unit in units:
        value = function(unit)
        if value is not None:
            values.append((unit.locale, unit.cluster_id, value))
    return values


def _cluster_means(pairs):
    buckets = {}
    for locale, cluster, value in pairs:
        buckets.setdefault((locale, cluster), []).append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items()}


def _locale_means(cluster_means):
    by_locale = {}
    for (locale, _cluster), mean in cluster_means.items():
        by_locale.setdefault(locale, []).append(mean)
    return {locale: sum(values) / len(values) for locale, values in by_locale.items()}


def _overall_equal_weight(locale_means):
    if not locale_means:
        return None
    return sum(locale_means.values()) / len(locale_means)


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _bootstrap(pairs, rng, resamples):
    """Hierarchical paired cluster bootstrap.

    Stage 1 resamples clusters with replacement inside each locale (locale weights
    stay equal). Stage 2 resamples that cluster's units with replacement, so both
    cluster-level and within-cluster dependence are represented.
    """
    by_locale = {}
    for locale, cluster, value in pairs:
        by_locale.setdefault(locale, {}).setdefault(cluster, []).append(value)
    locales = sorted(by_locale)
    draws = []
    for _ in range(resamples):
        locale_means = []
        for locale in locales:
            clusters = list(by_locale[locale].values())
            picked = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            sampled = []
            for values in picked:
                resampled = [values[rng.randrange(len(values))] for _ in values]
                sampled.append(sum(resampled) / len(resampled))
            locale_means.append(sum(sampled) / len(sampled))
        if locale_means:
            draws.append(sum(locale_means) / len(locale_means))
    return sorted(draws)


def _pooled_mean(pairs):
    if not pairs:
        return None
    return sum(value for _locale, _cluster, value in pairs) / len(pairs)


def _bootstrap_pooled(pairs, rng, resamples):
    """Cluster bootstrap of a POOLED statistic.

    Q is defined by the contract as a pooled ratio, so its interval must come from
    the pooled estimator. Review found the earlier version paired a pooled point
    estimate with a locale-equal-weighted interval, which produced bounds that
    were not bounds for the reported Q (a 1-win/100-loss split returned a lower
    bound of 0.50, and a 98%-win split was wrongly blocked by the safeguard).
    """
    by_cluster = {}
    for locale, cluster, value in pairs:
        by_cluster.setdefault((locale, cluster), []).append(value)
    keys = list(by_cluster)
    if not keys:
        return []
    draws = []
    for _ in range(resamples):
        pooled = []
        for _ in keys:
            pooled.extend(by_cluster[keys[rng.randrange(len(keys))]])
        draws.append(sum(pooled) / len(pooled))
    return sorted(draws)


def _envelope(unit, control, treatment):
    """(worst, best) D-minus-A bound for a unit excluded from the point estimate."""
    left, right = unit.observation(control), unit.observation(treatment)
    left_known = None if _excluded(left) else _resolved_value(left)
    right_known = None if _excluded(right) else _resolved_value(right)
    worst = (0.0 if right_known is None else right_known) - \
            (1.0 if left_known is None else left_known)
    best = (1.0 if right_known is None else right_known) - \
           (0.0 if left_known is None else left_known)
    return worst, best


def _missingness_bounds(units, control, treatment):
    """Worst/best-case bounds for every unit the point estimate had to exclude."""
    bounds = []
    for unit in units:
        value = _pair_difference(unit, control, treatment)
        if value is not None:
            bounds.append((unit.locale, unit.cluster_id, (value, value)))
        else:
            bounds.append((unit.locale, unit.cluster_id,
                           _envelope(unit, control, treatment)))

    def overall(index):
        cluster_means = {}
        for locale, cluster, pair in bounds:
            cluster_means.setdefault((locale, cluster), []).append(pair[index])
        means = {key: sum(values) / len(values) for key, values in cluster_means.items()}
        return _overall_equal_weight(_locale_means(means))

    return {"worst_case": overall(0), "best_case": overall(1)}


def _arm_resolved_summary(units, arm):
    """Pooled resolved count/rate for one arm over its assessable units (contract § 8)."""
    usable = [u for u in units if not _excluded(u.observation(arm))]
    resolved = sum(1 for u in usable if _resolved_value(u.observation(arm)) == 1.0)
    return {"assessable_units": len(usable), "resolved": resolved,
            "rate": (resolved / len(usable)) if usable else None}


def _status_breakdown(units):
    """Provider/infrastructure/grader/absent counts per arm, never folded into 'failed'."""
    breakdown = {}
    for arm in ARMS:
        counts = {kind: 0 for kind in INVALID_KINDS if kind is not None}
        for unit in units:
            kind = unit.observation(arm).invalid_kind
            if kind is not None:
                counts[kind] += 1
        breakdown[arm] = counts
    return breakdown


def _per_model(units, control, treatment):
    """Descriptive per-model heterogeneity. Deliberately interval-free: unadjusted
    per-model intervals would invite exactly the multiplicity overreach § 8 forbids."""
    models = {}
    for name in sorted({u.model for u in units}):
        subset = [u for u in units if u.model == name]
        pairs = _pairs(subset, lambda u: _pair_difference(u, control, treatment))
        models[name] = {
            "units_assigned": len(subset),
            "units_assessed": len(pairs),
            "clusters": len({(u.locale, u.cluster_id) for u in subset}),
            "observed_lift": _overall_equal_weight(_locale_means(_cluster_means(pairs))),
        }
    return models


def analyse(units, manifest, *, resamples=None):
    """Produce the W6 analysis report for one study manifest.

    The manifest is validated here as well as in ``analyse_path``: review showed a
    caller could otherwise run an analysis labelled with a non-frozen estimand or
    interval method.
    """
    assert_manifest(manifest)
    if not units:
        raise ManifestError("no units to analyse")
    control = manifest["estimand"]["arm_control"]
    treatment = manifest["estimand"]["arm_treatment"]
    seed = manifest["interval"]["seed"]
    resamples = resamples or manifest["interval"]["resamples"]
    confidence = manifest["interval"]["confidence"]

    lift_pairs = _pairs(units, lambda u: _pair_difference(u, control, treatment))
    quality_pairs = _pairs(units, lambda u: _quality_value(u, control, treatment))

    observed = _overall_equal_weight(_locale_means(_cluster_means(lift_pairs)))
    tail = (1.0 - confidence) / 2.0
    draws = _bootstrap(lift_pairs, random.Random(seed), resamples)
    lower, upper = _percentile(draws, tail), _percentile(draws, 1.0 - tail)

    quality_observations = [u.observation(treatment).quality for u in units]
    wins = quality_observations.count("win")
    ties = quality_observations.count("tie")
    losses = quality_observations.count("loss")
    cannot = quality_observations.count("cannot_assess")
    q_observed = _pooled_mean(quality_pairs)
    q_draws = _bootstrap_pooled(quality_pairs, random.Random(seed + 1), resamples)
    q_lower = _percentile(q_draws, tail)

    coverage = len(lift_pairs) / len(units)
    bounds = _missingness_bounds(units, control, treatment)

    lift_gate = float(manifest["benefit_gate"]["observed_lift_gate"])
    lift_lower_gate = float(manifest["benefit_gate"]["lower_bound_gate"])
    q_gate = float(manifest["quality_safeguard"]["observed_gate"])
    q_lower_gate = float(manifest["quality_safeguard"]["lower_bound_gate"])
    coverage_gate = float(manifest["coverage_threshold"])

    straddles = bounds["worst_case"] < lift_gate <= bounds["best_case"]
    met = (observed is not None and observed >= lift_gate
           and lower is not None and lower > lift_lower_gate
           and q_observed is not None and q_observed >= q_gate
           and q_lower is not None and q_lower > q_lower_gate
           and coverage >= coverage_gate)
    if straddles:
        benefit_status = "inconclusive_missingness"
    elif observed is None or lower is None or q_observed is None:
        benefit_status = "inconclusive"
    elif met:
        benefit_status = "met"
    else:
        benefit_status = "not_met"

    return {
        "study_id": manifest["study_id"],
        "kind": manifest["kind"],
        "manifest_status": manifest["status"],
        "estimand": manifest["estimand"]["primary"],
        "control_arm": control,
        "treatment_arm": treatment,
        "units_assigned": len(units),
        "units_assessed": len(lift_pairs),
        "clusters": len({(u.locale, u.cluster_id) for u in units}),
        "coverage": coverage,
        "coverage_threshold": coverage_gate,
        "observed_lift": observed,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval": {"method": manifest["interval"]["method"], "resamples": resamples,
                     "seed": seed, "confidence": confidence},
        "locale_means": _locale_means(_cluster_means(lift_pairs)),
        "resolved": {"control": _arm_resolved_summary(units, control),
                     "treatment": _arm_resolved_summary(units, treatment)},
        "status_breakdown": _status_breakdown(units),
        "per_model": _per_model(units, control, treatment),
        "quality": {"wins": wins, "ties": ties, "losses": losses,
                    "cannot_assess": cannot, "Q": q_observed, "Q_lower_bound": q_lower,
                    "valid_assessed_pairs": len(quality_pairs)},
        "authority_breaches": sum(1 for u in units for arm in ARMS
                                  if u.observation(arm).authority_breach),
        "all_attempt_cost_usd": str(sum((u.observation(arm).cost_usd for u in units
                                         for arm in ARMS), Decimal(0))),
        "latency_ms_total": sum(u.observation(arm).latency_ms for u in units
                                for arm in ARMS),
        "missingness": bounds,
        "benefit_status": benefit_status,
        "limits": [
            "A negative or inconclusive result completes the study; it is not a failure.",
            "The observed lift is not proof that the true effect reaches the gate.",
            "Q is a preference probability, not a literary-quality score.",
            "Provider/infrastructure/grader-invalid units are excluded from the point "
            "estimate and bounded, never counted as task failures.",
            "Model families are fixed strata; generalization beyond the roster is limited.",
            "Per-model lifts are descriptive and carry no interval: unadjusted "
            "per-model intervals would be a multiplicity overreach.",
            "Resolved rates in this report are pooled, not locale-equal-weighted; "
            "the primary estimand is the locale-equal-weighted paired difference.",
        ],
    }


def analyse_path(manifest_path, results_path, *, resamples=None):
    manifest = assert_manifest(load_manifest(manifest_path))
    return analyse(load_records(results_path), manifest, resamples=resamples)


# ------------------------------------------------------------------------------ power

@dataclass(frozen=True)
class PowerPoint:
    clusters_per_locale: int
    units_per_cluster: int
    power: float
    mean_interval_width: float
    mean_observed_lift: float | None


def _simulated_unit(rng, cluster_id, locale, sample_id, baseline, lift, cluster_sd,
                    unassessable_rate):
    intercept = rng.gauss(0.0, cluster_sd)

    def draw(rate):
        if rng.random() < unassessable_rate:
            return None
        probability = min(max(rate + intercept, 0.0), 1.0)
        return rng.random() < probability

    control = draw(baseline)
    treatment = draw(min(baseline + lift, 1.0))
    arms = {}
    for arm in ARMS:
        resolved = control if arm == "A" else treatment if arm == "D" else None
        # A simulated unassessable draw is agent-caused absence: it counts as
        # unresolved (contract § 8) rather than being dropped, and it must not carry
        # a resolved outcome alongside an invalid_kind (parse_units rejects that).
        absent = "absent" if resolved is None else None
        arms[arm] = Observation(arm=arm, resolved=resolved, invalid_kind=absent,
                                quality=None, authority_breach=False,
                                cost_usd=Decimal(0), latency_ms=0)
    return Unit(cluster_id, "simulated", locale, "simulated", sample_id, arms)


def simulate_units(*, baseline_rate, true_lift, cluster_sd, unassessable_rate,
                   clusters_per_locale, units_per_cluster, seed,
                   quality_win_rate=None, quality_tie_rate=0.2,
                   cost_per_unit="0", latency_ms=0, label="simulated"):
    """Generate one synthetic dataset of matched four-arm units.

    Used by the power simulation and by the clearly-labelled synthetic example in
    ``studies/examples/``. Quality outcomes stay absent unless ``quality_win_rate``
    is supplied, so a quantity the caller did not model can never appear as data.
    """
    rng = random.Random(seed)
    cost = Decimal(cost_per_unit)
    units = []
    for locale in LOCALES:
        for cluster in range(clusters_per_locale):
            for unit_index in range(units_per_cluster):
                unit = _simulated_unit(rng, f"c{cluster}-{locale}", locale,
                                       f"s{unit_index}", baseline_rate, true_lift,
                                       cluster_sd, unassessable_rate)
                quality = None
                if quality_win_rate is not None:
                    draw = rng.random()
                    quality = ("win" if draw < quality_win_rate else
                               "tie" if draw < quality_win_rate + quality_tie_rate
                               else "loss")
                arms = {}
                for arm in ARMS:
                    observation = unit.observation(arm)
                    arms[arm] = Observation(
                        arm=arm, resolved=observation.resolved,
                        invalid_kind=observation.invalid_kind,
                        quality=quality if arm == "D" else None,
                        authority_breach=False, cost_usd=cost, latency_ms=latency_ms)
                units.append(Unit(unit.cluster_id, label, locale, label,
                                  unit.sample_id, arms))
    return units


def power_simulation(*, baseline_rate, true_lift, cluster_sd, unassessable_rate,
                     clusters_per_locale, units_per_cluster, seed, trials=200,
                     resamples=200, gate=0.10, lower_gate=0.0, confidence=0.95):
    """Seeded simulation of the primary resolved gate at one candidate sample size.

    The simulation covers the PRIMARY estimand only. It deliberately does not claim
    to model the quality safeguard, so its power is an upper bound on the combined
    gate and must not be reported as combined power.
    """
    met, widths, lifts = 0, [], []
    tail = (1.0 - confidence) / 2.0
    for trial in range(trials):
        units = simulate_units(
            baseline_rate=baseline_rate, true_lift=true_lift, cluster_sd=cluster_sd,
            unassessable_rate=unassessable_rate,
            clusters_per_locale=clusters_per_locale,
            units_per_cluster=units_per_cluster, seed=seed * 1000 + trial)
        pairs = _pairs(units, lambda u: _pair_difference(u, "A", "D"))
        if not pairs:
            widths.append(0.0)
            continue
        observed = _overall_equal_weight(_locale_means(_cluster_means(pairs)))
        draws = _bootstrap(pairs, random.Random(seed + 7), resamples)
        lower = _percentile(draws, tail)
        upper = _percentile(draws, 1.0 - tail)
        lifts.append(observed)
        widths.append((upper or 0.0) - (lower or 0.0))
        if observed >= gate and lower is not None and lower > lower_gate:
            met += 1
    return {
        "power": met / trials,
        "mean_interval_width": sum(widths) / len(widths),
        "mean_observed_lift": (sum(lifts) / len(lifts)) if lifts else None,
        "gate": gate,
        "lower_gate": lower_gate,
        "trials": trials,
        # Declared in the RESULT, not only in a docstring: a consumer reading the
        # number must see which gates it does and does not stand in for.
        "covers": {"resolved_gate": True, "quality_safeguard": False,
                   "coverage_gate": False, "missingness_suppression": False},
    }


def power_curve(manifest, grid, *, trials=200, resamples=200):
    """Evaluate a preregistered grid of candidate (clusters/locale, units/cluster)."""
    simulation = manifest.get("power_simulation", {})
    points = []
    for index, (clusters_per_locale, units_per_cluster) in enumerate(grid):
        result = power_simulation(
            baseline_rate=simulation.get("baseline_rate", 0.55),
            true_lift=simulation.get("true_lift", 0.10),
            cluster_sd=simulation.get("cluster_sd", 0.15),
            unassessable_rate=simulation.get("unassessable_rate", 0.05),
            clusters_per_locale=clusters_per_locale,
            units_per_cluster=units_per_cluster,
            seed=manifest["interval"]["seed"] + index,
            trials=trials, resamples=resamples,
            gate=float(manifest["benefit_gate"]["observed_lift_gate"]),
            lower_gate=float(manifest["benefit_gate"]["lower_bound_gate"]),
            confidence=manifest["interval"]["confidence"])
        points.append(PowerPoint(clusters_per_locale, units_per_cluster,
                                 result["power"], result["mean_interval_width"],
                                 result["mean_observed_lift"]))
    return points


def render_power_curve(points):
    lines = ["clusters/locale  units/cluster  power   mean 95% width  mean lift",
             "-" * 62]
    for point in points:
        lift = "n/a" if point.mean_observed_lift is None else f"{point.mean_observed_lift:.3f}"
        lines.append(f"{point.clusters_per_locale:>15}  {point.units_per_cluster:>13}  "
                     f"{point.power:>5.2f}  {point.mean_interval_width:>13.3f}  {lift:>9}")
    lines.append("")
    lines.append("Power covers the RESOLVED gate only. It does NOT cover the quality "
                 "safeguard, the coverage threshold or the missingness-straddle "
                 "suppression that the real benefit verdict also requires.")
    return "\n".join(lines)
