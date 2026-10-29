"""Shared W6 study fixtures.

Kept out of conftest.py so the W6 suites own their own data shape. `manifest()`
returns a COMPLETE preregistration, because `analyse()` now validates the manifest
itself (review finding 10).
"""

ARMS = ("A", "B", "C", "D")


def manifest(**overrides):
    doc = {
        "schema_version": 1,
        "study_id": "revision-v1",
        "kind": "PREREGISTRATION",
        "status": "unfrozen",
        "roster": ["vendor/model-one", "vendor/model-two"],
        "roster_status": "candidates_pending_owner_approval",
        "sample_size": None,
        "estimand": {
            "primary": "weighted mean within-cluster paired D-minus-A resolved difference",
            "locale_weights": "equal",
            "arm_control": "A",
            "arm_treatment": "D",
        },
        "interval": {"method": "hierarchical_paired_cluster_bootstrap", "seed": 7,
                     "resamples": 1000, "confidence": 0.95},
        "benefit_gate": {"observed_lift_gate": 0.10, "lower_bound_gate": 0.0},
        "quality_safeguard": {"observed_gate": 0.50, "lower_bound_gate": 0.45},
        "missingness": {"policy": "full assigned denominator; worst/best bounds",
                        "report_full_denominator": True, "worst_best_bounds": True},
        "valid_assessment": {"definition": "paired outcome assessable under authority gates",
                             "unassessable_is_not_a_tie": True},
        "coverage_threshold": 0.90,
        "retry_limits": {"max_retries": 3, "new_attempt_id_per_retry": True,
                         "fixed_on_all_arms": True},
        "stop_criteria": ["predeclared cap reached", "frozen-version drift"],
        "multiplicity": {"policy": "one primary gate; B/C secondary and corrected"},
        "power_simulation": {"baseline_rate": 0.55, "true_lift": 0.10,
                             "cluster_sd": 0.15, "unassessable_rate": 0.05},
    }
    for path, value in overrides.items():
        keys = path.split(".")
        node = doc
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return doc


def unit(cluster, locale="en", a=True, d=True, a_invalid=None, d_invalid=None,
         quality=None, cost="0", sample="s0", breach=False, model="vendor/model-one"):
    def observation(arm, resolved, invalid_kind, qual):
        return {"resolved": resolved, "invalid_kind": invalid_kind, "quality": qual,
                "authority_breach": breach, "cost_usd": cost, "latency_ms": 1}

    arms = {arm: observation(arm, None, None, None) for arm in ("B", "C")}
    arms["A"] = observation("A", a, a_invalid, None)
    arms["D"] = observation("D", d, d_invalid, quality)
    return {"cluster_id": cluster, "model": model, "locale": locale,
            "task_family": "f", "sample_id": sample, "arms": arms}
