"""What a run can cost, as a bound rather than a guess — and the gate that enforces it.

Two numbers, and the difference is the point:

    worst case   every rollout fills `max_tokens`, and no prefill token is ever cached
    expected     completions at the length measured on the base model

A spend approval should be given against the worst case, because that is the number the run cannot
exceed. A quote that offers only the expected figure is not a quote.

Rates are per million tokens and are pinned in `config/prices.json`, fetched from the provider's
machine-readable file, so the arithmetic can be re-checked offline. The prompt length doubles as a
renderer check: the thinking-disabled renderer and the probe agree to the token, while the recommended
renderer is shorter because it prefills an open `<think>`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "prices.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
MILLION = 1_000_000

# Measured on the merged corpus, not assumed. Prompt: 80 free-decoded generations produced 14704 tokens.
# Completion: the T=1.0 arm, because that is the configuration the run uses.
MEASURED_PROMPT_TOKENS = 14704 / 80
MEASURED_COMPLETION_TOKENS = 2741 / 48


@dataclass(frozen=True)
class Rates:
    """Published per-million-token rates in USD."""

    prefill: float
    cached_prefill: float
    sample: float
    train: float


def _price(value: str | float) -> float:
    return float(str(value).lstrip("$").replace(",", ""))


def load_rates(path: Path = CONFIG) -> Rates:
    """Read the pinned row for the model this harness trains. A missing row is fatal, not defaulted."""
    snapshot = json.loads(path.read_text())
    row = next((entry for entry in snapshot if entry.get("tinker_id") == MODEL_ID), None)
    if row is None:
        raise ValueError(f"{path.name} carries no row for {MODEL_ID}")
    return Rates(
        prefill=_price(row["prefill"]),
        cached_prefill=_price(row["cached_prefill"]),
        sample=_price(row["sample"]),
        train=_price(row["train"]),
    )


def per_rollout(completion_tokens: float, rates: Rates, prompt_tokens: float = MEASURED_PROMPT_TOKENS) -> float:
    """One rollout: its prefill, its sample, and the same tokens once through train."""
    return (
        prompt_tokens * rates.prefill
        + completion_tokens * rates.sample
        + (prompt_tokens + completion_tokens) * rates.train
    ) / MILLION


def bound(steps: int, batch: int, group: int, rates: Rates, max_tokens: int) -> dict:
    """Worst case and expected for a run of `steps` batches of `batch` groups of `group`."""
    rollouts = steps * batch * group
    worst = per_rollout(max_tokens, rates) * rollouts
    expected = per_rollout(MEASURED_COMPLETION_TOKENS, rates) * rollouts
    return {
        "steps": steps,
        "batch": batch,
        "group": group,
        "rollouts": rollouts,
        "max_tokens": max_tokens,
        "worst_case_usd": round(worst, 2),
        "expected_usd": round(expected, 2),
        "per_step_worst_usd": round(worst / steps, 4) if steps else None,
        "rates": rates.__dict__,
        "measured": {
            "prompt_tokens": round(MEASURED_PROMPT_TOKENS, 1),
            "completion_tokens": round(MEASURED_COMPLETION_TOKENS, 1),
        },
    }


def require_approval(report: dict, approved_usd: float | None) -> None:
    """Refuse to spend without a figure, or above one. Both failures name the number."""
    if approved_usd is None:
        raise SystemExit(
            f"no approval figure set; this run's worst case is ${report['worst_case_usd']:.2f} "
            f"(expected ${report['expected_usd']:.2f}). Set the figure you are approving."
        )
    if report["worst_case_usd"] > approved_usd:
        raise SystemExit(
            f"worst case ${report['worst_case_usd']:.2f} exceeds the approved ${approved_usd:.2f}; "
            "reduce the step count, the group width, or the token cap — or approve more"
        )
