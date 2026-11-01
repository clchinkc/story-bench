"""Intervals and tests, in one place, so no report invents its own.

Two rules are encoded here rather than remembered in prose:

    cluster_bootstrap   the unit of analysis is the *task*, not the rollout. Four generations of one
                        passage share a passage, a truth and an error, so a rollout-level interval is
                        too narrow and would make every result look more certain than it is
    sign_test           with a handful of paired tasks, the exact test is the honest one; its floor is
                        `2/2**n` when every pair moves the same way, which is what "with eight tasks it
                        can only detect a near-uniform result" means numerically

`contrast` is the D019 group-uniformity model: `pass = (1 - e) ** breaks`, so a group is uniform with
probability `pass**group + (1-pass)**group` and the rest carry a gradient. It multiplies the error rate,
which is why a solved task cannot be made harder by adding breaks.
"""

from __future__ import annotations

import math
import random

Z95 = 1.959963984540054
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261101


def wilson(failures: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval — correct near 0 and 1, where a normal interval is not."""
    if trials <= 0:
        return (0.0, 1.0)
    phat = failures / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(
    per_task: dict[str, list[float]],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile interval over resampled **tasks**. Seeded, so a number is reproducible."""
    means = [sum(values) / len(values) for values in per_task.values() if values]
    if len(means) < 2:
        return (0.0, 1.0)
    rng = random.Random(seed)
    n = len(means)
    samples = sorted(sum(rng.choices(means, k=n)) / n for _ in range(draws))
    return (round(samples[int(0.025 * draws)], 4), round(samples[int(0.975 * draws)], 4))


def sign_test(improved: int, worsened: int) -> float:
    """Exact two-sided sign test over discordant pairs; ties are excluded, not counted as success."""
    n = improved + worsened
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(improved, worsened), n + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def contrast(error_rate: float, breaks: int, group: int) -> float | None:
    """Share of groups that carry a gradient, under the D019 break-density model."""
    if not 0.0 <= error_rate <= 1.0:
        return None
    passed = (1 - error_rate) ** breaks
    return round(1 - (passed**group + (1 - passed) ** group), 4)
