"""Coverage statistics for a threshold against sampled errors."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri


def wilson_interval(successes, n):
    """Return the 95% Wilson interval for an empirical coverage."""
    z = float(ndtri(0.975))
    coverage = successes / n
    denominator = 1.0 + z * z / n
    center = (coverage + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        coverage * (1.0 - coverage) / n + z * z / (4.0 * n * n)
    ) / denominator
    # The endpoints at 0 and n successes are exactly 0 and 1; set them to avoid rounding.
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return float(low), float(high)


def coverage_summary(values, threshold, p):
    """Compare sampled absolute errors with a threshold at nominal level p.

    Returns the empirical coverage with its Wilson interval, the empirical
    p-quantile and the ratio of the threshold to that quantile.
    """
    values = np.asarray(values, dtype=float)
    threshold = float(threshold)
    n = len(values)
    successes = int(np.count_nonzero(values <= threshold))
    low, high = wilson_interval(successes, n)
    quantile = float(np.quantile(values, p, method="linear"))
    return {
        "n": n,
        "successes": successes,
        "failures": n - successes,
        "coverage": successes / n,
        "coverage_wilson95_low": low,
        "coverage_wilson95_high": high,
        "threshold": threshold,
        "empirical_quantile": quantile,
        "threshold_over_empirical_quantile": threshold / quantile,
    }
