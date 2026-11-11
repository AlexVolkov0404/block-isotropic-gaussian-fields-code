"""Variance envelope, error thresholds and probability bounds.

Notation follows the manuscript. For a profile h, a radius R and a block dimension D,

    V_h(R)    pointwise variance envelope,
    C_3       2 sqrt(120),
    kappa_D   sqrt(2) Gamma((D + 1) / 2) / Gamma(D / 2),
    sigma_R   sqrt(2) C_3 / 3 * sqrt(gamma_3) R^3,
    a_D(R)    2 C_3 kappa_D sqrt(gamma_3) R^3,
    s_h(R)    min(sqrt(V_h(R)), sigma_R).

The pointwise bounds concern the error at one fixed input pair in the product of
balls of radius R. The simultaneous bounds concern its supremum over that product.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, gammaln, ndtri

C3 = 2.0 * math.sqrt(120.0)
SIGMA_FACTOR = math.sqrt(2.0) * C3 / 3.0


def kappa(dimension):
    """Return kappa_D, the expected Euclidean norm of a standard Gaussian vector in R^D."""
    dimension = np.asarray(dimension, dtype=float)
    if np.any(dimension < 1) or np.any(dimension != np.floor(dimension)):
        raise ValueError("Dimensions must be positive integers.")
    return np.sqrt(2.0) * np.exp(
        gammaln((dimension + 1.0) / 2.0) - gammaln(dimension / 2.0)
    )


def variance_envelope_terms(profile, radius):
    """Return the three terms A, B and C of V_h(R) = A + B + C."""
    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0):
        raise ValueError("R must be nonnegative.")

    h1 = float(profile.derivative(0.0, 1))
    h2 = float(profile.derivative(0.0, 2))
    t = 2.0 * radius**2

    a = 2.0 * profile.one_minus_h(8.0 * radius**2)
    b = 16.0 * radius**2 * np.maximum(-h1 + 2.0 * profile.derivative(t, 1), 0.0)
    c = 16.0 * radius**4 * np.maximum(h2 - 2.0 * profile.derivative(t, 2), 0.0)
    return a, b, c


def variance_envelope(profile, radius):
    """Evaluate the pointwise variance envelope V_h(R)."""
    a, b, c = variance_envelope_terms(profile, radius)
    return a + b + c


def activation_radii(profile):
    """Return the radii at which the terms B and C of V_h(R) become positive."""
    h1 = float(profile.derivative(0.0, 1))
    h2 = float(profile.derivative(0.0, 2))
    coefficients = (
        lambda r: -h1 + 2.0 * float(profile.derivative(2.0 * r * r, 1)),
        lambda r: h2 - 2.0 * float(profile.derivative(2.0 * r * r, 2)),
    )

    roots = []
    for coefficient in coefficients:
        upper = profile.ell
        while coefficient(upper) <= 0.0:
            upper *= 2.0
            if upper / profile.ell > 1e8:
                raise RuntimeError("Failed to bracket an activation radius.")
        roots.append(
            brentq(coefficient, 0.0, upper, xtol=2e-14 * profile.ell, rtol=1e-13)
        )
    return tuple(roots)


def cubic_sd(profile, radius):
    """Return sigma_R, the cubic bound on the error standard deviation."""
    radius = np.asarray(radius, dtype=float)
    return SIGMA_FACTOR * math.sqrt(profile.gamma3) * radius**3


def pointwise_sd(profile, radius):
    """Return s_h(R) = min(sqrt(V_h(R)), sigma_R)."""
    envelope_sd = np.sqrt(np.maximum(variance_envelope(profile, radius), 0.0))
    return np.minimum(envelope_sd, cubic_sd(profile, radius))


def mean_supremum_bound(profile, radius, dimension):
    """Return a_D(R), the bound on the expected supremum of the error."""
    radius = np.asarray(radius, dtype=float)
    return 2.0 * C3 * kappa(dimension) * math.sqrt(profile.gamma3) * radius**3


def pointwise_threshold(profile, radius, p):
    """Return b_pw(p, R), the absolute-error threshold at field confidence p."""
    return ndtri((1.0 + p) / 2.0) * pointwise_sd(profile, radius)


def simultaneous_threshold(profile, radius, dimension, p):
    """Return b_sim(p, D, R), the threshold for the supremum at field confidence p."""
    tail = np.sqrt(-2.0 * np.log1p(-np.asarray(p, dtype=float)))
    return mean_supremum_bound(profile, radius, dimension) + cubic_sd(profile, radius) * tail


def field_probability_bounds(profile, radius, dimension, epsilon):
    """Return lower bounds on the field probability of an absolute error at most epsilon.

    The three bounds are the pointwise bound from V_h(R) alone, the refined
    pointwise bound from s_h(R), and the simultaneous bound.
    """
    radius = np.asarray(radius, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("R must be positive.")

    envelope_sd = np.sqrt(np.maximum(variance_envelope(profile, radius), 0.0))
    sigma = cubic_sd(profile, radius)
    refined_sd = np.minimum(envelope_sd, sigma)
    gap = np.maximum(epsilon - mean_supremum_bound(profile, radius, dimension), 0.0)

    pointwise_envelope = erf(epsilon / (np.sqrt(2.0) * envelope_sd))
    pointwise_refined = erf(epsilon / (np.sqrt(2.0) * refined_sd))
    simultaneous = -np.expm1(-0.5 * (gap / sigma) ** 2)
    return pointwise_envelope, pointwise_refined, simultaneous


def admissible_radii(profile, dimension, epsilon, confidence):
    """Return the largest radii at which each bound reaches the field confidence.

    The result holds the radii for the pointwise envelope bound, the refined
    pointwise bound and the simultaneous bound. Only the last depends on D.
    """
    z = ndtri((1.0 + confidence) / 2.0)
    target_variance = (epsilon / z) ** 2

    upper = profile.ell
    while variance_envelope(profile, upper) < target_variance:
        upper *= 2.0
    envelope_radius = brentq(
        lambda r: variance_envelope(profile, r) - target_variance,
        0.0, upper, xtol=1e-13 * profile.ell,
    )

    sqrt_gamma3 = math.sqrt(profile.gamma3)
    cubic_radius = np.cbrt(epsilon / (z * SIGMA_FACTOR * sqrt_gamma3))
    refined_radius = max(envelope_radius, cubic_radius)

    tail = np.sqrt(-2.0 * np.log1p(-confidence))
    simultaneous_radius = np.cbrt(
        epsilon / (sqrt_gamma3 * (2.0 * C3 * kappa(dimension) + SIGMA_FACTOR * tail))
    )
    return envelope_radius, refined_radius, simultaneous_radius
