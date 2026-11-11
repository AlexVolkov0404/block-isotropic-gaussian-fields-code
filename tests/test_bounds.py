"""Variance envelope, thresholds and probability bounds."""

import math

import numpy as np
import pytest

from bigf import bounds, inputs, profiles
from bigf.covariance import error_variance

PROFILES = list(profiles.manuscript_profiles().values())
SMOOTH = [profile for profile in PROFILES if math.isfinite(profile.gamma3)]
RADII = np.geomspace(0.01, 4.0, 200)


def test_constants():
    assert bounds.C3 == 2.0 * math.sqrt(120.0)
    np.testing.assert_allclose(bounds.kappa(1), math.sqrt(2.0 / math.pi), rtol=1e-14)
    np.testing.assert_allclose(bounds.kappa(2), math.sqrt(math.pi / 2.0), rtol=1e-14)
    dimension = np.arange(1, 300)
    assert np.all(bounds.kappa(dimension) < np.sqrt(dimension))
    with pytest.raises(ValueError):
        bounds.kappa(1.5)


def test_squared_exponential_envelope_closed_form():
    profile = profiles.squared_exponential()
    active = np.maximum(1.0 - 2.0 * np.exp(-2.0 * RADII**2), 0.0)
    expected = (
        2.0 * (1.0 - np.exp(-8.0 * RADII**2))
        + 16.0 * RADII**2 * active + 16.0 * RADII**4 * active
    )
    np.testing.assert_allclose(bounds.variance_envelope(profile, RADII), expected, rtol=1e-12)
    np.testing.assert_allclose(
        bounds.activation_radii(profile), math.sqrt(math.log(2.0) / 2.0), rtol=1e-12
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_envelope_terms_switch_on_at_activation_radii(profile):
    r_b, r_c = bounds.activation_radii(profile)
    for index, radius in ((1, r_b), (2, r_c)):
        assert bounds.variance_envelope_terms(profile, 0.99 * radius)[index] == 0.0
        assert bounds.variance_envelope_terms(profile, 1.01 * radius)[index] > 0.0
    a, b, c = bounds.variance_envelope_terms(profile, RADII)
    np.testing.assert_array_equal(bounds.variance_envelope(profile, RADII), a + b + c)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_envelope_limits(profile):
    # V_h(R) = -16 h'(0) R^2 + O(R^4) as R -> 0 and 16 h''(0) R^4 + o(R^4) as R -> infinity.
    h1, h2 = float(profile.derivative(0.0, 1)), float(profile.derivative(0.0, 2))
    np.testing.assert_allclose(
        bounds.variance_envelope(profile, 1e-4), -16.0 * h1 * 1e-8, rtol=1e-6
    )
    np.testing.assert_allclose(
        bounds.variance_envelope(profile, 1e3), 16.0 * h2 * 1e12, rtol=1e-5
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("dimension", (2, 8, 32))
def test_envelope_dominates_exact_variance(profile, dimension):
    rng = np.random.default_rng(0)
    for radius in (0.05, 0.3, 1.0, 3.0):
        directions = rng.normal(size=(400, 2, dimension))
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        lengths = radius * rng.random(size=(400, 2, 1)) ** (1.0 / dimension)
        lengths[:200] = radius
        x = (directions * lengths).reshape(400, 2 * dimension)
        x[0] = 0.0
        x[0, 0], x[0, dimension] = radius, -radius
        envelope = bounds.variance_envelope(profile, radius)
        assert np.all(error_variance(x, profile) <= envelope * (1.0 + 1e-12))


@pytest.mark.parametrize("profile", SMOOTH, ids=lambda p: p.name)
def test_cubic_sd_dominates_exact_variance(profile):
    # sigma_R^2 bounds the error variance on the product of balls of radius R.
    rng = np.random.default_rng(1)
    dimension = 8
    for radius in (0.05, 0.15, 0.5):
        directions = rng.normal(size=(400, 2, dimension))
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        x = (radius * directions).reshape(400, 2 * dimension)
        x[0] = 0.0
        x[0, 0], x[0, dimension] = radius, -radius
        assert np.all(error_variance(x, profile) <= bounds.cubic_sd(profile, radius) ** 2)


@pytest.mark.parametrize("profile", SMOOTH, ids=lambda p: p.name)
def test_scales_match_manuscript_formulas(profile):
    radius, dimension = 0.15, 32
    root = math.sqrt(profile.gamma3)
    kappa = math.sqrt(2.0) * math.gamma((dimension + 1) / 2) / math.gamma(dimension / 2)
    np.testing.assert_allclose(
        bounds.cubic_sd(profile, radius),
        math.sqrt(2.0) * bounds.C3 / 3.0 * root * radius**3, rtol=1e-14,
    )
    np.testing.assert_allclose(
        bounds.mean_supremum_bound(profile, radius, dimension),
        2.0 * bounds.C3 * kappa * root * radius**3, rtol=1e-13,
    )
    np.testing.assert_allclose(
        bounds.pointwise_sd(profile, radius),
        min(math.sqrt(bounds.variance_envelope(profile, radius)),
            bounds.cubic_sd(profile, radius)),
        rtol=1e-14,
    )


@pytest.mark.parametrize("profile", SMOOTH, ids=lambda p: p.name)
@pytest.mark.parametrize("p", (0.80, 0.95, 0.999))
def test_thresholds_invert_probability_bounds(profile, p):
    # The thresholds of Section 6 are the tolerances at which the bounds equal p.
    for radius in (0.05, 0.15, 0.5):
        for dimension in (2, 32, 256):
            pointwise = bounds.pointwise_threshold(profile, radius, p)
            simultaneous = bounds.simultaneous_threshold(profile, radius, dimension, p)
            _, refined, _ = bounds.field_probability_bounds(
                profile, radius, dimension, pointwise
            )
            _, _, supremum = bounds.field_probability_bounds(
                profile, radius, dimension, simultaneous
            )
            np.testing.assert_allclose(refined, p, rtol=1e-12)
            np.testing.assert_allclose(supremum, p, rtol=1e-12)


@pytest.mark.parametrize("profile", SMOOTH, ids=lambda p: p.name)
def test_probability_bounds_are_ordered(profile):
    radii = np.geomspace(0.035, 0.35, 50)
    envelope, refined, simultaneous = bounds.field_probability_bounds(profile, radii, 16, 0.1)
    for bound in (envelope, refined, simultaneous):
        assert np.all((bound >= 0) & (bound <= 1))
        assert np.all(np.diff(bound) <= 1e-15)
    assert np.all(refined >= envelope)
    assert np.all(refined >= simultaneous)


@pytest.mark.parametrize("profile", SMOOTH, ids=lambda p: p.name)
def test_admissible_radii_attain_the_confidence(profile):
    epsilon, confidence = 0.1, 0.95 / 0.99
    dimensions = np.array([2, 16, 256])
    r_envelope, r_refined, r_simultaneous = bounds.admissible_radii(
        profile, dimensions, epsilon, confidence
    )
    envelope, _, _ = bounds.field_probability_bounds(profile, r_envelope, 2, epsilon)
    _, refined, _ = bounds.field_probability_bounds(profile, r_refined, 2, epsilon)
    np.testing.assert_allclose(envelope, confidence, rtol=1e-9)
    np.testing.assert_allclose(refined, confidence, rtol=1e-9)
    for dimension, radius in zip(dimensions, r_simultaneous):
        _, _, simultaneous = bounds.field_probability_bounds(
            profile, radius, dimension, epsilon
        )
        np.testing.assert_allclose(simultaneous, confidence, rtol=1e-9)
    assert r_refined >= r_envelope
    assert np.all(np.diff(r_simultaneous) < 0)


def test_infinite_gamma3_falls_back_to_the_envelope():
    profile = profiles.matern(2.5)
    envelope, refined, simultaneous = bounds.field_probability_bounds(profile, 0.1, 8, 0.1)
    assert refined == envelope
    assert simultaneous == 0.0


def test_gaussian_input_model():
    pair_coverage, radius = 0.99, 0.3
    for dimension in (2, 16, 64):
        q = inputs.chi2_quantile(dimension, pair_coverage)
        np.testing.assert_allclose(
            inputs.input_variance(radius, dimension, pair_coverage), radius**2 / q, rtol=1e-15
        )
        rng = np.random.default_rng(dimension)
        pairs = inputs.sample_input_pairs(radius, dimension, 200_000, rng, pair_coverage)
        assert pairs.shape == (200_000, 2 * dimension)
        norms = np.linalg.norm(pairs.reshape(-1, 2, dimension), axis=2)
        inside = np.mean(np.all(norms <= radius, axis=1))
        assert abs(inside - pair_coverage) < 5 * math.sqrt(0.99 * 0.01 / 200_000)

    profile = profiles.squared_exponential()
    joint = inputs.joint_probability_bounds(profile, radius, 16, 0.1, pair_coverage)
    field = bounds.field_probability_bounds(profile, radius, 16, 0.1)
    np.testing.assert_allclose(joint, pair_coverage * np.array(field), rtol=1e-15)
    np.testing.assert_allclose(inputs.field_confidence(0.95, 0.99), 0.95 / 0.99)
