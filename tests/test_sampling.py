"""Covariance construction, point sets and error sampling."""

import numpy as np
import pytest

from bigf import profiles
from bigf.covariance import (
    error_covariance,
    error_variance,
    field_covariance,
    flip,
    joint_field_taylor_covariance,
)
from bigf.sampling import antidiagonal_point, sample_errors, unit_pair_points

PROFILES = [
    profiles.get_profile(name)
    for name in ("squared_exponential", "rational_quadratic", "matern_7_2", "matern_9_2")
]


def points(dimension, count=24, radius=0.3, seed=0):
    return radius * unit_pair_points(dimension, count, np.random.default_rng(seed))


def test_unit_pair_points():
    dimension, count = 8, 64
    x = unit_pair_points(dimension, count, np.random.default_rng(3))
    assert x.shape == (count, 2 * dimension)
    np.testing.assert_array_equal(x[0], antidiagonal_point(dimension)[0])
    blocks = x.reshape(count, 2, dimension)
    norms = np.linalg.norm(blocks, axis=2)
    assert np.all(norms <= 1.0 + 1e-12)
    np.testing.assert_allclose(norms[::3], 1.0, rtol=1e-12)           # boundary pairs
    np.testing.assert_allclose(blocks[1::3, 1], -blocks[1::3, 0])     # antidiagonal pairs
    np.testing.assert_array_equal(
        x, unit_pair_points(dimension, count, np.random.default_rng(3))
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_field_is_skew_symmetric(profile):
    x = points(6)
    covariance = field_covariance(x, x, profile)
    np.testing.assert_allclose(field_covariance(x, flip(x), profile), -covariance, atol=1e-13)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-13)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("dimension", (2, 8, 32))
def test_joint_covariance_gives_the_closed_form_error_covariance(profile, dimension):
    x = points(dimension)
    m = len(x)
    joint = joint_field_taylor_covariance(x, profile)
    assert joint.shape == (2 * m, 2 * m)
    np.testing.assert_array_equal(joint, joint.T)
    assert np.linalg.eigvalsh(joint)[0] > -1e-10 * np.linalg.eigvalsh(joint)[-1]

    reconstructed = joint[:m, :m] + joint[m:, m:] - joint[:m, m:] - joint[m:, :m]
    reference = error_covariance(x, profile)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(reconstructed, reference, atol=1e-8 * scale)
    np.testing.assert_allclose(np.diag(reference), error_variance(x, profile), atol=1e-8 * scale)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_error_vanishes_on_the_diagonal(profile):
    # At a matched input (a, a) the skew-symmetric field and its approximation are both zero.
    a = np.random.default_rng(5).normal(size=(10, 4))
    x = np.concatenate((a, a), axis=1)
    np.testing.assert_allclose(error_variance(x, profile), 0.0, atol=1e-14)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_sampled_errors_have_the_closed_form_variance(profile):
    x = points(8, count=16, radius=0.5)
    n = 40_000
    errors, diagnostics = sample_errors(x, profile, n, np.random.default_rng(11))
    assert errors.shape == (n, len(x))
    assert diagnostics["error_covariance_relative_error"] < 1e-7

    variance = error_variance(x, profile)
    # Var of a sample variance of a centered Gaussian is 2 sigma^4 / n.
    tolerance = 6.0 * np.sqrt(2.0 / n) * variance
    assert np.all(np.abs(np.mean(errors**2, axis=0) - variance) <= tolerance)

    # Errors at different points come from the same field, so they are correlated.
    covariance = error_covariance(x, profile)
    empirical = errors.T @ errors / n
    bound = 6.0 * np.sqrt((np.outer(variance, variance) + covariance**2) / n)
    assert np.all(np.abs(empirical - covariance) <= bound)


def test_sampling_is_reproducible():
    profile = profiles.get_profile("matern_7_2")
    x = points(4)
    first, _ = sample_errors(x, profile, 100, np.random.default_rng(7))
    second, _ = sample_errors(x, profile, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
