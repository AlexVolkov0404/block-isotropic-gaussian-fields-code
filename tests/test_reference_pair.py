"""The variance-maximizing reference pair x*(h, R) over the product of two balls."""

import math

import numpy as np
import pytest

from bigf import bounds, fixed_input, profiles
from bigf.covariance import (
    error_variance,
    error_variance_from_squared_norms,
    max_error_variance,
)
from bigf.sampling import antidiagonal_point, ball_maximizing_pair, pair_angle_deg

NAMES = ("squared_exponential", "rational_quadratic", "matern_7_2", "matern_9_2")
RADII = (0.05, 0.15, 0.5)


def random_ball_pairs(dimension, radius, count, rng):
    """Pairs with both blocks uniform in the ball, plus pairs on the boundary."""
    directions = rng.normal(size=(count, 2, dimension))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    radii = radius * rng.random(size=(count, 2)) ** (1.0 / dimension)
    radii[: count // 2] = radius
    return (directions * radii[..., None]).reshape(count, 2 * dimension)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dimension", (2, 32))
def test_variance_from_squared_norms_agrees(name, dimension):
    profile = profiles.get_profile(name)
    rng = np.random.default_rng(7)
    x = random_ball_pairs(dimension, 0.6, 500, rng)
    a, b = x[:, :dimension], x[:, dimension:]
    reduced = error_variance_from_squared_norms(
        np.sum((a - b) ** 2, axis=1), np.sum((a + b) ** 2, axis=1), profile
    )
    np.testing.assert_allclose(reduced, error_variance(x, profile), rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("radius", RADII)
def test_max_error_variance_bounds(name, radius):
    profile = profiles.get_profile(name)
    maximum = max_error_variance(profile, radius)
    tol = 1e-12 * maximum.variance

    # Above the antidiagonal value and every sampled pair in the ball.
    assert maximum.variance >= float(error_variance(antidiagonal_point(3, radius), profile)[0]) - 1e-8 * maximum.variance
    rng = np.random.default_rng(11)
    sampled = error_variance(random_ball_pairs(3, radius, 20_000, rng), profile)
    assert maximum.variance >= sampled.max() - tol

    # Above the maximum over a polar grid (r_1, r_2, theta).
    r = np.linspace(0.0, radius, 41)
    theta = np.linspace(0.0, math.pi, 81)
    r1, r2, th = np.meshgrid(r, r, theta, indexing="ij")
    cross = r1 * r2 * np.cos(th)
    grid = error_variance_from_squared_norms(
        r1**2 + r2**2 - 2 * cross, r1**2 + r2**2 + 2 * cross, profile
    )
    assert maximum.variance >= grid.max() - tol
    assert maximum.variance <= grid.max() * (1 + 1e-3)

    # Proposition 16: sup sigma_E <= s_h(R).
    assert maximum.variance <= float(bounds.pointwise_sd(profile, radius)) ** 2

    # The maximizer lies in the triangle d, s >= 0, d + s <= 4 R^2.
    assert maximum.difference >= 0 and maximum.total >= 0
    assert maximum.difference + maximum.total <= 4 * radius**2 * (1 + 1e-12)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("radius", RADII)
def test_ball_maximizing_pair(name, radius):
    profile = profiles.get_profile(name)
    values = []
    for dimension in (2, 256):
        pair, info = ball_maximizing_pair(dimension, profile, radius)
        assert pair.shape == (1, 2 * dimension)
        a, b = pair[0, :dimension], pair[0, dimension:]
        assert np.linalg.norm(a) <= radius * (1 + 1e-12)
        assert np.linalg.norm(b) <= radius * (1 + 1e-12)
        variance = float(error_variance(pair, profile)[0])
        # Equal up to the rounding of the variance formula (about 1e-9 relative
        # for Matérn at R = 0.05), see sampling.ANTIDIAGONAL_TOLERANCE.
        np.testing.assert_allclose(variance, info["max_variance"], rtol=1e-8)
        np.testing.assert_allclose(pair_angle_deg(pair), info["angle_deg"])
        if info["is_antidiagonal"]:
            assert np.array_equal(pair, antidiagonal_point(dimension, radius))
        values.append(variance)
    np.testing.assert_allclose(values[0], values[1], rtol=1e-12)


def test_manuscript_grid_maximizers():
    """45 of the 60 scenarios keep the antidiagonal pair; 15 do not."""
    expected = {  # (d / 4R^2, angle in degrees, sup sigma) where x* is not antidiagonal
        ("rational_quadratic", 0.5): (0.679, 110.9, 1.47105),
        ("matern_7_2", 0.5): (0.679, 110.9, 1.28156),
        ("matern_9_2", 0.5): (0.766, 122.2, 1.14811),
    }
    config = fixed_input.FixedInputConfig()
    count = {True: 0, False: 0}
    for dimension in config.dimensions:
        for name in config.profiles:
            for radius in config.radii:
                _, info = ball_maximizing_pair(dimension, profiles.get_profile(name), radius)
                count[info["is_antidiagonal"]] += 1
                assert info["is_antidiagonal"] == ((name, radius) not in expected)
                if (name, radius) in expected:
                    d, angle, sd = expected[name, radius]
                    assert abs(info["d_over_4r2"] - d) < 1e-3
                    assert abs(info["s_over_4r2"] - (1 - d)) < 1e-3
                    assert abs(info["angle_deg"] - angle) < 0.1
                    np.testing.assert_allclose(math.sqrt(info["max_variance"]), sd, rtol=1e-5)
    assert count == {True: 45, False: 15}
