"""Covariance profiles: normalization, derivatives and closed forms."""

import math

import numpy as np
import pytest

from bigf import profiles

S = np.concatenate(([0.0], np.geomspace(1e-8, 60.0, 400)))
ALL = [
    profile
    for ell in (1.0, 0.7)
    for profile in profiles.manuscript_profiles(ell).values()
]
IDS = [f"{profile.name}-ell{profile.ell:g}" for profile in ALL]


@pytest.mark.parametrize("profile", ALL, ids=IDS)
def test_normalization(profile):
    np.testing.assert_allclose(profile.derivative(0.0), 1.0, rtol=1e-14)
    np.testing.assert_allclose(profile.derivative(0.0, 1), -profile.ell**-2, rtol=1e-14)


@pytest.mark.parametrize("profile", ALL, ids=IDS)
def test_values_lie_in_unit_interval_and_decrease(profile):
    h = profile(S)
    assert np.all((h > 0) & (h <= 1))
    assert np.all(np.diff(h) <= 0)


@pytest.mark.parametrize("profile", ALL, ids=IDS)
@pytest.mark.parametrize("order", (1, 2, 3))
def test_derivative_matches_finite_difference(profile, order):
    if profile.family == "matern" and profile.shape <= order:
        with pytest.raises(ValueError):
            profile.derivative(1.0, order)
        return
    s = np.geomspace(0.05, 5.0, 40)
    step = 1e-5 * s
    numerical = (
        profile.derivative(s + step, order - 1) - profile.derivative(s - step, order - 1)
    ) / (2.0 * step)
    np.testing.assert_allclose(profile.derivative(s, order), numerical, rtol=1e-7)


@pytest.mark.parametrize("profile", ALL, ids=IDS)
def test_one_minus_h(profile):
    np.testing.assert_allclose(profile.one_minus_h(S), 1.0 - profile(S), atol=1e-14)
    if profile.family != "matern":
        # Near zero, 1 - h(s) = s / ell^2 to first order. The direct difference loses this.
        tiny = 1e-12
        np.testing.assert_allclose(
            profile.one_minus_h(tiny), tiny / profile.ell**2, rtol=1e-9
        )


@pytest.mark.parametrize("profile", ALL, ids=IDS)
def test_gamma3_is_minus_third_derivative_at_zero(profile):
    if profile.family == "matern" and profile.shape <= 3:
        assert profile.gamma3 == math.inf
    else:
        np.testing.assert_allclose(profile.gamma3, -profile.derivative(0.0, 3), rtol=1e-13)


@pytest.mark.parametrize("name", list(profiles.manuscript_profiles()))
@pytest.mark.parametrize("order", (0, 1, 2))
def test_length_scale(name, order):
    ell = 1.7
    unit, scaled = profiles.get_profile(name), profiles.get_profile(name, ell)
    np.testing.assert_allclose(
        scaled.derivative(S, order),
        ell ** (-2 * order) * unit.derivative(S / ell**2, order),
        rtol=1e-12,
    )


def test_closed_forms():
    z = lambda nu: 2.0 * np.sqrt((nu - 1.0) * S)
    expected = {
        "squared_exponential": np.exp(-S),
        "rational_quadratic": 1.0 / (1.0 + S),
        "matern_5_2": (1 + z(2.5) + z(2.5) ** 2 / 3) * np.exp(-z(2.5)),
        "matern_7_2": (
            (1 + z(3.5) + 2 * z(3.5) ** 2 / 5 + z(3.5) ** 3 / 15) * np.exp(-z(3.5))
        ),
        "matern_9_2": (
            (1 + z(4.5) + 3 * z(4.5) ** 2 / 7 + 2 * z(4.5) ** 3 / 21 + z(4.5) ** 4 / 105)
            * np.exp(-z(4.5))
        ),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(profiles.get_profile(name)(S), values, rtol=1e-12)


def test_gamma3_values():
    values = {name: p.gamma3 for name, p in profiles.manuscript_profiles().items()}
    assert values["squared_exponential"] == 1.0
    assert values["rational_quadratic"] == 6.0
    np.testing.assert_allclose(values["matern_7_2"], 25.0 / 3.0, rtol=1e-15)
    np.testing.assert_allclose(values["matern_9_2"], 49.0 / 15.0, rtol=1e-15)


def test_invalid_arguments():
    profile = profiles.squared_exponential()
    with pytest.raises(ValueError):
        profile.derivative(-1.0)
    with pytest.raises(ValueError):
        profile.derivative(1.0, -1)
    with pytest.raises(ValueError):
        profiles.Profile("x", "unknown")
    with pytest.raises(ValueError):
        profiles.matern(1.0)
    with pytest.raises(KeyError):
        profiles.get_profile("unknown")
