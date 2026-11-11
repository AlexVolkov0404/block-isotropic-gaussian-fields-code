"""Covariance profiles of block-isotropic Gaussian fields.

A profile is a function h of the squared distance s = ||x - y||^2. Every profile
here is normalized so that h(0) = 1 and h'(0) = -1 / ell^2, where ell is the
length scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import gammaln, kve

FAMILIES = ("se", "rq", "matern")


@dataclass(frozen=True)
class Profile:
    """A covariance profile h(s).

    family  "se" (squared exponential), "rq" (rational quadratic) or "matern".
    shape   a for the rational quadratic profile, nu for the Matérn profile.
    ell     Length scale.
    """

    name: str
    family: str
    shape: float | None = None
    ell: float = 1.0
    label: str = ""

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown profile family: {self.family}")
        if self.ell <= 0:
            raise ValueError("The length scale must be positive.")
        if self.family == "rq" and not (self.shape is not None and self.shape > 0):
            raise ValueError("The rational quadratic profile needs a > 0.")
        if self.family == "matern" and not (self.shape is not None and self.shape > 1):
            raise ValueError("The Matérn profile needs nu > 1.")

    def _log_matern_factor(self, s, order):
        """Log of M_q(b s) with M_q(t) = 2 t^(q/2) K_q(2 sqrt(t)) / Gamma(q) and M_q(0) = 1.

        Here q = order and b = (nu - 1) / ell^2.
        """
        t = (self.shape - 1.0) / self.ell**2 * s
        positive = t > 0
        safe_t = np.where(positive, t, 1.0)
        z = 2.0 * np.sqrt(safe_t)
        log_factor = (
            np.log(2.0) + 0.5 * order * np.log(safe_t)
            - gammaln(order) + np.log(kve(order, z)) - z
        )
        return np.where(positive, np.minimum(log_factor, 0.0), 0.0)

    def derivative(self, s, order=0):
        """Evaluate the derivative h^(order)(s) with respect to the squared distance s."""
        if order != int(order) or order < 0:
            raise ValueError("The derivative order must be a nonnegative integer.")
        s = np.asarray(s, dtype=float)
        if np.any(s < 0):
            raise ValueError("The squared distance must be nonnegative.")
        k = int(order)
        sign = (-1.0) ** k

        if self.family == "se":
            return sign * self.ell ** (-2 * k) * np.exp(-s / self.ell**2)

        if self.family == "rq":
            a = self.shape
            moment = np.exp(gammaln(a + k) - gammaln(a)) / (a * self.ell**2) ** k
            return sign * moment * np.exp(-(a + k) * np.log1p(s / (a * self.ell**2)))

        nu = self.shape
        if nu <= k:
            raise ValueError(
                f"The Matérn derivative of order {k} is not finite at zero for nu = {nu:g}."
            )
        b = (nu - 1.0) / self.ell**2
        moment = b**k * np.exp(gammaln(nu - k) - gammaln(nu))
        return sign * moment * np.exp(self._log_matern_factor(s, nu - k))

    def __call__(self, s):
        """Evaluate h(s)."""
        return self.derivative(s, 0)

    def one_minus_h(self, s):
        """Evaluate 1 - h(s).

        For the squared exponential and rational quadratic profiles this avoids
        the cancellation of 1 - h(s) near s = 0. For the Matérn profile the
        absolute accuracy stays near 1e-15.
        """
        s = np.asarray(s, dtype=float)
        if np.any(s < 0):
            raise ValueError("The squared distance must be nonnegative.")
        if self.family == "se":
            return -np.expm1(-s / self.ell**2)
        if self.family == "rq":
            a = self.shape
            return -np.expm1(-a * np.log1p(s / (a * self.ell**2)))
        return -np.expm1(self._log_matern_factor(s, self.shape))

    @property
    def gamma3(self):
        """Return gamma_3 = -h'''(0+). It is infinite for the Matérn profile with nu <= 3."""
        if self.family == "se":
            return self.ell ** (-6)
        if self.family == "rq":
            a = self.shape
            return (a + 1.0) * (a + 2.0) / (a**2 * self.ell**6)
        nu = self.shape
        if nu <= 3:
            return math.inf
        return (nu - 1.0) ** 2 / ((nu - 2.0) * (nu - 3.0) * self.ell**6)


def _fraction(value):
    """Format a shape parameter for names and labels, e.g. 3.5 -> ("7_2", "7/2")."""
    fraction = Fraction(value).limit_denominator(1000)
    if fraction.denominator == 1:
        return str(fraction.numerator), str(fraction.numerator)
    return (f"{fraction.numerator}_{fraction.denominator}",
            f"{fraction.numerator}/{fraction.denominator}")


def squared_exponential(ell=1.0):
    """h(s) = exp(-s / ell^2)."""
    return Profile("squared_exponential", "se", None, ell, "Squared exponential")


def rational_quadratic(a=1.0, ell=1.0):
    """h(s) = (1 + s / (a ell^2))^(-a)."""
    tag, text = _fraction(a)
    name = "rational_quadratic" if a == 1 else f"rational_quadratic_a{tag}"
    return Profile(name, "rq", float(a), ell, rf"Rational quadratic ($a={text}$)")


def matern(nu, ell=1.0):
    """Matérn profile with smoothness nu in the variable z = 2 sqrt((nu - 1) s) / ell.

    h(s) = 2^(1 - nu) / Gamma(nu) * z^nu * K_nu(z).
    """
    tag, text = _fraction(nu)
    return Profile(f"matern_{tag}", "matern", float(nu), ell, rf"Matérn ($\nu={text}$)")


def manuscript_profiles(ell=1.0):
    """Return the profiles used in the manuscript, keyed by name."""
    profiles = (
        squared_exponential(ell),
        rational_quadratic(1.0, ell),
        matern(2.5, ell),
        matern(3.5, ell),
        matern(4.5, ell),
    )
    return {profile.name: profile for profile in profiles}


def get_profile(name, ell=1.0):
    """Look up one of the manuscript's profiles by name."""
    profiles = manuscript_profiles(ell)
    if name not in profiles:
        raise KeyError(f"Unknown profile {name!r}. Available: {', '.join(profiles)}")
    return profiles[name]
