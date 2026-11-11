"""Gaussian input model for the joint guarantees over the field and the inputs.

Two inputs are drawn independently from pi_{D,R} = N(z*, R^2 / q_D I_D), with
q_D = F^{-1}_{chi^2_D}(sqrt(p')). Then both inputs lie in the ball of radius R
about z* with probability p', the input-pair coverage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from . import bounds


def chi2_quantile(dimension, pair_coverage):
    """Return q_D, so that two independent inputs are both in the ball with probability p'."""
    return chi2.ppf(np.sqrt(pair_coverage), dimension)


def input_variance(radius, dimension, pair_coverage):
    """Return the coordinate variance R^2 / q_D of the input distribution."""
    return np.asarray(radius, dtype=float) ** 2 / chi2_quantile(dimension, pair_coverage)


def field_confidence(joint_confidence, pair_coverage):
    """Return the field confidence needed for a joint confidence at input-pair coverage p'."""
    return joint_confidence / pair_coverage


def joint_probability_bounds(profile, radius, dimension, epsilon, pair_coverage):
    """Return lower bounds on the success probability over both the field and the inputs."""
    return tuple(
        pair_coverage * bound
        for bound in bounds.field_probability_bounds(profile, radius, dimension, epsilon)
    )


def sample_input_pairs(radius, dimension, count, rng, pair_coverage):
    """Draw input pairs from pi_{D,R} x pi_{D,R} with z* = 0.

    Returns an array of shape (count, 2 D) whose rows stack the two inputs.
    """
    scale = np.sqrt(input_variance(radius, dimension, pair_coverage))
    return scale * rng.normal(size=(count, 2 * dimension))
