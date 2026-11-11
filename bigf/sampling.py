"""Point sets and sampling of approximation errors."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .covariance import (
    error_covariance,
    error_variance,
    field_covariance,
    flip,
    joint_field_taylor_covariance,
)


def antidiagonal_point(dimension, radius=1.0):
    """Return the pair (R e_1, -R e_1) as an array of shape (1, 2 D)."""
    x = np.zeros((1, 2 * dimension))
    x[0, 0], x[0, dimension] = radius, -radius
    return x


def unit_pair_points(dimension, count, rng):
    """Draw a set of input pairs in the product of two closed unit balls.

    The set mixes boundary pairs, interior pairs and antidiagonal pairs (a, -a).
    The first pair is (e_1, -e_1). Returns an array of shape (count, 2 D).
    """
    directions = rng.normal(size=(count, 2, dimension))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    radii = rng.random(size=(count, 2)) ** (1.0 / dimension)
    points = directions * radii[..., None]

    points[::3] = directions[::3]
    points[1::3, 0] = directions[1::3, 0]
    points[1::3, 1] = -directions[1::3, 0]

    points[0] = antidiagonal_point(dimension).reshape(2, dimension)
    return points.reshape(count, 2 * dimension)


def sample_errors(x, profile, n_fields, rng, tolerance=1e-7):
    """Sample independent fields and return their signed errors on the point set x.

    Field values and Taylor values are drawn jointly, so all errors in a row come
    from the same field. Returns an array of shape (n_fields, len(x)) and a
    dictionary of diagnostics for the covariance construction.
    """
    cov = joint_field_taylor_covariance(x, profile)
    m = len(x)

    eig, vectors = eigh(cov, driver="evr", check_finite=True)
    scale = max(float(eig[-1]), np.finfo(float).tiny)
    if eig[0] < -1e-10 * scale:
        raise ValueError(f"Materially indefinite joint covariance: {eig[0]=}, {scale=}")

    factor = vectors * np.sqrt(np.maximum(eig, 0.0))[None, :]
    draws = rng.normal(size=(n_fields, 2 * m)) @ factor.T
    field, taylor = draws[:, :m], draws[:, m:]
    errors = taylor - field

    # Check the joint covariance against the closed forms for the error.
    reconstructed = cov[:m, :m] + cov[m:, m:] - cov[:m, m:] - cov[m:, :m]
    reference = error_covariance(x, profile)
    ref_scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    relative_error = float(np.max(np.abs(reconstructed - reference)) / ref_scale)
    diagonal_error = float(
        np.max(np.abs(np.diag(reference) - error_variance(x, profile))) / ref_scale
    )
    if relative_error > tolerance or diagonal_error > tolerance:
        raise ValueError("Joint covariance disagrees with the closed-form error covariance.")

    # Skew-symmetry: K(x, P y) = -K(x, y).
    skew_error = float(np.max(np.abs(field_covariance(x, flip(x), profile) + cov[:m, :m])))

    diagnostics = {
        "min_eigenvalue": float(eig[0]),
        "largest_eigenvalue": scale,
        "clipped_negative_count": int(np.count_nonzero(eig < 0.0)),
        "error_covariance_relative_error": relative_error,
        "error_diagonal_relative_error": diagonal_error,
        "field_flip_covariance_absolute_error": skew_error,
    }
    return errors, diagnostics
