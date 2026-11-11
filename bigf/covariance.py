"""Covariances of the field, its quadratic Taylor approximation and the error.

An input pair is a row x = (a, b) in R^(2D). The skew-symmetric field has covariance

    K(x, y) = h(|x - y|^2) - h(|x - P y|^2) - h(|P x - y|^2) + h(|P x - P y|^2),

where P swaps the two blocks. The Taylor approximation is taken at the matched
input (0, 0), and the error is the approximation minus the field.
"""

from __future__ import annotations

import numpy as np


def flip(x):
    """Swap the two D-dimensional input blocks."""
    d = x.shape[-1] // 2
    return np.concatenate((x[..., d:], x[..., :d]), axis=-1)


def squared_distances(x, y):
    """Return the matrix of squared distances between the rows of x and y."""
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def _base_value_taylor(x, y, profile):
    """Covariance of a base field value at x with the Taylor terms at y."""
    sx = np.sum(x * x, axis=1)[:, None]
    sy = np.sum(y * y, axis=1)[None, :]
    dot = x @ y.T
    return (
        -2.0 * profile.derivative(sx, 1) * dot
        + profile.derivative(sx, 1) * sy
        + 2.0 * profile.derivative(sx, 2) * dot * dot
    )


def _base_taylor_taylor(x, y, profile):
    """Covariance of the base Taylor terms, including the Hessian trace term."""
    sx = np.sum(x * x, axis=1)[:, None]
    sy = np.sum(y * y, axis=1)[None, :]
    dot = x @ y.T
    return (
        -2.0 * profile.derivative(0.0, 1) * dot
        + profile.derivative(0.0, 2) * (sx * sy + 2.0 * dot * dot)
    )


def antisymmetrize(operator, x, y):
    """Apply the skew-symmetrization in both arguments of a base covariance."""
    px, py = flip(x), flip(y)
    return operator(x, y) - operator(x, py) - operator(px, y) + operator(px, py)


def field_covariance(x, y, profile):
    """Covariance K(x, y) of the skew-symmetric field between two point sets."""
    return antisymmetrize(
        lambda a, b: profile.derivative(squared_distances(a, b)), x, y
    )


def joint_field_taylor_covariance(x, profile):
    """Covariance of the stacked field and Taylor values on the point set x.

    For m points the result is 2m by 2m, with the field values first.
    """
    ff = field_covariance(x, x, profile)
    ft = antisymmetrize(lambda a, b: _base_value_taylor(a, b, profile), x, x)
    tt = antisymmetrize(lambda a, b: _base_taylor_taylor(a, b, profile), x, x)
    cov = np.block([[ff, ft], [ft.T, tt]])
    return (cov + cov.T) / 2.0


def error_covariance(x, profile):
    """Closed form of the error covariance on the point set x."""
    sx = np.sum(x * x, axis=1)
    dot = x @ x.T
    crossed = x @ flip(x).T
    p, q = dot - crossed, dot * dot - crossed * crossed
    return (
        2.0 * (profile.derivative(squared_distances(x, x))
               - profile.derivative(squared_distances(x, flip(x))))
        + 4.0 * (profile.derivative(sx, 1)[:, None]
                 + profile.derivative(sx, 1)[None, :]
                 - profile.derivative(0.0, 1)) * p
        - 4.0 * (profile.derivative(sx, 2)[:, None]
                 + profile.derivative(sx, 2)[None, :]
                 - profile.derivative(0.0, 2)) * q
    )


def error_variance(x, profile):
    """Closed form of the pointwise error variance at each row of x."""
    d = x.shape[-1] // 2
    a, b = x[:, :d], x[:, d:]
    t = np.sum(x * x, axis=1)
    distance = np.sum((a - b) ** 2, axis=1)
    q = distance * np.sum((a + b) ** 2, axis=1)
    return (
        2.0 * profile.one_minus_h(2.0 * distance)
        + 4.0 * (-profile.derivative(0.0, 1) + 2.0 * profile.derivative(t, 1)) * distance
        + 4.0 * (profile.derivative(0.0, 2) - 2.0 * profile.derivative(t, 2)) * q
    )
