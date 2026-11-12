"""Covariances of the field, its quadratic Taylor approximation and the error.

An input pair is a row x = (a, b) in R^(2D). The skew-symmetric field has covariance

    K(x, y) = h(|x - y|^2) - h(|x - P y|^2) - h(|P x - y|^2) + h(|P x - P y|^2),

where P swaps the two blocks. The Taylor approximation is taken at the matched
input (0, 0), and the error is the approximation minus the field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit


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


def error_variance_from_squared_norms(difference, total, profile):
    """Pointwise error variance as a function of d = |a - b|^2 and s = |a + b|^2.

    For a pair x = (a, b) the variance depends on x only through d and s, since
    |x|^2 = (d + s) / 2. The value agrees with error_variance and does not
    depend on the block dimension D.
    """
    d = np.asarray(difference, dtype=float)
    s = np.asarray(total, dtype=float)
    t = 0.5 * (d + s)
    return (
        2.0 * profile.one_minus_h(2.0 * d)
        + 4.0 * (-profile.derivative(0.0, 1) + 2.0 * profile.derivative(t, 1)) * d
        + 4.0 * (profile.derivative(0.0, 2) - 2.0 * profile.derivative(t, 2)) * d * s
    )


@dataclass(frozen=True)
class VarianceMaximum:
    """Maximum of the error variance over pairs in the product of two balls.

    variance    the maximal error variance over B_R x B_R
    difference  d* = |a - b|^2 at the maximizer
    total       s* = |a + b|^2 at the maximizer
    radius      the radius R of the balls
    """

    variance: float
    difference: float
    total: float
    radius: float

    @property
    def d_over_4r2(self):
        return self.difference / (4.0 * self.radius**2)

    @property
    def s_over_4r2(self):
        return self.total / (4.0 * self.radius**2)


def max_error_variance(profile, radius, grid=401, edge_points=20_001):
    """Maximize the error variance over pairs with |a|, |b| <= radius.

    For D >= 2 the pairs in B_R x B_R give exactly the triangle
    {d, s >= 0, d + s <= 4 R^2} of values (d, s) = (|a - b|^2, |a + b|^2),
    so the maximum is a two-variable problem that does not depend on D. The
    maximum is taken over a triangular grid, fine grids on the three edges and
    local refinements of the best candidates.
    """
    radius = float(radius)
    scale = 4.0 * radius**2

    def variance(u, v):
        return error_variance_from_squared_norms(scale * u, scale * v, profile)

    candidates = []

    # 1. Triangular grid in (u, v) = (d, s) / 4R^2.
    nodes = np.linspace(0.0, 1.0, grid)
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    inside = u + v <= 1.0 + 1e-15
    u, v = u[inside], np.minimum(v[inside], 1.0 - u[inside])
    values = variance(u, v)
    k = int(np.argmax(values))
    candidates.append((float(values[k]), float(u[k]), float(v[k])))

    # 2. The three edges u + v = 1, v = 0 and u = 0.
    line = np.linspace(0.0, 1.0, edge_points)
    step = 1.0 / (edge_points - 1)
    edges = (
        (lambda z: (z, 1.0 - z)),
        (lambda z: (z, np.zeros_like(z))),
        (lambda z: (np.zeros_like(z), z)),
    )
    for edge in edges:
        values = variance(*edge(line))
        k = int(np.argmax(values))
        ue, ve = edge(line[k:k + 1])
        candidates.append((float(values[k]), float(ue[0]), float(ve[0])))
        # Refine along the edge between the neighbours of the best edge point.
        low, high = max(line[k] - step, 0.0), min(line[k] + step, 1.0)
        if high > low:
            found = minimize_scalar(
                lambda z: -float(variance(*edge(np.array([z])))[0]),
                bounds=(low, high), method="bounded", options={"xatol": 1e-13},
            )
            ue, ve = edge(np.array([found.x]))
            candidates.append((float(variance(ue, ve)[0]), float(ue[0]), float(ve[0])))

    # 3. Nelder-Mead in the interior from the best candidate so far, with
    #    u = w z, v = w (1 - z) and logistic coordinates for w and z.
    best = max(candidates)
    _, u0, v0 = best
    w0 = min(max(u0 + v0, 1e-6), 1.0 - 1e-6)
    z0 = min(max(u0 / (u0 + v0) if u0 + v0 > 0 else 0.5, 1e-6), 1.0 - 1e-6)

    def interior(y):
        w, z = expit(y[0]), expit(y[1])
        return w * z, w * (1.0 - z)

    found = minimize(
        lambda y: -float(variance(*interior(y))),
        x0=np.array([logit(w0), logit(z0)]), method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-18, "maxiter": 4000},
    )
    ui, vi = interior(found.x)
    candidates.append((float(variance(ui, vi)), float(ui), float(vi)))

    value, u_best, v_best = max(candidates)
    return VarianceMaximum(
        variance=value, difference=scale * u_best, total=scale * v_best, radius=radius,
    )
