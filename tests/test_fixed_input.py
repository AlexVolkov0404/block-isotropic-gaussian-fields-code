"""The fixed-input experiment on a small configuration."""

import json

import numpy as np
import pytest

from bigf import bounds, fixed_input, profiles
from bigf.calibration import coverage_summary, wilson_interval

SMALL = fixed_input.FixedInputConfig(
    seed=1, profiles=("squared_exponential", "matern_7_2"), dimensions=(2, 8),
    radii=(0.15,), point_counts=(8, 16), n_fields=400, confidence_levels=(0.9, 0.99),
)


def test_wilson_interval():
    # The endpoints solve (c - pi)^2 = z^2 pi (1 - pi) / n for the observed coverage c.
    z, n = 1.959963984540054, 10_000
    for successes in (8000, 9500, 9990):
        coverage = successes / n
        low, high = wilson_interval(successes, n)
        assert low < coverage < high
        for endpoint in (low, high):
            np.testing.assert_allclose(
                (coverage - endpoint) ** 2, z**2 * endpoint * (1 - endpoint) / n, rtol=1e-9
            )
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


def test_coverage_summary():
    values = np.arange(1, 101, dtype=float)
    summary = coverage_summary(values, 90.0, 0.9)
    assert (summary["n"], summary["successes"], summary["failures"]) == (100, 90, 10)
    assert summary["coverage"] == 0.9
    np.testing.assert_allclose(summary["empirical_quantile"], 90.1)
    np.testing.assert_allclose(summary["threshold_over_empirical_quantile"], 90.0 / 90.1)


def test_default_config_is_the_manuscript_design():
    config = fixed_input.FixedInputConfig()
    assert config.seed == 2026102302
    assert config.n_scenarios == 60
    assert config.n_fields == 10_000
    assert len(config.record_keys()) == 600
    assert config.events["simultaneous"] == "maximum_on_128_fixed_points"


def test_thresholds_follow_the_bounds():
    profile = profiles.get_profile("rational_quadratic")
    levels = fixed_input.thresholds(profile, 0.15, 32, 0.95)
    assert levels == {
        "pointwise_refined": float(bounds.pointwise_threshold(profile, 0.15, 0.95)),
        "simultaneous": float(bounds.simultaneous_threshold(profile, 0.15, 32, 0.95)),
    }
    assert levels["simultaneous"] > levels["pointwise_refined"]


def test_run_writes_consistent_files(tmp_path):
    records = fixed_input.run(SMALL, tmp_path)
    assert len(records) == len(SMALL.record_keys()) == 16
    assert fixed_input.load_records(SMALL, tmp_path) == records

    # The records can be rebuilt from the saved errors alone.
    assert fixed_input.summarize(SMALL, tmp_path) == records

    for name in SMALL.profiles:
        for dimension in SMALL.dimensions:
            with np.load(fixed_input.raw_path(tmp_path, name, dimension, 0.15)) as raw:
                errors = raw["fixed_reference_point_errors"]
                maxima = raw["finite_set_maxima"]
            assert errors.shape == (SMALL.n_fields,)
            assert maxima.shape == (SMALL.n_fields, len(SMALL.point_counts))
            # The point sets are nested and the reference pair is their first point.
            assert np.all(np.diff(maxima, axis=1) >= 0)
            assert np.all(maxima[:, 0] >= np.abs(errors))
            points = np.load(fixed_input.unit_points_path(tmp_path, dimension))
            assert points.shape == (max(SMALL.point_counts), 2 * dimension)

    for row in records:
        assert row["n"] == SMALL.n_fields
        assert row["successes"] + row["failures"] == row["n"]
        assert row["coverage_wilson95_low"] <= row["coverage"] <= row["coverage_wilson95_high"]
        if row["scheme"] == "pointwise_refined":
            # The threshold is conservative for the exact Gaussian law at the fixed pair.
            assert row["exact_gaussian_fixed_point_coverage"] >= row["p"]
        else:
            assert row["exact_gaussian_fixed_point_coverage"] is None

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["config"]["seed"] == SMALL.seed
    assert set(info["environment"]) == {"python", "numpy", "scipy", "platform"}
    assert len(json.loads((tmp_path / "diagnostics.json").read_text())) == SMALL.n_scenarios


def test_run_is_reproducible(tmp_path):
    first = fixed_input.run(SMALL, tmp_path / "a")
    second = fixed_input.run(SMALL, tmp_path / "b")
    assert first == second


def test_load_records_rejects_other_settings(tmp_path):
    fixed_input.run(SMALL, tmp_path)
    with pytest.raises(ValueError):
        fixed_input.load_records(fixed_input.FixedInputConfig(), tmp_path)
    with pytest.raises(FileNotFoundError):
        fixed_input.load_records(SMALL, tmp_path / "missing")
