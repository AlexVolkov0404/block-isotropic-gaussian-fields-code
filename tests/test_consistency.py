"""The committed results agree with the code.

These tests read the files in figs/ and results/ and recompute their contents.
Values that depend only on formulas are compared with a small tolerance, since
they can differ in the last digits between numerical environments. Counts and
empirical quantiles are recomputed from the saved errors and must agree exactly.
"""

import csv
import json

import numpy as np
import pytest

from bigf import bounds, fixed_input, inputs, profiles
from bigf.paths import FIG_DIR

CONFIG = fixed_input.FixedInputConfig()
RESULTS_DIR = fixed_input.DEFAULT_RESULTS_DIR
RTOL = 1e-9
TABLE_LABELS = {
    "SE": "squared_exponential",
    "RQ": "rational_quadratic",
    "Matérn 7/2": "matern_7_2",
    "Matérn 9/2": "matern_9_2",
}


def read_csv(path):
    if not path.is_file():
        pytest.skip(f"{path.name} has not been generated")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def records():
    if not (RESULTS_DIR / "calibration_records.json").is_file():
        pytest.skip("run scripts/run_fixed_input.py to generate the records")
    return fixed_input.load_records(CONFIG, RESULTS_DIR)


def record_key(row):
    return row["profile"], row["dimension"], row["radius"], row["p"], row["scheme"]


def test_figure4_data():
    rows = read_csv(FIG_DIR / "probability_bounds_data.csv")
    by_name = {(p.family, p.shape): p for p in profiles.manuscript_profiles().values()}
    for row in rows[:: max(1, len(rows) // 600)]:
        shape = float(row["shape"]) if row["shape"] else None
        profile = by_name[row["family"], shape]
        dimension, radius = int(row["D"]), float(row["R"])
        coverage = float(row["input_pair_coverage"])
        expected = inputs.joint_probability_bounds(
            profile, radius, dimension, float(row["epsilon"]), coverage
        )
        actual = [float(row[column]) for column in (
            "joint_pointwise_envelope", "joint_pointwise_refined", "joint_simultaneous"
        )]
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=1e-300)
        np.testing.assert_allclose(
            float(row["q_D"]), inputs.chi2_quantile(dimension, coverage), rtol=RTOL
        )
        np.testing.assert_allclose(
            float(row["input_coordinate_variance"]),
            inputs.input_variance(radius, dimension, coverage), rtol=RTOL,
        )


def test_figure5_data():
    rows = read_csv(FIG_DIR / "radius_dimension_data.csv")
    assert rows
    for row in rows:
        profile = profiles.get_profile(row["kernel"])
        dimension = int(row["D"])
        expected = bounds.admissible_radii(
            profile, dimension, float(row["epsilon"]), float(row["field_confidence"])
        )
        actual = [float(row[c]) for c in ("R_envelope", "R_refined", "R_simultaneous")]
        np.testing.assert_allclose(actual, expected, rtol=RTOL)
        np.testing.assert_allclose(
            float(row["field_confidence"]),
            inputs.field_confidence(
                float(row["joint_confidence"]), float(row["input_pair_coverage"])
            ),
            rtol=RTOL,
        )


def test_records_follow_from_saved_errors(records):
    rebuilt = {record_key(row): row for row in fixed_input.summarize(CONFIG, RESULTS_DIR)}
    exact = ("n", "successes", "failures", "coverage", "empirical_quantile", "event",
             "coverage_wilson95_low", "coverage_wilson95_high")
    close = ("threshold", "threshold_over_empirical_quantile",
             "exact_gaussian_fixed_point_coverage")
    for row in records:
        expected = rebuilt[record_key(row)]
        for field in exact:
            assert row[field] == expected[field], (record_key(row), field)
        for field in close:
            if expected[field] is None:
                assert row[field] is None
            else:
                np.testing.assert_allclose(row[field], expected[field], rtol=RTOL)


def test_run_info_matches_settings(records):
    info = json.loads((RESULTS_DIR / "run_info.json").read_text(encoding="utf-8"))
    assert info["config"] == json.loads(json.dumps(fixed_input.run_info(CONFIG)["config"]))
    diagnostics = json.loads((RESULTS_DIR / "diagnostics.json").read_text(encoding="utf-8"))
    assert len(diagnostics) == CONFIG.n_scenarios
    for row in diagnostics:
        assert row["min_eigenvalue"] > -1e-10 * row["largest_eigenvalue"]
        assert row["error_covariance_relative_error"] < 1e-7
        assert row["error_diagonal_relative_error"] < 1e-7


def test_table1_matches_records(records):
    coverage = {record_key(row): row["coverage"] for row in records}
    rows = read_csv(RESULTS_DIR / "table1_coverage.csv")
    levels = CONFIG.confidence_levels
    columns = [f"p={100 * p:g}%" for p in levels]

    pointwise = [row for row in rows if row["Profile"] != "Simultaneous"]
    assert len(pointwise) == len(CONFIG.profiles) * len(CONFIG.dimensions)
    for row in pointwise:
        name, dimension = TABLE_LABELS[row["Profile"]], int(row["D"])
        for p, column in zip(levels, columns):
            expected = "/".join(
                f"{100 * coverage[name, dimension, radius, p, 'pointwise_refined']:.2f}"
                for radius in CONFIG.radii
            )
            assert row[column] == expected

    simultaneous = [row for row in rows if row["Profile"] == "Simultaneous"]
    assert len(simultaneous) == 1
    for p, column in zip(levels, columns):
        values = [v for key, v in coverage.items() if key[3] == p and key[4] == "simultaneous"]
        low, high = f"{100 * min(values):.2f}", f"{100 * max(values):.2f}"
        assert simultaneous[0][column] == (low if low == high else f"{low}-{high}")

    # The LaTeX table holds the same cells.
    tex = (RESULTS_DIR / "table1_coverage.tex").read_text(encoding="utf-8")
    tex = tex.replace(r"\phantom{0}", "")
    for row in pointwise:
        assert " & ".join(row[column] for column in columns) in tex


def test_figure6_data_matches_records(records):
    by_key = {record_key(row): row for row in records}
    rows = read_csv(RESULTS_DIR / "figure6_calibration_p95.csv")
    assert len(rows) == 2 * len(CONFIG.radii)
    for row in rows:
        source = by_key[
            row["profile"], int(row["dimension"]), float(row["radius"]),
            float(row["p"]), row["scheme"],
        ]
        for field in ("coverage", "threshold", "empirical_quantile",
                      "threshold_over_empirical_quantile"):
            assert float(row[field]) == source[field]
