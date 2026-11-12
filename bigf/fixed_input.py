"""Fixed-input Monte Carlo experiment (Table 1 and Figure 5 of the manuscript).

For each profile, block dimension D and radius R, the experiment samples
independent fields and records the absolute error at the fixed pair
(R e_1, -R e_1) and the maximum absolute error over a fixed set of input pairs.
These are compared with the refined pointwise threshold and the simultaneous
threshold at several field-confidence levels.

Files written under the results directory:

    raw/unit_points_D{D}.npy            the fixed point set for each D, before scaling by R
    raw/{profile}_D{D}_R{R}.npz         sampled errors for each scenario
    calibration_records.json            one record per scenario, level and threshold
    diagnostics.json                    checks of the sampling construction
    run_info.json                       settings and package versions of the run
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.special import erf

from . import bounds
from .calibration import coverage_summary
from .covariance import error_variance
from .paths import RESULTS_DIR
from .profiles import get_profile
from .sampling import antidiagonal_point, sample_errors, unit_pair_points

DEFAULT_RESULTS_DIR = RESULTS_DIR / "fixed_input"
SCHEMES = ("pointwise_refined", "simultaneous")


@dataclass(frozen=True)
class FixedInputConfig:
    """Settings of the experiment. The defaults are those of the manuscript."""

    seed: int = 2026102302
    profiles: tuple[str, ...] = (
        "squared_exponential", "rational_quadratic", "matern_7_2", "matern_9_2",
    )
    dimensions: tuple[int, ...] = (2, 8, 32, 128, 256)
    radii: tuple[float, ...] = (0.05, 0.15, 0.5)
    point_counts: tuple[int, ...] = (32, 64, 128)
    n_fields: int = 10_000
    confidence_levels: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99, 0.999)

    @property
    def n_scenarios(self):
        return len(self.profiles) * len(self.dimensions) * len(self.radii)

    @property
    def events(self):
        """Describe the error that each threshold is compared with."""
        return {
            "pointwise_refined": "absolute_error_at_fixed_reference_point",
            "simultaneous": f"maximum_on_{max(self.point_counts)}_fixed_points",
        }

    def record_keys(self):
        """Return the identifying key of every expected calibration record."""
        return {
            (name, dimension, radius, p, scheme)
            for name in self.profiles
            for dimension in self.dimensions
            for radius in self.radii
            for p in self.confidence_levels
            for scheme in SCHEMES
        }


def raw_path(results_dir, profile_name, dimension, radius):
    return Path(results_dir) / "raw" / f"{profile_name}_D{dimension}_R{radius:g}.npz"


def unit_points_path(results_dir, dimension):
    return Path(results_dir) / "raw" / f"unit_points_D{dimension}.npy"


def thresholds(profile, radius, dimension, p):
    """Return the absolute-error thresholds at field-confidence level p."""
    return {
        "pointwise_refined": float(bounds.pointwise_threshold(profile, radius, p)),
        "simultaneous": float(
            bounds.simultaneous_threshold(profile, radius, dimension, p)
        ),
    }


def scenario_records(config, profile, dimension, radius, reference_errors, maxima):
    """Summarize the sampled errors of one scenario at every confidence level.

    reference_errors  signed errors at the pair (R e_1, -R e_1), one per field
    maxima            maximum absolute error over the first k points of the set,
                      with one column for each k in config.point_counts
    """
    values = {
        "pointwise_refined": np.abs(reference_errors),
        "simultaneous": np.asarray(maxima)[:, -1],
    }
    variance = float(error_variance(antidiagonal_point(dimension, radius), profile)[0])

    records = []
    for p in config.confidence_levels:
        levels = thresholds(profile, radius, dimension, p)
        for scheme in SCHEMES:
            summary = coverage_summary(values[scheme], levels[scheme], p)
            exact_coverage = None
            if scheme == "pointwise_refined":
                exact_coverage = float(
                    erf(summary["threshold"] / math.sqrt(2.0 * variance))
                )
            records.append({
                "profile": profile.name,
                "dimension": int(dimension),
                "radius": float(radius),
                "p": float(p),
                "scheme": scheme,
                "event": config.events[scheme],
                **summary,
                "exact_gaussian_fixed_point_coverage": exact_coverage,
                "measured_continuum_supremum_coverage": False,
            })
    return records


def run_info(config):
    """Return the settings and the numerical environment of a run."""
    return {
        "experiment": "fixed_input",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
        },
    }


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run(config=FixedInputConfig(), results_dir=DEFAULT_RESULTS_DIR, progress=None):
    """Run the experiment, write its files and return the calibration records."""
    results_dir = Path(results_dir)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(config.seed)
    records = []
    diagnostics = []

    for dimension in config.dimensions:
        point_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        unit_points = unit_pair_points(dimension, max(config.point_counts), point_rng)
        np.save(unit_points_path(results_dir, dimension), unit_points)

        for name in config.profiles:
            profile = get_profile(name)
            for radius in config.radii:
                # Each scenario takes two child seeds. The second is reserved for
                # independent random inputs and is not used here.
                field_seed, _ = seed_sequence.spawn(2)
                field_rng = np.random.default_rng(field_seed)
                errors, diagnostic = sample_errors(
                    radius * unit_points, profile, config.n_fields, field_rng
                )

                absolute = np.abs(errors)
                reference_errors = errors[:, 0]
                maxima = np.column_stack([
                    absolute[:, :count].max(axis=1) for count in config.point_counts
                ])
                np.savez_compressed(
                    raw_path(results_dir, name, dimension, radius),
                    fixed_reference_point_errors=reference_errors,
                    finite_set_maxima=maxima,
                )

                records.extend(scenario_records(
                    config, profile, dimension, radius, reference_errors, maxima
                ))
                diagnostics.append(dict(
                    profile=name, dimension=dimension, radius=radius, **diagnostic
                ))
                if progress is not None:
                    progress(name, dimension, radius)

    _write_json(results_dir / "calibration_records.json", records)
    _write_json(results_dir / "diagnostics.json", diagnostics)
    _write_json(results_dir / "run_info.json", run_info(config))
    return records


def summarize(config=FixedInputConfig(), results_dir=DEFAULT_RESULTS_DIR):
    """Recompute the calibration records from the saved errors, without sampling."""
    records = []
    for dimension in config.dimensions:
        for name in config.profiles:
            profile = get_profile(name)
            for radius in config.radii:
                with np.load(raw_path(results_dir, name, dimension, radius)) as raw:
                    records.extend(scenario_records(
                        config, profile, dimension, radius,
                        raw["fixed_reference_point_errors"], raw["finite_set_maxima"],
                    ))
    return records


def load_records(config=FixedInputConfig(), results_dir=DEFAULT_RESULTS_DIR):
    """Load the saved calibration records and check them against the settings."""
    path = Path(results_dir) / "calibration_records.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_fixed_input.py first."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        (row["profile"], row["dimension"], row["radius"], row["p"], row["scheme"])
        for row in records
    }
    expected = config.record_keys()
    if (keys != expected or len(records) != len(expected)
            or any(row["n"] != config.n_fields for row in records)):
        raise ValueError("The saved calibration records do not match the settings.")
    return records
