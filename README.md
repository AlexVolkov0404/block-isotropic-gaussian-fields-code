# Block-Isotropic Gaussian Fields: Numerical Code

Companion code for the manuscript *On the Local Structure and Approximation Stability of Block Isotropic Gaussian Fields*. It evaluates bounds for the quadratic Taylor approximation about matched inputs and assesses their coverage by simulation.

| Notebook | Manuscript output | What it computes |
| --- | --- | --- |
| [01_variance_envelopes.ipynb](01_variance_envelopes.ipynb) | Figure 3 | Pointwise variance envelopes for squared exponential, rational quadratic, and Matérn covariance profiles. |
| [02_probability_bounds_and_radii.ipynb](02_probability_bounds_and_radii.ipynb) | Figures 4 and 5 | Lower bounds on the probability that the error stays within a given tolerance, jointly over the field and independent Gaussian inputs, and the radii certified by those bounds. |
| [03_fixed_input_calibration.ipynb](03_fixed_input_calibration.ipynb) | Table 1 and Figure 6 | Monte Carlo coverage for the refined pointwise and simultaneous thresholds, with a comparison of theoretical thresholds and empirical 95th percentiles. |

## Layout

```
bigf/                 shared Python modules
  profiles.py         covariance profiles h and their derivatives
  bounds.py           variance envelope, thresholds, and probability bounds
  inputs.py           Gaussian input model for the joint bounds
  covariance.py       covariances of the field, its Taylor approximation, and the error
  sampling.py         point sets and sampling of approximation errors
  calibration.py      coverage statistics
  fixed_input.py      the fixed-input Monte Carlo experiment
  plotting.py         figure styles and saving
  paths.py            output locations
scripts/
  run_fixed_input.py  runs the Monte Carlo experiment behind Table 1 and Figure 6
  sync_to_paper.sh    copies the manuscript's figures into the paper folder
tests/                tests of the modules and of the committed results
01–03 notebooks       build the figures and Table 1
figs/                 figures and the data plotted in Figures 4 and 5
results/              Monte Carlo results, Table 1, and the data for Figure 6
```

The notebooks hold the settings and the plotting code for each figure. The formulas and the simulation are in `bigf/`, so every notebook and script uses the same implementation. A new experiment adds a module in `bigf/`, a script in `scripts/`, a folder in `results/`, and a notebook.

## Run the code

Use Python 3.12 (see [Computing environment](#computing-environment)). On macOS or Linux, run:

```bash
git clone https://github.com/AlexVolkov0404/block-isotropic-gaussian-fields-code.git
cd block-isotropic-gaussian-fields-code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Start Jupyter in the repository root, open any notebook, and run its cells from top to bottom. To run all three from the command line instead:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Notebooks 01 and 02 compute everything they plot. Notebook 03 reads the Monte Carlo results saved in `results/fixed_input/`.

## Monte Carlo experiment

```bash
python scripts/run_fixed_input.py
```

This samples 10,000 independent field realizations for each of 60 scenarios: four covariance profiles, five block dimensions, and three radii. It uses seed `2026102302` and takes a few seconds. Coverage is evaluated at nominal levels of 80%, 90%, 95%, 99%, and 99.9%. The pointwise comparison uses the absolute error at a reference pair x*(h, R), a pair in the product of two balls of radius R at which the error standard deviation is largest, so the pointwise threshold is checked where it is tightest. It is the antidiagonal pair (R e_1, -R e_1) except for the rational quadratic and Matérn profiles at R = 0.5. The simultaneous comparison uses the maximum absolute error over 128 fixed input pairs per field realization.

`--reference-pair antidiagonal` uses (R e_1, -R e_1) in every scenario, the reference pair of results generated before this option existed, and `--results-dir DIR` writes the results to another folder.

The script writes these files to `results/fixed_input/`:

- `raw/`: the fixed input pairs and, for each scenario, the sampled errors at the reference pair and the maxima over the first 32, 64, and 128 pairs.
- `calibration_records.json`: coverage, threshold, and empirical quantile for each scenario, level, and threshold; the pointwise records also describe the reference pair.
- `diagnostics.json`: checks of the sampling construction.
- `run_info.json`: the settings and the Python, NumPy, and SciPy versions of the run.

The values in the manuscript's Table 1 and Figure 6 come from the committed files. Running the script again overwrites them. A new run uses the same design and seed, but it can give slightly different Monte Carlo values in a different numerical environment, for example with a different NumPy, SciPy, or BLAS build. To rebuild `calibration_records.json` from the committed errors without sampling, run:

```bash
python scripts/run_fixed_input.py --summarize
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests check the profiles, bounds, and sampling against closed forms and against each other. `tests/test_consistency.py` also recomputes the committed files in `figs/` and `results/`.

## Computing environment

`requirements.txt` pins the recommended environment, set on 2026-11-10. The package versions are the latest releases on that date, and each provides Python 3.12 wheels for macOS, Linux, and Windows:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| JupyterLab | 4.6.6 |

Python 3.12 receives source-only security releases, so python.org no longer offers macOS or Windows installers for its latest versions. Install it with a tool such as [uv](https://docs.astral.sh/uv/) (`uv python install 3.12`), Homebrew (`brew install python@3.12`), or pyenv.

The committed figures and results were generated in this environment. `results/fixed_input/run_info.json` records the versions used for the Monte Carlo run.

## Generated files

- `figs/`: Figures 3–6 in PDF and PNG format, plus the data for Figures 4 and 5 in CSV format. The PDFs carry no creation date, so an unchanged plot gives an unchanged file.
- `results/fixed_input/`: the Monte Carlo results listed above, Table 1 in LaTeX and CSV format, and the data for Figure 6.
