# Block-Isotropic Gaussian Fields: Numerical Notebooks

Companion code for the manuscript *On the Local Structure and Approximation Stability of Block Isotropic Gaussian Fields*. The notebooks evaluate bounds for the quadratic Taylor approximation about matched inputs and assess their coverage by simulation.

| Notebook | Manuscript output | What it computes |
| --- | --- | --- |
| [01_variance_envelopes.ipynb](01_variance_envelopes.ipynb) | Figures 2 and 3 | Pointwise variance envelopes for squared exponential, rational quadratic, and Matérn covariance profiles. |
| [02_probability_bounds_and_radii.ipynb](02_probability_bounds_and_radii.ipynb) | Figures 4 and 5 | Lower bounds on the probability that the error stays within a given tolerance, jointly over the field and independent Gaussian inputs, and the radii certified by those bounds. |
| [03_fixed_input_calibration.ipynb](03_fixed_input_calibration.ipynb) | Table 1 and Figure 6 | Monte Carlo coverage for the refined pointwise and simultaneous thresholds, with a comparison of theoretical thresholds and empirical 95th percentiles. |

## Run the notebooks

Use Python 3.12 (see [Computing environment](#computing-environment)). On macOS or Linux, run:

```bash
git clone https://github.com/AlexVolkov0404/block-isotropic-gaussian-fields-code.git
cd block-isotropic-gaussian-fields-code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open any notebook and run its cells from top to bottom. Start Jupyter in the repository root so the relative output paths resolve there.

For Notebook 03, set `RUN_NEW_SIMULATION = True` in the cell that selects the data source to run the experiment from a fresh clone. It uses seed `2026102302` and 10,000 independent field realizations for each of 60 scenarios: four covariance profiles, five block dimensions, and three radii. Coverage is evaluated at nominal levels of 80%, 90%, 95%, 99%, and 99.9%. The simultaneous comparison uses the maximum absolute error over 128 fixed input pairs per field realization.

The values in the manuscript's Table 1 and Figure 6 come from the run saved in `results/fixed_input/calibration_rerun.json` (seed `2026102302`). To rebuild Table 1 and the Figure 6 data from these records without rerunning the simulation, copy that file to `data/CALIBRATION_DATA.json` and set `RUN_NEW_SIMULATION = False`. A new run follows the same design and seed but can give slightly different Monte Carlo values in a different numerical environment, for example with a different NumPy, SciPy, or BLAS build.

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

The Monte Carlo records behind the manuscript's Table 1 and Figure 6 (`results/fixed_input/calibration_rerun.json`) and the committed Figures 2–6 were generated on 2026-11-08 with Python 3.13.5 and Matplotlib 3.10.6. Rebuilding Table 1 and the Figure 6 data from those records gives the same files in the recommended environment.

## Generated files

- `figs/`: Figures 2–6 in PDF and PNG format, plus the data for Figures 4 and 5 in CSV format.
- `results/fixed_input/`: Table 1 in LaTeX and CSV format and the data for Figure 6. A new simulation also saves calibration records, diagnostics, and the sampled fixed-pair errors and finite-set maxima used for the comparisons.
