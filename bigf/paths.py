"""Output locations, resolved from the repository root rather than the working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figs"
RESULTS_DIR = REPO_ROOT / "results"


def relative(path):
    """Return a path relative to the repository root, for printing."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
