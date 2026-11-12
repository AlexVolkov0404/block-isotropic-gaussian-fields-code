"""Figure styles and saving."""

from __future__ import annotations

from pathlib import Path

from .paths import FIG_DIR

# Figures 3-5 (variance envelopes, probability bounds, admissible radii).
SERIF_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}

# Figure 6 (fixed-input calibration).
SANS_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "text.color": "#20262c",
    "axes.labelcolor": "#20262c",
    "axes.edgecolor": "#646b73",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#343a40",
    "ytick.color": "#343a40",
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "figure.facecolor": "white",
}

BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#777777"


def save_figure(fig, stem, *, dpi, metadata=None, fig_dir=FIG_DIR, **savefig_kwargs):
    """Save a figure as figs/<stem>.pdf and figs/<stem>.png and return both paths.

    The PDF carries no creation date, so the same plot always gives the same file.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = fig_dir / f"{stem}.pdf", fig_dir / f"{stem}.png"
    fig.savefig(pdf_path, metadata={**(metadata or {}), "CreationDate": None},
                **savefig_kwargs)
    fig.savefig(png_path, dpi=dpi, **savefig_kwargs)
    return pdf_path, png_path
