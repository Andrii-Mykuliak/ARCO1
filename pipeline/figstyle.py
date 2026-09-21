"""One manuscript figure style, defined once and applied everywhere.

Every constant here is taken from the existing plotting sources rather than
invented: the series and region colours come from the granularity-selection
figure (the manuscript's style anchor), and the domain colours come from
``figures_ab.DOMAIN_COLOUR``, so a domain never changes colour between figures.

Usage::

    from .figstyle import apply, C, DOMAIN_COLOUR, save
    apply()
    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])
    ...
    save(fig, out_dir, "fig_name")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- semantic palette, carried over from the granularity-selection figure ---
C = {
    "primary": "#1f77b4",      # principal quantitative series (blue)
    "secondary": "#ff7f0e",    # secondary quantitative series (orange)
    "band_regime": "#c9e7c9",  # supported / stable region (pale green)
    "band_core": "#7fc97f",    # selected / core region (stronger pale green)
    "band_bad": "#f2c0c0",     # degenerate / collapse region (pale red)
    "selected": "#ffd700",     # selected setting marker (yellow, dark outline)
    "reference": "#d62728",    # comparison / default marker (red)
    "neutral": "#d9d9d9",      # unassigned / noise points
    "text": "#111111",
    "muted": "#555555",
    "grid": "#cccccc",
}

# --- the four descriptive domains: fixed mapping, never varied --------------
DOMAIN_COLOUR = {
    "Race dynamics": "#2a6f97",
    "Narrative & meta": "#8ab17d",
    "Strategy & technical": "#e76f51",
    "Regulatory": "#9d6bbf",
}
DOMAIN_ORDER = list(DOMAIN_COLOUR)

# --- sequential and diverging scales ---------------------------------------
SEQ = "Blues"      # one perceptually ordered scale for all [0,1] quantities
DIV = "RdBu_r"     # restrained diverging treatment, centred on zero

FIGSIZE = {
    "wide": (7.2, 4.0),      # Figure 1 and its supplementary siblings
    "square": (7.2, 6.4),    # projections
    "tall": (9.0, 11.5),     # per-category matrices and dot plots
}

DPI = 600          # minimum export density for the submission package


def apply() -> None:
    """Install the shared rcParams. Call once before building a figure."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.labelcolor": C["text"],
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "xtick.color": C["text"],
        "ytick.color": C["text"],
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "legend.fontsize": 7.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#999999",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.35,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "font.family": "sans-serif",
        "text.color": C["text"],
    })


def save(fig, out: Path, name: str, tight: bool = True) -> Path:
    """Write vector (SVG) and raster (PNG) at publication resolution."""
    out.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(out / f"{name}.pdf", dpi=DPI, **kw)
    fig.savefig(out / f"{name}.svg", **kw)
    fig.savefig(out / f"{name}.png", dpi=DPI, **kw)
    plt.close(fig)
    print(f"    {name}.pdf + {name}.svg + {name}.png ({DPI} dpi)")
    return out / f"{name}.pdf"
