"""Diagnostic support for the recovered highlight thematic structure.

Four complementary diagnostics, four panels, four independent vertical scales.
The dimensions probe different properties and are deliberately never normalised
onto a common scale, summed, ranked or averaged: there is no aggregate score
anywhere in this figure.

Scientific sources
------------------
A  tables/category_diagnostic_profiles.csv        alternative_model_recovery
B  diagnostics/resampling_stability_mc200_category_summary.csv
                                                  degenerate_excluded_mean
C  diagnostics/heldout_race_replication_factorial_runs.csv
                                                  n_pass over viable runs
D  diagnostics/alternative_geometry_mc100_category_summary.csv
                                                  new_mean_100

Every value is read from a frozen artefact; nothing is recomputed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

BLUE = "#2a6f97"
INK = "#111111"
MUTED = "#555555"
BOX = "#4a4a4a"

W_IN, H_IN = 468.3324 / 72.0, 4.45
FIG_NAME = "fig_diagnostic_support"
RNG = np.random.default_rng(0)      # jitter only; affects no reported value


def _box(ax, data, pos, width=0.52):
    ax.boxplot(data, positions=[pos], widths=width, showfliers=False,
               medianprops=dict(color=BLUE, lw=1.4),
               boxprops=dict(color=BOX, lw=0.8),
               whiskerprops=dict(color=BOX, lw=0.8),
               capprops=dict(color=BOX, lw=0.8))
    ax.scatter(pos + (RNG.random(len(data)) - 0.5) * width * 0.62, data,
               s=7, color=BLUE, alpha=0.45, linewidths=0, zorder=3)


def _note(ax, text):
    """Panel note, pinned to a reserved band at the top so it never overlaps ink."""
    ax.text(0.02, 0.975, text, transform=ax.transAxes, fontsize=5.9,
            color=MUTED, va="top", ha="left", linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none",
                      alpha=0.92))


def _headroom(ax, frac=0.30):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * frac)


def _tidy(ax, label):
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color="#e4e4e4", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.7)
    ax.set_title(label, loc="left", fontsize=7.4, fontweight="bold", pad=4)


def build(rel_root: Path) -> Path:
    T = rel_root / "results/f1_highlights/tables"
    D = rel_root / "results/f1_highlights/diagnostics"

    prof = pd.read_csv(T / "category_diagnostic_profiles.csv")
    res = pd.read_csv(D / "resampling_stability_mc200_category_summary.csv")
    geo = pd.read_csv(D / "alternative_geometry_mc100_category_summary.csv")
    runs = pd.read_csv(D / "heldout_race_replication_factorial_runs.csv")

    k = np.round(prof.alternative_model_recovery.to_numpy() * 7).astype(int)
    r = res["degenerate_excluded_mean"].dropna().to_numpy()
    g = geo["new_mean_100"].dropna().to_numpy()
    viable = runs[~runs.is_degenerate.astype(bool)]
    assert len(prof) == 34 and len(r) == 34 and len(g) == 34
    assert len(runs) == 200 and int(runs.is_degenerate.astype(bool).sum()) == 47

    fig, axes = plt.subplots(2, 2, figsize=(W_IN, H_IN), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.155,
                        wspace=0.26, hspace=0.42)
    (a, b), (c, d) = axes

    # --- A: recovery under alternative model specifications -----------------
    counts = [int((k == i).sum()) for i in range(8)]
    a.bar(range(8), counts, color=BLUE, width=0.68, edgecolor="white", lw=0.5)
    a.set_xticks(range(8))
    a.set_xlabel("alternative models recovering the category", fontsize=6.6)
    a.set_ylabel("highlight categories", fontsize=6.6)
    # No count-above-a-cut annotation: the dimension is the fraction of the seven
    # alternatives reaching the recovery threshold, and no secondary cut over that
    # fraction is defined anywhere in the manuscript (Section 4.4).
    a.set_ylim(0, max(counts) * 1.12)
    _tidy(a, "A  Alternative-model recovery")

    # --- B: resampling stability -------------------------------------------
    _box(b, r, 0)
    b.set_xticks([])
    # a Jaccard similarity is bounded at 1.0, so the axis stops there and the
    # note is given room by widening the view window, not by headroom above 1.0
    b.set_xlim(-1.20, 0.60)
    b.set_ylabel("category-level stability", fontsize=6.6)
    b.set_ylim(0.0, 1.0)
    b.text(0.02, 0.985, "200 runs: 165 viable,\n35 degenerate\n"
           "environment-conditioned\nestimate", transform=b.transAxes,
           fontsize=5.9, color=MUTED, va="top", ha="left", linespacing=1.35,
           bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none",
                     alpha=0.92))
    _tidy(b, "B  Resampling stability")

    # --- C: held-out race replication, four cells kept separate ------------
    cells = [("A", 30), ("A", 35), ("B", 30), ("B", 35)]
    for i, (cache, mcs) in enumerate(cells):
        sub = viable[(viable.cache == cache) & (viable.mcs == mcs)]["n_pass"]
        _box(c, sub.to_numpy(), i, width=0.46)
    # two-level axis: the setting on the ticks, the realisation grouped beneath,
    # so neither label has to be abbreviated into an internal code
    c.set_xticks(range(4))
    c.set_xticklabels(["30", "35", "30", "35"], fontsize=6.2)
    c.set_xlim(-0.6, 3.6)
    c.set_xlabel("minimum cluster size", fontsize=6.6, labelpad=22)
    # the two realisations are separated in the plot area and each pair is
    # bracketed beneath the ticks, so the grouping does not rest on label
    # position alone
    c.axvline(1.5, color="#d5d5d5", lw=0.7, zorder=0)
    span = mtransforms.blended_transform_factory(c.transData, c.transAxes)
    for (x0, x1), lab in (((0, 1), "realisation 1"), ((2, 3), "realisation 2")):
        c.plot([x0, x1], [-0.118, -0.118], transform=span, color=MUTED,
               lw=0.7, clip_on=False, zorder=5)
        c.annotate(lab, xy=((x0 + x1) / 2, -0.138),
                   xycoords=("data", "axes fraction"),
                   ha="center", va="top", fontsize=6.0, color=MUTED,
                   annotation_clip=False)
    c.set_ylabel("categories replicating per run", fontsize=6.6)
    _headroom(c, 0.18)
    _note(c, "200 runs; 47 degenerate")
    _tidy(c, "C  Held-out race replication")

    # --- D: alternative-geometry separability ------------------------------
    # No 0.50 reference line and no count above it: separability is a continuous
    # statistic whose absolute level is conditional on the fixed per-category
    # subsample size, so it is read comparatively across categories, not against
    # a cut (Section 4.4).
    _box(d, g, 0)
    d.set_xticks([])
    d.set_xlim(-0.6, 0.6)
    d.set_ylabel("category-level separability", fontsize=6.6)
    _headroom(d, 0.06)
    _tidy(d, "D  Alternative-geometry separability")

    figs = rel_root / "results/f1_highlights/figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"{FIG_NAME}.pdf", facecolor="white")
    fig.savefig(figs / f"{FIG_NAME}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"    {FIG_NAME}.pdf + .png   {W_IN:.3f} x {H_IN:.3f} in (vector)")
    return figs / f"{FIG_NAME}.pdf"


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
