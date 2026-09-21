"""Figures A and B - highlight thematic structure and the diagnostic profile.

Both are generated from frozen artefacts, deterministically, with no manual
editing step. Neither reads anything under results/*/legacy/.

FIGURE A  a 2-D map of the 34 canonical categories.

  The 2-D view is a DESCRIPTIVE projection. Clustering was performed in the
  5-component UMAP space, not here, so planar distance in this figure is not a
  quantitative representation of the semantic geometry. The caption says so, and
  the figure deliberately plots labelled centroids over a light sentence cloud
  rather than 34 colour-coded point clouds, which would be unreadable.

FIGURE B  category x diagnostic-dimension profile.

  Replaces the superseded "four-axis vote" visual. All four dimensions are
  already on a native [0, 1] scale, so NO normalisation is applied - values are
  shown as computed. There is no thresholding, no pass/fail shading, no aggregate
  column and no verdict band: the point of the figure is that support is
  heterogeneous, which a vote would hide.

  Category order is fixed: by domain, then by category id. It is never sorted
  by score, which would manufacture a visual gradient. The matrix is laid out
  horizontally - dimensions as rows, categories as columns - over two stacked
  panels of 17 categories so the cells stay legible at the text width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Column headings are wrapped so that no single word is wider than one cell.
HEADINGS = {
    "alternative_model_recovery": "alternative-\nmodel\nrecovery",
    "resampling_stability": "resampling\nstability",
    "heldout_replication": "held-out\nrace\nreplication",
    "geometry_separability": "alternative-\ngeometry\nseparability",
}

DIMENSIONS = [
    ("alternative_model_recovery", "alternative-model\nrecovery",
     "fraction of 7 alternative pipelines recovering the category"),
    ("resampling_stability", "resampling\nstability",
     "mean best-overlap Jaccard across 80% subsamples"),
    ("heldout_replication", "held-out race\nreplication",
     "replication of enriched vocabulary on held-out races"),
    ("geometry_separability", "alternative-geometry\nseparability",
     "mean best-overlap Jaccard under Ward at fixed K"),
]
DOMAIN_ORDER = ["Race dynamics", "Narrative & meta", "Strategy & technical",
                "Regulatory"]
DOMAIN_COLOUR = {"Race dynamics": "#2a6f97", "Narrative & meta": "#8ab17d",
                 "Strategy & technical": "#e76f51", "Regulatory": "#9d6bbf"}


DPI = 600          # minimum export density for the submission package
TEXT_WIDTH_IN = 468.3324 / 72.0   # \\linewidth, cas-sc single column


def _save(fig, out: Path, name: str, tight: bool = True):
    out.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    # PDF is the file LaTeX picks: vector text and axes, with any imshow layer
    # embedded at DPI. The PNG is a compatibility fallback at the same density.
    fig.savefig(out / f"{name}.pdf", dpi=DPI, **kw)
    fig.savefig(out / f"{name}.svg", **kw)
    fig.savefig(out / f"{name}.png", dpi=DPI, **kw)
    plt.close(fig)
    print(f"    {name}.pdf + {name}.svg + {name}.png ({DPI} dpi)")


def figure_a(emb2d: np.ndarray, labels: np.ndarray, meta: pd.DataFrame,
             out: Path, name: str = "figA_highlight_thematic_structure"):
    """emb2d: (n, 2) descriptive projection. labels: canonical cluster ids."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    noise = labels == -1
    ax.scatter(emb2d[noise, 0], emb2d[noise, 1], s=1.2, c="#d9d9d9",
               linewidths=0, zorder=1, rasterized=True)
    dom_of = dict(zip(meta.cluster_id, meta.domain))
    for c in sorted(set(labels.tolist()) - {-1}):
        m = labels == c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=2.0, linewidths=0, zorder=2,
                   alpha=0.35, c=DOMAIN_COLOUR.get(dom_of.get(c), "#777"),
                   rasterized=True)
    for c in sorted(set(labels.tolist()) - {-1}):
        m = labels == c
        cx, cy = emb2d[m, 0].mean(), emb2d[m, 1].mean()
        ax.scatter([cx], [cy], s=52, facecolor="white", zorder=3,
                   edgecolor=DOMAIN_COLOUR.get(dom_of.get(c), "#777"), linewidths=1.3)
        # No label is drawn on the centroid: the numeric cluster id is an
        # internal identifier and must not appear in a reader-facing figure.
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=v, mec=v, ms=6, label=k)
                       for k, v in DOMAIN_COLOUR.items() if k in set(meta.domain)],
              frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4)
    ax.set_title("Thematic structure of Formula 1 highlight commentary\n"
                 "34 categories; grey = unassigned sentences",
                 fontsize=9)
    fig.text(0.5, -0.10,
             "Descriptive two-dimensional projection. Clustering was performed in the "
             "five-component\nreduced space, not in this plane, so planar distance is "
             "not a quantitative representation\nof the semantic geometry.",
             ha="center", fontsize=6.6, color="#555")
    _save(fig, out, name)
    return out / f"{name}.svg"


def figure_b(profiles: pd.DataFrame, out: Path,
             name: str = "figB_category_diagnostic_profile",
             show_values: bool = True, show_note: bool = True):
    """Category x diagnostic-dimension heatmap, two blocks side by side.

    Rows are categories, columns are the four complementary dimensions. The 34
    categories keep their canonical order - by domain, then in canonical order,
    never by score - and are split into two 17-row blocks placed side by side,
    which makes the figure wide and short rather than tall. Both blocks share
    one sequential colour scale and one colour bar.
    """
    df = profiles.copy()
    df["_d"] = df.domain.map({d: i for i, d in enumerate(DOMAIN_ORDER)}).fillna(9)
    df = df.sort_values(["_d", "cluster_id"]).reset_index(drop=True)
    cols = [c for c, _, _ in DIMENSIONS]
    M = df[cols].to_numpy(float)                       # categories x dimensions

    n = len(df)
    half = (n + 1) // 2
    blocks = [(0, half), (half, n)]

    # --- geometry, in inches -------------------------------------------------
    # Cell width is solved from the text width so the figure is authored at
    # exactly \linewidth: pdflatex then places it at scale 1.0 and the 600 dpi
    # export density survives into the page.
    lab_w, strip_w = 1.02, 0.05
    row_h, head_h = 0.155, 0.44
    gap, cbar_w, top, bottom = 0.22, 0.44, 0.05, 0.26
    cell_w = (TEXT_WIDTH_IN - gap - cbar_w - 2 * (lab_w + strip_w)) / (2 * len(cols))
    blk_w = lab_w + strip_w + cell_w * len(cols)
    mat_h = row_h * half
    W = blk_w * 2 + gap + cbar_w
    H = top + head_h + mat_h + bottom

    fig = plt.figure(figsize=(W, H), facecolor="white")
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ffffff")
    n_na, im = 0, None

    for k, (lo, hi) in enumerate(blocks):
        sub = M[lo:hi, :]
        m = sub.shape[0]
        x0 = k * (blk_w + gap) + lab_w + strip_w
        ax = fig.add_axes([x0 / W, bottom / H, (cell_w * len(cols)) / W,
                           (row_h * m) / H])
        im = ax.imshow(np.ma.masked_invalid(sub), cmap=cmap, aspect="auto",
                       vmin=0.0, vmax=1.0)
        ax.set_yticks(range(m))
        ax.set_yticklabels([str(r.label)[:30] for r in df.iloc[lo:hi].itertuples()],
                           fontsize=5.6)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([HEADINGS[c] for c in cols], fontsize=5.4,
                           linespacing=1.05)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", length=0, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)

        for i, r in enumerate(df.iloc[lo:hi].itertuples()):
            # domain marker: a thin coloured strip immediately left of the label
            ax.add_patch(plt.Rectangle(
                (-0.5 - (strip_w / cell_w), i - .5), strip_w / cell_w * 0.55, 1,
                clip_on=False, color=DOMAIN_COLOUR.get(r.domain, "#777")))
            for j in range(len(cols)):
                v = sub[i, j]
                if not np.isfinite(v):
                    n_na += 1
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="#bbb", lw=0.0))
                    if show_values:
                        # no held-out estimate: the training-split refit did
                        # not reconstruct this category, so the rate has no
                        # denominator (heldout_replication.has_estimate)
                        ax.text(j, i, "no est.", ha="center", va="center",
                                fontsize=5.2, color="#777", style="italic")
                elif show_values:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if v > 0.58 else "#111")

    cax = fig.add_axes([(W - cbar_w + 0.10) / W, (bottom + mat_h * 0.10) / H,
                        0.085 / W, (mat_h * 0.80) / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Diagnostic value", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8)

    # compact domain key, inline beneath the colour bar label
    fig.legend(handles=[Line2D([], [], marker="s", ls="", mfc=DOMAIN_COLOUR[d],
                               mec=DOMAIN_COLOUR[d], ms=4, label=d)
                        for d in DOMAIN_ORDER if d in set(df.domain)],
               loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4,
               frameon=False, fontsize=5.8, handletextpad=0.3,
               columnspacing=1.2, borderpad=0.0)

    _save(fig, out, name, tight=False)
    if show_values:
        df[["cluster_id", "label", "domain", "size"] + cols].to_csv(
            out.parent / "tables" / "figB_source_category_diagnostic_profile.csv",
            index=False)
        print("    figB source CSV -> "
              "tables/figB_source_category_diagnostic_profile.csv")
    return out / f"{name}.svg"


FIGURE_NAME = "figB_category_diagnostic_profile"


def build_all(rel_root: Path) -> dict:
    """The manuscript diagnostic-profile figure: every cell value shown, and no
    in-plot note - the methodological qualification lives in the caption."""
    prof = pd.read_csv(rel_root /
                       "results/f1_highlights/tables/category_diagnostic_profiles.csv")
    out = rel_root / "results/f1_highlights/figures"
    print("  diagnostic-profile figure:")
    return {"figure": str(figure_b(prof, out, name=FIGURE_NAME,
                                   show_values=True, show_note=False))}


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parent.parent)
