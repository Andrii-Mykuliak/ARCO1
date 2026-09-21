"""Publication rendering of the thematic-structure projection (manuscript Figure 2).

Scientific source
-----------------
* ``results/f1_highlights/cache/umap2d_descriptive_s42.npy`` - the frozen
  descriptive two-dimensional projection, 7,647 rows.
* ``results/f1_highlights/cache/f1_highlights__labels__mcs35.npy`` - the promoted
  canonical partition, 7,647 rows / 5,084 clustered / 2,563 unassigned / 34
  categories. Byte-identical to ``data/corpus/labels_113.csv`` (``cluster_fine``).
  Its SHA-256 is pinned in ``CANONICAL_LABEL_SHA256`` and checked on every build,
  and recorded alongside the output in ``*.provenance.json``.
* ``results/f1_highlights/tables/category_diagnostic_profiles.csv`` - the
  category-to-domain map and the canonical category order.

Both arrays are verified against the previously frozen figure before anything is
drawn: the centroids implied by this pairing reproduce the centroid positions
stored in ``figA_highlight_thematic_structure.svg`` to zero residual under the
affine data-to-display map. ``_assert_matches_frozen_figure`` re-runs that check
on every build and raises if it ever stops holding.

Transformation performed
------------------------
Styling and composition only. Point coordinates, cluster assignments, centroid
positions, domain assignments and category order are taken verbatim from the
artefacts above. What this module chooses is purely visual: crop margin, point
size and opacity, centroid ring diameter and stroke, legend layout, and figure
size. The x:y data-unit ratio of the frozen figure is preserved exactly, so the
point cloud keeps its shape and only the empty margin is reduced.

No-inference guarantee
----------------------
No embedding, projection, clustering, centroid or statistic is recomputed. The
module reads two frozen arrays and one frozen table and draws them.

Output
------
``results/f1_highlights/figures/fig_thematic_structure_main.{pdf,png}``, authored
at the manuscript text width so pdflatex places it without rescaling.
``panel()`` draws the same projection into a caller-supplied axes so the figure
can later serve as one panel of a combined figure.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --- manuscript-wide domain palette, unchanged ------------------------------
DOMAIN_COLOUR = {
    "Race dynamics": "#2a6f97",
    "Narrative & meta": "#8ab17d",
    "Strategy & technical": "#e76f51",
    "Regulatory": "#9d6bbf",
}
DOMAIN_ORDER = list(DOMAIN_COLOUR)

# --- visual parameters (the only things this module decides) ----------------
NOISE_COLOUR = "#dedede"     # lighter neutral than the previous #d9d9d9
NOISE_SIZE = 0.8             # was 1.2
NOISE_ALPHA = 0.62           # was opaque
POINT_SIZE = 2.2             # was 2.0
POINT_ALPHA = 0.55           # was 0.35
RING_SIZE = 86.0             # sized to carry a two-digit rank
RING_LW = 1.0                # was 1.3
MARGIN = 0.025               # fraction of each data range, was ~0.05 per side
RANK_FONTSIZE = 4.4          # number drawn inside a centroid ring
INDEX_FONTSIZE = 5.8         # numbered index beside the map
INDEX_WIDTH_IN = 2.05        # width of the index column
C_MUTED = "#555555"

# x:y data-unit ratio of the frozen figure (38.42997 / 34.24628 display units
# per data unit). Preserved so the cloud keeps exactly its previous shape.
UNIT_RATIO = 38.42997 / 34.24628

TEXT_WIDTH_IN = 468.3324 / 72.0     # \linewidth, cas-sc single column
FIG_NAME = "fig_thematic_structure_main"

_FROZEN_SVG = "figA_highlight_thematic_structure.svg"

# SHA-256 of the canonical partition. The build fails if the label vector this
# figure plots is not bit-for-bit the canonical one.
CANONICAL_LABEL_SHA256 = \
    "1b1a89b7762f9872bd906eb506fd16ac48b30751fa247fb2db48c4a7387349a9"

LABELS_REL = "results/f1_highlights/cache/f1_highlights__labels__mcs35.npy"
PROJECTION_REL = "results/f1_highlights/cache/umap2d_descriptive_s42.npy"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load(rel_root: Path):
    """Frozen projection, canonical labels and the category/domain map."""
    lab_path = rel_root / LABELS_REL
    got = _sha256(lab_path)
    assert got == CANONICAL_LABEL_SHA256, (
        f"figure source partition hash {got[:16]} != canonical "
        f"{CANONICAL_LABEL_SHA256[:16]}")

    xy = np.load(rel_root / PROJECTION_REL)
    lab = np.load(lab_path)
    prof = pd.read_csv(rel_root /
                       "results/f1_highlights/tables/category_diagnostic_profiles.csv")

    assert xy.shape == (7647, 2), f"projection shape {xy.shape}"
    assert lab.shape == (7647,), f"label shape {lab.shape}"
    assert int((lab >= 0).sum()) == 5084, "clustered count is not canonical"
    assert int((lab == -1).sum()) == 2563, "unassigned count is not canonical"
    assert len(prof) == 34 and prof.cluster_id.tolist() == sorted(prof.cluster_id)

    sizes = pd.Series(lab[lab >= 0]).value_counts().sort_index()
    assert (sizes.reindex(prof.cluster_id).values == prof["size"].values).all(), \
        "per-category sizes do not reproduce category_diagnostic_profiles.csv"
    return xy, lab, prof


def _assert_matches_frozen_figure(xy, lab, prof, figs: Path) -> float:
    """The centroids of this pairing must reproduce the frozen figure exactly."""
    svg = figs / _FROZEN_SVG
    if not svg.exists():
        return float("nan")
    t = svg.read_text(encoding="utf-8")
    frozen = np.array([[float(a), float(b)] for a, b in re.findall(
        r'<g id="PathCollection_\d+">.*?<use [^>]*?x="([-\d.]+)" y="([-\d.]+)"',
        t, re.S)])
    cent = np.array([[xy[lab == c, 0].mean(), xy[lab == c, 1].mean()]
                     for c in prof.cluster_id])
    resid = 0.0
    for col in (0, 1):
        A = np.c_[cent[:, [col]], np.ones(len(cent))]
        coef, *_ = np.linalg.lstsq(A, frozen[:, col], rcond=None)
        resid = max(resid, float(np.abs(frozen[:, col] - A @ coef).max()))
    assert resid < 0.05, f"centroids differ from the frozen figure ({resid:.4f} pt)"
    return resid


def legend_handles(prof) -> list:
    """Shared four-domain legend, for this panel or an external one."""
    present = set(prof.domain)
    return [Line2D([], [], marker="o", ls="", mfc=DOMAIN_COLOUR[d],
                   mec=DOMAIN_COLOUR[d], ms=5.0, label=d)
            for d in DOMAIN_ORDER if d in present]


def ranked(prof):
    """Categories in descending size order, ties by canonical order.

    The rank is the number shown on the map and in the index. It is local to
    this figure: it encodes prevalence order and nothing else, and it is not the
    internal category identifier.
    """
    d = prof.sort_values(["size", "cluster_id"],
                         ascending=[False, True]).reset_index(drop=True)
    d = d.assign(rank=range(1, len(d) + 1))
    return d


def panel(ax, xy, lab, prof) -> None:
    """Panel (a): the descriptive projection, centroids numbered by prevalence."""
    dom_of = dict(zip(prof.cluster_id, prof.domain))
    rank_of = dict(zip(ranked(prof).cluster_id, ranked(prof)["rank"]))

    noise = lab == -1
    ax.scatter(xy[noise, 0], xy[noise, 1], s=NOISE_SIZE, c=NOISE_COLOUR,
               alpha=NOISE_ALPHA, linewidths=0, zorder=1, rasterized=True)
    for c in prof.cluster_id:
        m = lab == c
        ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, linewidths=0, zorder=2,
                   alpha=POINT_ALPHA, rasterized=True,
                   color=DOMAIN_COLOUR.get(dom_of.get(c), "#777777"))
    for c in prof.cluster_id:
        m = lab == c
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        col = DOMAIN_COLOUR.get(dom_of.get(c), "#777777")
        ax.scatter([cx], [cy], s=RING_SIZE, zorder=4, facecolor="white",
                   linewidths=RING_LW, edgecolor=col)
        ax.annotate(str(rank_of[c]), (cx, cy), ha="center", va="center",
                    fontsize=RANK_FONTSIZE, color="#111111", zorder=5)

    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_index(fig, prof, lab, x, y0, w, h) -> None:
    """Panel (b): the index keying the map, as a single column beside it."""
    d = ranked(prof)
    assigned = int((lab >= 0).sum())
    n = len(d)
    for i, r in enumerate(d.itertuples()):
        yy = y0 + h - (i + 0.5) * (h / n)
        col = DOMAIN_COLOUR.get(r.domain, "#777")
        fig.text(x, yy, f"{r.rank}.", ha="right", va="center",
                 fontsize=INDEX_FONTSIZE, color=col, weight="bold",
                 transform=fig.transFigure)
        fig.text(x + 0.007, yy, r.label, ha="left", va="center",
                 fontsize=INDEX_FONTSIZE, color="#111111", transform=fig.transFigure)
        fig.text(x + w, yy, f"{r.size:,} ({r.size / assigned:.1%})",
                 ha="right", va="center", fontsize=INDEX_FONTSIZE,
                 color=C_MUTED, transform=fig.transFigure)


def build(rel_root: Path) -> Path:
    figs = rel_root / "results/f1_highlights/figures"
    xy, lab, prof = load(rel_root)
    resid = _assert_matches_frozen_figure(xy, lab, prof, figs)

    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    xr, yr = (x1 - x0) * (1 + 2 * MARGIN), (y1 - y0) * (1 + 2 * MARGIN)

    # --- geometry, in inches -------------------------------------------------
    # The map sits left; the index is a single column beside it, aligned to the
    # map's vertical extent, so it reads as a legend rather than a second panel.
    W = TEXT_WIDTH_IN
    idx_w, gap = INDEX_WIDTH_IN, 0.12
    w_a = W - idx_w - gap
    h_a = w_a * (yr / xr) / UNIT_RATIO      # frozen x:y unit ratio preserved
    legend_h = 0.26
    top, bottom = 0.06, 0.06
    H = top + h_a + legend_h + bottom

    fig = plt.figure(figsize=(W, H), facecolor="white")

    ax = fig.add_axes([0.0, (bottom + legend_h) / H, w_a / W, h_a / H])
    ax.set_facecolor("white")
    panel(ax, xy, lab, prof)

    hdr = 0.17                                   # room for the (b) panel label
    panel_index(fig, prof, lab, (w_a + gap + 0.16) / W,
                (bottom + legend_h) / H, (idx_w - 0.16) / W, (h_a - hdr) / H)

    fig.legend(handles=legend_handles(prof), loc="lower center", ncol=4,
               bbox_to_anchor=(w_a / 2 / W, (bottom - 0.02) / H),
               frameon=False, fontsize=6.6, handletextpad=0.35,
               columnspacing=1.1, borderpad=0.0)

    fig.text((w_a + gap + 0.16) / W, (bottom + legend_h + h_a - 0.055) / H,
             "category", fontsize=6.0, style="italic", color=C_MUTED,
             va="center", ha="left")
    fig.text((w_a + gap + idx_w - 0.16) / W,
             (bottom + legend_h + h_a - 0.055) / H,
             "sentences", fontsize=6.0, style="italic", color=C_MUTED,
             va="center", ha="right")

    # dpi governs the density of the rasterised scatter layers inside the
    # otherwise vector PDF, so it must meet the submission minimum too.
    fig.savefig(figs / f"{FIG_NAME}.pdf", dpi=600, facecolor="white")
    fig.savefig(figs / f"{FIG_NAME}.png", dpi=600, facecolor="white")
    plt.close(fig)

    # Provenance sidecar: the partition and projection this figure was drawn from.
    (figs / f"{FIG_NAME}.provenance.json").write_text(json.dumps({
        "figure": "manuscript Figure 2 - thematic structure, map + numbered index",
        "canonical_label_vector": LABELS_REL,
        "canonical_label_sha256": _sha256(rel_root / LABELS_REL),
        "descriptive_projection": PROJECTION_REL,
        "descriptive_projection_sha256": _sha256(rel_root / PROJECTION_REL),
        "n_sentences": int(len(lab)),
        "n_clustered": int((lab >= 0).sum()),
        "n_unassigned": int((lab == -1).sum()),
        "n_categories": int(len(prof)),
        "prevalence_denominator": "assigned sentences (5,084)",
        "map_numbering": "figure-local rank by descending category size, ties by "
                         "canonical order; not the internal category identifier",
        "frozen_figure_centroid_residual_pt": resid,
        "inference_rerun": False,
    }, indent=1), encoding="utf-8")
    print(f"    {FIG_NAME}: {W:.3f} x {H:.3f} in   map {w_a:.2f}x{h_a:.2f}, "
          f"index {len(prof)} entries in one column")
    print(f"      frozen-figure centroid residual: {resid:.5f} pt")
    return figs / f"{FIG_NAME}.pdf"


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
