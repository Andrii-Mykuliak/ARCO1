"""Study-design overview (manuscript Figure 1).

Three side-by-side analytical components, not a pipeline: what the highlight
structure is and how it is characterised; how it is compared with an
independently reconstructed full-race structure; and how thematic composition is
compared over the events present in both corpora. Representation and clustering
detail belongs to Methodology and is deliberately absent.

Scientific source
-----------------
Nothing is computed here. The counts shown are asserted against the frozen
release summary before drawing, so the figure cannot drift from the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HL = "#2a6f97"        # highlight-derived
FR = "#6b7b88"        # full-race-derived
MX = "#e76f51"        # matched-event comparison
INK = "#111111"
MUTED = "#555555"

W_IN, H_IN = 468.3324 / 72.0, 2.62
FIG_NAME = "fig_study_design"


def box(ax, cx, y, w, h, text, colour, *, strong=False, fs=6.2):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.013",
        linewidth=1.5 if strong else 0.8, edgecolor=colour, facecolor="white",
        zorder=3))
    ax.text(cx, y + h / 2, text, ha="center", va="center", zorder=4,
            fontsize=fs + (0.6 if strong else 0), color=INK, linespacing=1.3,
            fontweight="bold" if strong else "normal")


def down(ax, cx, ytop, ybot, colour="#9a9a9a", label=None):
    ax.add_patch(FancyArrowPatch((cx, ytop), (cx, ybot), arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=0.8,
                                 color=colour, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(cx + 0.012, (ytop + ybot) / 2, label, ha="left", va="center",
                fontsize=5.0, color=MUTED, style="italic")


def build(rel_root: Path) -> Path:
    S = json.loads((rel_root / "results/paper_results_summary.json")
                   .read_text(encoding="utf-8"))
    assert S["highlight"]["n_sentences"] == 7647
    assert S["highlight"]["n_clusters"] == 34
    assert S["fullrace"]["n_sentences"] == 16876
    assert S["fullrace"]["n_clusters"] == 48

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cx = [0.175, 0.500, 0.825]                 # three column centres
    bw, bh = 0.288, 0.135
    for x in (0.3375, 0.6625):                 # hairline column separators
        ax.plot([x, x], [0.05, 0.915], color="#e0e0e0", lw=0.7, zorder=1)

    heads = ["THEMATIC DISCOVERY", "CROSS-REGISTER CORRESPONDENCE",
             "MATCHED-EVENT COMPOSITION"]
    for x, ttl, col in zip(cx, heads, (HL, FR, MX)):
        ax.text(x, 0.960, ttl, ha="center", va="center", fontsize=6.4,
                fontweight="bold", color=col)
        ax.plot([x - bw / 2, x + bw / 2], [0.915, 0.915], color=col, lw=1.1)

    # --- column 1: thematic discovery ---------------------------------------
    box(ax, cx[0], 0.745, bw, bh,
        "Highlight commentary\n113 events, 7,647 sentences", HL)
    down(ax, cx[0], 0.745, 0.660)
    box(ax, cx[0], 0.520, bw, bh, "34 thematic categories", HL, strong=True)
    ax.text(cx[0], 0.440, "characterised by four complementary diagnostics",
            ha="center", va="top", fontsize=5.4, color=MUTED, style="italic")
    ax.text(cx[0], 0.370,
            "alternative-model recovery\nresampling stability\n"
            "held-out race replication\nalternative-geometry separability",
            ha="center", va="top", fontsize=5.2, color="#444444", linespacing=1.45)

    # --- column 2: cross-register correspondence ----------------------------
    box(ax, cx[1], 0.745, bw, bh,
        "Full-race commentary\n10 events, 16,876 sentences", FR)
    down(ax, cx[1], 0.745, 0.660, label="independent\nreconstruction")
    box(ax, cx[1], 0.520, bw, bh, "48 full-race clusters", FR, strong=True)
    # the two fixed structures, shown as the paired inputs to the comparison.
    # The highlight side is drawn here, not above the reconstruction, so the
    # figure cannot be read as the taxonomy feeding the full-race clustering.
    ax.text(cx[1], 0.455, "compared, once both structures are fixed, with",
            ha="center", va="top", fontsize=5.3, color=MUTED, style="italic")
    ax.text(cx[1], 0.505, "\u2195", ha="center", va="center", fontsize=8,
            color=MUTED)
    box(ax, cx[1], 0.300, bw, 0.095, "34 thematic categories", HL, fs=6.0)
    ax.text(cx[1], 0.265,
            "single-match  |  split  |  merge  |  non-recovery",
            ha="center", va="top", fontsize=5.4, color=INK)

    # --- column 3: matched-event composition --------------------------------
    box(ax, cx[2], 0.745, bw, bh,
        "10 matched events\npresent in both registers", MX)
    down(ax, cx[2], 0.745, 0.660)
    box(ax, cx[2], 0.535, bw, 0.120,
        "Common assignment using\nfrozen highlight centroids", MX, fs=5.6)
    down(ax, cx[2], 0.535, 0.455)
    box(ax, cx[2], 0.330, bw, 0.120, "Event-level thematic\ncompositions", MX,
        fs=5.6)
    down(ax, cx[2], 0.330, 0.250)
    box(ax, cx[2], 0.125, bw, 0.120, "Paired cross-register\ncomparison", MX,
        strong=True, fs=5.6)

    figs = rel_root / "results/f1_highlights/figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"{FIG_NAME}.pdf", facecolor="white")
    fig.savefig(figs / f"{FIG_NAME}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"    {FIG_NAME}.pdf + .png   {W_IN:.3f} x {H_IN:.3f} in (vector)")
    return figs / f"{FIG_NAME}.pdf"


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
