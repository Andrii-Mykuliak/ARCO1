"""Conceptual problem figure (manuscript Figure 1).

Why the study exists, not how it is carried out: one underlying event stream,
two commentary registers derived from it, and the three questions that follow.
Deliberately carries no method, no parameter and no result value.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HL = "#2a6f97"        # highlight register
FR = "#6b7b88"        # full-race register
EV = "#b9c6d0"        # the shared event stream
ACC = "#e76f51"       # the questions band
INK = "#111111"
MUTED = "#555555"

W_IN, H_IN = 468.3324 / 72.0, 2.95
FIG_NAME = "fig_problem_overview"


def build(rel_root: Path) -> Path:
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- one underlying event stream ---------------------------------------
    ax.text(0.5, 0.965, "THE SAME UNDERLYING EVENTS", ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=MUTED)
    x0, x1, ys, hs = 0.215, 0.785, 0.865, 0.052
    ax.add_patch(Rectangle((x0, ys), x1 - x0, hs, facecolor="#f2f5f7",
                           edgecolor="#d3dde3", linewidth=0.7, zorder=2))
    n = 14
    for i in range(n):                     # abstract event markers, no real text
        w = (x1 - x0) / (2 * n + 1)
        ax.add_patch(Rectangle((x0 + w * (2 * i + 1), ys + 0.011), w, hs - 0.022,
                               facecolor=EV, edgecolor="none", zorder=3))
    ax.text(0.5, ys - 0.030, "a continuous, high-rate event stream",
            ha="center", va="top", fontsize=5.3, color=MUTED, style="italic")

    # --- two registers derived from it --------------------------------------
    bw, bh, yb = 0.300, 0.115, 0.560
    cxl, cxr = 0.285, 0.715
    for cx, col, title, sub in [
            (cxl, FR, "Full-race commentary", "continuous, as the event unfolds"),
            (cxr, HL, "Highlight commentary", "condensed, after the fact")]:
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, yb), bw, bh,
            boxstyle="round,pad=0.004,rounding_size=0.016",
            linewidth=1.4, edgecolor=col, facecolor="white", zorder=3))
        ax.text(cx, yb + bh * 0.62, title, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, yb + bh * 0.27, sub, ha="center", va="center",
                fontsize=5.4, color=MUTED, zorder=4)
        ax.add_patch(FancyArrowPatch((0.5, ys), (cx, yb + bh),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.9, color="#a9b4bc",
                                     shrinkA=2, shrinkB=2, zorder=2))

    # --- the three questions ------------------------------------------------
    yq = 0.395
    ax.plot([cxl, cxl, cxr, cxr], [yb, yq, yq, yb], color="#a9b4bc", lw=0.9,
            zorder=2, solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch((0.5, yq), (0.5, 0.345), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9,
                                 color="#a9b4bc", shrinkA=0, shrinkB=0, zorder=2))
    ax.text(0.5, 0.310, "WHAT HAPPENS TO THEMATIC CONTENT ACROSS THE TWO?",
            ha="center", va="center", fontsize=6.4, fontweight="bold", color=ACC)
    ax.plot([0.175, 0.825], [0.278, 0.278], color=ACC, lw=1.0)

    qs = [(0.175, "STRUCTURE",
           "Does fine-grained thematic\nstructure re-emerge\nacross registers?"),
          (0.500, "CORRESPONDENCE",
           "Which themes persist, split,\nmerge, or fail to recover\nacross registers?"),
          (0.825, "COMPOSITION",
           "For the same events, does\nrelative thematic emphasis\ndiffer between registers?")]
    for x, head, body in qs:
        ax.text(x, 0.235, head, ha="center", va="center", fontsize=6.2,
                fontweight="bold", color=INK)
        ax.text(x, 0.185, body, ha="center", va="top", fontsize=5.5,
                color="#333333", linespacing=1.45)

    figs = rel_root / "results/f1_highlights/figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"{FIG_NAME}.pdf", facecolor="white")
    fig.savefig(figs / f"{FIG_NAME}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"    {FIG_NAME}.pdf + .png   {W_IN:.3f} x {H_IN:.3f} in (vector)")
    return figs / f"{FIG_NAME}.pdf"


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
