"""Main-text figure: full-race granularity selection.

Same visual grammar as the highlight granularity figure — one panel, cluster
count on the left axis, unclustered fraction on the right, the degenerate zone
shaded, the selected setting marked — so the two selections read as the same
kind of evidence.

Scientific source
-----------------
``results/f1_full/tables/fullrace_granularity_sweep.csv``, frozen. Nothing is
recomputed; the figure reads stored sweep values and draws them.

The full-race sweep shows no cluster-count plateau, so no core region is shaded:
the granularity was fixed by the documented DBCV fallback, not by a plateau.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SELECTED = 40
DPI = 600


def build(rel_root: Path) -> Path:
    d = pd.read_csv(rel_root / "results/f1_full/tables/fullrace_granularity_sweep.csv")
    d = d.sort_values("mcs").reset_index(drop=True)
    assert {"mcs", "n_clusters", "noise_fraction", "degenerate_flag"} <= set(d.columns)
    sel = d[d.mcs == SELECTED].iloc[0]

    fig, (ax, axq) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.10})

    deg = d.loc[d.degenerate_flag.astype(bool), "mcs"]
    for a in (ax, axq):
        if len(deg):
            a.axvspan(deg.min() * 0.93, d.mcs.max() * 1.06, color="#f2c0c0",
                      alpha=.40, lw=0, zorder=0)
        a.axvline(SELECTED, color="#7fc97f", lw=6, alpha=.45, zorder=0)
        a.grid(alpha=.25)

    # --- upper panel: how many clusters, and how much is left unclustered ----
    ax.plot(d.mcs, d.n_clusters, "o-", color="#1f77b4", lw=2, ms=4.0,
            label="clusters recovered (left)")
    ax.plot(SELECTED, sel.n_clusters, "o", ms=11.2, mfc="#ffd700", mec="k",
            mew=1.6, zorder=5,
            label=f"selected setting ({int(sel.n_clusters)} clusters)")
    ax.set_ylabel("number of clusters", color="#1f77b4", fontsize=7.5)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7.5)

    ax2 = ax.twinx()
    ax2.plot(d.mcs, d.noise_fraction, "s--", color="#ff7f0e", lw=1.6, ms=3.2,
             label="unclustered fraction (right)")
    ax2.axhline(0.05, color="#ff7f0e", ls=":", lw=1)
    # wrapped: the single-line label does not fit the shortened axis
    ax2.set_ylabel("fraction of sentences\nleft unclustered", color="#ff7f0e",
                   fontsize=7.5)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=7.5)
    ax2.set_ylim(-0.02, 0.45)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    extra = [plt.Line2D([], [], color="#7fc97f", lw=6, alpha=.45)]
    if len(deg):
        extra.append(plt.Line2D([], [], color="#f2c0c0", lw=6, alpha=.40))
    lab = [f"selected setting ({SELECTED})"]
    if len(deg):
        lab.append(f"degenerate (setting $\\geq$ {int(deg.min())})")
    # both legends sit outside the axes so no series is covered
    ax.legend(h1 + h2 + extra, l1 + l2 + lab, loc="center left",
              bbox_to_anchor=(1.16, 0.5), fontsize=6.2, markerscale=0.5,
              frameon=False, labelspacing=0.42, handlelength=1.8,
              handletextpad=0.5, borderpad=0.0)

    # --- lower panel: the internal quality the fallback was decided on ------
    axq.plot(d.mcs, d.DBCV, "o-", color="#264653", lw=1.6, ms=3.6, label="DBCV")
    axq.plot(d.mcs, d.silhouette, "^-", color="#8ab17d", lw=1.6, ms=3.6,
             label="silhouette")
    if "frac_interpretable" in d.columns:
        axq.plot(d.mcs, d.frac_interpretable, "v:", color="#9d6bbf", lw=1.4,
                 ms=3.6, label="fraction with distinctive vocabulary")
    axq.set_ylabel("internal quality")
    axq.set_xlabel("minimum cluster size")
    axq.set_xscale("log")
    ticks = d.mcs.tolist()
    axq.set_xticks(ticks)
    axq.set_xticklabels([str(int(v)) for v in ticks], fontsize=8)
    axq.minorticks_off()
    axq.legend(loc="center left", bbox_to_anchor=(1.16, 0.5), fontsize=6.2,
               frameon=False, labelspacing=0.42, handlelength=1.8,
               handletextpad=0.5, borderpad=0.0)

    fig.tight_layout()

    out = rel_root / "results/f1_full/figures/fig_fullrace_granularity_selection"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"    {out.name}.pdf/.png/.svg  ({len(d)} sweep points, "
          f"selected={SELECTED} -> {int(sel.n_clusters)} clusters)")
    return out.with_suffix(".pdf")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
