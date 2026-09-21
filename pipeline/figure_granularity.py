"""Figure 1: highlight-corpus granularity selection.

Regenerated from the current release sweep (`mcs_sensitivity.csv`, whose
min_cluster_size = 35 row reproduces the frozen canonical partition exactly) plus
the untuned reference row from `mcs_sensitivity_extension.csv`, which carries
only a cluster count and a noise rate.

The earlier edition of this figure plotted a log-odds enrichment screen on the
right axis. That screen is not the FDR-corrected procedure the paper uses for
reported signatures, so its level disagrees with the enrichment reported in the
Results. The right axis here shows the noise fraction instead, which is the
quantity the pre-specified degeneracy rule is stated in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

X_MAX = 70            # displayed range; the sweep itself runs further
CORE = (30, 40)        # cluster count within +-1 of the selected solution
PLATEAU = (30, 50)     # evidence-supported regime
SELECTED = 35


def build(rel_root: Path) -> Path:
    t = rel_root / "results/f1_highlights/tables"
    sweep = pd.read_csv(t / "mcs_sensitivity.csv").rename(
        columns={"mcs": "min_cluster_size", "noise_rate": "noise"})
    ext = pd.read_csv(t / "mcs_sensitivity_extension.csv")
    ext = ext[ext["canonical"] == True]  # noqa: E712 - explicit canonical row only
    lib = ext.iloc[0]
    sweep = pd.concat([
        pd.DataFrame([{"min_cluster_size": int(lib["mcs"]),
                       "n_clusters": int(lib["n_clusters"]),
                       "noise": float(lib["noise_rate"])}]),
        sweep[["min_cluster_size", "n_clusters", "noise", "dbcv", "silhouette",
               "frac_interp"]],
    ]).sort_values("min_cluster_size").reset_index(drop=True)

    # the reference configuration is not a sweep point: it comes from a
    # different table and has no internal-quality values, so the curves in both
    # panels run over the sweep alone and it is drawn as a lone marker
    ref = sweep[sweep["dbcv"].isna()].iloc[0]
    sw = sweep.dropna(subset=["dbcv"])
    # the size-10 internal-quality point was recomputed under a later software
    # stack that does not reproduce the frozen rows, so it is carried
    # separately and drawn with open markers rather than merged into the sweep
    rq = pd.read_csv(t / "mcs_sensitivity_mcs10_recomputed.csv").iloc[0]

    fig, (ax, axq) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.10})
    # the regime shading spans both panels so the regions read as one reading
    for _a in (ax, axq):
        _a.axvspan(*PLATEAU, color="#c9e7c9", alpha=.45, lw=0, zorder=0)
        _a.axvspan(*CORE, color="#7fc97f", alpha=.45, lw=0, zorder=0)
    ax.axvspan(*PLATEAU, color="#c9e7c9", alpha=.45, lw=0,
               label=f"evidence-supported regime [{PLATEAU[0]}, {PLATEAU[1]}]")
    ax.axvspan(*CORE, color="#7fc97f", alpha=.45, lw=0,
               label=f"core region [{CORE[0]}, {CORE[1]}]")
    collapse = sw.loc[sw.n_clusters < 10, "min_cluster_size"].min()
    for _a in (ax, axq):
        _a.axvspan(collapse, X_MAX * 1.06, color="#f2c0c0", alpha=.40, lw=0,
                   zorder=0)
    ax.axvspan(collapse, X_MAX * 1.06, color="#f2c0c0", alpha=.40,
               lw=0, label=f"degenerate (setting $\\geq$ {int(collapse)})")

    ax.plot([ref.min_cluster_size] + list(sw.min_cluster_size),
            [ref.n_clusters] + list(sw.n_clusters), "o-", color="#1f77b4",
            lw=2, ms=4.0, label="categories recovered (left)")
    sel = sw[sw.min_cluster_size == SELECTED].iloc[0]
    ax.plot(SELECTED, sel.n_clusters, "o", ms=11.2, mfc="#ffd700", mec="k",
            mew=1.6, zorder=5, label=f"selected setting ({int(sel.n_clusters)} categories)")
    ax.annotate(f"selected\n{SELECTED}", (SELECTED, sel.n_clusters),
                # placed left of the marker, in the gap between the category
                # curve and the degeneracy level, so it crosses no plotted line
                textcoords="offset points", xytext=(-56, -28), ha="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=.8))
    ax.plot(ref.min_cluster_size, ref.n_clusters, "X", ms=9.6, color="#d62728",
            mec="k", zorder=5,
            label=f"untuned HDBSCAN reference, minimum cluster size "
                  f"{int(ref.min_cluster_size)}\n({int(ref.n_clusters)} categories)")

    # coloured axis text at 75% of the default label size
    ax.set_ylabel("number of categories", color="#1f77b4", fontsize=7.5)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7.5)
    ax.set_xscale("log")
    ticks = [v for v in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70)
             if v <= X_MAX]
    ax.set_xlim(9.0, X_MAX * 1.06)
    ax.minorticks_off()
    ax.grid(alpha=.25)

    ax2 = ax.twinx()
    ax2.plot([ref.min_cluster_size] + list(sw.min_cluster_size),
             [ref.noise] + list(sw.noise), "s--", color="#ff7f0e",
             lw=1.6, ms=3.2, label="unclustered fraction (right)")
    ax2.axhline(0.05, color="#ff7f0e", ls=":", lw=1)
    # wrapped: the single-line label no longer fits the shortened axis
    # wrapped: the single-line label does not fit the shortened axis
    ax2.set_ylabel("fraction of sentences\nleft unclustered", fontsize=7.5,
                   color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=7.5)
    ax2.set_ylim(-0.02, 0.45)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(1.16, 0.5),
              fontsize=6.2, markerscale=0.5, frameon=False,
              labelspacing=0.42, handlelength=1.8,
              handletextpad=0.5, borderpad=0.0)

    # --- lower panel: the internal quality behind the selection -------------
    q = sw
    series = [("dbcv", "o", "-", "#264653", 1.6, "DBCV"),
              ("silhouette", "^", "-", "#8ab17d", 1.6, "silhouette"),
              ("frac_interp", "v", ":", "#9d6bbf", 1.4,
               "fraction with distinctive vocabulary")]
    for col, mk, ls, colour, lw, lab in series:
        # one continuous series: the size-10 point is drawn like every other
        axq.plot([rq.mcs] + list(q.min_cluster_size), [rq[col]] + list(q[col]),
                 marker=mk, ls=ls, color=colour, lw=lw, ms=3.6, label=lab)
    axq.set_ylabel("internal quality")
    axq.set_xlabel("minimum cluster size")
    axq.set_xscale("log")
    axq.set_xticks(ticks)
    axq.set_xticklabels([str(v) for v in ticks], fontsize=8)
    axq.minorticks_off()
    axq.grid(alpha=.25)
    axq.legend(loc="center left", bbox_to_anchor=(1.16, 0.5), fontsize=6.2,
               frameon=False, labelspacing=0.42, handlelength=1.8,
               handletextpad=0.5, borderpad=0.0)

    fig.tight_layout()

    outdir = rel_root / "results/f1_highlights/figures"
    out = outdir / "fig_granularity_selection.png"
    fig.savefig(out.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"    {out.name} ({len(sw)} sweep points + 1 reference, "
          f"selected={SELECTED} -> {int(sel.n_clusters)} categories)")
    return out


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent)
