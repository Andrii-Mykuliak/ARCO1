"""Supplementary Figures S1-S4 in final publication form.

Rebuilt from frozen artefacts so that no notebook title, implementation subtitle,
configuration keyword or internal category identifier reaches the page. Captions
carry the methodological explanation instead.

Sources
-------
S1  results/f1_full/tables/fullrace_granularity_sweep.csv
S2  results/f1_cross_register/tables/matched_event_availability.csv
S3  results/f1_cross_register/tables/paired_category_prevalence.csv
S4  results/f1_cross_register/tables/matched_length_null_summary.csv

No inference is recomputed; each generator reads stored values and draws them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figstyle import C, DOMAIN_COLOUR, DOMAIN_ORDER

SELECTED_FULLRACE = 40
PRIMARY_THRESHOLD = 0.40
STRATEGY = "Strategy & technical"

AVAIL_LABEL = {"WELL_REPRESENTED": "well represented", "SPARSE": "sparse",
               "EFFECTIVELY_ABSENT": "effectively absent"}
AVAIL_COLOUR = {"WELL_REPRESENTED": "#2a6f97", "SPARSE": "#e9c46a",
                "EFFECTIVELY_ABSENT": "#c1666b"}


def _event(s: str) -> str:
    """'2024_LasVegas_GP' -> '2024 Las Vegas'."""
    s = (s.replace("_GP", "").replace("LasVegas", "Las Vegas")
          .replace("SaoPaulo", "S\u00e3o Paulo").replace("_", " "))
    return s.strip()


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # long reader-facing category names overflow the default subplot margin
    fig.savefig(out / f"{name}.pdf", dpi=600, facecolor="white",
                bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}.pdf + {name}.png")
    return out / f"{name}.pdf"


# ------------------------------------------------------------------- S1 -----
def s1_fullrace_granularity(rel: Path, out: Path,
                            name: str = "figS1_fullrace_granularity") -> Path:
    d = pd.read_csv(rel / "results/f1_full/tables/fullrace_granularity_sweep.csv")
    assert {"mcs", "n_clusters", "noise_fraction", "DBCV", "silhouette"} <= set(d.columns)
    d = d.sort_values("mcs")
    sel = d[d.mcs == SELECTED_FULLRACE].iloc[0]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.505, 5.0), sharex=True,
                                 facecolor="white",
                                 gridspec_kw={"height_ratios": [1.15, 1.0],
                                              "hspace": 0.12})
    for ax in (a1, a2):
        ax.axvline(SELECTED_FULLRACE, color="#7fc97f", lw=6, alpha=.45, zorder=0)
        ax.grid(color="#dddddd", lw=.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    a1.plot(d.mcs, d.n_clusters, "o-", color=C["primary"], lw=1.8, ms=4,
            label="clusters recovered (left)")
    a1.set_yscale("log")
    a1.set_ylabel("clusters recovered", fontsize=8.5, color=C["primary"])
    a1.tick_params(axis="y", labelcolor=C["primary"], labelsize=7.5)
    b = a1.twinx()
    b.plot(d.mcs, d.noise_fraction, "s--", color=C["secondary"], lw=1.5, ms=3.4,
           label="unclustered fraction (right)")
    b.set_ylabel("fraction left unclustered", fontsize=8.5, color=C["secondary"])
    b.tick_params(axis="y", labelcolor=C["secondary"], labelsize=7.5)
    b.spines["top"].set_visible(False)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a1.legend(h1 + h2 + [Line2D([], [], color="#7fc97f", lw=6, alpha=.45,
                                label=f"selected setting ({SELECTED_FULLRACE})")],
              l1 + l2 + [f"selected setting ({SELECTED_FULLRACE})"],
              loc="upper right", fontsize=6.8, frameon=True, framealpha=.92,
              edgecolor="#999999")

    a2.plot(d.mcs, d.DBCV, "o-", color="#264653", lw=1.6, ms=3.6, label="DBCV")
    a2.plot(d.mcs, d.silhouette, "^-", color="#8ab17d", lw=1.6, ms=3.6,
            label="silhouette")
    if "frac_interpretable" in d.columns:
        a2.plot(d.mcs, d.frac_interpretable, "v:", color="#9d6bbf", lw=1.4, ms=3.6,
                label="fraction with distinctive vocabulary")
    a2.set_ylabel("internal quality", fontsize=8.5)
    a2.set_xlabel("minimum cluster size", fontsize=8.5)
    a2.tick_params(labelsize=7.5)
    a2.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=.92,
              edgecolor="#999999")
    a2.set_xscale("log")
    a2.set_xticks(d.mcs.tolist())
    a2.set_xticklabels([str(int(v)) for v in d.mcs], fontsize=7.0)
    a2.minorticks_off()
    return _save(fig, out, name)


# ------------------------------------------------------------------- S2 -----
def s2_availability(rel: Path, out: Path,
                    name: str = "figS2_matched_event_availability") -> Path:
    d = pd.read_csv(rel / "results/f1_cross_register/tables/matched_event_availability.csv")
    assert len(d) == 34
    d = d.sort_values(["matched10_sentences", "highlight_cluster"],
                      ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.505, 6.2), facecolor="white")
    ax.barh(y, d.matched10_sentences, height=0.74, zorder=2,
            color=[AVAIL_COLOUR[a] for a in d.availability_class],
            edgecolor="white", linewidth=0.4)
    ax.axvline(15, color="#555555", ls="--", lw=0.9, zorder=3)
    ax.axvline(5, color="#555555", ls=":", lw=0.9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.highlight_name}  ({int(r.matched10_events)}/10 events)"
                        for r in d.itertuples()], fontsize=6.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("sentences contributed to the ten matched events", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", color="#dddddd", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AVAIL_COLOUR[k],
                                     label=AVAIL_LABEL[k])
                       for k in ["WELL_REPRESENTED", "SPARSE", "EFFECTIVELY_ABSENT"]]
              + [Line2D([], [], color="#555555", ls="--", lw=0.9,
                        label="well-represented floor (15)"),
                 Line2D([], [], color="#555555", ls=":", lw=0.9,
                        label="sparse floor (5)")],
              loc="lower right", fontsize=6.8, frameon=True, framealpha=.92,
              edgecolor="#999999")
    return _save(fig, out, name)


# ------------------------------------------------------------------- S3 -----
def s3_category_effects(rel: Path, out: Path,
                        name: str = "figS3_paired_category_effects") -> Path:
    xr = rel / "results/f1_cross_register/tables"
    pv = pd.read_csv(xr / "paired_category_prevalence.csv")
    prof = pd.read_csv(rel /
                       "results/f1_highlights/tables/category_diagnostic_profiles.csv")
    dom = dict(zip(prof.cluster_id, prof.domain))

    pv = pv[np.isclose(pv.threshold, PRIMARY_THRESHOLD)]
    w = pv.pivot_table(index=["category_id", "category_name", "event_id"],
                       columns="register", values="p_assigned").reset_index()
    w["delta"] = w["HIGHLIGHT"] - w["FULL_RACE"]
    med = (w.groupby(["category_id", "category_name"])["delta"]
             .median().reset_index().sort_values("delta"))
    assert len(med) == 34, f"expected 34 categories, got {len(med)}"

    fig, ax = plt.subplots(figsize=(6.505, 6.6), facecolor="white")
    ax.axvline(0.0, color="#888888", lw=0.9, zorder=1)
    for i, r in enumerate(med.itertuples()):
        col = DOMAIN_COLOUR.get(dom.get(r.category_id), "#777777")
        v = w[w.category_id == r.category_id]["delta"].to_numpy()
        ax.scatter(v, np.full(len(v), i), s=9, alpha=.32, color=col,
                   linewidths=0, zorder=2)
        ax.scatter([r.delta], [i], s=52, marker="|", color=col, linewidths=1.9,
                   zorder=4)
    ax.set_yticks(range(len(med)))
    ax.set_yticklabels(med.category_name, fontsize=6.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.8, len(med) - 0.2)
    ax.set_xlabel("difference in share of assigned sentences "
                  "(highlight minus full-race)", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", color="#dddddd", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=DOMAIN_COLOUR[d],
                              mec=DOMAIN_COLOUR[d], ms=5, label=d)
                       for d in DOMAIN_ORDER],
              loc="lower right", ncol=1, fontsize=6.8, frameon=True,
              framealpha=.92, edgecolor="#999999")
    return _save(fig, out, name)


# ------------------------------------------------------------------- S4 -----
def s4_matched_length_null(rel: Path, out: Path,
                           name: str = "figS4_matched_length_null") -> Path:
    d = pd.read_csv(rel / "results/f1_cross_register/tables/matched_length_null_summary.csv")
    d = d[(d.metric == "assigned") & (d.domain == STRATEGY)].copy()
    assert len(d) == 10, f"expected 10 events, got {len(d)}"
    d["ev"] = d.event_id.map(_event)
    d = d.sort_values("observed_effect").reset_index(drop=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(6.505, 3.5), facecolor="white")
    ax.axvline(0.0, color="#888888", lw=0.9, zorder=1)
    for i, r in d.iterrows():
        ax.plot([r["null_p2.5"], r["null_p97.5"]], [y[i], y[i]],
                color="#b9c6d0", lw=7, solid_capstyle="butt", zorder=2)
        ax.plot([r.null_mean, r.null_mean], [y[i] - .28, y[i] + .28],
                color="#6b7b88", lw=1.1, zorder=3)
    inside = ~d.outside_central_95.astype(bool)
    ax.scatter(d.observed_effect[~inside], y[~inside.to_numpy()], s=46, zorder=4,
               color=C["reference"], edgecolor=C["reference"], linewidths=1.2)
    ax.scatter(d.observed_effect[inside], y[inside.to_numpy()], s=46, zorder=4,
               facecolor="white", edgecolor=C["reference"], linewidths=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.ev, fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Strategy-and-technical share: highlight minus full-race",
                  fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", color="#dddddd", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color="#b9c6d0", lw=7, label="null interval, central 95%"),
        Line2D([], [], color="#6b7b88", lw=1.1, label="null mean"),
        Line2D([], [], marker="o", ls="", mfc=C["reference"], mec=C["reference"],
               ms=6, label="observed, outside the interval"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=C["reference"], ms=6,
               label="observed, inside the interval")],
        loc="lower right", fontsize=6.8, frameon=True, framealpha=.92,
        edgecolor="#999999")
    return _save(fig, out, name)


# ------------------------------------------------------------------- S5 -----
def s5_similarity_matrix(rel: Path, out: Path,
                         name: str = "figS5_cross_register_similarity") -> Path:
    """Highlight category x full-race cluster centroid-cosine matrix.

    Source: results/f1_cross_register/tables/cross_register_similarity_matrix.csv
    (frozen). Rebuilt so that the axes carry reader-facing category names and a
    figure-local full-race numbering instead of raw cluster identifiers.
    """
    xr = rel / "results/f1_cross_register/tables"
    m = pd.read_csv(xr / "cross_register_similarity_matrix.csv", index_col=0)
    prof = pd.read_csv(rel /
                       "results/f1_highlights/tables/category_diagnostic_profiles.csv")
    assert m.shape[0] == 34, f"expected 34 highlight rows, got {m.shape[0]}"

    # Seriation: order rows by the full-race cluster each one matches best, then
    # order the columns so those best matches run down the diagonal. Remaining
    # columns keep their canonical order and follow. This is a reading aid for a
    # bipartite matrix - it changes placement only, never a similarity value.
    rows_all = [f"H{c}" for c in prof.cluster_id]
    A = m.loc[rows_all].to_numpy(float)
    best = A.argmax(axis=1)
    row_order = sorted(range(len(prof)), key=lambda i: (best[i], -A[i, best[i]]))
    prof = prof.iloc[row_order].reset_index(drop=True)

    seen, col_order = set(), []
    for i in row_order:
        j = int(best[i])
        if j not in seen:
            seen.add(j)
            col_order.append(j)
    col_order += [j for j in range(A.shape[1]) if j not in seen]

    rows = [f"H{c}" for c in prof.cluster_id]
    M = m.loc[rows].to_numpy(float)[:, col_order]

    fig, ax = plt.subplots(figsize=(8.6, 6.6), facecolor="white")
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof.label, fontsize=6.4)
    ax.tick_params(axis="y", length=0, pad=2)
    step = 2
    ax.set_xticks(range(0, M.shape[1], step))
    ax.set_xticklabels([str(i + 1) for i in range(0, M.shape[1], step)],
                       fontsize=6.2)
    ax.set_xlabel("full-race theme (numbered locally to this figure)", fontsize=8.0)
    for i, r in enumerate(prof.itertuples()):
        ax.add_patch(plt.Rectangle((-2.4, i - .5), 1.1, 1, clip_on=False,
                                   color=DOMAIN_COLOUR.get(r.domain, "#777")))
    ax.set_xlim(-2.6, M.shape[1] - .5)
    cb = fig.colorbar(im, ax=ax, shrink=.72, pad=0.015)
    cb.set_label("centroid cosine similarity", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.0)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=DOMAIN_COLOUR[d],
                              mec=DOMAIN_COLOUR[d], ms=6, label=d)
                       for d in DOMAIN_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=4,
              frameon=False, fontsize=7.2)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, out, name)


def build_all(rel: Path) -> dict:
    rel = Path(rel)
    print("  supplementary figures:")
    return {
        "S1": str(s1_fullrace_granularity(rel, rel / "results/f1_full/figures")),
        "S2": str(s2_availability(rel, rel / "results/f1_cross_register/figures")),
        "S3": str(s3_category_effects(rel, rel / "results/f1_cross_register/figures")),
        "S4": str(s4_matched_length_null(rel, rel / "results/f1_cross_register/figures")),
        "S5": str(s5_similarity_matrix(rel, rel / "results/f1_cross_register/figures")),
    }


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parent.parent)
