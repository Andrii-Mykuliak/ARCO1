"""Figures and display helpers for every artefact the report carries."""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
# Only force a headless backend when there is no interactive/inline one, so
# figures still render inside the notebook while remaining scriptable.
if matplotlib.get_backend().lower() == "agg":
    pass
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from .io_utils import save_figure  # noqa: E402

PALETTE = ["#3b6ea5", "#c0603f", "#5c9a6a", "#8a6bab", "#c9a227", "#6b6b6b"]
plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_size_distribution(cfg, cards: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    d = cards.sort_values("size", ascending=False)
    ax.bar(range(len(d)), d["size"], color=PALETTE[0])
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["handle"], rotation=90, fontsize=6)
    ax.set_xlabel("cluster (sorted by size)")
    ax.set_ylabel("sentences")
    top5 = d.head(5)["share_of_clustered"].sum()
    ax.set_title(f"Cluster size distribution — top 5 hold {top5:.1%} of clustered sentences")
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_cluster_sizes"), fig


def fig_mcs_sweep(cfg, sweep: pd.DataFrame, plateau: dict):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(sweep["min_cluster_size"], sweep["n_clusters"], "o-", color=PALETTE[0],
            label="clusters")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("number of clusters", color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(sweep["min_cluster_size"], sweep["noise_pct"], "s--", color=PALETTE[1],
             label="noise %")
    ax2.set_ylabel("noise (%)", color=PALETTE[1])
    ax2.spines["right"].set_visible(True)
    if plateau.get("plateau_lo") is not None:
        ax.axvspan(plateau["plateau_lo"], plateau["plateau_hi"], color=PALETTE[2], alpha=0.15)
    ax.axvline(cfg.min_cluster_size, color=PALETTE[4], lw=2, alpha=0.9)
    ax.set_title("Granularity sweep (shaded = stability plateau, line = chosen value)")
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_granularity_sweep"), fig


def _with_identity(df: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    """Attach handle/keywords only if the caller has not already merged them."""
    need = [c for c in ("handle", "keywords") if c not in df.columns]
    if not need:
        return df.copy()
    return df.merge(cards[["cluster", *need]], on="cluster", how="left")


def fig_axis_heatmap(cfg, verdict: pd.DataFrame, cards: pd.DataFrame):
    d = _with_identity(verdict, cards)
    d = d.sort_values(["n_axes_passed", "axis1_score"], ascending=False)
    # A disabled axis is dropped from the figure rather than drawn as zero.
    labels = {"axis1_score": "Algorithm", "axis2_score": "Bootstrap",
              "axis3_score": "Held-out", "axis4_score": "Hierarchical"}
    cols = [c for c in labels if c in d.columns]
    n_ax = int(d["n_axes_enabled"].iloc[0]) if "n_axes_enabled" in d.columns else len(cols)
    M = d[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, max(3.2, 0.26 * len(d))))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([labels[c] for c in cols], rotation=20)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{h} ({n}/{n_ax})" for h, n in zip(d["handle"], d["n_axes_passed"])],
                       fontsize=7)
    for i in range(len(d)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="axis score (0-1)")
    ax.set_title("Per-cluster axis recovery")
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_axis_heatmap"), fig


def fig_dendrogram(cfg, Z, cards: pd.DataFrame, ids):
    order = {c: i for i, c in enumerate(ids)}
    labels = [""] * len(ids)
    for _, r in cards.iterrows():
        if r["cluster"] in order:
            kw = r["keywords"][:34]
            labels[order[r["cluster"]]] = f"{r['handle']} {kw}"
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.22 * len(ids))))
    dendrogram(Z, labels=labels, orientation="right", ax=ax,
               color_threshold=Z[-(cfg.n_domains - 1), 2] if len(Z) >= cfg.n_domains else None)
    ax.set_xlabel("Ward linkage distance (Euclidean on L2-normalised centroids)")
    ax.set_title(f"Centroid dendrogram — cut at K={cfg.n_domains}")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_dendrogram"), fig


def fig_coverage_curve(cfg, cov: pd.DataFrame):
    taus = np.round(np.arange(0.0, 1.001, 0.02), 3)
    per = []
    for s, sub in cov.groupby("session"):
        per.append([100 * float((sub["cosine"] >= t).mean()) for t in taus])
    per = np.array(per)
    pooled = np.array([100 * float((cov["cosine"] >= t).mean()) for t in taus])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if len(per):
        ax.fill_between(taus, per.min(axis=0), per.max(axis=0), color=PALETTE[0], alpha=0.18,
                        label="per-session min-max")
    ax.plot(taus, pooled, color=PALETTE[0], lw=2, label="pooled")
    ax.axvline(cfg.coverage_tau, ls="--", color=PALETTE[1])
    at = 100 * float((cov["cosine"] >= cfg.coverage_tau).mean())
    ax.annotate(f"τ={cfg.coverage_tau}\n{at:.1f}%",
                xy=(cfg.coverage_tau, at), xytext=(cfg.coverage_tau + 0.12, at + 12),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax.set_xlabel("nearest-centroid cosine threshold (τ)")
    ax.set_ylabel("sentences covered (%)")
    ax.set_title("Held-out coverage")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_coverage"), fig


def fig_null(cfg, uni: pd.DataFrame, n_sessions: int, null: dict, observed: int):
    """How many sessions each leftover cluster spans, against the null expectation.

    A leftover cluster touching every session looks like background chatter; one
    confined to a few sessions is race-specific. Plotting the whole spread keeps
    the figure informative even when the universal count has no variance under
    permutation, which is the regime small session counts fall into.
    """
    if uni is None or not len(uni) or "sessions" not in uni.columns:
        return None, None
    spread = uni["sessions"].astype(int)
    edges = np.arange(0.5, n_sessions + 1.5, 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(spread, bins=edges, color=PALETTE[0], alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax.axvline(n_sessions, color=PALETTE[1], ls="--", lw=2.0,
               label=f"present in all {n_sessions} sessions ({observed} clusters)")
    med = float(spread.median())
    ax.axvline(med, color="black", ls=":", lw=1.4, label=f"median spread = {med:.0f}")
    ax.set_xticks(range(1, n_sessions + 1))
    ax.set_xlabel("sessions in which the leftover cluster appears")
    ax.set_ylabel("leftover clusters")
    sub = ""
    if null.get("iterations"):
        c = np.array(null["counts"])
        sub = (f"  |  shuffle null: {c.mean():.1f} ± {c.std():.1f} universal"
               f" over {null['iterations']:,} permutations")
        if c.std() == 0:
            sub += " (no variance at this session count)"
    ax.set_title(f"Leftover-cluster session spread{sub}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return save_figure(cfg, fig, "fig_leftover_spread"), fig


def show(df: pd.DataFrame, n: int = 20, title: str | None = None):
    """Notebook-friendly display that never truncates the interesting columns."""
    from IPython.display import display, Markdown
    if title:
        display(Markdown(f"**{title}**"))
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        display(df.head(n))


def run_summary(configuration, cfg, docs, labels, runtime=None, ext=None,
                cfg2=None, docs2=None, second=None, corr=None, paired=None,
                tables=None, figures=None, release_root=None):
    """Print the values this execution computed.

    Nothing here is compared against a stored result: the numbers are whatever
    the analyses produced.
    """
    from pathlib import Path

    import numpy as np

    from . import clustering

    ext = ext or {}
    print(f"configuration   : {configuration['name']} - "
          f"{configuration['description']}")
    if runtime is not None:
        print(f"runtime         : {runtime:.0f}s")
    print()

    print("Primary corpus")
    print(f"  events                {docs['session'].nunique()}")
    print(f"  sentences             {len(docs):,}")
    print(f"  categories            {clustering.n_clusters(labels)}")
    print(f"  clustered             {int((labels != -1).sum()):,}")
    print(f"  unclustered           {int((labels == -1).sum()):,}")
    print(f"  unclustered fraction  {clustering.noise_rate(labels):.4f}")
    print(f"  minimum cluster size  {cfg.min_cluster_size}")

    if "resampling" in ext:
        s = ext["resampling"]["summary"]
        print("\nResampling stability")
        print(f"  replicates            {s['n_replicates']}")
        print(f"  viable                {s['n_viable']}")
        print(f"  degenerate            {s['n_degenerate']}")
        print(f"  median category mean  {s['median_category_mean_jaccard']}")
        print(f"  categories scored     {s['n_categories']}")

    if "factorial" in ext:
        s = ext["factorial"]["summary"]
        print("\nHeld-out replication factorial")
        print(f"  runs                  {s['n_runs']}")
        print(f"  degenerate            {s['n_degenerate']}")
        for _, r in ext["factorial"]["cell_summary"].iterrows():
            print(f"  {r.cell:<20}  median {r['median']:.0f}, "
                  f"IQR {r.iqr_lo:.0f}-{r.iqr_hi:.0f}, "
                  f"{int(r.n_viable)} viable, {int(r.n_degenerate)} degenerate")

    if "geometry" in ext:
        v = ext["geometry"]["category_summary"].mean_jaccard.to_numpy()
        print("\nAlternative geometry")
        print(f"  replicates            {ext['geometry']['summary']['n_replicates']}")
        print(f"  median                {np.median(v):.4f}")
        print(f"  IQR                   {np.percentile(v, 25):.4f}-"
              f"{np.percentile(v, 75):.4f}")
        print(f"  range                 {v.min():.4f}-{v.max():.4f}")
        print(f"  categories scored     {len(v)}")

    if second is not None:
        print("\nSecond register")
        print(f"  events                {docs2['session'].nunique()}")
        print(f"  sentences             {len(docs2):,}")
        print(f"  clusters              {second['n_clusters']}")
        print(f"  minimum cluster size  {second['selected_min_cluster_size']}")
        print(f"  unclustered fraction  {second['noise_fraction']:.6f}")
        print(f"  DBCV                  {second['dbcv']:.7f}")
        print(f"  silhouette            {second['silhouette']:.7f}")

    if corr is not None:
        c = corr["scalars"]
        rel = corr["correspondence"].relation.value_counts()
        av = corr["correspondence"].availability_class.value_counts()
        print("\nCross-register correspondence")
        print(f"  above own null        {c['exceeds_own_null']}/"
              f"{c['n_primary_categories']}")
        print(f"  above within-register {c['exceeds_within_register_reference']}/"
              f"{c['n_primary_categories']}")
        for k in ("SINGLE_MATCH", "SPLIT", "MERGE", "NON_RECOVERY"):
            print(f"  {k.lower().replace('_', ' '):<20}  {int(rel.get(k, 0))}")
        for k in ("WELL_REPRESENTED", "SPARSE", "EFFECTIVELY_ABSENT"):
            print(f"  {k.lower().replace('_', ' '):<20}  {int(av.get(k, 0))}")

    if paired is not None:
        print(f"\nMatched-event composition ({paired['primary_domain']})")
        for k, h in paired["headline"].items():
            print(f"  {k:<20}  median {h['median']:+.4f}, "
                  f"p {h['sign_test_p']:.4f}, "
                  f"{h['n_negative']}/{paired['n_matched_events']} negative")
        print(f"  outside own null      {paired['events_below_own_95_envelope']}/"
              f"{paired['n_matched_events']}")

    if tables is not None and figures is not None:
        where = Path(cfg.results_dir)
        if release_root:
            try:
                where = where.relative_to(Path(release_root))
            except ValueError:
                pass
        print(f"\n{len(tables)} tables and {len(figures)} figures written "
              f"under {where}")
