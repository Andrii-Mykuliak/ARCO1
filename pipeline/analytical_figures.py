"""Analytical figures drawn from a run's own artefacts.

Every function here takes data frames or arrays produced by the current run and
writes one figure under a caller-supplied name. Nothing reads a frozen result
path and nothing carries publication numbering: the caller decides both the
content name and the destination.

The renderers in ``figures_ab``/``figures_results``/``figures_supplement`` render
the archived study artefacts and resolve their own frozen inputs; these are the
general equivalents used when a notebook run must plot what it just computed.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

DPI = 200
GREY = "0.25"


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [text]


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def granularity_selection(sweep: pd.DataFrame, out: Path, name: str,
                          selected: int | None = None,
                          title: str = "Granularity selection") -> Path:
    """Cluster count, unclustered fraction and internal quality over the sweep."""
    d = sweep.sort_values(sweep.columns[0])
    x = d.iloc[:, 0]
    ncol = next(c for c in ("n_clusters", "clusters") if c in d)
    noise = next((c for c in ("noise_fraction", "noise_rate", "noise_pct")
                  if c in d), None)
    quality = [(c, col) for c, col in (("DBCV", "#1b7837"), ("dbcv", "#1b7837"),
                                       ("silhouette", "#7570b3")) if c in d]
    fig, axes = plt.subplots(2 if quality else 1, 1, figsize=(6.4, 4.4 if quality
                                                              else 2.9),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]} if quality
                             else None)
    a1, a2 = (axes[0], axes[1]) if quality else (np.atleast_1d(axes)[0], None)
    a1.plot(x, d[ncol], marker="o", ms=3.5, color="#1f77b4", label="clusters")
    a1.set_ylabel("clusters recovered", fontsize=8)
    a1.tick_params(labelsize=7)
    if noise is not None:
        v = d[noise] / 100 if noise == "noise_pct" else d[noise]
        a1b = a1.twinx()
        a1b.plot(x, v, marker="s", ms=3, ls="--", color="#d95f02",
                 label="unclustered fraction")
        a1b.set_ylabel("unclustered fraction", fontsize=8)
        a1b.tick_params(labelsize=7)
    if quality:
        for col, colour in quality:
            a2.plot(x, d[col], marker="o", ms=3, color=colour,
                    label=col.upper() if col.lower() == "dbcv" else col)
        a2.set_ylabel("internal quality", fontsize=8)
        a2.tick_params(labelsize=7)
        a2.legend(fontsize=7, frameon=False)
    (a2 or a1).set_xlabel("minimum cluster size", fontsize=8)
    if selected is not None:
        for ax in (a1, a2) if quality else (a1,):
            ax.axvline(selected, color="0.55", lw=1.0, ls=":")
        a1.annotate(f"selected: {selected}", xy=(selected, 0.03), xytext=(3, 0),
                    textcoords="offset points",
                    xycoords=("data", "axes fraction"), fontsize=7, ha="left",
                    va="bottom", color="0.35")
    a1.set_title(title, fontsize=9)
    return _save(fig, out, name)


def thematic_structure(xy: np.ndarray, labels: np.ndarray, out: Path, name: str,
                       category_label: dict | None = None,
                       title: str = "Recovered thematic structure") -> Path:
    """Descriptive two-dimensional view of the recovered categories."""
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], s=3, c="0.85", linewidths=0)
    ids = sorted({int(v) for v in labels if v >= 0})
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(ids):
        m = labels == cid
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color=cmap(i % 20), linewidths=0)
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.annotate(str(cid), (cx, cy), fontsize=6, ha="center", va="center",
                    color="0.15")
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("\n".join(_wrap(
        f"{title} - {len(ids)} categories, projection is descriptive only", 70)),
                 fontsize=9)
    return _save(fig, out, name)


def diagnostic_profiles(profiles: pd.DataFrame, dims, out: Path, name: str,
                        label_col: str = "label", id_col: str = "cluster_id",
                        group_col: str = "domain",
                        title: str = "Per-category diagnostic profile") -> Path:
    """Category by dimension heatmap on one shared scale, never summed to a score."""
    dims = [d for d in dims if d in profiles.columns]
    d = profiles.copy()
    if group_col in d.columns and d[group_col].astype(str).str.len().any():
        d = d.sort_values([group_col, id_col])
    else:
        d = d.sort_values(id_col)
    labels = (d[label_col] if label_col in d.columns else d[id_col]).astype(str)
    M = np.ma.masked_invalid(d[dims].to_numpy(float))
    fig, ax = plt.subplots(figsize=(1.05 * len(dims) + 3.2, 0.19 * len(d) + 1.5))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([s[:34] for s in labels], fontsize=6)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([x.replace("_", "\n") for x in dims], fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9, pad=26)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=6)
    return _save(fig, out, name)


def diagnostic_distributions(profiles: pd.DataFrame, dims, out: Path, name: str,
                             title: str = "Diagnostic dimensions") -> Path:
    """One distribution per complementary diagnostic dimension, never combined."""
    dims = [d for d in dims if d in profiles.columns]
    fig, axes = plt.subplots(1, len(dims), figsize=(2.1 * len(dims), 3.0),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, dim in zip(axes, dims):
        v = pd.to_numeric(profiles[dim], errors="coerce").dropna().to_numpy()
        ax.boxplot(v, widths=0.5, showfliers=False,
                   medianprops=dict(color=GREY))
        ax.scatter(np.random.default_rng(0).normal(1, 0.045, len(v)), v,
                   s=9, color="#1f77b4", alpha=0.65, linewidths=0)
        ax.set_title(dim.replace("_", " "), fontsize=7.5)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=9)
    return _save(fig, out, name)


def similarity_matrix(matrix: pd.DataFrame, out: Path, name: str,
                      title: str = "Cross-register centroid similarity") -> Path:
    """Primary-category by secondary-cluster centroid-cosine structure."""
    M = matrix.to_numpy(float)
    fig, ax = plt.subplots(figsize=(0.16 * M.shape[1] + 2.4,
                                    0.16 * M.shape[0] + 1.8))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M),
                   vmax=np.nanmax(M))
    ax.set_xlabel("secondary-register cluster", fontsize=8)
    ax.set_ylabel("primary category", fontsize=8)
    ax.set_xticks(range(0, M.shape[1], max(1, M.shape[1] // 12)))
    ax.set_yticks(range(0, M.shape[0], max(1, M.shape[0] // 12)))
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="centroid cosine")
    ax.set_title(f"{title} ({M.shape[0]} x {M.shape[1]})", fontsize=9)
    return _save(fig, out, name)


def availability_vs_correspondence(corr: pd.DataFrame, out: Path, name: str,
                                   title: str = "Availability and correspondence"
                                   ) -> Path:
    """Matched-event availability against best-match similarity, by outcome."""
    d = corr.copy()
    xcol = next((c for c in ("matched_event_sentences", "matched10_sentences",
                             "matched_sentences") if c in d), None)
    if xcol is None:
        raise KeyError("no matched-event availability column in the "
                       "correspondence table")
    ycol = "centroid_cosine"
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    classes = [c for c in d["relation"].dropna().unique()] if "relation" in d else []
    palette = {"SINGLE_MATCH": "#1b7837", "SPLIT": "#7570b3",
               "MERGE": "#d95f02", "NON_RECOVERY": "#b2182b"}
    for c in classes:
        m = d["relation"] == c
        ax.scatter(d.loc[m, xcol], d.loc[m, ycol], s=26, alpha=0.85,
                   color=palette.get(c, GREY), label=c.replace("_", "-").lower())
    if "null_p95" in d:
        ax.scatter(d[xcol], d["null_p95"], s=10, marker="_", color="0.6",
                   label="category null threshold")
    ax.set_xscale("symlog")
    ax.set_xlabel("sentences in matched events", fontsize=8)
    ax.set_ylabel("best-match centroid cosine", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("\n".join(_wrap(title, 62)), fontsize=9)
    return _save(fig, out, name)


def event_shares(shares: pd.DataFrame, out: Path, name: str,
                 event_col: str = "event_id", value_col: str = "share",
                 register_col: str = "register",
                 title: str = "Share by matched event") -> Path:
    """One row per matched event, both registers shown."""
    d = shares.copy()
    events = list(dict.fromkeys(d[event_col]))
    fig, ax = plt.subplots(figsize=(6.2, 0.30 * len(events) + 1.6))
    regs = list(dict.fromkeys(d[register_col]))
    colours = {regs[0]: "#1f77b4", regs[-1]: "#d95f02"}
    for i, ev in enumerate(events):
        sub = d[d[event_col] == ev]
        ax.plot(sub[value_col], [i] * len(sub), color="0.75", lw=1.0, zorder=1)
        for _, r in sub.iterrows():
            ax.scatter(r[value_col], i, s=30, zorder=2,
                       color=colours.get(r[register_col], GREY),
                       label=r[register_col] if i == 0 else None)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share of assigned sentences", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
    ax.set_title("\n".join(_wrap(title, 62)), fontsize=9)
    return _save(fig, out, name)


def domain_differences(effects: pd.DataFrame, out: Path, name: str,
                       domain_col: str = "domain", value_col: str = "delta",
                       title: str = "Domain-level compositional difference") -> Path:
    """Event-level differences per domain with the domain median marked."""
    d = effects.dropna(subset=[value_col])
    domains = list(dict.fromkeys(d[domain_col]))
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(domains) + 1.5))
    ax.axvline(0, color="0.6", lw=0.9)
    for i, dom in enumerate(domains):
        v = d.loc[d[domain_col] == dom, value_col].to_numpy(float)
        ax.scatter(v, np.full(len(v), i), s=22, color="#1f77b4", alpha=0.75,
                   linewidths=0)
        ax.plot([np.median(v)] * 2, [i - 0.22, i + 0.22], color="#b2182b", lw=2)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference (primary minus secondary register)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("\n".join(_wrap(f"{title}; bar marks the domain median", 62)),
                 fontsize=9)
    return _save(fig, out, name)


def category_effects(effects: pd.DataFrame, out: Path, name: str,
                     label_col: str = "category_name", value_col: str = "delta",
                     title: str = "Per-category compositional effect") -> Path:
    """Category-level effects across the matched-event comparison."""
    d = effects.dropna(subset=[value_col]).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(d) + 1.5))
    ax.axvline(0, color="0.6", lw=0.9)
    ax.scatter(d[value_col], range(len(d)), s=22, color="#1f77b4", linewidths=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d[label_col].astype(str), fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference (primary minus secondary register)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("\n".join(_wrap(title, 62)), fontsize=9)
    return _save(fig, out, name)


def matched_length_nulls(nulls: pd.DataFrame, out: Path, name: str,
                         event_col: str = "event_id",
                         obs_col: str = "observed_effect",
                         lo_col: str = "null_p2.5", hi_col: str = "null_p97.5",
                         title: str = "Observed difference against its own "
                                      "within-event null") -> Path:
    """Per-event observed value against the event's own null envelope."""
    d = nulls.dropna(subset=[obs_col]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.30 * len(d) + 1.6))
    ax.axvline(0, color="0.6", lw=0.9)
    for i, r in d.iterrows():
        if lo_col in d and hi_col in d:
            ax.plot([r[lo_col], r[hi_col]], [i, i], color="0.75", lw=3,
                    solid_capstyle="butt")
        outside = (lo_col in d and hi_col in d
                   and not (r[lo_col] <= r[obs_col] <= r[hi_col]))
        ax.scatter(r[obs_col], i, s=28, zorder=3,
                   color="#b2182b" if outside else "#1f77b4", linewidths=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d[event_col].astype(str), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("observed difference", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("\n".join(_wrap(f"{title}; bars are the null envelope", 62)),
                 fontsize=9)
    return _save(fig, out, name)


def distribution_panels(panels: dict, out: Path, name: str,
                        title: str = "Diagnostic dimensions") -> Path:
    """One panel per experiment, each showing that experiment's own distribution.

    ``panels`` maps panel title -> the values it distributes over. The panels are
    deliberately not forced onto a shared scale: they measure different
    quantities on different units and are reported separately, never combined.
    """
    keys = [k for k, v in panels.items() if v is not None and len(v)]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 3.2))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, keys):
        v = np.asarray(panels[k], dtype=float)
        v = v[np.isfinite(v)]
        ax.boxplot(v, widths=0.5, showfliers=False, medianprops=dict(color=GREY))
        ax.scatter(rng.normal(1, 0.045, len(v)), v, s=9, color="#1f77b4",
                   alpha=0.65, linewidths=0)
        ax.set_title("\n".join(_wrap(k, 22)), fontsize=7.2)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
        ax.set_xlabel(f"n = {len(v)}", fontsize=6.5)
    fig.suptitle(title, fontsize=9)
    return _save(fig, out, name)


DIAGNOSTIC_DIMENSIONS = ("alternative_model_recovery", "resampling_stability",
                         "heldout_replication", "geometry_separability")


def _diagnostic_panels(profiles, ext):
    """One panel per diagnostic dimension, each from its own experiment.

    This is not the per-category profile redrawn four times: resampling and
    alternative geometry use per-category summaries over their replicate
    experiments, and the held-out factorial uses viable run-level counts.
    """
    panels = {"alternative model recovery":
              profiles.alternative_model_recovery.dropna().to_numpy()}
    if "resampling" in ext:
        panels["resampling stability (200 replicates)"] = (
            ext["resampling"]["category_summary"].mean_jaccard.to_numpy())
    if "factorial" in ext:
        runs = ext["factorial"]["run_records"]
        panels["held-out replication (200 runs)"] = (
            runs[~runs.is_degenerate.astype(bool)].n_pass.to_numpy(float))
    if "geometry" in ext:
        panels["alternative geometry (100 replicates)"] = (
            ext["geometry"]["category_summary"].mean_jaccard.to_numpy())
    return panels


def write_all(release_root, cfg, tables, sweep, projection, labels,
              category_domains=None, ext=None, cfg2=None, second=None,
              corr=None, paired=None, cross_register_figures=None) -> dict:
    """Draw every figure this run supports; return name to path.

    Figures read the analytical tables and the objects the run computed. No
    figure writes or modifies an analytical table.
    """
    from pathlib import Path

    release_root = Path(release_root)
    ext = ext or {}
    category_domains = dict(category_domains or {})
    fd = Path(cfg.figures_dir)
    xf = Path(cross_register_figures) if cross_register_figures else None
    figures = {}

    def draw(key, fn, *a, **k):
        try:
            figures[key] = str(Path(fn(*a, **k)).relative_to(release_root))
        except Exception as exc:
            print(f"  {key}: not drawn - {type(exc).__name__}: {exc}")

    draw("primary_granularity_selection", granularity_selection,
         sweep, fd, "primary_granularity_selection",
         selected=cfg.min_cluster_size,
         title=f"{cfg.corpus_name}: granularity selection")
    draw("primary_thematic_structure", thematic_structure,
         projection[:, :2], labels, fd, "primary_thematic_structure",
         title=f"{cfg.corpus_name}: recovered thematic structure")

    profiles = pd.read_csv(release_root / tables["category_diagnostic_profiles"])
    draw("diagnostic_dimension_distributions", distribution_panels,
         _diagnostic_panels(profiles, ext), fd,
         "diagnostic_dimension_distributions",
         title="Complementary diagnostic dimensions, each from its own experiment")
    draw("category_diagnostic_profiles", diagnostic_profiles,
         profiles, list(DIAGNOSTIC_DIMENSIONS), fd, "category_diagnostic_profiles")

    if second is not None:
        draw("secondary_granularity_selection", granularity_selection,
             second["sweep"], xf, "secondary_granularity_selection",
             selected=second["selected_min_cluster_size"],
             title=f"{cfg2.corpus_name}: granularity selection")
    if corr is not None:
        draw("cross_register_similarity_structure", similarity_matrix,
             corr["similarity_matrix"], xf, "cross_register_similarity_structure")
        draw("category_availability_vs_correspondence",
             availability_vs_correspondence, corr["correspondence"], xf,
             "category_availability_vs_correspondence")

    if paired is not None:
        tau, domain = paired["tau_primary"], paired["primary_domain"]

        prev = paired["prevalence"]
        prev = prev[prev.threshold == tau].assign(
            domain=lambda f: f.category_id.map(category_domains))
        shares = (prev[prev.domain == domain]
                  .groupby(["event_id", "register"], as_index=False)
                  .agg(n=("n_category", "sum"), assigned=("n_assigned", "max")))
        shares["share"] = shares.n / shares.assigned
        draw("matched_event_domain_share", event_shares, shares, xf,
             "matched_event_domain_share",
             title=f"{domain}: share of assigned sentences, by matched event")

        comp = paired["compositional"]
        dcols = [c for c in comp.columns if c.startswith("delta_clr_")]
        long = comp.melt(id_vars="event_id", value_vars=dcols,
                         var_name="domain", value_name="delta")
        long["domain"] = long.domain.str.replace("delta_clr_", "", regex=False)
        draw("domain_composition_differences", domain_differences, long, xf,
             "domain_composition_differences",
             title="Domain-level compositional difference, one point per event")

        effects = paired["category_effects"]
        effects = effects[(effects.threshold == tau)
                          & (effects.metric == "assigned")]
        draw("per_category_composition_effects", category_effects, effects, xf,
             "per_category_composition_effects", label_col="category_name",
             value_col="median_delta")

        nulls = paired["matched_length_null_domain"]
        nulls = nulls[(nulls.domain == domain) & (nulls.metric == "assigned")]
        if len(nulls):
            draw("matched_length_null_envelopes", matched_length_nulls, nulls, xf,
                 "matched_length_null_envelopes",
                 title=f"{domain}: observed difference against its own "
                       f"within-event null")
    return figures
