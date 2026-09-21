"""Four-axis convergence validation.

Each axis perturbs a different source of contingency and asks whether a cluster
survives:

  Axis 1  Algorithm      encoder / clustering objective / density family
  Axis 2  Data           which sentences were drawn (bootstrap subsamples)
  Axis 3  Held-out       which sessions were drawn (train/test split)
  Axis 4  Hierarchical   the assumed cluster geometry (Ward at fixed K)

Every axis reports a per-cluster score in [0, 1]; a cluster "meets" an axis when
its score clears the configured criterion. Counting met axes gives the verdict
band. Thresholds live in config, never inline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .clustering import cluster_hdbscan, cluster_ids, reduce_umap
from .io_utils import best_jaccard, cached, jaccard, l2_normalise, log


# --------------------------------------------------------------- axis 1 ----
def paper_judges(cfg, with_encoders: bool = True) -> list[str]:
    """The seven pre-specified Axis-1 judges, and only those.

    Axis-1 scores are a fraction of this roster, so anything added here changes
    every published count. Granularity and seed perturbations are auxiliary
    sensitivity runs and are deliberately excluded - see ``auxiliary_judges``.
    """
    names = []
    if with_encoders:
        names += [f"enc_{e.split('/')[-1].split('-')[0].lower()}"
                  for e in cfg.axis1_encoders]
    names += ["kmeans", "ward", "optics", "leiden"]
    return names


def auxiliary_judges(cfg) -> list[str]:
    """Granularity and seed perturbations. Reported, never scored in Axis 1."""
    return [f"res_mcs{m}" for m in (max(10, cfg.min_cluster_size // 2),
                                    int(cfg.min_cluster_size * 1.5))] + ["seed_umap2"]


def expected_judges(cfg, with_encoders: bool = True) -> list[str]:
    """Back-compatible alias for the scored roster."""
    return paper_judges(cfg, with_encoders)


def missing_judges(cfg, alts: dict, with_encoders: bool = True) -> list[str]:
    """Judges the roster expects that this run did not produce."""
    return [j for j in expected_judges(cfg, with_encoders) if j not in alts]


def _alt_partitions(cfg, X, U, labels, texts=None):
    """Alternative pipelines, each varying one component from the primary.

    Three families, matching the study design:
      * **encoder** — re-embed with an independent sentence encoder
      * **clustering objective** — KMeans, Ward at the same K
      * **density family** — OPTICS, Leiden

    Granularity and reduction-seed variants are added as extra judges; they probe
    the same algorithm at a different resolution rather than a different
    algorithm, so they are labelled distinctly.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans

    k = len(cluster_ids(labels))
    alts = {}

    # --- encoder variation (the expensive family; needs the raw texts) ------
    if texts is not None:
        from .embedding import embed_with
        for enc in cfg.axis1_encoders:
            short = enc.split("/")[-1].split("-")[0].lower()

            def _mk(enc=enc, short=short):
                Xe = embed_with(enc, list(texts), cfg.batch_size)
                Ue = reduce_umap(cfg, Xe, tag=f"umap_enc_{short}", seed=cfg.seed)
                return cluster_hdbscan(Ue, cfg.min_cluster_size, cfg).astype(np.int32)

            try:
                alts[f"enc_{short}"] = cached(cfg, f"alt_enc_{short}.npy", _mk)
            except Exception as e:
                log(f"  encoder judge {short} unavailable: {e}")

    # --- clustering-objective variation on the same reduced space ----------
    alts["kmeans"] = KMeans(n_clusters=k, n_init=10,
                            random_state=cfg.seed).fit_predict(U)
    alts["ward"] = AgglomerativeClustering(n_clusters=k,
                                           linkage="ward").fit_predict(U)

    # --- density-family variation ------------------------------------------
    try:
        from sklearn.cluster import OPTICS
        alts["optics"] = OPTICS(min_samples=max(5, cfg.min_cluster_size // 3)).fit_predict(U)
    except Exception as e:  # pragma: no cover
        log(f"  optics unavailable: {e}")
    try:
        alts["leiden"] = _leiden_partition(U, cfg)
    except Exception as e:
        log(f"  leiden unavailable: {e}")

    # --- resolution / seed variation (same algorithm, different setting) ---
    for m in (max(10, cfg.min_cluster_size // 2), int(cfg.min_cluster_size * 1.5)):
        alts[f"res_mcs{m}"] = cluster_hdbscan(U, m, cfg)
    U2 = reduce_umap(cfg, X, tag="umap_alt", seed=cfg.seed + 1)
    alts["seed_umap2"] = cluster_hdbscan(U2, cfg.min_cluster_size, cfg)
    return alts


def _leiden_partition(U: np.ndarray, cfg) -> np.ndarray:
    """Leiden community detection over a kNN graph on the reduced space."""
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(U, n_neighbors=cfg.umap_n_neighbors, mode="connectivity")
    src, dst = A.nonzero()
    edges = [(int(a), int(b)) for a, b in zip(src, dst) if a < b]
    g = ig.Graph(n=U.shape[0], edges=edges)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=cfg.seed)
    return np.asarray(part.membership, dtype=np.int32)


def alternative_model_recovery(cfg, X, U, labels, texts=None) -> pd.DataFrame:
    """Cross-pipeline survival: recovered in a majority of alternatives?"""

    def _compute():
        all_alts = _alt_partitions(cfg, X, U, labels, texts)
        roster = paper_judges(cfg, with_encoders=texts is not None)
        # Axis 1 is scored on the seven pre-specified judges only; granularity
        # and seed perturbations stay available but never enter the denominator.
        alts = {k: v for k, v in all_alts.items() if k in roster}
        aux = sorted(k for k in all_alts if k not in roster)
        absent = [j for j in roster if j not in alts]
        expected = len(roster)

        if absent:
            log("!" * 68)
            log(f"!! AXIS 1 INCOMPLETE - {len(absent)} of {expected} judges did not run: "
                f"{', '.join(absent)}")
            log("!! alternative-model recovery scores are a fraction of the judges that ran, so these "
                "counts are NOT comparable to a full-roster run.")
            if any(j in ("leiden",) for j in absent):
                log("!! Leiden needs: pip install leidenalg python-igraph")
            log("!" * 68)

        idx = np.arange(len(labels))
        rows = []
        for c in cluster_ids(labels):
            members = set(idx[labels == c].tolist())
            hits = 0
            for name, alt in alts.items():
                if best_jaccard(members, alt, idx) >= cfg.recovery_jaccard:
                    hits += 1
            rows.append({"cluster": c, "axis1_recovered": hits,
                         "axis1_total": len(alts),
                         "axis1_judges_expected": expected,
                         "axis1_judges_missing": ";".join(absent) if absent else "",
                         "axis1_complete": not absent,
                         "axis1_score": hits / len(alts)})
        log(f"alternative-model recovery: {len(alts)} of {expected} paper judges ran"
            + ("" if not absent else "  <-- INCOMPLETE")
            + (f"; {len(aux)} auxiliary run(s) excluded from scoring: "
               f"{', '.join(aux)}" if aux else ""))
        return pd.DataFrame(rows)

    return cached(cfg, "axis1.csv", _compute)


def alternative_model_recovery_detail(cfg, X, U, labels, texts=None) -> pd.DataFrame:
    """Per-category best-overlap Jaccard against every pre-specified alternative.

    Same partitions and same overlap measure as ``alternative_model_recovery``;
    this keeps the per-alternative values that the scored table collapses to a
    count, so the full configuration-by-category matrix can be released.
    """

    def _compute():
        all_alts = _alt_partitions(cfg, X, U, labels, texts)
        roster = paper_judges(cfg, with_encoders=texts is not None)
        alts = {k: all_alts[k] for k in roster if k in all_alts}
        idx = np.arange(len(labels))
        rows = []
        for c in cluster_ids(labels):
            members = set(idx[labels == c].tolist())
            scores = {name: float(best_jaccard(members, alt, idx))
                      for name, alt in alts.items()}
            rows.append({"cluster": c, **scores,
                         "n_recovered": int(sum(v >= cfg.recovery_jaccard
                                                for v in scores.values())),
                         "n_alternatives": len(alts)})
        log(f"alternative-model detail: {len(alts)} alternatives x "
            f"{len(rows)} categories")
        return pd.DataFrame(rows)

    return cached(cfg, "axis1_per_alternative.csv", _compute)


# --------------------------------------------------------------- axis 2 ----
def resampling_stability(cfg, X, labels) -> pd.DataFrame:
    """Data robustness: re-cluster random subsamples, measure recovery."""

    def _compute():
        rng = np.random.default_rng(cfg.seed)
        n = len(labels)
        per_cluster = {c: [] for c in cluster_ids(labels)}
        for b in range(cfg.bootstrap_n):
            take = rng.choice(n, size=int(cfg.bootstrap_frac * n), replace=False)
            take.sort()
            Ub = reduce_umap(cfg, X[take], tag=f"umap_boot{b}", seed=cfg.seed + 100 + b)
            lab_b = cluster_hdbscan(Ub, cfg.min_cluster_size, cfg)
            for c in per_cluster:
                members = set(take[labels[take] == c].tolist())
                if not members:
                    continue
                per_cluster[c].append(best_jaccard(members, lab_b, take))
            log(f"  bootstrap {b + 1}/{cfg.bootstrap_n}")
        rows = [{"cluster": c,
                 "axis2_score": float(np.mean(v)) if v else 0.0,
                 "axis2_std": float(np.std(v)) if v else 0.0}
                for c, v in per_cluster.items()]
        return pd.DataFrame(rows)

    return cached(cfg, "axis2.csv", _compute)


# ----------------------------------------------- held-out centroid diagnostic --
def held_out_centroid_assignment_diagnostic(cfg, X, labels,
                                            sessions: pd.Series) -> pd.DataFrame:
    """Session-level split scored by nearest-centroid assignment.

    NOT Axis 3. Axis 3 refits the clustering on the training races and projects
    the held-out races onto it; see ``axis3.axis3_heldout_replication``. This
    remains only as a cheap geometric diagnostic.
    """

    def _compute():
        rng = np.random.default_rng(cfg.seed)
        uniq = np.array(sorted(sessions.unique()))
        rng.shuffle(uniq)
        n_train = max(1, int(cfg.heldout_train_frac * len(uniq)))
        train_s = set(uniq[:n_train].tolist())
        is_train = sessions.isin(train_s).to_numpy()
        log(f"held-out replication: {len(train_s)} train / {len(uniq) - len(train_s)} test sessions")

        rows = []
        # centroids from the training half only
        ids = cluster_ids(labels)
        train_cent, kept = [], []
        for c in ids:
            m = (labels == c) & is_train
            if m.sum() >= 3:
                v = X[m].mean(axis=0)
                train_cent.append(v)
                kept.append(c)
        if not kept:
            return pd.DataFrame([{"cluster": c, "axis3_score": 0.0} for c in ids])
        C = l2_normalise(np.stack(train_cent))

        test_idx = np.nonzero(~is_train)[0]
        assign = np.asarray(kept)[(X[test_idx] @ C.T).argmax(axis=1)] if len(test_idx) else np.array([])

        for c in ids:
            if c not in kept or not len(test_idx):
                rows.append({"cluster": c, "axis3_strict": 0.0,
                             "axis3_direction": 0.0, "axis3_score": 0.0})
                continue
            true_mask = labels[test_idx] == c
            if true_mask.sum() == 0:
                rows.append({"cluster": c, "axis3_strict": 0.0,
                             "axis3_direction": 0.0, "axis3_score": 0.0})
                continue
            # strict: share of this cluster's held-out sentences routed back to it
            strict = float((assign[true_mask] == c).mean())
            # direction: is the routing concentrated on one destination at all?
            dest = pd.Series(assign[true_mask]).value_counts(normalize=True)
            direction = float(dest.iloc[0]) if len(dest) else 0.0
            # the axis is met only when both hold, so a cluster that scatters
            # evenly cannot pass on strict recovery alone
            rows.append({"cluster": c, "axis3_strict": round(strict, 4),
                         "axis3_direction": round(direction, 4),
                         "axis3_score": min(strict, direction)})
        return pd.DataFrame(rows)

    return cached(cfg, "axis3.csv", _compute)


# --------------------------------------------------------------- axis 4 ----
def geometry_separability(cfg, X, labels) -> pd.DataFrame:
    """Ward at fixed K over stratified subsamples drawn without replacement.

    Ward linkage is applied to Euclidean distances over L2-normalised
    vectors; for unit vectors cosine distance is half the squared Euclidean
    distance, so the pairwise ordering is preserved.
    """

    def _compute():
        rng = np.random.default_rng(cfg.seed)
        ids = cluster_ids(labels)
        k = len(ids)
        per_cluster = {c: [] for c in ids}
        for b in range(cfg.axis4_n_replicates):
            picks = []
            for c in ids:
                pool = np.nonzero(labels == c)[0]
                take = min(cfg.axis4_per_cluster, len(pool))
                picks.extend(rng.choice(pool, take, replace=False).tolist())
            sub = np.array(sorted(picks))
            Z = linkage(pdist(X[sub], metric="euclidean"), method="ward")
            ward = fcluster(Z, t=k, criterion="maxclust")
            for c in ids:
                members = set(sub[labels[sub] == c].tolist())
                if members:
                    per_cluster[c].append(best_jaccard(members, ward, sub))
            log(f"  geometry-separability replicate {b + 1}/{cfg.axis4_n_replicates} "
                f"({len(sub)} sentences)")
        rows = [{"cluster": c,
                 "axis4_score": float(np.mean(v)) if v else 0.0,
                 "axis4_std": float(np.std(v)) if v else 0.0}
                for c, v in per_cluster.items()]
        return pd.DataFrame(rows)

    return cached(cfg, "axis4.csv", _compute)


# ------------------------------------------------------------ aggregation --
BANDS = ["fragile", "weak", "moderate", "robust", "strict-core"]


def combine_diagnostic_dimensions(cfg, a1, a2, a3, a4=None) -> pd.DataFrame:
    """Merge the per-axis frames into one verdict table.

    ``a3=None`` means Axis 3 was not run. A skipped axis is recorded as
    ``axis3_status="disabled"`` and excluded from the denominator - it is never
    folded in as a failure. The canonical five-band names describe the full
    four-axis protocol only, so a reduced run reports the enabled-axis count
    instead of a band.
    """
    # tolerate the legacy 4-positional call combine_axes(cfg, a1, a2, a3, a4)
    if a4 is None and a3 is not None and "axis4_score" in getattr(a3, "columns", []):
        a3, a4 = None, a3

    df = a1.merge(a2, on="cluster")
    if a3 is not None:
        df = df.merge(a3, on="cluster")
    df = df.merge(a4, on="cluster")

    t = cfg.axis_pass_threshold
    df["axis1_pass"] = df["axis1_score"] >= cfg.axis1_majority
    df["axis2_pass"] = df["axis2_score"] >= t
    df["axis4_pass"] = df["axis4_score"] >= t
    passcols = ["axis1_pass", "axis2_pass", "axis4_pass"]

    if a3 is not None:
        df["axis3_pass"] = df["axis3_score"] >= t
        df["axis3_status"] = "run"
        passcols.insert(2, "axis3_pass")
    else:
        df["axis3_status"] = "disabled"

    df["n_axes_enabled"] = len(passcols)
    df["n_axes_passed"] = df[passcols].sum(axis=1).astype(int)
    if len(passcols) == 4:
        df["band"] = df["n_axes_passed"].map(lambda n: BANDS[n])
    else:
        # Reduced protocol: report the profile, not a canonical verdict band.
        df["band"] = df["n_axes_passed"].map(
            lambda n: f"{n}/{len(passcols)} enabled axes")
    return df.sort_values(["n_axes_passed", "cluster"],
                          ascending=[False, True]).reset_index(drop=True)


def verdict_summary(v: pd.DataFrame) -> pd.DataFrame:
    n = len(v)
    n_ax = int(v["n_axes_enabled"].iloc[0]) if "n_axes_enabled" in v.columns and n else 4
    labels = list(reversed(BANDS)) if n_ax == 4 else         [f"{k}/{n_ax} enabled axes" for k in range(n_ax, -1, -1)]
    rows = []
    for b in labels:
        c = int((v["band"] == b).sum())
        rows.append({"band": b, "clusters": c, "pct": 100 * c / n if n else 0})
    cum = []
    for k in range(n_ax, 0, -1):
        c = int((v["n_axes_passed"] >= k).sum())
        cum.append({"criterion": f">= {k} axes", "clusters": c, "pct": 100 * c / n if n else 0})
    return pd.DataFrame(rows), pd.DataFrame(cum)


def dimension_counts(v: pd.DataFrame) -> pd.DataFrame:
    """Per-axis pass counts. A disabled axis reports "disabled", never 0."""
    rows = []
    for name, key in [("Algorithm", "axis1"), ("Bootstrap", "axis2"),
                      ("Held-out", "axis3"), ("Hierarchical", "axis4")]:
        col = f"{key}_pass"
        if col not in v.columns:
            rows.append({"axis": name, "clusters_meeting": "disabled", "of": len(v)})
        else:
            rows.append({"axis": name, "clusters_meeting": int(v[col].sum()),
                         "of": len(v)})
    return pd.DataFrame(rows)


def dimension_correlations(v: pd.DataFrame) -> pd.DataFrame:
    # Only enabled axes take part; a disabled axis contributes no synthetic value.
    cols = [c for c in ["axis1_score", "axis2_score", "axis3_score", "axis4_score"]
            if c in v.columns]
    names = {"axis1_score": "Algorithm", "axis2_score": "Bootstrap",
             "axis3_score": "Held-out", "axis4_score": "Hierarchical"}
    corr = v[cols].corr(method="pearson")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows.append({"pair": f"{names[a]} <-> {names[b]}",
                         "pearson_r": round(float(corr.loc[a, b]), 3)})
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


# ------------------------------------------------- intrinsic quality (T3) --
def intrinsic_indices(cfg, U: np.ndarray, labels: np.ndarray,
                      alts: dict | None = None) -> pd.DataFrame:
    """Density-aware and convex intrinsic indices for the primary partition and
    each alternative pipeline.

    DBCV suits density-based partitions; silhouette favours convex ones, so it is
    reported as a comparator rather than a verdict. Both are computed only over
    the clustered fraction, which is itself reported: a partition that discards
    most of the corpus can post a flattering score on what is left.
    """
    from sklearn.metrics import silhouette_score

    def _row(name, lab):
        clustered = lab != -1
        n_cl = int(len(set(lab[clustered].tolist())))
        out = {"pipeline": name, "n_clusters": n_cl,
               "noise_pct": round(100 * float((~clustered).mean()), 1),
               "scored_on_pct": round(100 * float(clustered.mean()), 1),
               "silhouette": np.nan, "dbcv": np.nan}
        if n_cl >= 2 and clustered.sum() > n_cl:
            try:
                out["silhouette"] = round(float(
                    silhouette_score(U[clustered], lab[clustered])), 4)
            except Exception:
                pass
            try:
                import hdbscan
                out["dbcv"] = round(float(hdbscan.validity.validity_index(
                    U[clustered].astype(np.float64), lab[clustered])), 4)
            except Exception:
                pass
        return out

    def _compute():
        rows = [_row("primary (UMAP+HDBSCAN)", labels)]
        for name, lab in (alts or {}).items():
            rows.append(_row(name, np.asarray(lab)))
        return pd.DataFrame(rows)

    return cached(cfg, "intrinsic_indices.csv", _compute)


def alternative_partitions(cfg, X, U, labels, texts=None) -> dict:
    """Expose the Axis-1 alternatives so other analyses can reuse them."""
    return _alt_partitions(cfg, X, U, labels, texts)


# ------------------------------------------- leave-one-out sensitivity (S8) --
def loo_sensitivity(cfg, X, U, labels, alts: dict | None = None,
                    texts=None) -> pd.DataFrame:
    """How far does the Axis-1 count move if any single judge is dropped?

    Guards against a headline survival number that rests on one alternative.
    """

    def _compute():
        a = alts if alts is not None else _alt_partitions(cfg, X, U, labels, texts)
        idx = np.arange(len(labels))
        ids = cluster_ids(labels)

        def count(subset):
            n = 0
            for c in ids:
                members = set(idx[labels == c].tolist())
                hits = sum(1 for nm in subset
                           if best_jaccard(members, a[nm], idx) >= cfg.recovery_jaccard)
                if hits / max(len(subset), 1) >= cfg.axis1_majority:
                    n += 1
            return n

        full = count(list(a))
        rows = [{"removed": "(none - all judges)", "majority_count": full, "delta": 0}]
        for nm in a:
            c = count([k for k in a if k != nm])
            rows.append({"removed": nm, "majority_count": c, "delta": c - full})
        df = pd.DataFrame(rows)
        df["abs_delta"] = df["delta"].abs()
        log(f"LOO: baseline {full}/{len(ids)}, max shift "
            f"{int(df['abs_delta'].max())}")
        return df

    return cached(cfg, "loo_sensitivity.csv", _compute)


# ------------------------------------------------ noise-bucket diagnostics --
def noise_profile(cfg, X, labels, texts, C, ids) -> pd.DataFrame:
    """What is in the -1 bucket? Reported so the discarded fraction is visible."""
    noise = labels == -1
    if not noise.any():
        return pd.DataFrame([{"noise_sentences": 0}])
    sims = (X[noise] @ C.T).max(axis=1)
    t = np.asarray(texts, dtype=object)[noise]
    order = np.argsort(sims)
    return pd.DataFrame([{
        "noise_sentences": int(noise.sum()),
        "noise_pct": round(100 * float(noise.mean()), 1),
        "median_nearest_cosine": round(float(np.median(sims)), 3),
        "pct_above_coverage_tau": round(100 * float((sims >= cfg.coverage_tau).mean()), 1),
        "mean_token_count": round(float(np.mean([len(str(s).split()) for s in t])), 1),
        "least_similar_example": str(t[order[0]])[:120],
        "most_similar_example": str(t[order[-1]])[:120],
    }])


# ------------------------------------------------------------- deprecated --
# Old internal names, kept importable for one release so existing scripts and
# the smoke suite do not break. DEPRECATED: no user-facing or publication code
# path uses them, and they will be removed in the next release.
#
#   axis1_algorithm    -> alternative_model_recovery
#   axis2_bootstrap    -> resampling_stability
#   axis4_hierarchical -> geometry_separability
#   combine_axes       -> combine_diagnostic_dimensions
#   axis_counts        -> dimension_counts
#   axis_correlations  -> dimension_correlations

DEPRECATED_FUNCTION_ALIASES = {
    "axis1_algorithm": "alternative_model_recovery",
    "axis2_bootstrap": "resampling_stability",
    "axis4_hierarchical": "geometry_separability",
    "combine_axes": "combine_diagnostic_dimensions",
    "axis_counts": "dimension_counts",
    "axis_correlations": "dimension_correlations",
}

axis1_algorithm = alternative_model_recovery
axis2_bootstrap = resampling_stability
axis4_hierarchical = geometry_separability
combine_axes = combine_diagnostic_dimensions
axis_counts = dimension_counts
axis_correlations = dimension_correlations


def characterise_categories(cfg, X, U, labels, masked, cards,
                            category_domains=None) -> dict:
    """Run the four complementary diagnostic dimensions and assemble the profile.

    Results are reported per dimension and per category. No aggregate vote,
    verdict band or "n of 4" count is produced: support is heterogeneous across
    categories and dimensions, and collapsing it into one number misrepresents it.
    """
    from pathlib import Path

    import pandas as pd

    from . import clustering, heldout_replication

    texts = masked["masked"]
    d1 = alternative_model_recovery(cfg, X, U, labels, texts)
    d2 = resampling_stability(cfg, X, labels)
    d3 = heldout_replication.axis3_heldout_replication(
        cfg, masked, X, labels, text_col="masked")
    d4 = geometry_separability(cfg, X, labels)
    # Same partitions and overlap measure as d1, keeping the per-alternative
    # values that the scored table collapses to a count.
    d1_detail = alternative_model_recovery_detail(cfg, X, U, labels, texts)

    profile = (d1[["cluster", "axis1_score"]]
               .merge(d2[["cluster", "axis2_score"]], on="cluster", how="outer")
               .merge(d3[["cluster", "axis3_score"]], on="cluster", how="outer")
               .merge(d4[["cluster", "axis4_score"]], on="cluster", how="outer")
               .rename(columns={"cluster": "cluster_id",
                                "axis1_score": "alternative_model_recovery",
                                "axis2_score": "resampling_stability",
                                "axis3_score": "heldout_replication",
                                "axis4_score": "geometry_separability"}))
    label_col = next((c for c in ("label", "handle", "external_label")
                      if c in cards.columns), None)
    keep = (["cluster"] + ([label_col] if label_col else [])
            + (["size"] if "size" in cards.columns else []))
    profile = profile.merge(
        cards[keep].rename(columns={"cluster": "cluster_id", label_col: "label"}),
        on="cluster_id", how="left")
    if "label" not in profile.columns:
        profile["label"] = profile.cluster_id.astype(str)
    if "size" not in profile.columns:
        profile["size"] = pd.NA

    # The domain grouping is a configured analysis input, not something
    # inferred: another corpus can recover category ids that overlap a stored
    # map numerically, and attaching that map would silently mislabel it. It is
    # used only when it was configured AND covers every category this run found.
    category_domains = dict(category_domains or {})
    applies = bool(category_domains) and set(
        clustering.cluster_ids(labels)) <= set(category_domains)
    profile["domain"] = (profile.cluster_id.map(category_domains) if applies
                         else "")

    # The profile is returned, not written: the analytical table stage is its
    # single producer, and writes it once from the same sources as the reported
    # per-category table.
    path = Path(cfg.tables_dir) / "category_diagnostic_profiles.csv"
    return {"alternative_model_recovery": d1,
            "alternative_model_recovery_detail": d1_detail,
            "resampling_stability": d2, "heldout_replication": d3,
            "geometry_separability": d4, "profile": profile,
            "profile_path": path, "domain_map_applies": applies}
