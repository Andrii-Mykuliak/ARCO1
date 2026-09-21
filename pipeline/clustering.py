"""UMAP reduction, HDBSCAN clustering, and the granularity sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import cached, l2_normalise, log


def assert_finite(A: np.ndarray, what: str, expected_rows: int | None = None,
                  remedy: str = "") -> np.ndarray:
    """Stop at the array that went wrong, not three stages later."""
    A = np.asarray(A)
    if expected_rows is not None and A.shape[0] != expected_rows:
        raise ValueError(f"{what}: {A.shape[0]} rows, expected {expected_rows}")
    finite = np.isfinite(A)
    if finite.all():
        return A
    bad_rows = int((~finite.all(axis=1)).sum())
    raise ValueError(
        f"{what} is not finite: {int(np.isnan(A).sum()):,} NaN and "
        f"{int(np.isinf(A).sum()):,} Inf over {A.size:,} values; "
        f"{bad_rows:,} of {A.shape[0]:,} rows carry a non-finite value. "
        f"Downstream clustering would drop those rows and report an empty "
        f"array instead of naming this cause. " + remedy)


UMAP_NONFINITE_REMEDY = (
    "A non-finite projection is a failure of the manifold fit in this "
    "environment, not of the corpus: check that the embedding is finite, then "
    "supply the archived projection through the configuration so the archived "
    "realisation is reused rather than refitted.")


def reduce_umap(cfg, X: np.ndarray, tag: str = "umap", seed: int | None = None) -> np.ndarray:
    seed = cfg.seed if seed is None else seed
    assert_finite(X, f"the embedding passed to UMAP ({tag})")

    def _compute():
        import umap
        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components,
            metric=cfg.umap_metric,
            random_state=seed,
        )
        return reducer.fit_transform(X).astype(np.float32)

    # UMAP output is version-sensitive: identical vectors still refit to a
    # different manifold across releases. Reproducing an archived partition
    # therefore needs the archived projection, not only the archived embedding.
    path = getattr(cfg, "umap_path", None)
    if path is not None and tag == "umap" and seed == cfg.seed:
        U = np.load(path)
        if len(U) != len(X):
            raise ValueError(f"archived UMAP has {len(U)} rows, corpus has {len(X)}")
        assert_finite(U, f"the archived UMAP projection {path.name}", len(X),
                      UMAP_NONFINITE_REMEDY)
        log(f"umap: loaded archived {path.name} {U.shape} (no refit)")
        return U

    U = cached(cfg, f"{tag}__s{seed}.npy", _compute)
    assert_finite(U, f"the UMAP projection ({tag}, seed {seed})", len(X),
                  UMAP_NONFINITE_REMEDY)
    log(f"umap: {U.shape}")
    return U


#: Archived two-dimensional descriptive layout, when one exists for the corpus.
DESCRIPTIVE_2D = "umap2d_descriptive_s{seed}.npy"


def descriptive_projection(cfg, X: np.ndarray) -> np.ndarray:
    """Two-dimensional layout for display only; never used for clustering.

    This is a two-component UMAP fit, not the first two axes of the clustering
    projection: UMAP optimises the whole layout for its target dimensionality,
    so a slice of the five-dimensional fit is a different arrangement. Where the
    archived descriptive projection exists it is reused, for the same reason the
    clustering projection is.
    """
    from pathlib import Path

    archived = Path(cfg.cache_dir) / DESCRIPTIVE_2D.format(seed=cfg.seed)
    if archived.exists():
        U2 = np.load(archived)
        if len(U2) != len(X):
            raise ValueError(
                f"archived descriptive projection {archived.name} has "
                f"{len(U2)} rows, corpus has {len(X)}")
        log(f"umap2d: loaded archived {archived.name} {U2.shape} (no refit)")
        return U2

    def _compute():
        import umap
        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=2,
            metric=cfg.umap_metric,
            random_state=cfg.seed,
        )
        return reducer.fit_transform(X).astype(np.float32)

    U2 = cached(cfg, f"umap2d_descriptive__s{cfg.seed}.npy", _compute)
    log(f"umap2d: {U2.shape}")
    return U2


def cluster_ids(labels: np.ndarray) -> list[int]:
    """Cluster labels present, excluding the HDBSCAN noise label."""
    return sorted(int(c) for c in set(labels.tolist()) if c != -1)


def n_clusters(labels: np.ndarray) -> int:
    return len(cluster_ids(labels))


def noise_rate(labels: np.ndarray) -> float:
    return float((labels == -1).mean())


def centroids(X: np.ndarray, labels: np.ndarray):
    """L2-normalised mean vector per cluster; returns (ids, matrix)."""
    ids = cluster_ids(labels)
    if not ids:
        return [], np.zeros((0, X.shape[1]), dtype=np.float32)
    C = np.stack([X[labels == c].mean(axis=0) for c in ids])
    return ids, l2_normalise(C)


def cluster_hdbscan(U: np.ndarray, min_cluster_size: int, cfg) -> np.ndarray:
    import hdbscan
    assert_finite(U, "the projection passed to HDBSCAN", remedy=UMAP_NONFINITE_REMEDY)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        metric=cfg.hdbscan_metric,
        cluster_selection_method=cfg.hdbscan_selection,
    )
    return cl.fit_predict(U)


def partition(cfg, U: np.ndarray) -> np.ndarray:
    """The primary partition at the configured granularity."""

    def _compute():
        return cluster_hdbscan(U, cfg.min_cluster_size, cfg).astype(np.int32)

    L = cached(cfg, f"labels__mcs{cfg.min_cluster_size}.npy", _compute)
    log(f"partition: {n_clusters(L)} clusters, {100 * noise_rate(L):.1f}% noise, "
        f"{int((L != -1).sum()):,} clustered")
    return L


def mcs_sweep(cfg, U: np.ndarray) -> pd.DataFrame:
    """Granularity sweep — the evidence behind the chosen min_cluster_size."""

    def _compute():
        rows = []
        for m in cfg.mcs_sweep:
            lab = cluster_hdbscan(U, m, cfg)
            rows.append({
                "min_cluster_size": int(m),
                "n_clusters": n_clusters(lab),
                "noise_pct": 100 * noise_rate(lab),
                "largest_cluster": int(pd.Series(lab[lab != -1]).value_counts().max())
                if (lab != -1).any() else 0,
            })
            log(f"  mcs={m:>3}: {rows[-1]['n_clusters']:>3} clusters, "
                f"{rows[-1]['noise_pct']:.1f}% noise")
        return pd.DataFrame(rows)

    return cached(cfg, "mcs_sweep.csv", _compute)


def plateau_summary(sweep: pd.DataFrame, chosen: int, tol: int = 1) -> dict:
    """Describe the stability plateau around the chosen granularity.

    Purely data-driven: finds the widest run of sweep points whose cluster count
    stays within ``tol`` of the count at ``chosen``.
    """
    s = sweep.sort_values("min_cluster_size").reset_index(drop=True)
    if chosen not in set(s["min_cluster_size"]):
        return {"chosen": chosen, "in_sweep": False}
    target = int(s.loc[s["min_cluster_size"] == chosen, "n_clusters"].iloc[0])
    ok = (s["n_clusters"] - target).abs() <= tol

    best_lo = best_hi = None
    run_start = None
    best_len = 0
    for i, good in enumerate(ok.tolist() + [False]):
        if good and run_start is None:
            run_start = i
        elif not good and run_start is not None:
            if i - run_start > best_len:
                best_len = i - run_start
                best_lo, best_hi = run_start, i - 1
            run_start = None
    if best_lo is None:
        return {"chosen": chosen, "in_sweep": True, "n_clusters": target, "plateau": None}
    return {
        "chosen": chosen,
        "in_sweep": True,
        "n_clusters": target,
        "plateau_lo": int(s.loc[best_lo, "min_cluster_size"]),
        "plateau_hi": int(s.loc[best_hi, "min_cluster_size"]),
        "plateau_points": int(best_len),
        "cluster_count_range": [int(s.loc[best_lo:best_hi, "n_clusters"].min()),
                                int(s.loc[best_lo:best_hi, "n_clusters"].max())],
        "noise_range": [float(s.loc[best_lo:best_hi, "noise_pct"].min()),
                        float(s.loc[best_lo:best_hi, "noise_pct"].max())],
    }
