"""UMAP reduction, HDBSCAN clustering, and the granularity sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import cached, l2_normalise, log


def reduce_umap(cfg, X: np.ndarray, tag: str = "umap", seed: int | None = None) -> np.ndarray:
    seed = cfg.seed if seed is None else seed

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
        log(f"umap: loaded archived {path.name} {U.shape} (no refit)")
        return U

    U = cached(cfg, f"{tag}__s{seed}.npy", _compute)
    log(f"umap: {U.shape}")
    return U


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
