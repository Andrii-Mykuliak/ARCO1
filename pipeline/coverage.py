"""Held-out coverage, the leftover partition, and the session-shuffle null.

Sentences from a held-out set are assigned to their nearest cluster centroid and
counted as covered above a cosine threshold. The uncovered residual is
re-clustered on its own; a leftover cluster is *universal* when it appears in
every held-out session. The shuffle null asks how many leftover clusters would
look universal if session identity carried no information at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clustering import cluster_hdbscan, cluster_ids, reduce_umap
from .io_utils import cached, l2_normalise, log


def nearest_centroid(Xq: np.ndarray, C: np.ndarray):
    S = Xq @ C.T
    return S.argmax(axis=1), S.max(axis=1)


def coverage(cfg, Xq: np.ndarray, C: np.ndarray, sessions: pd.Series) -> pd.DataFrame:
    """Per-sentence nearest-centroid assignment + similarity."""
    idx, sim = nearest_centroid(Xq, C)
    return pd.DataFrame({"session": sessions.to_numpy(),
                         "nearest": idx, "cosine": sim,
                         "covered": sim >= cfg.coverage_tau})


def coverage_by_session(cov: pd.DataFrame, tau: float) -> pd.DataFrame:
    g = cov.assign(hit=cov["cosine"] >= tau).groupby("session")
    df = g.agg(sentences=("cosine", "size"),
               median_cosine=("cosine", "median"),
               covered=("hit", "sum")).reset_index()
    df["coverage_pct"] = 100 * df["covered"] / df["sentences"]
    return df.sort_values("coverage_pct").reset_index(drop=True)


def tau_sweep(cov: pd.DataFrame, taus) -> pd.DataFrame:
    rows = []
    for t in taus:
        rows.append({"tau": t,
                     "pooled_coverage_pct": 100 * float((cov["cosine"] >= t).mean()),
                     "leftover_sentences": int((cov["cosine"] < t).sum())})
    return pd.DataFrame(rows)


def threshold_floor(cfg, X_train: np.ndarray, labels: np.ndarray, C: np.ndarray) -> dict:
    """Characterise tau against the pipeline's own clustered/noise decision.

    Reports what the threshold *does* (recall over known in-cluster sentences)
    rather than asserting a rationale for the value.
    """
    _, sim_cl = nearest_centroid(X_train[labels != -1], C)
    out = {"tau": cfg.coverage_tau,
           "retains_of_clustered_pct": round(100 * float((sim_cl >= cfg.coverage_tau).mean()), 2),
           "clustered_below_tau": int((sim_cl < cfg.coverage_tau).sum()),
           "clustered_total": int(len(sim_cl)),
           "p1_of_clustered": round(float(np.percentile(sim_cl, 1)), 3)}
    if (labels == -1).any():
        _, sim_no = nearest_centroid(X_train[labels == -1], C)
        out["admits_of_noise_pct"] = round(100 * float((sim_no >= cfg.coverage_tau).mean()), 2)
        out["youden_j"] = round(float((sim_cl >= cfg.coverage_tau).mean()
                                      - (sim_no >= cfg.coverage_tau).mean()), 3)
        grid = np.round(np.arange(0.20, 0.86, 0.01), 2)
        js = [(float((sim_cl >= t).mean() - (sim_no >= t).mean()), float(t)) for t in grid]
        bj, bt = max(js)
        out["separation_optimal_tau"] = bt
        out["separation_optimal_j"] = round(bj, 3)
    return out


def leftover_partition(cfg, X_left: np.ndarray, sessions_left: pd.Series):
    """Re-cluster the residual with the same hyperparameters."""

    def _compute():
        mcs = cfg.leftover_min_cluster_size or max(
            5, min(cfg.min_cluster_size, len(X_left) // 20))
        if len(X_left) < mcs * 2:
            log(f"leftover: only {len(X_left)} residual sentences - too few to cluster")
            return pd.DataFrame({"session": sessions_left.to_numpy(),
                                 "lc": np.full(len(X_left), -1)})
        log(f"leftover: re-clustering {len(X_left)} residual sentences at mcs={mcs}")
        U = reduce_umap(cfg, X_left, tag="umap_leftover", seed=cfg.seed)
        lc = cluster_hdbscan(U, mcs, cfg)
        return pd.DataFrame({"session": sessions_left.to_numpy(), "lc": lc})

    df = cached(cfg, "leftover.csv", _compute)
    log(f"leftover: {len(cluster_ids(df['lc'].to_numpy()))} clusters "
        f"from {len(df):,} residual sentences")
    return df


def universality(left: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Per-leftover-cluster session spread; universal = present in all sessions."""
    n_sessions = left["session"].nunique()
    sub = left[left["lc"] != -1]
    g = sub.groupby("lc").agg(size=("session", "size"),
                              sessions=("session", "nunique")).reset_index()
    g["universal"] = g["sessions"] == n_sessions
    return (g.sort_values("size", ascending=False).reset_index(drop=True),
            int(g["universal"].sum()), int(len(g)))


def shuffle_null(cfg, left: pd.DataFrame) -> dict:
    """Permute session labels across residual sentences, holding the leftover
    partition fixed, and recount universally-present clusters."""

    def _compute():
        sub = left[left["lc"] != -1].reset_index(drop=True)
        if not len(sub):
            return {"iterations": 0}
        rng = np.random.default_rng(cfg.seed)
        lc = sub["lc"].to_numpy()
        sess = sub["session"].to_numpy()
        n_sessions = len(set(sess.tolist()))
        counts = []
        for _ in range(cfg.null_n_permutations):
            perm = rng.permutation(sess)
            d = pd.DataFrame({"lc": lc, "s": perm})
            u = d.groupby("lc")["s"].nunique()
            counts.append(int((u == n_sessions).sum()))
        counts = np.array(counts)
        return {"iterations": int(len(counts)),
                "mean": float(counts.mean()),
                "sd": float(counts.std()),
                "min": int(counts.min()),
                "max": int(counts.max()),
                "counts": counts.tolist()}

    return cached(cfg, "leftover_null.json", _compute)


def null_verdict(observed: int, null: dict) -> dict:
    if not null.get("iterations"):
        return {"observed": observed, "testable": False}
    counts = np.array(null["counts"])
    p_low = float((counts <= observed).mean())
    p_high = float((counts >= observed).mean())
    return {
        "observed_universal": observed,
        "null_mean": round(null["mean"], 2),
        "null_sd": round(null["sd"], 2),
        "null_range": [null["min"], null["max"]],
        "iterations": null["iterations"],
        "p_observed_or_fewer": p_low,
        "p_observed_or_more": p_high,
        "direction": ("more session-concentrated than chance" if p_low <= 0.05
                      else "more session-spread than chance" if p_high <= 0.05
                      else "not distinguishable from chance"),
        "testable": True,
    }


def make_heldout_split(cfg, df: pd.DataFrame, n_sessions: int | None = None):
    """Split sessions into a discovery half and a held-out half.

    Used when no separate second-register corpus is supplied: the coverage
    analysis then measures transfer to *unseen sessions* rather than transfer to
    a different register. The distinction is reported, never blurred.
    """
    rng = np.random.default_rng(cfg.seed)
    uniq = np.array(sorted(df["session"].unique()))
    rng.shuffle(uniq)
    k = n_sessions if n_sessions else max(2, int(round(cfg.holdout_session_frac * len(uniq))))
    held = set(uniq[:k].tolist())
    mask = df["session"].isin(held).to_numpy()
    log(f"holdout split: {len(held)}/{len(uniq)} sessions held out "
        f"({int(mask.sum()):,} sentences)")
    return mask, sorted(held)


def coverage_bootstrap_ci(cfg, cov: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Session-level bootstrap CI for pooled coverage.

    Resamples *sessions* rather than sentences: sentences within a broadcast are
    not independent, so a sentence-level interval would be optimistically tight.
    """
    rng = np.random.default_rng(cfg.seed)
    sess = cov["session"].to_numpy()
    hit = (cov["cosine"] >= cfg.coverage_tau).to_numpy()
    uniq = np.array(sorted(set(sess.tolist())))
    by = {s: hit[sess == s] for s in uniq}

    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        arr = np.concatenate([by[uniq[i]] for i in pick])
        vals.append(100 * float(arr.mean()))
    vals = np.array(vals)
    per = [100 * float(v.mean()) for v in by.values()]
    return pd.DataFrame([{
        "tau": cfg.coverage_tau,
        "pooled_coverage_pct": round(100 * float(hit.mean()), 2),
        "bootstrap_mean_pct": round(float(vals.mean()), 2),
        "ci95_lo": round(float(np.percentile(vals, 2.5)), 2),
        "ci95_hi": round(float(np.percentile(vals, 97.5)), 2),
        "ci_width_pp": round(float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5)), 2),
        "sessions": len(uniq),
        "per_session_min_pct": round(min(per), 2),
        "per_session_max_pct": round(max(per), 2),
        "resamples": n_boot,
        "resampling_unit": "session",
    }])
