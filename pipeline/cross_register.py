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

from .clustering import (centroids, cluster_hdbscan, cluster_ids, n_clusters,
                         noise_rate, reduce_umap)
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


# ===========================================================================
# Cross-register producers
# ===========================================================================
# Ported from the frozen internal implementation:
#   scripts/fgc_sweep.py            granularity sweep, internal indices
#   scripts/fgc_regime_seed.py      core rule, documented fallback, seed roster
#   scripts/fgc_primary_matched.py  primary solution, lexical, availability
#   scripts/fgc_correspondence.py   null, thresholds, relation classification
#
# The scientific rules are unchanged. What changed is the interface: these take
# arrays and frames computed by the caller instead of reading internal output
# paths, so a fresh clone holding two authorised corpora can run them.

from . import frozen_f1 as F  # noqa: E402


def _weighted_jaccard(a: dict, b: dict) -> float:
    ks = set(a) | set(b)
    if not ks:
        return 0.0
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in ks)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in ks)
    return num / den if den else 0.0


def _term_weights(enr: pd.DataFrame, cluster_col: str):
    """term -> in-cluster incidence share, per cluster, over significant terms."""
    if enr is None or not len(enr):
        return {}, {}
    e = enr.copy()
    e["weight"] = e["in_cluster"] / e["cluster_size"]
    w = {int(c): dict(zip(g["term"], g["weight"])) for c, g in e.groupby(cluster_col)}
    top = {int(c): list(g.sort_values("p")["term"].head(F.CORRESPONDENCE_TOP_K))
           for c, g in e.groupby(cluster_col)}
    return w, top


# ------------------------------------------ second-register reconstruction --
def granularity_sweep_internal(cfg, U: np.ndarray, texts, mcs_values=None) -> pd.DataFrame:
    """Granularity sweep carrying the internal indices the selection rule needs.

    Unlike ``clustering.mcs_sweep`` this records DBCV, silhouette, enrichment
    breadth and the pre-specified degeneracy flag, which is what a register's
    granularity is selected from. Selection uses this corpus's own evidence
    only: corpus-size scaling is not a criterion and is not applied.
    """
    import re as _re

    import hdbscan
    from scipy.stats import hypergeom
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from sklearn.metrics import silhouette_score

    mcs_values = list(mcs_values or F.FULLRACE_MCS_SWEEP)
    ph = _re.compile("<(?:" + "|".join(cfg.entity_classes) + ")>")
    src = [ph.sub(" ", str(t)) for t in texts]
    vec = CountVectorizer(lowercase=True, min_df=F.ENRICHMENT_MIN_DF, ngram_range=(1, 1),
                          stop_words=sorted(ENGLISH_STOP_WORDS))
    M = (vec.fit_transform(src) > 0).astype(np.int32)
    N, K = M.shape[0], np.asarray(M.sum(axis=0)).ravel()

    def _screen(lab, ids):
        """Per-cluster count of BH-significant unigrams - the breadth check."""
        out = []
        for c in ids:
            idx = np.nonzero(lab == c)[0]
            k = np.asarray(M[idx].sum(axis=0)).ravel()
            p = hypergeom.sf(k - 1, N, K, len(idx))
            o = np.argsort(p)
            crit = F.ENRICHMENT_Q * np.arange(1, len(p) + 1) / len(p)
            ok = p[o] <= crit
            out.append(int(np.max(np.nonzero(ok)[0])) + 1 if ok.any() else 0)
        return np.array(out) if out else np.array([0])

    rows = []
    for m in mcs_values:
        cl = hdbscan.HDBSCAN(min_cluster_size=int(m), metric=cfg.hdbscan_metric,
                             cluster_selection_method=cfg.hdbscan_selection,
                             gen_min_span_tree=True)
        lab = cl.fit_predict(U)
        ids = cluster_ids(lab)
        assigned = lab != -1
        sizes = np.array([int((lab == c).sum()) for c in ids]) if ids else np.array([0])
        try:
            dbcv = float(cl.relative_validity_)
        except Exception:
            dbcv = float("nan")
        sil = (float(silhouette_score(U[assigned], lab[assigned]))
               if len(ids) > 1 and assigned.sum() > len(ids) else float("nan"))
        enr = _screen(lab, ids)
        noise = float((lab == -1).mean())
        rows.append({
            "mcs": int(m), "n_clusters": len(ids), "noise_fraction": noise,
            "assigned_fraction": float(assigned.mean()),
            "DBCV": dbcv, "silhouette": sil,
            "min_cluster_size_observed": int(sizes.min()),
            "median_cluster_size": float(np.median(sizes)),
            "max_cluster_size": int(sizes.max()),
            "mean_enriched_terms": float(enr.mean()),
            "degenerate_flag": bool(len(ids) < F.DEGENERACY_MIN_CLUSTERS
                                    or noise < F.DEGENERACY_MIN_NOISE)})
        log(f"  mcs={m:>3}: {len(ids):>3} clusters, noise {noise:.4f}, "
            f"DBCV {dbcv:.4f}, silhouette {sil:.4f}"
            + ("  DEGENERATE" if rows[-1]["degenerate_flag"] else ""))
    return pd.DataFrame(rows)


def select_granularity(sweep: pd.DataFrame, core_tol: float = 0.10,
                       min_core_width: int = 3) -> dict:
    """Pre-specified cluster-count core rule, with its documented fallback.

    The rule looks for a contiguous run of at least ``min_core_width``
    non-degenerate settings whose cluster count stays within ``core_tol`` of that
    run's median, and takes the arithmetic centre of the widest such run. The
    full-race corpus has no such region - its cluster count falls monotonically -
    so the documented fallback applies: maximum DBCV among non-degenerate
    settings, labelled as a fallback rather than presented as the primary rule.
    """
    nd = sweep[~sweep["degenerate_flag"]].reset_index(drop=True)
    if not len(nd):
        raise ValueError("every swept setting is degenerate; nothing to select")
    order = nd["mcs"].tolist()
    core = None
    for w in range(len(order), min_core_width - 1, -1):
        for s in range(len(order) - w + 1):
            sub = order[s:s + w]
            n = nd.set_index("mcs").loc[sub, "n_clusters"].to_numpy()
            med = np.median(n)
            if np.all(np.abs(n - med) <= core_tol * med):
                core = sub
                break
        if core:
            break
    if core:
        centre = 0.5 * (core[0] + core[-1])
        chosen = min(core, key=lambda m: (abs(m - centre),
                                          -nd.set_index("mcs").loc[m, "mean_enriched_terms"]))
        rule = "pre-specified: arithmetic centre of the cluster-count core region"
        plateau_exists = True
    else:
        chosen = int(nd.loc[nd["DBCV"].idxmax(), "mcs"])
        rule = ("FALLBACK - the pre-specified cluster-count core rule yielded no "
                "region for this corpus, so the representative setting is the "
                "maximum DBCV among non-degenerate settings. Reported as a "
                "fallback; conclusions are also reported across the regime.")
        plateau_exists = False
    row = nd[nd["mcs"] == chosen].iloc[0]
    log(f"granularity: mcs={chosen} ({'core rule' if core else 'DBCV fallback'})")
    return {"selected_mcs": int(chosen), "rule": rule,
            "cluster_count_plateau_exists": plateau_exists,
            "core_region": core, "n_clusters": int(row["n_clusters"]),
            "noise_fraction": float(row["noise_fraction"]),
            "dbcv": float(row["DBCV"]), "silhouette": float(row["silhouette"])}


def seed_stability(cfg, X: np.ndarray, mcs: int, seeds=None) -> pd.DataFrame:
    """Refit the projection under each seed and re-partition at ``mcs``.

    The pre-specified degeneracy rule is applied to each seed outcome as well as
    to the sweep; the selected setting is required to produce zero degenerate
    seeds.
    """
    import hdbscan
    seeds = list(seeds if seeds is not None else F.FULLRACE_SEEDS)
    rows = []
    for s in seeds:
        try:
            U = reduce_umap(cfg, X, tag=f"umap_seed{s}", seed=int(s))
            bad = not np.isfinite(U).all()
        except Exception as exc:
            log(f"  seed {s}: projection failed ({exc!r})"[:110])
            rows.append({"seed": int(s), "failed": True, "n_clusters": None,
                         "noise_fraction": None, "is_degenerate": None})
            continue
        if bad:
            rows.append({"seed": int(s), "failed": True, "n_clusters": None,
                         "noise_fraction": None, "is_degenerate": None})
            continue
        lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric=cfg.hdbscan_metric,
                              cluster_selection_method=cfg.hdbscan_selection
                              ).fit_predict(U)
        nc, nf = n_clusters(lab), noise_rate(lab)
        rows.append({"seed": int(s), "failed": False, "n_clusters": int(nc),
                     "noise_fraction": float(nf),
                     "is_degenerate": bool(nc < F.DEGENERACY_MIN_CLUSTERS
                                           or nf < F.DEGENERACY_MIN_NOISE)})
    return pd.DataFrame(rows)


def reconstruct_second_register(cfg, X: np.ndarray, texts, sessions,
                                mcs_values=None, seeds=None,
                                run_seed_stability: bool = True) -> dict:
    """Recover the second register's thematic structure from its own evidence.

    Nothing about the primary register enters this function. The granularity is
    selected from this corpus's own sweep and is never scaled from the other
    corpus's size, nor chosen to match its cluster count.

    Returns the labels, centroids, enrichment and per-cluster table that the
    correspondence stage consumes.
    """
    U = reduce_umap(cfg, X, tag="umap", seed=cfg.seed)
    sweep = granularity_sweep_internal(cfg, U, texts, mcs_values)
    selection = select_granularity(sweep)
    mcs = selection["selected_mcs"]

    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=mcs, metric=cfg.hdbscan_metric,
                             cluster_selection_method=cfg.hdbscan_selection
                             ).fit_predict(U).astype(np.int32)
    ids, C = centroids(X, labels)

    from .enrichment import _enrich_uncached
    enr = _enrich_uncached(cfg, list(texts), labels, verbose=False)
    if len(enr):
        enr = enr.rename(columns={"cluster": "fr_cluster"})

    sess = pd.Series(np.asarray(sessions))
    per_cluster = pd.DataFrame([{
        "fr_cluster": c,
        "size": int((labels == c).sum()),
        "n_enriched": int((enr["fr_cluster"] == c).sum()) if len(enr) else 0,
        "sessions_present": int(sess[labels == c].nunique()),
        "mean_cosine_to_centroid": float((X[labels == c] @ C[i]).mean()),
    } for i, c in enumerate(ids)])

    seedtab = (seed_stability(cfg, X, mcs, seeds) if run_seed_stability
               else pd.DataFrame(columns=["seed", "failed", "n_clusters",
                                          "noise_fraction", "is_degenerate"]))
    n_degen = (int(seedtab["is_degenerate"].fillna(False).sum())
               if len(seedtab) else None)

    out = {
        "status": "COMPLETED",
        "selected_min_cluster_size": mcs,
        "selection_rule": selection["rule"],
        "cluster_count_plateau_exists": selection["cluster_count_plateau_exists"],
        "n_clusters": len(ids),
        "noise_fraction": round(float(noise_rate(labels)), 6),
        "dbcv": selection["dbcv"], "silhouette": selection["silhouette"],
        "clusters_with_significant_vocabulary": int(per_cluster["n_enriched"].gt(0).sum()),
        "n_seeds": int(len(seedtab)), "n_degenerate_seeds": n_degen,
        "labels": labels, "ids": ids, "centroids": C,
        "enrichment": enr, "per_cluster": per_cluster,
        "sweep": sweep, "seed_stability": seedtab,
    }
    log(f"second register: {len(ids)} clusters at mcs={mcs}, "
        f"{100 * noise_rate(labels):.2f}% unclustered, DBCV {selection['dbcv']:.4f}")
    return out


# ----------------------------------------------------- availability control --
def matched_event_availability(labels: np.ndarray, sessions, matched_events,
                               category_name=None, category_domain=None) -> pd.DataFrame:
    """How well each primary category is represented in the matched events.

    The taxonomy is discovered over the whole primary corpus; the matched subset
    is far smaller. A category absent from the matched events cannot be expected
    to reappear in the second register regardless of any register effect, so
    non-recovery is only interpretable above this floor. The thresholds are fixed
    before any correspondence is computed.

    The matched subset is itself a product of the primary register's editing, so
    this is a LOWER BOUND on availability and is not independent of the register
    effect it controls for.
    """
    # As above: never defaulted to the frozen Formula 1 map.
    category_name = {} if category_name is None else category_name
    category_domain = {} if category_domain is None else category_domain
    sess = pd.Series(np.asarray(sessions))
    want = set(matched_events.values() if isinstance(matched_events, dict)
               else matched_events)
    in_matched = sess.isin(want).to_numpy()
    rows = []
    for c in cluster_ids(labels):
        m = (labels == c) & in_matched
        n_s, n_ev = int(m.sum()), int(sess[m].nunique())
        total = int((labels == c).sum())
        if n_s >= F.WELL_MIN_SENTENCES and n_ev >= F.WELL_MIN_EVENTS:
            cls = "WELL_REPRESENTED"
        elif n_s >= F.SPARSE_MIN_SENTENCES and n_ev >= F.SPARSE_MIN_EVENTS:
            cls = "SPARSE"
        else:
            cls = "EFFECTIVELY_ABSENT"
        rows.append({"highlight_cluster": c,
                     "highlight_name": category_name.get(c, str(c)),
                     "domain": category_domain.get(c, ""),
                     "canonical_total_sentences": total,
                     "matched10_sentences": n_s, "matched10_events": n_ev,
                     "share_of_canonical_total": n_s / total if total else np.nan,
                     "availability_class": cls})
    av = pd.DataFrame(rows)
    log("availability: " + ", ".join(f"{k} {v}" for k, v in
                                     av["availability_class"].value_counts().items()))
    return av


# ------------------------------------------------------------ correspondence --
def compute_correspondence(CH, h_ids, CF, f_ids, X_second, labels_second,
                           h_enrichment=None, f_enrichment=None,
                           availability=None, second_per_cluster=None,
                           category_name=None, category_domain=None,
                           n_null: int | None = None,
                           null_seed: int | None = None) -> dict:
    """Correspondence between two independently induced structures.

    Similarity is centroid cosine in the embedding space. Weighted Jaccard over
    enriched terms and overlap at the top K terms are reported alongside it as
    lexical views; the measures sit on different scales and their magnitudes are
    never compared directly.

    The null is size-preserving random membership over the second register's
    assigned sentences: cluster sizes are held fixed and membership is permuted,
    so a category's best match is compared against the best match it would obtain
    from a structure of the same granularity carrying no thematic content. Each
    category gets its OWN threshold - the 95th percentile of its own null - and
    the empirical p-value uses (r + 1) / (B + 1).

    A second, stricter reference scale is the 95th percentile of the off-diagonal
    within-primary-register centroid cosine: the similarity typical of two
    distinct categories of the same register.

    Relations are classified per category from the above-threshold set:
        SINGLE_MATCH  exactly one second-register cluster above threshold, and
                      that cluster is above threshold for this category only
        SPLIT         two or more second-register clusters above threshold
        MERGE         one second-register cluster above threshold, but that
                      cluster is also above threshold for another category
        NON_RECOVERY  nothing above threshold
    """
    # Category labels and the manual grouping are ANALYSIS INPUTS tied to a
    # particular partition. They are never defaulted to the frozen Formula 1 map:
    # a different corpus can easily recover ids that overlap it numerically, and
    # attaching those labels would silently mislabel another corpus.
    category_name = {} if category_name is None else category_name
    category_domain = {} if category_domain is None else category_domain
    n_null = int(n_null or F.CORRESPONDENCE_N_NULL)
    null_seed = int(null_seed if null_seed is not None else F.CORRESPONDENCE_NULL_SEED)
    pctl = F.CORRESPONDENCE_NULL_PERCENTILE
    CH, CF = np.asarray(CH), np.asarray(CF)
    h_ids, f_ids = list(h_ids), list(f_ids)
    labels_second = np.asarray(labels_second)
    X_second = np.asarray(X_second)

    Scos = CH @ CF.T
    sim = pd.DataFrame(Scos, index=[f"H{c}" for c in h_ids],
                       columns=[f"F{c}" for c in f_ids])

    hw, htop = _term_weights(h_enrichment, "cluster")
    fw, ftop = _term_weights(f_enrichment, "fr_cluster")
    Slex = (np.array([[_weighted_jaccard(hw.get(h, {}), fw.get(f, {})) for f in f_ids]
                      for h in h_ids]) if hw and fw else np.full(Scos.shape, np.nan))
    K = F.CORRESPONDENCE_TOP_K
    Stop = (np.array([[len(set(htop.get(h, [])) & set(ftop.get(f, []))) / K
                       for f in f_ids] for h in h_ids])
            if htop and ftop else np.full(Scos.shape, np.nan))

    # ---- size-preserving null over the second register's assigned sentences --
    assigned = np.nonzero(labels_second != -1)[0]
    sizes = [int((labels_second == c).sum()) for c in f_ids]
    bounds = np.cumsum([0] + sizes)
    rng = np.random.default_rng(null_seed)
    Xa = X_second[assigned]
    bn = np.empty((n_null, len(h_ids)))
    for b in range(n_null):
        Xp = Xa[rng.permutation(len(assigned))]
        cent = l2_normalise(np.stack([Xp[bounds[k]:bounds[k + 1]].mean(axis=0)
                                      for k in range(len(f_ids))]))
        bn[b] = (CH @ cent.T).max(axis=1)
    thr = np.percentile(bn, pctl, axis=0)
    obs = Scos.max(axis=1)
    p_emp = [(1 + int((bn[:, i] >= obs[i]).sum())) / (n_null + 1)
             for i in range(len(h_ids))]

    HH = CH @ CH.T
    ref = float(np.percentile(HH[~np.eye(len(h_ids), dtype=bool)],
                              F.WITHIN_REGISTER_REFERENCE_PERCENTILE))

    null_tab = pd.DataFrame({
        "highlight_cluster": h_ids,
        "highlight_name": [category_name.get(c, str(c)) for c in h_ids],
        "observed_best_cosine": obs, "null_median": np.median(bn, axis=0),
        "null_p95": thr, "null_p_empirical": p_emp,
        "exceeds_null_p95": obs > thr,
        "within_highlight_offdiag_p95": ref,
        "exceeds_within_highlight_p95": obs > ref})

    # ---- relation classification ----------------------------------------
    sig_h = {h: [f_ids[j] for j in range(len(f_ids)) if Scos[i, j] > thr[i]]
             for i, h in enumerate(h_ids)}
    thr_pooled = float(np.percentile(bn, pctl))
    sig_f = {f: [h_ids[i] for i in range(len(h_ids))
                 if Scos[i, j] > max(thr[i], thr_pooled)]
             for j, f in enumerate(f_ids)}

    av = (availability.set_index("highlight_cluster")
          if availability is not None else None)
    stats = (second_per_cluster.set_index("fr_cluster")
             if second_per_cluster is not None else None)
    rows = []
    for i, h in enumerate(h_ids):
        S = sig_h[h]
        j = int(Scos[i].argmax())
        rel = ("NON_RECOVERY" if not S else "SPLIT" if len(S) >= 2
               else ("MERGE" if len(sig_f[S[0]]) >= 2 else "SINGLE_MATCH"))
        fb = f_ids[j]
        row = {"highlight_cluster": h,
               "highlight_name": category_name.get(h, str(h)),
               "domain": category_domain.get(h, ""),
               "best_second_cluster": fb,
               "centroid_cosine": float(Scos[i, j]),
               "lexical_weighted_jaccard": float(Slex[i, j]),
               f"lexical_overlap_at_{K}": float(Stop[i, j]),
               "null_p95": float(thr[i]), "null_p_empirical": p_emp[i],
               "exceeds_null": bool(obs[i] > thr[i]),
               "exceeds_within_register_reference": bool(obs[i] > ref),
               "n_second_above_threshold": len(S),
               "second_clusters_above_threshold": ";".join(map(str, S)),
               "relation": rel}
        if av is not None and h in av.index:
            row["availability_class"] = av.loc[h, "availability_class"]
            row["matched_event_sentences"] = int(av.loc[h, "matched10_sentences"])
            row["interpretable_non_recovery"] = bool(
                rel == "NON_RECOVERY"
                and row["availability_class"] == "WELL_REPRESENTED")
        if stats is not None and fb in stats.index:
            row["second_cluster_size"] = int(stats.loc[fb, "size"])
        rows.append(row)
    corr = pd.DataFrame(rows)

    involved = set()
    for S in sig_h.values():
        involved.update(S)
    unmatched = [f for f in f_ids if f not in involved]

    counts = corr["relation"].value_counts().to_dict()
    out = {"status": "COMPLETED",
           "n_primary_categories": len(h_ids), "n_second_clusters": len(f_ids),
           "exceeds_own_null": int((obs > thr).sum()),
           "exceeds_within_register_reference": int((obs > ref).sum()),
           "within_register_reference": ref,
           "relations": {k: int(counts.get(k, 0)) for k in
                         ("SINGLE_MATCH", "SPLIT", "MERGE", "NON_RECOVERY")},
           "second_clusters_unmatched": len(unmatched),
           "n_null_replicates": n_null, "null_seed": null_seed,
           "median_best_cosine": float(np.median(obs)),
           "correspondence": corr, "null_table": null_tab,
           "similarity_matrix": sim}
    if availability is not None and "availability_class" in corr.columns:
        well = corr[corr["availability_class"] == "WELL_REPRESENTED"]
        out["well_represented"] = int(len(well))
        out["well_represented_non_recovery"] = int((well["relation"] == "NON_RECOVERY").sum())
        out["median_best_cosine_well_represented"] = (
            float(well["centroid_cosine"].median()) if len(well) else float("nan"))
    log(f"correspondence: {out['exceeds_own_null']}/{len(h_ids)} above own null, "
        f"{out['exceeds_within_register_reference']}/{len(h_ids)} above the "
        f"within-register reference; {out['relations']}; "
        f"{len(unmatched)}/{len(f_ids)} second-register clusters unmatched")
    return out


# ---------------------------------------------------- matched-event pairing --
#: Sao Paulo GP was called the Brazilian GP until 2020; the 2021+ event is the
#: same Interlagos race under a new name. Carried as an alias candidate and
#: reported as disputed, exactly as the frozen manifest records it.
MATCHED_EVENT_ALIASES = {"2024_brazil": "2024_SaoPaulo_GP"}


def _normalised_key(session: str) -> str:
    """Frozen normalisation: lowercase, drop a trailing "_gp"."""
    return str(session).lower().replace("_gp", "")


def matched_events_between(sessions_primary, sessions_second, aliases=None):
    """Pair the two registers' sessions that describe the same event.

    The registers label the same Grand Prix by different conventions, so the
    pairing is made on a normalised key rather than on string equality, with a
    documented alias for the one event that was renamed. Returns the mapping
    from second-register session to primary-register session, together with a
    report carrying the match type of every second-register session.
    """
    aliases = MATCHED_EVENT_ALIASES if aliases is None else aliases
    primary = sorted(set(pd.Series(sessions_primary).astype(str)))
    second = sorted(set(pd.Series(sessions_second).astype(str)))
    norm = {_normalised_key(r): r for r in primary}

    pairs, rows = {}, []
    for i, s in enumerate(second, start=1):
        direct = norm.get(_normalised_key(s))
        alias = aliases.get(s)
        match = direct or alias
        if match is not None and match in norm.values():
            pairs[s] = match
        rows.append({
            "pair_id": f"P{i:02d}",
            "second_session": s,
            "primary_session": match,
            "match_type": ("exact_normalised" if direct else
                           "alias_candidate" if alias else "none"),
            "match_disputed": bool(alias and not direct),
        })
    report = pd.DataFrame(rows)
    log(f"matched events: {len(pairs)}/{len(second)} second-register sessions "
        f"paired ({int(report.match_disputed.sum())} by documented alias)")
    return pairs, report


def reconstruct(cfg2, docs2, out_dir) -> dict:
    """Mask, embed and independently recover the second register's structure.

    Nothing about the primary register enters this stage: the granularity comes
    from this corpus's own sweep. The embedding is returned because the
    correspondence and matched-event stages assign against it.
    """
    from pathlib import Path

    from . import corpus, embedding

    out_dir = Path(out_dir)
    masked2 = corpus.apply_masking(cfg2, docs2)
    X2 = embedding.embed(cfg2, masked2["masked"])
    second = reconstruct_second_register(cfg2, X2, masked2["masked"],
                                         docs2["session"])
    second["embedding"] = X2
    second["masked"] = masked2
    out_dir.mkdir(parents=True, exist_ok=True)
    second["sweep"].to_csv(out_dir / "second_register_granularity_sweep.csv",
                           index=False)
    second["per_cluster"].to_csv(out_dir / "second_register_clusters.csv",
                                 index=False)
    second["seed_stability"].to_csv(
        out_dir / "second_register_seed_stability.csv", index=False)
    return second


def compare_registers(primary_centroids, primary_ids, second, enrichment,
                      labels, sessions, matched_events, out_dir,
                      category_name=None, category_domain=None) -> dict:
    """Correspondence between the categories recovered from each register."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    availability = None
    if matched_events:
        availability = matched_event_availability(
            labels, sessions, matched_events, category_name=category_name,
            category_domain=category_domain)
        availability.to_csv(out_dir / "matched_event_availability.csv",
                            index=False)
    corr = compute_correspondence(
        primary_centroids, primary_ids, second["centroids"], second["ids"],
        second["embedding"], second["labels"], h_enrichment=enrichment,
        f_enrichment=second["enrichment"], availability=availability,
        second_per_cluster=second["per_cluster"], category_name=category_name,
        category_domain=category_domain)
    corr["correspondence"].to_csv(out_dir / "cross_register_correspondence.csv",
                                  index=False)
    corr["null_table"].to_csv(out_dir / "cross_register_null.csv", index=False)
    corr["similarity_matrix"].to_csv(
        out_dir / "cross_register_similarity_matrix.csv")
    corr["scalars"] = {k: v for k, v in corr.items()
                       if not isinstance(v, (pd.DataFrame, np.ndarray))}
    return corr
