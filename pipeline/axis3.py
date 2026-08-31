"""Axis 3 - held-out race replication on a seeded 80/33 race-level split.

Two stages. Stage A reconstructs cluster structure from the training races
alone and projects the held-out races onto it; Stage B asks whether each
cluster's canonical R3 lexical signature survives that change of races.

Stage A is a genuine refit, not an assignment rule: UMAP and HDBSCAN are fitted
on the training races only, the resulting clusters are Hungarian-aligned to the
canonical full-corpus identifiers, and held-out sentences are projected with
``UMAP.transform`` and labelled with ``hdbscan.approximate_predict``. A cluster
the training subset never reconstructs has no held-out estimate and fails.

``AXIS3_TRAIN_MIN_CLUSTER_SIZE`` is the fixed train-split granularity used by
this validation protocol. It is not the primary pipeline's value and is not
derived from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import cached, log

SEED = 42
N_TRAIN_RACES = 80
AXIS3_TRAIN_MIN_CLUSTER_SIZE = 30


def race_split(sessions: pd.Series, n_train: int = N_TRAIN_RACES, seed: int = SEED):
    """The seeded split: choice without replacement over sorted race ids."""
    races_all = sorted(sessions.unique().tolist())
    rng = np.random.default_rng(seed)
    n_train = min(n_train, len(races_all))
    train = sorted(rng.choice(races_all, n_train, replace=False).tolist())
    test = sorted([r for r in races_all if r not in set(train)])
    return train, test


def hungarian_align(src_labels: np.ndarray, ref_labels: np.ndarray,
                    strict: bool = True) -> dict:
    """Map reconstructed cluster ids onto canonical ids by maximum overlap.

    With ``strict`` the mapping must be total: an unmatched source cluster is an
    error, because silently minting a pseudo-id turns a mismatched reference
    partition into plausible-looking output.
    """
    from scipy.optimize import linear_sum_assignment

    src_ids = sorted(set(src_labels.tolist()) - {-1})
    ref_ids = sorted(set(ref_labels.tolist()) - {-1})
    cost = np.zeros((len(src_ids), len(ref_ids)))
    for i, s in enumerate(src_ids):
        s_mask = src_labels == s
        for j, r in enumerate(ref_ids):
            cost[i, j] = -(s_mask & (ref_labels == r)).sum()
    n = max(len(src_ids), len(ref_ids))
    cost_sq = np.zeros((n, n))
    cost_sq[:len(src_ids), :len(ref_ids)] = cost
    row_ind, col_ind = linear_sum_assignment(cost_sq)

    mapping = {-1: -1}
    for i, j in zip(row_ind, col_ind):
        if i < len(src_ids) and j < len(ref_ids):
            mapping[src_ids[i]] = ref_ids[j]
    unmatched = [s for s in src_ids if s not in mapping]
    if unmatched and strict:
        raise ValueError(
            f"Hungarian alignment left {len(unmatched)} reconstructed clusters "
            f"unmatched against {len(ref_ids)} canonical ids. The reference "
            f"partition is incompatible with the corpus; refusing to mint "
            f"pseudo-ids. Unmatched: {unmatched}")
    for s in unmatched:
        mapping[s] = -100 - s
    return mapping


def validate_reference_partition(labels: np.ndarray, n_sentences: int,
                                 expect_clusters: int | None = None,
                                 expect_clustered: int | None = None) -> None:
    """Fail fast when the reference partition is not what it claims.

    Structural checks always run. ``expect_*`` pin a corpus to a known
    published partition and are supplied by the config; when absent (a corpus
    with no archived reference) only the structural checks apply.
    """
    if len(labels) != n_sentences:
        raise ValueError(f"reference has {len(labels)} rows, corpus has {n_sentences}")
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    n_clustered = int((labels != -1).sum())
    if not ids:
        raise ValueError("reference partition has no clusters")
    if ids != list(range(len(ids))):
        raise ValueError(f"reference cluster ids are not contiguous from 0: {ids[:5]}...")
    if expect_clusters is not None and len(ids) != expect_clusters:
        raise ValueError(f"reference has {len(ids)} clusters, expected {expect_clusters}")
    if expect_clustered is not None and n_clustered != expect_clustered:
        raise ValueError(f"reference has {n_clustered} clustered sentences, "
                         f"expected {expect_clustered}")


def _stage_a(cfg, embeddings: np.ndarray, sessions: pd.Series,
             canonical: np.ndarray) -> dict:
    """Train-only refit, alignment, and held-out projection."""
    import hdbscan
    import umap

    n_sessions = int(sessions.nunique())
    n_train = getattr(cfg, "axis3_n_train_sessions", None)
    if n_train is None:                       # corpus without a published split
        n_train = max(1, round(0.70 * n_sessions))
    n_train = min(n_train, n_sessions)
    if n_sessions - n_train < 1:
        raise ValueError(
            f"axis3: {n_sessions} sessions with n_train={n_train} leaves no held-out "
            f"sessions; set cfg.axis3_n_train_sessions below {n_sessions}")

    train_races, test_races = race_split(sessions, n_train=n_train)
    tr = sessions.isin(set(train_races)).to_numpy()
    te = ~tr

    reducer = umap.UMAP(n_neighbors=cfg.umap_n_neighbors,
                        min_dist=cfg.umap_min_dist,
                        n_components=cfg.umap_n_components,
                        metric=cfg.umap_metric,
                        random_state=cfg.seed)
    u_train = reducer.fit_transform(embeddings[tr])
    u_test = reducer.transform(embeddings[te])

    mcs = getattr(cfg, "axis3_train_min_cluster_size", None) or cfg.min_cluster_size
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                metric=cfg.hdbscan_metric,
                                cluster_selection_method=cfg.hdbscan_selection,
                                prediction_data=True)
    raw_train = clusterer.fit_predict(u_train)
    raw_test, _ = hdbscan.approximate_predict(clusterer, u_test)

    mapping = hungarian_align(raw_train, canonical[tr], strict=True)
    labels = np.empty(len(embeddings), dtype=int)
    labels[tr] = [mapping.get(int(l), -1) for l in raw_train]
    labels[te] = [mapping.get(int(l), -1) for l in raw_test]

    log(f"axis3 stage A: {len(train_races)} train / {len(test_races)} held-out sessions (mcs={mcs}); "
        f"{len(set(raw_train.tolist()) - {-1})} train clusters reconstructed; "
        f"{int((labels[tr] != -1).sum()):,} train / {int((labels[te] != -1).sum()):,} "
        f"held-out sentences assigned")
    return {"train_mask": tr, "test_mask": te, "labels": labels}


def _hypergeometric(labels: np.ndarray, M, cluster_ids: list[int]):
    """Exact upper-tail p and enrichment z for every (cluster, term) pair."""
    from scipy.stats import hypergeom

    n_sent, n_clu = len(labels), len(cluster_ids)
    pos = {c: i for i, c in enumerate(cluster_ids)}
    idx = np.array([pos.get(int(l), -1) for l in labels])
    ind = np.zeros((n_sent, n_clu), dtype=np.float32)
    ok = idx >= 0
    ind[np.arange(n_sent)[ok], idx[ok]] = 1.0

    k = ind.T @ M
    k = np.asarray(k.todense() if hasattr(k, "todense") else k)
    N = n_sent
    K = np.asarray(M.sum(axis=0)).ravel().astype(float)
    n = ind.sum(axis=0).astype(float)
    p = hypergeom.sf(k - 1, N, K[None, :], n[:, None])
    expected = n[:, None] * K[None, :] / N
    var = np.maximum(expected * (1 - K[None, :] / N) * (N - n[:, None]) / (N - 1), 1e-12)
    return p, (k - expected) / np.sqrt(var)


def axis3_heldout_replication(cfg, df: pd.DataFrame, embeddings: np.ndarray,
                              canonical: np.ndarray,
                              text_col: str = "masked") -> pd.DataFrame:
    """Per-cluster held-out race replication: the canonical Axis 3."""

    def _compute():
        from .enrichment import _bh_fdr, _doc_term_matrix

        validate_reference_partition(
            canonical, len(df),
            getattr(cfg, "axis3_expect_clusters", None),
            getattr(cfg, "axis3_expect_clustered", None))
        sessions = df[cfg.session_field] if cfg.session_field in df.columns else df["session"]
        A = _stage_a(cfg, embeddings, sessions, canonical)
        labels, tr, te = A["labels"], A["train_mask"], A["test_mask"]

        # The canonical R3 procedure re-applied to the training races: the
        # vocabulary is fitted on training text alone, so held-out races cannot
        # reach the feature universe through min_df. Held-out sentences are
        # transformed with that fitted vectoriser, never fitted on.
        M, _vocab = _doc_term_matrix(cfg, df[text_col].astype(str).tolist(),
                                     fit_mask=A["train_mask"])
        M = (M > 0).astype(np.int32)

        keep_tr = tr & (labels != -1)
        keep_te = te & (labels != -1)
        ids = sorted(set(labels[keep_tr].tolist()).intersection(labels[keep_te].tolist()))
        log(f"axis3: {len(ids)} clusters have both training and held-out support")

        # Background prevalence spans EVERY sentence in the split, noise
        # included, matching Section 4.3 where the corpus is the background and
        # noise is excluded only from the cluster loop. Unassigned sentences
        # carry label -1 and so contribute to N and K but never to any n or k.
        tr_p, _tr_z = _hypergeometric(labels[tr], M[tr], ids)
        te_p, te_z = _hypergeometric(labels[te], M[te], ids)

        tr_rej = _bh_fdr(tr_p.flatten(), cfg.r3_fdr_q)[0].reshape(tr_p.shape)
        sub = np.zeros_like(tr_rej)
        if tr_rej.any():
            sub[tr_rej] = _bh_fdr(te_p[tr_rej], cfg.r3_fdr_q)[0]

        strict = sub & tr_rej
        lenient = (te_z > 0) & (te_p < 0.10) & tr_rej
        direction = (te_z > 0) & tr_rej

        all_ids = sorted(int(c) for c in set(canonical.tolist()) if c != -1)
        rows = []
        for c in all_ids:
            if c not in ids:
                rows.append({"cluster": c, "has_estimate": False,
                             "n_train_enriched": 0, "n_test_strict": 0,
                             "n_test_lenient": 0, "n_test_direction": 0,
                             "rep_strict": np.nan, "rep_lenient": np.nan,
                             "rep_dir": np.nan})
                continue
            i = ids.index(c)
            n_tr = int(tr_rej[i].sum())
            s, l, d = int(strict[i].sum()), int(lenient[i].sum()), int(direction[i].sum())
            rows.append({"cluster": c, "has_estimate": n_tr > 0,
                         "n_train_enriched": n_tr, "n_test_strict": s,
                         "n_test_lenient": l, "n_test_direction": d,
                         "rep_strict": s / n_tr if n_tr else np.nan,
                         "rep_lenient": l / n_tr if n_tr else np.nan,
                         "rep_dir": d / n_tr if n_tr else np.nan})

        out = pd.DataFrame(rows)
        out["axis3_pass"] = ((out.rep_strict >= 0.5) & (out.rep_dir >= 0.5)).fillna(False)
        out["axis3_score"] = out[["rep_strict", "rep_dir"]].min(axis=1)
        log(f"axis3: {int(out.axis3_pass.sum())}/{len(out)} clusters meet "
            f"rep_strict >= 0.5 AND rep_dir >= 0.5")
        return out

    return cached(cfg, "axis3_heldout.csv", _compute)
