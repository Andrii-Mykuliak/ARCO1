"""R3 vocabulary enrichment: exact hypergeometric test + Benjamini-Hochberg FDR.

Each (cluster, term) pair is a sampling-without-replacement question: given a
corpus of N sentences of which K contain the term, and a cluster of n sentences,
how improbable is it to see k or more cluster sentences containing it? The upper
tail of the hypergeometric distribution answers this in closed form, with no
resampling and no resolution floor. A single BH family spans all pairs, so no
cluster's own size sets its own threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import cached, log


def _doc_term_matrix(cfg, texts, fit_mask=None):
    """Binary-ready doc-term matrix. ``fit_mask`` restricts which sentences may
    define the vocabulary; every sentence is still transformed. Axis 3 uses it
    so held-out races cannot influence the feature universe."""
    import re as _re
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    # Strip the placeholder TOKENS from the text before building the vocabulary,
    # matching Section 4.3. Stop-wording "person"/"team"/"place" after
    # tokenisation would also discard ordinary uses, which are genuine content.
    ph = _re.compile("<(?:" + "|".join(cfg.entity_classes) + ")>")
    texts = [ph.sub(" ", t) for t in texts]
    # sklearn's default token pattern: a custom one changes the vocabulary and
    # no longer matches scripts/r3_hypergeometric.py, which defines canonical R3.
    vec = CountVectorizer(
        min_df=cfg.r3_min_df,
        ngram_range=(1, cfg.r3_ngram_max),
        lowercase=True,
        max_features=cfg.r3_max_features,
        stop_words=sorted(ENGLISH_STOP_WORDS),
    )
    if fit_mask is None:
        M = vec.fit_transform(texts)
    else:
        vec.fit([t for t, keep in zip(texts, fit_mask) if keep])
        M = vec.transform(texts)
    return M, np.array(vec.get_feature_names_out())


def _bh_fdr(pvals: np.ndarray, q: float):
    """Benjamini-Hochberg. Returns (reject, qvalues)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    crit = q * (np.arange(1, n + 1) / n)
    passed = ranked <= crit
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k:
        reject[order[:k]] = True
    qvals = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out_q = np.empty(n)
    out_q[order] = np.clip(qvals, 0, 1)
    return reject, out_q


def _enrich_uncached(cfg, texts, labels: np.ndarray, verbose: bool = True) -> pd.DataFrame:
    """One row per FDR-significant (cluster, term) pair, exact hypergeometric."""
    from scipy.stats import hypergeom

    M, vocab = _doc_term_matrix(cfg, list(texts))
    M = (M > 0).astype(np.int32)          # binary sentence incidence
    N_docs = M.shape[0]
    K = np.asarray(M.sum(axis=0)).ravel()  # corpus sentences containing each term
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    if verbose:
        log(f"r3: vocabulary {len(vocab):,} terms over {N_docs:,} sentences, "
            f"{len(vocab) * len(ids):,} candidate pairs")

    rows = []
    for cid in ids:
        idx = np.nonzero(labels == cid)[0]
        n = len(idx)
        k = np.asarray(M[idx].sum(axis=0)).ravel()
        p = hypergeom.sf(k - 1, N_docs, K, n)      # P(X >= k)
        # Every vocab x cluster pair joins the BH family, including k=0 (p=1):
        # dropping them would shrink the family and inflate significance.
        for t, kk, KK, pp in zip(vocab, k, K, p):
            rows.append({"cluster": cid, "term": t, "in_cluster": int(kk),
                         "corpus_docfreq": int(KK), "cluster_size": n,
                         "expected": round(n * KK / N_docs, 3),
                         "p": float(pp), "is_bigram": int(" " in t)})

    df = pd.DataFrame(rows)
    if not len(df):
        return df
    # ONE global Benjamini-Hochberg family across every (cluster, term) test
    reject, qv = _bh_fdr(df["p"].to_numpy(), cfg.r3_fdr_q)
    df["q"], df["significant"] = qv, reject
    df = df[df["significant"]].drop(columns=["significant"])
    df = df.sort_values(["cluster", "p"]).reset_index(drop=True)
    if verbose:
        for cid in ids:
            log(f"  cluster {cid:>3}: enriched={int((df['cluster'] == cid).sum())}")
    return df


def r3_enrichment(cfg, texts, labels: np.ndarray) -> pd.DataFrame:
    """Return one row per FDR-significant (cluster, term) pair."""
    df = cached(cfg, "r3_enrichment.csv", lambda: _enrich_uncached(cfg, texts, labels))
    if len(df):
        log(f"r3: {len(df):,} significant (cluster, term) pairs, "
            f"{df['is_bigram'].mean():.0%} bigrams (exact hypergeometric, global BH)")
    return df


def top_terms(enr: pd.DataFrame, cluster: int, n: int = 8) -> list[str]:
    if not len(enr):
        return []
    sub = enr[enr["cluster"] == cluster].nsmallest(n, "p")
    return sub["term"].tolist()


def enrichment_summary(enr: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    rows = []
    for c in ids:
        sub = enr[enr["cluster"] == c] if len(enr) else enr
        rows.append({
            "cluster": c,
            "size": int((labels == c).sum()),
            "n_enriched": int(len(sub)),
            "pct_bigram": float(sub["is_bigram"].mean()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def vocabulary_size(cfg, texts) -> int:
    """Size of the R3 candidate vocabulary — the denominator behind the FDR."""
    _, vocab = _doc_term_matrix(cfg, list(texts))
    return int(len(vocab))


def enrichment_across_plateau(cfg, texts, U, mcs_values):
    """Vocabulary-signature richness at each granularity in the plateau.

    Cluster-count stability alone does not pick an operating point inside a
    plateau; richness is one of the tie-breakers, so it is measured rather than
    asserted. The hypergeometric test is exact, so every point here uses the
    same procedure as the headline run.
    """
    from .clustering import cluster_hdbscan, n_clusters

    def _compute():
        rows = []
        for m in mcs_values:
            lab = cluster_hdbscan(U, int(m), cfg)
            k = n_clusters(lab)
            if k < 2:
                rows.append({"min_cluster_size": int(m), "n_clusters": k,
                             "enriched_pairs": 0, "mean_per_cluster": 0.0,
                             "clusters_with_zero": 0})
                continue
            sub = _enrich_uncached(cfg, list(texts), lab, verbose=False)
            ids = sorted(int(c) for c in set(lab.tolist()) if c != -1)
            per = pd.Series({c: int((sub["cluster"] == c).sum()) if len(sub) else 0
                             for c in ids})
            rows.append({
                "min_cluster_size": int(m),
                "n_clusters": k,
                "enriched_pairs": int(len(sub)),
                "mean_per_cluster": round(float(len(sub) / k), 1),
                "clusters_with_zero": int((per == 0).sum()),
                # the "interpretability fraction": clusters with a usable signature
                "frac_clusters_ge5_terms": round(float((per >= 5).mean()), 3),
            })
            log(f"  richness mcs={m:>3}: {rows[-1]['mean_per_cluster']} terms/cluster")
        return pd.DataFrame(rows)

    return cached(cfg, "enrichment_richness.csv", _compute)
