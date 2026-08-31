"""Domain-level structure over cluster centroids.

Two independently derived groupings — Ward on the centroid distance matrix and
Louvain on the centroid similarity network — are compared with each other. For a
corpus with no hand-assigned domains, agreement between the two *is* the
evidence; where an external domain map exists, alignment against it is reported
as well.

Both groupings operate on the same centroid geometry, so agreement is a
geometric consistency check, not independent validation.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .io_utils import cached, log


def ward_groups(C: np.ndarray, k: int):
    """Ward linkage on Euclidean distances over L2-normalised centroids.

    Ward's variance-minimising update is defined for Euclidean geometry;
    supplying a cosine-distance matrix feeds squared-scale values into that
    recurrence and does not implement Ward."""
    Z = linkage(pdist(C, metric="euclidean"), method="ward")
    return fcluster(Z, t=k, criterion="maxclust"), Z


def louvain_groups(C: np.ndarray, threshold: float, seed: int = 42):
    import networkx as nx
    S = C @ C.T
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(C)))
    for i, j in itertools.combinations(range(len(C)), 2):
        if S[i, j] >= threshold:
            G.add_edge(i, j, weight=float(S[i, j]))
    try:
        comms = nx.community.louvain_communities(G, seed=seed, weight="weight")
    except Exception:
        comms = list(nx.connected_components(G))
    lab = np.zeros(len(C), dtype=int)
    for gi, com in enumerate(sorted(comms, key=lambda s: -len(s)), start=1):
        for n in com:
            lab[n] = gi
    density = nx.density(G) if len(G) > 1 else 0.0
    return lab, {"edges": G.number_of_edges(), "density": float(density),
                 "n_communities": int(len(set(lab.tolist())))}


def best_mapping_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, dict]:
    """Hungarian best-mapping agreement between two labelings."""
    from scipy.optimize import linear_sum_assignment
    ua, ub = sorted(set(a.tolist())), sorted(set(b.tolist()))
    M = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            M[i, j] = np.sum((a == x) & (b == y))
    r, c = linear_sum_assignment(-M)
    matched = M[r, c].sum()
    return float(matched / len(a)), {ua[i]: ub[j] for i, j in zip(r, c)}


def domain_analysis(cfg, ids, C, cards: pd.DataFrame, external: dict | None = None):
    """Return (per-cluster frame, summary dict, linkage matrix)."""

    def _compute():
        w, _ = ward_groups(C, cfg.n_domains)
        lv, meta = louvain_groups(C, cfg.network_cosine, cfg.seed)
        agree, _ = best_mapping_agreement(w, lv)
        df = pd.DataFrame({"cluster": ids, "ward_group": w, "louvain_group": lv})
        df.attrs = {}
        log(f"domains: Ward K={cfg.n_domains} vs Louvain "
            f"({meta['n_communities']} communities, density {meta['density']:.3f}) "
            f"-> {agree:.1%} best-mapping agreement")
        return df

    df = cached(cfg, "domains.csv", _compute)
    _, Z = ward_groups(C, cfg.n_domains)
    lv, meta = louvain_groups(C, cfg.network_cosine, cfg.seed)
    agree, _ = best_mapping_agreement(df["ward_group"].to_numpy(),
                                      df["louvain_group"].to_numpy())

    df = df.merge(cards[["cluster", "handle", "size", "keywords"]], on="cluster", how="left")

    summary = {
        "n_domains_requested": cfg.n_domains,
        "louvain_communities": meta["n_communities"],
        "network_edges": meta["edges"],
        "network_density": round(meta["density"], 3),
        "ward_louvain_agreement": round(agree, 4),
        "chance_baseline": round(monte_carlo_chance(len(ids), cfg.n_domains, cfg.seed), 4),
    }

    if external:
        ext = np.array([external.get(str(c), "?") for c in ids])
        if len(set(ext.tolist())) > 1:
            a_w, _ = best_mapping_agreement(ext, df["ward_group"].to_numpy())
            a_l, _ = best_mapping_agreement(ext, df["louvain_group"].to_numpy())
            df["external_domain"] = ext
            summary["external_vs_ward"] = round(a_w, 4)
            summary["external_vs_louvain"] = round(a_l, 4)

    return df, summary, Z


def monte_carlo_chance(n_items: int, k: int, seed: int = 42, trials: int = 2000) -> float:
    """Expected best-mapping agreement between two random k-way labelings."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        a = rng.integers(0, k, n_items)
        b = rng.integers(0, k, n_items)
        vals.append(best_mapping_agreement(a, b)[0])
    return float(np.mean(vals))


def domain_keywords(df: pd.DataFrame, enr: pd.DataFrame, group_col: str = "ward_group",
                    top_k: int = 6) -> pd.DataFrame:
    """Name each derived domain by its own aggregated enriched terms."""
    rows = []
    for g, sub in df.groupby(group_col):
        pool = enr[enr["cluster"].isin(sub["cluster"])] if len(enr) else enr
        if len(pool):
            agg = (pool.groupby("term")["p"].min()
                   .sort_values().head(top_k).index.tolist())
        else:
            agg = []
        rows.append({group_col: g,
                     "n_clusters": len(sub),
                     "n_sentences": int(sub["size"].sum()),
                     "domain_keywords": " / ".join(agg) if agg else "(none)"})
    return pd.DataFrame(rows).sort_values("n_clusters", ascending=False).reset_index(drop=True)


def ward_cut_profile(C: np.ndarray, ks=(2, 3, 4, 5, 6, 8, 10)) -> pd.DataFrame:
    """Group sizes at successive Ward cuts.

    Shows where the centroid hierarchy splits informatively and where a cut only
    shaves off a singleton, which is the evidence for the chosen domain count.
    """
    Z = linkage(pdist(C, metric="euclidean"), method="ward")
    rows = []
    for k in ks:
        if k >= len(C):
            continue
        lab = fcluster(Z, t=k, criterion="maxclust")
        sizes = sorted(np.bincount(lab)[1:].tolist(), reverse=True)
        rows.append({"K": k, "group_sizes": str(sizes),
                     "largest": sizes[0], "smallest": sizes[-1],
                     "singletons": int(sum(1 for s in sizes if s == 1))})
    return pd.DataFrame(rows)


def per_group_agreement(dom: pd.DataFrame) -> pd.DataFrame:
    """Ward-vs-Louvain agreement broken out per derived domain.

    The paper reports alignment per hand-assigned domain; with no hand assignment
    the derived Ward groups take that role, so the table shows which parts of the
    structure the two algorithms agree on and which they split.
    """
    a = dom["ward_group"].to_numpy()
    b = dom["louvain_group"].to_numpy()
    _, mapping = best_mapping_agreement(a, b)
    rows = []
    for g in sorted(set(a.tolist())):
        m = a == g
        tgt = mapping.get(g)
        matched = int(((b == tgt) & m).sum()) if tgt is not None else 0
        rows.append({
            "ward_group": g,
            "n_clusters": int(m.sum()),
            "louvain_counterpart": tgt,
            "matched": matched,
            "match_pct": round(100 * matched / max(int(m.sum()), 1), 1),
        })
    return pd.DataFrame(rows).sort_values("n_clusters", ascending=False).reset_index(drop=True)


def subcluster_profile(cfg, X, U, labels, cards: pd.DataFrame, texts,
                       top_n: int = 3, k_sub: int = 3) -> pd.DataFrame:
    """Split the largest clusters internally to expose latent sub-structure.

    Reported two ways, because they can disagree. A fixed-K cut always returns
    sub-groups — it is asked to. The density clusterer can return none, which
    would mean the region is one mode rather than several themes glued together.

    Each test runs in the space it belongs to: Ward on the sentence embeddings,
    matching the hierarchical axis, and the density pass on the reduced space
    where the primary partition was found. Running the density pass on the raw
    high-dimensional embeddings instead returns 100% noise for any input, which
    reads as homogeneity but is only a failure to estimate density.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from .clustering import cluster_hdbscan, n_clusters, noise_rate
    texts = np.asarray(list(texts), dtype=object)
    stop = sorted(ENGLISH_STOP_WORDS | {c.lower() for c in cfg.entity_classes})
    rows = []
    for _, r in cards.nlargest(top_n, "size").iterrows():
        cid = int(r["cluster"])
        idx = np.nonzero(labels == cid)[0]
        if len(idx) < k_sub * 5:
            continue
        Z = linkage(pdist(X[idx], metric="euclidean"), method="ward")
        sub = fcluster(Z, t=k_sub, criterion="maxclust")

        # the density view of the same region, at half the primary granularity
        mcs_in = max(5, cfg.min_cluster_size // 2)
        dens = cluster_hdbscan(U[idx], mcs_in, cfg)
        dens_k, dens_noise = n_clusters(dens), round(100 * noise_rate(dens), 1)

        # terms that separate each sub-group from its siblings, not from the corpus
        docs = [" ".join(texts[idx[sub == s]]) for s in sorted(set(sub.tolist()))]
        try:
            V = TfidfVectorizer(stop_words=stop, min_df=1)
            M = V.fit_transform(docs).toarray()
            vocab = np.array(V.get_feature_names_out())
        except ValueError:
            M, vocab = np.zeros((len(docs), 0)), np.array([])

        for k, s in enumerate(sorted(set(sub.tolist()))):
            members = idx[sub == s]
            kws = (" / ".join(vocab[np.argsort(-M[k])[:5]]) if len(vocab) else "(none)")
            rows.append({
                "parent_handle": r["handle"], "parent_size": int(r["size"]),
                "sub_id": int(s), "sub_size": int(len(members)),
                "share_of_parent": round(len(members) / len(idx), 3),
                "sub_keywords": kws,
                "density_subclusters": dens_k, "density_noise_pct": dens_noise,
            })
    return pd.DataFrame(rows)
