"""Entity-concentration diagnostic: how far is a partition organised around entities?

Given two clusterings of the SAME sentences and a per-sentence entity annotation,
this measures how strongly each partition groups sentences around recurring named
entities. It was written to test the rationale for entity masking -- masking should
reduce grouping around concrete people, teams and places -- but nothing here is
specific to that corpus or to motorsport.

It is a diagnostic, not a validation axis. It says nothing about which partition is
more correct.

Inputs are deliberately plain so no corpus ships with this module:

    entities   list, one entry per sentence, {class: set(entity_id)}
               sentence-level presence; an entity counts once per sentence
    labels_a   int array, one label per sentence, noise = -1
    labels_b   int array, same length, noise = -1

The caller supplies the entity annotation. Any recogniser may produce it, but the
comparison is only meaningful if the SAME annotation is used for both partitions
and is derived from the same underlying text.

Typical use:

    from pipeline.entity_concentration import compare_partitions
    res = compare_partitions(entities, labels_masked, labels_unmasked,
                             classes=("PERSON", "TEAM", "PLACE"))

"""
from __future__ import annotations

import numpy as np

NOISE = -1


# ----------------------------------------------------------------- metrics --
def shannon(counts) -> float:
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size == 0:
        return float("nan")
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def normalised_entropy(counts) -> float:
    """H / log(k).

    k = 0 -> nan (no entity of this class; the cluster is not evaluable and is
                  never silently counted as 0)
    k = 1 -> 0.0 (maximal concentration, defined explicitly)
    k > 1 -> H / log(k), in [0, 1]
    """
    c = [x for x in counts if x > 0]
    k = len(c)
    if k == 0:
        return float("nan")
    if k == 1:
        return 0.0
    return shannon(c) / float(np.log(k))


def shared_clustered(labels_a, labels_b) -> np.ndarray:
    """Indices non-noise in BOTH partitions.

    The primary comparison must use this set, otherwise a difference in entity
    concentration can be an artefact of the two partitions scoring different
    sentence populations.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label arrays differ in length: {a.shape} vs {b.shape}")
    return np.where((a != NOISE) & (b != NOISE))[0]


def cluster_metrics(labels, idx, entities, cls) -> list[dict]:
    """Per-cluster concentration for one entity class over the sentence set `idx`."""
    labels = np.asarray(labels)
    idx = np.asarray(idx)
    lab = labels[idx]
    rows = []
    for cid in sorted(set(lab.tolist())):
        members = idx[lab == cid]
        size = int(members.size)
        counts, n_with = {}, 0
        for s in members:
            e = entities[s].get(cls) or set()
            if e:
                n_with += 1
                for x in e:
                    counts[x] = counts.get(x, 0) + 1
        dom, dom_n = (max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                      if counts else (None, 0))
        rows.append({
            "cluster_id": int(cid),
            "entity_class": cls,
            "cluster_size": size,
            "sentences_with_entity": n_with,
            "entity_coverage": n_with / size if size else float("nan"),
            "n_distinct_entities": len(counts),
            "dominant_entity": dom,
            "dominant_entity_count": dom_n,
            "dominant_entity_share_all": dom_n / size if size else float("nan"),
            "dominant_entity_share_mentions": (dom_n / n_with) if n_with else float("nan"),
            "entity_entropy": shannon(counts.values()),
            "entity_entropy_normalised": normalised_entropy(counts.values()),
        })
    return rows


def aggregate(rows, cls) -> dict:
    """Median/IQR and cluster-size-weighted means.

    Scope differs by metric, deliberately:

      share_all, entity_coverage
          over ALL clusters. A cluster with no entity of this class has
          `dominant_entity_count = 0`, so `share_all = 0` -- a valid measurement,
          not a missing one. Dropping those clusters inflates the mean, and by
          different amounts in two partitions that have different numbers of
          zero-entity clusters, which biases the contrast between them.

      share_mentions, entropy_norm
          over EVALUABLE clusters only (`sentences_with_entity > 0`), where the
          denominator and `H/log(k)` are otherwise undefined.
    """
    allc = [r for r in rows if r["entity_class"] == cls]
    ev = [r for r in allc if r["sentences_with_entity"] > 0]
    out = {"entity_class": cls, "n_clusters": len(allc),
           "n_clusters_evaluable": len(ev)}
    if not allc:
        return out

    def summarise(subset, col, tag):
        if not subset:
            return
        v = np.asarray([r[col] for r in subset], dtype=float)
        w = np.asarray([r["cluster_size"] for r in subset], dtype=float)
        m = ~np.isnan(v)
        if not m.any():
            return
        out[f"{tag}_median"] = float(np.median(v[m]))
        out[f"{tag}_q1"] = float(np.percentile(v[m], 25))
        out[f"{tag}_q3"] = float(np.percentile(v[m], 75))
        out[f"{tag}_wmean"] = float(np.average(v[m], weights=w[m]))

    summarise(allc, "dominant_entity_share_all", "share_all")
    summarise(ev, "dominant_entity_share_mentions", "share_mentions")
    summarise(ev, "entity_entropy_normalised", "entropy_norm")
    out["entity_coverage_wmean"] = float(np.average(
        np.asarray([r["entity_coverage"] for r in allc], dtype=float),
        weights=np.asarray([r["cluster_size"] for r in allc], dtype=float)))
    return out


def aggregate_labels(labels, idx, entities, cls) -> dict:
    return aggregate(cluster_metrics(labels, idx, entities, cls), cls)


# ------------------------------------------------------------- uncertainty --
def paired_bootstrap(entities, labels_a, labels_b, idx, cls,
                     metric="share_all_wmean", n=1000, seed=42) -> dict:
    """Resample sentences once and score BOTH partitions on the same resample."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    diffs = []
    for _ in range(n):
        pick = rng.choice(idx, size=idx.size, replace=True)
        a = aggregate_labels(labels_a, pick, entities, cls).get(metric, np.nan)
        b = aggregate_labels(labels_b, pick, entities, cls).get(metric, np.nan)
        diffs.append(b - a)
    v = np.asarray(diffs, dtype=float)
    v = v[~np.isnan(v)]
    obs_a = aggregate_labels(labels_a, idx, entities, cls).get(metric, np.nan)
    obs_b = aggregate_labels(labels_b, idx, entities, cls).get(metric, np.nan)
    return {"entity_class": cls, "metric": metric, "a": obs_a, "b": obs_b,
            "point_estimate_b_minus_a": obs_b - obs_a,
            "ci_lo_95": float(np.percentile(v, 2.5)) if v.size else float("nan"),
            "ci_hi_95": float(np.percentile(v, 97.5)) if v.size else float("nan"),
            "n_replicates": int(v.size), "seed": seed}


def permutation_null(entities, labels, idx, cls,
                     metric="share_all_wmean", n=1000, seed=42) -> dict:
    """Null holding entity incidences and the cluster-size multiset fixed.

    Answers: how much concentration does this partition carry beyond what its
    own cluster-size distribution and the corpus entity frequencies imply?
    """
    labels = np.asarray(labels)
    idx = np.asarray(idx)
    rng = np.random.default_rng(seed)
    obs = aggregate_labels(labels, idx, entities, cls).get(metric, np.nan)
    lab = labels[idx]
    vals = []
    for _ in range(n):
        tmp = np.full(labels.shape, NOISE)
        tmp[idx] = rng.permutation(lab)          # size multiset preserved
        vals.append(aggregate_labels(tmp, idx, entities, cls).get(metric, np.nan))
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    lower_is_concentrated = "entropy" in metric
    p = (float((v <= obs).sum() + 1) / (v.size + 1) if lower_is_concentrated
         else float((v >= obs).sum() + 1) / (v.size + 1))
    return {"entity_class": cls, "metric": metric, "observed": obs,
            "null_mean": float(v.mean()) if v.size else float("nan"),
            "null_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "one_sided_empirical_p": p, "n_permutations": int(v.size), "seed": seed,
            "direction": "observed <= null" if lower_is_concentrated else "observed >= null"}


def compare_partitions(entities, labels_a, labels_b, classes,
                       n_boot=1000, n_perm=1000, seed=42) -> dict:
    """Full diagnostic on the shared non-noise population.

    `labels_a` is the reference arm (e.g. masked), `labels_b` the contrast arm
    (e.g. unmasked). Contrasts are reported as b - a. Cluster ids are never
    paired between partitions.
    """
    idx = shared_clustered(labels_a, labels_b)
    out = {"n_shared": int(idx.size), "classes": list(classes),
           "cluster_ids_paired": False, "seed": seed,
           "per_cluster": {}, "aggregate": {}, "bootstrap": [], "permutation": []}
    for cls in classes:
        out["per_cluster"][cls] = {
            "a": cluster_metrics(labels_a, idx, entities, cls),
            "b": cluster_metrics(labels_b, idx, entities, cls)}
        out["aggregate"][cls] = {
            "a": aggregate_labels(labels_a, idx, entities, cls),
            "b": aggregate_labels(labels_b, idx, entities, cls)}
        for m in ("share_all_wmean", "entropy_norm_wmean"):
            out["bootstrap"].append(
                paired_bootstrap(entities, labels_a, labels_b, idx, cls, m, n_boot, seed))
            out["permutation"].append(
                {"arm": "a", **permutation_null(entities, labels_a, idx, cls, m, n_perm, seed)})
            out["permutation"].append(
                {"arm": "b", **permutation_null(entities, labels_b, idx, cls, m, n_perm, seed)})
    return out
