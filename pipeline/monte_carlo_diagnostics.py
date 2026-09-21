"""Extended Monte-Carlo diagnostic experiments, ported from the frozen producers.

Two experiments share one frozen analytical input and one deterministic roster:

* resampling stability - 200 subsample replicates at 80 per cent, re-projected
  and re-clustered, scored by best-overlap Jaccard against the canonical
  partition;
* alternative geometry - 100 replicates that draw a fixed number of sentences
  per category and re-cluster them under a different geometry entirely (Ward
  linkage on Euclidean distances), scored the same way.

THE ROSTER IS NESTED, NOT INDEPENDENT. Each experiment draws from a single
sequential ``np.random.default_rng(seed)`` and, for resampling, a projection seed
that is a pure function of the replicate index. Extending the loop therefore
reproduces a shorter historical roster exactly as its own prefix, which makes
"shorter roster versus longer roster" a convergence question rather than two
independent estimates.

Ported from ``scripts/axis24_monte_carlo.py`` (historically Axis 2 and Axis 4).
The method is unchanged: subsample fraction, granularity, projection settings,
per-category sample size, cluster count, linkage, matching rule and Jaccard
definition all come from that producer. Only machine-specific paths and the
resume-from-JSONL plumbing were replaced by the package's own cache.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .io_utils import best_jaccard, log

# ---- pre-specified before execution, copied from the frozen producer --------
PROTOCOL = "monte-carlo-diagnostics/1"
SEED = 42
RESAMPLE_FRACTION = 0.80
RESAMPLE_MIN_CLUSTER_SIZE = 35
RESAMPLE_N = 200
RESAMPLE_HISTORICAL_PREFIX = 30
GEOMETRY_N = 100
GEOMETRY_HISTORICAL_PREFIX = 10
GEOMETRY_PER_CATEGORY = 15
UMAP_KW = dict(n_neighbors=15, min_dist=0.0, n_components=5, metric="cosine")
THRESHOLDS = (0.45, 0.50, 0.55)
CANONICAL_THRESHOLD = 0.50
# Pre-specified degeneracy rule, fixed for the held-out factorial before this
# pass and applied unchanged here: a replicate that recovers fewer than this many
# clusters, or leaves less than this fraction unclustered, did not produce a
# viable reconstruction to score against.
DEGENERATE_MIN_CLUSTERS = 10
DEGENERATE_MIN_NOISE = 0.05


def is_degenerate(rec) -> bool:
    return bool(rec.get("n_clusters", 99) < DEGENERATE_MIN_CLUSTERS
                or rec.get("noise_fraction", 1.0) < DEGENERATE_MIN_NOISE)


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def require_frozen_input(path, expected_sha256: str, expected_shape=None,
                         expected_dtype="float32") -> np.ndarray:
    """Load a frozen analytical input, or fail closed.

    A replacement is never generated automatically: these experiments are defined
    on one particular realisation, and re-encoding the corpus in a different
    environment produces a different array and therefore a different experiment.
    """
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"frozen analytical input {path.name} is not present at {path}. "
            "It is required to reproduce this experiment and is never "
            f"regenerated automatically; see data/f1_frozen_inputs/MANIFEST.json.")
    a = np.load(path)
    got = array_sha256(a)
    problems = []
    if got != expected_sha256:
        problems.append(f"sha256 {got[:16]}... != expected {expected_sha256[:16]}...")
    if expected_shape is not None and tuple(a.shape) != tuple(expected_shape):
        problems.append(f"shape {a.shape} != expected {tuple(expected_shape)}")
    if expected_dtype is not None and a.dtype.name != expected_dtype:
        problems.append(f"dtype {a.dtype.name} != expected {expected_dtype}")
    if problems:
        raise ValueError(
            f"frozen analytical input {path.name} does not match its recorded "
            "identity: " + "; ".join(problems) + ". This is a different "
            "realisation, not the one the experiment is defined on.")
    return a


def _identity(embedding, labels, experiment: str, n_replicates: int,
              **params) -> dict:
    return {"protocol": PROTOCOL, "experiment": experiment,
            "embedding_sha256": array_sha256(embedding),
            "labels_sha256": array_sha256(labels),
            "n_replicates": n_replicates, "seed": SEED, **params}


def _cached_experiment(cfg, name: str, identity: dict, compute):
    """Cache keyed on the full scientific identity; fail closed on mismatch."""
    from pathlib import Path
    base = Path(cfg.cache_dir) / name
    meta_path = base.with_suffix(".identity.json")
    data_path = base.with_suffix(".jsonl")
    if data_path.exists() and meta_path.exists() and not cfg.from_scratch:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        drift = {k: (stored.get(k), v) for k, v in identity.items()
                 if stored.get(k) != v}
        if drift:
            raise ValueError(
                f"the cached {name} was produced under a different scientific "
                f"identity and cannot be reused: "
                + "; ".join(f"{k} was {a!r}, is now {b!r}"
                            for k, (a, b) in drift.items())
                + ". Remove that cache to recompute the experiment.")
        rows = [json.loads(ln) for ln in
                data_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        log(f"cache hit  {data_path.name} ({len(rows)} replicates)")
        return rows
    rows = compute()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta_path.write_text(json.dumps(identity, indent=1), encoding="utf-8")
    return rows


def _category_ids(labels) -> list[int]:
    return sorted(int(c) for c in set(np.asarray(labels).tolist()) if c != -1)


# --------------------------------------------------- resampling stability ----
def resampling_stability_mc200(cfg, embedding, labels, n_replicates=RESAMPLE_N):
    """Subsample, re-project, re-cluster, score every category. 200 replicates.

    Historically Axis 2. The generator is advanced for every replicate index, so
    the first ``RESAMPLE_HISTORICAL_PREFIX`` replicates are the published ones
    rather than new draws.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)
    ids = _category_ids(labels)
    identity = _identity(embedding, labels, "resampling_stability", n_replicates,
                         fraction=RESAMPLE_FRACTION,
                         min_cluster_size=RESAMPLE_MIN_CLUSTER_SIZE)

    def _compute():
        import hdbscan
        import umap
        rng = np.random.default_rng(SEED)
        n = len(labels)
        size = int(RESAMPLE_FRACTION * n)
        rows = []
        for b in range(n_replicates):
            take = rng.choice(n, size=size, replace=False)
            take.sort()
            rec = {"replicate": b, "projection_seed": SEED + 100 + b,
                   "n_taken": int(size)}
            try:
                U = umap.UMAP(**UMAP_KW,
                              random_state=SEED + 100 + b).fit_transform(
                                  embedding[take]).astype(np.float32)
                lab_b = hdbscan.HDBSCAN(
                    min_cluster_size=RESAMPLE_MIN_CLUSTER_SIZE,
                    metric="euclidean",
                    cluster_selection_method="eom").fit_predict(U)
            except Exception as exc:
                rec.update(failed=True, failure=repr(exc)[:200])
                rows.append(rec)
                continue
            rec.update(failed=False,
                       n_clusters=int(len(set(lab_b.tolist()) - {-1})),
                       noise_fraction=float((lab_b == -1).mean()),
                       per_category={})
            for c in ids:
                members = set(take[labels[take] == c].tolist())
                if not members:
                    rec["per_category"][str(c)] = {
                        "jaccard": None, "matched": None,
                        "status": "absent_from_subsample"}
                    continue
                j, who = best_jaccard(members, lab_b, take, return_match=True)
                rec["per_category"][str(c)] = {
                    "jaccard": j, "matched": who,
                    "status": "ok" if who is not None else "no_overlapping_cluster"}
            rows.append(rec)
            if (b + 1) % 10 == 0:
                log(f"  resampling replicate {b + 1}/{n_replicates}")
        return rows

    return _cached_experiment(cfg, "resampling_stability_mc", identity, _compute)


# ------------------------------------------------- alternative geometry ------
def alternative_geometry_mc100(cfg, embedding, labels, n_replicates=GEOMETRY_N):
    """Re-cluster a per-category sample under Ward/Euclidean. 100 replicates.

    Historically Axis 4. A different geometry and a different linkage from the
    canonical solution, at a fixed per-category sample size so the statistic is
    comparable across categories of very different size.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)
    ids = _category_ids(labels)
    k = len(ids)
    identity = _identity(embedding, labels, "alternative_geometry", n_replicates,
                         per_category=GEOMETRY_PER_CATEGORY, k=k)

    def _compute():
        rng = np.random.default_rng(SEED)
        rows = []
        for b in range(n_replicates):
            picks = []
            for c in ids:
                pool = np.nonzero(labels == c)[0]
                picks.extend(rng.choice(pool, min(GEOMETRY_PER_CATEGORY,
                                                  len(pool)),
                                        replace=False).tolist())
            sub = np.array(sorted(picks))
            rec = {"replicate": b, "n_sampled": int(len(sub)), "k": k}
            try:
                Z = linkage(pdist(embedding[sub], metric="euclidean"),
                            method="ward")
                ward = fcluster(Z, t=k, criterion="maxclust")
            except Exception as exc:
                rec.update(failed=True, failure=repr(exc)[:200])
                rows.append(rec)
                continue
            rec.update(failed=False, per_category={})
            for c in ids:
                members = set(sub[labels[sub] == c].tolist())
                if not members:
                    rec["per_category"][str(c)] = {
                        "jaccard": None, "matched": None,
                        "status": "absent_from_sample"}
                    continue
                j, who = best_jaccard(members, ward, sub, return_match=True)
                rec["per_category"][str(c)] = {"jaccard": j, "matched": who,
                                               "status": "ok"}
            rows.append(rec)
            if (b + 1) % 20 == 0:
                log(f"  geometry replicate {b + 1}/{n_replicates}")
        return rows

    return _cached_experiment(cfg, "alternative_geometry_mc", identity, _compute)


# ----------------------------------------------------------- aggregation -----
def run_records(rows, experiment: str) -> pd.DataFrame:
    """One row per replicate, degenerate replicates retained."""
    out = []
    for r in rows:
        rec = {"replicate": r["replicate"], "failed": r.get("failed", False)}
        for k in ("projection_seed", "n_taken", "n_sampled", "k", "n_clusters",
                  "noise_fraction"):
            if k in r:
                rec[k] = r[k]
        if "n_clusters" in r and not r.get("failed"):
            rec["degenerate"] = is_degenerate(r)
        vals = [v["jaccard"] for v in r.get("per_category", {}).values()
                if v.get("jaccard") is not None]
        rec["n_categories_scored"] = len(vals)
        rec["mean_jaccard"] = round(float(np.mean(vals)), 6) if vals else None
        out.append(rec)
    return pd.DataFrame(out)


def category_records(rows) -> pd.DataFrame:
    """One row per (category, replicate): the full persisted overlap value."""
    out = []
    for r in rows:
        for c, v in r.get("per_category", {}).items():
            out.append({"replicate": r["replicate"], "category_id": int(c),
                        "jaccard": v["jaccard"], "matched_cluster": v["matched"],
                        "status": v["status"]})
    return pd.DataFrame(out)


def category_summary(rows, category_name=None,
                     thresholds=THRESHOLDS) -> pd.DataFrame:
    """Per-category conditional summary over the replicates that scored it."""
    category_name = category_name or {}
    per = {}
    for r in rows:
        if r.get("failed") or ("n_clusters" in r and is_degenerate(r)):
            continue
        for c, v in r.get("per_category", {}).items():
            if v.get("jaccard") is not None:
                per.setdefault(int(c), []).append(float(v["jaccard"]))
    out = []
    for c in sorted(per):
        v = np.asarray(per[c])
        rec = {"category_id": c, "category_label": category_name.get(c, str(c)),
               "n_replicates_scored": len(v),
               "mean_jaccard": round(float(v.mean()), 6),
               "median_jaccard": round(float(np.median(v)), 6),
               "sd": round(float(v.std(ddof=1)), 6) if len(v) > 1 else 0.0,
               "min": round(float(v.min()), 6), "max": round(float(v.max()), 6)}
        for t in thresholds:
            rec[f"at_or_above_{t:.2f}"] = bool(rec["mean_jaccard"] >= t)
        out.append(rec)
    return pd.DataFrame(out)


def experiment_summary(rows, cat_summary: pd.DataFrame, experiment: str,
                       historical_prefix: int) -> dict:
    """Global counts and the conditional headline, degeneracy always reported."""
    n = len(rows)
    failed = sum(1 for r in rows if r.get("failed"))
    degenerate = sum(1 for r in rows
                     if not r.get("failed") and "n_clusters" in r
                     and is_degenerate(r))
    viable = n - failed - degenerate
    med = (round(float(cat_summary.mean_jaccard.median()), 6)
           if len(cat_summary) else None)
    col = f"at_or_above_{CANONICAL_THRESHOLD:.2f}"
    return {"experiment": experiment, "protocol": PROTOCOL,
            "n_replicates": n, "n_viable": viable, "n_degenerate": degenerate,
            "n_failed": failed,
            "historical_prefix": historical_prefix,
            "prefix_is_exact": True,
            "median_category_mean_jaccard": med,
            "categories_at_or_above_canonical_threshold":
                int(cat_summary[col].sum()) if col in cat_summary else None,
            "n_categories": int(len(cat_summary)),
            "canonical_threshold": CANONICAL_THRESHOLD,
            "interpretation": "conditional on the recorded software environment; "
                              "the shorter historical roster is an exact prefix "
                              "of this one, so the two are a convergence "
                              "question rather than independent estimates"}


def convergence(rows, checkpoints, category_name=None) -> pd.DataFrame:
    """Headline statistic recomputed at each pre-specified checkpoint."""
    out = []
    for n in checkpoints:
        sub = [r for r in rows if r["replicate"] < n]
        if not sub:
            continue
        cs = category_summary(sub, category_name)
        out.append({"n_replicates": n, "n_categories": int(len(cs)),
                    "median_category_mean_jaccard":
                        round(float(cs.mean_jaccard.median()), 6) if len(cs) else None,
                    f"at_or_above_{CANONICAL_THRESHOLD:.2f}":
                        int(cs[f"at_or_above_{CANONICAL_THRESHOLD:.2f}"].sum())
                        if len(cs) else 0})
    return pd.DataFrame(out)
