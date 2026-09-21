"""Caching, logging and small shared helpers.

The cache is what makes the ``prepared`` / ``scratch`` switch work: every
expensive stage writes one file under ``results/cache`` and reads it back
unless the run mode says otherwise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# Configuration a cached artefact depends on. A cache directory records these
# once; a later run whose values differ is refusing to reuse it, because the
# filename alone cannot tell a re-masked or re-embedded corpus from the original.
CACHE_IDENTITY_FIELDS = (
    "corpus_name", "sentences_path", "apply_masking", "entity_classes",
    "gazetteer_path", "min_tokens", "encoder", "umap_n_neighbors",
    "umap_min_dist", "umap_n_components", "umap_metric", "hdbscan_metric",
    "hdbscan_selection", "seed")


def cache_identity(cfg) -> dict:
    out = {}
    for f in CACHE_IDENTITY_FIELDS:
        v = getattr(cfg, f, None)
        out[f] = (Path(v).name if isinstance(v, Path)
                  else list(v) if isinstance(v, tuple) else v)
    return out


def check_cache_identity(cfg):
    """Refuse a cache built under a different scientific configuration."""
    path = Path(cfg.cache_dir) / "_cache_identity.json"
    now = cache_identity(cfg)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=1), encoding="utf-8")
        return
    was = json.loads(path.read_text(encoding="utf-8"))
    drift = {k: (was.get(k), now[k]) for k in now if was.get(k) != now[k]}
    if drift:
        raise ValueError(
            f"the cache in {path.parent} was built under a different "
            f"configuration and cannot be reused: "
            + "; ".join(f"{k} was {a!r}, is now {b!r}" for k, (a, b) in drift.items())
            + ". Point the configuration at a different cache directory, or "
              "remove that one so the artefacts are recomputed.")


def cached(cfg, name: str, compute):
    """Return cached artefact, or compute + persist it.

    Supported by extension: ``.parquet`` (DataFrame), ``.npy`` (array),
    ``.csv`` (DataFrame), ``.json`` (dict/list).
    """
    check_cache_identity(cfg)
    path = cfg.cache_path(name)
    if path.exists() and not cfg.from_scratch:
        log(f"cache hit  {path.name}")
        return _load(path)
    log(f"computing  {path.name}")
    obj = compute()
    _save(path, obj)
    return obj


def _load(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"no loader for {path.suffix}")


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        obj.to_parquet(path, index=False)
    elif path.suffix == ".npy":
        np.save(path, obj)
    elif path.suffix == ".csv":
        obj.to_csv(path, index=False)
    elif path.suffix == ".json":
        import json
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    else:
        raise ValueError(f"no writer for {path.suffix}")


def save_table(cfg, df: pd.DataFrame, name: str) -> Path:
    """Persist a reportable table under results/tables and return its path."""
    path = cfg.tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    log(f"table   -> {path.name}  ({len(df)} rows)")
    return path


def save_figure(cfg, fig, name: str) -> Path:
    path = cfg.figures_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log(f"figure  -> {path.name}")
    return path


def l2_normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_jaccard(members: set, other_labels: np.ndarray, index: np.ndarray,
                 return_match: bool = False):
    """Best Jaccard of ``members`` against any cluster in ``other_labels``.

    ``return_match`` additionally reports which cluster achieved it, which the
    Monte-Carlo diagnostics persist per (category, replicate).
    """
    best, who = 0.0, None
    for lab in np.unique(other_labels):
        if lab == -1:
            continue
        cand = set(index[other_labels == lab].tolist())
        j = jaccard(members, cand)
        if j > best:
            best, who = j, int(lab)
    return (best, who) if return_match else best
