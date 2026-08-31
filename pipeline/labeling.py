"""Corpus-agnostic cluster identity.

The published paper names its clusters ("Overtake battles", "Pit stops", ...).
Those names are properties of *that* corpus and must not be baked into a
reproduction kit: a different corpus yields different clusters. So identity here
is derived, never asserted —

  * a stable **handle**  ``K07``  (index, ordered by size);
  * a **keyword signature** from the R3 enriched terms;
  * three **exemplar sentences** nearest the centroid.

An optional external label map can be supplied to attach human names *on top*
of the derived identity, for the case where you are re-running the original
corpus and want the paper's names back.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .enrichment import top_terms
from .io_utils import log


def _dedupe_keywords(terms, max_k=4):
    """Drop terms subsumed by an already-chosen one ('grand prix' vs 'prix')."""
    chosen = []
    for t in terms:
        if len(chosen) >= max_k:
            break
        if any(t in c or c in t for c in chosen):
            continue
        chosen.append(t)
    return chosen


def keyword_signature(enr: pd.DataFrame, cluster: int, max_k: int = 4) -> str:
    kws = _dedupe_keywords(top_terms(enr, cluster, n=12), max_k=max_k)
    return " / ".join(kws) if kws else "(no enriched terms)"


def exemplars(X: np.ndarray, labels: np.ndarray, texts, cluster: int, n: int = 3):
    """The n sentences nearest the centroid, with their corpus positions."""
    idx = np.nonzero(labels == cluster)[0]
    if not len(idx):
        return [], []
    c = X[idx].mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    sims = X[idx] @ c
    order = idx[np.argsort(-sims)][:n]
    return [str(texts[i]) for i in order], list(order)


def build_cluster_cards(cfg, X, labels, texts, enr, label_map: dict | None = None,
                        sources=None):
    """One row per cluster: handle, size, keyword signature, exemplars.

    ``sources`` is an optional per-sentence origin label; when given, every
    exemplar is shown next to the recording it was taken from, so no quoted
    sentence appears without its source.
    """
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    order = sorted(ids, key=lambda c: -int((labels == c).sum()))
    handle = {c: f"K{i:02d}" for i, c in enumerate(order)}

    clustered = int((labels != -1).sum())
    rows = []
    for c in order:
        ex, pos = exemplars(X, labels, texts, c, n=3)
        row = {
            "handle": handle[c],
            "cluster": c,
            "size": int((labels == c).sum()),
            "share_of_clustered": int((labels == c).sum()) / max(clustered, 1),
            "keywords": keyword_signature(enr, c),
            "n_enriched": int((enr["cluster"] == c).sum()) if len(enr) else 0,
            "external_label": (label_map or {}).get(str(c), ""),
        }
        for i in range(3):
            row[f"exemplar_{i + 1}"] = ex[i] if len(ex) > i else ""
            if sources is not None:
                row[f"exemplar_{i + 1}_source"] = (
                    str(sources[pos[i]]) if len(pos) > i else "")
        rows.append(row)
    df = pd.DataFrame(rows)
    log(f"cards: {len(df)} clusters, top-5 hold "
        f"{df.nlargest(5, 'size')['share_of_clustered'].sum():.1%} of clustered sentences")
    return df


def load_label_map(path: str | Path | None) -> dict | None:
    """Optional {cluster_id: human name} JSON. Absent by design for new corpora."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log(f"label map not found ({p.name}) — using derived keyword identity only")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def display_name(row) -> str:
    """Handle + keywords, with the external name appended when available."""
    base = f"{row['handle']} · {row['keywords']}"
    return f"{base}  [{row['external_label']}]" if row.get("external_label") else base
