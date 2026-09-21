"""Sentence embedding (cached)."""
from __future__ import annotations

import numpy as np

from .io_utils import cached, l2_normalise, log


def embed(cfg, texts, tag: str = "emb") -> np.ndarray:
    """Encode ``texts`` with the configured sentence-transformer, L2-normalised."""

    path = getattr(cfg, "embeddings_path", None)
    if path is not None and tag == "emb":
        X = np.load(path)
        if len(X) != len(list(texts)):
            raise ValueError(
                f"archived embedding has {len(X)} rows but the corpus has "
                f"{len(list(texts))} sentences; they must correspond row-for-row")
        log(f"embedding: loaded archived {path.name} {X.shape} (no encoding)")
        return l2_normalise(X)

    def _compute():
        from sentence_transformers import SentenceTransformer
        log(f"loading encoder {cfg.encoder}")
        model = SentenceTransformer(cfg.encoder)
        X = model.encode(
            list(texts),
            batch_size=cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return l2_normalise(np.asarray(X, dtype=np.float32))

    slug = cfg.encoder.split("/")[-1].replace("-", "_")
    X = cached(cfg, f"{tag}__{slug}.npy", _compute)
    log(f"embeddings: {X.shape}")
    return X


def embed_with(model_name: str, texts, batch_size: int = 256) -> np.ndarray:
    """Ad-hoc encode with an arbitrary model (used by the algorithm axis)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    X = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                     convert_to_numpy=True)
    return l2_normalise(np.asarray(X, dtype=np.float32))


def audit(A, what: str) -> dict:
    """Shape, dtype and numerical integrity of a sentence-by-feature matrix."""
    import numpy as np

    A = np.asarray(A)
    finite = np.isfinite(A)
    rep = {"what": what, "shape": list(A.shape), "dtype": str(A.dtype),
           "n_nan": int(np.isnan(A).sum()), "n_inf": int(np.isinf(A).sum()),
           "rows_all_finite": int(finite.all(axis=1).sum()),
           "rows_any_nonfinite": int((~finite.all(axis=1)).sum())}
    if rep["rows_all_finite"]:
        ok = A[finite.all(axis=1)]
        rep["min"] = round(float(ok.min()), 6)
        rep["max"] = round(float(ok.max()), 6)
        rep["mean_row_norm"] = round(float(np.linalg.norm(ok, axis=1).mean()), 6)
    print(f"{what}: shape={rep['shape']} dtype={rep['dtype']} "
          f"NaN={rep['n_nan']:,} Inf={rep['n_inf']:,} "
          f"finite rows={rep['rows_all_finite']:,}/{A.shape[0]:,}")
    return rep
