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
