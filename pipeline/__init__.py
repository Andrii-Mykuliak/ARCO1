"""Reproduction pipeline for the event-grounded commentary taxonomy study.

Flat, one-concern-per-module layout:

    config      run configuration, paths, scratch/prepared switch
    io_utils    caching, logging, small numeric helpers
    corpus      loading + entity masking (masking is part of the input)
    embedding   sentence encoding
    clustering  UMAP + HDBSCAN + granularity sweep
    enrichment  R3 permutation test + FDR
    labeling    corpus-agnostic cluster identity (handles + keywords)
    validation  four-axis convergence protocol
    domains     Ward + Louvain grouping over centroids
    coverage    held-out coverage, leftover partition, shuffle null
    ablation    masking and cross-encoder ablations
    reporting   figures and display helpers
"""
from . import (ablation, axis3, clustering, config, corpus, coverage, domains,
               embedding, enrichment, io_utils, labeling, reporting,
               validation)
from .config import CORPORA, Config, for_corpus

__all__ = ["Config", "config", "io_utils", "corpus", "embedding", "clustering",
           "enrichment", "labeling", "validation", "domains", "coverage",
           "ablation", "reporting"]
