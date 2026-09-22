"""Reproduction pipeline for the cross-register commentary structure study.

Flat, one-concern-per-module layout:

    config                run configuration, paths, scratch/prepared switch
    io_utils              caching, logging, small numeric helpers
    corpus                loading + entity masking (masking is part of the input)
    embedding             sentence encoding
    clustering            UMAP + HDBSCAN + granularity sweep
    enrichment            lexical enrichment (exact hypergeometric + global BH)
    labeling              corpus-agnostic cluster identity (handles + keywords)
    diagnostics           complementary diagnostic dimensions over categories
    heldout_replication   held-out race replication
    domains               Ward + Louvain grouping over centroids
    sensitivity           settings explored + extended robustness experiments
    cross_register        independent full-race structure and its correspondence
                          with the highlight categories
    paired_register       matched same-event highlight/full-race comparison
    ablation              masking and default-pipeline ablations
    reporting             manuscript tables, figures and the summary JSON

Terminology note. `diagnostics` reports COMPLEMENTARY DIAGNOSTIC DIMENSIONS and
a category-level diagnostic profile. It deliberately does not compute an
aggregate pass/fail vote or a verdict band: support is heterogeneous across
categories and dimensions, and collapsing it to a single count misrepresents it.
"""
from . import (ablation, analytical_figures, analytical_tables,  # noqa: F401
               monte_carlo_diagnostics, heldout_factorial,
               clustering, config, corpus,  # noqa: F401
               cross_register, diagnostics, domains, embedding, enrichment,
               frozen_f1, heldout_replication, io_utils,
               labeling, paired_register, reporting, sensitivity)
from .config import CORPORA, Config, for_corpus  # noqa: F401

__all__ = [
    "Config", "CORPORA", "for_corpus",
    "config", "io_utils", "corpus", "embedding", "clustering", "enrichment",
    "labeling", "diagnostics", "heldout_replication", "domains",
    "cross_register", "paired_register", "frozen_f1",
    "ablation", "reporting", "sensitivity", "analytical_figures", "analytical_tables", "monte_carlo_diagnostics", "heldout_factorial",
]


# ---------------------------------------------------------------- aliases --
# Deprecated module names stay importable for one release so external scripts
# do not break. All new code and documentation use the current names.
import sys as _sys  # noqa: E402

DEPRECATED_MODULE_ALIASES = {
    "validation": "diagnostics",
    "axis3": "heldout_replication",
    "coverage": "cross_register",
}

validation = diagnostics
axis3 = heldout_replication
coverage = cross_register
for _old, _new in DEPRECATED_MODULE_ALIASES.items():
    _sys.modules[f"{__name__}.{_old}"] = globals()[_new]
