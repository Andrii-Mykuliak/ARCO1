"""Run configuration, paths, and the from-scratch / prepared-data switch.

Everything downstream reads its knobs from a single ``Config`` instance so the
orchestrator notebook has exactly one place to change behaviour.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- corpus -----------------------------------------------------------
    corpus_name: str = "iracing"
    sentences_path: Path = RELEASE_ROOT / "data" / "iracing" / "sentences.jsonl"
    gazetteer_path: Path = RELEASE_ROOT / "data" / "iracing" / "gazetteer.json"
    text_field: str = "text"
    # field that groups sentences into "sessions" (races, episodes, documents).
    # Used by the held-out axis, the coverage split and the shuffle null.
    session_field: str = "video_id"
    min_tokens: int = 4          # set 0 for corpora already length-filtered
    #: Optional archived embedding (.npy, one row per corpus sentence, in
    #: order). When set, encoding is skipped entirely -- the only way to
    #: reproduce a partition produced under different library versions.
    embeddings_path: Path | None = None
    #: Optional archived UMAP projection (.npy). UMAP output is version-
    #: sensitive, so reproducing an archived partition needs this as well as
    #: the embedding: identical vectors still refit to a different manifold.
    umap_path: Path | None = None

    # ---- optional second corpus for the coverage analysis -----------------
    # If None, a session-level split of the primary corpus stands in for the
    # "different register" held-out set (see coverage.py).
    heldout_sentences_path: Path | None = None

    # ---- masking ----------------------------------------------------------
    apply_masking: bool = True
    use_ner_fallback: bool = False  # spaCy NER on top of the gazetteer
    entity_classes: tuple = ("PERSON", "TEAM", "PLACE")

    # ---- embedding --------------------------------------------------------
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256

    # ---- dimensionality reduction + clustering ----------------------------
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_n_components: int = 5
    umap_metric: str = "cosine"
    min_cluster_size: int = 35
    hdbscan_metric: str = "euclidean"
    hdbscan_selection: str = "eom"
    mcs_sweep: tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 90, 120)

    # ---- R3 vocabulary enrichment ----------------------------------------
    r3_min_df: int = 3
    r3_n_permutations: int = 1000
    r3_fdr_q: float = 0.05
    r3_ngram_max: int = 2
    r3_max_features: int = 6000

    # ---- Axis 3 (held-out race replication) -------------------------------
    # None = derive from the corpus. The F1 presets pin the published values;
    # any other corpus derives a split and granularity from its own scale.
    # Axis 3 is a full-protocol stage. Absent from a config -> enabled, so
    # existing research configurations never silently stop running it.
    axis3_enabled: bool = True
    axis3_n_train_sessions: int | None = None      # None -> ~70% of sessions
    axis3_train_min_cluster_size: int | None = None  # None -> min_cluster_size
    axis3_expect_clusters: int | None = None       # None -> no reference pinning
    axis3_expect_clustered: int | None = None

    # ---- validation -------------------------------------------------------
    bootstrap_n: int = 30
    bootstrap_frac: float = 0.80
    axis4_n_replicates: int = 10
    axis4_per_cluster: int = 15
    heldout_train_frac: float = 0.70
    recovery_jaccard: float = 0.30      # axis 1 "recovered" bar
    axis_pass_threshold: float = 0.50   # axes 2/3/4 criterion
    axis1_majority: float = 0.50        # fraction of alternative pipelines
    # Encoder-swap judges for Axis 1. Each re-embeds the corpus, so this is
    # the expensive part of the axis; set to () to skip encoder variation.
    axis1_encoders: tuple = (
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base",
    )
    # Axis 3 direction filter: share of the cluster's held-out sentences that
    # must route to a single dominant train cluster (paper: rep_dir).
    axis3_direction_threshold: float = 0.50

    # ---- domains ----------------------------------------------------------
    n_domains: int = 4
    network_cosine: float = 0.50

    # ---- coverage ---------------------------------------------------------
    coverage_tau: float = 0.40
    coverage_tau_sweep: tuple = (0.30, 0.35, 0.40, 0.45, 0.50)
    null_n_permutations: int = 1000
    # Granularity for re-clustering the residual. None = scale to the
    # residual size (the primary value is usually far too coarse for it).
    leftover_min_cluster_size: int | None = None
    # Fraction of sessions held out when no separate corpus is supplied.
    holdout_session_frac: float = 0.5

    # ---- execution --------------------------------------------------------
    seed: int = 42
    # "prepared" reuses anything already cached in results/cache;
    # "scratch" recomputes every stage and overwrites the cache.
    mode: str = "prepared"
    # Trim the corpus for a fast smoke run. None = use everything.
    limit_sentences: int | None = None

    # ---- paths ------------------------------------------------------------
    results_dir: Path = None   # defaults to results/<corpus_name>

    def __post_init__(self):
        self.sentences_path = Path(self.sentences_path)
        self.embeddings_path = (Path(self.embeddings_path)
                                if self.embeddings_path else None)
        self.umap_path = Path(self.umap_path) if self.umap_path else None
        self.gazetteer_path = Path(self.gazetteer_path) if self.gazetteer_path else None
        self.results_dir = (Path(self.results_dir) if self.results_dir
                            else RELEASE_ROOT / "results" / self.corpus_name)
        for d in (self.cache_dir, self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- derived paths ------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.results_dir / "cache"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def from_scratch(self) -> bool:
        return self.mode == "scratch"

    def cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{self.corpus_name}__{name}"

    # -- housekeeping -------------------------------------------------------
    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def summary(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        return d

    def save(self, path: Path | None = None):
        path = Path(path) if path else RELEASE_ROOT / "run_config.json"
        path.write_text(json.dumps(self.summary(), indent=2, default=str), encoding="utf-8")
        return path

# ---------------------------------------------------------------- corpora --
#: The three corpora this kit is set up for. Only iRacing ships: the two F1
#: corpora are broadcast-copyright, so their data/ folders hold a README and
#: nothing else. Selecting one of them without supplying your own licensed copy
#: fails loudly at load rather than silently producing an empty run.
CORPORA = {
    "iracing": dict(
        corpus_name="iracing",
        sentences_path=RELEASE_ROOT / "data" / "iracing" / "sentences.jsonl",
        gazetteer_path=RELEASE_ROOT / "data" / "iracing" / "gazetteer.json",
        session_field="video_id",
        apply_masking=True,
        min_cluster_size=35,
        # Eight sessions give a data-starved session-level split; the stage
        # ships off by default and can be switched on to inspect the protocol.
        axis3_enabled=False,
    ),
    "f1_highlights": dict(
        corpus_name="f1_highlights",
        min_tokens=0,                 # already length-filtered at construction
        sentences_path=RELEASE_ROOT / "data" / "f1_highlights" / "sentences.jsonl",
        gazetteer_path=None,
        session_field="race_id",
        apply_masking=False,          # that corpus ships pre-masked
        min_cluster_size=35,
        # Published Axis-3 configuration for this corpus; see Section 4.4.
        axis3_n_train_sessions=80,
        axis3_train_min_cluster_size=30,
        axis3_expect_clusters=34,
        axis3_expect_clustered=5084,
    ),
    "f1_full": dict(
        corpus_name="f1_full",
        min_tokens=0,                 # already length-filtered at construction
        sentences_path=RELEASE_ROOT / "data" / "f1_full" / "sentences.jsonl",
        gazetteer_path=None,
        session_field="race_id",
        apply_masking=False,
        min_cluster_size=35,
    ),
}


def for_corpus(name: str = "iracing", **overrides) -> "Config":
    """Build a Config for one of the three corpora; iRacing is the default."""
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    cfg = Config(**{**CORPORA[name], **overrides})
    if not cfg.sentences_path.exists():
        raise FileNotFoundError(
            f"corpus {name!r} is not present at {cfg.sentences_path}. "
            "The F1 corpora are broadcast-copyright and are not redistributed; "
            "supply your own licensed copy at that path, or use corpus 'iracing'."
        )
    return cfg
