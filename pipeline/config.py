"""Run configuration, paths, and the from-scratch / prepared-data switch.

Everything downstream reads its knobs from a single ``Config`` instance so the
orchestrator notebook has exactly one place to change behaviour.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import frozen_f1 as _FROZEN

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
    # "different register" held-out set (see cross_register.py).
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

    # ---- held-out race replication -------------------------------
    # None = derive from the corpus. The F1 presets pin the published values;
    # any other corpus derives a split and granularity from its own scale.
    # Held-out replication is a full-protocol stage. Absent from a config -> enabled, so
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
    # Encoder-swap judges for alternative-model recovery. Each re-embeds the corpus, so this is
    # the expensive part of the axis; set to () to skip encoder variation.
    axis1_encoders: tuple = (
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base",
    )
    # Held-out direction filter: share of the cluster's held-out sentences that
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
    # Where derived intermediates live. Defaults to results_dir/cache. Set it to
    # reuse an archived realisation while writing tables and figures elsewhere:
    # UMAP is version-sensitive, so reproducing an archived partition needs the
    # archived projection, and that lives in a cache directory.
    cache_dir_override: Path = None

    def __post_init__(self):
        self.sentences_path = Path(self.sentences_path)
        self.embeddings_path = (Path(self.embeddings_path)
                                if self.embeddings_path else None)
        self.umap_path = Path(self.umap_path) if self.umap_path else None
        self.gazetteer_path = Path(self.gazetteer_path) if self.gazetteer_path else None
        self.results_dir = (Path(self.results_dir) if self.results_dir
                            else RELEASE_ROOT / "results" / self.corpus_name)
        self.cache_dir_override = (Path(self.cache_dir_override)
                                   if self.cache_dir_override else None)
        for d in (self.cache_dir, self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- derived paths ------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.cache_dir_override or self.results_dir / "cache"

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
        # train-split granularity for held-out replication ONLY.
        # The canonical F1 highlight value is min_cluster_size=35;
        # 30 is the fixed train-split setting for this stage and is
        # not a canonical or corpus-size-derived value.
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
        # Selected from this corpus's OWN sweep, not scaled from the highlight
        # corpus. cross_register.reconstruct_second_register re-derives it; the
        # value here is the frozen outcome of that selection.
        min_cluster_size=40,
    ),
}


# --------------------------------------------------------------- the runs --
#: The two configurations the reproduction notebook offers. Assigning one of
#: them to CONFIG in the notebook selects the run; everything they do not state
#: comes from CORPORA and the Config defaults above. This is the single place
#: where a run is defined.
IRACING = {
    "name": "iracing",
    "description": "redistributable one-register demonstration corpus",
    "primary_corpus": "iracing",
    "secondary_corpus": None,
    # where tables and figures are written, under results/
    "results_root": "iracing",
    "secondary_results_root": None,
    "cross_register_results": "iracing_cross_register_run",
    # where derived intermediates live; None keeps them beside the results
    "cache_root": None,
    "secondary_cache_root": None,
    # analysis inputs tied to one particular partition; never inferred
    "matched_events": {},
    "category_names": {},
    "category_domains": {},
    # the Monte-Carlo and factorial experiments are defined on the frozen
    # Formula 1 realisations and do not transfer to another corpus
    "extended_experiments": False,
}

F1 = {
    "name": "f1",
    "description": "complete two-register Formula 1 workflow",
    "primary_corpus": "f1_highlights",
    "secondary_corpus": "f1_full",
    "results_root": "f1_run",
    "secondary_results_root": "f1_full_run",
    "cross_register_results": "f1_cross_register_run",
    # the archived caches: the manifold projection is version-sensitive, so
    # reproducing the published partition means reusing the archived
    # realisation rather than refitting it
    "cache_root": "f1_highlights",
    "secondary_cache_root": "f1_full",
    "matched_events": _FROZEN.MATCHED_EVENTS,
    "category_names": _FROZEN.CATEGORY_NAME,
    "category_domains": _FROZEN.CATEGORY_DOMAIN,
    "extended_experiments": True,
}

# --------------------------------------------------- provenance of the run --
#: Software stack the reported Formula 1 analysis ran under. The resampling and
#: alternative-model dimensions are version-sensitive, so this is reported with
#: the results rather than treated as an incidental detail.
REPORTED_ENVIRONMENT = {
    "python": "3.13.9",
    "numpy": "2.3.5",
    "scipy": "1.16.3",
    "scikit_learn": "1.7.2",
    "umap_learn": "0.5.12",
    "hdbscan": "0.8.43",
    "sentence_transformers": "5.5.1",
    "torch": "2.12.0",
}

#: Why `Config.min_samples` is absent: hdbscan resolves an unset min_samples to
#: min_cluster_size, so every granularity sweep is a coupled granularity and
#: density sweep. Passing 35 explicitly is bit-identical to omitting it.
MIN_SAMPLES_NOTE = (
    "left unset, which hdbscan resolves to min_cluster_size; every canonical "
    "run omits it, so granularity sweeps are coupled granularity+density "
    "sweeps. Verified: an explicit 35 is bit-identical to the omission."
)


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


# ------------------------------------------------- notebook configuration --
def resolve_corpora(configuration: dict, release_root: Path):
    """Build the primary and optional second-register Config from a preset.

    Availability is read from the configured path on disk, never inferred from a
    corpus name. A named-but-absent corpus stops the run: nothing is substituted
    for it, and a two-register configuration is never downgraded to one register.
    """
    release_root = Path(release_root)

    def path_of(name):
        return CORPORA.get(name, {}).get("sentences_path")

    def available(name):
        p = path_of(name)
        return bool(name) and bool(p) and Path(p).exists()

    primary = configuration["primary_corpus"]
    secondary = configuration["secondary_corpus"]

    if not available(primary):
        raise SystemExit(
            f"configuration error: the primary corpus {primary!r} is not present "
            f"at its configured path ({path_of(primary)}). This run is NOT "
            f"switched to another corpus: supply the corpus at that path, or "
            f"select a configuration whose corpora are available.")
    if secondary is not None and not available(secondary):
        raise SystemExit(
            f"configuration error: this is a two-register configuration, so the "
            f"second register ({secondary!r}) is required but is not available "
            f"at the configured path ({path_of(secondary)}). This run is NOT "
            f"downgraded to one-register mode and does not fall back to "
            f"precomputed cross-register outputs: supply that corpus, or select "
            f"the one-register configuration.")

    def cache_of(root):
        return (release_root / "results" / root / "cache") if root else None

    cfg = for_corpus(
        primary,
        results_dir=release_root / "results" / configuration["results_root"],
        cache_dir_override=cache_of(configuration["cache_root"]))
    cfg2 = None
    if secondary is not None:
        cfg2 = for_corpus(
            secondary,
            results_dir=(release_root / "results"
                         / configuration["secondary_results_root"]),
            cache_dir_override=cache_of(configuration["secondary_cache_root"]))
    return cfg, cfg2


def describe(configuration: dict, cfg: "Config", cfg2: "Config | None" = None,
             release_root: Path | None = None) -> str:
    """The settings that define the reported experiment, as printable text.

    Where a lightweight single pass and a larger reported experiment both exist,
    both are named: the reported figures come from the replicate experiments,
    not from the single pass.
    """
    from . import heldout_factorial as HF
    from . import monte_carlo_diagnostics as MC

    rel = Path(release_root) if release_root else RELEASE_ROOT
    extended = bool(configuration["extended_experiments"])

    def short(p):
        p = Path(p)
        try:
            return str(p.relative_to(rel))
        except ValueError:
            return str(p)

    lines = [
        f"configuration        {configuration['name']} - "
        f"{configuration['description']}",
        f"primary corpus       {cfg.corpus_name}  ({short(cfg.sentences_path)})",
        f"second register      "
        + (f"{cfg2.corpus_name}  ({short(cfg2.sentences_path)})" if cfg2
           else "none (one-register run)"),
        f"results              {short(cfg.results_dir)}",
        f"cache                {short(cfg.cache_dir)}",
        "",
        f"entity masking       "
        + ("applied by the pipeline" if cfg.apply_masking
           else "already applied when the corpus was built"),
        f"encoder              {cfg.encoder}",
        f"projection           UMAP {cfg.umap_n_components}d, "
        f"{cfg.umap_n_neighbors} neighbours, min_dist {cfg.umap_min_dist}, "
        f"{cfg.umap_metric}",
        f"clustering           HDBSCAN {cfg.hdbscan_metric}, "
        f"{cfg.hdbscan_selection} selection",
        f"min_cluster_size     {cfg.min_cluster_size}  "
        f"(swept over {list(cfg.mcs_sweep)})",
        f"random seed          {cfg.seed}",
        "",
        f"lexical enrichment   exact hypergeometric, global BH at q="
        f"{cfg.r3_fdr_q}, min_df {cfg.r3_min_df}, up to "
        f"{cfg.r3_ngram_max}-grams",
        f"resampling           "
        + (f"{MC.RESAMPLE_N} replicates at {MC.RESAMPLE_FRACTION:.0%} of the "
           f"corpus (reported); a {cfg.bootstrap_n}-replicate single pass is "
           f"also computed but not reported"
           if extended
           else f"{cfg.bootstrap_n} replicates at {cfg.bootstrap_frac:.0%} of "
                f"the corpus; the reported {MC.RESAMPLE_N}-replicate "
                f"experiment is defined on the frozen realisations only"),
        f"alternative geometry "
        + (f"{MC.GEOMETRY_N} replicates, {MC.GEOMETRY_PER_CATEGORY} sentences "
           f"per category (reported); a {cfg.axis4_n_replicates}-replicate "
           f"single pass is also computed but not reported"
           if extended
           else f"{cfg.axis4_n_replicates} replicates, "
                f"{cfg.axis4_per_cluster} sentences per category; the reported "
                f"{MC.GEOMETRY_N}-replicate experiment is defined on the "
                f"frozen realisations only"),
        f"held-out factorial   "
        + (f"{len(HF.SEEDS)} splits x 2 embedding realisations x "
           f"{len(HF.MCS)} training granularities {list(HF.MCS)} "
           f"= {len(HF.SEEDS) * 2 * len(HF.MCS)} runs"
           if extended else "defined on the frozen realisations only"),
        f"held-out replication "
        + (f"{cfg.axis3_n_train_sessions} training sessions"
           if cfg.axis3_n_train_sessions
           else f"{cfg.heldout_train_frac:.0%} of sessions, derived from this "
                f"corpus")
        + f", trained at min_cluster_size "
          f"{cfg.axis3_train_min_cluster_size or cfg.min_cluster_size}",
        f"correspondence null  {cfg.null_n_permutations} size-preserving "
        f"permutations",
        f"assignment threshold {cfg.coverage_tau}  "
        f"(swept over {list(cfg.coverage_tau_sweep)})",
        f"matched events       {len(configuration['matched_events'])} configured",
        f"extended experiments "
        + ("resampling, held-out factorial, alternative geometry, "
           "domain membership" if extended
           else "not defined for this corpus"),
    ]
    return "\n".join(lines)
