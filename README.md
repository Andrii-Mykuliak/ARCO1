# Method kit — event-grounded commentary taxonomy

Runs the computational analysis pipeline and principal validation stages used in
the study, end to end, from a single orchestrator notebook. The human-rater study
is documented in the accompanying materials and is not re-administered here.

```bash
pip install -r requirements.txt
python tests/test_scratch_smoke.py     # verify the cold-cache path (~8 min cold)
jupyter lab 00_orchestrator.ipynb      # Run All
```

The smoke test executes every pipeline stage on a reduced slice of the
redistributable corpus and finishes with:

```
45/45 stages passed
EXIT_FAILURES=0
```

Exit code is the number of failed stages, so it can gate CI. Tested with Python 3.13.9 (scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3,
scipy 1.16.3, umap-learn 0.5.12). On a cold cache the first run additionally downloads three alternative
sentence encoders (MPNet, BGE, GTE) for the Axis-1 comparison, so allow
noticeably longer than a warm re-run.

Versions in `requirements.txt` are **pinned exactly**, not bounded. The pipeline's
output is genuinely version-sensitive — UMAP especially, where the same corpus,
seed and parameters can give a different embedding across releases and a
different cluster count downstream.

## What the iRacing demo does and does not show

The demo runs the same computational framework on a different corpus, so its
numbers differ
from the paper's by construction. Several stages are additionally limited by
the demo corpus's size (8 sessions, 5,967 sentences).

- **Axis 3 is disabled by default** in the shipped iRacing configuration,
  because the redistributable demo contains only eight sessions and yields a
  data-starved session-level split. It remains enabled in the paper F1
  configuration. Enable it for the demo with `for_corpus(CORPUS, axis3_enabled=True)` to
  inspect and reproduce the full procedure.
  When enabled on the current demo the stage is expected to be weak: the run
  yields 1 of 31 categories meeting Axis 3.
- **With Axis 3 disabled, the default run reports a robustness profile over the
  three enabled axes** and does not claim the paper's canonical four-axis
  verdict bands. The five-band classification (strict-core / robust / moderate
  / weak / fragile) describes the full four-axis protocol only.
- **The leftover session-label shuffle null is degenerate** on the shipped
  demo: the held-out subset yields too few leftover clusters for the null to
  have variance, so that particular result is not informative. The paper's F1
  analysis operates on ten full-race broadcasts and 59 leftover clusters.
- **`min_cluster_size=35` is retained to mirror the paper configuration**, not
  because the iRacing sweep selected it. The sweep is reported as a diagnostic.
  For a new corpus, inspect that corpus's own sweep rather than assuming 35
  transfers.
- **The masking effect is corpus-dependent.** In the F1 corpus the
  masked/unmasked contrast is near-total (ARI = 0.0139); the demo shows a
  substantially smaller contrast. The demo reproduces the procedure, not the
  magnitude.
- **Domain analysis is a structural analogue.** The iRacing corpus carries no
  human-authored thematic-domain assignments, so that stage reports a
  Ward-Louvain comparison rather than the paper's domain-to-group concentration
  analysis.
- **Human components are not regenerated.** The human-rater study is not
  re-administered, and the paper's thematic labels and broad domains are
  human-authored reference artefacts that the clustering pipeline does not
  reproduce. The notebook derives corpus-specific keyword handles instead.
- **Canonical F1 results live in frozen artefacts** under
  `results/f1_highlights/` and are pinned by the regression suite in `tests/`.

## Two distinct things this repository provides

**A. Executable reproduction.** The included iRacing corpus (CC BY 3.0)
supports end-to-end execution of the computational pipeline on a fresh clone,
with no access request. Axis 3 is optional and disabled in the shipped demo
configuration; the remaining validation stages run and produce demonstration
outputs.

**B. Canonical paper artefacts.** The committed tables and figures under
`results/f1_highlights/`, together with the regression expectations in
`test_axis3_regression.py`, document the Formula 1 analysis reported in the
paper.

Reproducing the paper's Formula 1 numerical results requires the F1 corpus,
which is under broadcast copyright and is not redistributed. The immutable
configuration of that analysis is recorded in `pipeline/config.py`.

A different corpus yields different clusters, so **none of the numbers produced
here will match the paper, and they are not meant to.** What carries over is the
procedure and the shape of the findings: a granularity plateau exists and is
locatable from the data; masking measurably changes the partition; clusters
separate into robustness bands under the validation perturbations; a coverage
threshold behaves as a permissive floor; the residual is session-structured
rather than random. Cluster identity is therefore derived from each corpus's own
vocabulary rather than carried over as names from the paper.

The full paper configuration evaluates clusters under four robustness
perturbations. The shipped iRacing configuration runs three of those axes by
default, with Axis 3 available as an explicit opt-in.

Point the configuration at a corpus of your own to run the same computational
framework. Corpus-specific settings and optional validation stages should be
reviewed for the new data.

## What it does

| Stage | Produces |
|---|---|
| 1 Corpus + entity masking | corpus stats, masking coverage report, masked examples |
| 2 Embedding + granularity sweep | sweep table, plateau summary, partition stats, sweep figure |
| 3 R3 vocabulary enrichment (exact hypergeometric + global BH) | enriched (cluster, term) pairs, cluster cards, per-cluster spread, signature richness across the plateau, size concentration, size figure |
| 4 Robustness validation | per-cluster enabled-axis scores, criterion counts, full four-axis verdict bands when applicable, cumulative table, axis correlations, heatmap |
| 5 Domain structure | Ward + Louvain assignment, agreement vs Monte-Carlo chance, per-domain agreement breakdown, Ward cut profile across K, sub-partition of the largest clusters, domain keywords, dendrogram |
| 6 Coverage + leftover + null | per-session coverage, threshold sweep, threshold characterisation, leftover clusters, shuffle-null verdict, coverage + null figures |
| 7 Intrinsic quality + sensitivity | DBCV / silhouette per pipeline (with the scored fraction), leave-one-out judge sensitivity, noise-bucket profile |
| 8 Ablations | masked vs unmasked partition agreement; cross-encoder domain stability with a label-permutation test; chosen configuration vs library-default granularity, masked and unmasked |
| 9 Coverage CI | session-level bootstrap confidence interval |

### The four axes

| Axis | Perturbs | Judges / procedure |
|---|---|---|
| 1 Algorithm | encoder, clustering objective, density family | encoder swaps (MPNet, BGE, GTE) + KMeans, Ward, OPTICS and Leiden |
| 2 Data | which sentences were drawn | 30 x 80% subsamples, re-clustered |
| 3 Held-out | which sessions were drawn | session-level train/test; train-only UMAP + HDBSCAN reconstruction, Hungarian alignment, held-out approximate prediction, and lexical replication (**`rep_strict` and `rep_dir` must both be >= 0.5**) |
| 4 Hierarchical | assumed cluster geometry | Ward at fixed K over stratified subsamples drawn without replacement |

Resolution and UMAP-seed variants are reported as auxiliary sensitivity
diagnostics and do not contribute to the Axis-1 score.

Axis-1 scores are a **fraction of the judges that ran**, so a judge that fails to
construct would quietly shrink the denominator. The scored roster is declared in
`validation.paper_judges()` and checked against what actually ran: any shortfall
is printed as a warning and recorded per row as `axis1_judges_missing`,
`axis1_judges_expected` and `axis1_complete`. Leiden is the usual culprit — it needs
`leidenalg` and `python-igraph`, both pinned in `requirements.txt`.

Axis 1's encoder family is the expensive part — each judge re-embeds the whole
corpus. Set `axis1_encoders=()` in the config to drop it.

Tables land in `results/<corpus>/tables/`, figures in `results/<corpus>/figures/`,
and every intermediate in `results/<corpus>/cache/`.

### Partition agreement is always reported two ways

Any comparison between two partitions reports `ari_all` (noise treated as a
label) alongside `ari_co_clustered` (points both partitions clustered). The two
can differ several-fold when noise rates are high, and a single figure quoted
without saying which one it is cannot be interpreted.

## Three corpora, one switch

```python
CORPUS = "iracing"          # ships with the kit
CORPUS = "f1_highlights"    # needs your own licensed copy
CORPUS = "f1_full"          # needs your own licensed copy
```

Each corpus writes to its own `results/<corpus>/` tree, so runs never
overwrite one another. Selecting an F1 corpus without supplying the data
raises immediately rather than producing an empty run.

## Two run modes

```python
cfg = for_corpus(CORPUS, mode="prepared")   # reuse the cache — minutes
cfg = for_corpus(CORPUS, mode="scratch")    # recompute from the raw corpus
```

`prepared` is the default. Delete `results/<corpus>/cache/` to force a cold run without
touching the config.

For a fast smoke test set `limit_sentences=1500` and lower `bootstrap_n`,
`axis4_n_replicates`, `r3_n_permutations`, `null_n_permutations`.

## Why cluster names are absent

The paper names its clusters. Those names are properties of *that* corpus, so
baking them in would make the kit a replay rather than a reproduction. Here each
cluster carries:

* a positional **handle** — `K00`, `K01`, … ordered by size;
* a **keyword signature** derived from its own FDR-significant enriched terms;
* three **exemplar sentences** nearest the centroid.

Domains are likewise derived here (Ward cut + Louvain communities) and named by
their own aggregated keywords. The notebook derives corpus-specific keyword
handles and exemplars automatically; the thematic labels and broad-domain
assignments reported in the paper are human-authored reference artefacts and are
not regenerated by the clustering pipeline.

Running a different corpus therefore yields different clusters with honest
self-describing labels, while the protocol and the artefact set stay identical.
If you *do* want human names attached, drop a `data/label_map.json` of
`{"<cluster_id>": "Name"}` and it is shown alongside the derived identity.

## Corpora

**Shipped:** `data/iracing/` — 5,967 sentences of iRacing broadcast commentary
from 8 CC-BY 3.0 videos, with a 106-entry gazetteer for masking. Redistributable;
see `data/iracing/README.md` and `attribution.csv`.

### Every sentence says where it came from

Each row of `sentences.jsonl` carries `video_id`, `video_url`, `video_title`,
`channel`, `upload_date`, `series` and `license_spdx`. These survive the load and
travel with the sentence, so nothing quoted anywhere in the outputs is unsourced:
cluster exemplars name their recording, the noise-bucket examples name theirs,
and `01_source_provenance` gives the per-recording breakdown of the corpus with
the CC-BY attribution string each video requires.

A corpus without these fields still runs; the notebook logs that sentences will
not be traceable, and the provenance table says so rather than being silently
absent.

**Not shipped:** the F1 highlights and full-race corpora the paper reports on are
broadcast-copyright and cannot be redistributed under any licence available to
us. No request process will unlock them here. If you hold your own licensed copy,
this is the configuration the study used:

```python
cfg = Config(
    corpus_name="f1_highlights",
    sentences_path=".../corpus_113races_masked_v4.json",
    session_field="race_id",
    heldout_sentences_path=".../fullrace_sentences.jsonl",
    apply_masking=False,          # that corpus ships pre-masked
    min_cluster_size=35,
)
```

## Bringing your own corpus

Every corpus uses the same interface, so adding one is a data task rather than a
code task. Create `data/<your_corpus>/` with the file names below, register it in
`CORPORA`, and the computational framework can be run on the new corpus, with
corpus-specific settings and optional validation stages reviewed as needed.

### Required: `sentences.jsonl`

One JSON object per line. Two fields are required:

| Field | Type | Purpose |
|---|---|---|
| `text` | string | the sentence, as spoken/written |
| *session field* | string | groups sentences into recordings, races, documents |

The session field is named in the preset (`session_field`); the shipped corpus
uses `video_id`. Sessions are what Axis 3 holds out and what the coverage and
shuffle-null analyses permute, so a corpus with one session cannot support them.

Strongly recommended, so that nothing quoted in the outputs is unsourced:

| Field | Purpose |
|---|---|
| `video_url`, `video_title`, `channel`, `upload_date`, `series` | provenance shown beside every exemplar |
| `license_spdx` | the licence each sentence is used under |

These survive the load and travel with the sentence. A corpus without them still
runs, but the notebook logs that sentences will not be traceable and the
provenance table says so rather than being silently absent.

```json
{"text": "There is Logan Hamilton coming up in the 14th.", "video_id": "VYTJiGWe4fo",
 "video_title": "Victory Lane Outlaws", "channel": "Fearless Broadcasting",
 "upload_date": "2026-05-29", "series": "VLO", "license_spdx": "CC-BY-3.0"}
```

### Optional: `gazetteer.json`

Only needed when `apply_masking=True`. A dict with an optional `_meta` block and
one entry list per entity class, where each class becomes a placeholder token:

```json
{"_meta": {"version": "...", "category_codes": {"PERSON": "drivers", "TEAM": "...", "PLACE": "..."}},
 "entries": ["Logan Hamilton", "Robin Hamilton", "..."]}
```

Entity classes are set by `entity_classes` (default `PERSON`, `TEAM`, `PLACE`).
Masking runs before embedding to reduce dependence on recognised entity names; the masking
coverage report is a first-class artefact, and keywords dominated by proper nouns
mean the gazetteer has a gap. Set `use_ner_fallback=True` to add a spaCy pass.

Also conventional, though nothing reads them: `README.md` describing the corpus,
`attribution.csv` if the licence requires attribution, and `manifest.json`
recording how the sentences were produced.

### Registering the corpus

Add an entry to `CORPORA` in `pipeline/config.py`:

```python
"my_corpus": dict(
    corpus_name="my_corpus",
    sentences_path=RELEASE_ROOT / "data" / "my_corpus" / "sentences.jsonl",
    gazetteer_path=RELEASE_ROOT / "data" / "my_corpus" / "gazetteer.json",
    session_field="doc_id",
    apply_masking=True,
    min_cluster_size=35,
),
```

Then set `CORPUS = "my_corpus"` in the notebook. Results land in
`results/my_corpus/`, alongside the others rather than overwriting them.

### Applying it to a different domain

The protocol makes no assumption that the text is motorsport commentary. It needs
sentence-level text grouped into sessions, and enough of both for the axes to mean
something. Three settings usually need attention:

* **`min_cluster_size`** — the granularity sweep locates the plateau from the data.
  Run it first and read the plateau off the sweep figure rather than reusing 35.
* **`entity_classes` and the gazetteer** — masking is domain-specific. Decide which
  entity types would otherwise dominate the geometry, then check the masking
  coverage report.
* **`n_domains`** — the Ward cut and Louvain communities are compared at this
  value; the cut profile shows where the hierarchy actually splits.

Cluster identity is derived, never imported: each cluster is named by a positional
handle plus its own FDR-significant terms and centroid-nearest exemplars, so a new
corpus produces self-describing labels rather than categories borrowed from the
paper.

## Masking is part of the input

Embeddings are computed over masked text, which reduces the influence of recognised entity names on the
clustering. The gazetteer travels with the corpus; `use_ner_fallback=True` adds a
spaCy pass on top. The masking coverage report is a first-class artefact — if a
cluster's keywords are dominated by proper nouns, the gazetteer has a gap.

## Held-out semantics

With a second corpus configured, coverage measures transfer **across registers**.
Without one, sessions are split and coverage measures transfer to **unseen
sessions** — a weaker claim. The notebook prints which of the two it ran.

## Self-contained

Everything the kit needs is inside this folder: no path in `pipeline/` or the
notebook resolves outside it, and the corpus ships in
`data/`. Copy the folder anywhere and it runs.

`results/<corpus>/cache/` is git-ignored — it is ~40 MB of regenerable `.npy` and
`.parquet` intermediates. A fresh clone therefore starts cold and the first run
recomputes everything, which is the path `test_scratch_smoke.py` exercises
(45 stages, cold cache, exit code = failed stages).

## Layout

```
00_orchestrator.ipynb     run everything, render everything
LICENSE                   MIT for the code; data/ is CC-BY-3.0, see its README
CITATION.cff              machine-readable citation metadata
pipeline/                 flat, one concern per module
data/
  iracing/                shipped CC-BY corpus + gazetteer
  f1_highlights/          placeholder - broadcast copyright, not redistributed
  f1_full/                placeholder - broadcast copyright, not redistributed
results/
  iracing/{tables,figures,cache}
  f1_highlights/{tables,figures}   aggregate outputs behind the paper
  f1_full/{tables,figures}         held-out coverage outputs
supplements/              published supplementary material (PDF + pointers)
```

## Determinism

`seed=42` is applied to Python, NumPy, UMAP, HDBSCAN, subsampling and the
permutation nulls. UMAP with a fixed `random_state` is single-threaded and
reproducible; exact cluster *indices* may still differ across library versions,
which is why identity is keyword-derived rather than index-dependent.
