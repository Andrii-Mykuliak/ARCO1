# Cross-Register Thematic Structure in Formula 1 Commentary

Code, frozen configuration and derived results for **"What Commentators Talk
About: Cross-Register Thematic Structure in Formula 1 Commentary"**.

```bash
pip install -r requirements.txt
jupyter notebook 00_reproduce_pipeline.ipynb   # the reproduction entry point
```

---

## What this study does

Four layers, in order:

1. **Thematic discovery** — 113 Formula 1 highlight transcripts, 7,647 sentences.
   Entity-masked MiniLM embeddings → UMAP → HDBSCAN. Canonical solution:
   **34 categories at `min_cluster_size = 35`**, with lexical enrichment and
   interpretation.

2. **Diagnostic characterisation** — how well each category is supported across
   **complementary diagnostic dimensions**: alternative-model recovery,
   resampling stability, held-out race replication, alternative-geometry
   separability, masking diagnostics, human-rater evidence, domain-level
   description.

   > These are **not** "validation axes". No aggregate pass/fail vote and no
   > verdict band is computed. Support is heterogeneous across categories and
   > dimensions, and collapsing it into one number misrepresents it.

3. **Independent full-race structural replication** — the same pipeline applied
   to 10 full-race transcripts (16,876 sentences) with **no highlight
   information** until the solution is frozen. Granularity is selected from a
   **full-race-specific sweep**, never by scaling `min_cluster_size` with corpus
   size. Primary solution: **48 clusters at `mcs = 40`**, noise 0.3155,
   DBCV 0.3910, all 48 lexically enriched.

   Correspondence with the highlight categories: **29/34 above a
   category-specific null, 19 one-to-one, 5 split, 5 merge, 5 none** — and all
   five non-recoveries are categories that are absent or sparse in the matched
   events.

4. **Paired same-event register analysis** — 10 matched highlight/full-race event
   pairs, both registers assigned by the **same frozen canonical centroids**.

   > Across ten matched events, strategy-and-technical commentary occupied a
   > consistently smaller relative share in highlight commentary than in
   > full-race commentary. The direction persisted under log-ratio treatment and
   > leave-one-event-out analyses, and the aggregate shift exceeded that expected
   > from matched-length sampling alone.

   **The inferential unit is the event (n = 10).** Only 4 of 10 events
   individually exceed their own 95% matched-length null, so the result is
   aggregate, not per-event. Pooled sentences are never treated as independent
   observations.

---

## A. Reproduction in one notebook

`00_reproduce_pipeline.ipynb` is the **single reproduction entry point**, and it
has two predefined configurations. Selecting one is the entire configuration
procedure — change one line and run every cell:

```python
CONFIG = IRACING          # IRACING or F1
```

| | `IRACING` | `F1` |
|---|---|---|
| corpora | `iracing` | `f1_highlights` + `f1_full` |
| registers | one | two, matched |
| matched events, category labels, domain grouping | none | resolved from the frozen metadata |
| output root | `results/iracing/` | `results/f1_run/`, `results/f1_cross_register_run/` |
| ships with the package | yes | no — supply the licensed transcripts |

Everything else is resolved from the selected preset. No other line in the
notebook needs editing, and neither preset can write over the other's output or
over the frozen canonical artefacts in `results/f1_highlights/` and
`results/f1_cross_register/`.

Any other value fails immediately with *"Unknown CONFIG value … Supported
configurations: 'iracing', 'f1'."*, and a preset whose corpora are not present
fails naming the corpus and the path it is expected at.

**One register in → every single-register analysis runs.** Loading, masking,
embedding, the granularity sweep, clustering, enrichment, labelling, all four
complementary diagnostic dimensions — held-out replication included, since it
splits one corpus by session — a descriptive centroid-level grouping, the
single-register sensitivity views and the figures that follow. The sections that
need a second register print *"skipped: no second commentary register was
supplied"* and are passed over. That is a property of the data, not a failure.

**Two matched registers in → every analysis layer runs.** The second register is
reconstructed independently from its own granularity sweep, the two structures
are compared at category level against a category-specific null, and thematic
composition is compared across the matched events, with the dependent
sensitivity analyses and the tables and figures that follow.

Which sections run is decided from the configured **data**, never from a corpus
name. Naming a second corpus that is missing or invalid is a **configuration
error** and stops the run — it is never downgraded silently to a one-register
run.

Nothing is ever substituted for a missing register: no pseudo-registers, no
duplicated corpus, no synthetic data, and no train/test split of one register
standing in for two.

### What the run produces

Every analytical artefact the configuration supports is written from that run's
own results and named for its scientific content:

| Content | One register | Two registers |
|---|---|---|
| Corpus summary; granularity sweep with the selected setting | ✓ | ✓ |
| Category inventory, lexical enrichment summary, full term-enrichment records | ✓ | ✓ |
| Per-category diagnostic profiles across the four dimensions; diagnostic coverage | ✓ | ✓ |
| Granularity selection, thematic structure, diagnostic distributions and profiles (figures) | ✓ | ✓ |
| Second-register reconstruction summary, sweep, clusters, seed stability | — | ✓ |
| Cross-register similarity matrix, per-category outcomes, null thresholds, availability control | — | ✓ |
| Matched-event manifest, coverage, category shares and differences | — | ✓ |
| Compositional differences per event, robustness across representations, leave-one-out | — | ✓ |
| Matched-length null intervals and their variants | — | ✓ |

The notebook writes `outputs.csv`, listing each table and figure it produced and
where it went, generated from the files actually written. It closes with a
summary of the values this run computed.

## B. The public corpus and the restricted corpora

The shipped **iRacing** corpus (5,967 sentences, 8 sessions, CC BY 3.0) contains
**one commentary register**, so the public default is a one-register run.

> **The public iRacing corpus is a code-path demonstration.** It is not an
> analytical replication of the Formula 1 results, not an external validation,
> and not evidence for the Formula 1 cross-register result.

Its `min_cluster_size = 35` is a **fixed demonstration setting** with no claim of
optimality. It was *not* chosen to resemble the Formula 1 cluster count —
selecting granularity to match another corpus is not a valid criterion.

The Formula 1 transcripts are broadcast-copyright and **are not redistributed**.
Readers holding licensed copies can place them at
`data/f1_highlights/sentences.jsonl` and `data/f1_full/sentences.jsonl` — see the
README in each folder for the expected format — and then select `CONFIG = F1`.
Supplying them at the documented locations makes every analysis layer runnable
under the released configuration: the cross-register
producers ship with this package (`pipeline/cross_register.py`,
`pipeline/paired_register.py`, with the frozen inputs in
`pipeline/frozen_f1.py`), not only the code that reads their outputs.

**Running every layer is not the same as reproducing the archived numbers.**
A two-register run recomputes the second register from its own sweep, and that
sweep is sensitive to the embedding and manifold realisation: on the environment
used for this package it selects a different granularity from the archived one,
so the correspondence classes shift accordingly. Archived embeddings and
projections can be pinned through the preset's advanced `archived` field so an
archived realisation is used instead of a recomputed one; neither supported
preset sets it, so both compute everything from the corpora. The frozen
values in `results/` remain the reference for the published numbers.

## C. What the public release still provides

- the complete implementation (`pipeline/`, 18 modules), including the
  cross-register and matched-event **producers**, not only their consumers
- the complete run configuration, in one place (`pipeline/config.py`): the two
  run objects the notebook selects between, the per-corpus presets, the
  parameter defaults, and the software stack the reported run used
- derived numeric tables for all four layers, with broadcast text reduced to
  sha1 references by `publication.py`
- the manuscript figures (PNG + SVG)
- `results/paper_results_summary.json` — every headline number, read from
  computed artefacts rather than transcribed
- corpus manifests, attribution and hashes
- the executable iRacing demonstration (`00_reproduce_pipeline.ipynb`)

---

## Restricted-text policy

`publication.py` strips known text columns to sha1 references and then re-scans
the written file, aborting if prose-like text survives, so artefacts written by
the current pipeline carry no broadcast sentence. Artefacts committed before that
guard was added must be regenerated before release.

The previous edition leaked 104 verbatim F1 sentences; they were purged and the
guard added. See `RELEASE_AUDIT.md`.

iRacing exemplars *are* retained — CC BY 3.0 permits redistribution, and
`attribution.csv` maps every sentence to its source video.

---

## Layout

    pipeline/     implementation, one concern per module
    data/         iracing (ships) · f1_highlights, f1_full (README only)
    results/      f1_highlights · f1_full · f1_cross_register · iracing
    supplements/  supplementary materials

Module names use current terminology (`diagnostics`, `heldout_replication`,
`cross_register`, `paired_register`); the previous names
(`validation`, `axis3`, `coverage`) remain importable for one release.

See `CONTENTS.md` for the full inventory and `RELEASE_AUDIT.md` for what changed.

## Licence

MIT covers the **source code only**. `data/iracing/` is redistributed under
**CC BY 3.0**; attribution requirements travel with it.

## Citation

See `CITATION.cff`.
