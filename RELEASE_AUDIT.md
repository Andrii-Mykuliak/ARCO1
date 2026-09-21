# Release audit — pre-modification state

Read-only audit of `release/` before restructuring. A point-in-time record:
every path, module and filename below describes the state **as it was then**,
including names that have since been renamed or retired. Machine-readable copy
in `results/_release_audit.json`.

The audit and guard scripts it was produced by (`tests/`) are no longer part of
the release and are not shipped.

---

## 1. Inventory

| | files | size |
|---|---|---|
| on disk | 389 | 114.1 MB |
| tracked in git | 206 | 23.4 MB |
| untracked / ignored | 183 | 90.7 MB |

Separate git repository, branch `main`, single commit `929ecad`.

Uncommitted: `docs/`, `pipeline/entity_concentration.py` (both from later
entity-concentration work, never committed or documented).

---

## 2. CRITICAL — raw restricted F1 broadcast text is committed

**104 verbatim F1 broadcast sentences are in the published tree.**

| file | columns | sentences |
|---|---|---|
| `results/f1_highlights/tables/cluster_cards.csv` | `exemplar_1/2/3` | **102** |
| `results/f1_highlights/tables/noise_profile.csv` | `least_similar_example`, `most_similar_example` | **2** |

Entity masking (`<PERSON>`) does not make these redistributable — they are
still verbatim transcript lines from copyright broadcast material.

`supplements/README.md` names three files excluded for exactly this reason
(`cluster_overview_table.csv`, `multi_race_coverage.csv`,
`multi_race_leftover_clusters.csv`) but **missed these two**. The exclusion rule
existed and was applied incompletely.

iRacing `cluster_cards.csv` also carries 21 exemplars — that one is fine, CC BY 3.0.

**Must be removed from the published tree before release.**

> **RESOLVED — verified 2026-09-19.** Both files were purged of transcript
> text. `cluster_cards.csv` now carries `exemplar_1/2/3_sha1` hash columns and
> an `exemplars_withheld` note in place of the three exemplar text columns;
> `noise_profile.csv` carries `examples_withheld` in place of its two example
> columns. A repository-wide scan of every tracked CSV/TSV/JSON/JSONL artefact
> (139 csv/tsv, 2 json, 1 jsonl) for natural-language cells returns zero
> Formula 1 transcript sentences or sentence fragments. The only prose-bearing
> tracked artefacts are the CC BY 3.0 iRacing corpus, author-written `note`
> columns, and statistical enriched-term lists (`top_enriched_terms`,
> `P*_top_terms`), none of which are transcript excerpts.
> `tests/test_release_guards.py`: 152 passed, 0 failed.

---

## 3. CRITICAL — the iRacing granularity rationale is both wrong and forbidden

`data/iracing/README.md` lines 113–129:

> "The demo uses `min_cluster_size = 35` rather than the F1 canonical `mcs = 30`."
> "F1 canonical mcs=30 was validated specifically for **F1 highlights register**"
> "mcs=35 gives a cluster count **comparable in structural scale to F1's
>  34-cluster baseline** … enabling apples-to-apples thematic comparison."
> "It is an explicit choice to match structural scales for comparability."

Two separate defects:

1. **Factually wrong.** The canonical F1 highlight value is **mcs = 35**, not 30.
   The claim that the demo differs from F1 is itself false — they coincide.
2. **Methodologically forbidden.** Selecting granularity to match F1's cluster
   count is precisely the reasoning this release must not contain, and it is the
   same error that produced the superseded corpus-scaled full-race `mcs = 77`.

---

## 4. Stale terminology — 11 tracked files

| file | occurrences |
|---|---|
| `pipeline/validation.py` | four-axis ×2, verdict band ×1, Axis 1 ×5, Axis 2 ×2, Axis 3 ×4, Axis 4 ×2, axis3 ×23 |
| `pipeline/config.py` | Axis 1 ×2, Axis 3 ×3, axis3 ×11, min_cluster_size=30 ×1 |
| `pipeline/axis3.py` | Axis 3 ×2, axis3 ×16 |
| `README.md` | four-axis ×3, verdict band ×2, Axis 1 ×1, Axis 3 ×6, axis3 ×2 |
| `supplements/README.md` | four-axis ×1, validated cluster ×1, verdict band ×2 |
| `data/iracing/README.md` | mcs=30 ×1, comparability rationale ×2 |
| `CITATION.cff` | four-axis ×1 |
| `requirements.txt` | verdict band ×1 |
| `pipeline/__init__.py`, `pipeline/enrichment.py`, `pipeline/reporting.py` | axis3 / Axis 3 references |

`supplements/README.md` still describes "the 34 **validated** clusters, each with
domain, **four-axis outcome**, **verdict band**" and presents cross-register
**coverage** as the cross-register result — all superseded.

---

## 5. Scientific content that is superseded

| artefact | status |
|---|---|
| four-axis verdict-band tables | superseded — no aggregate vote in the current design |
| cross-register **coverage** as primary cross-register evidence | demoted to secondary/descriptive (register-level circularity is irreducible) |
| Axis-3 seed-42 held-out 24/34 presented as representative | superseded — 24/34 is the maximum of its distribution, median 17 |
| full-race corpus-size-scaled `mcs = 77` | never in this release, but must not be reintroduced |
| Axis 2 = 26/34, Axis 4 = 25/34 | Axis 4 is a Monte-Carlo budget artefact (29/34 at 100 replicates); Axis 2 is not reproducible |

The release predates the full-race independent replication and the paired
register analysis entirely — **neither exists here in any form**.

### 5.1 Diagnostic-dimensions closure (promoted 2026-09-12)

The closure runs for three of the four diagnostic dimensions are now **in** the
release and are the manuscript source of truth. They live under
`results/f1_highlights/diagnostics/`, with the combined provenance record in
`diagnostic_dimensions_closure_summary.json`.

| dimension | superseded value | final state |
|---|---|---|
| Resampling stability | 30-resample run (26/34) | 200 resamples, 35 degenerate, 165 viable; summaries conditional on viable runs. **The historical 30-run prefix does not reproduce under this environment** (`prefix_reproduces_canonical = false`, max abs diff 0.086). The extended run exposes version-sensitive non-reproducibility rather than refining the earlier estimate — neither pass count is canonical. |
| Held-out race replication | single seeded 80/33 split (24/34) | 50 seeds × 2 embedding caches × 2 granularities = 200 runs, 0 failed, 47 degenerate. Distributional reporting. The single-seed outcome is **atypically optimistic** — 97.7th percentile of its own cell, whose median is 17. |
| Alternative-geometry separability | 10-replicate run (25/34) | 100 replicates, 15 per category, K = 34, 0 failed, 0 degenerate, **prefix reproduction gate passed**. The 10-replicate value was a Monte-Carlo budget artefact. |
| Alternative-model recovery | — | unchanged by the closure pass. |

Historical settings were preserved verbatim. They lived in
`configs/paper_f1.json` under `highlight.diagnostics_historical_superseded`
until that file was consolidated into `pipeline/config.py`, and are now
recorded in `docs/superseded_artefacts.md`; nothing was silently erased.

Still true after promotion: no aggregate vote, no banded verdict and no
pass/fail ensemble is computed across the four dimensions. They are reported
separately because they probe different properties.

**Terminology.** `axis2` / `axis3` / `axis4` are legacy implementation
identifiers, retained only in provenance fields and in the generating script
names under `scripts/`. Release-facing filenames, configuration keys and summary
keys use the scientific dimension names. The correspondence label `one-to-one`
has been replaced by **`single-match`**, because the implemented relation is
category-local rather than globally bijective; the counts 19 / 5 / 5 / 5 are
unchanged by that relabelling.

**Headline science is untouched by this promotion.** The highlight and full-race
partitions, the cross-register correspondence counts and the paired
same-event results are byte-identical to their pre-promotion state.

---

## 6. Directory hygiene

| item | state |
|---|---|
| `y/`, `y/cache`, `y/figures`, `y/tables` | **empty**, NOT gitignored, undocumented — delete |
| `results/_ident/` | 1 scratch file, NOT gitignored |
| `_smoke_results/` | 32 files, correctly gitignored |
| `results/f1_full/cache`, `results/f1_full/figures` | empty |
| `results/.ipynb_checkpoints` | Jupyter autosave |

No `results/f1_cross_register/` exists.

---

## 7. Pipeline modules — 15 present

`__init__ ablation axis3 clustering config corpus coverage domains embedding
enrichment entity_concentration io_utils labeling reporting validation`

Required renames: `validation.py` → `diagnostics.py`,
`axis3.py` → `heldout_replication.py`, `coverage.py` → `cross_register.py`.
Missing entirely: `paired_register.py`.

---

## 8. CONTENTS.md drift

| claim | actual |
|---|---|
| 13 pipeline modules | 15 |
| 3 test files | 4 (+ this audit script) |
| 145 CSV tables | 165 |
| 112 MB on disk | 114.1 MB |
| 220 files / ~23 MB committed | 206 files / 23.4 MB (close) |
| `docs/` absent from layout | exists |

---

## Ordered remediation plan

1. **Purge raw F1 text** from `cluster_cards.csv` and `noise_profile.csv`; add a
   leakage guard test so it cannot recur.
2. **Rewrite the iRacing granularity rationale** — remove the F1-comparability
   logic and the `mcs = 30` error.
3. **Rename modules** to current terminology, add `paired_register.py`.
4. **Add `results/f1_cross_register/`** and populate from the frozen analyses.
5. **Update configs** to freeze the current three-layer analysis.
6. **Regenerate** README, CONTENTS, supplements README, `paper_results_summary.json`.
7. **Delete** `y/`; gitignore `results/_ident/`.
8. **Add scientific guard tests.**
