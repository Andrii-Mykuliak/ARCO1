# Legacy outputs — superseded analysis design

**These outputs are not part of the current manuscript and must not feed any
publication artefact.** The files themselves are no longer shipped: they were
removed from the release when the analysis design changed, and remain available
in git history at their original paths under `results/f1_highlights/tables/`.
This note records what they were and why they were retired.

## What they are

Outputs of an earlier analysis design in which the four diagnostic dimensions
were treated as **validation axes** and collapsed into an **aggregate pass/fail
vote** and a **verdict band** per category.

| file | what it was |
|---|---|
| `canonical_four_axis.csv` | per-category axis scores **plus** pass flags, `n_met` and `band` |
| `3axis_convergence_table.csv`, `convergence_band_ci.csv`, `convergence_bands_optionC.csv` | verdict-band aggregation and its confidence intervals |
| `axis_counts.csv`, `axis_pass_rate_ci.csv`, `axis_score_distributions.csv` | aggregate pass-rate summaries |
| `axis_correlation*.csv` (6 files) | inter-axis correlation, which presupposes the axes are commensurable votes |

## Why they are superseded

The current study reports **complementary diagnostic dimensions** and a
**category-level diagnostic profile**. It deliberately computes:

- **no aggregate pass/fail vote**
- **no verdict band**
- **no "4/4", "3/4" style counts**

Support is heterogeneous across categories and across dimensions, and collapsing
it into a single number misrepresents it. Two of the underlying numbers are also
now known to be unreliable in their stored form: the alternative-geometry score
was computed at an unconverged 10-replicate budget, and the resampling score is
not reproducible across `umap-learn` versions.

## What replaced them

The **continuous** per-dimension scores were preserved, without the vote, as:

    ../tables/category_diagnostic_profiles.csv

with dimensions renamed to current terminology:

| was | is |
|---|---|
| `A1_score` | `alternative_model_recovery` |
| `A2_score` | `resampling_stability` |
| `A3_score` | `heldout_replication` |
| `A4_score` | `geometry_separability` |

That table, not these files, is the source for Figure B and Table 3.

Three further tables held current per-dimension data under old names and were
**renamed rather than retired**: `axis3_heldout.csv` →
`heldout_replication_summary.csv`, and the two
`hierarchical_sentence_axis4*.csv` → `geometry_separability_*.csv`.

## Why they are not shipped

The manuscript states that the four diagnostic dimensions are reported
separately, per category, and are deliberately not collapsed into an aggregate
score, a pass/fail verdict or a count of validated categories. Shipping these
files alongside the current results would offer exactly that collapsed reading.
