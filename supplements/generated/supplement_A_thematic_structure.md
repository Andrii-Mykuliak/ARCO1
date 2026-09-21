# Supplement A — Thematic Structure

## `category_diagnostic_profiles.csv`
34 rows x 10 columns.

Columns: `cluster_id`, `label`, `domain`, `size`, `alternative_model_recovery`, `resampling_stability`, `heldout_replication`, `geometry_separability`, `heldout_rep_strict`, `heldout_rep_direction`

## `cluster_cards.csv`
34 rows x 11 columns.

Columns: `handle`, `cluster`, `size`, `share_of_clustered`, `keywords`, `n_enriched`, `external_label`, `exemplar_1_sha1`, `exemplar_2_sha1`, `exemplar_3_sha1`, `exemplars_withheld`

## `partition_stats.csv`
1 rows x 6 columns.

Columns: `min_cluster_size`, `n_clusters`, `noise_pct`, `clustered_sentences`, `plateau_lo`, `plateau_hi`

|   min_cluster_size |   n_clusters |   noise_pct |   clustered_sentences |   plateau_lo |   plateau_hi |
|-------------------:|-------------:|------------:|----------------------:|-------------:|-------------:|
|                 35 |           34 |        32.8 |                  5130 |           30 |           35 |

## `size_concentration.csv`
1 rows x 8 columns.

Columns: `clustered_sentences`, `top1_pct`, `top5_pct`, `top10_pct`, `largest`, `smallest`, `median`, `gini`

|   clustered_sentences |   top1_pct |   top5_pct |   top10_pct |   largest |   smallest |   median |   gini |
|----------------------:|-----------:|-----------:|------------:|----------:|-----------:|---------:|-------:|
|                  5130 |       15.4 |       47.1 |        62.6 |       792 |         41 |    112.5 |  0.456 |
