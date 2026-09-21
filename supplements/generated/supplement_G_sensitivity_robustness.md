# Supplement G — Sensitivity Robustness

## `fullrace_neighbour_stability.csv`
15 rows x 12 columns.

Columns: `mcs_low`, `mcs_high`, `n_clusters_low`, `n_clusters_high`, `ari_all`, `ari_co_clustered`, `hungarian_mean_jaccard`, `median_matched_jaccard`, `recovery_proportion`, `major_splits`, `major_merges`, `either_degenerate`

## `cross_register_regime_sensitivity.csv`
5 rows x 7 columns.

Columns: `mcs`, `is_primary`, `n_fr_clusters`, `n_above_within_ref`, `median_best_cosine`, `strategy_above_ref`, `strategy_median_cosine`

|   mcs | is_primary   |   n_fr_clusters |   n_above_within_ref |   median_best_cosine |   strategy_above_ref |   strategy_median_cosine |
|------:|:-------------|----------------:|---------------------:|---------------------:|---------------------:|-------------------------:|
|    40 | True         |              48 |                   27 |             0.928537 |                    6 |                 0.933766 |
|    30 | False        |              65 |                   28 |             0.921831 |                    7 |                 0.925293 |
|    35 | False        |              53 |                   27 |             0.923647 |                    6 |                 0.930044 |
|    45 | False        |              43 |                   25 |             0.929355 |                    6 |                 0.933058 |
|    50 | False        |              38 |                   26 |             0.930937 |                    6 |                 0.932307 |

## `matched_length_null_contiguous.csv`
40 rows x 11 columns.

Columns: `event_id`, `k_blocks`, `n_valid`, `n_failed`, `observed_effect`, `null_mean`, `null_sd`, `null_p2.5`, `null_p97.5`, `p_one_sided_depletion`, `below_null_p2.5`

## `mcs_sensitivity.csv`
14 rows x 8 columns.

Columns: `mcs`, `n_clusters`, `noise_rate`, `dbcv`, `silhouette`, `mean_enriched`, `frac_interp`, `time_sec`

|   mcs |   n_clusters |   noise_rate |     dbcv |   silhouette |   mean_enriched |   frac_interp |   time_sec |
|------:|-------------:|-------------:|---------:|-------------:|----------------:|--------------:|-----------:|
|    15 |           62 |     0.35151  | 0.352509 |     0.633141 |         11.2581 |      0.887097 |    2.35014 |
|    20 |           44 |     0.30391  | 0.345994 |     0.6404   |         13.8182 |      0.909091 |    2.22617 |
|    25 |           38 |     0.294495 | 0.303152 |     0.634598 |         15.2632 |      0.973684 |    2.37658 |
|    30 |           35 |     0.343795 | 0.360472 |     0.666739 |         15.2857 |      0.942857 |    2.19671 |
|    35 |           34 |     0.335164 | 0.34772  |     0.662435 |         17.1765 |      1        |    2.19761 |
|    40 |           33 |     0.341703 | 0.322213 |     0.669103 |         16.7273 |      0.969697 |    2.19021 |
|    45 |           30 |     0.345757 | 0.316425 |     0.657663 |         16.8    |      1        |    2.07163 |
|    50 |           29 |     0.356872 | 0.328375 |     0.659681 |         17.0345 |      1        |    2.13222 |
|    55 |            3 |     0        | 0.359663 |     0.346939 |         18.6667 |      1        |   14.5896  |
|    60 |            3 |     0        | 0.359663 |     0.346939 |         18.6667 |      1        |   13.7242  |
|    70 |            3 |     0        | 0.359663 |     0.346939 |         18.6667 |      1        |   13.3763  |
|    80 |            3 |     0        | 0.359663 |     0.346939 |         18.6667 |      1        |   13.4461  |
|   100 |            3 |     0        | 0.359663 |     0.346939 |         18.6667 |      1        |   13.2361  |
|   120 |            2 |     0        | 0.332189 |     0.494477 |         15.5    |      0.5      |   13.5874  |
