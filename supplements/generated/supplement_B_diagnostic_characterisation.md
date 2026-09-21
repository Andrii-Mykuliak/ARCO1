# Supplement B — Diagnostic Characterisation

## `category_diagnostic_profiles.csv`
34 rows x 10 columns.

Columns: `cluster_id`, `label`, `domain`, `size`, `alternative_model_recovery`, `resampling_stability`, `heldout_replication`, `geometry_separability`, `heldout_rep_strict`, `heldout_rep_direction`

## `heldout_replication_summary.csv`
34 rows x 11 columns.

Columns: `cluster`, `has_estimate`, `n_train_enriched`, `n_test_strict`, `n_test_lenient`, `n_test_direction`, `rep_strict`, `rep_lenient`, `rep_dir`, `axis3_pass`, `axis3_score`

## `geometry_separability_euclidean.csv`
34 rows x 8 columns.

Columns: `cluster_id`, `thematic_label`, `size`, `ax4_mean_jaccard`, `ax4_std_jaccard`, `ax4_min_jaccard`, `ax4_max_jaccard`, `ax4_stable`

|   cluster_id | thematic_label             |   size |   ax4_mean_jaccard |   ax4_std_jaccard |   ax4_min_jaccard |   ax4_max_jaccard | ax4_stable   |
|-------------:|:---------------------------|-------:|-------------------:|------------------:|------------------:|------------------:|:-------------|
|            7 | Race start lights          |     36 |           0.864485 |         0.100523  |          0.666667 |          1        | True         |
|            4 | Race start launch          |     42 |           0.861373 |         0.129862  |          0.625    |          1        | True         |
|           10 | Chicane moves              |     62 |           0.82326  |         0.110916  |          0.625    |          1        | True         |
|            0 | DRS battles                |    205 |           0.819902 |         0.0804555 |          0.6875   |          0.9375   | True         |
|            1 | Time penalties             |    118 |           0.797917 |         0.122435  |          0.6      |          0.933333 | True         |
|           11 | Team radio celebrations    |     55 |           0.768748 |         0.142708  |          0.533333 |          0.9375   | True         |
|           12 | Podium finishes            |    139 |           0.766865 |         0.107862  |          0.590909 |          0.9375   | True         |
|            8 | Flags signals              |    118 |           0.728824 |         0.151938  |          0.5      |          1        | True         |
|            3 | Mirror checks defending    |     47 |           0.727672 |         0.0806435 |          0.533333 |          0.8      | True         |
|           29 | Corner specific moves      |     60 |           0.716944 |         0.156349  |          0.444444 |          0.933333 | True         |
|            5 | Rain & weather             |     43 |           0.716034 |         0.119302  |          0.545455 |          0.933333 | True         |
|            2 | Lights out start           |     54 |           0.702917 |         0.0696232 |          0.6      |          0.875    | True         |
|            6 | Grand Prix wins            |    465 |           0.686303 |         0.180638  |          0.388889 |          0.933333 | True         |
|           28 | Corner battles early turns |     82 |           0.674589 |         0.191831  |          0.25     |          0.933333 | True         |
|            9 | Formula 1 milestones       |    134 |           0.671169 |         0.146653  |          0.45     |          0.933333 | True         |
|           20 | Wing & car damage          |     61 |           0.669559 |         0.12746   |          0.458333 |          0.875    | True         |
|           22 | Wheel-to-wheel             |     71 |           0.639465 |         0.163058  |          0.3125   |          0.866667 | True         |
|           16 | Safety car deployment      |    122 |           0.625658 |         0.149735  |          0.375    |          0.866667 | True         |
|           19 | Race lead holding          |    115 |           0.619597 |         0.249994  |          0.2      |          0.933333 | True         |
|           31 | Race leader changes        |    162 |           0.605533 |         0.185789  |          0.296296 |          0.866667 | True         |
|           17 | Gap & pace                 |     67 |           0.601582 |         0.188062  |          0.178571 |          0.875    | True         |
|           23 | Racetrack action           |     72 |           0.585244 |         0.139096  |          0.384615 |          0.8      | True         |
|           15 | Tires & compounds          |    379 |           0.575147 |         0.171566  |          0.352941 |          0.8125   | True         |
|           30 | Lap timing & fastest lap   |    136 |           0.534262 |         0.202683  |          0.264706 |          0.8125   | True         |
|           21 | Braking & lockups          |    128 |           0.531874 |         0.13885   |          0.333333 |          0.75     | True         |
|           13 | Championship & title       |    158 |           0.491832 |         0.194068  |          0.26087  |          0.9375   | False        |
|           26 | Defending & attacking      |    110 |           0.481424 |         0.188789  |          0.189189 |          0.705882 | False        |
|           25 | Indirect driver references |     69 |           0.480582 |         0.147071  |          0.243243 |          0.75     | False        |
|           24 | Car length gaps            |     60 |           0.47483  |         0.180866  |          0.222222 |          0.8125   | False        |
|           14 | Pit stops                  |    185 |           0.446841 |         0.228526  |          0.148148 |          0.882353 | False        |
|           32 | Front position changes     |     45 |           0.407461 |         0.0869679 |          0.264706 |          0.588235 | False        |
|           33 | Mid-field positions        |    122 |           0.391927 |         0.115486  |          0.192308 |          0.565217 | False        |
|           18 | Team & teammate dynamics   |    600 |           0.365429 |         0.171117  |          0.166667 |          0.75     | False        |
|           27 | Overtake battles           |    762 |           0.33875  |         0.0822495 |          0.166667 |          0.464286 | False        |

## `loo_sensitivity.csv`
11 rows x 4 columns.

Columns: `removed`, `majority_count`, `delta`, `abs_delta`

| removed             |   majority_count |   delta |   abs_delta |
|:--------------------|-----------------:|--------:|------------:|
| (none - all judges) |               28 |       0 |           0 |
| enc_all             |               28 |       0 |           0 |
| enc_bge             |               27 |      -1 |           1 |
| enc_gte             |               28 |       0 |           0 |
| kmeans              |               28 |       0 |           0 |
| ward                |               28 |       0 |           0 |
| optics              |               28 |       0 |           0 |
| leiden              |               27 |      -1 |           1 |
| res_mcs17           |               27 |      -1 |           1 |
| res_mcs52           |               27 |      -1 |           1 |
| seed_umap2          |               27 |      -1 |           1 |
