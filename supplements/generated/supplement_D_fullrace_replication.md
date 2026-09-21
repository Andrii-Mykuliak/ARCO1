# Supplement D — Fullrace Replication

## `fullrace_granularity_sweep.csv`
16 rows x 14 columns.

Columns: `mcs`, `effective_min_samples`, `min_samples_passed`, `n_clusters`, `noise_fraction`, `assigned_fraction`, `DBCV`, `silhouette`, `min_cluster_size_observed`, `median_cluster_size`, `max_cluster_size`, `mean_enriched_terms`, `frac_interpretable`, `degenerate_flag`

## `fullrace_primary_solution.csv`
48 rows x 9 columns.

Columns: `fr_cluster`, `size`, `n_enriched`, `top_terms`, `races_present`, `mean_cosine_to_centroid`, `representative_1_sha1`, `representative_2_sha1`, `representative_3_sha1`

## `fullrace_seed_stability.csv`
3 rows x 16 columns.

Columns: `mcs`, `n_seeds`, `viable_seeds`, `failed_seeds`, `degenerate_seeds`, `n_clusters_ref`, `n_clusters_min`, `n_clusters_median`, `n_clusters_max`, `noise_min`, `noise_median`, `noise_max`, `ari_median`, `ari_min`, `hungarian_median`, `hungarian_min`

## `fullrace_cluster_enrichment.csv`
1374 rows x 7 columns.

Columns: `fr_cluster`, `term`, `in_cluster`, `cluster_size`, `p_value`, `significant`, `weight`

## `cross_register_correspondence.csv`
34 rows x 18 columns.

Columns: `highlight_cluster`, `highlight_name`, `domain`, `availability_class`, `matched10_sentences`, `best_fr_cluster`, `fr_size`, `fr_races_present`, `centroid_cosine`, `lexical_weighted_jaccard`, `lexical_overlap_at_15`, `null_p95`, `null_p_empirical`, `exceeds_null`, `n_fr_above_threshold`, `fr_clusters_above_threshold`, `relation`, `interpretable_non_recovery`

## `cross_register_null.csv`
34 rows x 10 columns.

Columns: `highlight_cluster`, `highlight_name`, `availability_class`, `observed_best_cosine`, `null_median`, `null_p95`, `null_p_empirical`, `exceeds_null_p95`, `within_highlight_offdiag_p95`, `exceeds_within_highlight_p95`

## `matched_event_availability.csv`
34 rows x 8 columns.

Columns: `highlight_cluster`, `highlight_name`, `domain`, `canonical_total_sentences`, `matched10_sentences`, `matched10_events`, `share_of_canonical_total`, `availability_class`

|   highlight_cluster | highlight_name           | domain               |   canonical_total_sentences |   matched10_sentences |   matched10_events |   share_of_canonical_total | availability_class   |
|--------------------:|:-------------------------|:---------------------|----------------------------:|----------------------:|-------------------:|---------------------------:|:---------------------|
|                   0 | DRS-assisted battles     | Race dynamics        |                         205 |                    10 |                  5 |                  0.0487805 | SPARSE               |
|                   1 | Time penalties           | Regulatory           |                         118 |                    11 |                  4 |                  0.0932203 | SPARSE               |
|                   2 | Lights-out race starts   | Race dynamics        |                          54 |                     1 |                  1 |                  0.0185185 | EFFECTIVELY_ABSENT   |
|                   3 | Mirror checks defending  | Race dynamics        |                          47 |                     6 |                  6 |                  0.12766   | SPARSE               |
|                   4 | Race start launch        | Race dynamics        |                          42 |                     4 |                  4 |                  0.0952381 | EFFECTIVELY_ABSENT   |
|                   5 | Weather conditions       | Strategy & technical |                          43 |                     3 |                  3 |                  0.0697674 | EFFECTIVELY_ABSENT   |
|                   6 | Grand Prix wins          | Narrative & meta     |                         465 |                    55 |                 10 |                  0.11828   | WELL_REPRESENTED     |
|                   7 | Race start lights        | Race dynamics        |                          36 |                     3 |                  3 |                  0.0833333 | EFFECTIVELY_ABSENT   |
|                   8 | Race control flags       | Regulatory           |                         118 |                    11 |                  6 |                  0.0932203 | SPARSE               |
|                   9 | Formula 1 milestones     | Narrative & meta     |                         134 |                    12 |                  8 |                  0.0895522 | SPARSE               |
|                  10 | Chicane moves            | Race dynamics        |                          62 |                     5 |                  2 |                  0.0806452 | SPARSE               |
|                  11 | Team radio celebrations  | Narrative & meta     |                          55 |                     4 |                  3 |                  0.0727273 | EFFECTIVELY_ABSENT   |
|                  12 | Podium finishes          | Narrative & meta     |                         139 |                     8 |                  5 |                  0.057554  | SPARSE               |
|                  13 | Championship & title     | Narrative & meta     |                         158 |                    18 |                  8 |                  0.113924  | WELL_REPRESENTED     |
|                  14 | Pit stops                | Strategy & technical |                         185 |                    17 |                  8 |                  0.0918919 | WELL_REPRESENTED     |
|                  15 | Tires & compounds        | Strategy & technical |                         379 |                    32 |                  8 |                  0.0844327 | WELL_REPRESENTED     |
|                  16 | Safety car deployment    | Regulatory           |                         122 |                    12 |                  6 |                  0.0983607 | SPARSE               |
|                  17 | Gap & pace               | Strategy & technical |                          67 |                     6 |                  3 |                  0.0895522 | SPARSE               |
|                  18 | Team dynamics            | Narrative & meta     |                         600 |                    67 |                 10 |                  0.111667  | WELL_REPRESENTED     |
|                  19 | Race lead holding        | Race dynamics        |                         115 |                     8 |                  7 |                  0.0695652 | SPARSE               |
|                  20 | Wing & car damage        | Strategy & technical |                          61 |                     4 |                  3 |                  0.0655738 | EFFECTIVELY_ABSENT   |
|                  21 | Braking & lockups        | Race dynamics        |                         128 |                    13 |                  9 |                  0.101562  | SPARSE               |
|                  22 | Wheel-to-wheel           | Race dynamics        |                          71 |                     7 |                  5 |                  0.0985915 | SPARSE               |
|                  23 | Circuit action           | Race dynamics        |                          72 |                    10 |                  7 |                  0.138889  | SPARSE               |
|                  24 | Inter-car distance       | Strategy & technical |                          60 |                     1 |                  1 |                  0.0166667 | EFFECTIVELY_ABSENT   |
|                  25 | Driver & team references | Narrative & meta     |                          69 |                    13 |                  9 |                  0.188406  | SPARSE               |
|                  26 | Defending & attacking    | Race dynamics        |                         110 |                    10 |                  6 |                  0.0909091 | SPARSE               |
|                  27 | Overtake battles         | Race dynamics        |                         762 |                    68 |                 10 |                  0.0892388 | WELL_REPRESENTED     |
|                  28 | Corner battles           | Race dynamics        |                          82 |                     4 |                  3 |                  0.0487805 | EFFECTIVELY_ABSENT   |
|                  29 | Corner specific moves    | Race dynamics        |                          60 |                     8 |                  3 |                  0.133333  | SPARSE               |
|                  30 | Lap pace & timing        | Strategy & technical |                         136 |                    14 |                  7 |                  0.102941  | SPARSE               |
|                  31 | Race leader changes      | Race dynamics        |                         162 |                    13 |                  6 |                  0.0802469 | SPARSE               |
|                  32 | Front position changes   | Race dynamics        |                          45 |                     2 |                  2 |                  0.0444444 | EFFECTIVELY_ABSENT   |
|                  33 | Field position changes   | Race dynamics        |                         122 |                     9 |                  5 |                  0.0737705 | SPARSE               |

## `semantic_lexical_correspondence.csv`
34 rows x 49 columns.

Columns: `Unnamed: 0`, `F0`, `F1`, `F2`, `F3`, `F4`, `F5`, `F6`, `F7`, `F8`, `F9`, `F10`, `F11`, `F12`, `F13`, `F14`, `F15`, `F16`, `F17`, `F18`, `F19`, `F20`, `F21`, `F22`, `F23`, `F24`, `F25`, `F26`, `F27`, `F28`, `F29`, `F30`, `F31`, `F32`, `F33`, `F34`, `F35`, `F36`, `F37`, `F38`, `F39`, `F40`, `F41`, `F42`, `F43`, `F44`, `F45`, `F46`, `F47`
