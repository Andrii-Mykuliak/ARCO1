# Supplement E — Paired Cross Register

## `paired_event_manifest.csv`
10 rows x 10 columns.

Columns: `pair_id`, `event_id`, `season`, `event_name`, `highlight_source`, `fullrace_source`, `n_highlight_sentences`, `n_fullrace_sentences`, `match_type`, `notes`

## `paired_category_prevalence.csv`
2040 rows x 11 columns.

Columns: `pair_id`, `event_id`, `threshold`, `category_id`, `category_name`, `register`, `n_category`, `n_total`, `n_assigned`, `p_all`, `p_assigned`

## `paired_domain_prevalence.csv`
24 rows x 9 columns.

Columns: `threshold`, `domain`, `metric`, `n_events`, `mean_delta`, `median_delta`, `sd`, `n_positive`, `n_negative`

|   threshold | domain               | metric   |   n_events |   mean_delta |   median_delta |        sd |   n_positive |   n_negative |
|------------:|:---------------------|:---------|-----------:|-------------:|---------------:|----------:|-------------:|-------------:|
|        0.35 | Narrative & meta     | all      |         10 |   0.0673761  |     0.0521655  | 0.0643339 |            9 |            1 |
|        0.35 | Narrative & meta     | assigned |         10 |   0.0202984  |     0.00244385 | 0.0623258 |            5 |            5 |
|        0.35 | Race dynamics        | all      |         10 |   0.127472   |     0.140566   | 0.0607448 |           10 |            0 |
|        0.35 | Race dynamics        | assigned |         10 |   0.0661416  |     0.0679999  | 0.059969  |            8 |            2 |
|        0.35 | Regulatory           | all      |         10 |   0.0131633  |     0.00243721 | 0.0382548 |            6 |            4 |
|        0.35 | Regulatory           | assigned |         10 |   0.00408925 |    -0.00535805 | 0.0370832 |            4 |            6 |
|        0.35 | Strategy & technical | all      |         10 |  -0.04899    |    -0.0463073  | 0.053187  |            1 |            9 |
|        0.35 | Strategy & technical | assigned |         10 |  -0.0905293  |    -0.0983061  | 0.0512422 |            1 |            9 |
|        0.4  | Narrative & meta     | all      |         10 |   0.0786306  |     0.0698183  | 0.0620062 |            9 |            1 |
|        0.4  | Narrative & meta     | assigned |         10 |   0.0203257  |     0.00543917 | 0.0599415 |            5 |            5 |
|        0.4  | Race dynamics        | all      |         10 |   0.144312   |     0.154124   | 0.0577981 |           10 |            0 |
|        0.4  | Race dynamics        | assigned |         10 |   0.069208   |     0.0800807  | 0.0596178 |            8 |            2 |
|        0.4  | Regulatory           | all      |         10 |   0.0147565  |     0.00595435 | 0.0362037 |            6 |            4 |
|        0.4  | Regulatory           | assigned |         10 |   0.00261772 |    -0.00585462 | 0.0334511 |            4 |            6 |
|        0.4  | Strategy & technical | all      |         10 |  -0.0380626  |    -0.0332051  | 0.0492803 |            1 |            9 |
|        0.4  | Strategy & technical | assigned |         10 |  -0.0921514  |    -0.097257   | 0.0462664 |            0 |           10 |
|        0.45 | Narrative & meta     | all      |         10 |   0.0944436  |     0.0911328  | 0.06138   |           10 |            0 |
|        0.45 | Narrative & meta     | assigned |         10 |   0.0257057  |     0.00768095 | 0.0635626 |            6 |            4 |
|        0.45 | Race dynamics        | all      |         10 |   0.157553   |     0.1656     | 0.0624123 |           10 |            0 |
|        0.45 | Race dynamics        | assigned |         10 |   0.0695723  |     0.0862755  | 0.0623904 |            8 |            2 |
|        0.45 | Regulatory           | all      |         10 |   0.0105328  |     0.00066493 | 0.0287276 |            6 |            4 |
|        0.45 | Regulatory           | assigned |         10 |  -0.00436104 |    -0.0126026  | 0.0274606 |            4 |            6 |
|        0.45 | Strategy & technical | all      |         10 |  -0.0247684  |    -0.0277114  | 0.0552915 |            3 |            7 |
|        0.45 | Strategy & technical | assigned |         10 |  -0.0909169  |    -0.0987833  | 0.0511206 |            1 |            9 |

## `matched_length_null_summary.csv`
80 rows x 17 columns.

Columns: `event_id`, `domain`, `metric`, `H_r`, `F_r`, `observed_effect`, `null_mean`, `null_sd`, `null_p2.5`, `null_p50`, `null_p97.5`, `p_one_sided_depletion`, `p_one_sided_enrichment`, `p_two_sided`, `z_vs_null`, `outside_central_95`, `below_null_p2.5`

## `compositional_strategy_summary.csv`
12 rows x 16 columns.

Columns: `metric`, `column`, `median`, `mean`, `sd`, `q1`, `q3`, `iqr`, `min`, `max`, `n_negative`, `n_positive`, `sign_test_p`, `loo_sign_stable`, `loo_median_min`, `loo_median_max`

## `paired_leave_one_out.csv`
680 rows x 7 columns.

Columns: `category_id`, `category_name`, `omitted_event`, `metric`, `n_events_used`, `median_delta`, `sign`
