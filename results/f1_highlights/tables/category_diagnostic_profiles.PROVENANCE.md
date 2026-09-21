# category_diagnostic_profiles.csv - provenance

This file is not tracked in git. It was overwritten by a pipeline rerun
and restored from the values the compiled manuscript displays in
Figure 4, which predates the overwrite.

| column | source | precision |
|---|---|---|
| alternative_model_recovery | canonical A1 score | full |
| resampling_stability | canonical A2 score | full |
| geometry_separability | canonical A4 score | full |
| heldout_replication | manuscript Figure 4 | 2 dp |

The first three were verified cell by cell against Figure 4: all 34
categories agree at the displayed precision. Held-out replication is
computed per run and no full-precision artefact survives, so it carries
the displayed precision, which is what every consumer in this release
uses. Recomputing this profile under current library versions does NOT
reproduce it: the held-out and alternative-model dimensions are
version-sensitive, as pipeline/config.py REPORTED_ENVIRONMENT records.
