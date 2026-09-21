"""Sensitivity views and the extended robustness experiments.

Two different things live here.

`settings_explored` collects, from what the run already computed, the settings
each reported result was checked against: granularity settings swept, seeds,
assignment thresholds, null replicate counts.

`extended_experiments` runs the four large robustness experiments, each from its
own frozen analytical input and its own deterministic replicate roster:

* resampling stability, 200 subsample replicates re-projected and re-clustered;
* the held-out replication factorial, 50 splits x 2 embedding realisations x 2
  training granularities, which asks whether the single reference split is
  representative;
* alternative geometry, 100 replicates under a different geometry and linkage;
* domain-membership robustness, every primary-domain category dropped in turn
  plus an exploratory reassignment of the boundary categories.

The reference-split held-out diagnostic is not replaced by the factorial. That
diagnostic supplies the per-category replication values; the factorial supplies
the distribution over splits. Both are reported, separately.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import clustering, frozen_f1, heldout_factorial, monte_carlo_diagnostics
from . import paired_register

MC = monte_carlo_diagnostics
HF = heldout_factorial


def settings_explored(cfg, sweep, second=None, paired=None) -> dict:
    """The settings each reported result was checked against, from this run."""
    out = {"granularity_settings_swept": int(len(sweep)),
           "granularity_plateau": clustering.plateau_summary(
               sweep, cfg.min_cluster_size)}
    if second is not None:
        out["second_register_settings_swept"] = int(len(second["sweep"]))
        out["second_register_seeds"] = second["n_seeds"]
        out["second_register_degenerate_seeds"] = second["n_degenerate_seeds"]
    if paired is not None:
        out["assignment_thresholds"] = list(paired["tau_sensitivity"])
        out["leave_one_event_out_stable"] = {
            k: v["loo_sign_stable"] for k, v in paired["headline"].items()}
        out["matched_length_null_replicates"] = \
            frozen_f1.MATCHED_LENGTH_N_REPLICATES
        if len(paired["matched_length_null_contiguous"]):
            out["contiguous_block_settings"] = frozen_f1.CONTIGUOUS_BLOCK_KS
    return out


def extended_experiments(cfg, frozen_dir, docs, masked, labels,
                         category_names=None) -> dict:
    """Resampling, held-out factorial and alternative geometry, from frozen input.

    Each experiment is keyed to a specific embedding realisation whose hash is
    recorded in the manifest; a mismatch stops the run rather than quietly
    producing a different experiment.
    """
    frozen_dir = Path(frozen_dir)
    manifest = json.loads(
        (frozen_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    def frozen(name):
        m = manifest["inputs"][name]
        return (MC.require_frozen_input(frozen_dir / name, m["array_sha256"],
                                        m["shape"], m["dtype"]),
                m["array_sha256"])

    emb_a, sha_a = frozen("highlight_embedding_realisation_A.npy")
    emb_b, sha_b = frozen("highlight_embedding_realisation_B.npy")
    print("frozen analytical inputs verified against their recorded hashes")

    out = {}
    rs_rows = MC.resampling_stability_mc200(cfg, emb_a, labels)
    rs_cat = MC.category_summary(rs_rows, category_names)
    out["resampling"] = {
        "run_records": MC.run_records(rs_rows, "resampling_stability"),
        "category_records": MC.category_records(rs_rows),
        "category_summary": rs_cat,
        "summary": MC.experiment_summary(rs_rows, rs_cat, "resampling_stability",
                                         MC.RESAMPLE_HISTORICAL_PREFIX)}

    ag_rows = MC.alternative_geometry_mc100(cfg, emb_a, labels)
    ag_cat = MC.category_summary(ag_rows, category_names)
    out["geometry"] = {
        "run_records": MC.run_records(ag_rows, "alternative_geometry"),
        "category_summary": ag_cat,
        "convergence": MC.convergence(ag_rows, (5, 10, 20, 50, 75, 100),
                                      category_names),
        "summary": MC.experiment_summary(ag_rows, ag_cat, "alternative_geometry",
                                         MC.GEOMETRY_HISTORICAL_PREFIX)}

    fx_docs = docs.copy()
    fx_docs["masked"] = masked["masked"]
    fx_docs["session"] = docs["session"].astype(str)
    fx_rows = HF.heldout_replication_factorial(
        cfg, {"A": (emb_a, sha_a), "B": (emb_b, sha_b)}, fx_docs, labels)
    out["factorial"] = {
        "run_records": HF.run_records(fx_rows),
        "cell_summary": HF.cell_summary(fx_rows),
        "category_summary": HF.category_summary(fx_rows, category_names),
        "degeneracy_summary": HF.degeneracy_summary(fx_rows),
        "embedding_sensitivity": HF.embedding_sensitivity(fx_rows),
        "granularity_sensitivity": HF.granularity_sensitivity(fx_rows),
        "seed42_location": HF.reference_seed_location(fx_rows),
        "threshold_margins": HF.threshold_margins(fx_rows),
        "summary": HF.factorial_summary(fx_rows)}
    for k in ("resampling", "geometry", "factorial"):
        print(f"  {k}: {json.dumps(out[k]['summary'])[:150]}")
    return out


def domain_membership(paired, category_domains, category_names=None) -> dict:
    """Systematic leave-one-category-out plus exploratory boundary reassignment."""
    loo, boundary = paired_register.domain_membership_robustness(
        paired["prevalence"], paired["manifest"], category_domains,
        category_names)
    print(f"  domain-membership robustness: {len(loo) - 1} systematic removals, "
          f"{len(boundary) - 1} exploratory reassignments")
    return {"leave_one_category_out": loo, "boundary_reassignment": boundary}
