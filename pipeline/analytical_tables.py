"""Consolidated analytical tables built from a run's own artefacts.

Each builder takes data frames produced by the current run and returns one
table. Nothing here reads a finished historical table and re-saves it, and
nothing carries publication numbering: the caller decides the content name and
the destination.

``manuscript_tables``/``manuscript_tables_final`` assemble the archived study
tables and resolve their own frozen inputs; these are the general equivalents
used when a notebook run must consolidate what it just computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frozen_f1 as F


def corpus_roles(cfg, docs, labels, cfg2=None, docs2=None, second=None,
                 matched_events=None) -> pd.DataFrame:
    """Register, role and size for each corpus the run actually loaded."""
    n_matched = len(matched_events or {})
    rows = [{
        "register": cfg.corpus_name,
        "role_in_analysis": "thematic discovery and diagnostic characterisation",
        "events": int(docs["session"].nunique()),
        "sentences": int(len(docs)),
        "coverage_profile": "broad across events, comparatively shallow per event",
        "matched_event_role": (f"{n_matched} events used in the paired comparison"
                               if n_matched else "no paired comparison in this run"),
        "status": "analysed"}]
    if docs2 is not None:
        rows.append({
            "register": cfg2.corpus_name,
            "role_in_analysis": "independent thematic reconstruction",
            "events": int(docs2["session"].nunique()),
            "sentences": int(len(docs2)),
            "coverage_profile": ("narrow across events, comparatively deep "
                                 "per event"),
            "matched_event_role": (f"the same {n_matched} events used in the "
                                   f"paired comparison" if n_matched
                                   else "no paired comparison in this run"),
            "status": ("analysed" if second is not None
                       else "loaded, not reconstructed")})
    return pd.DataFrame(rows)


def correspondence_criteria(null_percentile: int | None = None) -> pd.DataFrame:
    """The four correspondence outcomes, stated from the classification rule.

    Assembled from the thresholds the classifier uses so the description cannot
    drift away from the implementation in ``cross_register``.
    """
    pctl = int(null_percentile or F.CORRESPONDENCE_NULL_PERCENTILE)
    above = (f"a second-register cluster whose centroid cosine exceeds this "
             f"category's own size-preserving null threshold (the {pctl}th "
             f"percentile of {F.CORRESPONDENCE_N_NULL} replicates)")
    rows = [
        ("NON_RECOVERY", "no cluster is above threshold", 0,
         "no supported counterpart is recovered"),
        ("SINGLE_MATCH",
         "exactly one cluster is above threshold, and that cluster is above "
         "threshold for this category only", 1,
         "one supported counterpart"),
        ("SPLIT", "two or more clusters are above threshold", "2 or more",
         "one primary category corresponds to several second-register clusters"),
        ("MERGE",
         "exactly one cluster is above threshold, and that cluster is also "
         "above threshold for at least one other primary category", 1,
         "several primary categories converge on one second-register cluster"),
    ]
    df = pd.DataFrame(rows, columns=["correspondence_outcome",
                                     "operational_criterion",
                                     "clusters_above_threshold",
                                     "interpretation"])
    df["above_threshold_means"] = above
    return df


def correspondence_summary(corr: pd.DataFrame) -> pd.DataFrame:
    """Counts per outcome, overall and among well-represented categories."""
    order = ["SINGLE_MATCH", "SPLIT", "MERGE", "NON_RECOVERY"]
    total = corr.relation.value_counts()
    out = pd.DataFrame({"correspondence_outcome": order})
    out["categories"] = out.correspondence_outcome.map(total).fillna(0).astype(int)
    if "availability_class" in corr.columns:
        well = corr[corr.availability_class == "WELL_REPRESENTED"]
        out["well_represented"] = (out.correspondence_outcome
                                   .map(well.relation.value_counts())
                                   .fillna(0).astype(int))
    return out


def lexical_signature_summary(enr: pd.DataFrame, labels: np.ndarray,
                              category_name: dict | None = None,
                              top_n: int = 3) -> pd.DataFrame:
    """Per category: how many terms survive FDR, and the most significant ones."""
    category_name = category_name or {}
    term_col = next(c for c in ("term", "ngram", "token") if c in enr.columns)
    rank_col = next((c for c in ("p_value", "pval", "p", "q_value", "qval")
                     if c in enr.columns), None)
    rows = []
    for c in sorted(int(v) for v in set(labels.tolist()) if v != -1):
        sub = enr[enr.cluster == c]
        if rank_col is not None:
            sub = sub.sort_values(rank_col)
        rows.append({
            "category_id": c,
            "category_label": category_name.get(c, str(c)),
            "n_significant_terms": int(len(sub)),
            "most_significant_terms": ", ".join(
                str(t) for t in sub[term_col].head(top_n))})
    return pd.DataFrame(rows)


def diagnostic_detail(d1, d2, d3, d4, category_name=None,
                      category_domain=None) -> pd.DataFrame:
    """Per-category values from all four dimensions, kept separate, never summed."""
    out = (d1.merge(d2, on="cluster", how="outer")
             .merge(d3, on="cluster", how="outer")
             .merge(d4, on="cluster", how="outer")
             .rename(columns={"cluster": "category_id"}))
    if category_name:
        out.insert(1, "category_label",
                   out.category_id.map(category_name).fillna(
                       out.category_id.astype(str)))
    if category_domain:
        out.insert(2, "domain", out.category_id.map(category_domain).fillna(""))
    return out.sort_values("category_id").reset_index(drop=True)


def matched_event_comparison(manifest: pd.DataFrame, prevalence: pd.DataFrame,
                             nulls: pd.DataFrame, category_domain: dict,
                             domain: str, tau: float) -> pd.DataFrame:
    """One row per matched event: both registers, the difference, its own null.

    Consolidated so the per-event result does not have to be reassembled by
    joining several files.
    """
    prev = prevalence[prevalence.threshold == tau].assign(
        domain=lambda f: f.category_id.map(category_domain))
    share = (prev[prev.domain == domain]
             .groupby(["event_id", "register"], as_index=False)
             .agg(n=("n_category", "sum"), assigned=("n_assigned", "max")))
    share["share"] = share.n / share.assigned
    wide = share.pivot(index="event_id", columns="register", values="share")
    # The register labels are read by name, never by column position: the pivot
    # sorts them alphabetically, which puts the second register first.
    regs = list(wide.columns)
    primary = next((r for r in regs if "HIGH" in str(r).upper()), regs[0])
    secondary = next((r for r in regs if r != primary), regs[-1])

    man = manifest.set_index("event_id")
    nl = nulls[(nulls.domain == domain) & (nulls.metric == "assigned")]
    nl = nl.set_index("event_id") if len(nl) else None

    def count(event, column):
        if event in man.index and column in man.columns:
            return int(man.loc[event, column])
        return None

    rows = []
    for event in wide.index:
        r = {"event_id": event,
             "primary_sentences": count(event, "n_primary_sentences"),
             "secondary_sentences": count(event, "n_second_sentences"),
             f"{domain} share (primary)": round(float(wide.loc[event, primary]), 6),
             f"{domain} share (secondary)": round(float(wide.loc[event, secondary]), 6),
             "difference": round(float(wide.loc[event, primary]
                                       - wide.loc[event, secondary]), 6)}
        if nl is not None and event in nl.index:
            r["null_lower"] = round(float(nl.loc[event, "null_p2.5"]), 6)
            r["null_upper"] = round(float(nl.loc[event, "null_p97.5"]), 6)
            r["outside_own_null"] = bool(nl.loc[event, "outside_central_95"])
        rows.append(r)
    return pd.DataFrame(rows)


def assignment_threshold_sensitivity(effects: pd.DataFrame, category_domain: dict,
                                     domain: str) -> pd.DataFrame:
    """The domain effect at each assignment threshold the run evaluated."""
    e = effects.assign(domain=lambda f: f.category_id.map(category_domain))
    e = e[e.domain == domain]
    rows = []
    for (tau, metric), g in e.groupby(["threshold", "metric"]):
        rows.append({
            "assignment_threshold": float(tau),
            "denominator": metric,
            "categories": int(g.category_id.nunique()),
            "median_delta": round(float(g.median_delta.median()), 6),
            "mean_delta": round(float(g.mean_delta.mean()), 6),
            "categories_negative": int((g.median_delta < 0).sum()),
            "categories_positive": int((g.median_delta > 0).sum()),
            "direction": ("negative" if g.median_delta.median() < 0
                          else "positive" if g.median_delta.median() > 0
                          else "no difference")})
    return pd.DataFrame(rows).sort_values(
        ["denominator", "assignment_threshold"]).reset_index(drop=True)


def compositional_sensitivity(summary: pd.DataFrame,
                              threshold_table: pd.DataFrame | None = None,
                              domain: str = "") -> pd.DataFrame:
    """Every compositional view the run produced, in one table.

    Each row is one representation with its effect, direction across events,
    exact sign-test p-value and leave-one-event-out range.
    """
    keep = ["metric", "median", "n_negative", "n_positive", "sign_test_p",
            "loo_sign_stable", "loo_median_min", "loo_median_max", "q1", "q3"]
    d = summary[[c for c in keep if c in summary.columns]].copy()
    d = d.rename(columns={"metric": "representation", "median": "effect",
                          "q1": "iqr_low", "q3": "iqr_high"})
    n_events = (d.n_negative + d.n_positive).max()
    d["direction_across_events"] = [
        f"{int(neg)}/{int(neg + pos)} negative" if neg >= pos
        else f"{int(pos)}/{int(neg + pos)} positive"
        for neg, pos in zip(d.n_negative, d.n_positive)]
    d["analysis"] = "pre-specified compositional representation"
    if threshold_table is not None and len(threshold_table):
        extra = pd.DataFrame({
            "representation": [f"assignment threshold {t:.2f} ({m})" for t, m in
                               zip(threshold_table.assignment_threshold,
                                   threshold_table.denominator)],
            "effect": threshold_table.median_delta,
            "direction_across_events": threshold_table.direction,
            "analysis": "assignment-threshold sensitivity"})
        d = pd.concat([d, extra], ignore_index=True)
    d.attrs["n_events"] = int(n_events) if pd.notna(n_events) else 0
    d.attrs["domain"] = domain
    return d


PAIRED_TABLES = {
    "matched_event_manifest": "paired_event_manifest.csv",
    "matched_event_coverage": "paired_register_coverage.csv",
    "matched_event_category_shares": "paired_category_prevalence.csv",
    "per_category_composition_effects": "paired_category_effects.csv",
    "leave_one_event_out": "paired_leave_one_out.csv",
    "compositional_event_differences": "compositional_domain_effects.csv",
    "compositional_robustness_summary": "compositional_strategy_summary.csv",
    "matched_length_null_intervals": "matched_length_null_summary.csv",
    "matched_length_null_by_event": "matched_length_null_event.csv",
    "matched_length_null_contiguous_variant": "matched_length_null_contiguous.csv",
}

ROBUSTNESS_TABLES = {
    "resampling": [("run_records", "resampling_run_records"),
                   ("category_records", "resampling_category_records"),
                   ("category_summary", "resampling_category_summary")],
    "factorial": [("run_records", "heldout_factorial_run_records"),
                  ("cell_summary", "heldout_factorial_cell_summary"),
                  ("category_summary", "heldout_factorial_category_summary"),
                  ("degeneracy_summary", "heldout_factorial_degeneracy_summary"),
                  ("embedding_sensitivity", "heldout_embedding_sensitivity"),
                  ("granularity_sensitivity", "heldout_granularity_sensitivity"),
                  ("seed42_location", "heldout_reference_split_location"),
                  ("threshold_margins", "heldout_threshold_margins")],
    "geometry": [("run_records", "alternative_geometry_run_records"),
                 ("category_summary", "alternative_geometry_category_summary"),
                 ("convergence", "alternative_geometry_convergence")],
    "domain_membership": [
        ("leave_one_category_out", "systematic_leave_one_strategy_category_out"),
        ("boundary_reassignment", "exploratory_boundary_reassignment")],
}


def _canonical_diagnostic_sources(diag, ext):
    """Prefer the replicate experiments over their single-pass equivalents.

    The reported resampling and geometry values come from the canonical
    Monte-Carlo producers wherever those apply to the corpus; the lighter
    single-pass interfaces are used only where they do not. The held-out column
    stays the REFERENCE-SPLIT value in every case.
    """
    d2, d4 = diag["resampling_stability"], diag["geometry_separability"]
    if "resampling" in ext:
        d2 = (ext["resampling"]["category_summary"]
              .rename(columns={"category_id": "cluster",
                               "mean_jaccard": "axis2_score"})
              [["cluster", "axis2_score"]])
    if "geometry" in ext:
        d4 = (ext["geometry"]["category_summary"]
              .rename(columns={"category_id": "cluster",
                               "mean_jaccard": "axis4_score"})
              [["cluster", "axis4_score"]])
    return d2, d4


def write_all(release_root, cfg, docs, labels, sweep, enr, diag,
              category_names=None, category_domains=None, ext=None,
              cfg2=None, docs2=None, second=None, corr=None, paired=None,
              matched_events=None, cross_register_dir=None) -> dict:
    """Write every analytical table this run supports; return name to path.

    This is the single producer of the analytical tables. Each artifact is named
    for its scientific content, and each is written exactly once.
    """
    from pathlib import Path

    from . import clustering, diagnostics, enrichment as enrichment_mod

    release_root = Path(release_root)
    ext = ext or {}
    category_names = dict(category_names or {})
    category_domains = dict(category_domains or {})
    domains_apply = diag["domain_map_applies"]
    td = Path(cfg.tables_dir)
    xt = Path(cross_register_dir) if cross_register_dir else None
    tables = {}

    def put(key, df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        tables[key] = str(path.relative_to(release_root))
        return df

    def register(key, path):
        if Path(path).exists():
            tables[key] = str(Path(path).relative_to(release_root))

    # ---- corpus, granularity and taxonomy ---------------------------------
    rows = [{"corpus": cfg.corpus_name, "role": "primary", "sentences": len(docs),
             "sessions": int(docs["session"].nunique()),
             "categories": int(clustering.n_clusters(labels)),
             "unclustered_fraction": round(float(clustering.noise_rate(labels)), 4),
             "minimum_cluster_size": cfg.min_cluster_size}]
    if second is not None:
        rows.append({"corpus": cfg2.corpus_name, "role": "secondary",
                     "sentences": len(docs2),
                     "sessions": int(docs2["session"].nunique()),
                     "categories": int(second["n_clusters"]),
                     "unclustered_fraction": round(float(second["noise_fraction"]), 4),
                     "minimum_cluster_size": second["selected_min_cluster_size"]})
    elif docs2 is not None:
        rows.append({"corpus": cfg2.corpus_name, "role": "secondary",
                     "sentences": len(docs2),
                     "sessions": int(docs2["session"].nunique()),
                     "categories": None, "unclustered_fraction": None,
                     "minimum_cluster_size": None})
    put("corpus_summary", pd.DataFrame(rows), td / "corpus_summary.csv")

    sweep_out = sweep.copy()
    sweep_out["selected_setting"] = (sweep_out[sweep_out.columns[0]]
                                     == cfg.min_cluster_size)
    put("granularity_sweep_primary", sweep_out, td / "granularity_sweep_primary.csv")

    sizes = pd.Series(labels[labels >= 0]).value_counts().sort_index()
    inv = pd.DataFrame({"category_id": sizes.index.astype(int),
                        "sentences": sizes.to_numpy()})
    inv["share_of_assigned"] = (inv.sentences / inv.sentences.sum()).round(6)
    inv["label"] = (inv.category_id.map(category_names) if category_names
                    else inv.category_id.astype(str))
    inv["domain"] = inv.category_id.map(category_domains) if domains_apply else ""
    inv["top_terms"] = [", ".join(enrichment_mod.top_terms(enr, int(c)))
                        for c in inv.category_id]
    put("category_inventory", inv, td / "category_inventory.csv")

    put("corpus_roles_and_characteristics",
        corpus_roles(cfg, docs, labels, cfg2, docs2, second, matched_events),
        td / "corpus_roles_and_characteristics.csv")

    # ---- lexical characterisation -----------------------------------------
    put("category_lexical_enrichment_summary",
        enrichment_mod.enrichment_summary(enr, labels),
        td / "category_lexical_enrichment_summary.csv")
    put("category_term_enrichment_records", enr,
        td / "category_term_enrichment_records.csv")
    put("category_lexical_signature_summary",
        lexical_signature_summary(enr, labels, category_names),
        td / "category_lexical_signature_summary.csv")

    # ---- diagnostic dimensions --------------------------------------------
    d1 = diag["alternative_model_recovery"]
    d3 = diag["heldout_replication"]
    d2, d4 = _canonical_diagnostic_sources(diag, ext)

    # The per-category profile is finalised here, once, from the same sources as
    # the reported per-category table. The figure stage reads it, never writes it.
    profile = diag["profile"].copy()
    if "resampling" in ext:
        src = ext["resampling"]["category_summary"].set_index("category_id")
        profile["resampling_stability"] = profile.cluster_id.map(src.mean_jaccard)
    if "geometry" in ext:
        src = ext["geometry"]["category_summary"].set_index("category_id")
        profile["geometry_separability"] = profile.cluster_id.map(src.mean_jaccard)
    put("category_diagnostic_profiles", profile, diag["profile_path"])

    put("diagnostic_run_summary", pd.DataFrame([
        {"dimension": "alternative_model_recovery",
         "categories_scored": int(d1.cluster.nunique()),
         "configurations": len(diagnostics.expected_judges(cfg)),
         "note": "fraction of pre-specified alternative configurations that "
                 "recover the category"},
        {"dimension": "resampling_stability",
         "categories_scored": int(d2.cluster.nunique()), "configurations": None,
         "note": "resampling runs, conditional on the recorded software "
                 "environment"},
        {"dimension": "heldout_replication",
         "categories_scored": int(d3.cluster.nunique()), "configurations": None,
         "note": "no estimate where the training split does not reconstruct the "
                 "category"},
        {"dimension": "geometry_separability",
         "categories_scored": int(d4.cluster.nunique()), "configurations": None,
         "note": "continuous overlap statistic, comparable at a fixed "
                 "per-category sample size"}]),
        td / "diagnostic_run_summary.csv")

    put("per_category_diagnostic_detail",
        diagnostic_detail(d1, d2, d3, d4, category_names,
                          category_domains if domains_apply else None),
        td / "per_category_diagnostic_detail.csv")
    put("alternative_model_recovery_per_category",
        diag["alternative_model_recovery_detail"],
        td / "alternative_model_recovery_per_category.csv")
    put("resampling_stability_per_category", d2,
        td / "resampling_stability_per_category.csv")
    put("heldout_replication_per_category", d3,
        td / "heldout_replication_per_category.csv")
    put("geometry_separability_per_category", d4,
        td / "geometry_separability_per_category.csv")

    # ---- second register and correspondence -------------------------------
    if second is not None:
        put("secondary_register_reconstruction_summary", pd.DataFrame([{
            "corpus": cfg2.corpus_name,
            "selected_minimum_cluster_size": second["selected_min_cluster_size"],
            "clusters": second["n_clusters"],
            "unclustered_fraction": round(float(second["noise_fraction"]), 4),
            "dbcv": second["dbcv"], "silhouette": second["silhouette"],
            "clusters_with_significant_vocabulary":
                second["clusters_with_significant_vocabulary"],
            "cluster_count_plateau_exists": second["cluster_count_plateau_exists"],
            "settings_swept": int(len(second["sweep"])),
            "seeds": second["n_seeds"],
            "degenerate_seeds": second["n_degenerate_seeds"]}]),
            xt / "secondary_register_reconstruction_summary.csv")
        register("secondary_register_granularity_sweep",
                 xt / "second_register_granularity_sweep.csv")
        register("secondary_register_clusters", xt / "second_register_clusters.csv")
        register("secondary_register_seed_stability",
                 xt / "second_register_seed_stability.csv")

    if corr is not None:
        register("cross_register_similarity_matrix",
                 xt / "cross_register_similarity_matrix.csv")
        register("cross_register_category_outcomes",
                 xt / "cross_register_correspondence.csv")
        register("cross_register_null_thresholds", xt / "cross_register_null.csv")
        register("category_availability_control",
                 xt / "matched_event_availability.csv")
        put("correspondence_outcome_criteria", correspondence_criteria(),
            xt / "correspondence_outcome_criteria.csv")
        put("cross_register_correspondence_summary",
            correspondence_summary(corr["correspondence"]),
            xt / "cross_register_correspondence_summary.csv")

    # ---- matched-event composition ----------------------------------------
    if paired is not None:
        for key, fname in PAIRED_TABLES.items():
            register(key, xt / fname)
        put("matched_event_category_differences", paired["event_deltas"],
            xt / "matched_event_category_differences.csv")
        tau, domain = paired["tau_primary"], paired["primary_domain"]
        put("exact_matched_event_comparison", matched_event_comparison(
            paired["manifest"], paired["prevalence"],
            paired["matched_length_null_domain"], category_domains, domain, tau),
            xt / "exact_matched_event_comparison.csv")
        thresholds = assignment_threshold_sensitivity(
            paired["category_effects"], category_domains, domain)
        put("assignment_threshold_sensitivity", thresholds,
            xt / "assignment_threshold_sensitivity.csv")
        put("complete_compositional_sensitivity", compositional_sensitivity(
            paired["compositional_summary"], thresholds, domain),
            xt / "complete_compositional_sensitivity.csv")

    # ---- extended robustness experiments ----------------------------------
    robustness_dir = td / "robustness"
    for family, items in ROBUSTNESS_TABLES.items():
        if family in ext:
            for key, name in items:
                put(name, ext[family][key], robustness_dir / f"{name}.csv")
    return tables
