"""Restructured main Tables 1-3, plus the supplementary tables they displace.

Main-text tables carry the substantive result only. Everything that is design
detail, per-cell distribution or sensitivity arithmetic moves to the supplement.
Every cell is read from a frozen artefact; nothing is transcribed, and no
inference is recomputed.

Sources
-------
tables/category_diagnostic_profiles.csv           alternative-model recovery
diagnostics/resampling_stability_mc200_*.csv      resampling stability
diagnostics/heldout_race_replication_*.json/.csv  held-out replication
diagnostics/alternative_geometry_mc100_*.csv      alternative geometry
f1_cross_register/tables/cross_register_*.csv     correspondence + null
f1_cross_register/tables/compositional_*.csv      paired composition
f1_cross_register/tables/matched_length_null_*    matched-length null
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .manuscript_tables import _write

STRATEGY = "Strategy & technical"


def _q(v):
    return f"{np.median(v):.3f} (IQR {np.quantile(v, .25):.3f}-{np.quantile(v, .75):.3f})"


# ------------------------------------------------------------------ main 1 --
def main_table1(rel: Path, out: Path) -> pd.DataFrame:
    prof = pd.read_csv(rel / "results/f1_highlights/tables/category_diagnostic_profiles.csv")
    dg = rel / "results/f1_highlights/diagnostics"
    cl = json.loads((dg / "diagnostic_dimensions_closure_summary.json").read_text("utf-8"))
    res = pd.read_csv(dg / "resampling_stability_mc200_category_summary.csv")
    geo = pd.read_csv(dg / "alternative_geometry_mc100_category_summary.csv")
    fac = json.loads((dg / "heldout_race_replication_factorial_summary.json").read_text("utf-8"))

    k = np.round(prof.alternative_model_recovery.to_numpy() * 7).astype(int)
    r = res["degenerate_excluded_mean"].dropna()
    g = geo["new_mean_100"].dropna()
    meds = sorted({int(c["median"]) for c in fac["distributions"]})

    rows = [
        {"Diagnostic dimension": "Alternative-model recovery",
         "Design": f"{cl['alternative_model_recovery']['n_pipelines']} pre-specified "
                   f"alternative configurations",
         "Main result": f"median {int(np.median(k))} of 7; "
                        f"{int((k >= 3).sum())} of 34 recovered under at least three",
         "Principal limitation": "recovery is near zero for a small minority of "
                                 "categories"},
        {"Diagnostic dimension": "Resampling stability",
         "Design": f"{cl['resampling_stability']['n_resamples']} resamples at "
                   f"{cl['resampling_stability']['subsample_fraction']:.0%}; "
                   f"{cl['resampling_stability']['n_viable']} viable",
         "Main result": f"median {np.median(r):.3f}; "
                        f"{int((r >= .50).sum())} of 34 at or above 0.50",
         "Principal limitation": "environment-sensitive; values are conditional on "
                                 "the recorded software stack"},
        {"Diagnostic dimension": "Held-out race replication",
         "Design": f"{cl['heldout_race_replication']['n_seeds']} seeds x "
                   f"{cl['heldout_race_replication']['n_embedding_caches']} embedding "
                   f"realisations x 2 granularities "
                   f"({cl['heldout_race_replication']['total_runs']} runs)",
         "Main result": f"median {min(meds)}-{max(meds)} categories replicating per "
                        f"run across the four cells",
         "Principal limitation": f"{cl['heldout_race_replication']['degenerate_runs']} "
                                 f"runs degenerate; reported distributionally, not "
                                 f"from any single split"},
        {"Diagnostic dimension": "Alternative-geometry separability",
         "Design": f"{cl['alternative_geometry_separability']['n_replicates']} "
                   f"replicates, {cl['alternative_geometry_separability']['samples_per_category']} "
                   f"sentences per category",
         "Main result": f"median {np.median(g):.3f}; "
                        f"{int((g >= .50).sum())} of 34 at or above 0.50",
         "Principal limitation": "rises with per-category sample size; comparable "
                                 "across categories, not against an external scale"},
    ]
    return _write(pd.DataFrame(rows), out, "main_table1_diagnostic_summary",
                  "The four complementary diagnostic dimensions. Each is reported "
                  "separately; no aggregate score is computed across them. Full "
                  "per-cell distributions are given in the supplement.",
                  "tab:diagnostics")


# ------------------------------------------------------------------ main 2 --
LABEL = {"ONE_TO_ONE": "Single-match", "SPLIT": "Split",
         "MERGE": "Merge", "NONE": "Non-recovery"}
ORDER = ["ONE_TO_ONE", "SPLIT", "MERGE", "NONE"]


def main_table2(rel: Path, out: Path) -> pd.DataFrame:
    xr = rel / "results/f1_cross_register/tables"
    co = pd.read_csv(xr / "cross_register_correspondence.csv")
    nl = pd.read_csv(xr / "cross_register_null.csv")
    well = co[co.availability_class == "WELL_REPRESENTED"]
    n_well = len(well)

    rows = []
    for r in ORDER:
        sub = co[co.relation == r]
        rows.append({
            "Correspondence outcome": LABEL[r],
            "Categories": int(len(sub)),
            "Of the well-represented categories":
                int((well.relation == r).sum()),
        })
    df = pd.DataFrame(rows)
    cap = ("Cross-register correspondence of the 34 highlight categories "
           "against the independently induced full-race structure.")
    return _write(df, out, "main_table2_cross_register_correspondence", cap,
                  "tab:correspondence")


# ------------------------------------------------------------------ main 3 --
def main_table3(rel: Path, out: Path) -> pd.DataFrame:
    xr = rel / "results/f1_cross_register/tables"
    cs = pd.read_csv(xr / "compositional_strategy_summary.csv").set_index("metric")
    mn = pd.read_csv(xr / "matched_length_null_summary.csv")
    a = mn[(mn.metric == "assigned") & (mn.domain == STRATEGY)]
    pe = pd.read_csv(xr / "paired_category_effects.csv")
    n_alt = len(sorted(pe.threshold.unique())) - 1   # alternatives to the primary

    def row(metric, analysis, interp):
        r = cs.loc[metric]
        n = int(r.n_negative)
        return {"Analysis": analysis,
                "Strategy-and-technical result":
                    f"{r['median']:+.3f} (p = {r.sign_test_p:.4f})",
                "Direction across events": f"{n} of 10 negative",
                "Interpretation": interp}

    rows = [
        row("raw proportion, Strategy", "Raw relative share",
            "lower share in highlight commentary in every event"),
        row("B1 logit, Strategy vs rest", "Binary logit",
            "direction holds on the log-odds scale"),
        row("B2 CLR, Strategy & technical", "Centred log-ratio",
            "direction holds under a compositional transform"),
    ]
    ilr = [m for m in cs.index if "ILR" in m.upper() and "Strategy" in m]
    if ilr:
        rows.append(row(ilr[0], "Isometric log-ratio balance",
                        "direction holds on a pre-specified balance"))

    loo = cs.loc["raw proportion, Strategy"]
    rows += [
        {"Analysis": "Assignment-threshold sensitivity",
         "Strategy-and-technical result":
             f"direction preserved at {n_alt} alternative thresholds",
         "Direction across events": "negative at every threshold",
         "Interpretation": "stable across assignment thresholds"},
        {"Analysis": "Leave-one-event-out",
         "Strategy-and-technical result":
             f"median {loo.loo_median_min:+.3f} to {loo.loo_median_max:+.3f}",
         "Direction across events": "sign stable across all omissions",
         "Interpretation": "aggregate direction stable under all event omissions"},
        {"Analysis": "Matched-length event-level comparison",
         "Strategy-and-technical result":
             f"{int(a.outside_central_95.sum())} of 10 outside their own 95% null",
         "Direction across events": "aggregate direction, not every event",
         "Interpretation": "individual-event separation in 4 of 10 events"},
    ]
    rd = [m for m in cs.index if "ILR" in m.upper() and "RaceDyn" in m]
    if rd:
        r = cs.loc[rd[0]]
        rows.append({
            "Analysis": "Race dynamics contrast",
            "Strategy-and-technical result":
                f"{r['median']:+.3f} (p = {r.sign_test_p:.4f})",
            "Direction across events": f"{int(r.n_positive)} of 10 positive",
            "Interpretation": "not independently supported"})

    cap = ("Robustness of the matched-event Strategy-and-technical result, with "
           "the Race dynamics contrast shown for comparison. The inferential unit "
           "is the event.")
    # readability only: the analysis names carry the longest words, so they get
    # the widest column, and the rows are opened up slightly
    return _write(pd.DataFrame(rows), out,
                  "main_table3_composition_robustness", cap, "tab:robustness",
                  col_fracs=[0.275, 0.235, 0.170, 0.290], arraystretch=1.22)


# ----------------------------------------------------------- supplementary --
def supp_diagnostic_detail(rel: Path, out: Path) -> pd.DataFrame:
    dg = rel / "results/f1_highlights/diagnostics"
    fac = json.loads((dg / "heldout_race_replication_factorial_summary.json").read_text("utf-8"))
    res = pd.read_csv(dg / "resampling_stability_mc200_category_summary.csv")
    geo = pd.read_csv(dg / "alternative_geometry_mc100_category_summary.csv")
    deg = {d["cache"] + str(d["mcs"]): d for d in fac["degeneracy_by_cell"]} \
        if "degeneracy_by_cell" in fac else {}

    rows = []
    for c in fac["distributions"]:
        key = f"{c['cache']}{c['mcs']}"
        rows.append({
            "Held-out cell": f"realisation {c['cache']}, granularity {c['mcs']}",
            "Viable runs": c["n_viable"],
            "Median categories": int(c["median"]),
            "IQR": f"{c['iqr_lo']:.0f}-{c['iqr_hi']:.0f}",
            "Degenerate": deg.get(key, {}).get("n_degenerate", ""),
        })
    df = pd.DataFrame(rows)
    _write(df, out, "supp_table_heldout_factorial",
           "Held-out race replication, per cell of the factorial.",
           "tab:supp-heldout")

    r = res["degenerate_excluded_mean"].dropna()
    g = geo["new_mean_100"].dropna()
    d2 = pd.DataFrame([
        {"Dimension": "Resampling stability", "Median": f"{np.median(r):.3f}",
         "IQR": f"{np.quantile(r, .25):.3f}-{np.quantile(r, .75):.3f}",
         "Range": f"{r.min():.3f}-{r.max():.3f}",
         "At or above 0.50": f"{int((r >= .5).sum())}/34"},
        {"Dimension": "Alternative-geometry separability", "Median": f"{np.median(g):.3f}",
         "IQR": f"{np.quantile(g, .25):.3f}-{np.quantile(g, .75):.3f}",
         "Range": f"{g.min():.3f}-{g.max():.3f}",
         "At or above 0.50": f"{int((g >= .5).sum())}/34"},
    ])
    return _write(d2, out, "supp_table_diagnostic_distributions",
                  "Category-level distributions for the two Monte-Carlo diagnostic "
                  "dimensions.", "tab:supp-diagdist")


def supp_composition_sensitivity(rel: Path, out: Path) -> pd.DataFrame:
    xr = rel / "results/f1_cross_register/tables"
    cs = pd.read_csv(xr / "compositional_strategy_summary.csv")
    df = cs[["metric", "median", "q1", "q3", "min", "max",
             "n_negative", "sign_test_p", "loo_median_min", "loo_median_max"]].copy()
    df.columns = ["Analysis", "Median", "Q1", "Q3", "Min", "Max",
                  "Events negative", "Sign-test p", "LOO median min", "LOO median max"]
    for c in df.columns[1:]:
        if df[c].dtype != object:
            df[c] = df[c].map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    return _write(df, out, "supp_table_composition_sensitivity",
                  "Full compositional sensitivity for the matched-event analysis.",
                  "tab:supp-composition")


def supp_per_event(rel: Path, out: Path) -> pd.DataFrame:
    """Per-event paired values, with reader-facing event names and fixed precision."""
    d = pd.read_csv(rel / "results/manuscript_tables/table5_paired_register_by_event.csv")
    d = d.copy()
    d["Event"] = (d["Event"].str.replace("SaoPaulo", "S\u00e3o Paulo", regex=False)
                            .str.replace("LasVegas", "Las Vegas", regex=False))
    for c in ["Strategy share (highlight)", "Strategy share (full-race)", "Difference"]:
        d[c] = d[c].astype(float).map(lambda v: f"{v:+.4f}" if c == "Difference"
                                      else f"{v:.4f}")
    d = d.rename(columns={"Strategy share (highlight)": "Highlight share",
                          "Strategy share (full-race)": "Full-race share",
                          "Null 95% interval": "Null 95% interval",
                          "Outside null": "Outside null"})
    return _write(d, out, "supp_table_per_event_paired",
                  "Paired same-event register comparison for the "
                  "Strategy-and-technical domain.", "tab:supp-perevent")


def main_table_corpora(rel: Path, out: Path) -> pd.DataFrame:
    """Side-by-side comparison of the two analytical corpora, frozen values."""
    S = json.loads((rel / "results/paper_results_summary.json").read_text("utf-8"))
    assert S["highlight"]["n_sentences"] == 7647
    assert S["fullrace"]["n_sentences"] == 16876
    rows = [
        ("Register", "condensed highlight commentary",
         "continuous full-race commentary"),
        ("Events", "113", "10"),
        ("Sentences", "7,647", "16,876"),
        ("Coverage profile", "broad across events, comparatively shallow per event",
         "narrow across events, comparatively deep per event"),
        ("Role in analysis", "thematic discovery and diagnostic characterisation",
         "independent thematic reconstruction"),
        ("Matched-event role", "10 events used in the paired comparison",
         "the same 10 events used in the paired comparison"),
    ]
    df = pd.DataFrame(rows, columns=["Characteristic", "Highlight commentary",
                                     "Full-race commentary"])
    return _write(df, out, "main_table_corpora",
                  "Formula 1 commentary corpora and their roles in the study.",
                  "tab:corpora")


def main_table_correspondence_classes(rel: Path, out: Path) -> pd.DataFrame:
    """Operational definition of the four correspondence outcomes."""
    rows = [
        ("Non-recovery",
         "No full-race cluster exceeds the highlight category's null threshold.",
         "No supported counterpart is recovered."),
        ("Single-match",
         "Exactly one full-race cluster exceeds the threshold and does not "
         "satisfy the merge criterion.",
         "One supported counterpart."),
        ("Split",
         "Two or more full-race clusters exceed the threshold.",
         "One highlight category corresponds to several full-race clusters."),
        ("Merge",
         "Exactly one full-race cluster exceeds the category threshold, and that "
         "cluster is also a supported counterpart of at least one other highlight "
         "category under the stricter merge threshold.",
         "Several highlight categories converge on one full-race cluster."),
    ]
    df = pd.DataFrame(rows, columns=["Correspondence", "Operational criterion",
                                     "Interpretation"])
    return _write(df, out, "main_table_correspondence_classes",
                  "Operational criteria for the four cross-register "
                  "correspondence outcomes.", "tab:corrclasses")


def build_all(rel_root: Path) -> dict:
    rel_root = Path(rel_root)
    out = rel_root / "results/manuscript_tables"
    print("  restructured manuscript tables:")
    main_table_corpora(rel_root, out)
    main_table_correspondence_classes(rel_root, out)
    main_table1(rel_root, out)
    main_table2(rel_root, out)
    main_table3(rel_root, out)
    supp_diagnostic_detail(rel_root, out)
    supp_composition_sensitivity(rel_root, out)
    supp_per_event(rel_root, out)
    return {"dir": str(out.relative_to(rel_root))}


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parent.parent)
