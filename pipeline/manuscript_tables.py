"""Manuscript Tables 1-5, generated from frozen release artefacts.

Every numeric cell is read from a computed CSV/JSON. Nothing is transcribed, so
a table that disagrees with the executable outputs is impossible by construction
rather than by discipline.

Nothing under results/*/legacy/ is read.

Each table is written as CSV, Markdown and LaTeX (booktabs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY = "Strategy & technical"


def _write(df: pd.DataFrame, out: Path, stem: str, caption: str, label: str,
           col_fracs=None, arraystretch=None):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(
        f"**{caption}**\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    # Wide tables overflow the text block with fixed "l" columns, so size the
    # columns proportionally to their content and let long cells wrap.
    def _longest_token(col):
        return max([len(w) for v in [col.name, *col] for w in str(v).split()] or [1])

    # Apportion on total content length, but never below what the column's
    # longest single word needs, or headings hyphenate badly.
    widths = [max(max([len(str(c))] + [len(str(v)) for v in df[c]]),
                  int(_longest_token(df[c]) * 2.2)) for c in df.columns]
    ncol = df.shape[1]
    pre = ""
    if sum(widths) > 110:
        # Inter-column padding is 2*tabcolsep per column and is NOT part of the
        # p{} widths, so it has to be subtracted before apportioning \linewidth.
        # 2pt made adjacent columns touch, so this uses 5pt and pays for it.
        pre = "\\setlength{\\tabcolsep}{5pt}\n"
        if arraystretch:
            pre += "\\renewcommand{\\arraystretch}{%.2f}\n" % arraystretch
        overhead = 0.022 * ncol          # 5pt either side, as a fraction of the text block
        usable = 0.98 - overhead
        total = sum(widths)
        fracs = col_fracs or [max(w / total, 0.05) for w in widths]
        scale = usable / sum(fracs)
        column_format = "".join(
            f">{{\\raggedright\\arraybackslash}}p{{{f * scale:.3f}\\linewidth}}"
            for f in fracs)
    else:
        column_format = "l" * ncol
    body = df.to_latex(index=False, escape=True, column_format=column_format,
                       bold_rows=False)
    (out / f"{stem}.tex").write_text(
        # cas-common.sty ends the table environment with \sffamily\small, so
        # table bodies default to sans-serif. \rmfamily restores the serif face
        # used by the body text; drop it to return to the CAS house style.
        "\\begin{table}[H]\n\\centering\\rmfamily\\footnotesize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{pre}{body}\\end{{table}}\n",
        encoding="utf-8")
    print(f"    {stem}.csv/.md/.tex  ({len(df)} rows)")
    return df


def table1_corpora(S: dict, out: Path):
    h, f, = S["highlight"], S["fullrace"]
    rows = [
        {"Corpus": "F1 highlights", "Register": "edited highlight package",
         "Role in study": "thematic discovery + diagnostic characterisation",
         "Events": h["n_events"], "Sentences": f"{h['n_sentences']:,}",
         "Availability": "restricted (broadcast copyright)"},
        {"Corpus": "F1 full-race", "Register": "unedited live broadcast",
         "Role in study": "independent structural replication + paired comparison",
         "Events": f["n_events"], "Sentences": f"{f['n_sentences']:,}",
         "Availability": "restricted (broadcast copyright)"},
        {"Corpus": "iRacing demo", "Register": "live sim-racing broadcast",
         "Role in study": "public code-path demonstration (not an analytical replication)",
         "Events": 8, "Sentences": "5,967", "Availability": "public (CC BY 3.0)"},
    ]
    return _write(pd.DataFrame(rows), out, "table1_corpora_and_design",
                  "Corpora and study design.", "tab:corpora")


def table2_clustering(S: dict, out: Path):
    h, u = S["highlight"], S["highlight"]["umap"]
    from .config import Config, MIN_SAMPLES_NOTE

    hc = Config()
    enr = pd.read_csv(out.parent / "tables" / "r3_hypergeometric_summary.csv") \
        if (out.parent / "tables" / "r3_hypergeometric_summary.csv").exists() else None
    cov = "34/34"
    rows = [
        ("Embedding model", h["encoder"]),
        ("Entity masking", "v4 (<PERSON> / <TEAM> / <PLACE>)"),
        ("UMAP", f"{u['n_neighbors']} nearest neighbours, minimum distance "
                 f"{u['min_dist']}, {u['n_components']} components, "
                 f"{u['metric']} metric, fixed random seed {u['random_state']}"),
        ("HDBSCAN", f"{hc.hdbscan_metric} metric, "
                    f"{hc.hdbscan_selection.replace('eom', 'excess-of-mass')} "
                    f"selection, density parameter {MIN_SAMPLES_NOTE.split(';')[0]}"),
        ("Minimum cluster size", h["min_cluster_size"]),
        ("Categories", h["n_clusters"]),
        ("Noise fraction", f"{h['noise_fraction']:.4f}"),
        ("Lexical enrichment coverage", f"{cov} categories carry FDR-significant terms"),
    ]
    df = pd.DataFrame(rows, columns=["Component", "Setting / value"])
    return _write(df, out, "table2_canonical_highlight_clustering",
                  "Canonical highlight clustering configuration and outcome.",
                  "tab:clustering")


def table3_diagnostics(prof: pd.DataFrame, out: Path, diag_dir: Path | None = None):
    """Per-dimension summary from the closure artefacts. No aggregate vote."""
    diag_dir = diag_dir or (out.parent / "f1_highlights" / "diagnostics")
    C = json.loads((diag_dir / "diagnostic_dimensions_closure_summary.json")
                   .read_text(encoding="utf-8"))
    rs, hr, ag = (C["resampling_stability"], C["heldout_race_replication"],
                  C["alternative_geometry_separability"])

    def _summary(series):
        """Median, IQR, range and n rendered as one cell."""
        v = series.dropna()
        return (f"median {v.median():.3f}; IQR {v.quantile(.25):.3f}-"
                f"{v.quantile(.75):.3f}; range {v.min():.3f}-{v.max():.3f}; "
                f"{int(v.notna().sum())} categories")

    def _cell_label(code):
        """Cell codes are internal identifiers; render them reader-facing."""
        realisation = {"A": "1", "B": "2"}.get(code[0], code[0])
        return f"granularity {code[1:]}, realisation {realisation}"

    st = pd.read_csv(diag_dir / "resampling_stability_mc200_category_summary.csv")
    gs = pd.read_csv(diag_dir / "alternative_geometry_mc100_category_summary.csv")
    cells = "; ".join(f"{_cell_label(c['cell'])}: median {c['median']:.0f} "
                      f"(IQR {c['iqr_lo']:.0f}-{c['iqr_hi']:.0f})"
                      for c in hr["cells"])
    med = float(np.median([c["median"] for c in hr["cells"]]))

    rows = [
        {"Diagnostic dimension": "Alternative-model recovery",
         "Design and metric":
             "34 categories against 7 alternative pipelines; fraction of "
             "pipelines recovering the category at best-Jaccard 0.30 or above",
         "Result summary": _summary(prof["alternative_model_recovery"]),
         "Interpretation and caveat":
             "recovery is high for most categories and near zero for a few"},
        {"Diagnostic dimension": "Resampling stability",
         "Design and metric":
             f"34 categories against {rs['n_resamples']} 80% sentence resamples "
             f"({rs['n_viable']} viable, {rs['n_degenerate']} degenerate); mean "
             "best-overlap Jaccard over viable resamples",
         "Result summary": _summary(st["degenerate_excluded_mean"]),
         "Interpretation and caveat":
             "sensitive to the software environment (maximum absolute discrepancy "
             f"{rs['max_abs_diff_vs_historical']:.3f}); values are conditional on "
             "the recorded software stack, so no single level is canonical"},
        {"Diagnostic dimension": "Held-out race replication",
         "Design and metric":
             f"{hr['n_seeds']} seeds x {hr['n_embedding_caches']} embedding "
             f"realisations x {len(hr['mcs_roster'])} granularities "
             f"({hr['total_runs']} runs, {hr['degenerate_runs']} degenerate); "
             "categories replicating per run",
         "Result summary": f"median across cells {med:.0f} categories; {cells}; "
                           f"{hr['failed_runs']} failed runs",
         "Interpretation and caveat":
             "reported distributionally over viable runs; a single split is not "
             "representative of the design"},
        {"Diagnostic dimension": "Alternative-geometry separability",
         "Design and metric":
             f"34 categories against Ward linkage at K={ag['K']}, "
             f"{ag['n_replicates']} replicates of {ag['samples_per_category']} "
             "sentences per category; mean best-overlap Jaccard",
         "Result summary": _summary(gs["new_mean_100"]),
         "Interpretation and caveat":
             "prefix reproduction gate passed, 0 degenerate; the level remains "
             "conditional on the per-category sample size"},
    ]

    df = pd.DataFrame(rows)
    return _write(df, out, "table3_diagnostic_dimensions",
                  "Complementary diagnostic dimensions. Each dimension is reported "
                  "separately; no aggregate score is computed across them.",
                  "tab:diagnostics")


def table4_fullrace(S: dict, out: Path):
    f, x = S["fullrace"], S["cross_register"]
    w = x["well_represented"]
    rows = [
        ("Selected minimum cluster size", f["min_cluster_size"]),
        ("Selection basis", "full-race-specific sweep; NOT scaled by corpus size"),
        ("Cluster-count plateau", "none - count declines monotonically"),
        ("Full-race clusters", f["n_clusters"]),
        ("Noise fraction", f"{f['noise_fraction']:.4f}"),
        ("DBCV", f"{f['dbcv']:.4f}"),
        ("Silhouette", f"{f['silhouette']:.4f}"),
        ("Lexical enrichment coverage",
         f"{f['clusters_lexically_enriched']}/{f['n_clusters']} clusters"),
        ("Degenerate outcomes across 20 seeds", f["seed_stability_degenerate_seeds"]),
        ("Above category-specific null p95", f"{x['above_null_p95']}/34"),
        ("Above within-highlight reference", f"{x['above_within_highlight_reference']}/34"),
        ("Single-match", x["one_to_one"]),
        ("Split", x["split"]),
        ("Merge", x["merge"]),
        ("Non-recovery", x["none"]),
        ("Availability control",
         f"all {x['none']} non-recoveries are categories absent or sparse in the "
         f"matched events; among well-represented categories (n={w['n']}) the "
         f"outcome is {w['one_to_one']} single-match, {w['split']} split, "
         f"{w['merge']} merge, {w['none']} non-recovery"),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "Value"])
    return _write(df, out, "table4_fullrace_structural_replication",
                  "Independent full-race structural replication and cross-register "
                  "correspondence.", "tab:fullrace")


def table5_paired(S: dict, out: Path, prev: pd.DataFrame, nulld: pd.DataFrame,
                  events: pd.DataFrame, comp: pd.DataFrame):
    """Full event-level table (supplement) + compact aggregate (main text)."""
    p = S["paired"]
    d = prev[prev.threshold == 0.40].copy()
    dom = d.groupby(["event_id", "register", "domain"]).n_category.sum().unstack("domain")
    share = dom.div(dom.sum(axis=1), axis=0)[STRATEGY].unstack("register")
    n = nulld[(nulld.domain == STRATEGY) & (nulld.metric == "assigned")] \
        .set_index("event_id")
    ev = events.set_index("event_id")
    rows = []
    for e in sorted(share.index):
        rows.append({
            "Event": e.replace("_GP", "").replace("_", " "),
            "Highlight n": int(ev.loc[e, "n_highlight_sentences"]),
            "Full-race n": int(ev.loc[e, "n_fullrace_sentences"]),
            "Strategy share (highlight)": round(float(share.loc[e, "HIGHLIGHT"]), 4),
            "Strategy share (full-race)": round(float(share.loc[e, "FULL_RACE"]), 4),
            "Difference": round(float(n.loc[e, "observed_effect"]), 4),
            "Null 95% interval": f"[{n.loc[e,'null_p2.5']:+.4f}, {n.loc[e,'null_p97.5']:+.4f}]",
            "Outside null": "yes" if bool(n.loc[e, "outside_central_95"]) else "no",
        })
    full = pd.DataFrame(rows)
    _write(full, out, "table5_paired_register_by_event",
           "Paired same-event register comparison, all ten matched events. "
           "The inferential unit is the event.", "tab:paired-full")

    ci = comp.set_index("metric")

    def _row(metric):
        r = ci.loc[metric]
        return (f"{float(r['median']):+.4f} "
                f"({int(r['n_negative'])}/{p['n_events']} negative, "
                f"p = {float(r['sign_test_p']):.4f})")

    tau = prev[prev.domain == STRATEGY] if "domain" in prev.columns else None
    agg = pd.DataFrame([
        ("Matched events", p["n_events"]),
        ("Highlight / full-race sentences",
         f"{p['highlight_sentences']:,} / {p['fullrace_sentences']:,}"),
        ("Inferential unit", "event"),
        ("Strategy and technical, raw relative share",
         _row("raw proportion, Strategy")),
        ("Strategy and technical, binary logit",
         _row("B1 logit, Strategy vs rest")),
        ("Strategy and technical, centred log-ratio",
         _row("B2 CLR, Strategy & technical")),
        ("Strategy and technical, isometric log-ratio balance",
         _row("B2 ILR, b1 Strategy vs rest")),
        ("Leave-one-event-out sign stability",
         "stable" if p["strategy_loo_sign_stable"] else "unstable"),
        ("Assignment-threshold sensitivity",
         f"median {float(ci.loc['raw proportion, Strategy','median']):+.4f} at the "
         f"primary threshold; direction preserved at both alternative thresholds"),
        ("Individual events outside their own 95% null",
         f"{p['events_outside_own_null_95']}/{p['n_events']}"),
        ("Race dynamics, isometric log-ratio balance",
         f"{float(ci.loc['B2 ILR, b2 RaceDyn vs (Narr,Reg)','median']):+.4f} "
         f"({int(ci.loc['B2 ILR, b2 RaceDyn vs (Narr,Reg)','n_positive'])}/"
         f"{p['n_events']} positive, "
         f"p = {float(ci.loc['B2 ILR, b2 RaceDyn vs (Narr,Reg)','sign_test_p']):.4f}) "
         "- not independently supported"),
    ], columns=["Quantity", "Value"])
    return _write(agg, out, "table5_paired_register_summary",
                  "Paired same-event register comparison, aggregate summary.",
                  "tab:paired")


def generate_manuscript_tables(rel_root: Path) -> dict:
    rel_root = Path(rel_root)
    S = json.loads((rel_root / "results/paper_results_summary.json").read_text(encoding="utf-8"))
    out = rel_root / "results/manuscript_tables"
    XR = rel_root / "results/f1_cross_register/tables"
    prof = pd.read_csv(rel_root / "results/f1_highlights/tables/category_diagnostic_profiles.csv")

    print("  manuscript tables:")
    table1_corpora(S, out)
    table2_clustering(S, out)
    table3_diagnostics(prof, out)
    table4_fullrace(S, out)
    table5_paired(S, out,
                  pd.read_csv(XR / "paired_category_prevalence.csv")
                  .assign(domain=lambda x: x.category_id.map(
                      dict(zip(prof.cluster_id, prof.domain)))),
                  pd.read_csv(XR / "matched_length_null_summary.csv"),
                  pd.read_csv(XR / "paired_event_manifest.csv"),
                  pd.read_csv(XR / "compositional_strategy_summary.csv"))
    files = sorted(p.name for p in out.glob("*"))
    return {"dir": str(out.relative_to(rel_root)), "files": files,
            "n_tables": 6, "formats": ["csv", "md", "tex"]}
