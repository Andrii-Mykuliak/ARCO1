"""Result-focused manuscript figures, built on the shared manuscript style.

Three additions that state the substantive findings directly:

* taxonomy profile   - what commentators talk about, as a ranked profile
* restructuring      - how the two registers correspond and depart
* paired shares      - the matched-event Strategy-and-technical result

Every value is read from a frozen release artefact. Reader-facing category
names only; no internal identifier reaches a label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figstyle import C, DOMAIN_COLOUR, DOMAIN_ORDER, apply, save

RELATION_LABEL = {
    "ONE_TO_ONE": "single-match",
    "SPLIT": "split",
    "MERGE": "merge",
    "NONE": "non-recovery",
}
AVAILABILITY_LABEL = {
    "WELL_REPRESENTED": "well represented",
    "SPARSE": "sparse",
    "EFFECTIVELY_ABSENT": "effectively absent",
}
RELATION_ORDER = ["single-match", "split", "merge", "non-recovery"]


def _domain_sort(df, col="domain"):
    order = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    return df.assign(_d=df[col].map(order).fillna(9))


# --------------------------------------------------------------- figure 1 --
def taxonomy_profile(rel: Path, out: Path,
                     name: str = "fig_taxonomy_profile") -> Path:
    """Ranked category profile: what commentators talk about."""
    prof = pd.read_csv(rel / "results/f1_highlights/tables/category_diagnostic_profiles.csv")
    total = float(prof["size"].sum())
    df = _domain_sort(prof).sort_values(["_d", "size"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    y = np.arange(len(df))
    ax.barh(y, df["size"], color=[DOMAIN_COLOUR.get(d, "#777") for d in df.domain],
            height=0.78, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df.label, fontsize=8.0)
    ax.set_ylim(-0.7, len(df) - 0.3)
    for i, (n, lab) in enumerate(zip(df["size"], df.label)):
        ax.text(n + total * 0.004, i, f"{n:,}  ({n / total:.1%})",
                va="center", fontsize=6.8, color=C["muted"])
    ax.set_xlabel("sentences assigned to the category")
    ax.set_xlim(0, df["size"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=DOMAIN_COLOUR[d],
                              mec=DOMAIN_COLOUR[d], ms=7, label=d)
                       for d in DOMAIN_ORDER if d in set(df.domain)],
              loc="lower right", frameon=True, fontsize=7.5, ncol=1)
    fig.text(0.5, -0.06,
             "Percentages are of all clustered sentences. Domain grouping is "
             "expository, not an independently recovered hierarchy.",
             ha="center", fontsize=7.0, color=C["muted"])
    return save(fig, out, name)


# --------------------------------------------------------------- figure 2 --
def restructuring_map(rel: Path, out: Path,
                      name: str = "fig_cross_register_restructuring") -> Path:
    """Bipartite map of how highlight categories correspond to full-race themes."""
    corr = pd.read_csv(rel / "results/f1_cross_register/tables/cross_register_correspondence.csv")
    corr["relation_r"] = corr.relation.map(RELATION_LABEL)
    df = _domain_sort(corr).sort_values(
        ["_d", "relation_r"], key=lambda s: s if s.name == "_d"
        else s.map({r: i for i, r in enumerate(RELATION_ORDER)})).reset_index(drop=True)

    # right-hand nodes: full-race themes actually involved, numbered manuscript-locally
    involved = []
    for _, r in df.iterrows():
        if r.relation == "NONE":
            continue
        raw = str(r.fr_clusters_above_threshold)
        for c in (raw.split(";") if raw and raw.lower() != "nan" else []):
            if c.strip():
                involved.append(int(c))
    # Order the full-race nodes by the mean row of the highlight categories that
    # link to them (a barycentre pass). This only changes vertical placement on
    # the right-hand axis; no link, relation or category is altered.
    rows_of = {}
    for i, r in df.iterrows():
        if r.relation == "NONE":
            continue
        raw = str(r.fr_clusters_above_threshold)
        for c in (raw.split(";") if raw and raw.lower() != "nan" else []):
            if c.strip():
                rows_of.setdefault(int(c), []).append(i)
    order = sorted(set(involved),
                   key=lambda c: (sum(rows_of[c]) / len(rows_of[c]), c))
    theme_no = {c: i + 1 for i, c in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8.4, 8.1))
    yl = np.arange(len(df))[::-1]
    yr_span = np.linspace(yl.min(), yl.max(), len(order)) if order else []
    yr = {c: yr_span[i] for i, c in enumerate(order)}

    link_style = {"single-match": dict(color="#444444", lw=1.0, alpha=.85, ls="-"),
                  "split": dict(color=C["secondary"], lw=1.1, alpha=.9, ls="-"),
                  "merge": dict(color="#7570b3", lw=1.1, alpha=.9, ls="-")}
    for i, r in df.iterrows():
        rel_r = r.relation_r
        if rel_r == "non-recovery":
            ax.plot([0.06, 0.16], [yl[i], yl[i]], color="#b03030", lw=1.0,
                    ls=":", alpha=.8, zorder=1)
            continue
        raw = str(r.fr_clusters_above_threshold)
        targets = [int(c) for c in raw.split(";") if c.strip() and raw.lower() != "nan"]
        for t in targets:
            ax.plot([0.06, 0.94], [yl[i], yr[t]], zorder=1,
                    **link_style.get(rel_r, link_style["single-match"]))
    for i, r in df.iterrows():
        ax.scatter([0.06], [yl[i]], s=26, zorder=3, clip_on=False,
                   color=DOMAIN_COLOUR.get(r.domain, "#777"))
        ax.text(0.045, yl[i], r.highlight_name, ha="right", va="center", fontsize=7.4)
    for c, yy in yr.items():
        ax.scatter([0.94], [yy], s=20, zorder=3, clip_on=False, color="#555555")
        ax.text(0.955, yy, f"Full-race theme {theme_no[c]}", ha="left",
                va="center", fontsize=7.0, color="#333333")

    ax.set_xlim(0, 1.16)
    ax.set_ylim(-1.2, len(df))
    ax.axis("off")
    ax.text(0.06, len(df) - 0.2, "highlight category", ha="center", fontsize=8.5,
            weight="bold")
    ax.text(0.94, len(df) - 0.2, "full-race theme", ha="center", fontsize=8.5,
            weight="bold")
    handles = [Line2D([], [], color="#444444", lw=1.2, label="single-match"),
               Line2D([], [], color=C["secondary"], lw=1.2, label="split"),
               Line2D([], [], color="#7570b3", lw=1.2, label="merge"),
               Line2D([], [], color="#b03030", lw=1.2, ls=":", label="non-recovery")]
    handles += [Line2D([], [], marker="o", ls="", mfc=DOMAIN_COLOUR[d],
                       mec=DOMAIN_COLOUR[d], ms=6, label=d)
                for d in DOMAIN_ORDER if d in set(df.domain)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.075),
              ncol=4, frameon=False, fontsize=7.2)
    # The local-numbering and non-bijection qualifications live in the
    # manuscript caption, not inside the plotting area.
    return save(fig, out, name)


# --------------------------------------------------------------- figure 3 --
def paired_strategy_shares(rel: Path, out: Path,
                           name: str = "fig_paired_strategy_shares") -> Path:
    """Slopegraph of the Strategy-and-technical share, per matched event."""
    ev = pd.read_csv(rel / "results/manuscript_tables/table5_paired_register_by_event.csv")
    ev["Event"] = (ev["Event"].str.replace("SaoPaulo", "S\u00e3o Paulo", regex=False)
                                .str.replace("LasVegas", "Las Vegas", regex=False))
    h = ev["Strategy share (highlight)"].astype(float)
    f = ev["Strategy share (full-race)"].astype(float)
    order = np.argsort(-(f - h).to_numpy())
    ev, h, f = ev.iloc[order], h.iloc[order], f.iloc[order]
    y = np.arange(len(ev))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    for i, (a, b) in enumerate(zip(h, f)):
        ax.plot([a, b], [y[i], y[i]], color="#999999", lw=1.1, zorder=1)
    ax.scatter(h, y, s=46, color=C["primary"], zorder=3, label="highlight commentary")
    ax.scatter(f, y, s=46, color=C["secondary"], zorder=3, marker="D",
               label="full-race commentary")
    ax.set_yticks(y)
    ax.set_yticklabels(ev["Event"], fontsize=8.0)
    ax.set_ylim(-0.9, len(ev) - 0.1)
    ax.set_xlabel("Strategy-and-technical share of assigned sentences")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=7.5, frameon=False)
    fig.text(0.5, 0.005,
             "Each row is one matched event; the line joins the two registers for "
             "that event.\nThe share is lower in highlight commentary in every one "
             "of the ten events.",
             ha="center", fontsize=7.0, color=C["muted"])
    return save(fig, out, name)


# --------------------------------------------------------------- figure 5 --
DOMAIN_CLR = {d: f"delta_clr_{d}" for d in DOMAIN_ORDER}


def _matched_events(rel: Path) -> pd.DataFrame:
    ev = pd.read_csv(rel / "results/manuscript_tables/table5_paired_register_by_event.csv")
    ev["Event"] = (ev["Event"].str.replace("SaoPaulo", "S\u00e3o Paulo", regex=False)
                              .str.replace("LasVegas", "Las Vegas", regex=False))
    assert len(ev) == 10, f"expected 10 matched events, got {len(ev)}"
    return ev


def paired_event_shares(rel: Path, out: Path,
                        name: str = "fig_paired_event_shares") -> Path:
    """Strategy-and-technical share in each register, per matched event.

    Source: results/manuscript_tables/table5_paired_register_by_event.csv, frozen.
    Nothing is recomputed; the panel reads and draws stored values.
    """
    ev = _matched_events(rel)
    h = ev["Strategy share (highlight)"].astype(float)
    f = ev["Strategy share (full-race)"].astype(float)
    o = np.argsort(-(f - h).to_numpy())
    ev, h, f = ev.iloc[o], h.iloc[o], f.iloc[o]
    y = np.arange(len(ev))[::-1]

    # compressed vertically: same width and font sizes, less page area
    W, H = 6.505, 2.25
    left, right, bottom, top = 1.22, 0.16, 0.46, 0.34
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W,
                       (H - bottom - top) / H])
    for i, (a, b) in enumerate(zip(h, f)):
        ax.plot([a, b], [y[i], y[i]], color="#b0b0b0", lw=1.2, zorder=1)
    ax.scatter(h, y, s=40, color=C["primary"], zorder=3, label="highlight commentary")
    ax.scatter(f, y, s=40, color=C["secondary"], zorder=3, marker="D",
               label="full-race commentary")
    ax.set_yticks(y)
    ax.set_yticklabels(ev["Event"], fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylim(-0.8, len(ev) - 0.2)
    ax.set_xlabel("Strategy-and-technical share of assigned sentences", fontsize=7.6)
    ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              fontsize=7.2, frameon=False, handletextpad=0.4, borderpad=0.0,
              columnspacing=2.2)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=600, facecolor="white")
    fig.savefig(out / f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"    {name}.pdf + {name}.png  (10 matched events)")
    return out / f"{name}.pdf"


def paired_domain_effects(rel: Path, out: Path,
                          name: str = "fig_paired_domain_effects") -> Path:
    """Domain-level difference in centred log-ratio, one point per matched event.

    Source: results/f1_cross_register/tables/compositional_domain_effects.csv,
    frozen. Nothing is recomputed.
    """
    dom = pd.read_csv(rel / "results/f1_cross_register/tables/compositional_domain_effects.csv")
    assert len(dom) == 10, f"expected 10 events, got {len(dom)}"

    # compressed vertically: same width and font sizes, less page area
    W, H = 6.505, 1.18
    left, right, bottom, top = 1.22, 0.16, 0.30, 0.10
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W,
                       (H - bottom - top) / H])
    yb = np.arange(len(DOMAIN_ORDER))[::-1]
    ax.axvline(0.0, color="#888888", lw=0.8, zorder=1)
    for i, d in enumerate(DOMAIN_ORDER):
        v = dom[DOMAIN_CLR[d]].astype(float).to_numpy()
        ax.scatter(v, np.full(len(v), yb[i]), s=15, alpha=0.62, zorder=2,
                   color=DOMAIN_COLOUR[d], linewidths=0)
        ax.scatter([np.median(v)], [yb[i]], s=64, zorder=4, marker="|",
                   color=DOMAIN_COLOUR[d], linewidths=2.1)
    ax.set_yticks(yb)
    ax.set_yticklabels(DOMAIN_ORDER, fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylim(-0.62, len(DOMAIN_ORDER) - 0.38)
    ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.6)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=600, facecolor="white")
    fig.savefig(out / f"{name}.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"    {name}.pdf + {name}.png  (4 domains x 10 events)")
    return out / f"{name}.pdf"


def build_all(rel: Path) -> dict:
    apply()
    hl = rel / "results/f1_highlights/figures"
    xr = rel / "results/f1_cross_register/figures"
    print("  result-focused figures:")
    return {"taxonomy": taxonomy_profile(rel, hl),
            "restructuring": restructuring_map(rel, xr),
            "paired": paired_strategy_shares(rel, xr),
            "event_shares": paired_event_shares(rel, xr),
            "domain_effects": paired_domain_effects(rel, xr)}


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parent.parent)
