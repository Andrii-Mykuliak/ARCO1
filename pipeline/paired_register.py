"""Paired same-event register analysis.

Compares highlight and full-race commentary for the SAME events, using the
frozen canonical highlight centroids for both registers so that register is not
confounded with assignment method.

THE INFERENTIAL UNIT IS THE EVENT, NOT THE SENTENCE. Every summary in this
module aggregates to n = 10 matched events. Pooled sentences are never treated
as independent observations, and no function here returns a sentence-level
significance test.

Two prevalence definitions are kept strictly separate:

    p_all       n_category / n_all_sentences        coverage + composition
    p_assigned  n_category / n_assigned_sentences   composition within coverage

Because assigned-only domain shares sum to 1, raw differences are compositional:
a decrease in one part mechanically implies an increase elsewhere. The log-ratio
functions below exist so that claim can be tested rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import l2_normalise, log

TAU_PRIMARY = 0.40
TAU_SENSITIVITY = (0.35, 0.40, 0.45)
DELTA_ZERO = 0.5                    # Jeffreys pseudocount for zero replacement


# ------------------------------------------------------------- assignment --
def assign_by_centroid(X: np.ndarray, C: np.ndarray, ids, tau: float):
    """Nearest-centroid cosine assignment, identical for both registers."""
    S = X @ C.T
    j = S.argmax(1)
    cos = S.max(1)
    ok = cos >= tau
    return np.where(ok, np.asarray(ids)[j], -1), cos, ok


def prevalence(labels: np.ndarray, ids, n_total: int | None = None):
    """Return (p_all, p_assigned) over `ids`, with the two denominators kept
    distinct. Zero-count categories are explicit zeros, never absences."""
    n_total = len(labels) if n_total is None else n_total
    n_asg = int((labels != -1).sum())
    counts = np.array([int((labels == c).sum()) for c in ids])
    p_all = counts / n_total if n_total else np.full(len(ids), np.nan)
    p_asg = counts / n_asg if n_asg else np.full(len(ids), np.nan)
    return counts, p_all, p_asg


# ------------------------------------------------------------ event pairs --
def event_deltas(prev: pd.DataFrame, key="category_id"):
    """Highlight minus full-race, within event. Returns one row per
    (event, threshold, category)."""
    w = prev.pivot_table(index=["event_id", "threshold", key],
                         columns="register", values=["p_all", "p_assigned"])
    return pd.DataFrame({
        "delta_all": w[("p_all", "HIGHLIGHT")] - w[("p_all", "FULL_RACE")],
        "delta_assigned": (w[("p_assigned", "HIGHLIGHT")]
                           - w[("p_assigned", "FULL_RACE")]),
    }).reset_index()


def summarise_events(d: pd.DataFrame, col: str):
    """Event-level summary. n is the number of EVENTS."""
    v = d[col].dropna().to_numpy()
    q1, q3 = np.percentile(v, [25, 75])
    return {"n_events": len(v), "mean": float(v.mean()),
            "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(v.min()), "max": float(v.max()),
            "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum()),
            "n_zero": int((v == 0).sum())}


def leave_one_event_out(d: pd.DataFrame, col: str, event_col="event_id"):
    """Median of `col` with each event removed in turn. One row per omission."""
    ev = sorted(d[event_col].unique())
    out = []
    for e in ev:
        v = d[d[event_col] != e][col].dropna().to_numpy()
        m = float(np.median(v))
        out.append({"omitted_event": e, "n_events_used": len(v),
                    "median": m, "sign": int(np.sign(m))})
    return pd.DataFrame(out)


# -------------------------------------------------- compositional analysis --
def bayes_multiplicative_replace(counts: np.ndarray, delta: float = DELTA_ZERO):
    """Zero replacement that preserves ratios among observed parts.

    A zero count becomes delta/n and the non-zero parts are rescaled
    multiplicatively, unlike an additive pseudocount which distorts them.
    Deterministic. Returns (composition, n_zeros_replaced).
    """
    n = counts.sum()
    if n == 0:
        return np.full(len(counts), 1 / len(counts)), int(len(counts))
    p = counts / n
    zero = counts == 0
    if not zero.any():
        return p, 0
    repl = delta / n
    out = p.copy()
    out[zero] = repl
    out[~zero] = p[~zero] * (1 - repl * zero.sum())
    return out / out.sum(), int(zero.sum())


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def clr(p: np.ndarray) -> np.ndarray:
    lg = np.log(p)
    return lg - lg.mean()


def ilr(p: np.ndarray, names, balances) -> np.ndarray:
    """Isometric log-ratio on a pre-specified sequential binary partition.

    The basis must be fixed before inspecting results; it is passed in rather
    than chosen here so that choice stays in configuration.
    """
    idx = {d: i for i, d in enumerate(names)}
    lg = np.log(p)
    out = []
    for num, den in balances:
        r, s = len(num), len(den)
        out.append(np.sqrt(r * s / (r + s))
                   * (lg[[idx[d] for d in num]].mean()
                      - lg[[idx[d] for d in den]].mean()))
    return np.array(out)


# ------------------------------------------------------ matched-length null --
def matched_length_null(labels_full: np.ndarray, assigned_full: np.ndarray,
                        ids, n_highlight: int, n_reps: int, seed: int,
                        index_map=None):
    """Null: highlight-sized samples drawn WITHOUT replacement from the SAME
    event's full-race sentences, i.e. shortening with no thematic selection.

    Returns an (n_reps, len(ids)) array of assigned-only prevalences. Real
    highlight sentences never enter this draw.
    """
    rng = np.random.default_rng(seed)
    F = len(labels_full)
    k = len(ids) if index_map is None else int(index_map.max()) + 1
    out = np.empty((n_reps, k))
    for b in range(n_reps):
        idx = rng.choice(F, size=n_highlight, replace=False)
        a = assigned_full[idx]
        c = labels_full[idx][a]
        n_asg = int(a.sum())
        cnt = (np.bincount(c if index_map is None else index_map[c], minlength=k)
               if n_asg else np.zeros(k, int))
        out[b] = cnt / n_asg if n_asg else np.nan
    return out


def empirical_p(null: np.ndarray, observed: float, side: str = "lower") -> float:
    """(r + 1) / (B + 1), so p is never zero and the floor is 1/(B+1)."""
    B = len(null)
    r = int((null <= observed).sum()) if side == "lower" else int((null >= observed).sum())
    return (1 + r) / (B + 1)


# ------------------------------------------------------------ availability --
def sentences_are_not_the_unit() -> str:
    """Guard string asserted by the test suite."""
    return ("The inferential unit is the event (n = 10). Pooled sentences are "
            "not independent observations and are never used as such.")


def not_applicable(reason: str = "a second matched commentary register is not "
                                 "available in this corpus") -> dict:
    """Explicit N/A status for corpora without a paired register.

    This is a legitimate outcome, NOT a failed stage.
    """
    log(f"paired register: NOT APPLICABLE - {reason}")
    return {"status": "NOT_APPLICABLE", "reason": reason, "applicable": False}


# ===========================================================================
# Matched-event producer
# ===========================================================================
# Ported from the frozen internal implementation:
#   scripts/paired_register_analysis.py    assignment, prevalence, deltas, LOO
#   scripts/closure_compositional.py       logit / CLR / ILR, sign tests
#   scripts/closure_matched_length_null.py matched-length and contiguous nulls
#
# The scientific rules are unchanged; only the interface is. Both registers are
# assigned by the SAME nearest-centroid procedure against the primary register's
# frozen centroids, so register is never confounded with assignment method. The
# primary register's own partition labels are deliberately not reused.

from . import frozen_f1 as F  # noqa: E402


def _sign_test_p(v: np.ndarray) -> float:
    """Two-sided exact binomial sign test over the event-level differences."""
    from scipy.stats import binomtest
    neg, pos = int((v < 0).sum()), int((v > 0).sum())
    return float(binomtest(max(neg, pos), len(v), 0.5).pvalue)


def contiguous_blocks(rng, n_source: int, n_draw: int, k: int):
    """``k`` non-overlapping contiguous blocks covering exactly ``n_draw`` rows.

    Starts are drawn from a reduced index space and then offset, the standard
    construction for non-overlapping equal-length blocks; the remainder is
    distributed one row at a time to the first blocks. Returns None when the
    geometry is infeasible for this event.
    """
    base, rem = divmod(n_draw, k)
    lens = [base + (1 if i < rem else 0) for i in range(k)]
    span = n_source - n_draw + k
    if span <= 0 or base == 0:
        return None
    starts = np.sort(rng.choice(span, size=k, replace=False))
    idx, off = [], 0
    for s, L in zip(starts, lens):
        idx.append(np.arange(s + off, s + off + L))
        off += L - 1
    out = np.concatenate(idx)
    return out if out.max() < n_source else None


def _prevalence_vector(cat_index: np.ndarray, assigned: np.ndarray, k: int,
                       index_map: np.ndarray):
    n_tot = len(assigned)
    n_asg = int(assigned.sum())
    counts = (np.bincount(index_map[cat_index[assigned]], minlength=k)
              if n_asg else np.zeros(k, int))
    return counts / n_tot, (counts / n_asg if n_asg else np.full(k, np.nan))


def compute_matched_event_composition(
        C: np.ndarray, ids, X_primary: np.ndarray, sessions_primary,
        X_second: np.ndarray, sessions_second, matched_events,
        taus=None, tau_primary: float | None = None,
        category_domain=None, category_name=None, domains=None,
        primary_domain: str | None = None,
        ilr_balances=None, n_null_replicates: int | None = None,
        null_seed: int | None = None, block_ks=None,
        run_matched_length_null: bool = True) -> dict:
    """Thematic composition of the two registers over the matched events.

    THE INFERENTIAL UNIT IS THE EVENT. Every reported quantity aggregates to the
    number of matched events; pooled sentences are never treated as independent
    observations and no sentence-level significance test is returned.

    Assigned-only domain shares sum to one, so a raw decrease in one part
    mechanically implies an increase elsewhere. The binary logit, the centred
    log-ratio and the pre-specified isometric log-ratio partition are therefore
    computed on the same event-level compositions, so that the direction can be
    tested rather than assumed. Zeros are replaced by the Bayesian-multiplicative
    rule before any log-ratio is taken.
    """
    taus = list(taus or F.TAU_SENSITIVITY)
    tau_primary = float(tau_primary if tau_primary is not None else F.TAU_PRIMARY)
    if category_domain is None:
        raise ValueError(
            "category_domain is required: the manual grouping is an analysis "
            "input tied to a particular partition and is never defaulted")
    domains = list(domains or sorted({category_domain[c] for c in ids}))
    ilr_balances = ilr_balances or F.ILR_BALANCES
    B = int(n_null_replicates or F.MATCHED_LENGTH_N_REPLICATES)
    null_seed = int(null_seed if null_seed is not None else F.MATCHED_LENGTH_SEED)
    block_ks = list(block_ks or F.CONTIGUOUS_BLOCK_KS)
    category_name = {} if category_name is None else category_name
    primary_domain = primary_domain or F.PRIMARY_DOMAIN
    if primary_domain not in domains:
        raise ValueError(f"primary_domain {primary_domain!r} is not one of "
                         f"the configured domains {domains}")

    ids = list(ids)
    sp = pd.Series(np.asarray(sessions_primary))
    ss = pd.Series(np.asarray(sessions_second))
    pairs = dict(sorted(matched_events.items()))

    p_idx = {rid: np.nonzero((sp == rid).to_numpy())[0] for rid in pairs.values()}
    s_idx = {stem: np.nonzero((ss == stem).to_numpy())[0] for stem in pairs}
    missing = ([rid for rid, v in p_idx.items() if not len(v)]
               + [stem for stem, v in s_idx.items() if not len(v)])
    if missing:
        raise ValueError("matched events absent from a corpus: " + ", ".join(missing))

    manifest = pd.DataFrame([{
        "pair_id": f"P{i:02d}", "event_id": rid, "second_register_event": stem,
        "n_primary_sentences": len(p_idx[rid]),
        "n_second_sentences": len(s_idx[stem]),
    } for i, (stem, rid) in enumerate(pairs.items(), start=1)])
    pid = dict(zip(manifest.event_id, manifest.pair_id))

    # ---------------- assignment and prevalence, both registers -----------
    prev_rows, cov_rows = [], []
    for tau in taus:
        for stem, rid in pairs.items():
            for reg, Xe in (("HIGHLIGHT", X_primary[p_idx[rid]]),
                            ("FULL_RACE", X_second[s_idx[stem]])):
                lab, _, ok = assign_by_centroid(Xe, C, ids, tau)
                counts, p_all, p_asg = prevalence(lab, ids, n_total=len(lab))
                cov_rows.append({"pair_id": pid[rid], "event_id": rid,
                                 "register": reg, "threshold": tau,
                                 "n_total": len(lab), "n_assigned": int(ok.sum()),
                                 "n_unassigned": int((~ok).sum()),
                                 "coverage": float(ok.mean()),
                                 "unassigned_rate": float((~ok).mean())})
                for c, n_c, pa, pg in zip(ids, counts, p_all, p_asg):
                    prev_rows.append({
                        "pair_id": pid[rid], "event_id": rid, "threshold": tau,
                        "category_id": c,
                        "category_name": category_name.get(c, str(c)),
                        "register": reg, "n_category": int(n_c),
                        "n_total": len(lab), "n_assigned": int(ok.sum()),
                        "p_all": float(pa), "p_assigned": float(pg)})
    prev = pd.DataFrame(prev_rows)
    coverage_tab = pd.DataFrame(cov_rows)

    # ---------------- per-category event deltas, effects, LOO -------------
    deltas = event_deltas(prev)
    eff_rows = []
    for (tau, cid), g in deltas.groupby(["threshold", "category_id"]):
        for metric in ("all", "assigned"):
            s = summarise_events(g, f"delta_{metric}")
            # released column vocabulary: the summarised quantity is a delta
            s = {("mean_delta" if k == "mean" else
                  "median_delta" if k == "median" else k): v for k, v in s.items()}
            npos, nneg = s["n_positive"], s["n_negative"]
            s.update({"threshold": tau, "category_id": cid, "metric": metric,
                      "category_name": F.CATEGORY_NAME.get(cid, str(cid)),
                      "majority_sign": 1 if npos > nneg else (-1 if nneg > npos else 0),
                      "n_majority": max(npos, nneg)})
            eff_rows.append(s)
    category_effects = pd.DataFrame(eff_rows)

    dp = deltas[deltas.threshold == tau_primary]
    loo_rows = []
    for cid, g in dp.groupby("category_id"):
        for metric in ("all", "assigned"):
            t = leave_one_event_out(g, f"delta_{metric}").rename(
                columns={"median": "median_delta"})
            t.insert(0, "metric", metric)
            t.insert(0, "category_name", category_name.get(cid, str(cid)))
            t.insert(0, "category_id", cid)
            loo_rows.append(t)
    category_loo = pd.concat(loo_rows, ignore_index=True) if loo_rows else pd.DataFrame()

    # ---------------- domain-level compositional analysis ------------------
    pr = prev[prev.threshold == tau_primary].copy()
    pr["domain"] = pr.category_id.map(category_domain)
    cnt = (pr.groupby(["event_id", "register", "domain"]).n_category.sum()
             .unstack("domain").reindex(columns=domains).fillna(0).astype(int))
    si = domains.index(primary_domain)

    comp_rows, n_zeros = [], 0
    comps_by_event = {}
    for e in manifest.event_id:
        rec = {"event_id": e}
        comps = {}
        for reg in ("HIGHLIGHT", "FULL_RACE"):
            p, nz = bayes_multiplicative_replace(cnt.loc[(e, reg)].to_numpy())
            n_zeros += nz
            comps[reg] = p
            rec[f"n_assigned_{reg}"] = int(cnt.loc[(e, reg)].sum())
            rec[f"n_zero_parts_{reg}"] = nz
            for i, d in enumerate(domains):
                rec[f"p_{reg}_{d}"] = float(p[i])
            rec[f"logit_S_{reg}"] = logit(float(p[si]))
        comps_by_event[e] = comps
        rec["delta_logit_S"] = rec["logit_S_HIGHLIGHT"] - rec["logit_S_FULL_RACE"]
        rec["delta_raw_p_S"] = float(comps["HIGHLIGHT"][si] - comps["FULL_RACE"][si])
        ch, cf = clr(comps["HIGHLIGHT"]), clr(comps["FULL_RACE"])
        for i, d in enumerate(domains):
            rec[f"delta_clr_{d}"] = float(ch[i] - cf[i])
        ih = ilr(comps["HIGHLIGHT"], domains, ilr_balances)
        iff = ilr(comps["FULL_RACE"], domains, ilr_balances)
        for j in range(len(ilr_balances)):
            rec[f"delta_ilr_b{j + 1}"] = float(ih[j] - iff[j])
        for i, d in enumerate(domains):
            if d == primary_domain:
                continue
            rec[f"delta_logratio_S_over_{d}"] = float(
                np.log(comps["HIGHLIGHT"][si] / comps["HIGHLIGHT"][i])
                - np.log(comps["FULL_RACE"][si] / comps["FULL_RACE"][i]))
        comp_rows.append(rec)
    compositional = pd.DataFrame(comp_rows)

    def _summarise(col, label):
        v = compositional[col].to_numpy()
        loo = [float(np.median(np.delete(v, i))) for i in range(len(v))]
        med = float(np.median(v))
        return {"metric": label, "column": col, "median": med,
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
                "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                "min": float(v.min()), "max": float(v.max()),
                "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum()),
                "sign_test_p": _sign_test_p(v),
                "loo_sign_stable": bool(len({int(np.sign(x)) for x in loo}) == 1
                                        and np.sign(loo[0]) == np.sign(med)),
                "loo_median_min": float(min(loo)), "loo_median_max": float(max(loo))}

    summ = [_summarise("delta_raw_p_S", f"raw proportion, {primary_domain}"),
            _summarise("delta_logit_S", f"B1 logit, {primary_domain} vs rest")]
    summ += [_summarise(f"delta_clr_{d}", f"B2 CLR, {d}") for d in domains]
    for j, (num, den) in enumerate(ilr_balances):
        summ.append(_summarise(f"delta_ilr_b{j + 1}",
                               f"B2 ILR, b{j + 1} {'+'.join(num)} vs {'+'.join(den)}"))
    summ += [_summarise(f"delta_logratio_S_over_{d}", f"B3 log(S/{d})")
             for d in domains if d != primary_domain]
    compositional_summary = pd.DataFrame(summ)

    # ---------------- matched-length null ---------------------------------
    null_domain, null_event, null_blocks = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if run_matched_length_null:
        dom_i = {d: i for i, d in enumerate(domains)}
        cat_i = {c: i for i, c in enumerate(ids)}
        cat_to_dom = np.array([dom_i[category_domain[c]] for c in ids])
        dom_rows, ev_rows, blk_rows = [], [], []
        for ei, (stem, rid) in enumerate(pairs.items()):
            Xe = X_second[s_idx[stem]]
            lab_s, _, ok_s = assign_by_centroid(Xe, C, ids, tau_primary)
            # nearest centroid regardless of threshold, so an unassigned row
            # still carries an index; the threshold mask decides what counts.
            near = np.asarray(ids)[(Xe @ C.T).argmax(1)]
            ci = np.array([cat_i[c] for c in near])
            Fn = len(Xe)
            Hn = int(len(p_idx[rid]))

            fr_all, fr_asg = _prevalence_vector(ci, ok_s, len(domains), cat_to_dom)
            hl = pr[(pr.event_id == rid) & (pr.register == "HIGHLIGHT")]
            hl_dom = (hl.groupby("domain")[["p_all", "p_assigned"]].sum()
                        .reindex(domains).fillna(0))

            rng = np.random.default_rng([null_seed, ei])
            nd_all = np.empty((B, len(domains)))
            nd_asg = np.empty((B, len(domains)))
            for b in range(B):
                idx = rng.choice(Fn, size=Hn, replace=False)
                nd_all[b], nd_asg[b] = _prevalence_vector(
                    ci[idx], ok_s[idx], len(domains), cat_to_dom)

            for metric, nullm, obs, ref in (
                    ("assigned", nd_asg, hl_dom.p_assigned.to_numpy(), fr_asg),
                    ("all", nd_all, hl_dom.p_all.to_numpy(), fr_all)):
                ne, oe = nullm - ref, obs - ref
                for di, dn in enumerate(domains):
                    v, o = ne[:, di], oe[di]
                    lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
                    dom_rows.append({
                        "event_id": rid, "domain": dn, "metric": metric,
                        "H_r": Hn, "F_r": Fn, "observed_effect": float(o),
                        "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
                        "null_p2.5": float(lo), "null_p50": float(md),
                        "null_p97.5": float(hi),
                        "p_one_sided_depletion": empirical_p(v, o, "lower"),
                        "p_one_sided_enrichment": empirical_p(v, o, "upper"),
                        "outside_central_95": bool(o < lo or o > hi),
                        "below_null_p2.5": bool(o < lo)})
            o_s = float(hl_dom.p_assigned.iloc[si] - fr_asg[si])
            v_s = nd_asg[:, si] - fr_asg[si]
            ev_rows.append({"event_id": rid, "H_r": Hn, "F_r": Fn,
                            "ratio": round(Fn / Hn, 1),
                            "observed_effect_assigned": o_s,
                            "null_p2.5_assigned": float(np.percentile(v_s, 2.5)),
                            "null_p97.5_assigned": float(np.percentile(v_s, 97.5)),
                            "p_depletion_assigned": empirical_p(v_s, o_s, "lower")})

            for k in block_ks:
                rngb = np.random.default_rng([null_seed, ei, k])
                vals, nfail = [], 0
                for b in range(B):
                    idx = contiguous_blocks(rngb, Fn, Hn, k)
                    if idx is None:
                        nfail += 1
                        continue
                    _, pa = _prevalence_vector(ci[idx], ok_s[idx],
                                               len(domains), cat_to_dom)
                    vals.append(pa[si])
                v = np.array(vals) - fr_asg[si]
                blk_rows.append({
                    "event_id": rid, "k_blocks": k, "n_valid": len(v),
                    "n_failed": nfail, "observed_effect": o_s,
                    "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
                    "null_p2.5": float(np.percentile(v, 2.5)),
                    "null_p97.5": float(np.percentile(v, 97.5)),
                    "p_one_sided_depletion": empirical_p(v, o_s, "lower"),
                    "below_null_p2.5": bool(o_s < np.percentile(v, 2.5))})
            log(f"  {rid}: H={Hn} F={Fn} observed {o_s:+.4f}")
        null_domain = pd.DataFrame(dom_rows)
        null_event = pd.DataFrame(ev_rows)
        null_blocks = pd.DataFrame(blk_rows)

    # ---------------- headline event-level summary -------------------------
    row = compositional_summary.set_index("column")
    headline = {}
    for key, col in (("raw", "delta_raw_p_S"), ("logit", "delta_logit_S"),
                     ("clr", f"delta_clr_{primary_domain}"), ("ilr", "delta_ilr_b1")):
        r = row.loc[col]
        headline[key] = {"median": round(float(r["median"]), 4),
                         "sign_test_p": round(float(r["sign_test_p"]), 4),
                         "n_negative": int(r["n_negative"]),
                         "loo_sign_stable": bool(r["loo_sign_stable"])}
    out = {"status": "COMPLETED",
           "n_matched_events": len(manifest),
           "inferential_unit": "event",
           "tau_primary": tau_primary, "tau_sensitivity": taus,
           "n_zero_parts_replaced": int(n_zeros),
           "primary_domain": primary_domain,
           "headline": headline,
           "manifest": manifest, "coverage": coverage_tab, "prevalence": prev,
           "event_deltas": deltas, "category_effects": category_effects,
           "category_loo": category_loo,
           "compositional": compositional,
           "compositional_summary": compositional_summary,
           "matched_length_null_domain": null_domain,
           "matched_length_null_event": null_event,
           "matched_length_null_contiguous": null_blocks}
    if len(null_domain):
        st = null_domain[(null_domain.domain == primary_domain)
                         & (null_domain.metric == "assigned")]
        out["events_below_own_95_envelope"] = int(st["below_null_p2.5"].sum())
    log(f"matched-event composition: {primary_domain} median "
        f"{headline['raw']['median']:+.4f} (p={headline['raw']['sign_test_p']:.4f}), "
        f"{headline['raw']['n_negative']}/{len(manifest)} events negative")
    return out


# ----------------------------------------- domain-membership robustness -----
def _composition_effects(prev, manifest, category_domain, domains,
                         primary_domain, ilr_balances, tau):
    """Event-level effects for one domain membership. Re-aggregation only.

    The counts, the zero replacement, the logit, the CLR and the pre-specified
    ILR balance are the canonical ones defined above; only the category-to-domain
    map varies. No clustering is rerun and no sentence is reassigned.
    """
    pr = prev[prev.threshold == tau].copy()
    pr["domain"] = pr.category_id.map(category_domain)
    cnt = (pr.groupby(["event_id", "register", "domain"]).n_category.sum()
             .unstack("domain").reindex(columns=domains).fillna(0).astype(int))
    si = domains.index(primary_domain)
    raw, lg, cl, il = [], [], [], []
    for e in manifest.event_id:
        comps = {}
        for reg in ("HIGHLIGHT", "FULL_RACE"):
            p, _ = bayes_multiplicative_replace(cnt.loc[(e, reg)].to_numpy())
            comps[reg] = p
        raw.append(float(comps["HIGHLIGHT"][si] - comps["FULL_RACE"][si]))
        lg.append(logit(float(comps["HIGHLIGHT"][si]))
                  - logit(float(comps["FULL_RACE"][si])))
        cl.append(float(clr(comps["HIGHLIGHT"])[si] - clr(comps["FULL_RACE"])[si]))
        il.append(float(ilr(comps["HIGHLIGHT"], domains, ilr_balances)[0]
                        - ilr(comps["FULL_RACE"], domains, ilr_balances)[0]))
    out = {}
    for key, v in (("raw", raw), ("logit", lg), ("clr", cl), ("ilr", il)):
        a = np.asarray(v, dtype=float)
        out[f"median_{key}_delta"] = round(float(np.median(a)), 4)
        out[f"n_negative_{key}"] = int((a < 0).sum())
        out[f"p_sign_{key}"] = round(_sign_test_p(a), 4)
    out["n_events"] = len(raw)
    return out


def domain_membership_robustness(prev, manifest, category_domain, category_name,
                                 domains=None, primary_domain=None,
                                 ilr_balances=None, tau=None,
                                 member_ids=None) -> tuple:
    """Systematic leave-one-category-out and exploratory boundary reassignment.

    Returns (leave_one_out, boundary_cases). The first drops each primary-domain
    category in turn, so that domain is defined by the remaining members. The
    second moves the pre-identified boundary categories to the comparison domain;
    it is an exploratory stress test, not part of the pre-specified plan.
    """
    domains = list(domains or F.DOMAINS)
    primary_domain = primary_domain or F.PRIMARY_DOMAIN
    ilr_balances = ilr_balances or F.ILR_BALANCES
    tau = F.TAU_PRIMARY if tau is None else tau
    members = sorted(member_ids if member_ids is not None else
                     [c for c, d in category_domain.items() if d == primary_domain])
    base = _composition_effects(prev, manifest, category_domain, domains,
                                primary_domain, ilr_balances, tau)

    loo = [{"variant": "ALL MEMBERS (baseline)",
            "analysis_type": "baseline", "removed_category": "",
            "n_primary_domain_categories": len(members), **base}]
    for c in members:
        m = {k: v for k, v in category_domain.items() if k != c}
        loo.append({"variant": f"drop {category_name.get(c, c)}",
                    "analysis_type": "systematic leave-one-category-out",
                    "removed_category": category_name.get(c, str(c)),
                    "n_primary_domain_categories": len(members) - 1,
                    **_composition_effects(prev, manifest, m, domains,
                                           primary_domain, ilr_balances, tau)})

    boundary = [{"variant": "NO REASSIGNMENT (baseline)",
                 "analysis_type": "baseline", "reassigned_categories": "",
                 "exploratory": False,
                 "n_primary_domain_categories": len(members), **base}]
    bnames = F.BOUNDARY_CATEGORIES
    bids = [c for c in members if category_name.get(c) in bnames]
    target = F.BOUNDARY_TARGET_DOMAIN
    for group in [[c] for c in bids] + ([bids] if len(bids) > 1 else []):
        m = dict(category_domain)
        for c in group:
            m[c] = target
        names = ", ".join(category_name.get(c, str(c)) for c in group)
        boundary.append({
            "variant": f"move {names} -> {target}",
            "analysis_type": "exploratory stress test",
            "reassigned_categories": names, "exploratory": True,
            "n_primary_domain_categories": len(members) - len(group),
            **_composition_effects(prev, manifest, m, domains, primary_domain,
                                   ilr_balances, tau)})
    return pd.DataFrame(loo), pd.DataFrame(boundary)


PAIRED_OUTPUTS = (("paired_event_manifest", "manifest"),
                  ("paired_register_coverage", "coverage"),
                  ("paired_category_prevalence", "prevalence"),
                  ("paired_category_effects", "category_effects"),
                  ("paired_leave_one_out", "category_loo"),
                  ("compositional_domain_effects", "compositional"),
                  ("compositional_strategy_summary", "compositional_summary"),
                  ("matched_length_null_summary", "matched_length_null_domain"),
                  ("matched_length_null_event", "matched_length_null_event"),
                  ("matched_length_null_contiguous",
                   "matched_length_null_contiguous"))


def compare_matched_events(primary_centroids, primary_ids, X, sessions,
                           X2, sessions2, matched_events, out_dir,
                           category_domain, category_name=None) -> dict:
    """Matched-event compositional comparison, written out under content names.

    Both registers are assigned by the same frozen primary centroids so that
    register is not confounded with assignment method. The inferential unit is
    the event; pooled sentences are never treated as independent observations.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = compute_matched_event_composition(
        primary_centroids, primary_ids, X, sessions, X2, sessions2,
        matched_events, category_domain=category_domain,
        category_name=category_name)
    for name, key in PAIRED_OUTPUTS:
        if len(paired[key]):
            paired[key].to_csv(out_dir / f"{name}.csv", index=False)
    return paired
