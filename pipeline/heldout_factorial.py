"""Held-out race replication over a factorial of splits, embeddings and granularity.

The canonical held-out diagnostic uses one deterministic race-level split. One
split cannot show whether its result is representative of plausible race
compositions, so this repeats the SAME pipeline over a fixed roster of split
seeds, at two training granularities, on two embedding realisations.

WHAT VARIES   the split seed, the embedding realisation, the training-split
              minimum cluster size.
WHAT DOES NOT the split algorithm, the projection seed (fixed, so split
              composition is not confounded with projection stochasticity), the
              alignment, the train-only vocabulary refit, the two-stage FDR and
              the within-split criterion.

Ported from ``scripts/axis3_factorial.py`` and ``scripts/axis3_factorial_analyse.py``
(historically Axis 3). The historical runner monkeypatched ``race_split`` to
inject the per-run seed and wrapped the stage to capture internal counts; neither
is reproduced here. The split is passed explicitly, and every captured quantity
is derived in this module from what the stage already returns.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from .heldout_replication import axis3_heldout_replication, race_split
from .io_utils import log

PROTOCOL = "heldout-factorial/1"
SEEDS = tuple(range(50))
MCS = (30, 35)
N_TRAIN = 80
PROJECTION_SEED = 42
THRESHOLD = 0.50
# Pre-specified degeneracy rule for this experiment. NOTE the first operand is
# the number of ALIGNED canonical categories recovered on the training split,
# not the raw training cluster count: a run can produce clusters that align to
# nothing. The resampling experiment uses a different quantity and its helper
# must not be reused here.
DEGENERATE_MIN_ALIGNED = 10
DEGENERATE_MIN_TRAIN_NOISE = 0.05


def is_degenerate(aligned_categories: int, train_noise_fraction: float) -> bool:
    return bool(aligned_categories < DEGENERATE_MIN_ALIGNED
                or train_noise_fraction < DEGENERATE_MIN_TRAIN_NOISE)


def split_for_seed(sessions, seed: int, n_train: int = N_TRAIN):
    """The per-seed race split, from the canonical split function."""
    return race_split(sessions, n_train=n_train, seed=seed)


def _run_config(base_cfg, mcs: int):
    """A per-run configuration: training granularity varies, nothing else."""
    from copy import copy
    cfg = copy(base_cfg)
    cfg.axis3_train_min_cluster_size = mcs
    cfg.axis3_n_train_sessions = N_TRAIN
    cfg.seed = PROJECTION_SEED
    return cfg


def run_once(base_cfg, docs, embedding, canonical, realisation: str,
             embedding_sha256: str, mcs: int, seed: int,
             text_col: str = "masked") -> dict:
    """One factorial member: one split, one embedding, one training granularity."""
    sessions = docs["session"]
    train_races, heldout_races = split_for_seed(sessions, seed)
    cfg = _run_config(base_cfg, mcs)
    t0 = time.time()
    out, stage = axis3_heldout_replication(
        cfg, docs, embedding, canonical, text_col=text_col,
        split=(train_races, heldout_races), return_stage=True)

    # The historical spy captured these five; all are derivable from the stage.
    labels, tr, te = stage["labels"], stage["train_mask"], stage["test_mask"]
    train_total, heldout_total = int(tr.sum()), int(te.sum())
    train_assigned = int((labels[tr] != -1).sum())
    heldout_assigned = int((labels[te] != -1).sum())
    aligned = len(set(labels[tr].tolist()) - {-1})
    noise = 1.0 - train_assigned / train_total

    return {
        "cache": realisation, "embedding_sha256": embedding_sha256,
        "mcs": mcs, "seed": seed, "umap_seed": PROJECTION_SEED,
        "n_train_races": len(train_races), "n_heldout_races": len(heldout_races),
        "train_races": ";".join(train_races),
        "heldout_races": ";".join(heldout_races),
        "aligned_categories": aligned,
        "train_total": train_total, "heldout_total": heldout_total,
        "train_assigned": train_assigned, "heldout_assigned": heldout_assigned,
        "train_noise_fraction": round(noise, 6),
        "is_degenerate": is_degenerate(aligned, noise),
        "n_categories": int(len(out)),
        "n_not_reconstructed": int((~out.has_estimate).sum()),
        "n_pass": int(out.axis3_pass.sum()),
        "pass_vector": ";".join(str(int(c))
                                for c in out.loc[out.axis3_pass, "cluster"]),
        "elapsed_sec": round(time.time() - t0, 1),
        "categories": [
            {"cluster": int(r.cluster), "reconstructed": bool(r.has_estimate),
             "num_strict": int(r.n_test_strict),
             "den_enriched": int(r.n_train_enriched),
             "num_dir": int(r.n_test_direction),
             "rep_strict": None if pd.isna(r.rep_strict) else float(r.rep_strict),
             "rep_dir": None if pd.isna(r.rep_dir) else float(r.rep_dir),
             "pass": bool(r.axis3_pass),
             "margin": (None if (pd.isna(r.rep_strict) or pd.isna(r.rep_dir))
                        else float(min(r.rep_strict - THRESHOLD,
                                       r.rep_dir - THRESHOLD)))}
            for r in out.itertuples()],
    }


def _identity(embeddings: dict, canonical, docs) -> dict:
    corpus = hashlib.sha256(
        "\n".join(docs["masked"].astype(str)).encode("utf-8")).hexdigest()
    return {"protocol": PROTOCOL,
            "embedding_sha256": {k: v[1] for k, v in sorted(embeddings.items())},
            "labels_sha256": hashlib.sha256(np.asarray(canonical).tobytes()).hexdigest(),
            "corpus_sha256": corpus,
            "seeds": list(SEEDS), "mcs": list(MCS), "n_train": N_TRAIN,
            "projection_seed": PROJECTION_SEED, "threshold": THRESHOLD}


def heldout_replication_factorial(cfg, embeddings: dict, docs, canonical,
                                  seeds=SEEDS, mcs=MCS,
                                  text_col: str = "masked") -> list[dict]:
    """The full factorial: len(seeds) x len(embeddings) x len(mcs) runs.

    ``embeddings`` maps realisation name -> (array, expected sha256). Results are
    cached under an identity covering every scientific dependency; a cache built
    under any other identity is refused rather than reused.
    """
    from pathlib import Path
    identity = _identity(embeddings, canonical, docs)
    base = Path(cfg.cache_dir) / "heldout_replication_factorial"
    meta_path, data_path = (base.with_suffix(".identity.json"),
                            base.with_suffix(".jsonl"))
    if data_path.exists() and meta_path.exists() and not cfg.from_scratch:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        drift = {k: (stored.get(k), v) for k, v in identity.items()
                 if stored.get(k) != v}
        if drift:
            raise ValueError(
                "the cached held-out factorial was produced under a different "
                "scientific identity and cannot be reused: "
                + "; ".join(f"{k} differs" for k in drift)
                + ". Remove that cache to recompute the experiment.")
        rows = [json.loads(ln) for ln in
                data_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        log(f"cache hit  {data_path.name} ({len(rows)} runs)")
        return rows

    rows = []
    total = len(seeds) * len(embeddings) * len(mcs)
    for name in sorted(embeddings):
        arr, sha = embeddings[name]
        for m in mcs:
            for s in seeds:
                rows.append(run_once(cfg, docs, arr, canonical, name, sha, m, s,
                                     text_col))
                if len(rows) % 10 == 0:
                    log(f"  factorial run {len(rows)}/{total}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta_path.write_text(json.dumps(identity, indent=1), encoding="utf-8")
    return rows


# ------------------------------------------------------------ aggregation ----
def run_records(rows) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "categories"}
                         for r in rows])


def category_records(rows) -> pd.DataFrame:
    out = []
    for r in rows:
        for c in r["categories"]:
            out.append({"cache": r["cache"], "mcs": r["mcs"], "seed": r["seed"],
                        "is_degenerate": r["is_degenerate"], **c})
    return pd.DataFrame(out)


def cell_summary(rows) -> pd.DataFrame:
    """Per (realisation, granularity): viable runs and their replication spread."""
    runs = run_records(rows)
    out = []
    for (cache, m), g in runs.groupby(["cache", "mcs"]):
        viable = g[~g.is_degenerate.astype(bool)]
        v = viable.n_pass.to_numpy(float)
        out.append({"cell": f"{cache}{m}", "cache": cache, "mcs": int(m),
                    "n_runs": int(len(g)),
                    "n_degenerate": int(g.is_degenerate.astype(bool).sum()),
                    "n_viable": int(len(viable)),
                    "mean": round(float(v.mean()), 6) if len(v) else None,
                    "median": float(np.median(v)) if len(v) else None,
                    "sd": round(float(v.std(ddof=1)), 6) if len(v) > 1 else None,
                    "iqr_lo": float(np.percentile(v, 25)) if len(v) else None,
                    "iqr_hi": float(np.percentile(v, 75)) if len(v) else None,
                    "min": float(v.min()) if len(v) else None,
                    "max": float(v.max()) if len(v) else None})
    return pd.DataFrame(out).sort_values(["cache", "mcs"]).reset_index(drop=True)


def degeneracy_summary(rows) -> pd.DataFrame:
    runs = run_records(rows)
    d = (runs.groupby(["cache", "mcs"])
         .agg(n_runs=("seed", "size"),
              n_degenerate=("is_degenerate", "sum"),
              min_aligned=("aligned_categories", "min"),
              max_aligned=("aligned_categories", "max"),
              min_noise=("train_noise_fraction", "min"),
              max_noise=("train_noise_fraction", "max")).reset_index())
    d["degeneracy_rate"] = (d.n_degenerate / d.n_runs).round(6)
    return d


def category_summary(rows, category_name=None) -> pd.DataFrame:
    """How often each category replicates, over viable runs only."""
    category_name = category_name or {}
    cats = category_records(rows)
    viable = cats[~cats.is_degenerate.astype(bool)]
    out = (viable.groupby("cluster")
           .agg(n_viable_runs=("pass", "size"), n_pass=("pass", "sum"),
                median_rep_strict=("rep_strict", "median"),
                median_rep_dir=("rep_dir", "median"),
                n_reconstructed=("reconstructed", "sum")).reset_index())
    out["pass_rate"] = (out.n_pass / out.n_viable_runs).round(6)
    out.insert(1, "category_label",
               out.cluster.map(category_name).fillna(out.cluster.astype(str)))
    return out


def embedding_sensitivity(rows) -> pd.DataFrame:
    """Paired A-minus-B difference at each granularity, both runs viable."""
    runs = run_records(rows)
    out = []
    for m, g in runs.groupby("mcs"):
        a = g[g.cache == "A"].set_index("seed")
        b = g[g.cache == "B"].set_index("seed")
        both = [s for s in a.index if s in b.index
                and not a.loc[s, "is_degenerate"] and not b.loc[s, "is_degenerate"]]
        d = np.array([a.loc[s, "n_pass"] - b.loc[s, "n_pass"] for s in both],
                     dtype=float)
        out.append({"mcs": int(m), "n_seeds": int(len(a)),
                    "n_pairs_both_viable": len(both),
                    "median_delta_A_minus_B": float(np.median(d)) if len(d) else None,
                    "iqr_lo": float(np.percentile(d, 25)) if len(d) else None,
                    "iqr_hi": float(np.percentile(d, 75)) if len(d) else None,
                    "min_delta": float(d.min()) if len(d) else None,
                    "max_delta": float(d.max()) if len(d) else None,
                    "n_A_gt_B": int((d > 0).sum()), "n_equal": int((d == 0).sum()),
                    "n_A_lt_B": int((d < 0).sum()),
                    "mean_abs_delta": float(np.abs(d).mean()) if len(d) else None})
    return pd.DataFrame(out)


def granularity_sensitivity(rows) -> pd.DataFrame:
    """Paired mcs30-minus-mcs35 difference within each realisation."""
    runs = run_records(rows)
    out = []
    for cache, g in runs.groupby("cache"):
        a = g[g.mcs == 30].set_index("seed")
        b = g[g.mcs == 35].set_index("seed")
        both = [s for s in a.index if s in b.index
                and not a.loc[s, "is_degenerate"] and not b.loc[s, "is_degenerate"]]
        d = np.array([a.loc[s, "n_pass"] - b.loc[s, "n_pass"] for s in both],
                     dtype=float)
        out.append({"cache": cache, "n_seeds": int(len(a)),
                    "degenerate_mcs30": int(a.is_degenerate.astype(bool).sum()),
                    "degenerate_mcs35": int(b.is_degenerate.astype(bool).sum()),
                    "n_pairs_both_viable": len(both),
                    "median_delta_30_minus_35": float(np.median(d)) if len(d) else None,
                    "iqr_lo": float(np.percentile(d, 25)) if len(d) else None,
                    "iqr_hi": float(np.percentile(d, 75)) if len(d) else None,
                    "min_delta": float(d.min()) if len(d) else None,
                    "max_delta": float(d.max()) if len(d) else None,
                    "n_30_gt_35": int((d > 0).sum()), "n_equal": int((d == 0).sum()),
                    "n_30_lt_35": int((d < 0).sum())})
    return pd.DataFrame(out)


def reference_seed_location(rows, reference_seed: int = PROJECTION_SEED) -> pd.DataFrame:
    """Where the reference split sits within its cell's viable distribution."""
    runs = run_records(rows)
    out = []
    for (cache, m), g in runs.groupby(["cache", "mcs"]):
        viable = g[~g.is_degenerate.astype(bool)]
        row = g[g.seed == reference_seed]
        if not len(row):
            continue
        n_pass = int(row.n_pass.iloc[0])
        degen = bool(row.is_degenerate.iloc[0])
        v = viable.n_pass.to_numpy(float)
        pct = (float((v < n_pass).mean() * 100) if len(v) and not degen else None)
        out.append({"cell": f"{cache}{m}", "cache": cache, "mcs": int(m),
                    "reference_seed": reference_seed,
                    "reference_n_pass": n_pass,
                    "reference_degenerate": degen,
                    "n_viable": int(len(viable)),
                    "percentile_among_viable": pct})
    return pd.DataFrame(out)


def threshold_margins(rows) -> pd.DataFrame:
    """Distance to the replication threshold, per category over viable runs."""
    cats = category_records(rows)
    v = cats[(~cats.is_degenerate.astype(bool)) & cats.margin.notna()]
    out = (v.groupby("cluster")
           .agg(n=("margin", "size"), median_margin=("margin", "median"),
                min_margin=("margin", "min"), max_margin=("margin", "max"),
                n_within_0_05=("margin", lambda s: int((s.abs() <= 0.05).sum())))
           .reset_index())
    return out


def factorial_summary(rows) -> dict:
    runs = run_records(rows)
    return {"protocol": PROTOCOL, "n_runs": int(len(runs)),
            "expected_runs": len(SEEDS) * 2 * len(MCS),
            "n_degenerate": int(runs.is_degenerate.astype(bool).sum()),
            "n_viable": int((~runs.is_degenerate.astype(bool)).sum()),
            "seeds": len(SEEDS), "mcs": list(MCS),
            "realisations": sorted(runs.cache.unique().tolist()),
            "projection_seed": PROJECTION_SEED, "threshold": THRESHOLD}
