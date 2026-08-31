"""Ablations that test design choices rather than measure the result.

Two questions the study's design invites:

  * **Does masking matter?** Masking is asserted to decouple thematic structure
    from entity frequency. Re-running the pipeline on unmasked text and
    comparing partitions turns that assertion into a measurement.
  * **Is the structure specific to one encoder?** Re-deriving the centroid
    grouping from an independent encoder family, with a label-permutation test,
    separates real structure from an artefact of one embedding space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clustering import cluster_hdbscan, cluster_ids, n_clusters, noise_rate, reduce_umap
from .domains import best_mapping_agreement, ward_groups
from .io_utils import cached, l2_normalise, log


def _agreement(a: np.ndarray, b: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    both = (a != -1) & (b != -1)
    return {
        "ari_all": round(float(adjusted_rand_score(a, b)), 4),
        "nmi_all": round(float(normalized_mutual_info_score(a, b)), 4),
        "ari_co_clustered": round(float(adjusted_rand_score(a[both], b[both])), 4)
        if both.sum() > 1 else np.nan,
        "n_co_clustered": int(both.sum()),
    }


def masking_ablation(cfg, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Re-run reduction + clustering on unmasked text and compare partitions.

    A high agreement would mean masking changes little on this corpus; a low one
    means entity names were shaping the geometry, which is precisely what the
    masking step exists to prevent.
    """

    def _compute():
        from .embedding import embed
        if not cfg.apply_masking:
            log("masking ablation skipped: corpus is not being masked")
            return pd.DataFrame([{"comparison": "n/a - masking disabled"}])

        Xu = embed(cfg, df["text"], tag="emb_unmasked")
        Uu = reduce_umap(cfg, Xu, tag="umap_unmasked", seed=cfg.seed)
        lab_u = cluster_hdbscan(Uu, cfg.min_cluster_size, cfg)

        row = {"comparison": "masked vs unmasked",
               "masked_clusters": n_clusters(labels),
               "unmasked_clusters": n_clusters(lab_u),
               "masked_noise_pct": round(100 * noise_rate(labels), 1),
               "unmasked_noise_pct": round(100 * noise_rate(lab_u), 1)}
        row.update(_agreement(labels, lab_u))
        log(f"masking ablation: ARI={row['ari_all']} "
            f"({row['masked_clusters']} vs {row['unmasked_clusters']} clusters)")
        return pd.DataFrame([row])

    return cached(cfg, "ablation_masking.csv", _compute)


def cross_encoder_stability(cfg, df: pd.DataFrame, labels: np.ndarray,
                            ids, C, alt_encoder: str) -> pd.DataFrame:
    """Re-derive the domain grouping from a second encoder family.

    The permutation test asks whether the agreement between the two encoders'
    K-way centroid groupings beats random relabelling.
    """

    def _compute():
        from .embedding import embed
        from sklearn.metrics import normalized_mutual_info_score

        cfg_alt = type(cfg)(**{**cfg.summary(),
                               "encoder": alt_encoder,
                               "results_dir": cfg.results_dir})
        cfg_alt.mode = cfg.mode
        Xb = embed(cfg_alt, df["masked"], tag="emb_alt_encoder")

        # same primary partition, centroids recomputed in the other space
        Cb = l2_normalise(np.stack([Xb[labels == c].mean(axis=0) for c in ids]))
        ga, _ = ward_groups(C, cfg.n_domains)
        gb, _ = ward_groups(Cb, cfg.n_domains)

        nmi = float(normalized_mutual_info_score(ga, gb))
        agree, _ = best_mapping_agreement(ga, gb)

        rng = np.random.default_rng(cfg.seed)
        null = [normalized_mutual_info_score(ga, rng.permutation(gb))
                for _ in range(2000)]
        p = float((np.array(null) >= nmi).mean())

        log(f"cross-encoder: NMI={nmi:.3f} vs permutation null "
            f"{np.mean(null):.3f} (p={p:.4f})")
        return pd.DataFrame([{
            "encoder_a": cfg.encoder.split("/")[-1],
            "encoder_b": alt_encoder.split("/")[-1],
            "n_domains": cfg.n_domains,
            "nmi": round(nmi, 4),
            "best_mapping_agreement": round(agree, 4),
            "permutation_null_mean_nmi": round(float(np.mean(null)), 4),
            "permutations": len(null),
            "p_value": p,
        }])

    return cached(cfg, "ablation_cross_encoder.csv", _compute)


def default_pipeline_ablation(cfg, df: pd.DataFrame, U: np.ndarray,
                              labels: np.ndarray, default_mcs: int = 10) -> pd.DataFrame:
    """Compare the chosen configuration against library-default granularity.

    Two contrasts: default granularity on the same (masked) representation,
    isolating the effect of the parameter; and default granularity on unmasked
    text, which is what an out-of-the-box run would produce.
    """

    def _compute():
        from .embedding import embed
        rows = []

        lab_d = cluster_hdbscan(U, default_mcs, cfg)
        r = {"variant": f"default granularity (mcs={default_mcs}), masked",
             "n_clusters": n_clusters(lab_d),
             "noise_pct": round(100 * noise_rate(lab_d), 1)}
        r.update(_agreement(labels, lab_d))
        rows.append(r)

        if cfg.apply_masking:
            Xu = embed(cfg, df["text"], tag="emb_unmasked")
            Uu = reduce_umap(cfg, Xu, tag="umap_unmasked", seed=cfg.seed)
            lab_u = cluster_hdbscan(Uu, default_mcs, cfg)
            r = {"variant": f"default granularity (mcs={default_mcs}), unmasked",
                 "n_clusters": n_clusters(lab_u),
                 "noise_pct": round(100 * noise_rate(lab_u), 1)}
            r.update(_agreement(labels, lab_u))
            rows.append(r)

        out = pd.DataFrame(rows)
        log(f"default-pipeline ablation: "
            f"{', '.join(f'{a}->ARI {b}' for a, b in zip(out['variant'], out['ari_all']))}")
        return out

    return cached(cfg, "ablation_default_pipeline.csv", _compute)
