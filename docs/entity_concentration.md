# Entity-concentration diagnostic

`pipeline/entity_concentration.py` measures how strongly a clustering is organised
around recurring named entities, and compares two clusterings of the same
sentences.

> **This diagnostic evaluates whether entity masking reduces within-cluster entity
> concentration. It is not a taxonomy-validation axis and does not determine which
> partition is semantically correct.**

It exists to test a **representation rationale**: if a preprocessing step is meant
to suppress grouping by entity identity, this quantifies whether it does. It is a
diagnostic, **not** a validation axis, and it says nothing about which partition is
more correct.

## Inputs

| Argument | Shape | Meaning |
|---|---|---|
| `entities` | list, one entry per sentence | `{class: set(entity_id)}` — sentence-level presence |
| `labels_a` | int array | reference partition; noise = `-1` |
| `labels_b` | int array | contrast partition; noise = `-1` |
| `classes` | tuple of str | entity classes to score, e.g. `("PERSON", "TEAM", "PLACE")` |

No corpus ships with this module. The caller supplies the entity annotation, from
whatever recogniser is appropriate. Two rules make the comparison meaningful:

1. **The same annotation must be used for both partitions**, derived from the same
   underlying text. If one partition is built on transformed text, the entity
   statistics must still come from the original.
2. **Incidence is sentence-level.** An entity counts once per sentence no matter how
   often it occurs; several distinct entities of one class may co-occur.

## Analysis population

`shared_clustered(a, b)` returns the sentences that are non-noise in **both**
partitions. Use it. Scoring each partition on its own non-noise set lets a
difference in noise membership masquerade as a difference in entity concentration.
Per-partition sets are still worth reporting as a sensitivity.

## Metrics, per cluster and entity class

| Metric | Definition |
|---|---|
| `cluster_size` | sentences in the cluster |
| `sentences_with_entity` | sentences carrying ≥ 1 entity of the class |
| `entity_coverage` | `sentences_with_entity / cluster_size` |
| `dominant_entity` | entity with the most sentence-level incidences |
| `dominant_entity_share_all` | `dominant_entity_count / cluster_size` |
| `dominant_entity_share_mentions` | `dominant_entity_count / sentences_with_entity` |
| `entity_entropy` | Shannon entropy of the incidence distribution |
| `entity_entropy_normalised` | `H / log(k)` |

The two denominators answer different questions: `share_all` is "how much of this
cluster is about one entity", `share_mentions` is "when an entity appears, how often
is it the same one". Report both.

**Entropy conventions are defined, not coerced:**

- `k = 0` distinct entities → `nan`. The cluster is **not evaluable** for that class
  and is excluded from aggregates. It is never counted as 0, which would fake
  maximal concentration.
- `k = 1` → `0.0`. Genuinely maximal concentration.
- `k > 1` → `H / log(k)` in `[0, 1]`.

## Aggregation

`aggregate()` reports median, IQR and cluster-size-weighted means. Weighting
matters — an unweighted median over many tiny clusters can move independently of
where the sentences actually are.

**Scope differs by metric.** `share_all` and `entity_coverage` are aggregated over
**all** non-noise clusters: a cluster with no entity of the class scores a valid 0,
and dropping such clusters inflates the mean — by different amounts in two
partitions that have different numbers of them, which biases the contrast between
them. `share_mentions` and `entropy_norm` are aggregated over **evaluable** clusters
only (`sentences_with_entity > 0`), where they are otherwise undefined.

**Cluster ids are never paired between partitions.** Two partitions of the same
sentences do not have corresponding clusters, and `compare_partitions` records
`cluster_ids_paired: False` to make that explicit.

## Uncertainty

`paired_bootstrap()` resamples sentences with replacement from the shared population
and rescores **both** partitions on the same resample, so the contrast keeps its
pairing. Returns the point estimate and a bootstrap 95 % CI on `b - a`.

`permutation_null()` holds the entity incidences and the observed **cluster-size
multiset** fixed and permutes sentence-to-cluster assignment. It answers: how much
concentration does this partition carry beyond what its own size distribution and
the corpus entity frequencies already imply? Reports observed, null mean/SD and a
one-sided empirical *p*, with the tail chosen by metric direction (higher share =
more concentrated; lower entropy = more concentrated).

Both take a `seed` and default to 1,000 replicates.

## Interpreting the output

Classify conservatively and per class. An effect on one entity class is not an
effect on entities in general — report the class. In the study this module was
written for, the effect was large for PERSON, modest for TEAM and absent for PLACE,
and the write-up says exactly that.

Do not use these numbers to claim one partition is more correct, more true, or more
semantically valid. They measure organisation around entity identity, nothing else.

## Example

```python
from pipeline.entity_concentration import compare_partitions

res = compare_partitions(entities, labels_masked, labels_unmasked,
                         classes=("PERSON", "TEAM", "PLACE"),
                         n_boot=1000, n_perm=1000, seed=42)

for b in res["bootstrap"]:
    print(b["entity_class"], b["metric"],
          b["point_estimate_b_minus_a"], b["ci_lo_95"], b["ci_hi_95"])
```

The stage runs on a synthetic fixture without a corpus, so it can be exercised
independently of the restricted data.
