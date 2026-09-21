**Canonical highlight clustering configuration and outcome.**

| Component                   | Setting / value                                                                                                      |
|:----------------------------|:---------------------------------------------------------------------------------------------------------------------|
| Embedding model             | sentence-transformers/all-MiniLM-L6-v2                                                                               |
| Entity masking              | v4 (<PERSON> / <TEAM> / <PLACE>)                                                                                     |
| UMAP                        | 15 nearest neighbours, minimum distance 0.0, 5 components, cosine metric, fixed random seed 42                       |
| HDBSCAN                     | euclidean metric, excess-of-mass selection, density parameter left unset, which hdbscan resolves to min_cluster_size |
| Minimum cluster size        | 35                                                                                                                   |
| Categories                  | 34                                                                                                                   |
| Noise fraction              | 0.3352                                                                                                               |
| Lexical enrichment coverage | 34/34 categories carry FDR-significant terms                                                                         |
