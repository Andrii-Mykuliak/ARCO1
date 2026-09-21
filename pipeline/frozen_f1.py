"""Frozen Formula 1 constants used by the cross-register and paired layers.

Every value here is a pre-specified analysis input, not a result: the manual
four-domain membership, the matched event pairs, the availability thresholds and
the null settings. They are stored as code so a fresh clone can run the
cross-register layers without any internal repository.

Category labels and domain names are the only text in this file. No broadcast
transcript text is stored here or derivable from it.

Changing anything below changes the science. Do not tune these.
"""
from __future__ import annotations

# ------------------------------------------------- manual domain grouping --
#: The four broad domains are a manual, descriptive grouping of the 34 canonical
#: highlight categories (Section 5.1). They are used for exposition and for the
#: pre-specified matched-event compositional comparison; they are not an
#: independently recovered higher-level taxonomy.
DOMAINS = ["Race dynamics", "Narrative & meta", "Strategy & technical", "Regulatory"]
PRIMARY_DOMAIN = "Strategy & technical"

#: (canonical category id, category label, domain). Keyed by the canonical
#: highlight partition at min_cluster_size = 35.
CATEGORY_DOMAINS: list[tuple[int, str, str]] = [
    ( 0, 'DRS-assisted battles', 'Race dynamics'),
    ( 1, 'Time penalties', 'Regulatory'),
    ( 2, 'Lights-out race starts', 'Race dynamics'),
    ( 3, 'Mirror checks defending', 'Race dynamics'),
    ( 4, 'Race start launch', 'Race dynamics'),
    ( 5, 'Weather conditions', 'Strategy & technical'),
    ( 6, 'Grand Prix wins', 'Narrative & meta'),
    ( 7, 'Race start lights', 'Race dynamics'),
    ( 8, 'Race control flags', 'Regulatory'),
    ( 9, 'Formula 1 milestones', 'Narrative & meta'),
    (10, 'Chicane moves', 'Race dynamics'),
    (11, 'Team radio celebrations', 'Narrative & meta'),
    (12, 'Podium finishes', 'Narrative & meta'),
    (13, 'Championship & title', 'Narrative & meta'),
    (14, 'Pit stops', 'Strategy & technical'),
    (15, 'Tires & compounds', 'Strategy & technical'),
    (16, 'Safety car deployment', 'Regulatory'),
    (17, 'Gap & pace', 'Strategy & technical'),
    (18, 'Team dynamics', 'Narrative & meta'),
    (19, 'Race lead holding', 'Race dynamics'),
    (20, 'Wing & car damage', 'Strategy & technical'),
    (21, 'Braking & lockups', 'Race dynamics'),
    (22, 'Wheel-to-wheel', 'Race dynamics'),
    (23, 'Circuit action', 'Race dynamics'),
    (24, 'Inter-car distance', 'Strategy & technical'),
    (25, 'Driver & team references', 'Narrative & meta'),
    (26, 'Defending & attacking', 'Race dynamics'),
    (27, 'Overtake battles', 'Race dynamics'),
    (28, 'Corner battles', 'Race dynamics'),
    (29, 'Corner specific moves', 'Race dynamics'),
    (30, 'Lap pace & timing', 'Strategy & technical'),
    (31, 'Race leader changes', 'Race dynamics'),
    (32, 'Front position changes', 'Race dynamics'),
    (33, 'Field position changes', 'Race dynamics'),
]

CATEGORY_NAME = {c: n for c, n, _ in CATEGORY_DOMAINS}
CATEGORY_DOMAIN = {c: d for c, _, d in CATEGORY_DOMAINS}
N_CATEGORIES = len(CATEGORY_DOMAINS)

# ------------------------------------------------------- matched events --
#: full-race corpus race_id -> highlight corpus race_id. All ten events are
#: matched. 2024_brazil is the Sao Paulo GP, renamed from the Brazilian GP in
#: 2021; the alias is confirmed from transcript content.
MATCHED_EVENTS: dict[str, str] = {
    "2023_australia": "2023_Australia_GP",
    "2023_qatar": "2023_Qatar_GP",
    "2023_singapore": "2023_Singapore_GP",
    "2024_bahrain": "2024_Bahrain_GP",
    "2024_brazil": "2024_SaoPaulo_GP",
    "2024_italy": "2024_Italy_GP",
    "2024_lasvegas": "2024_LasVegas_GP",
    "2024_qatar": "2024_Qatar_GP",
    "2025_belgium": "2025_Belgium_GP",
    "2025_japan": "2025_Japan_GP",
}
N_MATCHED_EVENTS = len(MATCHED_EVENTS)

# ------------------------------------ second-register granularity selection --
#: The full-race sweep grid, and the documented fallback used because the
#: full-race corpus has no cluster-count plateau: maximum DBCV among
#: non-degenerate settings, cross-checked against adjacent-setting stability and
#: requiring zero degenerate outcomes across the seed roster.
FULLRACE_MCS_SWEEP = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 125, 150, 200]
FULLRACE_SEEDS = list(range(20))
FULLRACE_SELECTED_MCS = 40
DEGENERACY_MIN_CLUSTERS = 10          # n_clusters < 10 is degenerate
DEGENERACY_MIN_NOISE = 0.05           # noise_fraction < 0.05 is degenerate

# --------------------------------------------------- availability control --
#: Fixed BEFORE correspondence is computed. The taxonomy was discovered over 113
#: events; the full-race corpus has 10, so non-recovery is only interpretable
#: where the theme is present in the matched events at all.
WELL_MIN_SENTENCES, WELL_MIN_EVENTS = 15, 5
SPARSE_MIN_SENTENCES, SPARSE_MIN_EVENTS = 5, 2

# -------------------------------------------------- correspondence null --
CORRESPONDENCE_N_NULL = 1000
CORRESPONDENCE_NULL_SEED = 4242
CORRESPONDENCE_NULL_PERCENTILE = 95
CORRESPONDENCE_TOP_K = 15             # overlap@K over top enriched terms
#: Second reference scale: the 95th percentile of the off-diagonal
#: within-highlight centroid cosine, i.e. the similarity typical of two distinct
#: categories of the same register.
WITHIN_REGISTER_REFERENCE_PERCENTILE = 95

# ------------------------------------------------------ paired register --
TAU_PRIMARY = 0.40
TAU_SENSITIVITY = (0.35, 0.40, 0.45)
DELTA_ZERO = 0.5                      # Jeffreys pseudocount, zero replacement

#: Pre-specified sequential binary partition for the ILR basis. Fixed before
#: results were inspected; b1 is the Strategy-and-technical balance reported in
#: the manuscript.
ILR_BALANCES = [
    (["Strategy & technical"], ["Race dynamics", "Narrative & meta", "Regulatory"]),
    (["Race dynamics"], ["Narrative & meta", "Regulatory"]),
    (["Narrative & meta"], ["Regulatory"]),
]

# ------------------------------------------------------ matched-length null --
MATCHED_LENGTH_N_REPLICATES = 10_000
MATCHED_LENGTH_SEED = 42
CONTIGUOUS_BLOCK_KS = [1, 5, 10, 20]

# ---------------------------------------------------------- enrichment --
ENRICHMENT_Q = 0.05
ENRICHMENT_MIN_DF = 3
ENRICHMENT_NGRAM = (1, 2)
ENRICHMENT_MAX_FEATURES = 6000

# ------------------------------------------- domain-membership robustness --
# Categories at the Strategy/Race-dynamics boundary, identified before the
# exploratory reassignment stress test. Moving them is secondary evidence: the
# rule was not part of the original analysis plan.
BOUNDARY_CATEGORIES = ["Pit stops", "Gap & pace", "Wing & car damage"]
BOUNDARY_TARGET_DOMAIN = "Race dynamics"
