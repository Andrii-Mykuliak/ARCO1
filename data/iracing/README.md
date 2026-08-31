# Demo corpus — iRacing CC-BY broadcast commentary

Supplementary corpus for **reproducibility-without-canonical-data**: lets readers
run the full F1-paper pipeline (masking → embedding → UMAP → HDBSCAN → R3)
end-to-end on a real, openly-licensed sport-broadcast corpus without needing the
restricted F1 canonical data.

## Scope

| Use case | Supported |
|---|:---:|
| Code-path verification (does the pipeline run? do outputs have expected shape?) | ✅ |
| Course / teaching exercises (students experiment on real broadcast data) | ✅ |
| CI smoke-tests (catch pipeline regressions) | ✅ |
| Reproducing F1 paper's 34-cluster taxonomy | ❌ different register |
| Claiming cross-corpus transferability of the F1 protocol | ❌ not validated |
| Benchmarking against F1 results | ❌ apples-to-oranges (live ASR vs post-edited highlights) |

The demo is **a code-path artefact, not an analytical replication**.
F1 highlights and iRacing live broadcasts are structurally different registers
(post-edited prose vs live ASR auto-captions); the cluster output will differ
from the F1 taxonomy by register design, not by methodological failure.

## Source

8 Creative Commons Attribution (CC-BY 3.0) videos from the
**Fearless Broadcasting** YouTube channel
([@fearlessbroadcasting](https://www.youtube.com/@fearlessbroadcasting)),
2026 vintage. License of each source video verified via `yt-dlp --print "%(license)s"`
on **2026-06-09** and recorded in `manifest.json` → each video's
`license.verified_via` field.

## Files

| File | Content |
|---|---|
| `manifest.json` | Per-video metadata: title, channel, upload date, duration, license, race-segment cut boundaries, attribution string. |
| `sentences.jsonl` | **5,967 race-segment sentences**, one per line, with provenance fields denormalised for join-free use. |
| `gazetteer.json` | **106 verified gazetteer entries** (83 PERSON, 9 TEAM, 6 PLACE, 8 broadcaster-meta DROP). Every variant verified to appear ≥1× in the corpus; built from the corpus itself, not speculated. |
| `attribution.csv` | Per-video CC-BY attribution string, for use in publications using this demo. |
| `code/` | Python scripts: end-to-end pipeline. |

## Code

| Script | Stage |
|---|---|
| `00_generate_attribution.py` | Builds `attribution.csv` from `manifest.json` |
| `01_download_subs.py` | yt-dlp download of json3 word-level YouTube auto-captions for all CC-BY-verified videos in manifest |
| `02_clean_and_cut.py` | Cleans ASR text, sentence-splits, cuts race segments by automatic cue detection (green flag → checkered flag) |
| `03_pipeline.py` | Applies gazetteer + spaCy NER masking → MiniLM embedding → UMAP → HDBSCAN → R3 log-odds per cluster |
| `run_demo.py` | End-to-end orchestrator (calls 01–03) |

## Quick start

```bash
# Dependencies: yt-dlp, sentence-transformers, umap-learn, hdbscan, spacy, pandas
pip install yt-dlp sentence-transformers umap-learn hdbscan spacy pandas
python -m spacy download en_core_web_sm

# Option A — end-to-end (re-downloads from YouTube):
python code/run_demo.py

# Option B — skip download, use pre-shipped sentences.jsonl:
python code/run_demo.py --skip-download

# Output: pipeline_out/cluster_cards.json, sentences_clustered.parquet
```

## License

- **Source videos**: Creative Commons Attribution 3.0 Unported (CC-BY 3.0).
  See <https://creativecommons.org/licenses/by/3.0/>. Per-video verification in
  `manifest.json`.
- **Derivative release** (this corpus): Creative Commons Attribution 4.0 International
  (CC-BY 4.0), per CC-BY 3.0 §4(b) compatible-upgrade clause. See
  <https://creativecommons.org/licenses/by/4.0/>.

### Attribution requirement

When using this corpus or any derivative output, you must cite each source
video used. See `attribution.csv` for ready-made citation strings. Example:

> Fearless Broadcasting. *Victory Lane Outlaws.* YouTube, 2026-05-29.
> <https://www.youtube.com/watch?v=VYTJiGWe4fo>. Licensed under Creative
> Commons Attribution license 3.0 (CC-BY-3.0),
> <https://creativecommons.org/licenses/by/3.0/>.

## Extraction pipeline notes

- **yt-dlp** json3 format: word-level YouTube auto-generated captions (no rolling-cue duplication).
- **Race-segment cut**: sentence-aligned automatic boundary detection.
  Start cues (priority): `green flag in the air`, `here we go`, `away we go`, `waving the flag`, `coming to the green flag`, `we're underway`.
  End cues priority-tiered: Tier 1 (race-end moment) — `there's your winner`, `(coming to|taking|takes|gets) the checkered`, `checkered flag is (out|up|in the air)`; Tier 2 (post-race transition) — `your winner of (tonight's race|the race)`, `congratulations on your (win|finish)`, `brought home (first|second|third)`, `(he|she|that) (is|was) your winner`.
- **Sentence segmentation**: regex `(?<=[.!?])\s*(?=[A-Z])`, then drop sentences with <4 tokens (ASR fragments).

## Gazetteer notes

`gazetteer.json` (106 entries) was built from the corpus itself with the
following discipline:
- Every entry was verified to appear ≥1× in the cleaned corpus (no
  speculatively-added "ASR variants" that never occur)
- ASR variants are collected from corpus inspection, not invented (e.g.,
  Stephen Hasset has 14 verified spelling variants, Sage Greco has 5, etc.)
- Common-word collisions excluded (`tank`, `chase`, `looks`, etc. removed
  even when they overlap with driver-name patterns)
- 4 categories: PERSON (driver/personnel), TEAM (team/car livery), PLACE
  (track/venue/region), DROP (broadcaster/sponsor meta — replaced with
  space rather than masked).

## Pipeline configuration choice

The demo uses **`min_cluster_size = 35`** rather than the F1 canonical
`mcs = 30`. Rationale:

- F1 canonical mcs=30 was validated specifically for **F1 highlights register**
  (post-edited prose). iRacing is **live broadcast ASR** — different register.
- Within the iRacing demo corpus, mcs=35 sits inside a stable region
  (cluster count 36, noise 34.0%, largest cluster 9.5% of corpus) — neither
  at the lower edge of plateau (where over-fragmentation begins) nor at
  the upper edge (where mega-cluster collapse begins; ≥40).
- mcs=35 gives a cluster count **comparable in structural scale to F1's
  34-cluster baseline** (Δ=2 clusters, Δ=0.5pp noise, Δ=0.5pp largest cluster
  proportion) — enabling apples-to-apples thematic comparison.

This is **not a claim** that mcs=35 is "the right value" for iRacing
universally. It is an explicit choice to match structural scales for
comparability. The F1 paper's `mcs_sensitivity.csv` describes the full sweep
behavior across mcs values; users should re-sweep on any new corpus.

## Honest limitations

- **Live-broadcast register** — fragmentary sentences, name variants
  ("Stephen Hasset" vs "Stephen Hazit" vs "Stephen has it" — 14 forms in
  the gazetteer), mid-race driver interview chunks embedded in race-segment cuts.
- **Single channel / single commentator family** — `Fearless Broadcasting`
  only. Generalisation to other sim-racing channels not validated.
- **Sponsor-read residue** — some clusters surface broadcast-meta content
  (sponsor mentions, "ladies and gentlemen" transitions, camera direction)
  absent from F1 highlights register. This is a register difference, not
  a pipeline bug.
- **Race-segment cut is approximate** — cue patterns are register-specific
  and tuned to this commentator's style; other broadcasters may need
  re-anchored cues.

## Citation

If you use this demo in published work, please cite:

1. The F1 paper (the protocol this demo exercises) — see repository root
2. Each source video (per-video strings in `attribution.csv`)
3. (Optional) this demo corpus itself: *iRacing-FB-RaceSeg-2026 supplementary
   demo corpus for ["paper title"], `data/demo_iracing/`, 2026-06-10.*

---

**Generated**: 2026-06-10
**Pipeline**: see `code/`
**Source channel**: Fearless Broadcasting, <https://www.youtube.com/@fearlessbroadcasting>