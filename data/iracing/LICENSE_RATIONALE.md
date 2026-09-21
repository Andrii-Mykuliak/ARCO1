# Supplementary CC-BY Commentary Corpus — License Rationale

**Corpus.** `data/iracing/` contains **5,967 race-segment sentences** of transcribed spoken-word commentary from eight sim-racing live broadcasts on the *Fearless Broadcasting* YouTube channel (2026 vintage). The corpus is supplied as a supplementary demo for code-path verification, course exercises, and CI smoke-tests of the canonical Paper 1 pipeline. It does not replicate the canonical 34-cluster F1 highlights taxonomy and is not interpretable as transferability validation (cluster output differs from F1 highlights by register design: live ASR vs post-edited prose).

## Scope of the corpus

The corpus contains **only the transcribed commentary speech** — the channel author's spoken-word output, captured as text. It contains:

- **No iRacing audiovisual assets** — no game footage, no game audio, no interface elements, no track models, no in-game telemetry overlays.
- **No raw video files or audio waveforms** — only the resulting sentence-level text after transcription.
- **No commercial sim-racing or platform metadata beyond what is required for CC-BY attribution.**

This scope distinction matters: the object of release is the commentator's spoken-word expression as text, not anything covered by the iRacing platform's End-User License Agreement.

## License chain

The source videos are published by the *Fearless Broadcasting* YouTube channel under **Creative Commons Attribution 3.0 Unported (CC-BY 3.0)**. CC-BY permits users to:

1. **Share** the work (copy and redistribute in any medium or format), and
2. **Adapt** the work (remix, transform, and build upon),

subject to two conditions:

- **Attribution** — credit the source, provide a link to the license, and indicate if changes were made.
- **Indication of modifications** — clearly mark modified versions as such.

Following the interpretation adopted by **The People's Speech** (Galvez et al., NeurIPS 2021) §3.1 Licensing Description for transcript-based corpora derived from CC-BY-licensed audiovisual sources — under which "the steps to create a machine learning dataset" (including transcription, alignment, and downstream processing) are read as falling within CC-BY's share + adapt allowances — we treat the automatic transcription and downstream NLP pipeline applied here as a permitted format change and adaptation of the commentary text.

Comparable transcript-from-CC-BY-YouTube corpora published in this interpretive frame include:

- **YouTube-Commons** (PleIAs, 2024) — 2,063,066 videos, ~45 billion words, distributed on YouTube under CC-BY; full provenance per video (title, link, channel name, upload date); published under CC-BY 4.0. Backed by the French Ministry of Culture.
- **YODAS** (Li, Takamichi, Saeki, Chen, Shiota, Watanabe, 2024) — 500k+ hours of speech across 100+ languages from CC-licensed YouTube videos; data-collection methodology explicitly requires the video to be accompanied by a Creative Commons license; distributed under Creative Commons.
- **The People's Speech** (Galvez et al., NeurIPS 2021, MLCommons) — 30,000-hour supervised English speech-recognition dataset licensed for academic and commercial usage under CC-BY-SA (with a CC-BY subset). The canonical academic precedent for transcript-based corpora derived from CC-BY-licensed audiovisual sources.

## Compliance steps taken

In compliance with CC-BY:

- **License tag.** We rely on the CC-BY license tag as published by the channel. Recorded per video in `manifest.json` (per-video fields: title, video ID, upload date, license tag at time of access).
- **Attribution.** Each source video is attributed per CC-BY in `attribution.csv` (one row per source: channel name, video title, video URL, license version).
- **Indication of modifications.** Modifications applied to produce the derivative corpus are: (i) automatic transcription of the audio track; (ii) sentence-level segmentation by automatic cue detection (green-flag / checkered / winner-announcement cues); (iii) downstream NLP pipeline application (canonical entity masking, MiniLM embedding, UMAP, HDBSCAN, R3 enrichment) per Paper 1 §3.2 + §4.1-4.3.

The resulting text corpus is released as a **derivative of the commentary text** under **CC-BY 4.0**, the same-author compatible upgrade per CC-BY 3.0 §4(b).

## Limitations and known unresolved questions

1. **Channel-author identity.** The CC-BY publication confers reuse rights on the spoken text on the assumption that the commentator is the channel owner (and thus the copyright holder of the spoken-word expression). This has not been independently verified per source video; we rely on the channel's CC-BY publication as published.
2. **Third-party platform terms.** Sim-racing broadcasts on platforms with end-user terms (iRacing EULA, YouTube ToS) may impose obligations on the channel author. The contractual axis between the channel author and any underlying platform is between those two parties; whether and how it reaches a third-party recipient relying on the channel's CC-BY publication is a legal question outside the scope of this release. We rely on the channel's CC-BY publication as published; a formal legal review of this third-party-reliance axis is recommended before broader redistribution.

Until both points are independently closed, the corpus is shipped as a supplementary demo only.

## Files in this directory

- `manifest.json` — per-video metadata (title, ID, upload date, license tag, processing date)
- `attribution.csv` — CC-BY attribution per source video (channel name, title, URL, license version)
- `corpus.jsonl` — the 5,967-sentence derivative text corpus (one sentence per line, with source video ID and intra-video position)
- `LICENSE_RATIONALE.md` — this file
- `README.md` — short technical README for using the demo with the canonical Paper 1 pipeline

## Cross-references

Paper 1 §3.5 (one-paragraph mention with cross-reference here). Paper 1 §6.7 (one-sentence callback in future-work context). Paper 1 status report Open Item #5 documents the two unresolved questions above as prerequisites for inclusion in the formal submission package.

The full text of the CC-BY 3.0 and CC-BY 4.0 licenses is available at https://creativecommons.org/licenses/by/3.0/ and https://creativecommons.org/licenses/by/4.0/ respectively.