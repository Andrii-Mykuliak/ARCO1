"""Corpus loading and entity masking.

Masking is part of the input pipeline, not a preprocessing afterthought: the
embeddings the whole study runs on are computed over masked text, so the
gazetteer travels with the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .io_utils import cached, log


# ---------------------------------------------------------------- loading --
def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json_any(path: Path) -> list[dict]:
    """Accept .jsonl, or .json holding a list, or {sessions: {id: [sent...]}}."""
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        rows = []
        for key, val in obj.items():
            if isinstance(val, list):
                for i, s in enumerate(val):
                    if isinstance(s, str):
                        rows.append({"text": s, "session": key, "idx": i})
                    elif isinstance(s, dict):
                        s.setdefault("session", key)
                        rows.append(s)
        if rows:
            return rows
    raise ValueError(f"Unrecognised corpus layout: {path}")


#: Source-provenance fields carried through the pipeline when the corpus has
#: them, so any sentence on display can be traced back to the recording it came
#: from and the licence it is used under.
PROVENANCE_FIELDS = ("video_id", "video_url", "video_title", "channel",
                     "upload_date", "series", "license_spdx")


def load_corpus(cfg, sentences_path=None) -> pd.DataFrame:
    """Return sentence_id | session | text | token_count, plus any provenance.

    ``sentences_path`` overrides the configured primary corpus. The held-out
    coverage analysis must pass ``cfg.heldout_sentences_path`` explicitly:
    silently reusing the primary path would make the holdout meaningless.
    """
    src = sentences_path or cfg.sentences_path
    rows = _read_json_any(src)
    df = pd.DataFrame(rows)

    if cfg.text_field not in df.columns:
        raise KeyError(
            f"text field {cfg.text_field!r} not in corpus columns {list(df.columns)[:12]}"
        )
    df = df.rename(columns={cfg.text_field: "text"})

    session = cfg.session_field
    if session in df.columns:
        df["session"] = df[session].astype(str)
    elif "session" not in df.columns:
        df["session"] = "all"
    df["session"] = df["session"].astype(str)

    df["text"] = df["text"].astype(str).str.strip()
    df["token_count"] = df["text"].str.split().str.len()
    df = df[df["token_count"] >= cfg.min_tokens].reset_index(drop=True)

    if cfg.limit_sentences:
        df = df.head(cfg.limit_sentences).reset_index(drop=True)

    # A corpus may already carry its own sentence_id; reassign contiguously so
    # downstream positional indexing stays valid either way.
    if "sentence_id" in df.columns:
        df = df.drop(columns=["sentence_id"])
    df.insert(0, "sentence_id", range(len(df)))
    present = [c for c in PROVENANCE_FIELDS if c in df.columns]
    keep = ["sentence_id", "session", "text", "token_count"] + present
    log(f"corpus: {len(df):,} sentences across {df['session'].nunique()} sessions "
        f"[{Path(src).name}]")
    if present:
        log(f"corpus: provenance carried - {', '.join(present)}")
    else:
        log("corpus: no provenance fields found; sentences will not be traceable to a source")
    return df[keep]


def provenance_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per source recording: what it is, and how much of the corpus it is."""
    present = [c for c in PROVENANCE_FIELDS if c in df.columns]
    if "video_id" not in present:
        return pd.DataFrame([{"note": "corpus carries no per-sentence source fields"}])
    meta = [c for c in present if c != "video_id"]
    out = (df.groupby("video_id")
             .agg(sentences=("sentence_id", "size"),
                  **{c: (c, "first") for c in meta})
             .reset_index())
    out["pct_of_corpus"] = (100 * out["sentences"] / len(df)).round(1)
    return out.sort_values("sentences", ascending=False).reset_index(drop=True)


def source_labels(df: pd.DataFrame) -> list[str]:
    """Per-sentence origin label, for output that has its own source column."""
    if "video_title" not in df.columns:
        return [""] * len(df)
    tag = df["video_title"].astype(str)
    if "upload_date" in df.columns:
        tag = tag + ", " + df["upload_date"].astype(str)
    return tag.tolist()


def with_source(df: pd.DataFrame, texts) -> list[str]:
    """Sentence and origin as one string, for output that shows a bare sentence."""
    tags = source_labels(df)
    return [f"{t}  [{s}]" if s else str(t) for t, s in zip(texts, tags)]


# ---------------------------------------------------------------- masking --
def load_gazetteer(path: Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = obj.get("entries", obj) if isinstance(obj, dict) else obj
    return entries


def build_mask_patterns(entries, entity_classes):
    """Longest-variant-first so 'Max Verstappen' wins over 'Max'."""
    pats = []
    for e in entries:
        cat = e.get("category", "PERSON")
        for v in e.get("variants", [e.get("canonical", "")]):
            v = (v or "").strip()
            if len(v) < 2:
                continue
            repl = f"<{cat}>" if cat in entity_classes else " "
            pats.append((len(v), re.compile(rf"\b{re.escape(v)}\b('s)?", re.I), repl))
    pats.sort(key=lambda t: -t[0])
    return [(p, r) for _, p, r in pats]


def mask_texts(texts, patterns, ner=None, entity_classes=("PERSON", "TEAM", "PLACE")):
    """Gazetteer masking, with optional spaCy NER as a fallback pass."""
    NER_MAP = {"PERSON": "PERSON", "ORG": "TEAM", "GPE": "PLACE",
               "LOC": "PLACE", "FAC": "PLACE", "NORP": "PLACE"}
    out = []
    for t in texts:
        s = t
        for pat, repl in patterns:
            s = pat.sub(repl, s)
        if ner is not None:
            doc = ner(s)
            spans = [(e.start_char, e.end_char, NER_MAP[e.label_])
                     for e in doc.ents
                     if e.label_ in NER_MAP and NER_MAP[e.label_] in entity_classes
                     and "<" not in e.text]
            for a, b, lab in sorted(spans, reverse=True):
                s = s[:a] + f"<{lab}>" + s[b:]
        out.append(re.sub(r"\s+", " ", s).strip())
    return out


def source_hashes(texts) -> list[str]:
    """SHA-256 of the exact source text. Identity only - no normalisation."""
    import hashlib
    return [hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in texts]


def _masking_cache(cfg, df: pd.DataFrame, compute):
    """Reuse a masking cache only if it provably belongs to this corpus.

    sentence_id is positional, so ids and row counts cannot distinguish a cache
    written for a reordered or different corpus. Every row must carry the hash
    of the source text it was masked from.
    """
    path = None
    cache_dir = getattr(cfg, "cache_dir", None)
    if cache_dir is not None:
        path = Path(cache_dir) / f"{cfg.corpus_name}__masked.parquet"

    want = pd.Series(source_hashes(df["text"]), index=df["sentence_id"].to_numpy())
    if path is not None and path.exists():
        try:
            cache = pd.read_parquet(path)
        except Exception as exc:
            cache, reason = None, f"unreadable ({exc})"
        else:
            reason = None
            if "source_text_hash" not in cache.columns:
                reason = "legacy cache without source_text_hash"
            elif len(cache) != len(df):
                reason = f"row count {len(cache)} != corpus {len(df)}"
            elif not cache["sentence_id"].is_unique:
                reason = "duplicate sentence_id"
            elif set(cache["sentence_id"]) != set(df["sentence_id"]):
                reason = "sentence_id set differs from corpus"
            elif cache["source_text_hash"].isna().any():
                reason = "null source_text_hash"
            else:
                got = cache.set_index("sentence_id")["source_text_hash"]
                bad = int((got.reindex(want.index).to_numpy() != want.to_numpy()).sum())
                if bad:
                    reason = f"{bad} row(s) hash-mismatched against current source"
        if reason is None:
            log(f"masking: cache verified against source ({len(cache):,} rows)")
            return cache
        log(f"masking: REJECTED cache - {reason}; recomputing")

    out = compute()
    out["source_text_hash"] = want.reindex(out["sentence_id"].to_numpy()).to_numpy()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    return out


def apply_masking(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``masked`` column. Cached, because NER is slow."""

    # No masking configured -> the answer is exactly the source text. Caching it
    # would let a parquet written for some earlier corpus win over the current
    # one: sentence_id is positional, so a stale file of a different length
    # reattaches the wrong sentence while keeping the row count plausible.
    if not cfg.apply_masking:
        out = df.copy()
        out["masked"] = out["text"]
        log(f"masking: disabled; masked = text for {len(out):,} sentences")
        return out

    def _compute():
        entries = load_gazetteer(cfg.gazetteer_path)
        patterns = build_mask_patterns(entries, cfg.entity_classes)
        log(f"masking: {len(entries)} gazetteer entries -> {len(patterns)} patterns")
        ner = None
        if cfg.use_ner_fallback:
            import spacy
            ner = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            log("masking: spaCy NER fallback enabled")
        masked = mask_texts(df["text"].tolist(), patterns, ner, cfg.entity_classes)
        return pd.DataFrame({"sentence_id": df["sentence_id"], "masked": masked})

    masked_df = _masking_cache(cfg, df, _compute)
    out = df.merge(masked_df[["sentence_id", "masked"]], on="sentence_id", how="left")
    n_marked = out["masked"].str.contains("<", regex=False).sum()
    log(f"masking: {n_marked:,}/{len(out):,} sentences carry >=1 placeholder")
    return out


def masking_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-placeholder counts — an artefact in its own right."""
    rows = []
    for tag in ("<PERSON>", "<TEAM>", "<PLACE>"):
        n_sent = int(df["masked"].str.contains(tag, regex=False).sum())
        n_tot = int(df["masked"].str.count(re.escape(tag)).sum())
        rows.append({"placeholder": tag, "sentences": n_sent, "occurrences": n_tot})
    rows.append({"placeholder": "any",
                 "sentences": int(df["masked"].str.contains("<", regex=False).sum()),
                 "occurrences": int(df["masked"].str.count("<").sum())})
    return pd.DataFrame(rows)
