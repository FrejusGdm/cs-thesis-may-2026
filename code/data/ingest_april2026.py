#!/usr/bin/env python3
"""
ingest_april2026.py — Preprocess the April-2026 new-data batch.

Consumes whatever files are dropped under april-2026-new-data/raw/
(.docx, .csv, .tsv, .xlsx) and produces:

    april-2026-new-data/processed/new6k.tsv          # cleaned 2-col TSV (no header)
    april-2026-new-data/processed/ingest_report.md   # provenance + drop counts

Pipeline:
  1. Extract French/Adja pairs from each raw file.
  2. Apply Unicode NFC normalization + control-char cleanup.
  3. Deduplicate within-batch on the French side.
  4. Drop anything that also appears in the existing training pool
     (shared/random_train.tsv, shared/structured_train.tsv) — prevents
     double-counting if a Tatoeba sentence was re-translated.
  5. Hard-block any row whose French matches shared/test.tsv
     (test-set contamination guard).

Run from repo root:
    python experiments/data/ingest_april2026.py
Optional:
    --raw-dir / --output-dir to override default paths
    --min-tokens N   drop rows with < N French tokens (useful if GATITOS
                     contributes many 1-3 word dictionary entries)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW = REPO_ROOT / "april-2026-new-data" / "raw"
DEFAULT_OUT = REPO_ROOT / "april-2026-new-data" / "processed"
SHARED = REPO_ROOT / "experiments" / "data" / "splits" / "shared"
TEST_PATH = SHARED / "test.tsv"
RANDOM_POOL = SHARED / "random_train.tsv"
STRUCT_POOL = SHARED / "structured_train.tsv"

FR_COL_ALIASES = {"french", "fr", "source", "src", "fr_text", "fr_sentence"}
AJ_COL_ALIASES = {"adja", "aja", "aj", "target", "tgt", "translation", "adja_translation"}


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

_NONPRINT_CATS = {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
_NONPRINT_MAP = {
    ord(c): " "
    for c in (chr(i) for i in range(sys.maxunicode + 1))
    if unicodedata.category(c) in _NONPRINT_CATS
}
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_NONPRINT_MAP)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def _pick_cols(df: pd.DataFrame) -> tuple[str, str] | None:
    """Find the French and Adja columns in a DataFrame, or return None."""
    lowered = {c: str(c).strip().lower() for c in df.columns}
    fr_col = next((c for c, lc in lowered.items() if lc in FR_COL_ALIASES), None)
    aj_col = next((c for c, lc in lowered.items() if lc in AJ_COL_ALIASES), None)
    if fr_col is None or aj_col is None:
        # Fallback: first two columns, in order.
        if len(df.columns) >= 2:
            return df.columns[0], df.columns[1]
        return None
    return fr_col, aj_col


def extract_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported tabular extension: {path}")
    cols = _pick_cols(df)
    if cols is None:
        raise ValueError(f"Could not identify FR/AJ columns in {path}: {list(df.columns)}")
    fr, aj = cols
    return pd.DataFrame({"french": df[fr].astype(str), "adja": df[aj].astype(str)})


# Characters almost never seen in French but common in Adja (IPA / extended Latin).
_ADJA_CHARS = set("ɖɛɔŋɣʋɥʔᴐ")

# Leading "<num>." or "<num>.<tab>" prefix on every paragraph of these batches.
_NUM_PREFIX_RE = re.compile(r"^\s*\d+\s*[.)]\s*")


def _score_boundary(tokens: list[str], k: int, fr_vocab: set[str], aj_vocab: set[str]) -> float:
    """Score a candidate split point k (tokens[:k] = FR side, tokens[k:] = AJ side).

    A token helps the French side if it's in the French vocab and contains no
    Adja-specific character; it helps the Adja side if it's in the Adja vocab
    or contains an Adja-specific character.
    """
    if k == 0 or k == len(tokens):
        return -1.0  # degenerate splits get penalised
    fr_hits = aj_hits = 0
    for t in tokens[:k]:
        tl = t.lower()
        has_adja = any(c in _ADJA_CHARS for c in tl)
        if has_adja:
            fr_hits -= 1
        if tl in fr_vocab:
            fr_hits += 1
    for t in tokens[k:]:
        tl = t.lower()
        has_adja = any(c in _ADJA_CHARS for c in tl)
        if has_adja:
            aj_hits += 1
        if tl in aj_vocab:
            aj_hits += 1
    return fr_hits + aj_hits


def _lexicon_split(
    line: str,
    fr_vocab: set[str],
    aj_vocab: set[str],
) -> tuple[str, str, float]:
    """Split a French+Adja concatenated line using per-side vocab scoring.

    Returns (french, adja, confidence). Confidence is the raw score;
    callers can threshold it to flag low-quality splits.
    """
    tokens = line.split()
    if len(tokens) < 2:
        return line, "", 0.0
    best_k = len(tokens) // 2
    best_score = -1e9
    for k in range(1, len(tokens)):
        s = _score_boundary(tokens, k, fr_vocab, aj_vocab)
        if s > best_score:
            best_score, best_k = s, k
    return " ".join(tokens[:best_k]), " ".join(tokens[best_k:]), float(best_score)


def _strip_leading_number(line: str) -> str:
    # Some paragraphs have the line number twice (once with a tab, once with a
    # period+space) — apply repeatedly until the prefix no longer matches.
    prev = None
    while line != prev:
        prev = line
        line = _NUM_PREFIX_RE.sub("", line, count=1)
    return line


def extract_docx(
    path: Path,
    fr_vocab: set[str] | None = None,
    aj_vocab: set[str] | None = None,
    low_conf_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Pull French/Adja pairs from a .docx.

    Strategy, in order:
      1. If the document has tables, read 2+ column rows from them.
      2. For each remaining paragraph, strip the leading "<num>." prefix and:
         a. If the line contains a colon, split on the LAST colon.
         b. Else, if French and Adja vocabs are supplied, pick the best
            boundary via per-side vocab scoring.
      3. Rows we can't confidently split are appended to `low_conf_rows` for
         reporting and skipped here.
    """
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError(
            "python-docx is required for .docx ingestion. "
            "Install with: pip install python-docx"
        ) from e

    doc = docx.Document(str(path))

    rows: list[tuple[str, str]] = []

    # --- 1. Tables ---
    for tbl in doc.tables:
        for r in tbl.rows:
            cells = [c.text for c in r.cells]
            if len(cells) < 2:
                continue
            fr, aj = cells[0], cells[1]
            if fr.strip().lower() in FR_COL_ALIASES and aj.strip().lower() in AJ_COL_ALIASES:
                continue
            rows.append((fr, aj))
    if rows:
        return pd.DataFrame(rows, columns=["french", "adja"])

    # --- 2. Paragraphs ---
    fr_vocab = fr_vocab or set()
    aj_vocab = aj_vocab or set()
    # A heuristic threshold: below this the split is almost certainly wrong.
    CONF_MIN = 2.0

    for para in doc.paragraphs:
        raw = para.text.strip()
        if not raw:
            continue
        line = _strip_leading_number(raw)
        if not line:
            continue

        # 2a. Colon-separator path (PATRICE, AJAGBE).
        if ":" in line:
            left, _, right = line.rpartition(":")
            left = left.strip(" \t-—–|")
            right = right.strip()
            if left and right:
                rows.append((left, right))
                continue

        # 2b. Lexicon split (HUREINE-style concatenated pairs).
        fr, aj, conf = _lexicon_split(line, fr_vocab, aj_vocab)
        if fr and aj and conf >= CONF_MIN:
            rows.append((fr, aj))
        elif low_conf_rows is not None:
            low_conf_rows.append({
                "source": path.name,
                "line": line,
                "best_french": fr,
                "best_adja": aj,
                "confidence": conf,
            })

    if not rows:
        raise ValueError(
            f"No usable tables, colon-separated, or vocab-splittable paragraphs "
            f"found in {path}. Extend extract_docx() with a matching parser."
        )
    return pd.DataFrame(rows, columns=["french", "adja"])


def _convert_doc_to_docx(doc_path: Path) -> Path:
    """Convert a legacy .doc to .docx using macOS textutil (required on Darwin).

    Writes the converted file to a sibling temp location and returns its path.
    Raises if textutil is unavailable (caller can decide to skip).
    """
    textutil = shutil.which("textutil")
    if not textutil:
        raise RuntimeError(
            f".doc files need macOS `textutil` (or LibreOffice `soffice`) to "
            f"convert to .docx first; couldn't find either. File: {doc_path}"
        )
    out_dir = Path(tempfile.mkdtemp(prefix="doc2docx_"))
    out_path = out_dir / (doc_path.stem + ".docx")
    subprocess.run(
        [textutil, "-convert", "docx", "-output", str(out_path), str(doc_path)],
        check=True,
        capture_output=True,
    )
    return out_path


def build_side_vocabs() -> tuple[set[str], set[str]]:
    """Build French / Adja token vocabularies from the existing training pools.

    Used by the lexicon-based splitter when a source file lacks a clean FR/AJ
    separator (e.g. translator notebooks where sentences are concatenated).
    """
    fr_vocab: set[str] = set()
    aj_vocab: set[str] = set()
    for pool in (RANDOM_POOL, STRUCT_POOL):
        if not pool.exists():
            continue
        df = pd.read_csv(pool, sep="\t", header=None, names=["fr", "aj"], dtype=str)
        for s in df["fr"].dropna():
            fr_vocab.update(t.lower() for t in clean(s).split())
        for s in df["aj"].dropna():
            aj_vocab.update(t.lower() for t in clean(s).split())
    return fr_vocab, aj_vocab


def load_raw_dir(raw_dir: Path, low_conf: list[dict]) -> list[tuple[str, pd.DataFrame]]:
    """Return [(filename, df), ...] for every ingestable file under raw_dir."""
    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    batches: list[tuple[str, pd.DataFrame]] = []

    fr_vocab: set[str] | None = None
    aj_vocab: set[str] | None = None

    for p in files:
        ext = p.suffix.lower()
        if ext == ".doc":
            try:
                converted = _convert_doc_to_docx(p)
                print(f"  converted: {p.name} -> {converted.name}")
            except Exception as e:
                print(f"  skip: {p.name} ({e})")
                continue
            target, label = converted, p.name
            ext = ".docx"
        else:
            target, label = p, p.name

        if ext == ".docx":
            if fr_vocab is None:
                fr_vocab, aj_vocab = build_side_vocabs()
                print(f"  built vocabs: |FR|={len(fr_vocab):,}  |AJ|={len(aj_vocab):,}")
            df = extract_docx(target, fr_vocab=fr_vocab, aj_vocab=aj_vocab, low_conf_rows=low_conf)
        elif ext in {".csv", ".tsv", ".xlsx", ".xls"}:
            df = extract_tabular(target)
        else:
            print(f"  skip: {label} (unsupported extension {ext})")
            continue
        batches.append((label, df))
        print(f"  loaded: {label} ({len(df):,} rows)")
    return batches


# ---------------------------------------------------------------------------
# Pool loading (existing splits)
# ---------------------------------------------------------------------------

def load_french_set(path: Path) -> set[str]:
    if not path.exists():
        print(f"  WARN: pool file missing: {path}")
        return set()
    df = pd.read_csv(path, sep="\t", header=None, names=["french", "adja"], dtype=str)
    return {clean(x).lower() for x in df["french"].dropna()}


# ---------------------------------------------------------------------------
# Simple French language heuristic (no external deps)
# ---------------------------------------------------------------------------

_FR_COMMON = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "en",
    "est", "il", "elle", "je", "tu", "nous", "vous", "ils", "elles",
    "que", "qui", "pas", "ne", "sur", "dans", "pour", "avec", "ce", "ça",
    "mon", "ma", "mes", "ton", "son", "sa", "ses", "au", "aux",
}


def looks_french(text: str) -> bool:
    tokens = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _FR_COMMON)
    return hits >= 1 or len(tokens) <= 2  # short dictionary entries get a pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Drop rows with fewer than N French tokens (default 0 = keep all)")
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"ERROR: {raw_dir} is empty. Drop raw files there first.")
        return 1

    print(f"Raw dir:    {raw_dir}")
    print(f"Output dir: {out_dir}")
    print()

    # ----- 1. Load all raw files -----
    print("=== 1. Loading raw files ===")
    low_conf: list[dict] = []
    batches = load_raw_dir(raw_dir, low_conf)
    if not batches:
        print("ERROR: no ingestable files found.")
        return 1
    if low_conf:
        lc_path = out_dir / "needs_review.tsv"
        pd.DataFrame(low_conf).to_csv(lc_path, sep="\t", index=False)
        print(f"  flagged {len(low_conf):,} low-confidence rows -> {lc_path}")

    per_file_counts = {name: len(df) for name, df in batches}
    combined = pd.concat(
        [df.assign(_source=name) for name, df in batches],
        ignore_index=True,
    )
    raw_total = len(combined)
    print(f"  raw total: {raw_total:,}")

    # ----- 2. Clean / normalize -----
    print("\n=== 2. Clean + NFC normalize ===")
    combined["french"] = combined["french"].map(clean)
    combined["adja"] = combined["adja"].map(clean)

    empty_mask = (combined["french"] == "") | (combined["adja"] == "")
    identical_mask = combined["french"] == combined["adja"]
    drop_mask = empty_mask | identical_mask
    n_empty = int(empty_mask.sum())
    n_identical = int((identical_mask & ~empty_mask).sum())
    combined = combined[~drop_mask].reset_index(drop=True)
    print(f"  dropped empty: {n_empty}")
    print(f"  dropped FR==AJ: {n_identical}")
    print(f"  after clean: {len(combined):,}")

    # ----- 2c. Expand alternative translations (Adja "A/ B" → two rows) -----
    # Source format: when an Adja translation has "/" with whitespace, the
    # author meant "either of these works". We split into one row per alt so
    # the model sees both surface forms paired with the same French sentence.
    print("\n=== 2c. Expand Adja alternatives on '/' ===")
    expanded_rows = []
    n_expansions = 0
    for _, r in combined.iterrows():
        aj = r["adja"]
        # Match "A/ B", "A/B", "A / B" — split on slash with optional whitespace.
        # Skip lines that are just one slash inside a single word (unlikely here).
        parts = [p.strip() for p in re.split(r"\s*/\s*", aj) if p.strip()]
        if len(parts) >= 2:
            n_expansions += len(parts) - 1
            for alt in parts:
                expanded_rows.append({**r.to_dict(), "adja": alt})
        else:
            expanded_rows.append(r.to_dict())
    combined = pd.DataFrame(expanded_rows).reset_index(drop=True)
    print(f"  rows added by alternative expansion: {n_expansions}")

    # ----- 2b. Optional length filter -----
    n_short = 0
    if args.min_tokens > 0:
        token_counts = combined["french"].str.split().map(len)
        keep = token_counts >= args.min_tokens
        n_short = int((~keep).sum())
        combined = combined[keep].reset_index(drop=True)
        print(f"  dropped <{args.min_tokens} tokens: {n_short}")

    # ----- 3. Within-batch dedup -----
    # Dedup on the (french, adja) pair so that legitimate alternative
    # translations (one French sentence with multiple valid Adja renderings,
    # introduced by step 2c) all survive.
    print("\n=== 3. Within-batch dedup (exact (french, adja) pair) ===")
    before = len(combined)
    combined = combined.drop_duplicates(subset=["french", "adja"], keep="first").reset_index(drop=True)
    n_internal_dup = before - len(combined)
    print(f"  dropped internal dups: {n_internal_dup}")

    # ----- 4. Contamination guard against existing test set -----
    print("\n=== 4. Contamination guard vs shared/test.tsv ===")
    test_fr = load_french_set(TEST_PATH)
    fr_lower = combined["french"].str.lower()
    contam_mask = fr_lower.isin(test_fr)
    n_contam = int(contam_mask.sum())
    combined = combined[~contam_mask].reset_index(drop=True)
    print(f"  dropped test-set matches: {n_contam}")

    # ----- 5. Dedup against existing train pools -----
    print("\n=== 5. Dedup vs existing train pools ===")
    random_fr = load_french_set(RANDOM_POOL)
    struct_fr = load_french_set(STRUCT_POOL)
    fr_lower = combined["french"].str.lower()
    in_random = fr_lower.isin(random_fr)
    in_struct = fr_lower.isin(struct_fr)
    n_random_dup = int(in_random.sum())
    n_struct_dup = int(in_struct.sum())
    combined = combined[~(in_random | in_struct)].reset_index(drop=True)
    print(f"  dropped random-pool matches: {n_random_dup}")
    print(f"  dropped structured-pool matches: {n_struct_dup}")

    # ----- 6. Language-id spot-check -----
    print("\n=== 6. French-side language-id spot-check ===")
    fr_ok = combined["french"].map(looks_french)
    pct_fr = 100.0 * fr_ok.sum() / max(len(combined), 1)
    print(f"  looks French: {fr_ok.sum():,}/{len(combined):,} ({pct_fr:.2f}%)")

    # ----- Write output (TSV for training, CSV for inspection) -----
    out_tsv = out_dir / "new6k.tsv"
    combined[["french", "adja"]].to_csv(out_tsv, sep="\t", header=False, index=False)
    out_csv = out_dir / "new6k.csv"
    combined[["french", "adja"]].to_csv(out_csv, index=False)  # has header
    print(f"\nWrote {out_tsv} ({len(combined):,} rows)")
    print(f"Wrote {out_csv} ({len(combined):,} rows, with header)")

    # ----- Sanity stats for report -----
    token_counts = combined["french"].str.split().map(len)
    buckets = {
        "1": int((token_counts == 1).sum()),
        "2-3": int(((token_counts >= 2) & (token_counts <= 3)).sum()),
        "4-7": int(((token_counts >= 4) & (token_counts <= 7)).sum()),
        "8-15": int(((token_counts >= 8) & (token_counts <= 15)).sum()),
        "16+": int((token_counts >= 16).sum()),
    }

    per_source_final = Counter(combined["_source"].tolist()) if "_source" in combined.columns else {}

    report_path = out_dir / "ingest_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# April-2026 Ingest Report\n\n")
        f.write(f"**Raw dir:** `{raw_dir.relative_to(REPO_ROOT)}`\n\n")
        f.write(f"**Output file:** `{out_tsv.relative_to(REPO_ROOT)}`\n\n")
        f.write("## Per-file raw counts\n\n")
        f.write("| File | Raw rows | Kept (after all filters) |\n")
        f.write("|---|---:|---:|\n")
        for name, n in per_file_counts.items():
            kept = per_source_final.get(name, 0)
            f.write(f"| {name} | {n:,} | {kept:,} |\n")
        f.write("\n## Drop summary\n\n")
        f.write(f"- raw total: **{raw_total:,}**\n")
        f.write(f"- dropped empty: {n_empty}\n")
        f.write(f"- dropped FR==AJ: {n_identical}\n")
        if args.min_tokens > 0:
            f.write(f"- dropped <{args.min_tokens} tokens: {n_short}\n")
        f.write(f"- dropped internal dups: {n_internal_dup}\n")
        f.write(f"- dropped test-set matches: **{n_contam}** (removed before writing; "
                "Run A comparability preserved)\n")
        f.write(f"- dropped random-pool matches: {n_random_dup}\n")
        f.write(f"- dropped structured-pool matches: {n_struct_dup}\n")
        if low_conf:
            f.write(f"- flagged low-confidence splits (see needs_review.tsv): "
                    f"**{len(low_conf):,}**\n")
        f.write(f"- **final kept: {len(combined):,}**\n\n")
        f.write("## French-side length distribution (token buckets)\n\n")
        f.write("| Tokens | Count |\n|---|---:|\n")
        for k, v in buckets.items():
            f.write(f"| {k} | {v:,} |\n")
        f.write("\n## French-side language-id\n\n")
        f.write(f"- heuristic match rate: **{pct_fr:.2f}%** "
                f"({fr_ok.sum():,}/{len(combined):,})\n")
        f.write("- heuristic uses common French function words; short dictionary "
                "entries (≤2 tokens) pass by default.\n\n")
        f.write("## Sample\n\n```\n")
        for _, row in combined.head(5).iterrows():
            f.write(f"{row['french']}\t{row['adja']}\n")
        f.write("```\n")
    print(f"Wrote {report_path}")

    # Machine-readable sidecar for pipeline automation.
    (out_dir / "ingest_report.json").write_text(json.dumps({
        "raw_total": raw_total,
        "final_kept": len(combined),
        "dropped": {
            "empty": n_empty,
            "fr_eq_aj": n_identical,
            "short": n_short,
            "internal_dup": n_internal_dup,
            "test_contamination": n_contam,
            "random_pool_dup": n_random_dup,
            "struct_pool_dup": n_struct_dup,
        },
        "low_confidence_flagged": len(low_conf),
        "per_file_raw": per_file_counts,
        "per_file_kept": dict(per_source_final),
        "french_lang_id_pct": pct_fr,
        "length_buckets": buckets,
    }, indent=2))

    # The contamination guard already removed matching rows before writing.
    # Surface the count prominently so the user knows to audit the source,
    # but don't fail — downstream pipeline can safely consume new6k.tsv.
    if n_contam > 0:
        print(f"\nNOTE: {n_contam} row(s) matched the held-out test set and were "
              "removed from new6k.tsv. Verify the source file for why a test "
              "sentence leaked in, but Run A comparability is preserved.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
