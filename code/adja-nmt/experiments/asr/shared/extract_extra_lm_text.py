#!/usr/bin/env python3
"""Extract Adja sentences from external NMT-project CSVs for LM training.

Reads two CSVs from the neurosymbolic NMT project:
  1. 10_000_for_data_paper_LREC_cleaned_v2_normalized.csv  (col: Translation)
  2. simple-dataset-enriched.csv                           (col: adja_translation)

Dedups, applies preprocessing (NFC + punctuation/spacing), writes one
sentence per line to data/extra_adja_text.txt.

Usage:
    python experiments/asr/shared/extract_extra_lm_text.py \\
        --csv1 /path/to/10_000_for_data_paper_LREC_cleaned_v2_normalized.csv \\
        --csv2 /path/to/simple-dataset-enriched.csv \\
        --output data/extra_adja_text.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

# Make shared package importable when run from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from preprocessing import preprocess_for_lm


def extract_column(csv_path: Path, column: str) -> list[str]:
    """Read a CSV and return non-empty string values from `column`."""
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in {csv_path.name}. "
                         f"Available: {list(df.columns)}")
    raw = df[column].dropna().astype(str).tolist()
    return [s.strip() for s in raw if s.strip()]


def main():
    p = argparse.ArgumentParser(description="Extract Adja text for LM training")
    p.add_argument("--csv1", type=str, required=True,
                   help="Path to 10_000_..._normalized.csv (col: Translation)")
    p.add_argument("--csv2", type=str, required=True,
                   help="Path to simple-dataset-enriched.csv (col: adja_translation)")
    p.add_argument("--output", type=str, default="data/extra_adja_text.txt")
    args = p.parse_args()

    # --- Load ---
    print(f"Loading {args.csv1} ...")
    sents1 = extract_column(Path(args.csv1), "Translation")
    print(f"  {len(sents1)} raw sentences")

    print(f"Loading {args.csv2} ...")
    sents2 = extract_column(Path(args.csv2), "adja_translation")
    print(f"  {len(sents2)} raw sentences")

    combined = sents1 + sents2
    print(f"\nCombined raw: {len(combined)} sentences")

    # --- Preprocess ---
    print("Applying NFC + punctuation/spacing cleanup ...")
    normalized = [preprocess_for_lm(s) for s in combined]
    normalized = [s for s in normalized if s]  # drop empties
    print(f"After preprocessing: {len(normalized)}")

    # --- Dedupe ---
    unique = list(dict.fromkeys(normalized))  # preserves insertion order
    print(f"After dedup: {len(unique)} unique sentences")

    # --- Stats ---
    chars: Counter = Counter()
    for s in unique:
        chars.update(s)
    print(f"\nCharacter inventory: {len(chars)} unique chars")
    # Show ONLY the top 15 most-common plus any Adja-special chars at the bottom
    print("Top 15 chars by frequency:")
    for ch, n in chars.most_common(15):
        display = repr(ch) if ch in (" ", "\n", "\t") else ch
        print(f"  {display}  {n}")
    specials = [c for c in "ɛɔŋɖ" if c in chars]
    if specials:
        print(f"Adja specials found: {', '.join(f'{c}={chars[c]}' for c in specials)}")

    # --- Write ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in unique:
            f.write(s + "\n")
    print(f"\nWrote {len(unique)} sentences to {out}")


if __name__ == "__main__":
    main()
