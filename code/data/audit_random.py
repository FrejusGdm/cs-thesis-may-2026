"""
audit_random.py — Quality audit for the 10K random Tatoeba data.

Checks for:
  - Exact duplicates (source side and source+target)
  - Empty or whitespace-only translations
  - Source-target length ratio outliers (> 3x median ratio)
  - Untranslated sentences (source == target or high character overlap)
  - Very short sentences (< 2 words on either side)

Outputs:
  - Quality report to stdout
  - Clean CSV with flagged rows removed (RANDOM-CLEAN)
  - Flagged rows CSV for manual review

Usage:
    python audit_random.py \
        --input /path/to/tatoeba_random_10k.csv \
        --output /path/to/tatoeba_random_clean.csv \
        --flagged /path/to/tatoeba_flagged.csv

    # With custom column names:
    python audit_random.py \
        --input data.csv \
        --src-col french \
        --tgt-col adja_translation \
        --output clean.csv
"""

import argparse
import os
from collections import Counter

import pandas as pd
import numpy as np


def check_duplicates(df, src_col, tgt_col):
    """Flag exact duplicates."""
    flags = pd.Series(False, index=df.index)

    # Source-side duplicates
    src_dups = df.duplicated(subset=[src_col], keep="first")
    # Source+target duplicates (exact parallel pair)
    pair_dups = df.duplicated(subset=[src_col, tgt_col], keep="first")

    flags |= pair_dups

    n_src_dups = src_dups.sum()
    n_pair_dups = pair_dups.sum()
    print(f"  Source-side duplicates: {n_src_dups}")
    print(f"  Exact pair duplicates: {n_pair_dups} (flagged for removal)")

    return flags, "duplicate"


def check_empty(df, src_col, tgt_col):
    """Flag empty or whitespace-only entries."""
    flags = (
        df[src_col].isna() |
        df[tgt_col].isna() |
        (df[src_col].astype(str).str.strip() == "") |
        (df[tgt_col].astype(str).str.strip() == "")
    )
    print(f"  Empty/whitespace entries: {flags.sum()}")
    return flags, "empty"


def check_length_ratio(df, src_col, tgt_col, threshold_multiplier=3.0):
    """Flag sentences with extreme source-target length ratios."""
    src_len = df[src_col].astype(str).str.split().str.len()
    tgt_len = df[tgt_col].astype(str).str.split().str.len()

    # Avoid division by zero
    tgt_len_safe = tgt_len.clip(lower=1)
    src_len_safe = src_len.clip(lower=1)

    ratio = src_len_safe / tgt_len_safe
    median_ratio = ratio.median()

    flags = (ratio > median_ratio * threshold_multiplier) | (ratio < median_ratio / threshold_multiplier)
    print(f"  Length ratio outliers (>{threshold_multiplier}x median): {flags.sum()}")
    print(f"    Median src/tgt ratio: {median_ratio:.2f}")

    return flags, "length_ratio"


def check_untranslated(df, src_col, tgt_col, overlap_threshold=0.8):
    """Flag likely untranslated sentences (source ~ target)."""
    flags = pd.Series(False, index=df.index)

    for idx in df.index:
        src = str(df.loc[idx, src_col]).lower().strip()
        tgt = str(df.loc[idx, tgt_col]).lower().strip()

        # Exact match
        if src == tgt:
            flags[idx] = True
            continue

        # Character overlap (Jaccard on character sets)
        src_chars = set(src)
        tgt_chars = set(tgt)
        if src_chars and tgt_chars:
            jaccard = len(src_chars & tgt_chars) / len(src_chars | tgt_chars)
            if jaccard > overlap_threshold and len(src) > 5:
                flags[idx] = True

    print(f"  Likely untranslated (high overlap): {flags.sum()}")
    return flags, "untranslated"


def check_very_short(df, src_col, tgt_col, min_words=2):
    """Flag very short sentences."""
    src_len = df[src_col].astype(str).str.split().str.len()
    tgt_len = df[tgt_col].astype(str).str.split().str.len()

    flags = (src_len < min_words) | (tgt_len < min_words)
    print(f"  Very short sentences (< {min_words} words): {flags.sum()}")
    return flags, "too_short"


def main():
    parser = argparse.ArgumentParser(description="Quality audit for random parallel data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", help="Output clean CSV (flagged rows removed)")
    parser.add_argument("--flagged", help="Output CSV of flagged rows for review")
    parser.add_argument("--src-col", default="french", help="Source column name")
    parser.add_argument("--tgt-col", default="adja_translation", help="Target column name")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    # Auto-detect column names
    aliases = {
        args.src_col: ["French", "french", "src", "source"],
        args.tgt_col: ["Translation", "adja_translation", "tgt", "target"],
    }
    col_map = {}
    for expected, candidates in aliases.items():
        if expected in df.columns:
            continue
        for alias in candidates:
            if alias in df.columns:
                col_map[alias] = expected
                break
        else:
            raise ValueError(
                f"Column '{expected}' not found in {args.input}. "
                f"Columns: {list(df.columns)}"
            )
    if col_map:
        df = df.rename(columns=col_map)
    print(f"  Loaded {len(df)} rows")

    # Run all checks
    print("\n=== Quality Checks ===")
    all_flags = pd.DataFrame(index=df.index)

    for check_fn in [check_duplicates, check_empty, check_length_ratio,
                     check_untranslated, check_very_short]:
        flags, name = check_fn(df, args.src_col, args.tgt_col)
        all_flags[name] = flags

    # Combine flags
    any_flagged = all_flags.any(axis=1)
    n_flagged = any_flagged.sum()
    n_clean = len(df) - n_flagged

    print(f"\n=== Summary ===")
    print(f"  Total rows: {len(df)}")
    print(f"  Flagged: {n_flagged} ({n_flagged/len(df)*100:.1f}%)")
    print(f"  Clean: {n_clean} ({n_clean/len(df)*100:.1f}%)")

    # Breakdown by flag type
    print(f"\n  Breakdown (rows can have multiple flags):")
    for col in all_flags.columns:
        count = all_flags[col].sum()
        if count > 0:
            print(f"    {col}: {count}")

    # Save clean data
    if args.output:
        clean_df = df[~any_flagged].reset_index(drop=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        clean_df.to_csv(args.output, index=False)
        print(f"\n  Saved {len(clean_df)} clean rows to {args.output}")

    # Save flagged data for review
    if args.flagged:
        flagged_df = df[any_flagged].copy()
        flagged_df["flag_reasons"] = all_flags[any_flagged].apply(
            lambda row: ", ".join([col for col in all_flags.columns if row[col]]),
            axis=1
        )
        os.makedirs(os.path.dirname(args.flagged) or ".", exist_ok=True)
        flagged_df.to_csv(args.flagged, index=False)
        print(f"  Saved {len(flagged_df)} flagged rows to {args.flagged}")

    # Show examples of flagged rows
    if n_flagged > 0:
        print(f"\n=== Sample Flagged Rows ===")
        flagged_sample = df[any_flagged].head(5)
        for idx, row in flagged_sample.iterrows():
            reasons = [col for col in all_flags.columns if all_flags.loc[idx, col]]
            print(f"  [{', '.join(reasons)}]")
            print(f"    src: {row[args.src_col]}")
            print(f"    tgt: {row[args.tgt_col]}")


if __name__ == "__main__":
    main()
