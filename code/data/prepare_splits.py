"""
prepare_splits.py — Prepare train/val/test splits for all experiments.

Critical: For structured data, splits are GROUP-AWARE. All variants of a base
sentence (M1, M2, M3, M4, M5) stay in the same split. This prevents minimal-pair
leakage where the model could see a transformed version of a test sentence during
training.

This script:
1. Loads structured data (Run-1 + Run-2) and random Tatoeba data
2. Creates 80-10-10 train/val/test splits (grouped by base_sentence_id)
3. Creates per-experiment condition splits from the TRAIN portion only
4. Validates no data leakage between train and test
5. Reports vocabulary overlap statistics

Usage:
    # With single combined structured CSV:
    python prepare_splits.py \
        --structured /path/to/structured_combined.csv \
        --random /path/to/tatoeba_10k.csv \
        --output-dir ./splits

    # Or with separate run files:
    python prepare_splits.py \
        --structured-run1 /path/to/run1_combined.csv \
        --structured-run2 /path/to/run2_combined.csv \
        --random /path/to/tatoeba_10k.csv \
        --output-dir ./splits
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parallel_csv(path, src_col="french", tgt_col="adja_translation"):
    """Load a parallel corpus CSV and validate columns exist.

    Auto-detects common column name variants (e.g. 'French' -> 'french',
    'Translation' -> 'adja_translation') and renames them so downstream
    code always sees the expected names.
    """
    df = pd.read_csv(path)
    # Auto-detect column names — map common aliases to expected names
    aliases = {
        src_col: ["French", "french", "src", "source"],
        tgt_col: ["Translation", "adja_translation", "tgt", "target"],
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
                f"Column '{expected}' not found in {path}. "
                f"Columns: {list(df.columns)}"
            )
    if col_map:
        df = df.rename(columns=col_map)
    df = df.dropna(subset=[src_col, tgt_col])
    df = df[df[tgt_col].str.strip() != ""]
    return df


# ---------------------------------------------------------------------------
# Group-aware splitting (the critical fix)
# ---------------------------------------------------------------------------

def extract_base_group(df):
    """Extract the base sentence group for each row.

    For structured data, the group is the base_sentence_id (e.g., M1_0042).
    - Module 1 rows: use their own sentence_id (they ARE the base)
    - Module 2-5 rows: use base_sentence_id (links back to M1)

    For random data (no base_sentence_id column): each row is its own group.
    """
    if "base_sentence_id" in df.columns and "sentence_id" in df.columns:
        # Structured data: use base_sentence_id, falling back to sentence_id for M1
        groups = df["base_sentence_id"].fillna(df["sentence_id"])
        # For M1 rows where base_sentence_id might be NaN or empty
        mask = groups.isna() | (groups == "")
        groups[mask] = df.loc[mask, "sentence_id"]
        return groups
    elif "sentence_id" in df.columns:
        return df["sentence_id"]
    else:
        # Random data: each row is its own group
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def group_aware_split(df, val_ratio, test_ratio, seed):
    """Split data at the GROUP level so all variants of a base sentence stay together.

    Returns (train_df, val_df, test_df) with no group leakage.
    """
    rng = np.random.RandomState(seed)

    groups = extract_base_group(df)
    df = df.copy()
    df["_group"] = groups

    unique_groups = df["_group"].unique().tolist()
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_test = max(1, int(n_groups * test_ratio))
    n_val = max(1, int(n_groups * val_ratio))

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test:n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val:])

    test_df = df[df["_group"].isin(test_groups)].drop(columns=["_group"])
    val_df = df[df["_group"].isin(val_groups)].drop(columns=["_group"])
    train_df = df[df["_group"].isin(train_groups)].drop(columns=["_group"])

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def verify_no_group_leakage(train_df, val_df, test_df):
    """Assert that no base_sentence_id group appears in multiple splits."""
    train_groups = set(extract_base_group(train_df))
    val_groups = set(extract_base_group(val_df))
    test_groups = set(extract_base_group(test_df))

    train_test = train_groups & test_groups
    train_val = train_groups & val_groups
    val_test = val_groups & test_groups

    ok = True
    if train_test:
        print(f"  LEAK: {len(train_test)} groups in both train and test!")
        ok = False
    if train_val:
        print(f"  LEAK: {len(train_val)} groups in both train and val!")
        ok = False
    if val_test:
        print(f"  LEAK: {len(val_test)} groups in both val and test!")
        ok = False

    if ok:
        print("  OK: No group leakage across splits.")
    return ok


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_vocab(texts):
    """Compute vocabulary (set of unique lowercase tokens) from a series of texts."""
    vocab = set()
    for text in texts:
        vocab.update(str(text).lower().split())
    return vocab


def check_leakage(train_df, test_df, src_col="french"):
    """Check for exact-match sentence leakage between train and test."""
    train_sents = set(train_df[src_col].str.strip().str.lower())
    test_sents = set(test_df[src_col].str.strip().str.lower())
    overlap = train_sents & test_sents
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping sentences between train and test!")
        for s in list(overlap)[:5]:
            print(f"    - {s}")
    else:
        print("  OK: No exact-match sentence leakage.")
    return overlap


def vocab_overlap_stats(train_df, test_df, src_col="french"):
    """Compute vocabulary overlap between train and test sets."""
    train_vocab = compute_vocab(train_df[src_col])
    test_vocab = compute_vocab(test_df[src_col])
    overlap = train_vocab & test_vocab
    return {
        "train_vocab_size": len(train_vocab),
        "test_vocab_size": len(test_vocab),
        "overlap_size": len(overlap),
        "overlap_pct_of_test": len(overlap) / len(test_vocab) * 100 if test_vocab else 0,
        "test_oov_count": len(test_vocab - train_vocab),
        "test_oov_pct": len(test_vocab - train_vocab) / len(test_vocab) * 100 if test_vocab else 0,
    }


def subsample_structured(df, size, seed, stratify_col="module"):
    """Subsample structured data with stratification by module."""
    if size >= len(df):
        return df.copy()
    return df.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, size // df[stratify_col].nunique())),
                           random_state=seed)
    ).head(size).reset_index(drop=True)


def subsample_random(df, size, seed):
    """Subsample random data uniformly."""
    if size >= len(df):
        return df.copy()
    return df.sample(n=size, random_state=seed).reset_index(drop=True)


def make_train_val_split(df, val_ratio, seed):
    """Split data into train and validation sets (for per-condition splits)."""
    if len(df) == 0:
        return df.copy(), df.copy()
    val_size = max(1, int(len(df) * val_ratio))
    val_size = min(val_size, len(df) - 1)
    if val_size <= 0:
        return df.copy(), df.head(0)
    val_df = df.sample(n=val_size, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_split(df, output_dir, name, src_col="french", tgt_col="adja_translation"):
    """Save a split as both CSV (full metadata) and TSV (src\ttgt for training)."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    tsv_path = os.path.join(output_dir, f"{name}.tsv")
    df[[src_col, tgt_col]].to_csv(tsv_path, sep="\t", index=False, header=False)
    return csv_path, tsv_path


def data_fingerprint(df, src_col="french"):
    """Compute a deterministic fingerprint of a dataset for reproducibility."""
    content = "\n".join(sorted(df[src_col].astype(str).tolist()))
    return hashlib.md5(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Experiment-specific split preparation
# ---------------------------------------------------------------------------

def prepare_experiment1(random_train, structured_train, output_dir, val_ratio, seed,
                        src_col="french", tgt_col="adja_translation"):
    """Prepare all Experiment 1 conditions from TRAIN portion only."""
    print("\n=== Experiment 1: Primary Hypothesis ===")
    conditions = {}

    def _save(name, df):
        train, val = make_train_val_split(df, val_ratio, seed)
        save_split(train, os.path.join(output_dir, "exp1", name), "train", src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "exp1", name), "val", src_col, tgt_col)
        conditions[name] = {"train": len(train), "val": len(val)}
        print(f"  {name}: train={len(train)}, val={len(val)}")

    # RANDOM-10K (or however many random train sentences we have)
    _save("RANDOM-10K", random_train)

    # RANDOM-6K + STRUCTURED-4K
    random_6k = subsample_random(random_train, 6000, seed)
    _save("RANDOM-6K_STRUCTURED-4K", pd.concat([random_6k, structured_train], ignore_index=True))

    # RANDOM-10K + STRUCTURED-4K (full data)
    _save("RANDOM-10K_STRUCTURED-4K",
          pd.concat([random_train, structured_train], ignore_index=True))

    # STRUCTURED-4K-ONLY
    _save("STRUCTURED-4K-ONLY", structured_train)

    # RANDOM-4K
    _save("RANDOM-4K", subsample_random(random_train, 4000, seed))

    # STRUCTURED-2K
    _save("STRUCTURED-2K", subsample_structured(structured_train, 2000, seed))

    return conditions


def prepare_experiment2(random_train, structured_train, output_dir, val_ratio, seed,
                        src_col="french", tgt_col="adja_translation"):
    """Prepare all Experiment 2 (scaling curve) conditions from TRAIN portion only."""
    print("\n=== Experiment 2: Scaling Curves ===")
    conditions = {}

    def _save(name, df):
        train, val = make_train_val_split(df, val_ratio, seed)
        save_split(train, os.path.join(output_dir, "exp2", name), "train", src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "exp2", name), "val", src_col, tgt_col)
        conditions[name] = {"train": len(train), "val": len(val)}

    # Structured scaling
    for size in [200, 500, 1000, 2000, 3000, 4000]:
        _save(f"STRUCTURED-{size}", subsample_structured(structured_train, size, seed))

    # Random scaling
    for size in [200, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
        _save(f"RANDOM-{size}", subsample_random(random_train, size, seed))

    # Combined additive: 6K random + X structured
    random_6k = subsample_random(random_train, 6000, seed)
    for struct_size in [500, 1000, 2000, 4000]:
        struct_subset = subsample_structured(structured_train, struct_size, seed)
        _save(f"RANDOM-6K_STRUCTURED-{struct_size}",
              pd.concat([random_6k, struct_subset], ignore_index=True))

    # Combined replacement: (10K - X) random + X structured = 10K total
    for struct_size in [500, 1000, 2000, 4000]:
        random_size = 10000 - struct_size
        random_subset = subsample_random(random_train, random_size, seed)
        struct_subset = subsample_structured(structured_train, struct_size, seed)
        _save(f"REPLACE-R{random_size}_S{struct_size}",
              pd.concat([random_subset, struct_subset], ignore_index=True))

    print(f"  Created {len(conditions)} scaling conditions")
    return conditions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare 80-10-10 splits with group-aware splitting for structured data"
    )
    parser.add_argument("--structured-run1", help="Path to Run-1 structured CSV")
    parser.add_argument("--structured-run2", help="Path to Run-2 structured CSV")
    parser.add_argument("--structured", help="Path to single combined structured CSV (alternative to --structured-run1/run2)")
    parser.add_argument("--random", required=True, help="Path to 10K Tatoeba random CSV")
    parser.add_argument("--output-dir", default="./splits", help="Output directory for splits")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--src-col", default="french", help="Source language column name")
    parser.add_argument("--tgt-col", default="adja_translation", help="Target language column name")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Validate structured args
    if args.structured:
        if args.structured_run1 or args.structured_run2:
            parser.error("Use --structured OR (--structured-run1 + --structured-run2), not both")
    elif not (args.structured_run1 and args.structured_run2):
        parser.error("Provide either --structured or both --structured-run1 and --structured-run2")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading data...")
    if args.structured:
        structured_df = load_parallel_csv(args.structured, args.src_col, args.tgt_col)
        print(f"  Structured (combined): {len(structured_df)} sentences")
    else:
        run1 = load_parallel_csv(args.structured_run1, args.src_col, args.tgt_col)
        run2 = load_parallel_csv(args.structured_run2, args.src_col, args.tgt_col)
        structured_df = pd.concat([run1, run2], ignore_index=True)
        print(f"  Structured Run-1: {len(run1)} sentences")
        print(f"  Structured Run-2: {len(run2)} sentences")
        print(f"  Structured Combined: {len(structured_df)} sentences")
    random_df = load_parallel_csv(args.random, args.src_col, args.tgt_col)
    print(f"  Random Tatoeba: {len(random_df)} sentences")

    # Count base sentence groups
    groups = extract_base_group(structured_df)
    n_groups = groups.nunique()
    print(f"  Structured base groups: {n_groups} "
          f"(avg {len(structured_df)/n_groups:.1f} variants per group)")

    # -----------------------------------------------------------------------
    # Create 80-10-10 splits
    # -----------------------------------------------------------------------
    print(f"\nCreating splits (train={1-args.val_ratio-args.test_ratio:.0%} / "
          f"val={args.val_ratio:.0%} / test={args.test_ratio:.0%})...")

    # Structured data: GROUP-AWARE split (all variants stay together)
    print("\n  Structured data (group-aware split):")
    struct_train, struct_val, struct_test = group_aware_split(
        structured_df, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"    Train: {len(struct_train)} sentences")
    print(f"    Val:   {len(struct_val)} sentences")
    print(f"    Test:  {len(struct_test)} sentences")

    # Random data: simple row-level split
    print("\n  Random data (row-level split):")
    rand_train, rand_val, rand_test = group_aware_split(
        random_df, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"    Train: {len(rand_train)} sentences")
    print(f"    Val:   {len(rand_val)} sentences")
    print(f"    Test:  {len(rand_test)} sentences")

    # Combined test set (structured test + random test)
    test_df = pd.concat([struct_test, rand_test], ignore_index=True)
    print(f"\n  Combined test set: {len(test_df)} sentences "
          f"({len(struct_test)} structured + {len(rand_test)} random)")

    # -----------------------------------------------------------------------
    # Verify no leakage
    # -----------------------------------------------------------------------
    print("\n=== Leakage Verification ===")

    print("\nGroup-level leakage (structured):")
    verify_no_group_leakage(struct_train, struct_val, struct_test)

    print("\nSentence-level leakage (structured train vs test):")
    check_leakage(struct_train, struct_test, args.src_col)

    print("\nSentence-level leakage (random train vs test):")
    check_leakage(rand_train, rand_test, args.src_col)

    # Cross-dataset check: structured train vs random test and vice versa
    print("\nCross-dataset leakage (structured train vs random test):")
    check_leakage(struct_train, rand_test, args.src_col)

    # -----------------------------------------------------------------------
    # Save shared test set and train portions
    # -----------------------------------------------------------------------
    shared_dir = os.path.join(args.output_dir, "shared")
    save_split(test_df, shared_dir, "test", args.src_col, args.tgt_col)
    # Also save structured and random train CSVs (for baselines/ablations scripts)
    save_split(struct_train, shared_dir, "structured_train", args.src_col, args.tgt_col)
    save_split(rand_train, shared_dir, "random_train", args.src_col, args.tgt_col)
    print(f"\nSaved shared test set: {os.path.join(shared_dir, 'test.tsv')}")
    print(f"Saved structured train: {os.path.join(shared_dir, 'structured_train.tsv')}")
    print(f"Saved random train: {os.path.join(shared_dir, 'random_train.tsv')}")

    # -----------------------------------------------------------------------
    # Vocabulary overlap stats
    # -----------------------------------------------------------------------
    print("\n=== Vocabulary Overlap ===")
    for name, train_part in [("Structured", struct_train), ("Random", rand_train),
                              ("All train", pd.concat([struct_train, rand_train]))]:
        stats = vocab_overlap_stats(train_part, test_df, args.src_col)
        print(f"  {name} train vs Test:")
        print(f"    Train vocab: {stats['train_vocab_size']}, "
              f"Test vocab: {stats['test_vocab_size']}")
        print(f"    Overlap: {stats['overlap_size']} ({stats['overlap_pct_of_test']:.1f}% of test)")
        print(f"    Test OOV: {stats['test_oov_count']} ({stats['test_oov_pct']:.1f}%)")

    # -----------------------------------------------------------------------
    # Prepare per-experiment condition splits (from train portions only)
    # -----------------------------------------------------------------------
    manifest = {}
    manifest["split_info"] = {
        "structured_train": len(struct_train),
        "structured_val": len(struct_val),
        "structured_test": len(struct_test),
        "random_train": len(rand_train),
        "random_val": len(rand_val),
        "random_test": len(rand_test),
        "test_total": len(test_df),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    manifest["exp1"] = prepare_experiment1(
        rand_train, struct_train, args.output_dir, args.val_ratio, args.seed,
        args.src_col, args.tgt_col
    )
    manifest["exp2"] = prepare_experiment2(
        rand_train, struct_train, args.output_dir, args.val_ratio, args.seed,
        args.src_col, args.tgt_col
    )

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")

    # Data fingerprints
    print("\nData fingerprints:")
    print(f"  Structured combined: {data_fingerprint(structured_df, args.src_col)}")
    print(f"  Random: {data_fingerprint(random_df, args.src_col)}")
    print(f"  Test set: {data_fingerprint(test_df, args.src_col)}")

    print(f"\nDone. All splits saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
