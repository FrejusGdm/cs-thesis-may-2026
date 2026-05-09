#!/usr/bin/env python3
"""
check_test_contamination.py — Detect test set overlap with training data.

Checks every condition's train.tsv against the shared test.tsv to find
sentences that appear in both. This is critical for ensuring fair evaluation.

Usage:
    python experiments/analysis/check_test_contamination.py

Output:
    Prints a contamination report for all conditions, grouped by severity.
"""

import glob
import os
import sys
from collections import defaultdict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(REPO_ROOT, "experiments", "data", "splits")
TEST_PATH = os.path.join(SPLITS_DIR, "shared", "test.tsv")
STRUCTURED_TRAIN_PATH = os.path.join(SPLITS_DIR, "shared", "structured_train.tsv")


def load_tsv_no_header(path):
    """Load a TSV file without header, returning src_text and tgt_text columns."""
    df = pd.read_csv(path, sep="\t", header=None, names=["src_text", "tgt_text"])
    return df


def find_all_train_files():
    """Find all train.tsv files under the splits directory."""
    patterns = [
        os.path.join(SPLITS_DIR, "**", "train.tsv"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def get_condition_name(train_path):
    """Extract a human-readable condition name from the train.tsv path."""
    rel = os.path.relpath(train_path, SPLITS_DIR)
    # Remove the trailing /train.tsv
    return os.path.dirname(rel)


def main():
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Test file not found at {TEST_PATH}")
        sys.exit(1)

    # Load test set
    test = load_tsv_no_header(TEST_PATH)
    test_src = set(test["src_text"].str.strip())
    test_tgt = set(test["tgt_text"].str.strip())
    print(f"Test set: {len(test)} sentences ({len(test_src)} unique source)")
    print()

    # Load structured train for attribution
    struct_src = set()
    if os.path.exists(STRUCTURED_TRAIN_PATH):
        struct = load_tsv_no_header(STRUCTURED_TRAIN_PATH)
        struct_src = set(struct["src_text"].str.strip())
        struct_overlap = struct_src & test_src
        print(f"Structured train: {len(struct)} sentences, {len(struct_overlap)} test overlap")
    print()

    # Check all conditions
    train_files = find_all_train_files()
    print(f"Found {len(train_files)} train.tsv files")
    print()

    results = []
    for train_path in train_files:
        condition = get_condition_name(train_path)
        train = load_tsv_no_header(train_path)
        train_src = set(train["src_text"].str.strip())

        overlap_src = train_src & test_src
        overlap_from_struct = overlap_src & struct_src
        overlap_from_other = overlap_src - struct_src

        results.append({
            "condition": condition,
            "train_size": len(train),
            "overlap_count": len(overlap_src),
            "overlap_pct": 100 * len(overlap_src) / len(test_src) if test_src else 0,
            "overlap_from_structured": len(overlap_from_struct),
            "overlap_from_other": len(overlap_from_other),
        })

    # Sort by overlap percentage
    results.sort(key=lambda x: x["overlap_pct"], reverse=True)

    # Print report
    print("=" * 85)
    print("CONTAMINATION REPORT")
    print("=" * 85)
    print()
    print(f"{'Condition':<45} {'Train':>6} {'Overlap':>8} {'%Test':>7} {'Source':>10}")
    print("-" * 85)

    contaminated = []
    clean = []
    for r in results:
        source = "baseline" if r["overlap_from_other"] > 0 else ("struct" if r["overlap_from_structured"] > 0 else "—")
        marker = " ⚠️" if r["overlap_pct"] > 0 else ""
        print(f"{r['condition']:<45} {r['train_size']:>6} {r['overlap_count']:>8} {r['overlap_pct']:>6.1f}% {source:>10}{marker}")
        if r["overlap_count"] > 0:
            contaminated.append(r)
        else:
            clean.append(r)

    print()
    print("=" * 85)
    print(f"CLEAN conditions: {len(clean)}")
    print(f"CONTAMINATED conditions: {len(contaminated)}")
    if contaminated:
        print()
        print("⚠️  Contaminated conditions have test sentences in their training data.")
        print("   This inflates evaluation metrics. Options:")
        print("   1. Re-split: remove test sentences from train, retrain")
        print("   2. Re-evaluate: score only on non-overlapping test subset")
        print("   3. Disclose: note the overlap in the paper")
    print("=" * 85)


if __name__ == "__main__":
    main()
