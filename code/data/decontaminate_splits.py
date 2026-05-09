#!/usr/bin/env python3
"""
decontaminate_splits.py — Remove test-set sentences from ALL contaminated splits.

Two contamination sources:
  1. Standalone baselines (TF-IDF, Length, Vocab) — selected from full Tatoeba pool
     that also sourced the test set (~13% overlap)
  2. Ablation conditions (26 conditions) — built from larger structured pool that
     overlaps with test set (3.8–29.3% overlap)

Fix: Remove any sentence whose source text appears in test.tsv from both
train.tsv and val.tsv in all affected conditions.

Usage:
    python experiments/data/decontaminate_splits.py [--dry-run] [--in-place] [--ablations-only]

    --dry-run:        Print what would be removed without writing files
    --in-place:       Overwrite original files (default: write to _clean/ suffix dirs)
    --ablations-only: Only process ablation conditions (skip baselines/combos)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "experiments" / "data" / "splits"
TEST_PATH = SPLITS_DIR / "shared" / "test.tsv"

# Standalone baselines + combo conditions
BASELINE_DIRS = {
    "baselines/TF-IDF-DIVERSE": SPLITS_DIR / "baselines" / "TF-IDF-DIVERSE",
    "baselines/LENGTH-STRATIFIED": SPLITS_DIR / "baselines" / "LENGTH-STRATIFIED",
    "baselines/VOCAB-MAXIMIZED": SPLITS_DIR / "baselines" / "VOCAB-MAXIMIZED",
    "combos/STRUCT4K-TFIDF2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-TFIDF2K",
    "combos/STRUCT4K-LENGTH2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-LENGTH2K",
    "combos/STRUCT4K-VOCAB2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-VOCAB2K",
    "combos/STRUCT4K-ALL-BASELINES": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-ALL-BASELINES",
}

# All 26 ablation conditions (Tables 10-13)
ABLATION_GROUPS = {
    "module_loo": ["FULL", "NO-PAST", "NO-NEGATION", "NO-QUESTIONS", "NO-FUTURE", "BASE-ONLY"],
    "module_size_ctrl": ["FULL-1K", "NO-PAST-1K", "NO-QUEST-1K", "NO-NEG-1K", "NO-FUT-1K", "BASE-1K"],
    "minimal_pairs": ["PAIRS-INTACT", "PAIRS-BROKEN"],
    "verb": ["1-VERB", "3-VERBS-a", "3-VERBS-b", "3-VERBS-c", "5-VERBS-a", "5-VERBS-b", "5-VERBS-c", "10-VERBS"],
    "pronoun": ["ALL-8", "MINIMAL-1", "REDUCED-4", "SINGULAR-3"],
}

ABLATION_DIRS = {}
for group, conditions in ABLATION_GROUPS.items():
    for cond in conditions:
        label = f"ablations/{group}/{cond}"
        ABLATION_DIRS[label] = SPLITS_DIR / "ablations" / group / cond


def load_test_sources():
    """Load the set of source-side sentences in the test set."""
    test = pd.read_csv(TEST_PATH, sep="\t", header=None, names=["src_text", "tgt_text"])
    return set(test["src_text"].str.strip())


def decontaminate_file(path, test_src, dry_run=False):
    """Remove rows whose source text is in the test set. Returns (before, after, removed_count)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["src_text", "tgt_text"])
    before = len(df)

    mask = df["src_text"].str.strip().isin(test_src)
    removed = mask.sum()
    clean = df[~mask]

    if not dry_run:
        clean.to_csv(path, sep="\t", header=False, index=False)

    return before, len(clean), removed


def process_dirs(dirs, test_src, args):
    """Process a dict of {label: path} directories, removing test sentences."""
    total_removed = 0
    for label, src_dir in dirs.items():
        if not src_dir.exists():
            print(f"{label:<40} SKIP (directory not found)")
            continue

        if args.in_place or args.dry_run:
            work_dir = src_dir
        else:
            work_dir = src_dir.parent / (src_dir.name + "_clean")
            if not args.dry_run:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                shutil.copytree(src_dir, work_dir)

        for fname in ["train.tsv", "val.tsv"]:
            fpath = work_dir / fname
            if not fpath.exists():
                continue
            before, after, removed = decontaminate_file(fpath, test_src, dry_run=args.dry_run)
            total_removed += removed
            marker = " ***" if removed > 0 else ""
            print(f"{label:<40} {fname:<8} {before:>7} {removed:>8} {after:>7}{marker}")
    return total_removed


def main():
    parser = argparse.ArgumentParser(description="Remove test sentences from all contaminated splits")
    parser.add_argument("--dry-run", action="store_true", help="Print report without modifying files")
    parser.add_argument("--in-place", action="store_true",
                        help="Modify original files (default: copy to *_clean/ dirs first)")
    parser.add_argument("--ablations-only", action="store_true",
                        help="Only process ablation conditions (skip baselines/combos)")
    args = parser.parse_args()

    if not TEST_PATH.exists():
        print(f"ERROR: Test file not found at {TEST_PATH}")
        sys.exit(1)

    test_src = load_test_sources()
    print(f"Test set: {len(test_src)} unique source sentences")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    header = f"{'Condition':<40} {'File':<8} {'Before':>7} {'Removed':>8} {'After':>7}"
    print(header)
    print("-" * len(header))

    total_removed = 0

    if not args.ablations_only:
        print("\n=== BASELINES & COMBOS ===")
        total_removed += process_dirs(BASELINE_DIRS, test_src, args)

    print("\n=== ABLATION CONDITIONS (26 conditions) ===")
    total_removed += process_dirs(ABLATION_DIRS, test_src, args)

    print()
    print(f"Total sentences removed: {total_removed}")
    if not args.dry_run and not args.in_place:
        print(f"\nCleaned files written to *_clean/ directories alongside originals.")
    elif args.dry_run:
        print("\nDry run complete. No files were modified.")
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
