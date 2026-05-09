#!/usr/bin/env python3
"""
upload_decontaminated_ablations.py — Upload decontaminated ablation splits to HF Hub.

After running decontaminate_splits.py --in-place, the ablation splits in
experiments/data/splits/ablations/ are clean. This script uploads them
to JosueG/adja-nmt-splits, overwriting the contaminated versions.

Usage:
    python experiments/data/upload_decontaminated_ablations.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import upload_folder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "experiments" / "data" / "splits"
HF_REPO = "JosueG/adja-nmt-splits"

# All ablation groups and their conditions
ABLATION_GROUPS = {
    "module_loo": ["FULL", "NO-PAST", "NO-NEGATION", "NO-QUESTIONS", "NO-FUTURE", "BASE-ONLY"],
    "module_size_ctrl": ["FULL-1K", "NO-PAST-1K", "NO-QUEST-1K", "NO-NEG-1K", "NO-FUT-1K", "BASE-1K"],
    "minimal_pairs": ["PAIRS-INTACT", "PAIRS-BROKEN"],
    "verb": ["1-VERB", "3-VERBS-a", "3-VERBS-b", "3-VERBS-c", "5-VERBS-a", "5-VERBS-b", "5-VERBS-c", "10-VERBS"],
    "pronoun": ["ALL-8", "MINIMAL-1", "REDUCED-4", "SINGULAR-3"],
}

# Also re-upload standalone baselines (already decontaminated earlier)
BASELINE_CONDITIONS = ["TF-IDF-DIVERSE", "LENGTH-STRATIFIED", "VOCAB-MAXIMIZED"]


def upload_condition(local_dir: Path, path_in_repo: str, dry_run: bool):
    """Upload a single condition directory to HF Hub."""
    if not local_dir.exists():
        print(f"  SKIP {path_in_repo} (not found)")
        return False

    train_path = local_dir / "train.tsv"
    if not train_path.exists():
        print(f"  SKIP {path_in_repo} (no train.tsv)")
        return False

    import pandas as pd
    df = pd.read_csv(train_path, sep="\t", header=None)
    n_rows = len(df)

    if dry_run:
        print(f"  [DRY RUN] {path_in_repo}: {n_rows} train rows")
        return True

    print(f"  Uploading {path_in_repo} ({n_rows} train rows)...")
    upload_folder(
        repo_id=HF_REPO,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_type="dataset",
        commit_message=f"Decontaminate {path_in_repo}: remove test-set overlap",
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload decontaminated ablation splits to HF Hub")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded")
    args = parser.parse_args()

    if not args.dry_run:
        from huggingface_hub import get_token
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            print("ERROR: Not logged in. Run: huggingface-cli login")
            sys.exit(1)

    uploaded = 0
    total = 0

    # Upload standalone baselines
    print("\n=== STANDALONE BASELINES ===")
    for cond in BASELINE_CONDITIONS:
        total += 1
        local_dir = SPLITS_DIR / "baselines" / cond
        if upload_condition(local_dir, f"baselines/{cond}", args.dry_run):
            uploaded += 1

    # Upload ablation conditions
    for group, conditions in ABLATION_GROUPS.items():
        print(f"\n=== ABLATION: {group} ===")
        for cond in conditions:
            total += 1
            local_dir = SPLITS_DIR / "ablations" / group / cond
            path_in_repo = f"ablations/{group}/{cond}"
            if upload_condition(local_dir, path_in_repo, args.dry_run):
                uploaded += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Uploaded {uploaded}/{total} conditions to {HF_REPO}")

    if not args.dry_run:
        print("\nNext: launch ablation rerun jobs:")
        print("  bash experiments/training/launch_new_experiments.sh ablations-rerun")


if __name__ == "__main__":
    main()
