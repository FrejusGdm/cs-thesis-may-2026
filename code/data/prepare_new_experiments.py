#!/usr/bin/env python3
"""
prepare_new_experiments.py

Creates train/val TSV splits for 7 new experimental conditions and uploads
them to HF Hub as new subdirectories in JosueG/adja-nmt-splits.

Conditions created:
  Additive ablation (experiment=ablations/additive):
    ADD-M1M2        — modules M1+M2 (present + negation)
    ADD-M1M2M3      — modules M1+M2+M3 (+ past)
    ADD-M1M2M3M4    — modules M1+M2+M3+M4 (+ future)

  Structured + smart-selection combos (experiment=baselines):
    STRUCT4K-TFIDF2K       — structured_train + TF-IDF-DIVERSE
    STRUCT4K-LENGTH2K      — structured_train + LENGTH-STRATIFIED
    STRUCT4K-VOCAB2K       — structured_train + VOCAB-MAXIMIZED
    STRUCT4K-ALL-BASELINES — structured_train + deduped union of all 3

Safety: upload_folder() only adds files; never deletes existing HF Hub content.

Usage:
    python experiments/data/prepare_new_experiments.py [--dry-run] [--no-upload]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from huggingface_hub import HfApi, upload_folder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = REPO_ROOT / "experiments" / "data" / "splits" / "shared"
BASELINES_DIR = REPO_ROOT / "experiments" / "data" / "splits" / "baselines"
OUTPUT_DIR = REPO_ROOT / "experiments" / "data" / "splits" / "new_experiments"

STRUCTURED_CSV = SHARED_DIR / "structured_train.csv"
STRUCTURED_TSV = SHARED_DIR / "structured_train.tsv"

BASELINE_TFIDF = BASELINES_DIR / "TF-IDF-DIVERSE" / "train.tsv"
BASELINE_LENGTH = BASELINES_DIR / "LENGTH-STRATIFIED" / "train.tsv"
BASELINE_VOCAB = BASELINES_DIR / "VOCAB-MAXIMIZED" / "train.tsv"

HF_REPO = "JosueG/adja-nmt-splits"
RANDOM_STATE = 42
VAL_FRAC = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tsv(path: Path) -> pd.DataFrame:
    """Read a two-column TSV (french, adja) with no header."""
    df = pd.read_csv(path, sep="\t", header=None, names=["french", "adja"])
    return df


def write_tsv(df: pd.DataFrame, path: Path):
    """Write two-column TSV (french, adja) with no header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["french", "adja"]].to_csv(path, sep="\t", header=False, index=False)
    print(f"  Wrote {len(df):>5} rows → {path.relative_to(REPO_ROOT)}")


def group_aware_split(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val keeping all rows with the same group_col value
    in the same split (prevents minimal-pair leakage).
    Returns (train_df, val_df).
    """
    groups = df[group_col].values
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def simple_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple random 90/10 split."""
    train_df, val_df = train_test_split(df, test_size=VAL_FRAC, random_state=RANDOM_STATE, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_condition(experiment: str, condition: str, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Write train.tsv and val.tsv to OUTPUT_DIR/{experiment}/{condition}/."""
    out_dir = OUTPUT_DIR / experiment / condition
    write_tsv(train_df, out_dir / "train.tsv")
    write_tsv(val_df, out_dir / "val.tsv")


def upload_condition(experiment: str, condition: str, dry_run: bool):
    """Upload OUTPUT_DIR/{experiment}/{condition}/ to HF Hub."""
    local_path = OUTPUT_DIR / experiment / condition
    path_in_repo = f"{experiment}/{condition}"
    if dry_run:
        print(f"  [DRY RUN] Would upload {local_path} → {HF_REPO}/{path_in_repo}")
        return
    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    print(f"  Uploading → {HF_REPO}/{path_in_repo} ...")
    upload_folder(
        repo_id=HF_REPO,
        folder_path=str(local_path),
        path_in_repo=path_in_repo,
        repo_type="dataset",
        token=token,
        commit_message=f"Add {condition} splits for {experiment}",
    )
    print(f"  Done.")


# ---------------------------------------------------------------------------
# Additive ablation conditions
# ---------------------------------------------------------------------------

def make_additive_conditions(dry_run: bool):
    print("\n" + "=" * 60)
    print("ADDITIVE ABLATION CONDITIONS")
    print("=" * 60)

    df_csv = pd.read_csv(STRUCTURED_CSV)
    # Rename columns to standard names
    df_csv = df_csv.rename(columns={"french": "french", "adja_translation": "adja"})
    print(f"Loaded structured_train.csv: {len(df_csv)} rows")
    print(f"Module distribution:\n{df_csv['module'].value_counts().sort_index()}\n")

    conditions = {
        "ADD-M1M2":     ["M1", "M2"],
        "ADD-M1M2M3":   ["M1", "M2", "M3"],
        "ADD-M1M2M3M4": ["M1", "M2", "M3", "M4"],
    }

    for cond_name, modules in conditions.items():
        print(f"\n{cond_name} ({'+'.join(modules)}):")
        subset = df_csv[df_csv["module"].isin(modules)][["french", "adja", "base_sentence_id"]].copy()
        print(f"  Total rows: {len(subset)}")
        train_df, val_df = group_aware_split(subset, group_col="base_sentence_id")
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        save_condition("ablations/additive", cond_name, train_df, val_df)
        upload_condition("ablations/additive", cond_name, dry_run)


# ---------------------------------------------------------------------------
# Structured + smart-selection combo conditions
# ---------------------------------------------------------------------------

def make_combo_conditions(dry_run: bool):
    print("\n" + "=" * 60)
    print("STRUCTURED + SMART-SELECTION COMBO CONDITIONS")
    print("=" * 60)

    struct_df = read_tsv(STRUCTURED_TSV)
    print(f"Loaded structured_train.tsv: {len(struct_df)} rows")

    tfidf_df = read_tsv(BASELINE_TFIDF)
    length_df = read_tsv(BASELINE_LENGTH)
    vocab_df = read_tsv(BASELINE_VOCAB)
    print(f"Loaded TF-IDF baseline: {len(tfidf_df)} rows")
    print(f"Loaded LENGTH baseline: {len(length_df)} rows")
    print(f"Loaded VOCAB baseline: {len(vocab_df)} rows")

    combos = {
        "STRUCT4K-TFIDF2K":       tfidf_df,
        "STRUCT4K-LENGTH2K":      length_df,
        "STRUCT4K-VOCAB2K":       vocab_df,
    }

    for cond_name, baseline_df in combos.items():
        print(f"\n{cond_name}:")
        combined = pd.concat([struct_df, baseline_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["french"])
        print(f"  Combined (after dedup): {len(combined)} rows")
        train_df, val_df = simple_split(combined)
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        save_condition("baselines", cond_name, train_df, val_df)
        upload_condition("baselines", cond_name, dry_run)

    # ALL-BASELINES: structured + deduped union of all 3
    print(f"\nSTRUCT4K-ALL-BASELINES:")
    all_baselines = pd.concat([tfidf_df, length_df, vocab_df], ignore_index=True)
    all_baselines = all_baselines.drop_duplicates(subset=["french"])
    print(f"  Deduped union of 3 baselines: {len(all_baselines)} rows")
    combined = pd.concat([struct_df, all_baselines], ignore_index=True)
    combined = combined.drop_duplicates(subset=["french"])
    print(f"  Combined with structured (after dedup): {len(combined)} rows")
    train_df, val_df = simple_split(combined)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
    save_condition("baselines", "STRUCT4K-ALL-BASELINES", train_df, val_df)
    upload_condition("baselines", "STRUCT4K-ALL-BASELINES", dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create and upload new experiment splits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create local files but skip HF Hub upload")
    parser.add_argument("--no-upload", action="store_true",
                        help="Same as --dry-run")
    args = parser.parse_args()

    dry_run = args.dry_run or args.no_upload

    if not dry_run:
        from huggingface_hub import get_token
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            print("ERROR: Not logged in. Run: hf auth login")
            print("Or use --dry-run to create local files only.")
            sys.exit(1)

    print(f"Output directory: {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"HF Hub repo: {HF_REPO}")
    if dry_run:
        print("MODE: DRY RUN (local files only, no upload)")
    else:
        print("MODE: UPLOAD TO HF HUB")

    make_additive_conditions(dry_run)
    make_combo_conditions(dry_run)

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)
    if not dry_run:
        print("\nNext steps:")
        print("  bash experiments/training/launch_new_experiments.sh additive  # 9 jobs")
        print("  bash experiments/training/launch_new_experiments.sh table9    # 20 jobs")
    else:
        print(f"\nLocal splits written to: {OUTPUT_DIR.relative_to(REPO_ROOT)}")
        print("Run without --dry-run to upload to HF Hub.")


if __name__ == "__main__":
    main()
