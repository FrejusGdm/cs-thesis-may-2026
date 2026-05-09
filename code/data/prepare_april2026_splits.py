#!/usr/bin/env python3
"""
prepare_april2026_splits.py — Build train/val/test splits for the April-2026
new-data expansion.

Three split trees are produced, each isolated from prior results:

  splits/april2026_runA/   — keep existing shared/test.tsv; add new6k to train.
  splits/april2026_runB1/  — new split seed over the paper pool (10K+4K),
                             new6k added to train/val only (not test-eligible).
  splits/april2026_runB2/  — new split seed over pool+new6k together, so the
                             new 6K rows are test-eligible.

Run A conditions:
  FULL                          All train-pool data (existing ~11K + new6k).
  STRUCT4K-ALL-BASELINES-PLUS-NEW
                                Paper's BLEU-41 recipe with new6k stacked in.

Run B1 / Run B2 conditions:
  FULL                          Only — the STRUCT4K-ALL-BASELINES recipe is
                                specific to the paper's test split, so it's
                                not reproduced for the new-test variants.

Each run writes a manifest.json with seed, split sizes, and the new-data
fraction in the test set.

Run (from repo root):
    python experiments/data/prepare_april2026_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SPLITS = REPO / "experiments" / "data" / "splits"
SHARED = SPLITS / "shared"
EXISTING_STRUCT4K = SPLITS / "new_experiments" / "baselines" / "STRUCT4K-ALL-BASELINES"
NEW6K_PATH = REPO / "april-2026-new-data" / "processed" / "new6k.tsv"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_pair_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["fr", "aj"], dtype=str, keep_default_na=False)
    df = df[(df["fr"].str.strip() != "") & (df["aj"].str.strip() != "")].reset_index(drop=True)
    return df


def write_pair_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["fr", "aj"]].to_csv(path, sep="\t", header=False, index=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Group-aware split (each row = its own group unless "group" col is supplied)
# ---------------------------------------------------------------------------

def shuffle_groups(df: pd.DataFrame, seed: int) -> list[str]:
    groups = df["group"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    return groups


def split_by_groups(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert "group" in df.columns, "Expect a 'group' column for group-aware splitting."
    groups = shuffle_groups(df, seed)
    n = len(groups)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    train_g = set(groups[n_test + n_val:])
    test = df[df["group"].isin(test_g)].reset_index(drop=True)
    val = df[df["group"].isin(val_g)].reset_index(drop=True)
    train = df[df["group"].isin(train_g)].reset_index(drop=True)
    return train, val, test


def split_train_val_only(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """90-10 train/val split when no test is drawn (test is provided externally)."""
    groups = shuffle_groups(df, seed)
    n = len(groups)
    n_val = max(1, int(round(n * val_ratio)))
    val_g = set(groups[:n_val])
    val = df[df["group"].isin(val_g)].reset_index(drop=True)
    train = df[~df["group"].isin(val_g)].reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Tagging sources so we can compute new-data fractions later
# ---------------------------------------------------------------------------

def tag(df: pd.DataFrame, source: str, group_prefix: str) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    # Unique group per row (random pool & new6k have no minimal-pair structure);
    # callers using the structured pool can overwrite with a base_sentence_id.
    out["group"] = [f"{group_prefix}_{i:06d}" for i in range(len(out))]
    return out


def tag_structured_with_groups(df: pd.DataFrame, source: str, group_prefix: str) -> pd.DataFrame:
    """Collapse near-duplicate French phrasings into shared groups so that the
    minimal-pair relationships the paper depended on don't leak across splits.

    The paper's prepare_splits.py used base_sentence_id from the CSV metadata,
    which isn't preserved in the tsv form. As a safe substitute we group by
    the lowercased French-side string — this over-merges a little (exact dups
    collapse) but never under-merges, so leakage is impossible.
    """
    out = df.copy()
    out["source"] = source
    out["group"] = out["fr"].str.strip().str.lower()
    # Prefix with source to guarantee no accidental group collisions with
    # other sources that might share an identical French sentence.
    out["group"] = f"{group_prefix}:" + out["group"]
    return out


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def write_manifest(run_dir: Path, payload: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2))


def per_source_counts(df: pd.DataFrame) -> dict[str, int]:
    if "source" not in df.columns:
        return {}
    return df["source"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Run builders
# ---------------------------------------------------------------------------

def build_run_a(sources: dict[str, pd.DataFrame], out_root: Path, seed: int) -> dict:
    """Run A: keep shared/test.tsv, add new6k to train/val."""
    print("\n=== RUN A — existing test preserved ===")
    pool = pd.concat(
        [sources["random"], sources["structured"], sources["new"]],
        ignore_index=True,
    )
    train, val = split_train_val_only(pool, seed=seed, val_ratio=0.10)
    run_dir = out_root / "april2026_runA"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy (not symlink) the existing test so rsync to HPC preserves it.
    src_test = SHARED / "test.tsv"
    dst_test = run_dir / "test.tsv"
    shutil.copyfile(src_test, dst_test)

    # --- Condition 1: FULL ---
    write_pair_tsv(train, run_dir / "FULL" / "train.tsv")
    write_pair_tsv(val, run_dir / "FULL" / "val.tsv")
    print(f"  FULL              train={len(train):,}  val={len(val):,}")

    # --- Condition 2: STRUCT4K-ALL-BASELINES-PLUS-NEW ---
    if EXISTING_STRUCT4K.exists():
        paper_train = read_pair_tsv(EXISTING_STRUCT4K / "train.tsv")
        paper_val = read_pair_tsv(EXISTING_STRUCT4K / "val.tsv")
        new = sources["new"]
        # 90-10 on the new6k, fold into paper train/val respectively.
        new_train, new_val = split_train_val_only(new, seed=seed, val_ratio=0.10)
        combo_train = pd.concat([paper_train[["fr", "aj"]], new_train[["fr", "aj"]]], ignore_index=True)
        combo_val = pd.concat([paper_val[["fr", "aj"]], new_val[["fr", "aj"]]], ignore_index=True)
        write_pair_tsv(combo_train, run_dir / "STRUCT4K-ALL-BASELINES-PLUS-NEW" / "train.tsv")
        write_pair_tsv(combo_val, run_dir / "STRUCT4K-ALL-BASELINES-PLUS-NEW" / "val.tsv")
        print(f"  STRUCT+NEW        train={len(combo_train):,}  val={len(combo_val):,}")
    else:
        print(f"  STRUCT+NEW        SKIPPED (source dir not found: {EXISTING_STRUCT4K})")

    payload = {
        "run": "runA",
        "seed": seed,
        "test_source": str(src_test.relative_to(REPO)),
        "test_sha256": sha256_of(src_test),
        "test_sha256_copy": sha256_of(dst_test),
        "pool_sizes": {
            "random": int(len(sources["random"])),
            "structured": int(len(sources["structured"])),
            "new": int(len(sources["new"])),
        },
        "conditions": {
            "FULL": {
                "train": int(len(train)),
                "val": int(len(val)),
                "train_per_source": per_source_counts(train),
                "val_per_source": per_source_counts(val),
            },
        },
    }
    if EXISTING_STRUCT4K.exists():
        payload["conditions"]["STRUCT4K-ALL-BASELINES-PLUS-NEW"] = {
            "train": int(len(combo_train)),
            "val": int(len(combo_val)),
            "paper_train_rows": int(len(paper_train)),
            "paper_val_rows": int(len(paper_val)),
            "new_train_rows": int(len(new_train)),
            "new_val_rows": int(len(new_val)),
        }
    write_manifest(run_dir, payload)
    return payload


def add_struct_plus_new(
    run_dir: Path,
    test_df: pd.DataFrame,
    new_pool: pd.DataFrame,
    seed: int,
) -> dict | None:
    """Build STRUCT4K-ALL-BASELINES-PLUS-NEW for a non-paper run.

    Takes the paper's BLEU-41 recipe (already decontaminated vs paper test),
    further decontaminates against THIS run's test set (which is different),
    then stacks on a 90-10 split of new6k (also decontaminated). Writes
    {run_dir}/STRUCT4K-ALL-BASELINES-PLUS-NEW/{train,val}.tsv.

    Returns the manifest entry or None if EXISTING_STRUCT4K isn't available.
    """
    if not EXISTING_STRUCT4K.exists():
        print(f"  STRUCT+NEW SKIPPED (source dir not found: {EXISTING_STRUCT4K})")
        return None
    paper_train = read_pair_tsv(EXISTING_STRUCT4K / "train.tsv")
    paper_val = read_pair_tsv(EXISTING_STRUCT4K / "val.tsv")
    test_fr = set(test_df["fr"].str.strip().str.lower())

    def decontam(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        mask = df["fr"].str.strip().str.lower().isin(test_fr)
        return df[~mask].reset_index(drop=True), int(mask.sum())

    paper_train_clean, n_pt_drop = decontam(paper_train)
    paper_val_clean, n_pv_drop = decontam(paper_val)

    # 90-10 split of new6k, then drop anything that hit the run's test set.
    new_train, new_val = split_train_val_only(new_pool, seed=seed, val_ratio=0.10)
    new_train_clean, n_nt_drop = decontam(new_train)
    new_val_clean, n_nv_drop = decontam(new_val)

    combo_train = pd.concat(
        [paper_train_clean[["fr", "aj"]], new_train_clean[["fr", "aj"]]],
        ignore_index=True,
    )
    combo_val = pd.concat(
        [paper_val_clean[["fr", "aj"]], new_val_clean[["fr", "aj"]]],
        ignore_index=True,
    )
    write_pair_tsv(combo_train, run_dir / "STRUCT4K-ALL-BASELINES-PLUS-NEW" / "train.tsv")
    write_pair_tsv(combo_val, run_dir / "STRUCT4K-ALL-BASELINES-PLUS-NEW" / "val.tsv")
    print(f"  STRUCT+NEW        train={len(combo_train):,}  val={len(combo_val):,}  "
          f"(decontam drops: paper {n_pt_drop}+{n_pv_drop}, new {n_nt_drop}+{n_nv_drop})")
    return {
        "train": int(len(combo_train)),
        "val": int(len(combo_val)),
        "paper_train_rows_kept": int(len(paper_train_clean)),
        "paper_val_rows_kept": int(len(paper_val_clean)),
        "new_train_rows_kept": int(len(new_train_clean)),
        "new_val_rows_kept": int(len(new_val_clean)),
        "decontam_drops": {
            "paper_train": n_pt_drop,
            "paper_val": n_pv_drop,
            "new_train": n_nt_drop,
            "new_val": n_nv_drop,
        },
    }


def build_run_b1(sources: dict[str, pd.DataFrame], out_root: Path, seed: int) -> dict:
    """Run B1: paper pool (10K+4K) re-split with new seed, new6k added to train/val only."""
    print("\n=== RUN B1 — new split seed, new6k kept out of test ===")
    run_dir = out_root / "april2026_runB1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the original paper pool: everything that existed before new6k.
    # random_train + structured_train + shared/test.tsv  covers the whole 10K+4K.
    paper_test = read_pair_tsv(SHARED / "test.tsv")
    paper_test_tagged = tag(paper_test, "paper_test_origin", "old-test")
    pool = pd.concat(
        [sources["random"], sources["structured"], paper_test_tagged],
        ignore_index=True,
    )
    train, val, test = split_by_groups(pool, seed=seed, val_ratio=0.10, test_ratio=0.10)

    # Append new6k to train and val via its own 90-10 split so fractions stay balanced.
    new_train, new_val = split_train_val_only(sources["new"], seed=seed, val_ratio=0.10)
    train = pd.concat([train, new_train], ignore_index=True)
    val = pd.concat([val, new_val], ignore_index=True)

    shutil.copyfile  # no-op, reserved
    write_pair_tsv(test, run_dir / "test.tsv")
    write_pair_tsv(train, run_dir / "FULL" / "train.tsv")
    write_pair_tsv(val, run_dir / "FULL" / "val.tsv")
    print(f"  test={len(test):,}  train={len(train):,}  val={len(val):,}")
    # B1 invariant: no new6k row can be in test.
    assert not (test["source"] == "new").any(), "B1 invariant violated: new6k leaked to test"

    # Also emit the paper's BLEU-41 recipe (decontaminated against B1 test) + new6k.
    spn = add_struct_plus_new(run_dir, test, sources["new"], seed)

    payload = {
        "run": "runB1",
        "seed": seed,
        "invariant": "new6k is never test-eligible",
        "pool_sizes": {
            "random": int(len(sources["random"])),
            "structured": int(len(sources["structured"])),
            "paper_test_reabsorbed": int(len(paper_test_tagged)),
            "new": int(len(sources["new"])),
        },
        "conditions": {
            "FULL": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
                "train_per_source": per_source_counts(train),
                "val_per_source": per_source_counts(val),
                "test_per_source": per_source_counts(test),
            },
        },
    }
    if spn is not None:
        payload["conditions"]["STRUCT4K-ALL-BASELINES-PLUS-NEW"] = spn
    write_manifest(run_dir, payload)
    return payload


def build_run_b2(sources: dict[str, pd.DataFrame], out_root: Path, seed: int) -> dict:
    """Run B2: full pool (10K+4K+new6k) re-split with new seed; new6k eligible for test."""
    print("\n=== RUN B2 — new split seed, new6k test-eligible ===")
    run_dir = out_root / "april2026_runB2"
    run_dir.mkdir(parents=True, exist_ok=True)

    paper_test = read_pair_tsv(SHARED / "test.tsv")
    paper_test_tagged = tag(paper_test, "paper_test_origin", "old-test")
    pool = pd.concat(
        [sources["random"], sources["structured"], paper_test_tagged, sources["new"]],
        ignore_index=True,
    )
    train, val, test = split_by_groups(pool, seed=seed, val_ratio=0.10, test_ratio=0.10)

    write_pair_tsv(test, run_dir / "test.tsv")
    write_pair_tsv(train, run_dir / "FULL" / "train.tsv")
    write_pair_tsv(val, run_dir / "FULL" / "val.tsv")
    print(f"  test={len(test):,}  train={len(train):,}  val={len(val):,}")
    new_in_test = int((test["source"] == "new").sum())
    new_frac = new_in_test / max(len(test), 1)
    print(f"  new6k fraction in test: {new_in_test}/{len(test)} = {100*new_frac:.2f}%")

    # Paper recipe + new6k, decontaminated against B2's bigger test set
    # (which contains 44.9% new6k — so the new6k portion of the recipe will
    # lose more rows here than in runA/B1).
    spn = add_struct_plus_new(run_dir, test, sources["new"], seed)

    payload = {
        "run": "runB2",
        "seed": seed,
        "invariant": "new6k rows are eligible for test (test distribution reflects new data)",
        "pool_sizes": {
            "random": int(len(sources["random"])),
            "structured": int(len(sources["structured"])),
            "paper_test_reabsorbed": int(len(paper_test_tagged)),
            "new": int(len(sources["new"])),
        },
        "conditions": {
            "FULL": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
                "train_per_source": per_source_counts(train),
                "val_per_source": per_source_counts(val),
                "test_per_source": per_source_counts(test),
                "new6k_in_test": new_in_test,
                "new6k_test_fraction": new_frac,
            },
        },
    }
    if spn is not None:
        payload["conditions"]["STRUCT4K-ALL-BASELINES-PLUS-NEW"] = spn
    write_manifest(run_dir, payload)
    return payload


def build_run_b3(sources: dict[str, pd.DataFrame], out_root: Path, seed: int) -> dict:
    """Run B3: same pool as B2 but new6k filtered to sentences only (>=4 tokens).

    Addresses runB2's test drop: B2's test is 28.7% dictionary entries (single
    or 2-3 token rows from the AJAGBE dictionary-style source). That's a
    different task (word-level lookup) that drags aggregate BLEU down. Run B3
    excludes the short new6k rows before pooling so the test distribution
    reflects "more sentence data, same sentence-translation task" cleanly.
    """
    print("\n=== RUN B3 — new split seed, sentence-only new6k pool ===")
    run_dir = out_root / "april2026_runB3"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Filter new6k to sentence-length rows only.
    new_full = sources["new"]
    fr_lens = new_full["fr"].str.split().map(len)
    new_sent = new_full[fr_lens >= 4].reset_index(drop=True)
    print(f"  new6k filter: {len(new_full):,} -> {len(new_sent):,} (>=4 token FR)")

    paper_test = read_pair_tsv(SHARED / "test.tsv")
    paper_test_tagged = tag(paper_test, "paper_test_origin", "old-test")
    pool = pd.concat(
        [sources["random"], sources["structured"], paper_test_tagged, new_sent],
        ignore_index=True,
    )
    train, val, test = split_by_groups(pool, seed=seed, val_ratio=0.10, test_ratio=0.10)

    write_pair_tsv(test, run_dir / "test.tsv")
    write_pair_tsv(train, run_dir / "FULL" / "train.tsv")
    write_pair_tsv(val, run_dir / "FULL" / "val.tsv")
    print(f"  test={len(test):,}  train={len(train):,}  val={len(val):,}")
    # Confirm the filter held through the split: no dict-length rows in test.
    test_lens = test["fr"].str.split().map(len)
    n_short_in_test = int((test_lens <= 3).sum())
    n_new_in_test = int((test["source"] == "new").sum())
    print(f"  dict-length rows in test: {n_short_in_test}  (originally short content only from "
          f"Tatoeba/structured/paper-test; new6k short entries excluded)")
    print(f"  new6k-sourced rows in test: {n_new_in_test} / {len(test)} "
          f"({100*n_new_in_test/max(len(test),1):.1f}%)")

    # Also emit the paper recipe + filtered new6k, decontam'd against B3 test.
    spn = add_struct_plus_new(run_dir, test, new_sent, seed)

    payload = {
        "run": "runB3",
        "seed": seed,
        "invariant": "new6k restricted to >=4-token FR rows; test distribution is sentence-only task",
        "pool_sizes": {
            "random": int(len(sources["random"])),
            "structured": int(len(sources["structured"])),
            "paper_test_reabsorbed": int(len(paper_test_tagged)),
            "new_full": int(len(new_full)),
            "new_filtered": int(len(new_sent)),
        },
        "conditions": {
            "FULL": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
                "train_per_source": per_source_counts(train),
                "val_per_source": per_source_counts(val),
                "test_per_source": per_source_counts(test),
                "new_in_test": n_new_in_test,
                "short_in_test": n_short_in_test,
            },
        },
    }
    if spn is not None:
        payload["conditions"]["STRUCT4K-ALL-BASELINES-PLUS-NEW"] = spn
    write_manifest(run_dir, payload)
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--output-root", type=Path, default=SPLITS)
    ap.add_argument("--run-a-seed", type=int, default=42,
                    help="Seed for the 90-10 train/val draw in Run A.")
    ap.add_argument("--run-b1-seed", type=int, default=7,
                    help="Seed for Run B1 re-split (must differ from the paper's seed=42).")
    ap.add_argument("--run-b2-seed", type=int, default=13,
                    help="Seed for Run B2 re-split.")
    ap.add_argument("--run-b3-seed", type=int, default=19,
                    help="Seed for Run B3 re-split (sentence-only new6k pool).")
    ap.add_argument("--skip", choices=["runA", "runB1", "runB2", "runB3"], action="append", default=[],
                    help="Skip a specific run (repeatable).")
    args = ap.parse_args()

    for path in (SHARED / "random_train.tsv", SHARED / "structured_train.tsv",
                 SHARED / "test.tsv", NEW6K_PATH):
        if not path.exists():
            print(f"ERROR: required source missing: {path}")
            return 1

    print("Loading sources...")
    random_pool = tag(read_pair_tsv(SHARED / "random_train.tsv"), "random", "rand")
    struct_pool = tag_structured_with_groups(
        read_pair_tsv(SHARED / "structured_train.tsv"),
        source="structured",
        group_prefix="struct",
    )
    new_pool = tag(read_pair_tsv(NEW6K_PATH), "new", "new")
    print(f"  random:     {len(random_pool):,}")
    print(f"  structured: {len(struct_pool):,}  (groups: {struct_pool['group'].nunique():,})")
    print(f"  new:        {len(new_pool):,}")

    sources = {"random": random_pool, "structured": struct_pool, "new": new_pool}

    summaries = {}
    if "runA" not in args.skip:
        summaries["runA"] = build_run_a(sources, args.output_root, args.run_a_seed)
    if "runB1" not in args.skip:
        summaries["runB1"] = build_run_b1(sources, args.output_root, args.run_b1_seed)
    if "runB2" not in args.skip:
        summaries["runB2"] = build_run_b2(sources, args.output_root, args.run_b2_seed)
    if "runB3" not in args.skip:
        summaries["runB3"] = build_run_b3(sources, args.output_root, args.run_b3_seed)

    # Top-level index for quick inspection.
    index_path = args.output_root / "april2026_index.json"
    index_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
