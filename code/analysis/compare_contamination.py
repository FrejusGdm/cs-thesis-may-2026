#!/usr/bin/env python3
"""
compare_contamination.py — Compare contaminated vs decontaminated results.

Reads the pre-decontamination archive and current (clean) results,
then produces a side-by-side comparison table showing the impact
of removing test-set overlap from training data.

Usage:
    python experiments/analysis/compare_contamination.py

Output:
    - Prints comparison table to stdout
    - Writes experiments/results/summary/contamination_comparison.csv
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_CSV = os.path.join(REPO_ROOT, "experiments", "results", "summary", "all_results_pre_decontamination.csv")
CURRENT_CSV = os.path.join(REPO_ROOT, "experiments", "results", "summary", "all_results.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "experiments", "results", "summary", "contamination_comparison.csv")


def load_and_aggregate(csv_path):
    """Load results CSV and compute mean/std per experiment/condition."""
    df = pd.read_csv(csv_path)
    grouped = df.groupby(["experiment", "condition"]).agg(
        n_seeds=("seed", "count"),
        bleu_mean=("test_bleu", "mean"),
        bleu_std=("test_bleu", "std"),
        chrf_mean=("test_chrf", "mean"),
        chrf_std=("test_chrf", "std"),
        chrfpp_mean=("test_chrfpp", "mean"),
        chrfpp_std=("test_chrfpp", "std"),
    ).reset_index()
    return grouped


def main():
    if not os.path.exists(ARCHIVE_CSV):
        print(f"ERROR: Archive not found at {ARCHIVE_CSV}")
        print("Run this after collecting clean results.")
        sys.exit(1)

    if not os.path.exists(CURRENT_CSV):
        print(f"ERROR: Current results not found at {CURRENT_CSV}")
        print("Run collect_hf_results.py first.")
        sys.exit(1)

    old = load_and_aggregate(ARCHIVE_CSV)
    new = load_and_aggregate(CURRENT_CSV)

    # Merge on experiment + condition
    merged = old.merge(
        new,
        on=["experiment", "condition"],
        suffixes=("_contaminated", "_clean"),
        how="outer",
        indicator=True,
    )

    # Build comparison rows
    rows = []
    for _, r in merged.iterrows():
        exp = r["experiment"]
        cond = r["condition"]

        has_old = r["_merge"] in ["both", "left_only"]
        has_new = r["_merge"] in ["both", "right_only"]

        row = {
            "experiment": exp,
            "condition": cond,
        }

        if has_old:
            row["contaminated_bleu"] = f"{r['bleu_mean_contaminated']:.1f}"
            row["contaminated_chrfpp"] = f"{r['chrfpp_mean_contaminated']:.1f}"
            row["contaminated_seeds"] = int(r["n_seeds_contaminated"])
        else:
            row["contaminated_bleu"] = "—"
            row["contaminated_chrfpp"] = "—"
            row["contaminated_seeds"] = 0

        if has_new:
            row["clean_bleu"] = f"{r['bleu_mean_clean']:.1f}"
            row["clean_chrfpp"] = f"{r['chrfpp_mean_clean']:.1f}"
            row["clean_seeds"] = int(r["n_seeds_clean"])
        else:
            row["clean_bleu"] = "—"
            row["clean_chrfpp"] = "—"
            row["clean_seeds"] = 0

        if has_old and has_new:
            delta = r["bleu_mean_clean"] - r["bleu_mean_contaminated"]
            row["delta_bleu"] = f"{delta:+.1f}"
        else:
            row["delta_bleu"] = "—"

        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(["experiment", "condition"])

    # Save full comparison
    result.to_csv(OUTPUT_CSV, index=False)

    # Print affected conditions only
    print("=" * 95)
    print("CONTAMINATION IMPACT: Contaminated vs Clean Results")
    print("=" * 95)
    print()
    print(f"{'Experiment/Condition':<45} {'Contam BLEU':>12} {'Clean BLEU':>11} {'Delta':>7} {'Seeds':>6}")
    print("-" * 95)

    for _, r in result.iterrows():
        label = f"{r['experiment']}/{r['condition']}"
        # Only show rows where both exist and they differ
        if r["delta_bleu"] != "—":
            delta_val = float(r["delta_bleu"])
            marker = " ***" if abs(delta_val) > 1.0 else ""
            print(f"{label:<45} {r['contaminated_bleu']:>12} {r['clean_bleu']:>11} {r['delta_bleu']:>7}{marker}")
        elif r["clean_bleu"] == "—":
            print(f"{label:<45} {r['contaminated_bleu']:>12} {'PENDING':>11} {'':>7}")

    print()
    print(f"Saved to: {OUTPUT_CSV}")
    print("*** = delta > 1.0 BLEU (likely contamination effect)")


if __name__ == "__main__":
    main()
