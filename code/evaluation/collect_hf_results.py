"""
collect_hf_results.py — Download and aggregate results from HuggingFace Hub.

Scans the results repo for test_metrics.json files, downloads them all,
and assembles a single CSV summary for analysis + significance testing.

Usage:
    python collect_hf_results.py \
        --results-repo username/adja-nmt-results \
        --output-dir ./results/summary

    # Then run significance tests:
    python significance.py \
        --all-dirs ./results/exp1/RANDOM-10K/ ./results/exp1/STRUCTURED-4K-ONLY/ \
        --output ./results/exp1/pairwise_significance.json
"""

import argparse
import csv
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree


def list_metrics_files(repo_id, token=None):
    """List all test_metrics.json files in the results repo."""
    api = HfApi(token=token)
    metrics_files = []

    for entry in list_repo_tree(repo_id, repo_type="dataset", token=token, recursive=True):
        if hasattr(entry, "rfilename"):
            path = entry.rfilename
        elif hasattr(entry, "path"):
            path = entry.path
        else:
            continue
        if path.endswith("test_metrics.json"):
            metrics_files.append(path)

    return sorted(metrics_files)


def parse_result_path(path):
    """Parse experiment/condition/seed from the file path.

    Expected patterns:
      exp1/RANDOM-10K/seed42/test_metrics.json
      ablations/module_loo/FULL/seed42/test_metrics.json
    """
    parts = path.replace("test_metrics.json", "").strip("/").split("/")

    # Last part (before filename) is seed
    seed_str = parts[-1] if parts else "unknown"
    seed = seed_str.replace("seed", "") if seed_str.startswith("seed") else seed_str

    # Condition is second-to-last
    condition = parts[-2] if len(parts) >= 2 else "unknown"

    # Experiment is everything before condition
    experiment = "/".join(parts[:-2]) if len(parts) >= 3 else "unknown"

    return experiment, condition, seed


def download_all_results(repo_id, output_dir, token=None):
    """Download all test_metrics.json files and return parsed results."""
    metrics_files = list_metrics_files(repo_id, token)
    print(f"Found {len(metrics_files)} result files in {repo_id}")

    results = []
    for path in metrics_files:
        experiment, condition, seed = parse_result_path(path)

        # Download
        local_path = hf_hub_download(
            repo_id=repo_id, filename=path,
            repo_type="dataset", token=token,
            local_dir=os.path.join(output_dir, "raw"),
        )

        with open(local_path) as f:
            metrics = json.load(f)

        # Build row
        row = {
            "experiment": metrics.get("experiment", experiment),
            "condition": metrics.get("condition", condition),
            "seed": metrics.get("seed", seed),
            "model": metrics.get("model", "unknown"),
            "train_size": metrics.get("train_size", ""),
            "test_bleu": metrics.get("test_bleu"),
            "test_chrf": metrics.get("test_chrf"),
            "test_chrfpp": metrics.get("test_chrfpp"),
            "test_ter": metrics.get("test_ter"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "actual_epochs": metrics.get("actual_epochs"),
            "test_n_samples": metrics.get("test_n_samples"),
        }
        results.append(row)
        print(f"  {experiment}/{condition}/seed{seed}: "
              f"BLEU={row['test_bleu']:.1f}, chrF={row['test_chrf']:.1f}")

    return results


def save_csv(results, output_file):
    """Save results as CSV."""
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "experiment", "condition", "seed", "model", "train_size",
        "test_bleu", "test_chrf", "test_chrfpp", "test_ter",
        "training_time_seconds", "actual_epochs", "test_n_samples",
    ]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {output_file}")


def save_local_structure(results, output_dir):
    """Save results in the local directory structure expected by significance.py.

    Creates: {output_dir}/{experiment}/{condition}/seed{seed}/test_metrics.json
    """
    for row in results:
        exp = row["experiment"]
        cond = row["condition"]
        seed = row["seed"]
        local_dir = os.path.join(output_dir, exp, cond, f"seed{seed}")
        os.makedirs(local_dir, exist_ok=True)

        metrics = {k: v for k, v in row.items() if v is not None}
        with open(os.path.join(local_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


def print_summary(results):
    """Print a summary table grouped by experiment."""
    from collections import defaultdict

    by_exp = defaultdict(list)
    for r in results:
        by_exp[r["experiment"]].append(r)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for exp, rows in sorted(by_exp.items()):
        print(f"\n--- {exp} ---")
        # Group by condition
        by_cond = defaultdict(list)
        for r in rows:
            by_cond[r["condition"]].append(r)

        print(f"{'Condition':<35} {'Seeds':>5} {'BLEU':>8} {'chrF':>8} {'chrF++':>8}")
        print("-" * 70)
        for cond, cond_rows in sorted(by_cond.items()):
            n = len(cond_rows)
            bleu_vals = [r["test_bleu"] for r in cond_rows if r["test_bleu"] is not None]
            chrf_vals = [r["test_chrf"] for r in cond_rows if r["test_chrf"] is not None]
            chrfpp_vals = [r["test_chrfpp"] for r in cond_rows if r["test_chrfpp"] is not None]

            def fmt(vals):
                if not vals:
                    return "—"
                import numpy as np
                mean = np.mean(vals)
                if len(vals) > 1:
                    return f"{mean:.1f}+/-{np.std(vals):.1f}"
                return f"{mean:.1f}"

            print(f"{cond:<35} {n:>5} {fmt(bleu_vals):>8} {fmt(chrf_vals):>8} {fmt(chrfpp_vals):>8}")

    # Coverage check
    total = len(results)
    missing_metrics = sum(1 for r in results if r["test_bleu"] is None)
    print(f"\nTotal: {total} results collected")
    if missing_metrics:
        print(f"WARNING: {missing_metrics} results missing BLEU scores")


def main():
    parser = argparse.ArgumentParser(description="Collect results from HF Hub")
    parser.add_argument("--results-repo", required=True, help="HF dataset repo with results")
    parser.add_argument("--output-dir", default="./results", help="Local output directory")
    parser.add_argument("--token", help="HF token (defaults to HF_TOKEN env or cached login)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    results = download_all_results(args.results_repo, args.output_dir, token)

    if not results:
        print("No results found. Have experiments finished running?")
        sys.exit(0)

    # Save CSV summary
    csv_file = os.path.join(args.output_dir, "summary", "all_results.csv")
    save_csv(results, csv_file)

    # Save local directory structure (for significance.py compatibility)
    save_local_structure(results, args.output_dir)
    print(f"Local structure saved to {args.output_dir}/ (compatible with significance.py)")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
