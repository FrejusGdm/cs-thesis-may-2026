"""
collect_results.py — Aggregate results across all experiments into summary tables.

Scans the results directory for completed experiments and generates:
  1. A comprehensive JSON summary
  2. A markdown summary table (for README updates)
  3. CSV exports suitable for plotting

Usage:
    python collect_results.py \
        --results-dir ./results \
        --output-dir ./results/summary
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def collect_condition_results(condition_dir):
    """Collect results for a single condition across all seeds."""
    seed_dirs = sorted(glob.glob(os.path.join(condition_dir, "seed*")))
    if not seed_dirs:
        return None

    all_metrics = {}
    configs = []

    for seed_dir in seed_dirs:
        # Load test metrics
        metrics_file = os.path.join(seed_dir, "test_metrics.json")
        full_metrics_file = os.path.join(seed_dir, "full_metrics.json")

        # Prefer full_metrics.json (from evaluate.py) over test_metrics.json (from train)
        mf = full_metrics_file if os.path.exists(full_metrics_file) else metrics_file
        if os.path.exists(mf):
            with open(mf) as f:
                metrics = json.load(f)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    all_metrics.setdefault(key, []).append(value)

        # Load config
        config_file = os.path.join(seed_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                configs.append(json.load(f))

    if not all_metrics:
        return None

    result = {
        "condition": os.path.basename(condition_dir),
        "n_seeds": len(seed_dirs),
        "n_completed": len([d for d in seed_dirs
                           if os.path.exists(os.path.join(d, "test_metrics.json"))
                           or os.path.exists(os.path.join(d, "full_metrics.json"))]),
    }

    # Add config info
    if configs:
        c = configs[0]
        result["model"] = c.get("model", "unknown")
        result["train_file"] = c.get("train_file", "unknown")

    # Aggregate metrics
    for key, values in all_metrics.items():
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))

    return result


def scan_experiment(experiment_dir):
    """Scan all conditions in an experiment directory."""
    results = []
    for condition_dir in sorted(glob.glob(os.path.join(experiment_dir, "*"))):
        if not os.path.isdir(condition_dir):
            continue
        # Skip 'summary' and similar meta-directories
        if os.path.basename(condition_dir) in ("summary", "comparisons"):
            continue

        result = collect_condition_results(condition_dir)
        if result:
            result["experiment"] = os.path.basename(experiment_dir)
            results.append(result)
    return results


def generate_markdown_table(results, title="Results"):
    """Generate a markdown table from results."""
    lines = [
        f"## {title}",
        "",
        "| Condition | Size | BLEU | chrF | chrF++ | TER | COMET | Seeds |",
        "|-----------|------|------|------|--------|-----|-------|-------|",
    ]

    for r in results:
        def fmt(key):
            mean = r.get(f"{key}_mean")
            std = r.get(f"{key}_std")
            if mean is not None:
                if std is not None and std > 0:
                    return f"{mean:.1f} +/- {std:.1f}"
                return f"{mean:.1f}"
            return "—"

        condition = r["condition"]
        n_seeds = r.get("n_completed", r.get("n_seeds", "?"))
        # Try to extract size from test metrics
        size = r.get("n_samples_mean", "—")
        if isinstance(size, float):
            size = int(size)

        lines.append(
            f"| {condition} | {size} | {fmt('bleu')} | {fmt('chrf')} | "
            f"{fmt('chrfpp')} | {fmt('ter')} | {fmt('comet')} | {n_seeds} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_csv(results, output_file):
    """Export results as CSV for plotting."""
    if not results:
        return

    # Collect all keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Collect and aggregate all results")
    parser.add_argument("--results-dir", default="./results", help="Base results directory")
    parser.add_argument("--output-dir", default="./results/summary", help="Summary output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    experiment_results = {}

    # Scan each experiment directory
    for exp_dir in sorted(glob.glob(os.path.join(args.results_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        if exp_name == "summary":
            continue

        results = scan_experiment(exp_dir)
        if results:
            experiment_results[exp_name] = results
            all_results.extend(results)
            print(f"{exp_name}: {len(results)} conditions with results")

    if not all_results:
        print("No results found. Run experiments first.")
        sys.exit(0)

    # Save comprehensive JSON
    json_file = os.path.join(args.output_dir, "all_results.json")
    with open(json_file, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nJSON summary: {json_file}")

    # Generate markdown
    md_lines = ["# Experiment Results Summary\n"]
    for exp_name, results in experiment_results.items():
        md_lines.append(generate_markdown_table(results, title=exp_name))

    md_file = os.path.join(args.output_dir, "results_summary.md")
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown summary: {md_file}")

    # Export CSV
    csv_file = os.path.join(args.output_dir, "all_results.csv")
    generate_csv(all_results, csv_file)
    print(f"CSV export: {csv_file}")

    # Print summary
    print(f"\nTotal: {len(all_results)} conditions across {len(experiment_results)} experiments")


if __name__ == "__main__":
    main()
