"""
run_experiment.py — Run a single experiment condition across all seeds.

Wraps train_nllb.py and runs it for each seed, then aggregates results.

Usage:
    python run_experiment.py \
        --experiment exp1 \
        --condition RANDOM-10K \
        --splits-dir ./splits \
        --results-dir ./results \
        --test-file ./splits/shared/test.tsv \
        --seeds 42 123 456 789 1024

    # Or run a single seed:
    python run_experiment.py \
        --experiment exp1 \
        --condition RANDOM-10K \
        --splits-dir ./splits \
        --results-dir ./results \
        --seeds 42
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_single_seed(train_file, val_file, test_file, output_dir, model, seed, **kwargs):
    """Run training for a single seed."""
    cmd = [
        sys.executable, "train_nllb.py",
        "--train-file", train_file,
        "--val-file", val_file,
        "--output-dir", output_dir,
        "--model", model,
        "--seed", str(seed),
    ]
    if test_file:
        cmd.extend(["--test-file", test_file])

    # Forward optional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def aggregate_results(results_dir, seeds, condition_name):
    """Aggregate test metrics across all seeds."""
    all_metrics = {}

    for seed in seeds:
        metrics_file = os.path.join(results_dir, f"seed{seed}", "test_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics.setdefault(key, []).append(value)

    if not all_metrics:
        print("WARNING: No test metrics found for any seed.")
        return {}

    # Compute mean and std
    summary = {"condition": condition_name, "n_seeds": len(seeds)}
    for key, values in all_metrics.items():
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_values"] = values

    # Save summary
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {condition_name}")
    print(f"{'='*60}")
    for key in ["test_bleu", "test_chrf", "test_chrfpp", "test_ter"]:
        if f"{key}_mean" in summary:
            print(f"  {key}: {summary[f'{key}_mean']:.2f} +/- {summary[f'{key}_std']:.2f}")
    print(f"  Seeds completed: {len(all_metrics.get('test_bleu', []))}/{len(seeds)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run experiment across all seeds")
    parser.add_argument("--experiment", required=True, help="Experiment name (exp1, exp2, ...)")
    parser.add_argument("--condition", required=True, help="Condition name (RANDOM-10K, ...)")
    parser.add_argument("--splits-dir", default="./splits", help="Base splits directory")
    parser.add_argument("--results-dir", default="./results", help="Base results directory")
    parser.add_argument("--test-file", help="Shared test file path")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip seeds that already have results")
    args = parser.parse_args()

    # Locate data files
    condition_dir = os.path.join(args.splits_dir, args.experiment, args.condition)
    train_file = os.path.join(condition_dir, "train.tsv")
    val_file = os.path.join(condition_dir, "val.tsv")

    if not os.path.exists(train_file):
        print(f"ERROR: Train file not found: {train_file}")
        print(f"Run prepare_splits.py first to create data splits.")
        sys.exit(1)

    # Results directory for this condition
    condition_results_dir = os.path.join(args.results_dir, args.experiment, args.condition)

    # Run each seed
    succeeded = []
    failed = []
    for seed in args.seeds:
        seed_output = os.path.join(condition_results_dir, f"seed{seed}")

        # Skip if results already exist
        if args.skip_existing and os.path.exists(os.path.join(seed_output, "test_metrics.json")):
            print(f"Skipping seed {seed} (results exist)")
            succeeded.append(seed)
            continue

        optional_kwargs = {}
        if args.lr is not None:
            optional_kwargs["lr"] = args.lr
        if args.batch_size is not None:
            optional_kwargs["batch_size"] = args.batch_size
        if args.max_epochs is not None:
            optional_kwargs["max_epochs"] = args.max_epochs

        ok = run_single_seed(
            train_file, val_file, args.test_file, seed_output,
            args.model, seed, **optional_kwargs
        )
        if ok:
            succeeded.append(seed)
        else:
            failed.append(seed)

    print(f"\nCompleted: {len(succeeded)}/{len(args.seeds)} seeds")
    if failed:
        print(f"Failed seeds: {failed}")

    # Aggregate results
    if succeeded:
        aggregate_results(condition_results_dir, args.seeds, args.condition)


if __name__ == "__main__":
    main()
