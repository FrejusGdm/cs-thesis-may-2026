"""
significance.py — Statistical significance testing for MT experiments.

Implements:
  - Paired bootstrap resampling (Koehn 2004)
  - Wilcoxon signed-rank test across seeds
  - Bonferroni correction for multiple comparisons
  - Cohen's d effect sizes
  - 95% confidence intervals

Usage:
    # Compare two systems' predictions:
    python significance.py \
        --system-a ./results/exp1/RANDOM-10K/seed42/predictions.tsv \
        --system-b ./results/exp1/STRUCTURED-4K-ONLY/seed42/predictions.tsv \
        --output ./results/comparisons/random10k_vs_structured4k.json

    # Compare across seeds:
    python significance.py \
        --dir-a ./results/exp1/RANDOM-10K/ \
        --dir-b ./results/exp1/STRUCTURED-4K-ONLY/ \
        --output ./results/comparisons/random10k_vs_structured4k_seeds.json

    # Full pairwise comparison of all conditions:
    python significance.py \
        --all-dirs ./results/exp1/RANDOM-10K/ ./results/exp1/STRUCTURED-4K-ONLY/ \
            ./results/exp1/RANDOM-6K_STRUCTURED-4K/ \
        --output ./results/exp1/pairwise_significance.json
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations

import numpy as np


def load_predictions(pred_file):
    """Load predictions TSV: src\\tref\\tpred per line."""
    sources, references, predictions = [], [], []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                sources.append(parts[0])
                references.append(parts[1])
                predictions.append(parts[2])
    return sources, references, predictions


def sentence_bleu_scores(references, predictions):
    """Compute sentence-level BLEU scores."""
    import sacrebleu
    scores = []
    for ref, pred in zip(references, predictions):
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        scores.append(bleu.score)
    return np.array(scores)


def sentence_chrf_scores(references, predictions):
    """Compute sentence-level chrF scores."""
    import sacrebleu
    scores = []
    for ref, pred in zip(references, predictions):
        chrf = sacrebleu.sentence_chrf(pred, [ref])
        scores.append(chrf.score)
    return np.array(scores)


def paired_bootstrap(scores_a, scores_b, n_samples=1000, seed=42):
    """
    Paired bootstrap resampling (Koehn 2004).
    Returns p-value: probability that system B is NOT better than system A.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n, "Systems must have same number of test sentences"

    wins_b = 0
    for _ in range(n_samples):
        indices = rng.randint(0, n, size=n)
        sample_a = scores_a[indices].mean()
        sample_b = scores_b[indices].mean()
        if sample_b > sample_a:
            wins_b += 1

    p_value = 1 - (wins_b / n_samples)
    return p_value


def cohens_d(scores_a, scores_b):
    """Compute Cohen's d effect size."""
    diff = scores_b - scores_a
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    return float(d)


def confidence_interval(values, confidence=0.95):
    """Compute confidence interval using bootstrap."""
    rng = np.random.RandomState(42)
    n = len(values)
    means = []
    for _ in range(10000):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = 1 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    return float(np.percentile(means, lower_pct)), float(np.percentile(means, upper_pct))


def wilcoxon_test(values_a, values_b):
    """Wilcoxon signed-rank test across seeds."""
    from scipy.stats import wilcoxon
    if len(values_a) < 5:
        return None  # Need at least 5 pairs for meaningful test
    try:
        stat, p_value = wilcoxon(values_a, values_b, alternative="two-sided")
        return float(p_value)
    except Exception:
        return None


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return {
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_comparisons": n,
        "significant": [p < corrected_alpha for p in p_values],
    }


def compare_two_systems(pred_file_a, pred_file_b, n_bootstrap=1000):
    """Compare two prediction files with all significance tests."""
    _, refs_a, preds_a = load_predictions(pred_file_a)
    _, refs_b, preds_b = load_predictions(pred_file_b)

    assert refs_a == refs_b, "Reference sets must be identical"

    results = {}

    for metric_name, scorer in [("bleu", sentence_bleu_scores), ("chrf", sentence_chrf_scores)]:
        scores_a = scorer(refs_a, preds_a)
        scores_b = scorer(refs_b, preds_b)

        mean_a = float(scores_a.mean())
        mean_b = float(scores_b.mean())
        diff = mean_b - mean_a

        p_value = paired_bootstrap(scores_a, scores_b, n_bootstrap)
        d = cohens_d(scores_a, scores_b)
        ci = confidence_interval(scores_b - scores_a)

        results[metric_name] = {
            "system_a_mean": mean_a,
            "system_b_mean": mean_b,
            "diff": diff,
            "p_value": p_value,
            "cohens_d": d,
            "ci_95": ci,
            "significant_at_005": p_value < 0.05,
        }

    return results


def compare_across_seeds(dir_a, dir_b, metric="bleu"):
    """Compare two conditions across multiple seeds using Wilcoxon."""
    # Collect per-seed metrics, keyed by seed name for proper pairing
    metrics_a = {}
    metrics_b = {}

    for seed_dir in sorted(glob.glob(os.path.join(dir_a, "seed*"))):
        seed_name = os.path.basename(seed_dir)
        metrics_file = os.path.join(seed_dir, "test_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                m = json.load(f)
            metrics_a[seed_name] = m.get(f"test_{metric}", 0)

    for seed_dir in sorted(glob.glob(os.path.join(dir_b, "seed*"))):
        seed_name = os.path.basename(seed_dir)
        metrics_file = os.path.join(seed_dir, "test_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                m = json.load(f)
            metrics_b[seed_name] = m.get(f"test_{metric}", 0)

    if not metrics_a or not metrics_b:
        return None

    # Ensure proper pairing: only compare seeds that exist in both conditions
    common_seeds = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
    if not common_seeds:
        return None

    values_a = np.array([metrics_a[s] for s in common_seeds])
    values_b = np.array([metrics_b[s] for s in common_seeds])

    result = {
        "system_a": os.path.basename(dir_a),
        "system_b": os.path.basename(dir_b),
        "metric": metric,
        "system_a_mean": float(values_a.mean()),
        "system_a_std": float(values_a.std()),
        "system_b_mean": float(values_b.mean()),
        "system_b_std": float(values_b.std()),
        "diff": float(values_b.mean() - values_a.mean()),
        "n_seeds_a": len(values_a),
        "n_seeds_b": len(values_b),
    }

    # Wilcoxon (if enough seeds)
    if len(values_a) == len(values_b) and len(values_a) >= 5:
        p_val = wilcoxon_test(values_a, values_b)
        result["wilcoxon_p"] = p_val
        if p_val is not None:
            result["wilcoxon_significant_005"] = p_val < 0.05

    # Cohen's d on seed-level means
    if len(values_a) > 1 and len(values_b) > 1:
        pooled_std = np.sqrt((values_a.std()**2 + values_b.std()**2) / 2)
        if pooled_std > 0:
            result["cohens_d"] = float((values_b.mean() - values_a.mean()) / pooled_std)

    return result


def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system-a", help="Predictions TSV for system A")
    group.add_argument("--dir-a", help="Results directory for system A (has seed*/)")
    group.add_argument("--all-dirs", nargs="+", help="All condition directories for pairwise comparison")

    parser.add_argument("--system-b", help="Predictions TSV for system B")
    parser.add_argument("--dir-b", help="Results directory for system B")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.system_a and args.system_b:
        # Single file comparison
        results = compare_two_systems(args.system_a, args.system_b, args.n_bootstrap)
        results["system_a_file"] = args.system_a
        results["system_b_file"] = args.system_b

    elif args.dir_a and args.dir_b:
        # Cross-seed comparison
        results = {}
        for metric in ["bleu", "chrf"]:
            result = compare_across_seeds(args.dir_a, args.dir_b, metric)
            if result:
                results[metric] = result

    elif args.all_dirs:
        # Pairwise comparison of all conditions
        results = {"comparisons": [], "bonferroni": {}}
        p_values = []

        for dir_a, dir_b in combinations(args.all_dirs, 2):
            for metric in ["bleu", "chrf"]:
                result = compare_across_seeds(dir_a, dir_b, metric)
                if result:
                    results["comparisons"].append(result)
                    if "wilcoxon_p" in result and result["wilcoxon_p"] is not None:
                        p_values.append(result["wilcoxon_p"])

        if p_values:
            results["bonferroni"] = bonferroni_correction(p_values, args.alpha)

    else:
        print("ERROR: Must provide either --system-a/--system-b, --dir-a/--dir-b, or --all-dirs")
        sys.exit(1)

    # Print results
    print(json.dumps(results, indent=2))

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
