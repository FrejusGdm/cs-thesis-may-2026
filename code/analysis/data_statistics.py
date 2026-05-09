"""
data_statistics.py — Compute and report data statistics for the paper.

Generates:
  - Vocabulary size, type-token ratio, entropy for each dataset
  - Sentence length distributions
  - Vocabulary overlap matrices between conditions and test set
  - Information-theoretic diversity metrics

Usage:
    python data_statistics.py \
        --structured /path/to/structured_combined.csv \
        --random /path/to/tatoeba_10k.csv \
        --test /path/to/heldout_diverse.csv \
        --output ./results/data_statistics.json
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd


def load_texts(path, col="french"):
    """Load text column from CSV. Auto-detects common column name variants."""
    df = pd.read_csv(path)
    if col not in df.columns:
        for alias in ["French", "french", "src", "source"]:
            if alias in df.columns:
                df = df.rename(columns={alias: col})
                break
        else:
            raise ValueError(
                f"Column '{col}' not found in {path}. Columns: {list(df.columns)}"
            )
    return df[col].dropna().astype(str).tolist()


def compute_statistics(texts, name=""):
    """Compute vocabulary and distributional statistics for a list of texts."""
    all_tokens = []
    lengths = []
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        lengths.append(len(tokens))

    vocab = set(all_tokens)
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)

    # Type-token ratio
    ttr = len(vocab) / total_tokens if total_tokens > 0 else 0

    # Vocabulary entropy (Shannon)
    entropy = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        if p > 0:
            entropy -= p * math.log2(p)

    # Hapax legomena (words appearing once)
    hapax = sum(1 for c in token_counts.values() if c == 1)

    stats = {
        "name": name,
        "n_sentences": len(texts),
        "n_tokens": total_tokens,
        "n_types": len(vocab),
        "type_token_ratio": round(ttr, 4),
        "vocabulary_entropy": round(entropy, 4),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax / len(vocab), 4) if vocab else 0,
        "mean_length": round(np.mean(lengths), 2),
        "std_length": round(np.std(lengths), 2),
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "median_length": round(float(np.median(lengths)), 1),
    }

    return stats, vocab


def vocab_overlap_matrix(datasets):
    """Compute pairwise vocabulary overlap between datasets."""
    names = [d[0] for d in datasets]
    vocabs = [d[1] for d in datasets]
    n = len(names)

    matrix = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                overlap_pct = 100.0
            else:
                overlap = len(vocabs[i] & vocabs[j])
                overlap_pct = overlap / len(vocabs[j]) * 100 if vocabs[j] else 0
            matrix[f"{names[i]}_covers_{names[j]}"] = round(overlap_pct, 1)

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Compute data statistics")
    parser.add_argument("--structured", help="Structured combined CSV")
    parser.add_argument("--random", help="Random Tatoeba CSV")
    parser.add_argument("--test", help="Test set CSV")
    parser.add_argument("--src-col", default="french", help="Source column name")
    parser.add_argument("--output", default="data_statistics.json", help="Output JSON")
    args = parser.parse_args()

    results = {}
    datasets = []

    for name, path in [("structured", args.structured), ("random", args.random),
                        ("test", args.test)]:
        if path and os.path.exists(path):
            print(f"Computing stats for {name}...")
            texts = load_texts(path, args.src_col)
            stats, vocab = compute_statistics(texts, name)
            results[name] = stats
            datasets.append((name, vocab))

            print(f"  {name}: {stats['n_sentences']} sentences, "
                  f"{stats['n_types']} types, "
                  f"TTR={stats['type_token_ratio']:.4f}, "
                  f"H={stats['vocabulary_entropy']:.2f} bits")

    # Overlap matrix
    if len(datasets) > 1:
        print("\nVocabulary overlap matrix:")
        overlap = vocab_overlap_matrix(datasets)
        results["vocab_overlap"] = overlap
        for key, val in overlap.items():
            print(f"  {key}: {val}%")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
