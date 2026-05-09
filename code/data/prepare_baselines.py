"""
prepare_baselines.py — Construct all baseline training sets from 10K random data.

Baselines:
  1. RANDOM-FULL: All 10K (no construction needed — handled by prepare_splits.py)
  2. LENGTH-STRATIFIED: 2K stratified by sentence length
  3. VOCAB-MAXIMIZED: 2K via greedy vocabulary coverage
  4. TF-IDF DIVERSE: 2K via k-means in TF-IDF space

Usage:
    python prepare_baselines.py \
        --random /path/to/tatoeba_10k.csv \
        --output-dir ./splits/baselines \
        --size 2000 \
        --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def load_parallel_csv(path, src_col="french", tgt_col="adja_translation"):
    """Load a parallel corpus CSV.

    Auto-detects common column name variants (e.g. 'French' -> 'french',
    'Translation' -> 'adja_translation') and renames them so downstream
    code always sees the expected names.
    """
    df = pd.read_csv(path)
    aliases = {
        src_col: ["French", "french", "src", "source"],
        tgt_col: ["Translation", "adja_translation", "tgt", "target"],
    }
    col_map = {}
    for expected, candidates in aliases.items():
        if expected in df.columns:
            continue
        for alias in candidates:
            if alias in df.columns:
                col_map[alias] = expected
                break
        else:
            raise ValueError(
                f"Column '{expected}' not found in {path}. "
                f"Columns: {list(df.columns)}"
            )
    if col_map:
        df = df.rename(columns=col_map)
    df = df.dropna(subset=[src_col, tgt_col])
    df = df[df[tgt_col].str.strip() != ""]
    return df


def make_train_val_split(df, val_ratio, seed):
    """Split into train and val."""
    val_size = max(1, int(len(df) * val_ratio))
    val_df = df.sample(n=val_size, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_split(df, output_dir, name, src_col="french", tgt_col="adja_translation"):
    """Save as CSV and TSV."""
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    df[[src_col, tgt_col]].to_csv(
        os.path.join(output_dir, f"{name}.tsv"), sep="\t", index=False, header=False
    )


def length_stratified(df, size, seed, src_col="french"):
    """Select sentences stratified by length bins: short (3-5), medium (6-9), long (10+)."""
    df = df.copy()
    df["_wc"] = df[src_col].str.split().str.len()
    df["_bin"] = pd.cut(
        df["_wc"], bins=[0, 5, 9, float("inf")], labels=["short", "medium", "long"]
    )

    per_bin = size // 3
    selected = df.groupby("_bin", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), per_bin), random_state=seed)
    )
    # Top up from remaining if any bin was short
    remaining = size - len(selected)
    if remaining > 0:
        pool = df.drop(selected.index)
        extra = pool.sample(n=min(len(pool), remaining), random_state=seed)
        selected = pd.concat([selected, extra])

    selected = selected.head(size).drop(columns=["_wc", "_bin"])
    return selected.reset_index(drop=True)


def vocab_maximized(df, size, seed, src_col="french"):
    """Select sentences via greedy set-cover to maximize vocabulary coverage."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_tokens"] = df[src_col].str.lower().str.split()

    # Precompute token sets for speed
    token_sets = [set(tokens) for tokens in df["_tokens"]]
    indices = list(range(len(df)))

    selected = []
    covered = set()

    for _ in range(min(size, len(df))):
        best_idx = -1
        best_gain = -1

        # Shuffle to break ties randomly
        rng.shuffle(indices)
        for idx in indices:
            if idx in selected:
                continue
            gain = len(token_sets[idx] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
            # Early termination: can't do better than all tokens being new
            if gain == len(token_sets[idx]):
                break

        if best_idx == -1:
            break
        selected.append(best_idx)
        covered.update(token_sets[best_idx])

        if len(selected) % 500 == 0:
            print(f"    Vocab-maximized: {len(selected)}/{size} selected, "
                  f"vocab={len(covered)} types")

    result = df.iloc[selected].drop(columns=["_tokens"])
    return result.reset_index(drop=True)


def tfidf_diverse(df, size, seed, src_col="french", n_clusters=20):
    """Select sentences via k-means clustering in TF-IDF space."""
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df[src_col])

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    df = df.copy()
    df["_cluster"] = kmeans.fit_predict(tfidf_matrix)

    per_cluster = size // n_clusters
    selected = df.groupby("_cluster", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), per_cluster), random_state=seed)
    )
    # Top up
    remaining = size - len(selected)
    if remaining > 0:
        pool = df.drop(selected.index)
        extra = pool.sample(n=min(len(pool), remaining), random_state=seed)
        selected = pd.concat([selected, extra])

    selected = selected.head(size).drop(columns=["_cluster"], errors="ignore")
    return selected.reset_index(drop=True)


def report_properties(df, name, src_col="french"):
    """Report data properties for a baseline."""
    vocab = set()
    lengths = []
    for text in df[src_col]:
        tokens = str(text).lower().split()
        vocab.update(tokens)
        lengths.append(len(tokens))

    total_tokens = sum(lengths)
    ttr = len(vocab) / total_tokens if total_tokens > 0 else 0

    print(f"  {name}:")
    print(f"    Sentences: {len(df)}")
    print(f"    Vocab size: {len(vocab)}")
    print(f"    Mean length: {np.mean(lengths):.1f} words")
    print(f"    Length range: [{min(lengths)}, {max(lengths)}]")
    print(f"    Type-token ratio: {ttr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Construct baseline training sets")
    parser.add_argument("--random", required=True, help="Path to 10K random Tatoeba CSV")
    parser.add_argument("--output-dir", default="./splits/baselines", help="Output directory")
    parser.add_argument("--size", type=int, default=2000, help="Target size for selection baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--src-col", default="french")
    parser.add_argument("--tgt-col", default="adja_translation")
    args = parser.parse_args()

    print("Loading random data...")
    df = load_parallel_csv(args.random, args.src_col, args.tgt_col)
    print(f"  Loaded {len(df)} sentences")

    print(f"\nConstructing baselines (target size: {args.size})...")

    # Baseline 2: LENGTH-STRATIFIED
    print("\n  Building LENGTH-STRATIFIED...")
    df_length = length_stratified(df, args.size, args.seed, args.src_col)
    train, val = make_train_val_split(df_length, args.val_ratio, args.seed)
    save_split(train, os.path.join(args.output_dir, "LENGTH-STRATIFIED"), "train",
               args.src_col, args.tgt_col)
    save_split(val, os.path.join(args.output_dir, "LENGTH-STRATIFIED"), "val",
               args.src_col, args.tgt_col)

    # Baseline 3: VOCAB-MAXIMIZED
    print("\n  Building VOCAB-MAXIMIZED...")
    df_vocab = vocab_maximized(df, args.size, args.seed, args.src_col)
    train, val = make_train_val_split(df_vocab, args.val_ratio, args.seed)
    save_split(train, os.path.join(args.output_dir, "VOCAB-MAXIMIZED"), "train",
               args.src_col, args.tgt_col)
    save_split(val, os.path.join(args.output_dir, "VOCAB-MAXIMIZED"), "val",
               args.src_col, args.tgt_col)

    # Baseline 4: TF-IDF DIVERSE
    print("\n  Building TF-IDF-DIVERSE...")
    df_tfidf = tfidf_diverse(df, args.size, args.seed, args.src_col)
    train, val = make_train_val_split(df_tfidf, args.val_ratio, args.seed)
    save_split(train, os.path.join(args.output_dir, "TF-IDF-DIVERSE"), "train",
               args.src_col, args.tgt_col)
    save_split(val, os.path.join(args.output_dir, "TF-IDF-DIVERSE"), "val",
               args.src_col, args.tgt_col)

    # Report properties
    print("\n=== Data Properties ===")
    report_properties(df, "RANDOM-FULL (10K)", args.src_col)
    report_properties(df_length, "LENGTH-STRATIFIED", args.src_col)
    report_properties(df_vocab, "VOCAB-MAXIMIZED", args.src_col)
    report_properties(df_tfidf, "TF-IDF-DIVERSE", args.src_col)

    print(f"\nAll baselines saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
