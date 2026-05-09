"""
prepare_ablations.py — Construct all ablation training sets from structured data.

Ablations:
  1. Module leave-one-out (FULL, -NEG, -PAST, -FUTURE, -QUEST, BASE-ONLY)
  1b. Size-controlled module ablation (all at 1,000 sentences)
  2. Pronoun coverage (ALL-8, REDUCED-4, SINGULAR-3, MINIMAL-1)
  3. Verb diversity (10, 5, 3, 1 verbs)
  4. Minimal-pair structure (PAIRS-INTACT vs PAIRS-BROKEN)

Usage:
    python prepare_ablations.py \
        --structured /path/to/structured_combined.csv \
        --output-dir ./splits/ablations \
        --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd


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
    if len(df) == 0:
        return df.copy(), df.copy()
    val_size = max(1, int(len(df) * val_ratio))
    val_size = min(val_size, len(df) - 1)  # Ensure at least 1 training sample
    if val_size <= 0:
        return df.copy(), df.head(0)
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


# Module name to number mapping
MODULE_MAP = {
    "module1_base": 1,
    "module2_negation": 2,
    "module3_past": 3,
    "module4_future": 4,
    "module5_questions": 5,
    "module5_questions_dedup": 5,
    # Short forms (e.g. enriched CSV uses "M1", "M2", ...)
    "m1": 1,
    "m2": 2,
    "m3": 3,
    "m4": 4,
    "m5": 5,
}


def get_module_number(module_name):
    """Map module column value to module number."""
    name = str(module_name).lower().strip()
    # Exact match first (for short forms like "m1")
    if name in MODULE_MAP:
        return MODULE_MAP[name]
    # Substring match (for long forms like "module1_base")
    for key, val in MODULE_MAP.items():
        if key in name:
            return val
    return None


def ablation_module_loo(df, val_ratio, seed, output_dir, src_col, tgt_col):
    """Module leave-one-out ablation."""
    print("\n=== Ablation 1: Module Leave-One-Out ===")
    df = df.copy()
    df["_module_num"] = df["module"].apply(get_module_number)

    conditions = {
        "FULL": [1, 2, 3, 4, 5],
        "NO-NEGATION": [1, 3, 4, 5],
        "NO-PAST": [1, 2, 4, 5],
        "NO-FUTURE": [1, 2, 3, 5],
        "NO-QUESTIONS": [1, 2, 3, 4],
        "BASE-ONLY": [1],
    }

    for name, modules in conditions.items():
        subset = df[df["_module_num"].isin(modules)].drop(columns=["_module_num"])
        train, val = make_train_val_split(subset, val_ratio, seed)
        save_split(train, os.path.join(output_dir, "module_loo", name), "train",
                   src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "module_loo", name), "val",
                   src_col, tgt_col)
        print(f"  {name}: {len(subset)} sentences (modules {modules})")


def ablation_module_size_controlled(df, val_ratio, seed, output_dir, src_col, tgt_col,
                                     total_size=1000):
    """Size-controlled module ablation — all conditions at fixed total size."""
    print(f"\n=== Ablation 1b: Size-Controlled Module ({total_size} each) ===")
    df = df.copy()
    df["_module_num"] = df["module"].apply(get_module_number)

    conditions = {
        "FULL-1K": ([1, 2, 3, 4, 5], 200),
        "NO-NEG-1K": ([1, 3, 4, 5], 250),
        "NO-PAST-1K": ([1, 2, 4, 5], 250),
        "NO-FUT-1K": ([1, 2, 3, 5], 250),
        "NO-QUEST-1K": ([1, 2, 3, 4], 250),
        "BASE-1K": ([1], 1000),
    }

    for name, (modules, per_module) in conditions.items():
        subset_parts = []
        for mod in modules:
            mod_df = df[df["_module_num"] == mod]
            n = min(len(mod_df), per_module)
            subset_parts.append(mod_df.sample(n=n, random_state=seed))
        subset = pd.concat(subset_parts, ignore_index=True).drop(columns=["_module_num"])
        subset = subset.head(total_size)

        train, val = make_train_val_split(subset, val_ratio, seed)
        save_split(train, os.path.join(output_dir, "module_size_ctrl", name), "train",
                   src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "module_size_ctrl", name), "val",
                   src_col, tgt_col)
        print(f"  {name}: {len(subset)} sentences ({per_module}/module, modules {modules})")


def ablation_pronoun_coverage(df, val_ratio, seed, output_dir, src_col, tgt_col):
    """Pronoun coverage ablation."""
    print("\n=== Ablation 2: Pronoun Coverage ===")

    conditions = {
        "ALL-8": ["je", "tu", "il", "elle", "nous", "vous", "ils", "elles"],
        "REDUCED-4": ["je", "tu", "il", "nous"],
        "SINGULAR-3": ["je", "tu", "il"],
        "MINIMAL-1": ["je"],
    }

    for name, pronouns in conditions.items():
        subset = df[df["pronoun"].isin(pronouns)]
        train, val = make_train_val_split(subset, val_ratio, seed)
        save_split(train, os.path.join(output_dir, "pronoun", name), "train",
                   src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "pronoun", name), "val",
                   src_col, tgt_col)
        print(f"  {name}: {len(subset)} sentences (pronouns: {pronouns})")


def ablation_verb_diversity(df, val_ratio, seed, output_dir, src_col, tgt_col):
    """Verb diversity ablation."""
    print("\n=== Ablation 3: Verb Diversity ===")
    rng = np.random.RandomState(seed)
    all_verbs = sorted(df["verb"].unique().tolist())
    print(f"  Available verbs: {all_verbs}")

    # 10 verbs (all)
    subset = df.copy()
    train, val = make_train_val_split(subset, val_ratio, seed)
    save_split(train, os.path.join(output_dir, "verb", "10-VERBS"), "train",
               src_col, tgt_col)
    save_split(val, os.path.join(output_dir, "verb", "10-VERBS"), "val",
               src_col, tgt_col)
    print(f"  10-VERBS: {len(subset)} sentences")

    # 5 verbs — 3 random subsets
    for i in range(3):
        verbs = sorted(rng.choice(all_verbs, size=5, replace=False).tolist())
        subset = df[df["verb"].isin(verbs)]
        name = f"5-VERBS-{chr(97+i)}"  # a, b, c
        train, val = make_train_val_split(subset, val_ratio, seed + i)
        save_split(train, os.path.join(output_dir, "verb", name), "train",
                   src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "verb", name), "val",
                   src_col, tgt_col)
        print(f"  {name}: {len(subset)} sentences (verbs: {verbs})")

    # 3 verbs — 3 random subsets
    for i in range(3):
        verbs = sorted(rng.choice(all_verbs, size=3, replace=False).tolist())
        subset = df[df["verb"].isin(verbs)]
        name = f"3-VERBS-{chr(97+i)}"
        train, val = make_train_val_split(subset, val_ratio, seed + i)
        save_split(train, os.path.join(output_dir, "verb", name), "train",
                   src_col, tgt_col)
        save_split(val, os.path.join(output_dir, "verb", name), "val",
                   src_col, tgt_col)
        print(f"  {name}: {len(subset)} sentences (verbs: {verbs})")

    # 1 verb (manger — most common/basic)
    subset = df[df["verb"] == "manger"]
    if len(subset) == 0:
        # Fall back to first available verb
        fallback_verb = all_verbs[0]
        subset = df[df["verb"] == fallback_verb]
        print(f"  WARNING: 'manger' not found, using '{fallback_verb}'")
    train, val = make_train_val_split(subset, val_ratio, seed)
    save_split(train, os.path.join(output_dir, "verb", "1-VERB"), "train",
               src_col, tgt_col)
    save_split(val, os.path.join(output_dir, "verb", "1-VERB"), "val",
               src_col, tgt_col)
    print(f"  1-VERB: {len(subset)} sentences")


def ablation_minimal_pairs(df, val_ratio, seed, output_dir, src_col, tgt_col):
    """Minimal-pair structure ablation: intact vs. broken pairings."""
    print("\n=== Ablation 4: Minimal-Pair Structure ===")

    # PAIRS-INTACT: original data as-is
    train, val = make_train_val_split(df, val_ratio, seed)
    save_split(train, os.path.join(output_dir, "minimal_pairs", "PAIRS-INTACT"), "train",
               src_col, tgt_col)
    save_split(val, os.path.join(output_dir, "minimal_pairs", "PAIRS-INTACT"), "val",
               src_col, tgt_col)
    print(f"  PAIRS-INTACT: {len(df)} sentences")

    # PAIRS-BROKEN: shuffle transformations within each module independently
    df_broken = df.copy()
    df_broken["_module_num"] = df_broken["module"].apply(get_module_number)

    rng = np.random.RandomState(seed)
    for mod_num in [2, 3, 4, 5]:
        mask = df_broken["_module_num"] == mod_num
        mod_indices = df_broken.index[mask]
        # Shuffle the French sentences within this module
        shuffled_french = df_broken.loc[mod_indices, src_col].values.copy()
        rng.shuffle(shuffled_french)
        df_broken.loc[mod_indices, src_col] = shuffled_french
        # Also shuffle the Adja translations correspondingly
        shuffled_tgt = df_broken.loc[mod_indices, tgt_col].values.copy()
        rng.shuffle(shuffled_tgt)
        df_broken.loc[mod_indices, tgt_col] = shuffled_tgt

    df_broken = df_broken.drop(columns=["_module_num"])
    train, val = make_train_val_split(df_broken, val_ratio, seed)
    save_split(train, os.path.join(output_dir, "minimal_pairs", "PAIRS-BROKEN"), "train",
               src_col, tgt_col)
    save_split(val, os.path.join(output_dir, "minimal_pairs", "PAIRS-BROKEN"), "val",
               src_col, tgt_col)
    print(f"  PAIRS-BROKEN: {len(df_broken)} sentences (shuffled M2-M5)")


def main():
    parser = argparse.ArgumentParser(description="Construct ablation training sets")
    parser.add_argument("--structured", required=True, help="Path to structured combined CSV")
    parser.add_argument("--output-dir", default="./splits/ablations", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--src-col", default="french")
    parser.add_argument("--tgt-col", default="adja_translation")
    args = parser.parse_args()

    print("Loading structured data...")
    df = load_parallel_csv(args.structured, args.src_col, args.tgt_col)
    # Drop rows with missing pronoun or verb (needed for ablation grouping)
    n_before = len(df)
    df = df.dropna(subset=["pronoun", "verb"])
    df = df[df["pronoun"].str.strip() != ""]
    df = df[df["verb"].str.strip() != ""]
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing pronoun/verb")
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df)} sentences")
    print(f"  Modules: {df['module'].value_counts().to_dict()}")
    print(f"  Pronouns: {sorted(df['pronoun'].unique().tolist())}")
    print(f"  Verbs: {sorted(df['verb'].unique().tolist())}")

    ablation_module_loo(df, args.val_ratio, args.seed, args.output_dir,
                        args.src_col, args.tgt_col)
    ablation_module_size_controlled(df, args.val_ratio, args.seed, args.output_dir,
                                     args.src_col, args.tgt_col)
    ablation_pronoun_coverage(df, args.val_ratio, args.seed, args.output_dir,
                              args.src_col, args.tgt_col)
    ablation_verb_diversity(df, args.val_ratio, args.seed, args.output_dir,
                            args.src_col, args.tgt_col)
    ablation_minimal_pairs(df, args.val_ratio, args.seed, args.output_dir,
                           args.src_col, args.tgt_col)

    print(f"\nAll ablation splits saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
