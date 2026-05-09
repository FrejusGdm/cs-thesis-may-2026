"""
generate_jobs_tsv.py — Generate the jobs.tsv parameter file for SLURM array submission.

Produces a TSV with one line per training job, covering:
  - NLLB-1.3B (lines 1-233)
  - NLLB-600M (lines 234-466)
  - mBART-50 French-init (lines 467-699)
  - mBART-50 Random-init (lines 700-932)

Usage:
    python generate_jobs_tsv.py [--output jobs.tsv]

Output columns:
    job_id  experiment  condition  seed  model_key  results_subdir  similar_lang

The SLURM array script reads line $SLURM_ARRAY_TASK_ID from this file.
Adjust --array to skip already-completed jobs (e.g. --array=41-699).
"""

import argparse
from datetime import datetime, timezone

# ── Condition definitions (must match launch_jobs.sh exactly) ──

SEEDS_5 = [42, 123, 456, 789, 2024]
SEEDS_3 = [42, 123, 456]

EXP1_CONDITIONS = [
    "RANDOM-10K",
    "RANDOM-6K_STRUCTURED-4K",
    "RANDOM-10K_STRUCTURED-4K",
    "STRUCTURED-4K-ONLY",
    "RANDOM-4K",
    "STRUCTURED-2K",
]

BASELINES = ["LENGTH-STRATIFIED", "VOCAB-MAXIMIZED", "TF-IDF-DIVERSE"]

STRUCTURED_SIZES = [200, 500, 1000, 2000, 3000, 4000]
RANDOM_SIZES = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
ADDITIVE_STRUCT_SIZES = [500, 1000, 2000, 4000]
REPLACEMENT_STRUCT_SIZES = [500, 1000, 2000, 4000]

MODULE_LOO = ["FULL", "NO-NEGATION", "NO-PAST", "NO-FUTURE", "NO-QUESTIONS", "BASE-ONLY"]
MODULE_SC = ["FULL-1K", "NO-NEG-1K", "NO-PAST-1K", "NO-FUT-1K", "NO-QUEST-1K", "BASE-1K"]
PRONOUNS = ["ALL-8", "REDUCED-4", "SINGULAR-3", "MINIMAL-1"]
VERBS = ["10-VERBS", "5-VERBS-a", "5-VERBS-b", "5-VERBS-c", "3-VERBS-a", "3-VERBS-b", "3-VERBS-c", "1-VERB"]
PAIRS = ["PAIRS-INTACT", "PAIRS-BROKEN"]


def generate_all_conditions():
    """Generate all (experiment, condition, seed) tuples in tier order."""
    jobs = []

    # ── Exp1: 6 conditions × 5 seeds = 30 ──
    for cond in EXP1_CONDITIONS:
        for seed in SEEDS_5:
            jobs.append(("exp1", cond, seed))

    # ── Baselines: 3 × 5 seeds = 15 ──
    for cond in BASELINES:
        for seed in SEEDS_5:
            jobs.append(("baselines", cond, seed))

    # ── Exp2 structured scaling: 6 sizes × 5 seeds = 30 ──
    for size in STRUCTURED_SIZES:
        for seed in SEEDS_5:
            jobs.append(("exp2", f"STRUCTURED-{size}", seed))

    # ── Exp2 random scaling: 8 sizes × 5 seeds = 40 ──
    for size in RANDOM_SIZES:
        for seed in SEEDS_5:
            jobs.append(("exp2", f"RANDOM-{size}", seed))

    # ── Exp2 additive: 4 × 5 seeds = 20 ──
    for struct_size in ADDITIVE_STRUCT_SIZES:
        for seed in SEEDS_5:
            jobs.append(("exp2", f"RANDOM-6K_STRUCTURED-{struct_size}", seed))

    # ── Exp2 replacement: 4 × 5 seeds = 20 ──
    for struct_size in REPLACEMENT_STRUCT_SIZES:
        random_size = 10000 - struct_size
        for seed in SEEDS_5:
            jobs.append(("exp2", f"REPLACE-R{random_size}_S{struct_size}", seed))

    # ── Ablations: module LOO 6×3 = 18 ──
    for cond in MODULE_LOO:
        for seed in SEEDS_3:
            jobs.append(("ablations/module_loo", cond, seed))

    # ── Ablations: size-controlled 6×3 = 18 ──
    for cond in MODULE_SC:
        for seed in SEEDS_3:
            jobs.append(("ablations/module_size_ctrl", cond, seed))

    # ── Ablations: pronouns 4×3 = 12 ──
    for cond in PRONOUNS:
        for seed in SEEDS_3:
            jobs.append(("ablations/pronoun", cond, seed))

    # ── Ablations: verbs 8×3 = 24 ──
    for cond in VERBS:
        for seed in SEEDS_3:
            jobs.append(("ablations/verb", cond, seed))

    # ── Ablations: minimal pairs 2×3 = 6 ──
    for cond in PAIRS:
        for seed in SEEDS_3:
            jobs.append(("ablations/minimal_pairs", cond, seed))

    return jobs  # 233 total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="jobs.tsv")
    args = parser.parse_args()

    conditions = generate_all_conditions()
    assert len(conditions) == 233, f"Expected 233 conditions, got {len(conditions)}"

    # Model configurations: (model_key, results_subdir, similar_lang)
    model_configs = [
        ("nllb-1.3b", "nllb-1.3b", "ewe_Latn"),
        ("nllb-600m", "nllb-600m", "ewe_Latn"),
        ("mbart-50", "mbart-fr", "fr_XX"),
        ("mbart-50", "mbart-rand", "none"),
    ]

    lines = []
    job_id = 0
    for model_key, results_subdir, similar_lang in model_configs:
        for experiment, condition, seed in conditions:
            job_id += 1
            lines.append(f"{job_id}\t{experiment}\t{condition}\t{seed}\t{model_key}\t{results_subdir}\t{similar_lang}")

    # Write CLEAN TSV — no comments, no header. Line N = job N.
    # This is critical: submit_array.sbatch uses sed -n "${SLURM_ARRAY_TASK_ID}p"
    # so line numbers must match job IDs exactly.
    with open(args.output, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Generated {args.output} with {len(lines)} jobs")
    print(f"  NLLB-1.3B:     lines 1-233")
    print(f"  NLLB-600M:     lines 234-466")
    print(f"  mBART-fr:      lines 467-699")
    print(f"  mBART-rand:    lines 700-932")
    print(f"  Total:         {len(lines)} jobs")


if __name__ == "__main__":
    main()
