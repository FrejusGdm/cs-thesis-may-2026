"""
generate_jobs_tsv_rebuttal.py — emit jobs_rebuttal.tsv for ACL rebuttal subset-BLEU reruns.

Why: paper-era results in experiments/results/hpc_new/ and exp1/ were produced by an
older training script that did NOT save per-sentence JSONL predictions, so subset
BLEU on the structured/Tatoeba split cannot be recomputed offline. This sweep re-runs
the marquee paper conditions with the current training script (which saves
test_predictions.jsonl alongside test_metrics.json), under a fresh results subfolder
results/rebuttal_rerun/ so paper-era and april-2026 results are untouched.

Tiers (each layered on top of the previous so you can submit incrementally):
  TIER 1 (20 jobs): NLLB-600M × {STRUCTURED-2K, RANDOM-10K, RANDOM-10K_STRUCTURED-4K,
                                 STRUCTURED-4K-ONLY} × 5 seeds
                    — the four conditions cited in the paper's headline claims.
  TIER 2 (+10):     NLLB-600M × {RANDOM-6K_STRUCTURED-4K, RANDOM-4K} × 5 seeds
                    — completes exp1, needed for tab:replacement and tab:scaling.
  TIER 3 (+30):     mBART-fr × all six exp1 conditions × 5 seeds
                    — needed to argue subset BLEU pattern holds across architectures.
  TIER 4 (+30):     NLLB-1.3B × all six exp1 conditions × 5 seeds — best-effort.

Output schema (matches generate_jobs_tsv.py exactly so submit_rebuttal_rerun.sbatch
can use the same sed-by-line-number pattern):

    job_id  experiment  condition  seed  model_key  results_subdir  similar_lang

Usage:
    python generate_jobs_tsv_rebuttal.py [--output jobs_rebuttal.tsv]

Submit on HPC:
    sbatch --array=1-20  submit_rebuttal_rerun.sbatch   # Tier 1
    sbatch --array=1-30  submit_rebuttal_rerun.sbatch   # Tier 1 + 2
    sbatch --array=1-60  submit_rebuttal_rerun.sbatch   # Tier 1 + 2 + 3
    sbatch --array=1-90  submit_rebuttal_rerun.sbatch   # All four tiers
"""

import argparse

SEEDS = [42, 123, 456, 789, 2024]

# Model order: NLLB-600M FIRST (priority for rebuttal), then mBART-fr, then NLLB-1.3B.
# This is intentionally different from generate_jobs_tsv.py's ordering.
TIER1_CONDITIONS = [
    "STRUCTURED-2K",
    "RANDOM-10K",
    "RANDOM-10K_STRUCTURED-4K",
    "STRUCTURED-4K-ONLY",
]
TIER2_CONDITIONS = [
    "RANDOM-6K_STRUCTURED-4K",
    "RANDOM-4K",
]
ALL_EXP1_CONDITIONS = TIER1_CONDITIONS + TIER2_CONDITIONS  # 6 conditions

EXPERIMENT = "exp1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="jobs_rebuttal.tsv")
    args = ap.parse_args()

    rows: list[str] = []
    job_id = 0

    def add_jobs(model_key: str, results_subdir: str, similar_lang: str, conditions: list[str]):
        nonlocal job_id
        for cond in conditions:
            for seed in SEEDS:
                nonlocal_id = job_id + 1  # noqa: F841 (placeholder to satisfy linter; not used)
                job_id += 1
                rows.append(
                    f"{job_id}\t{EXPERIMENT}\t{cond}\t{seed}\t"
                    f"{model_key}\t{results_subdir}\t{similar_lang}"
                )

    # TIER 1: NLLB-600M, 4 conditions, 5 seeds = 20 jobs (lines 1-20)
    add_jobs("nllb-600m", "nllb-600m", "ewe_Latn", TIER1_CONDITIONS)

    # TIER 2: NLLB-600M, 2 more conditions, 5 seeds = 10 jobs (lines 21-30)
    add_jobs("nllb-600m", "nllb-600m", "ewe_Latn", TIER2_CONDITIONS)

    # TIER 3: mBART-fr, all 6 conditions, 5 seeds = 30 jobs (lines 31-60)
    add_jobs("mbart-50", "mbart-fr", "fr_XX", ALL_EXP1_CONDITIONS)

    # TIER 4: NLLB-1.3B, all 6 conditions, 5 seeds = 30 jobs (lines 61-90) — best-effort
    add_jobs("nllb-1.3b", "nllb-1.3b", "ewe_Latn", ALL_EXP1_CONDITIONS)

    with open(args.output, "w") as f:
        for r in rows:
            f.write(r + "\n")

    print(f"Generated {args.output} with {len(rows)} jobs")
    print()
    print("Tier ranges (use with sbatch --array=...):")
    print(f"  Tier 1 (NLLB-600M, headline 4 conditions): lines 1-20    [20 jobs]")
    print(f"  Tier 1+2 (NLLB-600M, all exp1 conditions): lines 1-30    [30 jobs]")
    print(f"  Tier 1+2+3 (+ mBART-fr × exp1):            lines 1-60    [60 jobs]")
    print(f"  Tier 1+2+3+4 (+ NLLB-1.3B × exp1):         lines 1-90    [90 jobs]")
    print()
    print("Smoke test (one job per model, STRUCTURED-2K, seed 42):")
    print(f"  --array=1,31,61")


if __name__ == "__main__":
    main()
