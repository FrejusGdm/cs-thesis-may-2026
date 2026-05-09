"""
generate_jobs_tsv_april2026.py — emit jobs_april2026.tsv for the new-data sweep.

Layout (matches the existing generate_jobs_tsv.py schema so submit_april2026.sbatch
can use the same sed-by-line-number pattern):

    job_id  experiment  condition  seed  model_key  results_subdir  similar_lang

  - experiment ∈ {april2026_runA, april2026_runB1, april2026_runB2}
  - condition  ∈ {FULL, STRUCT4K-ALL-BASELINES-PLUS-NEW}
                 (Run A has both; Run B1/B2 have FULL only — the BLEU-41 recipe
                  is tied to the paper test set, so it isn't reproduced for the
                  new-test-distribution variants.)
  - seed       ∈ {42, 123, 456, 789, 2024}
  - models     ∈ NLLB-1.3B, NLLB-600M, mBART-fr, mBART-rand

Job count: (2 + 1 + 1) conditions × 5 seeds × 4 models = 80 jobs.

Usage:
    python generate_jobs_tsv_april2026.py [--output jobs_april2026.tsv]
"""

import argparse

SEEDS = [42, 123, 456, 789, 2024]

# Lines 1-80 of jobs_april2026.tsv — DO NOT REORDER OR REMOVE.
# These are the conditions you've already submitted; the SLURM array reads
# rows by line number at job-execution time, so any shuffle here would
# silently misroute already-queued tasks. Add new conditions to
# EXTRA_RUN_CONDITIONS instead.
BASE_RUN_CONDITIONS = [
    ("april2026_runA", "FULL"),
    ("april2026_runA", "STRUCT4K-ALL-BASELINES-PLUS-NEW"),
    ("april2026_runB1", "FULL"),
    ("april2026_runB2", "FULL"),
]

# Lines 81-120 — appended in a second pass so they get strictly higher line
# numbers. Submit these incrementally with `sbatch --array=81-120 …`.
EXTRA_RUN_CONDITIONS = [
    ("april2026_runB1", "STRUCT4K-ALL-BASELINES-PLUS-NEW"),
    ("april2026_runB2", "STRUCT4K-ALL-BASELINES-PLUS-NEW"),
]

# Lines 121-140 — second extras wave (runB3: sentence-only new6k pool).
# Added to answer "did the new data help on the same sentence-translation
# task", without the dict-entry skew that drags runB2 scores.
# Submit with `sbatch --array=121-140 …`.
EXTRA_RUN_CONDITIONS_V2 = [
    ("april2026_runB3", "FULL"),
]

# (model_key, results_subdir, similar_lang)
MODELS = [
    ("nllb-1.3b", "nllb-1.3b", "ewe_Latn"),
    ("nllb-600m", "nllb-600m", "ewe_Latn"),
    ("mbart-50", "mbart-fr", "fr_XX"),
    ("mbart-50", "mbart-rand", "none"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="jobs_april2026.tsv")
    args = ap.parse_args()

    rows: list[str] = []
    job_id = 0

    # --- Pass 1: BASE rows (lines 1..80). Identical bytes to the version
    #     already deployed/submitted on HPC; do not change ordering here. ---
    for model_key, results_subdir, similar_lang in MODELS:
        for experiment, condition in BASE_RUN_CONDITIONS:
            for seed in SEEDS:
                job_id += 1
                rows.append(
                    f"{job_id}\t{experiment}\t{condition}\t{seed}\t"
                    f"{model_key}\t{results_subdir}\t{similar_lang}"
                )
    base_count = job_id

    # --- Pass 2: EXTRA rows (lines 81..120). Same model loop so each model's
    #     extras stay grouped (10 per model = 2 conds × 5 seeds). ---
    for model_key, results_subdir, similar_lang in MODELS:
        for experiment, condition in EXTRA_RUN_CONDITIONS:
            for seed in SEEDS:
                job_id += 1
                rows.append(
                    f"{job_id}\t{experiment}\t{condition}\t{seed}\t"
                    f"{model_key}\t{results_subdir}\t{similar_lang}"
                )
    pass2_end = job_id

    # --- Pass 3: EXTRA_V2 rows (lines 121..140). runB3 FULL across models. ---
    for model_key, results_subdir, similar_lang in MODELS:
        for experiment, condition in EXTRA_RUN_CONDITIONS_V2:
            for seed in SEEDS:
                job_id += 1
                rows.append(
                    f"{job_id}\t{experiment}\t{condition}\t{seed}\t"
                    f"{model_key}\t{results_subdir}\t{similar_lang}"
                )

    with open(args.output, "w") as f:
        for r in rows:
            f.write(r + "\n")

    base_per_model = len(BASE_RUN_CONDITIONS) * len(SEEDS)
    extra_per_model = len(EXTRA_RUN_CONDITIONS) * len(SEEDS)
    extra_v2_per_model = len(EXTRA_RUN_CONDITIONS_V2) * len(SEEDS)
    print(f"Generated {args.output} with {len(rows)} jobs "
          f"({base_count} base + {pass2_end - base_count} extras + {len(rows) - pass2_end} extras_v2)")
    for i, (mk, rsd, _) in enumerate(MODELS):
        lo = i * base_per_model + 1
        hi = (i + 1) * base_per_model
        print(f"  {rsd:<12} ({mk}) base:     lines {lo:>3}-{hi:<3}")
    for i, (mk, rsd, _) in enumerate(MODELS):
        lo = base_count + i * extra_per_model + 1
        hi = base_count + (i + 1) * extra_per_model
        print(f"  {rsd:<12} ({mk}) extra:    lines {lo:>3}-{hi:<3}")
    for i, (mk, rsd, _) in enumerate(MODELS):
        lo = pass2_end + i * extra_v2_per_model + 1
        hi = pass2_end + (i + 1) * extra_v2_per_model
        print(f"  {rsd:<12} ({mk}) extra_v2: lines {lo:>3}-{hi:<3}")


if __name__ == "__main__":
    main()
