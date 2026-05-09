"""
generate_jobs_tsv_pipeline.py — emit jobs_pipeline.tsv for downstream-pipeline checkpoint runs.

Goal: produce the smallest, cleanest set of trained NLLB-600M models the user can
plug into a downstream serving pipeline (later wired up via Hugging Face).

Scope (per user decision 2026-05-02):
  - Model:     NLLB-600M only (NLLB-1.3B and mBART-fr skipped — performance gap is marginal,
               storage cost on HPC is high)
  - Condition: RANDOM-10K_STRUCTURED-4K only (paper's strongest mixed model, BLEU ~23 combined)
  - Seeds:     seed 42 only (1 best seed; reproduces fine, fast iteration)
  - Direction: BOTH (forward fr->aj AND reverse aj->fr)
  - Total:     2 jobs, ~5 GB storage on HPC

Output schema (extends generate_jobs_tsv.py with one extra column for direction):
    job_id  experiment  condition  seed  model_key  results_subdir  similar_lang  direction

Usage:
    python generate_jobs_tsv_pipeline.py [--output jobs_pipeline.tsv]

Submit on HPC:
    sbatch --array=1-2 submit_pipeline.sbatch
"""

import argparse

SEEDS = [42]                                         # single best seed
EXPERIMENT = "exp1"
CONDITION  = "RANDOM-10K_STRUCTURED-4K"
DIRECTIONS = ["forward", "reverse"]

# (model_key, results_subdir, similar_lang)
MODELS = [
    ("nllb-600m", "nllb-600m", "ewe_Latn"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="jobs_pipeline.tsv")
    args = ap.parse_args()

    rows: list[str] = []
    job_id = 0
    for model_key, results_subdir, similar_lang in MODELS:
        for direction in DIRECTIONS:
            for seed in SEEDS:
                job_id += 1
                rows.append(
                    f"{job_id}\t{EXPERIMENT}\t{CONDITION}\t{seed}\t"
                    f"{model_key}\t{results_subdir}\t{similar_lang}\t{direction}"
                )

    with open(args.output, "w") as f:
        for r in rows:
            f.write(r + "\n")

    print(f"Generated {args.output} with {len(rows)} jobs")
    print()
    print("Layout:")
    for i, r in enumerate(rows, start=1):
        cols = r.split("\t")
        print(f"  line {i:>2}: {cols[4]:<11} {cols[2]:<28} seed{cols[3]:<5} direction={cols[7]}")
    print()
    print("Submit on HPC:")
    print("  sbatch --array=1-2 submit_pipeline.sbatch       # both directions, ~5 GB")
    print("  sbatch --array=1   submit_pipeline.sbatch       # forward only (smoke test)")
    print("  sbatch --array=2   submit_pipeline.sbatch       # reverse only (smoke test)")


if __name__ == "__main__":
    main()
