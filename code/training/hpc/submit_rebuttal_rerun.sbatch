#!/bin/bash
#SBATCH -J nmt-rebuttal
#SBATCH -o <HPC_WORKDIR>
#SBATCH -e <HPC_WORKDIR>
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-30
#
# Rebuttal subset-BLEU rerun — paper-era exp1 conditions on the paper test set,
# using the current training script that saves per-sentence JSONL predictions.
# Output isolated under results/rebuttal_rerun/ so the historical results in
# results/ and results/april2026/ are untouched.
#
# Tier ranges (see generate_jobs_tsv_rebuttal.py header for details):
#   --array=1-20   = Tier 1: NLLB-600M × {STRUCT-2K, RAND-10K, R10K+S4K, S4K-ONLY} × 5 seeds
#   --array=1-30   = Tier 1+2: NLLB-600M × all 6 exp1 conditions × 5 seeds  ← default
#   --array=1-60   = +Tier 3: mBART-fr × all 6 conditions × 5 seeds
#   --array=1-90   = +Tier 4: NLLB-1.3B × all 6 conditions × 5 seeds (best-effort)
#
# Smoke test:
#   --array=1,31,61
#
# Already-completed jobs are skipped automatically (test_metrics.json check).
#
# Prerequisites (run locally first):
#   1. python experiments/training/hpc/generate_jobs_tsv_rebuttal.py \
#        --output experiments/training/hpc/jobs_rebuttal.tsv
#   2. bash experiments/training/hpc/deploy_rebuttal_rerun.sh

set -euo pipefail

# ── Paths ──
HPC_DIR="<HPC_WORKDIR>"
JOBS_FILE="${HPC_DIR}/jobs_rebuttal.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
# Isolate rebuttal-rerun results from paper-era and april-2026 results.
RESULTS_DIR="${HPC_DIR}/results/rebuttal_rerun"

# ── Read job params ──
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
if [ -z "$LINE" ]; then
    echo "ERROR: No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 1
fi

JOB_ID=$(echo "$LINE" | cut -f1)
EXPERIMENT=$(echo "$LINE" | cut -f2)
CONDITION=$(echo "$LINE" | cut -f3)
SEED=$(echo "$LINE" | cut -f4)
MODEL_KEY=$(echo "$LINE" | cut -f5)
RESULTS_SUBDIR=$(echo "$LINE" | cut -f6)
SIMILAR_LANG=$(echo "$LINE" | cut -f7)

# Map model_key to local model path
case "$MODEL_KEY" in
    nllb-1.3b)     MODEL_PATH="${MODELS_DIR}/nllb-200-1.3B" ;;
    nllb-600m)     MODEL_PATH="${MODELS_DIR}/nllb-200-distilled-600M" ;;
    mbart-50)      MODEL_PATH="${MODELS_DIR}/mbart-large-50" ;;
    *)             echo "Unknown model_key: $MODEL_KEY"; exit 1 ;;
esac

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Experiment:       ${EXPERIMENT}"
echo "Condition:        ${CONDITION}"
echo "Seed:             ${SEED}"
echo "Model:            ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Results subdir:   ${RESULTS_SUBDIR}"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "Results root:     ${RESULTS_DIR}"
echo "========================================"

# ── Skip if results already exist (in the rebuttal_rerun tree) ──
RESULT_FILE="${RESULTS_DIR}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run training in container ──
# Note: --no-save-checkpoint to keep storage usage flat.
# Default test path = /data/shared/test.tsv (paper test set) — no override.
apptainer exec --nv \
    --bind "${DATA_DIR}:/data:ro" \
    --bind "${MODEL_PATH}:/model:ro" \
    --bind "${RESULTS_DIR}:/results" \
    --bind "${TRAIN_SCRIPT}:/train.py:ro" \
    "$CONTAINER" \
    python3 /train.py \
    --experiment "$EXPERIMENT" \
    --condition "$CONDITION" \
    --seed "$SEED" \
    --model-path /model \
    --data-dir /data \
    --results-dir "/results/${RESULTS_SUBDIR}" \
    --similar-lang "$SIMILAR_LANG" \
    --no-save-checkpoint

echo "Job ${JOB_ID} completed successfully."
