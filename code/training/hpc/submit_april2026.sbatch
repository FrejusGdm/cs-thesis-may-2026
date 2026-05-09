#!/bin/bash
#SBATCH -J nmt-april2026
#SBATCH -o <HPC_WORKDIR>
#SBATCH -e <HPC_WORKDIR>
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-140
# April-2026 new-data sweep — 140 jobs total (80 base + 40 extras + 20 extras_v2).
#
# BASE (lines 1-80 — matches the version originally submitted):
#   --array=1-20    = NLLB-1.3B    (FULL on A/B1/B2 + STRUCT+NEW on A)
#   --array=21-40   = NLLB-600M    (same conditions)
#   --array=41-60   = mBART-50 French-init
#   --array=61-80   = mBART-50 Random-init
#
# EXTRAS (lines 81-120 — added 2026-04-20: STRUCT4K-ALL-BASELINES-PLUS-NEW
# for runB1 and runB2):
#   --array=81-90   = NLLB-1.3B    extras
#   --array=91-100  = NLLB-600M    extras
#   --array=101-110 = mBART-50 French-init extras
#   --array=111-120 = mBART-50 Random-init extras
#   --array=81-120  = all extras
#
# EXTRAS_V2 (lines 121-140 — added 2026-04-21: runB3 FULL — sentence-only
# new6k pool. Clean "more data, same task" comparison vs runB2's mixed
# dict+sentence test):
#   --array=121-125 = NLLB-1.3B    runB3/FULL (5 seeds)
#   --array=126-130 = NLLB-600M    runB3/FULL
#   --array=131-135 = mBART-50 French-init runB3/FULL
#   --array=136-140 = mBART-50 Random-init runB3/FULL
#   --array=121-140 = all runB3
#
# Smoke test (4 jobs, one per model, runA/FULL/seed42):
#   --array=1,21,41,61
#
# Already-completed jobs are skipped automatically — the script checks for
# the per-job test_metrics.json before launching the container, so re-running
# with --array=1-140 after a partial sweep is also safe (just wasteful of
# SLURM allocations vs running only the missing tasks).
#
# Data must be uploaded to HPC under data/april2026_runA/, runB1/, runB2/:
#   rsync -av experiments/data/splits/april2026_runA/ \
#     f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR>
#   (repeat for runB1, runB2)
# jobs_april2026.tsv must also be uploaded to ${HPC_DIR}/jobs_april2026.tsv.

set -euo pipefail

# ── Paths ──
HPC_DIR="<HPC_WORKDIR>"
JOBS_FILE="${HPC_DIR}/jobs_april2026.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
# Isolate april-2026 results from prior paper results.
RESULTS_DIR="${HPC_DIR}/results/april2026"

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

# Per-run test set: each april2026_run* tree has its own test.tsv at the
# experiment-root level (see prepare_april2026_splits.py). Override the
# training script's default {data_dir}/shared/test.tsv path.
TEST_PATH="/data/${EXPERIMENT}/test.tsv"

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Experiment:       ${EXPERIMENT}"
echo "Condition:        ${CONDITION}"
echo "Seed:             ${SEED}"
echo "Model:            ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Results subdir:   ${RESULTS_SUBDIR}"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "Test path:        ${TEST_PATH}"
echo "========================================"

# ── Skip if results already exist ──
RESULT_FILE="${RESULTS_DIR}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run training in container ──
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
    --test-path "$TEST_PATH" \
    --no-save-checkpoint

echo "Job ${JOB_ID} completed successfully."
