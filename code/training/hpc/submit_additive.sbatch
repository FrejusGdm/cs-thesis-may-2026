#!/bin/bash
#SBATCH -J nmt-additive
#SBATCH -o <HPC_WORKDIR>
#SBATCH -e <HPC_WORKDIR>
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-36
# Additive ablations: 3 conditions x 3 seeds x 4 models = 36 jobs
# Conditions: ADD-M1M2, ADD-M1M2M3, ADD-M1M2M3M4
# Data at: data/new_experiments/ablations/additive/{condition}/

set -euo pipefail

# ── Paths ──
HPC_DIR="<HPC_WORKDIR>"
JOBS_FILE="${HPC_DIR}/jobs_additive.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
RESULTS_DIR="${HPC_DIR}/results"

# ── Read job params from jobs_additive.tsv (line $SLURM_ARRAY_TASK_ID) ──
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
echo "========================================"

# ── Skip if results already exist ──
RESULT_FILE="${RESULTS_DIR}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

# ── Create output directory ──
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
    --similar-lang "$SIMILAR_LANG"

echo "Job ${JOB_ID} completed successfully."
