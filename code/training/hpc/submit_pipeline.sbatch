#!/bin/bash
#SBATCH -J nmt-pipeline
#SBATCH -o <HPC_WORKDIR>
#SBATCH -e <HPC_WORKDIR>
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-2
#
# Pipeline-checkpoint training — produce serving-ready NLLB-600M models for the
# downstream pipeline (later uploaded to Hugging Face via upload_checkpoints_to_hf.py).
#
# Differences from submit_rebuttal_rerun.sbatch:
#   - Reads jobs_pipeline.tsv (with extra "direction" column at position 8)
#   - Writes to results/pipeline/{forward|reverse}/<...> (segregated by direction)
#   - Saves checkpoints (NO --no-save-checkpoint flag)
#   - Passes --reverse to the training script for the reverse-direction job
#
# Submit:
#   sbatch --array=1-2 submit_pipeline.sbatch     # both directions
#   sbatch --array=1   submit_pipeline.sbatch     # forward only
#   sbatch --array=2   submit_pipeline.sbatch     # reverse only

set -euo pipefail

# ── Paths ──
HPC_DIR="<HPC_WORKDIR>"
JOBS_FILE="${HPC_DIR}/jobs_pipeline.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
RESULTS_DIR="${HPC_DIR}/results/pipeline"   # isolated tree for checkpoints

# ── Read job params (now 8 columns: extra "direction" at the end) ──
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
DIRECTION=$(echo "$LINE" | cut -f8)

case "$MODEL_KEY" in
    nllb-1.3b)     MODEL_PATH="${MODELS_DIR}/nllb-200-1.3B" ;;
    nllb-600m)     MODEL_PATH="${MODELS_DIR}/nllb-200-distilled-600M" ;;
    mbart-50)      MODEL_PATH="${MODELS_DIR}/mbart-large-50" ;;
    *)             echo "Unknown model_key: $MODEL_KEY"; exit 1 ;;
esac

# Reverse flag passed to the training script
REVERSE_FLAG=""
if [ "$DIRECTION" = "reverse" ]; then
    REVERSE_FLAG="--reverse"
fi

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Direction:        ${DIRECTION}"
echo "Experiment:       ${EXPERIMENT}"
echo "Condition:        ${CONDITION}"
echo "Seed:             ${SEED}"
echo "Model:            ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Results subdir:   ${RESULTS_SUBDIR} (under ${DIRECTION}/)"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "Reverse flag:     ${REVERSE_FLAG:-<none>}"
echo "========================================"

# ── Skip if results already exist for this direction × condition × seed ──
RESULT_FILE="${RESULTS_DIR}/${DIRECTION}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run training in container — note: NO --no-save-checkpoint, so default save happens ──
apptainer exec --nv \
    --bind "${DATA_DIR}:/data:ro" \
    --bind "${MODEL_PATH}:/model:ro" \
    --bind "${RESULTS_DIR}/${DIRECTION}:/results" \
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
    ${REVERSE_FLAG}

echo "Job ${JOB_ID} (${DIRECTION}) completed successfully."
