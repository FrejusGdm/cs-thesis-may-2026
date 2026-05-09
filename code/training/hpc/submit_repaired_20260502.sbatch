#!/bin/bash
#SBATCH -J nmt-repaired
#SBATCH -o <HPC_WORKDIR>
#SBATCH -e <HPC_WORKDIR>
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-934
#
# Array ranges are generated in jobs_repaired_20260502.ranges.sh:
#   TIER1_NLLB600_PAPER
#   TIER2_PIPELINE
#   TIER3_MBART_FORWARD
#   TIER4_NLLB13_FORWARD
#   TIER5_MBART_REVERSE_NICE_TO_HAVE

set -euo pipefail

HPC_DIR="<HPC_WORKDIR>"
JOBS_FILE="${HPC_DIR}/jobs_repaired_20260502.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data_repaired_20260502"
MODELS_DIR="${HPC_DIR}/models"
REBUTTAL_RESULTS_DIR="${HPC_DIR}/results/rebuttal_repaired_20260502"
PIPELINE_RESULTS_DIR="${HPC_DIR}/results/pipeline_repaired_20260502"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
if [ -z "$LINE" ]; then
    echo "ERROR: No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 1
fi

JOB_ID=$(echo "$LINE" | cut -f1)
TIER=$(echo "$LINE" | cut -f2)
EXPERIMENT=$(echo "$LINE" | cut -f3)
CONDITION=$(echo "$LINE" | cut -f4)
SEED=$(echo "$LINE" | cut -f5)
MODEL_KEY=$(echo "$LINE" | cut -f6)
RESULTS_SUBDIR=$(echo "$LINE" | cut -f7)
SIMILAR_LANG=$(echo "$LINE" | cut -f8)
DIRECTION=$(echo "$LINE" | cut -f9)
SAVE_CHECKPOINT=$(echo "$LINE" | cut -f10)

case "$MODEL_KEY" in
    nllb-1.3b) MODEL_PATH="${MODELS_DIR}/nllb-200-1.3B" ;;
    nllb-600m) MODEL_PATH="${MODELS_DIR}/nllb-200-distilled-600M" ;;
    mbart-50)  MODEL_PATH="${MODELS_DIR}/mbart-large-50" ;;
    *)         echo "Unknown model_key: $MODEL_KEY"; exit 1 ;;
esac

if [ "$SAVE_CHECKPOINT" = "1" ] || [ "$TIER" = "TIER2_PIPELINE" ]; then
    RESULTS_DIR="$PIPELINE_RESULTS_DIR"
else
    RESULTS_DIR="$REBUTTAL_RESULTS_DIR"
fi

EXTRA_ARGS=()
if [ "$SAVE_CHECKPOINT" != "1" ]; then
    EXTRA_ARGS+=(--no-save-checkpoint)
fi
case "$DIRECTION" in
    forward) ;;
    reverse) EXTRA_ARGS+=(--reverse) ;;
    *)       echo "Unknown direction: $DIRECTION"; exit 1 ;;
esac

RESULT_FILE="${RESULTS_DIR}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Tier:             ${TIER}"
echo "Experiment:       ${EXPERIMENT}"
echo "Condition:        ${CONDITION}"
echo "Seed:             ${SEED}"
echo "Model:            ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Results root:     ${RESULTS_DIR}"
echo "Results subdir:   ${RESULTS_SUBDIR}"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "Direction:        ${DIRECTION}"
echo "Save checkpoint:  ${SAVE_CHECKPOINT}"
echo "========================================"

if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

mkdir -p "$(dirname "$RESULT_FILE")"

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
    "${EXTRA_ARGS[@]}"

echo "Job ${JOB_ID} completed successfully."
