#!/bin/bash
# T1: Full Sesame CSM fine-tuning pipeline on HPC.
#
# Runs all stages:
#   1. Data preparation (resample 48kHz -> 24kHz)
#   2. Fine-tuning with Unsloth LoRA
#   3. Generate sample audio for evaluation
#
# This script is called by SLURM — all paths are absolute (no $(dirname "$0")).
#
# Usage (called by SLURM, not directly):
#   This runs inside the tts_unsloth.sif container via apptainer exec.

set -euo pipefail

# --- Configuration (overridable via environment) ---
BASE_DIR="${BASE_DIR:-<HPC_WORKDIR>"
DATA_DIR="${BASE_DIR}/data"
EXP_DIR="${BASE_DIR}/experiments/tts/T1_sesame_csm_finetune"
OUTPUT_DIR="${BASE_DIR}/results/T1/seed42"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"
MODEL_CACHE="${BASE_DIR}/models/csm-1b"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "T1: Sesame CSM (1B) Fine-tuning Pipeline"
echo "============================================"
echo "Base dir: ${BASE_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Dry run: ${DRY_RUN}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python3 --version)"
echo "============================================"
echo ""

# ===== STAGE 1: Data Preparation =====
echo "===== STAGE 1: Data Preparation ====="
echo "Resampling audio from 48kHz -> 24kHz..."

if [ -d "${PREPARED_DATA_DIR}/hf_dataset" ]; then
    echo "  Prepared data already exists, skipping."
else
    DRY_RUN_FLAG=""
    if [ "${DRY_RUN}" = "true" ]; then
        DRY_RUN_FLAG="--dry-run"
    fi

    python3 "${EXP_DIR}/data_prep.py" \
        --source local \
        --data-dir "${DATA_DIR}" \
        --output-dir "${PREPARED_DATA_DIR}" \
        ${DRY_RUN_FLAG}
fi
echo ""

# ===== STAGE 2: Fine-tuning =====
echo "===== STAGE 2: Fine-tuning with Unsloth ====="

# Check if model weights are pre-downloaded
if [ -d "${MODEL_CACHE}" ]; then
    echo "  Using pre-downloaded model from ${MODEL_CACHE}"
    # Symlink so transformers finds it in HF cache
    export TRANSFORMERS_CACHE="${BASE_DIR}/models"
    export HF_HOME="${BASE_DIR}/models"
fi

TRAIN_DRY_RUN_FLAG=""
if [ "${DRY_RUN}" = "true" ]; then
    TRAIN_DRY_RUN_FLAG="--dry-run"
fi

python3 "${EXP_DIR}/train.py" \
    --config "${EXP_DIR}/conf/config.yaml" \
    --data-dir "${PREPARED_DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}/training" \
    ${TRAIN_DRY_RUN_FLAG}

echo ""

# ===== STAGE 3: Generate Sample Audio =====
echo "===== STAGE 3: Generate Sample Audio ====="

ADAPTER_DIR="${OUTPUT_DIR}/training/adapter"
GEN_DIR="${OUTPUT_DIR}/generated"
mkdir -p "${GEN_DIR}"

if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "  ERROR: Adapter not found at ${ADAPTER_DIR}"
    echo "  Training may have failed. Check logs."
    exit 1
fi

# Create test sentences file from test manifest
TEST_MANIFEST="${DATA_DIR}/manifests/test.tsv"
TEST_SENTENCES="${GEN_DIR}/test_sentences.txt"

if [ -f "${TEST_MANIFEST}" ]; then
    # Extract text column (3rd column) from TSV, take first 20 for evaluation
    tail -n +2 "${TEST_MANIFEST}" | head -20 | cut -f3 > "${TEST_SENTENCES}"
    echo "  Extracted $(wc -l < "${TEST_SENTENCES}") test sentences"
else
    # Fallback: a few sample Adja sentences
    cat > "${TEST_SENTENCES}" << 'SENTENCES'
Tom trɔ yi kpanŋkɔ wezexu
Ele lɔ awu ehoci lɔ sa
Ɛ yi gbɛ̀
Mì ɖo alɔ ji
Nye ŋkɔ nyé Tom
SENTENCES
    echo "  Using fallback test sentences"
fi

python3 "${EXP_DIR}/generate.py" \
    --adapter-dir "${ADAPTER_DIR}" \
    --input-file "${TEST_SENTENCES}" \
    --output-dir "${GEN_DIR}" \
    --max-new-tokens 2048

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Results: ${OUTPUT_DIR}"
echo "  Training: ${OUTPUT_DIR}/training/"
echo "  Generated: ${GEN_DIR}/"
echo "============================================"
