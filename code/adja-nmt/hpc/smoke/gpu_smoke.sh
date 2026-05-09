#!/usr/bin/env bash
# Interactive GPU smoke: runs every HPC training script in --smoke mode
# (4 samples, 2 steps, Stage A only, no checkpoint save). Validates the whole
# pipe (data load → model → forward → backward → optimizer step) on a real GPU.
# Last updated: 2026-04-22
#
# Usage (inside an interactive srun allocation):
#   srun --pty --partition=gpuq \
#        --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 \
#        --mem=64G --time=30:00 bash
#   cd <HPC_WORKDIR>
#   export HF_TOKEN=hf_xxx
#   bash smoke/gpu_smoke.sh
#
# Total runtime ~10-15 minutes for all 4 models.

set -uo pipefail  # do NOT set -e — we want to run all 4 even if one fails

HPC_DIR="<HPC_WORKDIR>"
CONTAINER="${HPC_DIR}/speech-training.sif"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: ${CONTAINER} not found."; exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."; exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no GPU visible. Are you inside an srun allocation?"
    exit 1
fi

echo "GPU on this node:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

FAILED=()

run_smoke() {
    local label="$1"
    local script="$2"
    shift 2
    echo "============================================================"
    echo "SMOKE: ${label}"
    echo "Script: ${script}"
    echo "============================================================"
    if apptainer exec --nv \
        --bind "${HPC_DIR}/models:/models:ro" \
        --bind "${HPC_DIR}/data:/data" \
        --bind "${HPC_DIR}/results:/results" \
        --bind "${HPC_DIR}/scripts:/scripts:ro" \
        --env PYTHONUNBUFFERED=1 \
        --env "HF_TOKEN=${HF_TOKEN}" \
        --env HF_HOME=/data/hf_cache \
        --env HF_DATASETS_CACHE=/data/hf_cache/datasets \
        --env TRANSFORMERS_CACHE=/data/hf_cache/transformers \
        "${CONTAINER}" \
        python3 "${script}" \
            --models-dir /models \
            --data-dir /data \
            --output-dir "/results/smoke_${label}" \
            --smoke "$@"
    then
        echo "✅ ${label} PASSED"
    else
        echo "❌ ${label} FAILED"
        FAILED+=("${label}")
    fi
    echo ""
}

# 1. wav2vec2 SSL — XLS-R 300M (baseline, lightest)
run_smoke "xlsr-300m-ssl" \
    "/scripts/pretraining/wav2vec2_ssl_ewe.py" \
    --base facebook/wav2vec2-xls-r-300m --model-key xlsr-300m

# 2. audio-LM CSM (Mimi)
run_smoke "audiolm-csm" "/scripts/pretraining/audio_lm_csm_ewe.py"

# 3. audio-LM Orpheus (SNAC, 3B — biggest memory hit)
run_smoke "audiolm-orpheus" "/scripts/pretraining/audio_lm_orpheus_ewe.py"

# 4. Whisper large-v3 (1.5B — other big memory hit)
run_smoke "whisper-largev3" "/scripts/large/whisper_largev3_ewe_hpc.py"

echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL 4 SMOKES PASSED ✅"
    echo "Ready to sbatch real jobs."
    exit 0
else
    echo "FAILED: ${FAILED[*]} ❌"
    echo "Do NOT sbatch real jobs until these pass. Check logs above."
    exit 1
fi
