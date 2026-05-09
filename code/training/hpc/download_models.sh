#!/bin/bash
# download_models.sh — Download model weights to local directory on HPC login node.
#
# Run this ONCE on the login node before submitting jobs.
# Downloads are routed through the Apptainer container (which has huggingface-hub installed).
# Requires: nmt-training.sif built first, HF_TOKEN env var set.
#
# Usage:
#   bash download_models.sh [models_dir]

set -euo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${HPC_DIR}/nmt-training.sif"
MODELS_DIR="${1:-${HPC_DIR}/models}"
mkdir -p "$MODELS_DIR"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "Build it first: apptainer build --fakeroot nmt-training.sif nmt-training.def"
    exit 1
fi

echo "Downloading models to: ${MODELS_DIR}"
echo "Using container: ${CONTAINER}"
echo ""

# ── NLLB-200-1.3B (~5GB) ──
echo "=== NLLB-200-1.3B ==="
if [ -d "${MODELS_DIR}/nllb-200-1.3B" ] && [ -f "${MODELS_DIR}/nllb-200-1.3B/config.json" ]; then
    echo "  Already exists, skipping."
else
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/nllb-200-1.3B', local_dir='/models/nllb-200-1.3B')
"
    echo "  Done."
fi
echo ""

# ── NLLB-200-distilled-600M (~1.2GB) ──
echo "=== NLLB-200-distilled-600M ==="
if [ -d "${MODELS_DIR}/nllb-200-distilled-600M" ] && [ -f "${MODELS_DIR}/nllb-200-distilled-600M/config.json" ]; then
    echo "  Already exists, skipping."
else
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/nllb-200-distilled-600M', local_dir='/models/nllb-200-distilled-600M')
"
    echo "  Done."
fi
echo ""

# ── mBART-large-50-many-to-many-mmt (~2.4GB) ──
echo "=== mBART-large-50-many-to-many-mmt ==="
if [ -d "${MODELS_DIR}/mbart-large-50" ] && [ -f "${MODELS_DIR}/mbart-large-50/config.json" ]; then
    echo "  Already exists, skipping."
else
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/mbart-large-50-many-to-many-mmt', local_dir='/models/mbart-large-50')
"
    echo "  Done."
fi
echo ""

echo "All models downloaded to ${MODELS_DIR}/"
ls -lh "${MODELS_DIR}/"
