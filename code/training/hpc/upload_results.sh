#!/bin/bash
# upload_results.sh — Upload HPC results to HuggingFace Hub repos.
#
# Run this AFTER all SLURM jobs complete to push results for collection.
# Uploads are routed through the Apptainer container (which has huggingface-hub installed).
# Requires: nmt-training.sif built first, HF_TOKEN env var set.
#
# Usage:
#   bash upload_results.sh [results_dir]

set -euo pipefail

HF_USER="JosueG"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${HPC_DIR}/nmt-training.sif"
RESULTS_DIR="${1:-${HPC_DIR}/results}"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "Build it first: apptainer build --fakeroot nmt-training.sif nmt-training.def"
    exit 1
fi

# Map result subdirectory to HF repo
declare -A REPOS=(
    ["nllb-1.3b"]="${HF_USER}/adja-nmt-results-1.3b"
    ["mbart-fr"]="${HF_USER}/adja-nmt-results-mbart-fr"
    ["mbart-rand"]="${HF_USER}/adja-nmt-results-mbart-random"
)

echo "Uploading results from: ${RESULTS_DIR}"
echo "Using container: ${CONTAINER}"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

for subdir in "${!REPOS[@]}"; do
    REPO="${REPOS[$subdir]}"
    LOCAL_PATH="${RESULTS_DIR}/${subdir}"

    if [ ! -d "$LOCAL_PATH" ]; then
        echo "[SKIP] ${subdir}: No directory at ${LOCAL_PATH}"
        continue
    fi

    # Count result files
    N_FILES=$(find "$LOCAL_PATH" -name "test_metrics.json" | wc -l)
    echo "=== ${subdir} -> ${REPO} (${N_FILES} result files) ==="

    if [ "$N_FILES" -eq 0 ]; then
        echo "  No results to upload."
        continue
    fi

    # Create repo (if needed) and upload results via container
    apptainer exec --bind "${RESULTS_DIR}:/results" "$CONTAINER" \
        python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${REPO}', repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(folder_path='/results/${subdir}', repo_id='${REPO}', repo_type='dataset')
"

    echo "  Uploaded ${N_FILES} files to ${REPO}"
    echo ""
done

echo "Upload complete."
echo "Collect results with: python experiments/evaluation/collect_hf_results.py"
