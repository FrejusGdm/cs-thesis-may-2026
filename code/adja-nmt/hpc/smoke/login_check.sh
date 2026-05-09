#!/usr/bin/env bash
# Login-node pre-flight smoke. Runs in speech-training.sif with the SAME bind
# pattern the real sbatch files use, so any bind/auth/cache bug shows up here
# before we spend queue time on it.
# Last updated: 2026-04-22
#
# Usage on the login node:
#   cd <HPC_WORKDIR>
#   export HF_TOKEN=hf_xxx
#   bash smoke/login_check.sh

set -euo pipefail

HPC_DIR="<HPC_WORKDIR>"
CONTAINER="${HPC_DIR}/speech-training.sif"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: ${CONTAINER} not found. Run apptainer/build.sh first."
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. export HF_TOKEN=hf_xxx first."
    exit 1
fi

# Match the real sbatch exactly: /data is rw so FileLock can write.
apptainer exec \
    --bind "${HPC_DIR}/models:/models:ro" \
    --bind "${HPC_DIR}/data:/data" \
    --bind "${HPC_DIR}/smoke:/smoke:ro" \
    --bind "${HPC_DIR}/scripts:/scripts:ro" \
    --env PYTHONUNBUFFERED=1 \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --env HF_HOME=/data/hf_cache \
    --env HF_DATASETS_CACHE=/data/hf_cache/datasets \
    --env TRANSFORMERS_CACHE=/data/hf_cache/transformers \
    "${CONTAINER}" \
    python3 /smoke/login_check.py
