#!/bin/bash
# build_container.sh — Build NMT training container on HPC.
# Pulls base image with retries to handle flaky HPC networks, then builds locally.
set -euo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$HPC_DIR"

BASE_SIF="pytorch-base.sif"
FINAL_SIF="nmt-training.sif"
MAX_RETRIES=5
RETRY_DELAY=30

# Step 1: Pull base image with retries
if [ -f "$BASE_SIF" ]; then
    echo "Base image already exists: $BASE_SIF (skipping pull)"
else
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "[Attempt $attempt/$MAX_RETRIES] Pulling pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime..."
        if apptainer pull "$BASE_SIF" docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime; then
            echo "Pull succeeded."
            break
        fi
        rm -f "$BASE_SIF"  # clean up partial file
        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo "ERROR: Failed after $MAX_RETRIES attempts."
            exit 1
        fi
        echo "Failed. Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
    done
fi

# Step 2: Build final container from local base (fast, no network needed)
echo "Building ${FINAL_SIF}..."
apptainer build --fakeroot "$FINAL_SIF" nmt-training.def
echo "Done: $(ls -lh "$FINAL_SIF" | awk '{print $5}') — $FINAL_SIF"
