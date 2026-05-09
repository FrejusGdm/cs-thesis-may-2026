#!/usr/bin/env bash
# fetch_results.sh — pull test_metrics.json + test_predictions.jsonl back
# to the laptop. Checkpoints are NOT fetched (they're on HF Hub at
# JosueG/adja-mt-best).
#
# Usage:
#   bash experiments/mt/retraining-nllb-for-weights/fetch_results.sh

set -uo pipefail
REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_RESULTS="<HPC_WORKDIR>"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCAL_DIR="${REPO_ROOT}/results/mt/bidir-paper-repro"
mkdir -p "${LOCAL_DIR}"

echo "Pulling test_metrics.json + test_predictions.jsonl from ${HPC_RESULTS}..."

# scp -r the directory but only metrics + predictions, not checkpoints.
# Easier: rsync with --include works on Linux server side via ssh tar.
ssh "$REMOTE" "cd ${HPC_RESULTS} && tar c \
        --exclude='checkpoint' \
        --exclude='hf_cache' \
        ." | tar xv -C "${LOCAL_DIR}"

echo "Done. Local: ${LOCAL_DIR}"
