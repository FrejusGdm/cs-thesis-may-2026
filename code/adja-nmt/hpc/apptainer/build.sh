#!/usr/bin/env bash
# build.sh — build speech-training.sif on Dartmouth Discovery login node.
# Last updated: 2026-04-21
#
# Run this on the login node after first deploy:
#   ssh f006g5b@discovery.dartmouth.edu
#   cd <HPC_WORKDIR>
#   bash apptainer/build.sh

set -euo pipefail
HPC_DIR="<HPC_WORKDIR>"
DEF="${HPC_DIR}/apptainer/speech-training.def"
SIF="${HPC_DIR}/speech-training.sif"
BASE="${HPC_DIR}/pytorch-base.sif"

if [ ! -f "$BASE" ]; then
    echo "ERROR: pytorch-base.sif missing at ${BASE}."
    echo "Copy it from <HPC_WORKDIR> or rebuild."
    exit 1
fi

cd "${HPC_DIR}"
apptainer build --fakeroot "${SIF}" "${DEF}"
echo "Built ${SIF}"
