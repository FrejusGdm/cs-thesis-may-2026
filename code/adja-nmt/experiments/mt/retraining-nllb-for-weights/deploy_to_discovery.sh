#!/usr/bin/env bash
# deploy_to_discovery.sh — scp this folder to Dartmouth Discovery.
#
# Mirrors hpc/deploy/deploy_to_discovery.sh in style, but targets the NMT
# HPC root: <HPC_WORKDIR> (separate from the
# audio-exploration root used by the speech experiments).
#
# Uses scp + SSH ControlMaster (one password prompt). Do NOT use rsync on
# macOS 15+ — openrsync segfaults against GNU rsync on the server.
#
# Usage (from anywhere):
#   bash experiments/mt/retraining-nllb-for-weights/deploy_to_discovery.sh

set -uo pipefail
REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

CM_SOCKET="/tmp/ssh-cm-adja-mt-$$"
SSH_OPTS=(
    -o "ControlMaster=auto"
    -o "ControlPath=${CM_SOCKET}"
    -o "ControlPersist=600"
)
cleanup() {
    ssh "${SSH_OPTS[@]}" -O exit "$REMOTE" 2>/dev/null || true
    rm -f "$CM_SOCKET" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Open SSH master connection (one password prompt) ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "mkdir -p ${HPC_ROOT}/{scripts,data/shared,data/bidir-paper-repro,models,results,logs,apptainer}"

FAILED=()
run_scp() {
    local label="$1"; shift
    echo "  $label"
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    echo "    RETRY..."; sleep 2
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    FAILED+=("$label"); return 1
}

echo "=== Syncing training script ==="
run_scp "scripts/train_nllb_hpc.py" "${LOCAL_DIR}/train_nllb_hpc.py" "${REMOTE}:${HPC_ROOT}/scripts/"
run_scp "scripts/prep_data.py"      "${LOCAL_DIR}/prep_data.py"      "${REMOTE}:${HPC_ROOT}/scripts/"
run_scp "scripts/select_and_publish_best.py" \
    "${LOCAL_DIR}/select_and_publish_best.py" "${REMOTE}:${HPC_ROOT}/scripts/"

echo "=== Syncing sbatch + jobs TSV ==="
run_scp "submit_array.sbatch" "${LOCAL_DIR}/submit_array.sbatch" "${REMOTE}:${HPC_ROOT}/"
run_scp "jobs.tsv"            "${LOCAL_DIR}/jobs.tsv"            "${REMOTE}:${HPC_ROOT}/"

echo "=== Syncing README ==="
if [ -f "${LOCAL_DIR}/README.md" ]; then
    run_scp "README.md" "${LOCAL_DIR}/README.md" "${REMOTE}:${HPC_ROOT}/"
fi

echo "=== chmod scripts ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "chmod +x ${HPC_ROOT}/scripts/*.py ${HPC_ROOT}/submit_array.sbatch 2>/dev/null || true"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Deploy complete. On Discovery, do:"
echo "  ssh ${REMOTE}"
echo "  cd ${HPC_ROOT}"
echo "  export HF_TOKEN=<token>"
echo "  python3 scripts/prep_data.py --data-root data       # one-time"
echo "  # If the container/model are not yet present, build/download once."
echo "  sbatch --array=1 --time=00:30:00 submit_array.sbatch  # smoke first"
echo "  sbatch submit_array.sbatch                            # full 1-6 array"
