#!/usr/bin/env bash
# deploy_to_discovery.sh — scp adja-nmt/hpc/ to Dartmouth Discovery.
# Last updated: 2026-04-21
#
# Uses scp + SSH ControlMaster — single password prompt. Do NOT use rsync on
# macOS 15+ (openrsync segfaults against GNU rsync on the server).
#
# Usage:
#   bash hpc/deploy/deploy_to_discovery.sh
#
# Idempotent: scp overwrites per file; re-running is safe.

set -uo pipefail
REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"

# repo root resolved relative to this script
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_HPC="${REPO_ROOT}/hpc"

if [ ! -d "${LOCAL_HPC}" ]; then
    echo "ERROR: local hpc/ dir missing at ${LOCAL_HPC}"
    exit 1
fi

# --- SSH ControlMaster: one connection for all transfers ---
CM_SOCKET="/tmp/ssh-cm-adja-speech-$$"
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
    "mkdir -p ${HPC_ROOT}/{slurm,apptainer,deploy,jobs,scripts/pretraining,scripts/large,smoke,logs,models,data,results}"

FAILED=()
run_scp() {
    local label="$1"; shift
    echo "  $label"
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    echo "    RETRY..."; sleep 2
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    FAILED+=("$label"); return 1
}

echo "=== Syncing SLURM sbatch files ==="
for f in "${LOCAL_HPC}"/slurm/*.sbatch; do
    [ -f "$f" ] || continue
    run_scp "slurm/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/slurm/"
done

echo "=== Syncing Apptainer recipe + build script ==="
for f in "${LOCAL_HPC}"/apptainer/*; do
    [ -f "$f" ] || continue
    run_scp "apptainer/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/apptainer/"
done

echo "=== Syncing deploy scripts ==="
for f in "${LOCAL_HPC}"/deploy/*.sh; do
    [ -f "$f" ] || continue
    run_scp "deploy/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/deploy/"
done

echo "=== Syncing smoke tests ==="
for f in "${LOCAL_HPC}"/smoke/*; do
    # Ignore directories like __pycache__ which scp refuses without -r.
    [ -f "$f" ] || continue
    run_scp "smoke/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/smoke/"
done

echo "=== Syncing jobs TSVs ==="
for f in "${LOCAL_HPC}"/jobs/*.tsv; do
    [ -f "$f" ] || continue
    run_scp "jobs/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/jobs/"
done

echo "=== Syncing Python scripts ==="
for f in "${LOCAL_HPC}"/scripts/*.py; do
    [ -f "$f" ] || continue
    run_scp "scripts/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/scripts/"
done
for f in "${LOCAL_HPC}"/scripts/pretraining/*.py; do
    [ -f "$f" ] || continue
    run_scp "scripts/pretraining/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/scripts/pretraining/"
done
for f in "${LOCAL_HPC}"/scripts/large/*.py; do
    [ -f "$f" ] || continue
    run_scp "scripts/large/$(basename "$f")" "$f" "${REMOTE}:${HPC_ROOT}/scripts/large/"
done

echo "=== Syncing README ==="
if [ -f "${LOCAL_HPC}/README.md" ]; then
    run_scp "README.md" "${LOCAL_HPC}/README.md" "${REMOTE}:${HPC_ROOT}/"
fi

echo "=== chmod +x shell scripts ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "chmod +x ${HPC_ROOT}/deploy/*.sh ${HPC_ROOT}/apptainer/build.sh ${HPC_ROOT}/smoke/*.sh 2>/dev/null || true"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Deploy complete. Next: ssh ${REMOTE} → cd ${HPC_ROOT} → bash apptainer/build.sh"
