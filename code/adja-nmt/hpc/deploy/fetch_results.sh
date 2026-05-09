#!/usr/bin/env bash
# fetch_results.sh — pull metrics.json + generated audio back from Discovery.
# Last updated: 2026-04-21
#
# Usage:
#   bash hpc/deploy/fetch_results.sh
# Local destination: adja-nmt/results/hpc/

set -uo pipefail
REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DEST="${REPO_ROOT}/results/hpc"
mkdir -p "$LOCAL_DEST"

CM_SOCKET="/tmp/ssh-cm-adja-speech-fetch-$$"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=${CM_SOCKET}" -o "ControlPersist=600")
cleanup() {
    ssh "${SSH_OPTS[@]}" -O exit "$REMOTE" 2>/dev/null || true
    rm -f "$CM_SOCKET"
}
trap cleanup EXIT

echo "=== Listing remote results subdirs ==="
REMOTE_SUBDIRS=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "ls -1 ${HPC_ROOT}/results/ 2>/dev/null || true")
if [ -z "$REMOTE_SUBDIRS" ]; then
    echo "No results yet on cluster."
    exit 0
fi

for subdir in $REMOTE_SUBDIRS; do
    echo "=== ${subdir} ==="
    mkdir -p "${LOCAL_DEST}/${subdir}"
    # Pull metrics + generated audio only; skip raw checkpoints (multi-GB).
    scp "${SSH_OPTS[@]}" -q "${REMOTE}:${HPC_ROOT}/results/${subdir}/metrics.json" \
        "${LOCAL_DEST}/${subdir}/" 2>/dev/null || echo "  (no metrics.json)"
    scp "${SSH_OPTS[@]}" -q -r "${REMOTE}:${HPC_ROOT}/results/${subdir}/generated" \
        "${LOCAL_DEST}/${subdir}/" 2>/dev/null || echo "  (no generated/)"
done

echo "Fetched to ${LOCAL_DEST}"
