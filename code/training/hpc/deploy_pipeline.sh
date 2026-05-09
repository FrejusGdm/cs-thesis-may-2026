#!/usr/bin/env bash
# deploy_pipeline.sh — push pipeline-checkpoint artifacts to Discovery via scp +
# SSH ControlMaster. Run from the repo root.
#
# What it pushes (always):
#   - jobs_pipeline.tsv
#   - submit_pipeline.sbatch
#   - hf_job_train_hpc.py (UPDATED with --reverse flag — must overwrite for the
#     pipeline jobs to pick up the patched logic)
#
# Idempotent: scp will overwrite per file. Re-running is safe.
#
# Prerequisites (run locally first):
#   1. python experiments/training/hpc/generate_jobs_tsv_pipeline.py \
#        --output experiments/training/hpc/jobs_pipeline.tsv

set -uo pipefail

REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HPC_DIR="${REPO_ROOT}/experiments/training/hpc"

JOBS_TSV="${HPC_DIR}/jobs_pipeline.tsv"
SBATCH="${HPC_DIR}/submit_pipeline.sbatch"
TRAIN_PY="${HPC_DIR}/hf_job_train_hpc.py"

for f in "$JOBS_TSV" "$SBATCH" "$TRAIN_PY"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 1
    fi
done

CM_SOCKET="/tmp/ssh-cm-deploy-pipeline-$$"
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

FAILED=()
run_scp() {
    local label="$1"; shift
    echo "  $label"
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    echo "    RETRY..."
    sleep 2
    if scp "${SSH_OPTS[@]}" -q "$@"; then return 0; fi
    FAILED+=("$label")
    return 1
}

echo "=== Open SSH master connection (single password prompt) ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "mkdir -p ${HPC_ROOT}/results/pipeline ${HPC_ROOT}/logs"

echo
echo "=== Push job TSV + sbatch + UPDATED training script (reverse flag) ==="
run_scp "jobs_pipeline.tsv"     "$JOBS_TSV"  "${REMOTE}:${HPC_ROOT}/jobs_pipeline.tsv"
run_scp "submit_pipeline.sbatch" "$SBATCH"   "${REMOTE}:${HPC_ROOT}/submit_pipeline.sbatch"
run_scp "hf_job_train_hpc.py"   "$TRAIN_PY"  "${REMOTE}:${HPC_ROOT}/hf_job_train_hpc.py"

echo
echo "=== Verify remote layout ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    cd ${HPC_ROOT}
    echo '--- new files at top level ---'
    ls -la jobs_pipeline.tsv submit_pipeline.sbatch hf_job_train_hpc.py 2>&1
    echo
    echo '--- jobs_pipeline.tsv contents (should be 2 lines) ---'
    cat jobs_pipeline.tsv
    echo
    echo '--- results/pipeline (should be empty) ---'
    ls -la results/pipeline/ 2>/dev/null | head
    echo
    echo '--- HPC space check ---'
    df -h ${HPC_ROOT} 2>/dev/null | head -3 || true
    du -sh ${HPC_ROOT}/results 2>/dev/null || true
"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "=== ${#FAILED[@]} transfer(s) FAILED ==="
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

cat <<EOF

=== Done. Next: ssh in and submit ===

  ssh ${REMOTE}
  cd ${HPC_ROOT}

  # Smoke test: forward direction only (1 job)
  sbatch --array=1 submit_pipeline.sbatch

  # Both directions (2 jobs, ~5 GB checkpoints when finished)
  sbatch --array=1-2 submit_pipeline.sbatch

When done, fetch checkpoints + metrics back:

  bash experiments/training/hpc/fetch_results.sh pipeline   # extend fetch_results.sh first

Then upload to Hugging Face:

  python experiments/training/upload_checkpoints_to_hf.py \\
    --root experiments/results/pipeline \\
    --hf-username JosueG
EOF
