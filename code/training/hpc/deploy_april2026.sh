#!/usr/bin/env bash
# deploy_april2026.sh — push the april-2026 sweep artifacts to Discovery via
# scp + SSH ControlMaster. Run from the repo root.
#
# Why scp (not rsync)?
#   macOS 15 (Sequoia) replaced GNU rsync with openrsync, which segfaults
#   against this server's GNU rsync (status 11 / io_read_blocking) regardless
#   of flags. The historical project pattern (rsync -av) worked on macOS ≤14.
#   scp is what the project's table9 deploy doc already uses for individual
#   files, and it works with stock macOS.
#
# SSH ControlMaster reuses one connection across all 6 transfers → single
# password prompt instead of nine.
#
# Prerequisites (run locally first):
#   1. python experiments/data/ingest_april2026.py
#   2. python experiments/data/prepare_april2026_splits.py
#   3. python experiments/training/hpc/generate_jobs_tsv_april2026.py \
#        --output experiments/training/hpc/jobs_april2026.tsv
#
# Idempotent: scp will overwrite per file. Re-running is safe.

set -uo pipefail

REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SPLITS_DIR="${REPO_ROOT}/experiments/data/splits"
HPC_DIR="${REPO_ROOT}/experiments/training/hpc"

JOBS_TSV="${HPC_DIR}/jobs_april2026.tsv"
SBATCH="${HPC_DIR}/submit_april2026.sbatch"
TRAIN_PY="${HPC_DIR}/hf_job_train_hpc.py"

for f in "$JOBS_TSV" "$SBATCH" "$TRAIN_PY"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — see prerequisites in script header." >&2
        exit 1
    fi
done

# --- SSH ControlMaster: one connection for all transfers ---
CM_SOCKET="/tmp/ssh-cm-deploy-april2026-$$"
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
    # run_scp <label> <args...>
    local label="$1"; shift
    echo "  $label"
    if scp "${SSH_OPTS[@]}" -q "$@"; then
        return 0
    fi
    echo "    RETRY..."
    sleep 2
    if scp "${SSH_OPTS[@]}" -q "$@"; then
        return 0
    fi
    FAILED+=("$label")
    return 1
}

echo "=== Open SSH master connection (single password prompt) ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "mkdir -p ${HPC_ROOT}/data/april2026_runA \
              ${HPC_ROOT}/data/april2026_runB1 \
              ${HPC_ROOT}/data/april2026_runB2"

echo
echo "=== Push split trees (recursive) ==="
for run in april2026_runA april2026_runB1 april2026_runB2; do
    src="${SPLITS_DIR}/${run}/"
    if [ ! -d "$src" ]; then
        echo "  WARN: split dir not found: $src — skipping." >&2
        continue
    fi
    # Glob to copy contents into the (already-existing) remote dir.
    # shellcheck disable=SC2086
    run_scp "$run" -r ${src}* "${REMOTE}:${HPC_ROOT}/data/${run}/"
done

echo
echo "=== Push job TSV + sbatch + training script (with --test-path patch) ==="
run_scp "jobs_april2026.tsv"      "$JOBS_TSV" "${REMOTE}:${HPC_ROOT}/jobs_april2026.tsv"
run_scp "submit_april2026.sbatch" "$SBATCH"   "${REMOTE}:${HPC_ROOT}/submit_april2026.sbatch"
run_scp "hf_job_train_hpc.py"     "$TRAIN_PY" "${REMOTE}:${HPC_ROOT}/hf_job_train_hpc.py"

echo
echo "=== Verify remote layout ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    cd ${HPC_ROOT}
    echo '--- data/april2026_runA ---';  ls -1R data/april2026_runA  2>/dev/null | head -20
    echo '--- data/april2026_runB1 ---'; ls -1R data/april2026_runB1 2>/dev/null | head -20
    echo '--- data/april2026_runB2 ---'; ls -1R data/april2026_runB2 2>/dev/null | head -20
    echo '--- top-level new files ---'
    ls -la jobs_april2026.tsv submit_april2026.sbatch hf_job_train_hpc.py 2>&1
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

  # Smoke test (4 jobs: one per model, runA / FULL / seed 42)
  sbatch --array=1,21,41,61 submit_april2026.sbatch

  # Full sweep (80 jobs)
  sbatch submit_april2026.sbatch

  # Subsets:
  #   --array=1-20    NLLB-1.3B
  #   --array=21-40   NLLB-600M
  #   --array=41-60   mBART-50 fr-init
  #   --array=61-80   mBART-50 rand-init

Once jobs land, sync results back. Same scp+ControlMaster pattern works:

  ssh ${REMOTE} "find ${HPC_ROOT}/results/april2026 -name test_metrics.json" \\
    | while read f; do
        rel="\${f#${HPC_ROOT}/results/april2026/}"
        mkdir -p "experiments/results/april2026/\$(dirname "\$rel")"
        scp "${REMOTE}:\$f" "experiments/results/april2026/\$rel"
      done

Then aggregate:

  python experiments/analysis/compare_april2026_vs_acl.py
  python experiments/analysis/compute_robustness_table.py \\
    --results-root experiments/results/april2026 \\
    --skip-paper-reference --include-conditions auto --include-models auto \\
    --experiment april2026_runA \\
    --output experiments/results/summary/april2026_runA_table.csv
EOF
