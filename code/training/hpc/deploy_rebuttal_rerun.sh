#!/usr/bin/env bash
# deploy_rebuttal_rerun.sh — push the ACL-rebuttal subset-BLEU rerun artifacts to
# Discovery via scp + SSH ControlMaster. Run from the repo root.
#
# What it ALWAYS pushes (NEW filenames — can't overwrite anything pre-existing):
#   - jobs_rebuttal.tsv
#   - submit_rebuttal_rerun.sbatch
#
# What it pushes ONLY if --force-train is passed (overwrites):
#   - hf_job_train_hpc.py — the same version is already on HPC from
#     deploy_april2026.sh; re-pushing is idempotent unless you've made local
#     edits since the last april-2026 deploy. Skipping by default avoids any
#     risk of perturbing in-flight april-2026 jobs that haven't started yet.
#
# What it does NOT push:
#   - data/exp1/* — already on HPC from the original 932-job run
#   - data/shared/test.tsv — already on HPC (paper test set, unchanged)
#   - container, models — already on HPC
#
# Idempotent: scp will overwrite per file. Re-running is safe.
#
# Usage:
#   bash experiments/training/hpc/deploy_rebuttal_rerun.sh
#   bash experiments/training/hpc/deploy_rebuttal_rerun.sh --force-train
#
# Prerequisites (run locally first):
#   1. python experiments/training/hpc/generate_jobs_tsv_rebuttal.py \
#        --output experiments/training/hpc/jobs_rebuttal.tsv

set -uo pipefail

FORCE_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --force-train) FORCE_TRAIN=1 ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

REMOTE="f006g5b@discovery.dartmouth.edu"
HPC_ROOT="<HPC_WORKDIR>"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HPC_DIR="${REPO_ROOT}/experiments/training/hpc"

JOBS_TSV="${HPC_DIR}/jobs_rebuttal.tsv"
SBATCH="${HPC_DIR}/submit_rebuttal_rerun.sbatch"
TRAIN_PY="${HPC_DIR}/hf_job_train_hpc.py"

for f in "$JOBS_TSV" "$SBATCH" "$TRAIN_PY"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — see prerequisites in script header." >&2
        exit 1
    fi
done

# --- SSH ControlMaster: one connection for all transfers ---
CM_SOCKET="/tmp/ssh-cm-deploy-rebuttal-$$"
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
    "mkdir -p ${HPC_ROOT}/results/rebuttal_rerun"

echo
echo "=== Push job TSV + sbatch (NEW files, can't overwrite anything) ==="
run_scp "jobs_rebuttal.tsv"          "$JOBS_TSV"  "${REMOTE}:${HPC_ROOT}/jobs_rebuttal.tsv"
run_scp "submit_rebuttal_rerun.sbatch" "$SBATCH"  "${REMOTE}:${HPC_ROOT}/submit_rebuttal_rerun.sbatch"

if [ "$FORCE_TRAIN" -eq 1 ]; then
    echo
    echo "=== Push hf_job_train_hpc.py (--force-train: OVERWRITES existing copy) ==="
    run_scp "hf_job_train_hpc.py"    "$TRAIN_PY"  "${REMOTE}:${HPC_ROOT}/hf_job_train_hpc.py"
else
    echo
    echo "=== Skipping hf_job_train_hpc.py (use --force-train to push it) ==="
    echo "    The version on HPC from deploy_april2026.sh is already current."
fi

echo
echo "=== Verify remote layout (sanity check) ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    cd ${HPC_ROOT}
    echo '--- new files at top level ---'
    ls -la jobs_rebuttal.tsv submit_rebuttal_rerun.sbatch hf_job_train_hpc.py 2>&1
    echo
    echo '--- results/rebuttal_rerun (should be empty so far) ---'
    ls -la results/rebuttal_rerun/ 2>/dev/null | head
    echo
    echo '--- data/exp1 conditions (must already exist) ---'
    ls -1 data/exp1 2>/dev/null
    echo
    echo '--- data/shared/test.tsv (paper test set) ---'
    ls -la data/shared/test.tsv 2>/dev/null
    wc -l data/shared/test.tsv 2>/dev/null
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

  # Smoke test (3 jobs: NLLB-600M + mBART-fr + NLLB-1.3B, STRUCT-2K, seed 42)
  sbatch --array=1,31,61 submit_rebuttal_rerun.sbatch

  # Tier 1 (NLLB-600M, 4 headline conditions, 5 seeds = 20 jobs)
  sbatch --array=1-20 submit_rebuttal_rerun.sbatch

  # Tier 1+2 (NLLB-600M, all 6 exp1 conditions, 5 seeds = 30 jobs)
  sbatch --array=1-30 submit_rebuttal_rerun.sbatch

  # + Tier 3 (mBART-fr × exp1 × 5 seeds, +30 jobs = 60 total)
  sbatch --array=1-60 submit_rebuttal_rerun.sbatch

  # + Tier 4 (NLLB-1.3B × exp1 × 5 seeds, +30 jobs = 90 total — best-effort)
  sbatch --array=1-90 submit_rebuttal_rerun.sbatch

Once jobs land, sync results back:

  bash experiments/training/hpc/fetch_results.sh rebuttal_rerun

(You may need to add a 'rebuttal_rerun' case to fetch_results.sh's scope dispatch
if it doesn't already accept it — see the case statement near the top of that script.)

Then aggregate:

  python experiments/analysis/recompute_subset_metrics.py \\
    --glob "experiments/results/rebuttal_rerun/**/test_predictions.jsonl" \\
    --output experiments/results/summary/rebuttal_subset_bleu.csv
EOF
