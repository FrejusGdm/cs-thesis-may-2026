#!/usr/bin/env bash
# deploy_repaired_20260502.sh — deploy May 2 repaired splits and SLURM files.
#
# This script does not touch the historical HPC data/ or results/ roots. It
# deploys repaired splits under:
#   <HPC_WORKDIR>
#
# If that repaired root already exists, it is preserved as:
#   data_repaired_20260502.previous_<UTC timestamp>

set -uo pipefail

REMOTE="${REMOTE:-f006g5b@discovery.dartmouth.edu}"
HPC_ROOT="${REMOTE_ROOT:-<HPC_WORKDIR>"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HPC_DIR="${REPO_ROOT}/experiments/training/hpc"
CLEANED_DIR="${REPO_ROOT}/experiments/data/cleaned"
SPLITS_DIR="${REPO_ROOT}/experiments/data/splits"

DATA_NAME="data_repaired_20260502"
RESULTS_REBUTTAL="results/rebuttal_repaired_20260502"
RESULTS_PIPELINE="results/pipeline_repaired_20260502"

JOBS_TSV="${HPC_DIR}/jobs_repaired_20260502.tsv"
RANGES_SH="${HPC_DIR}/jobs_repaired_20260502.ranges.sh"
SBATCH="${HPC_DIR}/submit_repaired_20260502.sbatch"
TRAIN_PY="${HPC_DIR}/hf_job_train_hpc.py"

DATA_TAR="/tmp/data_repaired_20260502_splits.tar.gz"
META_TAR="/tmp/data_repaired_20260502_metadata.tar.gz"

for f in "$JOBS_TSV" "$RANGES_SH" "$SBATCH" "$TRAIN_PY"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        echo "Run: python experiments/training/hpc/generate_jobs_tsv_repaired_20260502.py --output $JOBS_TSV --ranges-output $RANGES_SH" >&2
        exit 1
    fi
done

for f in \
    "${CLEANED_DIR}/simple-dataset-enriched-clean.csv" \
    "${CLEANED_DIR}/source_target_repair.freeze_manifest.json" \
    "${CLEANED_DIR}/source_target_repair.alternative_references.tsv" \
    "${CLEANED_DIR}/source_target_repair.slash_alternatives.tsv"
do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing repaired metadata file: $f" >&2
        exit 1
    fi
done

if [ ! -d "$SPLITS_DIR/shared" ] || [ ! -d "$SPLITS_DIR/exp1" ]; then
    echo "ERROR: expected regenerated split tree under $SPLITS_DIR" >&2
    exit 1
fi

echo "=== Build local deployment archives ==="
tar -C "$SPLITS_DIR" -czf "$DATA_TAR" .
tar -C "$CLEANED_DIR" -czf "$META_TAR" \
    simple-dataset-enriched-clean.csv \
    source_target_repair.freeze_manifest.json \
    source_target_repair.alternative_references.tsv \
    source_target_repair.slash_alternatives.tsv \
    source_target_repair.accepted_review.tsv \
    source_target_repair.check.json \
    source_target_repair.summary.json
ls -lh "$DATA_TAR" "$META_TAR"

CM_SOCKET="/tmp/ssh-cm-deploy-repaired-$$"
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

echo
echo "=== Open SSH master connection ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    mkdir -p ${HPC_ROOT}/incoming_repaired_20260502 \
             ${HPC_ROOT}/${RESULTS_REBUTTAL} \
             ${HPC_ROOT}/${RESULTS_PIPELINE} \
             ${HPC_ROOT}/metadata/${DATA_NAME} \
             ${HPC_ROOT}/logs
"

echo
echo "=== Upload archives and job files ==="
run_scp "split archive" "$DATA_TAR" "${REMOTE}:${HPC_ROOT}/incoming_repaired_20260502/splits.tar.gz"
run_scp "metadata archive" "$META_TAR" "${REMOTE}:${HPC_ROOT}/incoming_repaired_20260502/metadata.tar.gz"
run_scp "jobs_repaired_20260502.tsv" "$JOBS_TSV" "${REMOTE}:${HPC_ROOT}/jobs_repaired_20260502.tsv"
run_scp "jobs_repaired_20260502.ranges.sh" "$RANGES_SH" "${REMOTE}:${HPC_ROOT}/jobs_repaired_20260502.ranges.sh"
run_scp "submit_repaired_20260502.sbatch" "$SBATCH" "${REMOTE}:${HPC_ROOT}/submit_repaired_20260502.sbatch"
run_scp "hf_job_train_hpc.py" "$TRAIN_PY" "${REMOTE}:${HPC_ROOT}/hf_job_train_hpc.py"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "=== ${#FAILED[@]} transfer(s) FAILED ==="
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

echo
echo "=== Install repaired data tree on HPC ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    set -euo pipefail
    cd ${HPC_ROOT}
    TMP=${DATA_NAME}.tmp
    FINAL=${DATA_NAME}
    META_TMP=metadata/${DATA_NAME}.tmp
    META_FINAL=metadata/${DATA_NAME}
    rm -rf \"\$TMP\"
    mkdir -p \"\$TMP\"
    tar -xzf incoming_repaired_20260502/splits.tar.gz -C \"\$TMP\"
    rm -rf \"\$META_TMP\"
    mkdir -p \"\$META_TMP\"
    tar -xzf incoming_repaired_20260502/metadata.tar.gz -C \"\$META_TMP\"
    if [ -d \"\$FINAL\" ]; then
        BACKUP=\"\${FINAL}.previous_\$(date -u +%Y%m%dT%H%M%SZ)\"
        echo \"Preserving existing \$FINAL as \$BACKUP\"
        mv \"\$FINAL\" \"\$BACKUP\"
    fi
    if [ -d \"\$META_FINAL\" ]; then
        META_BACKUP=\"\${META_FINAL}.previous_\$(date -u +%Y%m%dT%H%M%SZ)\"
        echo \"Preserving existing \$META_FINAL as \$META_BACKUP\"
        mv \"\$META_FINAL\" \"\$META_BACKUP\"
    fi
    mv \"\$TMP\" \"\$FINAL\"
    mv \"\$META_TMP\" \"\$META_FINAL\"
"

echo
echo "=== Verify remote layout ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "
    set -e
    cd ${HPC_ROOT}
    echo '--- space ---'
    df -h .
    echo
    echo '--- repaired split counts ---'
    wc -l ${DATA_NAME}/shared/test.tsv
    wc -l ${DATA_NAME}/shared/structured_train.tsv
    wc -l ${DATA_NAME}/shared/random_train.tsv
    echo
    echo '--- train.tsv counts ---'
    find ${DATA_NAME}/exp1 -name train.tsv | wc -l
    find ${DATA_NAME} -name train.tsv | wc -l
    echo
    echo '--- job files ---'
    ls -lh jobs_repaired_20260502.tsv jobs_repaired_20260502.ranges.sh submit_repaired_20260502.sbatch hf_job_train_hpc.py
    echo
    echo '--- ranges ---'
    . ./jobs_repaired_20260502.ranges.sh
    echo \"TIER1_NLLB600_PAPER=\$TIER1_NLLB600_PAPER\"
    echo \"TIER2_PIPELINE=\$TIER2_PIPELINE\"
    echo \"TIER3_MBART_FORWARD=\$TIER3_MBART_FORWARD\"
    echo \"TIER4_NLLB13_FORWARD=\$TIER4_NLLB13_FORWARD\"
    echo \"TIER5_MBART_REVERSE_NICE_TO_HAVE=\$TIER5_MBART_REVERSE_NICE_TO_HAVE\"
    echo
    echo '--- repaired metadata ---'
    ls -lh metadata/${DATA_NAME}
"

cat <<EOF

=== Done. Next: submit on Discovery ===

  ssh ${REMOTE}
  cd ${HPC_ROOT}
  source jobs_repaired_20260502.ranges.sh

  sbatch --array="\$TIER1_NLLB600_PAPER" submit_repaired_20260502.sbatch
  sbatch --array="\$TIER2_PIPELINE" submit_repaired_20260502.sbatch
  sbatch --array="\$TIER3_MBART_FORWARD" submit_repaired_20260502.sbatch
  sbatch --array="\$TIER4_NLLB13_FORWARD" submit_repaired_20260502.sbatch

Optional later:

  sbatch --array="\$TIER5_MBART_REVERSE_NICE_TO_HAVE" submit_repaired_20260502.sbatch
EOF
