#!/usr/bin/env bash
# fetch_results.sh — pull training outputs off Discovery.
#
# Drop-in replacement for the historical rsync recipe in CLAUDE.md:
#   rsync -av --include="*/test_metrics.json" --include="*/" --exclude="*" …
# which is broken by macOS 15 openrsync. Uses scp + SSH ControlMaster so
# you get one password prompt for the whole transfer.
#
# Safe by design:
#   - Only READS from the remote subtree you select (default: results/april2026).
#   - Only WRITES into the matching local subtree (default: experiments/results/april2026).
#   - Paper-era results on either side are never touched.
#   - Re-runs are idempotent: scp overwrites each file with the current
#     HPC version. Same (job, seed) training is deterministic, so
#     re-syncing a completed job is effectively a no-op.
#
# Usage (from repo root):
#   bash experiments/training/hpc/fetch_results.sh                     # april2026 (default)
#   bash experiments/training/hpc/fetch_results.sh april2026           # same, explicit
#   bash experiments/training/hpc/fetch_results.sh hpc_new             # pull paper re-run results
#   bash experiments/training/hpc/fetch_results.sh rebuttal_repaired_20260502
#   bash experiments/training/hpc/fetch_results.sh pipeline_repaired_20260502
#   bash experiments/training/hpc/fetch_results.sh all                 # everything under results/
#
# Env overrides (rarely needed):
#   REMOTE=user@host       REMOTE_ROOT=/abs/path     LOCAL_ROOT=experiments/results

set -uo pipefail

# --- config ---
REMOTE="${REMOTE:-f006g5b@discovery.dartmouth.edu}"
HPC_ROOT="${REMOTE_ROOT:-<HPC_WORKDIR>"

SCOPE="${1:-april2026}"
case "$SCOPE" in
    april2026)       REMOTE_SUBDIR="results/april2026";       LOCAL_SUBDIR="experiments/results/april2026" ;;
    rebuttal_rerun)  REMOTE_SUBDIR="results/rebuttal_rerun";  LOCAL_SUBDIR="experiments/results/rebuttal_rerun" ;;
    pipeline)        REMOTE_SUBDIR="results/pipeline";        LOCAL_SUBDIR="experiments/results/pipeline" ;;
    rebuttal_repaired_20260502)
                     REMOTE_SUBDIR="results/rebuttal_repaired_20260502"; LOCAL_SUBDIR="experiments/results/rebuttal_repaired_20260502" ;;
    pipeline_repaired_20260502)
                     REMOTE_SUBDIR="results/pipeline_repaired_20260502"; LOCAL_SUBDIR="experiments/results/pipeline_repaired_20260502" ;;
    hpc_new)         REMOTE_SUBDIR="results";                 LOCAL_SUBDIR="experiments/results/hpc_new" ;;
    all)             REMOTE_SUBDIR="results";                 LOCAL_SUBDIR="experiments/results/hpc_all" ;;
    *)               echo "Unknown scope '$SCOPE'. Valid: april2026 | rebuttal_rerun | pipeline | rebuttal_repaired_20260502 | pipeline_repaired_20260502 | hpc_new | all" >&2; exit 1 ;;
esac

REMOTE_PATH="${HPC_ROOT}/${REMOTE_SUBDIR}"
LOCAL_PATH="${LOCAL_ROOT:-$LOCAL_SUBDIR}"

# Files to pull (test_metrics.json is the canonical result file; the other
# two are produced by the april-2026 sweep's extended training script).
FILENAMES=( test_metrics.json test_predictions.jsonl test_neural_metrics.json )

# --- SSH ControlMaster (single password prompt for the whole session) ---
CM_SOCKET="/tmp/ssh-cm-fetch-$$"
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

echo "Fetch  scope: ${SCOPE}"
echo "Source:       ${REMOTE}:${REMOTE_PATH}"
echo "Destination:  ${LOCAL_PATH}"
echo

# --- Discover remote files (one find, all filenames) ---
find_expr=""
for name in "${FILENAMES[@]}"; do
    find_expr+=" -name ${name} -o"
done
find_expr="${find_expr% -o}"  # strip trailing -o

echo "=== Listing remote files (opens SSH master connection) ==="
LIST_FILE="$(mktemp)"
if ! ssh "${SSH_OPTS[@]}" "$REMOTE" \
        "find ${REMOTE_PATH} \\( ${find_expr} \\) 2>/dev/null" > "$LIST_FILE"; then
    echo "ERROR: ssh/find failed." >&2
    rm -f "$LIST_FILE"
    exit 1
fi
total=$(wc -l < "$LIST_FILE" | tr -d ' ')
echo "Found ${total} file(s) to fetch."
if [ "$total" -eq 0 ]; then
    echo "(nothing to do — maybe jobs haven't landed any results yet?)"
    rm -f "$LIST_FILE"
    exit 0
fi

# --- Copy them, reusing the master connection ---
echo
echo "=== Copying ==="
copied=0 failed=0
while IFS= read -r f; do
    [ -z "$f" ] && continue
    rel="${f#${REMOTE_PATH}/}"
    dst="${LOCAL_PATH}/${rel}"
    mkdir -p "$(dirname "$dst")"
    if scp -q "${SSH_OPTS[@]}" "${REMOTE}:${f}" "$dst"; then
        copied=$((copied + 1))
    else
        echo "  FAIL: $rel" >&2
        failed=$((failed + 1))
    fi
done < "$LIST_FILE"
rm -f "$LIST_FILE"

echo
echo "Copied:  ${copied}"
echo "Failed:  ${failed}"

# --- Quick summary: how many completed jobs + per-model counts ---
if [ -d "$LOCAL_PATH" ]; then
    echo
    echo "=== Local summary (${LOCAL_PATH}) ==="
    n_done=$(find "$LOCAL_PATH" -name test_metrics.json 2>/dev/null | wc -l | tr -d ' ')
    echo "test_metrics.json files: ${n_done}"
    if [ "$n_done" -gt 0 ]; then
        echo "Per model:"
        find "$LOCAL_PATH" -name test_metrics.json 2>/dev/null \
            | awk -F/ '{print $(NF-4)}' | sort | uniq -c \
            | awk '{printf "  %-15s %s\n", $2, $1}'
    fi
fi

echo
echo "Next step: aggregate."
echo "  python experiments/analysis/compare_april2026_vs_acl.py"
echo "  python experiments/analysis/compute_robustness_table.py \\"
echo "    --results-root ${LOCAL_PATH} --skip-paper-reference \\"
echo "    --include-conditions auto --include-models auto \\"
echo "    --experiment april2026_runA"

[ "$failed" -eq 0 ] || exit 1
