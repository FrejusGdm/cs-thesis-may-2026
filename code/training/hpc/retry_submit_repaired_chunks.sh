#!/usr/bin/env bash
# retry_submit_repaired_chunks.sh — submit the May 2 repaired rerun in chunks.
#
# Run on Discovery from:
#   <HPC_WORKDIR>
#
# It submits all 1400 repaired jobs as six arrays and retries when Slurm
# temporarily refuses sbatch submissions. Note that sbatch itself may block
# before returning "Resource temporarily unavailable"; this script waits 10s
# after that returned failure before trying again.

set -u

SBATCH_FILE="submit_repaired_20260502.sbatch"
CHUNKS=(
  "1-250"
  "251-500"
  "501-750"
  "751-1000"
  "1001-1250"
  "1251-1400"
)

retry_seconds="${RETRY_SECONDS:-10}"
submitted_job_ids=()

if [ ! -f "$SBATCH_FILE" ]; then
  echo "[$(date)] ERROR: missing $SBATCH_FILE in $(pwd)"
  exit 1
fi

echo "[$(date)] Starting repaired chunk submission loop."
echo "[$(date)] Working directory: $(pwd)"
echo "[$(date)] Chunks: ${CHUNKS[*]}"

submit_chunk() {
  local chunk="$1"
  local attempt=1

  while true; do
    echo "[$(date)] Submitting chunk ${chunk}, attempt ${attempt}..."

    output="$(sbatch --parsable --array="${chunk}" "$SBATCH_FILE" 2>&1)"
    status=$?

    echo "$output"

    if [ "$status" -eq 0 ]; then
      echo "[$(date)] Chunk ${chunk} submitted. Job ID: ${output}"
      submitted_job_ids+=("$output")
      return 0
    fi

    echo "[$(date)] Chunk ${chunk} failed with status ${status}. Retrying in ${retry_seconds}s..."
    sleep "$retry_seconds"

    attempt=$(( attempt + 1 ))
  done
}

for chunk in "${CHUNKS[@]}"; do
  submit_chunk "$chunk"
  echo "[$(date)] Moving immediately to next chunk."
done

echo "[$(date)] All chunks submitted."
echo "[$(date)] Job IDs: ${submitted_job_ids[*]}"
echo "[$(date)] Current queue:"
squeue -u "$USER"
