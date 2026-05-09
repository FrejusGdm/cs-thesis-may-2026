#!/usr/bin/env bash
# launch_api_finetune.sh — Launch all experiment fine-tuning runs via OpenAI / Gemini APIs.
#
# Mirrors the condition structure of launch_jobs.sh (233 jobs per platform).
# Each job blocks until training + eval completes, so run inside tmux.
#
# Prerequisites:
#   1. Convert data first (one-time):
#        python experiments/training/convert_to_finetune_format.py \
#            --data-dir experiments/data/splits --batch \
#            --output-dir experiments/data/finetune_formats
#   2. .env with OPENAI_API_KEY and/or GOOGLE_API_KEY
#   3. pip install openai google-genai sacrebleu sacremoses python-dotenv
#
# Usage:
#   bash experiments/training/launch_api_finetune.sh --platform openai
#   bash experiments/training/launch_api_finetune.sh --platform gemini
#   bash experiments/training/launch_api_finetune.sh --platform both \
#       --gcp-project YOUR_PROJECT --gcs-bucket YOUR_BUCKET
#
# Options:
#   --platform openai|gemini|both    Which API to use (required)
#   --data-dir DIR                   Finetune JSONL root (default: experiments/data/finetune_formats)
#   --splits-dir DIR                 Original splits root (default: experiments/data/splits)
#   --results-dir DIR                Results output root (default: results)
#   --gcp-project PROJECT            GCP project ID (required for Gemini Vertex)
#   --gcs-bucket BUCKET              GCS bucket name (required for Gemini Vertex)
#   --gcp-location LOCATION          GCP region (default: us-central1)
#   --use-ai-studio                  Use Google AI Studio instead of Vertex AI for Gemini
#   --tier TIER                      Run a specific tier: smoke|core|scale|ablate|all (default: all)
#   --dry-run                        Print jobs without running them

set -uo pipefail

# ============================================================================
# Resolve paths relative to this script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# Defaults
# ============================================================================
PLATFORM=""
DATA_DIR="${PROJECT_ROOT}/experiments/data/finetune_formats"
SPLITS_DIR="${PROJECT_ROOT}/experiments/data/splits"
RESULTS_DIR="${PROJECT_ROOT}/results"
GCP_PROJECT=""
GCS_BUCKET=""
GCP_LOCATION="us-central1"
USE_AI_STUDIO=false
TIER="all"
DRY_RUN=false
LOG_FILE="${PROJECT_ROOT}/launch_api_finetune.log"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)     PLATFORM="$2"; shift 2 ;;
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        --splits-dir)   SPLITS_DIR="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --gcp-project)  GCP_PROJECT="$2"; shift 2 ;;
        --gcs-bucket)   GCS_BUCKET="$2"; shift 2 ;;
        --gcp-location) GCP_LOCATION="$2"; shift 2 ;;
        --use-ai-studio) USE_AI_STUDIO=true; shift ;;
        --tier)         TIER="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-load .env if present (try project root, then script dir)
for envfile in "${PROJECT_ROOT}/.env" "${PROJECT_ROOT}/experiments/training/hpc/.env" "${SCRIPT_DIR}/.env"; do
    if [[ -f "$envfile" ]]; then
        set -a
        source "$envfile"
        set +a
        echo "[env] Loaded env from: $envfile"
        break
    fi
done

if [[ -z "$PLATFORM" ]]; then
    echo "ERROR: --platform is required (openai|gemini|both)"
    exit 1
fi

if [[ "$PLATFORM" == "gemini" || "$PLATFORM" == "both" ]]; then
    if [[ "$USE_AI_STUDIO" == false && ( -z "$GCP_PROJECT" || -z "$GCS_BUCKET" ) ]]; then
        echo "ERROR: Gemini Vertex AI requires --gcp-project and --gcs-bucket"
        echo "  Or use --use-ai-studio for Google AI Studio (no GCP needed)"
        exit 1
    fi
fi

# ============================================================================
# Seeds (same as launch_jobs.sh)
# ============================================================================
SEEDS_5=(42 123 456 789 2024)
SEEDS_3=(42 123 456)

# ============================================================================
# Condition lists (same as launch_jobs.sh)
# ============================================================================
EXP1_CONDITIONS=(
    "RANDOM-10K"
    "RANDOM-6K_STRUCTURED-4K"
    "RANDOM-10K_STRUCTURED-4K"
    "STRUCTURED-4K-ONLY"
    "RANDOM-4K"
    "STRUCTURED-2K"
)

BASELINES=(
    "LENGTH-STRATIFIED"
    "VOCAB-MAXIMIZED"
    "TF-IDF-DIVERSE"
)

MODULE_LOO=("FULL" "NO-NEGATION" "NO-PAST" "NO-FUTURE" "NO-QUESTIONS" "BASE-ONLY")
MODULE_SC=("FULL-1K" "NO-NEG-1K" "NO-PAST-1K" "NO-FUT-1K" "NO-QUEST-1K" "BASE-1K")
PRONOUNS=("ALL-8" "REDUCED-4" "SINGULAR-3" "MINIMAL-1")
VERBS=("10-VERBS" "5-VERBS-a" "5-VERBS-b" "5-VERBS-c" "3-VERBS-a" "3-VERBS-b" "3-VERBS-c" "1-VERB")
PAIRS=("PAIRS-INTACT" "PAIRS-BROKEN")

# ============================================================================
# Counters
# ============================================================================
TOTAL=0
COMPLETED=0
SKIPPED=0
FAILED=0

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ============================================================================
# Job runners
# ============================================================================

run_openai_job() {
    local experiment=$1
    local condition=$2
    local seed=$3

    local output_dir="${RESULTS_DIR}/openai/${experiment}/${condition}/seed${seed}"
    local metrics_file="${output_dir}/test_metrics.json"

    TOTAL=$((TOTAL + 1))

    if [[ "$DRY_RUN" == true ]]; then
        log "  [DRY] openai/${experiment}/${condition}/seed${seed}"
        return 0
    fi

    # Skip if already done
    if [[ -f "$metrics_file" ]]; then
        log "  [SKIP] openai/${experiment}/${condition}/seed${seed} — already done"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local train_jsonl="${DATA_DIR}/${experiment}/${condition}/openai_train.jsonl"
    local val_jsonl="${DATA_DIR}/${experiment}/${condition}/openai_val.jsonl"
    local test_tsv="${SPLITS_DIR}/shared/test.tsv"

    # Check train file exists
    if [[ ! -f "$train_jsonl" ]]; then
        log "  [MISS] openai/${experiment}/${condition}/seed${seed} — train JSONL not found: ${train_jsonl}"
        FAILED=$((FAILED + 1))
        return 1
    fi

    log "  [RUN]  openai/${experiment}/${condition}/seed${seed}"

    local val_flag=""
    if [[ -f "$val_jsonl" ]]; then
        val_flag="--val-jsonl ${val_jsonl}"
    fi

    if python "${SCRIPT_DIR}/openai_finetune.py" \
        --train-jsonl "$train_jsonl" \
        $val_flag \
        --test-tsv "$test_tsv" \
        --experiment "$experiment" \
        --condition "$condition" \
        --seed "$seed" \
        --output-dir "$output_dir" 2>&1 | tee -a "$LOG_FILE"; then
        log "  [DONE] openai/${experiment}/${condition}/seed${seed}"
        COMPLETED=$((COMPLETED + 1))
    else
        log "  [FAIL] openai/${experiment}/${condition}/seed${seed}"
        FAILED=$((FAILED + 1))
    fi
}

run_gemini_job() {
    local experiment=$1
    local condition=$2
    local seed=$3

    local output_dir="${RESULTS_DIR}/gemini/${experiment}/${condition}/seed${seed}"
    local metrics_file="${output_dir}/test_metrics.json"

    TOTAL=$((TOTAL + 1))

    if [[ "$DRY_RUN" == true ]]; then
        log "  [DRY] gemini/${experiment}/${condition}/seed${seed}"
        return 0
    fi

    # Skip if already done
    if [[ -f "$metrics_file" ]]; then
        log "  [SKIP] gemini/${experiment}/${condition}/seed${seed} — already done"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    local train_jsonl="${DATA_DIR}/${experiment}/${condition}/gemini_train.jsonl"
    local test_tsv="${SPLITS_DIR}/shared/test.tsv"

    if [[ ! -f "$train_jsonl" ]]; then
        log "  [MISS] gemini/${experiment}/${condition}/seed${seed} — train JSONL not found: ${train_jsonl}"
        FAILED=$((FAILED + 1))
        return 1
    fi

    log "  [RUN]  gemini/${experiment}/${condition}/seed${seed}"

    local vertex_flags=""
    local gcs_uri=""

    if [[ "$USE_AI_STUDIO" == false ]]; then
        # Upload JSONL to GCS if not already there
        local gcs_path="gs://${GCS_BUCKET}/adja-nmt/${experiment}/${condition}/gemini_train.jsonl"
        if ! gsutil -q stat "$gcs_path" 2>/dev/null; then
            log "    Uploading to GCS: ${gcs_path}"
            gsutil cp "$train_jsonl" "$gcs_path" 2>&1 | tee -a "$LOG_FILE"
        fi
        vertex_flags="--use-vertex --project ${GCP_PROJECT} --location ${GCP_LOCATION} --gcs-uri ${gcs_path}"
    fi

    if python "${SCRIPT_DIR}/gemini_finetune.py" \
        --train-jsonl "$train_jsonl" \
        --test-tsv "$test_tsv" \
        --experiment "$experiment" \
        --condition "$condition" \
        --seed "$seed" \
        --output-dir "$output_dir" \
        $vertex_flags 2>&1 | tee -a "$LOG_FILE"; then
        log "  [DONE] gemini/${experiment}/${condition}/seed${seed}"
        COMPLETED=$((COMPLETED + 1))
    else
        log "  [FAIL] gemini/${experiment}/${condition}/seed${seed}"
        FAILED=$((FAILED + 1))
    fi
}

run_job() {
    # Dispatch to openai and/or gemini based on PLATFORM
    local experiment=$1
    local condition=$2
    local seed=$3

    if [[ "$PLATFORM" == "openai" || "$PLATFORM" == "both" ]]; then
        run_openai_job "$experiment" "$condition" "$seed"
    fi
    if [[ "$PLATFORM" == "gemini" || "$PLATFORM" == "both" ]]; then
        run_gemini_job "$experiment" "$condition" "$seed"
    fi
}

# ============================================================================
# Tier functions
# ============================================================================

tier_smoke() {
    log ""
    log "=========================================="
    log "SMOKE TEST (Exp1 x seed 42)"
    log "=========================================="
    for cond in "${EXP1_CONDITIONS[@]}"; do
        run_job "exp1" "$cond" 42
    done
}

tier_core() {
    log ""
    log "=========================================="
    log "CORE (Exp1 x 5 seeds + baselines)"
    log "=========================================="

    # Exp1 x 5 seeds
    for cond in "${EXP1_CONDITIONS[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            run_job "exp1" "$cond" "$seed"
        done
    done

    # Baselines x 5 seeds
    for cond in "${BASELINES[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            run_job "baselines" "$cond" "$seed"
        done
    done
}

tier_scale() {
    log ""
    log "=========================================="
    log "SCALING CURVES (Exp2 x 5 seeds)"
    log "=========================================="

    # Structured scaling
    for size in 200 500 1000 2000 3000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            run_job "exp2" "STRUCTURED-${size}" "$seed"
        done
    done

    # Random scaling
    for size in 200 500 1000 2000 4000 6000 8000 10000; do
        for seed in "${SEEDS_5[@]}"; do
            run_job "exp2" "RANDOM-${size}" "$seed"
        done
    done

    # Combined additive
    for struct_size in 500 1000 2000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            run_job "exp2" "RANDOM-6K_STRUCTURED-${struct_size}" "$seed"
        done
    done

    # Combined replacement
    for struct_size in 500 1000 2000 4000; do
        local random_size=$((10000 - struct_size))
        for seed in "${SEEDS_5[@]}"; do
            run_job "exp2" "REPLACE-R${random_size}_S${struct_size}" "$seed"
        done
    done
}

tier_ablate() {
    log ""
    log "=========================================="
    log "ABLATIONS (x 3 seeds)"
    log "=========================================="

    # Module leave-one-out
    for cond in "${MODULE_LOO[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            run_job "ablations/module_loo" "$cond" "$seed"
        done
    done

    # Size-controlled module
    for cond in "${MODULE_SC[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            run_job "ablations/module_size_ctrl" "$cond" "$seed"
        done
    done

    # Pronoun coverage
    for cond in "${PRONOUNS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            run_job "ablations/pronoun" "$cond" "$seed"
        done
    done

    # Verb diversity
    for cond in "${VERBS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            run_job "ablations/verb" "$cond" "$seed"
        done
    done

    # Minimal pairs
    for cond in "${PAIRS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            run_job "ablations/minimal_pairs" "$cond" "$seed"
        done
    done
}

# ============================================================================
# Main
# ============================================================================

log "============================================================"
log "API Fine-Tuning Launcher — Adja NMT Experiments"
log "============================================================"
log "Platform:    ${PLATFORM}"
log "Data dir:    ${DATA_DIR}"
log "Splits dir:  ${SPLITS_DIR}"
log "Results dir: ${RESULTS_DIR}"
log "Tier:        ${TIER}"
log "Dry run:     ${DRY_RUN}"
if [[ "$PLATFORM" == "gemini" || "$PLATFORM" == "both" ]]; then
    if [[ "$USE_AI_STUDIO" == true ]]; then
        log "Gemini mode: AI Studio"
    else
        log "Gemini mode: Vertex AI (project=${GCP_PROJECT}, bucket=${GCS_BUCKET})"
    fi
fi
log "Log file:    ${LOG_FILE}"
log ""

case "$TIER" in
    smoke)  tier_smoke ;;
    core)   tier_core ;;
    scale)  tier_scale ;;
    ablate) tier_ablate ;;
    all)
        tier_smoke
        tier_core
        tier_scale
        tier_ablate
        ;;
    *)
        echo "Unknown tier: $TIER"
        echo "Usage: $0 --platform openai|gemini|both [--tier smoke|core|scale|ablate|all]"
        exit 1
        ;;
esac

log ""
log "============================================================"
log "SUMMARY"
log "============================================================"
log "Total jobs:     ${TOTAL}"
log "Completed:      ${COMPLETED}"
log "Skipped (done): ${SKIPPED}"
log "Failed:         ${FAILED}"
log "============================================================"
