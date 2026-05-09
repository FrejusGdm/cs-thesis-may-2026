#!/usr/bin/env bash
# launch_jobs.sh — Launch all experiment training runs on HuggingFace Jobs.
#
# Prerequisites:
#   1. hf auth login
#   2. Data uploaded to DATASET_REPO
#   3. hf_job_train.py in experiments/training/
#
# Usage:
#   ./launch_jobs.sh [tier]
#
# Tiers (600M — original):
#   smoke  — T1: Exp1 × seed 42 only (6 jobs)
#   core   — T2: Exp1 remaining seeds + baselines (39 jobs)
#   scale  — T3: Exp2 all conditions × 5 seeds (110 jobs)
#   ablate — T4: Ablations × 3 seeds (78 jobs)
#   arch   — T5: Architecture variants (6 jobs)
#   all    — T1 through T5 sequentially
#
# Tiers (NLLB-1.3B — broader replication):
#   1.3b-smoke  — Exp1 × seed 42 (6 jobs, a10g-small)
#   1.3b-core   — Exp1 + baselines (45 jobs)
#   1.3b-scale  — Exp2 scaling curves (110 jobs)
#   1.3b-ablate — Ablations (78 jobs)
#   1.3b-all    — All 1.3B tiers sequentially (239 jobs)
#
# Tiers (mBART-50, French init):
#   mbart-fr-smoke  — Exp1 × seed 42 (6 jobs, t4-small)
#   mbart-fr-core   — Exp1 + baselines (45 jobs)
#   mbart-fr-scale  — Exp2 scaling curves (110 jobs)
#   mbart-fr-ablate — Ablations (78 jobs)
#   mbart-fr-all    — All mBART-fr tiers (239 jobs)
#
# Tiers (mBART-50, Random init):
#   mbart-rand-smoke  — Exp1 × seed 42 (6 jobs, t4-small)
#   mbart-rand-core   — Exp1 + baselines (45 jobs)
#   mbart-rand-scale  — Exp2 scaling curves (110 jobs)
#   mbart-rand-ablate — Ablations (78 jobs)
#   mbart-rand-all    — All mBART-rand tiers (239 jobs)

set -uo pipefail

# ============================================================================
# Configuration — EDIT THESE
# ============================================================================
HF_USER="JosueG"
DATASET_REPO="${HF_USER}/adja-nmt-splits"
RESULTS_REPO="${HF_USER}/adja-nmt-results"
RESULTS_REPO_1_3B="${HF_USER}/adja-nmt-results-1.3b"
RESULTS_REPO_MBART_FR="${HF_USER}/adja-nmt-results-mbart-fr"
RESULTS_REPO_MBART_RAND="${HF_USER}/adja-nmt-results-mbart-random"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/hf_job_train.py"
MODEL_600M="facebook/nllb-200-distilled-600M"
MODEL_1_3B="facebook/nllb-200-1.3B"
MODEL_MBART="facebook/mbart-large-50-many-to-many-mmt"

SEEDS_5=(42 123 456 789 2024)
SEEDS_3=(42 123 456)
DELAY=5   # seconds between job launches
RETRY_WAIT=65  # seconds to wait on 429 (rate limit window is 5 min)
MAX_RETRIES=3

# ============================================================================
# Job launcher (with retry on 429 rate limit)
# ============================================================================
JOB_COUNT=0
FAIL_COUNT=0

launch_job() {
    # Args: experiment condition seed hardware model [timeout [results_repo [extra_env_args...]]]
    local experiment=$1
    local condition=$2
    local seed=$3
    local hardware=$4
    local model=$5
    local timeout=${6:-"3h"}
    local results_repo=${7:-"$RESULTS_REPO"}
    shift 7 2>/dev/null || true
    local extra_env=("$@")  # remaining args are extra -e KEY=VALUE pairs

    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[Job #${JOB_COUNT}] ${experiment}/${condition}/seed${seed} on ${hardware} → ${results_repo}"

    # Build extra -e flags (safe with set -u for empty arrays)
    local extra_flags=()
    if [ ${#extra_env[@]} -gt 0 ]; then
        for kv in "${extra_env[@]}"; do
            extra_flags+=(-e "$kv")
        done
    fi

    local attempt=0
    while true; do
        if hf jobs uv run \
            --flavor "$hardware" \
            --timeout "$timeout" \
            --detach \
            --secrets HF_TOKEN \
            -e EXPERIMENT="$experiment" \
            -e CONDITION="$condition" \
            -e SEED="$seed" \
            -e MODEL="$model" \
            -e DATASET_REPO="$DATASET_REPO" \
            -e RESULTS_REPO="$results_repo" \
            ${extra_flags[@]+"${extra_flags[@]}"} \
            "$SCRIPT_PATH" 2>&1; then
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -ge $MAX_RETRIES ]; then
                echo "FAILED after ${MAX_RETRIES} retries: ${experiment}/${condition}/seed${seed}"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                break
            fi
            echo "  Rate limited — waiting ${RETRY_WAIT}s before retry ${attempt}/${MAX_RETRIES}..."
            sleep "$RETRY_WAIT"
        fi
    done

    sleep "$DELAY"
}

# ============================================================================
# Tier 1: Smoke test — Exp1, seed 42 only
# ============================================================================
tier_smoke() {
    echo ""
    echo "=========================================="
    echo "TIER 1: SMOKE TEST (Exp1 × seed 42)"
    echo "=========================================="

    local conditions=(
        "RANDOM-10K"
        "RANDOM-6K_STRUCTURED-4K"
        "RANDOM-10K_STRUCTURED-4K"
        "STRUCTURED-4K-ONLY"
        "RANDOM-4K"
        "STRUCTURED-2K"
    )

    for cond in "${conditions[@]}"; do
        launch_job "exp1" "$cond" 42 "t4-small" "$MODEL_600M"
    done

    echo ""
    echo "Smoke test: ${#conditions[@]} jobs launched."
    echo "Check results before proceeding to next tier."
}

# ============================================================================
# Tier 2: Core — Exp1 remaining seeds + baselines
# ============================================================================
tier_core() {
    echo ""
    echo "=========================================="
    echo "TIER 2: CORE (Exp1 seeds 123-2024 + baselines)"
    echo "=========================================="

    local exp1_conditions=(
        "RANDOM-10K"
        "RANDOM-6K_STRUCTURED-4K"
        "RANDOM-10K_STRUCTURED-4K"
        "STRUCTURED-4K-ONLY"
        "RANDOM-4K"
        "STRUCTURED-2K"
    )

    # Exp1 remaining seeds (123, 456, 789, 2024)
    for cond in "${exp1_conditions[@]}"; do
        for seed in 123 456 789 2024; do
            launch_job "exp1" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Baselines × 5 seeds
    local baselines=(
        "LENGTH-STRATIFIED"
        "VOCAB-MAXIMIZED"
        "TF-IDF-DIVERSE"
    )

    for cond in "${baselines[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "baselines" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Core tier launched."
}

# ============================================================================
# Tier 3: Scaling curves — Exp2 all conditions × 5 seeds
# ============================================================================
tier_scale() {
    echo ""
    echo "=========================================="
    echo "TIER 3: SCALING CURVES (Exp2 × 5 seeds)"
    echo "=========================================="

    # Structured scaling
    for size in 200 500 1000 2000 3000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "STRUCTURED-${size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Random scaling
    for size in 200 500 1000 2000 4000 6000 8000 10000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-${size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Combined additive
    for struct_size in 500 1000 2000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-6K_STRUCTURED-${struct_size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Combined replacement
    for struct_size in 500 1000 2000 4000; do
        local random_size=$((10000 - struct_size))
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "REPLACE-R${random_size}_S${struct_size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Scaling tier launched."
}

# ============================================================================
# Tier 4: Ablations × 3 seeds
# ============================================================================
tier_ablate() {
    echo ""
    echo "=========================================="
    echo "TIER 4: ABLATIONS (× 3 seeds)"
    echo "=========================================="

    # Module leave-one-out
    local module_loo=(
        "FULL" "NO-NEGATION" "NO-PAST" "NO-FUTURE" "NO-QUESTIONS" "BASE-ONLY"
    )
    for cond in "${module_loo[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_loo" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Size-controlled module
    local module_sc=(
        "FULL-1K" "NO-NEG-1K" "NO-PAST-1K" "NO-FUT-1K" "NO-QUEST-1K" "BASE-1K"
    )
    for cond in "${module_sc[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_size_ctrl" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Pronoun coverage
    local pronouns=("ALL-8" "REDUCED-4" "SINGULAR-3" "MINIMAL-1")
    for cond in "${pronouns[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/pronoun" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Verb diversity
    local verbs=("10-VERBS" "5-VERBS-a" "5-VERBS-b" "5-VERBS-c" "3-VERBS-a" "3-VERBS-b" "3-VERBS-c" "1-VERB")
    for cond in "${verbs[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/verb" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Minimal pairs
    local pairs=("PAIRS-INTACT" "PAIRS-BROKEN")
    for cond in "${pairs[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/minimal_pairs" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Ablation tier launched."
}

# ============================================================================
# Tier 5: Architecture variants
# ============================================================================
tier_arch() {
    echo ""
    echo "=========================================="
    echo "TIER 5: ARCHITECTURE VARIANTS"
    echo "=========================================="

    # NLLB-1.3B on A10G (3 seeds, best structured condition)
    for seed in "${SEEDS_3[@]}"; do
        launch_job "exp1" "RANDOM-10K_STRUCTURED-4K" "$seed" "a10g-small" "$MODEL_1_3B" "4h"
    done

    # mBART-50 on T4 (3 seeds)
    for seed in "${SEEDS_3[@]}"; do
        launch_job "exp1" "RANDOM-10K_STRUCTURED-4K" "$seed" "t4-small" "$MODEL_MBART" "3h"
    done

    echo ""
    echo "Architecture tier launched."
}

# ============================================================================
# Shared condition lists (used by multiple model tiers)
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
# Generic tier helpers — parameterized by model, hardware, results repo, extra env
# ============================================================================

_tier_smoke() {
    local model=$1 hw=$2 repo=$3 timeout=$4
    shift 4
    for cond in "${EXP1_CONDITIONS[@]}"; do
        launch_job "exp1" "$cond" 42 "$hw" "$model" "$timeout" "$repo" "$@"
    done
}

_tier_core() {
    local model=$1 hw=$2 repo=$3 timeout=$4
    shift 4
    # Exp1 × 5 seeds
    for cond in "${EXP1_CONDITIONS[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp1" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    # Baselines × 5 seeds
    for cond in "${BASELINES[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "baselines" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
}

_tier_scale() {
    local model=$1 hw=$2 repo=$3 timeout=$4
    shift 4
    # Structured scaling
    for size in 200 500 1000 2000 3000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "STRUCTURED-${size}" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    # Random scaling
    for size in 200 500 1000 2000 4000 6000 8000 10000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-${size}" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    # Combined additive
    for struct_size in 500 1000 2000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-6K_STRUCTURED-${struct_size}" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    # Combined replacement
    for struct_size in 500 1000 2000 4000; do
        local random_size=$((10000 - struct_size))
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "REPLACE-R${random_size}_S${struct_size}" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
}

_tier_ablate() {
    local model=$1 hw=$2 repo=$3 timeout=$4
    shift 4
    for cond in "${MODULE_LOO[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_loo" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    for cond in "${MODULE_SC[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_size_ctrl" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    for cond in "${PRONOUNS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/pronoun" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    for cond in "${VERBS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/verb" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
    for cond in "${PAIRS[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/minimal_pairs" "$cond" "$seed" "$hw" "$model" "$timeout" "$repo" "$@"
        done
    done
}

# ============================================================================
# NLLB-1.3B Tiers (a10g-small, 4h timeout)
# ============================================================================

tier_1_3b_smoke() {
    echo ""
    echo "=========================================="
    echo "NLLB-1.3B SMOKE (Exp1 × seed 42, 6 jobs)"
    echo "=========================================="
    _tier_smoke "$MODEL_1_3B" "a10g-small" "$RESULTS_REPO_1_3B" "4h"
    echo "NLLB-1.3B smoke: 6 jobs launched."
}

tier_1_3b_core() {
    echo ""
    echo "=========================================="
    echo "NLLB-1.3B CORE (Exp1 5 seeds + baselines, 45 jobs)"
    echo "=========================================="
    _tier_core "$MODEL_1_3B" "a10g-small" "$RESULTS_REPO_1_3B" "4h"
    echo "NLLB-1.3B core: 45 jobs launched."
}

tier_1_3b_scale() {
    echo ""
    echo "=========================================="
    echo "NLLB-1.3B SCALE (Exp2 scaling, 110 jobs)"
    echo "=========================================="
    _tier_scale "$MODEL_1_3B" "a10g-small" "$RESULTS_REPO_1_3B" "4h"
    echo "NLLB-1.3B scale: 110 jobs launched."
}

tier_1_3b_ablate() {
    echo ""
    echo "=========================================="
    echo "NLLB-1.3B ABLATE (ablations, 78 jobs)"
    echo "=========================================="
    _tier_ablate "$MODEL_1_3B" "a10g-small" "$RESULTS_REPO_1_3B" "4h"
    echo "NLLB-1.3B ablate: 78 jobs launched."
}

# ============================================================================
# mBART-50 French init Tiers (t4-small, 3h timeout, SIMILAR_LANG=fr_XX)
# ============================================================================

tier_mbart_fr_smoke() {
    echo ""
    echo "=========================================="
    echo "mBART-50 FRENCH-INIT SMOKE (6 jobs)"
    echo "=========================================="
    _tier_smoke "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_FR" "3h" "SIMILAR_LANG=fr_XX"
    echo "mBART-fr smoke: 6 jobs launched."
}

tier_mbart_fr_core() {
    echo ""
    echo "=========================================="
    echo "mBART-50 FRENCH-INIT CORE (45 jobs)"
    echo "=========================================="
    _tier_core "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_FR" "3h" "SIMILAR_LANG=fr_XX"
    echo "mBART-fr core: 45 jobs launched."
}

tier_mbart_fr_scale() {
    echo ""
    echo "=========================================="
    echo "mBART-50 FRENCH-INIT SCALE (110 jobs)"
    echo "=========================================="
    _tier_scale "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_FR" "3h" "SIMILAR_LANG=fr_XX"
    echo "mBART-fr scale: 110 jobs launched."
}

tier_mbart_fr_ablate() {
    echo ""
    echo "=========================================="
    echo "mBART-50 FRENCH-INIT ABLATE (78 jobs)"
    echo "=========================================="
    _tier_ablate "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_FR" "3h" "SIMILAR_LANG=fr_XX"
    echo "mBART-fr ablate: 78 jobs launched."
}

# ============================================================================
# mBART-50 Random init Tiers (t4-small, 3h timeout, SIMILAR_LANG=none)
# ============================================================================

tier_mbart_rand_smoke() {
    echo ""
    echo "=========================================="
    echo "mBART-50 RANDOM-INIT SMOKE (6 jobs)"
    echo "=========================================="
    _tier_smoke "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_RAND" "3h" "SIMILAR_LANG=none"
    echo "mBART-rand smoke: 6 jobs launched."
}

tier_mbart_rand_core() {
    echo ""
    echo "=========================================="
    echo "mBART-50 RANDOM-INIT CORE (45 jobs)"
    echo "=========================================="
    _tier_core "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_RAND" "3h" "SIMILAR_LANG=none"
    echo "mBART-rand core: 45 jobs launched."
}

tier_mbart_rand_scale() {
    echo ""
    echo "=========================================="
    echo "mBART-50 RANDOM-INIT SCALE (110 jobs)"
    echo "=========================================="
    _tier_scale "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_RAND" "3h" "SIMILAR_LANG=none"
    echo "mBART-rand scale: 110 jobs launched."
}

tier_mbart_rand_ablate() {
    echo ""
    echo "=========================================="
    echo "mBART-50 RANDOM-INIT ABLATE (78 jobs)"
    echo "=========================================="
    _tier_ablate "$MODEL_MBART" "t4-small" "$RESULTS_REPO_MBART_RAND" "3h" "SIMILAR_LANG=none"
    echo "mBART-rand ablate: 78 jobs launched."
}

# ============================================================================
# Main
# ============================================================================
TIER=${1:-smoke}

echo "HF Jobs Launcher — Adja NMT Experiments"
echo "Dataset repo: ${DATASET_REPO}"
echo "Results repos:"
echo "  600M:       ${RESULTS_REPO}"
echo "  1.3B:       ${RESULTS_REPO_1_3B}"
echo "  mBART-fr:   ${RESULTS_REPO_MBART_FR}"
echo "  mBART-rand: ${RESULTS_REPO_MBART_RAND}"
echo "Script: ${SCRIPT_PATH}"
echo ""

case "$TIER" in
    # ── Original 600M tiers ──
    smoke)  tier_smoke ;;
    core)   tier_core ;;
    scale)  tier_scale ;;
    ablate) tier_ablate ;;
    arch)   tier_arch ;;
    all)
        tier_smoke
        echo ""
        echo ">>> Launching remaining 600M tiers <<<"
        echo ""
        tier_core
        tier_scale
        tier_ablate
        tier_arch
        ;;

    # ── NLLB-1.3B tiers ──
    1.3b-smoke)  tier_1_3b_smoke ;;
    1.3b-core)   tier_1_3b_core ;;
    1.3b-scale)  tier_1_3b_scale ;;
    1.3b-ablate) tier_1_3b_ablate ;;
    1.3b-all)
        tier_1_3b_smoke
        tier_1_3b_core
        tier_1_3b_scale
        tier_1_3b_ablate
        ;;

    # ── mBART-50 French init tiers ──
    mbart-fr-smoke)  tier_mbart_fr_smoke ;;
    mbart-fr-core)   tier_mbart_fr_core ;;
    mbart-fr-scale)  tier_mbart_fr_scale ;;
    mbart-fr-ablate) tier_mbart_fr_ablate ;;
    mbart-fr-all)
        tier_mbart_fr_smoke
        tier_mbart_fr_core
        tier_mbart_fr_scale
        tier_mbart_fr_ablate
        ;;

    # ── mBART-50 Random init tiers ──
    mbart-rand-smoke)  tier_mbart_rand_smoke ;;
    mbart-rand-core)   tier_mbart_rand_core ;;
    mbart-rand-scale)  tier_mbart_rand_scale ;;
    mbart-rand-ablate) tier_mbart_rand_ablate ;;
    mbart-rand-all)
        tier_mbart_rand_smoke
        tier_mbart_rand_core
        tier_mbart_rand_scale
        tier_mbart_rand_ablate
        ;;

    # ── Convenience combos ──
    new-models-smoke)
        tier_1_3b_smoke
        tier_mbart_fr_smoke
        tier_mbart_rand_smoke
        ;;
    new-models-all)
        tier_1_3b_smoke
        tier_1_3b_core
        tier_1_3b_scale
        tier_1_3b_ablate
        tier_mbart_fr_smoke
        tier_mbart_fr_core
        tier_mbart_fr_scale
        tier_mbart_fr_ablate
        tier_mbart_rand_smoke
        tier_mbart_rand_core
        tier_mbart_rand_scale
        tier_mbart_rand_ablate
        ;;

    *)
        echo "Unknown tier: $TIER"
        echo ""
        echo "Usage: $0 [tier]"
        echo ""
        echo "600M tiers:        smoke | core | scale | ablate | arch | all"
        echo "NLLB-1.3B tiers:   1.3b-smoke | 1.3b-core | 1.3b-scale | 1.3b-ablate | 1.3b-all"
        echo "mBART French-init: mbart-fr-smoke | mbart-fr-core | mbart-fr-scale | mbart-fr-ablate | mbart-fr-all"
        echo "mBART Random-init: mbart-rand-smoke | mbart-rand-core | mbart-rand-scale | mbart-rand-ablate | mbart-rand-all"
        echo "Combos:            new-models-smoke | new-models-all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Total jobs launched: ${JOB_COUNT}"
if [ "${FAIL_COUNT:-0}" -gt 0 ]; then
    echo "FAILED submissions: ${FAIL_COUNT}"
fi
echo "Monitor at: https://huggingface.co/settings/jobs"
echo "=========================================="
