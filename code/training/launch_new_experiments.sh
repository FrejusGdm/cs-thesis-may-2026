#!/usr/bin/env bash
# launch_new_experiments.sh
#
# Launches new training jobs needed for professor feedback revisions:
#   1. NO-PAST-1K fix: 2 additional seeds (42, 123) — fixes missing std in Table 11
#   2. Additive ablation: 3 new conditions × 3 seeds = 9 jobs — for appendix table
#   3. Structured + smart-selection combos: 4 conditions × 5 seeds = 20 jobs — Table 9 additions
#
# Usage:
#   ./experiments/training/launch_new_experiments.sh [tier]
#
# Tiers:
#   fix       — Only the NO-PAST-1K fix (2 jobs) — run first
#   additive  — Additive ablation (9 jobs) — requires data on HF Hub
#   table9    — Struct + smart selection (20 jobs) — requires combined data on HF Hub
#   all       — All of the above sequentially
#
# Prerequisites:
#   - hf auth login
#   - For 'additive' and 'table9' tiers: run create_new_datasets.py first to upload data

set -uo pipefail

HF_USER="JosueG"
DATASET_REPO="${HF_USER}/adja-nmt-splits"
RESULTS_REPO="${HF_USER}/adja-nmt-results"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/hf_job_train.py"
MODEL_600M="facebook/nllb-200-distilled-600M"

DELAY=5
RETRY_WAIT=65
MAX_RETRIES=3
JOB_COUNT=0
FAIL_COUNT=0

launch_job() {
    local experiment=$1
    local condition=$2
    local seed=$3
    local hardware=$4
    local model=$5
    local timeout=${6:-"3h"}
    local results_repo=${7:-"$RESULTS_REPO"}

    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[Job #${JOB_COUNT}] ${experiment}/${condition}/seed${seed} on ${hardware}"

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
            "$SCRIPT_PATH" 2>&1; then
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -ge $MAX_RETRIES ]; then
                echo "FAILED after ${MAX_RETRIES} retries"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                break
            fi
            echo "  Rate limited — waiting ${RETRY_WAIT}s..."
            sleep "$RETRY_WAIT"
        fi
    done
    sleep "$DELAY"
}

# ============================================================================
# Tier: Fix — NO-PAST-1K missing seeds (2 jobs)
# ============================================================================
tier_fix() {
    echo ""
    echo "=========================================="
    echo "FIX: NO-PAST-1K missing seeds (2 jobs)"
    echo "=========================================="
    echo "Data already on HF Hub from seed456 run."

    launch_job "ablations/module_size_ctrl" "NO-PAST-1K" 42  "t4-small" "$MODEL_600M" "2h"
    launch_job "ablations/module_size_ctrl" "NO-PAST-1K" 123 "t4-small" "$MODEL_600M" "2h"

    echo "Done. After jobs complete, sync results and update tab:module_size_ctrl in acl_latex.tex."
}

# ============================================================================
# Tier: Additive — Additive ablation (9 jobs)
# Conditions build up modules: M1+M2, M1+M2+M3, M1+M2+M3+M4
# (ADD-M1 = BASE-ONLY already exists; ADD-FULL = FULL already exists)
# ============================================================================
tier_additive() {
    echo ""
    echo "=========================================="
    echo "ADDITIVE ABLATION (9 jobs)"
    echo "=========================================="
    echo "NOTE: Requires ADD-M1M2, ADD-M1M2M3, ADD-M1M2M3M4 data on HF Hub."
    echo "Run 'python experiments/training/create_additive_ablation_data.py' first."

    local conditions=("ADD-M1M2" "ADD-M1M2M3" "ADD-M1M2M3M4")
    local seeds=(42 123 456)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/additive" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo "Done. After jobs complete, run:"
    echo "  python experiments/analysis/compute_additive_ablation_table.py"
}

# ============================================================================
# Tier: Table9 — Structured + smart-selection combinations (20 jobs)
# ============================================================================
tier_table9() {
    echo ""
    echo "=========================================="
    echo "TABLE 9 NEW ROWS (20 jobs)"
    echo "=========================================="
    echo "NOTE: Requires combined dataset files on HF Hub."
    echo "Run 'python experiments/training/create_table9_datasets.py' first."

    local conditions=("STRUCT4K-TFIDF2K" "STRUCT4K-LENGTH2K" "STRUCT4K-VOCAB2K" "STRUCT4K-ALL-BASELINES")
    local seeds=(42 123 456 789 2024)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "baselines" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo "Done. After jobs complete, sync and update tab:baselines in acl_latex.tex."
}

# ============================================================================
# Tier: Baselines-rerun — Decontaminated standalone baselines (15 jobs)
# ============================================================================
tier_baselines_rerun() {
    echo ""
    echo "=========================================="
    echo "BASELINES RERUN — decontaminated (15 jobs)"
    echo "=========================================="
    echo "NOTE: Run decontaminate_splits.py first, then re-upload baseline data."

    local conditions=("TF-IDF-DIVERSE" "LENGTH-STRATIFIED" "VOCAB-MAXIMIZED")
    local seeds=(42 123 456 789 2024)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "baselines" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo "Done. After jobs complete, update tab:baselines in acl_latex.tex."
}

# ============================================================================
# Tier: Ablations-rerun — Decontaminated ablation conditions (78 jobs)
# Tables 10-13: module_loo, module_size_ctrl, minimal_pairs, verb, pronoun
# ============================================================================
tier_ablations_rerun() {
    echo ""
    echo "=========================================="
    echo "ABLATIONS RERUN — decontaminated (78 jobs)"
    echo "=========================================="
    echo "NOTE: Run decontaminate_splits.py + upload_decontaminated_ablations.py first."

    local seeds=(42 123 456)

    # Table 10: module_loo (6 conditions × 3 seeds = 18 jobs)
    echo ""
    echo "--- Table 10: module_loo (18 jobs) ---"
    local module_loo=("FULL" "NO-PAST" "NO-NEGATION" "NO-QUESTIONS" "NO-FUTURE" "BASE-ONLY")
    for cond in "${module_loo[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/module_loo" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    # Table 11: module_size_ctrl (6 conditions × 3 seeds = 18 jobs)
    echo ""
    echo "--- Table 11: module_size_ctrl (18 jobs) ---"
    local size_ctrl=("FULL-1K" "NO-PAST-1K" "NO-QUEST-1K" "NO-NEG-1K" "NO-FUT-1K" "BASE-1K")
    for cond in "${size_ctrl[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/module_size_ctrl" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    # Table 12: minimal_pairs (2 conditions × 3 seeds = 6 jobs)
    echo ""
    echo "--- Table 12: minimal_pairs (6 jobs) ---"
    local pairs=("PAIRS-INTACT" "PAIRS-BROKEN")
    for cond in "${pairs[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/minimal_pairs" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    # Table 12: verb diversity (8 conditions × 3 seeds = 24 jobs)
    echo ""
    echo "--- Table 12: verb diversity (24 jobs) ---"
    local verbs=("1-VERB" "3-VERBS-a" "3-VERBS-b" "3-VERBS-c" "5-VERBS-a" "5-VERBS-b" "5-VERBS-c" "10-VERBS")
    for cond in "${verbs[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/verb" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    # Table 13: pronoun coverage (4 conditions × 3 seeds = 12 jobs)
    echo ""
    echo "--- Table 13: pronoun coverage (12 jobs) ---"
    local pronouns=("ALL-8" "MINIMAL-1" "REDUCED-4" "SINGULAR-3")
    for cond in "${pronouns[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/pronoun" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo ""
    echo "Done. After jobs complete, re-collect results and update Tables 10-13."
}

# ============================================================================
# Tier: NLLB-1.3B exp1 — Robustness table (20 jobs)
# ============================================================================
tier_nllb_1_3b_exp1() {
    echo ""
    echo "=========================================="
    echo "NLLB-1.3B EXP1 — Robustness table (20 jobs)"
    echo "=========================================="
    echo "Model: facebook/nllb-200-1.3B on a10g-large (OOM on a10g-small)"

    local model="facebook/nllb-200-1.3B"
    local conditions=("RANDOM-10K" "STRUCTURED-4K-ONLY" "RANDOM-6K_STRUCTURED-4K" "RANDOM-10K_STRUCTURED-4K")
    local seeds=(42 123 456 789 2024)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "exp1" "$cond" "$seed" "a10g-large" "$model" "16h"
        done
    done

    echo "Done. After jobs complete, run:"
    echo "  python experiments/analysis/compute_robustness_table.py"
}

# ============================================================================
# Main dispatch
# ============================================================================
TIER="${1:-all}"

case "$TIER" in
    fix)              tier_fix ;;
    additive)         tier_additive ;;
    table9)           tier_table9 ;;
    baselines-rerun)  tier_baselines_rerun ;;
    ablations-rerun)  tier_ablations_rerun ;;
    nllb-1.3b-exp1)   tier_nllb_1_3b_exp1 ;;
    all-decontaminated)
        tier_baselines_rerun
        tier_ablations_rerun
        ;;
    all)
        tier_fix
        echo ""
        echo "=== fix jobs launched. Run additive/table9 after creating datasets. ==="
        ;;
    *)
        echo "Unknown tier: $TIER"
        echo "Usage: $0 [fix|additive|table9|baselines-rerun|ablations-rerun|nllb-1.3b-exp1|all-decontaminated|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Total jobs launched: ${JOB_COUNT}"
echo "Total failures:      ${FAIL_COUNT}"
echo "=========================================="
