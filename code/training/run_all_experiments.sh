#!/bin/bash
# run_all_experiments.sh — Orchestrate the full experiment suite.
#
# Prerequisites:
#   1. Run prepare_splits.py to create all data splits
#   2. Run prepare_baselines.py to create baseline splits
#   3. Run prepare_ablations.py to create ablation splits
#
# Usage:
#   bash run_all_experiments.sh [--phase PHASE] [--dry-run]
#
# Phases:
#   1 = Core experiments (Exp 1 + Exp 2 + Baselines)
#   2 = Ablations (Exp 3 + module/pronoun/verb/pair ablations)
#   3 = Transfer & architecture (Exp 4 + Exp 5 + Exp 6)
#   all = Run everything

set -euo pipefail

PHASE="${1:---phase}"
PHASE_VAL="${2:-all}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE_VAL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPLITS_DIR="${SCRIPT_DIR}/../data/splits"
RESULTS_DIR="${SCRIPT_DIR}/../results"
TEST_FILE="${SPLITS_DIR}/shared/test.tsv"
MODEL="facebook/nllb-200-distilled-600M"
SEEDS="42 123 456 789 1024"
SEEDS_3="42 123 456"  # Fewer seeds for ablations

run_condition() {
    local experiment="$1"
    local condition="$2"
    local seeds="${3:-$SEEDS}"
    local extra_args="${4:-}"

    echo ""
    echo "================================================================"
    echo "  $experiment / $condition (seeds: $seeds)"
    echo "================================================================"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: python run_experiment.py --experiment $experiment --condition $condition --seeds $seeds $extra_args"
        return
    fi

    python "${SCRIPT_DIR}/run_experiment.py" \
        --experiment "$experiment" \
        --condition "$condition" \
        --splits-dir "$SPLITS_DIR" \
        --results-dir "$RESULTS_DIR" \
        --test-file "$TEST_FILE" \
        --model "$MODEL" \
        --seeds $seeds \
        --skip-existing \
        $extra_args
}

# ============================================================
# Phase 1: Core Experiments
# ============================================================
run_phase_1() {
    echo ""
    echo "========================================"
    echo "  PHASE 1: CORE EXPERIMENTS"
    echo "========================================"

    # Experiment 1: Primary Hypothesis
    for condition in RANDOM-10K RANDOM-6K_STRUCTURED-4K RANDOM-10K_STRUCTURED-4K \
                     STRUCTURED-4K-ONLY RANDOM-4K STRUCTURED-2K; do
        run_condition "exp1" "$condition" "$SEEDS"
    done

    # Experiment 2: Scaling Curves — Structured
    for size in 200 500 1000 2000 3000 4000; do
        run_condition "exp2" "STRUCTURED-$size" "$SEEDS"
    done

    # Experiment 2: Scaling Curves — Random
    for size in 200 500 1000 2000 4000 6000 8000 10000; do
        run_condition "exp2" "RANDOM-$size" "$SEEDS"
    done

    # Experiment 2: Scaling Curves — Combined additive
    for size in 500 1000 2000 4000; do
        run_condition "exp2" "RANDOM-6K_STRUCTURED-$size" "$SEEDS"
    done

    # Experiment 2: Scaling Curves — Combined replacement
    for struct_size in 500 1000 2000 4000; do
        random_size=$((10000 - struct_size))
        run_condition "exp2" "REPLACE-R${random_size}_S${struct_size}" "$SEEDS"
    done

    # Baselines (that need training)
    for baseline in LENGTH-STRATIFIED VOCAB-MAXIMIZED TF-IDF-DIVERSE; do
        run_condition "baselines" "$baseline" "$SEEDS"
    done

    echo ""
    echo "Phase 1 complete!"
}

# ============================================================
# Phase 2: Ablations
# ============================================================
run_phase_2() {
    echo ""
    echo "========================================"
    echo "  PHASE 2: ABLATIONS"
    echo "========================================"

    # Module leave-one-out
    for condition in FULL NO-NEGATION NO-PAST NO-FUTURE NO-QUESTIONS BASE-ONLY; do
        run_condition "ablations/module_loo" "$condition" "$SEEDS_3"
    done

    # Module size-controlled
    for condition in FULL-1K NO-NEG-1K NO-PAST-1K NO-FUT-1K NO-QUEST-1K BASE-1K; do
        run_condition "ablations/module_size_ctrl" "$condition" "$SEEDS_3"
    done

    # Pronoun coverage
    for condition in ALL-8 REDUCED-4 SINGULAR-3 MINIMAL-1; do
        run_condition "ablations/pronoun" "$condition" "$SEEDS_3"
    done

    # Verb diversity
    for condition in 10-VERBS 5-VERBS-a 5-VERBS-b 5-VERBS-c \
                     3-VERBS-a 3-VERBS-b 3-VERBS-c 1-VERB; do
        run_condition "ablations/verb" "$condition" "$SEEDS_3"
    done

    # Minimal-pair structure
    for condition in PAIRS-INTACT PAIRS-BROKEN; do
        run_condition "ablations/minimal_pairs" "$condition" "$SEEDS_3"
    done

    # Experiment 3: Curriculum vs Shuffled
    for condition in CURRICULUM SHUFFLED COMPETENCE; do
        run_condition "exp3" "$condition" "$SEEDS"
    done

    echo ""
    echo "Phase 2 complete!"
}

# ============================================================
# Phase 3: Transfer & Architecture
# ============================================================
run_phase_3() {
    echo ""
    echo "========================================"
    echo "  PHASE 3: TRANSFER & ARCHITECTURE"
    echo "========================================"

    # Experiment 4: Cross-verb generalization
    for condition in RUN1_TRAIN_RUN2_TEST RUN2_TRAIN_RUN1_TEST COMBINED_HELDOUT; do
        run_condition "exp4" "$condition" "$SEEDS"
    done

    # Experiment 5: Architecture comparison
    run_condition "exp5" "NLLB-200-600M" "$SEEDS" "--model facebook/nllb-200-distilled-600M"
    run_condition "exp5" "NLLB-200-1.3B" "$SEEDS" "--model facebook/nllb-200-1.3B --batch-size 8"
    run_condition "exp5" "MBART-50" "$SEEDS" "--model facebook/mbart-large-50-many-to-many-mmt"

    # Note: Transformer from-scratch requires a separate training script (not NLLB fine-tuning)
    echo "NOTE: Transformer-base and Transformer-tiny from scratch require a separate script."

    echo ""
    echo "Phase 3 complete!"
}

# ============================================================
# Main dispatcher
# ============================================================
echo "Experiment Suite: Data Composition vs. Quantity in Low-Resource NMT"
echo "Phase: $PHASE_VAL"
echo "Dry run: $DRY_RUN"
echo ""

case "$PHASE_VAL" in
    1)   run_phase_1 ;;
    2)   run_phase_2 ;;
    3)   run_phase_3 ;;
    all)
        run_phase_1
        run_phase_2
        run_phase_3
        ;;
    *)
        echo "Unknown phase: $PHASE_VAL"
        echo "Usage: bash run_all_experiments.sh --phase {1|2|3|all} [--dry-run]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  ALL REQUESTED PHASES COMPLETE"
echo "========================================"
