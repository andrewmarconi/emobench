#!/bin/bash
# Train all models on all datasets
#
# Usage:
#   ./scripts/train_all.sh [dataset]
#
# If no dataset is specified, trains on all datasets
#
# Examples:
#   ./scripts/train_all.sh           # Train all models on all datasets
#   ./scripts/train_all.sh imdb      # Train all models on IMDB only

set -e  # Exit on error

# Default models and datasets
MODELS=(
    "DistilBERT-base"
    "RoBERTa-base"
    "Phi-3-mini"
    "Gemma-2-2B"
    "TinyLlama-1.1B"
    "Qwen2.5-1.5B"
    "SmolLM2-1.7B"
)

if [ $# -eq 1 ]; then
    DATASETS=("$1")
else
    DATASETS=(
        "imdb"
        "sst2"
        "amazon"
        "yelp"
    )
fi

# Output base directory
OUTPUT_BASE="./experiments/checkpoints"
LOG_DIR="./experiments/logs"
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="$LOG_DIR/training_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "=================================================="
echo "SentiCompare - Full Training Suite"
echo "=================================================="
echo "Models:   ${#MODELS[@]} models"
echo "Datasets: ${#DATASETS[@]} datasets"
echo "Total:    $((${#MODELS[@]} * ${#DATASETS[@]})) training runs"
echo "Output:   $OUTPUT_BASE"
echo "Logs:     $LOG_DIR"
echo "=================================================="
echo ""

# Initialize summary
{
    echo "SentiCompare Training Summary"
    echo "Started: $(date)"
    echo "=================================================="
    echo ""
} > "$SUMMARY_FILE"

# Track success/failure
TOTAL=0
SUCCESS=0
FAILED=0
FAILED_RUNS=()

# Train each model on each dataset
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        TOTAL=$((TOTAL + 1))

        echo ""
        echo "=================================================="
        echo "Run $TOTAL: $MODEL on $DATASET"
        echo "=================================================="

        OUTPUT_DIR="$OUTPUT_BASE/${MODEL}_${DATASET}"
        LOG_FILE="$LOG_DIR/${MODEL}_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

        # Run training and capture output
        START_TIME=$(date +%s)

        if ./scripts/train_model.sh "$MODEL" "$DATASET" "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            SUCCESS=$((SUCCESS + 1))

            echo "✓ Success: $MODEL on $DATASET (${DURATION}s)" | tee -a "$SUMMARY_FILE"
        else
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            FAILED=$((FAILED + 1))
            FAILED_RUNS+=("$MODEL on $DATASET")

            echo "✗ Failed: $MODEL on $DATASET (${DURATION}s)" | tee -a "$SUMMARY_FILE"
        fi

        echo ""
    done
done

# Print final summary
echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Total Runs:    $TOTAL"
echo "Successful:    $SUCCESS"
echo "Failed:        $FAILED"
    echo "Success Rate:  $(printf "%.1f" $(echo "scale=1; ($SUCCESS/$TOTAL)*100" | bc))%"
echo "=================================================="

# Append to summary file
{
    echo ""
    echo "=================================================="
    echo "Final Summary"
    echo "=================================================="
    echo "Completed: $(date)"
    echo "Total Runs:    $TOTAL"
    echo "Successful:    $SUCCESS"
    echo "Failed:        $FAILED"
echo "Success Rate:  $(printf "%.1f" $(echo "scale=1; ($SUCCESS/$TOTAL)*100" | bc))%"

    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "Failed Runs:"
        for RUN in "${FAILED_RUNS[@]}"; do
            echo "  - $RUN"
        done
    fi

    echo "=================================================="
} >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"

# Exit with error if any runs failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
