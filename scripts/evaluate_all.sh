#!/bin/bash
# Evaluate all trained models
#
# Usage:
#   ./scripts/evaluate_all.sh [checkpoints_dir] [output_dir]
#
# Examples:
#   ./scripts/evaluate_all.sh
#   ./scripts/evaluate_all.sh ./experiments/checkpoints ./experiments/evaluation

set -e  # Exit on error

# Default directories
CHECKPOINTS_DIR=${1:-"./experiments/checkpoints"}
OUTPUT_DIR=${2:-"./experiments/evaluation"}
EXPERIMENT_NAME="full_evaluation_$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "SentiCompare - Full Evaluation Suite"
echo "=================================================="
echo "Checkpoints: $CHECKPOINTS_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Experiment:  $EXPERIMENT_NAME"
echo "=================================================="
echo ""

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track success/failure
TOTAL=0
SUCCESS=0
FAILED=0
FAILED_RUNS=()

# Find all final model directories
find "$CHECKPOINTS_DIR" -type d -name "final" | while read MODEL_DIR; do
    # Extract model and dataset from path
    # Expected format: checkpoints/ModelName_DatasetName/final
    PARENT_DIR=$(dirname "$MODEL_DIR")
    MODEL_DATASET=$(basename "$PARENT_DIR")

    # Try to extract dataset name (assumes format: Model_Dataset)
    if [[ $MODEL_DATASET =~ _([^_]+)$ ]]; then
        DATASET="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not extract dataset from $MODEL_DATASET, skipping..."
        continue
    fi

    TOTAL=$((TOTAL + 1))

    echo ""
    echo "=================================================="
    echo "Evaluating: $MODEL_DATASET"
    echo "=================================================="

    # Run evaluation
    if uv run python scripts/evaluate.py \
        --model-path "$MODEL_DIR" \
        --dataset "$DATASET" \
        --output "$OUTPUT_DIR" \
        --experiment-name "$EXPERIMENT_NAME" \
        --max-samples 1000 \
        --batch-sizes "1,4,8,16"; then

        SUCCESS=$((SUCCESS + 1))
        echo "✓ Success: $MODEL_DATASET"
    else
        FAILED=$((FAILED + 1))
        FAILED_RUNS+=("$MODEL_DATASET")
        echo "✗ Failed: $MODEL_DATASET"
    fi
done

# Print final summary
echo ""
echo "=================================================="
echo "Evaluation Complete!"
echo "=================================================="
echo "Total Models:  $TOTAL"
echo "Successful:    $SUCCESS"
echo "Failed:        $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed Evaluations:"
    for RUN in "${FAILED_RUNS[@]}"; do
        echo "  - $RUN"
    done
fi

echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="

# Exit with error if any runs failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
