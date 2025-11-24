#!/bin/bash
# Train a single sentiment analysis model
#
# Usage:
#   ./scripts/train_model.sh <model_alias> <dataset> [output_dir]
#
# Examples:
#   ./scripts/train_model.sh DistilBERT-base imdb
#   ./scripts/train_model.sh RoBERTa-base disney ./experiments/custom_output

set -e  # Exit on error

# Check for required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_alias> <dataset> [output_dir]"
    echo ""
    echo "Available models:"
    echo "  - DistilBERT-base"
    echo "  - RoBERTa-base"
    echo "  - Phi-3-mini"
    echo "  - Gemma-2-2B"
    echo "  - TinyLlama-1.1B"
    echo "  - Qwen2.5-1.5B"
    echo "  - SmolLM2-1.7B"
    echo ""
    echo "Available datasets:"
    echo "  - imdb"
    echo "  - sst2"
    echo "  - amazon"
    echo "  - yelp"
    echo "  - disney"
    exit 1
fi

MODEL_ALIAS=$1
DATASET=$2
OUTPUT_DIR=${3:-"./experiments/checkpoints/${MODEL_ALIAS}_${DATASET}"}

echo "=================================================="
echo "SentiCompare - Model Training"
echo "=================================================="
echo "Model:       $MODEL_ALIAS"
echo "Dataset:     $DATASET"
echo "Output Dir:  $OUTPUT_DIR"
echo "=================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create a Python training script
TRAIN_SCRIPT=$(cat <<'PYTHON'
import sys
import logging
import os
from pathlib import Path

# Load .env file for HF_TOKEN and other environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env' if Path(__file__).parent.name != '-' else Path.cwd() / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded environment variables from {env_path}")

from src.training.trainer import train_model
from src.utils.device import get_device, print_device_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    model_alias = sys.argv[1]
    dataset = sys.argv[2]
    output_dir = sys.argv[3]

    # Print device info
    print("\nDevice Information:")
    print("=" * 60)
    print_device_info()
    print("=" * 60)
    print()

    # Get device
    device = get_device()
    logger.info(f"Training on device: {device.type}")

    # Train model
    logger.info(f"Starting training: {model_alias} on {dataset}")
    trainer = train_model(
        model_name=model_alias,
        dataset_name=dataset,
        output_dir=output_dir,
        device=device
    )

    logger.info("Training completed successfully!")

    # Save final model
    final_output = f"{output_dir}/final"
    logger.info(f"Saving final model to {final_output}")
    trainer.model.save_pretrained(final_output)
    trainer.processing_class.save_pretrained(final_output)  # Updated from deprecated trainer.tokenizer

    logger.info("Model saved successfully!")

if __name__ == "__main__":
    main()
PYTHON
)

# Run training
echo "$TRAIN_SCRIPT" | uv run python - "$MODEL_ALIAS" "$DATASET" "$OUTPUT_DIR"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Training failed!"
    echo "=================================================="
    exit 1
fi
