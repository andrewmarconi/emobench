#!/bin/bash
# Download models from Hugging Face Hub for SentiCompare benchmark

set -e  # Exit on error

echo "=========================================="
echo "SentiCompare - Model Download"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON_CMD="uv run python"

echo -e "${BLUE}Models will be cached to: ${CACHE_DIR}${NC}"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. Some models may not be accessible.${NC}"
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Models to download (from config/models.yaml)
MODELS=(
    "microsoft/Phi-3-mini-4k-instruct"
    "google/gemma-2-2b"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "Qwen/Qwen2.5-1.5B"
    "HuggingFaceTB/SmolLM2-1.7B"
    "distilbert-base-uncased"
    "roberta-base"
)

# Create download script
cat > /tmp/download_models.py << 'EOF'
"""Download models from Hugging Face Hub."""
import sys
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str) -> bool:
    """Download model and tokenizer."""
    try:
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Downloading model weights for {model_name}...")
        # Try sequence classification first, fall back to base model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        except Exception:
            model = AutoModel.from_pretrained(model_name)

        logger.info(f"✓ Successfully downloaded {model_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1]
    success = download_model(model_name)
    sys.exit(0 if success else 1)
EOF

# Download each model
SUCCESS_COUNT=0
FAIL_COUNT=0

for model in "${MODELS[@]}"; do
    echo -e "${BLUE}Downloading: ${model}${NC}"
    if $PYTHON_CMD /tmp/download_models.py "$model"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

# Clean up
rm /tmp/download_models.py

# Summary
echo -e "${GREEN}=========================================="
echo "Model download complete!"
echo "==========================================${NC}"
echo ""
echo "Successfully downloaded: $SUCCESS_COUNT models"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Failed to download: $FAIL_COUNT models${NC}"
fi
echo ""
echo "Models cached to: $CACHE_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify downloads: ls -lh $CACHE_DIR/hub/"
echo "  2. Start training: uv run python -m src.training.trainer --model distilbert-base-uncased"
echo ""
