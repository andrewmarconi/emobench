#!/bin/bash
# Download and prepare datasets for SentiCompare benchmark

set -e  # Exit on error

echo "=========================================="
echo "SentiCompare - Data Preparation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="${DATA_DIR:-./data/raw}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# Create directories
echo -e "${BLUE}Creating data directories...${NC}"
mkdir -p "$DATA_DIR"
mkdir -p ./data/processed
mkdir -p ./data/splits

# Check Python environment
echo -e "${BLUE}Checking Python environment...${NC}"
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.12+"
    exit 1
fi

PYTHON_CMD="uv run python"

# Download datasets
echo -e "${BLUE}Downloading datasets...${NC}"
echo ""

# Create a Python script to download all datasets
cat > /tmp/download_datasets.py << 'EOF'
"""Download all datasets for SentiCompare."""
import logging
from src.data.loader import SentimentDataLoader, list_available_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    datasets = list_available_datasets()
    logger.info(f"Downloading {len(datasets)} datasets...")

    # Use a lightweight tokenizer for downloading
    tokenizer = "distilbert-base-uncased"

    for dataset_name in datasets:
        logger.info(f"\nDownloading: {dataset_name}")
        try:
            loader = SentimentDataLoader(dataset_name, tokenizer)
            data = loader.load()
            logger.info(f"✓ Successfully loaded {dataset_name}")

            # Print basic info
            if hasattr(data, 'num_rows'):
                logger.info(f"  Rows: {data.num_rows}")
            elif isinstance(data, dict):
                total = sum(len(split) for split in data.values())
                logger.info(f"  Total rows: {total}")

        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_name}: {e}")
            continue

    logger.info("\nDataset download complete!")

if __name__ == "__main__":
    main()
EOF

# Run download script
echo -e "${GREEN}Starting dataset downloads...${NC}"
$PYTHON_CMD /tmp/download_datasets.py

# Clean up
rm /tmp/download_datasets.py

echo ""
echo -e "${GREEN}=========================================="
echo "Data preparation complete!"
echo "==========================================${NC}"
echo ""
echo "Datasets downloaded to: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Explore data with: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "  2. Start training with: uv run python -m src.training.trainer"
echo ""
