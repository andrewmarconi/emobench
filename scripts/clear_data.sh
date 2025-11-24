#!/bin/bash

# Script to clear all generated data from the project
# This removes processed data, model checkpoints, evaluation results, logs, and results
# while preserving directory structure and .gitkeep files

set -e  # Exit on any error

echo "Clearing project data..."

# Clear data directories (keep .gitkeep files)
find data/processed -type f -not -name '.gitkeep' -delete
find data/raw -type f -not -name '.gitkeep' -delete
find data/splits -type f -not -name '.gitkeep' -delete

# Clear experiments directories
find experiments/checkpoints -type f -not -name '.gitkeep' -delete
find experiments/evaluation -type f -not -name '.gitkeep' -delete
find experiments/logs -type f -not -name '.gitkeep' -delete
find experiments/results -type f -not -name '.gitkeep' -delete

# Clear any __pycache__ directories
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "Data cleared successfully."