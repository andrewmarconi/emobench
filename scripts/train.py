#!/usr/bin/env python
"""
Training script for SentiCompare models.

This script provides a command-line interface for training sentiment analysis
models with LoRA fine-tuning.

Usage:
    python scripts/train.py --model DistilBERT-base --dataset imdb
    python scripts/train.py --model RoBERTa-base --dataset imdb --output ./custom/output
    python scripts/train.py --model Phi-3-mini --dataset sst2 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data.loader import SentimentDataLoader, list_available_datasets
from src.models.lora_config import prepare_model
from src.models.model_registry import ModelRegistry, list_all_models
from src.training.trainer import SentiCompareTrainer
from src.utils.device import get_device, print_device_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a sentiment analysis model with LoRA fine-tuning"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model alias (e.g., DistilBERT-base, RoBERTa-base)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., imdb, sst2, amazon, yelp)",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: ./experiments/checkpoints/<model>_<dataset>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training (default: auto-detect)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (default: senticompare)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--model-info",
        type=str,
        default=None,
        help="Show information for a specific model and exit",
    )

    return parser.parse_args()


def list_models():
    """List all available models."""
    registry = ModelRegistry()
    models = registry.list_models()

    print("\nAvailable Models:")
    print("=" * 60)
    for model in models:
        config = registry.get_model_config(model)
        print(f"  {model:<20} ({config['size_params']:<8} - {config['architecture']})")
    print("=" * 60)
    print(f"Total: {len(models)} models\n")


def list_datasets():
    """List all available datasets."""
    datasets = list_available_datasets()

    print("\nAvailable Datasets:")
    print("=" * 60)
    for dataset in datasets:
        from src.data.loader import SentimentDataLoader

        info = SentimentDataLoader.SUPPORTED_DATASETS[dataset]
        print(f"  {dataset:<15} ({info['source']:<12} - {info['name']})")
    print("=" * 60)
    print(f"Total: {len(datasets)} datasets\n")


def show_model_info(model_alias: str, device_str: str = None):
    """Show detailed information for a model."""
    registry = ModelRegistry()

    # Get device
    if device_str:
        device = get_device(force_device=device_str)
    else:
        device = get_device()

    registry.print_model_info(model_alias, device)


def main():
    """Main training function."""
    args = parse_args()

    # Handle list commands
    if args.list_models:
        list_models()
        return

    if args.list_datasets:
        list_datasets()
        return

    if args.model_info:
        show_model_info(args.model_info, args.device)
        return

    # Print header
    print("\n" + "=" * 60)
    print("SentiCompare - Model Training".center(60))
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Config:      {args.config}")
    print("=" * 60)
    print()

    # Print device information
    print("Device Information:")
    print("=" * 60)
    print_device_info()
    print("=" * 60)
    print()

    # Get device
    if args.device:
        device = get_device(force_device=args.device)
    else:
        device = get_device()

    logger.info(f"Using device: {device.type}")

    # Validate model
    registry = ModelRegistry()
    available_models = registry.list_models()
    if args.model not in available_models:
        logger.error(f"Model '{args.model}' not found.")
        logger.error(f"Available models: {', '.join(available_models)}")
        sys.exit(1)

    # Validate dataset
    available_datasets = list_available_datasets()
    if args.dataset not in available_datasets:
        logger.error(f"Dataset '{args.dataset}' not found.")
        logger.error(f"Available datasets: {', '.join(available_datasets)}")
        sys.exit(1)

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"./experiments/checkpoints/{args.model}_{args.dataset}"

    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model_name = registry.get_model_name(args.model)
    model = prepare_model(model_name, num_labels=2, device=device)

    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    loader = SentimentDataLoader(args.dataset, model_name)
    train_data, val_data, test_data = loader.load_and_prepare()

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples:   {len(val_data)}")
    logger.info(f"Test samples:  {len(test_data)}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = SentiCompareTrainer(
        model=model,
        tokenizer=loader.tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        output_dir=output_dir,
        config_path=args.config,
        experiment_name=args.experiment_name,
        device=device,
    )

    # Train
    logger.info("Starting training...")
    print("\n" + "=" * 60)
    print("Training Started".center(60))
    print("=" * 60)
    print()

    trained_trainer = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!".center(60))
    print("=" * 60)
    print()

    # Save final model
    final_output = f"{output_dir}/final"
    logger.info(f"Saving final model to {final_output}")
    trainer.save_model(final_output)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trained_trainer.evaluate(test_data)

    print("\n" + "=" * 60)
    print("Test Set Results".center(60))
    print("=" * 60)
    print(f"Loss:       {test_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"Accuracy:   {test_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"F1:         {test_metrics.get('eval_f1', 'N/A'):.4f}")
    print(f"Precision:  {test_metrics.get('eval_precision', 'N/A'):.4f}")
    print(f"Recall:     {test_metrics.get('eval_recall', 'N/A'):.4f}")
    print("=" * 60)
    print()

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
