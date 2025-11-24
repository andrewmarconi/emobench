"""
Training CLI commands for SentiCompare.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train_model(args):
    """
    Train a single model.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Training model: {args.model} on dataset: {args.dataset}")

    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = Path(f"experiments/checkpoints/{args.model}_{args.dataset}")

    try:
        # Import training components
        from src.training.trainer import train_model as train_model_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        logger.info(f"Using device: {device}")

        # Train the model
        train_model_func(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=str(args.output_dir),
            device=device,
        )

        logger.info(f"Training completed successfully! Model saved to: {args.output_dir}")

    except ImportError as e:
        logger.error(f"Training module not implemented yet: {e}")
        logger.info("Please implement the training module first")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def train_all_models(args):
    """
    Train all models on specified dataset(s).

    Args:
        args: Parsed command line arguments
    """
    logger.info("Training all models")

    try:
        # Import training components
        from src.training.trainer import train_all_models as train_all_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        logger.info(f"Using device: {device}")

        # Train all models
        train_all_func(dataset_name=args.dataset, model_names=args.models, device=device)

        logger.info("All training completed successfully!")

    except ImportError as e:
        logger.error(f"Training module not implemented yet: {e}")
        logger.info("Please implement the training module first")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
