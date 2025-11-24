"""
Evaluation CLI commands for SentiCompare.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_model(args):
    """
    Evaluate a single trained model.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Evaluating model: {args.model} on dataset: {args.dataset}")

    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = Path("experiments/results")

    try:
        # Import evaluation components
        from src.evaluation.benchmark import evaluate_model as evaluate_model_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        logger.info(f"Using device: {device}")

        # Evaluate the model
        evaluate_model_func(
            model_name=args.model,
            dataset_name=args.dataset,
            checkpoint_path=str(args.checkpoint),
            output_dir=str(args.output_dir),
            device=device,
        )

        logger.info(f"Evaluation completed! Results saved to: {args.output_dir}")

    except ImportError as e:
        logger.error(f"Evaluation module not implemented yet: {e}")
        logger.info("Please implement the evaluation module first")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def benchmark_models(args):
    """
    Run full benchmark suite on trained models.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Running benchmark on dataset: {args.dataset}")

    try:
        # Import evaluation components
        from src.evaluation.benchmark import benchmark_models as benchmark_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        logger.info(f"Using device: {device}")

        # Run benchmark
        benchmark_func(
            models_dir=str(args.models_dir),
            dataset_name=args.dataset,
            output_dir=str(args.output_dir),
            device=device,
        )

        logger.info(f"Benchmark completed! Results saved to: {args.output_dir}")

    except ImportError as e:
        logger.error(f"Evaluation module not implemented yet: {e}")
        logger.info("Please implement the evaluation module first")
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
