"""
Evaluation CLI commands for EmoBench.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def benchmark_models(args):
    """
    Run benchmark on selected models and datasets.

    Args:
        args: Parsed command line arguments
    """
    # Handle model selection
    if args.all_models and args.models:
        raise ValueError("Cannot specify both --models and --all-models. Choose one.")
    elif not args.all_models and not args.models:
        raise ValueError("Must specify either --models or --all-models.")

    if args.all_models:
        # Load all available models from registry
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry()
        models = list(registry.models.keys())
        logger.info(f"Benchmarking all {len(models)} available models: {models}")
    else:
        models = args.models
        logger.info(f"Benchmarking selected models: {models}")

    # Handle dataset selection
    if args.all_datasets and args.datasets:
        raise ValueError("Cannot specify both --datasets and --all-datasets. Choose one.")
    elif not args.all_datasets and not args.datasets:
        raise ValueError("Must specify either --datasets or --all-datasets.")

    if args.all_datasets:
        # Load all available datasets from config
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        datasets = list(config["datasets"].keys())
        logger.info(f"Benchmarking on all {len(datasets)} available datasets: {datasets}")
    else:
        datasets = args.datasets
        logger.info(f"Running benchmark on datasets: {datasets}")

    try:
        # Import evaluation components
        from src.evaluation.benchmark import benchmark_models as benchmark_func
        from src.utils.device import get_device

        # Get device
        device = get_device() if args.device == "auto" else args.device
        device_str = str(device) if hasattr(device, "type") else str(device)
        logger.info(f"Using device: {device_str}")

        # Run benchmark
        benchmark_func(
            models=models,
            datasets=datasets,
            checkpoints_dir=str(args.checkpoints_dir),
            output_dir=str(args.output_dir),
            device=device_str,
        )

        logger.info(f"Benchmark completed! Results saved to: {args.output_dir}")

    except ImportError as e:
        logger.error(f"Evaluation module not implemented yet: {e}")
        logger.info("Please implement the evaluation module first")
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
