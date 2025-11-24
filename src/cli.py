"""
SentiCompare CLI - Multi-LLM Sentiment Analysis Benchmark Framework

Command-line interface for training, evaluating, and comparing sentiment analysis models.
"""

import argparse
import logging
import sys
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="SentiCompare - Multi-LLM Sentiment Analysis Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  senticompare train --model DistilBERT-base --dataset imdb
  senticompare train-all --dataset imdb
  senticompare evaluate --model DistilBERT-base --dataset imdb
  senticompare benchmark --models-dir experiments/checkpoints
  senticompare dashboard
  senticompare report --results-dir experiments/results
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--config-dir",
        "-c",
        type=Path,
        default=Path("config"),
        help="Configuration directory (default: config/)",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a single model")
    train_parser.add_argument("--model", "-m", required=True, help="Model name/alias to train")
    train_parser.add_argument("--dataset", "-d", required=True, help="Dataset name to train on")
    train_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for checkpoints (default: experiments/checkpoints/{model}_{dataset})",
    )
    train_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for training (default: auto)",
    )

    # Train-all command
    train_all_parser = subparsers.add_parser(
        "train-all", help="Train all models on specified dataset(s)"
    )
    train_all_parser.add_argument(
        "--dataset", "-d", help="Dataset name (if not specified, trains on all datasets)"
    )
    train_all_parser.add_argument(
        "--models", nargs="+", help="Specific models to train (default: all models)"
    )
    train_all_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for training (default: auto)",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a single trained model")
    evaluate_parser.add_argument(
        "--model", "-m", required=True, help="Model name/alias to evaluate"
    )
    evaluate_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset name to evaluate on"
    )
    evaluate_parser.add_argument(
        "--checkpoint", "-ckpt", type=Path, required=True, help="Path to model checkpoint"
    )
    evaluate_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for results (default: experiments/results/)",
    )
    evaluate_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for evaluation (default: auto)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run full benchmark suite on trained models"
    )
    benchmark_parser.add_argument(
        "--models-dir",
        "-m",
        type=Path,
        default=Path("experiments/checkpoints"),
        help="Directory containing trained model checkpoints",
    )
    benchmark_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset name to benchmark on"
    )
    benchmark_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory for benchmark results",
    )
    benchmark_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for benchmarking (default: auto)",
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch interactive Streamlit dashboard"
    )
    dashboard_parser.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        default=Path("experiments/evaluation"),
        help="Directory containing benchmark results",
    )
    dashboard_parser.add_argument(
        "--port", "-p", type=int, default=8501, help="Port to run dashboard on (default: 8501)"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comparison reports")
    report_parser.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        default=Path("experiments/results"),
        help="Directory containing benchmark results",
    )
    report_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("experiments/reports"),
        help="Output directory for reports",
    )
    report_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="Report format (default: all)",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"SentiCompare CLI - Command: {args.command}")

    try:
        # Import here to avoid circular imports and allow CLI to work even if modules are incomplete
        if args.command == "train":
            from src.training.cli import train_model

            train_model(args)
        elif args.command == "train-all":
            from src.training.cli import train_all_models

            train_all_models(args)
        elif args.command == "evaluate":
            from src.evaluation.cli import evaluate_model

            evaluate_model(args)
        elif args.command == "benchmark":
            from src.evaluation.cli import benchmark_models

            benchmark_models(args)
        elif args.command == "dashboard":
            from src.visualization.cli import launch_dashboard

            launch_dashboard(args)
        elif args.command == "report":
            from src.visualization.cli import generate_reports

            generate_reports(args)
        else:
            parser.error(f"Unknown command: {args.command}")

    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        logger.error("Make sure all dependencies are installed and modules are implemented")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
