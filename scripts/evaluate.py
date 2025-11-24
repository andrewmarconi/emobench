#!/usr/bin/env python
"""
Evaluation script for EmoBench models.

Performs comprehensive evaluation including:
- Classification metrics (accuracy, F1, precision, recall)
- Speed benchmarking (latency, throughput)
- Memory profiling
- Results aggregation

Usage:
    python scripts/evaluate.py --model-path ./experiments/checkpoints/DistilBERT-base_imdb/final --dataset imdb
    python scripts/evaluate.py --model-path ./model --dataset sst2 --batch-sizes 1,4,8,16
    python scripts/evaluate.py --model-path ./model --dataset imdb --skip-benchmark
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.loader import SentimentDataLoader, list_available_datasets
from src.evaluation.benchmark import SpeedBenchmark, print_benchmark_results
from src.evaluation.metrics import MetricsCalculator, print_metrics
from src.evaluation.profiler import MemoryProfiler, print_memory_stats, print_model_memory
from src.evaluation.results import ResultsAggregator
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
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment analysis model")

    # Add list-datasets first
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    # Required arguments (unless --list-datasets is used)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., imdb, sst2, amazon, yelp)",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./experiments/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for evaluation (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16",
        help="Comma-separated list of batch sizes for benchmarking",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples for evaluation",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip speed benchmarking",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory profiling",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for results aggregation",
    )

    return parser.parse_args()


def evaluate_metrics(model, tokenizer, dataset, device, max_samples=None):
    """
    Evaluate classification metrics.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for preprocessing
        dataset: Dataset to evaluate on
        device: Device to use
        max_samples: Maximum number of samples

    Returns:
        Dict: Evaluation metrics
    """
    logger.info("Evaluating classification metrics...")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    labels = []
    probabilities = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(dataset), 32):  # Process in batches of 32
            batch = dataset[i : i + 32]

            # Tokenize
            inputs = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict
            outputs = model(**inputs)
            logits = outputs.logits

            # Store results
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            probabilities.extend(probs)
            labels.extend(batch["label"])

    # Calculate metrics
    calculator = MetricsCalculator(num_classes=2)
    metrics = calculator.compute_metrics(
        predictions=np.array(predictions),
        labels=np.array(labels),
        probabilities=np.array(probabilities),
    )

    logger.info(f"Evaluated {len(predictions)} samples")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Handle list commands
    if args.list_datasets:
        datasets = list_available_datasets()
        print("\nAvailable Datasets:")
        print("=" * 60)
        for dataset in datasets:
            print(f"  - {dataset}")
        print("=" * 60 + "\n")
        return

    # Validate required arguments for evaluation
    if not args.model_path or not args.dataset:
        print("Error: --model-path and --dataset are required unless --list-datasets is used")
        print("Usage: python scripts/evaluate.py --model-path <path> --dataset <dataset>")
        print("Or:    python scripts/evaluate.py --list-datasets")
        sys.exit(1)

    # Print header
    print("\n" + "=" * 60)
    print("EmoBench - Model Evaluation".center(60))
    print("=" * 60)
    print(f"Model Path:  {args.model_path}")
    print(f"Dataset:     {args.dataset}")
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

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model_path_obj = Path(args.model_path)

    if not model_path_obj.exists():
        logger.error(f"Model path does not exist: {model_path_obj}")
        sys.exit(1)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path_obj)
        tokenizer = AutoTokenizer.from_pretrained(model_path_obj)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Print model memory
    print_model_memory(model, "Model Memory Footprint")

    # Load dataset
    try:
        loader = SentimentDataLoader(args.dataset, str(args.model_path))
        _, _, test_data = loader.load_and_prepare()
        logger.info(f"Loaded {len(test_data)} test samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Determine model name from path
    model_name = (
        Path(args.model_path).parent.name
        if Path(args.model_path).name == "final"
        else Path(args.model_path).name
    )

    # Create results aggregator
    aggregator = ResultsAggregator(experiment_name=args.experiment_name)

    # 1. Evaluate Classification Metrics
    print("\n" + "=" * 60)
    print("Classification Metrics".center(60))
    print("=" * 60)

    metrics = evaluate_metrics(model, tokenizer, test_data, device, max_samples=args.max_samples)
    print_metrics(metrics, "Classification Results")

    # 2. Speed Benchmarking
    benchmark_results = None
    if not args.skip_benchmark:
        print("\n" + "=" * 60)
        print("Speed Benchmarking".center(60))
        print("=" * 60)

        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]

        benchmark = SpeedBenchmark(
            model=model,
            tokenizer=tokenizer,
            device=device,
            warmup_steps=10,
            num_runs=100,
        )

        benchmark_results = benchmark.run_benchmark(
            dataset=test_data,
            batch_sizes=batch_sizes,
            max_samples=args.max_samples,
        )

        print_benchmark_results(benchmark_results)

    # 3. Memory Profiling
    memory_stats = None
    if not args.skip_memory:
        print("\n" + "=" * 60)
        print("Memory Profiling".center(60))
        print("=" * 60)

        profiler = MemoryProfiler(device)
        profiler.start()

        # Run some inference to measure memory
        sample_texts = test_data["text"][:100]
        for i in range(0, len(sample_texts), 16):
            batch = sample_texts[i : i + 16]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

        memory_stats = profiler.get_stats()
        print_memory_stats(memory_stats, "Memory Usage")

        peak_memory = profiler.get_peak_memory()
        print(f"Peak Memory: {peak_memory:.3f} GB\n")

    # 4. Save Results
    logger.info("Saving results...")

    aggregator.add_result(
        model_name=model_name,
        dataset_name=args.dataset,
        metrics=metrics,
        benchmark=benchmark_results,
        memory=memory_stats,
        metadata={
            "model_path": str(args.model_path),
            "device": str(device),
            "max_samples": args.max_samples,
        },
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregator.save(str(output_dir))

    # Print summary
    aggregator.print_summary()

    print("\n" + "=" * 60)
    print("Evaluation Complete!".center(60))
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
