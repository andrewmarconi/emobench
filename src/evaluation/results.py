"""
Results aggregation and storage for SentiCompare.

Collects, stores, and analyzes evaluation results from multiple models and datasets.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    Aggregate and store evaluation results.

    Collects results from multiple model/dataset combinations and provides
    utilities for analysis and comparison.

    Examples:
        >>> aggregator = ResultsAggregator()
        >>> aggregator.add_result("DistilBERT-base", "imdb", metrics, benchmark, memory)
        >>> aggregator.save("./experiments/results")
        >>> df = aggregator.to_dataframe()
    """

    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize results aggregator.

        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []

    def add_result(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict[str, float],
        benchmark: Optional[Dict] = None,
        memory: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add a result to the aggregator.

        Args:
            model_name: Model name or alias
            dataset_name: Dataset name
            metrics: Classification metrics
            benchmark: Speed benchmark results
            memory: Memory profiling results
            metadata: Additional metadata
        """
        result = {
            "experiment": self.experiment_name,
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        if benchmark is not None:
            result["benchmark"] = benchmark

        if memory is not None:
            result["memory"] = memory

        if metadata is not None:
            result["metadata"] = metadata

        self.results.append(result)
        logger.info(f"Added result: {model_name} on {dataset_name}")

    def to_dataframe(self, include_benchmark: bool = True, include_memory: bool = True) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Args:
            include_benchmark: Include benchmark metrics
            include_memory: Include memory metrics

        Returns:
            pd.DataFrame: Results as DataFrame
        """
        if not self.results:
            return pd.DataFrame()

        rows = []

        for result in self.results:
            row = {
                "experiment": result["experiment"],
                "model": result["model"],
                "dataset": result["dataset"],
                "timestamp": result["timestamp"],
            }

            # Add metrics
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    row[f"metric_{key}"] = value

            # Add benchmark data
            if include_benchmark and "benchmark" in result:
                benchmark = result["benchmark"]

                # Extract key benchmark metrics
                if "batch_sizes" in benchmark:
                    # Get metrics for batch_size=1 (single sample latency)
                    if 1 in benchmark["batch_sizes"]:
                        bs1 = benchmark["batch_sizes"][1]
                        row["latency_mean_ms"] = bs1["latency"]["mean_ms"]
                        row["latency_p95_ms"] = bs1["latency"]["p95_ms"]

                    # Get best throughput
                    if "best_throughput_batch_size" in benchmark:
                        best_bs = benchmark["best_throughput_batch_size"]
                        if best_bs in benchmark["batch_sizes"]:
                            best = benchmark["batch_sizes"][best_bs]
                            row["throughput_samples_per_sec"] = best["throughput"][
                                "throughput_samples_per_sec"
                            ]
                            row["best_throughput_batch_size"] = best_bs

            # Add memory data
            if include_memory and "memory" in result:
                memory = result["memory"]
                if "peak_allocated_gb" in memory:
                    row["peak_memory_gb"] = memory["peak_allocated_gb"]
                elif "allocated_gb" in memory:
                    row["peak_memory_gb"] = memory["allocated_gb"]

            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, output_dir: str) -> None:
        """
        Save results to disk.

        Saves as both JSON and CSV formats.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_file = output_path / f"{self.experiment_name}_results.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "experiment": self.experiment_name,
                    "num_results": len(self.results),
                    "results": self.results,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved JSON results to {json_file}")

        # Save as CSV
        df = self.to_dataframe()
        if not df.empty:
            csv_file = output_path / f"{self.experiment_name}_results.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved CSV results to {csv_file}")

    def load(self, json_file: str) -> None:
        """
        Load results from JSON file.

        Args:
            json_file: Path to JSON results file
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        self.experiment_name = data.get("experiment", self.experiment_name)
        self.results = data.get("results", [])

        logger.info(f"Loaded {len(self.results)} results from {json_file}")

    def get_summary(self) -> Dict:
        """
        Get summary statistics across all results.

        Returns:
            Dict: Summary statistics
        """
        if not self.results:
            return {}

        df = self.to_dataframe()

        # Get metric columns
        metric_cols = [col for col in df.columns if col.startswith("metric_")]

        summary = {
            "experiment": self.experiment_name,
            "num_results": len(self.results),
            "models": df["model"].unique().tolist(),
            "datasets": df["dataset"].unique().tolist(),
        }

        # Aggregate metrics
        if metric_cols:
            summary["metrics"] = {}
            for col in metric_cols:
                metric_name = col.replace("metric_", "")
                summary["metrics"][metric_name] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        return summary

    def get_best_models(
        self, metric: str = "f1", n: int = 5, per_dataset: bool = False
    ) -> pd.DataFrame:
        """
        Get top N models by a specific metric.

        Args:
            metric: Metric to sort by
            n: Number of top models to return
            per_dataset: Return top N per dataset

        Returns:
            pd.DataFrame: Top models
        """
        df = self.to_dataframe()

        if df.empty:
            return df

        metric_col = f"metric_{metric}"
        if metric_col not in df.columns:
            logger.warning(f"Metric '{metric}' not found in results")
            return pd.DataFrame()

        if per_dataset:
            # Get top N per dataset
            result_dfs = []
            for dataset in df["dataset"].unique():
                dataset_df = df[df["dataset"] == dataset]
                top_n = dataset_df.nlargest(n, metric_col)
                result_dfs.append(top_n)
            return pd.concat(result_dfs, ignore_index=True)
        else:
            # Get top N overall
            return df.nlargest(n, metric_col)

    def compare_models(
        self, model1: str, model2: str, dataset: Optional[str] = None
    ) -> Dict:
        """
        Compare two models.

        Args:
            model1: First model name
            model2: Second model name
            dataset: Dataset to compare on (None = all datasets)

        Returns:
            Dict: Comparison results
        """
        df = self.to_dataframe()

        # Filter by models
        df1 = df[df["model"] == model1]
        df2 = df[df["model"] == model2]

        if dataset:
            df1 = df1[df1["dataset"] == dataset]
            df2 = df2[df2["dataset"] == dataset]

        if df1.empty or df2.empty:
            logger.warning(f"No results found for comparison")
            return {}

        # Get metric columns
        metric_cols = [col for col in df.columns if col.startswith("metric_")]

        comparison = {
            "model1": model1,
            "model2": model2,
            "dataset": dataset or "all",
            "metrics": {},
        }

        for col in metric_cols:
            metric_name = col.replace("metric_", "")

            val1 = df1[col].mean()
            val2 = df2[col].mean()

            comparison["metrics"][metric_name] = {
                model1: float(val1),
                model2: float(val2),
                "diff": float(val2 - val1),
                "improvement_pct": float((val2 - val1) / val1 * 100) if val1 != 0 else 0.0,
            }

        return comparison

    def print_summary(self) -> None:
        """Print summary of results."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("Results Summary".center(60))
        print("=" * 60)
        print(f"Experiment:  {summary.get('experiment', 'N/A')}")
        print(f"Num Results: {summary.get('num_results', 0)}")
        print(f"Models:      {', '.join(summary.get('models', []))}")
        print(f"Datasets:    {', '.join(summary.get('datasets', []))}")

        if "metrics" in summary:
            print("\n" + "-" * 60)
            print("Metric Aggregates (across all results):".center(60))
            print("-" * 60)

            for metric, stats in summary["metrics"].items():
                print(f"\n{metric}:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std:  {stats['std']:.4f}")
                print(f"  Min:  {stats['min']:.4f}")
                print(f"  Max:  {stats['max']:.4f}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: Results aggregation
    print("Results Aggregator Module")
    print("=" * 60)

    # Create aggregator
    aggregator = ResultsAggregator(experiment_name="demo_experiment")

    # Add some sample results
    models = ["DistilBERT-base", "RoBERTa-base"]
    datasets = ["imdb", "sst2"]

    for model in models:
        for dataset in datasets:
            # Sample metrics
            metrics = {
                "accuracy": 0.85 + (hash(model + dataset) % 10) / 100,
                "f1": 0.84 + (hash(model + dataset) % 10) / 100,
                "precision": 0.83 + (hash(model + dataset) % 10) / 100,
                "recall": 0.85 + (hash(model + dataset) % 10) / 100,
            }

            # Sample benchmark
            benchmark = {
                "batch_sizes": {
                    1: {
                        "latency": {"mean_ms": 15.5, "p95_ms": 18.2},
                        "throughput": {"throughput_samples_per_sec": 64.5},
                    }
                },
                "best_throughput_batch_size": 1,
            }

            # Sample memory
            memory = {"peak_allocated_gb": 1.2}

            aggregator.add_result(model, dataset, metrics, benchmark, memory)

    # Print summary
    aggregator.print_summary()

    # Get top models
    print("\nTop 3 Models by F1:")
    print("=" * 60)
    top_models = aggregator.get_best_models(metric="f1", n=3)
    print(top_models[["model", "dataset", "metric_f1"]])

    # Compare models
    print("\nModel Comparison:")
    print("=" * 60)
    comparison = aggregator.compare_models("DistilBERT-base", "RoBERTa-base")
    for metric, data in comparison["metrics"].items():
        print(f"{metric}: {data[comparison['model1']]:.4f} vs {data[comparison['model2']]:.4f} "
              f"(diff: {data['diff']:+.4f}, {data['improvement_pct']:+.2f}%)")
