"""
Result aggregation utilities for SentiCompare.

Handles loading, combining, and aggregating benchmark results from multiple models
and experiments into unified DataFrames for analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .ranking import ModelRanker

logger = logging.getLogger(__name__)


class BenchmarkAggregator:
    """
    Aggregate and analyze benchmark results from multiple models.

    Examples:
        >>> aggregator = BenchmarkAggregator()
        >>> aggregator.load_results("experiments/results/")
        >>> aggregated = aggregator.aggregate_all()
        >>> rankings = aggregator.rank_models(["f1", "accuracy"])
    """

    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize aggregator.

        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir) if results_dir else Path("experiments/evaluation")
        self.raw_results = {}
        self.aggregated_results = None
        self.normalization_params = {}

    def load_results(
        self,
        source: Union[str, Path, Dict[str, pd.DataFrame], List[Dict]],
        pattern: str = "*.json",
        recursive: bool = True,
    ) -> None:
        """
        Load benchmark results from various sources.

        Args:
            source: Source of results (directory, file, dict, or list of result dicts)
            pattern: File pattern to match (for directory sources)
            recursive: Whether to search recursively
        """
        if isinstance(source, (str, Path)):
            self._load_from_directory(Path(source), pattern, recursive)
        elif isinstance(source, dict):
            if all(isinstance(v, pd.DataFrame) for v in source.values()):
                self.raw_results = source
            else:
                self._load_from_dict(source)
        elif isinstance(source, list):
            self._load_from_list(source)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        logger.info(f"Loaded results for {len(self.raw_results)} models")

    def _load_from_directory(self, directory: Path, pattern: str, recursive: bool) -> None:
        """Load results from directory containing JSON/YAML files."""
        if not directory.exists():
            logger.warning(f"Results directory not found: {directory}")
            return

        # Find result files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        logger.info(f"Found {len(files)} result files in {directory}")

        for file_path in files:
            try:
                # Load based on extension
                if file_path.suffix.lower() == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                elif file_path.suffix.lower() in [".yaml", ".yml"]:
                    with open(file_path, "r") as f:
                        data = yaml.safe_load(f)
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    continue

                # Extract model name from filename or data
                model_name = data.get("model_name", file_path.stem)

                # Convert to DataFrame
                if "results" in data:
                    df = pd.DataFrame(data["results"])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    # Single result entry
                    df = pd.DataFrame([data])

                # Add metadata
                df["model_name"] = model_name
                df["source_file"] = str(file_path)

                self.raw_results[model_name] = df

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

    def _load_from_dict(self, data: Dict) -> None:
        """Load results from dictionary format."""
        for model_name, results in data.items():
            if isinstance(results, list):
                df = pd.DataFrame(results)
            elif isinstance(results, dict):
                df = pd.DataFrame([results])
            else:
                logger.warning(f"Unsupported data format for {model_name}")
                continue

            df["model_name"] = model_name
            self.raw_results[model_name] = df

    def _load_from_list(self, data: List[Dict]) -> None:
        """Load results from list of result dictionaries."""
        # Group by model name
        model_results = {}
        for result in data:
            model_name = result.get("model_name", "unknown")
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        # Convert to DataFrames
        for model_name, results in model_results.items():
            self.raw_results[model_name] = pd.DataFrame(results)

    def aggregate_all(
        self,
        group_by: Optional[List[str]] = None,
        agg_functions: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate all loaded results into a unified DataFrame.

        Args:
            group_by: Columns to group by (default: ["model_name", "dataset"])
            agg_functions: Aggregation functions for columns

        Returns:
            pd.DataFrame: Aggregated results
        """
        if not self.raw_results:
            logger.warning("No results loaded to aggregate")
            return pd.DataFrame()

        # Combine all results
        all_dfs = []
        for model_name, df in self.raw_results.items():
            if df.empty:
                continue
            all_dfs.append(df)

        if not all_dfs:
            logger.warning("No non-empty DataFrames to aggregate")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Set default grouping
        if group_by is None:
            group_by = ["model_name"]
            if "dataset" in combined_df.columns:
                group_by.append("dataset")

        # Set default aggregation functions
        if agg_functions is None:
            agg_functions = {}

            # Only aggregate numeric columns
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

            # Auto-detect metric columns (numeric only)
            metric_cols = [col for col in numeric_cols if col.startswith("metric_")]
            for col in metric_cols:
                agg_functions[col] = ["mean", "std", "min", "max", "count"]

            # Add speed/memory metrics (numeric only)
            speed_cols = [
                col
                for col in numeric_cols
                if any(x in col.lower() for x in ["latency", "throughput", "memory", "time"])
            ]
            for col in speed_cols:
                agg_functions[col] = ["mean", "std", "min", "max"]

            # Add count for non-numeric grouping columns
            if not agg_functions:  # If no numeric columns found
                agg_functions = {"model_name": "count"}  # Just count occurrences

        # For now, just return the combined data without complex aggregation
        # TODO: Fix the aggregation logic to handle MultiIndex columns properly
        aggregated = combined_df

        self.aggregated_results = aggregated
        logger.info(f"Aggregated {len(combined_df)} results into {len(aggregated)} rows")

        return aggregated

    def normalize_metrics(
        self,
        metrics: Optional[List[str]] = None,
        method: str = "minmax",
        range_bounds: Tuple[float, float] = (0, 1),
    ) -> pd.DataFrame:
        """
        Normalize metrics to specified range.

        Args:
            metrics: List of metrics to normalize (None = auto-detect)
            method: Normalization method ("minmax", "zscore", "robust")
            range_bounds: Target range for minmax normalization

        Returns:
            pd.DataFrame: DataFrame with normalized metrics
        """
        if self.aggregated_results is None:
            logger.warning("No aggregated results to normalize")
            return pd.DataFrame()

        df = self.aggregated_results.copy()

        # Auto-detect metrics if not provided
        if metrics is None:
            metric_cols = [col for col in df.columns if col.startswith("metric_")]
        else:
            metric_cols = [f"metric_{m}" if not m.startswith("metric_") else m for m in metrics]

        normalized_cols = []

        for col in metric_cols:
            if col not in df.columns:
                logger.warning(f"Metric column '{col}' not found")
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            if method == "minmax":
                min_val = values.min()
                max_val = values.max()
                normalized = (df[col] - min_val) / (max_val - min_val)
                normalized = normalized * (range_bounds[1] - range_bounds[0]) + range_bounds[0]

                # Store parameters for inverse transform
                self.normalization_params[col] = {
                    "method": "minmax",
                    "min": min_val,
                    "max": max_val,
                    "range": range_bounds,
                }

            elif method == "zscore":
                mean_val = values.mean()
                std_val = values.std()
                normalized = (df[col] - mean_val) / std_val

                self.normalization_params[col] = {
                    "method": "zscore",
                    "mean": mean_val,
                    "std": std_val,
                }

            elif method == "robust":
                median_val = values.median()
                mad_val = (values - median_val).abs().median()
                normalized = (df[col] - median_val) / mad_val

                self.normalization_params[col] = {
                    "method": "robust",
                    "median": median_val,
                    "mad": mad_val,
                }

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            normalized_col = f"{col}_normalized"
            df[normalized_col] = normalized
            normalized_cols.append(normalized_col)

        logger.info(f"Normalized {len(normalized_cols)} metrics using {method} method")
        return df

    def rank_models(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None,
        ranking_method: str = "composite",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Rank models using specified metrics and method.

        Args:
            metrics: Metrics to use for ranking
            weights: Weights for each metric (None = equal weights)
            ranking_method: Method to use ("composite", "pareto", "efficiency")
            **kwargs: Additional arguments for ranking method

        Returns:
            pd.DataFrame: Model rankings
        """
        if self.aggregated_results is None:
            logger.warning("No aggregated results to rank")
            return pd.DataFrame()

        # Create ranker
        ranker = ModelRanker(self.aggregated_results)

        # Route to appropriate ranking method
        if ranking_method == "composite":
            perf_weight = weights[0] if weights and len(weights) > 0 else 0.7
            speed_weight = weights[1] if weights and len(weights) > 1 else 0.3
            return ranker.rank_by_composite(
                performance_metrics=metrics,
                performance_weight=perf_weight,
                speed_weight=speed_weight,
                **kwargs,
            )
        elif ranking_method == "pareto":
            if len(metrics) != 2:
                raise ValueError("Pareto ranking requires exactly 2 metrics")
            return ranker.get_pareto_frontier(metrics[0], metrics[1], **kwargs)
        elif ranking_method == "efficiency":
            if len(metrics) != 2:
                raise ValueError("Efficiency ranking requires exactly 2 metrics")
            return ranker.rank_by_efficiency(
                performance_metric=metrics[0],
                resource_metric=metrics[1],
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported ranking method: {ranking_method}")

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for all loaded results.

        Returns:
            Dict: Summary statistics
        """
        if not self.raw_results:
            return {}

        summary = {
            "total_models": len(self.raw_results),
            "model_names": list(self.raw_results.keys()),
            "total_results": sum(len(df) for df in self.raw_results.values()),
        }

        # Get dataset information
        all_datasets = set()
        for df in self.raw_results.values():
            if "dataset" in df.columns:
                all_datasets.update(df["dataset"].unique())

        summary["datasets"] = list(all_datasets)
        summary["num_datasets"] = len(all_datasets)

        # Get metric information
        all_metrics = set()
        for df in self.raw_results.values():
            metric_cols = [col for col in df.columns if col.startswith("metric_")]
            all_metrics.update(metric_cols)

        summary["metrics"] = [col.replace("metric_", "") for col in all_metrics]
        summary["num_metrics"] = len(all_metrics)

        return summary

    def export_results(
        self,
        output_path: Union[str, Path],
        format: str = "json",
        include_raw: bool = False,
        include_aggregated: bool = True,
    ) -> None:
        """
        Export results to file.

        Args:
            output_path: Output file path
            format: Export format ("json", "csv", "excel", "yaml")
            include_raw: Include raw results
            include_aggregated: Include aggregated results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {}

        if include_raw and self.raw_results:
            export_data["raw_results"] = {
                model: df.to_dict("records") for model, df in self.raw_results.items()
            }

        if include_aggregated and self.aggregated_results is not None:
            export_data["aggregated_results"] = self.aggregated_results.to_dict("records")

        export_data["summary"] = self.get_summary_statistics()

        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False)
        elif format.lower() == "csv":
            if include_aggregated and self.aggregated_results is not None:
                self.aggregated_results.to_csv(output_path, index=False)
            else:
                logger.warning("No aggregated results to export as CSV")
        elif format.lower() == "excel":
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                if include_aggregated and self.aggregated_results is not None:
                    self.aggregated_results.to_excel(writer, sheet_name="Aggregated", index=False)

                if include_raw and self.raw_results:
                    for model_name, df in self.raw_results.items():
                        # Clean sheet name
                        sheet_name = model_name.replace("/", "_").replace("\\", "_")[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported results to {output_path}")


if __name__ == "__main__":
    # Demo: Benchmark aggregation
    print("Benchmark Aggregator Module")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    sample_results = {
        "DistilBERT-base": [
            {
                "model_name": "DistilBERT-base",
                "dataset": "imdb",
                "metric_accuracy": 0.85,
                "metric_f1": 0.84,
                "latency_mean_ms": 12.5,
                "throughput_samples_per_sec": 80.0,
            },
            {
                "model_name": "DistilBERT-base",
                "dataset": "sst2",
                "metric_accuracy": 0.88,
                "metric_f1": 0.87,
                "latency_mean_ms": 10.2,
                "throughput_samples_per_sec": 95.0,
            },
        ],
        "RoBERTa-base": [
            {
                "model_name": "RoBERTa-base",
                "dataset": "imdb",
                "metric_accuracy": 0.87,
                "metric_f1": 0.86,
                "latency_mean_ms": 18.3,
                "throughput_samples_per_sec": 55.0,
            },
            {
                "model_name": "RoBERTa-base",
                "dataset": "sst2",
                "metric_accuracy": 0.90,
                "metric_f1": 0.89,
                "latency_mean_ms": 15.7,
                "throughput_samples_per_sec": 65.0,
            },
        ],
    }

    # Create aggregator
    aggregator = BenchmarkAggregator()
    aggregator.load_results([item for sublist in sample_results.values() for item in sublist])

    # Aggregate results
    aggregated = aggregator.aggregate_all()
    print("\nAggregated Results:")
    print(aggregated)

    # Normalize metrics
    normalized = aggregator.normalize_metrics()
    print("\nNormalized Results:")
    print(normalized[["model_name", "metric_accuracy_normalized", "metric_f1_normalized"]])

    # Rank models
    rankings = aggregator.rank_models(
        metrics=["accuracy", "f1"], weights=[0.6, 0.4], ranking_method="composite"
    )
    print("\nModel Rankings:")
    print(rankings)

    # Summary statistics
    summary = aggregator.get_summary_statistics()
    print("\nSummary Statistics:")
    print(summary)
