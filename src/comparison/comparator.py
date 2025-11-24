"""
Model comparison utilities for SentiCompare.

Provides tools for comparing multiple models across different metrics and datasets.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare multiple models across metrics and datasets.

    Examples:
        >>> comparator = ModelComparator()
        >>> comparator.add_model_results("DistilBERT", results_df)
        >>> comparator.add_model_results("RoBERTa", results_df)
        >>> comparison = comparator.compare_all(metric="f1")
    """

    def __init__(self):
        """Initialize model comparator."""
        self.models = {}

    def add_model_results(self, model_name: str, results: pd.DataFrame) -> None:
        """
        Add results for a model.

        Args:
            model_name: Model name
            results: DataFrame with model results
        """
        self.models[model_name] = results
        logger.info(f"Added results for {model_name}")

    def compare_on_metric(self, metric: str, dataset: Optional[str] = None) -> pd.DataFrame:
        """
        Compare all models on a specific metric.

        Args:
            metric: Metric to compare (e.g., "f1", "accuracy")
            dataset: Dataset to filter by (None = all datasets)

        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.models:
            logger.warning("No models added for comparison")
            return pd.DataFrame()

        metric_col = f"metric_{metric}"
        results = []

        for model_name, model_df in self.models.items():
            df = model_df.copy()

            if dataset:
                df = df[df["dataset"] == dataset]

            if metric_col not in df.columns:
                logger.warning(f"Metric '{metric}' not found for {model_name}")
                continue

            # Aggregate if multiple results
            result = {
                "model": model_name,
                "mean": df[metric_col].mean(),
                "std": df[metric_col].std(),
                "min": df[metric_col].min(),
                "max": df[metric_col].max(),
                "count": len(df),
            }

            results.append(result)

        comparison_df = pd.DataFrame(results)

        if not comparison_df.empty:
            # Sort by mean (descending)
            comparison_df = comparison_df.sort_values("mean", ascending=False)

        return comparison_df

    def compare_all_metrics(
        self, metrics: Optional[List[str]] = None, dataset: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare models on multiple metrics.

        Args:
            metrics: List of metrics to compare (None = all available)
            dataset: Dataset to filter by

        Returns:
            Dict[str, pd.DataFrame]: Comparison for each metric
        """
        if metrics is None:
            # Auto-detect metrics from first model
            if self.models:
                first_model_df = next(iter(self.models.values()))
                metric_cols = [col for col in first_model_df.columns if col.startswith("metric_")]
                metrics = [col.replace("metric_", "") for col in metric_cols]
            else:
                metrics = []

        comparisons = {}
        for metric in metrics:
            comparisons[metric] = self.compare_on_metric(metric, dataset)

        return comparisons

    def get_winner_counts(
        self, metrics: Optional[List[str]] = None, per_dataset: bool = False
    ) -> pd.DataFrame:
        """
        Count how many times each model wins on each metric.

        Args:
            metrics: Metrics to consider (None = all)
            per_dataset: Count wins per dataset

        Returns:
            pd.DataFrame: Winner counts
        """
        if metrics is None:
            # Auto-detect metrics
            if self.models:
                first_model_df = next(iter(self.models.values()))
                metric_cols = [col for col in first_model_df.columns if col.startswith("metric_")]
                metrics = [col.replace("metric_", "") for col in metric_cols]
            else:
                return pd.DataFrame()

        winner_counts = {model: {metric: 0 for metric in metrics} for model in self.models.keys()}

        if per_dataset:
            # Get all datasets
            all_datasets = set()
            for df in self.models.values():
                all_datasets.update(df["dataset"].unique())

            for dataset in all_datasets:
                for metric in metrics:
                    comparison = self.compare_on_metric(metric, dataset=dataset)
                    if not comparison.empty:
                        winner = comparison.iloc[0]["model"]
                        winner_counts[winner][metric] += 1
        else:
            for metric in metrics:
                comparison = self.compare_on_metric(metric)
                if not comparison.empty:
                    winner = comparison.iloc[0]["model"]
                    winner_counts[winner][metric] += 1

        # Convert to DataFrame
        winner_df = pd.DataFrame(winner_counts).T
        winner_df["total_wins"] = winner_df.sum(axis=1)
        winner_df = winner_df.sort_values("total_wins", ascending=False)

        return winner_df

    def get_pairwise_comparison(
        self, model1: str, model2: str, dataset: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare two models pairwise across all metrics.

        Args:
            model1: First model name
            model2: Second model name
            dataset: Dataset to filter by

        Returns:
            pd.DataFrame: Pairwise comparison
        """
        if model1 not in self.models or model2 not in self.models:
            logger.error(f"One or both models not found: {model1}, {model2}")
            return pd.DataFrame()

        df1 = self.models[model1].copy()
        df2 = self.models[model2].copy()

        if dataset:
            df1 = df1[df1["dataset"] == dataset]
            df2 = df2[df2["dataset"] == dataset]

        # Get metric columns
        metric_cols = [col for col in df1.columns if col.startswith("metric_")]

        comparisons = []
        for col in metric_cols:
            metric_name = col.replace("metric_", "")

            val1 = df1[col].mean()
            val2 = df2[col].mean()

            comparison = {
                "metric": metric_name,
                model1: val1,
                model2: val2,
                "diff": val2 - val1,
                "improvement_pct": ((val2 - val1) / val1 * 100) if val1 != 0 else 0,
                "winner": model2 if val2 > val1 else model1,
            }

            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def get_summary_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary table with all models and metrics.

        Args:
            metrics: Metrics to include (None = all)

        Returns:
            pd.DataFrame: Summary table
        """
        if not self.models:
            return pd.DataFrame()

        if metrics is None:
            # Auto-detect metrics
            first_model_df = next(iter(self.models.values()))
            metric_cols = [col for col in first_model_df.columns if col.startswith("metric_")]
            metrics = [col.replace("metric_", "") for col in metric_cols]

        summary_data = []

        for model_name, model_df in self.models.items():
            row = {"model": model_name}

            for metric in metrics:
                metric_col = f"metric_{metric}"
                if metric_col in model_df.columns:
                    row[f"{metric}_mean"] = model_df[metric_col].mean()
                    row[f"{metric}_std"] = model_df[metric_col].std()

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Sort by first metric (mean)
        if metrics and f"{metrics[0]}_mean" in summary_df.columns:
            summary_df = summary_df.sort_values(f"{metrics[0]}_mean", ascending=False)

        return summary_df

    def get_best_model_per_dataset(self, metric: str = "f1") -> pd.DataFrame:
        """
        Get the best model for each dataset.

        Args:
            metric: Metric to use for ranking

        Returns:
            pd.DataFrame: Best model per dataset
        """
        if not self.models:
            return pd.DataFrame()

        # Get all datasets
        all_datasets = set()
        for df in self.models.values():
            all_datasets.update(df["dataset"].unique())

        results = []

        for dataset in all_datasets:
            comparison = self.compare_on_metric(metric, dataset=dataset)

            if not comparison.empty:
                best_model = comparison.iloc[0]
                results.append(
                    {
                        "dataset": dataset,
                        "best_model": best_model["model"],
                        f"{metric}_mean": best_model["mean"],
                        f"{metric}_std": best_model["std"],
                    }
                )

        return pd.DataFrame(results)


def print_comparison(comparison_df: pd.DataFrame, title: str = "Model Comparison") -> None:
    """
    Print comparison results in a formatted way.

    Args:
        comparison_df: Comparison DataFrame
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

    if comparison_df.empty:
        print("No comparison data available")
        print("=" * 80 + "\n")
        return

    # Format numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns

    formatted_df = comparison_df.copy()
    for col in numeric_cols:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
        )

    print(formatted_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Demo: Model comparison
    print("Model Comparator Module")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini"]
    datasets = ["imdb", "sst2", "amazon", "yelp"]

    comparator = ModelComparator()

    for model in models:
        data = []
        for dataset in datasets:
            # Generate sample metrics
            base_acc = 0.82 + hash(model + dataset) % 10 / 100
            data.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "metric_accuracy": base_acc,
                    "metric_f1": base_acc - 0.01,
                    "metric_precision": base_acc - 0.02,
                    "metric_recall": base_acc + 0.01,
                }
            )

        model_df = pd.DataFrame(data)
        comparator.add_model_results(model, model_df)

    # Compare on F1
    print("\nComparison on F1 Score:")
    f1_comparison = comparator.compare_on_metric("f1")
    print_comparison(f1_comparison, "F1 Score Comparison")

    # Get winner counts
    print("\nWinner Counts:")
    winner_counts = comparator.get_winner_counts(per_dataset=True)
    print(winner_counts)

    # Pairwise comparison
    print("\nPairwise Comparison (DistilBERT vs RoBERTa):")
    pairwise = comparator.get_pairwise_comparison("DistilBERT-base", "RoBERTa-base")
    print_comparison(pairwise, "DistilBERT-base vs RoBERTa-base")

    # Summary table
    print("\nSummary Table:")
    summary = comparator.get_summary_table()
    print(summary)
