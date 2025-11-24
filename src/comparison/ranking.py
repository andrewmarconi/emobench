"""
Ranking system for SentiCompare models.

Provides multiple ranking strategies based on different criteria:
- Performance-based (metrics)
- Speed-based (latency, throughput)
- Efficiency-based (performance/speed ratio)
- Composite scores
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelRanker:
    """
    Rank models using various strategies.

    Examples:
        >>> ranker = ModelRanker(results_df)
        >>> rankings = ranker.rank_by_metric("f1")
        >>> composite = ranker.rank_by_composite(["accuracy", "f1", "throughput"])
    """

    def __init__(self, results: pd.DataFrame):
        """
        Initialize ranker.

        Args:
            results: DataFrame with model results
        """
        self.results = results.copy()

    def rank_by_metric(
        self, metric: str, ascending: bool = False, per_dataset: bool = False
    ) -> pd.DataFrame:
        """
        Rank models by a single metric.

        Args:
            metric: Metric to rank by
            ascending: Whether lower is better
            per_dataset: Rank per dataset separately

        Returns:
            pd.DataFrame: Ranked models
        """
        metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric

        if metric_col not in self.results.columns:
            logger.error(f"Metric column '{metric_col}' not found")
            return pd.DataFrame()

        if per_dataset:
            # Rank within each dataset
            ranked_dfs = []

            for dataset in self.results["dataset"].unique():
                dataset_df = self.results[self.results["dataset"] == dataset].copy()
                dataset_df = dataset_df.sort_values(by=metric_col, ascending=ascending)
                dataset_df["rank"] = range(1, len(dataset_df) + 1)
                ranked_dfs.append(dataset_df)

            return pd.concat(ranked_dfs, ignore_index=True)
        else:
            # Global ranking (average across datasets)
            grouped = self.results.groupby("model")[metric_col].agg(["mean", "std", "count"])
            grouped = grouped.sort_values(by="mean", ascending=ascending)
            grouped["rank"] = range(1, len(grouped) + 1)
            grouped = grouped.reset_index()

            return grouped

    def rank_by_multiple_metrics(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None,
        ascending: Optional[List[bool]] = None,
    ) -> pd.DataFrame:
        """
        Rank models by multiple metrics with optional weights.

        Args:
            metrics: List of metrics
            weights: Weight for each metric (None = equal weights)
            ascending: Whether lower is better for each metric

        Returns:
            pd.DataFrame: Ranked models
        """
        if weights is None:
            weights = [1.0] * len(metrics)

        if ascending is None:
            ascending = [False] * len(metrics)

        if len(weights) != len(metrics) or len(ascending) != len(metrics):
            raise ValueError("Metrics, weights, and ascending must have same length")

        # Normalize weights
        weights = list(np.array(weights) / sum(weights))

        # Calculate weighted score for each model
        scores = []

        for model in self.results["model"].unique():
            model_df = self.results[self.results["model"] == model]

            weighted_score = 0.0

            for metric, weight, asc in zip(metrics, weights, ascending):
                metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric

                if metric_col not in model_df.columns:
                    logger.warning(f"Metric '{metric_col}' not found for {model}")
                    continue

                # Get mean value
                value = model_df[metric_col].mean()

                # Invert if ascending (lower is better)
                if asc:
                    value = 1.0 / (value + 1e-10)  # Avoid division by zero

                weighted_score += weight * value

            scores.append({"model": model, "weighted_score": weighted_score})

        score_df = pd.DataFrame(scores)
        score_df = score_df.sort_values("weighted_score", ascending=False)
        score_df["rank"] = range(1, len(score_df) + 1)

        return score_df

    def rank_by_composite(
        self,
        performance_metrics: List[str],
        speed_metrics: Optional[List[str]] = None,
        performance_weight: float = 0.7,
        speed_weight: float = 0.3,
    ) -> pd.DataFrame:
        """
        Rank by composite score combining performance and speed.

        Args:
            performance_metrics: Performance metrics (e.g., ["accuracy", "f1"])
            speed_metrics: Speed metrics (e.g., ["latency_mean_ms", "throughput_samples_per_sec"])
            performance_weight: Weight for performance (0-1)
            speed_weight: Weight for speed (0-1)

        Returns:
            pd.DataFrame: Ranked models with composite scores
        """
        if speed_metrics is None:
            speed_metrics = []

        # Ensure weights sum to 1
        total_weight = performance_weight + speed_weight
        performance_weight /= total_weight
        speed_weight /= total_weight

        composite_scores = []

        for model in self.results["model"].unique():
            model_df = self.results[self.results["model"] == model]

            # Calculate performance score
            perf_score = 0.0
            perf_count = 0

            for metric in performance_metrics:
                metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric

                if metric_col in model_df.columns:
                    perf_score += model_df[metric_col].mean()
                    perf_count += 1

            if perf_count > 0:
                perf_score /= perf_count

            # Calculate speed score
            speed_score = 0.0
            speed_count = 0

            for metric in speed_metrics:
                if metric in model_df.columns:
                    value = model_df[metric].mean()

                    # For latency, lower is better (invert)
                    if "latency" in metric.lower() or "time" in metric.lower():
                        value = 1.0 / (value + 1e-10)

                    speed_score += value
                    speed_count += 1

            if speed_count > 0:
                speed_score /= speed_count

            # Composite score
            composite = (performance_weight * perf_score) + (speed_weight * speed_score)

            composite_scores.append(
                {
                    "model": model,
                    "performance_score": perf_score,
                    "speed_score": speed_score,
                    "composite_score": composite,
                }
            )

        score_df = pd.DataFrame(composite_scores)
        score_df = score_df.sort_values("composite_score", ascending=False)
        score_df["rank"] = range(1, len(score_df) + 1)

        return score_df

    def rank_by_efficiency(
        self, performance_metric: str = "f1", resource_metric: str = "latency_mean_ms"
    ) -> pd.DataFrame:
        """
        Rank by efficiency (performance per unit resource).

        Args:
            performance_metric: Performance metric
            resource_metric: Resource metric (latency, memory, etc.)

        Returns:
            pd.DataFrame: Ranked models by efficiency
        """
        perf_col = (
            f"metric_{performance_metric}"
            if not performance_metric.startswith("metric_")
            else performance_metric
        )

        if perf_col not in self.results.columns or resource_metric not in self.results.columns:
            logger.error("Required columns not found")
            return pd.DataFrame()

        efficiency_scores = []

        for model in self.results["model"].unique():
            model_df = self.results[self.results["model"] == model]

            perf = model_df[perf_col].mean()
            resource = model_df[resource_metric].mean()

            # Efficiency = performance / resource
            # Higher performance with lower resource = higher efficiency
            if resource > 0:
                efficiency = perf / resource
            else:
                efficiency = 0.0

            efficiency_scores.append(
                {
                    "model": model,
                    performance_metric: perf,
                    resource_metric: resource,
                    "efficiency_score": efficiency,
                }
            )

        score_df = pd.DataFrame(efficiency_scores)
        score_df = score_df.sort_values("efficiency_score", ascending=False)
        score_df["rank"] = range(1, len(score_df) + 1)

        return score_df

    def get_pareto_frontier(
        self, metric1: str, metric2: str, maximize_both: bool = True
    ) -> pd.DataFrame:
        """
        Get Pareto frontier for two metrics.

        Args:
            metric1: First metric
            metric2: Second metric
            maximize_both: Whether to maximize both metrics

        Returns:
            pd.DataFrame: Models on Pareto frontier
        """
        col1 = f"metric_{metric1}" if not metric1.startswith("metric_") else metric1
        col2 = f"metric_{metric2}" if not metric2.startswith("metric_") else metric2

        if col1 not in self.results.columns or col2 not in self.results.columns:
            logger.error("Required columns not found")
            return pd.DataFrame()

        # Aggregate by model
        model_data = []
        for model in self.results["model"].unique():
            model_df = self.results[self.results["model"] == model]
            model_data.append(
                {
                    "model": model,
                    metric1: model_df[col1].mean(),
                    metric2: model_df[col2].mean(),
                }
            )

        df = pd.DataFrame(model_data)

        # Find Pareto frontier
        pareto_front = []

        for i, row in df.iterrows():
            is_pareto = True

            for j, other_row in df.iterrows():
                if i == j:
                    continue

                if maximize_both:
                    # Other dominates if both metrics are better
                    if (
                        other_row[metric1] >= row[metric1] and other_row[metric2] > row[metric2]
                    ) or (other_row[metric1] > row[metric1] and other_row[metric2] >= row[metric2]):
                        is_pareto = False
                        break
                else:
                    # For minimize-maximize or other combinations
                    # This is simplified - extend as needed
                    pass

            if is_pareto:
                pareto_front.append(row)

        pareto_df = pd.DataFrame(pareto_front)
        pareto_df["on_pareto_frontier"] = True

        return pareto_df


def print_rankings(rankings_df: pd.DataFrame, title: str = "Model Rankings") -> None:
    """
    Print rankings in a formatted way.

    Args:
        rankings_df: Rankings DataFrame
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

    if rankings_df.empty:
        print("No ranking data available")
        print("=" * 80 + "\n")
        return

    # Format numeric columns
    numeric_cols = rankings_df.select_dtypes(include=[np.number]).columns

    formatted_df = rankings_df.copy()
    for col in numeric_cols:
        if col == "rank":
            continue  # Don't format rank
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
        )

    print(formatted_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Demo: Model ranking
    print("Model Ranking Module")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini", "TinyLlama-1.1B"]
    datasets = ["imdb", "sst2"]

    data = []
    for model in models:
        for dataset in datasets:
            base_f1 = 0.80 + hash(model) % 10 / 100
            data.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "metric_accuracy": base_f1 + 0.02,
                    "metric_f1": base_f1,
                    "metric_precision": base_f1 - 0.01,
                    "metric_recall": base_f1 + 0.01,
                    "latency_mean_ms": 10 + hash(model + dataset) % 20,
                    "throughput_samples_per_sec": 50 + hash(model) % 30,
                }
            )

    results_df = pd.DataFrame(data)

    # Create ranker
    ranker = ModelRanker(results_df)

    # Rank by F1
    print("\nRanking by F1 Score:")
    f1_rankings = ranker.rank_by_metric("f1")
    print_rankings(f1_rankings, "F1 Score Rankings")

    # Rank by multiple metrics
    print("\nRanking by Multiple Metrics (F1 + Accuracy):")
    multi_rankings = ranker.rank_by_multiple_metrics(metrics=["f1", "accuracy"], weights=[0.6, 0.4])
    print_rankings(multi_rankings, "Multi-Metric Rankings")

    # Composite ranking
    print("\nComposite Rankings (Performance + Speed):")
    composite_rankings = ranker.rank_by_composite(
        performance_metrics=["f1", "accuracy"],
        speed_metrics=["throughput_samples_per_sec"],
        performance_weight=0.7,
        speed_weight=0.3,
    )
    print_rankings(composite_rankings, "Composite Rankings")

    # Efficiency ranking
    print("\nEfficiency Rankings (F1 per ms latency):")
    efficiency_rankings = ranker.rank_by_efficiency(
        performance_metric="f1", resource_metric="latency_mean_ms"
    )
    print_rankings(efficiency_rankings, "Efficiency Rankings")
