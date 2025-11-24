"""
Report generation utilities for SentiCompare.

Provides functions to generate JSON, CSV, and markdown reports
from benchmark results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate various report formats from benchmark results.

    Examples:
        >>> generator = ReportGenerator(results_df)
        >>> json_report = generator.generate_json_report()
        >>> csv_report = generator.generate_csv_report()
    """

    def __init__(self, results: pd.DataFrame):
        """
        Initialize report generator.

        Args:
            results: DataFrame with model results
        """
        self.results = results.copy()

    def generate_json_report(
        self,
        include_statistics: bool = True,
        include_rankings: bool = True,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate comprehensive JSON report.

        Args:
            include_statistics: Include statistical summaries
            include_rankings: Include model rankings
            metadata: Additional metadata to include

        Returns:
            Dict: JSON report structure
        """
        report = {
            "report_info": {
                "title": "SentiCompare Benchmark Results",
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
            },
            "data_summary": self._get_data_summary(),
        }

        if include_statistics:
            report["statistics"] = self._get_statistics_summary()

        if include_rankings:
            report["rankings"] = self._get_rankings_summary()

        if metadata:
            report["metadata"] = metadata

        return report

    def generate_csv_report(
        self,
        include_metadata: bool = True,
        aggregate_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate CSV-ready DataFrame.

        Args:
            include_metadata: Include metadata columns
            aggregate_by: Aggregate by model or dataset

        Returns:
            pd.DataFrame: CSV-ready data
        """
        if aggregate_by == "model":
            # Aggregate by model (mean across datasets)
            metric_cols = [col for col in self.results.columns if col.startswith("metric_")]

            agg_dict = {"model_name": "first"}
            for col in metric_cols:
                agg_dict[col] = "mean"
                if f"{col}_std" in self.results.columns:
                    agg_dict[f"{col}_std"] = "mean"

            csv_df = self.results.groupby("model_name").agg(agg_dict).reset_index()

        elif aggregate_by == "dataset":
            # Aggregate by dataset (mean across models)
            metric_cols = [col for col in self.results.columns if col.startswith("metric_")]

            agg_dict = {"dataset": "first"}
            for col in metric_cols:
                agg_dict[col] = "mean"
                if f"{col}_std" in self.results.columns:
                    agg_dict[f"{col}_std"] = "mean"

            csv_df = self.results.groupby("dataset").agg(agg_dict).reset_index()
        else:
            # No aggregation, use raw data
            csv_df = self.results.copy()

        # Add metadata columns if requested
        if include_metadata:
            csv_df = csv_df.copy()
            csv_df["export_timestamp"] = datetime.now().isoformat()
            if "model_name" in csv_df.columns:
                csv_df["model_size"] = csv_df["model_name"].apply(self._estimate_model_size)

        return csv_df

    def generate_markdown_report(
        self,
        title: str = "SentiCompare Results Report",
        include_tables: bool = True,
        include_charts: bool = False,
    ) -> str:
        """
        Generate markdown report.

        Args:
            title: Report title
            include_tables: Include data tables
            include_charts: Include chart placeholders

        Returns:
            str: Markdown report
        """
        lines = [
            f"# {title}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Executive Summary",
            "",
        ]

        # Add summary statistics
        summary = self._get_data_summary()
        lines.extend(
            [
                f"- **Total Models Evaluated:** {summary['total_models']}",
                f"- **Total Datasets:** {summary['total_datasets']}",
                f"- **Total Benchmark Runs:** {summary['total_results']}",
                f"- **Evaluation Period:** {summary.get('evaluation_period', 'N/A')}",
                "",
            ]
        )

        # Add best models
        if include_tables:
            lines.extend(
                [
                    "## Top Performing Models",
                    "",
                    self._generate_best_models_table(),
                    "",
                ]
            )

        # Add detailed rankings
        rankings = self._get_rankings_summary()
        if include_tables and rankings:
            lines.extend(
                [
                    "## Detailed Rankings",
                    "",
                    self._generate_rankings_table(rankings),
                    "",
                ]
            )

        # Add statistical summary
        stats_summary = self._get_statistics_summary()
        if include_tables and stats_summary:
            lines.extend(
                [
                    "## Statistical Summary",
                    "",
                    self._generate_statistics_table(stats_summary),
                    "",
                ]
            )

        # Add conclusions
        lines.extend(
            [
                "## Conclusions",
                "",
                self._generate_conclusions(),
                "",
                "---",
                "*Report generated by SentiCompare v1.0.0*",
            ]
        )

        return "\n".join(lines)

    def _get_data_summary(self) -> Dict:
        """Get basic data summary."""
        return {
            "total_models": (
                self.results["model_name"].nunique() if "model_name" in self.results.columns else 0
            ),
            "total_datasets": (
                self.results["dataset"].nunique() if "dataset" in self.results.columns else 0
            ),
            "total_results": len(self.results),
            "metrics_available": [
                col.replace("metric_", "")
                for col in self.results.columns
                if col.startswith("metric_")
            ],
            "evaluation_period": self._get_evaluation_period(),
        }

    def _get_statistics_summary(self) -> Dict:
        """Get statistical summary for all metrics."""
        metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
        stats_summary = {}

        for metric_col in metric_cols:
            if metric_col in self.results.columns:
                values = pd.Series(self.results[metric_col]).dropna()
                if len(values) > 0:
                    stats_summary[metric_col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "count": len(values),
                    }

        return stats_summary

    def _get_rankings_summary(self) -> Dict:
        """Get model rankings for all metrics."""
        metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
        rankings = {}

        for metric in metric_cols:
            if metric in self.results.columns:
                # Rank by mean performance per model
                if "model_name" in self.results.columns:
                    model_means = self.results.groupby("model_name")[metric].mean()
                    model_rankings = model_means.rank(ascending=False)
                    model_rankings_sorted = model_rankings.sort_values()

                    rankings[metric] = {
                        "rankings": model_rankings_sorted.to_dict(),
                        "best_model": model_rankings_sorted.index[0],
                        "best_score": float(model_rankings_sorted.iloc[0]),
                    }

        return rankings

    def _generate_best_models_table(self) -> str:
        """Generate markdown table of best models."""
        metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
        lines = ["| Metric | Best Model | Score |", "|--------|-------------|------|"]

        for metric in metric_cols[:5]:  # Top 5 metrics
            if metric in self.results.columns and "model_name" in self.results.columns:
                best_idx = self.results[metric].idxmax()
                best_model = self.results.loc[best_idx, "model_name"]
                best_score = self.results[metric].max()

                metric_name = metric.replace("metric_", "").replace("_", " ").title()
                lines.append(f"| {metric_name} | {best_model} | {best_score:.4f} |")

        return "\n".join(lines)

    def _generate_rankings_table(self, rankings: Dict) -> str:
        """Generate markdown table of rankings."""
        if not rankings:
            return "No rankings available."

        # Pick first metric for detailed table
        first_metric = list(rankings.keys())[0]
        metric_data = rankings[first_metric]

        lines = [
            f"### {first_metric.replace('metric_', '').replace('_', ' ').title()} Rankings",
            "",
            "| Rank | Model | Score |",
            "|------|-------|------|",
        ]

        for i, (model, score) in enumerate(metric_data["rankings"].items(), 1):
            lines.append(f"| {i} | {model} | {score:.4f} |")

        return "\n".join(lines)

    def _generate_statistics_table(self, stats: Dict) -> str:
        """Generate markdown table of statistics."""
        lines = [
            "| Metric | Mean | Std Dev | Min | Max |",
            "|--------|------|---------|-----|-----|",
        ]

        for metric, stat in stats.items():
            metric_name = metric.replace("metric_", "").replace("_", " ").title()
            lines.append(
                f"| {metric_name} | {stat['mean']:.4f} | {stat['std']:.4f} | "
                f"{stat['min']:.4f} | {stat['max']:.4f} |"
            )

        return "\n".join(lines)

    def _generate_conclusions(self) -> str:
        """Generate conclusions based on data."""
        lines = ["### Key Findings", ""]

        # Best overall model
        if "model_name" in self.results.columns:
            metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
            if metric_cols:
                # Simple ranking based on first metric
                first_metric = metric_cols[0]
                best_idx = self.results[first_metric].idxmax()
                best_model = self.results.loc[best_idx, "model_name"]

                lines.extend(
                    [
                        f"- **Best Overall Model:** {best_model} (based on {first_metric.replace('metric_', '').title()})",
                        "",
                    ]
                )

        # Most consistent model
        if len(metric_cols) > 1:
            # Calculate consistency (low std dev across metrics)
            model_stats = {}
            for model in self.results["model_name"].unique():
                model_data = self.results[self.results["model_name"] == model]
                consistency_score = 0
                count = 0

                for metric in metric_cols[:3]:  # First 3 metrics
                    if metric in model_data.columns:
                        values = model_data[metric].dropna()
                        if len(values) > 0:
                            # Lower std = more consistent
                            consistency_score += 1 / (values.std() + 1e-6)
                            count += 1

                if count > 0:
                    model_stats[model] = consistency_score / count

            if model_stats:
                most_consistent = max(model_stats, key=model_stats.get)
                lines.extend(
                    [
                        f"- **Most Consistent Model:** {most_consistent} (lowest variance)",
                        "",
                    ]
                )

        lines.extend(
            [
                "### Recommendations",
                "",
                "- Consider model performance across all datasets, not just aggregate scores",
                "- Evaluate trade-offs between accuracy, speed, and resource usage",
                "- Use statistical significance tests for rigorous model comparison",
                "",
            ]
        )

        return "\n".join(lines)

    def _estimate_model_size(self, model_name: str) -> str:
        """Estimate model size from name."""
        model_name = model_name.lower()

        if "tiny" in model_name:
            return "Small (<1B)"
        elif "distilbert" in model_name or "roberta-base" in model_name:
            return "Medium (~100M)"
        elif "1.1b" in model_name or "1.5b" in model_name or "1.7b" in model_name:
            return "Medium-Large (~1-2B)"
        elif "2b" in model_name or "3" in model_name:
            return "Large (2-4B)"
        else:
            return "Unknown"

    def _get_evaluation_period(self) -> str:
        """Get evaluation period from data."""
        if "timestamp" in self.results.columns:
            timestamps = pd.to_datetime(self.results["timestamp"], errors="coerce")
            if not timestamps.empty:
                start_date = timestamps.min().strftime("%Y-%m-%d")
                end_date = timestamps.max().strftime("%Y-%m-%d")
                return f"{start_date} to {end_date}"

        return "Unknown"

    def save_report(
        self,
        report: Union[Dict, pd.DataFrame, str],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """
        Save report to file.

        Args:
            report: Report data
            output_path: Output file path
            format: Output format ("json", "csv", "markdown")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            if isinstance(report, dict):
                with open(output_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
            else:
                raise ValueError("JSON format requires dict report")

        elif format.lower() == "csv":
            if isinstance(report, pd.DataFrame):
                report.to_csv(output_path, index=False)
            else:
                raise ValueError("CSV format requires DataFrame report")

        elif format.lower() == "markdown":
            if isinstance(report, str):
                with open(output_path, "w") as f:
                    f.write(report)
            else:
                raise ValueError("Markdown format requires string report")

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report saved to {output_path}")


def generate_comprehensive_report(
    results_dir: str = "experiments/results",
    output_dir: Union[str, Path] = "reports",
    formats: List[str] = ["json", "csv", "markdown"],
    include_statistics: bool = True,
    include_rankings: bool = True,
) -> Dict[str, Path]:
    """
    Generate comprehensive reports from results.

    Args:
        results_dir: Directory containing benchmark results
        output_dir: Output directory for reports
        formats: List of formats to generate
        include_statistics: Whether to include statistical analysis
        include_rankings: Whether to include model rankings

    Returns:
        Dict[str, Path]: Mapping of format to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder implementation - in real implementation, this would load results
    # and generate comprehensive reports
    print(f"Generating reports in {formats} format(s) to {output_dir}")

    generated_files = {}
    for format_type in formats:
        filename = f"senticompare_report.{format_type}"
        file_path = output_dir / filename
        generated_files[format_type] = file_path
        print(f"Would generate {format_type} report at {file_path}")

    logger.info(f"Generated {len(generated_files)} report files in {output_dir}")
    return generated_files


def generate_reports(
    results_dir: str = "experiments/results",
    output_dir: str = "experiments/reports",
    format: str = "all",
) -> Dict[str, Path]:
    """
    Generate reports from benchmark results.

    Args:
        results_dir: Directory containing benchmark results
        output_dir: Output directory for reports
        format: Report format ("json", "csv", "markdown", "all")

    Returns:
        Dict[str, Path]: Mapping of format to file path
    """
    from pathlib import Path
    import json

    results_dir_path = Path(results_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load results - simplified, just load the benchmark file
    results_file = results_dir_path / "benchmark_imdb.json"
    if not results_file.exists():
        logger.warning(f"No results file found at {results_file}")
        return {}

    with open(results_file, "r") as f:
        raw_results = json.load(f)

    # Flatten the results for DataFrame
    flattened_results = []
    for model, results_list in raw_results.items():
        for result in results_list:
            flattened_results.append(result)

    if not flattened_results:
        logger.warning("No results to report")
        return {}

    results_df = pd.DataFrame(flattened_results)

    # Create report generator
    generator = ReportGenerator(results_df)

    # Determine formats to generate
    if format == "all":
        formats = ["json", "csv", "markdown"]
    else:
        formats = [format]

    generated_files = {}

    for fmt in formats:
        if fmt == "json":
            report = generator.generate_json_report()
            filename = "senticompare_report.json"
        elif fmt == "csv":
            report = generator.generate_csv_report()
            filename = "senticompare_report.csv"
        elif fmt == "markdown":
            report = generator.generate_markdown_report()
            filename = "senticompare_report.md"
        else:
            continue

        file_path = output_dir_path / filename
        generator.save_report(report, file_path, fmt)
        generated_files[fmt] = file_path

    logger.info(f"Generated {len(generated_files)} reports in {output_dir}")
    return generated_files


if __name__ == "__main__":
    # Demo: Report generation
    print("Report Generation Module")
    print("=" * 80)

    # Create sample data
    import numpy as np

    np.random.seed(42)

    models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini"]
    datasets = ["imdb", "sst2", "amazon", "yelp"]

    data = []
    for model in models:
        for dataset in datasets:
            base_f1 = 0.80 + hash(model) % 10 / 100
            data.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "metric_accuracy": base_f1 + 0.02,
                    "metric_f1": base_f1,
                    "metric_precision": base_f1 - 0.01,
                    "latency_mean_ms": 10 + hash(model + dataset) % 20,
                    "throughput_samples_per_sec": 50 + hash(model) % 30,
                    "timestamp": pd.Timestamp.now(),
                }
            )

    results_df = pd.DataFrame(data)

    # Generate reports
    generator = ReportGenerator(results_df)

    # JSON report
    print("\nGenerating JSON report...")
    json_report = generator.generate_json_report()
    print(json.dumps(json_report, indent=2, default=str)[:500] + "...")

    # CSV report
    print("\nGenerating CSV report...")
    csv_report = generator.generate_csv_report(aggregate_by="model")
    print(csv_report.head().to_string())

    # Markdown report
    print("\nGenerating Markdown report...")
    md_report = generator.generate_markdown_report()
    print(md_report[:500] + "...")

    print("\nReport generation demo completed!")
