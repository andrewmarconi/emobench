"""
Report generation utilities for SentiCompare.

Generates comprehensive ranking reports in various formats including
JSON, CSV, markdown, and visual summaries.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Import modules directly to avoid circular imports
import sys

sys.path.append(str(Path(__file__).parent))

from aggregator import BenchmarkAggregator
from ranking import ModelRanker
from statistical import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class RankingReportGenerator:
    """
    Generate comprehensive ranking reports from benchmark results.

    Examples:
        >>> generator = RankingReportGenerator(results_df)
        >>> report = generator.generate_full_report()
        >>> generator.save_report(report, "report.json")
    """

    def __init__(self, results: pd.DataFrame):
        """
        Initialize report generator.

        Args:
            results: DataFrame with model results
        """
        self.results = results.copy()
        self.aggregator = BenchmarkAggregator()
        self.aggregator.raw_results = {"all": results}
        self.aggregator.aggregated_results = results
        self.ranker = ModelRanker(results)
        self.analyzer = StatisticalAnalyzer(results)

    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report with key findings.

        Returns:
            Dict: Summary report
        """
        # Get basic statistics
        models = self.results["model_name"].unique() if "model_name" in self.results.columns else []
        datasets = self.results["dataset"].unique() if "dataset" in self.results.columns else []

        # Get metric columns
        metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
        metrics = [col.replace("metric_", "") for col in metric_cols]

        # Get best models for each metric
        best_models = {}
        for metric in metrics:
            rankings = self.ranker.rank_by_metric(metric)
            if not rankings.empty:
                best_models[metric] = rankings.iloc[0]["model"]

        # Get composite ranking
        composite_rankings = self.ranker.rank_by_composite(
            performance_metrics=metrics[:2] if len(metrics) >= 2 else metrics,
            performance_weight=0.7,
            speed_weight=0.3,
        )

        # Get Pareto frontier (if we have at least 2 metrics)
        pareto_frontier = None
        if len(metrics) >= 2:
            pareto_frontier = self.ranker.get_pareto_frontier(metrics[0], metrics[1])

        return {
            "report_type": "summary",
            "generated_at": datetime.now().isoformat(),
            "overview": {
                "total_models": len(models),
                "total_datasets": len(datasets),
                "metrics_evaluated": metrics,
                "total_comparisons": len(self.results),
            },
            "best_models": best_models,
            "composite_rankings": (
                composite_rankings.to_dict("records") if not composite_rankings.empty else []
            ),
            "pareto_frontier": (
                pareto_frontier.to_dict("records")
                if pareto_frontier is not None and not pareto_frontier.empty
                else []
            ),
            "model_list": list(models),
            "dataset_list": list(datasets),
        }

    def generate_detailed_report(
        self,
        include_statistical_tests: bool = True,
        include_bootstrap_ci: bool = True,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Generate a detailed report with statistical analysis.

        Args:
            include_statistical_tests: Include pairwise statistical tests
            include_bootstrap_ci: Include bootstrap confidence intervals
            alpha: Significance level for statistical tests

        Returns:
            Dict: Detailed report
        """
        # Start with summary
        report = self.generate_summary_report()
        report["report_type"] = "detailed"

        # Add metric rankings
        metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
        metrics = [col.replace("metric_", "") for col in metric_cols]

        detailed_rankings = {}
        for metric in metrics:
            rankings = self.ranker.rank_by_metric(metric, per_dataset=True)
            if not rankings.empty:
                detailed_rankings[metric] = rankings.to_dict("records")

        report["detailed_rankings"] = detailed_rankings

        # Add statistical tests
        if include_statistical_tests and len(metrics) > 0:
            statistical_results = {}
            for metric in metrics[:3]:  # Limit to first 3 metrics to avoid too many tests
                pairwise_tests = self.analyzer.compare_all_pairs(
                    metric, test="paired_t_test", alpha=alpha
                )
                if not pairwise_tests.empty:
                    statistical_results[metric] = pairwise_tests.to_dict("records")

            report["statistical_tests"] = statistical_results

        # Add bootstrap confidence intervals
        if include_bootstrap_ci and len(metrics) > 0:
            bootstrap_results = {}
            models = self.results["model_name"].unique()

            for model in models:
                model_results = {}
                for metric in metrics[:2]:  # Limit to first 2 metrics
                    ci_result = self.analyzer.bootstrap_ci(model, metric, n_bootstrap=1000)
                    if "error" not in ci_result:
                        model_results[metric] = {
                            "mean": ci_result["sample_mean"],
                            "ci_lower": ci_result["ci_lower"],
                            "ci_upper": ci_result["ci_upper"],
                            "confidence_level": ci_result["confidence_level"],
                        }

                if model_results:
                    bootstrap_results[model] = model_results

            report["bootstrap_confidence_intervals"] = bootstrap_results

        # Add efficiency analysis
        if len(metrics) >= 2:
            efficiency_rankings = self.ranker.rank_by_efficiency(
                performance_metric=metrics[0],
                resource_metric=(
                    "latency_mean_ms" if "latency_mean_ms" in self.results.columns else metrics[1]
                ),
            )
            if not efficiency_rankings.empty:
                report["efficiency_rankings"] = efficiency_rankings.to_dict("records")

        return report

    def generate_comparison_report(
        self,
        reference_model: str,
        metrics: Optional[List[str]] = None,
        include_effect_size: bool = True,
    ) -> Dict:
        """
        Generate a comparison report against a reference model.

        Args:
            reference_model: Model to compare against
            metrics: Metrics to include in comparison
            include_effect_size: Include effect size calculations

        Returns:
            Dict: Comparison report
        """
        if metrics is None:
            metric_cols = [col for col in self.results.columns if col.startswith("metric_")]
            metrics = [col.replace("metric_", "") for col in metric_cols]

        models = self.results["model_name"].unique()
        if reference_model not in models:
            return {"error": f"Reference model '{reference_model}' not found"}

        comparisons = {}

        for model in models:
            if model == reference_model:
                continue

            model_comparison = {"model": model}

            for metric in metrics:
                # Get paired comparison
                if include_effect_size:
                    test_result = self.analyzer.paired_t_test(reference_model, model, metric)
                    if "error" not in test_result:
                        model_comparison[f"{metric}_p_value"] = test_result["p_value"]
                        model_comparison[f"{metric}_significant"] = test_result["significant"]
                        model_comparison[f"{metric}_effect_size"] = test_result["cohens_d"]
                        model_comparison[f"{metric}_mean_diff"] = test_result["mean_difference"]

                # Get raw performance difference
                ref_data = self.results[self.results["model_name"] == reference_model]
                model_data = self.results[self.results["model_name"] == model]

                metric_col = f"metric_{metric}"
                if metric_col in ref_data.columns and metric_col in model_data.columns:
                    ref_mean = ref_data[metric_col].mean()
                    model_mean = model_data[metric_col].mean()
                    model_comparison[f"{metric}_performance_diff"] = model_mean - ref_mean
                    model_comparison[f"{metric}_performance_pct"] = (
                        ((model_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
                    )

            comparisons[model] = model_comparison

        return {
            "report_type": "comparison",
            "reference_model": reference_model,
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "comparisons": comparisons,
        }

    def save_report(
        self,
        report: Dict,
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """
        Save report to file.

        Args:
            report: Report dictionary
            output_path: Output file path
            format: Output format ("json", "markdown", "csv", "html")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        elif format.lower() == "markdown":
            self._save_markdown_report(report, output_path)

        elif format.lower() == "csv":
            self._save_csv_report(report, output_path)

        elif format.lower() == "html":
            self._save_html_report(report, output_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report saved to {output_path}")

    def _save_markdown_report(self, report: Dict, output_path: Path) -> None:
        """Save report as markdown."""
        lines = []

        # Title
        lines.append(f"# {report.get('report_type', 'Report').title()} Report")
        lines.append(f"Generated: {report.get('generated_at', 'Unknown')}")
        lines.append("")

        # Overview
        if "overview" in report:
            overview = report["overview"]
            lines.append("## Overview")
            lines.append(f"- **Models evaluated**: {overview.get('total_models', 'N/A')}")
            lines.append(f"- **Datasets**: {overview.get('total_datasets', 'N/A')}")
            lines.append(f"- **Metrics**: {', '.join(overview.get('metrics_evaluated', []))}")
            lines.append("")

        # Best models
        if "best_models" in report:
            lines.append("## Best Models by Metric")
            for metric, model in report["best_models"].items():
                lines.append(f"- **{metric}**: {model}")
            lines.append("")

        # Composite rankings
        if "composite_rankings" in report and report["composite_rankings"]:
            lines.append("## Composite Rankings")
            lines.append("| Rank | Model | Score |")
            lines.append("|------|-------|-------|")
            for item in report["composite_rankings"][:10]:  # Top 10
                lines.append(
                    f"| {item.get('rank', 'N/A')} | {item.get('model', 'N/A')} | {item.get('composite_score', 'N/A'):.4f} |"
                )
            lines.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def _save_csv_report(self, report: Dict, output_path: Path) -> None:
        """Save report as CSV (focus on rankings)."""
        if "composite_rankings" in report and report["composite_rankings"]:
            df = pd.DataFrame(report["composite_rankings"])
            df.to_csv(output_path, index=False)

    def _save_html_report(self, report: Dict, output_path: Path) -> None:
        """Save report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.get("report_type", "Report").title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{report.get("report_type", "Report").title()} Report</h1>
            <p>Generated: {report.get("generated_at", "Unknown")}</p>
            
            {self._html_overview(report.get("overview", {}))}
            {self._html_best_models(report.get("best_models", {}))}
            {self._html_rankings(report.get("composite_rankings", []))}
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html)

    def _html_overview(self, overview: Dict) -> str:
        """Generate HTML for overview section."""
        if not overview:
            return ""

        return f"""
        <h2>Overview</h2>
        <ul>
            <li><strong>Models evaluated</strong>: {overview.get("total_models", "N/A")}</li>
            <li><strong>Datasets</strong>: {overview.get("total_datasets", "N/A")}</li>
            <li><strong>Metrics</strong>: {", ".join(overview.get("metrics_evaluated", []))}</li>
        </ul>
        """

    def _html_best_models(self, best_models: Dict) -> str:
        """Generate HTML for best models section."""
        if not best_models:
            return ""

        items = "".join(
            [
                f"<li><strong>{metric}</strong>: {model}</li>"
                for metric, model in best_models.items()
            ]
        )

        return f"""
        <h2>Best Models by Metric</h2>
        <ul>
            {items}
        </ul>
        """

    def _html_rankings(self, rankings: List[Dict]) -> str:
        """Generate HTML for rankings table."""
        if not rankings:
            return ""

        rows = "".join(
            [
                f"<tr><td>{r.get('rank', 'N/A')}</td><td>{r.get('model', 'N/A')}</td><td>{r.get('composite_score', 'N/A'):.4f}</td></tr>"
                for r in rankings[:10]
            ]
        )

        return f"""
        <h2>Composite Rankings</h2>
        <table>
            <tr><th>Rank</th><th>Model</th><th>Score</th></tr>
            {rows}
        </table>
        """


def generate_auto_report(
    results: pd.DataFrame,
    output_dir: Union[str, Path],
    report_types: List[str] = ["summary", "detailed"],
    formats: List[str] = ["json", "markdown"],
) -> Dict[str, str]:
    """
    Automatically generate multiple report types and formats.

    Args:
        results: DataFrame with model results
        output_dir: Output directory for reports
        report_types: Types of reports to generate
        formats: Output formats to generate

    Returns:
        Dict[str, str]: Mapping of report names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = RankingReportGenerator(results)
    generated_files = {}

    for report_type in report_types:
        if report_type == "summary":
            report = generator.generate_summary_report()
        elif report_type == "detailed":
            report = generator.generate_detailed_report()
        else:
            logger.warning(f"Unknown report type: {report_type}")
            continue

        for format_type in formats:
            filename = f"{report_type}_report.{format_type}"
            file_path = output_dir / filename
            generator.save_report(report, file_path, format_type)
            generated_files[f"{report_type}_{format_type}"] = str(file_path)

    return generated_files


if __name__ == "__main__":
    # Demo: Report generation
    print("Ranking Report Generator Module")
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
                }
            )

    results_df = pd.DataFrame(data)

    # Generate reports
    generator = RankingReportGenerator(results_df)

    # Summary report
    summary = generator.generate_summary_report()
    print("\nSummary Report:")
    print(json.dumps(summary, indent=2, default=str))

    # Detailed report
    detailed = generator.generate_detailed_report()
    print(f"\nDetailed report generated with {len(detailed)} sections")

    # Comparison report
    comparison = generator.generate_comparison_report("DistilBERT-base")
    print("\nComparison report generated vs DistilBERT-base")

    print("\nReport generation demo completed!")
