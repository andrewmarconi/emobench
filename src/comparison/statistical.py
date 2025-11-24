"""
Statistical analysis utilities for SentiCompare.

Provides statistical significance testing, confidence intervals, effect size calculations,
and bootstrapping for robust model comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, bootstrap

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Perform statistical analysis on model comparison results.

    Examples:
        >>> analyzer = StatisticalAnalyzer(results_df)
        >>> t_test = analyzer.paired_t_test("DistilBERT", "RoBERTa", "f1")
        >>> ci = analyzer.bootstrap_ci("DistilBERT", "f1")
    """

    def __init__(self, results: pd.DataFrame):
        """
        Initialize analyzer with results DataFrame.

        Args:
            results: DataFrame with model results including model_name, dataset, and metrics
        """
        self.results = results.copy()
        self.models = results["model_name"].unique() if "model_name" in results.columns else []

    def paired_t_test(
        self,
        model1: str,
        model2: str,
        metric: str,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Perform paired t-test between two models.

        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
            alpha: Significance level
            alternative: Alternative hypothesis ("two-sided", "less", "greater")

        Returns:
            Dict: Test results including statistic, p-value, and interpretation
        """
        # Get paired data
        data1, data2 = self._get_paired_data(model1, model2, metric)

        if data1 is None or data2 is None:
            return {"error": f"No paired data found for {model1} vs {model2} on {metric}"}

        if len(data1) < 2:
            return {"error": "Insufficient data for paired t-test (need at least 2 pairs)"}

        # Perform paired t-test
        try:
            statistic, p_value = ttest_rel(data1, data2, alternative=alternative)

            # Calculate effect size (Cohen's d for paired samples)
            differences = data1 - data2
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)

            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            # Calculate confidence interval for mean difference
            differences = np.array(data1) - np.array(data2)
            se = np.std(differences, ddof=1) / np.sqrt(len(differences))
            mean_diff = np.mean(differences)
            t_critical = stats.t.ppf(1 - alpha / 2, len(differences) - 1)
            ci_lower = mean_diff - t_critical * se
            ci_upper = mean_diff + t_critical * se

            return {
                "test": "paired_t_test",
                "model1": model1,
                "model2": model2,
                "metric": metric,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "significant": p_value < alpha,
                "mean_difference": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": 1 - alpha,
                "cohens_d": cohens_d,
                "effect_interpretation": effect_interpretation,
                "sample_size": len(data1),
                "degrees_freedom": len(data1) - 1,
            }

        except Exception as e:
            logger.error(f"Error in paired t-test: {e}")
            return {"error": str(e)}

    def wilcoxon_signed_rank_test(
        self,
        model1: str,
        model2: str,
        metric: str,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
            alpha: Significance level
            alternative: Alternative hypothesis

        Returns:
            Dict: Test results
        """
        # Get paired data
        data1, data2 = self._get_paired_data(model1, model2, metric)

        if data1 is None or data2 is None:
            return {"error": f"No paired data found for {model1} vs {model2} on {metric}"}

        if len(data1) < 6:  # Wilcoxon test minimum
            return {"error": "Insufficient data for Wilcoxon test (need at least 6 pairs)"}

        try:
            statistic, p_value = wilcoxon(data1, data2, alternative=alternative)

            # Calculate effect size (rank-biserial correlation)
            n = len(data1)
            z_score = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            r = z_score / np.sqrt(n)

            return {
                "test": "wilcoxon_signed_rank",
                "model1": model1,
                "model2": model2,
                "metric": metric,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "significant": p_value < alpha,
                "effect_size_r": r,
                "sample_size": len(data1),
            }

        except Exception as e:
            logger.error(f"Error in Wilcoxon test: {e}")
            return {"error": str(e)}

    def bootstrap_ci(
        self,
        model_name: str,
        metric: str,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        Calculate bootstrap confidence interval for a model's metric.

        Args:
            model_name: Model name
            metric: Metric to analyze
            confidence_level: Confidence level (0-1)
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility

        Returns:
            Dict: Bootstrap results
        """
        # Get model data
        model_data = self.results[self.results["model_name"] == model_name]
        metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric

        if metric_col not in model_data.columns:
            return {"error": f"Metric '{metric}' not found for model '{model_name}'"}

        data = model_data[metric_col].dropna().to_numpy()

        if len(data) < 2:
            return {"error": "Insufficient data for bootstrap CI (need at least 2 observations)"}

        try:
            # Define statistic function
            def statistic_func(data, axis):
                return np.mean(data, axis=axis)

            # Perform bootstrap
            bootstrap_result = bootstrap(
                data=(data,),
                statistic=statistic_func,
                n_resamples=n_bootstrap,
                confidence_level=confidence_level,
                method="percentile",
            )

            # Calculate additional statistics
            sample_mean = float(np.mean(data))
            sample_std = float(np.std(data, ddof=1))
            sample_median = float(np.median(data))

            return {
                "model": model_name,
                "metric": metric,
                "method": "bootstrap",
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "sample_median": sample_median,
                "ci_lower": bootstrap_result.confidence_interval.low,
                "ci_upper": bootstrap_result.confidence_interval.high,
                "bootstrap_distribution": bootstrap_result.bootstrap_distribution,
                "sample_size": len(data),
            }

        except Exception as e:
            logger.error(f"Error in bootstrap CI: {e}")
            return {"error": str(e)}

    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni",
        alpha: float = 0.05,
    ) -> Dict:
        """
        Apply multiple comparison correction to p-values.

        Args:
            p_values: List of p-values
            method: Correction method ("bonferroni", "holm", "fdr_bh")
            alpha: Original significance level

        Returns:
            Dict: Corrected p-values and significance decisions
        """
        try:
            from statsmodels.stats.multitest import multipletests

            p_array = np.array(p_values)
            rejected, p_corrected, _, _ = multipletests(p_array, alpha=alpha, method=method)

            return {
                "method": method,
                "original_p_values": p_values,
                "corrected_p_values": p_corrected.tolist(),
                "significant_original": [p < alpha for p in p_values],
                "significant_corrected": rejected.tolist(),
                "alpha": alpha,
                "num_tests": len(p_values),
            }

        except ImportError:
            logger.warning("statsmodels not available, using simple Bonferroni correction")
            # Simple Bonferroni correction
            p_corrected = [min(p * len(p_values), 1.0) for p in p_values]
            return {
                "method": "bonferroni_simple",
                "original_p_values": p_values,
                "corrected_p_values": p_corrected,
                "significant_original": [p < alpha for p in p_values],
                "significant_corrected": [p < alpha for p in p_corrected],
                "alpha": alpha,
                "num_tests": len(p_values),
            }

    def compare_all_pairs(
        self,
        metric: str,
        test: str = "paired_t_test",
        alpha: float = 0.05,
        correction_method: str = "bonferroni",
    ) -> pd.DataFrame:
        """
        Perform pairwise comparisons between all models.

        Args:
            metric: Metric to compare
            test: Statistical test to use
            alpha: Significance level
            correction_method: Multiple comparison correction method

        Returns:
            pd.DataFrame: All pairwise comparisons
        """
        if len(self.models) < 2:
            return pd.DataFrame()

        comparisons = []
        p_values = []

        # Generate all unique pairs
        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models[i + 1 :], i + 1):
                if test == "paired_t_test":
                    result = self.paired_t_test(model1, model2, metric, alpha)
                elif test == "wilcoxon":
                    result = self.wilcoxon_signed_rank_test(model1, model2, metric, alpha)
                else:
                    result = {"error": f"Unknown test: {test}"}

                if "error" not in result:
                    comparisons.append(result)
                    p_values.append(result["p_value"])

        # Apply multiple comparison correction
        if p_values and correction_method:
            correction = self.multiple_comparison_correction(p_values, correction_method, alpha)

            # Add corrected p-values to comparisons
            for i, comp in enumerate(comparisons):
                comp["p_value_corrected"] = correction["corrected_p_values"][i]
                comp["significant_corrected"] = correction["significant_corrected"][i]

        return pd.DataFrame(comparisons)

    def _get_paired_data(
        self, model1: str, model2: str, metric: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get paired data for two models on the same datasets.

        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare

        Returns:
            Tuple of arrays with paired data, or (None, None) if no pairing possible
        """
        metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric

        # Get data for both models
        data1_df = self.results[self.results["model_name"] == model1]
        data2_df = self.results[self.results["model_name"] == model2]

        if metric_col not in data1_df.columns or metric_col not in data2_df.columns:
            return None, None

        # Pair by dataset (assuming each model has one result per dataset)
        paired_data = []

        for dataset in self.results["dataset"].unique():
            val1 = data1_df[data1_df["dataset"] == dataset][metric_col]
            val2 = data2_df[data2_df["dataset"] == dataset][metric_col]

            if len(val1) > 0 and len(val2) > 0:
                paired_data.append((float(val1.iloc[0]), float(val2.iloc[0])))

        if not paired_data:
            return None, None

        data1, data2 = zip(*paired_data)
        return np.array(data1), np.array(data2)

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test: str = "paired_t",
    ) -> Dict:
        """
        Perform power analysis for sample size determination.

        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power
            test: Type of test

        Returns:
            Dict: Power analysis results
        """
        try:
            from statsmodels.stats.power import TTestPower

            power_analysis = TTestPower()

            if test == "paired_t":
                # For paired t-test, use related samples
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    alternative="two-sided",
                )
            else:
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    alternative="two-sided",
                )

            return {
                "test": test,
                "effect_size": effect_size,
                "alpha": alpha,
                "desired_power": power,
                "required_sample_size": int(np.ceil(sample_size)),
                "actual_power": power_analysis.power(
                    effect_size=effect_size,
                    nobs=int(np.ceil(sample_size)),
                    alpha=alpha,
                ),
            }

        except ImportError:
            logger.warning("statsmodels not available, using approximate calculation")
            # Approximate sample size calculation
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            n_approx = 2 * ((z_alpha + z_beta) / effect_size) ** 2

            return {
                "test": test,
                "effect_size": effect_size,
                "alpha": alpha,
                "desired_power": power,
                "required_sample_size": int(np.ceil(n_approx)),
                "method": "approximate",
            }


def print_statistical_results(results: Dict, title: str = "Statistical Test Results") -> None:
    """
    Print statistical results in a formatted way.

    Args:
        results: Statistical test results
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

    if "error" in results:
        print(f"Error: {results['error']}")
        print("=" * 80 + "\n")
        return

    # Print key results
    if "p_value" in results:
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Significant (α={results.get('alpha', 0.05)}): {results.get('significant', 'N/A')}")

    if "statistic" in results:
        print(f"Test statistic: {results['statistic']:.6f}")

    if "cohens_d" in results:
        print(
            f"Cohen's d: {results['cohens_d']:.6f} ({results.get('effect_interpretation', 'N/A')})"
        )

    if "ci_lower" in results and "ci_upper" in results:
        ci_level = results.get("confidence_level", 0.95) * 100
        print(f"{ci_level:.0f}% CI: [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")

    if "mean_difference" in results:
        print(f"Mean difference: {results['mean_difference']:.6f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Demo: Statistical analysis
    print("Statistical Analysis Module")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini"]
    datasets = ["imdb", "sst2", "amazon", "yelp"]

    data = []
    for model in models:
        for dataset in datasets:
            # Generate correlated performance (models tend to be good/bad on same datasets)
            base_score = 0.80 + hash(dataset) % 15 / 100
            model_offset = (hash(model) % 10 - 5) / 100
            score = base_score + model_offset + np.random.normal(0, 0.02)

            data.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "metric_accuracy": max(0.5, min(1.0, score)),
                    "metric_f1": max(0.5, min(1.0, score - 0.01)),
                }
            )

    results_df = pd.DataFrame(data)

    # Create analyzer
    analyzer = StatisticalAnalyzer(results_df)

    # Paired t-test
    print("\nPaired t-test (DistilBERT vs RoBERTa, F1):")
    t_test = analyzer.paired_t_test("DistilBERT-base", "RoBERTa-base", "f1")
    print_statistical_results(t_test)

    # Bootstrap CI
    print("\nBootstrap CI (DistilBERT, F1):")
    bootstrap_ci = analyzer.bootstrap_ci("DistilBERT-base", "f1", n_bootstrap=1000)
    print_statistical_results(bootstrap_ci, "Bootstrap Confidence Interval")

    # All pairwise comparisons
    print("\nAll pairwise comparisons (F1):")
    pairwise = analyzer.compare_all_pairs("f1", test="paired_t_test")
    print(pairwise[["model1", "model2", "p_value", "significant", "cohens_d"]])

    # Power analysis
    print("\nPower analysis (medium effect size):")
    power = analyzer.power_analysis(effect_size=0.5, power=0.8)
    print(power)
