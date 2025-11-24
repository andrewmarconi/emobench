"""
Streamlit dashboard for EmoBench model comparison.

Interactive web interface for visualizing and comparing model performance
across multiple metrics with filtering, sorting, and export capabilities.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# Import modules directly to avoid circular imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.comparison.aggregator import BenchmarkAggregator
from src.visualization.plots import PlotGenerator

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Streamlit dashboard for model comparison visualization.

    Examples:
        >>> dashboard = Dashboard()
        >>> dashboard.run()
    """

    # Metric definitions for user education
    METRIC_DEFINITIONS = {
        "accuracy": "Fraction of predictions that are correct. Higher values (closer to 1.0) are better.",
        "f1": "Harmonic mean of precision and recall. Balances false positives and false negatives. Higher values (closer to 1.0) are better.",
        "precision": "Fraction of positive predictions that are actually correct. Higher values (closer to 1.0) are better.",
        "recall": "Fraction of actual positives that are correctly identified. Higher values (closer to 1.0) are better.",
        "latency_mean_ms": "Average time (in milliseconds) to process one sample. Lower values are better (faster).",
        "throughput_samples_per_sec": "Number of samples processed per second. Higher values are better (more efficient).",
        "latency_std_ms": "Standard deviation of latency. Lower values indicate more consistent performance.",
        "memory_usage_mb": "Memory used by the model during inference (in MB). Lower values are better.",
        "model_size_mb": "Size of the model file on disk (in MB). Lower values are better for deployment.",
        "inference_time_ms": "Time taken for model inference (excluding preprocessing). Lower values are better.",
        "mcc": "Matthews Correlation Coefficient. Measures quality of binary classifications. Range: -1 to +1, higher is better.",
        "auc": "Area Under the ROC Curve. Measures ability to distinguish between classes. Higher values (closer to 1.0) are better.",
        "loss": "Model's loss function value. Lower values indicate better fit to the data.",
    }

    def __init__(self, results_dir: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize dashboard.

        Args:
            results_dir: Directory containing benchmark results
            port: Port to run the dashboard on
        """
        self.results_dir = results_dir or "experiments/evaluation"
        self.port = port
        self.results_df = pd.DataFrame()
        self.plot_generator = PlotGenerator(pd.DataFrame())

    def load_data(self) -> bool:
        """
        Load benchmark results from directory.

        Returns:
            bool: True if data loaded successfully
        """
        try:
            aggregator = BenchmarkAggregator(self.results_dir)
            aggregator.load_results(self.results_dir)

            # Aggregate results
            self.results_df = aggregator.aggregate_all()

            if self.results_df.empty:
                st.error("No benchmark results found. Please run benchmarks first.")
                return False

            # Initialize plot generator
            self.plot_generator = PlotGenerator(self.results_df)

            logger.info(f"Loaded {len(self.results_df)} benchmark results")
            return True

        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Failed to load data: {e}")
            return False

    def run(self) -> None:
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="EmoBench Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🤖 EmoBench Dashboard")
        st.markdown("Interactive model comparison and analysis dashboard")

        # Add help section
        with st.expander("📚 Help & Metric Definitions"):
            st.markdown("""
            ### Understanding the Metrics

            **Performance Metrics:**
            - **Accuracy**: Fraction of predictions that are correct (higher is better)
            - **F1 Score**: Harmonic mean of precision and recall, balances false positives and negatives (higher is better)
            - **Precision**: Fraction of positive predictions that are actually correct (higher is better)
            - **Recall**: Fraction of actual positives that are correctly identified (higher is better)

            **Speed Metrics:**
            - **Latency (ms)**: Average time to process one sample (lower is better)
            - **Throughput**: Number of samples processed per second (higher is better)

            **Model Characteristics:**
            - **Model Size**: File size on disk (lower is better for deployment)
            - **Memory Usage**: RAM used during inference (lower is better)

            ### How to Use This Dashboard
            1. **Filter** models and datasets in the sidebar
            2. **Select metrics** to compare
            3. **Explore** different visualization types
            4. **Export** results for further analysis
            """)

        # Load data
        if not self.load_data():
            st.stop()

        # Sidebar controls
        self._render_sidebar()

        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Overview", "📈 Visualizations", "📋 Detailed Results", "📤 Export"]
        )

        with tab1:
            self._render_overview()

        with tab2:
            self._render_visualizations()

        with tab3:
            self._render_detailed_results()

        with tab4:
            self._render_export()

    def _render_sidebar(self) -> None:
        """Render sidebar controls."""
        st.sidebar.header("⚙️ Controls")

        # Model selection
        available_models = self.results_df["model_name"].unique().tolist()
        selected_models = st.sidebar.multiselect(
            "Select Models",
            options=available_models,
            default=available_models,
            help="Filter results by model(s)",
        )

        # Dataset selection
        available_datasets = self.results_df["dataset"].unique().tolist()
        selected_datasets = st.sidebar.multiselect(
            "Select Datasets",
            options=available_datasets,
            default=available_datasets,
            help="Filter results by dataset(s)",
        )

        # Metric selection
        if "metrics" in self.results_df.columns:
            # Extract metrics from nested structure
            sample_metrics = self.results_df["metrics"].iloc[0] if not self.results_df.empty else {}
            available_metrics = (
                list(sample_metrics.keys()) if isinstance(sample_metrics, dict) else []
            )
        else:
            # Fallback to flat structure
            metric_cols = [col for col in self.results_df.columns if col.startswith("metric_")]
            available_metrics = [col.replace("metric_", "") for col in metric_cols]

        selected_metrics = st.sidebar.multiselect(
            "Select Metrics",
            options=available_metrics,
            default=available_metrics[:4] if available_metrics else [],  # First 4 metrics or empty
            help="Select metrics to display and compare models",
        )

        # Show metric definitions
        if available_metrics:
            with st.sidebar.expander("📚 Metric Definitions"):
                st.markdown("**Click on metrics to learn what they mean:**")
                for metric in available_metrics:
                    definition = self.METRIC_DEFINITIONS.get(metric, "No definition available.")
                    st.markdown(f"**{metric.replace('_', ' ').title()}:** {definition}")

        # Store filters in session state
        st.session_state.selected_models = selected_models
        st.session_state.selected_datasets = selected_datasets
        st.session_state.selected_metrics = selected_metrics

        # Apply filters
        filtered_df = pd.DataFrame(self.results_df)

        if selected_models:
            filtered_df = filtered_df[filtered_df["model_name"].isin(selected_models)]  # type: ignore

        if selected_datasets:
            filtered_df = filtered_df[filtered_df["dataset"].isin(selected_datasets)]  # type: ignore

        st.session_state.filtered_df = filtered_df

        st.sidebar.markdown("---")
        st.sidebar.info(f"Showing {len(filtered_df)} results")

    def _render_overview(self) -> None:
        """Render overview tab with key metrics."""
        st.header("📊 Performance Overview")

        filtered_df = getattr(st.session_state, "filtered_df", self.results_df)
        selected_metrics = getattr(st.session_state, "selected_metrics", [])

        if filtered_df.empty:
            st.warning("No data to display with current filters.")
            return

        # Key metrics in columns
        if len(selected_metrics) >= 4:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                self._metric_card(selected_metrics[0], filtered_df)

            with col2:
                self._metric_card(selected_metrics[1], filtered_df)

            with col3:
                self._metric_card(selected_metrics[2], filtered_df)

            with col4:
                self._metric_card(selected_metrics[3], filtered_df)
        elif len(selected_metrics) > 0:
            # Dynamic columns based on available metrics
            num_cols = min(len(selected_metrics), 4)
            cols = st.columns(num_cols)
            for i, metric in enumerate(selected_metrics):
                with cols[i % num_cols]:
                    self._metric_card(metric, filtered_df)
        else:
            st.info("No metrics selected. Please select metrics in the sidebar.")

        # Summary statistics
        st.markdown("---")
        st.subheader("📈 Summary Statistics")

        # Model performance summary
        model_summary = (
            filtered_df.groupby("model_name")
            .agg({col: "mean" for col in filtered_df.columns if col.startswith("metric_")})
            .round(4)
        )

        st.dataframe(model_summary, use_container_width=True)

    def _metric_card(self, metric: str, df: pd.DataFrame) -> None:
        """Render a single metric card."""
        if "metrics" in df.columns:
            # Extract from nested metrics structure
            try:
                values = (
                    df["metrics"]
                    .apply(lambda x: x.get(metric) if isinstance(x, dict) else None)
                    .dropna()
                )
            except (KeyError, AttributeError):
                st.warning(f"Metric '{metric}' not found in nested metrics")
                return
        else:
            # Fallback to flat structure
            metric_col = f"metric_{metric}"
            if metric_col not in df.columns:
                st.warning(f"Metric '{metric}' not found in data")
                return
            values = df[metric_col].dropna()

        if len(values) == 0:
            st.warning(f"No valid values for metric '{metric}'")
            return

        # Calculate statistics
        mean_val = values.mean()
        std_val = values.std()
        max_val = values.max()
        min_val = values.min()

        # Display card
        st.metric(
            label=metric.replace("_", " ").title(),
            value=f"{mean_val:.4f}",
            delta=f"±{std_val:.4f}",
            delta_color="normal",
        )

        # Additional info in expander
        with st.expander("Details"):
            # Show metric definition
            definition = self.METRIC_DEFINITIONS.get(metric, "No definition available.")
            st.markdown(f"**Definition:** {definition}")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Max:** {max_val:.4f}")
                st.write(f"**Min:** {min_val:.4f}")
            with col2:
                st.write(f"**Std Dev:** {std_val:.4f}")
                st.write(f"**Count:** {len(values)}")

    def _render_visualizations(self) -> None:
        """Render visualizations tab."""
        st.header("📈 Interactive Visualizations")

        filtered_df = st.session_state.get("filtered_df", self.results_df)
        selected_metrics = st.session_state.get("selected_metrics", [])

        if filtered_df.empty or not selected_metrics:
            st.warning("No data or metrics selected for visualization.")
            return

        # Visualization type selection
        viz_type = st.selectbox(
            "Select Visualization Type",
            options=[
                "Scatter Plot (Accuracy vs Latency)",
                "Radar Chart (Multi-Metric)",
                "Bar Chart (Metric Comparison)",
                "Box Plot (Distributions)",
                "Heatmap (Correlation)",
            ],
            index=0,
            help="Choose how to visualize your model performance metrics",
        )

        # Show selected metrics definitions
        if selected_metrics:
            with st.expander("📊 Selected Metrics Info"):
                st.markdown("**Definitions for selected metrics:**")
                for metric in selected_metrics:
                    definition = self.METRIC_DEFINITIONS.get(metric, "No definition available.")
                    st.markdown(f"• **{metric.replace('_', ' ').title()}:** {definition}")

        # Generate visualization based on selection
        if "Scatter Plot" in viz_type:
            self._render_scatter_plot(filtered_df, selected_metrics)
        elif "Radar Chart" in viz_type:
            self._render_radar_chart(filtered_df, selected_metrics)
        elif "Bar Chart" in viz_type:
            self._render_bar_chart(filtered_df, selected_metrics)
        elif "Box Plot" in viz_type:
            self._render_box_plot(filtered_df, selected_metrics)
        elif "Heatmap" in viz_type:
            self._render_heatmap(filtered_df, selected_metrics)

    def _render_scatter_plot(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Render scatter plot."""
        st.subheader("📊 Accuracy vs Latency Scatter Plot")

        # Select metrics for axes
        col1, col2 = st.columns(2)

        with col1:
            x_metric = st.selectbox(
                "X-axis Metric",
                options=metrics,
                index=0 if metrics else 0,
            )

        with col2:
            # Find latency/throughput metrics for y-axis
            speed_metrics = [
                col
                for col in df.columns
                if any(x in col.lower() for x in ["latency", "throughput", "time"])
            ]
            if speed_metrics:
                y_metric = st.selectbox(
                    "Y-axis Metric (Speed)",
                    options=speed_metrics,
                    index=0,
                )
            else:
                y_metric = st.selectbox(
                    "Y-axis Metric", options=metrics, index=min(1, len(metrics) - 1)
                )

        # Create scatter plot
        fig = self.plot_generator.scatter_plot(
            f"metric_{x_metric}",
            y_metric or "metric_latency_mean_ms",
            color_by="model_name",
            show_pareto=st.checkbox("Show Pareto Frontier", value=True),
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_radar_chart(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Render radar chart."""
        st.subheader("🕸️ Multi-Metric Radar Chart")

        # Select metrics for radar (max 6 for readability)
        selected_radar_metrics = st.multiselect(
            "Select Metrics for Radar Chart",
            options=metrics,
            default=metrics[: min(6, len(metrics))],
            max_selections=6,
        )

        if len(selected_radar_metrics) < 3:
            st.warning("Select at least 3 metrics for radar chart.")
            return

        # Group by selection
        group_by = st.radio(
            "Group By",
            options=["model_name", "dataset"],
            index=0,
        )

        # Create radar chart
        fig = self.plot_generator.radar_chart(
            [f"metric_{m}" for m in selected_radar_metrics],
            group_by=group_by,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_bar_chart(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Render bar chart."""
        st.subheader("📊 Metric Comparison Bar Chart")

        selected_bar_metric = st.selectbox(
            "Select Metric",
            options=metrics,
            index=0,
        )

        group_by = st.radio(
            "Group By",
            options=["model_name", "dataset"],
            index=0,
        )

        # Create bar chart
        fig = self.plot_generator.bar_chart(
            f"metric_{selected_bar_metric}",
            group_by=group_by,
            show_error_bars=st.checkbox("Show Error Bars", value=True),
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_box_plot(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Render box plot."""
        st.subheader("📊 Distribution Box Plot")

        selected_box_metric = st.selectbox(
            "Select Metric",
            options=metrics,
            index=0,
        )

        group_by = st.radio(
            "Group By",
            options=["model_name", "dataset"],
            index=0,
        )

        # Create box plot
        fig = self.plot_generator.box_plot(
            f"metric_{selected_box_metric}",
            group_by=group_by,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_heatmap(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Render heatmap."""
        st.subheader("🔥 Model Performance Heatmap")

        # Select metrics for heatmap
        col1, col2 = st.columns(2)

        with col1:
            index_col = st.selectbox("Y-axis", options=["model_name", "dataset"], index=0)

        with col2:
            value_col = st.selectbox(
                "Values",
                options=[f"metric_{m}" for m in metrics],
                index=0,
            )

        columns_col = "dataset" if index_col == "model_name" else "model_name"

        # Create heatmap
        fig = self.plot_generator.heatmap(
            index_col=index_col,
            columns_col=columns_col,
            value_col=value_col,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_detailed_results(self) -> None:
        """Render detailed results table."""
        st.header("📋 Detailed Results Table")

        st.markdown("""
        View individual benchmark results for each model. Use the sorting and filtering options to explore the data.
        All metrics are computed on the test dataset.
        """)

        filtered_df = st.session_state.get("filtered_df", self.results_df)

        if filtered_df.empty:
            st.warning("No data to display.")
            return

        # Sorting options
        sort_col = st.selectbox(
            "Sort by",
            options=filtered_df.columns.tolist(),
            index=0,
        )
        sort_ascending = st.checkbox("Ascending", value=False)

        # Apply sorting
        sorted_df = filtered_df.sort_values(by=sort_col, ascending=sort_ascending)

        # Display table
        st.dataframe(sorted_df, use_container_width=True)

        # Download button
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="detailed_results.csv",
            mime="text/csv",
        )

    def _render_export(self) -> None:
        """Render export tab."""
        st.header("📤 Export Reports")

        filtered_df = st.session_state.get("filtered_df", self.results_df)
        selected_metrics = st.session_state.get("selected_metrics", [])

        if filtered_df.empty:
            st.warning("No data to export.")
            return

        # Export format selection
        export_format = st.selectbox(
            "Export Format",
            options=["JSON Summary", "CSV Data", "Markdown Report"],
            index=0,
        )

        # Generate and provide download
        if st.button("🚀 Generate Export"):
            if export_format == "JSON Summary":
                self._export_json_summary(filtered_df, selected_metrics)
            elif export_format == "CSV Data":
                self._export_csv_data(filtered_df)
            elif export_format == "Markdown Report":
                self._export_markdown_report(filtered_df, selected_metrics)

    def _export_json_summary(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Export JSON summary."""
        # Create summary
        summary = {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "total_results": len(df),
            "models": df["model_name"].unique().tolist(),
            "datasets": df["dataset"].unique().tolist(),
            "metrics": metrics,
            "summary_statistics": {},
        }

        # Add statistics for each metric
        for metric in metrics:
            metric_col = f"metric_{metric}"
            if metric_col in df.columns:
                values = df[metric_col].dropna()
                summary["summary_statistics"][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": len(values),
                }

        # Provide download
        import json

        json_str = json.dumps(summary, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Summary",
            data=json_str,
            file_name="emobench_summary.json",
            mime="application/json",
        )

    def _export_csv_data(self, df: pd.DataFrame) -> None:
        """Export CSV data."""
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Data",
            data=csv,
            file_name="emobench_data.csv",
            mime="text/csv",
        )

    def _export_markdown_report(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Export Markdown report."""
        # Generate markdown
        lines = [
            "# EmoBench Results Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            f"- **Total Results:** {len(df)}",
            f"- **Models:** {', '.join(df['model_name'].unique())}",
            f"- **Datasets:** {', '.join(df['dataset'].unique())}",
            "",
            "## Performance Summary",
        ]

        # Add metric summaries
        for metric in metrics:
            metric_col = f"metric_{metric}"
            if metric_col in df.columns:
                values = df[metric_col].dropna()
                lines.extend(
                    [
                        f"### {metric.replace('_', ' ').title()}",
                        f"- **Mean:** {values.mean():.4f}",
                        f"- **Std Dev:** {values.std():.4f}",
                        f"- **Range:** [{values.min():.4f}, {values.max():.4f}]",
                        "",
                    ]
                )

        markdown = "\n".join(lines)
        st.download_button(
            label="📥 Download Markdown Report",
            data=markdown,
            file_name="emobench_report.md",
            mime="text/markdown",
        )


def run_dashboard(results_dir: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    Run the Streamlit dashboard.

    Args:
        results_dir: Directory containing benchmark results
        port: Port to run the dashboard on
    """
    dashboard = Dashboard(results_dir, port)
    dashboard.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EmoBench Dashboard")
    parser.add_argument("--results-dir", type=str, help="Directory containing benchmark results")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")

    args = parser.parse_args()

    # Run dashboard with parsed arguments
    run_dashboard(results_dir=args.results_dir, port=args.port)
