"""
Plot generation utilities for EmoBench.

Provides Plotly-based visualization functions for model comparison
including scatter plots, radar charts, bar charts, and line plots.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Generate interactive Plotly charts for model comparison.

    Examples:
        >>> plotter = PlotGenerator(results_df)
        >>> fig = plotter.scatter_plot("accuracy", "latency_mean_ms")
        >>> fig.show()
    """

    def __init__(self, results: pd.DataFrame, theme: str = "plotly_white"):
        """
        Initialize plot generator.

        Args:
            results: DataFrame with model results
            theme: Plotly theme name
        """
        self.results = results.copy()
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set1

    def scatter_plot(
        self,
        x_metric: str,
        y_metric: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        title: Optional[str] = None,
        show_pareto: bool = False,
    ) -> go.Figure:
        """
        Create scatter plot comparing two metrics.

        Args:
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Column to color points by
            size_by: Column to size points by
            hover_data: Additional columns for hover
            title: Plot title
            show_pareto: Whether to highlight Pareto frontier

        Returns:
            go.Figure: Interactive scatter plot
        """
        # Prepare data
        plot_data = self._prepare_plot_data([x_metric, y_metric])

        if plot_data.empty:
            logger.warning("No data available for scatter plot")
            return go.Figure()

        # Set default title
        if title is None:
            title = f"{y_metric.replace('metric_', '').title()} vs {x_metric.replace('metric_', '').title()}"

        # Create scatter plot
        fig = px.scatter(
            plot_data,
            x=x_metric,
            y=y_metric,
            color=color_by,
            size=size_by,
            hover_data=hover_data,
            title=title,
            template=self.theme,
            color_discrete_sequence=self.color_palette,
        )

        # Update layout
        fig.update_layout(
            xaxis_title=self._format_metric_name(x_metric),
            yaxis_title=self._format_metric_name(y_metric),
            font=dict(size=12),
            title_font=dict(size=16),
        )

        # Add Pareto frontier if requested
        if show_pareto:
            pareto_data = self._get_pareto_frontier(plot_data, x_metric, y_metric)
            if not pareto_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pareto_data[x_metric],
                        y=pareto_data[y_metric],
                        mode="lines+markers",
                        name="Pareto Frontier",
                        line=dict(color="red", width=2, dash="dash"),
                        marker=dict(color="red", size=8),
                    )
                )

        return fig

    def radar_chart(
        self,
        metrics: List[str],
        group_by: str = "model_name",
        title: Optional[str] = None,
        max_rings: int = 5,
    ) -> go.Figure:
        """
        Create radar chart comparing multiple metrics.

        Args:
            metrics: List of metrics to include
            group_by: Column to group by (model_name, dataset, etc.)
            title: Plot title
            max_rings: Number of concentric rings

        Returns:
            go.Figure: Interactive radar chart
        """
        # Prepare data
        plot_data = self._prepare_plot_data(metrics)

        if plot_data.empty:
            logger.warning("No data available for radar chart")
            return go.Figure()

        # Group data
        grouped = plot_data.groupby(group_by).mean()

        # Create radar chart
        fig = go.Figure()

        for i, (name, row) in enumerate(grouped.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the radar shape

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[self._format_metric_name(m) for m in metrics]
                    + [self._format_metric_name(metrics[0])],
                    fill="toself",
                    name=str(name),
                    line_color=self.color_palette[i % len(self.color_palette)],
                )
            )

        # Update layout
        if title is None:
            title = f"Multi-Metric Comparison by {group_by.replace('_', ' ').title()}"

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_rings],
                    tickfont=dict(size=10),
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                ),
            ),
            title=title,
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16),
            showlegend=True,
        )

        return fig

    def bar_chart(
        self,
        metric: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        orientation: str = "v",
        show_error_bars: bool = True,
    ) -> go.Figure:
        """
        Create bar chart for metric comparison.

        Args:
            metric: Metric to plot
            group_by: Column to group by
            title: Plot title
            orientation: "v" for vertical, "h" for horizontal
            show_error_bars: Whether to show error bars (std dev)

        Returns:
            go.Figure: Interactive bar chart
        """
        # Prepare data
        plot_data = self._prepare_plot_data([metric])

        if plot_data.empty:
            logger.warning("No data available for bar chart")
            return go.Figure()

        # Aggregate data if grouping
        if group_by:
            if show_error_bars:
                agg_data = plot_data.groupby(group_by)[metric].agg(["mean", "std"]).reset_index()
                error_y = agg_data["std"]
                y_values = agg_data["mean"]
            else:
                agg_data = plot_data.groupby(group_by)[metric].mean().reset_index()
                y_values = agg_data[metric]
                error_y = None

            x_values = agg_data[group_by]
        else:
            y_values = plot_data[metric]
            error_y = plot_data[f"{metric}_std"] if f"{metric}_std" in plot_data.columns else None
            x_values = (
                plot_data.index
                if "model_name" not in plot_data.columns
                else plot_data["model_name"]
            )

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                error_y=error_y,
                name=self._format_metric_name(metric),
                marker_color=self.color_palette[0],
                orientation=orientation,
            )
        )

        # Update layout
        if title is None:
            title = self._format_metric_name(metric)

        if orientation == "v":
            fig.update_layout(
                xaxis_title=group_by.replace("_", " ").title() if group_by else "Model",
                yaxis_title=self._format_metric_name(metric),
                title=title,
                template=self.theme,
                font=dict(size=12),
                title_font=dict(size=16),
            )
        else:
            fig.update_layout(
                yaxis_title=group_by.replace("_", " ").title() if group_by else "Model",
                xaxis_title=self._format_metric_name(metric),
                title=title,
                template=self.theme,
                font=dict(size=12),
                title_font=dict(size=16),
            )

        return fig

    def line_plot(
        self,
        x_metric: str,
        y_metrics: List[str],
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        show_markers: bool = True,
    ) -> go.Figure:
        """
        Create line plot for trends over time or parameters.

        Args:
            x_metric: Metric for x-axis (often time or epoch)
            y_metrics: List of metrics for y-axis
            group_by: Column to group lines by
            title: Plot title
            show_markers: Whether to show markers on lines

        Returns:
            go.Figure: Interactive line plot
        """
        # Prepare data
        all_metrics = [x_metric] + y_metrics
        plot_data = self._prepare_plot_data(all_metrics)

        if plot_data.empty:
            logger.warning("No data available for line plot")
            return go.Figure()

        # Create line plot
        fig = go.Figure()

        for i, y_metric in enumerate(y_metrics):
            if group_by:
                for j, (group_name, group_data) in enumerate(plot_data.groupby(group_by)):
                    fig.add_trace(
                        go.Scatter(
                            x=group_data[x_metric],
                            y=group_data[y_metric],
                            mode="lines+markers" if show_markers else "lines",
                            name=f"{group_name} - {self._format_metric_name(y_metric)}",
                            line=dict(color=self.color_palette[j % len(self.color_palette)]),
                            marker=dict(size=6),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data[x_metric],
                        y=plot_data[y_metric],
                        mode="lines+markers" if show_markers else "lines",
                        name=self._format_metric_name(y_metric),
                        line=dict(color=self.color_palette[i % len(self.color_palette)]),
                        marker=dict(size=6),
                    )
                )

        # Update layout
        if title is None:
            title = f"{' vs '.join([self._format_metric_name(m) for m in y_metrics])} over {self._format_metric_name(x_metric)}"

        fig.update_layout(
            xaxis_title=self._format_metric_name(x_metric),
            yaxis_title="Value",
            title=title,
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16),
            showlegend=True,
        )

        return fig

    def heatmap(
        self,
        index_col: str,
        columns_col: str,
        value_col: str,
        title: Optional[str] = None,
        colorscale: str = "Viridis",
    ) -> go.Figure:
        """
        Create heatmap for matrix-style comparisons.

        Args:
            index_col: Column for y-axis indices
            columns_col: Column for x-axis columns
            value_col: Column for heatmap values
            title: Plot title
            colorscale: Plotly colorscale name

        Returns:
            go.Figure: Interactive heatmap
        """
        # Pivot data for heatmap
        pivot_data = self.results.pivot(
            index=index_col,
            columns=columns_col,
            values=value_col,
        )

        if pivot_data.empty:
            logger.warning("No data available for heatmap")
            return go.Figure()

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale=colorscale,
                hoverongaps=False,
            )
        )

        # Update layout
        if title is None:
            title = f"{value_col.replace('metric_', '').title()} Heatmap"

        fig.update_layout(
            xaxis_title=columns_col.replace("_", " ").title(),
            yaxis_title=index_col.replace("_", " ").title(),
            title=title,
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16),
        )

        return fig

    def box_plot(
        self,
        metric: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create box plot for distribution comparison.

        Args:
            metric: Metric to plot
            group_by: Column to group by
            title: Plot title

        Returns:
            go.Figure: Interactive box plot
        """
        # Prepare data
        plot_data = self._prepare_plot_data([metric])

        if plot_data.empty:
            logger.warning("No data available for box plot")
            return go.Figure()

        # Create box plot
        if group_by:
            fig = px.box(
                plot_data,
                x=group_by,
                y=metric,
                title=title or self._format_metric_name(metric),
                template=self.theme,
                color_discrete_sequence=self.color_palette,
            )
        else:
            fig = px.box(
                plot_data,
                y=metric,
                title=title or self._format_metric_name(metric),
                template=self.theme,
            )

        # Update layout
        fig.update_layout(
            xaxis_title=group_by.replace("_", " ").title() if group_by else "Models",
            yaxis_title=self._format_metric_name(metric),
            font=dict(size=12),
            title_font=dict(size=16),
        )

        return fig

    def multi_metric_comparison(
        self,
        metrics: List[str],
        plot_type: str = "bar",
        subplot_dims: Tuple[int, int] = (2, 2),
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create subplots comparing multiple metrics.

        Args:
            metrics: List of metrics to compare
            plot_type: Type of subplot ("bar", "box", "violin")
            subplot_dims: Tuple of (rows, cols) for subplot grid
            title: Overall plot title

        Returns:
            go.Figure: Figure with subplots
        """
        rows, cols = subplot_dims

        # Create subplots
        subplot_titles = tuple(self._format_metric_name(m) for m in metrics)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False,
        )

        # Add each metric to subplot
        for i, metric in enumerate(metrics):
            row = (i // cols) + 1
            col = (i % cols) + 1

            plot_data = self._prepare_plot_data([metric])

            if plot_data.empty:
                continue

            if plot_type == "bar":
                # Group by model for bar plot
                model_means = plot_data.groupby("model_name")[metric].mean()

                fig.add_trace(
                    go.Bar(
                        x=model_means.index,
                        y=model_means.values,
                        name=self._format_metric_name(metric),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            elif plot_type == "box":
                for model in plot_data["model_name"].unique():
                    model_data = plot_data[plot_data["model_name"] == model][metric]

                    fig.add_trace(
                        go.Box(
                            y=model_data,
                            name=model,
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

        # Update layout
        if title is None:
            title = "Multi-Metric Comparison"

        fig.update_layout(
            title=title,
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16),
            height=300 * rows,
        )

        return fig

    def _prepare_plot_data(self, metrics: List[str]) -> pd.DataFrame:
        """Prepare data for plotting by ensuring metric columns exist."""
        plot_data = self.results.copy()

        # Check if metrics exist
        available_metrics = []
        for metric in metrics:
            if metric in plot_data.columns:
                available_metrics.append(metric)
            elif f"metric_{metric}" in plot_data.columns:
                plot_data[metric] = plot_data[f"metric_{metric}"]
                available_metrics.append(metric)
            else:
                logger.warning(f"Metric '{metric}' not found in data")

        return plot_data if available_metrics else pd.DataFrame()

    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        name = metric.replace("metric_", "").replace("_", " ").title()

        # Add units for common metrics
        if "latency" in name.lower():
            return f"{name} (ms)"
        elif "throughput" in name.lower():
            return f"{name} (samples/sec)"
        elif "memory" in name.lower():
            return f"{name} (MB)"
        else:
            return name

    def _get_pareto_frontier(
        self, data: pd.DataFrame, x_metric: str, y_metric: str
    ) -> pd.DataFrame:
        """Calculate Pareto frontier points."""
        points = data[[x_metric, y_metric]].values

        # Sort by x (ascending) and y (descending) for efficiency frontier
        x_vals = [float(p[0]) for p in points]
        y_vals = [float(p[1]) for p in points]
        sorted_indices = np.lexsort((y_vals, [-x for x in x_vals]))

        # Find Pareto frontier
        pareto_points = []
        max_y = -np.inf

        for i in sorted_indices:
            x_val = x_vals[i]
            y_val = y_vals[i]
            if y_val > max_y:
                pareto_points.append([x_val, y_val])
                max_y = y_val

        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            pareto_df.columns = [x_metric, y_metric]
            # Sort by x for proper line drawing
            pareto_df = pareto_df.sort_values(by=x_metric)
            return pareto_df

        return pd.DataFrame()

    def save_plot(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html",
        width: int = 1200,
        height: int = 800,
    ) -> None:
        """
        Save plot to file.

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ("html", "png", "pdf", "svg")
            width: Image width (for image formats)
            height: Image height (for image formats)
        """
        if format.lower() == "html":
            fig.write_html(filename)
        elif format.lower() == "png":
            fig.write_image(filename, width=width, height=height)
        elif format.lower() == "pdf":
            fig.write_image(filename, width=width, height=height)
        elif format.lower() == "svg":
            fig.write_image(filename, width=width, height=height)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Plot saved to {filename}")


if __name__ == "__main__":
    # Demo: Plot generation
    print("Plot Generator Module")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini", "TinyLlama-1.1B"]
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

    # Create plot generator
    plotter = PlotGenerator(results_df)

    # Scatter plot
    print("\nCreating scatter plot...")
    scatter_fig = plotter.scatter_plot(
        "metric_f1", "latency_mean_ms", color_by="model_name", show_pareto=True
    )
    scatter_fig.show()

    # Radar chart
    print("\nCreating radar chart...")
    radar_fig = plotter.radar_chart(
        ["metric_accuracy", "metric_f1", "metric_precision"], group_by="model_name"
    )
    radar_fig.show()

    # Bar chart
    print("\nCreating bar chart...")
    bar_fig = plotter.bar_chart("metric_f1", group_by="model_name")
    bar_fig.show()

    # Multi-metric comparison
    print("\nCreating multi-metric comparison...")
    multi_fig = plotter.multi_metric_comparison(
        ["metric_accuracy", "metric_f1", "metric_precision"], plot_type="bar"
    )
    multi_fig.show()

    print("\nPlot generation demo completed!")
