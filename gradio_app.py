"""
Simple Gradio Web UI for MoodBench - Multi-LLM Sentiment Analysis Benchmark Framework

A basic web interface demonstrating MoodBench functionality.
Install required dependencies: pip install gradio
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Check if gradio is available
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not available. Install with: pip install gradio")
    sys.exit(1)

# Available models and datasets (fallback if registry not available)
DEFAULT_MODELS = [
    "BERT-tiny",
    "BERT-mini",
    "BERT-small",
    "ELECTRA-small",
    "MiniLM-L12",
    "DistilBERT-base",
    "RoBERTa-base",
    # "GPT2-small",  # Temporarily disabled for Gradio - too slow for demo
]

DEFAULT_DATASETS = ["imdb", "sst2", "amazon", "yelp"]


def run_command_stream(command_list, timeout=300):
    """Run a command and stream output in real-time."""
    try:
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        # Enable test mode for faster training in Gradio
        env["MOODBENCH_TEST_MODE"] = "1"

        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=Path(__file__).parent,
        )

        # Stream output in real-time
        output_lines = []
        import time

        start_time = time.time()

        if process.stdout:
            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    yield f"❌ Command timed out after {timeout} seconds"
                    return

                if process.poll() is not None:
                    # Process finished, read remaining output
                    try:
                        remaining = process.stdout.read()
                        if remaining:
                            lines = remaining.strip().split("\n")
                            output_lines.extend(lines)
                            if lines:
                                yield "\n".join(output_lines[-20:])  # Keep last 20 lines
                    except Exception:
                        pass
                    break

                try:
                    # Read output with timeout
                    output = process.stdout.readline()
                    if not output:
                        # Check if process is still running
                        if process.poll() is None:
                            time.sleep(0.1)  # Small delay to prevent busy waiting
                            continue
                        else:
                            break

                    line = output.strip()
                    if line:
                        output_lines.append(line)
                        yield "\n".join(output_lines[-20:])  # Keep last 20 lines

                except Exception as e:
                    yield f"❌ Error reading output: {str(e)}"
                    break

        # Wait for process to finish if not already done
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = -1
            yield "❌ Command killed due to timeout"

        if return_code == 0:
            yield "✅ Command completed successfully!"
        else:
            yield f"❌ Command failed with return code {return_code}"

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def run_benchmark(models, datasets):
    """Run benchmark on selected models and datasets with progress tracking."""
    if not models:
        yield (0, "❌ Please select at least one model to benchmark.")
        return
    if not datasets:
        yield (0, "❌ Please select at least one dataset to test on.")
        return

    total_combinations = len(models) * len(datasets)
    completed = 0

    yield (
        0,
        f"🏁 Starting benchmark of {len(models)} models on {len(datasets)} datasets ({total_combinations} total combinations)...",
    )

    # Build command with all models and datasets
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "benchmark",
        "--checkpoints-dir",
        "experiments/checkpoints",
        "--device",
        "auto",
    ]

    # Add models
    command.extend(["--models"] + models)

    # Add datasets
    command.extend(["--datasets"] + datasets)

    for update in run_command_stream(command):
        # Calculate progress based on completion messages in the output
        # For now, we'll show 50% progress during execution and 100% at end
        if "completed" in update.lower() or "finished" in update.lower():
            completed += 1
            progress_percent = min(int((completed / total_combinations) * 100), 100)
            yield (progress_percent, update)
        else:
            yield (50, update)  # Show 50% during execution

    yield (100, f"🎉 Benchmark completed! Tested {len(models)} models on {len(datasets)} datasets.")


def train_models(models, datasets):
    """Train multiple models on multiple datasets with progress tracking."""
    if not models:
        yield (0, "❌ Please select at least one model to train.")
        return
    if not datasets:
        yield (0, "❌ Please select at least one dataset to train on.")
        return

    total_combinations = len(models) * len(datasets)
    completed = 0

    # Show progress bar and initial message
    yield (
        0,
        f"🚀 Starting training of {len(models)} models on {len(datasets)} datasets ({total_combinations} total combinations)...",
    )

    for dataset in datasets:
        for model in models:
            progress_percent = int((completed / total_combinations) * 100)
            yield (
                progress_percent,
                f"📋 Training {model} on {dataset}... ({completed + 1}/{total_combinations})",
            )

            command = [
                sys.executable,
                "-m",
                "src.cli",
                "train",
                "--model",
                model,
                "--dataset",
                dataset,
                "--device",
                "auto",
            ]

            for update in run_command_stream(command):
                # Keep progress at current level during training
                yield (progress_percent, f"[{model} on {dataset}] {update}")

            completed += 1
            progress_percent = int((completed / total_combinations) * 100)
            yield (
                progress_percent,
                f"✅ Completed {model} on {dataset} ({completed}/{total_combinations})",
            )

    yield (
        100,
        f"🎉 All training completed! Trained {len(models)} models on {len(datasets)} datasets.",
    )


def generate_reports():
    """Generate reports with real-time progress and return markdown content."""
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "report",
        "--format",
        "all",
    ]

    yield (0, "📋 Generating reports...", "")

    progress_updates = []
    json_done = False
    csv_done = False
    markdown_done = False

    for update in run_command_stream(command):
        progress_updates.append(update)

        # Track completion of each format
        update_lower = update.lower()
        if "json" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            json_done = True
        if "csv" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            csv_done = True
        if "markdown" in update_lower and ("saved" in update_lower or "generated" in update_lower):
            markdown_done = True

        # Calculate progress
        completed = sum([json_done, csv_done, markdown_done])
        progress = int((completed / 3) * 100)

        yield (progress, "\n".join(progress_updates[-10:]), "")

    # Load and return the markdown report content
    try:
        import os

        markdown_path = "experiments/reports/moodbench_report.md"
        if os.path.exists(markdown_path):
            with open(markdown_path, "r") as f:
                markdown_content = f.read()
            yield (100, "✅ Reports generated successfully!", markdown_content)
        else:
            yield (100, "❌ Reports generated but markdown file not found", "")
    except Exception as e:
        yield (100, f"❌ Error loading markdown report: {str(e)}", "")


def load_benchmark_results():
    """Load benchmark results from the results directory."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return pd.DataFrame()

    # Find all benchmark result files
    benchmark_files = list(results_dir.glob("benchmark_*.json"))

    if not benchmark_files:
        return pd.DataFrame()

    # Load and combine all results
    all_results = []
    for results_file in benchmark_files:
        try:
            with open(results_file, "r") as f:
                raw_results = json.load(f)

            # Flatten the results for DataFrame
            for model, results_list in raw_results.items():
                for result in results_list:
                    result["_source_file"] = results_file.name
                    all_results.append(result)

        except Exception:
            continue

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def create_dashboard_summary(results_df):
    """Create a summary of benchmark results."""
    if results_df.empty:
        return {
            "total_models": 0,
            "total_datasets": 0,
            "total_results": 0,
            "last_updated": "No data available",
            "top_model": "N/A",
        }

    summary = {
        "total_models": results_df["model_name"].nunique(),
        "total_datasets": results_df["dataset"].nunique(),
        "total_results": len(results_df),
        "last_updated": "Unknown",
    }

    # Find top performing model (highest accuracy)
    if "metric_accuracy" in results_df.columns:
        valid_data = results_df.dropna(subset=["metric_accuracy"])
        if not valid_data.empty:
            best_idx = valid_data["metric_accuracy"].idxmax()
            summary["top_model"] = valid_data.loc[best_idx, "model_name"]
        else:
            summary["top_model"] = "N/A"
    else:
        summary["top_model"] = "N/A"

    # Try to get timestamp
    if "timestamp" in results_df.columns:
        timestamps = pd.to_datetime(results_df["timestamp"], unit="s", errors="coerce")
        if not timestamps.empty:
            summary["last_updated"] = timestamps.max().strftime("%Y-%m-%d %H:%M:%S")

    return summary


def create_training_matrix():
    """Create an HTML table showing trained model-dataset combinations."""
    checkpoints_dir = Path("experiments/checkpoints")

    # Get all available models and datasets from the constants
    all_models = DEFAULT_MODELS
    all_datasets = DEFAULT_DATASETS

    # Check which combinations have been trained
    trained_combinations = {}

    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                # Parse model_dataset from directory name
                dir_name = item.name
                if "_" in dir_name:
                    parts = dir_name.split("_", 1)  # Split only on first underscore
                    if len(parts) == 2:
                        model, dataset = parts
                        # Check if final checkpoint exists
                        final_checkpoint = item / "final"
                        if final_checkpoint.exists():
                            trained_combinations[(model, dataset)] = True

    # Create HTML table
    html = """
    <style>
        .training-matrix {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
            width: 100%;
        }
        .training-matrix th, .training-matrix td {
            border: 1px solid var(--border-color-primary, #e5e7eb);
            padding: 8px 12px;
            text-align: center;
        }
        .training-matrix th {
            background-color: var(--background-fill-secondary, #f9fafb);
            font-weight: bold;
            color: var(--body-text-color, #111827);
        }
        .training-matrix .model-header {
            background-color: var(--background-fill-primary, #ffffff);
            font-weight: 600;
        }
        .trained {
            background-color: rgba(34, 197, 94, 0.1);
            color: #16a34a;
        }
        .not-trained {
            background-color: rgba(239, 68, 68, 0.1);
            color: #dc2626;
        }
        .status-icon {
            font-size: 16px;
            font-weight: bold;
        }
        @media (prefers-color-scheme: dark) {
            .trained {
                background-color: rgba(34, 197, 94, 0.2);
                color: #4ade80;
            }
            .not-trained {
                background-color: rgba(239, 68, 68, 0.2);
                color: #f87171;
            }
        }
    </style>
    <table class="training-matrix">
        <thead>
            <tr>
                <th class="model-header">Model ↓ / Dataset →</th>
    """

    # Add dataset headers
    for dataset in all_datasets:
        html += f"<th>{dataset}</th>"
    html += "</tr></thead><tbody>"

    # Add rows for each model
    for model in all_models:
        html += f"<tr><td class='model-header'>{model}</td>"
        for dataset in all_datasets:
            is_trained = (model, dataset) in trained_combinations
            status_class = "trained" if is_trained else "not-trained"
            status_icon = "✅" if is_trained else "❌"
            status_text = "Trained" if is_trained else "Not Trained"
            html += f'<td class="{status_class}" title="{model} on {dataset}: {status_text}"><span class="status-icon">{status_icon}</span></td>'
        html += "</tr>"

    html += "</tbody></table>"

    return html


def create_scatter_plot(results_df):
    """Create a scatter plot of latency vs accuracy with individual runs and model means."""
    if results_df.empty or "metric_accuracy" not in results_df.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available", xaxis_title="Latency (ms)", yaxis_title="Accuracy"
        )
        return fig

    # Prepare data
    plot_data = results_df.dropna(subset=["metric_accuracy"])

    if plot_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Valid Data for Plotting", xaxis_title="Latency (ms)", yaxis_title="Accuracy"
        )
        return fig

    # Calculate per-model means
    model_means = (
        plot_data.groupby("model_name")
        .agg({"metric_accuracy": "mean", "latency_mean_ms": "mean"})
        .reset_index()
    )

    # Get unique models and assign colors
    unique_models = sorted(plot_data["model_name"].unique())
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create color mapping
    model_colors = {}
    for i, model in enumerate(unique_models):
        model_colors[model] = color_palette[i % len(color_palette)]

    # Create figure
    fig = go.Figure()

    # Add traces for each model (individual runs)
    for model in unique_models:
        model_data = plot_data[plot_data["model_name"] == model]
        model_color = model_colors[model]

        # Individual runs for this model
        fig.add_trace(
            go.Scatter(
                x=model_data["latency_mean_ms"],
                y=model_data["metric_accuracy"],
                mode="markers",
                name=f"{model} (runs)",
                marker=dict(
                    size=6, opacity=0.6, color=model_color, line=dict(width=1, color=model_color)
                ),
                hovertemplate=f"<b>{model}</b><br>"
                + "Latency: %{x:.2f}ms<br>"
                + "Accuracy: %{y:.4f}<br>"
                + "Dataset: %{customdata}<extra></extra>",
                customdata=model_data["dataset"],
                legendgroup=model,
                showlegend=True,
            )
        )

    # Add traces for model means
    for model in unique_models:
        model_mean = model_means[model_means["model_name"] == model]
        if not model_mean.empty:
            model_color = model_colors[model]

            fig.add_trace(
                go.Scatter(
                    x=model_mean["latency_mean_ms"],
                    y=model_mean["metric_accuracy"],
                    mode="markers",
                    name=f"{model} (mean)",
                    marker=dict(
                        size=14,
                        opacity=1.0,
                        color=model_color,
                        symbol="diamond",
                        line=dict(width=2, color="black"),
                    ),
                    hovertemplate=f"<b>{model} (Mean)</b><br>"
                    + "Avg Latency: %{x:.2f}ms<br>"
                    + "Avg Accuracy: %{y:.4f}<extra></extra>",
                    legendgroup=model,
                    showlegend=True,
                )
            )

    # Update layout
    fig.update_layout(
        title="Model Performance: Latency vs Accuracy",
        xaxis_title="Latency (ms)",
        yaxis_title="Accuracy",
        font=dict(size=12),
        title_font=dict(size=16),
        showlegend=False,  # Hide legend since hover provides details
        hovermode="closest",
    )

    return fig

    # Prepare data
    plot_data = results_df.dropna(subset=["metric_accuracy"])

    if plot_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Valid Data for Plotting", xaxis_title="Accuracy", yaxis_title="Latency (ms)"
        )
        return fig

    # Calculate per-model means
    model_means = (
        plot_data.groupby("model_name")
        .agg({"metric_accuracy": "mean", "latency_mean_ms": "mean"})
        .reset_index()
    )

    # Create figure with two traces
    fig = go.Figure()

    # Trace 1: Individual runs (small, transparent markers)
    fig.add_trace(
        go.Scatter(
            x=plot_data["metric_accuracy"],
            y=plot_data["latency_mean_ms"],
            mode="markers",
            name="Individual Runs",
            marker=dict(
                size=6, opacity=0.4, color="lightblue", line=dict(width=1, color="darkblue")
            ),
            hovertemplate="<b>%{text}</b><br>"
            + "Accuracy: %{x:.4f}<br>"
            + "Latency: %{y:.2f}ms<br>"
            + "Dataset: %{customdata}<extra></extra>",
            text=plot_data["model_name"],
            customdata=plot_data["dataset"],
        )
    )

    # Trace 2: Per-model means (bigger, opaque markers)
    fig.add_trace(
        go.Scatter(
            x=model_means["metric_accuracy"],
            y=model_means["latency_mean_ms"],
            mode="markers",
            name="Model Means",
            marker=dict(
                size=12,
                opacity=1.0,
                color="red",
                symbol="diamond",
                line=dict(width=2, color="darkred"),
            ),
            hovertemplate="<b>%{text}</b> (Mean)<br>"
            + "Avg Accuracy: %{x:.4f}<br>"
            + "Avg Latency: %{y:.2f}ms<extra></extra>",
            text=model_means["model_name"],
        )
    )

    # Update layout
    fig.update_layout(
        title="Model Performance: Accuracy vs Latency",
        xaxis_title="Accuracy",
        yaxis_title="Latency (ms)",
        font=dict(size=12),
        title_font=dict(size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            groupclick="togglegroup",
        ),
        hovermode="closest",
        margin=dict(b=120),  # Add bottom margin for legend
    )

    return fig


def create_accuracy_by_dataset_chart(results_df):
    """Create bar chart showing model accuracy by dataset."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "metric_accuracy": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no accuracy data
    aggregated_df = aggregated_df.dropna(subset=["metric_accuracy"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Accuracy Data Available")
        return fig

    # Create bar chart for accuracy
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="metric_accuracy",
        color="dataset",
        barmode="group",
        title="Model Accuracy by Dataset",
        labels={"model_name": "Model", "metric_accuracy": "Accuracy", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig

    # Prepare data for accuracy only
    accuracy_data = []
    for _, row in results_df.iterrows():
        model = row["model_name"]
        dataset = row["dataset"]
        if pd.notna(row.get("metric_accuracy")):
            accuracy_data.append(
                {"model_name": model, "dataset": dataset, "accuracy": row["metric_accuracy"]}
            )

    if not accuracy_data:
        fig = go.Figure()
        fig.update_layout(title="No Accuracy Data Available")
        return fig

    accuracy_df = pd.DataFrame(accuracy_data)

    # Create bar chart for accuracy
    fig = px.bar(
        accuracy_df,
        x="model_name",
        y="accuracy",
        color="dataset",
        barmode="group",
        title="Model Accuracy by Dataset",
        labels={"model_name": "Model", "accuracy": "Accuracy", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig


def create_f1_by_dataset_chart(results_df):
    """Create bar chart showing model F1 score by dataset."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "metric_f1": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no F1 data
    aggregated_df = aggregated_df.dropna(subset=["metric_f1"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No F1 Data Available")
        return fig

    # Create bar chart for F1
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="metric_f1",
        color="dataset",
        barmode="group",
        title="Model F1 Score by Dataset",
        labels={"model_name": "Model", "metric_f1": "F1 Score", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig

    # Prepare data for F1 only
    f1_data = []
    for _, row in results_df.iterrows():
        model = row["model_name"]
        dataset = row["dataset"]
        if pd.notna(row.get("metric_f1")):
            f1_data.append({"model_name": model, "dataset": dataset, "f1_score": row["metric_f1"]})

    if not f1_data:
        fig = go.Figure()
        fig.update_layout(title="No F1 Data Available")
        return fig

    f1_df = pd.DataFrame(f1_data)

    # Create bar chart for F1
    fig = px.bar(
        f1_df,
        x="model_name",
        y="f1_score",
        color="dataset",
        barmode="group",
        title="Model F1 Score by Dataset",
        labels={"model_name": "Model", "f1_score": "F1 Score", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig

    # Prepare data for both metrics
    plot_data = []
    for _, row in results_df.iterrows():
        model = row["model_name"]
        dataset = row["dataset"]

        # Add accuracy data
        if pd.notna(row.get("metric_accuracy")):
            plot_data.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "metric": "Accuracy",
                    "value": row["metric_accuracy"],
                }
            )

        # Add F1 data
        if pd.notna(row.get("metric_f1")):
            plot_data.append(
                {
                    "model_name": model,
                    "dataset": dataset,
                    "metric": "F1 Score",
                    "value": row["metric_f1"],
                }
            )

    if not plot_data:
        fig = go.Figure()
        fig.update_layout(title="No Performance Data Available")
        return fig

    plot_df = pd.DataFrame(plot_data)

    # Create faceted bar chart
    fig = px.bar(
        plot_df,
        x="model_name",
        y="value",
        color="dataset",
        facet_col="metric",
        barmode="group",
        title="Model Performance by Dataset",
        labels={"model_name": "Model", "value": "Score", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig


def create_latency_breakdown_chart(results_df):
    """Create chart showing latency metrics breakdown per model (Mean, Median, P95, P99)."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "latency_mean_ms": "mean",
                "latency_median_ms": "mean",
                "latency_p95_ms": "mean",
                "latency_p99_ms": "mean",
            }
        )
        .reset_index()
    )

    # Prepare data for bar chart - we'll show mean, median, p95, p99 as separate traces
    latency_data = []

    for _, row in aggregated_df.iterrows():
        model = row["model_name"]
        dataset = row["dataset"]

        # Create entries for each latency metric (excluding TTFT)
        latency_data.extend(
            [
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "Mean",
                    "value": row.get("latency_mean_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "Median",
                    "value": row.get("latency_median_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "P95",
                    "value": row.get("latency_p95_ms", 0),
                },
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "P99",
                    "value": row.get("latency_p99_ms", 0),
                },
            ]
        )

    if not latency_data:
        fig = go.Figure()
        fig.update_layout(title="No Latency Data Available")
        return fig

    latency_df = pd.DataFrame(latency_data)

    # Create grouped bar chart - group by model and dataset combination
    latency_df["model_dataset"] = latency_df["model"] + " (" + latency_df["dataset"] + ")"

    fig = px.bar(
        latency_df,
        x="model_dataset",
        y="value",
        color="metric",
        barmode="group",
        title="Latency Breakdown per Model",
        labels={
            "model_dataset": "Model (Dataset)",
            "value": "Latency (ms)",
            "metric": "Latency Metric",
        },
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_tickangle=-45,
    )

    return fig


def create_ttft_chart(results_df):
    """Create bar chart showing Time to First Token (TTFT) per model."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "latency_ttft_ms": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    # Filter out rows with no TTFT data
    aggregated_df = aggregated_df.dropna(subset=["latency_ttft_ms"])

    if aggregated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No TTFT Data Available")
        return fig

    # Create bar chart for TTFT
    fig = px.bar(
        aggregated_df,
        x="model_name",
        y="latency_ttft_ms",
        color="dataset",
        barmode="group",
        title="Time to First Token (TTFT) per Model",
        labels={"model_name": "Model", "latency_ttft_ms": "TTFT (ms)", "dataset": "Dataset"},
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)

    return fig


def create_throughput_accuracy_scatter(results_df):
    """Create scatter plot of throughput vs accuracy."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    fig = px.scatter(
        results_df,
        x="throughput_samples_per_sec",
        y="metric_accuracy",
        color="model_name",
        symbol="dataset",
        title="Throughput vs. Accuracy Scatter",
        labels={
            "throughput_samples_per_sec": "Throughput (samples/sec)",
            "metric_accuracy": "Accuracy",
            "model_name": "Model",
            "dataset": "Dataset",
        },
        hover_data=["model_name", "dataset", "metric_f1"],
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def create_throughput_comparison_chart(results_df):
    """Create horizontal bar chart comparing throughput."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Aggregate data by model and dataset to prevent stacking of duplicates
    aggregated_df = (
        results_df.groupby(["model_name", "dataset"])
        .agg(
            {
                "throughput_samples_per_sec": "mean"  # Take mean if multiple runs exist
            }
        )
        .reset_index()
    )

    fig = px.bar(
        aggregated_df,  # Use aggregated data instead of raw results_df
        x="throughput_samples_per_sec",
        y="model_name",
        color="dataset",
        orientation="h",
        barmode="group",
        title="Comparative Throughput Analysis",
        labels={
            "throughput_samples_per_sec": "Throughput (samples/sec)",
            "model_name": "Model",
            "dataset": "Dataset",
        },
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def create_efficiency_bubble_chart(results_df):
    """Create bubble chart showing latency vs accuracy with throughput as size."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Filter out rows with NaN values in critical columns for bubble chart
    plot_data = results_df.dropna(
        subset=["latency_mean_ms", "metric_accuracy", "throughput_samples_per_sec"]
    )

    if plot_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Complete Data Available for Efficiency Chart",
            annotations=[
                dict(
                    text="Run benchmarks to generate throughput and latency data",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return fig

    fig = px.scatter(
        plot_data,
        x="latency_mean_ms",
        y="metric_accuracy",
        size="throughput_samples_per_sec",
        color="model_name",
        hover_data=["dataset", "metric_f1"],
        title="Model Efficiency Chart (Latency vs Accuracy, Size = Throughput)",
        labels={
            "latency_mean_ms": "Latency (ms)",
            "metric_accuracy": "Accuracy",
            "throughput_samples_per_sec": "Throughput",
            "model_name": "Model",
        },
    )

    fig.update_layout(height=400, showlegend=True)

    return fig


def load_nps_results():
    """Load NPS estimation results from the results directory."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return pd.DataFrame()

    # Find NPS result files
    nps_files = list(results_dir.glob("estimated_nps_results.json"))

    if not nps_files:
        return pd.DataFrame()

    # Load the most recent NPS results
    nps_file = max(nps_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(nps_file, "r") as f:
            data = json.load(f)

        if "results" in data:
            return pd.DataFrame(data["results"])
        else:
            return pd.DataFrame(data)

    except Exception as e:
        print(f"Error loading NPS results: {e}")
        return pd.DataFrame()


def create_nps_stacked_chart(nps_df):
    """Create stacked bar chart showing NPS categories grouped by branch."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    # Filter for Disneyland data
    dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
    if "branch" in nps_df.columns:
        disneyland_data = nps_df[
            (nps_df[dataset_col] == "disneyland") & (nps_df["branch"].notna())
        ].copy()
    else:
        disneyland_data = nps_df[nps_df[dataset_col] == "disneyland"].copy()

    if disneyland_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No Disneyland NPS Data Available")
        return fig

    # Prepare data for stacked bar chart grouped by branch
    has_branch_col = "branch" in disneyland_data.columns

    if has_branch_col:
        # When we have branch data, show branch + model combinations as before
        plot_data = []
        for _, row in disneyland_data.iterrows():
            model_name = row["model_name"]
            branch = row["branch"].replace("Disneyland_", "")
            label = f"{branch}<br>{model_name}"

            # Add detractors (bottom layer)
            plot_data.append(
                {
                    "branch_model": label,
                    "category": "Detractors",
                    "percentage": row["detractors_percent"],
                    "count": row["detractors_count"],
                    "nps_score": row["nps_score"],
                }
            )

            # Add passives (middle layer)
            plot_data.append(
                {
                    "branch_model": label,
                    "category": "Passives",
                    "percentage": row["passives_percent"],
                    "count": row["passives_count"],
                    "nps_score": row["nps_score"],
                }
            )

            # Add promoters (top layer)
            plot_data.append(
                {
                    "branch_model": label,
                    "category": "Promoters",
                    "percentage": row["promoters_percent"],
                    "count": row["promoters_count"],
                    "nps_score": row["nps_score"],
                }
            )
    else:
        # When there's no branch data, aggregate by model across all training datasets
        aggregated_data = []
        for model_name, group in disneyland_data.groupby("model_name"):
            # Sum counts across all training datasets for this model
            total_promoters = group["promoters_count"].sum()
            total_passives = group["passives_count"].sum()
            total_detractors = group["detractors_count"].sum()
            total_samples = total_promoters + total_passives + total_detractors

            # Recalculate percentages so they add up to 100%
            if total_samples > 0:
                promoters_pct = (total_promoters / total_samples) * 100
                passives_pct = (total_passives / total_samples) * 100
                detractors_pct = (total_detractors / total_samples) * 100
                nps_score = promoters_pct - detractors_pct
            else:
                promoters_pct = passives_pct = detractors_pct = nps_score = 0

            aggregated_data.append({
                "model_name": model_name,
                "promoters_count": total_promoters,
                "passives_count": total_passives,
                "detractors_count": total_detractors,
                "promoters_percent": promoters_pct,
                "passives_percent": passives_pct,
                "detractors_percent": detractors_pct,
                "nps_score": nps_score,
            })

        # Prepare plot data from aggregated results
        plot_data = []
        for model_data in aggregated_data:
            model_name = model_data["model_name"]

            # Add detractors (bottom layer)
            plot_data.append(
                {
                    "branch_model": model_name,
                    "category": "Detractors",
                    "percentage": model_data["detractors_percent"],
                    "count": model_data["detractors_count"],
                    "nps_score": model_data["nps_score"],
                }
            )

            # Add passives (middle layer)
            plot_data.append(
                {
                    "branch_model": model_name,
                    "category": "Passives",
                    "percentage": model_data["passives_percent"],
                    "count": model_data["passives_count"],
                    "nps_score": model_data["nps_score"],
                }
            )

            # Add promoters (top layer)
            plot_data.append(
                {
                    "branch_model": model_name,
                    "category": "Promoters",
                    "percentage": model_data["promoters_percent"],
                    "count": model_data["promoters_count"],
                    "nps_score": model_data["nps_score"],
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart with appropriate title and labels
    chart_title = "NPS Categories Grouped by Branch" if has_branch_col else "NPS Categories by Model"
    x_label = "Branch<br>Model" if has_branch_col else "Model"

    fig = px.bar(
        plot_df,
        x="branch_model",
        y="percentage",
        color="category",
        title=chart_title,
        labels={
            "branch_model": x_label,
            "percentage": "Percentage (%)",
            "category": "NPS Category",
        },
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=True, barmode="stack")

    return fig


def calculate_actual_nps_from_ratings(nps_df, branch):
    """Calculate actual NPS from customer ratings in the Disney dataset for a specific branch.

    Maps star ratings to NPS categories:
    - 5 stars → Promoter (9-10 on NPS scale)
    - 4 stars → Passive (7-8 on NPS scale)
    - 1-3 stars → Detractor (0-6 on NPS scale)

    Args:
        nps_df: DataFrame with NPS results (not used, but kept for compatibility)
        branch: Branch name (California, Paris, or HongKong)

    Returns:
        Dictionary with NPS metrics or None if data unavailable
    """
    try:
        import kagglehub
        import yaml

        # Load dataset configuration
        config_path = Path("config/datasets.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dataset_config = config["datasets"]["disneyland"]

        # Check for cached dataset
        cache_dir = Path("data/raw")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_slug = dataset_config["dataset_id"].replace("/", "_")
        local_cache_path = cache_dir / f"{dataset_slug}.csv"

        if local_cache_path.exists():
            # Load from cache
            try:
                df = pd.read_csv(local_cache_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(local_cache_path, encoding="latin-1")
        else:
            # Download and cache
            dataset_path = kagglehub.dataset_download(dataset_config["dataset_id"])

            # Find the CSV file
            import os

            csv_files = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                return None

            csv_path = csv_files[0]

            # Load CSV
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1")

            # Cache locally
            df.to_csv(local_cache_path, index=False)

        # Filter by branch
        branch_df = df[df["Branch"] == f"Disneyland_{branch}"].copy()

        if branch_df.empty:
            return None

        # Calculate NPS from star ratings
        # NPS mapping: 5★→Promoter, 4★→Passive, 1-3★→Detractor
        promoters = len(branch_df[branch_df["Rating"] == 5])
        passives = len(branch_df[branch_df["Rating"] == 4])
        detractors = len(branch_df[branch_df["Rating"] <= 3])

        total_samples = len(branch_df)

        if total_samples == 0:
            return None

        promoters_pct = (promoters / total_samples) * 100
        passives_pct = (passives / total_samples) * 100
        detractors_pct = (detractors / total_samples) * 100
        nps_score = promoters_pct - detractors_pct

        return {
            "nps_score": nps_score,
            "promoters_percent": promoters_pct,
            "passives_percent": passives_pct,
            "detractors_percent": detractors_pct,
            "branch": branch,
            "total_samples": total_samples,
        }

    except Exception as e:
        print(f"Error calculating actual NPS for {branch}: {e}")
        return None


def create_nps_gauge_chart(nps_data, branch_name):
    """Create a gauge chart showing NPS score for a branch."""
    if not nps_data:
        # Empty gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=0,
                gauge={"axis": {"range": [-100, 100]}},
            )
        )
        fig.update_layout(height=300)
        return fig

    nps_score = nps_data["nps_score"]

    # Determine color based on NPS score
    if nps_score >= 30:
        color = "green"
    elif nps_score >= 0:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=nps_score,
            gauge={
                "axis": {"range": [-100, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [-100, 0], "color": "lightcoral"},
                    {"range": [0, 30], "color": "lightyellow"},
                    {"range": [30, 100], "color": "lightgreen"},
                ],
            },
        )
    )

    fig.update_layout(height=300)
    return fig


def create_nps_stacked_chart_for_branch(nps_df, branch):
    """Create stacked bar chart showing NPS categories for a specific branch."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No NPS Data for {branch}")
        return fig

    # Filter for specific branch
    dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
    if "branch" in nps_df.columns:
        branch_data = nps_df[
            (nps_df[dataset_col] == "disneyland") & (nps_df["branch"] == f"Disneyland_{branch}")
        ].copy()
    else:
        # For aggregated results, use all Disneyland data (no branch filtering)
        branch_data = nps_df[nps_df[dataset_col] == "disneyland"].copy()

    if branch_data.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No Data for {branch}")
        return fig

    # Aggregate counts by model (across all training datasets)
    aggregated_data = []
    for model_name, group in branch_data.groupby("model_name"):
        # Sum counts across all training datasets for this model
        total_promoters = group["promoters_count"].sum()
        total_passives = group["passives_count"].sum()
        total_detractors = group["detractors_count"].sum()
        total_samples = total_promoters + total_passives + total_detractors

        # Recalculate percentages so they add up to 100%
        if total_samples > 0:
            promoters_pct = (total_promoters / total_samples) * 100
            passives_pct = (total_passives / total_samples) * 100
            detractors_pct = (total_detractors / total_samples) * 100
            nps_score = promoters_pct - detractors_pct
        else:
            promoters_pct = passives_pct = detractors_pct = nps_score = 0

        aggregated_data.append({
            "model_name": model_name,
            "promoters_count": total_promoters,
            "passives_count": total_passives,
            "detractors_count": total_detractors,
            "promoters_percent": promoters_pct,
            "passives_percent": passives_pct,
            "detractors_percent": detractors_pct,
            "nps_score": nps_score,
        })

    # Prepare data for stacked bar chart
    plot_data = []
    for model_data in aggregated_data:
        model_name = model_data["model_name"]

        # Add detractors (bottom layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Detractors",
                "percentage": model_data["detractors_percent"],
                "count": model_data["detractors_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add passives (middle layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Passives",
                "percentage": model_data["passives_percent"],
                "count": model_data["passives_count"],
                "nps_score": model_data["nps_score"],
            }
        )

        # Add promoters (top layer)
        plot_data.append(
            {
                "model": model_name,
                "category": "Promoters",
                "percentage": model_data["promoters_percent"],
                "count": model_data["promoters_count"],
                "nps_score": model_data["nps_score"],
            }
        )

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart
    fig = px.bar(
        plot_df,
        x="model",
        y="percentage",
        color="category",
        labels={"model": "Model", "percentage": "Percentage (%)", "category": "NPS Category"},
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=280,
        showlegend=False,
        barmode="stack",
        margin=dict(t=10, b=40, l=40, r=10),
    )

    return fig


def create_nps_accuracy_chart_by_model(nps_df, model_name):
    """Create bar chart showing accuracy by training dataset for a specific model.

    Args:
        nps_df: DataFrame with NPS results
        model_name: Model name (e.g., "BERT-tiny", "DistilBERT-base")

    Returns:
        Plotly figure with bars for each training dataset and an average line
    """
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    # Filter for Disneyland data with valid accuracy for this model
    disneyland_data = nps_df[
        (nps_df["test_dataset"] == "disneyland")
        & (nps_df["accuracy_percent"].notna())
        & (nps_df["model_name"] == model_name)
    ].copy()

    # Remove entries with errors
    if "error" in disneyland_data.columns:
        disneyland_data = disneyland_data[disneyland_data["error"].isna()]

    if disneyland_data.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    # Group by training dataset and calculate average accuracy
    accuracy_data = []
    for dataset, group in disneyland_data.groupby("training_dataset"):
        avg_accuracy = group["accuracy_percent"].mean()
        accuracy_data.append({"training_dataset": dataset, "accuracy_percent": avg_accuracy})

    if not accuracy_data:
        fig = go.Figure()
        fig.update_layout(margin=dict(t=10, b=40, l=40, r=10))
        return fig

    accuracy_df = pd.DataFrame(accuracy_data)

    # Calculate overall average across all training datasets
    overall_avg = accuracy_df["accuracy_percent"].mean()

    # Calculate dynamic y-axis range
    min_accuracy = accuracy_df["accuracy_percent"].min()
    max_accuracy = 100

    # Set y-axis to start slightly below the minimum value (5% below or at least at 0)
    y_min = max(0, min(min_accuracy - 5, overall_avg - 10))

    # Color map for each training dataset
    color_map = {
        "imdb": "#3b82f6",  # blue
        "sst2": "#8b5cf6",  # purple
        "amazon": "#06b6d4",  # cyan
        "yelp": "#10b981",  # emerald
    }

    # Create bar chart
    fig = go.Figure()

    # Add bars for each training dataset
    for _, row in accuracy_df.iterrows():
        dataset = row["training_dataset"]
        accuracy = row["accuracy_percent"]
        fig.add_trace(
            go.Bar(
                x=[dataset],
                y=[accuracy],
                name=dataset,
                marker_color=color_map.get(dataset, "#3b82f6"),
                text=f"{accuracy:.1f}%",
                textposition="outside",
                textfont_size=9,
                showlegend=False,
            )
        )

    # Add average line
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["training_dataset"].tolist(),
            y=[overall_avg] * len(accuracy_df),
            mode="lines",
            name=f"Average: {overall_avg:.1f}%",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
        )
    )

    fig.update_layout(
        xaxis_title="Training Dataset",
        yaxis_title="Accuracy (%)",
        xaxis_tickangle=0,
        height=280,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=10, b=40, l=40, r=10),
        yaxis=dict(range=[y_min, max_accuracy]),
    )

    return fig


def create_nps_distribution_chart(nps_df):
    """Create stacked bar chart showing NPS category distribution."""
    if nps_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No NPS Data Available")
        return fig

    # Filter out entries with errors
    valid_data = nps_df.dropna(
        subset=["promoters_percent", "passives_percent", "detractors_percent"]
    )

    if valid_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No Valid NPS Data Available")
        return fig

    # Prepare data for stacked bar chart
    plot_data = []
    for _, row in valid_data.iterrows():
        model_dataset = f"{row['model_name']}<br>({row['dataset']})"
        plot_data.extend(
            [
                {
                    "model_dataset": model_dataset,
                    "category": "Promoters",
                    "percentage": row["promoters_percent"],
                    "count": row["promoters_count"],
                },
                {
                    "model_dataset": model_dataset,
                    "category": "Passives",
                    "percentage": row["passives_percent"],
                    "count": row["passives_count"],
                },
                {
                    "model_dataset": model_dataset,
                    "category": "Detractors",
                    "percentage": row["detractors_percent"],
                    "count": row["detractors_count"],
                },
            ]
        )

    plot_df = pd.DataFrame(plot_data)

    # Create stacked bar chart
    fig = px.bar(
        plot_df,
        x="model_dataset",
        y="percentage",
        color="category",
        title="NPS Category Distribution by Model",
        labels={
            "model_dataset": "Model (Dataset)",
            "percentage": "Percentage (%)",
            "category": "NPS Category",
        },
        color_discrete_map={
            "Promoters": "#22c55e",  # green
            "Passives": "#eab308",  # yellow
            "Detractors": "#ef4444",  # red
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=300, showlegend=False, barmode="stack")

    return fig


def create_nps_summary_table(nps_df):
    """Create formatted summary table for NPS results."""
    if nps_df.empty:
        return pd.DataFrame()

    # Filter out entries with errors and select relevant columns
    valid_data = nps_df.dropna(subset=["nps_score"]).copy()

    if valid_data.empty:
        return pd.DataFrame()

    # Handle column name differences between old and new formats
    dataset_col = "test_dataset" if "test_dataset" in valid_data.columns else "dataset"

    # Format the table
    columns_to_select = [
        "model_name",
        dataset_col,
        "nps_score",
        "promoters_percent",
        "passives_percent",
        "detractors_percent",
    ]

    # Add training_dataset column if it exists (for new format)
    if "training_dataset" in valid_data.columns:
        columns_to_select.insert(2, "training_dataset")

    display_df = valid_data[columns_to_select].copy()

    # Rename columns for display
    column_names = [
        "Model",
        "Dataset",
        "NPS Score",
        "Promoters (%)",
        "Passives (%)",
        "Detractors (%)",
    ]
    if "training_dataset" in valid_data.columns:
        column_names.insert(2, "Training Dataset")

    display_df.columns = column_names

    # Format percentages
    for col in ["Promoters (%)", "Passives (%)", "Detractors (%)"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

    # Format NPS score
    display_df["NPS Score"] = display_df["NPS Score"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )

    return display_df


def update_nps_dashboard():
    """Update NPS dashboard with latest results."""
    # Load latest NPS data
    nps_df = load_nps_results()

    # Get available model families from the data
    model_families = []
    if not nps_df.empty:
        dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
        # For cross-dataset evaluation, we don't have branch-specific data
        if "branch" in nps_df.columns:
            disneyland_data = nps_df[
                (nps_df[dataset_col] == "disneyland") & (nps_df["branch"].notna())
            ]
        else:
            disneyland_data = nps_df[nps_df[dataset_col] == "disneyland"]
        if not disneyland_data.empty:
            # Extract model families (e.g., "BERT" from "BERT-tiny")
            model_names = disneyland_data["model_name"].tolist()
            model_families = sorted(set(name.split("-")[0] for name in model_names))

    # Create visualizations showing all models
    nps_stacked_fig = create_nps_stacked_chart(nps_df)

    # Get unique models for accuracy charts
    accuracy_chart_models = []
    if not nps_df.empty:
        dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
        disneyland_data = nps_df[
            (nps_df[dataset_col] == "disneyland") & (nps_df["accuracy_percent"].notna())
        ]
        if not disneyland_data.empty:
            # Get unique models, sorted alphabetically
            accuracy_chart_models = sorted(disneyland_data["model_name"].unique().tolist())

    # Create accuracy charts for all available models
    nps_accuracy_charts = []
    for model_name in accuracy_chart_models:
        chart = create_nps_accuracy_chart_by_model(nps_df, model_name)
        nps_accuracy_charts.append((model_name, chart))

    nps_table = create_nps_summary_table(nps_df)

    # Create summary markdown
    if not nps_df.empty:
        dataset_col = "test_dataset" if "test_dataset" in nps_df.columns else "dataset"
        if "branch" in nps_df.columns:
            disneyland_results = nps_df[
                (nps_df[dataset_col] == "disneyland") & (nps_df["branch"].notna())
            ].copy()  # Ensure it's a DataFrame
        else:
            disneyland_results = nps_df[nps_df[dataset_col] == "disneyland"].copy()

        if not disneyland_results.empty:
            # Calculate stats across all branches/models
            avg_nps = disneyland_results["nps_score"].mean()
            has_branch_col = "branch" in disneyland_results.columns

            # Find best performing entry (branch or model)
            best_nps = float("-inf")
            best_identifier = "N/A"
            for idx, row in disneyland_results.iterrows():
                if row["nps_score"] > best_nps:
                    best_nps = row["nps_score"]
                    if has_branch_col and pd.notna(row.get("branch")):
                        best_identifier = str(row["branch"]).replace("Disneyland_", "")
                    else:
                        # Use model name and training dataset instead
                        model = row["model_name"]
                        training_ds = row.get("training_dataset", "unknown")
                        best_identifier = f"{model} (trained on {training_ds})"

            best_nps_score = best_nps
            avg_accuracy = disneyland_results["accuracy_percent"].mean()

            if has_branch_col:
                summary_md = f"""## 🎯 Estimated Net Promoter Score (e-NPS) Analysis

**Disneyland Parks Overview:**
- **Branches Analyzed:** California, Paris, HongKong
- **Models Evaluated:** {len(set(disneyland_results["model_name"]))}
- **Average NPS Score:** {avg_nps:.1f}
- **Best Performing Branch:** {best_identifier} ({best_nps_score:.1f} NPS)
- **Average Accuracy:** {avg_accuracy:.1f}%

**NPS Categories:**
- **Promoters (Green):** High-confidence positive predictions (9-10 on NPS scale)
- **Passives (Yellow):** Medium-confidence positive or uncertain predictions (7-8)
- **Detractors (Red):** Negative predictions (0-6)

**NPS Formula:** % Promoters - % Detractors

*💡 NPS scores range from -100 to 100. Higher scores indicate better customer loyalty.*
"""
            else:
                training_datasets = "N/A"
                if "training_dataset" in disneyland_results.columns:
                    training_datasets = len(set(disneyland_results["training_dataset"]))

                summary_md = f"""## 🎯 Estimated Net Promoter Score (e-NPS) Analysis

**Cross-Dataset Evaluation:**
- **Models Evaluated:** {len(set(disneyland_results["model_name"]))}
- **Training Datasets:** {training_datasets}
- **Average NPS Score:** {avg_nps:.1f}
- **Best Performing:** {best_identifier} ({best_nps_score:.1f} NPS)
- **Average Accuracy:** {avg_accuracy:.1f}%

**NPS Categories:**
- **Promoters (Green):** High-confidence positive predictions (9-10 on NPS scale)
- **Passives (Yellow):** Medium-confidence positive or uncertain predictions (7-8)
- **Detractors (Red):** Negative predictions (0-6)

**NPS Formula:** % Promoters - % Detractors

*💡 NPS scores range from -100 to 100. Higher scores indicate better customer loyalty.*
"""
        else:
            summary_md = """
### No Disneyland NPS Results

Run NPS estimation on Disneyland dataset:

```bash
moodbench estimated-nps --all-models --datasets disneyland
```

This will evaluate all trained models on Disneyland customer reviews.
"""
    else:
        summary_md = """
### No NPS Results Available

Run NPS estimation first:

```bash
moodbench estimated-nps --all-models --datasets disneyland
```

This will evaluate all trained models on Disneyland customer reviews.
"""

    # Calculate actual NPS for each branch
    california_nps = calculate_actual_nps_from_ratings(nps_df, "California")
    paris_nps = calculate_actual_nps_from_ratings(nps_df, "Paris")
    hongkong_nps = calculate_actual_nps_from_ratings(nps_df, "HongKong")

    california_gauge_fig = create_nps_gauge_chart(california_nps, "California")
    paris_gauge_fig = create_nps_gauge_chart(paris_nps, "Paris")
    hongkong_gauge_fig = create_nps_gauge_chart(hongkong_nps, "Hong Kong")

    california_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "California")
    paris_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "Paris")
    hongkong_chart_fig = create_nps_stacked_chart_for_branch(nps_df, "HongKong")

    if not nps_df.empty and not nps_table.empty:
        return (
            summary_md,
            california_gauge_fig,
            paris_gauge_fig,
            hongkong_gauge_fig,
            california_chart_fig,
            paris_chart_fig,
            hongkong_chart_fig,
            nps_accuracy_charts,  # Return list of (model_name, chart) tuples
            nps_table,
            gr.Markdown(visible=False),
        )
    else:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Data Available")
        return (
            summary_md,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            [],  # Empty list of charts
            gr.Dataframe(visible=False),
            gr.Markdown(value="### No NPS data available. Run estimation first."),
        )


def create_latency_distribution_chart(results_df):
    """Create range plot showing latency percentiles per model."""
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Create figure
    fig = go.Figure()

    # Get unique models and datasets
    unique_combinations = results_df.groupby(["model_name", "dataset"]).size().reset_index()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    color_idx = 0

    for _, combo in unique_combinations.iterrows():
        model = combo["model_name"]
        dataset = combo["dataset"]

        # Get data for this model-dataset combination
        row = results_df[(results_df["model_name"] == model) & (results_df["dataset"] == dataset)]
        if row.empty:
            continue

        row = row.iloc[0]

        # Get percentile values
        median = row.get("latency_median_ms", 0)
        p95 = row.get("latency_p95_ms", 0)
        p99 = row.get("latency_p99_ms", 0)

        if median == 0 and p95 == 0 and p99 == 0:
            continue

        model_dataset = f"{model}<br>({dataset})"
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # Add range from median to p95
        fig.add_trace(
            go.Scatter(
                x=[model_dataset, model_dataset],
                y=[median, p95],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{model} ({dataset}) - Median to P95",
                showlegend=True,
            )
        )

        # Add range from p95 to p99
        fig.add_trace(
            go.Scatter(
                x=[model_dataset, model_dataset],
                y=[p95, p99],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{model} ({dataset}) - P95 to P99",
                showlegend=True,
            )
        )

        # Add markers for percentiles
        fig.add_trace(
            go.Scatter(
                x=[model_dataset, model_dataset, model_dataset],
                y=[median, p95, p99],
                mode="markers",
                marker=dict(color=color, size=8, symbol=["diamond", "square", "triangle-up"]),
                name=f"{model} ({dataset}) - Percentiles",
                text=[f"Median: {median:.1f}ms", f"P95: {p95:.1f}ms", f"P99: {p99:.1f}ms"],
                hovertemplate="%{text}",
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title="Latency Distribution per Model (Percentile Ranges)",
        xaxis_title="Model (Dataset)",
        yaxis_title="Latency (ms)",
        height=400,
        showlegend=False,
    )

    return fig


def update_dashboard():
    """Update dashboard with latest benchmark results."""
    # Load latest data
    results_df = load_benchmark_results()
    summary = create_dashboard_summary(results_df)

    # Create all charts
    scatter_fig = create_scatter_plot(results_df)
    accuracy_by_dataset_fig = create_accuracy_by_dataset_chart(results_df)
    f1_by_dataset_fig = create_f1_by_dataset_chart(results_df)
    latency_breakdown_fig = create_latency_breakdown_chart(results_df)
    ttft_fig = create_ttft_chart(results_df)
    throughput_accuracy_scatter_fig = create_throughput_accuracy_scatter(results_df)
    throughput_comparison_fig = create_throughput_comparison_chart(results_df)
    efficiency_bubble_fig = create_efficiency_bubble_chart(results_df)
    latency_distribution_fig = create_latency_distribution_chart(results_df)

    # Create summary markdown
    summary_md = f"""## 📊 Results Summary

- **Total Models:** {summary["total_models"]}
- **Total Datasets:** {summary["total_datasets"]}
- **Total Results:** {summary["total_results"]}
- **Last Updated:** {summary["last_updated"]}
- **Top Model:** {summary["top_model"]}

*💡 Use the **Reports** tab for data export and detailed analysis*"""

    if not results_df.empty:
        # Format the results table with proper column names and data formatting
        display_df = format_benchmark_results_table(results_df)
        return (
            summary_md,
            scatter_fig,
            accuracy_by_dataset_fig,
            f1_by_dataset_fig,
            latency_breakdown_fig,
            ttft_fig,
            throughput_accuracy_scatter_fig,
            throughput_comparison_fig,
            efficiency_bubble_fig,
            latency_distribution_fig,
            display_df,
            gr.Markdown(visible=False),
        )
    else:
        no_data_md = """
### No Results Available

Run some benchmarks first to see results here:

1. Go to the **Benchmark** tab
2. Select models and datasets
3. Click **Start Benchmark**
4. Click **🔄 Refresh Analysis** to view results
"""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Data Available", xaxis_title="Accuracy", yaxis_title="Latency (ms)"
        )
        return (
            summary_md,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            gr.Dataframe(visible=False),
            no_data_md,
        )


def format_benchmark_results_table(results_df):
    """Format benchmark results dataframe with proper column names and data formatting."""
    if results_df.empty:
        return pd.DataFrame()

    # Create a copy to avoid modifying the original
    formatted_df = results_df.copy()

    # Rename columns
    column_mapping = {
        "model_name": "Model",
        "dataset": "Dataset",
        "metric_accuracy": "Accuracy",
        "metric_f1": "F1 Score",
        "latency_mean_ms": "Mean (ms)",
        "latency_median_ms": "Median (ms)",
        "latency_p95_ms": "P95 (ms)",
        "latency_p99_ms": "P99 (ms)",
        "latency_ttft_ms": "TTFT (ms)",
        "throughput_samples_per_sec": "Throughput",
    }

    formatted_df = formatted_df.rename(columns=column_mapping)

    # Format percentage columns (Accuracy and F1 Score)
    if "Accuracy" in formatted_df.columns:
        formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    if "F1 Score" in formatted_df.columns:
        formatted_df["F1 Score"] = formatted_df["F1 Score"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    # Format latency columns (trim to hundredths place)
    latency_columns = ["Mean (ms)", "Median (ms)", "P95 (ms)", "P99 (ms)", "TTFT (ms)"]
    for col in latency_columns:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A"
            )

    # Format throughput column (trim to hundredths place)
    if "Throughput" in formatted_df.columns:
        formatted_df["Throughput"] = formatted_df["Throughput"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A"
        )

    # Remove internal columns
    formatted_df = formatted_df.drop(columns=["timestamp", "_source_file"], errors="ignore")

    return formatted_df


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="MoodBench - LLM Benchmark UI") as interface:
        gr.Markdown("""
        # 🤖 MoodBench - Multi-LLM Sentiment Analysis Benchmark

        **Fast benchmarking of small language models (4M-410M parameters) for sentiment analysis.**

        This interface provides access to all MoodBench functionality through an intuitive web UI.
        """)

        with gr.Tabs():
            # Tab 1: Train Models
            with gr.TabItem("🚀 Train Models"):
                gr.Markdown("### Train multiple models on multiple datasets")

                gr.Markdown("""
                ⚠️ **Note about training timeouts:** Due to Gradio's request timeout limitations, training larger models or complex configurations may fail in the web interface.
                For long-running trainings, use the command line interface instead:

                ```bash
                uv run moodbench train --model <model_name> --dataset <dataset_name>
                ```

                The web interface works best for quick training runs on smaller models like BERT-tiny and BERT-mini.
                """)

                with gr.Row():
                    # Column 1: Inputs and training matrix
                    with gr.Column():
                        models_checkboxes = gr.CheckboxGroup(
                            choices=DEFAULT_MODELS,
                            label="Models to Train",
                            value=["BERT-tiny", "BERT-mini"],
                            info="Select one or more models to train",
                        )
                        datasets_checkboxes = gr.CheckboxGroup(
                            choices=DEFAULT_DATASETS,
                            label="Datasets to Train On",
                            value=["imdb"],
                            info="Select one or more datasets to train on",
                        )

                        # Training status matrix
                        training_matrix = create_training_matrix()
                        matrix_display = gr.HTML(training_matrix)
                        gr.Markdown("*Green cells (✅) indicate trained combinations*")

                        # Refresh button for the matrix
                        refresh_matrix_btn = gr.Button("🔄 Refresh Status", size="sm")
                        refresh_matrix_btn.click(
                            fn=lambda: create_training_matrix(), outputs=matrix_display
                        )

                    # Column 2: Progress and output
                    with gr.Column():
                        train_progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Training Progress (%)",
                            interactive=False,
                        )
                        with gr.Accordion("Training Details", open=False):
                            train_output = gr.Textbox(
                                label="Status",
                                lines=15,
                                interactive=False,
                                show_copy_button=True,
                                autoscroll=True,
                            )

                        train_button = gr.Button("🚀 Start Training", variant="primary")
                        train_button.click(
                            fn=train_models,
                            inputs=[models_checkboxes, datasets_checkboxes],
                            outputs=[train_progress_bar, train_output],
                        )

            # Tab 3: Benchmark
            with gr.TabItem("📊 Benchmark"):
                gr.Markdown("### Run benchmark on selected models and datasets")

                with gr.Row():
                    with gr.Column():
                        models_checkboxes_benchmark = gr.CheckboxGroup(
                            choices=DEFAULT_MODELS,
                            label="Models to Benchmark",
                            value=["BERT-tiny"],
                            info="Select models to include in benchmark",
                        )
                        datasets_checkboxes_benchmark = gr.CheckboxGroup(
                            choices=DEFAULT_DATASETS,
                            label="Datasets to Test On",
                            value=["imdb"],
                            info="Select datasets to evaluate models on",
                        )

                    with gr.Column():
                        benchmark_progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Progress (%)",
                            interactive=False,
                        )
                        with gr.Accordion("Benchmark Details", open=False):
                            benchmark_output = gr.Textbox(
                                label="Status",
                                lines=15,
                                interactive=False,
                                show_copy_button=True,
                                autoscroll=True,
                            )

                benchmark_button = gr.Button("🏁 Start Benchmark", variant="primary")
                benchmark_button.click(
                    fn=run_benchmark,
                    inputs=[models_checkboxes_benchmark, datasets_checkboxes_benchmark],
                    outputs=[benchmark_progress_bar, benchmark_output],
                )

            # Tab 5: Generate Reports
            with gr.TabItem("📋 Generate Reports"):
                gr.Markdown("### Generate comparison reports from benchmark results")

                with gr.Row():
                    with gr.Column():
                        report_markdown = gr.Markdown(
                            label="📊 Generated Report",
                            value="",
                            height=400,
                        )

                    with gr.Column():
                        report_progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Generation Progress (%)",
                            interactive=False,
                        )
                        with gr.Accordion("Generation Details", open=False):
                            report_output = gr.Textbox(
                                label="Status",
                                lines=10,
                                interactive=False,
                                show_copy_button=True,
                                autoscroll=True,
                            )

                report_button = gr.Button("📋 Generate Reports", variant="primary")
                report_button.click(
                    fn=generate_reports,
                    inputs=[],
                    outputs=[report_progress, report_output, report_markdown],
                )

            # Tab 6: Analysis
            with gr.TabItem("📊 Analysis"):
                gr.Markdown("### Results Analysis")

                # Refresh button at the top
                refresh_btn = gr.Button("🔄 Refresh Analysis", variant="secondary")

                # Create updatable components with initial data
                initial_results = load_benchmark_results()
                initial_summary = create_dashboard_summary(initial_results)
                initial_scatter_fig = create_scatter_plot(initial_results)
                initial_accuracy_fig = create_accuracy_by_dataset_chart(initial_results)
                initial_f1_fig = create_f1_by_dataset_chart(initial_results)
                initial_latency_breakdown_fig = create_latency_breakdown_chart(initial_results)
                initial_ttft_fig = create_ttft_chart(initial_results)
                initial_throughput_accuracy_fig = create_throughput_accuracy_scatter(
                    initial_results
                )
                initial_throughput_comparison_fig = create_throughput_comparison_chart(
                    initial_results
                )
                initial_efficiency_bubble_fig = create_efficiency_bubble_chart(initial_results)
                initial_latency_distribution_fig = create_latency_distribution_chart(
                    initial_results
                )

                initial_summary_md = f"""## 📊 Results Summary

- **Total Models:** {initial_summary["total_models"]}
- **Total Datasets:** {initial_summary["total_datasets"]}
- **Total Results:** {initial_summary["total_results"]}
- **Last Updated:** {initial_summary["last_updated"]}
- **Top Model:** {initial_summary["top_model"]}

*💡 Use the **Reports** tab for data export and detailed analysis*"""

                summary_text = gr.Markdown(value=initial_summary_md)

                # Row 1: Accuracy and Latency charts
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model Accuracy by Dataset")
                        accuracy_by_dataset_plot = gr.Plot(value=initial_accuracy_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Model Accuracy by Dataset** shows how well each model performs on different datasets.

                            - **Accuracy**: The percentage of predictions that are correct (higher is better)
                            - **Grouped bars**: Each model shows separate bars for different datasets
                            - **Color coding**: Different colors represent different datasets
                            - **Comparison**: Use this to see which models work best on specific types of data
                            """)
                    with gr.Column():
                        gr.Markdown("#### Latency Breakdown per Model")
                        latency_breakdown_plot = gr.Plot(value=initial_latency_breakdown_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Latency Breakdown per Model** shows the speed characteristics of each model across different latency metrics.

                            - **Mean**: Average response time across all predictions
                            - **Median**: Middle value when response times are sorted (less affected by outliers)
                            - **P95**: 95th percentile - 95% of responses are faster than this time
                            - **P99**: 99th percentile - 99% of responses are faster than this time
                            - **Lower values**: Indicate faster model inference (better for real-time applications)
                            """)

                # Row 1.5: F1 Score and TTFT charts
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model F1 Score by Dataset")
                        f1_by_dataset_plot = gr.Plot(value=initial_f1_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Model F1 Score by Dataset** measures the balance between precision and recall for each model.

                            - **F1 Score**: Harmonic mean of precision and recall (ranges from 0 to 1)
                            - **Precision**: Of all positive predictions, what percentage were actually correct
                            - **Recall**: Of all actual positive cases, what percentage were correctly identified
                            - **F1 Formula**: 2 × (precision × recall) / (precision + recall)
                            - **Higher values**: Indicate better balance between avoiding false positives and false negatives
                            """)
                    with gr.Column():
                        gr.Markdown("#### Time to First Token (TTFT)")
                        ttft_plot = gr.Plot(value=initial_ttft_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Time to First Token (TTFT)** measures how quickly each model starts generating responses.

                            - **TTFT**: Time from when a request is made until the first token is generated
                            - **Lower values**: Indicate faster initial response times (better user experience)
                            - **Important for**: Streaming responses, interactive applications, and perceived responsiveness
                            - **Different from total latency**: TTFT is just the "time to start", not the complete response time
                            """)

                # Row 2: Throughput and Efficiency charts
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Throughput vs. Accuracy Scatter")
                        throughput_accuracy_scatter_plot = gr.Plot(
                            value=initial_throughput_accuracy_fig
                        )
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Throughput vs. Accuracy Scatter** shows the trade-off between speed and quality for each model.

                            - **X-axis (Throughput)**: Number of samples processed per second (higher = faster)
                            - **Y-axis (Accuracy)**: Percentage of correct predictions (higher = more accurate)
                            - **Color coding**: Different colors represent different models
                            - **Shape coding**: Different shapes represent different datasets
                            - **Trade-off analysis**: Helps identify models that balance speed and accuracy well
                            """)
                    with gr.Column():
                        gr.Markdown("#### Model Efficiency Chart")
                        efficiency_bubble_plot = gr.Plot(value=initial_efficiency_bubble_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Model Efficiency Chart** combines three key performance dimensions in one visualization.

                            - **X-axis (Latency)**: Mean response time in milliseconds (lower = faster)
                            - **Y-axis (Accuracy)**: Percentage of correct predictions (higher = better)
                            - **Bubble size (Throughput)**: Larger bubbles = higher throughput (more samples/second)
                            - **Color coding**: Different colors represent different models
                            - **Multi-dimensional analysis**: Identifies the "sweet spot" balancing speed, quality, and capacity
                            """)

                # Row 3: Throughput comparison and latency distribution
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Comparative Throughput Analysis")
                        throughput_comparison_plot = gr.Plot(
                            value=initial_throughput_comparison_fig
                        )
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Comparative Throughput Analysis** shows the processing capacity of each model across datasets.

                            - **Throughput**: Number of samples/inferences processed per second
                            - **Horizontal bars**: Each model shows separate bars for different datasets
                            - **Higher values**: Indicate models that can handle more requests per second
                            - **Color coding**: Different colors represent different datasets
                            - **Capacity planning**: Use this to understand which models can handle higher loads
                            """)
                    with gr.Column():
                        gr.Markdown("#### Latency Distribution per Model")
                        latency_distribution_plot = gr.Plot(value=initial_latency_distribution_fig)
                        with gr.Accordion("📖 What does this chart show?", open=False):
                            gr.Markdown("""
                            **Latency Distribution per Model** shows the range and consistency of response times.

                            - **Median to P95**: Solid lines show the range from typical response time to 95th percentile
                            - **P95 to P99**: Dashed lines show the range from 95th to 99th percentile (tail latency)
                            - **Markers**: Diamond (median), square (P95), triangle (P99) show key percentiles
                            - **Shorter ranges**: Indicate more consistent/predictable response times
                            - **Tail latency analysis**: Helps identify models with reliable performance under load
                            """)

                # Row 4: Original scatter plot
                gr.Markdown("#### Model Performance: Latency vs Accuracy")
                scatter_plot = gr.Plot(value=initial_scatter_fig)
                with gr.Accordion("📖 What does this chart show?", open=False):
                    gr.Markdown("""
                    **Average Accuracy by Model** shows the overall accuracy of each model across all Disneyland locations.

                    - **Accuracy**: Average percentage of correct predictions against actual star ratings (1-5)
                    - **Higher values**: Indicate better alignment between sentiment analysis and customer ratings
                    - **Averaged across locations**: Single value per model combining California, Paris, and Hong Kong results
                    - **Cross-domain evaluation**: Models trained on other datasets (Yelp, IMDB) tested on Disneyland reviews
                    - **Percentage labels**: Values shown as formatted percentages on each bar
                    """)

                if not initial_results.empty:
                    display_df = format_benchmark_results_table(initial_results)
                    results_table = gr.Dataframe(value=display_df, label="Benchmark Results Table")
                    no_data_message = gr.Markdown(visible=False)
                else:
                    results_table = gr.Dataframe(visible=False)
                    no_data_message = gr.Markdown(
                        value="""
### No Results Available

Run some benchmarks first to see results here:

1. Go to the **Benchmark** tab
2. Select models and datasets
3. Click **Start Benchmark**
4. Click **🔄 Refresh Analysis** to view results
"""
                    )

                # Set up refresh functionality
                refresh_btn.click(
                    fn=update_dashboard,
                    inputs=[],
                    outputs=[
                        summary_text,
                        scatter_plot,
                        accuracy_by_dataset_plot,
                        f1_by_dataset_plot,
                        latency_breakdown_plot,
                        ttft_plot,
                        throughput_accuracy_scatter_plot,
                        throughput_comparison_plot,
                        efficiency_bubble_plot,
                        latency_distribution_plot,
                        results_table,
                        no_data_message,
                    ],
                )

                # Initial load
                refresh_btn.click(
                    fn=update_dashboard,
                    inputs=[],
                    outputs=[
                        summary_text,
                        scatter_plot,
                        accuracy_by_dataset_plot,
                        f1_by_dataset_plot,
                        latency_breakdown_plot,
                        ttft_plot,
                        throughput_accuracy_scatter_plot,
                        throughput_comparison_plot,
                        efficiency_bubble_plot,
                        latency_distribution_plot,
                        results_table,
                        no_data_message,
                    ],
                    queue=False,  # Don't queue the initial load
                )

            # Tab 7: e-NPS (Estimated Net Promoter Score)
            with gr.TabItem("🎯 e-NPS"):
                gr.Markdown("### Estimated Net Promoter Score Analysis")
                gr.Markdown("""
                **Estimated NPS (e-NPS)** measures customer loyalty based on model predictions.

                This analysis estimates Net Promoter Score from sentiment analysis predictions,
                mapping confidence scores to NPS categories (Promoters, Passives, Detractors).
                """)

                # Refresh button
                nps_refresh_btn = gr.Button("🔄 Refresh e-NPS Analysis", variant="secondary")

                # Create updatable components with initial NPS data
                initial_nps_df = load_nps_results()

                # Calculate actual NPS for each branch
                california_nps = calculate_actual_nps_from_ratings(initial_nps_df, "California")
                paris_nps = calculate_actual_nps_from_ratings(initial_nps_df, "Paris")
                hongkong_nps = calculate_actual_nps_from_ratings(initial_nps_df, "HongKong")

                initial_california_gauge = create_nps_gauge_chart(california_nps, "California")
                initial_paris_gauge = create_nps_gauge_chart(paris_nps, "Paris")
                initial_hongkong_gauge = create_nps_gauge_chart(hongkong_nps, "Hong Kong")

                initial_california_chart = create_nps_stacked_chart_for_branch(
                    initial_nps_df, "California"
                )
                initial_paris_chart = create_nps_stacked_chart_for_branch(initial_nps_df, "Paris")
                initial_hongkong_chart = create_nps_stacked_chart_for_branch(
                    initial_nps_df, "HongKong"
                )

                # Get unique models for accuracy charts
                initial_accuracy_chart_models = []
                if not initial_nps_df.empty:
                    dataset_col = "test_dataset" if "test_dataset" in initial_nps_df.columns else "dataset"
                    disneyland_data = initial_nps_df[
                        (initial_nps_df[dataset_col] == "disneyland")
                        & (initial_nps_df["accuracy_percent"].notna())
                    ]
                    if not disneyland_data.empty:
                        initial_accuracy_chart_models = sorted(
                            disneyland_data["model_name"].unique().tolist()
                        )

                # Create accuracy charts for all available models
                initial_nps_accuracy_charts = []
                for model_name in initial_accuracy_chart_models:
                    chart = create_nps_accuracy_chart_by_model(initial_nps_df, model_name)
                    initial_nps_accuracy_charts.append((model_name, chart))

                initial_nps_table = create_nps_summary_table(initial_nps_df)

                # Summary
                nps_summary_text = gr.Markdown(value="Loading NPS data...")

                # NPS Analysis by Location
                gr.Markdown("#### NPS Analysis by Location")
                gr.Markdown(
                    "*NPS Mapping: 5★ = Promoter, 4★ = Passive, 1-3★ = Detractor. "
                    "NPS Score = % Promoters - % Detractors*",
                    elem_classes=["text-sm", "text-gray-600"],
                )

                # California Row
                gr.Markdown("##### 🇺🇸 California")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                        california_gauge = gr.Plot(value=initial_california_gauge)
                    with gr.Column(scale=2):
                        gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                        california_plot = gr.Plot(value=initial_california_chart)

                # Paris Row
                gr.Markdown("##### 🇫🇷 Paris")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                        paris_gauge = gr.Plot(value=initial_paris_gauge)
                    with gr.Column(scale=2):
                        gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                        paris_plot = gr.Plot(value=initial_paris_chart)

                # Hong Kong Row
                gr.Markdown("##### 🇭🇰 Hong Kong")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Actual NPS Score**<br>(from customer ratings)")
                        hongkong_gauge = gr.Plot(value=initial_hongkong_gauge)
                    with gr.Column(scale=2):
                        gr.Markdown("**NPS Categories by Model**<br>(estimated from predictions)")
                        hongkong_plot = gr.Plot(value=initial_hongkong_chart)

                # Accuracy Charts by Model
                gr.Markdown("#### Model Accuracy by Training Dataset")
                gr.Markdown(
                    "*Comparison of how each model performs when trained on different datasets*"
                )

                # Create charts in rows of 3
                accuracy_chart_components = []
                num_charts = len(initial_nps_accuracy_charts)

                for i in range(0, num_charts, 3):
                    with gr.Row():
                        for j in range(3):
                            idx = i + j
                            if idx < num_charts:
                                model_name, chart = initial_nps_accuracy_charts[idx]
                                with gr.Column(scale=1):
                                    name_component = gr.Markdown(value=f"**{model_name}**")
                                    chart_component = gr.Plot(value=chart)
                                    accuracy_chart_components.append((name_component, chart_component))
                            else:
                                # Placeholder empty column
                                with gr.Column(scale=1):
                                    pass

                with gr.Accordion("📖 What do these charts show?", open=False):
                    gr.Markdown("""
                    **NPS Categories by Location** shows the breakdown of predictions into NPS categories for each Disneyland park separately.

                    - **Promoters (Green)**: High-confidence positive predictions (9-10 on NPS scale)
                    - **Passives (Yellow)**: Medium-confidence positive or uncertain predictions (7-8)
                    - **Detractors (Red)**: Negative predictions (0-6)
                    - **Separate charts**: One chart per Disneyland location (California, Paris, Hong Kong)
                    - **Model comparison**: Each location chart compares all evaluated models
                    """)

                # NPS Results Table
                if not initial_nps_table.empty:
                    nps_results_table = gr.Dataframe(
                        value=initial_nps_table, label="NPS Results Summary"
                    )
                    nps_no_data_message = gr.Markdown(visible=False)
                else:
                    nps_results_table = gr.Dataframe(visible=False)
                    nps_no_data_message = gr.Markdown(
                        value="""
### No NPS Data Available

Run NPS estimation first:

```bash
moodbench estimated-nps --all-models --all-datasets
```

This will evaluate all trained models and estimate NPS from their predictions.
"""
                    )

                # Create wrapper function to handle dynamic charts
                def update_nps_dashboard_wrapper():
                    result = update_nps_dashboard()
                    summary_md, ca_gauge, pa_gauge, hk_gauge, ca_plot, pa_plot, hk_plot, charts_list, table, no_data = result

                    # Flatten charts_list into individual name/chart pairs
                    flattened_outputs = [summary_md, ca_gauge, pa_gauge, hk_gauge, ca_plot, pa_plot, hk_plot]

                    # Add chart names and plots
                    for name_comp, chart_comp in accuracy_chart_components:
                        idx = accuracy_chart_components.index((name_comp, chart_comp))
                        if idx < len(charts_list):
                            model_name, chart = charts_list[idx]
                            flattened_outputs.extend([f"**{model_name}**", chart])
                        else:
                            flattened_outputs.extend(["", go.Figure()])

                    flattened_outputs.extend([table, no_data])
                    return flattened_outputs

                # Flatten accuracy chart components into outputs list
                accuracy_outputs = []
                for name_comp, chart_comp in accuracy_chart_components:
                    accuracy_outputs.extend([name_comp, chart_comp])

                # Set up refresh functionality
                nps_refresh_btn.click(
                    fn=update_nps_dashboard_wrapper,
                    inputs=[],
                    outputs=[
                        nps_summary_text,
                        california_gauge,
                        paris_gauge,
                        hongkong_gauge,
                        california_plot,
                        paris_plot,
                        hongkong_plot,
                    ]
                    + accuracy_outputs
                    + [
                        nps_results_table,
                        nps_no_data_message,
                    ],
                )

            # Tab 6: Methodology & Data Documentation
            with gr.TabItem("📚 Methodology"):
                gr.Markdown("""
# Methodology & Data Documentation

## 📊 Data Provenance

### Disney Customer Reviews Dataset
- **Source**: [Kaggle - Disneyland Reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews)
- **Size**: 42,000+ customer reviews
- **Locations**:
  - Disneyland California (USA)
  - Disneyland Paris (France)
  - Hong Kong Disneyland
- **Rating Scale**: 1-5 stars
- **Content**: Customer-written text reviews with associated ratings

### Training Datasets
Models are fine-tuned on four standard sentiment analysis benchmarks:
- **IMDB**: [Stanford Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) - 50k movie reviews
- **SST-2**: [Stanford Sentiment Treebank](https://huggingface.co/datasets/stanfordnlp/sst2) - Single sentence reviews
- **Amazon**: [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity) - Product reviews
- **Yelp**: [Yelp Polarity](https://huggingface.co/datasets/yelp_polarity) - Business reviews

All datasets are publicly available and accessed via Hugging Face Datasets or Kaggle.

---

## 🧮 Calculation Methodology

### Actual NPS from Customer Ratings
**Actual NPS** gauges show the true customer sentiment based on their star ratings:

**Rating Mapping:**
```
5 stars → Promoter    (9-10 on NPS scale)
4 stars → Passive     (7-8 on NPS scale)
1-3 stars → Detractor (0-6 on NPS scale)
```

**Calculation:**
```python
NPS Score = (% Promoters) - (% Detractors)
```

**Example:**
- If 50% gave 5★, 30% gave 4★, and 20% gave 1-3★
- NPS = 50% - 20% = **+30**

**Range**: -100 (all detractors) to +100 (all promoters)

### Estimated NPS from Model Predictions
**Estimated NPS** charts show how models predict sentiment:

**Model Output Mapping:**
- Models predict **binary sentiment** (positive/negative) with **confidence scores**
- High-confidence positive (>0.85) → Promoter
- Medium-confidence positive (0.60-0.85) → Passive
- Negative predictions → Detractor

**Purpose**: Compare model predictions against actual customer sentiment to assess model quality.

### Model Accuracy by Training Dataset
**Accuracy** measures how well models predict the correct sentiment (positive/negative):

**Calculation:**
```python
Accuracy = (Correct Predictions / Total Predictions) × 100%
```

**Charts show:**
- X-axis: Training dataset (IMDB, SST-2, Amazon, Yelp)
- Y-axis: Accuracy percentage
- Red dashed line: Average accuracy across all training datasets

**Interpretation**: Higher accuracy = better transfer learning from training dataset to Disney reviews.

### NPS Categories by Model
**Stacked bar charts** show the distribution of predictions:
- **Green (Promoters)**: Models predict high customer satisfaction
- **Yellow (Passives)**: Models predict moderate satisfaction
- **Red (Detractors)**: Models predict dissatisfaction

**Aggregation**: Counts are summed across all training dataset variants of each base model, then normalized to 100%.

---

## ⚠️ Caveats & Limitations

### Data Quality
- **Sample Bias**: Kaggle dataset may not represent all Disney customers
- **Self-Selection**: Online reviewers may have stronger opinions than average visitors
- **Time Period**: Reviews reflect historical sentiment, not current conditions
- **Language**: Analysis assumes English-language reviews only

### Model Limitations
- **Binary Sentiment**: Models only predict positive/negative, missing nuanced emotions
- **Context Window**: Longer reviews may be truncated (max 512 tokens)
- **Training Domain Mismatch**: Models trained on movies/products may not fully understand theme park experiences
- **Confidence Calibration**: Confidence thresholds (0.85, 0.60) are heuristic, not rigorously calibrated

### NPS Methodology
- **Non-Standard Mapping**: Traditional NPS uses 0-10 scale; we map 1-5 stars with assumptions
- **Missing Neutrality**: 4-star reviews (Passives) may include both satisfied and slightly dissatisfied customers
- **Aggregation Effects**: Combining different training datasets may introduce noise

### Cross-Dataset Evaluation
- **Domain Shift**: Theme park reviews differ significantly from movie/product reviews
- **Generalization**: High accuracy on training data doesn't guarantee Disney review accuracy
- **Class Imbalance**: Disney reviews may skew more positive than training datasets

### Technical Constraints
- **Model Size**: Small models (4M-410M parameters) have inherent performance limits
- **Fine-Tuning**: LoRA adapters may not capture full semantic complexity
- **Quantization**: 4-bit quantization trades accuracy for memory efficiency

---

## 🎯 Recommended Interpretation

### When to Trust the Data
- **Relative Comparisons**: Comparing models or locations within this dataset
- **Trend Identification**: Identifying which training datasets transfer better
- **Model Selection**: Choosing which model architecture works best for this domain

### When to Be Cautious
- **Absolute NPS Values**: Don't compare these NPS scores to industry benchmarks
- **Real-World Decisions**: Don't base business decisions solely on these estimates
- **Causation Claims**: Correlation between training data and accuracy doesn't imply causation
- **Generalization**: Results may not apply to other theme parks or time periods

### Best Practices
1. **Use as Proxy**: Treat estimated NPS as a proxy for model quality, not ground truth
2. **Compare Within Dataset**: Only compare metrics within this controlled experiment
3. **Validate Externally**: Cross-reference findings with official Disney satisfaction data if available
4. **Consider Ensemble**: Average predictions from multiple models for more robust estimates
5. **Monitor Drift**: Re-evaluate if applying to significantly newer review data

---

## 📖 Additional Resources

- **Hugging Face Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **LoRA Fine-Tuning**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **Net Promoter Score**: [https://en.wikipedia.org/wiki/Net_promoter_score](https://en.wikipedia.org/wiki/Net_promoter_score)
- **Sentiment Analysis Datasets**: [https://paperswithcode.com/task/sentiment-analysis](https://paperswithcode.com/task/sentiment-analysis)

---

*This documentation was generated as part of the MoodBench project to provide transparency about data sources, calculation methods, and appropriate interpretation of results.*
                """)

    return interface


def main():
    """Main entry point."""
    if not GRADIO_AVAILABLE:
        print("Gradio is required. Install with: pip install gradio")
        return

    interface = create_interface()

    # Launch the interface
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
