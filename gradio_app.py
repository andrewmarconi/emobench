"""
Simple Gradio Web UI for EmoBench - Multi-LLM Sentiment Analysis Benchmark Framework

A basic web interface demonstrating EmoBench functionality.
Install required dependencies: pip install gradio
"""

import os
import sys
import subprocess
import time
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
    "GPT2-small",
]

DEFAULT_DATASETS = ["imdb", "sst2", "amazon", "yelp"]


def run_command_stream(command_list, timeout=300):
    """Run a command and stream output in real-time."""
    try:
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)

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

        markdown_path = "experiments/reports/emobench_report.md"
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

        except Exception as e:
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
        timestamps = pd.to_datetime(results_df["timestamp"], errors="coerce")
        if not timestamps.empty:
            summary["last_updated"] = timestamps.max().strftime("%Y-%m-%d %H:%M:%S")

    return summary


def create_scatter_plot(results_df):
    """Create a scatter plot of accuracy vs latency."""
    if results_df.empty or "metric_accuracy" not in results_df.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available", xaxis_title="Accuracy", yaxis_title="Latency (ms)"
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

    # Create scatter plot
    fig = px.scatter(
        plot_data,
        x="metric_accuracy",
        y="latency_mean_ms",
        color="model_name",
        hover_data=["dataset", "model_name"],
        title="Model Performance: Accuracy vs Latency",
        labels={
            "metric_accuracy": "Accuracy",
            "latency_mean_ms": "Latency (ms)",
            "model_name": "Model",
        },
    )

    fig.update_layout(
        xaxis_title="Accuracy",
        yaxis_title="Latency (ms)",
        font=dict(size=12),
        title_font=dict(size=16),
    )

    return fig


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="EmoBench - LLM Benchmark UI") as interface:
        gr.Markdown("""
        # 🤖 EmoBench - Multi-LLM Sentiment Analysis Benchmark

        **Fast benchmarking of small language models (4M-410M parameters) for sentiment analysis.**

        This interface provides access to all EmoBench functionality through an intuitive web UI.
        """)

        with gr.Tabs():
            # Tab 1: Train Models
            with gr.TabItem("🚀 Train Models"):
                gr.Markdown("### Train multiple models on multiple datasets")

                with gr.Row():
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

                    with gr.Column():
                        progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Training Progress (%)",
                            interactive=False,
                        )
                        with gr.Accordion("Training Details", open=False):
                            train_output = gr.Textbox(
                                label="Training Status",
                                lines=15,
                                interactive=False,
                                show_copy_button=True,
                                autoscroll=True,
                            )

                train_button = gr.Button("🚀 Start Training", variant="primary")
                train_button.click(
                    fn=train_models,
                    inputs=[models_checkboxes, datasets_checkboxes],
                    outputs=[progress_bar, train_output],
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

            # Tab 6: Dashboard
            with gr.TabItem("📈 Dashboard"):
                gr.Markdown("### Results Dashboard")

                # Load data and create dashboard components
                results_df = load_benchmark_results()
                summary = create_dashboard_summary(results_df)
                scatter_fig = create_scatter_plot(results_df)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 📊 Results Summary")
                        gr.Markdown(f"""
                        - **Total Models:** {summary["total_models"]}
                        - **Total Datasets:** {summary["total_datasets"]}
                        - **Total Results:** {summary["total_results"]}
                        - **Last Updated:** {summary["last_updated"]}
                        - **Top Model:** {summary["top_model"]}
                        """)

                        gr.Markdown(
                            "*💡 Use the **Reports** tab for data export and detailed analysis*"
                        )

                    with gr.Column():
                        gr.Markdown("## 📈 Performance Visualization")
                        gr.Plot(scatter_fig)

                if not results_df.empty:
                    gr.Markdown("## 📋 Detailed Results")
                    # Remove internal columns from display
                    display_df = results_df.drop(
                        columns=["timestamp", "_source_file"], errors="ignore"
                    )
                    gr.Dataframe(display_df, label="Benchmark Results Table")
                else:
                    gr.Markdown("""
                    ### No Results Available

                    Run some benchmarks first to see results here:

                    1. Go to the **Benchmark** tab
                    2. Select models and datasets
                    3. Click **Start Benchmark**
                    4. Return here to view results
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
