"""
Visualization CLI commands for EmoBench.
"""

import logging

logger = logging.getLogger(__name__)


def launch_dashboard(args):
    """
    Launch the Streamlit dashboard.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Launching dashboard on port {args.port}")

    try:
        import subprocess
        import sys
        import os

        # Launch dashboard using subprocess to allow port configuration
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/visualization/dashboard.py",
            "--server.port",
            str(args.port),
            "--",
            "--results-dir",
            str(args.results_dir),
        ]

        # Set environment variables to disable prompts
        env = os.environ.copy()
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env)

        logger.info(f"Dashboard launched successfully on port {args.port}")
        logger.info("Press Ctrl+C to stop the dashboard")

        # Wait for the process (it will run until interrupted)
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping dashboard...")
            process.terminate()
            process.wait()

    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard launch failed: {e}")
        raise
    except ImportError as e:
        logger.error(f"Streamlit not installed: {e}")
        logger.info("Install streamlit with: uv add streamlit")
        raise
    except Exception as e:
        logger.error(f"Dashboard launch failed: {e}")
        raise


def generate_reports(args):
    """
    Generate comparison reports.

    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Generating reports in format: {args.format}")

    try:
        # Import visualization components
        from src.visualization.reports import generate_reports as generate_reports_func

        # Generate reports
        reports = generate_reports_func(
            results_dir=str(args.results_dir), output_dir=str(args.output_dir), format=args.format
        )

        logger.info(f"Reports generated in: {args.output_dir}")

        # Print summary
        for format_name, file_path in reports.items():
            logger.info(f"  {format_name.upper()}: {file_path}")

    except ImportError as e:
        logger.error(f"Visualization module not implemented yet: {e}")
        logger.info("Please implement the visualization module first")
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise
