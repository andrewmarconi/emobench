"""
Visualization CLI commands for EmoBench.
"""

import logging

logger = logging.getLogger(__name__)


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
