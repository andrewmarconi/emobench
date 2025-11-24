"""
Evaluation metrics for SentiCompare.

Comprehensive metrics computation for sentiment analysis models including:
- Classification metrics (accuracy, F1, precision, recall)
- Per-class metrics
- Confusion matrix
- ROC-AUC, PR-AUC
- Statistical analysis
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for classification tasks.

    Examples:
        >>> calculator = MetricsCalculator(num_classes=2)
        >>> predictions = [0, 1, 1, 0, 1]
        >>> labels = [0, 1, 0, 0, 1]
        >>> metrics = calculator.compute_metrics(predictions, labels)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Predicted class labels
            labels: True class labels
            probabilities: Predicted probabilities (optional, for AUC metrics)

        Returns:
            Dict[str, float]: Dictionary of all metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(predictions, labels))

        # Confusion matrix
        metrics.update(self._compute_confusion_metrics(predictions, labels))

        # Per-class metrics
        if self.num_classes == 2:
            metrics.update(self._compute_binary_metrics(predictions, labels, probabilities))

        # Statistical metrics
        metrics.update(self._compute_statistical_metrics(predictions, labels))

        return metrics

    def _compute_basic_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        avg = "binary" if self.num_classes == 2 else "weighted"

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average=avg, zero_division=0),
            "precision": precision_score(labels, predictions, average=avg, zero_division=0),
            "recall": recall_score(labels, predictions, average=avg, zero_division=0),
        }

    def _compute_confusion_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute confusion matrix based metrics."""
        cm = confusion_matrix(labels, predictions)

        metrics = {}

        if self.num_classes == 2:
            # Binary classification
            tn, fp, fn, tp = cm.ravel()

            # Specificity (True Negative Rate)
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Sensitivity (same as recall)
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # False Positive Rate
            metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # False Negative Rate
            metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # True positives, etc.
            metrics["true_positives"] = float(tp)
            metrics["true_negatives"] = float(tn)
            metrics["false_positives"] = float(fp)
            metrics["false_negatives"] = float(fn)

        return metrics

    def _compute_binary_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute binary classification specific metrics."""
        metrics = {}

        if probabilities is not None:
            try:
                # ROC-AUC
                if len(probabilities.shape) > 1:
                    # If probabilities are 2D, use positive class
                    probs = probabilities[:, 1]
                else:
                    probs = probabilities

                metrics["roc_auc"] = roc_auc_score(labels, probs)

                # PR-AUC (Average Precision)
                metrics["pr_auc"] = average_precision_score(labels, probs)
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")

        return metrics

    def _compute_statistical_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistical agreement metrics."""
        return {
            "matthews_corrcoef": matthews_corrcoef(labels, predictions),
            "cohen_kappa": cohen_kappa_score(labels, predictions),
        }

    def get_classification_report(
        self, predictions: np.ndarray, labels: np.ndarray, target_names: Optional[List[str]] = None
    ) -> str:
        """
        Get detailed classification report.

        Args:
            predictions: Predicted labels
            labels: True labels
            target_names: Names for each class

        Returns:
            str: Classification report
        """
        return classification_report(
            labels, predictions, target_names=target_names, zero_division=0
        )

    def get_confusion_matrix(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            predictions: Predicted labels
            labels: True labels

        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(labels, predictions)


def compute_metrics_from_trainer(eval_pred) -> Dict[str, float]:
    """
    Compute metrics from HuggingFace Trainer evaluation.

    This is a convenience function for use with Trainer.compute_metrics.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dict[str, float]: Computed metrics
    """
    predictions, labels = eval_pred

    # If predictions are logits, convert to class predictions
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
        probs = predictions
    else:
        preds = predictions
        probs = None

    calculator = MetricsCalculator(num_classes=2)
    return calculator.compute_metrics(preds, labels, probs)


def compare_metrics(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two sets of metrics.

    Args:
        metrics1: First metrics dictionary
        metrics2: Second metrics dictionary
        metric_names: Specific metrics to compare (None = all)

    Returns:
        Dict: Comparison results with differences and improvements
    """
    if metric_names is None:
        metric_names = set(metrics1.keys()) & set(metrics2.keys())

    comparison = {}

    for metric in metric_names:
        if metric not in metrics1 or metric not in metrics2:
            continue

        val1 = metrics1[metric]
        val2 = metrics2[metric]

        comparison[metric] = {
            "model1": val1,
            "model2": val2,
            "diff": val2 - val1,
            "improvement": ((val2 - val1) / val1 * 100) if val1 != 0 else 0.0,
        }

    return comparison


def aggregate_metrics(
    metrics_list: List[Dict[str, float]], aggregate_fn: str = "mean"
) -> Dict[str, float]:
    """
    Aggregate multiple metric dictionaries.

    Args:
        metrics_list: List of metric dictionaries
        aggregate_fn: Aggregation function ("mean", "median", "std")

    Returns:
        Dict[str, float]: Aggregated metrics
    """
    if not metrics_list:
        return {}

    # Get all metric names
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    aggregated = {}

    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]

        if not values:
            continue

        if aggregate_fn == "mean":
            aggregated[key] = np.mean(values)
        elif aggregate_fn == "median":
            aggregated[key] = np.median(values)
        elif aggregate_fn == "std":
            aggregated[key] = np.std(values)
        elif aggregate_fn == "min":
            aggregated[key] = np.min(values)
        elif aggregate_fn == "max":
            aggregated[key] = np.max(values)

    return aggregated


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

    # Group metrics by category
    basic = ["accuracy", "f1", "precision", "recall"]
    auc = ["roc_auc", "pr_auc"]
    stat = ["matthews_corrcoef", "cohen_kappa"]
    confusion = ["specificity", "sensitivity", "fpr", "fnr"]

    def print_group(group_name: str, keys: List[str]):
        printed = False
        for key in keys:
            if key in metrics:
                if not printed:
                    print(f"\n{group_name}:")
                    printed = True
                print(f"  {key:<20} {metrics[key]:.4f}")

    print_group("Classification Metrics", basic)
    print_group("AUC Metrics", auc)
    print_group("Statistical Metrics", stat)
    print_group("Confusion Metrics", confusion)

    # Print any remaining metrics
    remaining = set(metrics.keys()) - set(basic + auc + stat + confusion)
    if remaining:
        print(f"\nOther Metrics:")
        for key in sorted(remaining):
            value = metrics[key]
            if isinstance(value, (int, float)):
                print(f"  {key:<20} {value:.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: Calculate metrics
    np.random.seed(42)

    # Generate sample predictions and labels
    n_samples = 100
    labels = np.random.randint(0, 2, n_samples)
    predictions = labels.copy()

    # Add some errors (10% error rate)
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]

    # Generate probabilities
    probabilities = np.random.rand(n_samples, 2)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Calculate metrics
    calculator = MetricsCalculator(num_classes=2)
    metrics = calculator.compute_metrics(predictions, labels, probabilities)

    print_metrics(metrics, "Sample Evaluation Metrics")

    # Print classification report
    print("Classification Report:")
    print("=" * 60)
    report = calculator.get_classification_report(
        predictions, labels, target_names=["Negative", "Positive"]
    )
    print(report)

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("=" * 60)
    cm = calculator.get_confusion_matrix(predictions, labels)
    print(cm)
    print()
