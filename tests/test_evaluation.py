"""
Unit tests for evaluation modules.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.evaluation.metrics import (
    MetricsCalculator,
    compute_metrics_from_trainer,
    compare_metrics,
    aggregate_metrics,
)
from src.evaluation.profiler import (
    MemoryProfiler,
    get_model_memory,
)
from src.evaluation.results import ResultsAggregator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and labels."""
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        predictions = labels.copy()
        # Add 10% errors
        error_idx = np.random.choice(100, 10, replace=False)
        predictions[error_idx] = 1 - predictions[error_idx]
        probabilities = np.random.rand(100, 2)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return predictions, labels, probabilities

    def test_initialization(self):
        """Test calculator initialization."""
        calculator = MetricsCalculator(num_classes=2)
        assert calculator.num_classes == 2

    def test_compute_basic_metrics(self, sample_data):
        """Test basic metrics computation."""
        predictions, labels, _ = sample_data
        calculator = MetricsCalculator(num_classes=2)

        metrics = calculator._compute_basic_metrics(predictions, labels)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1

    def test_compute_confusion_metrics(self, sample_data):
        """Test confusion matrix metrics."""
        predictions, labels, _ = sample_data
        calculator = MetricsCalculator(num_classes=2)

        metrics = calculator._compute_confusion_metrics(predictions, labels)

        assert "specificity" in metrics
        assert "sensitivity" in metrics
        assert "fpr" in metrics
        assert "fnr" in metrics
        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics

    def test_compute_binary_metrics(self, sample_data):
        """Test binary classification metrics."""
        predictions, labels, probabilities = sample_data
        calculator = MetricsCalculator(num_classes=2)

        metrics = calculator._compute_binary_metrics(predictions, labels, probabilities)

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1

    def test_compute_statistical_metrics(self, sample_data):
        """Test statistical agreement metrics."""
        predictions, labels, _ = sample_data
        calculator = MetricsCalculator(num_classes=2)

        metrics = calculator._compute_statistical_metrics(predictions, labels)

        assert "matthews_corrcoef" in metrics
        assert "cohen_kappa" in metrics
        assert -1 <= metrics["matthews_corrcoef"] <= 1
        assert -1 <= metrics["cohen_kappa"] <= 1

    def test_compute_metrics_complete(self, sample_data):
        """Test complete metrics computation."""
        predictions, labels, probabilities = sample_data
        calculator = MetricsCalculator(num_classes=2)

        metrics = calculator.compute_metrics(predictions, labels, probabilities)

        # Check all expected metrics are present
        expected_keys = [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "sensitivity",
            "roc_auc",
            "pr_auc",
            "matthews_corrcoef",
            "cohen_kappa",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_get_classification_report(self, sample_data):
        """Test classification report generation."""
        predictions, labels, _ = sample_data
        calculator = MetricsCalculator(num_classes=2)

        report = calculator.get_classification_report(
            predictions, labels, target_names=["Negative", "Positive"]
        )

        assert isinstance(report, str)
        assert "Negative" in report
        assert "Positive" in report
        assert "precision" in report.lower()
        assert "recall" in report.lower()

    def test_get_confusion_matrix(self, sample_data):
        """Test confusion matrix generation."""
        predictions, labels, _ = sample_data
        calculator = MetricsCalculator(num_classes=2)

        cm = calculator.get_confusion_matrix(predictions, labels)

        assert cm.shape == (2, 2)
        assert np.sum(cm) == len(predictions)


class TestMetricsUtilities:
    """Tests for metrics utility functions."""

    def test_compute_metrics_from_trainer(self):
        """Test compute_metrics_from_trainer function."""
        # Create sample data
        logits = np.random.rand(100, 2)
        labels = np.random.randint(0, 2, 100)

        metrics = compute_metrics_from_trainer((logits, labels))

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_compare_metrics(self):
        """Test metrics comparison."""
        metrics1 = {"accuracy": 0.85, "f1": 0.84, "precision": 0.83}
        metrics2 = {"accuracy": 0.87, "f1": 0.86, "precision": 0.85}

        comparison = compare_metrics(metrics1, metrics2)

        assert "accuracy" in comparison
        assert "f1" in comparison

        # Check structure
        assert "model1" in comparison["accuracy"]
        assert "model2" in comparison["accuracy"]
        assert "diff" in comparison["accuracy"]
        assert "improvement" in comparison["accuracy"]

        # Check values
        assert comparison["accuracy"]["diff"] == pytest.approx(0.02, rel=1e-9)

    def test_aggregate_metrics_mean(self):
        """Test metrics aggregation with mean."""
        metrics_list = [
            {"accuracy": 0.85, "f1": 0.84},
            {"accuracy": 0.87, "f1": 0.86},
            {"accuracy": 0.86, "f1": 0.85},
        ]

        aggregated = aggregate_metrics(metrics_list, aggregate_fn="mean")

        assert "accuracy" in aggregated
        assert "f1" in aggregated
        assert aggregated["accuracy"] == pytest.approx(0.86, rel=1e-2)
        assert aggregated["f1"] == pytest.approx(0.85, rel=1e-2)

    def test_aggregate_metrics_median(self):
        """Test metrics aggregation with median."""
        metrics_list = [
            {"accuracy": 0.85},
            {"accuracy": 0.87},
            {"accuracy": 0.86},
        ]

        aggregated = aggregate_metrics(metrics_list, aggregate_fn="median")

        assert aggregated["accuracy"] == 0.86

    def test_aggregate_metrics_empty(self):
        """Test aggregation with empty list."""
        aggregated = aggregate_metrics([], aggregate_fn="mean")
        assert aggregated == {}


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        device = torch.device("cpu")
        profiler = MemoryProfiler(device)

        assert profiler.device == device
        assert profiler.device_type == "cpu"

    def test_start(self):
        """Test starting profiler."""
        device = torch.device("cpu")
        profiler = MemoryProfiler(device)

        profiler.start()

        assert profiler.initial_memory is not None

    def test_get_stats(self):
        """Test getting memory statistics."""
        device = torch.device("cpu")
        profiler = MemoryProfiler(device)

        profiler.start()
        stats = profiler.get_stats()

        assert "device" in stats
        assert stats["device"] == "cpu"
        assert "current_memory_bytes" in stats
        assert "current_memory_gb" in stats

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_stats(self):
        """Test CUDA-specific stats."""
        device = torch.device("cuda")
        profiler = MemoryProfiler(device)

        profiler.start()
        stats = profiler.get_stats()

        assert "allocated_bytes" in stats
        assert "reserved_bytes" in stats
        assert "peak_allocated_gb" in stats

    def test_reset(self):
        """Test profiler reset."""
        device = torch.device("cpu")
        profiler = MemoryProfiler(device)

        profiler.start()
        profiler.reset()

        assert profiler.initial_memory is None


class TestModelMemory:
    """Tests for model memory utilities."""

    def test_get_model_memory(self):
        """Test model memory calculation."""
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 2)
        )

        memory_info = get_model_memory(model)

        assert "total_params" in memory_info
        assert "trainable_params" in memory_info
        assert "frozen_params" in memory_info
        assert "total_memory_mb" in memory_info

        # Check parameter counts
        assert memory_info["total_params"] > 0
        assert memory_info["trainable_params"] > 0

    def test_frozen_parameters(self):
        """Test memory calculation with frozen parameters."""
        model = torch.nn.Linear(100, 2)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        memory_info = get_model_memory(model)

        assert memory_info["trainable_params"] == 0
        assert memory_info["frozen_params"] > 0


class TestResultsAggregator:
    """Tests for ResultsAggregator class."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = ResultsAggregator(experiment_name="test_exp")
        assert aggregator.experiment_name == "test_exp"
        assert len(aggregator.results) == 0

    def test_add_result(self):
        """Test adding results."""
        aggregator = ResultsAggregator()

        metrics = {"accuracy": 0.85, "f1": 0.84}
        aggregator.add_result("model1", "dataset1", metrics)

        assert len(aggregator.results) == 1
        assert aggregator.results[0]["model"] == "model1"
        assert aggregator.results[0]["dataset"] == "dataset1"
        assert aggregator.results[0]["metrics"] == metrics

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        aggregator = ResultsAggregator()

        # Add multiple results
        for i in range(3):
            metrics = {"accuracy": 0.85 + i * 0.01, "f1": 0.84 + i * 0.01}
            aggregator.add_result(f"model{i}", f"dataset{i}", metrics)

        df = aggregator.to_dataframe()

        assert len(df) == 3
        assert "model" in df.columns
        assert "dataset" in df.columns
        assert "metric_accuracy" in df.columns
        assert "metric_f1" in df.columns

    def test_save_and_load(self, tmp_path):
        """Test saving and loading results."""
        aggregator = ResultsAggregator(experiment_name="test_save")

        metrics = {"accuracy": 0.85, "f1": 0.84}
        aggregator.add_result("model1", "dataset1", metrics)

        # Save
        output_dir = str(tmp_path)
        aggregator.save(output_dir)

        # Check files exist
        json_file = tmp_path / "test_save_results.json"
        csv_file = tmp_path / "test_save_results.csv"

        assert json_file.exists()
        assert csv_file.exists()

        # Load
        new_aggregator = ResultsAggregator()
        new_aggregator.load(str(json_file))

        assert len(new_aggregator.results) == 1
        assert new_aggregator.results[0]["model"] == "model1"

    def test_get_summary(self):
        """Test getting summary statistics."""
        aggregator = ResultsAggregator()

        # Add multiple results
        for i in range(5):
            metrics = {"accuracy": 0.80 + i * 0.02, "f1": 0.79 + i * 0.02}
            aggregator.add_result(f"model{i % 2}", f"dataset{i % 3}", metrics)

        summary = aggregator.get_summary()

        assert "experiment" in summary
        assert "num_results" in summary
        assert "models" in summary
        assert "datasets" in summary
        assert "metrics" in summary

        assert summary["num_results"] == 5
        assert len(summary["models"]) == 2  # model0, model1
        assert len(summary["datasets"]) == 3  # dataset0, dataset1, dataset2

    def test_get_best_models(self):
        """Test getting best models."""
        aggregator = ResultsAggregator()

        # Add results with different F1 scores
        models_f1 = [("model_a", 0.85), ("model_b", 0.90), ("model_c", 0.88)]

        for model, f1 in models_f1:
            metrics = {"accuracy": 0.80, "f1": f1}
            aggregator.add_result(model, "dataset1", metrics)

        # Get top 2 models
        top_models = aggregator.get_best_models(metric="f1", n=2)

        assert len(top_models) == 2
        assert top_models.iloc[0]["model"] == "model_b"  # Highest F1
        assert top_models.iloc[1]["model"] == "model_c"  # Second highest F1

    def test_compare_models(self):
        """Test comparing two models."""
        aggregator = ResultsAggregator()

        # Add results for two models
        metrics1 = {"accuracy": 0.85, "f1": 0.84}
        metrics2 = {"accuracy": 0.87, "f1": 0.86}

        aggregator.add_result("model1", "dataset1", metrics1)
        aggregator.add_result("model2", "dataset1", metrics2)

        # Compare
        comparison = aggregator.compare_models("model1", "model2", dataset="dataset1")

        assert "model1" in comparison
        assert "model2" in comparison
        assert "metrics" in comparison

        assert comparison["metrics"]["accuracy"]["model1"] == 0.85
        assert comparison["metrics"]["accuracy"]["model2"] == 0.87
        assert comparison["metrics"]["accuracy"]["diff"] == pytest.approx(0.02, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
