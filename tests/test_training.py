"""
Unit tests for training modules.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from transformers import TrainingArguments

from src.training.trainer import SentiCompareTrainer
from src.training.optimizer import (
    get_optimizer,
    create_scheduler,
    get_optimizer_info,
)
from src.training.callbacks import (
    DetailedProgressCallback,
    MetricsLoggerCallback,
    CheckpointCleanupCallback,
    MemoryTrackerCallback,
    EvaluationCallback,
    get_default_callbacks,
)


class TestSentiCompareTrainer:
    """Tests for SentiCompareTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.config = Mock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.save_pretrained = Mock()
        return tokenizer

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=100)
        return dataset

    def test_trainer_initialization(self, mock_model, mock_tokenizer, mock_dataset):
        """Test trainer initialization."""
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            device=torch.device("cpu"),
        )

        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.train_dataset == mock_dataset
        assert trainer.eval_dataset == mock_dataset
        assert trainer.device.type == "cpu"

    def test_get_training_args(self, mock_model, mock_tokenizer, mock_dataset):
        """Test training arguments generation."""
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            device=torch.device("cpu"),
        )

        args = trainer.get_training_args()

        assert isinstance(args, TrainingArguments)
        assert args.output_dir == trainer.output_dir
        assert args.per_device_train_batch_size > 0
        assert args.per_device_eval_batch_size > 0

    def test_device_auto_detection(self, mock_model, mock_tokenizer, mock_dataset):
        """Test automatic device detection."""
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
        )

        # Should auto-detect a device
        assert trainer.device is not None
        assert trainer.device.type in ["cuda", "mps", "cpu"]

    def test_device_specific_batch_size(self, mock_model, mock_tokenizer, mock_dataset):
        """Test device-specific batch size selection."""
        # Test CPU
        trainer_cpu = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            device=torch.device("cpu"),
        )
        args_cpu = trainer_cpu.get_training_args()

        # CPU should have smaller batch size
        assert args_cpu.per_device_train_batch_size >= 1

    def test_compute_metrics(self):
        """Test metrics computation."""
        # Mock predictions and labels
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        labels = torch.tensor([1, 0, 1, 0])

        eval_pred = (predictions.numpy(), labels.numpy())
        metrics = SentiCompareTrainer.compute_metrics(eval_pred)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1

    def test_save_model(self, mock_model, mock_tokenizer, mock_dataset, tmp_path):
        """Test model saving."""
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            device=torch.device("cpu"),
        )

        output_dir = str(tmp_path / "test_model")
        trainer.save_model(output_dir)

        # Check that save methods were called
        mock_model.save_pretrained.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)

        # Check directory was created
        assert Path(output_dir).exists()


class TestOptimizerConfig:
    """Tests for optimizer configuration."""

    def test_get_optimizer(self):
        """Test optimizer creation."""
        # Create a simple model
        model = torch.nn.Linear(10, 2)

        optimizer = get_optimizer(
            model, learning_rate=2e-4, weight_decay=0.01, adam_epsilon=1e-8
        )

        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # Decay and no-decay groups

    def test_create_scheduler_linear(self):
        """Test linear scheduler creation."""
        model = torch.nn.Linear(10, 2)
        optimizer = get_optimizer(model)

        scheduler = create_scheduler(
            optimizer, scheduler_type="linear", num_warmup_steps=100, num_training_steps=1000
        )

        assert scheduler is not None

    def test_create_scheduler_cosine(self):
        """Test cosine scheduler creation."""
        model = torch.nn.Linear(10, 2)
        optimizer = get_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None

    def test_create_scheduler_constant(self):
        """Test constant scheduler creation."""
        model = torch.nn.Linear(10, 2)
        optimizer = get_optimizer(model)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="constant",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None

    def test_get_optimizer_info(self):
        """Test optimizer info retrieval."""
        model = torch.nn.Linear(10, 2)
        optimizer = get_optimizer(model, learning_rate=2e-4, weight_decay=0.01)

        info = get_optimizer_info(optimizer)

        assert "type" in info
        assert "num_param_groups" in info
        assert "learning_rate" in info
        assert "total_parameters" in info
        assert info["num_param_groups"] == 2  # Decay and no-decay groups

    def test_optimizer_parameter_groups(self):
        """Test optimizer parameter grouping (decay vs no-decay)."""
        model = torch.nn.Linear(10, 2)
        optimizer = get_optimizer(model, weight_decay=0.01)

        # Should have 2 parameter groups
        assert len(optimizer.param_groups) == 2

        # One group should have weight decay, one should not
        decay_groups = [pg for pg in optimizer.param_groups if pg["weight_decay"] > 0]
        no_decay_groups = [pg for pg in optimizer.param_groups if pg["weight_decay"] == 0]

        assert len(decay_groups) == 1
        assert len(no_decay_groups) == 1


class TestCallbacks:
    """Tests for training callbacks."""

    def test_detailed_progress_callback(self):
        """Test DetailedProgressCallback."""
        callback = DetailedProgressCallback()

        assert callback.start_time is None
        assert callback.epoch_start_time is None

    def test_metrics_logger_callback(self, tmp_path):
        """Test MetricsLoggerCallback."""
        log_dir = str(tmp_path / "logs")
        callback = MetricsLoggerCallback(log_dir=log_dir)

        assert callback.log_dir.exists()
        assert callback.log_file.exists()

        # Check header was written
        with open(callback.log_file, "r") as f:
            header = f.readline().strip()
            assert "step" in header
            assert "epoch" in header
            assert "loss" in header

    def test_checkpoint_cleanup_callback(self):
        """Test CheckpointCleanupCallback."""
        callback = CheckpointCleanupCallback(keep_best_n=3, metric="eval_f1")

        assert callback.keep_best_n == 3
        assert callback.metric == "eval_f1"
        assert len(callback.checkpoint_scores) == 0

    def test_memory_tracker_callback(self):
        """Test MemoryTrackerCallback."""
        callback = MemoryTrackerCallback(log_interval=100)

        assert callback.log_interval == 100
        assert callback.step_count == 0

    def test_evaluation_callback(self):
        """Test EvaluationCallback."""
        callback = EvaluationCallback(eval_dataset_name="validation")

        assert callback.eval_dataset_name == "validation"
        assert callback.best_score is None

    def test_get_default_callbacks(self, tmp_path):
        """Test default callbacks creation."""
        log_dir = str(tmp_path / "logs")

        callbacks = get_default_callbacks(
            log_dir=log_dir, keep_best_n=3, track_memory=True
        )

        assert len(callbacks) == 5  # All 5 callbacks

        # Check types
        assert isinstance(callbacks[0], DetailedProgressCallback)
        assert isinstance(callbacks[1], MetricsLoggerCallback)
        assert isinstance(callbacks[2], CheckpointCleanupCallback)
        assert isinstance(callbacks[3], EvaluationCallback)
        assert isinstance(callbacks[4], MemoryTrackerCallback)

    def test_get_default_callbacks_no_memory(self, tmp_path):
        """Test default callbacks without memory tracking."""
        log_dir = str(tmp_path / "logs")

        callbacks = get_default_callbacks(
            log_dir=log_dir, keep_best_n=3, track_memory=False
        )

        assert len(callbacks) == 4  # Without MemoryTracker

    def test_memory_tracker_get_stats_cpu(self):
        """Test memory stats on CPU."""
        callback = MemoryTrackerCallback()
        stats = callback._get_memory_stats()

        # CPU doesn't have detailed memory stats
        assert stats is None or isinstance(stats, str)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_tracker_get_stats_cuda(self):
        """Test memory stats on CUDA."""
        callback = MemoryTrackerCallback()
        stats = callback._get_memory_stats()

        assert stats is not None
        assert "Allocated" in stats
        assert "GB" in stats

    def test_callback_on_log(self, tmp_path):
        """Test MetricsLoggerCallback on_log method."""
        log_dir = str(tmp_path / "logs")
        callback = MetricsLoggerCallback(log_dir=log_dir)

        # Mock state and logs
        state = Mock()
        state.global_step = 100
        state.epoch = 1.5

        logs = {
            "loss": 0.5,
            "learning_rate": 2e-4,
            "eval_loss": 0.4,
            "eval_accuracy": 0.85,
            "eval_f1": 0.82,
        }

        # Call on_log
        callback.on_log(None, state, None, logs=logs)

        # Check log file
        with open(callback.log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 log entry
            assert "100" in lines[1]  # step
            assert "1.5" in lines[1]  # epoch


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_training_config_loading(self, tmp_path):
        """Test training configuration loading."""
        # Create a minimal config
        config_path = tmp_path / "training.yaml"
        config_content = """
training:
  num_epochs: 2
  learning_rate: 2e-4
  batch_size:
    cpu: 4
    cuda: 8
    mps: 4

evaluation:
  strategy: "steps"
  eval_steps: 100

logging:
  steps: 50
"""
        config_path.write_text(config_content)

        # Create mock objects
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config._name_or_path = "test-model"

        mock_tokenizer = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Create trainer
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            config_path=str(config_path),
            device=torch.device("cpu"),
        )

        # Check config was loaded
        assert trainer.config is not None
        assert "training" in trainer.config
        assert trainer.config["training"]["num_epochs"] == 2

    def test_mlflow_setup_without_mlflow(self, tmp_path):
        """Test MLflow setup when MLflow is not available."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config._name_or_path = "test-model"

        mock_tokenizer = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # This should not raise an error even if MLflow is not installed
        trainer = SentiCompareTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            device=torch.device("cpu"),
        )

        # use_mlflow might be True or False depending on whether mlflow is installed
        assert isinstance(trainer.use_mlflow, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
