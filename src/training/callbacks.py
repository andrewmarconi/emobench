"""
Training callbacks for EmoBench.

Custom callbacks for progress tracking, logging, and model checkpointing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import ProgressCallback

logger = logging.getLogger(__name__)


class DetailedProgressCallback(ProgressCallback):
    """
    Enhanced progress callback with additional information.

    Shows training speed, ETA, and memory usage.
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info("=" * 60)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        logger.info(f"\nEpoch {int(state.epoch) + 1}/{int(args.num_train_epochs)}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(f"Epoch completed in {epoch_time:.2f}s")

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info("=" * 60)
            logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
            logger.info("=" * 60)


class MetricsLoggerCallback(TrainerCallback):
    """
    Callback for detailed metrics logging.

    Logs metrics to file and optionally to external services.
    """

    def __init__(self, log_dir: str = "./experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "metrics.log"

        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write("step,epoch,loss,learning_rate,eval_loss,eval_accuracy,eval_f1\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics when available."""
        if logs is None:
            return

        # Write to log file
        with open(self.log_file, "a") as f:
            step = state.global_step
            epoch = state.epoch if state.epoch is not None else 0

            # Training metrics
            loss = logs.get("loss", "")
            lr = logs.get("learning_rate", "")

            # Evaluation metrics
            eval_loss = logs.get("eval_loss", "")
            eval_acc = logs.get("eval_accuracy", "")
            eval_f1 = logs.get("eval_f1", "")

            f.write(f"{step},{epoch},{loss},{lr},{eval_loss},{eval_acc},{eval_f1}\n")


class CheckpointCleanupCallback(TrainerCallback):
    """
    Callback to clean up old checkpoints and save disk space.

    Keeps only the best N checkpoints based on metric.
    """

    def __init__(self, keep_best_n: int = 3, metric: str = "eval_f1"):
        self.keep_best_n = keep_best_n
        self.metric = metric
        self.checkpoint_scores = {}

    def on_save(self, args, state, control, **kwargs):
        """Track checkpoint scores when saved."""
        if state.best_metric is not None:
            checkpoint_folder = f"checkpoint-{state.global_step}"
            self.checkpoint_scores[checkpoint_folder] = state.best_metric

            # Clean up old checkpoints
            self._cleanup_checkpoints(args.output_dir)

    def _cleanup_checkpoints(self, output_dir: str):
        """Remove old checkpoints, keeping only the best N."""
        if len(self.checkpoint_scores) <= self.keep_best_n:
            return

        # Sort by score
        sorted_checkpoints = sorted(
            self.checkpoint_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Keep top N
        to_keep = set(cp[0] for cp in sorted_checkpoints[: self.keep_best_n])

        # Remove others
        output_path = Path(output_dir)
        for checkpoint_folder in self.checkpoint_scores.keys():
            if checkpoint_folder not in to_keep:
                checkpoint_path = output_path / checkpoint_folder
                if checkpoint_path.exists():
                    import shutil

                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_folder}")


class MemoryTrackerCallback(TrainerCallback):
    """
    Track memory usage during training.

    Logs memory statistics for debugging and optimization.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log memory usage periodically."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            memory_stats = self._get_memory_stats()
            if memory_stats:
                logger.info(f"Memory (step {state.global_step}): {memory_stats}")

    def _get_memory_stats(self) -> Optional[str]:
        """Get memory statistics for current device."""
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"

        elif torch.backends.mps.is_available():
            try:
                if hasattr(torch.mps, "current_allocated_memory"):
                    allocated = torch.mps.current_allocated_memory() / 1024**3
                    return f"Allocated: {allocated:.2f}GB"
            except Exception:
                pass

        return None


class EvaluationCallback(TrainerCallback):
    """
    Custom evaluation callback for additional metrics and logging.
    """

    def __init__(self, eval_dataset_name: str = "validation"):
        self.eval_dataset_name = eval_dataset_name
        self.best_score = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation results."""
        if metrics is None:
            return

        # Log evaluation metrics
        logger.info(f"\nEvaluation on {self.eval_dataset_name}:")
        logger.info(f"  Loss:      {metrics.get('eval_loss', 'N/A'):.4f}")
        logger.info(f"  Accuracy:  {metrics.get('eval_accuracy', 'N/A'):.4f}")
        logger.info(f"  F1:        {metrics.get('eval_f1', 'N/A'):.4f}")
        logger.info(f"  Precision: {metrics.get('eval_precision', 'N/A'):.4f}")
        logger.info(f"  Recall:    {metrics.get('eval_recall', 'N/A'):.4f}")

        # Track best score
        f1_score = metrics.get("eval_f1")
        if f1_score is not None:
            if self.best_score is None or f1_score > self.best_score:
                self.best_score = f1_score
                logger.info(f"  ✓ New best F1 score: {f1_score:.4f}")


def get_default_callbacks(
    log_dir: str = "./experiments/logs",
    keep_best_n: int = 3,
    track_memory: bool = True,
) -> list:
    """
    Get list of default callbacks for training.

    Args:
        log_dir: Directory for logs
        keep_best_n: Number of best checkpoints to keep
        track_memory: Whether to track memory usage

    Returns:
        list: List of callback instances
    """
    callbacks = [
        DetailedProgressCallback(),
        MetricsLoggerCallback(log_dir=log_dir),
        CheckpointCleanupCallback(keep_best_n=keep_best_n),
        EvaluationCallback(),
    ]

    if track_memory:
        callbacks.append(MemoryTrackerCallback())

    return callbacks


if __name__ == "__main__":
    # Demo: Show callback usage
    print("Available Callbacks:")
    print("=" * 60)
    print("1. DetailedProgressCallback - Enhanced progress tracking")
    print("2. MetricsLoggerCallback - Log metrics to file")
    print("3. CheckpointCleanupCallback - Clean up old checkpoints")
    print("4. MemoryTrackerCallback - Track memory usage")
    print("5. EvaluationCallback - Enhanced evaluation logging")
    print("=" * 60)

    # Show how to use
    print("\nUsage Example:")
    print("""
from src.training.callbacks import get_default_callbacks

callbacks = get_default_callbacks(
    log_dir='./experiments/logs',
    keep_best_n=3,
    track_memory=True
)

trainer = Trainer(
    model=model,
    args=args,
    callbacks=callbacks,
    ...
)
    """)
