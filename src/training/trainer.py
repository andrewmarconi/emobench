"""
Training module for EmoBench.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

logger = logging.getLogger(__name__)


class EmoBenchTrainer:
    """
    Custom trainer for EmoBench with MLflow logging.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        config_dir: str = "config",
        device: str = "auto",
    ):
        """
        Initialize trainer.

        Args:
            model_name: Name of the model to train
            dataset_name: Name of the dataset
            config_dir: Configuration directory
            device: Device to use (auto, cuda, mps, cpu)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_dir = Path(config_dir)
        self.device = device
        self.model = None  # type: ignore
        self.tokenizer = None  # type: ignore

    def train(self, training_config: Optional[Dict] = None) -> Dict:
        """
        Train the model.

        Args:
            training_config: Training configuration overrides

        Returns:
            Dict: Training results
        """
        logger.info(f"Starting training for {self.model_name} on {self.dataset_name}")

        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Load the model and tokenizer
        # 2. Load the dataset
        # 3. Set up training arguments
        # 4. Initialize the Trainer
        # 5. Train with MLflow logging
        # 6. Return results

        results = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "status": "completed",
            "message": "Training completed (placeholder implementation)",
        }

        logger.info(f"Training completed for {self.model_name}")
        return results

    def train_all(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
    ) -> Dict:
        """
        Train multiple models.

        Args:
            models: List of models to train
            datasets: List of datasets to use
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dict: Training results for all models
        """
        logger.info("Starting training for all models")

        # Placeholder implementation
        all_results = {}

        if models is None:
            models = ["DistilBERT-base", "RoBERTa-base", "Phi-3-mini"]

        if datasets is None:
            datasets = ["imdb", "sst2", "amazon", "yelp"]

        for model in models:
            model_results = []
            for dataset in datasets:
                # Create new trainer for each model-dataset combination
                trainer = EmoBenchTrainer(model, dataset, str(self.config_dir), self.device)
                result = trainer.train()
                result["dataset"] = dataset
                model_results.append(result)

            all_results[model] = model_results

        logger.info(f"Training completed for {len(models)} models")
        return all_results


def train_model(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    device: str = "auto",
    config_dir: str = "config",
):
    """
    Train a single model.

    Args:
        model_name: Model alias or full HuggingFace model name
        dataset_name: Name of the dataset
        output_dir: Output directory for checkpoints
        device: Device to use
        config_dir: Configuration directory

    Returns:
        EmoBenchTrainer: Trained trainer object with model and tokenizer
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from src.data.loader import SentimentDataLoader
    from src.models.model_registry import ModelRegistry
    from src.utils.device import get_device
    import torch

    # Get actual device
    if device == "auto":
        device_obj = get_device()
        device_str = str(device_obj.type)
    else:
        device_obj = torch.device(device)
        device_str = device

    # Convert model alias to full model name if needed
    registry = ModelRegistry()
    try:
        # Try to get full model name from registry (assumes model_name is an alias)
        full_model_name = registry.get_model_name(model_name)
        logger.info(f"Using model: {full_model_name} (alias: {model_name})")
    except ValueError:
        # model_name might already be a full name, use it directly
        full_model_name = model_name
        logger.info(f"Using model: {full_model_name}")

    logger.info(f"Training {model_name} on {dataset_name} using device: {device_str}")

    # Load dataset (already tokenized by the loader)
    try:
        loader = SentimentDataLoader(dataset_name, full_model_name)
        train_dataset, val_dataset, test_dataset = loader.load_and_prepare()
        logger.info(
            f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples"
        )

        # Get the tokenizer from the loader
        tokenizer = loader.tokenizer

        # Ensure tokenizer has pad_token for batching (important for decoder-only models)
        logger.info(f"Tokenizer pad_token before: {tokenizer.pad_token}")
        logger.info(f"Tokenizer eos_token: {tokenizer.eos_token}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(
                f"Set pad_token to eos_token ({tokenizer.pad_token}) and pad_token_id ({tokenizer.pad_token_id}) for batching"
            )
        else:
            logger.info(f"Tokenizer already has pad_token: {tokenizer.pad_token}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Load model (with HF_TOKEN support for gated models)
    try:
        import os

        hf_token = os.environ.get("HF_TOKEN")

        model = AutoModelForSequenceClassification.from_pretrained(
            full_model_name,
            num_labels=2,  # Binary classification for sentiment
            token=hf_token,
            trust_remote_code=True,
        )

        model = model.to(device_obj)
        logger.info(f"Loaded model: {full_model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Datasets are already tokenized by the loader, just set format for PyTorch
    logger.info("Preparing datasets for PyTorch...")
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Check memory requirements and warn if potentially problematic
    model_size = registry.get_model_size(model_name)
    memory_req = registry.get_memory_requirements(model_name, device_str)
    logger.info(f"Model size: {model_size}, Memory requirement: {memory_req}")

    if device_str == "mps" and "B" in model_size:
        size_b = float(model_size.replace("B", ""))
        if size_b >= 3.0:
            logger.warning(
                f"⚠️  {model_name} ({model_size}) may require significant memory on MPS. "
                f"If you encounter OOM errors, consider:\n"
                f"  1. Closing other applications\n"
                f"  2. Setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0\n"
                f"  3. Skipping this model with --models flag"
            )

    # Get recommended batch size from model registry
    batch_size = registry.get_recommended_batch_size(model_name, device_obj)

    # For decoder-only models, ensure batch_size works with padding
    model_config = registry.get_model_config(model_name)
    if model_config.get("architecture") == "decoder-only":
        # Temporarily reduce batch size for decoder models if needed
        if batch_size > 1:
            batch_size = 1
            logger.info("Reduced batch size to 1 for decoder-only model to avoid padding issues")

    logger.info(f"Using batch size: {batch_size} (recommended for {device_str})")

    # Calculate gradient accumulation to maintain effective batch size
    target_effective_batch_size = 16
    gradient_accumulation_steps = max(1, target_effective_batch_size // batch_size)
    logger.info(
        f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {batch_size * gradient_accumulation_steps})"
    )

    # Enable gradient checkpointing for large models on MPS to save memory
    use_gradient_checkpointing = False
    if device_str == "mps":
        if "B" in model_size:  # Models in billions of parameters
            size_b = float(model_size.replace("B", ""))
            if size_b >= 1.0:  # 1B+ parameters
                use_gradient_checkpointing = True
                model.gradient_checkpointing_enable()
                logger.info("✓ Enabled gradient checkpointing for memory efficiency")

    # Data collator (datasets are already padded during tokenization)
    from transformers import default_data_collator

    data_collator = default_data_collator

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=use_gradient_checkpointing,
        # Device-specific settings
        fp16=(device_str == "cuda"),  # Only enable fp16 on CUDA
        bf16=False,  # Disable bf16 for now
        dataloader_pin_memory=(device_str == "cuda"),  # Pin memory only on CUDA
    )

    # Data collator with proper padding for variable-length sequences
    # (important for decoder-only models that may not have padding tokens)

    # Custom compute metrics function
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
        }

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Save the model and tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_model_path = output_path / "final"
    final_model_path.mkdir(exist_ok=True)

    logger.info(f"Saving model to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training results
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "training_results": trainer.state.log_history,
        "validation_results": eval_results,
        "device": device_str,
        "completed": True,
    }

    with open(output_path / "training_results.json", "w") as f:
        import json

        json.dump(results, f, indent=2, default=str)

    logger.info("Training completed successfully!")

    # Return trainer object with model and tokenizer
    trainer.model = model
    trainer.processing_class = tokenizer  # Updated from deprecated trainer.tokenizer

    return trainer


def train_all_models(
    dataset_name: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    device: str = "auto",
    config_dir: str = "config",
) -> Dict:
    """
    Train all models on specified dataset(s).

    Args:
        dataset_name: Name of the dataset (if None, train on all datasets)
        model_names: List of model names to train (if None, train all models)
        device: Device to use
        config_dir: Configuration directory

    Returns:
        Dict: Training results for all models
    """
    from src.models.model_registry import ModelRegistry

    # Get default models if not specified
    if model_names is None:
        registry = ModelRegistry()
        model_names = registry.list_models()
        logger.info(f"Training all {len(model_names)} models from registry")

    # Get default datasets if not specified
    if dataset_name is None:
        datasets = ["imdb", "sst2", "amazon", "yelp"]
        logger.info(f"No dataset specified, will train on all datasets: {datasets}")
    else:
        datasets = [dataset_name]
        logger.info(f"Training on dataset: {dataset_name}")

    all_results = {}

    for model in model_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training model: {model}")
        logger.info(f"{'=' * 60}\n")

        model_results = []

        for dataset in datasets:
            logger.info(f"\nTraining {model} on {dataset}...")

            # Create output directory for this model-dataset combination
            output_dir = Path(f"experiments/checkpoints/{model}_{dataset}")

            try:
                # Train the model using the real implementation
                trainer = train_model(
                    model_name=model,
                    dataset_name=dataset,
                    output_dir=str(output_dir),
                    device=device,
                    config_dir=config_dir,
                )

                # Collect results
                result = {
                    "model_name": model,
                    "dataset": dataset,
                    "status": "completed",
                    "checkpoint_dir": str(output_dir),
                }
                model_results.append(result)
                logger.info(f"✓ Completed training {model} on {dataset}")

            except Exception as e:
                logger.error(f"✗ Failed to train {model} on {dataset}: {e}")
                result = {
                    "model_name": model,
                    "dataset": dataset,
                    "status": "failed",
                    "error": str(e),
                }
                model_results.append(result)

        all_results[model] = model_results

    # Save aggregated results
    output_dir = Path("experiments/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    results_file = output_dir / "all_training_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"All training completed!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'=' * 60}\n")

    return all_results
