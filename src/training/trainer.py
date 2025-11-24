"""
Training module for SentiCompare.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

logger = logging.getLogger(__name__)


class SentiCompareTrainer:
    """
    Custom trainer for SentiCompare with MLflow logging.
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
                trainer = SentiCompareTrainer(model, dataset, str(self.config_dir), self.device)
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
        SentiCompareTrainer: Trained trainer object with model and tokenizer
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

    # Load dataset
    try:
        loader = SentimentDataLoader(dataset_name, full_model_name)
        train_dataset, val_dataset, test_dataset = loader.load_and_prepare()
        logger.info(
            f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Load model and tokenizer (with HF_TOKEN support for gated models)
    try:
        import os
        hf_token = os.environ.get("HF_TOKEN")

        model = AutoModelForSequenceClassification.from_pretrained(
            full_model_name,
            num_labels=2,  # Binary classification for sentiment
            token=hf_token,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            full_model_name,
            token=hf_token,
            trust_remote_code=True
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device_obj)
        logger.info(f"Loaded model and tokenizer: {full_model_name}")
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Remove original text columns and rename labels
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_val = tokenized_val.remove_columns(["text"])

    # Set format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,  # Reduced for testing
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,  # More frequent logging for testing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],  # Disable wandb/tensorboard for now
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
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
    trainer.tokenizer = tokenizer

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
    # Use a dummy model name for the trainer since we'll override
    trainer = SentiCompareTrainer("dummy", "dummy", config_dir, device)

    # Convert single dataset to list
    datasets = [dataset_name] if dataset_name else None

    results = trainer.train_all(models=model_names, datasets=datasets)

    # Save results
    output_dir = Path("experiments/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_dir / "all_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
