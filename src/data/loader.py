"""
Data loading utilities for EmoBench.

Handles loading sentiment analysis datasets from various sources
(HuggingFace Hub, KaggleHub) and prepares them for training.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_DATASETS_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/datasets.yaml")


class SentimentDataLoader:
    """
    Load and prepare sentiment analysis datasets.

    Supports multiple datasets from HuggingFace Hub and KaggleHub,
    with automatic tokenization and label mapping.

    Examples:
        >>> loader = SentimentDataLoader("imdb", "distilbert-base-uncased")
        >>> dataset = loader.load()
        >>> train_data = loader.prepare_splits(dataset)
    """

    # Dataset registry with source information
    SUPPORTED_DATASETS = {
        "imdb": {
            "name": "IMDB Movie Reviews",
            "source": "huggingface",
            "dataset_id": "stanfordnlp/imdb",
        },
        "sst2": {
            "name": "Stanford Sentiment Treebank v2",
            "source": "huggingface",
            "dataset_id": "stanfordnlp/sst2",
        },
        "amazon": {
            "name": "Amazon Polarity",
            "source": "huggingface",
            "dataset_id": "amazon_polarity",
        },
        "yelp": {
            "name": "Yelp Polarity",
            "source": "huggingface",
            "dataset_id": "yelp_polarity",
        },
    }

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        config_path: str = DEFAULT_DATASETS_CONFIG,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the data loader.

        Args:
            dataset_name: Name of the dataset (e.g., 'imdb', 'sst2', 'amazon', 'yelp')
            tokenizer_name: Name of the tokenizer model
            config_path: Path to dataset configuration file
            cache_dir: Directory for caching downloaded datasets
        """
        self.dataset_name = dataset_name.lower()
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir or "./data/raw"

        # Validate dataset name
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Choose from: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize tokenizer (with HF_TOKEN support for gated models)
        import os

        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, token=hf_token, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset metadata
        self.dataset_info = self.SUPPORTED_DATASETS[self.dataset_name]
        self.dataset_config = self.config["datasets"].get(self.dataset_name, {})

        # For testing, use smaller dataset sizes
        import os

        if os.environ.get("EMOBENCH_TEST_MODE"):
            self.dataset_config["max_train_samples"] = 100
            self.dataset_config["max_val_samples"] = 50
            self.dataset_config["max_test_samples"] = 50

        logger.info(f"Initialized loader for {self.dataset_info['name']}")

    def _load_config(self, config_path: str) -> Dict:
        """Load dataset configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {"datasets": {}, "splits": {}, "tokenization": {}}

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def load(self) -> DatasetDict:
        """
        Load dataset from source.

        Returns:
            DatasetDict: Loaded dataset with train/test splits

        Raises:
            ValueError: If dataset source is not supported
        """
        source = self.dataset_info["source"]
        dataset_id = self.dataset_info["dataset_id"]

        logger.info(f"Loading dataset from {source}: {dataset_id}")

        if source == "huggingface":
            return self._load_from_huggingface(dataset_id)
        elif source == "kagglehub":
            return self._load_from_kagglehub(dataset_id)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def _load_from_huggingface(self, dataset_id: str) -> DatasetDict:
        """Load dataset from HuggingFace Hub."""
        try:
            dataset = load_dataset(dataset_id, cache_dir=self.cache_dir)
            logger.info(f"Successfully loaded {dataset_id} from HuggingFace Hub")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            raise

    def _load_from_kagglehub(self, dataset_id: str) -> DatasetDict:
        """Load dataset from KaggleHub."""
        try:
            import kagglehub

            # Download dataset
            path = kagglehub.dataset_download(dataset_id)
            logger.info(f"Downloaded dataset from Kaggle to: {path}")

            # Load CSV file (assuming Disney reviews format)
            import pandas as pd

            # Find CSV files in the downloaded path
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {path}")

            # Load the first CSV file
            df = pd.read_csv(csv_files[0])
            logger.info(f"Loaded CSV with {len(df)} rows")

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_pandas(df)

            # Create train/test split (will be further split in prepare_splits)
            dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)

            return dataset_dict

        except Exception as e:
            logger.error(f"Failed to load dataset from Kaggle: {e}")
            raise

    def preprocess(self, examples: Dict) -> Dict:
        """
        Tokenize and preprocess examples.

        Args:
            examples: Batch of examples from dataset

        Returns:
            Dict: Tokenized examples
        """
        text_column = self.dataset_config.get("text_column", "text")
        max_length = self.dataset_config.get("max_length", 512)

        # Get tokenization settings from config
        tokenization_config = self.config.get("tokenization", {})
        padding = tokenization_config.get("padding", "max_length")
        truncation = tokenization_config.get("truncation", True)

        # Tokenize
        tokenized = self.tokenizer(
            examples[text_column],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_attention_mask=True,
        )

        # Map labels if needed
        label_column = self.dataset_config.get("label_column", "label")
        label_mapping = self.dataset_config.get("label_mapping")

        if label_mapping:
            # Apply label mapping (e.g., 1-5 star ratings to binary)
            tokenized["labels"] = [
                label_mapping.get(label, label) for label in examples[label_column]
            ]
        else:
            tokenized["labels"] = examples[label_column]

        return tokenized

    def prepare_splits(
        self,
        dataset: DatasetDict,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        stratify: bool = True,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create balanced train/val/test splits.

        Args:
            dataset: Dataset to split
            train_size: Number of training examples (default from config)
            val_size: Number of validation examples (default from config)
            test_size: Number of test examples (default from config)
            stratify: Whether to stratify by label

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train, val, test datasets
        """
        # Get split sizes from config if not provided
        split_config = self.config.get("splits", {})
        train_size = train_size or split_config.get("train_size", 10000)
        val_size = val_size or split_config.get("val_size", 2000)
        test_size = test_size or split_config.get("test_size", 5000)

        # Combine all available data
        if isinstance(dataset, DatasetDict):
            # Combine train and test if available
            all_data = []
            if "train" in dataset:
                all_data.append(dataset["train"])
            if "test" in dataset:
                all_data.append(dataset["test"])
            if "validation" in dataset:
                all_data.append(dataset["validation"])

            if len(all_data) > 1:
                from datasets import concatenate_datasets

                combined = concatenate_datasets(all_data)
            else:
                combined = all_data[0]
        else:
            combined = dataset

        # Shuffle
        seed = split_config.get("random_seed", 42)
        if split_config.get("shuffle", True):
            combined = combined.shuffle(seed=seed)

        # Create splits
        total_needed = train_size + val_size + test_size

        if len(combined) < total_needed:
            logger.warning(
                f"Dataset has only {len(combined)} examples, "
                f"but {total_needed} requested. Using all available data."
            )
            # Adjust sizes proportionally
            ratio = len(combined) / total_needed
            train_size = int(train_size * ratio)
            val_size = int(val_size * ratio)
            test_size = int(test_size * ratio)

        # Split dataset
        train_data = combined.select(range(train_size))
        val_data = combined.select(range(train_size, train_size + val_size))
        test_data = combined.select(range(train_size + val_size, train_size + val_size + test_size))

        logger.info(
            f"Created splits - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def load_and_prepare(
        self,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load dataset and prepare train/val/test splits in one step.

        Args:
            train_size: Number of training examples
            val_size: Number of validation examples
            test_size: Number of test examples

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train, val, test datasets (tokenized)
        """
        # Load dataset
        dataset = self.load()

        # Create splits
        train_data, val_data, test_data = self.prepare_splits(
            dataset, train_size, val_size, test_size
        )

        # Tokenize all splits
        logger.info("Tokenizing datasets...")
        train_data = train_data.map(
            self.preprocess,
            batched=True,
            desc="Tokenizing train data",
        )
        val_data = val_data.map(
            self.preprocess,
            batched=True,
            desc="Tokenizing validation data",
        )
        test_data = test_data.map(
            self.preprocess,
            batched=True,
            desc="Tokenizing test data",
        )

        # Remove original text columns (keep only tokenized data)
        text_column = self.dataset_config.get("text_column", "text")
        label_column = self.dataset_config.get("label_column", "label")

        # Get all column names
        cols_to_remove = [col for col in train_data.column_names if col not in ["input_ids", "attention_mask", "labels"]]

        if cols_to_remove:
            logger.info(f"Removing original columns: {cols_to_remove}")
            train_data = train_data.remove_columns(cols_to_remove)
            val_data = val_data.remove_columns(cols_to_remove)
            test_data = test_data.remove_columns(cols_to_remove)

        return train_data, val_data, test_data

    def get_dataset_info(self) -> Dict:
        """Get information about the current dataset."""
        return {
            "name": self.dataset_name,
            "full_name": self.dataset_info["name"],
            "source": self.dataset_info["source"],
            "num_labels": self.dataset_config.get("num_labels", 2),
            "max_length": self.dataset_config.get("max_length", 512),
            "tokenizer": self.tokenizer_name,
        }


def list_available_datasets() -> List[str]:
    """
    List all available datasets.

    Returns:
        List[str]: List of dataset names
    """
    return list(SentimentDataLoader.SUPPORTED_DATASETS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dict: Dataset information

    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in SentimentDataLoader.SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {list_available_datasets()}"
        )

    return SentimentDataLoader.SUPPORTED_DATASETS[dataset_name]
