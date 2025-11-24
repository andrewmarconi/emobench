"""
Model registry for EmoBench.

Manages model configurations and metadata for all supported models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing model configurations.

    Loads model metadata from config/models.yaml and provides
    utilities to access model information, recommended settings, etc.

    Examples:
        >>> registry = ModelRegistry()
        >>> models = registry.list_models()
        >>> config = registry.get_model_config("distilbert-base-uncased")
    """

    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Initialize the model registry.

        Args:
            config_path: Path to models configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models = {model["alias"]: model for model in self.config.get("models", [])}

        logger.info(f"Loaded {len(self.models)} models from registry")

    def _load_config(self) -> Dict:
        """Load model configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {"models": [], "lora_defaults": {}, "quantization": {}}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def list_models(self) -> List[str]:
        """
        List all available model aliases.

        Returns:
            List[str]: List of model aliases
        """
        return list(self.models.keys())

    def get_model_config(self, model_alias: str) -> Dict:
        """
        Get configuration for a specific model.

        Args:
            model_alias: Model alias (e.g., "DistilBERT-base")

        Returns:
            Dict: Model configuration

        Raises:
            ValueError: If model not found
        """
        if model_alias not in self.models:
            raise ValueError(
                f"Model '{model_alias}' not found. "
                f"Available models: {self.list_models()}"
            )

        return self.models[model_alias]

    def get_model_name(self, model_alias: str) -> str:
        """
        Get the full HuggingFace model name for a model alias.

        Args:
            model_alias: Model alias

        Returns:
            str: Full model name (e.g., "distilbert-base-uncased")
        """
        config = self.get_model_config(model_alias)
        return config["name"]

    def get_lora_config(self, model_alias: str) -> Dict:
        """
        Get LoRA configuration for a model.

        Args:
            model_alias: Model alias

        Returns:
            Dict: LoRA configuration
        """
        model_config = self.get_model_config(model_alias)
        lora_config = model_config.get("lora", {})

        # Merge with defaults
        defaults = self.config.get("lora_defaults", {})
        return {**defaults, **lora_config}

    def get_recommended_batch_size(
        self, model_alias: str, device: Optional[torch.device] = None
    ) -> int:
        """
        Get recommended batch size for a model on a specific device.

        Args:
            model_alias: Model alias
            device: Target device (auto-detect if None)

        Returns:
            int: Recommended batch size
        """
        if device is None:
            from src.utils.device import get_device

            device = get_device()

        model_config = self.get_model_config(model_alias)
        batch_sizes = model_config.get("recommended_batch_size", {})

        device_type = device.type if isinstance(device, torch.device) else str(device)
        return batch_sizes.get(device_type, 4)  # Default to 4

    def get_memory_requirements(self, model_alias: str, device_type: str) -> str:
        """
        Get memory requirements for a model on a specific device.

        Args:
            model_alias: Model alias
            device_type: Device type ("cuda", "mps", or "cpu")

        Returns:
            str: Memory requirement (e.g., "4GB", "16GB")
        """
        model_config = self.get_model_config(model_alias)
        memory_reqs = model_config.get("memory_requirements", {})

        if device_type == "cuda":
            return memory_reqs.get("cuda_4bit", "Unknown")
        elif device_type == "mps":
            return memory_reqs.get("mps_fp32", "Unknown")
        else:
            return memory_reqs.get("cpu", "Unknown")

    def get_model_size(self, model_alias: str) -> str:
        """
        Get model size in parameters.

        Args:
            model_alias: Model alias

        Returns:
            str: Model size (e.g., "1.1B", "125M")
        """
        model_config = self.get_model_config(model_alias)
        return model_config.get("size_params", "Unknown")

    def get_architecture_type(self, model_alias: str) -> str:
        """
        Get model architecture type.

        Args:
            model_alias: Model alias

        Returns:
            str: Architecture type ("encoder-only", "decoder-only", etc.)
        """
        model_config = self.get_model_config(model_alias)
        return model_config.get("architecture", "unknown")

    def filter_models(
        self,
        max_size: Optional[str] = None,
        architecture: Optional[str] = None,
        min_memory: Optional[int] = None,
    ) -> List[str]:
        """
        Filter models by criteria.

        Args:
            max_size: Maximum model size (e.g., "2B")
            architecture: Architecture type filter
            min_memory: Minimum memory requirement in GB

        Returns:
            List[str]: Filtered model aliases
        """
        filtered = []

        for alias in self.list_models():
            config = self.get_model_config(alias)

            # Filter by architecture
            if architecture and config.get("architecture") != architecture:
                continue

            # Filter by size (simple string comparison for now)
            if max_size:
                size = config.get("size_params", "")
                # This is a simple check, could be made more sophisticated
                if size > max_size:
                    continue

            filtered.append(alias)

        return filtered

    def get_models_by_size(self) -> Dict[str, List[str]]:
        """
        Group models by size category.

        Returns:
            Dict[str, List[str]]: Models grouped by size
        """
        categories = {
            "tiny": [],  # < 200M
            "small": [],  # 200M - 2B
            "medium": [],  # 2B - 4B
            "large": [],  # > 4B
        }

        for alias in self.list_models():
            size_str = self.get_model_size(alias)

            # Parse size
            if "M" in size_str:
                size_mb = float(size_str.replace("M", ""))
                if size_mb < 200:
                    categories["tiny"].append(alias)
                else:
                    categories["small"].append(alias)
            elif "B" in size_str:
                size_b = float(size_str.replace("B", ""))
                if size_b < 2:
                    categories["small"].append(alias)
                elif size_b < 4:
                    categories["medium"].append(alias)
                else:
                    categories["large"].append(alias)

        return categories

    def get_quantization_config(self) -> Dict:
        """
        Get global quantization configuration.

        Returns:
            Dict: Quantization settings
        """
        return self.config.get("quantization", {})

    def supports_quantization(self, model_alias: str, device_type: str) -> bool:
        """
        Check if quantization is supported for model on device.

        Args:
            model_alias: Model alias
            device_type: Device type

        Returns:
            bool: True if quantization supported
        """
        # Only CUDA supports quantization currently
        return device_type == "cuda"

    def get_model_info(self, model_alias: str, device: Optional[torch.device] = None) -> Dict:
        """
        Get comprehensive information about a model.

        Args:
            model_alias: Model alias
            device: Target device

        Returns:
            Dict: Complete model information
        """
        if device is None:
            from src.utils.device import get_device

            device = get_device()

        config = self.get_model_config(model_alias)
        device_type = device.type if isinstance(device, torch.device) else str(device)

        return {
            "alias": model_alias,
            "name": config["name"],
            "size": config.get("size_params"),
            "architecture": config.get("architecture"),
            "lora_config": self.get_lora_config(model_alias),
            "recommended_batch_size": self.get_recommended_batch_size(model_alias, device),
            "memory_requirement": self.get_memory_requirements(model_alias, device_type),
            "supports_quantization": self.supports_quantization(model_alias, device_type),
        }

    def print_model_info(self, model_alias: str, device: Optional[torch.device] = None) -> None:
        """Print formatted model information."""
        info = self.get_model_info(model_alias, device)

        print("\n" + "=" * 60)
        print(f"Model: {info['alias']}".center(60))
        print("=" * 60)
        print(f"Full Name:            {info['name']}")
        print(f"Size:                 {info['size']}")
        print(f"Architecture:         {info['architecture']}")
        print(f"\nLoRA Configuration:")
        print(f"  Rank:               {info['lora_config'].get('rank')}")
        print(f"  Alpha:              {info['lora_config'].get('alpha')}")
        print(f"  Dropout:            {info['lora_config'].get('dropout')}")
        print(f"  Target Modules:     {info['lora_config'].get('target_modules')}")
        print(f"\nRecommended Settings:")
        print(f"  Batch Size:         {info['recommended_batch_size']}")
        print(f"  Memory Required:    {info['memory_requirement']}")
        print(f"  Quantization:       {'✓ Supported' if info['supports_quantization'] else '✗ Not Supported'}")
        print("=" * 60 + "\n")


def list_all_models() -> List[str]:
    """
    Convenience function to list all available models.

    Returns:
        List[str]: List of model aliases
    """
    registry = ModelRegistry()
    return registry.list_models()


def get_model_info(model_alias: str) -> Dict:
    """
    Convenience function to get model information.

    Args:
        model_alias: Model alias

    Returns:
        Dict: Model information
    """
    registry = ModelRegistry()
    return registry.get_model_info(model_alias)


if __name__ == "__main__":
    # Demo: Print information for all models
    registry = ModelRegistry()

    print("\nAvailable Models:")
    print("=" * 60)
    for alias in registry.list_models():
        print(f"  - {alias}")
    print()

    # Group by size
    print("\nModels by Size:")
    print("=" * 60)
    grouped = registry.get_models_by_size()
    for category, models in grouped.items():
        if models:
            print(f"{category.upper()}: {', '.join(models)}")
    print()

    # Print detailed info for a sample model
    if registry.list_models():
        sample_model = registry.list_models()[0]
        registry.print_model_info(sample_model)
