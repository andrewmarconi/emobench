"""
LoRA/QLoRA configuration for EmoBench.

Handles device-aware model loading with LoRA fine-tuning,
including 4-bit quantization on CUDA and full precision on MPS.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class LoRAConfigManager:
    """
    Manage LoRA and quantization configurations.

    Provides device-aware model loading with appropriate quantization
    strategy based on hardware capabilities.

    Examples:
        >>> from src.utils.device import get_device
        >>> device = get_device()
        >>> model = LoRAConfigManager.prepare_model(
        ...     "distilbert-base-uncased",
        ...     num_labels=2,
        ...     device=device
        ... )
    """

    @staticmethod
    def get_4bit_config() -> "BitsAndBytesConfig":
        """
        Get 4-bit quantization config for QLoRA (CUDA only).

        Returns:
            BitsAndBytesConfig: Quantization configuration

        Raises:
            ImportError: If bitsandbytes not installed
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. "
                "Install with: uv sync --extra cuda"
            )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    @staticmethod
    def get_lora_config(
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        task_type: Union[str, TaskType] = TaskType.SEQ_CLS,
    ) -> LoraConfig:
        """
        Get LoRA configuration for fine-tuning.

        Args:
            rank: LoRA rank (r)
            alpha: LoRA alpha parameter
            dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to
            task_type: Task type (SEQ_CLS for classification)

        Returns:
            LoraConfig: LoRA configuration
        """
        if target_modules is None:
            # Default to common attention modules
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=task_type if isinstance(task_type, TaskType) else TaskType[task_type],
        )

    @staticmethod
    def get_model_specific_lora_config(model_name: str) -> LoraConfig:
        """
        Get model-specific LoRA configuration from registry.

        Args:
            model_name: Model name or alias

        Returns:
            LoraConfig: Model-specific LoRA configuration
        """
        from src.models.model_registry import ModelRegistry

        registry = ModelRegistry()

        # Try to find by alias first, then by name
        try:
            lora_cfg = registry.get_lora_config(model_name)
        except ValueError:
            # Try to find by full model name
            for alias in registry.list_models():
                if registry.get_model_name(alias) == model_name:
                    lora_cfg = registry.get_lora_config(alias)
                    break
            else:
                # Use defaults if not found
                logger.warning(
                    f"Model {model_name} not in registry, using default LoRA config"
                )
                lora_cfg = {}

        return LoRAConfigManager.get_lora_config(
            rank=lora_cfg.get("rank", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
        )

    @staticmethod
    def prepare_model(
        model_name: str,
        num_labels: int = 2,
        device: Optional[Union[str, torch.device]] = None,
        use_quantization: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        **model_kwargs,
    ) -> PeftModel:
        """
        Load model with device-aware quantization and LoRA.

        This is the main entry point for loading models. It automatically
        handles quantization based on device capabilities.

        Args:
            model_name: HuggingFace model name or alias
            num_labels: Number of classification labels
            device: Target device (auto-detect if None)
            use_quantization: Force quantization on/off (auto if None)
            lora_config: Custom LoRA config (auto-generate if None)
            **model_kwargs: Additional arguments for model loading

        Returns:
            PeftModel: Model with LoRA adapters

        Examples:
            >>> # Auto-detect device and quantization
            >>> model = LoRAConfigManager.prepare_model("distilbert-base-uncased")
            >>>
            >>> # Force specific device
            >>> model = LoRAConfigManager.prepare_model(
            ...     "distilbert-base-uncased",
            ...     device="mps"
            ... )
        """
        # Auto-detect device if not provided
        if device is None:
            from src.utils.device import get_device

            device = get_device()
        elif isinstance(device, str):
            device = torch.device(device)

        # Determine quantization strategy
        supports_quant = device.type == "cuda"
        if use_quantization is None:
            use_quantization = supports_quant
        elif use_quantization and not supports_quant:
            logger.warning(
                f"Quantization requested but not supported on {device.type}. "
                "Falling back to full precision."
            )
            use_quantization = False

        # Get LoRA config
        if lora_config is None:
            lora_config = LoRAConfigManager.get_model_specific_lora_config(model_name)

        logger.info(f"Loading {model_name} on {device.type}")
        logger.info(f"Quantization: {'enabled' if use_quantization else 'disabled'}")
        logger.info(f"LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")

        # Load model with appropriate configuration
        if use_quantization:
            # CUDA with 4-bit quantization
            model = LoRAConfigManager._load_model_with_quantization(
                model_name, num_labels, **model_kwargs
            )
        elif device.type == "mps":
            # MPS with full precision
            model = LoRAConfigManager._load_model_mps(
                model_name, num_labels, **model_kwargs
            )
        else:
            # CPU or fallback
            model = LoRAConfigManager._load_model_cpu(
                model_name, num_labels, **model_kwargs
            )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params, all_params = LoRAConfigManager.count_trainable_parameters(model)
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )

        return model

    @staticmethod
    def _load_model_with_quantization(
        model_name: str, num_labels: int, **model_kwargs
    ) -> AutoModelForSequenceClassification:
        """Load model with 4-bit quantization (CUDA only)."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes required for quantization. "
                "Install with: uv sync --extra cuda"
            )

        quant_config = LoRAConfigManager.get_4bit_config()

        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=quant_config,
            device_map="auto",
            **model_kwargs,
        )

    @staticmethod
    def _load_model_mps(
        model_name: str, num_labels: int, **model_kwargs
    ) -> AutoModelForSequenceClassification:
        """Load model for MPS (Apple Silicon) with full precision."""
        from src.utils.device import get_optimal_dtype

        device = torch.device("mps")
        dtype = get_optimal_dtype(device)

        logger.info(f"Loading on MPS with dtype: {dtype}")

        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            device_map="auto",
            **model_kwargs,
        )

    @staticmethod
    def _load_model_cpu(
        model_name: str, num_labels: int, **model_kwargs
    ) -> AutoModelForSequenceClassification:
        """Load model for CPU."""
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **model_kwargs
        )

    @staticmethod
    def count_trainable_parameters(model: PeftModel) -> tuple[int, int]:
        """
        Count trainable parameters in a PEFT model.

        Args:
            model: PEFT model

        Returns:
            tuple[int, int]: (trainable_params, all_params)
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        return trainable_params, all_params

    @staticmethod
    def print_trainable_parameters(model: PeftModel) -> None:
        """Print trainable parameters information."""
        trainable_params, all_params = LoRAConfigManager.count_trainable_parameters(model)
        percentage = 100 * trainable_params / all_params

        print("\n" + "=" * 60)
        print("Model Parameters".center(60))
        print("=" * 60)
        print(f"Trainable params:     {trainable_params:,}")
        print(f"All params:           {all_params:,}")
        print(f"Trainable %:          {percentage:.2f}%")
        print("=" * 60 + "\n")

    @staticmethod
    def save_lora_model(model: PeftModel, save_path: str) -> None:
        """
        Save only LoRA adapter weights.

        Args:
            model: PEFT model to save
            save_path: Directory to save adapter weights
        """
        model.save_pretrained(save_path)
        logger.info(f"Saved LoRA adapters to {save_path}")

    @staticmethod
    def load_lora_model(
        base_model_name: str,
        adapter_path: str,
        num_labels: int = 2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> PeftModel:
        """
        Load base model and attach LoRA adapters.

        Args:
            base_model_name: Base model name
            adapter_path: Path to LoRA adapter weights
            num_labels: Number of labels
            device: Target device

        Returns:
            PeftModel: Model with loaded adapters
        """
        from peft import PeftModel

        # Load base model
        if device is None:
            from src.utils.device import get_device

            device = get_device()

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=num_labels
        )
        base_model = base_model.to(device)

        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)

        logger.info(f"Loaded LoRA adapters from {adapter_path}")
        return model


def prepare_model(
    model_name: str,
    num_labels: int = 2,
    device: Optional[Union[str, torch.device]] = None,
) -> PeftModel:
    """
    Convenience function to prepare a model with LoRA.

    Args:
        model_name: Model name or alias
        num_labels: Number of classification labels
        device: Target device

    Returns:
        PeftModel: Configured model
    """
    return LoRAConfigManager.prepare_model(model_name, num_labels, device)


if __name__ == "__main__":
    # Demo: Load a small model with LoRA
    from src.utils.device import print_device_info

    print_device_info()

    print("Loading DistilBERT with LoRA...")
    model = prepare_model("distilbert-base-uncased", num_labels=2)

    LoRAConfigManager.print_trainable_parameters(model)
