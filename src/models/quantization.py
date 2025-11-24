"""
Quantization utilities for EmoBench.

Provides device-aware quantization configuration and utilities
for working with quantized models.
"""

import logging
from typing import Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def supports_quantization(device: Union[str, torch.device]) -> bool:
    """
    Check if device supports bitsandbytes quantization.

    Args:
        device: Device to check

    Returns:
        bool: True if quantization is supported

    Note:
        Currently only CUDA devices support bitsandbytes quantization.
        MPS and CPU do not support it.
    """
    if isinstance(device, str):
        device_type = device
    else:
        device_type = device.type

    return device_type == "cuda"


def supports_8bit_quantization(device: Union[str, torch.device]) -> bool:
    """
    Check if device supports 8-bit quantization.

    Args:
        device: Device to check

    Returns:
        bool: True if 8-bit quantization is supported
    """
    # Same as 4-bit for now (both require bitsandbytes)
    return supports_quantization(device)


def get_optimal_dtype(device: Union[str, torch.device]) -> torch.dtype:
    """
    Get optimal data type for the device.

    Args:
        device: Target device

    Returns:
        torch.dtype: Optimal dtype for the device

    Note:
        - CUDA: Uses bfloat16 for better stability with quantization
        - MPS: Uses float32 for stability (fp16 can be unstable on MPS)
        - CPU: Uses float32
    """
    if isinstance(device, str):
        device_type = device
    else:
        device_type = device.type

    if device_type == "cuda":
        return torch.bfloat16
    elif device_type == "mps":
        # MPS can be unstable with fp16, use fp32
        return torch.float32
    else:  # CPU or other
        return torch.float32


def get_quantization_config(
    device: Union[str, torch.device],
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
) -> Optional[Dict]:
    """
    Get quantization configuration for device.

    Args:
        device: Target device
        load_in_4bit: Use 4-bit quantization if supported
        load_in_8bit: Use 8-bit quantization if supported

    Returns:
        Optional[Dict]: Quantization config or None if not supported

    Raises:
        ValueError: If both 4-bit and 8-bit requested
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot use both 4-bit and 8-bit quantization")

    if not supports_quantization(device):
        logger.info(f"Quantization not supported on {device}, using full precision")
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "bitsandbytes not installed, skipping quantization. "
            "Install with: uv sync --extra cuda"
        )
        return None

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def verify_quantization(model: torch.nn.Module) -> Dict[str, any]:
    """
    Verify if model is quantized and get quantization details.

    Args:
        model: Model to check

    Returns:
        Dict: Quantization information
    """
    info = {
        "is_quantized": False,
        "quantization_method": None,
        "bits": None,
    }

    # Check if model has quantization config
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        quant_config = model.config.quantization_config
        info["is_quantized"] = True

        if hasattr(quant_config, "load_in_4bit") and quant_config.load_in_4bit:
            info["bits"] = 4
            info["quantization_method"] = "bitsandbytes (4-bit)"
        elif hasattr(quant_config, "load_in_8bit") and quant_config.load_in_8bit:
            info["bits"] = 8
            info["quantization_method"] = "bitsandbytes (8-bit)"

    return info


def estimate_model_size(
    num_parameters: int,
    dtype: torch.dtype = torch.float32,
    quantized: bool = False,
    quantization_bits: int = 4,
) -> Dict[str, float]:
    """
    Estimate model memory footprint.

    Args:
        num_parameters: Number of model parameters
        dtype: Data type for non-quantized model
        quantized: Whether model is quantized
        quantization_bits: Number of bits for quantization (4 or 8)

    Returns:
        Dict[str, float]: Memory estimates in MB and GB
    """
    if quantized:
        # Quantized models use fewer bits per parameter
        bytes_per_param = quantization_bits / 8
    else:
        # Standard precision
        if dtype == torch.float32:
            bytes_per_param = 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            bytes_per_param = 2
        else:
            bytes_per_param = 4  # Default

    total_bytes = num_parameters * bytes_per_param
    mb = total_bytes / (1024**2)
    gb = total_bytes / (1024**3)

    return {
        "size_mb": mb,
        "size_gb": gb,
        "bytes_per_param": bytes_per_param,
    }


def compare_quantization_methods(num_parameters: int) -> None:
    """
    Compare memory usage across different quantization methods.

    Args:
        num_parameters: Number of model parameters
    """
    methods = [
        ("Float32 (Full Precision)", torch.float32, False, None),
        ("Float16", torch.float16, False, None),
        ("BFloat16", torch.bfloat16, False, None),
        ("8-bit Quantization", torch.float32, True, 8),
        ("4-bit Quantization (QLoRA)", torch.float32, True, 4),
    ]

    print("\n" + "=" * 70)
    print("Memory Comparison for Quantization Methods".center(70))
    print("=" * 70)
    print(f"Model Parameters: {num_parameters:,}")
    print("-" * 70)
    print(f"{'Method':<30} {'Memory (MB)':<15} {'Memory (GB)':<15}")
    print("-" * 70)

    for name, dtype, quantized, bits in methods:
        estimate = estimate_model_size(num_parameters, dtype, quantized, bits)
        print(
            f"{name:<30} {estimate['size_mb']:>12,.1f}   {estimate['size_gb']:>12,.2f}"
        )

    print("=" * 70 + "\n")


def get_device_quantization_strategy(device: Union[str, torch.device]) -> Dict[str, any]:
    """
    Get recommended quantization strategy for device.

    Args:
        device: Target device

    Returns:
        Dict: Recommended quantization strategy
    """
    if isinstance(device, str):
        device_type = device
    else:
        device_type = device.type

    strategy = {
        "device": device_type,
        "supports_quantization": supports_quantization(device),
        "recommended_dtype": get_optimal_dtype(device),
        "recommended_precision": None,
        "notes": [],
    }

    if device_type == "cuda":
        strategy["recommended_precision"] = "4-bit"
        strategy["notes"].append("Use 4-bit quantization (QLoRA) for memory efficiency")
        strategy["notes"].append("Enables training larger models on limited VRAM")
    elif device_type == "mps":
        strategy["recommended_precision"] = "float32"
        strategy["notes"].append("Quantization not supported on MPS")
        strategy["notes"].append("Use full precision with LoRA")
        strategy["notes"].append("Reduce batch size to manage memory")
    else:  # CPU
        strategy["recommended_precision"] = "float32"
        strategy["notes"].append("Quantization not recommended on CPU")
        strategy["notes"].append("Training will be slower than GPU")

    return strategy


def print_quantization_strategy(device: Union[str, torch.device]) -> None:
    """Print quantization strategy for device."""
    strategy = get_device_quantization_strategy(device)

    print("\n" + "=" * 60)
    print(f"Quantization Strategy for {strategy['device'].upper()}".center(60))
    print("=" * 60)
    print(f"Supports Quantization:  {'✓ Yes' if strategy['supports_quantization'] else '✗ No'}")
    print(f"Recommended dtype:      {strategy['recommended_dtype']}")
    print(f"Recommended Precision:  {strategy['recommended_precision']}")
    print(f"\nNotes:")
    for note in strategy["notes"]:
        print(f"  • {note}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: Show quantization strategies for different devices
    from src.utils.device import get_device, print_device_info

    # Print device info
    print_device_info()

    # Show quantization strategy for current device
    device = get_device()
    print_quantization_strategy(device)

    # Compare quantization methods for a typical model
    # Example: DistilBERT has ~66M parameters
    print("\nExample: DistilBERT (66M parameters)")
    compare_quantization_methods(66_000_000)

    print("\nExample: TinyLlama (1.1B parameters)")
    compare_quantization_methods(1_100_000_000)
