"""
Device detection and management utilities for SentiCompare.

Handles device selection (CUDA, MPS, CPU) and provides device-specific
synchronization and memory tracking utilities.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_device(force_device: Optional[str] = None) -> torch.device:
    """
    Detect and return the best available device.

    Args:
        force_device: Optional device to force ("cuda", "mps", or "cpu")

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device()
        >>> print(device)  # cuda:0, mps, or cpu
    """
    if force_device:
        device_str = force_device.lower()
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_str = "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            device_str = "cpu"
        device = torch.device(device_str)
    else:
        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Selected device: {device}")
    return device


def get_device_name(device: torch.device) -> str:
    """
    Get human-readable device name.

    Args:
        device: torch.device object

    Returns:
        str: Human-readable device name

    Examples:
        >>> device = torch.device("cuda")
        >>> print(get_device_name(device))
        'NVIDIA GeForce RTX 3080'
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"


def get_device_memory(device: torch.device) -> Dict[str, float]:
    """
    Get available memory for the device (in MB).

    Args:
        device: torch.device object

    Returns:
        Dict[str, float]: Memory information

    Examples:
        >>> device = torch.device("cuda")
        >>> memory = get_device_memory(device)
        >>> print(f"Total: {memory['total_mb']:.2f} MB")
    """
    memory_info = {}

    if device.type == "cuda":
        memory_info["total_mb"] = torch.cuda.get_device_properties(device).total_memory / 1024**2
        memory_info["allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        memory_info["reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        memory_info["free_mb"] = memory_info["total_mb"] - memory_info["allocated_mb"]
    elif device.type == "mps":
        # MPS uses unified memory, harder to get exact numbers
        # Approximate using PyTorch's memory tracking if available
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                memory_info["allocated_mb"] = torch.mps.current_allocated_memory() / 1024**2
            else:
                memory_info["allocated_mb"] = 0.0
            # Total unified memory is system RAM on Apple Silicon
            import psutil

            memory_info["total_mb"] = psutil.virtual_memory().total / 1024**2
            memory_info["free_mb"] = psutil.virtual_memory().available / 1024**2
        except Exception as e:
            logger.warning(f"Could not get MPS memory info: {e}")
            memory_info["total_mb"] = 0.0
            memory_info["allocated_mb"] = 0.0
            memory_info["free_mb"] = 0.0
    else:  # CPU
        import psutil

        vm = psutil.virtual_memory()
        memory_info["total_mb"] = vm.total / 1024**2
        memory_info["allocated_mb"] = vm.used / 1024**2
        memory_info["free_mb"] = vm.available / 1024**2

    return memory_info


def supports_quantization(device: torch.device) -> bool:
    """
    Check if the device supports 4-bit quantization (bitsandbytes).

    Args:
        device: torch.device object

    Returns:
        bool: True if quantization is supported

    Note:
        Currently only CUDA devices support bitsandbytes quantization.
        MPS and CPU do not support it.
    """
    return device.type == "cuda"


def sync_device(device: torch.device) -> None:
    """
    Synchronize device operations for accurate timing measurements.

    Args:
        device: torch.device object

    Examples:
        >>> device = get_device()
        >>> sync_device(device)
        >>> # Now timing measurements are accurate
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # MPS doesn't have a direct synchronization method
        # PyTorch 2.0+ might have torch.mps.synchronize()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        else:
            # No-op, rely on CPU synchronization
            pass
    # CPU doesn't need explicit synchronization


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Get optimal data type for the device.

    Args:
        device: torch.device object

    Returns:
        torch.dtype: Optimal dtype for the device

    Note:
        - CUDA: Uses bfloat16 for better stability with quantization
        - MPS: Uses float32 for stability (fp16 can be unstable)
        - CPU: Uses float32
    """
    if device.type == "cuda":
        return torch.bfloat16
    elif device.type == "mps":
        # MPS can be unstable with fp16, use fp32
        return torch.float32
    else:  # CPU
        return torch.float32


def get_device_info() -> Dict[str, any]:
    """
    Get comprehensive device information.

    Returns:
        Dict[str, any]: Complete device information

    Examples:
        >>> info = get_device_info()
        >>> print(info['device_type'])
        'cuda'
    """
    device = get_device()

    info = {
        "device": device,
        "device_type": device.type,
        "device_name": get_device_name(device),
        "supports_quantization": supports_quantization(device),
        "optimal_dtype": get_optimal_dtype(device),
        "memory": get_device_memory(device),
    }

    # Add CUDA-specific info
    if device.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["num_gpus"] = torch.cuda.device_count()
        info["current_gpu"] = torch.cuda.current_device()
        info["gpu_capability"] = torch.cuda.get_device_capability(device)

    # Add PyTorch info
    info["pytorch_version"] = torch.__version__

    return info


def print_device_info() -> None:
    """Print device information in a human-readable format."""
    info = get_device_info()

    print("\n" + "=" * 60)
    print("Device Information".center(60))
    print("=" * 60)
    print(f"Device Type:          {info['device_type'].upper()}")
    print(f"Device Name:          {info['device_name']}")
    print(f"PyTorch Version:      {info['pytorch_version']}")

    if info["device_type"] == "cuda":
        print(f"CUDA Version:         {info['cuda_version']}")
        print(f"Number of GPUs:       {info['num_gpus']}")
        print(f"Current GPU:          {info['current_gpu']}")
        print(f"GPU Capability:       {info['gpu_capability']}")

    print(f"\nMemory Information:")
    print(f"  Total Memory:       {info['memory']['total_mb']:.2f} MB")
    print(f"  Allocated Memory:   {info['memory']['allocated_mb']:.2f} MB")
    print(f"  Free Memory:        {info['memory']['free_mb']:.2f} MB")

    print(f"\nCapabilities:")
    print(f"  4-bit Quantization: {'✓ Yes' if info['supports_quantization'] else '✗ No'}")
    print(f"  Optimal dtype:      {info['optimal_dtype']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # When run as a script, print device information
    print_device_info()
