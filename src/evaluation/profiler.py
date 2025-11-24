"""
Memory profiling for SentiCompare models.

Tracks memory usage during training and inference across different devices
(CUDA, MPS, CPU).
"""

import logging
import os
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Profile memory usage for models across different devices.

    Supports:
    - CUDA GPU memory (allocated, reserved, peak)
    - MPS (Apple Silicon) memory
    - System CPU memory
    - Model parameter memory

    Examples:
        >>> profiler = MemoryProfiler(device)
        >>> profiler.start()
        >>> # ... run inference ...
        >>> stats = profiler.get_stats()
        >>> print(f"Peak memory: {stats['peak_memory_gb']:.2f} GB")
    """

    def __init__(self, device: torch.device):
        """
        Initialize memory profiler.

        Args:
            device: Device to profile
        """
        self.device = device
        self.device_type = device.type
        self.initial_memory = None
        self.peak_memory = 0

    def start(self) -> None:
        """Start memory profiling."""
        self._reset_peak()
        self.initial_memory = self._get_current_memory()
        logger.info(f"Started memory profiling on {self.device_type}")

    def get_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.

        Returns:
            Dict[str, float]: Memory statistics in bytes and GB
        """
        current_memory = self._get_current_memory()

        stats = {
            "device": self.device_type,
            "current_memory_bytes": current_memory,
            "current_memory_gb": current_memory / (1024**3),
        }

        if self.initial_memory is not None:
            memory_used = current_memory - self.initial_memory
            stats["memory_used_bytes"] = memory_used
            stats["memory_used_gb"] = memory_used / (1024**3)

        # Device-specific stats
        if self.device_type == "cuda":
            stats.update(self._get_cuda_stats())
        elif self.device_type == "mps":
            stats.update(self._get_mps_stats())
        elif self.device_type == "cpu":
            stats.update(self._get_cpu_stats())

        return stats

    def get_peak_memory(self) -> float:
        """
        Get peak memory usage in GB.

        Returns:
            float: Peak memory in GB
        """
        if self.device_type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024**3)
        elif self.device_type == "mps":
            # MPS doesn't have built-in peak tracking
            return self.peak_memory / (1024**3)
        else:
            return 0.0

    def reset(self) -> None:
        """Reset memory profiling."""
        self._reset_peak()
        self.initial_memory = None

    def _get_current_memory(self) -> float:
        """Get current memory usage in bytes."""
        if self.device_type == "cuda":
            return float(torch.cuda.memory_allocated(self.device))
        elif self.device_type == "mps":
            try:
                if hasattr(torch.mps, "current_allocated_memory"):
                    memory = torch.mps.current_allocated_memory()
                    self.peak_memory = max(self.peak_memory, memory)
                    return float(memory)
            except Exception:
                pass
            return 0.0
        else:
            # CPU - use process memory
            try:
                import psutil

                process = psutil.Process(os.getpid())
                return float(process.memory_info().rss)
            except ImportError:
                logger.warning("psutil not available, cannot track CPU memory")
                return 0.0

    def _reset_peak(self) -> None:
        """Reset peak memory tracking."""
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0

    def _get_cuda_stats(self) -> Dict[str, float]:
        """Get CUDA-specific memory stats."""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)

        return {
            "allocated_bytes": float(allocated),
            "allocated_gb": allocated / (1024**3),
            "reserved_bytes": float(reserved),
            "reserved_gb": reserved / (1024**3),
            "peak_allocated_bytes": float(max_allocated),
            "peak_allocated_gb": max_allocated / (1024**3),
            "peak_reserved_bytes": float(max_reserved),
            "peak_reserved_gb": max_reserved / (1024**3),
        }

    def _get_mps_stats(self) -> Dict[str, float]:
        """Get MPS-specific memory stats."""
        stats = {}

        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory()
                stats["allocated_bytes"] = float(allocated)
                stats["allocated_gb"] = allocated / (1024**3)
                stats["peak_allocated_gb"] = self.peak_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get MPS memory stats: {e}")

        return stats

    def _get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU memory stats."""
        stats = {}

        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            stats["rss_bytes"] = float(mem_info.rss)
            stats["rss_gb"] = mem_info.rss / (1024**3)
            stats["vms_bytes"] = float(mem_info.vms)
            stats["vms_gb"] = mem_info.vms / (1024**3)

            # System memory
            virtual_mem = psutil.virtual_memory()
            stats["system_total_gb"] = virtual_mem.total / (1024**3)
            stats["system_available_gb"] = virtual_mem.available / (1024**3)
            stats["system_percent"] = virtual_mem.percent

        except ImportError:
            logger.warning("psutil not available, cannot get detailed CPU stats")

        return stats


def get_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model memory footprint.

    Args:
        model: PyTorch model

    Returns:
        Dict[str, float]: Memory information
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    # Estimate memory (assuming float32)
    bytes_per_param = 4  # float32
    total_memory = total_params * bytes_per_param

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_memory_bytes": total_memory,
        "total_memory_mb": total_memory / (1024**2),
        "total_memory_gb": total_memory / (1024**3),
        "trainable_memory_mb": (trainable_params * bytes_per_param) / (1024**2),
        "frozen_memory_mb": (frozen_params * bytes_per_param) / (1024**2),
    }


def print_memory_stats(stats: Dict[str, float], title: str = "Memory Statistics") -> None:
    """
    Print memory statistics in a formatted way.

    Args:
        stats: Memory statistics dictionary
        title: Title to display
    """
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    print(f"Device: {stats.get('device', 'Unknown')}")
    print("-" * 60)

    # Current memory
    if "current_memory_gb" in stats:
        print(f"Current Memory: {stats['current_memory_gb']:.3f} GB")

    if "memory_used_gb" in stats:
        print(f"Memory Used:    {stats['memory_used_gb']:.3f} GB")

    # Device-specific stats
    device = stats.get("device", "")

    if device == "cuda":
        print(f"\nCUDA Memory:")
        print(f"  Allocated:      {stats.get('allocated_gb', 0):.3f} GB")
        print(f"  Reserved:       {stats.get('reserved_gb', 0):.3f} GB")
        print(f"  Peak Allocated: {stats.get('peak_allocated_gb', 0):.3f} GB")
        print(f"  Peak Reserved:  {stats.get('peak_reserved_gb', 0):.3f} GB")

    elif device == "mps":
        print(f"\nMPS Memory:")
        print(f"  Allocated:      {stats.get('allocated_gb', 0):.3f} GB")
        if "peak_allocated_gb" in stats:
            print(f"  Peak Allocated: {stats.get('peak_allocated_gb', 0):.3f} GB")

    elif device == "cpu":
        print(f"\nCPU Memory:")
        if "rss_gb" in stats:
            print(f"  RSS:            {stats['rss_gb']:.3f} GB")
        if "vms_gb" in stats:
            print(f"  VMS:            {stats['vms_gb']:.3f} GB")
        if "system_total_gb" in stats:
            print(f"\nSystem Memory:")
            print(f"  Total:          {stats['system_total_gb']:.2f} GB")
            print(f"  Available:      {stats['system_available_gb']:.2f} GB")
            print(f"  Used:           {stats.get('system_percent', 0):.1f}%")

    print("=" * 60 + "\n")


def print_model_memory(model: torch.nn.Module, title: str = "Model Memory") -> None:
    """
    Print model memory information.

    Args:
        model: PyTorch model
        title: Title to display
    """
    stats = get_model_memory(model)

    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    print(f"Total Parameters:     {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")
    print(f"Frozen Parameters:    {stats['frozen_params']:,}")
    print("-" * 60)
    print(f"Total Memory:         {stats['total_memory_mb']:.2f} MB")
    print(f"Trainable Memory:     {stats['trainable_memory_mb']:.2f} MB")
    print(f"Frozen Memory:        {stats['frozen_memory_mb']:.2f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: Memory profiling
    from src.utils.device import get_device, print_device_info

    print("Memory Profiler Module")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device.type}")

    # Create profiler
    profiler = MemoryProfiler(device)

    # Start profiling
    profiler.start()

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 2),
    )
    model = model.to(device)

    # Print model memory
    print_model_memory(model, "Demo Model Memory")

    # Get memory stats
    stats = profiler.get_stats()
    print_memory_stats(stats, "Memory After Model Creation")

    # Run some operations
    x = torch.randn(32, 1000).to(device)
    for _ in range(10):
        y = model(x)

    # Final stats
    final_stats = profiler.get_stats()
    print_memory_stats(final_stats, "Memory After Inference")

    print(f"Peak Memory: {profiler.get_peak_memory():.3f} GB")
