"""
Evaluation and benchmarking module for SentiCompare.

This module provides comprehensive evaluation tools:
- Performance metrics (accuracy, F1, precision, recall)
- Speed benchmarking (latency, throughput, memory usage)
- Device-aware evaluation with CUDA/MPS/CPU support
- Statistical significance testing

Classes:
    BenchmarkRunner: Main evaluation orchestrator
    SpeedBenchmark: Performance and latency measurement
    MemoryProfiler: GPU/CPU memory tracking
"""
