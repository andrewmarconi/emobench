"""
Training module for EmoBench.

This module provides training utilities for fine-tuning language models:
- Custom trainer with device-aware configuration
- LoRA/QLoRA fine-tuning support
- MLflow experiment tracking
- Multi-GPU and MPS support

Classes:
    EmoBenchTrainer: Main training class with LoRA support
"""
