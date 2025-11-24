"""
Training module for SentiCompare.

This module provides training utilities for fine-tuning language models:
- Custom trainer with device-aware configuration
- LoRA/QLoRA fine-tuning support
- MLflow experiment tracking
- Multi-GPU and MPS support

Classes:
    SentiCompareTrainer: Main training class with LoRA support
"""
