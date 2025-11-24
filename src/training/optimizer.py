"""
Optimizer configuration for SentiCompare.

Provides optimizer and scheduler configurations optimized for LoRA fine-tuning.
"""

import logging
from typing import Dict, Optional

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR
from transformers import get_scheduler

logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> Optimizer:
    """
    Get AdamW optimizer configured for LoRA fine-tuning.

    Args:
        model: Model to optimize
        learning_rate: Learning rate (2e-4 recommended for LoRA)
        weight_decay: Weight decay for regularization
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        adam_epsilon: Adam epsilon for numerical stability

    Returns:
        Optimizer: Configured optimizer
    """
    # Separate parameters with and without weight decay
    decay_parameters = []
    nodecay_parameters = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to bias and LayerNorm parameters
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            nodecay_parameters.append(param)
        else:
            decay_parameters.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_parameters,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )

    logger.info(
        f"Optimizer: AdamW(lr={learning_rate}, weight_decay={weight_decay}, "
        f"betas=({adam_beta1}, {adam_beta2}))"
    )

    return optimizer


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a linear warmup and then linear decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch

    Returns:
        LambdaLR: Learning rate scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a linear warmup and then cosine decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: The index of the last epoch

    Returns:
        LambdaLR: Learning rate scheduler
    """
    import math

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(
    optimizer: Optimizer,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate.

    Args:
        optimizer: Optimizer to schedule
        last_epoch: The index of the last epoch

    Returns:
        LambdaLR: Learning rate scheduler
    """
    return LambdaLR(optimizer, lambda _: 1.0, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a warmup period followed by constant learning rate.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: The index of the last epoch

    Returns:
        LambdaLR: Learning rate scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "linear",
    num_warmup_steps: int = 500,
    num_training_steps: Optional[int] = None,
) -> Optional[LambdaLR]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("linear", "cosine", "constant")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps (required for linear/cosine)

    Returns:
        Optional[LambdaLR]: Learning rate scheduler
    """
    if scheduler_type == "linear":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for linear scheduler")
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    elif scheduler_type == "cosine":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for cosine scheduler")
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    elif scheduler_type == "constant":
        if num_warmup_steps > 0:
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        else:
            return get_constant_schedule(optimizer)

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using linear")
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps or 1000
        )


def get_optimizer_info(optimizer: Optimizer) -> Dict:
    """
    Get information about optimizer configuration.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dict: Optimizer information
    """
    info = {
        "type": type(optimizer).__name__,
        "num_param_groups": len(optimizer.param_groups),
    }

    # Get learning rates
    lrs = [group["lr"] for group in optimizer.param_groups]
    info["learning_rates"] = lrs
    info["learning_rate"] = lrs[0] if lrs else None

    # Get weight decay
    weight_decays = [group.get("weight_decay", 0.0) for group in optimizer.param_groups]
    info["weight_decays"] = weight_decays

    # Count parameters
    total_params = sum(
        sum(p.numel() for p in group["params"]) for group in optimizer.param_groups
    )
    info["total_parameters"] = total_params

    return info


def print_optimizer_info(optimizer: Optimizer) -> None:
    """Print optimizer configuration."""
    info = get_optimizer_info(optimizer)

    print("\n" + "=" * 60)
    print("Optimizer Configuration".center(60))
    print("=" * 60)
    print(f"Type:                {info['type']}")
    print(f"Learning Rate:       {info['learning_rate']:.2e}")
    print(f"Parameter Groups:    {info['num_param_groups']}")
    print(f"Total Parameters:    {info['total_parameters']:,}")

    if len(info['weight_decays']) > 1:
        print(f"\nParameter Group Details:")
        for i, (lr, wd) in enumerate(zip(info['learning_rates'], info['weight_decays'])):
            print(f"  Group {i}: lr={lr:.2e}, weight_decay={wd}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo: Create optimizer for a simple model
    import torch.nn as nn

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 2),
    )

    # Create optimizer
    optimizer = get_optimizer(model, learning_rate=2e-4)
    print_optimizer_info(optimizer)

    # Create scheduler
    scheduler = create_scheduler(
        optimizer, scheduler_type="linear", num_warmup_steps=100, num_training_steps=1000
    )

    print(f"Scheduler type: {type(scheduler).__name__}")
    print(f"Initial LR: {scheduler.get_last_lr()[0]:.2e}")

    # Simulate training
    print("\nLearning rate schedule (first 20 steps):")
    for step in range(20):
        lr = scheduler.get_last_lr()[0]
        print(f"  Step {step:3d}: lr={lr:.6f}")
        scheduler.step()
