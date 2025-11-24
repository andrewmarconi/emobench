"""
Unit tests for model configuration modules.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.models.model_registry import ModelRegistry, list_all_models, get_model_info
from src.models.quantization import (
    supports_quantization,
    get_optimal_dtype,
    estimate_model_size,
    get_device_quantization_strategy,
)


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = ModelRegistry()
        assert registry.config is not None
        assert isinstance(registry.models, dict)
        assert len(registry.models) > 0

    def test_list_models(self):
        """Test listing all models."""
        registry = ModelRegistry()
        models = registry.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "DistilBERT-base" in models or "RoBERTa-base" in models

    def test_get_model_config(self):
        """Test getting model configuration."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            config = registry.get_model_config(models[0])
            assert "name" in config
            assert "size_params" in config
            assert "architecture" in config

    def test_get_model_config_invalid(self):
        """Test getting config for invalid model."""
        registry = ModelRegistry()

        with pytest.raises(ValueError):
            registry.get_model_config("invalid_model_alias")

    def test_get_model_name(self):
        """Test getting HuggingFace model name."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            name = registry.get_model_name(models[0])
            assert isinstance(name, str)
            assert len(name) > 0

    def test_get_lora_config(self):
        """Test getting LoRA configuration."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            lora_config = registry.get_lora_config(models[0])
            assert "rank" in lora_config
            assert "alpha" in lora_config
            assert "target_modules" in lora_config

    def test_get_recommended_batch_size(self):
        """Test getting recommended batch size."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            # Test with CPU device
            batch_size = registry.get_recommended_batch_size(
                models[0], torch.device("cpu")
            )
            assert isinstance(batch_size, int)
            assert batch_size > 0

    def test_get_memory_requirements(self):
        """Test getting memory requirements."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            mem_req = registry.get_memory_requirements(models[0], "cuda")
            assert isinstance(mem_req, str)

    def test_get_model_size(self):
        """Test getting model size."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            size = registry.get_model_size(models[0])
            assert isinstance(size, str)
            assert ("B" in size or "M" in size)

    def test_get_architecture_type(self):
        """Test getting architecture type."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            arch = registry.get_architecture_type(models[0])
            assert isinstance(arch, str)
            assert arch in ["encoder-only", "decoder-only", "encoder-decoder", "unknown"]

    def test_filter_models(self):
        """Test filtering models."""
        registry = ModelRegistry()

        # Filter by architecture
        encoder_models = registry.filter_models(architecture="encoder-only")
        assert isinstance(encoder_models, list)

    def test_get_models_by_size(self):
        """Test grouping models by size."""
        registry = ModelRegistry()
        grouped = registry.get_models_by_size()

        assert isinstance(grouped, dict)
        assert "tiny" in grouped
        assert "small" in grouped
        assert "medium" in grouped
        assert "large" in grouped

    def test_get_quantization_config(self):
        """Test getting quantization configuration."""
        registry = ModelRegistry()
        quant_config = registry.get_quantization_config()

        assert isinstance(quant_config, dict)

    def test_supports_quantization(self):
        """Test quantization support checking."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            # CUDA supports quantization
            assert registry.supports_quantization(models[0], "cuda") is True
            # MPS does not
            assert registry.supports_quantization(models[0], "mps") is False

    def test_get_model_info(self):
        """Test getting complete model information."""
        registry = ModelRegistry()
        models = registry.list_models()

        if models:
            info = registry.get_model_info(models[0])
            assert "alias" in info
            assert "name" in info
            assert "size" in info
            assert "architecture" in info
            assert "lora_config" in info
            assert "recommended_batch_size" in info

    def test_list_all_models_function(self):
        """Test convenience function."""
        models = list_all_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_model_info_function(self):
        """Test convenience function."""
        models = list_all_models()

        if models:
            info = get_model_info(models[0])
            assert isinstance(info, dict)


class TestQuantization:
    """Tests for quantization utilities."""

    def test_supports_quantization_cuda(self):
        """Test quantization support on CUDA."""
        assert supports_quantization("cuda") is True
        assert supports_quantization(torch.device("cuda:0")) is True

    def test_supports_quantization_mps(self):
        """Test quantization support on MPS."""
        assert supports_quantization("mps") is False
        assert supports_quantization(torch.device("mps")) is False

    def test_supports_quantization_cpu(self):
        """Test quantization support on CPU."""
        assert supports_quantization("cpu") is False
        assert supports_quantization(torch.device("cpu")) is False

    def test_get_optimal_dtype_cuda(self):
        """Test optimal dtype for CUDA."""
        dtype = get_optimal_dtype("cuda")
        assert dtype == torch.bfloat16

    def test_get_optimal_dtype_mps(self):
        """Test optimal dtype for MPS."""
        dtype = get_optimal_dtype("mps")
        assert dtype == torch.float32

    def test_get_optimal_dtype_cpu(self):
        """Test optimal dtype for CPU."""
        dtype = get_optimal_dtype("cpu")
        assert dtype == torch.float32

    def test_get_optimal_dtype_device_object(self):
        """Test optimal dtype with device object."""
        dtype = get_optimal_dtype(torch.device("cpu"))
        assert dtype == torch.float32

    def test_estimate_model_size_float32(self):
        """Test model size estimation with float32."""
        estimate = estimate_model_size(1_000_000, torch.float32, quantized=False)

        assert "size_mb" in estimate
        assert "size_gb" in estimate
        assert "bytes_per_param" in estimate
        assert estimate["bytes_per_param"] == 4
        assert estimate["size_mb"] > 0

    def test_estimate_model_size_float16(self):
        """Test model size estimation with float16."""
        estimate = estimate_model_size(1_000_000, torch.float16, quantized=False)

        assert estimate["bytes_per_param"] == 2
        # float16 should use half the memory of float32
        estimate_fp32 = estimate_model_size(1_000_000, torch.float32, quantized=False)
        assert estimate["size_mb"] < estimate_fp32["size_mb"]

    def test_estimate_model_size_4bit(self):
        """Test model size estimation with 4-bit quantization."""
        estimate = estimate_model_size(
            1_000_000, torch.float32, quantized=True, quantization_bits=4
        )

        assert estimate["bytes_per_param"] == 0.5
        # 4-bit should use much less memory than float32
        estimate_fp32 = estimate_model_size(1_000_000, torch.float32, quantized=False)
        assert estimate["size_mb"] < estimate_fp32["size_mb"]

    def test_estimate_model_size_8bit(self):
        """Test model size estimation with 8-bit quantization."""
        estimate = estimate_model_size(
            1_000_000, torch.float32, quantized=True, quantization_bits=8
        )

        assert estimate["bytes_per_param"] == 1
        # 8-bit should be between 4-bit and float16
        estimate_4bit = estimate_model_size(
            1_000_000, torch.float32, quantized=True, quantization_bits=4
        )
        estimate_fp16 = estimate_model_size(1_000_000, torch.float16, quantized=False)

        assert estimate["size_mb"] > estimate_4bit["size_mb"]
        assert estimate["size_mb"] < estimate_fp16["size_mb"]

    def test_get_device_quantization_strategy_cuda(self):
        """Test quantization strategy for CUDA."""
        strategy = get_device_quantization_strategy("cuda")

        assert strategy["device"] == "cuda"
        assert strategy["supports_quantization"] is True
        assert strategy["recommended_dtype"] == torch.bfloat16
        assert strategy["recommended_precision"] == "4-bit"
        assert isinstance(strategy["notes"], list)
        assert len(strategy["notes"]) > 0

    def test_get_device_quantization_strategy_mps(self):
        """Test quantization strategy for MPS."""
        strategy = get_device_quantization_strategy("mps")

        assert strategy["device"] == "mps"
        assert strategy["supports_quantization"] is False
        assert strategy["recommended_dtype"] == torch.float32
        assert strategy["recommended_precision"] == "float32"

    def test_get_device_quantization_strategy_cpu(self):
        """Test quantization strategy for CPU."""
        strategy = get_device_quantization_strategy("cpu")

        assert strategy["device"] == "cpu"
        assert strategy["supports_quantization"] is False
        assert strategy["recommended_dtype"] == torch.float32

    def test_get_device_quantization_strategy_device_object(self):
        """Test strategy with torch.device object."""
        strategy = get_device_quantization_strategy(torch.device("cpu"))
        assert strategy["device"] == "cpu"


class TestLoRAConfiguration:
    """Tests for LoRA configuration."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_get_4bit_config_cuda(self):
        """Test 4-bit quantization config on CUDA."""
        from src.models.lora_config import LoRAConfigManager

        config = LoRAConfigManager.get_4bit_config()
        assert config is not None

    def test_get_lora_config(self):
        """Test LoRA configuration."""
        from src.models.lora_config import LoRAConfigManager

        config = LoRAConfigManager.get_lora_config(rank=8, alpha=16)

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05

    def test_get_lora_config_custom_modules(self):
        """Test LoRA config with custom target modules."""
        from src.models.lora_config import LoRAConfigManager

        custom_modules = ["query", "value"]
        config = LoRAConfigManager.get_lora_config(target_modules=custom_modules)

        # target_modules may be converted to set by peft
        assert set(config.target_modules) == set(custom_modules)

    def test_get_model_specific_lora_config(self):
        """Test getting model-specific LoRA config."""
        from src.models.lora_config import LoRAConfigManager

        # Use a model alias from registry
        config = LoRAConfigManager.get_model_specific_lora_config("DistilBERT-base")

        assert config.r > 0
        assert config.lora_alpha > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
