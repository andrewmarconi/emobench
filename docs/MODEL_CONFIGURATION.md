# Model Configuration Guide

This guide explains how to configure, add, and manage models in the EmoBench benchmark framework.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Model Categories](#model-categories)
- [Adding New Models](#adding-new-models)
- [Configuration Parameters](#configuration-parameters)
- [Recommended Model Sets](#recommended-model-sets)
- [Target Module Mapping](#target-module-mapping)
- [Troubleshooting](#troubleshooting)

## Overview

EmoBench supports 18 pre-configured models ranging from 4M to 3.8B parameters, optimized for sentiment analysis benchmarking. Models are configured in `config/models.yaml` with architecture-specific LoRA settings, batch sizes, and memory requirements.

## Configuration Structure

Models are defined in `config/models.yaml` with the following structure:

```yaml
models:
  - name: "huggingface/model-name"        # HuggingFace model ID
    alias: "Friendly-Name"                # Short name for CLI
    size_params: "100M"                   # Parameter count
    architecture: "encoder-only"          # Model architecture type
    lora:                                 # LoRA fine-tuning settings
      rank: 8
      alpha: 16
      dropout: 0.05
      target_modules: ["query", "value"]  # Attention layers to fine-tune
    recommended_batch_size:               # Device-specific batch sizes
      cuda: 32
      mps: 16
      cpu: 8
    memory_requirements:                  # Estimated memory usage
      cuda_4bit: "1GB"
      mps_fp32: "3GB"
      cpu: "6GB"
```

## Model Categories

### Ultra-Tiny Models (4M-30M Parameters)

**Use case:** Rapid experimentation, CPU training, proof-of-concept

| Model | Size | Parameters | Best For |
|-------|------|------------|----------|
| BERT-tiny | 4M | 4 million | Ultra-fast baseline, CPU training |
| BERT-mini | 11M | 11 million | Quick experiments |
| ELECTRA-small | 14M | 14 million | Efficient pre-training approach |
| BERT-small | 29M | 29 million | Balanced speed/performance |
| MiniLM-L12 | 33M | 33 million | Distilled from larger BERT |

**Training time (1 epoch, Amazon dataset):** 5-15 minutes
**Memory requirement (MPS):** 0.5-1.2GB
**Batch size (MPS):** 32-64

### Tiny Models (60M-170M Parameters)

**Use case:** Production benchmarks, standard comparisons

| Model | Size | Parameters | Best For |
|-------|------|------------|----------|
| DistilBERT-base | 66M | 66 million | Industry standard, fast baseline |
| Pythia-70m | 70M | 70 million | Decoder architecture comparison |
| DistilRoBERTa | 82M | 82 million | Improved RoBERTa distillation |
| DeBERTa-v3-small | 86M | 86 million | State-of-the-art small model |
| BERT-base | 110M | 110 million | Original BERT architecture |
| GPT2-small | 124M | 124 million | Generative model baseline |
| RoBERTa-base | 125M | 125 million | Robust BERT variant |
| Pythia-160m | 160M | 160 million | Larger Pythia variant |
| Gemma-2-2B | 270M | 270 million | Google's efficient model |

**Training time (1 epoch, Amazon dataset):** 20-40 minutes
**Memory requirement (MPS):** 2-4GB
**Batch size (MPS):** 8-16

### Small LLMs (1B-4B Parameters)

**Use case:** State-of-the-art comparisons, research benchmarks

| Model | Size | Parameters | Best For |
|-------|------|------------|----------|
| TinyLlama-1.1B | 1.1B | 1.1 billion | Smallest capable LLM |
| Qwen2.5-1.5B | 1.5B | 1.5 billion | Alibaba's efficient model |
| SmolLM2-1.7B | 1.7B | 1.7 billion | HuggingFace's optimized model |
| Phi-3-mini | 3.8B | 3.8 billion | Microsoft's high-performance model |

**Training time (1 epoch, Amazon dataset):** 2-6 hours
**Memory requirement (MPS):** 8-16GB
**Batch size (MPS):** 2-4

## Adding New Models

### Step 1: Identify Model Information

Before adding a model, gather:

1. **HuggingFace model ID** (e.g., `google/bert-base-uncased`)
2. **Parameter count** (e.g., `110M`)
3. **Architecture type** (`encoder-only` or `decoder-only`)
4. **Attention layer names** (check model architecture)

### Step 2: Find Target Modules

Target modules are the attention layers that will be fine-tuned with LoRA. Common patterns:

```python
# Check model architecture
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name")
print(model)
```

See [Target Module Mapping](#target-module-mapping) for common patterns.

### Step 3: Add Configuration

Add the model to `config/models.yaml`:

```yaml
- name: "your-org/model-name"
  alias: "YourModel"
  size_params: "100M"
  architecture: "encoder-only"
  lora:
    rank: 8                              # Use 4 for <50M, 8 for 50M-1B
    alpha: 16                            # Typically 2x rank
    dropout: 0.05
    target_modules: ["query", "value"]   # Model-specific
  recommended_batch_size:
    cuda: 32                             # GPU with 4-bit quantization
    mps: 16                              # Apple Silicon
    cpu: 8                               # CPU fallback
  memory_requirements:
    cuda_4bit: "1GB"                     # With quantization
    mps_fp32: "3GB"                      # Full precision
    cpu: "6GB"                           # Full precision
```

### Step 4: Test Configuration

```bash
# List all models to verify registration
uv run python -c "from src.models.model_registry import ModelRegistry; print(ModelRegistry().list_models())"

# Test training on small dataset
export EMOBENCH_TEST_MODE=1
uv run emobench train --model YourModel --dataset imdb --device=mps
```

## Configuration Parameters

### Model Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | HuggingFace model ID | `"distilbert-base-uncased"` |
| `alias` | string | Short CLI-friendly name | `"DistilBERT-base"` |
| `size_params` | string | Parameter count with unit | `"66M"`, `"1.1B"` |
| `architecture` | string | Model architecture type | `"encoder-only"`, `"decoder-only"` |

### LoRA Parameters

Parameter-Efficient Fine-Tuning (PEFT) configuration:

| Parameter | Type | Description | Recommended Values |
|-----------|------|-------------|-------------------|
| `rank` | int | LoRA rank (lower = fewer params) | 4 for <50M, 8 for 50M-1B, 16 for >1B |
| `alpha` | int | LoRA scaling factor | Typically 2× rank |
| `dropout` | float | Dropout probability | 0.05 (5%) |
| `target_modules` | list | Attention layers to fine-tune | See [Target Module Mapping](#target-module-mapping) |

**LoRA Rank Guidelines:**
- **Rank 4:** Ultra-tiny models (<50M) - minimal overhead
- **Rank 8:** Small-medium models (50M-1B) - balanced
- **Rank 16:** Large models (>1B) - more capacity

### Batch Size Guidelines

Batch sizes are optimized per device type:

| Device | Memory | Ultra-Tiny | Tiny | Small | Large |
|--------|--------|------------|------|-------|-------|
| **CUDA** | 8GB+ | 64 | 32 | 16 | 4 |
| **MPS** | 16GB+ | 32 | 16 | 8 | 2 |
| **CPU** | 32GB+ | 16 | 8 | 4 | 1 |

**Formula:**
```
effective_batch_size = per_device_batch_size × gradient_accumulation_steps
```

EmoBench automatically uses gradient accumulation to maintain an effective batch size of 16.

### Memory Requirements

Estimate memory based on model size and precision:

| Model Size | CUDA 4-bit | MPS FP32 | CPU FP32 |
|------------|------------|----------|----------|
| 4M-30M | 0.1-0.4GB | 0.5-1.2GB | 1-3GB |
| 60M-170M | 0.5-1.5GB | 2-4GB | 4-8GB |
| 1B-4B | 2-6GB | 8-16GB | 16-32GB |

**Formula (rough estimate):**
```
Memory (GB) ≈ Parameters × Bytes_Per_Param × Overhead_Factor

FP32: 4 bytes/param, overhead ≈ 1.5×
FP16: 2 bytes/param, overhead ≈ 1.5×
4-bit: 0.5 bytes/param, overhead ≈ 2×
```

## Recommended Model Sets

### Set 1: Ultra-Fast Benchmark (1 hour)

**Use case:** Quick validation, CI/CD testing

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12
```

**Models:** 5 ultra-tiny encoders (4M-33M)
**Training time:** ~1 hour total
**Memory:** <2GB

### Set 2: Encoder Comparison (2-3 hours)

**Use case:** Comprehensive encoder architecture study

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small DistilBERT-base \
           DistilRoBERTa DeBERTa-v3-small BERT-base RoBERTa-base
```

**Models:** 8 encoder models (4M-125M)
**Training time:** ~2-3 hours total
**Memory:** <3GB

### Set 3: Decoder Comparison (2 hours)

**Use case:** Generative model baseline

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models Pythia-70m Pythia-160m GPT2-small Gemma-2-2B
```

**Models:** 4 decoder models (70M-270M)
**Training time:** ~2 hours total
**Memory:** <4GB

### Set 4: Size Scaling Study (2 hours)

**Use case:** Analyze parameter count vs performance

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small BERT-base \
           DistilBERT-base RoBERTa-base
```

**Models:** 6 models (4M-125M) in same family
**Training time:** ~2 hours total
**Memory:** <3GB

### Set 5: Production Baseline (1 hour)

**Use case:** Fast, reliable comparison for production

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models DistilBERT-base DistilRoBERTa DeBERTa-v3-small RoBERTa-base
```

**Models:** 4 proven production models (66M-125M)
**Training time:** ~1 hour total
**Memory:** <3GB

### Set 6: Full Tiny Benchmark (4-5 hours)

**Use case:** Comprehensive lightweight model comparison

```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini ELECTRA-small BERT-small MiniLM-L12 \
           DistilBERT-base Pythia-70m DistilRoBERTa DeBERTa-v3-small \
           BERT-base GPT2-small RoBERTa-base Pythia-160m Gemma-2-2B
```

**Models:** 14 models (4M-270M)
**Training time:** ~4-5 hours total
**Memory:** <4GB

## Target Module Mapping

Different model architectures use different names for attention layers. Here's a reference:

### BERT Family (Encoder-Only)

```yaml
# google/bert-base-uncased, bert-large-uncased
target_modules: ["query", "value"]

# distilbert-base-uncased
target_modules: ["q_lin", "v_lin"]

# roberta-base, roberta-large, distilroberta-base
target_modules: ["query", "value"]

# microsoft/deberta-v3-small, deberta-v3-base
target_modules: ["query_proj", "value_proj"]

# google/electra-small-discriminator
target_modules: ["query", "value"]

# microsoft/MiniLM-L12-H384-uncased
target_modules: ["query", "value"]
```

### GPT Family (Decoder-Only)

```yaml
# openai-community/gpt2, gpt2-medium
target_modules: ["c_attn"]  # Combined QKV projection

# EleutherAI/pythia-70m, pythia-160m, pythia-410m
target_modules: ["query_key_value"]  # Combined QKV

# TinyLlama/TinyLlama-1.1B-Chat-v1.0
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Qwen/Qwen2.5-1.5B
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# HuggingFaceTB/SmolLM2-1.7B
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# microsoft/Phi-3-mini-4k-instruct
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# google/gemma-2-2b, google/gemma-3-270m
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Finding Target Modules

To find target modules for a new model:

```python
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("model-name")

# Print architecture
print(model)

# Look for attention layers - common patterns:
# - query, key, value (BERT family)
# - q_proj, k_proj, v_proj, o_proj (Llama family)
# - c_attn (GPT-2)
# - query_key_value (Pythia)
# - q_lin, k_lin, v_lin (DistilBERT)
# - query_proj, value_proj (DeBERTa)
```

Or use the model registry helper:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model-name")
print(config)  # Look for attention-related parameters
```

## Troubleshooting

### Model Not Found

**Error:** `Model 'YourModel' not found. Available models: [...]`

**Solution:**
1. Check that the alias in `config/models.yaml` matches the name used in CLI
2. Verify YAML syntax (indentation, quotes)
3. Reload configuration: restart Python or clear cache

### Target Module Errors

**Error:** `Target modules not found in model`

**Solution:**
1. Print model architecture to find correct layer names
2. Check [Target Module Mapping](#target-module-mapping)
3. Verify model architecture type (encoder vs decoder)

### Out of Memory (OOM)

**Error:** `MPS backend out of memory`

**Solutions:**
1. **Reduce batch size:** Edit `recommended_batch_size.mps` in config
2. **Enable memory optimization:** Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
3. **Skip large models:** Use `--models` flag to train subset
4. **Close other applications:** Free up unified memory

### Slow Training

**Issue:** Training taking too long

**Solutions:**
1. **Start with ultra-tiny models:** Use Set 1 (BERT-tiny, BERT-mini)
2. **Reduce dataset size:** Set `EMOBENCH_TEST_MODE=1` for quick testing
3. **Increase batch size:** If memory allows, use larger batches
4. **Use CUDA:** Train on NVIDIA GPU for 4-bit quantization

### Model Not Loading

**Error:** `Failed to load model from HuggingFace`

**Solutions:**
1. **Check internet connection:** Models download from HuggingFace Hub
2. **Set HF_TOKEN:** Some models require authentication:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```
3. **Check model name:** Verify model exists on HuggingFace
4. **Clear cache:** Delete `~/.cache/huggingface/`

## Best Practices

### 1. Start Small
Begin with Set 1 (ultra-tiny models) to validate your setup before running larger benchmarks.

### 2. Monitor Memory
Use Activity Monitor (macOS) or `nvidia-smi` (Linux) to track memory usage during training.

### 3. Document Results
Save training results and configurations for reproducibility:
```bash
# Results automatically saved to:
experiments/checkpoints/{model}_{dataset}/training_results.json
```

### 4. Use Test Mode
For quick validation, use test mode with reduced dataset:
```bash
export EMOBENCH_TEST_MODE=1
uv run emobench train-all --dataset amazon --device=mps --models BERT-tiny
```

### 5. Encoder vs Decoder
- **Encoder models** (BERT family) typically perform better on sentiment classification
- **Decoder models** (GPT, Llama family) are more versatile but slower and may need more data

### 6. Batch Size Tuning
If you encounter OOM errors, reduce batch size in config:
```yaml
recommended_batch_size:
  mps: 8  # Reduce from 16 to 8
```

## Additional Resources

- **HuggingFace Model Hub:** https://huggingface.co/models
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **PEFT Library:** https://github.com/huggingface/peft
- **Model Architecture Guide:** See `docs/ARCHITECTURE.md`
- **Training Guide:** See `docs/TRAINING.md`

## Examples

### Example 1: Quick Validation

```bash
# Test framework with smallest model
export EMOBENCH_TEST_MODE=1
uv run emobench train --model BERT-tiny --dataset imdb --device=mps
```

### Example 2: Compare BERT Variants

```bash
# Compare original BERT vs distilled versions
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small DistilBERT-base BERT-base
```

### Example 3: Encoder vs Decoder

```bash
# Compare encoder (RoBERTa) vs decoder (GPT2) on same task
uv run emobench train-all --dataset sst2 --device=mps \
  --models RoBERTa-base GPT2-small
```

### Example 4: Production Model Selection

```bash
# Test top production candidates
uv run emobench train-all --dataset amazon --device=mps \
  --models DistilBERT-base DistilRoBERTa DeBERTa-v3-small

# Review results and select best model
uv run emobench report --results-dir experiments/results
```

---

**Last updated:** 2025-11-24
**EmoBench version:** 0.1.0
