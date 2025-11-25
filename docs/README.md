# MoodBench Documentation

Welcome to the MoodBench documentation! This directory contains comprehensive guides for using the multi-LLM sentiment analysis benchmark framework.

## 📚 Documentation Index

### Getting Started

- **[Quick Reference](QUICK_REFERENCE.md)** - Start here! Quick commands, model lists, and common workflows
- **[Model Configuration Guide](MODEL_CONFIGURATION.md)** - Complete guide to models, configuration, and customization

### Core Documentation

Located in the root directory:

- **[CLAUDE.md](../CLAUDE.md)** - Project overview, architecture, and technical details for Claude Code
- **[README.md](../README.md)** - Project introduction and setup instructions

## 🚀 Quick Navigation

### I want to...

**...get started quickly**
→ Read [Quick Reference](QUICK_REFERENCE.md) and run the Ultra-Fast benchmark

**...understand all available models**
→ See [Model Configuration Guide](MODEL_CONFIGURATION.md) Section: Model Categories

**...add a new model**
→ See [Model Configuration Guide](MODEL_CONFIGURATION.md) Section: Adding New Models

**...choose the right model set**
→ See [Quick Reference](QUICK_REFERENCE.md) Section: Recommended Benchmark Sets

**...troubleshoot an error**
→ See [Model Configuration Guide](MODEL_CONFIGURATION.md) Section: Troubleshooting

**...understand the architecture**
→ Read [CLAUDE.md](../CLAUDE.md) Section: Architecture

**...optimize memory usage**
→ See [Quick Reference](QUICK_REFERENCE.md) Section: Troubleshooting Quick Fixes

## 📖 Documentation by Use Case

### For Data Scientists

**Goal:** Compare model performance on sentiment analysis

1. Start with [Quick Reference](QUICK_REFERENCE.md) - Recommended Benchmark Sets
2. Run the Production Baseline set (1 hour, 4 models)
3. Analyze results using the comparison reports
4. Expand to more models as needed

**Recommended reading:**
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands and model selection
- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - Understanding model trade-offs

### For ML Engineers

**Goal:** Optimize model selection for production deployment

1. Review [Model Configuration Guide](MODEL_CONFIGURATION.md) - Model Categories
2. Understand memory/speed trade-offs
3. Run comprehensive benchmarks on your target datasets
4. Analyze latency, throughput, and memory metrics

**Recommended reading:**
- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - Full guide including memory requirements
- [CLAUDE.md](../CLAUDE.md) - Hardware considerations and optimizations

### For Researchers

**Goal:** Study model architectures and scaling laws

1. Read [Model Configuration Guide](MODEL_CONFIGURATION.md) - Target Module Mapping
2. Understand LoRA configuration and parameter-efficient fine-tuning
3. Run Size Scaling Study benchmark set
4. Analyze performance vs parameter count

**Recommended reading:**
- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - Complete technical details
- [CLAUDE.md](../CLAUDE.md) - LoRA/QLoRA implementation details

### For Platform Engineers

**Goal:** Deploy and maintain the benchmark framework

1. Review [CLAUDE.md](../CLAUDE.md) - Package management and dependencies
2. Understand hardware requirements and device-specific optimizations
3. Set up CI/CD using ultra-fast benchmark set
4. Configure monitoring and result tracking

**Recommended reading:**
- [CLAUDE.md](../CLAUDE.md) - Complete architecture and technical guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Environment variables and troubleshooting

## 🎯 Common Tasks

### Running Your First Benchmark

```bash
# 1. Quick validation (5 minutes)
export MOODBENCH_TEST_MODE=1
uv run moodbench train --model BERT-tiny --dataset imdb --device=mps

# 2. Train models (1 hour)
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12

# 3. Evaluate models
uv run moodbench benchmark --dataset amazon

# 4. Analyze results
uv run moodbench report --results-dir experiments/results
```

See: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

### Adding a Custom Model

```yaml
# 1. Add to config/models.yaml
- name: "org/model-name"
  alias: "ModelName"
  size_params: "100M"
  architecture: "encoder-only"
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["query", "value"]
  recommended_batch_size:
    cuda: 32
    mps: 16
    cpu: 8
  memory_requirements:
    cuda_4bit: "1GB"
    mps_fp32: "3GB"
    cpu: "6GB"

# 2. Test the configuration
uv run python -c "from src.models.model_registry import ModelRegistry; \
  print(ModelRegistry().list_models())"

# 3. Train on small dataset
export MOODBENCH_TEST_MODE=1
uv run moodbench train --model ModelName --dataset imdb --device=mps
```

See: [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) Section: Adding New Models

### Troubleshooting Memory Issues

```bash
# Option 1: Use smaller models
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small DistilBERT-base

# Option 2: Allow higher memory allocation (MPS)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
uv run moodbench train-all --dataset amazon --device=mps

# Option 3: Check which models fit in memory
# Models < 200M typically use < 4GB on MPS
uv run python -c "from src.models.model_registry import ModelRegistry; \
  r = ModelRegistry(); \
  print([m for m in r.list_models() if 'M' in r.get_model_size(m)])"
```

See: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) Section: Troubleshooting Quick Fixes

## 📊 Model Selection Guide

### By Speed

| Speed | Models | Use Case |
|-------|--------|----------|
| ⚡⚡⚡⚡⚡ Fastest | BERT-tiny, BERT-mini | Quick experiments, CI/CD |
| ⚡⚡⚡⚡ Very Fast | BERT-small, ELECTRA-small, MiniLM-L12 | Rapid prototyping |
| ⚡⚡⚡ Fast | DistilBERT-base, DistilRoBERTa, DeBERTa-v3-small | Production baseline |
| ⚡⚡ Moderate | BERT-base, RoBERTa-base, GPT2-small | Full comparison |
| ⚡ Slow | Pythia-160m, Gemma-2-2B | Research quality |
| 🐌 Very Slow | TinyLlama-1.1B+, Phi-3-mini | State-of-the-art |

### By Memory

| Memory | Models | Count |
|--------|--------|-------|
| <1GB | BERT-tiny, BERT-mini, ELECTRA-small | 3 |
| 1-2GB | BERT-small, MiniLM-L12, DistilBERT-base | 3 |
| 2-4GB | Pythia-70m, DistilRoBERTa, DeBERTa-v3-small, BERT-base, GPT2-small, RoBERTa-base | 6 |
| 4-8GB | Pythia-160m, Gemma-2-2B, TinyLlama-1.1B | 3 |
| 8-16GB | Qwen2.5-1.5B, SmolLM2-1.7B, Phi-3-mini | 3 |

### By Use Case

| Use Case | Recommended Models |
|----------|-------------------|
| **Fastest validation** | BERT-tiny |
| **Quick comparison** | BERT-tiny, BERT-mini, BERT-small |
| **Production baseline** | DistilBERT-base, RoBERTa-base |
| **State-of-the-art** | DeBERTa-v3-small, Gemma-2-2B |
| **Encoder vs Decoder** | RoBERTa-base vs GPT2-small |
| **Scaling laws** | BERT-tiny → BERT-mini → BERT-small → BERT-base |

## 🔧 Configuration Files

```
config/
├── models.yaml      # Model configurations (17 models)
├── datasets.yaml    # Dataset settings (4 datasets)
├── training.yaml    # Training hyperparameters
└── evaluation.yaml  # Evaluation metrics
```

**Key configurations:**
- **Batch sizes**: Automatically adjusted per device and model size
- **LoRA settings**: Rank 4-16 depending on model size
- **Memory requirements**: Estimated per device type
- **Target modules**: Architecture-specific attention layers

## 🐛 Common Issues

### "Model not found"
→ Check `uv run python -c "from src.models.model_registry import ModelRegistry; print(ModelRegistry().list_models())"`

### "Out of memory"
→ Use smaller models or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

### "Target modules not found"
→ Check [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) Section: Target Module Mapping

### "Training too slow"
→ Start with ultra-tiny models (BERT-tiny, BERT-mini) or enable test mode: `export MOODBENCH_TEST_MODE=1`

## 📝 Additional Resources

- **HuggingFace Model Hub**: https://huggingface.co/models
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **PEFT Library**: https://github.com/huggingface/peft
- **Transformers Docs**: https://huggingface.co/docs/transformers

## 🤝 Contributing

To add new documentation:

1. Create a new `.md` file in the `docs/` directory
2. Add it to this index
3. Update cross-references in related documents
4. Test all code examples

## 📧 Support

For issues or questions:
- Check [Troubleshooting](#-common-issues) section
- Review [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) Section: Troubleshooting
- Open an issue on GitHub

---

**Documentation version:** 1.0.0
**Last updated:** 2025-11-24
**MoodBench version:** 0.1.0
