# EmoBench Quick Reference

## Models by Speed/Memory

| Model | Size | Speed | Memory | Best For |
|-------|------|-------|--------|----------|
| BERT-tiny | 4M | ⚡⚡⚡⚡⚡ | 0.5GB | Fastest baseline |
| BERT-mini | 11M | ⚡⚡⚡⚡⚡ | 0.8GB | Quick experiments |
| ELECTRA-small | 14M | ⚡⚡⚡⚡ | 0.8GB | Efficient training |
| BERT-small | 29M | ⚡⚡⚡⚡ | 1GB | Speed/accuracy balance |
| MiniLM-L12 | 33M | ⚡⚡⚡⚡ | 1.2GB | Distilled performance |
| DistilBERT-base | 66M | ⚡⚡⚡ | 2GB | Industry standard |
| Pythia-70m | 70M | ⚡⚡⚡ | 2GB | Decoder baseline |
| DistilRoBERTa | 82M | ⚡⚡⚡ | 2.5GB | Robust distillation |
| DeBERTa-v3-small | 86M | ⚡⚡⚡ | 2.5GB | State-of-the-art small |
| BERT-base | 110M | ⚡⚡⚡ | 3GB | Original BERT |
| GPT2-small | 124M | ⚡⚡⚡ | 3GB | Generative baseline |
| RoBERTa-base | 125M | ⚡⚡⚡ | 3GB | Production quality |
| Pythia-160m | 160M | ⚡⚡ | 4GB | Larger decoder |
| Gemma-2-2B | 270M | ⚡⚡ | 3GB | Google's efficient |
| TinyLlama-1.1B | 1.1B | ⚡ | 8GB | Small LLM |
| Qwen2.5-1.5B | 1.5B | ⚡ | 10GB | Alibaba's LLM |
| SmolLM2-1.7B | 1.7B | ⚡ | 10GB | HF's optimized LLM |
| Phi-3-mini | 3.8B | 🐌 | 16GB | High performance |

## Common Commands

### Train Single Model
```bash
uv run emobench train --model DistilBERT-base --dataset amazon --device=mps
```

### Train Multiple Models
```bash
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini DistilBERT-base
```

### Quick Test (Small Dataset)
```bash
export EMOBENCH_TEST_MODE=1
uv run emobench train --model BERT-tiny --dataset imdb --device=mps
```

### Skip Large Models (Avoid OOM)
```bash
# Train only models < 200M
uv run emobench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small DistilBERT-base \
           DistilRoBERTa DeBERTa-v3-small BERT-base RoBERTa-base
```

### Set Memory Limit (MPS)
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
uv run emobench train-all --dataset amazon --device=mps
```

## Recommended Benchmark Sets

### 🚀 Ultra-Fast (1 hour, <2GB)
```bash
--models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12
```

### 📊 Encoder Comparison (2-3 hours, <3GB)
```bash
--models BERT-tiny BERT-mini BERT-small DistilBERT-base \
         DistilRoBERTa DeBERTa-v3-small BERT-base RoBERTa-base
```

### 🔄 Decoder Comparison (2 hours, <4GB)
```bash
--models Pythia-70m Pythia-160m GPT2-small Gemma-2-2B
```

### 📈 Size Scaling (2 hours, <3GB)
```bash
--models BERT-tiny BERT-mini BERT-small BERT-base \
         DistilBERT-base RoBERTa-base
```

### 🏭 Production Baseline (1 hour, <3GB)
```bash
--models DistilBERT-base DistilRoBERTa DeBERTa-v3-small RoBERTa-base
```

### 🌟 Full Tiny Benchmark (4-5 hours, <4GB)
```bash
--models BERT-tiny BERT-mini ELECTRA-small BERT-small MiniLM-L12 \
         DistilBERT-base Pythia-70m DistilRoBERTa DeBERTa-v3-small \
         BERT-base GPT2-small RoBERTa-base Pythia-160m Gemma-2-2B
```

## Dataset Options

| Dataset | Name | Size | Domain |
|---------|------|------|--------|
| IMDB | `imdb` | 50K reviews | Movies |
| SST2 | `sst2` | 67K sentences | Movies |
| Amazon | `amazon` | 4M reviews | Products |
| Yelp | `yelp` | 650K reviews | Businesses |

## Troubleshooting Quick Fixes

### Out of Memory
```bash
# Option 1: Allow higher memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Option 2: Train smaller models only
--models BERT-tiny BERT-mini BERT-small DistilBERT-base

# Option 3: Skip specific large models
--models $(python -c "from src.models.model_registry import ModelRegistry; \
  print(' '.join([m for m in ModelRegistry().list_models() \
  if 'B' not in ModelRegistry().get_model_size(m)]))")
```

### Slow Training
```bash
# Use test mode with small dataset
export EMOBENCH_TEST_MODE=1

# Start with fastest models
--models BERT-tiny BERT-mini

# Increase batch size if memory allows (edit config/models.yaml)
```

### Model Not Found
```bash
# List available models
uv run python -c "from src.models.model_registry import ModelRegistry; \
  print('\n'.join(ModelRegistry().list_models()))"
```

## Performance Expectations (MPS, 64GB)

| Model Size | Training Time | Inference Speed | Memory |
|------------|---------------|-----------------|--------|
| 4M-30M | 5-15 min | 1000-5000 samples/sec | <2GB |
| 60M-170M | 20-40 min | 500-1000 samples/sec | 2-4GB |
| 1B-4B | 2-6 hours | 50-200 samples/sec | 8-16GB |

*Based on 1 epoch, 10K samples, Amazon dataset*

## Model Selection Guide

### I need the fastest results
→ **BERT-tiny** (4M) or **BERT-mini** (11M)

### I need production quality
→ **DistilBERT-base** (66M) or **RoBERTa-base** (125M)

### I need state-of-the-art
→ **DeBERTa-v3-small** (86M) or **Phi-3-mini** (3.8B)

### I need to compare encoder vs decoder
→ **RoBERTa-base** (125M encoder) vs **GPT2-small** (124M decoder)

### I need to study scaling laws
→ **BERT-tiny** (4M), **BERT-mini** (11M), **BERT-small** (29M), **BERT-base** (110M)

### I have limited memory (<4GB available)
→ Ultra-tiny set: **BERT-tiny**, **BERT-mini**, **BERT-small**, **ELECTRA-small**, **MiniLM-L12**

### I have plenty of memory (16GB+)
→ Full benchmark: All 18 models including **Phi-3-mini** (3.8B)

## File Locations

```
experiments/
├── checkpoints/          # Trained model checkpoints
│   └── {model}_{dataset}/
│       ├── final/        # Final model weights
│       └── training_results.json
└── results/              # Benchmark results
    └── aggregated_results.csv

config/
├── models.yaml           # Model configurations
├── datasets.yaml         # Dataset configurations
└── training.yaml         # Training hyperparameters

docs/
├── MODEL_CONFIGURATION.md    # Full model guide
└── QUICK_REFERENCE.md        # This file
```

## Environment Variables

```bash
# Enable test mode (small datasets)
export EMOBENCH_TEST_MODE=1

# Allow higher MPS memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# HuggingFace token (for gated models)
export HF_TOKEN=your_token_here

# Verbose logging
uv run emobench train --verbose
```

## Next Steps

1. **Validate setup:** Run BERT-tiny in test mode
   ```bash
   export EMOBENCH_TEST_MODE=1
   uv run emobench train --model BERT-tiny --dataset imdb --device=mps
   ```

2. **Run first benchmark:** Ultra-fast set (1 hour)
   ```bash
   uv run emobench train-all --dataset amazon --device=mps \
     --models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12
   ```

3. **Analyze results:** Generate comparison report
   ```bash
   uv run emobench report --results-dir experiments/results
   ```

4. **Expand benchmark:** Add more models as needed
   ```bash
   uv run emobench train-all --dataset amazon --device=mps \
     --models DistilBERT-base DistilRoBERTa DeBERTa-v3-small RoBERTa-base
   ```

---

For detailed documentation, see [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md)
