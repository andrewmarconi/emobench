![MoodBench](moodbench-logo.svg)
## Multi-LLM Sentiment Analysis Benchmark Framework

**Fast, efficient benchmarking of 17 small language models (4M-410M parameters) for sentiment analysis using LoRA fine-tuning.**



---

## 🚀 Quick Start

### Command Line Interface
```bash
# Install dependencies
uv sync

# Train models (1 hour, 5 models)
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12

# Evaluate models
uv run moodbench benchmark --dataset amazon

# View results
uv run moodbench report --results-dir experiments/results
```

### Web Interface (Alternative)
```bash
# Install additional dependencies
uv add gradio

# Launch web UI
python gradio_app.py

# Open http://localhost:7860 in your browser
```

The web interface provides modular tabs for training, benchmarking, analysis, NPS estimation, and methodology documentation.

## 📚 Documentation

### Getting Started
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Commands, model lists, and common workflows ⚡ **Start here!**
- **[Model Configuration Guide](docs/MODEL_CONFIGURATION.md)** - Complete guide to all 18 models and configurations

### User Interfaces
- **[Gradio Web UI](gradio_app.py)** - Interactive web interface with modular tabs for training, benchmarking, analysis, NPS estimation, and methodology

### Technical Details
- **[CLAUDE.md](CLAUDE.md)** - Architecture, technical implementation, and development guide
- **[Documentation Index](docs/README.md)** - Navigate all documentation by role and use case

## 🎯 What is MoodBench?

MoodBench is an automated benchmarking framework that fine-tunes, evaluates, and compares small language models for sentiment analysis. It uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to enable efficient training on consumer hardware.

**Key Features:**
- 🏃 **17 optimized models** from 4M to 410M parameters
- ⚡ **Fast benchmarking** - Ultra-tiny models train in 5-15 minutes
- 💾 **Memory efficient** - All models <6GB on Apple Silicon M4
- 📊 **Comprehensive metrics** - Accuracy, F1, balanced accuracy, latency percentiles, throughput, memory, statistical significance, robustness
- 🔧 **Production ready** - CI/CD-friendly, reproducible benchmarks
- 🌐 **Web Interface** - Interactive Gradio UI for all operations

## 📊 Available Models

### Ultra-Tiny (4M-30M) - Fastest
`BERT-tiny` `BERT-mini` `ELECTRA-small` `BERT-small` `MiniLM-L12`

### Tiny (60M-170M) - Production Quality
`DistilBERT-base` `Pythia-70m` `DistilRoBERTa` `DeBERTa-v3-small` `BERT-base` `GPT2-small` `RoBERTa-base` `Pythia-160m` `DialoGPT-small` `DistilGPT2`

### Medium (200M-500M) - Research Quality
`Gemma-2-2B` `Pythia-410m`

**See [Quick Reference](docs/QUICK_REFERENCE.md) for full details and benchmarks.**

## 💡 Common Use Cases

### Quick Validation
```bash
export MOODBENCH_TEST_MODE=1
uv run moodbench train --model BERT-tiny --dataset imdb --device=mps
```

### Production Model Selection
```bash
uv run moodbench train-all --dataset amazon --device=mps \
  --models DistilBERT-base DistilRoBERTa DeBERTa-v3-small RoBERTa-base
```

### Research Comparison
```bash
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small BERT-base DistilBERT-base RoBERTa-base
```

## 🎨 Architecture

```
Data Pipeline → Training Engine → Evaluation Engine → Comparison Module → Visualization
     ↓               ↓                    ↓                    ↓               ↓
   Loader      LoRA/QLoRA           Metrics             Statistical      Dashboard
Preprocessor   4-bit Quant     Speed Benchmark          Analysis          Reports
 Tokenizer     Multi-Device     Memory Profile           Ranking          Charts
```

**Web Interface:** Modular Gradio UI with dedicated tabs for training, benchmarking, analysis, NPS estimation, and methodology documentation.

## 📦 Supported Datasets

- **IMDB** - Movie reviews (50K samples)
- **SST2** - Stanford Sentiment Treebank (67K sentences)
- **Amazon** - Product reviews (4M samples)
- **Yelp** - Business reviews (650K samples)

## 🖥️ Hardware Support

| Platform | Status | Optimizations |
|----------|--------|---------------|
| **CUDA (NVIDIA)** | ✅ Full support | 4-bit quantization, fp16 |
| **MPS (Apple Silicon)** | ✅ Full support | Dynamic batching, gradient checkpointing |
| **CPU** | ✅ Supported | Optimized for ultra-tiny models |

**Recommended:**
- **CUDA**: 16GB+ RAM, 8GB+ VRAM
- **MPS**: M2/M3 with 32GB+ unified memory
- **CPU**: 32GB+ RAM (ultra-tiny models only)

## 🛠️ Installation

### Using uv (Recommended)
```bash
git clone https://github.com/yourusername/moodbench.git
cd moodbench
uv sync
```

### Using pip
```bash
git clone https://github.com/yourusername/moodbench.git
cd moodbench
pip install -e .
```

**Requirements:**
- Python 3.12+
- PyTorch 2.1+
- 50GB+ storage for datasets and models

## 📖 CLI Commands

```bash
# Train single model
uv run moodbench train --model <model-name> --dataset <dataset>

# Train multiple models
uv run moodbench train-all --dataset <dataset> --models <model1> <model2> ...

# Evaluate model
uv run moodbench evaluate --model <model> --dataset <dataset> --checkpoint <path>

# Run benchmarks
uv run moodbench benchmark --models BERT-tiny DistilBERT-base --datasets imdb sst2

# Generate reports
uv run moodbench report --results-dir experiments/results
```

See [Quick Reference](docs/QUICK_REFERENCE.md) for detailed usage.

## 📁 Project Structure

```
moodbench/
├── config/              # Model, dataset, and training configurations
├── src/                 # Core framework code
│   ├── data/           # Dataset loading and preprocessing
│   ├── models/         # Model registry and LoRA configurations
│   ├── training/       # Training engine and optimizers
│   ├── evaluation/     # Metrics and benchmarking
│   ├── comparison/     # Result aggregation and ranking
│   ├── ui/             # Modular Gradio web interface components
│   └── visualization/  # Dashboard and reporting
├── experiments/        # Training logs, checkpoints, results
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit and integration tests
├── scripts/            # Shell scripts for common tasks
└── docs/               # Comprehensive documentation
```

## 🔧 Configuration

Models are configured in `config/models.yaml`:

```yaml
- name: "prajjwal1/bert-tiny"
  alias: "BERT-tiny"
  size_params: "4M"
  architecture: "encoder-only"
  lora:
    rank: 4
    alpha: 8
    dropout: 0.05
    target_modules: ["query", "value"]
  recommended_batch_size:
    cuda: 64
    mps: 32
    cpu: 16
  memory_requirements:
    cuda_4bit: "0.1GB"
    mps_fp32: "0.5GB"
    cpu: "1GB"
```

See [Model Configuration Guide](docs/MODEL_CONFIGURATION.md) for details on adding custom models.

## 🐛 Troubleshooting

### Out of Memory (MPS)
```bash
# Use smaller models
--models BERT-tiny BERT-mini BERT-small DistilBERT-base

# Or allow higher memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Training Too Slow
```bash
# Enable test mode with small dataset
export MOODBENCH_TEST_MODE=1

# Start with ultra-tiny models
--models BERT-tiny BERT-mini
```

### Model Not Found
```bash
# List available models
uv run python -c "from src.models.model_registry import ModelRegistry; \
  print('\n'.join(ModelRegistry().list_models()))"
```

See [Model Configuration Guide - Troubleshooting](docs/MODEL_CONFIGURATION.md#troubleshooting) for more solutions.

## 📊 Example Results

| Model | Size | Accuracy | F1 | Latency (ms) | Throughput (tok/s) | Memory (MB) |
|-------|------|----------|----|--------------|--------------------|-------------|
| BERT-tiny | 4M | 0.823 | 0.815 | 8.2 | 5000+ | 500 |
| DistilBERT-base | 66M | 0.915 | 0.910 | 18.5 | 2500 | 2000 |
| RoBERTa-base | 125M | 0.932 | 0.928 | 32.1 | 1800 | 3000 |
| DeBERTa-v3-small | 86M | 0.935 | 0.931 | 24.3 | 2100 | 2500 |

*Results on IMDB dataset, Apple Silicon M3 Max, 1 epoch*

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Adding new models to the registry
- Supporting additional datasets
- Improving benchmarking metrics
- Enhancing visualization
- Documentation improvements

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [Transformers](https://github.com/huggingface/transformers) - Model implementations
- [PEFT](https://github.com/huggingface/peft) - LoRA fine-tuning
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gradio](https://gradio.app/) - Interactive web interface

---

**For detailed documentation, see:**
- 📖 [Quick Reference](docs/QUICK_REFERENCE.md) - Get started in 5 minutes
- 🔧 [Model Configuration Guide](docs/MODEL_CONFIGURATION.md) - Complete technical guide
- 🏗️ [CLAUDE.md](CLAUDE.md) - Architecture and implementation details
- 📚 [Documentation Index](docs/README.md) - Navigate all docs

**Project version:** 0.1.0
**Last updated:** 2025-11-24
