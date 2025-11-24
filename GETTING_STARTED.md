# Getting Started with SentiCompare

Welcome to **SentiCompare** - a comprehensive framework for benchmarking and comparing small language models (SLMs) for sentiment analysis tasks. This guide will help you get up and running in minutes.

## 🎯 What is SentiCompare?

SentiCompare is an automated benchmarking framework that:
- **Fine-tunes** multiple SLMs using LoRA/QLoRA with 4-bit quantization
- **Evaluates** model performance on accuracy, F1-score, precision, and recall
- **Benchmarks** inference speed (latency, throughput) and memory usage
- **Compares** models across multiple metrics with interactive visualizations
- **Supports** CUDA, MPS (Apple Silicon), and CPU devices

Perfect for researchers and practitioners who want to systematically evaluate SLMs for sentiment analysis while maintaining memory efficiency.

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16GB | 32GB+ |
| **GPU** | None (CPU only) | 8GB+ VRAM (CUDA) or Apple Silicon (MPS) |
| **Storage** | 20GB | 50GB+ |
| **OS** | Linux/macOS/Windows | Linux/macOS with CUDA |

### Software Requirements

- **Python**: 3.12 or higher
- **Git**: For cloning the repository
- **uv**: Modern Python package manager (recommended)

### Hardware Compatibility

- ✅ **CUDA**: Full support with 4-bit quantization
- ✅ **MPS**: Apple Silicon support (no quantization, LoRA only)
- ✅ **CPU**: Fallback support for any system

## 🚀 Quick Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/senticompare.git
cd senticompare

# Install dependencies (fast and reliable)
uv sync

# Verify installation
uv run senticompare --help
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/senticompare.git
cd senticompare

# Install in development mode
pip install -e .

# Verify installation
senticompare --help
```

### Option 3: Using Docker (Coming Soon)

```bash
# Docker support planned for future release
# Will support both CPU and CUDA versions
```

## ⚡ Quick Start (5 minutes)

Let's train and evaluate your first model!

### Step 1: Train a Model

```bash
# Train DistilBERT on IMDB dataset (fastest option)
senticompare train --model DistilBERT-base --dataset imdb
```

This will:
- Download the IMDB dataset (~200MB)
- Download DistilBERT model (~250MB)
- Fine-tune with LoRA for 3 epochs
- Save checkpoints to `experiments/checkpoints/`

### Step 2: Evaluate Performance

```bash
# Evaluate the trained model
senticompare evaluate --model DistilBERT-base --dataset imdb \
  --checkpoint experiments/checkpoints/DistilBERT-base_imdb/final
```

This will:
- Load your trained model
- Evaluate on test set
- Measure accuracy, F1, latency, memory usage
- Save results to `experiments/results/`

### Step 3: View Results

```bash
# Launch interactive dashboard
senticompare dashboard
```

This opens a web interface at http://localhost:8501 showing:
- Performance metrics comparison
- Speed vs accuracy trade-offs
- Detailed results table

## 📊 Basic Usage Examples

### Training Different Models

```bash
# Train on different datasets
senticompare train --model DistilBERT-base --dataset sst2
senticompare train --model TinyLlama-1.1B --dataset imdb

# Train all models at once
senticompare train-all --dataset imdb

# Custom output directory
senticompare train --model DistilBERT-base --dataset imdb \
  --output-dir my_experiments/
```

### Benchmarking Multiple Models

```bash
# Benchmark all trained models
senticompare benchmark --dataset imdb

# Generate comparison reports
senticompare report --format all
```

### Device-Specific Training

```bash
# Auto-detect device (recommended)
senticompare train --model DistilBERT-base --dataset imdb

# Force specific device
senticompare train --model DistilBERT-base --dataset imdb --device cuda
senticompare train --model DistilBERT-base --dataset imdb --device mps
senticompare train --model DistilBERT-base --dataset imdb --device cpu
```

## 🔧 Configuration

### Model Configuration

Edit `config/models.yaml` to customize models:

```yaml
models:
  - name: "distilbert-base-uncased"
    alias: "DistilBERT-base"
    lora_rank: 8
    lora_alpha: 16
    target_modules: ["q_lin", "k_lin", "v_lin", "out_lin"]
```

### Dataset Configuration

Edit `config/datasets.yaml` for custom datasets:

```yaml
datasets:
  imdb:
    name: "IMDB Movie Reviews"
    source: "huggingface"
    dataset_id: "stanfordnlp/imdb"
    text_column: "text"
    label_column: "label"
```

## 📈 Understanding Results

### Performance Metrics

After evaluation, you'll see:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Speed Metrics

- **TTFT (ms)**: Time to first token
- **Median Latency (ms)**: Median inference time
- **P99 Latency (ms)**: 99th percentile latency
- **Throughput (tokens/sec)**: Processing speed
- **Memory Usage (MB)**: GPU/CPU memory consumption

## 🎛️ Advanced Usage

### Custom Training Parameters

```bash
# Advanced training (when modules are implemented)
senticompare train --model DistilBERT-base --dataset imdb \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4
```

### Multi-Dataset Evaluation

```bash
# Train on multiple datasets
senticompare train-all --dataset imdb
senticompare train-all --dataset sst2
senticompare train-all --dataset amazon

# Benchmark across datasets
senticompare benchmark --dataset imdb
senticompare benchmark --dataset sst2
```

### Exporting Results

```bash
# Generate reports in different formats
senticompare report --format json     # Machine-readable
senticompare report --format csv      # Spreadsheet-friendly
senticompare report --format markdown # Human-readable
```

## 🐛 Troubleshooting

### Common Issues

**"Module not implemented yet"**
- Some advanced features are still in development
- Start with basic commands like `train` and `evaluate`

**Memory errors on CUDA**
```bash
# Use smaller batch size or model
senticompare train --model TinyLlama-1.1B --dataset imdb
```

**Memory errors on MPS**
```bash
# MPS has unified memory - reduce model size
senticompare train --model DistilBERT-base --dataset imdb --device mps
```

**Download failures**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
senticompare train --model DistilBERT-base --dataset imdb
```

### Getting Help

```bash
# CLI help
senticompare --help
senticompare train --help

# Verbose logging
senticompare train --model DistilBERT-base --dataset imdb --verbose

# Check logs
tail -f experiments/logs/*.log
```

## 📚 Next Steps

### Learn More

1. **Explore the Dashboard**: Run `senticompare dashboard` to see interactive visualizations
2. **Compare Models**: Train multiple models and use `senticompare benchmark`
3. **Read the Full Documentation**: See `README.md` for detailed API reference
4. **Customize Configurations**: Modify `config/` files for your use case

### Advanced Topics

- **Adding New Models**: Extend `config/models.yaml`
- **Custom Datasets**: Add support for your own datasets
- **Performance Optimization**: Memory efficiency and training speed tips
- **Production Deployment**: Export optimized models

### Community & Support

- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Contributing**: See contribution guidelines

## 🎉 You're All Set!

You've successfully installed SentiCompare and trained your first model! The framework is designed to be extensible and production-ready. As you explore more features, you'll discover the full power of systematic model benchmarking.

Happy benchmarking! 🚀</content>
<parameter name="filePath">GETTING_STARTED.md