# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MoodBench** is a multi-LLM sentiment analysis benchmark framework that fine-tunes, evaluates, and compares small language models (SLMs) using Parameter-Efficient Fine-Tuning (PEFT) techniques (LoRA/QLoRA) with 4-bit quantization. The framework systematically measures both performance metrics (accuracy, F1, precision, recall) and speed metrics (latency, throughput, memory usage) across multiple models.

## Package Management

This project uses **uv** as the package manager (not pip or poetry).

### Common Commands

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run the main script
uv run python main.py

# Run a specific module
uv run python -m src.training.trainer

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Python Version
- Required: Python 3.12 (specified in `.python-version`)

## Architecture

### High-Level System Flow

The framework follows a modular pipeline architecture:

```
Data Pipeline → Training Engine → Evaluation Engine → Comparison Module → Visualization
```

1. **Data Pipeline** (`src/data/`): Loads datasets (Disney reviews, IMDB, SST2, Amazon, Yelp), tokenizes, and creates balanced splits
2. **Training Engine** (`src/training/`): Applies LoRA/QLoRA with 4-bit quantization to train models efficiently
3. **Evaluation Engine** (`src/evaluation/`): Measures performance metrics and speed benchmarks
4. **Comparison Module** (`src/comparison/`): Aggregates results, performs statistical analysis, ranks models
5. **Visualization** (`src/visualization/`): Gradio dashboard and static reports

### Key Technical Decisions

#### LoRA/QLoRA Fine-Tuning
- Uses **4-bit quantization** (NF4) with double quantization for memory efficiency on CUDA GPUs
- **MPS Support**: On Apple Silicon, uses full precision with LoRA (bitsandbytes not supported on MPS)
- Target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]` (attention layers)
- Default LoRA rank: 8, alpha: 16
- Enables training on 10GB VRAM GPUs (e.g., RTX 3080) or Apple Silicon with 16GB+ unified memory

#### Model Registry

The framework supports **18 models** ranging from 4M to 3.8B parameters, optimized for fast benchmarking:

**Ultra-Tiny Models (4M-30M) - Fastest:**
- BERT-tiny (4M), BERT-mini (11M), ELECTRA-small (14M)
- BERT-small (29M), MiniLM-L12 (33M)

**Tiny Models (60M-170M) - Production Baselines:**
- DistilBERT-base (66M), Pythia-70m (70M), DistilRoBERTa (82M)
- DeBERTa-v3-small (86M), BERT-base (110M), GPT2-small (124M)
- RoBERTa-base (125M), Pythia-160m (160M), Gemma-2-2B (270M)

**Small LLMs (1B-4B) - Research Quality:**
- TinyLlama-1.1B (1.1B), Qwen2.5-1.5B (1.5B), SmolLM2-1.7B (1.7B), Phi-3-mini (3.8B)

**Documentation:**
- Full model guide: `docs/MODEL_CONFIGURATION.md`
- Quick reference: `docs/QUICK_REFERENCE.md`
- Configuration: `config/models.yaml`

#### Benchmarking Approach
- **Performance Metrics**: Accuracy, F1-score, Precision, Recall (via sklearn)
- **Speed Metrics**:
  - Latency: TTFT (time to first token), median, P99
  - Throughput: tokens/second, samples/second
  - Memory: GPU VRAM allocation and reservation
- **Comparison**: Weighted composite scoring with normalization

### Core Libraries

**Fine-tuning stack:**
- `transformers` - Model loading and training
- `peft` - LoRA/QLoRA implementation
- `bitsandbytes` - 4-bit quantization (CUDA only, not supported on MPS)
- `accelerate` - Multi-GPU and device management support

**Data & Evaluation:**
- `datasets` - Dataset loading (Hugging Face Hub)
- `kagglehub` - Disney reviews dataset
- `scikit-learn` - Metrics computation
- `evaluate` - Additional evaluation utilities

**Experiment tracking:**
- `mlflow` - Training metrics logging
- `tensorboard` - Loss curves and visualizations

**Visualization:**
- `gradio` - Interactive dashboard
- `plotly` - Interactive charts
- `matplotlib`/`seaborn` - Static plots

## Project Structure

```
src/
├── data/              # Dataset loading, preprocessing, augmentation
├── models/            # Model registry, LoRA configs, quantization setup
├── training/          # Training loop, optimizers, callbacks
├── evaluation/        # Metrics, speed benchmarking, memory profiling
├── comparison/        # Result aggregation, statistical tests, ranking
└── visualization/     # Gradio dashboard, plots, reports

config/                # YAML configs for models, datasets, training, evaluation
experiments/           # Training logs, checkpoints, benchmark results
scripts/               # Shell scripts for downloading models/data, training, benchmarking
```

## Development Workflow

### Training Workflow
1. Load dataset using `src/data/loader.py`
2. Configure LoRA and quantization via `src/models/lora_config.py`
3. Train with `src/training/trainer.py` (uses Hugging Face `Trainer` with MLflow logging)
4. Checkpoints saved to `experiments/checkpoints/`

### Evaluation Workflow
1. Load trained model checkpoint
2. Run performance metrics via `src/evaluation/metrics.py`
3. Run speed benchmarks via `src/evaluation/speed_benchmark.py`
4. Measure memory usage via `src/evaluation/memory_profiler.py`
5. Save results to `experiments/results/{model_name}_results.json`

### Comparison Workflow
1. Aggregate all model results using `src/comparison/aggregator.py`
2. Normalize metrics and compute composite scores
3. Rank models based on weighted criteria (default: 30% accuracy, 30% F1, 20% throughput, 20% latency)
4. Generate statistical significance tests via `src/comparison/statistical.py`

### Visualization
- Launch dashboard: `python gradio_app.py` (Modular UI with training, benchmarking, analysis, NPS, and methodology tabs)
- Key visualizations:
  - Analysis tab: Accuracy vs Latency scatter plot, F1 score charts, latency distributions
  - NPS tab: Customer vs model NPS comparisons, category breakdowns, actual value markers
  - Benchmarking tab: Real-time progress tracking and results
  - Training tab: Model status matrix and progress monitoring

## Important Patterns

### Device Detection
The framework automatically detects available hardware:

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### LoRA Model Preparation
Always use the `LoRAConfigManager.prepare_model()` pattern to ensure proper quantization (CUDA) or full precision (MPS) and LoRA application:

```python
from src.models.lora_config import LoRAConfigManager

model = LoRAConfigManager.prepare_model(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    num_labels=2,  # Binary sentiment
    device=get_device()  # Automatically selects quantization strategy
)
```

**Note**: On MPS devices, 4-bit quantization is not available. The framework will use full precision (float32) or half precision (float16) with LoRA instead.

### Training Arguments
Standard training configuration uses:
- Mixed precision (fp16=True on CUDA, bf16=False on MPS due to stability issues)
- Gradient accumulation (4 steps)
- Learning rate: 2e-4
- Evaluation strategy: "steps" with eval_steps=500
- Best model selection based on F1 score

**MPS Considerations**: Apple Silicon GPUs require slightly different training arguments. The framework automatically adjusts based on device type.

### Speed Benchmarking
Always synchronize device operations before/after inference to get accurate timing:

```python
# CUDA
if device.type == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
# ... inference ...
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.perf_counter()
```

**Note**: MPS does not have an equivalent to `cuda.synchronize()`. Use `torch.mps.synchronize()` if available, or rely on CPU synchronization for timing.

### Dataset Support
Use `SentimentDataLoader.SUPPORTED_DATASETS` mapping:
- `'imdb'` → "stanfordnlp/imdb"
- `'sst2'` → "stanfordnlp/sst2"
- `'amazon'` → "amazon_polarity"
- `'yelp'` → "yelp_polarity"

## Configuration Files

Configuration is managed via YAML files in `config/`:

- `models.yaml` - Model names, aliases, LoRA hyperparameters, target modules
- `datasets.yaml` - Dataset settings, split sizes, preprocessing options
- `training.yaml` - Training hyperparameters (epochs, batch size, learning rate, etc.)
- `evaluation.yaml` - Metrics to compute, benchmarking parameters

## Recommended Workflows

### Quick Start: Ultra-Fast Benchmark (1 hour)
For rapid experimentation and CI/CD testing:
```bash
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small ELECTRA-small MiniLM-L12
```
- 5 ultra-tiny models (4M-33M parameters)
- Training time: ~1 hour total
- Memory: <2GB
- Best for initial validation

### Production: Encoder Comparison (2-3 hours)
For comprehensive encoder architecture study:
```bash
uv run moodbench train-all --dataset amazon --device=mps \
  --models DistilBERT-base DistilRoBERTa DeBERTa-v3-small BERT-base RoBERTa-base
```
- 5 proven production models (66M-125M)
- Training time: ~2-3 hours
- Memory: <3GB

### Research: Full Tiny Benchmark (4-5 hours)
For comprehensive lightweight model comparison:
```bash
uv run moodbench train-all --dataset amazon --device=mps \
  --models BERT-tiny BERT-mini BERT-small DistilBERT-base DistilRoBERTa \
           DeBERTa-v3-small BERT-base RoBERTa-base GPT2-small Pythia-70m
```
- 10 models (4M-160M)
- Training time: ~4-5 hours
- Memory: <4GB

## Hardware Considerations

### CUDA (NVIDIA GPUs)
- **Minimum**: 16GB RAM, 8GB VRAM GPU (e.g., RTX 3070)
- **Recommended**: 32GB RAM, 16-24GB VRAM GPU (e.g., RTX 4090, A100)
- Development hardware: i9 CPU, 64GB RAM, RTX 3080 (10GB VRAM)
- 4-bit quantization + LoRA enables training on 10GB VRAM

### MPS (Apple Silicon)
- **Minimum**: M1 with 16GB unified memory (ultra-tiny models only)
- **Recommended**: M2/M3 Pro/Max with 32GB+ unified memory (all tiny models)
- **Optimal**: M3 Max with 64GB+ unified memory (all models including LLMs)
- No 4-bit quantization support (bitsandbytes incompatible)
- Uses full precision (FP32) with LoRA
- Memory pressure managed via unified memory architecture
- **Automatic optimizations:**
  - Dynamic batch sizing based on model size
  - Gradient checkpointing for models ≥1B parameters
  - Gradient accumulation to maintain effective batch size
  - Pin memory disabled (not applicable to unified memory)

### Device-Specific Features
| Feature | CUDA | MPS | CPU |
|---------|------|-----|-----|
| 4-bit Quantization | ✅ Yes | ❌ No | ❌ No |
| 8-bit Quantization | ✅ Yes | ❌ No | ❌ No |
| Mixed Precision (fp16) | ✅ Yes | ⚠️ Limited | ❌ No |
| LoRA Fine-tuning | ✅ Yes | ✅ Yes | ✅ Yes |
| Gradient Checkpointing | ✅ Yes | ✅ Yes | ✅ Yes |

## Experiment Tracking

- Training runs logged to MLflow
- Key logged parameters: model_name, lora_rank, lora_alpha, learning_rate
- Key logged metrics: accuracy, f1, precision, recall, loss
- Results saved as JSON in `experiments/results/`

## Result Format

Each model's benchmark results follow this structure:

```json
{
  "model_name": "Phi-3-mini",
  "accuracy": 0.9245,
  "f1": 0.9198,
  "precision": 0.9156,
  "recall": 0.9241,
  "ttft_mean_ms": 42.3,
  "latency_median_ms": 45.1,
  "latency_p99_ms": 89.7,
  "throughput_tokens_per_sec": 1847,
  "memory_allocated_mb": 2345,
  "training_time_hours": 1.2
}
```
