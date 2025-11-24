# **Project: SentiCompare - Multi-LLM Sentiment Analysis Benchmark Framework**

## **Project Overview**

SentiCompare is an automated benchmarking framework that fine-tunes, evaluates, and compares multiple small language models (SLMs) for sentiment analysis tasks. The system leverages Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA with 4-bit quantization to enable efficient training and deployment on consumer-grade hardware while systematically measuring performance and inference speed across models.

> 🚀 **New to SentiCompare?** Start with our [Getting Started Guide](GETTING_STARTED.md) for a quick 5-minute setup!

***

## **Core Objectives**

1. **Efficient Fine-Tuning**: Implement LoRA/QLoRA fine-tuning with 4-bit quantization for memory-efficient training
2. **Multi-Model Comparison**: Systematically evaluate 5-7 small language models on identical datasets and metrics
3. **Performance Benchmarking**: Measure accuracy, F1-score, precision, recall on sentiment classification
4. **Speed Benchmarking**: Track inference latency (TTFT, median, P99), throughput (tokens/sec), and memory usage
5. **Automated Pipeline**: Create reproducible CI/CD-ready evaluation workflows
6. **Visualization Dashboard**: Generate comparative charts and performance reports

***

## **Architecture**

### **System Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    SentiCompare Framework                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  Data Pipeline  │  │  Training Engine │  │ Evaluation │ │
│  │                 │  │                  │  │   Engine   │ │
│  │ • Loader        │→ │ • LoRA/QLoRA    │→ │ • Metrics  │ │
│  │ • Preprocessor  │  │ • 4-bit Quant   │  │ • Speed    │ │
│  │ • Tokenizer     │  │ • Multi-GPU     │  │ • Memory   │ │
│  └─────────────────┘  └──────────────────┘  └────────────┘ │
│           ↓                      ↓                    ↓      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Benchmark Comparison Module               │    │
│  │  • Statistical Analysis  • Model Ranking            │    │
│  │  • Cost Estimation      • Performance Profiling     │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Visualization & Reporting                  │    │
│  │  • Interactive Dashboards  • Export Results         │    │
│  │  • Comparative Charts     • JSON/CSV Reports        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

***

## **Target Models for Comparison**

Based on your prior interest in small models, the framework will support:

1. **Phi-3-mini** (3.8B) - Microsoft's efficient reasoning model
2. **Gemma-2-2B** (2B) - Google's compact high-performer
3. **TinyLlama-1.1B** (1.1B) - Ultra-lightweight model
4. **Qwen2.5-1.5B** (1.5B) - Alibaba's multilingual model
5. **SmolLM2-1.7B** (1.7B) - Hugging Face's instruction-tuned model
6. **DistilBERT-base** (66M) - Baseline encoder model
7. **RoBERTa-base** (125M) - Strong sentiment analysis baseline

***

## **Technical Stack**

### **Core Libraries**
```python
# Fine-tuning & Model Management
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0

# Data & Evaluation
datasets>=2.16.0
evaluate>=0.4.1
scikit-learn>=1.3.0

# Performance Benchmarking
torch>=2.1.0
tensorboard>=2.15.0
mlflow>=2.9.0

# Visualization
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.29.0  # For dashboard

# Utilities
pandas>=2.1.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.66.0
```

### **Hardware Requirements**
- **Minimum**: 16GB RAM, 8GB VRAM GPU (e.g., RTX 3070)
- **Recommended**: 32GB RAM, 16-24GB VRAM GPU (e.g., RTX 4090, A100)
- **Your Setup**: Perfect - i9 with 64GB RAM + RTX 3080 (10GB VRAM)

***

## **Project Structure**

```
senticompare/
├── config/
│   ├── models.yaml           # Model configurations
│   ├── datasets.yaml         # Dataset settings
│   ├── training.yaml         # LoRA/QLoRA hyperparameters
│   └── evaluation.yaml       # Metrics & benchmarks
│
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Tokenized data
│   └── splits/               # Train/val/test splits
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py         # Dataset loading utilities
│   │   ├── preprocessor.py   # Text cleaning & tokenization
│   │   └── augmentation.py   # Optional data augmentation
│   │
│   ├── models/
│   │   ├── model_registry.py # Model definitions & configs
│   │   ├── lora_config.py    # LoRA hyperparameters
│   │   └── quantization.py   # 4-bit quantization setup
│   │
│   ├── training/
│   │   ├── trainer.py        # Training loop with LoRA
│   │   ├── optimizer.py      # Optimizer configurations
│   │   └── callbacks.py      # Early stopping, logging
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # Accuracy, F1, precision, recall
│   │   ├── speed_benchmark.py # Latency & throughput tests
│   │   └── memory_profiler.py # VRAM & RAM tracking
│   │
│   ├── comparison/
│   │   ├── aggregator.py     # Collect results across models
│   │   ├── statistical.py    # Statistical significance tests
│   │   └── ranker.py         # Model ranking logic
│   │
│   └── visualization/
│       ├── dashboard.py      # Streamlit interactive UI
│       ├── plots.py          # Chart generation
│       └── reports.py        # JSON/CSV export
│
├── experiments/
│   ├── logs/                 # Training logs
│   ├── checkpoints/          # Model checkpoints
│   └── results/              # Benchmark results
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── test_data.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── scripts/
│   ├── download_models.sh    # Fetch models from HF Hub
│   ├── prepare_data.sh       # Download & preprocess datasets
│   ├── train_all.sh          # Train all models sequentially
│   └── benchmark_all.sh      # Run full benchmark suite
│
├── requirements.txt
├── setup.py
├── README.md
├── .env.example              # API keys, paths
└── docker-compose.yml        # Optional containerization
```

***

## **Implementation Details**

### **1. Data Pipeline (`src/data/loader.py`)**

```python
from datasets import load_dataset
from transformers import AutoTokenizer

class SentimentDataLoader:
    """Load and prepare sentiment analysis datasets"""
    
    SUPPORTED_DATASETS = {
        'imdb': 'stanfordnlp/imdb',
        'sst2': 'stanfordnlp/sst2',
        'amazon': 'amazon_polarity',
        'yelp': 'yelp_polarity'
    }
    
    def __init__(self, dataset_name: str, tokenizer_name: str):
        self.dataset = load_dataset(self.SUPPORTED_DATASETS[dataset_name])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def preprocess(self, examples):
        """Tokenize text and prepare labels"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    def prepare_splits(self, train_size=10000, val_size=2000, test_size=5000):
        """Create balanced train/val/test splits"""
        # Implementation for stratified sampling
        pass
```

### **2. LoRA Configuration (`src/models/lora_config.py`)**

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig
import torch

class LoRAConfigManager:
    """Manage LoRA and quantization configurations"""
    
    @staticmethod
    def get_4bit_config():
        """4-bit quantization config for QLoRA"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    
    @staticmethod
    def get_lora_config(rank=8, alpha=16, target_modules=None):
        """LoRA configuration for sentiment classification"""
        if target_modules is None:
            # Target attention layers (q_proj, v_proj)
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS  # Sequence classification
        )
    
    @staticmethod
    def prepare_model(model_name: str, num_labels=2):
        """Load model with 4-bit quantization and LoRA"""
        from transformers import AutoModelForSequenceClassification
        
        # Load base model with quantization
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=LoRAConfigManager.get_4bit_config(),
            device_map="auto"
        )
        
        # Apply LoRA
        lora_config = LoRAConfigManager.get_lora_config()
        model = get_peft_model(model, lora_config)
        
        print(f"Trainable parameters: {model.print_trainable_parameters()}")
        return model
```

### **3. Training Engine (`src/training/trainer.py`)**

```python
from transformers import Trainer, TrainingArguments
import mlflow

class SentiCompareTrainer:
    """Custom trainer with MLflow logging"""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
    def get_training_args(self):
        """Optimized training arguments for LoRA fine-tuning"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            learning_rate=2e-4,
            fp16=True,  # Mixed precision
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="mlflow"
        )
    
    def train(self):
        """Execute training with MLflow tracking"""
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        with mlflow.start_run():
            mlflow.log_params({
                "model_name": self.model.config._name_or_path,
                "lora_rank": self.model.peft_config['default'].r,
                "lora_alpha": self.model.peft_config['default'].lora_alpha
            })
            
            trainer.train()
            
        return trainer
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Calculate accuracy, F1, precision, recall"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        preds = predictions.argmax(-1)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='binary'),
            'precision': precision_score(labels, preds, average='binary'),
            'recall': recall_score(labels, preds, average='binary')
        }
```

### **4. Speed Benchmarking (`src/evaluation/speed_benchmark.py`)**

```python
import time
import torch
from statistics import median

class SpeedBenchmark:
    """Measure inference performance metrics"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def measure_latency(self, texts, num_runs=100):
        """Measure TTFT, median latency, P99 latency"""
        latencies = []
        ttfts = []
        
        for text in texts[:num_runs]:
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            
            # Measure time to first token
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latency = (end - start) * 1000  # Convert to ms
            latencies.append(latency)
            ttfts.append(latency)  # For classification, TTFT ≈ total latency
        
        return {
            'ttft_mean': sum(ttfts) / len(ttfts),
            'latency_median': median(latencies),
            'latency_p99': sorted(latencies)[int(len(latencies) * 0.99)],
            'latency_mean': sum(latencies) / len(latencies)
        }
    
    def measure_throughput(self, texts, batch_size=16):
        """Measure tokens processed per second"""
        total_tokens = 0
        total_time = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            total_tokens += inputs['input_ids'].numel()
            total_time += (end - start)
        
        return {
            'throughput_tokens_per_sec': total_tokens / total_time,
            'throughput_samples_per_sec': len(texts) / total_time
        }
    
    def measure_memory(self):
        """Measure GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
        return {}
```

### **5. Comparison & Ranking (`src/comparison/aggregator.py`)**

```python
import pandas as pd
import json

class BenchmarkAggregator:
    """Aggregate and compare results across models"""
    
    def __init__(self, results_dir='experiments/results'):
        self.results_dir = results_dir
        self.results = []
    
    def load_results(self, model_name):
        """Load evaluation results for a model"""
        with open(f"{self.results_dir}/{model_name}_results.json", 'r') as f:
            return json.load(f)
    
    def aggregate_all(self, model_names):
        """Combine results from all models"""
        df_list = []
        
        for model_name in model_names:
            results = self.load_results(model_name)
            results['model_name'] = model_name
            df_list.append(pd.DataFrame([results]))
        
        return pd.concat(df_list, ignore_index=True)
    
    def rank_models(self, df, weights=None):
        """Rank models based on weighted scoring"""
        if weights is None:
            weights = {
                'accuracy': 0.3,
                'f1': 0.3,
                'throughput_tokens_per_sec': 0.2,
                'latency_median': -0.2  # Lower is better
            }
        
        # Normalize metrics to 0-1 scale
        df_norm = df.copy()
        for metric in weights.keys():
            if weights[metric] > 0:
                df_norm[f'{metric}_norm'] = (
                    (df[metric] - df[metric].min()) / 
                    (df[metric].max() - df[metric].min())
                )
            else:  # Inverse for latency (lower is better)
                df_norm[f'{metric}_norm'] = (
                    (df[metric].max() - df[metric]) / 
                    (df[metric].max() - df[metric].min())
                )
        
        # Calculate weighted score
        df_norm['composite_score'] = sum(
            df_norm[f'{metric}_norm'] * abs(weight)
            for metric, weight in weights.items()
        )
        
        return df_norm.sort_values('composite_score', ascending=False)
```

### **6. Visualization Dashboard (`src/visualization/dashboard.py`)**

```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class SentiCompareDashboard:
    """Interactive Streamlit dashboard for results"""
    
    def __init__(self, df):
        self.df = df
    
    def render(self):
        st.title("🎯 SentiCompare: Multi-LLM Sentiment Analysis Benchmark")
        
        # Model selection
        models = st.multiselect(
            "Select models to compare:",
            options=self.df['model_name'].tolist(),
            default=self.df['model_name'].tolist()
        )
        
        df_filtered = self.df[self.df['model_name'].isin(models)]
        
        # Performance metrics
        st.header("📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Accuracy", f"{df_filtered['accuracy'].max():.4f}")
        with col2:
            st.metric("Best F1", f"{df_filtered['f1'].max():.4f}")
        with col3:
            st.metric("Fastest (median)", f"{df_filtered['latency_median'].min():.2f}ms")
        with col4:
            st.metric("Highest Throughput", f"{df_filtered['throughput_tokens_per_sec'].max():.0f} tok/s")
        
        # Comparative charts
        st.header("📈 Model Comparison")
        
        # Accuracy vs Latency scatter
        fig1 = px.scatter(
            df_filtered,
            x='latency_median',
            y='accuracy',
            size='memory_allocated_mb',
            color='model_name',
            hover_data=['f1', 'throughput_tokens_per_sec'],
            title="Accuracy vs Latency Trade-off"
        )
        st.plotly_chart(fig1)
        
        # Multi-metric radar chart
        fig2 = go.Figure()
        for _, row in df_filtered.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row['accuracy'], row['f1'], row['precision'], row['recall']],
                theta=['Accuracy', 'F1', 'Precision', 'Recall'],
                name=row['model_name']
            ))
        fig2.update_layout(title="Multi-Metric Performance Comparison")
        st.plotly_chart(fig2)
        
        # Detailed table
        st.header("📋 Detailed Results")
        st.dataframe(df_filtered)
```

***

## **CLI Usage**

SentiCompare provides a comprehensive command-line interface for all operations:

### **Available Commands**

- **`senticompare train`** - Train a single model
- **`senticompare train-all`** - Train all models on specified dataset(s)
- **`senticompare evaluate`** - Evaluate a single trained model
- **`senticompare benchmark`** - Run full benchmark suite on trained models
- **`senticompare dashboard`** - Launch interactive Streamlit dashboard
- **`senticompare report`** - Generate comparison reports

### **Common Usage Patterns**

```bash
# Get help
senticompare --help
senticompare train --help

# Quick start: Train and evaluate a model
senticompare train --model DistilBERT-base --dataset imdb
senticompare evaluate --model DistilBERT-base --dataset imdb --checkpoint experiments/checkpoints/DistilBERT-base_imdb/final
senticompare dashboard

# Full pipeline: Train all models and benchmark
senticompare train-all --dataset imdb
senticompare benchmark --dataset imdb
senticompare report --format all
```

### **Device Configuration**

SentiCompare automatically detects your hardware:

```bash
# Auto-detect (recommended)
senticompare train --model DistilBERT-base --dataset imdb

# Force specific device
senticompare train --model DistilBERT-base --dataset imdb --device cuda
senticompare train --model DistilBERT-base --dataset imdb --device mps
senticompare train --model DistilBERT-base --dataset imdb --device cpu
```

***

## **Execution Workflow**

### **Step 1: Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/senticompare.git
cd senticompare

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .

# Download datasets and models (optional - will download on first use)
bash scripts/prepare_data.sh
bash scripts/download_models.sh
```

### **Step 2: Configuration**
Edit `config/models.yaml`:
```yaml
models:
  - name: "microsoft/Phi-3-mini-4k-instruct"
    alias: "Phi-3-mini"
    lora_rank: 8
    lora_alpha: 16
    target_modules: ["q_proj", "v_proj"]
  
  - name: "google/gemma-2-2b"
    alias: "Gemma-2-2B"
    lora_rank: 8
    lora_alpha: 16
    target_modules: ["q_proj", "v_proj"]
  
  # ... additional models
```

### **Step 3: Training**
```bash
# Train all models
senticompare train-all --dataset imdb

# Or train single model
senticompare train --model Phi-3-mini --dataset imdb

# Train with custom output directory
senticompare train --model DistilBERT-base --dataset imdb --output-dir experiments/custom_output
```

### **Step 4: Evaluation**
```bash
# Evaluate single model
senticompare evaluate --model DistilBERT-base --dataset imdb --checkpoint experiments/checkpoints/DistilBERT-base_imdb/final

# Run full benchmark suite
senticompare benchmark --models-dir experiments/checkpoints --dataset imdb
```

### **Step 5: Visualization**
```bash
# Launch interactive dashboard
senticompare dashboard

# Generate reports in multiple formats
senticompare report --results-dir experiments/results --format all
```

***

## **Expected Outputs**

### **1. Training Metrics**
- Training loss curves
- Validation accuracy/F1 over epochs
- Learning rate schedules
- GPU utilization logs

### **2. Performance Benchmarks**
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
  "training_time_hours": 1.2,
  "inference_cost_per_1m_tokens": 0.032
}
```

### **3. Comparative Visualizations**
- Accuracy vs Latency scatter plots
- Multi-metric radar charts
- Throughput comparisons
- Memory usage bar charts
- Cost efficiency analysis

### **4. Final Ranking**
| Rank | Model | Accuracy | F1 | Latency (ms) | Throughput (tok/s) | Memory (MB) | Score |
|------|-------|----------|----|--------------|--------------------|-------------|-------|
| 1 | Gemma-2-2B | 0.928 | 0.922 | 38.2 | 2145 | 1890 | 0.89 |
| 2 | Phi-3-mini | 0.925 | 0.920 | 45.1 | 1847 | 2345 | 0.86 |
| 3 | SmolLM2-1.7B | 0.918 | 0.911 | 32.5 | 2567 | 1456 | 0.85 |

***

## **Key Features**

✅ **Memory Efficient**: 4-bit quantization enables training on 10GB VRAM  
✅ **Fast Training**: LoRA reduces training time by 60-80% vs full fine-tuning  
✅ **Reproducible**: Containerized with Docker, seeded experiments  
✅ **Extensible**: Easy to add new models, datasets, metrics  
✅ **Production-Ready**: Export optimized models for deployment  
✅ **CI/CD Integration**: Automated testing with pytest, GitHub Actions ready  

***

## **Timeline Estimate**

- **Week 1**: Setup infrastructure, data pipeline, model configs
- **Week 2**: Implement LoRA training, basic evaluation
- **Week 3**: Speed benchmarking, memory profiling
- **Week 4**: Comparison logic, visualization dashboard
- **Week 5**: Testing, documentation, refinement

***

## **Troubleshooting Guide**

### **Common Issues**

#### **CUDA/Memory Issues**
```bash
# Check device availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU memory
nvidia-smi

# Use smaller batch size for memory issues
senticompare train --model TinyLlama-1.1B --dataset imdb  # Start with smaller models
```

#### **MPS (Apple Silicon) Issues**
```bash
# MPS doesn't support quantization - use full precision LoRA
senticompare train --model DistilBERT-base --dataset imdb --device mps

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### **Dataset Download Issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets
senticompare train --model DistilBERT-base --dataset imdb
```

#### **Model Download Issues**
```bash
# Use different mirror or check network
export HF_ENDPOINT=https://hf-mirror.com
senticompare train --model DistilBERT-base --dataset imdb
```

#### **Training Instability**
```bash
# Reduce learning rate
# Check model configuration in config/models.yaml
# Try different batch size or gradient accumulation
```

### **Performance Optimization**

#### **Memory Efficiency**
- Use 4-bit quantization (CUDA only)
- Enable gradient checkpointing
- Reduce batch size
- Use smaller models for testing

#### **Training Speed**
- Use mixed precision (fp16) on CUDA
- Increase batch size if memory allows
- Use gradient accumulation
- Enable data loading optimizations

### **Getting Help**

```bash
# Check CLI help
senticompare --help
senticompare train --help

# Verbose logging
senticompare train --model DistilBERT-base --dataset imdb --verbose

# Check logs
tail -f experiments/logs/*.log
```

### **System Requirements**

- **Minimum**: 16GB RAM, modern CPU
- **Recommended for CUDA**: 32GB RAM, 8GB+ VRAM GPU
- **Recommended for MPS**: 16GB RAM, Apple Silicon Mac
- **Storage**: 50GB+ for datasets and models

***

This project would provide you with a comprehensive framework to systematically evaluate different SLMs for sentiment analysis while maintaining efficiency through LoRA and quantization—perfectly suited for your hardware setup and interest in both performance benchmarking and practical deployment.

