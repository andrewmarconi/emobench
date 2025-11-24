# EmoBench Implementation Plan

## Project Status
Currently implemented:
- Basic project structure with uv package manager
- Minimal dependencies (huggingface, kagglehub)
- Comprehensive design documentation (README.md)
- Python 3.12 environment

## Implementation Roadmap

### Phase 1: Foundation & Infrastructure (Week 1)

#### 1.1 Project Structure Setup
- [ ] Create directory structure:
  - `config/` - YAML configuration files
  - `src/` - Source code modules
  - `data/` - Dataset storage (raw/, processed/, splits/)
  - `experiments/` - Training artifacts (logs/, checkpoints/, results/)
  - `scripts/` - Shell scripts for automation
  - `tests/` - Unit and integration tests
  - `notebooks/` - Jupyter notebooks for exploration

#### 1.2 Dependencies & Environment
- [ ] Update pyproject.toml with all required dependencies:
  - Fine-tuning: transformers, peft, bitsandbytes (optional for CUDA), accelerate
  - Data: datasets, evaluate, scikit-learn
  - Benchmarking: torch, tensorboard, mlflow
  - Visualization: plotly, matplotlib, seaborn, streamlit
  - Utilities: pandas, numpy, pyyaml, tqdm
- [ ] Run `uv sync` to install dependencies
- [ ] Create `.env.example` for environment variables (HF_TOKEN, MLFLOW_URI)
- [ ] Add device detection script (CUDA/MPS/CPU)
- [ ] Make bitsandbytes optional (only for CUDA)

#### 1.3 Configuration Files
- [ ] Create `config/models.yaml` - Model definitions and LoRA hyperparameters
- [ ] Create `config/datasets.yaml` - Dataset settings and split configurations
- [ ] Create `config/training.yaml` - Training hyperparameters
- [ ] Create `config/evaluation.yaml` - Benchmark metrics and parameters

---

### Phase 2: Data Pipeline (Week 1-2)

#### 2.1 Data Loading (`src/data/loader.py`)
- [ ] Implement `SentimentDataLoader` class
  - [ ] Support for Disney reviews (via KaggleHub)
  - [ ] Support for IMDB dataset
  - [ ] Support for SST-2 dataset
  - [ ] Support for Amazon Polarity
  - [ ] Support for Yelp Polarity
- [ ] Add dataset caching mechanism
- [ ] Implement data validation

#### 2.2 Preprocessing (`src/data/preprocessor.py`)
- [ ] Implement tokenization with max_length=512
- [ ] Add text cleaning utilities
- [ ] Handle label mapping for different datasets
- [ ] Implement stratified train/val/test splitting
  - Default: 10k train, 2k val, 5k test
- [ ] Add data augmentation utilities (optional)

#### 2.3 Data Scripts
- [ ] Create `scripts/download_models.sh` - Download models from HF Hub
- [ ] Create `scripts/prepare_data.sh` - Download and preprocess datasets
- [ ] Add data exploration notebook (`notebooks/01_data_exploration.ipynb`)

#### 2.4 Testing
- [ ] Write `tests/test_data.py` - Unit tests for data loading and preprocessing

---

### Phase 3: Model Configuration & LoRA Setup (Week 2)

#### 3.0 Device Utilities (`src/utils/device.py`)
- [ ] Implement device detection utilities:
  - [ ] `get_device()` - Returns "cuda", "mps", or "cpu"
  - [ ] `get_device_name()` - Returns human-readable device name
  - [ ] `get_device_memory()` - Returns available memory
  - [ ] `supports_quantization(device)` - Check if 4-bit quantization supported
- [ ] Add device synchronization utilities:
  - [ ] `sync_device(device)` - Device-aware synchronization wrapper
- [ ] Create device compatibility matrix
- [ ] Add logging for device selection and capabilities

#### 3.1 Model Registry (`src/models/model_registry.py`)
- [ ] Define model configurations for all 7 target models:
  - Phi-3-mini (3.8B)
  - Gemma-2-2B (2B)
  - TinyLlama-1.1B (1.1B)
  - Qwen2.5-1.5B (1.5B)
  - SmolLM2-1.7B (1.7B)
  - DistilBERT-base (66M)
  - RoBERTa-base (125M)
- [ ] Add model metadata (size, architecture type)

#### 3.2 LoRA Configuration (`src/models/lora_config.py`)
- [ ] Implement `LoRAConfigManager` class
  - [ ] `get_4bit_config()` - BitsAndBytes quantization setup (CUDA only)
  - [ ] `get_lora_config()` - LoRA hyperparameters
  - [ ] `prepare_model()` - Load model with device-aware quantization + LoRA
  - [ ] Auto-detect device (CUDA/MPS/CPU) and adjust strategy
- [ ] Add support for model-specific target modules
- [ ] Implement trainable parameters counter
- [ ] Add MPS-specific model loading (full precision or fp16 with LoRA)
- [ ] Handle bitsandbytes import gracefully when unavailable

#### 3.3 Quantization (`src/models/quantization.py`)
- [ ] 4-bit NF4 quantization configuration (CUDA only)
- [ ] Double quantization support (CUDA only)
- [ ] bfloat16 compute dtype setup
- [ ] Add quantization verification utilities
- [ ] Implement device detection: CUDA vs MPS vs CPU
- [ ] Add fallback strategy for MPS (no quantization, LoRA only)
- [ ] Create utility: `get_optimal_dtype(device)` returns fp16/fp32 based on device

---

### Phase 4: Training Engine (Week 2-3)

#### 4.1 Trainer Implementation (`src/training/trainer.py`)
- [ ] Implement `EmoBenchTrainer` class
  - [ ] Wrap Hugging Face `Trainer` with custom logic
  - [ ] Configure device-aware `TrainingArguments`:
    - CUDA: fp16=True, bf16=False
    - MPS: fp16=False, bf16=False (use fp32 for stability)
    - CPU: fp16=False, bf16=False
  - [ ] Implement `compute_metrics()` for accuracy/F1/precision/recall
  - [ ] Add MLflow integration for experiment tracking
- [ ] Add early stopping callback
- [ ] Implement learning rate scheduler
- [ ] Add checkpoint saving strategy
- [ ] Add device type to logged metadata

#### 4.2 Optimizer Configuration (`src/training/optimizer.py`)
- [ ] Configure AdamW optimizer
- [ ] Add warmup steps (500)
- [ ] Set learning rate (2e-4 for LoRA)

#### 4.3 Callbacks (`src/training/callbacks.py`)
- [ ] Early stopping callback
- [ ] Logging callback for TensorBoard/MLflow
- [ ] Best model checkpoint callback

#### 4.4 Training Scripts
- [ ] Create `scripts/train_all.sh` - Train all models sequentially
- [ ] Add CLI arguments for single model training
- [ ] Implement training resumption from checkpoints

#### 4.5 Testing
- [ ] Write `tests/test_training.py` - Test training loop with tiny model

---

### Phase 5: Evaluation Engine (Week 3)

#### 5.1 Performance Metrics (`src/evaluation/metrics.py`)
- [ ] Implement classification metrics:
  - [ ] Accuracy
  - [ ] F1-score (binary)
  - [ ] Precision
  - [ ] Recall
- [ ] Add confusion matrix generation
- [ ] Implement per-class metrics for multi-class scenarios

#### 5.2 Speed Benchmarking (`src/evaluation/speed_benchmark.py`)
- [ ] Implement `SpeedBenchmark` class
  - [ ] `measure_latency()` - TTFT, median, P99
  - [ ] `measure_throughput()` - Tokens/sec, samples/sec
  - [ ] `measure_memory()` - Device memory usage (CUDA VRAM or MPS unified memory)
- [ ] Add device-aware synchronization:
  - CUDA: `torch.cuda.synchronize()`
  - MPS: `torch.mps.synchronize()` if available, else use CPU timing
  - CPU: No synchronization needed
- [ ] Implement batch processing for throughput tests
- [ ] Add warmup runs to stabilize measurements
- [ ] Track device type in benchmark results

#### 5.3 Memory Profiling (`src/evaluation/memory_profiler.py`)
- [ ] Track device memory allocation:
  - CUDA: `torch.cuda.memory_allocated()`
  - MPS: `torch.mps.current_allocated_memory()` if available
  - CPU: Use psutil for RAM tracking
- [ ] Track device memory reservation (CUDA only)
- [ ] Track peak memory usage
- [ ] Add unified memory tracking for Apple Silicon
- [ ] Handle different memory APIs per device type

#### 5.4 Evaluation Scripts
- [ ] Create `scripts/benchmark_all.sh` - Run full benchmark suite
- [ ] Add JSON output for results
- [ ] Implement batch evaluation for multiple checkpoints

#### 5.5 Testing
- [ ] Write `tests/test_evaluation.py` - Test metrics and benchmarking

---

### Phase 6: Comparison & Ranking (Week 3-4)

#### 6.1 Result Aggregation (`src/comparison/aggregator.py`)
- [ ] Implement `BenchmarkAggregator` class
  - [ ] `load_results()` - Load individual model results
  - [ ] `aggregate_all()` - Combine results into DataFrame
  - [ ] `rank_models()` - Weighted composite scoring
- [ ] Implement metric normalization (0-1 scale)
- [ ] Add configurable ranking weights

#### 6.2 Statistical Analysis (`src/comparison/statistical.py`)
- [ ] Implement statistical significance tests
  - [ ] Paired t-tests for accuracy comparisons
  - [ ] Confidence intervals
  - [ ] Effect size calculations
- [ ] Add bootstrapping for robust comparisons

#### 6.3 Model Ranking (`src/comparison/ranker.py`)
- [ ] Implement composite scoring algorithm
- [ ] Add Pareto frontier analysis (accuracy vs speed)
- [ ] Generate ranking reports

---

### Phase 7: Visualization & Reporting (Week 4)

#### 7.1 Dashboard (Gradio App)
- [x] Implement simplified Gradio dashboard
  - [x] Results summary display
  - [x] Accuracy vs Latency scatter plot
  - [x] Detailed results table
- [x] Integrated into main Gradio web UI

#### 7.2 Plot Generation (`src/visualization/plots.py`)
- [ ] Create Plotly chart functions:
  - [ ] Scatter plots (accuracy vs latency)
  - [ ] Radar charts (multi-metric)
  - [ ] Bar charts (throughput, memory)
  - [ ] Line plots (training curves)
- [ ] Add customizable styling

#### 7.3 Report Generation (`src/visualization/reports.py`)
- [ ] Generate JSON summary reports
- [ ] Generate CSV exports for Excel
- [ ] Create markdown reports
- [ ] Add PDF export (optional)

#### 7.4 Notebooks
- [ ] Create `notebooks/02_model_comparison.ipynb` - Compare trained models
- [ ] Create `notebooks/03_results_analysis.ipynb` - Deep dive into results

---

### Phase 8: End-to-End Integration & Testing (Week 4-5)

#### 8.1 Integration Testing
- [ ] Test full pipeline: data → training → evaluation → comparison
- [ ] Verify MLflow tracking works end-to-end
- [ ] Test with at least 2 models (e.g., TinyLlama, DistilBERT)

#### 8.2 CLI Interface
- [ ] Create main CLI entry point (`src/cli.py`)
- [ ] Add commands:
  - `train` - Train single model
  - `train-all` - Train all models
  - `evaluate` - Evaluate single model
  - `benchmark` - Run benchmark on selected models and datasets
  - `report` - Generate reports

#### 8.3 Documentation
- [ ] Update main.py to use CLI
- [ ] Add docstrings to all classes and functions
- [ ] Create usage examples in README
- [ ] Add troubleshooting guide

---

### Phase 9: Optimization & Refinement (Week 5)

#### 9.1 Performance Optimization
- [ ] Profile training loop for bottlenecks
- [ ] Optimize data loading with multiprocessing
- [ ] Add gradient checkpointing for memory efficiency
- [ ] Implement mixed precision training

#### 9.2 Error Handling
- [ ] Add robust error handling throughout codebase
- [ ] Implement checkpoint recovery for failed training
- [ ] Add validation for configuration files
- [ ] Improve error messages

#### 9.3 Code Quality
- [ ] Add type hints throughout codebase
- [ ] Run linting (ruff or pylint)
- [ ] Format code consistently (black)
- [ ] Add pre-commit hooks

---

### Phase 10: Production Readiness (Optional)

#### 10.1 Containerization
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml for multi-service setup
- [ ] Add CUDA-enabled base image

#### 10.2 CI/CD
- [ ] Set up GitHub Actions for testing
- [ ] Add automated linting and formatting checks
- [ ] Implement automated benchmarking on PR

#### 10.3 Model Deployment
- [ ] Export optimized models (ONNX or TorchScript)
- [ ] Create inference API (FastAPI)
- [ ] Add model serving documentation

---

## Critical Path Items

### Minimum Viable Product (MVP)
1. Complete Phases 1-3 (Infrastructure, Data, Models)
2. Implement basic training for 1-2 models
3. Add simple evaluation metrics
4. Generate basic comparison report

### Full Feature Set
Complete all phases 1-9 for production-ready framework

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation | 2-3 days | None |
| Phase 2: Data Pipeline | 2-3 days | Phase 1 |
| Phase 3: Model Config | 2-3 days | Phase 1 |
| Phase 4: Training | 3-4 days | Phases 2, 3 |
| Phase 5: Evaluation | 3-4 days | Phase 4 |
| Phase 6: Comparison | 2-3 days | Phase 5 |
| Phase 7: Visualization | 2-3 days | Phase 6 |
| Phase 8: Integration | 2-3 days | All previous |
| Phase 9: Refinement | 3-5 days | Phase 8 |
| Phase 10: Production | 3-5 days | Phase 9 |

**Total Estimated Time**: 4-6 weeks

---

## Risk Mitigation

### Technical Risks
- **GPU Memory Issues**:
  - CUDA: Use 4-bit quantization, reduce batch size, enable gradient checkpointing
  - MPS: Reduce batch size, use gradient checkpointing, consider smaller models first
  - Monitor unified memory pressure on Apple Silicon
- **Training Instability**: Implement gradient clipping, warmup steps, early stopping
- **Dataset Access**: Cache datasets locally, handle download failures gracefully
- **Model Download Failures**: Implement retry logic, use HF Hub mirrors
- **MPS Compatibility**: bitsandbytes not supported on MPS, must use full precision LoRA
- **Device-Specific Bugs**: Test on both CUDA and MPS, have fallback to CPU

### Validation Strategy
- Start with smallest models (DistilBERT, TinyLlama) to validate pipeline
- Test on both CUDA and MPS if available
- Test each component in isolation before integration
- Use small dataset subsets for rapid iteration
- Checkpoint frequently to avoid losing progress
- Validate device detection and fallback mechanisms early

---

## Success Metrics

### Phase Completion
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Code reviewed and refactored

### Final Deliverables
1. Fully functional training pipeline for 7 models
2. Comprehensive benchmark suite with speed + performance metrics
3. Interactive Gradio dashboard
4. Detailed comparison reports (JSON, CSV, visualizations)
5. Complete test coverage (>80%)
6. Production-ready documentation

---

## MPS-Specific Implementation Notes

### Key Differences from CUDA
1. **No Quantization Support**:
   - bitsandbytes library does not support MPS
   - Must use full precision (float32) or half precision (float16) with LoRA
   - Larger memory footprint compared to CUDA with 4-bit quantization

2. **Training Configuration**:
   - Disable fp16/bf16 for stability (use float32)
   - May need smaller batch sizes due to higher memory usage
   - Gradient checkpointing becomes more critical

3. **Synchronization**:
   - No direct equivalent to `torch.cuda.synchronize()`
   - Use `torch.mps.synchronize()` if available (PyTorch 2.0+)
   - Fallback to CPU synchronization for timing

4. **Memory Management**:
   - Unified memory architecture (shared between CPU and GPU)
   - Use `torch.mps.current_allocated_memory()` for memory tracking
   - No separate VRAM allocation/reservation APIs

5. **Model Selection**:
   - Prioritize smaller models on MPS (DistilBERT, TinyLlama, Qwen2.5-1.5B)
   - Larger models (Phi-3-mini, Gemma-2-2B) require 32GB+ unified memory

### Implementation Strategy for MPS
```python
# Example device-aware model loading
def load_model(model_name, device):
    if device == "cuda":
        # Use 4-bit quantization
        model = load_with_quantization(model_name)
    elif device == "mps":
        # Use full precision with LoRA
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # or torch.float16
            device_map="auto"
        )
        model = apply_lora(model)
    else:
        # CPU fallback
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = apply_lora(model)
    return model
```

### Testing Requirements
- [ ] Validate on macOS with Apple Silicon (M1/M2/M3)
- [ ] Test training with DistilBERT and TinyLlama on MPS
- [ ] Compare training speed and memory usage: CUDA vs MPS
- [ ] Document memory requirements per model on MPS
- [ ] Add MPS-specific troubleshooting guide

---

## Next Steps

1. **Start with Phase 1.1**: Create directory structure
2. **Update dependencies**: Add all required libraries to pyproject.toml (make bitsandbytes optional)
3. **Implement device detection**: Create `src/utils/device.py` first
4. **Create configuration files**: Set up YAML configs for models, datasets, training
5. **Begin Phase 2**: Implement data loading for Disney reviews dataset
