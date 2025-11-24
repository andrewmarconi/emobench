# AGENTS.md

## Commands

**Package Management (uv):**
```bash
uv sync                    # Install dependencies
uv add <package>           # Add dependency
uv run python main.py       # Run main script
```

**Development:**
```bash
uv run ruff check .         # Lint code
uv run black .             # Format code
uv run pytest              # Run all tests
uv run pytest tests/test_data.py::TestClass::test_function  # Run single test
uv run pytest -k "test_name"  # Run tests by name pattern
uv run pytest --cov=src    # Run with coverage
```

**Training Scripts:**
```bash
./scripts/train_model.sh <model> <dataset>  # Train single model
./scripts/train_all.sh [dataset]           # Train all models
./scripts/evaluate_all.sh                  # Run full evaluation
```

## Code Style

**Formatting:** Black (100 char line length), Ruff for linting
**Types:** Full type hints required (Python 3.12+)
**Imports:** Group stdlib, third-party, local imports; use `from typing import`
**Naming:** snake_case for functions/variables, PascalCase for classes
**Error Handling:** Log errors with `logger.error()` and raise descriptive exceptions
**Device Detection:** Always use `src.utils.device.get_device()` for hardware compatibility
**LoRA Models:** Use `LoRAConfigManager.prepare_model()` pattern for proper quantization
**Configs:** Load YAML configs from `config/` directory with fallback defaults
**Logging:** Use `logging.getLogger(__name__)` for module-level loggers
**Docstrings:** Google-style docstrings with Args/Returns sections