# Tech Stack

## Build System

- **Build backend**: setuptools (legacy backend)
- **Python version**: >=3.11

## Core Dependencies

- **numpy** (>=1.26): Numerical operations, array handling
- **scikit-learn** (>=1.4): Machine learning utilities
- **nilearn** (>=0.10): Neuroimaging data processing
- **tribev2**: Brain encoding model (install from GitHub: `pip install git+https://github.com/facebookresearch/tribev2.git`)

## Testing & Development

- **pytest** (>=8.0): Test runner
- **hypothesis** (>=6.100): Property-based testing framework

## Common Commands

### Installation
```bash
# Install package with dependencies
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install TRIBE v2 (not on PyPI)
pip install git+https://github.com/facebookresearch/tribev2.git
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_facade.py

# Run with verbose output
pytest -v

# Run property-based tests (may take longer)
pytest tests/ -v
```

### Development
```bash
# Check for syntax/type errors (if using type checker)
# Note: No mypy/pyright configured in this project

# Clean build artifacts
rm -rf build/ dist/ *.egg-info __pycache__/ .pytest_cache/
```

## Key Technical Notes

- **Cortical response shape**: (n_timesteps, 20484) float32 — only response the public TRIBE v2 checkpoint produces; subcortical variant is not released
- **Collapsed response shape**: (20484,) float32
- **ICA projection matrix**: (2048, 20484) for 5 networks
- **Cache format**: NumPy .npz files for raw/collapsed responses
- **Hypothesis database**: Stored in `.hypothesis/` for test case replay
