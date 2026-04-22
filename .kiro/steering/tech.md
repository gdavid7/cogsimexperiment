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

### End-to-end validation (requires HF access + torch)
```bash
# After syncing the Colab-produced cache to local, run:
python scripts/validate_ibc.py --cache-dir /path/to/cognitive-similarity-cache
# Downloads best.ckpt from HuggingFace (~676 MB) on first run,
# runs FastICA once (cached to <cache>/ica_masks.npz), then evaluates
# ValidationSuite against the materialized collapsed tensors.
```

## Key Technical Notes

- **Cortical response shape**: (n_timesteps, 20484) float32 — only response the public TRIBE v2 checkpoint produces; subcortical variant is not released
- **Collapsed response shape**: (20484,) float32
- **ICA projection source**: `model.predictor.weights` in `best.ckpt`, shape `(1, 2048, 20484)` — the leading singleton is the subject axis (S=1 in unseen-subject mode); squeezed to `(2048, 20484)` before FastICA
- **Stimulus duration**: 10 s per static video for single-stimulus inference (allows TRIBE's output to span the t+5 s hemodynamic peak). 1 s — as used in the paper's streamed protocol — collapses to T=1 locally and loses peak-response information
- **Cache format**: NumPy `.npy` files per stimulus under `tensors/<content_hash>/` (`raw_cortical.npy` written by Colab, `collapsed.npy` materialized locally)
- **Hypothesis database**: Stored in `.hypothesis/` for test case replay
