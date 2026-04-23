# Tech Stack

## Build System

- **Build backend**: setuptools (legacy backend)
- **Python version**: >=3.11

## Core Dependencies

- **numpy** (>=1.26): Numerical operations, array handling
- **scikit-learn** (>=1.4): Machine learning utilities (FastICA)
- **scipy**: `linear_sum_assignment` for §5.10 ICA-component label matching
- **nilearn** (>=0.10): Neuroimaging data processing, `surface.vol_to_surf` for projecting NeuroSynth maps to fsaverage5
- **statsmodels**: Benjamini-Hochberg FDR correction in validation output
- **tribev2**: Brain encoding model (install from GitHub: `pip install git+https://github.com/facebookresearch/tribev2.git`)

## Optional Dependencies

- `[neurosynth]`: **nimare** (>=0.1) — only required by `scripts/fetch_neurosynth_maps.py` (one-shot §5.10 term-map generation). The runtime library consumes the cached `.npz` output and doesn't import NiMARE.
- `[inference]`: **torch**, **torchaudio**, **huggingface-hub**, **transformers** — for running TRIBE v2 inference (typically in the Modal container, not locally).

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

### End-to-end pipeline (runs from the local Mac terminal via Modal)

One-time setup:
```bash
pip install modal                                          # installs CLI + Python SDK
modal token new                                            # browser auth
modal secret create hf-token HF_TOKEN=<your_hf_token>      # gated Llama 3.2 access
```

Run:
```bash
# Smoke test (3 stimuli, one per modality):
modal run scripts/run_inference_modal.py --smoke-only

# Full batch (14 stimuli) + pull results to local cache:
modal run scripts/run_inference_modal.py \
    --download-to /path/to/local/cognitive-similarity-cache

# Long or flaky connection? Detach so the job survives local disconnects:
modal run --detach scripts/run_inference_modal.py
# ...then later pull results yourself:
modal volume get cogsim-cache manifest.json /path/to/local/cache/
modal volume get cogsim-cache tensors /path/to/local/cache/

# One-time: NeuroSynth term-map fetch for §5.10 ICA labeling (~3 min, ~100 MB)
python scripts/fetch_neurosynth_maps.py --cache-dir /path/to/local/cache

# Local analysis + validation (requires torch + hf_hub in whichever venv):
python scripts/validate_ibc.py --cache-dir /path/to/local/cache
# First run downloads best.ckpt from HF (~676 MB) and runs FastICA once
# (cached to <cache>/ica_masks.npz); subsequent runs are fast. If
# neurosynth_maps.npz exists in the cache dir, labels are §5.10-assigned
# automatically; otherwise WARNING + positional fallback.

# Optional: paper-faithful Figure 4E replication (§5.9 contrast maps)
python scripts/replicate_figure_4e.py --cache-dir /path/to/local/cache
```

## Key Technical Notes

- **Cortical response shape**: (n_timesteps, 20484) float32 — only response the public TRIBE v2 checkpoint produces; subcortical variant is not released
- **Collapsed response shape**: (20484,) float32
- **ICA projection source**: `model.predictor.weights` in `best.ckpt`, shape `(1, 2048, 20484)` — the leading singleton is the subject axis (S=1 in unseen-subject mode); squeezed to `(2048, 20484)` before FastICA
- **Stimulus protocol**: 1 s stimulus + 7 s blank = 8 s total (matches TRIBEv2.pdf §5.9 "one second every eight seconds"). TRIBE's 1 Hz output produces T=8 timepoints; `TemporalCollapser` indexes `cortical_response[5]` for the §5.8 / Fig 4A hemodynamic peak. An earlier revision used 10 s sustained stimuli; Slice 3 reverted to the paper's protocol.
- **Cache format**: NumPy `.npy` files per stimulus under `tensors/<content_hash>/` (`raw_cortical.npy` written by Modal worker, `collapsed.npy` materialized locally)
- **Hypothesis database**: Stored in `.hypothesis/` for test case replay
