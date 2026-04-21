# Project Structure

## Directory Layout

```
cognitive_similarity/          # Main package
├── __init__.py               # Public API exports
├── models.py                 # Data models (Stimulus, ICANetwork, SimilarityResult, etc.)
├── facade.py                 # CognitiveSimilarity - top-level API
├── similarity_engine.py      # SimilarityEngine - Pearson correlation computation
├── ica_atlas.py             # ICANetworkAtlas - brain network masks/components
├── collapsing.py            # TemporalCollapser - temporal response reduction
├── cache.py                 # ResponseCache - disk-based caching
├── stimulus_runner.py       # StimulusRunner - TRIBE v2 inference (Colab)
└── validation.py            # ValidationSuite - correctness validation

tests/                        # Test suite
├── test_facade.py           # Integration & property tests for facade
├── test_similarity_engine.py # Property tests for similarity computation
├── test_models.py           # Property tests for data models
├── test_ica_atlas.py        # Tests for ICA network atlas
├── test_collapsing.py       # Tests for temporal collapsing
└── test_cache.py            # Tests for caching system

.kiro/                        # Kiro configuration
├── specs/                   # Feature specifications
│   └── cognitive-similarity/ # Current feature spec
└── steering/                # AI assistant guidance (this file)

.hypothesis/                  # Hypothesis test database
├── examples/                # Saved test cases
└── unicode_data/            # Unicode data for text generation
```

## Module Responsibilities

### Core API Layer
- **facade.py**: User-facing API (`CognitiveSimilarity` class)
  - `compare(stimulus_a, stimulus_b)` → `SimilarityResult`
  - `rank(query, corpus)` → `RankedResult`
  - `get_collapsed_response(stimulus)` → `np.ndarray`

### Computation Layer
- **similarity_engine.py**: Pearson correlation across brain networks
  - `compute_profile()` → all 5 networks + whole-cortex score
  - `compute_network_score()` → single network score
  - Supports binary mask and continuous weighting modes

- **ica_atlas.py**: Brain network definitions
  - Loads ICA projection matrix from HuggingFace or synthetic
  - Provides vertex masks and component vectors for 5 networks
  - Top 10% percentile for binary masks

- **collapsing.py**: Temporal dimension reduction (shape `(T, 20484)` → `(20484,)`)
  - Method selected automatically from stimulus duration — not caller-configurable
  - Peak extraction at t+5s for duration ≤ 10s
  - GLM+HRF fitting (nilearn design matrix + lstsq) for duration > 10s

### Data & Caching Layer
- **cache.py**: Disk-based response storage
  - Raw responses: `{stimulus_id}_raw.npz` (cortical + subcortical)
  - Collapsed responses: `{stimulus_id}_collapsed.npy`
  - Avoids re-inference for repeated queries

- **models.py**: Type-safe data structures
  - All dataclasses with type hints
  - Enums for ICANetwork, ICAMode
  - Validation methods where needed

### Inference Layer (Colab-only)
- **stimulus_runner.py**: TRIBE v2 model inference
  - Not used in local testing (requires GPU)
  - Pre-computed responses stored in cache for local development

## Architecture Patterns

### Facade Pattern
`CognitiveSimilarity` orchestrates all components, hiding complexity from users.

### Strategy Pattern
Collapsing strategies (AUTO, PEAK, GLM_HRF) selected at runtime.

### Cache-Aside Pattern
Check cache → if miss, compute → store → return.

### Dependency Injection
Atlas and cache injected into facade for testability.

## Testing Strategy

### Property-Based Testing (PBT)
- Uses Hypothesis to generate test cases
- Validates correctness properties across input space
- Test names follow pattern: `test_property_N_description`
- Properties documented in test docstrings with requirement IDs

### Test Organization
- One test file per module
- Property tests grouped by feature requirement
- Integration tests at end of test files
- Fixtures shared via pytest fixtures (module scope for expensive setup)

### Synthetic Test Data
- Tests use synthetic ICA projection matrices (no HuggingFace dependency)
- Random seeds for reproducibility
- Temporary directories for cache isolation

## Code Conventions

### Imports
- `from __future__ import annotations` for forward references
- Standard library → third-party → local imports
- Absolute imports preferred

### Type Hints
- All public functions have type hints
- Return types always specified
- Use `Optional[T]` for nullable types

### Docstrings
- Module-level docstrings list requirement IDs
- Function docstrings for public API
- Property test docstrings include feature name and requirement validation

### Naming
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE
- Private members: leading underscore

### Error Handling
- Raise `ValueError` for invalid inputs with descriptive messages
- Raise `RuntimeError` for system-level failures
- Log warnings for non-fatal issues (zero variance, etc.)
