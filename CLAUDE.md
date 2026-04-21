# CLAUDE.md

Persistent context for this repository. Migrated from `.kiro/` (AWS Kiro) on 2026-04-20.

## Product

**Cognitive Similarity** — a brain-grounded stimulus similarity system using TRIBE v2 (Meta FAIR's tri-modal brain encoding foundation model, HuggingFace `facebook/tribev2`).

The pipeline:
1. Generate brain response predictions via TRIBE v2 (cortical model, shape `[T, 20484]` on fsaverage5).
2. Collapse temporal responses to a single cortical vector.
3. Compute Pearson correlation across 5 ICA-derived brain networks (primary auditory cortex, language, motion MT+, default mode, visual).
4. Produce similarity profiles and rankings.

Supports video, audio, and text stimuli. Two ICA modes: binary mask (top 10% vertices) and continuous weighting. Disk cache for raw and collapsed responses.

**Why cortical, not semantic:** cortical similarity captures *how it is to experience* a stimulus (visual rhythm, auditory texture, cognitive load, cross-modal convergence) — things semantic embeddings miss. Subcortical prediction is out of scope: the TRIBE v2 paper describes a subcortical variant, but Meta has not published that checkpoint on HuggingFace (`facebook/tribev2` ships only the cortical `best.ckpt` hard-wired to TribeSurfaceProjector / fsaverage5).

Use cases: content recommendation, stimulus validation for neuroscience, brain-grounded semantic search.

## Tech Stack

- Python ≥ 3.11, setuptools backend
- Core: `numpy` ≥ 1.26, `scikit-learn` ≥ 1.4, `nilearn` ≥ 0.10
- `tribev2` — not on PyPI, install via `pip install git+https://github.com/facebookresearch/tribev2.git`
- Tests: `pytest` ≥ 8.0, `hypothesis` ≥ 6.100 (property-based testing)

Install: `pip install -e ".[dev]"` then install TRIBE v2 separately.
Test: `pytest` / `pytest tests/test_facade.py -v`.

**Shapes to remember:**
- Cortical response: `(T, 20484)` float32 — only response TRIBE v2's public checkpoint produces
- Collapsed cortical: `(20484,)` float32
- ICA projection matrix: `(2048, 20484)` for 5 networks
- Cache files: `.npy` under `tensors/<content_hash>/` (`raw_cortical.npy`, `collapsed.npy`)

No mypy / pyright configured. Hypothesis DB lives in `.hypothesis/`.

## Project Structure

```
cognitive_similarity/       # Main package
├── __init__.py            # Public API exports
├── models.py              # Dataclasses + enums (Stimulus, ICANetwork, SimilarityResult, ...)
├── facade.py              # CognitiveSimilarity — top-level API
├── similarity_engine.py   # Pearson correlation across networks
├── ica_atlas.py           # ICANetworkAtlas — network masks / components
├── collapsing.py          # TemporalCollapser (auto-selects method from duration)
├── cache.py               # ResponseCache — disk-based
├── stimulus_runner.py     # TRIBE v2 inference (Colab-only, GPU)
└── validation.py          # ValidationSuite
tests/                     # One test file per module; property tests via Hypothesis
.kiro/specs/cognitive-similarity/   # Authoritative specs (see below)
TRIBEv2.pdf                # Primary research source
```

Facade: `compare(a, b) → SimilarityResult`, `rank(query, corpus) → RankedResult`, `get_collapsed_response(stimulus) → np.ndarray`.

Architectural patterns: facade (CognitiveSimilarity), strategy (collapsing), cache-aside, dependency injection (atlas + cache into facade for testability).

Inference is split: Colab notebook runs TRIBE v2 on GPU and populates the cache; local Mac library does all analysis against cached responses.

## Code Conventions

- `from __future__ import annotations` at top of modules
- Import order: stdlib → third-party → local, absolute imports
- Type hints on all public functions, return types always specified, `Optional[T]` for nullable
- Classes `PascalCase`, functions / methods `snake_case`, constants `UPPER_SNAKE_CASE`, private `_leading_underscore`
- Module docstrings list requirement IDs; public API has docstrings; property-test docstrings name the feature + requirement
- `ValueError` for bad inputs (descriptive message), `RuntimeError` for system failures, log warnings for non-fatal issues (e.g. zero variance)

### Testing

- Property-based testing with Hypothesis; test names follow `test_property_N_description`
- One test file per module; integration tests at the end of the file
- Module-scope pytest fixtures for expensive setup
- Tests use synthetic ICA matrices (no HuggingFace dependency), fixed seeds, tempdir-isolated caches

## Evidence-Based Development (non-negotiable)

Every technical decision, parameter, and implementation choice **must be traceable to a credible source**. Do not guess; do not rely on general knowledge.

**Source hierarchy** (consult in order):
1. `TRIBEv2.pdf` (repo root) — the primary research paper. Cite specific sections / figures / equations.
2. `.kiro/specs/cognitive-similarity/requirements.md` — functional specs. Quote requirement IDs.
3. Library docs — TRIBE v2 GitHub, NumPy, scikit-learn, nilearn, Hypothesis.
4. Existing code and tests in `cognitive_similarity/` and `tests/`.

**Before writing code:** identify the requirement, find the source, quote the source, verify interpretation.

**Phrasing:**
- Avoid: "I think…", "this is standard…", "typically…", "let me try…"
- Prefer: "Per TRIBEv2.pdf §X…", "Requirements.md 4.5 specifies…", "NumPy docs state…"

**Cite sources in code comments, commit messages, and test docstrings** when implementing non-trivial logic. Example:

```python
# Per TRIBEv2.pdf §4.1: binary masks select top 10% of vertices
# by absolute component weight for each ICA network
top_percentile = 0.10
```

**When sources conflict:** document each, apply the hierarchy (paper > requirements > library docs), verify against existing tests, and flag for review.

**When uncertain:** stop and search — do not guess. Use web search for library docs. Re-read the paper. Check existing property tests (they often encode requirements).

## Specs (authoritative — read when implementing)

Large, detailed, and load-bearing. Read directly rather than relying on this summary:

- `.kiro/specs/cognitive-similarity/requirements.md` — functional spec, API contracts, validation criteria (requirement IDs cited throughout code)
- `.kiro/specs/cognitive-similarity/design.md` — architecture and implementation design
- `.kiro/specs/cognitive-similarity/tasks.md` — implementation plan with checkbox progress

## Migration Notes

Migrated from Kiro steering files:
- `.kiro/steering/product.md` → **Product** section
- `.kiro/steering/tech.md` → **Tech Stack** section
- `.kiro/steering/structure.md` → **Project Structure** + **Code Conventions**
- `.kiro/steering/evidence-based-development.md` → **Evidence-Based Development**

No `.kiro/hooks/` directory exists in this project — nothing to migrate there. If automated behaviors are added later, configure them as Claude Code hooks in `.claude/settings.json` (they run in the harness, not as Claude instructions).

The `.kiro/specs/` directory is kept in place and referenced above; its contents remain authoritative.
