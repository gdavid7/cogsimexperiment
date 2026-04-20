# Implementation Plan: Cognitive Similarity

## Overview

Implement the Cognitive Similarity library in Python, split across two environments: a Google Colab notebook for TRIBE v2 inference (GPU), and a local Mac library for all analysis. The implementation proceeds bottom-up: data models → cache → temporal collapsing → ICA atlas → similarity engine → facade → validation → demo notebook.

## Tasks

- [x] 1. Define data models and project structure
  - Create `cognitive_similarity/` package with `__init__.py`
  - Implement all dataclasses and enums from the design: `ICANetwork`, `ICAMode`, `CollapsingStrategy`, `BrainResponse`, `Stimulus`, `NetworkScore`, `CognitiveSimilarityProfile`, `SimilarityResult`, `RankedEntry`, `RankedResult`, `ValidationCheck`, `ValidationReport`
  - Implement `Stimulus.validate()` raising `ValueError` when all modality paths are `None`
  - _Requirements: 1.6_

  - [x] 1.1 Write property test for invalid stimulus rejection
    - **Property 2: Invalid Stimulus Rejection**
    - **Validates: Requirements 1.6**

- [x] 2. Implement `ResponseCache`
  - Create `cognitive_similarity/cache.py`
  - Implement `_content_hash()` using SHA-256 over raw file bytes (chunked 65536-byte reads)
  - Implement `put_raw()` / `get_raw()` for `raw_cortical.npy` + `raw_subcortical.npy` under `tensors/<hash>/`
  - Implement `put_collapsed()` / `get_collapsed()` for `collapsed.npy` under `tensors/<hash>/`
  - Use `numpy.save()` / `numpy.load()` with `.npy` format (float32 precision preserved)
  - Log `WARNING` and return `None` on corrupted or wrong-shape cache files
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 2.1 Write property test for serialization round-trip
    - **Property 12: Collapsed Response Serialization Round-Trip**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 2.2 Write property test for content hash determinism and uniqueness
    - **Property 13: Content Hash Determinism and Uniqueness**
    - **Validates: Requirements 6.4**

- [x] 3. Implement `TemporalCollapser`
  - Create `cognitive_similarity/collapsing.py`
  - Implement `collapse()` with `CollapsingStrategy.AUTO` default
  - Peak extraction: index at `round((onset_s + 5.0) / tr_s)`; fall back to `T - 1` with `WARNING` log if out of bounds
  - GLM+HRF: use `nilearn.glm.first_level.make_first_level_design_matrix` + `numpy.linalg.lstsq`; return `beta[0]` (shape `(20484,)`)
  - Duration inferred as `T * tr_s` when `stimulus.duration_s` is `None`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.1 Write property test for strategy selection and output shape
    - **Property 3: Temporal Collapsing Strategy Selection and Output Shape**
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 3.2 Write property test for peak extraction correctness
    - **Property 4: Peak Extraction Correctness**
    - **Validates: Requirements 2.1**

  - [x] 3.3 Write unit tests for `TemporalCollapser`
    - Test peak fallback when `T < peak_idx` (logs warning, uses last timepoint)
    - Test GLM+HRF output shape `(20484,)` for `T > 10`
    - Test `CollapsingStrategy.AUTO` is the default parameter value
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement `ICANetworkAtlas`
  - Create `cognitive_similarity/ica_atlas.py`
  - Load `best.ckpt` from HuggingFace (`facebook/tribev2`) and extract the unseen-subject projection layer (shape `2048 × 20484`)
  - Run `sklearn.decomposition.FastICA(n_components=5)` on the extracted layer
  - Threshold each component at top 10% of absolute values to produce binary boolean masks of shape `(20484,)`
  - Implement `get_mask()`, `get_vertex_indices()`, `get_component()` per the design interface
  - Cache computed masks locally (e.g., `ica_masks.npz`) to avoid recomputation
  - _Requirements: 3.1, 3.2, 3.3_

  - [x] 5.1 Write unit tests for `ICANetworkAtlas`
    - Verify each network mask has shape `(20484,)` and dtype `bool`
    - Verify vertex count per network is approximately 2,048 (top 10% of 20,484)
    - Verify all vertex indices are in range `[0, 20483]`
    - Verify masks are NOT mutually exclusive (do not assert zero overlap)
    - _Requirements: 3.1, 3.2_

- [x] 6. Implement `SimilarityEngine`
  - Create `cognitive_similarity/similarity_engine.py`
  - Implement `pearson_correlation()` helper: mean-centered dot product, returns `0.0` on zero-variance input
  - Implement `compute_network_score()` for binary mask mode (restrict to top-10% vertices, equal weight)
  - Implement `compute_network_score()` for continuous weighting mode: `w = abs(component) / abs(component).sum()`, apply `sqrt(w)` weighting before Pearson
  - Implement `compute_profile()` returning `CognitiveSimilarityProfile` with all 5 network scores + `whole_cortex_score`
  - Whole-cortex score: vertex-count-weighted average of 5 network scores
  - Set `NetworkScore.warning` when zero-variance input detected
  - Raise `ValueError` for unknown network names (listing valid identifiers)
  - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 6.1 Write property test for ICA network masking isolation
    - **Property 5: ICA Network Masking Isolation**
    - **Validates: Requirements 3.5, 3.6**

  - [x] 6.2 Write property test for invalid ROI rejection
    - **Property 6: Invalid ROI Rejection**
    - **Validates: Requirements 3.7**

  - [x] 6.3 Write property test for continuous ICA weight normalization
    - **Property 7: Continuous ICA Weight Normalization**
    - **Validates: Requirements 3.2**

  - [x] 6.4 Write property test for cognitive similarity profile structure and score range
    - **Property 8: Cognitive Similarity Profile Structure and Score Range**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 6.5 Write property test for whole-cortex score formula
    - **Property 9: Whole-Cortex Score Is Vertex-Count-Weighted Average**
    - **Validates: Requirements 4.4**

  - [x] 6.6 Write unit tests for `SimilarityEngine`
    - Test zero-variance vector → score `0.0` with warning in `NetworkScore`
    - Test binary mask mode and continuous weighting mode produce different scores for the same pair
    - Test single-network query returns a single `Cognitive_Similarity_Score`
    - _Requirements: 4.3, 4.6, 4.8_

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement `CognitiveSimilarity` facade
  - Create `cognitive_similarity/facade.py`
  - Implement `__init__()`: instantiate `ICANetworkAtlas`, `ResponseCache`, `TemporalCollapser`, `SimilarityEngine`
  - Implement `get_collapsed_response()`: check cache → if miss, load raw cortical from cache → collapse → store collapsed → return
  - Implement `compare()`: get collapsed responses for both stimuli (via cache), call `engine.compute_profile()`, return `SimilarityResult`
  - Implement `rank()`: call `compare()` for each corpus stimulus, collect scores per network, sort descending with tie handling, return `RankedResult`
  - Raise `ValueError` when corpus has fewer than 2 stimuli
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 4.5, 4.7, 4.8, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 8.1 Write property test for stimulus isolation (one inference per stimulus)
    - **Property 1: Stimulus Isolation — One Inference Per Stimulus**
    - **Validates: Requirements 1.4**

  - [x] 8.2 Write property test for cache hit avoids re-inference
    - **Property 14: Cache Hit Avoids Re-Inference**
    - **Validates: Requirements 6.5**

  - [x] 8.3 Write property test for similarity result structural completeness
    - **Property 10: Similarity Result Structural Completeness**
    - **Validates: Requirements 3.8, 4.5**

  - [x] 8.4 Write property test for batch result ordering
    - **Property 11: Batch Result Ordering**
    - **Validates: Requirements 4.7**

  - [x] 8.5 Write property test for ranked list descending order
    - **Property 16: Ranked List Is Sorted in Descending Order**
    - **Validates: Requirements 7.1**

  - [x] 8.6 Write property test for tie handling in rankings
    - **Property 17: Tie Handling — Equal Scores Share Rank**
    - **Validates: Requirements 7.5**

  - [x] 8.7 Write property test for JSON output completeness
    - **Property 15: JSON Output Contains All Required Fields**
    - **Validates: Requirements 6.6**

  - [x] 8.8 Write integration tests for `CognitiveSimilarity`
    - End-to-end `compare()` with mocked TRIBE v2 responses (verifies full pipeline)
    - Cache population and retrieval: two `compare()` calls for the same stimulus, verify `predict()` called only once
    - `rank()` with corpus < 2 stimuli raises `ValueError`
    - _Requirements: 1.1, 1.4, 4.1, 6.5, 7.1_

- [x] 9. Implement `StimulusRunner` and `remote_inference.ipynb`
  - Create `cognitive_similarity/stimulus_runner.py`
  - Implement `StimulusRunner.run()`: call `get_events_dataframe()` + `predict()` on cortical model, then separately on subcortical model; return `BrainResponse`
  - Create `remote_inference.ipynb` with 5 cells per the design:
    - Cell 1: mount Drive, clone repo, install deps, HuggingFace login
    - Cell 2: load cortical and subcortical `TribeModel` instances
    - Cell 3: download IBC stimuli, build manifest
    - Cell 4: resume-from-checkpoint inference loop (skip if `raw_cortical.npy` exists)
    - Cell 5: verify cache (list all cached stimuli with shapes)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 10. Implement `ValidationSuite`
  - Create `cognitive_similarity/validation.py`
  - Implement `ValidationSuite.run()`: load stimuli via `manifest.json` content hashes, retrieve `Collapsed_Response` from `ResponseCache`, compute and compare similarity scores for all 9 expected orderings
  - Implement all 9 checks from Requirement 5.3 (4 Visual System, 2 Primary Auditory Cortex, 2 Language Network, 1 Motion Detection MT+)
  - Return `ValidationReport` with per-check `ValidationCheck` objects and summary `passed` / `total` counts
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement `demo.ipynb`
  - Create `demo.ipynb` with 6 sections per the design:
    - Setup: point to local cache dir, load `CognitiveSimilarity`
    - Single comparison: compare two IBC stimuli, print `CognitiveSimilarityProfile`
    - Ranked similarity: query + full IBC corpus, display ranked table per network
    - Per-network bar chart: visualize 5 network scores for a stimulus pair
    - Validation suite: run 9 checks, display pass/fail
    - Cache inspection: list cached stimuli, sizes, confirm round-trip serialization
  - _Requirements: (demo only — no direct requirement number)_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use **Hypothesis** (`@given` / `@settings(max_examples=100)`)
- Tag each property test: `# Feature: cognitive-similarity, Property N: <property_text>`
- The Colab notebook (task 10) and demo notebook (task 13) can be done in parallel with or after the local library tasks
- `StimulusRunner` is only used in Colab — local library reads raw tensors from cache written by Colab
