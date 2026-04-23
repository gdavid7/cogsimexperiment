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
- ICA projection: stored in `best.ckpt` as `model.predictor.weights` of shape `(1, 2048, 20484)` — the leading singleton is the subject axis (S=1, unseen-subject mode); `ICANetworkAtlas` squeezes it to `(2048, 20484)` before running FastICA
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
├── stimulus_runner.py     # TRIBE v2 inference (used inside the Modal worker)
└── validation.py          # ValidationSuite
scripts/                   # Operational scripts (driven from the local Mac terminal)
├── ibc_exemplars.py       # Single source of truth — the 23 stimulus spec
├── run_inference_modal.py # Modal App + GPU worker; runs TRIBE v2 in the cloud
└── validate_ibc.py        # Local-side: materialize collapsed.npy + run ValidationSuite
tests/                     # One test file per module; property tests via Hypothesis
demo.ipynb                 # Local exploration notebook using synthetic data
.kiro/specs/cognitive-similarity/   # Authoritative specs (see below)
TRIBEv2.pdf                # Primary research source
```

Facade: `compare(a, b) → SimilarityResult`, `rank(query, corpus) → RankedResult`, `get_collapsed_response(stimulus) → np.ndarray`.

Architectural patterns: facade (CognitiveSimilarity), strategy (collapsing), cache-aside, dependency injection (atlas + cache into facade for testability).

## End-to-end Workflow

Inference is split across two environments, both driven from the local Mac terminal (no browser, no Colab session):

1. **Modal (GPU inference)** — `modal run scripts/run_inference_modal.py [--smoke-only] [--ids ...] [--download-to <path>]` spins up an A100 container with TRIBE v2, clones IBC `public_protocols` inside the container, preprocesses each stimulus (image→10 s static MP4, WAV→10 s padded with silence, French→English text→gTTS+WhisperX inside TRIBE, BioMvt clip→10 s looped MP4), runs cortical inference, and writes `raw_cortical.npy` to a persistent Modal Volume (`cogsim-cache`). `--download-to` optionally mirrors the Volume to a local cache dir on completion.
2. **Local Mac (CPU analysis)** — `python scripts/validate_ibc.py --cache-dir <cache>` materializes `collapsed.npy` (peak extraction at t+5 s) and runs `ValidationSuite` against the real HuggingFace-loaded ICA atlas.

**Prerequisites (one-time):** `pip install modal` + `modal token new` (browser auth) + `modal secret create hf-token HF_TOKEN=<your_token>` for the gated Llama 3.2 inside TRIBE's text pipeline. The first `modal run` builds the container image (~5 min, mostly the tribev2 pip install); subsequent runs start in seconds.

**Resuming after a dropped session:** if your laptop goes offline mid-run, the Modal Volume keeps whatever stimuli committed, and re-running the same command skips any cached `raw_cortical.npy`. For long runs or unstable connections, use `modal run --detach ...` so the GPU job continues even when the local client disconnects.

Stimulus duration is **10 s per static video/audio** (deviation from TRIBEv2.pdf §5.9's 1 s cited for the paper's streamed-GLM protocol). Rationale: single-stimulus inference needs enough timepoints for TRIBE's output to reach the t+5 s hemodynamic peak (§5.8); a 1-s input gives T=1 and the collapser falls back to the earliest timepoint with badly compressed Δs (0.01–0.05), whereas 10 s gives T≈10 and lets the collapser take `cortical_response[5]` with meaningful Δs (0.09–0.45).

## Validation Status

End-to-end validated on 23 IBC-curated stimuli (all 9 neuroscientific orderings evaluated) with two significance tests per check:
- **perm p** — random-mask permutation (n=1000); tests whether the ICA mask beats a random same-size cortex subset.
- **boot CI** — vertex bootstrap 95% CI on Δ (n=1000); tests whether Δ is a stable positive estimate.

| # | Check | Δ | perm p | boot CI | Verdict |
|---|---|---|---|---|---|
| 1 | sim(face,face) > sim(face,place) | +0.436 | >0.999 | [+0.40, +0.47] | ordering only |
| 2 | sim(place,place) > sim(place,body) | +0.340 | >0.999 | [+0.31, +0.38] | ordering only |
| 3 | sim(body,body) > sim(body,face) | +0.308 | <0.001 | [+0.29, +0.33] | **ordering + mask-specific** |
| 4 | sim(wc,wc) > sim(wc,place) | +0.091 | >0.999 | [+0.08, +0.10] | ordering only |
| 5 | sim(speech,speech) > sim(speech,non_speech) | +0.028 | >0.999 | [+0.03, +0.03] | ordering only (tiny Δ) |
| 6 | sim(audio_segment,audio_segment) > sim(audio_segment,silence) | −0.168 | 0.749 | [−0.19, −0.15] | FAILED (wrong direction) |
| 7 | sim(sentence,sentence) > sim(sentence,word_list) | +0.171 | <0.001 | [+0.15, +0.19] | **ordering + mask-specific** |
| 8 | sim(complex,complex) > sim(complex,simple) | −0.069 | <0.001 | [−0.08, −0.06] | FAILED (wrong direction) |
| 9 | sim(motion,motion) > sim(motion,static) | +0.304 | >0.999 | [+0.27, +0.34] | ordering only |

**7/9 orderings hold with stable Δ > 0. Only 2/7 also pass the mask-specificity test**: body>face and sentence>word_list. For the other 5 "ordering-only" checks, the effect is driven by global TRIBE response structure — a random same-size cortex subset gives a comparable or bigger Δ. The two failures (audio>silence, complex>simple) flip direction; both involve subtle contrasts whose known issues are documented in `README.md#known-limitations`.

To re-run: `python scripts/validate_ibc.py --cache-dir <local_cache>`.

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
