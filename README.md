# Cognitive Similarity

Brain-grounded stimulus similarity using [TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta FAIR's tri-modal brain-encoding foundation model). Two stimuli are compared by predicting how similarly the cortex would respond to each, rather than by semantic embedding distance.

The pipeline:
1. Predict each stimulus's cortical activation pattern via TRIBE v2 (`(T, 20484)` on fsaverage5).
2. Collapse temporal responses to a single 20,484-vertex vector at the t+5 s hemodynamic peak ([TRIBEv2.pdf §5.8](TRIBEv2.pdf)).
3. Split the cortex into 5 functional networks via ICA on TRIBE's own learned subject-projection layer, and assign each network label by spatial Pearson correlation against NeuroSynth keyword maps ([§5.10, Figure 6B](TRIBEv2.pdf)).
4. Compute pairwise Pearson correlation between two collapsed vectors *within each network mask*.

**How this relates to the paper.** The spatial Pearson metric comes from [§2.5 / §2.6 / §5.10](TRIBEv2.pdf), where it's used to compare predicted vs. measured (or NeuroSynth-reference) maps. We apply the same metric pairwise between two predicted single-stimulus responses, which is a novel construction not independently validated by §5.9. A paper-faithful Figure 4E replication (contrast-map methodology per §5.9) ships as a separate script (`scripts/replicate_figure_4e.py`) and reuses the same cached tensors.

## Validation status

Tested end-to-end against 6 pairwise ordering predictions from the neuroscience literature (e.g. `sim(face, face) > sim(face, place)` from FFA face-selectivity). Per-check statistics include vertex-bootstrap CI on Δ and a random-mask permutation test for mask specificity, with Benjamini-Hochberg FDR correction across the batch.

| Check | Δ | BH q | CI excludes 0 | Notes |
|---|---|---|---|---|
| face > place (visual)                  | +0.705 | >0.999 | ✓ | ordering only |
| place > body (visual)                  | +0.465 | >0.999 | ✓ | ordering only |
| body > face (visual)                   | +0.230 | >0.999 | ✓ | ordering only |
| written_char > place (visual/VWFA)     | +0.164 | >0.999 | ✓ | ordering only |
| speech > non_speech (primary auditory) | +0.028 | >0.999 | ✓ | ordering only, tiny Δ |
| sentence > word_list (language)        | −0.037 | >0.999 | — | FAILED (wrong direction) |

5/6 orderings hold with stable Δ > 0. 0/6 show mask-specific selectivity after FDR correction — i.e. a random same-size subset of cortex gives Δs comparable to the labeled mask. This is not a metric failure; it reflects that TRIBE's predicted responses carry broad category information throughout cortex, not just in narrow ICA-defined networks. The one failing check is structural, not a bug: pairwise Pearson on a correctly-labeled language mask doesn't replicate the paper's §2.6 *magnitude*-contrast finding between sentences and word lists (different statistical question).

*The numbers above are from the current cache. Re-running Modal with the §5.9-aligned 8 s stimulus protocol (Slice 3) may shift them; see `CLAUDE.md` for per-slice provenance.*

Full run with significance stats: `python scripts/validate_ibc.py --cache-dir <local_cache>`.
Paper-faithful Figure 4E contrast-map replication: `python scripts/replicate_figure_4e.py --cache-dir <local_cache>`.

## Prerequisites

- **Python ≥ 3.11** (a venv strongly recommended — torch + nilearn + Modal combined won't coexist nicely with a system python locked under PEP 668).
- **ffmpeg** on your PATH (`brew install ffmpeg` on macOS). Only needed if you're running the *local* analysis; Modal has its own ffmpeg inside the container image.
- **A HuggingFace account with access to the gated [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B) model** — TRIBE v2's text pipeline requires it for TTS token embeddings.
- **A Modal account** (<https://modal.com>) — free tier is plenty for this project (~$1 of GPU time per full run).
- **~2 GB free disk** (torch, best.ckpt download, cache).

## One-time setup

```bash
# 1. Clone
git clone https://github.com/gdavid7/cogsimexperiment.git
cd cogsimexperiment

# 2. Create/activate a venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,inference,neurosynth]"
# or: pip install torch huggingface_hub scikit-learn nilearn pandas modal hypothesis pytest nimare

# 3. Install TRIBE v2 (not on PyPI)
pip install git+https://github.com/facebookresearch/tribev2.git

# 4. Link Modal to your account (opens a browser)
modal token new

# 5. Store your HF token as a Modal Secret (used inside the GPU container
#    for gated Llama 3.2 access). Get the token from
#    https://huggingface.co/settings/tokens after joining the Llama 3.2 gate.
modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 6. One-shot fetch of NeuroSynth term maps for §5.10 ICA labeling.
#    ~100 MB download + 5 term meta-analyses; ~3 min wall-clock.
python scripts/fetch_neurosynth_maps.py \
    --cache-dir ~/ibc_cache/cognitive-similarity-cache
```

## Running the full experiment

```bash
# Step 1 — smoke-test the remote pipeline (runs 3 stimuli, one per modality).
# First invocation builds the container image (~5 min, mostly tribev2 pip install).
modal run scripts/run_inference_modal.py --smoke-only

# Step 2 — run the full 14-stimulus batch AND mirror results to a local cache.
#   Idempotent: if the batch is interrupted, just re-run — already-cached
#   stimuli are skipped via content-hash lookup on the Modal Volume.
#   For long runs or unstable connections, use `modal run --detach ...` so
#   the GPU job survives local-side disconnects.
modal run scripts/run_inference_modal.py \
    --download-to ~/ibc_cache/cognitive-similarity-cache

# Step 3 — local analysis: materialize collapsed.npy, run ValidationSuite,
#   compute permutation + bootstrap significance tests (with FDR correction).
#   First invocation downloads best.ckpt from HuggingFace (~676 MB) and runs
#   FastICA once (masks cached to <cache-dir>/ica_masks.npz). If NeuroSynth
#   term maps exist in the cache dir, the atlas applies the §5.10 bipartite
#   label assignment automatically.
python scripts/validate_ibc.py --cache-dir ~/ibc_cache/cognitive-similarity-cache

# Step 4 (optional) — paper-faithful Figure 4E replication (contrast maps per §5.9).
python scripts/replicate_figure_4e.py --cache-dir ~/ibc_cache/cognitive-similarity-cache
```

### Targeted runs (optional)

```bash
# Only re-process a subset (cached ones are skipped automatically)
modal run scripts/run_inference_modal.py \
    --ids face_01,speech_01,sentence_01 \
    --download-to ~/ibc_cache/cognitive-similarity-cache

# Skip significance tests (faster; just shows ValidationSuite pass/fail)
python scripts/validate_ibc.py --cache-dir <cache> --no-significance

# Pull the cache manually if --download-to fails
modal volume get cogsim-cache manifest.json ~/ibc_cache/cognitive-similarity-cache/
modal volume get cogsim-cache tensors      ~/ibc_cache/cognitive-similarity-cache/

# Force-regenerate NeuroSynth term maps (e.g. after a vocabulary update)
python scripts/fetch_neurosynth_maps.py --cache-dir <cache> --overwrite --invalidate-ica-cache
```

### Interpreting the output

`validate_ibc.py` prints three sections:

1. **Collapsed-tensor materialization** — one line per stimulus, showing the raw cortical shape (T, 20484) being reduced to (20484,) via peak extraction at t+5 s (per [TRIBEv2.pdf §5.8](TRIBEv2.pdf)).
2. **ValidationSuite ordering results** — 6 checks, PASS/FAIL on whether `sim(same-category) > sim(cross-category)`.
3. **Significance tests** per check:
   - **perm p** — random-mask permutation: does the ICA mask beat a random same-size vertex subset?
   - **BH q** — Benjamini-Hochberg FDR-corrected p across the batch. `q<0.05` → mask-specificity survives multiple-comparison correction.
   - **boot CI** — 95% confidence interval on Δ from vertex bootstrap. CI excluding zero means the ordering is a stable estimate. *Caveat: i.i.d. vertex resampling underestimates CI width — cortical vertices are spatially autocorrelated. See `_significance_test` docstring.*

The verdict labels:
- `ordering + mask-specific` — Δ > 0, CI excludes zero, BH q<0.05.
- `ordering only (mask ≈ random)` — Δ > 0, CI excludes zero, but the mask doesn't outperform random cortex after FDR. Indicates a real but non-network-specific effect.
- `unstable (CI spans 0)` — Δ > 0 but the bootstrap isn't reliable.
- `FAILED (wrong direction)` — Δ ≤ 0.

## Where things live

```
cognitive_similarity/                     # Python package (the library)
├── __init__.py                          # public API exports
├── models.py                            # dataclasses (Stimulus, SimilarityResult, ...)
├── facade.py                            # CognitiveSimilarity — compare/rank
├── similarity_engine.py                 # pearson + weighted_pearson + SimilarityEngine
├── ica_atlas.py                         # loads best.ckpt, runs FastICA, persists §5.10 labels
├── neurosynth_labels.py                 # §5.10 fetch + bipartite label assignment
├── collapsing.py                        # TemporalCollapser (peak @ t+5 s or GLM+HRF)
├── cache.py                             # ResponseCache — content-addressed .npy cache
├── stimulus_runner.py                   # thin wrapper around TribeModel.predict
├── validation.py                        # ValidationSuite — 6 pairwise ordering checks
└── paper_replication.py                 # §5.9 Figure 4E contrast-map replication

scripts/                                 # Operational scripts (driven from your terminal)
├── ibc_exemplars.py                     # 14-stimulus spec (single source of truth)
├── run_inference_modal.py               # Modal App + GPU worker (1 s + 7 s blank preprocessor)
├── validate_ibc.py                      # local analysis + FDR-corrected significance
├── fetch_neurosynth_maps.py             # one-shot §5.10 term-map fetch (NiMARE)
└── replicate_figure_4e.py               # Fig 4E contrast-map replication CLI

tests/                                   # pytest suite (110 tests, property-based via Hypothesis)
TRIBEv2.pdf                              # authoritative research source
demo.ipynb                               # local exploration on synthetic data

.kiro/specs/cognitive-similarity/
├── requirements.md                      # functional spec (cited as Req X.Y throughout code)
├── design.md                            # architecture + component interfaces
└── tasks.md                             # historical implementation plan

.kiro/steering/
├── product.md                           # what the project is
├── tech.md                              # stack + commands
├── structure.md                         # conventions
└── evidence-based-development.md        # rule: all decisions cite a source

CLAUDE.md                                # AI-context file (same info, AI-flavored)
```

### Data locations (external)

| What | Where | Size |
|---|---|---|
| TRIBE v2 checkpoint | HuggingFace `facebook/tribev2` (`best.ckpt`) | ~676 MB |
| IBC stimuli | GitHub `individual-brain-charting/public_protocols` at pinned SHA `cbbb7715` (cloned fresh inside the Modal container) | ~300 MB repo |
| Modal Volume `cogsim-cache` | Modal's persistent cloud storage | ~30 MB per full run |
| Local cache (mirrored from Volume) | `~/ibc_cache/cognitive-similarity-cache/` (user-specified) | ~30 MB |
| ICA atlas cache | `<cache-dir>/ica_masks.npz` | ~900 KB |
| NeuroSynth term maps | `<cache-dir>/neurosynth_maps.npz` | ~400 KB |
| NeuroSynth v7 database (one-shot) | `<cache-dir>/_neurosynth/` (configurable) | ~100 MB |

## Testing the library

```bash
# Unit + property-based tests (synthetic data; no Modal / HF / torch required)
pytest

# 110 tests total as of Slice 4. Synthetic-projection FastICA still emits one
# ConvergenceWarning — that's expected for random-normal test matrices, which
# have no true independent components.
```

## Design notes worth reading

- [TRIBEv2.pdf §2.7 + Figure 6](TRIBEv2.pdf) — how the ICA component labels were derived and validated against NeuroSynth. Our implementation of §5.10 is in `cognitive_similarity/neurosynth_labels.py`.
- [TRIBEv2.pdf §5.9](TRIBEv2.pdf) — in-silico experiment methodology (1 s flashed stimuli, GLM fits, contrast maps). Our `_preprocess_stimulus` + `paper_replication.py` mirror this for image experiments.
- `.kiro/specs/cognitive-similarity/design.md` — our architecture, component interfaces, Modal driver structure.
- `.kiro/specs/cognitive-similarity/requirements.md` — the pairwise-Pearson orderings with their neuroscience sources, and an honest note about how pairwise similarity relates to the paper's magnitude contrasts.

## Known limitations

- **Subcortical predictions are out of scope.** The TRIBE v2 paper describes a subcortical variant, but Meta has not published that checkpoint on HuggingFace; the public release is cortical-only (one `best.ckpt` hard-wired to TribeSurfaceProjector).
- **Pairwise Pearson is not the paper's validation metric.** We reuse the §5.10 metric for a different question (pairwise similarity) than the paper validates (predicted-vs-measured contrast maps). Paper-faithful Figure 4E-style replication is in `scripts/replicate_figure_4e.py` for direct comparison.
- **Bootstrap CIs ignore spatial autocorrelation.** Cortical vertices are not independent; our i.i.d. vertex bootstrap underestimates CI width. Current |Δ|s are large enough that this doesn't flip any verdict, but don't report these as formal 95% intervals. Upgrade to spin tests via `neuromaps` if you need to publish.
- **Language ICA component is a weaker match to NeuroSynth** than the other four (|r|≈0.40 vs. 0.47–0.52 for the others). TRIBE v2's unseen-subject projection doesn't produce a strongly separated, left-lateralized language component; matches are still the best-available by Hungarian assignment, but pairwise orderings inside the language mask are correspondingly noisier.

## License

See `LICENSE` (MIT).
