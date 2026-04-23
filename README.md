# Cognitive Similarity

Brain-grounded stimulus similarity using [TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta FAIR's tri-modal brain-encoding foundation model). Two stimuli are compared by predicting how similarly the cortex would respond to each, rather than by semantic embedding distance.

The pipeline:
1. Predict each stimulus's cortical activation pattern via TRIBE v2 (`(T, 20484)` on fsaverage5).
2. Collapse temporal responses to a single 20,484-vertex vector at the t+5 s hemodynamic peak.
3. Split the cortex into 5 functional networks via ICA on TRIBE's own learned subject-projection layer (primary auditory, language, motion MT+, default mode, visual — matching [TRIBEv2.pdf §2.7](TRIBEv2.pdf)).
4. Compute Pearson correlation between two collapsed vectors *within each network mask*.

## Validation status

Tested end-to-end against 9 ordering predictions from the neuroscience literature (e.g. `sim(face, face) > sim(face, place)` from FFA face-selectivity). Results on 23 curated IBC stimuli (see `scripts/ibc_exemplars.py` for the full list):

| Outcome | Count | Checks |
|---|---|---|
| ordering holds + mask-specific (ICA mask beats random cortex subset) | **2** | body>face, sentence>word_list |
| ordering holds, mask ≈ random (effect driven by global TRIBE structure) | **5** | face>place, place>body, wc>place, speech>non_speech, motion>static |
| ordering fails (wrong direction) | 2 | audio_segment>silence, complex_sentence>simple_sentence |

Full run output and significance stats: `python scripts/validate_ibc.py --cache-dir <local_cache>`.

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
pip install torch huggingface_hub scikit-learn nilearn pandas modal hypothesis pytest

# 3. Link Modal to your account (opens a browser)
modal token new

# 4. Store your HF token as a Modal Secret (used inside the GPU container
#    for gated Llama 3.2 access). Get the token from
#    https://huggingface.co/settings/tokens after joining the Llama 3.2 gate.
modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Running the full experiment

```bash
# Step 1 — smoke-test the remote pipeline (runs 4 stimuli, one per modality).
# First invocation builds the container image (~5 min, mostly tribev2 pip install).
modal run scripts/run_inference_modal.py --smoke-only

# Step 2 — run the full 23-stimulus batch AND mirror results to a local cache.
#   Idempotent: if the batch is interrupted, just re-run — already-cached
#   stimuli are skipped via content-hash lookup on the Modal Volume.
#   For long runs or unstable connections, use `modal run --detach ...` so
#   the GPU job survives local-side disconnects.
modal run scripts/run_inference_modal.py \
    --download-to ~/ibc_cache/cognitive-similarity-cache

# Step 3 — local analysis: materialize collapsed.npy, run ValidationSuite,
#   compute permutation + bootstrap significance tests.
#   First invocation downloads best.ckpt from HuggingFace (~676 MB) and runs
#   FastICA once (masks cached to <cache-dir>/ica_masks.npz).
python scripts/validate_ibc.py --cache-dir ~/ibc_cache/cognitive-similarity-cache
```

### Targeted runs (optional)

```bash
# Only re-process a subset (cached ones are skipped automatically)
modal run scripts/run_inference_modal.py \
    --ids face_01,speech_01,motion_video_01 \
    --download-to ~/ibc_cache/cognitive-similarity-cache

# Skip significance tests (faster; just shows ValidationSuite pass/fail)
python scripts/validate_ibc.py --cache-dir <cache> --no-significance

# Pull the cache manually if --download-to fails
modal volume get cogsim-cache manifest.json ~/ibc_cache/cognitive-similarity-cache/
modal volume get cogsim-cache tensors      ~/ibc_cache/cognitive-similarity-cache/
```

### Interpreting the output

`validate_ibc.py` prints three sections:

1. **Collapsed-tensor materialization** — one line per stimulus, showing the raw cortical shape (T, 20484) being reduced to (20484,) via peak extraction at t+5 s (per [TRIBEv2.pdf §5.8](TRIBEv2.pdf)).
2. **ValidationSuite ordering results** — 9 checks, PASS/FAIL on whether `sim(same-category) > sim(cross-category)`.
3. **Significance tests** per check:
   - **perm p** — random-mask permutation test: does the ICA mask beat a random same-size vertex subset? `p<0.05` means yes.
   - **boot CI** — 95% confidence interval on Δ from vertex bootstrap. CI excluding zero means the ordering is a stable estimate.

The verdict labels:
- `ordering + mask-specific` — both tests pass (the gold standard).
- `ordering only (mask ≈ random)` — Δ > 0 with stable CI, but the ICA mask doesn't outperform random cortex for this contrast. Indicates a real but non-network-specific effect.
- `unstable (CI spans 0)` — Δ > 0 but the bootstrap isn't reliable.
- `FAILED (wrong direction)` — Δ ≤ 0.

## Where things live

```
cognitive_similarity/           # Python package (the library)
├── __init__.py                 # public API exports
├── models.py                   # dataclasses (Stimulus, SimilarityResult, ...)
├── facade.py                   # CognitiveSimilarity — compare/rank
├── similarity_engine.py        # Pearson per-network, whole-cortex score
├── ica_atlas.py                # loads best.ckpt, runs FastICA, caches masks
├── collapsing.py               # TemporalCollapser (peak @ t+5s or GLM+HRF)
├── cache.py                    # ResponseCache — content-addressed .npy cache
├── stimulus_runner.py          # thin wrapper around TribeModel.predict
└── validation.py               # ValidationSuite — the 9 ordering checks

scripts/                        # Operational scripts (driven from your terminal)
├── ibc_exemplars.py            # the 23-stimulus spec (single source of truth)
├── run_inference_modal.py      # Modal App + GPU worker
└── validate_ibc.py             # local analysis + significance tests

tests/                          # pytest suite (83 tests, property-based via Hypothesis)
TRIBEv2.pdf                     # authoritative research source
demo.ipynb                      # local exploration on synthetic data

.kiro/specs/cognitive-similarity/
├── requirements.md             # functional spec (cited as Req X.Y throughout code)
├── design.md                   # architecture + component interfaces
└── tasks.md                    # historical implementation plan

.kiro/steering/
├── product.md                  # what the project is
├── tech.md                     # stack + commands
├── structure.md                # conventions
└── evidence-based-development.md  # rule: all decisions cite a source

CLAUDE.md                       # AI-context file (same info, AI-flavored)
```

### Data locations (external)

| What | Where | Size |
|---|---|---|
| TRIBE v2 checkpoint | HuggingFace `facebook/tribev2` (`best.ckpt`) | ~676 MB |
| IBC stimuli | GitHub `individual-brain-charting/public_protocols` (cloned fresh inside the Modal container) | ~300 MB repo |
| Modal Volume `cogsim-cache` | Modal's persistent cloud storage | ~50 MB per full run |
| Local cache (mirrored from Volume) | `~/ibc_cache/cognitive-similarity-cache/` (user-specified) | ~50 MB |
| ICA atlas cache | `<cache-dir>/ica_masks.npz` | ~900 KB |

## Testing the library

```bash
# Unit + property-based tests (synthetic data; no Modal / HF / torch required)
pytest

# Type-check (no mypy configured — syntax check only)
python -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/run_inference_modal.py', 'scripts/validate_ibc.py', 'scripts/ibc_exemplars.py']]"
```

## Design notes worth reading

- `TRIBEv2.pdf` §2.7 (+ Figure 6) — how the ICA component labels were derived and validated against NeuroSynth.
- `TRIBEv2.pdf` §5.9 — in-silico experiment methodology (static-video conversion, peak-response extraction).
- `.kiro/specs/cognitive-similarity/design.md` — our architecture, component interfaces, Modal driver structure.
- `.kiro/specs/cognitive-similarity/requirements.md` — the 9 validation orderings with their neuroscience sources.

## Known limitations

- **Subcortical predictions are out of scope.** The TRIBE v2 paper describes a subcortical variant, but Meta has not published that checkpoint on HuggingFace; the public release is cortical-only.
- **Language stimuli are short.** The gTTS-generated audio for short English sentences produces T=4–6 output timepoints, which isn't quite enough for the collapser to cleanly index the t+5 s peak. The `complex > simple sentence` check fails partly because of this.
- **ICA mask labels are positional.** Our `NETWORKS` list assumes FastICA with `random_state=42` returns components in the paper's IC1→IC5 order. Empirically this holds, but we don't currently auto-verify via NeuroSynth correlation (which would be the rigorous step).

## License

See `LICENSE` (MIT).
