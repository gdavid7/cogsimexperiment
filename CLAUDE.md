# CLAUDE.md

Persistent context for this repository. Migrated from `.kiro/` (AWS Kiro) on 2026-04-20; paper-alignment sweep (Slices 1–5) on 2026-04-23.

## Product

**Cognitive Similarity** — a brain-grounded stimulus similarity system using TRIBE v2 (Meta FAIR's tri-modal brain encoding foundation model, HuggingFace `facebook/tribev2`).

The pipeline:
1. Generate brain response predictions via TRIBE v2 (cortical model, shape `[T, 20484]` on fsaverage5).
2. Collapse temporal responses to a single cortical vector at the t+5 s hemodynamic peak (§5.8).
3. Compute pairwise Pearson correlation across 5 ICA-derived brain networks (primary auditory cortex, language, motion MT+, default mode, visual). ICA labels are assigned via the paper's §5.10 procedure — spatial Pearson correlation against five NeuroSynth keyword maps, with a bipartite 1-to-1 assignment. Not positional.
4. Produce similarity profiles and rankings.

Supports video, audio, and text stimuli. Two ICA modes: binary mask (top 10% vertices, default) and continuous weighting (standard weighted Pearson over all 20484 vertices). Disk cache for raw and collapsed responses.

**Why cortical, not semantic:** cortical similarity captures *how it is to experience* a stimulus (visual rhythm, auditory texture, cognitive load, cross-modal convergence) — things semantic embeddings miss. Subcortical prediction is out of scope: the TRIBE v2 paper describes a subcortical variant, but Meta has not published that checkpoint on HuggingFace (`facebook/tribev2` ships only the cortical `best.ckpt` hard-wired to TribeSurfaceProjector / fsaverage5).

**How this relates to the paper** (important — avoid overclaiming): the spatial Pearson metric we use is the same one the paper applies in §2.5 / §2.6 / §5.10. The paper uses it to compare predicted vs measured / NeuroSynth-reference maps. We use the same metric pairwise between two predicted single-stimulus responses inside an ICA mask — a novel construction not independently validated by §5.9. Paper-faithful replication of Figure 4E is provided as a separate module (`cognitive_similarity.paper_replication` + `scripts/replicate_figure_4e.py`) that follows §5.9's contrast-map methodology exactly.

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
├── similarity_engine.py   # pearson + weighted_pearson + SimilarityEngine
├── ica_atlas.py           # ICANetworkAtlas — network masks / components (+ §5.10 labels)
├── neurosynth_labels.py   # §5.10 NeuroSynth fetching + bipartite label assignment
├── collapsing.py          # TemporalCollapser (auto-selects method from duration)
├── cache.py               # ResponseCache — disk-based
├── stimulus_runner.py     # TRIBE v2 inference (used inside the Modal worker)
├── validation.py          # ValidationSuite (6 pairwise-Pearson ordering checks)
└── paper_replication.py   # §5.9 Fig 4E contrast-map replication (not pairwise)
scripts/                   # Operational scripts (driven from the local Mac terminal)
├── ibc_exemplars.py       # Single source of truth — 14-stimulus spec
├── run_inference_modal.py # Modal App + GPU worker; runs TRIBE v2 in the cloud
├── validate_ibc.py        # Local-side: materialize collapsed.npy + run ValidationSuite
├── fetch_neurosynth_maps.py    # One-shot §5.10 term-map fetch (requires nimare)
└── replicate_figure_4e.py      # Runs the Fig 4E contrast-map replication end-to-end
tests/                     # One test file per module; property tests via Hypothesis
demo.ipynb                 # Local exploration notebook using synthetic data
.kiro/specs/cognitive-similarity/   # Authoritative specs (see below)
TRIBEv2.pdf                # Primary research source
```

Facade: `compare(a, b) → SimilarityResult`, `rank(query, corpus) → RankedResult`, `get_collapsed_response(stimulus) → np.ndarray`.

Architectural patterns: facade (CognitiveSimilarity), strategy (collapsing), cache-aside, dependency injection (atlas + cache into facade for testability).

## End-to-end Workflow

Inference is split across two environments, both driven from the local Mac terminal (no browser, no Colab session):

1. **Modal (GPU inference)** — `modal run scripts/run_inference_modal.py [--smoke-only] [--ids ...] [--download-to <path>]` spins up an A100 container with TRIBE v2, clones IBC `public_protocols` at a pinned commit (SHA in `run_inference_modal.py:IBC_PROTOCOLS_SHA`), preprocesses each stimulus to 1 s content + 7 s blank = 8 s total (image → static frame + black pad, WAV → signal + silence pad, text → .txt), runs cortical inference, and writes `raw_cortical.npy` to a persistent Modal Volume (`cogsim-cache`). `--download-to` optionally mirrors the Volume to a local cache dir on completion.
2. **Local Mac (CPU analysis)** — `python scripts/validate_ibc.py --cache-dir <cache>` materializes `collapsed.npy` (peak extraction at t+5 s per §5.8) and runs `ValidationSuite` against the real HuggingFace-loaded ICA atlas. The atlas is §5.10-labeled once `scripts/fetch_neurosynth_maps.py` has been run; before that, it falls back to positional labels and logs a loud WARNING.

**Prerequisites (one-time):**
- `pip install modal` + `modal token new` (browser auth) + `modal secret create hf-token HF_TOKEN=<your_token>` for the gated Llama 3.2 inside TRIBE's text pipeline. The first `modal run` builds the container image (~5 min, mostly the tribev2 pip install); subsequent runs start in seconds.
- For paper-faithful ICA labels: `pip install -e ".[neurosynth]"` then `python scripts/fetch_neurosynth_maps.py --cache-dir <local_cache>`. This downloads NeuroSynth v7 (~100 MB) and computes 5 MKDAChi2 term meta-analyses (§5.10). ~3 min one-shot cost; subsequent atlas loads are instant.

**Resuming after a dropped session:** if your laptop goes offline mid-run, the Modal Volume keeps whatever stimuli committed, and re-running the same command skips any cached `raw_cortical.npy`. For long runs or unstable connections, use `modal run --detach ...` so the GPU job continues even when the local client disconnects.

**Stimulus protocol** (per TRIBEv2.pdf §5.9): **1 s stimulus + 7 s blank = 8 s total**, matching the paper's "one second every eight seconds" trial structure. This gives TRIBE's 1 Hz output T≈8 timepoints, so the collapser indexes `cortical_response[5]` — the §5.8 / Fig 4A hemodynamic peak of a true 1 s onset. Earlier revisions used 10 s sustained stimuli as a workaround for early-pipeline T=1 fallback behavior; Slice 3 reverted to the paper's protocol now that the rest of the pipeline can support it.

## Validation Status

End-to-end validated on 14 IBC-curated stimuli with three per-check statistics:
- **Δ** — `sim(same-category) − sim(cross-category)` inside the NeuroSynth-labeled (§5.10) ICA mask.
- **perm p** — random-mask permutation (n=1000, one-sided): does the ICA mask beat a random same-size cortex subset?
- **BH q** — Benjamini-Hochberg FDR over all `perm p`s in the batch (α=0.05).
- **boot CI** — vertex bootstrap 95% CI on Δ (n=1000). Note: ignores spatial autocorrelation and is narrower than a formal 95% spin-test CI (see `_significance_test` docstring).

**Current state** (post-Slice-3 Modal re-inference on fresh 1 s + 7 s blank tensors; NeuroSynth-labeled ICA masks from Slice 2; FDR from Slice 5 F1):

| # | Check | Δ | perm p | BH q | boot CI | Verdict |
|---|---|---|---|---|---|---|
| 1 | sim(face,face) > sim(face,place)                 | +0.002 | >0.999 | >0.999 | [+0.002, +0.002] | ordering only |
| 2 | sim(place,place) > sim(place,body)               | −0.000 | >0.999 | >0.999 | [−0.000, −0.000] | FAILED (wrong direction) |
| 3 | sim(body,body) > sim(body,face)                  | +0.003 | 0.005 | **0.010** | [+0.003, +0.003] | **ordering + mask-specific** |
| 4 | sim(wc,wc) > sim(wc,place)                       | +0.002 | 0.003 | **0.009** | [+0.002, +0.002] | **ordering + mask-specific** |
| 5 | sim(speech,speech) > sim(speech,non_speech)      | +0.005 | <0.001 | **<0.001** | [+0.003, +0.006] | **ordering + mask-specific** |
| 6 | sim(sentence,sentence) > sim(sentence,word_list) | −0.042 | >0.999 | >0.999 | [−0.05, −0.03] | FAILED (wrong direction) |

**4/6 orderings hold with stable Δ > 0; 3/6 show mask-specific selectivity at BH q<0.05** — the strongest result the project has reported to date.

Key reading: Δs shrank by 2–3 orders of magnitude vs the pre-re-inference 10 s-sustained numbers. That's expected and correct — the 1 s onset response is transient and narrow, so categorical differences only show up as small perturbations on the baseline. The pre-re-inference large Δs came from sustained activation across 5 s of continuous input, which diffuses category information across cortex (hence 0/6 mask-specific then). The tiny-but-mask-specific signature here mirrors the paper's Figure 4E, where 1 s flashes produce focal, category-selective activation that a GLM picks up by averaging many trials — we pick it up in a single trial per stimulus because TRIBE is the GLM now.

The two failing checks:
- `place > body`: Δ ≈ 0. Place and body stimuli produce very similar predicted patterns at t=5; their difference is below TRIBE's single-trial noise floor on the visual mask.
- `sentence > word_list`: Δ = −0.042 (note: bigger in magnitude than most passing checks). On a correctly-labeled language mask (|r|=0.40 with NeuroSynth "language"), two sentences about different topics do not pairwise-correlate more than a sentence/word-list pair. Pairwise Pearson-on-an-ICA-mask isn't the same as the paper's §2.6 magnitude-contrast findings; see `requirements.md` §4.3 for the framing note.

Figure 4E contrast-map replication (`scripts/replicate_figure_4e.py`, §5.9 methodology): 0/4 visual categories localize ≥50 % of top-1 % contrast vertices into the NeuroSynth-labeled visual_system mask. Honest reading: NeuroSynth's "visual" keyword picks up V1/V2 primarily, while category-selective responses (FFA, PPA, EBA, VWFA) live in ventral temporal cortex — which the other ICA components (labeled "motion" / "language" / "DMN" at |r|=0.34–0.47) partially cover. Upgrading the labeling reference beyond five NeuroSynth terms (e.g. Glasser HCP-MMP parcellation) would improve this; currently a documented open item.

3 checks were deliberately dropped (Slice 3):
- `audio_segment > silence` — Santoro clean-sound vs silence is not a §5.9 construction.
- `complex_sentence > simple_sentence` — paper §2.6 finding is a Broca magnitude contrast, not a pairwise similarity.
- `motion_video > static_image` — no 1 s in-silico motion protocol in §5.9.

To re-run: `python scripts/validate_ibc.py --cache-dir <local_cache>`.
To regenerate with post-Modal-re-inference tensors: re-run Modal first, then validate_ibc.
For paper-faithful Figure 4E replication: `python scripts/replicate_figure_4e.py --cache-dir <local_cache>` (direct §5.9 contrast-map methodology).

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
