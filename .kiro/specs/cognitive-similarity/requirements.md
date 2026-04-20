# Requirements Document

## Introduction

This feature implements a **cognitive similarity** system that uses TRIBE v2 — Meta FAIR's tri-modal brain encoding foundation model — to measure how similarly two stimuli would be processed by the human brain. Rather than comparing stimuli by semantic embedding cosine similarity, this system grounds similarity in predicted whole-brain fMRI activation patterns. Two stimuli are considered cognitively similar if the brain would respond to them in a similar way.

TRIBE v2 accepts video, audio, and/or text as input. The model has two separate configurations: a cortical model that predicts a `Cortical_Response` of shape `[T, 20,484]` (fsaverage5 surface vertices), and a subcortical model that predicts a `Subcortical_Response` of shape `[T, 8,802]` (Harvard-Oxford atlas voxels). These are separate model runs — `TribeModel.from_pretrained("facebook/tribev2")` loads the cortical model by default; the subcortical model requires a separate configuration (`MaskProjector(mask="subcortical")`). The cognitive similarity system operates exclusively on the Cortical_Response. Cortical similarity measures how similarly two stimuli are *perceptually represented and understood* by the brain — capturing visual features, auditory features, semantic content, and language processing. It does not measure emotional similarity, memorability, or reward value, which are primarily subcortical functions. The system must handle the temporal nature of the cortical output and support both whole-cortex and region-specific similarity queries.

## Why Cortical Similarity, Not Semantic Similarity?

Semantic similarity — as produced by language embedding models — captures what things *mean* in text. Cortical similarity captures what it is *like* for a brain to experience them. These diverge in ways that matter:

- **Content recommendation beyond language**: Two films can be semantically similar (both sci-fi) but cortically very different — one slow and meditative, one fast-paced. The brain responds to visual rhythm, auditory texture, and cognitive load, not just genre labels. Cortical similarity surfaces experiential likeness that semantic similarity misses.

- **Media fatigue and cognitive repetition**: Two ads can be semantically unrelated (cars vs. perfume) but cortically near-identical — same visual pacing, same auditory arc. The brain habituates to *patterns of processing*, not semantic content. Cortical similarity can detect when a media schedule is cognitively repetitive even when content is semantically diverse.

- **Cognitive load equivalence**: A dense academic paper and a complex legal contract are semantically different but may produce nearly identical cortical activation in the language network — both demand the same effortful syntactic processing. Cortical similarity identifies equivalent cognitive demands that semantic similarity cannot.

- **Cross-modal equivalence**: A thunderstorm described in text, heard as audio, and seen as video are semantically identical but perceptually very different. Cortical similarity measures how much the brain's representation actually converges across modalities — something semantic similarity, being unimodal, cannot do.

- **Non-linguistic media**: Semantic similarity breaks down for instrumental music — there is no text to embed. Cortical similarity works regardless: two pieces that activate the auditory cortex and default mode network similarly are being processed similarly by the brain.

---

## Glossary

- **TRIBE_v2**: The tri-modal brain encoding foundation model from Meta FAIR (HuggingFace: `facebook/tribev2`) that predicts whole-brain fMRI responses from video, audio, and/or text stimuli.
- **Brain_Response**: The pair of predicted fMRI activation tensors from two separate TRIBE_v2 model runs: a Cortical_Response of shape `[T, 20,484]` from the default cortical model, and a Subcortical_Response of shape `[T, 8,802]` from a separately configured subcortical model (`MaskProjector(mask="subcortical")`). Only the Cortical_Response is used for similarity computation; the Subcortical_Response is cached for future investigation.
- **Cortical_Response**: The cortical component of a Brain_Response — a predicted fMRI activation tensor of shape `[T, 20,484]`, where T is the number of timepoints and 20,484 dimensions are cortical surface vertices in fsaverage5 space. This is the sole input to all similarity computations.
- **Subcortical_Response**: The subcortical component of a Brain_Response — a predicted fMRI activation tensor of shape `[T, 8,802]` covering Harvard-Oxford atlas voxels (hippocampus, amygdala, thalamus, basal ganglia, ventricles, etc.). Excluded from similarity computation because subcortical regions serve memory encoding, emotional salience, sensory relay, and reward/motor functions rather than perceptual or representational processing, and their predictions are 2–3× lower quality than cortical predictions.
- **Stimulus**: A multimodal input consisting of one or more of: video, audio, or text, passed to TRIBE_v2 independently to obtain a Brain_Response. Each stimulus is run through TRIBE_v2 in isolation — stimuli are never concatenated or batched into a single inference pass, because TRIBE_v2 is context-sensitive (text embeddings use the preceding 1,024 words, video uses the preceding 4 seconds of frames, audio is bidirectional within its window). Running stimuli together would contaminate each response with the context of adjacent stimuli.
- **Stimulus_Isolation_Protocol**: The procedure for presenting a single stimulus to TRIBE_v2 in a way that produces a clean, uncontaminated Brain_Response. Each stimulus is run via its own independent `get_events_dataframe()` + `predict()` call. The TRIBE_v2 API enforces single-stimulus input (`get_events_dataframe()` accepts exactly one path), and `predict()` automatically discards empty segments, so isolation is achieved simply by never combining stimuli into a single call. For images, the stimulus is converted to a 1-second static video (same frame repeated). For text, `get_events_dataframe(text_path=...)` handles TTS conversion and word-timing extraction automatically — no external preprocessing required.
- **TRIBE_v2_API**: The public Python interface for TRIBE v2 inference. `TribeModel.from_pretrained("facebook/tribev2")` loads the model. `model.get_events_dataframe(video_path=, audio_path=, text_path=)` prepares the events dataframe (handling TTS internally for text inputs). `model.predict(events=df)` returns `(preds, segments)` where `preds` has shape `(n_timesteps, n_vertices)` covering the 20,484 fsaverage5 cortical vertices.
- **ICA_Network_Mask**: A continuous vector of 20,484 values derived by running FastICA on TRIBE v2's "unseen subject" projection layer — a matrix of shape `(2048, 20,484)` embedded in the model weights (`best.ckpt`). The 2,048 dimension comes from the `low_rank_head` bottleneck in the model architecture (confirmed in `config.yaml`). Each of the 5 ICA components is a vector over all cortical vertices. The system supports two modes for using these components: (1) **binary mask mode** — threshold at the top 10% of absolute values (~2,048 vertices), consistent with the paper's Figure 6A visualization; (2) **continuous weighting mode** — use the full component vector as per-vertex weights, preserving all information from the ICA decomposition. Binary mask mode is the default.
- **Temporal_Collapsing**: The process of reducing a Cortical_Response timeseries `[T, 20,484]` to a single spatial vector `[20,484]` using either peak-response extraction (for brief stimuli) or GLM-with-HRF fitting (for longer stimuli).
- **Collapsed_Response**: The single spatial vector `[20,484]` produced by Temporal_Collapsing of the Cortical_Response.
- **Pairwise Activation Similarity**: The core computation of this system — computing Pearson correlation between the predicted activation patterns of two stimuli within a given brain region. This directly mirrors the spatial Pearson correlation used by the TRIBE v2 paper (sections 2.5, 2.6, 5.10) to compare predicted maps against ground-truth maps. We apply the same metric to compare two predicted maps against each other. Pearson correlation is preferred over cosine similarity because it is mean-centered, making it robust to differences in overall activation magnitude between stimuli.
- **Cognitive_Similarity_Profile**: The primary output of the system — a structured result containing one Pearson correlation similarity score per ICA_Network (5 scores total), plus a derived whole-cortex summary score. This replaces a single scalar with a cognitively interpretable vector, revealing *which* perceptual/cognitive dimensions drive similarity between two stimuli.
- **Cognitive_Similarity_Score**: A scalar value in `[-1, 1]` representing similarity within a single ICA_Network or across the whole cortex, computed as Pearson correlation between two Collapsed_Responses restricted to that region's vertices. Pearson correlation is preferred over cosine similarity because it is mean-centered, making it robust to differences in overall activation magnitude between stimuli. This is the same metric used by the TRIBE v2 paper for spatial map comparison.
- **Whole_Cortex_Score**: A derived summary scalar computed as the average of the five ICA_Network scores, weighted by the number of vertices in each network. Provided for convenience but secondary to the per-network profile.
- **ROI**: A Region of Interest — a subset of the 20,484 cortical dimensions selected for targeted similarity computation. ROIs may be defined by the Glasser 360-parcel parcellation or by the five canonical ICA networks identified in the TRIBE v2 paper.
- **Glasser_Parcellation**: The Glasser 360-parcel cortical atlas used in the TRIBE v2 paper for region-level analysis, covering the 20,484 cortical vertices.
- **ICA_Network**: One of the five canonical brain networks identified via ICA on TRIBE v2's final layer: (1) Primary Auditory Cortex, (2) Language Network, (3) Motion Detection (MT+), (4) Default Mode Network (DMN), (5) Visual System. These networks are the primary unit of analysis for cognitive similarity.
- **Cortical_Vertex**: One of the 20,484 spatial dimensions in fsaverage5 space corresponding to the cortical surface.
- **Subcortical_Voxel**: One of the 8,802 spatial dimensions corresponding to subcortical brain structures (not used in similarity computation).
- **Peak_Response**: The Collapsed_Response obtained by selecting the timepoint at t+5s after stimulus onset — the hemodynamic peak established by the paper (section 5.9) — used for brief stimuli.
- **GLM_HRF**: A General Linear Model with hemodynamic response function convolution used to estimate the sustained response for longer stimuli, producing a beta-weight vector as the Collapsed_Response. Implemented using `nilearn.glm.first_level.make_first_level_design_matrix` to build the HRF-convolved design matrix, then solved via `numpy.linalg.lstsq` applied directly to the `(T, 20,484)` cortical surface array.
- **Similarity_Request**: A structured input specifying two stimuli and an optional comparison scope (defaults to full per-ICA-network profile).
- **Similarity_Result**: A structured output containing the Cognitive_Similarity_Profile (per-ICA-network scores), the Whole_Cortex_Score, and metadata (temporal collapsing method used for each stimulus, vertex counts per network).

---

## Requirements

### Requirement 1: Brain Response Prediction

**User Story:** As a developer, I want to obtain predicted whole-brain fMRI responses for arbitrary stimuli run in isolation, so that I have clean, uncontaminated neural activation patterns for computing cognitive similarity.

#### Acceptance Criteria

1. WHEN a Stimulus containing at least one of video, audio, or text is provided, THE System SHALL invoke `model.get_events_dataframe()` with the appropriate path argument (`video_path`, `audio_path`, or `text_path`), then call `model.predict(events=df)` on the cortical model to obtain a `preds` tensor of shape `(n_timesteps, 20,484)`.
2. THE System SHALL obtain the Subcortical_Response by running the same Stimulus through a separately configured subcortical model (`MaskProjector(mask="subcortical")`), producing a tensor of shape `(n_timesteps, 8,802)`. Both tensors SHALL be stored in cache; only the cortical tensor is used for similarity computation.
3. THE System SHALL rely on TRIBE_v2's built-in `get_events_dataframe()` for all stimulus preprocessing — including TTS conversion and word-timing extraction for text inputs, and static video conversion for image inputs. No external preprocessing pipeline is required.
4. THE System SHALL never concatenate multiple stimuli into a single TRIBE_v2 inference pass, as the model's context-sensitive embeddings (text: 1,024-word context; video: 4-second preceding frames; audio: bidirectional window) would cause each stimulus response to be contaminated by adjacent stimuli.
5. THE TRIBE_v2 SHALL operate in zero-shot "unseen subject" mode, predicting group-average brain responses without requiring subject-specific calibration data.
6. IF a Stimulus contains none of video, audio, or text, THEN THE System SHALL return a descriptive error indicating that at least one modality is required.

---

### Requirement 2: Temporal Collapsing

**User Story:** As a developer, I want to reduce a Brain_Response timeseries to a single spatial vector, so that I can compute a scalar similarity score between two stimuli.

#### Acceptance Criteria

1. WHEN a Stimulus is classified as brief (duration ≤ 10 seconds), THE Temporal_Collapsing SHALL extract the Peak_Response at the timepoint corresponding to t+5s after stimulus onset.
2. WHEN a Stimulus is classified as long (duration > 10 seconds), THE Temporal_Collapsing SHALL fit a GLM_HRF to the Cortical_Response timeseries and return the resulting beta-weight vector as the Collapsed_Response.
3. THE Temporal_Collapsing SHALL produce a Collapsed_Response of shape `[20,484]` regardless of the input Cortical_Response duration T.
4. IF the Cortical_Response timeseries does not contain a timepoint at t+5s (e.g., stimulus is too short), THEN THE Temporal_Collapsing SHALL fall back to the final available timepoint and log a warning.
5. THE System SHALL expose the temporal collapsing strategy as a configurable parameter, defaulting to automatic selection based on stimulus duration.

---

### Requirement 3: ICA Network Atlas and ROI Masking

**User Story:** As a developer, I want to use TRIBE v2's own ICA network masks to restrict similarity computation to specific brain regions, so that I can measure cognitively meaningful per-network similarity.

#### Acceptance Criteria

1. THE System SHALL compute the five ICA_Network_Masks at initialization by loading `best.ckpt` from HuggingFace (`facebook/tribev2`), extracting the unseen-subject projection layer (shape `2048 × 20,484`, where 2,048 is the `low_rank_head` bottleneck), running `FastICA(n_components=5)`, and thresholding each component at the top 10% of absolute values. Computed masks SHALL be cached locally to avoid recomputation on subsequent runs.
2. THE System SHALL support two ICA network similarity modes:
   - **Binary mask mode** (default): restrict similarity computation to the top 10% of vertices (~2,048) per ICA component, treating all selected vertices equally.
   - **Continuous weighting mode**: use the full ICA component vector (all 20,484 values) as per-vertex weights, so vertices with stronger component association contribute more to the similarity score.
3. THE System SHALL expose the ICA similarity mode as a configurable parameter, defaulting to binary mask mode.
4. THE System SHALL support whole-cortex similarity using all 20,484 dimensions of the Collapsed_Response.
5. THE System SHALL support ICA_Network-masked similarity using any of the five canonical ICA networks: Primary Auditory Cortex, Language Network, Motion Detection (MT+), Default Mode Network, or Visual System.
6. THE System SHALL support ROI-masked similarity using any subset of the 20,484 cortical dimensions defined by a named Glasser_Parcellation parcel (1–360).
7. WHEN an ROI mask is applied, THE System SHALL extract only the dimensions corresponding to that ROI before computing the similarity score.
8. IF a requested ROI name does not match any known Glasser parcel or ICA network, THEN THE System SHALL return a descriptive error listing valid ROI identifiers.
9. THE System SHALL include the vertex count used in the comparison in every Similarity_Result.

---

### Requirement 4: Cognitive Similarity Computation via Per-Network Pearson Correlation

**User Story:** As a developer, I want to compute a per-ICA-network cognitive similarity profile between two stimuli using Pearson correlation of their predicted activation patterns, so that I can understand which cognitive dimensions drive their similarity.

#### Acceptance Criteria

1. WHEN a Similarity_Request specifying two stimuli is provided, THE System SHALL return a Similarity_Result containing a Cognitive_Similarity_Profile with one Pearson correlation score per ICA_Network.
2. FOR EACH ICA_Network, THE System SHALL compute the Pearson correlation between the two Collapsed_Responses restricted to the vertices belonging to that network, producing a score in `[-1, 1]`.
3. THE System SHALL use Pearson correlation (not cosine similarity) as the similarity metric, as it is mean-centered and robust to differences in overall activation magnitude between stimuli. This mirrors the spatial Pearson correlation used by the TRIBE v2 paper for map comparison (sections 2.5, 2.6, 5.10).
4. THE System SHALL compute a Whole_Cortex_Score as the vertex-count-weighted average of the five ICA_Network scores and include it in the Similarity_Result as a secondary summary.
5. THE System SHALL include in the Similarity_Result: the per-network profile, the whole-cortex summary score, the temporal collapsing method used for each stimulus, and the vertex count per ICA_Network used in each comparison.
6. IF either Collapsed_Response restricted to a given ICA_Network has zero variance, THE System SHALL return a score of 0.0 for that network and include a warning in the Similarity_Result.
7. THE System SHALL support batch Similarity_Requests comparing one stimulus against a list of N stimuli, returning N Similarity_Results in the same order as the input list.
8. THE System SHALL also support targeted single-network queries where only one ICA_Network score is requested, returning a single Cognitive_Similarity_Score for that network.

---

### Requirement 5: Optional Glasser Parcel-Level Similarity

**User Story:** As a developer, I want to optionally compute similarity at the finer Glasser 360-parcel level, so that I can investigate specific brain areas beyond the five ICA networks.

#### Acceptance Criteria

1. THE System SHALL support ROI-masked similarity using any subset of the 20,484 cortical dimensions defined by a named Glasser_Parcellation parcel (1–360).
2. WHEN a Glasser parcel ROI is requested, THE System SHALL compute Pearson correlation between the two Collapsed_Responses restricted to that parcel's vertices.
3. IF a requested ROI name does not match any known Glasser parcel or ICA network, THEN THE System SHALL return a descriptive error listing valid ROI identifiers.
4. THE System SHALL include the vertex count of the requested parcel in the Similarity_Result.

---

### Requirement 6: Validation Against Known Ground Truths

**User Story:** As a developer, I want to verify that the cognitive similarity system produces results consistent with known neuroscientific findings from the TRIBE v2 paper, so that I can trust the system's outputs.

#### Acceptance Criteria

1. THE System SHALL include a validation suite using publicly available IBC stimuli from https://github.com/individual-brain-charting/public_protocols/tree/master/FaceBody/stimuli that tests per-network similarity scores against expected orderings derived from the paper's in-silico experiments.
2. WHEN the validation suite is executed, THE System SHALL report whether each expected ordering holds (pair A scores higher than pair B for the specified ICA_Network).
3. THE System SHALL validate the following expected orderings, each grounded in a specific finding from the TRIBE v2 paper:

   **Visual System ICA network** (section 2.5 — FFA, PPA, EBA, VWFA selectivity):
   - sim(face, face) > sim(face, place) — FFA is face-selective
   - sim(place, place) > sim(place, body) — PPA is place-selective
   - sim(body, body) > sim(body, face) — EBA is body-selective
   - sim(written_character, written_character) > sim(written_character, place) — VWFA is written-word selective

   **Primary Auditory Cortex ICA network** (section 2.6 — Bang/Audio tasks):
   - sim(speech, speech) > sim(speech, non_speech) — auditory cortex responds selectively to speech
   - sim(audio_segment, audio_segment) > sim(audio_segment, silence) — any sound activates auditory cortex more than silence

   **Language Network ICA network** (section 2.6 — RSVP/EmotionalPain tasks):
   - sim(sentence, sentence) > sim(sentence, word_list) — sentences engage full syntactic processing; word lists do not
   - sim(complex_sentence, complex_sentence) > sim(complex_sentence, simple_sentence) — complex syntax drives Broca's area more strongly

   **Motion Detection (MT+) ICA network** (section 2.5 — Visu task):
   - sim(motion_video, motion_video) > sim(motion_video, static_image) — MT+ is motion-selective; static images produce minimal MT+ activation

4. THE System SHALL NOT define expected orderings for the Default Mode Network, as the paper does not run a direct DMN localizer experiment that would provide ground truth for pairwise similarity ordering.
5. WHEN the validation suite is executed, THE System SHALL report a pass/fail result per ordering and a summary count of how many orderings hold.

---

### Requirement 7: Serialization and Caching of Collapsed Responses

**User Story:** As a developer, I want to serialize and reload Collapsed_Responses, so that I can avoid re-running TRIBE v2 inference for stimuli I have already processed.

#### Acceptance Criteria

1. THE System SHALL serialize a Collapsed_Response to a file format that preserves the full `[20,484]` float32 tensor without loss.
2. THE System SHALL deserialize a Collapsed_Response from a previously serialized file and produce a tensor numerically identical to the original.
3. FOR ALL valid Collapsed_Responses, serializing then deserializing SHALL produce a tensor equal to the original within float32 precision (round-trip property).
4. THE System SHALL associate each serialized Collapsed_Response with a content-addressed identifier derived from the Stimulus (e.g., a hash of the input modality data) to enable cache lookup.
5. IF a Collapsed_Response for a given Stimulus is already cached, THEN THE System SHALL load it from cache rather than re-running TRIBE_v2 inference.
6. THE Pretty_Printer SHALL format Similarity_Results as human-readable JSON, including all metadata fields defined in Requirement 5.

---

### Requirement 8: Ranked Similarity

**User Story:** As a developer, I want to rank a corpus of stimuli by their cognitive similarity to a query stimulus, so that I can see which stimuli are most and least similar to it across each cognitive dimension.

#### Acceptance Criteria

1. WHEN a Similarity_Request specifies a query stimulus and a corpus of N stimuli, THE System SHALL return a ranked list of all N stimuli ordered by their Cognitive_Similarity_Score to the query, from most similar to least similar.
2. THE System SHALL produce one ranked list per ICA_Network (5 lists total), plus one ranked list for the Whole_Cortex_Score, so that rankings can be compared across cognitive dimensions.
3. EACH entry in the ranked list SHALL include: the stimulus identifier, its Cognitive_Similarity_Score for that network, and its rank position (1 = most similar).
4. THE System SHALL support ranking by a single specified ICA_Network when a full per-network ranking is not required.
5. THE System SHALL handle ties in Cognitive_Similarity_Score by assigning the same rank to tied stimuli.
6. IF the corpus contains fewer than 2 stimuli, THE System SHALL return a descriptive error indicating that at least 2 stimuli are required for ranking.

---

## References

Key resources for implementation and further research. These should be consulted when making design decisions or resolving ambiguities.

### Primary Source
- **TRIBE v2 Paper**: d'Ascoli et al. (2026). *A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience.* March 25, 2026. — The authoritative source for all model architecture, training, and in-silico experiment methodology decisions.

### Model Artifacts and Code
- **HuggingFace Model + Weights**: https://huggingface.co/facebook/tribev2 — Model weights (`best.ckpt`), config, and README with API usage. ICA network masks are computed from these weights at initialization.
- **GitHub Repository**: https://github.com/facebookresearch/tribev2 — Full source code including `demo_utils.py` (TribeModel inference API), `utils_fmri.py` (surface projection and ROI analysis), and `tribe_demo.ipynb` (Colab walkthrough).
- **Interactive Demo**: https://aidemos.atmeta.com/tribev2 — Live demo with brain visualizations for exploring predicted responses.

### Neuroscience Methodology
- **Representational Similarity Analysis (RSA)**: Kriegeskorte et al. (2008). *Representational similarity analysis — connecting the branches of systems neuroscience.* Frontiers in Systems Neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/ — Background context for pairwise activation pattern comparison using Pearson correlation. Note: we use the core Pearson correlation metric from this framework but do not implement full RSA (no RDMs or second-order comparisons).
- **Glasser 360-Parcel Parcellation**: Glasser et al. (2016). *A multi-modal parcellation of human cerebral cortex.* Nature, 536, 171–178. — The cortical atlas used by TRIBE v2 for parcel-level analysis.
- **NeuroSynth (for ICA network validation)**: Kent et al. (2026). *NeuroSynth Compose.* https://neurosynth.org — Meta-analysis tool used by the paper to validate ICA network components against known functional maps.

### Validation Stimuli
- **IBC FaceBody/Visu Stimuli** (visual localizer images — faces, places, bodies, written characters): https://github.com/individual-brain-charting/public_protocols/tree/master/FaceBody/stimuli
- **IBC Full Public Protocols** (all tasks including language experiments — Bang, Audio, EmotionalPain, RSVP): https://github.com/individual-brain-charting/public_protocols

### External Tools Used by TRIBE v2
- **pocket-tts** (TTS pipeline): https://github.com/kyutai-labs/pocket-tts — Used in the paper's in-silico language experiments. Note: the actual `TribeModel` API uses `gTTS` (Google Text-to-Speech) internally via the `TextToEvents` class in `demo_utils.py`.
- **WhisperX** (word timing extraction): https://github.com/m-bain/whisperX — Used internally by `get_events_dataframe()` to obtain word-level timestamps from audio.
- **nilearn** (neuroimaging analysis): https://nilearn.github.io — Used for GLM+HRF fitting (`FirstLevelModel`) in temporal collapsing for longer stimuli.
