# Product Overview

**Cognitive Similarity** is a brain-grounded stimulus similarity system using TRIBE v2 (Transformer for Representations Informed by Brain Encodings).

## Core Functionality

Computes cognitive similarity between stimuli (video, audio, text) by:
1. Generating brain response predictions via TRIBE v2 model.
2. Collapsing temporal responses to single cortical vectors at the t+5 s hemodynamic peak (TRIBEv2.pdf §5.8).
3. Computing pairwise Pearson correlation across 5 ICA-derived brain networks (labels assigned via §5.10 NeuroSynth bipartite matching, not positionally).
4. Producing similarity profiles and rankings.

## Key Features

- **Multi-modal input**: Video, audio, and text stimuli (1 s + 7 s blank protocol per §5.9).
- **Brain network analysis**: 5 ICA networks (primary auditory cortex, language network, motion detection MT+, default mode network, visual system), each labeled by spatial correlation against a NeuroSynth keyword map.
- **Two ICA modes**: Binary mask (top 10% vertices, default) and continuous weighting (standard weighted Pearson over all 20,484 vertices).
- **Caching system**: Stores raw and collapsed brain responses (content-addressed SHA-256) to avoid re-inference; ICA masks + assignment + provenance cached at `<cache>/ica_masks.npz`.
- **Ranking API**: Compare query stimulus against corpus, rank by similarity per-network + whole-cortex.
- **Paper-faithful replication**: `scripts/replicate_figure_4e.py` runs the §5.9 Figure 4E contrast-map methodology as a direct paper validation, separate from the novel pairwise-similarity metric.

## Use Cases

- Content recommendation based on cognitive similarity
- Stimulus validation for neuroscience research
- Brain-grounded semantic search
