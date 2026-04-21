# Product Overview

**Cognitive Similarity** is a brain-grounded stimulus similarity system using TRIBE v2 (Transformer for Representations Informed by Brain Encodings).

## Core Functionality

Computes cognitive similarity between stimuli (video, audio, text) by:
1. Generating brain response predictions via TRIBE v2 model
2. Collapsing temporal responses to single cortical vectors
3. Computing Pearson correlation across 5 ICA-derived brain networks
4. Producing similarity profiles and rankings

## Key Features

- **Multi-modal input**: Video, audio, and text stimuli
- **Brain network analysis**: 5 ICA networks (primary auditory cortex, language network, motion detection MT+, default mode network, visual system)
- **Two ICA modes**: Binary mask (top 10% vertices) and continuous weighting
- **Caching system**: Stores raw and collapsed brain responses to avoid re-inference
- **Ranking API**: Compare query stimulus against corpus, rank by similarity

## Use Cases

- Content recommendation based on cognitive similarity
- Stimulus validation for neuroscience research
- Brain-grounded semantic search
