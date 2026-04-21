"""CognitiveSimilarity — top-level facade orchestrating all components."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from cognitive_similarity.cache import ResponseCache
from cognitive_similarity.collapsing import TemporalCollapser
from cognitive_similarity.ica_atlas import ICANetworkAtlas
from cognitive_similarity.models import (
    ICAMode,
    ICANetwork,
    RankedEntry,
    RankedResult,
    SimilarityResult,
    Stimulus,
)
from cognitive_similarity.similarity_engine import SimilarityEngine

log = logging.getLogger(__name__)


def _ensure_stimulus_id(stimulus: Stimulus) -> str:
    """Return stimulus_id, auto-generating one if None."""
    if stimulus.stimulus_id is not None:
        return stimulus.stimulus_id
    return str(uuid.uuid4())


def _rank_entries(entries: list[tuple[str, float]]) -> list[RankedEntry]:
    """Sort entries descending by score with tie handling (equal scores share rank)."""
    sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
    result: list[RankedEntry] = []
    rank = 1
    for i, (sid, score) in enumerate(sorted_entries):
        if i > 0 and score < sorted_entries[i - 1][1]:
            rank = i + 1
        result.append(RankedEntry(stimulus_id=sid, score=score, rank=rank))
    return result


class CognitiveSimilarity:
    """Top-level facade for cognitive similarity computation.

    Reads collapsed responses from cache (raw tensors must be pre-computed
    via Colab and stored in the cache before calling compare/rank locally).
    """

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        cache_dir: Optional[str] = None,
        _atlas: Optional[ICANetworkAtlas] = None,
    ) -> None:
        self._cache = ResponseCache(cache_dir or ".")
        self._collapser = TemporalCollapser()

        if _atlas is not None:
            self._atlas = _atlas
        else:
            self._atlas = ICANetworkAtlas(model_id=model_id, cache_dir=cache_dir)

        self._engine = SimilarityEngine(self._atlas, ica_mode=ICAMode.BINARY_MASK)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_collapsed_response(self, stimulus: Stimulus) -> np.ndarray:
        """Return the collapsed cortical response [20484] for a stimulus.

        Flow:
        1. Check cache for collapsed response → return if found.
        2. Check cache for raw cortical → collapse, store, return if found.
        3. Raise RuntimeError if neither is cached.
        """
        # Step 1: collapsed already cached?
        collapsed = self._cache.get_collapsed(stimulus)
        if collapsed is not None:
            log.debug("Cache hit (collapsed) for stimulus %s", stimulus.stimulus_id)
            return collapsed

        # Step 2: raw cortical cached?
        raw = self._cache.get_raw(stimulus)
        if raw is not None:
            raw_cortical, _ = raw
            log.debug("Cache hit (raw) for stimulus %s — collapsing", stimulus.stimulus_id)
            collapsed = self._collapser.collapse(raw_cortical, stimulus)
            self._cache.put_collapsed(stimulus, collapsed)
            return collapsed

        # Step 3: nothing cached
        raise RuntimeError(
            f"No cached response found for stimulus '{stimulus.stimulus_id}'. "
            "Raw cortical tensors must be pre-computed (e.g. via Colab) and stored "
            "in the cache before calling compare/rank locally."
        )

    def compare(
        self,
        stimulus_a: Stimulus,
        stimulus_b: Stimulus,
        ica_mode: ICAMode = ICAMode.BINARY_MASK,
    ) -> SimilarityResult:
        """Compare two stimuli and return a SimilarityResult.

        The returned profile always contains scores for all 5 ICA networks,
        plus a whole-cortex score computed as the vertex-count-weighted
        average of those 5 networks (Req 4.1, 4.4). This is not configurable.
        For a single-network query use ``rank(..., network=...)`` or the
        lower-level ``SimilarityEngine.compute_network_score()``.

        Parameters
        ----------
        stimulus_a, stimulus_b:
            Stimuli to compare. Their collapsed responses must be in cache.
        ica_mode:
            ICA mode to use for similarity computation.
        """
        id_a = _ensure_stimulus_id(stimulus_a)
        id_b = _ensure_stimulus_id(stimulus_b)

        collapsed_a = self.get_collapsed_response(stimulus_a)
        collapsed_b = self.get_collapsed_response(stimulus_b)

        profile = self._engine.compute_profile(collapsed_a, collapsed_b, ica_mode=ica_mode)

        return SimilarityResult(
            profile=profile,
            stimulus_a_id=id_a,
            stimulus_b_id=id_b,
        )

    def rank(
        self,
        query: Stimulus,
        corpus: list[Stimulus],
        network: Optional[ICANetwork] = None,
    ) -> RankedResult:
        """Rank corpus stimuli by cognitive similarity to query.

        Parameters
        ----------
        query:
            The query stimulus.
        corpus:
            List of stimuli to rank. Must have at least 2 entries.
        network:
            If provided, rank by this single network's score.
            If None, rank by all networks + whole cortex.

        Raises
        ------
        ValueError if corpus has fewer than 2 stimuli.
        """
        if len(corpus) < 2:
            raise ValueError(
                f"corpus must contain at least 2 stimuli for ranking, got {len(corpus)}."
            )

        query_id = _ensure_stimulus_id(query)

        results = [self.compare(query, s) for s in corpus]

        # Build per-network ranked lists
        if network is not None:
            networks_to_rank = [network]
        else:
            networks_to_rank = list(ICANetwork)

        rankings_by_network: dict[ICANetwork, list[RankedEntry]] = {}
        for net in networks_to_rank:
            entries = [
                (r.stimulus_b_id, r.profile.network_scores[net].score)
                for r in results
            ]
            rankings_by_network[net] = _rank_entries(entries)

        # Whole-cortex ranking
        whole_cortex_entries = [
            (r.stimulus_b_id, r.profile.whole_cortex_score)
            for r in results
        ]
        rankings_whole_cortex = _rank_entries(whole_cortex_entries)

        return RankedResult(
            query_id=query_id,
            rankings_by_network=rankings_by_network,
            rankings_whole_cortex=rankings_whole_cortex,
        )

