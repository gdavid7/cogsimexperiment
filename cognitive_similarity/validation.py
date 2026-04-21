"""ValidationSuite — runs 9 expected-ordering checks against cached IBC stimuli."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from cognitive_similarity.cache import ResponseCache
from cognitive_similarity.models import (
    ICANetwork,
    ValidationCheck,
    ValidationReport,
)
from cognitive_similarity.similarity_engine import SimilarityEngine

log = logging.getLogger(__name__)


class ValidationSuite:
    """Runs 9 expected-ordering checks against cached IBC stimuli.

    Loads stimuli by looking up their content hashes in manifest.json,
    retrieves their Collapsed_Response from ResponseCache directly by hash
    (bypassing _content_hash since we already have the hash from the manifest),
    then computes and compares similarity scores.
    """

    def __init__(
        self,
        engine: SimilarityEngine,
        cache: ResponseCache,
        manifest_path: str,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._manifest_path = Path(manifest_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ValidationReport:
        """Run all 9 validation checks and return a ValidationReport."""
        manifest = self._load_manifest()
        # Build lookup: stimulus_id -> content_hash
        id_to_hash: dict[str, str] = {
            entry["stimulus_id"]: entry["content_hash"]
            for entry in manifest
        }

        checks: list[ValidationCheck] = []

        # --- Visual System (4 checks) ---
        checks.append(self._check(
            description="sim(face, face) > sim(face, place)",
            network=ICANetwork.VISUAL_SYSTEM,
            pair_a_ids=("face_01", "face_02"),
            pair_b_ids=("face_01", "place_01"),
            id_to_hash=id_to_hash,
        ))
        checks.append(self._check(
            description="sim(place, place) > sim(place, body)",
            network=ICANetwork.VISUAL_SYSTEM,
            pair_a_ids=("place_01", "place_02"),
            pair_b_ids=("place_01", "body_01"),
            id_to_hash=id_to_hash,
        ))
        checks.append(self._check(
            description="sim(body, body) > sim(body, face)",
            network=ICANetwork.VISUAL_SYSTEM,
            pair_a_ids=("body_01", "body_02"),
            pair_b_ids=("body_01", "face_01"),
            id_to_hash=id_to_hash,
        ))
        checks.append(self._check(
            description="sim(written_character, written_character) > sim(written_character, place)",
            network=ICANetwork.VISUAL_SYSTEM,
            pair_a_ids=("written_character_01", "written_character_02"),
            pair_b_ids=("written_character_01", "place_01"),
            id_to_hash=id_to_hash,
        ))

        # --- Primary Auditory Cortex (2 checks) ---
        checks.append(self._check(
            description="sim(speech, speech) > sim(speech, non_speech)",
            network=ICANetwork.PRIMARY_AUDITORY_CORTEX,
            pair_a_ids=("speech_01", "speech_02"),
            pair_b_ids=("speech_01", "non_speech_01"),
            id_to_hash=id_to_hash,
        ))
        checks.append(self._check(
            description="sim(audio_segment, audio_segment) > sim(audio_segment, silence)",
            network=ICANetwork.PRIMARY_AUDITORY_CORTEX,
            pair_a_ids=("audio_segment_01", "audio_segment_02"),
            pair_b_ids=("audio_segment_01", "silence_01"),
            id_to_hash=id_to_hash,
        ))

        # --- Language Network (2 checks) ---
        checks.append(self._check(
            description="sim(sentence, sentence) > sim(sentence, word_list)",
            network=ICANetwork.LANGUAGE_NETWORK,
            pair_a_ids=("sentence_01", "sentence_02"),
            pair_b_ids=("sentence_01", "word_list_01"),
            id_to_hash=id_to_hash,
        ))
        checks.append(self._check(
            description="sim(complex_sentence, complex_sentence) > sim(complex_sentence, simple_sentence)",
            network=ICANetwork.LANGUAGE_NETWORK,
            pair_a_ids=("complex_sentence_01", "complex_sentence_02"),
            pair_b_ids=("complex_sentence_01", "simple_sentence_01"),
            id_to_hash=id_to_hash,
        ))

        # --- Motion Detection MT+ (1 check) ---
        checks.append(self._check(
            description="sim(motion_video, motion_video) > sim(motion_video, static_image)",
            network=ICANetwork.MOTION_DETECTION_MT_PLUS,
            pair_a_ids=("motion_video_01", "motion_video_02"),
            pair_b_ids=("motion_video_01", "static_image_01"),
            id_to_hash=id_to_hash,
        ))

        passed = sum(1 for c in checks if c.passed)
        return ValidationReport(checks=checks, passed=passed, total=len(checks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Load and parse manifest.json."""
        with open(self._manifest_path, "r") as f:
            return json.load(f)

    def _get_response(
        self, stimulus_id: str, id_to_hash: dict[str, str]
    ) -> Optional[np.ndarray]:
        """Retrieve collapsed response by stimulus_id via content hash lookup."""
        content_hash = id_to_hash.get(stimulus_id)
        if content_hash is None:
            log.warning("Stimulus '%s' not found in manifest", stimulus_id)
            return None
        response = self._cache.get_collapsed_by_hash(content_hash)
        if response is None:
            log.warning(
                "No collapsed response in cache for stimulus '%s' (hash=%s)",
                stimulus_id,
                content_hash,
            )
        return response

    def _compute_similarity(
        self,
        response_a: np.ndarray,
        response_b: np.ndarray,
        network: ICANetwork,
    ) -> float:
        """Compute network similarity score between two collapsed responses."""
        return self._engine.compute_network_score(response_a, response_b, network)

    def _check(
        self,
        description: str,
        network: ICANetwork,
        pair_a_ids: tuple[str, str],
        pair_b_ids: tuple[str, str],
        id_to_hash: dict[str, str],
    ) -> ValidationCheck:
        """Build a single ValidationCheck, handling missing stimuli gracefully."""
        id_a1, id_a2 = pair_a_ids
        id_b1, id_b2 = pair_b_ids

        resp_a1 = self._get_response(id_a1, id_to_hash)
        resp_a2 = self._get_response(id_a2, id_to_hash)
        resp_b1 = self._get_response(id_b1, id_to_hash)
        resp_b2 = self._get_response(id_b2, id_to_hash)

        missing = [
            sid
            for sid, resp in [(id_a1, resp_a1), (id_a2, resp_a2), (id_b1, resp_b1), (id_b2, resp_b2)]
            if resp is None
        ]

        if missing:
            desc = f"{description} [MISSING: {', '.join(missing)}]"
            log.warning("Check '%s' skipped — missing stimuli: %s", description, missing)
            return ValidationCheck(
                description=desc,
                network=network,
                pair_a=pair_a_ids,
                pair_b=pair_b_ids,
                expected="sim(pair_a) > sim(pair_b)",
                passed=False,
                score_a=0.0,
                score_b=0.0,
            )

        score_a = self._compute_similarity(resp_a1, resp_a2, network)
        score_b = self._compute_similarity(resp_b1, resp_b2, network)

        return ValidationCheck(
            description=description,
            network=network,
            pair_a=pair_a_ids,
            pair_b=pair_b_ids,
            expected="sim(pair_a) > sim(pair_b)",
            passed=score_a > score_b,
            score_a=score_a,
            score_b=score_b,
        )
