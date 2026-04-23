"""Tests for TRIBEv2.pdf §5.9 contrast-map replication.

Uses synthetic responses so tests don't depend on HuggingFace or NeuroSynth.

Requirements: TRIBEv2.pdf §5.9 ("Contrast maps").
"""
from __future__ import annotations

import numpy as np
import pytest

from cognitive_similarity.models import ICANetwork
from cognitive_similarity.paper_replication import (
    _PEAK_INDEX,
    ContrastMapResult,
    compute_all_contrasts,
    compute_contrast_map,
    localize_contrast_maps,
)

_VERTICES = 20484


# ----------------------------------------------------------------------
# compute_contrast_map — §5.9 formula
# ----------------------------------------------------------------------


def test_contrast_map_zero_when_category_equals_others() -> None:
    """If the category response exactly equals the others' mean, contrast
    must be zero — the §5.9 "X − mean(others)" construction.
    """
    rng = np.random.default_rng(0)
    mean_vec = rng.standard_normal(_VERTICES).astype(np.float32)
    cmap = compute_contrast_map(mean_vec, [mean_vec, mean_vec, mean_vec])
    np.testing.assert_allclose(cmap, 0.0, atol=1e-6)


def test_contrast_map_subtracts_average_of_others() -> None:
    """Algebraic check: contrast = category − mean(others) vertex-wise."""
    rng = np.random.default_rng(1)
    cat = rng.standard_normal(_VERTICES).astype(np.float32)
    a = rng.standard_normal(_VERTICES).astype(np.float32)
    b = rng.standard_normal(_VERTICES).astype(np.float32)
    c = rng.standard_normal(_VERTICES).astype(np.float32)

    cmap = compute_contrast_map(cat, [a, b, c])

    expected = cat - (a + b + c) / 3.0
    np.testing.assert_allclose(cmap, expected.astype(np.float32), atol=1e-6)


def test_contrast_map_rejects_empty_others() -> None:
    """Cannot compute a contrast with no other-category baselines."""
    with pytest.raises(ValueError, match="at least one"):
        compute_contrast_map(np.zeros(_VERTICES), [])


# ----------------------------------------------------------------------
# compute_all_contrasts — end-to-end across multiple stimuli per category
# ----------------------------------------------------------------------


def test_compute_all_contrasts_identifies_category_specific_vertex() -> None:
    """Plant a category-specific spike in one synthetic response and verify
    compute_all_contrasts's peak_vertex lands on that spike.
    """
    T, N = 8, _VERTICES
    rng = np.random.default_rng(2)

    # Baseline response: small random signal at t=5
    def baseline(T=T):
        r = rng.standard_normal((T, N)).astype(np.float32) * 0.01
        return r

    # Plant a large face-specific signal at vertex 12345 at t=5
    face_responses = []
    for _ in range(2):
        r = baseline()
        r[_PEAK_INDEX, 12345] += 5.0  # strong face-specific peak
        face_responses.append(r)

    # Three other categories with only baseline noise
    place_responses = [baseline() for _ in range(2)]
    body_responses = [baseline() for _ in range(2)]
    word_responses = [baseline() for _ in range(2)]

    contrasts = compute_all_contrasts({
        "face": face_responses,
        "place": place_responses,
        "body": body_responses,
        "word": word_responses,
    })

    assert contrasts["face"].peak_vertex == 12345
    # Peak value should be ~5 (minus a tiny fraction from the "others" noise mean)
    assert contrasts["face"].peak_value == pytest.approx(5.0, abs=0.1)

    # Non-face categories must not peak at 12345 (the planted-face vertex)
    for cat in ("place", "body", "word"):
        assert contrasts[cat].peak_vertex != 12345
        # contrast at 12345 for non-face categories should be strongly negative
        # (the others' mean at that vertex is dominated by face's +5 signal)
        assert contrasts[cat].contrast_map[12345] < -1.0


def test_compute_all_contrasts_falls_back_when_T_too_short(caplog) -> None:
    """T<6 → use T-1 as the peak index and log WARNING (matches
    collapsing.py fallback behavior).
    """
    import logging
    rng = np.random.default_rng(3)
    T = 2
    resps = {
        "face":  [rng.standard_normal((T, _VERTICES)).astype(np.float32)],
        "place": [rng.standard_normal((T, _VERTICES)).astype(np.float32)],
    }
    with caplog.at_level(logging.WARNING,
                         logger="cognitive_similarity.paper_replication"):
        out = compute_all_contrasts(resps)
    assert any("Peak index" in r.message for r in caplog.records)
    assert out["face"].contrast_map.shape == (_VERTICES,)


# ----------------------------------------------------------------------
# localize_contrast_maps — assigns top-1% → ICA-mask fractions
# ----------------------------------------------------------------------


class _FakeAtlas:
    """Stand-in for ICANetworkAtlas exposing only what localize_* needs."""

    NETWORKS = [
        ICANetwork.PRIMARY_AUDITORY_CORTEX,
        ICANetwork.LANGUAGE_NETWORK,
        ICANetwork.MOTION_DETECTION_MT_PLUS,
        ICANetwork.DEFAULT_MODE_NETWORK,
        ICANetwork.VISUAL_SYSTEM,
    ]

    def __init__(self) -> None:
        # Each mask owns 2048 disjoint contiguous vertices (non-overlapping
        # for a clean test — real ICA masks may overlap).
        self._masks = {
            n: np.zeros(_VERTICES, dtype=bool) for n in self.NETWORKS
        }
        BAND = 2048
        for i, n in enumerate(self.NETWORKS):
            self._masks[n][i * BAND : (i + 1) * BAND] = True

    def get_mask(self, network: ICANetwork) -> np.ndarray:
        return self._masks[network].copy()


def test_localize_fractions_sum_to_one_when_masks_partition_cortex() -> None:
    """With disjoint masks covering 10240 of the 20484 vertices, the top-1%
    fractions across masks should be bounded and land where the signal is.
    """
    atlas = _FakeAtlas()

    # Build a contrast whose top-1% is entirely inside VISUAL_SYSTEM
    # (vertices [8192, 10240)).
    cmap = np.zeros(_VERTICES, dtype=np.float32)
    cmap[8192:10240] = 10.0  # strong signal in visual_system band
    result = ContrastMapResult(
        category="synthetic", contrast_map=cmap, peak_vertex=8192,
        peak_value=10.0,
    )
    localize_contrast_maps({"synthetic": result}, atlas, top_percentile=0.01)

    fr = result.top_pct_fraction_per_network
    assert fr[ICANetwork.VISUAL_SYSTEM] == pytest.approx(1.0)
    for other in atlas.NETWORKS:
        if other is not ICANetwork.VISUAL_SYSTEM:
            assert fr[other] == 0.0


def test_localize_fractions_reflect_partial_overlap() -> None:
    """When top-1% is split between two masks (e.g., signal straddles
    visual and language bands), fractions should reflect the split.
    """
    atlas = _FakeAtlas()

    # Put signal half in LANGUAGE (band [2048, 4096)) and half in VISUAL
    # (band [8192, 10240)).
    cmap = np.zeros(_VERTICES, dtype=np.float32)
    # top-1% = ceil(20484 * 0.01) = 205 vertices
    cmap[3900:3900 + 100] = 5.0    # inside LANGUAGE
    cmap[9800:9800 + 105] = 5.0    # inside VISUAL
    result = ContrastMapResult(
        category="split", contrast_map=cmap, peak_vertex=3900, peak_value=5.0,
    )
    localize_contrast_maps({"split": result}, atlas, top_percentile=0.01)

    fr = result.top_pct_fraction_per_network
    # Each half should contribute ~ its count / 205
    assert fr[ICANetwork.LANGUAGE_NETWORK] == pytest.approx(100 / 205, abs=0.02)
    assert fr[ICANetwork.VISUAL_SYSTEM] == pytest.approx(105 / 205, abs=0.02)
