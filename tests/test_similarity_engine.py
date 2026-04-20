"""Tests for SimilarityEngine — property-based and unit tests.

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.ica_atlas import ICANetworkAtlas, N_VERTICES
from cognitive_similarity.models import ICAMode, ICANetwork
from cognitive_similarity.similarity_engine import SimilarityEngine, pearson_correlation

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Use a small projection matrix for speed; still (2048, 20484)
_RNG = np.random.default_rng(seed=99)
_SYNTHETIC_PROJECTION = _RNG.standard_normal((2048, N_VERTICES)).astype(np.float32)


@pytest.fixture(scope="module")
def atlas() -> ICANetworkAtlas:
    """ICANetworkAtlas built from a synthetic projection matrix (no HuggingFace)."""
    return ICANetworkAtlas.from_projection_matrix(_SYNTHETIC_PROJECTION, top_percentile=0.10)


@pytest.fixture(scope="module")
def engine_binary(atlas: ICANetworkAtlas) -> SimilarityEngine:
    return SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)


@pytest.fixture(scope="module")
def engine_continuous(atlas: ICANetworkAtlas) -> SimilarityEngine:
    return SimilarityEngine(atlas, ica_mode=ICAMode.CONTINUOUS_WEIGHTS)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def collapsed_response_strategy(seed: int | None = None):
    """Generate a (20484,) float32 array with non-trivial variance."""
    return st.integers(min_value=0, max_value=2**31 - 1).map(
        lambda s: np.random.default_rng(s).standard_normal(N_VERTICES).astype(np.float32)
    )


def collapsed_response_pair_strategy():
    """Generate a pair of (20484,) float32 arrays."""
    return st.tuples(collapsed_response_strategy(), collapsed_response_strategy())


# ---------------------------------------------------------------------------
# Property 5: ICA Network Masking Isolation
# Feature: cognitive-similarity, Property 5: ICA Network Masking Isolation
# ---------------------------------------------------------------------------

@given(
    network=st.sampled_from(list(ICANetwork)),
    pair=collapsed_response_pair_strategy(),
    noise_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property_5_ica_network_masking_isolation(
    atlas: ICANetworkAtlas,
    network: ICANetwork,
    pair: tuple[np.ndarray, np.ndarray],
    noise_seed: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 5: ICA Network Masking Isolation
    Validates: Requirements 3.5, 3.6

    When binary mask mode is used, the score for network N only uses vertices
    in that network's mask. Modifying vertices outside the mask should not
    change the score.
    """
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    response_a, response_b = pair

    score_original = engine.compute_network_score(
        response_a, response_b, network, ica_mode=ICAMode.BINARY_MASK
    )

    # Perturb vertices OUTSIDE the mask
    mask = atlas.get_mask(network)
    outside_indices = np.where(~mask)[0]

    rng = np.random.default_rng(noise_seed)
    response_a_perturbed = response_a.copy()
    response_b_perturbed = response_b.copy()
    response_a_perturbed[outside_indices] += rng.standard_normal(len(outside_indices)).astype(np.float32) * 100.0
    response_b_perturbed[outside_indices] += rng.standard_normal(len(outside_indices)).astype(np.float32) * 100.0

    score_perturbed = engine.compute_network_score(
        response_a_perturbed, response_b_perturbed, network, ica_mode=ICAMode.BINARY_MASK
    )

    assert score_original == pytest.approx(score_perturbed, abs=1e-5), (
        f"Score changed after perturbing vertices outside {network.value} mask: "
        f"{score_original} vs {score_perturbed}"
    )


# ---------------------------------------------------------------------------
# Property 6: Invalid ROI Rejection
# Feature: cognitive-similarity, Property 6: Invalid ROI Rejection
# ---------------------------------------------------------------------------

@given(
    invalid_value=st.one_of(
        st.text(min_size=1).filter(lambda s: s not in [n.value for n in ICANetwork]),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=None)
def test_property_6_invalid_roi_rejection(
    atlas: ICANetworkAtlas,
    invalid_value,
) -> None:
    """
    # Feature: cognitive-similarity, Property 6: Invalid ROI Rejection
    Validates: Requirements 3.7

    Passing a non-ICANetwork value raises ValueError listing valid identifiers.
    """
    engine = SimilarityEngine(atlas)
    rng = np.random.default_rng(0)
    response_a = rng.standard_normal(N_VERTICES).astype(np.float32)
    response_b = rng.standard_normal(N_VERTICES).astype(np.float32)

    with pytest.raises(ValueError) as exc_info:
        engine.compute_network_score(response_a, response_b, invalid_value)

    error_msg = str(exc_info.value)
    # The error message must list valid identifiers
    for network in ICANetwork:
        assert network.value in error_msg, (
            f"Valid identifier '{network.value}' not listed in error: {error_msg}"
        )


# ---------------------------------------------------------------------------
# Property 7: Continuous ICA Weight Normalization
# Feature: cognitive-similarity, Property 7: Continuous ICA Weight Normalization
# ---------------------------------------------------------------------------

@given(
    network=st.sampled_from(list(ICANetwork)),
)
@settings(max_examples=100, deadline=None)
def test_property_7_continuous_ica_weight_normalization(
    atlas: ICANetworkAtlas,
    network: ICANetwork,
) -> None:
    """
    # Feature: cognitive-similarity, Property 7: Continuous ICA Weight Normalization
    Validates: Requirements 3.2

    In continuous weighting mode, w = abs(component) / abs(component).sum()
    must sum to 1.0 within float32 tolerance.
    """
    component = atlas.get_component(network)
    w = np.abs(component)
    w_normalized = w / w.sum()

    assert abs(w_normalized.sum() - 1.0) < 1e-5, (
        f"Weights for {network.value} do not sum to 1.0: sum={w_normalized.sum()}"
    )


# ---------------------------------------------------------------------------
# Property 8: Cognitive Similarity Profile Structure and Score Range
# Feature: cognitive-similarity, Property 8: Cognitive Similarity Profile Structure and Score Range
# ---------------------------------------------------------------------------

@given(pair=collapsed_response_pair_strategy())
@settings(max_examples=100, deadline=None)
def test_property_8_profile_structure_and_score_range(
    atlas: ICANetworkAtlas,
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    # Feature: cognitive-similarity, Property 8: Cognitive Similarity Profile Structure and Score Range
    Validates: Requirements 4.1, 4.2

    compute_profile() returns exactly 5 NetworkScore entries (one per ICANetwork),
    and all scores are in [-1, 1].
    """
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    response_a, response_b = pair

    profile = engine.compute_profile(response_a, response_b)

    # Exactly 5 entries
    assert len(profile.network_scores) == 5, (
        f"Expected 5 network scores, got {len(profile.network_scores)}"
    )

    # One entry per ICANetwork enum value
    assert set(profile.network_scores.keys()) == set(ICANetwork), (
        "network_scores keys do not match ICANetwork enum values"
    )

    # All scores in [-1, 1]
    for network, ns in profile.network_scores.items():
        assert -1.0 <= ns.score <= 1.0, (
            f"Score {ns.score} for {network.value} is outside [-1, 1]"
        )


# ---------------------------------------------------------------------------
# Property 9: Whole-Cortex Score Is Vertex-Count-Weighted Average
# Feature: cognitive-similarity, Property 9: Whole-Cortex Score Is Vertex-Count-Weighted Average
# ---------------------------------------------------------------------------

@given(pair=collapsed_response_pair_strategy())
@settings(max_examples=100, deadline=None)
def test_property_9_whole_cortex_score_formula(
    atlas: ICANetworkAtlas,
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    # Feature: cognitive-similarity, Property 9: Whole-Cortex Score Is Vertex-Count-Weighted Average
    Validates: Requirements 4.4

    whole_cortex_score == sum(score_i * vertex_count_i) / sum(vertex_count_i)
    """
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    response_a, response_b = pair

    profile = engine.compute_profile(response_a, response_b)

    total_vertices = sum(ns.vertex_count for ns in profile.network_scores.values())
    expected_whole_cortex = sum(
        ns.score * ns.vertex_count / total_vertices
        for ns in profile.network_scores.values()
    )

    assert profile.whole_cortex_score == pytest.approx(expected_whole_cortex, abs=1e-5), (
        f"whole_cortex_score {profile.whole_cortex_score} != "
        f"expected {expected_whole_cortex}"
    )


# ---------------------------------------------------------------------------
# Unit tests for SimilarityEngine (sub-task 6.6)
# ---------------------------------------------------------------------------

def test_zero_variance_returns_zero_with_warning(
    atlas: ICANetworkAtlas,
) -> None:
    """
    Zero-variance input → score 0.0 with warning set in NetworkScore.
    Validates: Requirements 4.6
    """
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)

    # Constant vector has zero variance
    response_a = np.ones(N_VERTICES, dtype=np.float32)
    response_b = np.random.default_rng(0).standard_normal(N_VERTICES).astype(np.float32)

    profile = engine.compute_profile(response_a, response_b)

    for network, ns in profile.network_scores.items():
        assert ns.score == 0.0, (
            f"Expected score 0.0 for zero-variance input on {network.value}, got {ns.score}"
        )
        assert ns.warning is not None, (
            f"Expected warning for zero-variance input on {network.value}"
        )
        assert "zero variance" in ns.warning.lower(), (
            f"Warning '{ns.warning}' does not mention 'zero variance'"
        )


def test_binary_and_continuous_modes_produce_different_scores(
    atlas: ICANetworkAtlas,
) -> None:
    """
    Binary mask mode and continuous weighting mode should produce different scores
    for the same stimulus pair (since they use different vertex subsets/weights).
    Validates: Requirements 3.2, 3.3
    """
    rng = np.random.default_rng(42)
    response_a = rng.standard_normal(N_VERTICES).astype(np.float32)
    response_b = rng.standard_normal(N_VERTICES).astype(np.float32)

    engine_bin = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    engine_cont = SimilarityEngine(atlas, ica_mode=ICAMode.CONTINUOUS_WEIGHTS)

    profile_bin = engine_bin.compute_profile(response_a, response_b)
    profile_cont = engine_cont.compute_profile(response_a, response_b)

    # At least one network should differ between modes
    any_different = any(
        profile_bin.network_scores[n].score != profile_cont.network_scores[n].score
        for n in ICANetwork
    )
    assert any_different, (
        "Binary mask and continuous weighting modes produced identical scores for all networks"
    )


def test_single_network_query_returns_float_in_range(
    atlas: ICANetworkAtlas,
) -> None:
    """
    compute_network_score() returns a float in [-1, 1].
    Validates: Requirements 4.8
    """
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    rng = np.random.default_rng(7)
    response_a = rng.standard_normal(N_VERTICES).astype(np.float32)
    response_b = rng.standard_normal(N_VERTICES).astype(np.float32)

    for network in ICANetwork:
        score = engine.compute_network_score(response_a, response_b, network)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert -1.0 <= score <= 1.0, (
            f"Score {score} for {network.value} is outside [-1, 1]"
        )


def test_pearson_correlation_known_values() -> None:
    """Pearson correlation of identical vectors should be 1.0."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert pearson_correlation(v, v) == pytest.approx(1.0, abs=1e-6)


def test_pearson_correlation_opposite_vectors() -> None:
    """Pearson correlation of a vector and its negation should be -1.0."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert pearson_correlation(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_pearson_correlation_zero_variance() -> None:
    """Zero-variance input returns 0.0."""
    a = np.ones(10, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    assert pearson_correlation(a, b) == 0.0
    assert pearson_correlation(b, a) == 0.0


def test_ica_mode_override_in_compute_profile(atlas: ICANetworkAtlas) -> None:
    """ica_mode parameter in compute_profile() overrides the instance default."""
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    rng = np.random.default_rng(13)
    response_a = rng.standard_normal(N_VERTICES).astype(np.float32)
    response_b = rng.standard_normal(N_VERTICES).astype(np.float32)

    profile_bin = engine.compute_profile(response_a, response_b, ica_mode=ICAMode.BINARY_MASK)
    profile_cont = engine.compute_profile(response_a, response_b, ica_mode=ICAMode.CONTINUOUS_WEIGHTS)

    assert profile_bin.ica_mode is ICAMode.BINARY_MASK
    assert profile_cont.ica_mode is ICAMode.CONTINUOUS_WEIGHTS


def test_ica_mode_override_in_compute_network_score(atlas: ICANetworkAtlas) -> None:
    """ica_mode parameter in compute_network_score() overrides the instance default."""
    engine = SimilarityEngine(atlas, ica_mode=ICAMode.BINARY_MASK)
    rng = np.random.default_rng(17)
    response_a = rng.standard_normal(N_VERTICES).astype(np.float32)
    response_b = rng.standard_normal(N_VERTICES).astype(np.float32)

    score_bin = engine.compute_network_score(
        response_a, response_b, ICANetwork.VISUAL_SYSTEM, ica_mode=ICAMode.BINARY_MASK
    )
    score_cont = engine.compute_network_score(
        response_a, response_b, ICANetwork.VISUAL_SYSTEM, ica_mode=ICAMode.CONTINUOUS_WEIGHTS
    )

    # They should differ (different vertex subsets/weights)
    assert score_bin != score_cont
