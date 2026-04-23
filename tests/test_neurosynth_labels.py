"""Tests for NeuroSynth-based ICA component labeling.

Tests use synthetic components + synthetic "term maps" to exercise the
assignment logic without any NeuroSynth or HuggingFace dependency.

Requirements: 3.1 (paper §5.10 labeling procedure)
"""
from __future__ import annotations

import numpy as np
import pytest

from cognitive_similarity.models import ICANetwork
from cognitive_similarity.neurosynth_labels import (
    NEUROSYNTH_TERMS,
    compute_label_assignment,
)


N_VERTICES = 20484
N_COMPONENTS = 5

_ORDERED_NETWORKS = [
    ICANetwork.PRIMARY_AUDITORY_CORTEX,
    ICANetwork.LANGUAGE_NETWORK,
    ICANetwork.MOTION_DETECTION_MT_PLUS,
    ICANetwork.DEFAULT_MODE_NETWORK,
    ICANetwork.VISUAL_SYSTEM,
]


def _random_components(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_COMPONENTS, N_VERTICES)).astype(np.float32)


def _term_maps_from_components(
    components: np.ndarray,
    permutation: list[int],
    sign_flips: list[int],
    noise_std: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build synthetic NeuroSynth term maps that match a known ground-truth
    assignment. Each term map = component at ``permutation[i]`` + noise,
    optionally sign-flipped.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for i, network in enumerate(_ORDERED_NETWORKS):
        src = components[permutation[i]].copy()
        if sign_flips[i] < 0:
            src = -src
        src = src + rng.standard_normal(N_VERTICES).astype(np.float32) * noise_std
        out[network.value] = src
    return out


# ----------------------------------------------------------------------
# NEUROSYNTH_TERMS dict covers every ICANetwork
# ----------------------------------------------------------------------


def test_neurosynth_terms_covers_all_networks() -> None:
    """NEUROSYNTH_TERMS must have one entry per ICANetwork enum value."""
    assert set(NEUROSYNTH_TERMS.keys()) == set(ICANetwork)


def test_neurosynth_terms_match_paper_keywords() -> None:
    """§5.10 enumerates exactly these five terms; guard against typos."""
    expected = {
        ICANetwork.PRIMARY_AUDITORY_CORTEX: "primary auditory",
        ICANetwork.LANGUAGE_NETWORK:        "language",
        ICANetwork.MOTION_DETECTION_MT_PLUS: "motion",
        ICANetwork.DEFAULT_MODE_NETWORK:    "default network",
        ICANetwork.VISUAL_SYSTEM:           "visual",
    }
    assert NEUROSYNTH_TERMS == expected


# ----------------------------------------------------------------------
# compute_label_assignment — recovers known permutations
# ----------------------------------------------------------------------


def test_assignment_recovers_identity_permutation() -> None:
    """With term maps equal to components (+noise), assignment must be
    identity: network i → component i for all i.
    """
    components = _random_components(seed=1)
    # Build maps that exactly match the positional order with no sign flips
    term_maps = _term_maps_from_components(
        components,
        permutation=[0, 1, 2, 3, 4],
        sign_flips=[+1, +1, +1, +1, +1],
        noise_std=0.01,
    )

    adjusted, assignment, correlations = compute_label_assignment(
        components, term_maps
    )

    for i, network in enumerate(_ORDERED_NETWORKS):
        assert assignment[network] == i, (
            f"{network.value} assigned to comp {assignment[network]}, expected {i}"
        )
        assert correlations[network] > 0.9, (
            f"{network.value} |r|={correlations[network]} is too low for clean synthetic data"
        )


def test_assignment_recovers_permuted_order() -> None:
    """Shuffle the term maps; assignment must recover the ground-truth
    permutation via bipartite matching.
    """
    components = _random_components(seed=2)
    # Ground truth: network i should map to component permutation[i]
    permutation = [3, 0, 4, 1, 2]
    term_maps = _term_maps_from_components(
        components,
        permutation=permutation,
        sign_flips=[+1] * 5,
        noise_std=0.01,
    )

    _, assignment, _ = compute_label_assignment(components, term_maps)

    for i, network in enumerate(_ORDERED_NETWORKS):
        assert assignment[network] == permutation[i], (
            f"{network.value} assigned to {assignment[network]}, expected {permutation[i]}"
        )


def test_assignment_handles_sign_flips() -> None:
    """When a term map is anti-correlated with its matching component,
    compute_label_assignment must (a) still assign correctly, (b) flip
    the component's sign in the returned adjusted-components array, and
    (c) report a positive |r| in ``correlations``.
    """
    components = _random_components(seed=3)
    # Match identity but flip sign on networks 0 and 2
    term_maps = _term_maps_from_components(
        components,
        permutation=[0, 1, 2, 3, 4],
        sign_flips=[-1, +1, -1, +1, +1],
        noise_std=0.01,
    )

    adjusted, assignment, correlations = compute_label_assignment(
        components, term_maps
    )

    # Assignment is identity
    for i, network in enumerate(_ORDERED_NETWORKS):
        assert assignment[network] == i

    # All reported |r| are positive (sign was flipped during assignment)
    for network, r in correlations.items():
        assert r > 0.9, f"{network.value} should have positive |r|, got {r}"

    # Components 0 and 2 were sign-flipped, 1/3/4 unchanged
    assert np.allclose(adjusted[0], -components[0])
    assert np.allclose(adjusted[1], components[1])
    assert np.allclose(adjusted[2], -components[2])
    assert np.allclose(adjusted[3], components[3])
    assert np.allclose(adjusted[4], components[4])


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


def test_assignment_raises_on_weak_match() -> None:
    """When components are uncorrelated with term maps, all |r| fall below
    the error threshold — must RuntimeError rather than cache garbage.
    """
    components = _random_components(seed=4)
    # Term maps are an independent random matrix with no relation to components
    rng = np.random.default_rng(seed=999)
    term_maps = {
        network.value: rng.standard_normal(N_VERTICES).astype(np.float32)
        for network in _ORDERED_NETWORKS
    }

    with pytest.raises(RuntimeError, match="below error threshold"):
        compute_label_assignment(components, term_maps)


def test_assignment_rejects_wrong_component_count() -> None:
    """compute_label_assignment requires exactly N_COMPONENTS=5 rows."""
    components = np.random.default_rng(5).standard_normal(
        (4, N_VERTICES)  # wrong count
    ).astype(np.float32)
    term_maps = {
        network.value: components[0] + 0.01 * np.random.default_rng(6).standard_normal(N_VERTICES).astype(np.float32)
        for network in _ORDERED_NETWORKS
    }

    with pytest.raises(ValueError, match="components.shape"):
        compute_label_assignment(components, term_maps)


def test_assignment_rejects_missing_term() -> None:
    """compute_label_assignment requires every ICANetwork to have a term map."""
    components = _random_components(seed=7)
    incomplete_maps = {
        network.value: components[i]
        for i, network in enumerate(_ORDERED_NETWORKS[:-1])  # missing VISUAL_SYSTEM
    }

    with pytest.raises(KeyError, match="visual_system"):
        compute_label_assignment(components, incomplete_maps)
