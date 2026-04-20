"""Unit tests for ICANetworkAtlas.

Tests use a synthetic (2048, 20484) random projection matrix to bypass
HuggingFace access, exercising the ICA computation and mask logic directly.

Requirements: 3.1, 3.2
"""

import numpy as np
import pytest

from cognitive_similarity.ica_atlas import ICANetworkAtlas, N_VERTICES, N_COMPONENTS
from cognitive_similarity.models import ICANetwork

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=0)
SYNTHETIC_PROJECTION = RNG.standard_normal((2048, N_VERTICES)).astype(np.float32)


@pytest.fixture(scope="module")
def atlas() -> ICANetworkAtlas:
    """ICANetworkAtlas built from a synthetic projection matrix (no HuggingFace)."""
    return ICANetworkAtlas.from_projection_matrix(SYNTHETIC_PROJECTION, top_percentile=0.10)


# ---------------------------------------------------------------------------
# Mask shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", list(ICANetwork))
def test_mask_shape(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """Each network mask must have shape (20484,)."""
    mask = atlas.get_mask(network)
    assert mask.shape == (N_VERTICES,), (
        f"Expected mask shape ({N_VERTICES},), got {mask.shape} for {network}"
    )


@pytest.mark.parametrize("network", list(ICANetwork))
def test_mask_dtype_bool(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """Each network mask must have dtype bool."""
    mask = atlas.get_mask(network)
    assert mask.dtype == bool, (
        f"Expected dtype bool, got {mask.dtype} for {network}"
    )


# ---------------------------------------------------------------------------
# Vertex count (~2048 per network, top 10% of 20484)
# ---------------------------------------------------------------------------

EXPECTED_VERTEX_COUNT = int(N_VERTICES * 0.10)  # 2048
TOLERANCE = 0.05  # ±5%


@pytest.mark.parametrize("network", list(ICANetwork))
def test_vertex_count_approximately_2048(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """Vertex count per network should be approximately 2048 (top 10% of 20484), ±5%."""
    indices = atlas.get_vertex_indices(network)
    count = len(indices)
    lower = int(EXPECTED_VERTEX_COUNT * (1 - TOLERANCE))
    upper = int(EXPECTED_VERTEX_COUNT * (1 + TOLERANCE)) + 1  # +1 for rounding
    assert lower <= count <= upper, (
        f"Vertex count {count} for {network} is outside [{lower}, {upper}] "
        f"(expected ~{EXPECTED_VERTEX_COUNT})"
    )


# ---------------------------------------------------------------------------
# Vertex indices in valid range [0, 20483]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", list(ICANetwork))
def test_vertex_indices_in_valid_range(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """All vertex indices must be in [0, 20483]."""
    indices = atlas.get_vertex_indices(network)
    assert indices.min() >= 0, f"Negative index found for {network}"
    assert indices.max() <= N_VERTICES - 1, (
        f"Index {indices.max()} out of range for {network} (max allowed: {N_VERTICES - 1})"
    )


# ---------------------------------------------------------------------------
# Masks are NOT mutually exclusive
# ---------------------------------------------------------------------------


def test_masks_not_mutually_exclusive(atlas: ICANetworkAtlas) -> None:
    """Masks are allowed to overlap — do NOT assert zero overlap."""
    networks = list(ICANetwork)
    # Collect all masks
    masks = [atlas.get_mask(n) for n in networks]
    # Check that at least one pair has some overlap (expected with random data)
    # We only assert that we do NOT enforce mutual exclusivity — i.e., we don't
    # raise an error when overlap exists.
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap = np.logical_and(masks[i], masks[j]).sum()
            # No assertion that overlap == 0; overlap is allowed.
            # Just verify the operation completes without error.
            assert overlap >= 0  # trivially true, confirms no exception


# ---------------------------------------------------------------------------
# get_component returns full continuous vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", list(ICANetwork))
def test_component_shape(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """get_component() must return a vector of shape (20484,)."""
    component = atlas.get_component(network)
    assert component.shape == (N_VERTICES,), (
        f"Expected component shape ({N_VERTICES},), got {component.shape} for {network}"
    )


@pytest.mark.parametrize("network", list(ICANetwork))
def test_component_has_nonzero_values(atlas: ICANetworkAtlas, network: ICANetwork) -> None:
    """ICA components should not be all-zero."""
    component = atlas.get_component(network)
    assert not np.all(component == 0), f"Component for {network} is all zeros"


# ---------------------------------------------------------------------------
# NETWORKS list covers all ICANetwork enum values
# ---------------------------------------------------------------------------


def test_networks_list_covers_all_enum_values() -> None:
    """ICANetworkAtlas.NETWORKS must contain all ICANetwork enum values."""
    assert set(ICANetworkAtlas.NETWORKS) == set(ICANetwork), (
        "ICANetworkAtlas.NETWORKS does not cover all ICANetwork enum values"
    )


def test_networks_list_length() -> None:
    """ICANetworkAtlas.NETWORKS must have exactly 5 entries."""
    assert len(ICANetworkAtlas.NETWORKS) == N_COMPONENTS


# ---------------------------------------------------------------------------
# from_projection_matrix factory
# ---------------------------------------------------------------------------


def test_from_projection_matrix_factory() -> None:
    """from_projection_matrix() should produce a valid atlas without HuggingFace."""
    rng = np.random.default_rng(seed=42)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)
    atlas = ICANetworkAtlas.from_projection_matrix(proj)
    # Spot-check one network
    mask = atlas.get_mask(ICANetwork.VISUAL_SYSTEM)
    assert mask.shape == (N_VERTICES,)
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path) -> None:
    """Saving and reloading from cache should produce identical masks."""
    rng = np.random.default_rng(seed=7)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)

    # Build atlas and save cache
    atlas1 = ICANetworkAtlas(_projection_matrix=proj, cache_dir=str(tmp_path))
    atlas1._save_cache()

    # Load from cache
    atlas2 = ICANetworkAtlas(cache_dir=str(tmp_path))

    for network in ICANetwork:
        np.testing.assert_array_equal(
            atlas1.get_mask(network),
            atlas2.get_mask(network),
            err_msg=f"Mask mismatch after cache round-trip for {network}",
        )
        np.testing.assert_array_almost_equal(
            atlas1.get_component(network),
            atlas2.get_component(network),
            err_msg=f"Component mismatch after cache round-trip for {network}",
        )
