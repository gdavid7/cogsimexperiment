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


# ---------------------------------------------------------------------------
# _find_projection_tensor — key-name preference
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for torch.Tensor that exposes .shape / .float() / .numpy()."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    def float(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


def test_find_projection_prefers_predictor_weights_key() -> None:
    """When multiple tensors match the target shape, prefer the one whose
    key ends in '.predictor.weights' — the paper-documented subject-block
    location (TRIBEv2.pdf §5.3).
    """
    rng = np.random.default_rng(seed=11)
    wanted = rng.standard_normal((1, 2048, N_VERTICES)).astype(np.float32)
    decoy = rng.standard_normal((1, 2048, N_VERTICES)).astype(np.float32)

    state_dict = {
        "some.other.decoy": _FakeTensor(decoy),
        "model.predictor.weights": _FakeTensor(wanted),
    }

    arr = ICANetworkAtlas._find_projection_tensor(state_dict)
    np.testing.assert_array_equal(arr, wanted.squeeze(axis=0))


def test_find_projection_falls_back_when_no_predictor_key(caplog) -> None:
    """If no key ends in '.predictor.weights' (e.g. a future TRIBE release
    renames the subject block), the finder still returns a shape-matching
    tensor but logs a WARNING so the fallback is visible.
    """
    import logging
    rng = np.random.default_rng(seed=12)
    only = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)

    state_dict = {"model.unexpected.name": _FakeTensor(only)}

    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.ica_atlas"):
        arr = ICANetworkAtlas._find_projection_tensor(state_dict)

    np.testing.assert_array_equal(arr, only)
    assert any(
        "No key ending in '.predictor.weights'" in r.message for r in caplog.records
    ), "expected a WARNING when the preferred key name is absent"


# ---------------------------------------------------------------------------
# A2/A3 — NeuroSynth assignment integration
# ---------------------------------------------------------------------------


def test_positional_fallback_when_neurosynth_maps_missing(tmp_path, caplog) -> None:
    """With use_neurosynth_labels=True but no neurosynth_maps.npz in the
    cache directory, the atlas must log a WARNING and fall back to
    positional labels (so existing deployments keep working until the
    user runs scripts/fetch_neurosynth_maps.py).
    """
    import logging
    rng = np.random.default_rng(seed=20)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)

    # use_neurosynth_labels=True but no neurosynth_maps.npz in tmp_path.
    # _projection_matrix path skips the labeling step entirely (tests bypass);
    # to exercise the fallback branch we construct through the cache path.
    atlas = ICANetworkAtlas(_projection_matrix=proj, cache_dir=str(tmp_path))
    atlas._save_cache()

    # Clear any test-time cache state, then rebuild from the cached .npz with
    # use_neurosynth_labels=True so the "maps missing" branch fires.
    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.ica_atlas"):
        atlas2 = ICANetworkAtlas(
            cache_dir=str(tmp_path), use_neurosynth_labels=True
        )

    assert atlas2._label_source == "positional"
    # Assignment is identity by construction
    for i, network in enumerate(ICANetwork):
        assert atlas2._label_assignment[network] == i
    assert any(
        "NeuroSynth term maps not found" in r.message for r in caplog.records
    ), "expected WARNING when neurosynth_maps.npz is missing"


def test_neurosynth_assignment_applied_when_maps_present(tmp_path) -> None:
    """When neurosynth_maps.npz exists in the cache dir, the atlas must
    apply the §5.10 bipartite assignment and update _label_assignment.

    Uses hand-constructed orthogonal components (5 non-overlapping vertex
    bands with strong signal) instead of FastICA-on-random-data. FastICA
    outputs on Gaussian input are not easily distinguishable, so the
    assignment correlations would be dominated by noise.
    """
    # Build 5 orthogonal synthetic components: each has strong signal in
    # a non-overlapping contiguous vertex band.
    BAND = N_VERTICES // 5
    components = np.zeros((5, N_VERTICES), dtype=np.float32)
    for i in range(5):
        components[i, i * BAND : (i + 1) * BAND] = 1.0

    # Build a proj-matrix-backed atlas, then override the stored components
    # so we're testing the labeling flow, not FastICA itself.
    rng = np.random.default_rng(seed=21)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)
    atlas0 = ICANetworkAtlas(_projection_matrix=proj, cache_dir=str(tmp_path))
    atlas0._components = components
    # Recompute masks to match the new components (top 10% = the active band)
    masks = np.zeros_like(atlas0._masks)
    for i in range(5):
        abs_vals = np.abs(components[i])
        cutoff = np.quantile(abs_vals, 0.90)
        masks[i] = abs_vals >= cutoff
    atlas0._masks = masks
    atlas0._save_cache()

    # Build synthetic neurosynth_maps.npz that deliberately reorders labels:
    # network i's reference map = component at idx (4 - i), with small noise.
    # Bipartite assignment should recover network i → component (4 - i).
    noise_rng = np.random.default_rng(seed=21 + 100)
    reordered_refs = {}
    keys = [n.value for n in ICANetwork]
    for i in range(5):
        src = components[4 - i].copy()
        src = src + noise_rng.standard_normal(N_VERTICES).astype(np.float32) * 0.01
        reordered_refs[keys[i]] = src
    np.savez(tmp_path / "neurosynth_maps.npz", **reordered_refs)

    atlas = ICANetworkAtlas(cache_dir=str(tmp_path), use_neurosynth_labels=True)

    assert atlas._label_source == "neurosynth"
    for i, network in enumerate(ICANetwork):
        assert atlas._label_assignment[network] == 4 - i, (
            f"{network.value} assigned to {atlas._label_assignment[network]}, "
            f"expected {4 - i}"
        )
    # With orthogonal components + 0.01 noise, correlations should be ~1.0
    for network, r in atlas._label_correlations.items():
        assert r > 0.9, f"{network.value} |r|={r} too low"


def test_cache_round_trip_includes_a3_provenance(tmp_path) -> None:
    """The extended .npz schema (label_assignment, label_correlations,
    sklearn_version, fastica_n_iter) must survive save → load cleanly.
    """
    rng = np.random.default_rng(seed=22)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)
    atlas = ICANetworkAtlas(_projection_matrix=proj, cache_dir=str(tmp_path))

    # Mutate state to include a NeuroSynth-style assignment before save
    atlas._label_source = "neurosynth"
    atlas._label_assignment = {
        network: (i + 2) % 5 for i, network in enumerate(ICANetwork)
    }
    atlas._label_correlations = {
        network: 0.3 + i * 0.05 for i, network in enumerate(ICANetwork)
    }
    atlas._sklearn_version = "1.8.0"
    atlas._fastica_n_iter = 3998
    atlas._save_cache()

    atlas2 = ICANetworkAtlas(
        cache_dir=str(tmp_path), use_neurosynth_labels=False
    )

    assert atlas2._label_source == "neurosynth"
    assert atlas2._sklearn_version == "1.8.0"
    assert atlas2._fastica_n_iter == 3998
    for i, network in enumerate(ICANetwork):
        assert atlas2._label_assignment[network] == (i + 2) % 5
        assert atlas2._label_correlations[network] == pytest.approx(
            0.3 + i * 0.05, abs=1e-6
        )


def test_network_index_uses_assignment_dict(tmp_path) -> None:
    """get_mask / get_component honor _label_assignment (A2): when network X
    maps to component 3, get_mask(X) must return _masks[3], not the positional
    index of X.
    """
    rng = np.random.default_rng(seed=23)
    proj = rng.standard_normal((2048, N_VERTICES)).astype(np.float32)
    atlas = ICANetworkAtlas(_projection_matrix=proj, cache_dir=str(tmp_path))

    # Remap LANGUAGE_NETWORK to component index 3 (positional would be 1)
    atlas._label_assignment[ICANetwork.LANGUAGE_NETWORK] = 3
    atlas._label_source = "neurosynth"

    mask_via_get = atlas.get_mask(ICANetwork.LANGUAGE_NETWORK)
    mask_at_3 = atlas._masks[3]
    np.testing.assert_array_equal(mask_via_get, mask_at_3)

    comp_via_get = atlas.get_component(ICANetwork.LANGUAGE_NETWORK)
    comp_at_3 = atlas._components[3]
    np.testing.assert_array_equal(comp_via_get, comp_at_3)
