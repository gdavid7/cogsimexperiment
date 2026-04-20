"""Tests for ResponseCache — serialization round-trip and content hash properties."""

# Feature: cognitive-similarity, Property 12: Collapsed Response Serialization Round-Trip
# Feature: cognitive-similarity, Property 13: Content Hash Determinism and Uniqueness

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.cache import ResponseCache
from cognitive_similarity.models import Stimulus


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid Collapsed_Response: shape [20484], dtype float32.
# We use numpy arrays directly via st.integers for the random seed, then
# generate the array deterministically — this avoids Hypothesis's list-size limit.
collapsed_response_strategy = st.integers(min_value=0, max_value=2**31 - 1).map(
    lambda seed: np.random.default_rng(seed).random(20484).astype(np.float32) * 2e6 - 1e6
)

# Strategy for arbitrary file byte content (non-empty to ensure a real hash)
file_bytes_strategy = st.binary(min_size=1, max_size=65536 * 3)


# ---------------------------------------------------------------------------
# Property 12: Collapsed Response Serialization Round-Trip
# ---------------------------------------------------------------------------

@given(collapsed=collapsed_response_strategy)
@settings(max_examples=100)
def test_property_12_collapsed_response_serialization_round_trip(
    collapsed: np.ndarray,
) -> None:
    """
    **Validates: Requirements 6.1, 6.2, 6.3**

    For ALL valid Collapsed_Responses (shape [20484] float32), serializing then
    deserializing SHALL produce a tensor equal to the original within float32 precision.
    """
    assert collapsed.shape == (20484,)
    assert collapsed.dtype == np.float32

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a real temp file so _content_hash can open it
        stimulus_file = os.path.join(tmp_dir, "stimulus.bin")
        with open(stimulus_file, "wb") as f:
            f.write(b"test-stimulus-content")

        stimulus = Stimulus(audio_path=stimulus_file)
        cache = ResponseCache(tmp_dir)

        cache.put_collapsed(stimulus, collapsed)
        loaded = cache.get_collapsed(stimulus)

    assert loaded is not None, "get_collapsed should return the stored tensor"
    assert loaded.shape == (20484,), f"Expected shape (20484,), got {loaded.shape}"
    assert loaded.dtype == np.float32, f"Expected float32, got {loaded.dtype}"

    # Round-trip must be exact for float32 (numpy.save/.load preserves float32 exactly)
    np.testing.assert_array_equal(
        loaded,
        collapsed,
        err_msg="Round-trip serialization must be exact for float32 tensors",
    )


# ---------------------------------------------------------------------------
# Property 13: Content Hash Determinism and Uniqueness
# ---------------------------------------------------------------------------

@given(content=file_bytes_strategy)
@settings(max_examples=100)
def test_property_13_content_hash_determinism(content: bytes) -> None:
    """
    **Validates: Requirements 6.4**

    The same stimulus file content always produces the same hash (determinism).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)

        # Write the same content to two separate files
        path_a = os.path.join(tmp_dir, "stim_a.bin")
        path_b = os.path.join(tmp_dir, "stim_b.bin")
        with open(path_a, "wb") as f:
            f.write(content)
        with open(path_b, "wb") as f:
            f.write(content)

        stimulus_a = Stimulus(audio_path=path_a)
        stimulus_b = Stimulus(audio_path=path_b)

        hash_a1 = cache._content_hash(stimulus_a)
        hash_a2 = cache._content_hash(stimulus_a)  # same stimulus, called twice
        hash_b = cache._content_hash(stimulus_b)   # different file, same content

    assert hash_a1 == hash_a2, "Same stimulus must always produce the same hash"
    assert hash_a1 == hash_b, "Same file content must produce the same hash regardless of path"


@given(content_a=file_bytes_strategy, content_b=file_bytes_strategy)
@settings(max_examples=100)
def test_property_13_content_hash_uniqueness(
    content_a: bytes, content_b: bytes
) -> None:
    """
    **Validates: Requirements 6.4**

    Different file contents produce different hashes (collision resistance).
    """
    # Only test when contents actually differ
    if content_a == content_b:
        return  # skip — same content, same hash is expected

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)

        path_a = os.path.join(tmp_dir, "stim_a.bin")
        path_b = os.path.join(tmp_dir, "stim_b.bin")
        with open(path_a, "wb") as f:
            f.write(content_a)
        with open(path_b, "wb") as f:
            f.write(content_b)

        stimulus_a = Stimulus(audio_path=path_a)
        stimulus_b = Stimulus(audio_path=path_b)

        hash_a = cache._content_hash(stimulus_a)
        hash_b = cache._content_hash(stimulus_b)

    assert hash_a != hash_b, (
        f"Different file contents must produce different hashes. "
        f"Got collision: {hash_a}"
    )


# ---------------------------------------------------------------------------
# Unit tests for ResponseCache
# ---------------------------------------------------------------------------

def test_get_collapsed_returns_none_when_not_cached(tmp_path):
    """Cache miss returns None."""
    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"hello")
    stimulus = Stimulus(audio_path=str(stim_file))
    assert cache.get_collapsed(stimulus) is None


def test_get_raw_returns_none_when_not_cached(tmp_path):
    """Cache miss for raw tensors returns None."""
    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"hello")
    stimulus = Stimulus(audio_path=str(stim_file))
    assert cache.get_raw(stimulus) is None


def test_put_get_raw_round_trip(tmp_path):
    """put_raw / get_raw preserves cortical and subcortical tensors."""
    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"raw-test")
    stimulus = Stimulus(audio_path=str(stim_file))

    rng = np.random.default_rng(42)
    cortical = rng.random((5, 20484), dtype=np.float32)
    subcortical = rng.random((5, 8802), dtype=np.float32)

    cache.put_raw(stimulus, cortical, subcortical)
    result = cache.get_raw(stimulus)

    assert result is not None
    loaded_cortical, loaded_subcortical = result
    np.testing.assert_array_equal(loaded_cortical, cortical)
    np.testing.assert_array_equal(loaded_subcortical, subcortical)


def test_get_collapsed_returns_none_on_corrupted_file(tmp_path, caplog):
    """Corrupted .npy file logs WARNING and returns None."""
    import logging

    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"corrupt-test")
    stimulus = Stimulus(audio_path=str(stim_file))

    # Write a valid collapsed first so the directory exists
    valid = np.zeros(20484, dtype=np.float32)
    cache.put_collapsed(stimulus, valid)

    # Overwrite with garbage
    h = cache._content_hash(stimulus)
    corrupt_path = tmp_path / "tensors" / h / "collapsed.npy"
    corrupt_path.write_bytes(b"not a numpy file at all")

    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.cache"):
        result = cache.get_collapsed(stimulus)

    assert result is None
    assert any("corrupted" in r.message.lower() or "corrupt" in r.message.lower() for r in caplog.records)


def test_get_collapsed_returns_none_on_wrong_shape(tmp_path, caplog):
    """Wrong-shape .npy file logs WARNING and returns None."""
    import logging

    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"shape-test")
    stimulus = Stimulus(audio_path=str(stim_file))

    # Save a tensor with wrong shape
    h = cache._content_hash(stimulus)
    tensor_dir = tmp_path / "tensors" / h
    tensor_dir.mkdir(parents=True, exist_ok=True)
    np.save(tensor_dir / "collapsed.npy", np.zeros((100,), dtype=np.float32))

    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.cache"):
        result = cache.get_collapsed(stimulus)

    assert result is None
    assert any("shape" in r.message.lower() for r in caplog.records)


def test_content_hash_is_hex_string(tmp_path):
    """_content_hash returns a 64-character hex string (SHA-256)."""
    cache = ResponseCache(str(tmp_path))
    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"some content")
    stimulus = Stimulus(audio_path=str(stim_file))
    h = cache._content_hash(stimulus)
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_multimodal_order(tmp_path):
    """Hash over multiple modalities is order-dependent (video, audio, text)."""
    cache = ResponseCache(str(tmp_path))

    video_file = tmp_path / "v.bin"
    audio_file = tmp_path / "a.bin"
    video_file.write_bytes(b"video-bytes")
    audio_file.write_bytes(b"audio-bytes")

    stim_va = Stimulus(video_path=str(video_file), audio_path=str(audio_file))
    stim_v_only = Stimulus(video_path=str(video_file))
    stim_a_only = Stimulus(audio_path=str(audio_file))

    h_va = cache._content_hash(stim_va)
    h_v = cache._content_hash(stim_v_only)
    h_a = cache._content_hash(stim_a_only)

    # All three should be distinct
    assert h_va != h_v
    assert h_va != h_a
    assert h_v != h_a
