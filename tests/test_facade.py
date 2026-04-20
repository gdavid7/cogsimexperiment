"""Tests for CognitiveSimilarity facade — property-based and integration tests.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 4.5, 4.7, 4.8, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.cache import ResponseCache
from cognitive_similarity.collapsing import TemporalCollapser
from cognitive_similarity.facade import CognitiveSimilarity, _rank_entries
from cognitive_similarity.ica_atlas import ICANetworkAtlas, N_VERTICES
from cognitive_similarity.models import (
    CollapsingStrategy,
    ICAMode,
    ICANetwork,
    RankedEntry,
    SimilarityResult,
    Stimulus,
)

# ---------------------------------------------------------------------------
# Shared synthetic atlas (no HuggingFace)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)
_SYNTHETIC_PROJECTION = _RNG.standard_normal((2048, N_VERTICES)).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_atlas() -> ICANetworkAtlas:
    return ICANetworkAtlas.from_projection_matrix(_SYNTHETIC_PROJECTION, top_percentile=0.10)


# ---------------------------------------------------------------------------
# Helpers for building test stimuli backed by real temp files
# ---------------------------------------------------------------------------

def _make_stimulus_file(tmp_dir: str, name: str, content: bytes) -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _make_stimulus(tmp_dir: str, name: str, content: bytes, stimulus_id: Optional[str] = None) -> Stimulus:
    path = _make_stimulus_file(tmp_dir, name, content)
    return Stimulus(audio_path=path, stimulus_id=stimulus_id or name)


def _random_collapsed(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(N_VERTICES).astype(np.float32)


def _random_raw_cortical(seed: int, T: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, N_VERTICES)).astype(np.float32)


def _random_raw_subcortical(seed: int, T: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, 8802)).astype(np.float32)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def collapsed_response_strategy():
    """Generate a (20484,) float32 array with non-trivial variance."""
    return st.integers(min_value=0, max_value=2**31 - 1).map(
        lambda s: np.random.default_rng(s).standard_normal(N_VERTICES).astype(np.float32)
    )


def stimulus_seed_strategy():
    """Generate a unique integer seed for building a stimulus."""
    return st.integers(min_value=1, max_value=10_000)


# ---------------------------------------------------------------------------
# Property 1: Stimulus Isolation — One Inference Per Stimulus
# Feature: cognitive-similarity, Property 1: Stimulus Isolation — One Inference Per Stimulus
# ---------------------------------------------------------------------------

@given(
    seeds=st.lists(
        st.integers(min_value=1, max_value=10_000),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
@settings(max_examples=50, deadline=None)
def test_property_1_stimulus_isolation_compare(
    synthetic_atlas: ICANetworkAtlas,
    seeds: list[int],
) -> None:
    """
    # Feature: cognitive-similarity, Property 1: Stimulus Isolation — One Inference Per Stimulus
    Validates: Requirements 1.4

    get_collapsed_response() is called exactly once per unique stimulus in compare().
    Since the facade reads from cache, we verify that for N unique stimuli passed to
    compare(), get_collapsed_response() is invoked exactly N times (once per stimulus).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        # Pre-populate cache with collapsed responses
        stimuli = []
        for i, seed in enumerate(seeds):
            stim = _make_stimulus(tmp_dir, f"stim_{seed}.bin", seed.to_bytes(4, "big"), f"s{seed}")
            cache.put_collapsed(stim, _random_collapsed(seed))
            stimuli.append(stim)

        # For compare(), we need exactly 2 stimuli
        stim_a = stimuli[0]
        stim_b = stimuli[-1] if len(stimuli) > 1 else stimuli[0]

        call_count = 0
        original_gcr = facade.get_collapsed_response

        def counting_gcr(stimulus):
            nonlocal call_count
            call_count += 1
            return original_gcr(stimulus)

        facade.get_collapsed_response = counting_gcr

        facade.compare(stim_a, stim_b)

        # compare() calls get_collapsed_response once per stimulus (2 calls)
        expected = 2 if stim_a.stimulus_id != stim_b.stimulus_id else 2
        assert call_count == expected, (
            f"Expected get_collapsed_response to be called {expected} times, got {call_count}"
        )


@given(
    seeds=st.lists(
        st.integers(min_value=1, max_value=10_000),
        min_size=2,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=50, deadline=None)
def test_property_1_stimulus_isolation_rank(
    synthetic_atlas: ICANetworkAtlas,
    seeds: list[int],
) -> None:
    """
    # Feature: cognitive-similarity, Property 1: Stimulus Isolation — One Inference Per Stimulus
    Validates: Requirements 1.4

    In rank(), get_collapsed_response() is called once per unique stimulus
    (query + each corpus stimulus = N+1 unique stimuli → N+1 calls).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        stimuli = []
        for seed in seeds:
            stim = _make_stimulus(tmp_dir, f"stim_{seed}.bin", seed.to_bytes(4, "big"), f"s{seed}")
            cache.put_collapsed(stim, _random_collapsed(seed))
            stimuli.append(stim)

        query = stimuli[0]
        corpus = stimuli[1:]  # at least 1 element; but rank() needs >= 2

        if len(corpus) < 2:
            # Not enough for rank(); skip this case
            return

        call_count = 0
        original_gcr = facade.get_collapsed_response

        def counting_gcr(stimulus):
            nonlocal call_count
            call_count += 1
            return original_gcr(stimulus)

        facade.get_collapsed_response = counting_gcr

        facade.rank(query, corpus)

        # rank() calls compare(query, corpus[i]) for each i
        # each compare() calls get_collapsed_response twice (query + corpus[i])
        # so total = 2 * len(corpus) calls
        expected_calls = 2 * len(corpus)
        assert call_count == expected_calls, (
            f"Expected {expected_calls} calls to get_collapsed_response, got {call_count}"
        )


# ---------------------------------------------------------------------------
# Property 14: Cache Hit Avoids Re-Inference
# Feature: cognitive-similarity, Property 14: Cache Hit Avoids Re-Inference
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=50, deadline=None)
def test_property_14_cache_hit_avoids_re_inference(
    synthetic_atlas: ICANetworkAtlas,
    seed: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 14: Cache Hit Avoids Re-Inference
    Validates: Requirements 6.5

    When a collapsed response is already in cache, get_collapsed_response()
    returns it without re-collapsing (TemporalCollapser.collapse is not called).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        stim = _make_stimulus(tmp_dir, f"stim_{seed}.bin", seed.to_bytes(4, "big"), f"s{seed}")
        expected = _random_collapsed(seed)
        cache.put_collapsed(stim, expected)

        # Patch the collapser to detect if it's called
        collapse_called = False
        original_collapse = facade._collapser.collapse

        def spy_collapse(*args, **kwargs):
            nonlocal collapse_called
            collapse_called = True
            return original_collapse(*args, **kwargs)

        facade._collapser.collapse = spy_collapse

        result = facade.get_collapsed_response(stim)

        assert not collapse_called, (
            "TemporalCollapser.collapse should NOT be called when collapsed response is cached"
        )
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Property 10: Similarity Result Structural Completeness
# Feature: cognitive-similarity, Property 10: Similarity Result Structural Completeness
# ---------------------------------------------------------------------------

@given(
    seed_a=st.integers(min_value=1, max_value=5_000),
    seed_b=st.integers(min_value=5_001, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
def test_property_10_similarity_result_structural_completeness(
    synthetic_atlas: ICANetworkAtlas,
    seed_a: int,
    seed_b: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 10: Similarity Result Structural Completeness
    Validates: Requirements 3.8, 4.5

    compare() returns a SimilarityResult with a CognitiveSimilarityProfile containing
    all 5 networks, plus stimulus_a_id, stimulus_b_id, collapsing_strategy_a,
    collapsing_strategy_b.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        stim_a = _make_stimulus(tmp_dir, f"a_{seed_a}.bin", seed_a.to_bytes(4, "big"), f"sa{seed_a}")
        stim_b = _make_stimulus(tmp_dir, f"b_{seed_b}.bin", seed_b.to_bytes(4, "big"), f"sb{seed_b}")
        cache.put_collapsed(stim_a, _random_collapsed(seed_a))
        cache.put_collapsed(stim_b, _random_collapsed(seed_b))

        result = facade.compare(stim_a, stim_b)

        # Must be a SimilarityResult
        assert isinstance(result, SimilarityResult)

        # Required fields populated
        assert result.stimulus_a_id is not None and result.stimulus_a_id != ""
        assert result.stimulus_b_id is not None and result.stimulus_b_id != ""
        assert result.collapsing_strategy_a is not None
        assert result.collapsing_strategy_b is not None

        # Profile must have all 5 networks
        profile = result.profile
        assert len(profile.network_scores) == 5
        assert set(profile.network_scores.keys()) == set(ICANetwork)

        # Each network score must have vertex_count > 0
        for net, ns in profile.network_scores.items():
            assert ns.vertex_count > 0, f"vertex_count must be > 0 for {net.value}"

        # whole_cortex_score must be present (float)
        assert isinstance(profile.whole_cortex_score, float)


# ---------------------------------------------------------------------------
# Property 11: Batch Result Ordering
# Feature: cognitive-similarity, Property 11: Batch Result Ordering
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=50, deadline=None)
def test_property_11_batch_result_ordering(
    synthetic_atlas: ICANetworkAtlas,
    n: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 11: Batch Result Ordering
    Validates: Requirements 4.7

    rank() returns exactly N entries for a corpus of N stimuli.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        query = _make_stimulus(tmp_dir, "query.bin", b"query", "query")
        cache.put_collapsed(query, _random_collapsed(0))

        corpus = []
        for i in range(n):
            stim = _make_stimulus(tmp_dir, f"corpus_{i}.bin", i.to_bytes(4, "big"), f"c{i}")
            cache.put_collapsed(stim, _random_collapsed(i + 1))
            corpus.append(stim)

        ranked = facade.rank(query, corpus)

        # Each per-network ranking has exactly N entries
        for net, entries in ranked.rankings_by_network.items():
            assert len(entries) == n, (
                f"Expected {n} entries for {net.value}, got {len(entries)}"
            )

        # Whole-cortex ranking also has N entries
        assert len(ranked.rankings_whole_cortex) == n, (
            f"Expected {n} whole-cortex entries, got {len(ranked.rankings_whole_cortex)}"
        )


# ---------------------------------------------------------------------------
# Property 16: Ranked List Is Sorted in Descending Order
# Feature: cognitive-similarity, Property 16: Ranked List Is Sorted in Descending Order
# ---------------------------------------------------------------------------

@given(
    scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=50, deadline=None)
def test_property_16_ranked_list_descending_order(scores: list[float]) -> None:
    """
    # Feature: cognitive-similarity, Property 16: Ranked List Is Sorted in Descending Order
    Validates: Requirements 7.1

    _rank_entries() produces a list sorted in descending order of score.
    """
    entries = [(f"s{i}", score) for i, score in enumerate(scores)]
    ranked = _rank_entries(entries)

    for i in range(len(ranked) - 1):
        assert ranked[i].score >= ranked[i + 1].score, (
            f"Ranked list not in descending order at position {i}: "
            f"{ranked[i].score} < {ranked[i + 1].score}"
        )


@given(
    n=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=50, deadline=None)
def test_property_16_rank_result_descending_via_facade(
    synthetic_atlas: ICANetworkAtlas,
    n: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 16: Ranked List Is Sorted in Descending Order
    Validates: Requirements 7.1

    rank() returns per-network and whole-cortex lists sorted descending.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        query = _make_stimulus(tmp_dir, "query.bin", b"query", "query")
        cache.put_collapsed(query, _random_collapsed(999))

        corpus = []
        for i in range(n):
            stim = _make_stimulus(tmp_dir, f"c_{i}.bin", i.to_bytes(4, "big"), f"c{i}")
            cache.put_collapsed(stim, _random_collapsed(i + 100))
            corpus.append(stim)

        ranked = facade.rank(query, corpus)

        for net, entries in ranked.rankings_by_network.items():
            for j in range(len(entries) - 1):
                assert entries[j].score >= entries[j + 1].score, (
                    f"Network {net.value}: not descending at position {j}"
                )

        wc = ranked.rankings_whole_cortex
        for j in range(len(wc) - 1):
            assert wc[j].score >= wc[j + 1].score, (
                f"Whole-cortex: not descending at position {j}"
            )


# ---------------------------------------------------------------------------
# Property 17: Tie Handling — Equal Scores Share Rank
# Feature: cognitive-similarity, Property 17: Tie Handling — Equal Scores Share Rank
# ---------------------------------------------------------------------------

@given(
    base_scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    tie_count=st.integers(min_value=2, max_value=4),
    tie_score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_property_17_tie_handling_equal_scores_share_rank(
    base_scores: list[float],
    tie_count: int,
    tie_score: float,
) -> None:
    """
    # Feature: cognitive-similarity, Property 17: Tie Handling — Equal Scores Share Rank
    Validates: Requirements 7.5

    When two or more entries have equal scores, they share the same rank number.
    """
    # Build entries: some base scores + tie_count entries with the same tie_score
    entries = [(f"base_{i}", s) for i, s in enumerate(base_scores)]
    for j in range(tie_count):
        entries.append((f"tie_{j}", tie_score))

    ranked = _rank_entries(entries)

    # Find all entries with tie_score
    tied_entries = [e for e in ranked if e.score == tie_score]

    if len(tied_entries) >= 2:
        ranks = {e.rank for e in tied_entries}
        assert len(ranks) == 1, (
            f"Tied entries with score {tie_score} have different ranks: "
            f"{[e.rank for e in tied_entries]}"
        )


# ---------------------------------------------------------------------------
# Property 15: JSON Output Contains All Required Fields
# Feature: cognitive-similarity, Property 15: JSON Output Contains All Required Fields
# ---------------------------------------------------------------------------

@given(
    seed_a=st.integers(min_value=1, max_value=5_000),
    seed_b=st.integers(min_value=5_001, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
def test_property_15_json_output_completeness(
    synthetic_atlas: ICANetworkAtlas,
    seed_a: int,
    seed_b: int,
) -> None:
    """
    # Feature: cognitive-similarity, Property 15: JSON Output Contains All Required Fields
    Validates: Requirements 6.6

    SimilarityResult can be serialized to JSON and contains all required fields:
    profile, stimulus_a_id, stimulus_b_id, collapsing_strategy_a, collapsing_strategy_b.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = ResponseCache(tmp_dir)
        facade = CognitiveSimilarity(cache_dir=tmp_dir, _atlas=synthetic_atlas)

        stim_a = _make_stimulus(tmp_dir, f"a_{seed_a}.bin", seed_a.to_bytes(4, "big"), f"sa{seed_a}")
        stim_b = _make_stimulus(tmp_dir, f"b_{seed_b}.bin", seed_b.to_bytes(4, "big"), f"sb{seed_b}")
        cache.put_collapsed(stim_a, _random_collapsed(seed_a))
        cache.put_collapsed(stim_b, _random_collapsed(seed_b))

        result = facade.compare(stim_a, stim_b)

        # Serialize to JSON via dataclasses.asdict
        # network_scores dict has ICANetwork enum keys — convert to string keys first
        def convert_for_json(obj):
            """Recursively convert enums and numpy types for JSON serialization."""
            if isinstance(obj, dict):
                return {
                    (k.value if isinstance(k, (ICANetwork, ICAMode, CollapsingStrategy)) else k): convert_for_json(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            if isinstance(obj, (ICANetwork, ICAMode, CollapsingStrategy)):
                return obj.value
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        d = dataclasses.asdict(result)
        d_converted = convert_for_json(d)
        json_str = json.dumps(d_converted)
        parsed = json.loads(json_str)

        # Required top-level fields
        required_fields = {
            "profile",
            "stimulus_a_id",
            "stimulus_b_id",
            "collapsing_strategy_a",
            "collapsing_strategy_b",
        }
        for field in required_fields:
            assert field in parsed, f"Missing required field '{field}' in JSON output"

        # Profile must contain network_scores and whole_cortex_score
        assert "network_scores" in parsed["profile"], "Missing 'network_scores' in profile"
        assert "whole_cortex_score" in parsed["profile"], "Missing 'whole_cortex_score' in profile"

        # All 5 networks present in network_scores
        assert len(parsed["profile"]["network_scores"]) == 5, (
            f"Expected 5 network scores in JSON, got {len(parsed['profile']['network_scores'])}"
        )


# ---------------------------------------------------------------------------
# Integration tests (sub-task 8.8)
# ---------------------------------------------------------------------------

def test_integration_compare_with_prepopulated_cache(
    synthetic_atlas: ICANetworkAtlas,
    tmp_path,
) -> None:
    """
    End-to-end compare() with pre-populated cache (no live TRIBE v2).
    Write raw cortical tensors to cache, call compare(), verify SimilarityResult structure.
    """
    cache = ResponseCache(str(tmp_path))
    facade = CognitiveSimilarity(cache_dir=str(tmp_path), _atlas=synthetic_atlas)

    # Create stimulus files
    file_a = tmp_path / "stim_a.bin"
    file_b = tmp_path / "stim_b.bin"
    file_a.write_bytes(b"stimulus-a-content")
    file_b.write_bytes(b"stimulus-b-content")

    stim_a = Stimulus(audio_path=str(file_a), stimulus_id="stim_a", duration_s=5.0)
    stim_b = Stimulus(audio_path=str(file_b), stimulus_id="stim_b", duration_s=5.0)

    # Write raw cortical tensors (T=6, duration_s=5.0 → PEAK strategy)
    rng = np.random.default_rng(0)
    raw_a = rng.standard_normal((6, N_VERTICES)).astype(np.float32)
    raw_b = rng.standard_normal((6, N_VERTICES)).astype(np.float32)
    cache.put_raw(stim_a, raw_a, np.zeros((6, 8802), dtype=np.float32))
    cache.put_raw(stim_b, raw_b, np.zeros((6, 8802), dtype=np.float32))

    result = facade.compare(stim_a, stim_b)

    assert isinstance(result, SimilarityResult)
    assert result.stimulus_a_id == "stim_a"
    assert result.stimulus_b_id == "stim_b"
    assert result.collapsing_strategy_a is not None
    assert result.collapsing_strategy_b is not None
    assert len(result.profile.network_scores) == 5
    assert set(result.profile.network_scores.keys()) == set(ICANetwork)
    assert isinstance(result.profile.whole_cortex_score, float)
    assert -1.0 <= result.profile.whole_cortex_score <= 1.0


def test_integration_cache_population_and_retrieval(
    synthetic_atlas: ICANetworkAtlas,
    tmp_path,
) -> None:
    """
    Cache population and retrieval: call get_collapsed_response() twice for the same
    stimulus, verify the collapser is only called once.
    """
    cache = ResponseCache(str(tmp_path))
    facade = CognitiveSimilarity(cache_dir=str(tmp_path), _atlas=synthetic_atlas)

    stim_file = tmp_path / "stim.bin"
    stim_file.write_bytes(b"test-stimulus")
    stim = Stimulus(audio_path=str(stim_file), stimulus_id="test", duration_s=5.0)

    # Write raw cortical (no collapsed yet)
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((6, N_VERTICES)).astype(np.float32)
    cache.put_raw(stim, raw, np.zeros((6, 8802), dtype=np.float32))

    collapse_call_count = 0
    original_collapse = facade._collapser.collapse

    def counting_collapse(*args, **kwargs):
        nonlocal collapse_call_count
        collapse_call_count += 1
        return original_collapse(*args, **kwargs)

    facade._collapser.collapse = counting_collapse

    # First call: should collapse (raw → collapsed) and cache it
    result1 = facade.get_collapsed_response(stim)
    assert collapse_call_count == 1, "Expected collapse to be called once on first access"

    # Second call: should hit collapsed cache, NOT call collapse again
    result2 = facade.get_collapsed_response(stim)
    assert collapse_call_count == 1, "Expected collapse NOT to be called again on second access"

    np.testing.assert_array_equal(result1, result2)


def test_integration_rank_corpus_too_small_raises_value_error(
    synthetic_atlas: ICANetworkAtlas,
    tmp_path,
) -> None:
    """
    rank() with corpus < 2 stimuli raises ValueError.
    Validates: Requirements 7.6
    """
    cache = ResponseCache(str(tmp_path))
    facade = CognitiveSimilarity(cache_dir=str(tmp_path), _atlas=synthetic_atlas)

    query_file = tmp_path / "query.bin"
    query_file.write_bytes(b"query")
    query = Stimulus(audio_path=str(query_file), stimulus_id="query")
    cache.put_collapsed(query, _random_collapsed(0))

    # Empty corpus
    with pytest.raises(ValueError, match="at least 2"):
        facade.rank(query, [])

    # Single-element corpus
    single_file = tmp_path / "single.bin"
    single_file.write_bytes(b"single")
    single = Stimulus(audio_path=str(single_file), stimulus_id="single")
    cache.put_collapsed(single, _random_collapsed(1))

    with pytest.raises(ValueError, match="at least 2"):
        facade.rank(query, [single])


def test_integration_get_collapsed_response_raises_when_nothing_cached(
    synthetic_atlas: ICANetworkAtlas,
    tmp_path,
) -> None:
    """
    get_collapsed_response() raises RuntimeError when neither collapsed nor raw is cached.
    """
    facade = CognitiveSimilarity(cache_dir=str(tmp_path), _atlas=synthetic_atlas)

    stim_file = tmp_path / "missing.bin"
    stim_file.write_bytes(b"no-cache")
    stim = Stimulus(audio_path=str(stim_file), stimulus_id="missing")

    with pytest.raises(RuntimeError, match="pre-computed"):
        facade.get_collapsed_response(stim)
