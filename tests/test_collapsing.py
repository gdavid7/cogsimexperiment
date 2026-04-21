"""Tests for TemporalCollapser — unit tests and property-based tests."""

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.collapsing import TemporalCollapser
from cognitive_similarity.models import Stimulus

_VERTICES = 20484
_TR_S = 1.0


# ---------------------------------------------------------------------------
# Property 3: Output Shape Is Always (20484,)
# Feature: cognitive-similarity, Property 3: Temporal Collapsing Output Shape
# ---------------------------------------------------------------------------

@given(
    T=st.integers(min_value=1, max_value=30),
    duration_s=st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_property_3_output_shape(T, duration_s):
    """
    # Feature: cognitive-similarity, Property 3: Temporal Collapsing Output Shape
    Validates: Requirements 2.3

    collapse() produces shape (20484,) for any valid (T, duration_s) combo.
    """
    rng = np.random.default_rng(0)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=duration_s)
    collapser = TemporalCollapser()

    collapsed = collapser.collapse(cortical_response, stimulus, tr_s=_TR_S)

    assert collapsed.shape == (_VERTICES,), f"Expected ({_VERTICES},), got {collapsed.shape}"


# ---------------------------------------------------------------------------
# Property 4: Peak Extraction Correctness
# Feature: cognitive-similarity, Property 4: Peak Extraction Correctness
# ---------------------------------------------------------------------------

@given(
    T=st.integers(min_value=6, max_value=30),
)
@settings(max_examples=100)
def test_property_4_peak_extraction_correctness(T):
    """
    # Feature: cognitive-similarity, Property 4: Peak Extraction Correctness
    Validates: Requirements 2.1

    For short stimuli (duration ≤ 10s) with sufficient T, the collapsed output
    must equal cortical_response at the t+5s timepoint (index = round(5.0 / tr_s) = 5).
    """
    rng = np.random.default_rng(T)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    # duration ≤ 10s forces peak extraction
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=5.0)
    collapser = TemporalCollapser()

    collapsed = collapser.collapse(cortical_response, stimulus, tr_s=_TR_S)

    expected_idx = round(5.0 / _TR_S)  # = 5
    np.testing.assert_array_equal(collapsed, cortical_response[expected_idx])


# ---------------------------------------------------------------------------
# Unit tests for TemporalCollapser
# ---------------------------------------------------------------------------

def test_peak_fallback_uses_last_timepoint_and_logs_warning(caplog):
    """
    When T is shorter than the peak index (T=3, peak_idx=5), fall back to T-1
    and log WARNING.
    Validates: Requirements 2.4
    """
    T = 3
    rng = np.random.default_rng(1)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    # duration ≤ 10s → peak extraction; T=3 means peak_idx=5 is out of bounds
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=5.0)
    collapser = TemporalCollapser()

    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.collapsing"):
        collapsed = collapser.collapse(cortical_response, stimulus, tr_s=_TR_S)

    assert collapsed.shape == (_VERTICES,)
    np.testing.assert_array_equal(collapsed, cortical_response[T - 1])
    assert any("Peak timepoint unavailable" in r.message for r in caplog.records)


def test_long_stimulus_output_differs_from_any_single_timepoint():
    """
    For long stimuli (duration > 10s, T ≥ 2), the collapser uses GLM+HRF fitting.
    The resulting beta-weight vector should not exactly match any single
    timepoint of the input (which would indicate peak extraction was used).
    Validates: Requirements 2.2, 2.3
    """
    T = 20
    rng = np.random.default_rng(2)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=15.0)
    collapser = TemporalCollapser()

    collapsed = collapser.collapse(cortical_response, stimulus, tr_s=_TR_S)

    assert collapsed.shape == (_VERTICES,)
    for t in range(T):
        assert not np.array_equal(collapsed, cortical_response[t]), (
            f"Long-stimulus output matched cortical_response[{t}]; "
            "expected GLM+HRF fit, not peak extraction"
        )


def test_short_stimulus_matches_peak_timepoint():
    """Short stimulus (duration ≤ 10s) produces output equal to cortical_response[5]."""
    T = 10
    rng = np.random.default_rng(3)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=8.0)
    collapser = TemporalCollapser()

    collapsed = collapser.collapse(cortical_response, stimulus)
    np.testing.assert_array_equal(collapsed, cortical_response[5])


def test_duration_inferred_from_T_when_none():
    """
    When stimulus.duration_s is None, duration is inferred as T * tr_s.
    T=8, tr_s=1.0 → duration=8.0 ≤ 10 → peak extraction (output = cortical_response[5]).
    T=15, tr_s=1.0 → duration=15.0 > 10 → GLM+HRF (output ≠ any single timepoint).
    """
    collapser = TemporalCollapser()
    rng = np.random.default_rng(5)

    # Short inferred duration → peak extraction
    cortical_short = rng.random((8, _VERTICES), dtype=np.float32)
    stimulus_no_dur = Stimulus(video_path="dummy.mp4", duration_s=None)
    collapsed_short = collapser.collapse(cortical_short, stimulus_no_dur, tr_s=1.0)
    np.testing.assert_array_equal(collapsed_short, cortical_short[5])

    # Long inferred duration → GLM+HRF (should differ from any single timepoint)
    cortical_long = rng.random((15, _VERTICES), dtype=np.float32)
    collapsed_long = collapser.collapse(cortical_long, stimulus_no_dur, tr_s=1.0)
    for t in range(15):
        assert not np.array_equal(collapsed_long, cortical_long[t])
