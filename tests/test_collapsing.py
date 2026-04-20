"""Tests for TemporalCollapser — unit tests and property-based tests."""

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.collapsing import TemporalCollapser
from cognitive_similarity.models import CollapsingStrategy, Stimulus

_VERTICES = 20484
_TR_S = 1.0


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def cortical_response_strategy(min_T: int = 1, max_T: int = 30):
    """Generate a (T, 20484) float32 array."""
    return st.integers(min_value=min_T, max_value=max_T).flatmap(
        lambda T: st.just(
            np.random.default_rng(42).random((T, _VERTICES), dtype=np.float32)
        )
    )


# ---------------------------------------------------------------------------
# Property 3: Temporal Collapsing Strategy Selection and Output Shape
# Feature: cognitive-similarity, Property 3: Temporal Collapsing Strategy Selection and Output Shape
# ---------------------------------------------------------------------------

@given(
    T=st.integers(min_value=1, max_value=30),
    duration_s=st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_property_3_strategy_selection_and_output_shape(T, duration_s):
    """
    # Feature: cognitive-similarity, Property 3: Temporal Collapsing Strategy Selection and Output Shape
    Validates: Requirements 2.1, 2.2, 2.3
    """
    rng = np.random.default_rng(0)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=duration_s)
    collapser = TemporalCollapser()

    collapsed, strategy_used = collapser.collapse(
        cortical_response, stimulus, strategy=CollapsingStrategy.AUTO, tr_s=_TR_S
    )

    # Output shape must always be (20484,)
    assert collapsed.shape == (_VERTICES,), f"Expected ({_VERTICES},), got {collapsed.shape}"

    # Strategy selection based on duration.
    # When T == 1, GLM+HRF cannot compute TR (needs np.diff on ≥2 frame_times),
    # so the implementation falls back to PEAK regardless of duration.
    if duration_s <= 10.0 or T < 2:
        assert strategy_used is CollapsingStrategy.PEAK
    else:
        assert strategy_used is CollapsingStrategy.GLM_HRF


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

    For T >= 6, peak at round(5.0 / 1.0) = 5 is always available.
    The PEAK strategy must return exactly cortical_response[5].
    """
    rng = np.random.default_rng(T)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    # duration <= 10s forces PEAK
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=5.0)
    collapser = TemporalCollapser()

    collapsed, strategy_used = collapser.collapse(
        cortical_response, stimulus, strategy=CollapsingStrategy.AUTO, tr_s=_TR_S
    )

    assert strategy_used is CollapsingStrategy.PEAK
    expected_idx = round(5.0 / _TR_S)  # = 5
    np.testing.assert_array_equal(collapsed, cortical_response[expected_idx])


# ---------------------------------------------------------------------------
# Unit tests for TemporalCollapser (sub-task 3.3)
# ---------------------------------------------------------------------------

def test_peak_fallback_uses_last_timepoint_and_logs_warning(caplog):
    """
    When T < peak_idx (T=3, peak_idx=5), fall back to T-1 and log WARNING.
    Validates: Requirements 2.4
    """
    T = 3
    rng = np.random.default_rng(1)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    # duration <= 10s → PEAK strategy; T=3 means peak_idx=5 is out of bounds
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=5.0)
    collapser = TemporalCollapser()

    with caplog.at_level(logging.WARNING, logger="cognitive_similarity.collapsing"):
        collapsed, strategy_used = collapser.collapse(
            cortical_response, stimulus, strategy=CollapsingStrategy.PEAK, tr_s=_TR_S
        )

    assert strategy_used is CollapsingStrategy.PEAK
    assert collapsed.shape == (_VERTICES,)
    np.testing.assert_array_equal(collapsed, cortical_response[T - 1])
    assert any("Peak timepoint unavailable" in r.message for r in caplog.records)


def test_glm_hrf_output_shape_for_long_stimulus():
    """
    GLM+HRF must return shape (20484,) for T > 10.
    Validates: Requirements 2.2, 2.3
    """
    T = 20
    rng = np.random.default_rng(2)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=15.0)
    collapser = TemporalCollapser()

    collapsed, strategy_used = collapser.collapse(
        cortical_response, stimulus, strategy=CollapsingStrategy.GLM_HRF, tr_s=_TR_S
    )

    assert strategy_used is CollapsingStrategy.GLM_HRF
    assert collapsed.shape == (_VERTICES,)


def test_auto_is_default_parameter():
    """
    CollapsingStrategy.AUTO must be the default value for the strategy parameter.
    Validates: Requirements 2.5
    """
    import inspect
    sig = inspect.signature(TemporalCollapser.collapse)
    default = sig.parameters["strategy"].default
    assert default is CollapsingStrategy.AUTO


def test_auto_selects_peak_for_short_stimulus():
    """AUTO picks PEAK when duration <= 10s."""
    T = 10
    rng = np.random.default_rng(3)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=8.0)
    collapser = TemporalCollapser()

    _, strategy_used = collapser.collapse(cortical_response, stimulus)
    assert strategy_used is CollapsingStrategy.PEAK


def test_auto_selects_glm_hrf_for_long_stimulus():
    """AUTO picks GLM_HRF when duration > 10s."""
    T = 20
    rng = np.random.default_rng(4)
    cortical_response = rng.random((T, _VERTICES), dtype=np.float32)
    stimulus = Stimulus(video_path="dummy.mp4", duration_s=12.0)
    collapser = TemporalCollapser()

    _, strategy_used = collapser.collapse(cortical_response, stimulus)
    assert strategy_used is CollapsingStrategy.GLM_HRF


def test_duration_inferred_from_T_when_none():
    """
    When stimulus.duration_s is None, duration is inferred as T * tr_s.
    T=8, tr_s=1.0 → duration=8.0 ≤ 10 → PEAK.
    T=15, tr_s=1.0 → duration=15.0 > 10 → GLM_HRF.
    Validates: Requirements 2.5 (duration inference)
    """
    collapser = TemporalCollapser()
    rng = np.random.default_rng(5)

    # Short inferred duration → PEAK
    cortical_short = rng.random((8, _VERTICES), dtype=np.float32)
    stimulus_no_dur = Stimulus(video_path="dummy.mp4", duration_s=None)
    _, strategy = collapser.collapse(cortical_short, stimulus_no_dur, tr_s=1.0)
    assert strategy is CollapsingStrategy.PEAK

    # Long inferred duration → GLM_HRF
    cortical_long = rng.random((15, _VERTICES), dtype=np.float32)
    _, strategy = collapser.collapse(cortical_long, stimulus_no_dur, tr_s=1.0)
    assert strategy is CollapsingStrategy.GLM_HRF
