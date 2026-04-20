"""Tests for cognitive_similarity data models."""

# Feature: cognitive-similarity, Property 2: Invalid Stimulus Rejection

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_similarity.models import Stimulus


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

# A strategy that always produces a Stimulus with all modality paths set to None.
# We use st.builds so Hypothesis can vary the optional non-modality fields
# (duration_s, stimulus_id) while keeping all paths None.
invalid_stimulus_strategy = st.builds(
    Stimulus,
    video_path=st.none(),
    audio_path=st.none(),
    text_path=st.none(),
    duration_s=st.one_of(st.none(), st.floats(min_value=0.1, max_value=3600.0, allow_nan=False)),
    stimulus_id=st.one_of(st.none(), st.text(max_size=64)),
)


# ---------------------------------------------------------------------------
# Property 2: Invalid Stimulus Rejection
# ---------------------------------------------------------------------------

@given(stimulus=invalid_stimulus_strategy)
@settings(max_examples=100)
def test_property_2_invalid_stimulus_rejection(stimulus: Stimulus) -> None:
    """
    **Validates: Requirements 1.6**

    For any Stimulus where video_path, audio_path, and text_path are all None,
    calling validate() SHALL raise a ValueError with a descriptive message.
    """
    # Precondition: all modality paths are None
    assert stimulus.video_path is None
    assert stimulus.audio_path is None
    assert stimulus.text_path is None

    with pytest.raises(ValueError) as exc_info:
        stimulus.validate()

    # The error message must be descriptive (non-empty)
    assert len(str(exc_info.value)) > 0, "ValueError message should be descriptive"


# ---------------------------------------------------------------------------
# Unit tests for Stimulus.validate() — valid stimuli should NOT raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"video_path": "/some/video.mp4"},
    {"audio_path": "/some/audio.wav"},
    {"text_path": "/some/text.txt"},
    {"video_path": "/v.mp4", "audio_path": "/a.wav"},
    {"video_path": "/v.mp4", "audio_path": "/a.wav", "text_path": "/t.txt"},
])
def test_valid_stimulus_does_not_raise(kwargs):
    """A Stimulus with at least one modality path set should not raise on validate()."""
    s = Stimulus(**kwargs)
    s.validate()  # should not raise
