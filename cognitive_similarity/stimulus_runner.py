"""StimulusRunner — runs a Stimulus through the TRIBE v2 cortical model."""

try:
    from tribev2 import TribeModel
except ImportError:
    TribeModel = None  # type: ignore[assignment,misc]
    _TRIBEV2_MISSING_MSG = (
        "tribev2 is required to use StimulusRunner but is not installed. "
        "Install it with: pip install tribev2"
    )

from cognitive_similarity.models import BrainResponse, Stimulus


class StimulusRunner:
    """Runs a single Stimulus through the TRIBE v2 cortical model.

    Only the cortical model is exposed: the public `facebook/tribev2`
    checkpoint ships a single weight file hard-wired to TribeSurfaceProjector
    on fsaverage5. A subcortical checkpoint is described in Meta's training
    grid (`tribev2/grids/run_subcortical.py`) but not published — so at
    inference time we can only obtain a (T, 20484) cortical prediction.

    Args:
        cortical_model: A ``TribeModel`` instance loaded with the default
            ``facebook/tribev2`` configuration. Produces predictions of shape
            ``(n_timesteps, 20484)``.
    """

    def __init__(self, cortical_model) -> None:
        if TribeModel is None:
            raise ImportError(_TRIBEV2_MISSING_MSG)
        self._cortical_model = cortical_model

    def run(self, stimulus: Stimulus) -> BrainResponse:
        """Run *stimulus* through the cortical model and return a :class:`BrainResponse`.

        The stimulus is validated first (raises ``ValueError`` if no modality is
        set).  For multimodal input (video + audio + text), ``video_path`` is
        preferred because TRIBE v2 extracts audio and transcribes internally.

        Args:
            stimulus: The stimulus to process.  At least one of ``video_path``,
                ``audio_path``, or ``text_path`` must be set.

        Returns:
            A :class:`BrainResponse` with:
            - ``cortical`` — shape ``(n_timesteps, 20484)`` float32
            - ``segments`` — segment objects from the cortical model run
        """
        stimulus.validate()

        # Determine which path argument to pass to get_events_dataframe().
        # video_path is preferred for multimodal input because TRIBE v2 handles
        # audio extraction and transcription internally.
        kwargs = _modality_kwargs(stimulus)

        df = self._cortical_model.get_events_dataframe(**kwargs)
        preds, segments = self._cortical_model.predict(events=df)

        return BrainResponse(cortical=preds, segments=segments)


def _modality_kwargs(stimulus: Stimulus) -> dict:
    """Return the single keyword argument for ``get_events_dataframe()``.

    Priority: video_path > audio_path > text_path.
    """
    if stimulus.video_path is not None:
        return {"video_path": stimulus.video_path}
    if stimulus.audio_path is not None:
        return {"audio_path": stimulus.audio_path}
    return {"text_path": stimulus.text_path}
