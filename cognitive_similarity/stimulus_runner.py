"""StimulusRunner — runs a Stimulus through both cortical and subcortical TRIBE v2 models."""

try:
    from tribev2 import TribeModel
except ImportError as _tribev2_import_error:
    TribeModel = None  # type: ignore[assignment,misc]
    _TRIBEV2_MISSING_MSG = (
        "tribev2 is required to use StimulusRunner but is not installed. "
        "Install it with: pip install tribev2"
    )

from cognitive_similarity.models import BrainResponse, Stimulus


class StimulusRunner:
    """Runs a single Stimulus through both the cortical and subcortical TRIBE v2 models.

    Each model is called independently via its own ``get_events_dataframe()`` +
    ``predict()`` pair, ensuring stimulus isolation (no cross-contamination from
    context-sensitive embeddings).

    Args:
        cortical_model: A ``TribeModel`` instance loaded with the default
            ``facebook/tribev2`` configuration.  Produces predictions of shape
            ``(n_timesteps, 20484)``.
        subcortical_model: A ``TribeModel`` instance configured with
            ``MaskProjector(mask="subcortical")``.  Produces predictions of
            shape ``(n_timesteps, 8802)``.
    """

    def __init__(self, cortical_model, subcortical_model) -> None:
        if TribeModel is None:
            raise ImportError(_TRIBEV2_MISSING_MSG)
        self._cortical_model = cortical_model
        self._subcortical_model = subcortical_model

    def run(self, stimulus: Stimulus) -> BrainResponse:
        """Run *stimulus* through both models and return a :class:`BrainResponse`.

        The stimulus is validated first (raises ``ValueError`` if no modality is
        set).  For multimodal input (video + audio + text), ``video_path`` is
        preferred because TRIBE v2 extracts audio and transcribes internally.

        Args:
            stimulus: The stimulus to process.  At least one of ``video_path``,
                ``audio_path``, or ``text_path`` must be set.

        Returns:
            A :class:`BrainResponse` with:
            - ``cortical``   — shape ``(n_timesteps, 20484)`` float32
            - ``subcortical`` — shape ``(n_timesteps, 8802)``  float32
            - ``segments``   — segment objects from the cortical model run
        """
        stimulus.validate()

        # Determine which path argument to pass to get_events_dataframe().
        # video_path is preferred for multimodal input because TRIBE v2 handles
        # audio extraction and transcription internally.
        kwargs = _modality_kwargs(stimulus)

        # --- Cortical model ---
        df_cortical = self._cortical_model.get_events_dataframe(**kwargs)
        preds_cortical, segments_cortical = self._cortical_model.predict(events=df_cortical)

        # --- Subcortical model (independent call — isolation guaranteed) ---
        df_subcortical = self._subcortical_model.get_events_dataframe(**kwargs)
        preds_subcortical, _ = self._subcortical_model.predict(events=df_subcortical)

        return BrainResponse(
            cortical=preds_cortical,
            subcortical=preds_subcortical,
            segments=segments_cortical,
        )


def _modality_kwargs(stimulus: Stimulus) -> dict:
    """Return the single keyword argument for ``get_events_dataframe()``.

    Priority: video_path > audio_path > text_path.
    """
    if stimulus.video_path is not None:
        return {"video_path": stimulus.video_path}
    if stimulus.audio_path is not None:
        return {"audio_path": stimulus.audio_path}
    return {"text_path": stimulus.text_path}
