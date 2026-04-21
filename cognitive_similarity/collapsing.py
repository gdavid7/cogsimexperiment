"""Temporal collapsing: reduce (T, 20484) cortical response to (20484,)."""

import logging

import numpy as np
import pandas as pd

from cognitive_similarity.models import Stimulus

log = logging.getLogger(__name__)

_VERTICES = 20484
_PEAK_OFFSET_S = 5.0
_LONG_STIMULUS_THRESHOLD_S = 10.0


class TemporalCollapser:
    """Reduces a cortical response timeseries to a single spatial vector.

    Method is selected automatically from stimulus duration (Req 2.1, 2.2):
    peak extraction at t+5s for stimuli ≤10s, GLM+HRF fitting for longer ones.
    Not caller-configurable.
    """

    def collapse(
        self,
        cortical_response: np.ndarray,  # shape (T, 20484)
        stimulus: Stimulus,
        tr_s: float = 1.0,
    ) -> np.ndarray:
        """Return the collapsed cortical response of shape (20484,).

        Duration comes from ``stimulus.duration_s``; if None, inferred as ``T * tr_s``.
        """
        T = cortical_response.shape[0]
        duration_s = stimulus.duration_s if stimulus.duration_s is not None else T * tr_s

        # GLM+HRF needs ≥2 timepoints to build a design matrix; otherwise PEAK.
        if duration_s > _LONG_STIMULUS_THRESHOLD_S and T >= 2:
            return self._glm_hrf(cortical_response, duration_s, tr_s)
        return self._peak(cortical_response, tr_s)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _peak(self, cortical_response: np.ndarray, tr_s: float) -> np.ndarray:
        T = cortical_response.shape[0]
        peak_idx = round(_PEAK_OFFSET_S / tr_s)
        if peak_idx >= T:
            peak_idx = T - 1
            log.warning("Peak timepoint unavailable; using last timepoint")
        return cortical_response[peak_idx]

    def _glm_hrf(
        self,
        cortical_response: np.ndarray,
        duration_s: float,
        tr_s: float,
    ) -> np.ndarray:
        from nilearn.glm.first_level import make_first_level_design_matrix

        T = cortical_response.shape[0]
        frame_times = np.arange(T) * tr_s
        events = pd.DataFrame({
            "onset": [0.0],
            "duration": [duration_s],
            "trial_type": ["stimulus"],
        })
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events=events,
            hrf_model="spm",
            drift_model=None,
        )
        X = design_matrix.values  # (T, n_regressors)
        Y = cortical_response      # (T, 20484)
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return beta[0]             # shape (20484,)
