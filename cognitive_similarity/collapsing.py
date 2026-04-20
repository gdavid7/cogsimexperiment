"""Temporal collapsing: reduce (T, 20484) cortical response to (20484,)."""

import logging

import numpy as np
import pandas as pd

from cognitive_similarity.models import CollapsingStrategy, Stimulus

log = logging.getLogger(__name__)

_VERTICES = 20484
_PEAK_OFFSET_S = 5.0


class TemporalCollapser:
    """Reduces a cortical response timeseries to a single spatial vector."""

    def collapse(
        self,
        cortical_response: np.ndarray,  # shape (T, 20484)
        stimulus: Stimulus,
        strategy: CollapsingStrategy = CollapsingStrategy.AUTO,
        tr_s: float = 1.0,
    ) -> tuple[np.ndarray, CollapsingStrategy]:
        """
        Returns (collapsed_response [20484], strategy_used).

        Duration is taken from stimulus.duration_s; if None, inferred as T * tr_s.
        AUTO: peak if duration <= 10s, GLM+HRF if duration > 10s.
        """
        T = cortical_response.shape[0]
        duration_s = stimulus.duration_s if stimulus.duration_s is not None else T * tr_s

        if strategy is CollapsingStrategy.AUTO:
            strategy = CollapsingStrategy.PEAK if duration_s <= 10.0 else CollapsingStrategy.GLM_HRF

        # GLM+HRF requires at least 2 timepoints to compute TR via np.diff;
        # fall back to PEAK when T == 1.
        if strategy is CollapsingStrategy.GLM_HRF and T < 2:
            log.warning("GLM+HRF requires T >= 2; falling back to PEAK (T=%d)", T)
            strategy = CollapsingStrategy.PEAK

        if strategy is CollapsingStrategy.PEAK:
            collapsed = self._peak(cortical_response, tr_s)
        else:
            collapsed = self._glm_hrf(cortical_response, duration_s, tr_s)

        return collapsed, strategy

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
