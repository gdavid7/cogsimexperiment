"""SimilarityEngine — computes per-network Pearson correlation similarity scores."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from cognitive_similarity.ica_atlas import ICANetworkAtlas
from cognitive_similarity.models import (
    CognitiveSimilarityProfile,
    ICAMode,
    ICANetwork,
    NetworkScore,
)

log = logging.getLogger(__name__)

# Valid network identifiers for error messages
_VALID_NETWORKS = [n.value for n in ICANetwork]


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-centered dot product normalized by norms.

    Returns 0.0 if either vector has zero variance.
    """
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(a_c, b_c) / (norm_a * norm_b))
    # Clamp to [-1, 1] to guard against float32 rounding errors
    return max(-1.0, min(1.0, raw))


def weighted_pearson_correlation(
    a: np.ndarray, b: np.ndarray, w: np.ndarray
) -> float:
    """Weighted Pearson correlation across same-length vectors.

    Uses the standard weighted-mean + weighted-covariance formulation:

        a_wm = Σᵢ wᵢ aᵢ   (weights already normalized to sum to 1)
        b_wm = Σᵢ wᵢ bᵢ
        numer = Σᵢ wᵢ (aᵢ - a_wm)(bᵢ - b_wm)
        denom = √(Σᵢ wᵢ (aᵢ - a_wm)²  ·  Σᵢ wᵢ (bᵢ - b_wm)²)
        r     = numer / denom

    With uniform weights (wᵢ = 1/N) this reduces exactly to the
    standard Pearson correlation — hence valid as a drop-in
    generalization. This differs from the earlier
    "``sqrt(w) * a`` then plain Pearson" heuristic, which centered
    by the unweighted mean and did not compute a weighted covariance.

    Parameters
    ----------
    a, b:
        Same-length 1-D arrays.
    w:
        Same-length non-negative weights. May be unnormalized;
        will be normalized to sum to 1 internally. Must not be
        all-zero.

    Returns
    -------
    Pearson r in [-1, 1]. Returns 0.0 when either vector has
    zero weighted variance (analogue of zero-variance in the
    unweighted case).
    """
    w = np.asarray(w, dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0.0:
        raise ValueError("Weights must have a positive sum.")
    w = w / w_sum
    a_wm = np.sum(w * a)
    b_wm = np.sum(w * b)
    a_c = a - a_wm
    b_c = b - b_wm
    var_a = np.sum(w * a_c * a_c)
    var_b = np.sum(w * b_c * b_c)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    cov = np.sum(w * a_c * b_c)
    raw = float(cov / np.sqrt(var_a * var_b))
    return max(-1.0, min(1.0, raw))


class SimilarityEngine:
    """Computes per-network and whole-cortex cognitive similarity scores."""

    def __init__(
        self,
        ica_atlas: ICANetworkAtlas,
        ica_mode: ICAMode = ICAMode.BINARY_MASK,
    ) -> None:
        self._atlas = ica_atlas
        self._ica_mode = ica_mode

    def compute_network_score(
        self,
        response_a: np.ndarray,
        response_b: np.ndarray,
        network: ICANetwork,
        ica_mode: Optional[ICAMode] = None,
    ) -> float:
        """Compute Pearson correlation for a single network.

        Parameters
        ----------
        response_a, response_b:
            Collapsed cortical responses of shape (20484,).
        network:
            The ICA network to restrict computation to.
        ica_mode:
            Overrides the instance default if provided.

        Returns
        -------
        float in [-1, 1].

        Raises
        ------
        ValueError if network is not a valid ICANetwork enum member.
        """
        if not isinstance(network, ICANetwork):
            raise ValueError(
                f"Unknown network {network!r}. "
                f"Valid identifiers: {_VALID_NETWORKS}"
            )

        mode = ica_mode if ica_mode is not None else self._ica_mode

        if mode is ICAMode.BINARY_MASK:
            indices = self._atlas.get_vertex_indices(network)
            a_masked = response_a[indices]
            b_masked = response_b[indices]
            return pearson_correlation(a_masked, b_masked)
        else:
            # CONTINUOUS_WEIGHTS — use proper weighted Pearson (E1 fix)
            component = self._atlas.get_component(network)
            w = np.abs(component)
            return weighted_pearson_correlation(response_a, response_b, w)

    def compute_profile(
        self,
        response_a: np.ndarray,
        response_b: np.ndarray,
        ica_mode: Optional[ICAMode] = None,
    ) -> CognitiveSimilarityProfile:
        """Compute a full CognitiveSimilarityProfile for all 5 networks.

        Parameters
        ----------
        response_a, response_b:
            Collapsed cortical responses of shape (20484,).
        ica_mode:
            Overrides the instance default if provided.

        Returns
        -------
        CognitiveSimilarityProfile with 5 NetworkScore entries and whole_cortex_score.
        """
        mode = ica_mode if ica_mode is not None else self._ica_mode

        network_scores: dict[ICANetwork, NetworkScore] = {}

        for network in ICANetwork:
            warning: Optional[str] = None

            if mode is ICAMode.BINARY_MASK:
                indices = self._atlas.get_vertex_indices(network)
                a_masked = response_a[indices]
                b_masked = response_b[indices]
                vertex_count = len(indices)

                # Detect zero variance before computing
                a_c = a_masked - a_masked.mean()
                b_c = b_masked - b_masked.mean()
                if np.linalg.norm(a_c) == 0.0 or np.linalg.norm(b_c) == 0.0:
                    warning = "zero variance"
                    log.warning(
                        "Zero-variance input for network %s (binary mask mode); returning 0.0",
                        network.value,
                    )

                score = pearson_correlation(a_masked, b_masked)

            else:
                # CONTINUOUS_WEIGHTS — weighted Pearson over all 20484 (E1 fix)
                component = self._atlas.get_component(network)
                w = np.abs(component)
                vertex_count = len(component)

                # Zero-variance under the weight distribution: weighted
                # mean-centered norm is zero. Same semantics as the
                # unweighted zero-variance guard, but respects the weights.
                w_norm = w / w.sum() if w.sum() > 0 else w
                a_wm = float(np.sum(w_norm * response_a))
                b_wm = float(np.sum(w_norm * response_b))
                var_a = float(np.sum(w_norm * (response_a - a_wm) ** 2))
                var_b = float(np.sum(w_norm * (response_b - b_wm) ** 2))
                if var_a <= 0.0 or var_b <= 0.0:
                    warning = "zero variance"
                    log.warning(
                        "Zero-variance input for network %s (continuous mode); returning 0.0",
                        network.value,
                    )

                score = weighted_pearson_correlation(response_a, response_b, w)

            network_scores[network] = NetworkScore(
                network=network,
                score=score,
                vertex_count=vertex_count,
                ica_mode=mode,
                warning=warning,
            )

        # Whole-cortex score: vertex-count-weighted average
        total_vertices = sum(ns.vertex_count for ns in network_scores.values())
        whole_cortex_score = sum(
            ns.score * ns.vertex_count / total_vertices
            for ns in network_scores.values()
        )

        return CognitiveSimilarityProfile(
            network_scores=network_scores,
            whole_cortex_score=whole_cortex_score,
            ica_mode=mode,
        )
