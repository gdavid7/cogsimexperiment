"""Data models for the Cognitive Similarity system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ICANetwork(Enum):
    PRIMARY_AUDITORY_CORTEX = "primary_auditory_cortex"
    LANGUAGE_NETWORK = "language_network"
    MOTION_DETECTION_MT_PLUS = "motion_detection_mt_plus"
    DEFAULT_MODE_NETWORK = "default_mode_network"
    VISUAL_SYSTEM = "visual_system"


class ICAMode(Enum):
    BINARY_MASK = "binary_mask"        # top 10% vertices, equal weight (default)
    CONTINUOUS_WEIGHTS = "continuous"  # full component vector as per-vertex weights


@dataclass
class BrainResponse:
    """Raw output from StimulusRunner — both model runs preserved for future use."""
    cortical: np.ndarray        # shape (n_timesteps, 20484) float32 — from cortical model
    subcortical: np.ndarray     # shape (n_timesteps, 8802) float32  — from subcortical model
    segments: list              # TRIBE v2 segment objects aligned with cortical


@dataclass
class Stimulus:
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    duration_s: Optional[float] = None  # if None, inferred from media
    stimulus_id: Optional[str] = None   # for ranking output; auto-generated if None

    def validate(self) -> None:
        """Raises ValueError if no modality is provided."""
        if self.video_path is None and self.audio_path is None and self.text_path is None:
            raise ValueError(
                "Stimulus must have at least one modality: "
                "video_path, audio_path, or text_path. All are None."
            )


@dataclass
class NetworkScore:
    network: ICANetwork
    score: float                   # Pearson r in [-1, 1]
    vertex_count: int              # ~2048 for binary mask, 20484 for continuous
    ica_mode: ICAMode              # which ICA mode was used
    warning: Optional[str] = None  # e.g., "zero variance"


@dataclass
class CognitiveSimilarityProfile:
    network_scores: dict[ICANetwork, NetworkScore]
    whole_cortex_score: float      # vertex-count-weighted average of 5 network scores
    ica_mode: ICAMode              # which ICA mode was used


@dataclass
class SimilarityResult:
    profile: CognitiveSimilarityProfile
    stimulus_a_id: str
    stimulus_b_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RankedEntry:
    stimulus_id: str
    score: float
    rank: int                      # 1 = most similar; ties share rank


@dataclass
class RankedResult:
    query_id: str
    rankings_by_network: dict[ICANetwork, list[RankedEntry]]
    rankings_whole_cortex: list[RankedEntry]


@dataclass
class ValidationCheck:
    description: str
    network: ICANetwork
    pair_a: tuple[str, str]        # (stimulus_id_1, stimulus_id_2)
    pair_b: tuple[str, str]
    expected: str                  # "sim(pair_a) > sim(pair_b)"
    passed: bool
    score_a: float
    score_b: float


@dataclass
class ValidationReport:
    checks: list[ValidationCheck]
    passed: int
    total: int
