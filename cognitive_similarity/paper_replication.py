"""Figure 4E replication: predicted contrast maps per TRIBEv2.pdf §5.9.

Per §5.9 (p. 22) "Contrast maps":

    "To obtain contrast maps for the visual experiments, we obtain contrast
    maps by simply selecting the predicted response at t=5 after the image
    is shown (which is the peak of the response as shown in figure 4A),
    and substracting the average responses at t=5 for the other categories."

This module computes exactly that contrast — selecting each stimulus'
response at the hemodynamic peak (cortical_response[5]) and subtracting
the mean peak response across the other categories. The result per
category is a (20484,) surface map that, per Figure 4C/D, should be
spatially focal in the paper's expected ROIs (FFA for faces, PPA for
places, EBA for bodies, VWFA for written characters).

A full replication of Figure 4E requires the Glasser HCP-MMP parcellation
(§5.9 footnote 3: FFC for FFA, PH for EBA/PPA, TE1a/PGi for VWFA), which
is not shipped in nilearn's built-in atlas set. Instead we validate the
spatial localization at the coarser ICA-mask level: each visual-category
contrast peak should fall within the NeuroSynth-derived VISUAL_SYSTEM
mask (Slice 2 A1/A2), and not spill out into auditory/language/motion
masks. This is a weaker check than ROI-level parcels, but still
directly tests what §5.9 claims: TRIBE's predicted responses localize
category-selective activation to visual cortex.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cognitive_similarity.ica_atlas import ICANetworkAtlas
from cognitive_similarity.models import ICANetwork

log = logging.getLogger(__name__)

_VERTICES = 20484
# Paper §5.9 / Fig 4A: hemodynamic peak is at t=5 s after stimulus onset
# in TRIBE's predicted fMRI timeseries (the 5-s offset is trained-in per
# §5.8). With our 1 Hz output that's index 5.
_PEAK_INDEX = 5


@dataclass
class ContrastMapResult:
    """One category's contrast map + localization summary."""

    category: str                     # e.g. "face", "place", "body", "written_character"
    contrast_map: np.ndarray          # (20484,) float32 — t=5 peak minus other-categories mean
    peak_vertex: int                  # argmax of the contrast map (most category-selective vertex)
    peak_value: float                 # contrast_map[peak_vertex]
    # Fraction of contrast_map's top-1% vertices that fall inside each
    # ICA network mask — used to test spatial localization.
    top_pct_fraction_per_network: dict[ICANetwork, float] = field(default_factory=dict)


@dataclass
class ContrastReplicationReport:
    """Per-category results + pass/fail summary."""

    results: dict[str, ContrastMapResult]
    # expected_network holds the ICA network each category should localize
    # to (all four visual categories → VISUAL_SYSTEM per §2.5 Fig 4C).
    expected_network: ICANetwork
    # Fraction threshold above which the localization check passes.
    localization_threshold: float

    @property
    def passed(self) -> int:
        return sum(
            1 for r in self.results.values()
            if r.top_pct_fraction_per_network.get(self.expected_network, 0.0)
               >= self.localization_threshold
        )

    @property
    def total(self) -> int:
        return len(self.results)


def compute_contrast_map(
    category_response: np.ndarray,
    other_responses: list[np.ndarray],
) -> np.ndarray:
    """Implement §5.9 contrast-map formula.

    Parameters
    ----------
    category_response:
        (20484,) float array — the category's predicted response at t=5.
    other_responses:
        List of (20484,) float arrays — one per *other* category. Typically
        these are also peak-extracted responses.

    Returns
    -------
    (20484,) float32 contrast map.
    """
    if len(other_responses) == 0:
        raise ValueError("other_responses must contain at least one entry.")
    cat = np.asarray(category_response, dtype=np.float32)
    others = np.stack(other_responses, axis=0).astype(np.float32)
    return (cat - others.mean(axis=0)).astype(np.float32)


def _peak_response(raw_cortical: np.ndarray) -> np.ndarray:
    """Per §5.9 / Fig 4A: return cortical_response[5] or last-available if T<6."""
    T = raw_cortical.shape[0]
    idx = _PEAK_INDEX if T > _PEAK_INDEX else T - 1
    if idx != _PEAK_INDEX:
        log.warning(
            "Peak index %d unavailable (T=%d); falling back to last timepoint %d",
            _PEAK_INDEX, T, idx,
        )
    return raw_cortical[idx].astype(np.float32)


def compute_all_contrasts(
    category_to_responses: dict[str, list[np.ndarray]],
) -> dict[str, ContrastMapResult]:
    """Compute §5.9 contrast maps for every category given the raw responses.

    Parameters
    ----------
    category_to_responses:
        ``{"face": [raw_cortical_face_01, raw_cortical_face_02, ...], ...}``.
        Each raw_cortical is (T, 20484) float — the full timeseries; we
        take the t=5 peak inside this function.

    Returns
    -------
    Dict mapping category → ContrastMapResult (contrast_map + peak info;
    top_pct_fraction_per_network left empty for the caller to fill in).
    """
    # Stage 1: per-category mean peak response
    peak_by_category: dict[str, np.ndarray] = {}
    for category, responses in category_to_responses.items():
        if not responses:
            raise ValueError(f"No responses provided for category {category!r}")
        peaks = np.stack([_peak_response(r) for r in responses], axis=0)
        peak_by_category[category] = peaks.mean(axis=0).astype(np.float32)

    # Stage 2: per-category contrast against the mean of other categories
    out: dict[str, ContrastMapResult] = {}
    categories = list(peak_by_category.keys())
    for category in categories:
        others = [peak_by_category[c] for c in categories if c != category]
        cmap = compute_contrast_map(peak_by_category[category], others)
        peak_vtx = int(np.argmax(cmap))
        out[category] = ContrastMapResult(
            category=category,
            contrast_map=cmap,
            peak_vertex=peak_vtx,
            peak_value=float(cmap[peak_vtx]),
        )
    return out


def localize_contrast_maps(
    contrasts: dict[str, ContrastMapResult],
    atlas: ICANetworkAtlas,
    *,
    top_percentile: float = 0.01,
) -> None:
    """Annotate each contrast map with its overlap with each ICA mask.

    For each category's contrast map, take the top ``top_percentile`` of
    vertices by absolute contrast value, and compute what fraction of
    those vertices fall inside each ICA network mask. Stored in
    ``result.top_pct_fraction_per_network``.

    Mutates the ContrastMapResult objects in-place.
    """
    k = int(np.ceil(_VERTICES * top_percentile))
    masks_by_network = {n: atlas.get_mask(n) for n in atlas.NETWORKS}

    for result in contrasts.values():
        # Top-k vertex indices by |contrast|
        top_indices = np.argsort(np.abs(result.contrast_map))[-k:]
        top_set = set(top_indices.tolist())

        fractions: dict[ICANetwork, float] = {}
        for network, mask in masks_by_network.items():
            mask_vertices = set(np.where(mask)[0].tolist())
            overlap = len(top_set & mask_vertices)
            fractions[network] = overlap / k if k else 0.0
        result.top_pct_fraction_per_network = fractions


def replicate_figure_4e(
    cache_dir: Path,
    atlas: ICANetworkAtlas,
    categories_to_stimuli: dict[str, list[str]],
    manifest_path: Optional[Path] = None,
    *,
    expected_network: ICANetwork = ICANetwork.VISUAL_SYSTEM,
    localization_threshold: float = 0.5,
) -> ContrastReplicationReport:
    """End-to-end Figure 4E-style replication.

    Loads each stimulus' cached raw_cortical.npy by content hash (from
    manifest.json), computes the §5.9 contrast maps, and reports whether
    each category's top-1% vertices concentrate inside
    ``expected_network`` (default: VISUAL_SYSTEM, the paper's expectation
    per §2.5 for all four visual localizer categories).

    Parameters
    ----------
    cache_dir:
        Cache root holding tensors/<hash>/raw_cortical.npy files and
        manifest.json (the file Modal's worker writes).
    atlas:
        An ICANetworkAtlas already loaded with the NeuroSynth-assigned
        labels.
    categories_to_stimuli:
        ``{"face": ["face_01", "face_02"], "place": [...], ...}`` —
        the stimulus ids to pool per category.
    manifest_path:
        Defaults to ``<cache_dir>/manifest.json``.
    expected_network:
        Which ICA network each category's contrast should localize to.
    localization_threshold:
        Minimum fraction of top-1% vertices that must fall in
        ``expected_network`` for the category to PASS.

    Returns
    -------
    ContrastReplicationReport
    """
    import json
    manifest_path = manifest_path or (cache_dir / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    id_to_hash = {e["stimulus_id"]: e["content_hash"] for e in manifest}

    category_to_responses: dict[str, list[np.ndarray]] = {}
    for category, stim_ids in categories_to_stimuli.items():
        responses = []
        for sid in stim_ids:
            h = id_to_hash.get(sid)
            if h is None:
                log.warning("Stimulus %r not in manifest; skipping", sid)
                continue
            raw_path = cache_dir / "tensors" / h / "raw_cortical.npy"
            if not raw_path.exists():
                log.warning("raw_cortical.npy missing for %r at %s", sid, raw_path)
                continue
            responses.append(np.load(raw_path))
        if not responses:
            raise RuntimeError(
                f"No cached responses found for category {category!r}; "
                "run scripts/run_inference_modal.py first."
            )
        category_to_responses[category] = responses

    contrasts = compute_all_contrasts(category_to_responses)
    localize_contrast_maps(contrasts, atlas)

    return ContrastReplicationReport(
        results=contrasts,
        expected_network=expected_network,
        localization_threshold=localization_threshold,
    )
