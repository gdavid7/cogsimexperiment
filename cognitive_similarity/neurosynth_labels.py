"""NeuroSynth-based ICA component labeling.

Per TRIBEv2.pdf §5.10 (p. 23):

    "at odds with Principal Component Analysis, the order of the components
    does not carry any meaning for ICA. [...] To characterize these brain
    maps functionally, we compare them with five functional maps obtained
    by NeuroSynth (Kent et al., 2026) with the following keywords:
    'primary auditory', 'language', 'motion', 'default network' and
    'visual'. These maps are resampled from volumetric to cortical space
    via nilearn's vol_to_surf, then compared with the ICA components via
    Pearson correlation across all vertices of the cortical surface."

This module implements that labeling step:

1. ``fetch_and_project_term_maps`` downloads the NeuroSynth v7 database via
   NiMARE (Kent et al. 2026), computes an MKDAChi2 association-test map
   per term, projects each volumetric map to fsaverage5, and caches the
   result as a .npz of (20484,) float32 arrays.
2. ``compute_label_assignment`` computes the 5×5 |Pearson r| matrix
   between ICA components and term maps, solves a bipartite assignment
   (``scipy.optimize.linear_sum_assignment``) so each ICANetwork maps to
   exactly one component, and returns the sign-adjusted components +
   assignment + correlations.

The fetching step is separate from the atlas so it can be run once (CLI
via ``scripts/fetch_neurosynth_maps.py``) and cached alongside the ICA
masks. The assignment step is cheap and runs on every ICA-atlas rebuild.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cognitive_similarity.models import ICANetwork

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants (all paper-grounded; cite source in changes)
# ----------------------------------------------------------------------

# Mapping ICANetwork → NeuroSynth term string. Terms are the five keywords
# §5.10 enumerates verbatim.
NEUROSYNTH_TERMS: dict[ICANetwork, str] = {
    ICANetwork.PRIMARY_AUDITORY_CORTEX: "primary auditory",
    ICANetwork.LANGUAGE_NETWORK:        "language",
    ICANetwork.MOTION_DETECTION_MT_PLUS: "motion",
    ICANetwork.DEFAULT_MODE_NETWORK:    "default network",
    ICANetwork.VISUAL_SYSTEM:           "visual",
}

# NiMARE's annotations column prefix for Neurosynth v7 abstract-TF-IDF terms.
_ANN_PREFIX = "terms_abstract_tfidf__"

# Standard NeuroSynth association-test threshold (Yarkoni et al. 2011):
# include a study if the term's TF-IDF weight exceeds 0.001.
_TERM_FREQUENCY_THRESHOLD = 0.001

# Radius (mm) of the MKDA kernel. NeuroSynth Compose (Kent et al. 2026)
# and the original neurosynth Python package both default to 10 mm.
_MKDA_KERNEL_RADIUS_MM = 10

# Minimum |Pearson r| we expect between an ICA component and its assigned
# NeuroSynth term map. Below this, the assignment is suspect — either
# ICA didn't find that network, or something upstream is wrong. The
# paper's Figure 6B shows diagonal values roughly 0.35-0.7; we flag
# anything below 0.15 as a loud warning, below 0.05 as an error.
_ASSIGNMENT_WARN_THRESHOLD = 0.15
_ASSIGNMENT_ERROR_THRESHOLD = 0.05


# ----------------------------------------------------------------------
# NeuroSynth term-map fetching + projection
# ----------------------------------------------------------------------


def fetch_and_project_term_maps(
    data_dir: Path,
    cache_file: Path,
    *,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Ensure ``cache_file`` exists and return its contents.

    If ``cache_file`` does not exist (or ``overwrite`` is True), download
    NeuroSynth v7 into ``data_dir`` via NiMARE, compute an MKDAChi2
    association-test map for each of the five §5.10 terms, project each
    volumetric map to fsaverage5 via ``nilearn.surface.vol_to_surf``,
    and save the resulting (20484,) float32 arrays as a .npz keyed by
    ``ICANetwork.value``.

    Returns
    -------
    dict mapping ``ICANetwork.value`` (str) to (20484,) float32 array.

    Notes
    -----
    NiMARE's ``fetch_neurosynth`` downloads ~100 MB. ``MKDAChi2.fit`` for
    each term takes ~5 s. First-run wall-clock is ~2-3 minutes; cached
    runs are <1 s.
    """
    if cache_file.exists() and not overwrite:
        data = np.load(cache_file)
        out = {k: data[k] for k in data.files}
        log.info(
            "Loaded NeuroSynth term maps from cache %s (%d keys)",
            cache_file, len(out),
        )
        return out

    # Import heavy deps lazily so importing this module doesn't incur
    # NiMARE / nilearn cost for callers that don't fetch.
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn.surface import vol_to_surf
    from nimare.extract import fetch_neurosynth
    from nimare.io import convert_neurosynth_to_dataset
    from nimare.meta.cbma import MKDAChi2

    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("NeuroSynth: fetching v7 database to %s", data_dir)
    files = fetch_neurosynth(
        data_dir=str(data_dir),
        version="7",
        return_type="files",
        source="abstract",
        vocab="terms",
    )
    entry = files[0] if isinstance(files, list) else files

    log.info("NeuroSynth: building NiMARE Dataset from fetched files")
    ds = convert_neurosynth_to_dataset(
        coordinates_file=entry["coordinates"],
        metadata_file=entry["metadata"],
        annotations_files=entry["features"],
    )
    log.info("NeuroSynth: Dataset has %d studies", len(ds.ids))

    log.info("NeuroSynth: fetching fsaverage5 surface mesh")
    fsavg = fetch_surf_fsaverage(mesh="fsaverage5")

    term_maps: dict[str, np.ndarray] = {}
    for network, term in NEUROSYNTH_TERMS.items():
        col = _ANN_PREFIX + term
        if col not in ds.annotations.columns:
            matches = [c for c in ds.annotations.columns if term.split()[0] in c.lower()][:6]
            raise ValueError(
                f"NeuroSynth annotation column {col!r} not found for term "
                f"{term!r}. Closest matches: {matches}"
            )

        term_ids = ds.get_studies_by_label(
            labels=[col], label_threshold=_TERM_FREQUENCY_THRESHOLD,
        )
        other_ids = list(set(ds.ids) - set(term_ids))
        log.info(
            "  term=%r: %d term-studies vs %d other-studies",
            term, len(term_ids), len(other_ids),
        )

        ds_term = ds.slice(term_ids)
        ds_other = ds.slice(other_ids)
        meta = MKDAChi2(kernel__r=_MKDA_KERNEL_RADIUS_MM)
        result = meta.fit(ds_term, ds_other)
        # "z_desc-association" is the Neurosynth "association test" map —
        # the one the paper's Figure 6B compares against (studies with the
        # term vs without, Chi-square test).
        img = result.get_map("z_desc-association")

        # Project volumetric map → fsaverage5 surface. Concatenate L/R to
        # match the canonical (20484,) layout: indices [0, 10242) = LH,
        # [10242, 20484) = RH.
        lh = vol_to_surf(img, fsavg["pial_left"])
        rh = vol_to_surf(img, fsavg["pial_right"])
        surf = np.concatenate([lh, rh]).astype(np.float32)
        if surf.shape != (20484,):
            raise RuntimeError(
                f"Unexpected surface shape {surf.shape} for term {term!r}; "
                "expected (20484,)."
            )
        term_maps[network.value] = surf

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, **term_maps)
    log.info("NeuroSynth: cached %d term maps to %s", len(term_maps), cache_file)
    return term_maps


# ----------------------------------------------------------------------
# Component → ICANetwork assignment
# ----------------------------------------------------------------------


def compute_label_assignment(
    components: np.ndarray,
    term_maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[ICANetwork, int], dict[ICANetwork, float]]:
    """Match each ICANetwork to the best ICA component by spatial Pearson r.

    Implements the §5.10 procedure: for each of the five components and
    each of the five NeuroSynth term maps, compute Pearson correlation
    across all 20484 cortical vertices, then solve a bipartite assignment
    that maximizes total ``|r|`` under the constraint that each network
    maps to a distinct component. If the best-matched component is
    negatively correlated with the term map (ICA components have
    sign ambiguity), flip the component's sign so the returned
    components are all positively oriented with respect to the
    paper's reference maps.

    Parameters
    ----------
    components:
        ``(N_COMPONENTS, N_VERTICES)`` float array from FastICA.
    term_maps:
        Dict ``ICANetwork.value → (N_VERTICES,) float array`` (output of
        ``fetch_and_project_term_maps``).

    Returns
    -------
    adjusted_components:
        Copy of ``components`` with signs possibly flipped so each
        matched pair has positive Pearson r.
    assignment:
        ``ICANetwork → int`` — component index assigned to that network.
    correlations:
        ``ICANetwork → float`` — *signed* r after sign adjustment (always
        ≥ 0 by construction).
    """
    from scipy.optimize import linear_sum_assignment

    ordered = [
        ICANetwork.PRIMARY_AUDITORY_CORTEX,
        ICANetwork.LANGUAGE_NETWORK,
        ICANetwork.MOTION_DETECTION_MT_PLUS,
        ICANetwork.DEFAULT_MODE_NETWORK,
        ICANetwork.VISUAL_SYSTEM,
    ]
    N = len(ordered)
    if components.shape[0] != N:
        raise ValueError(
            f"Expected components.shape[0]={N}, got {components.shape[0]}"
        )
    for network in ordered:
        key = network.value
        if key not in term_maps:
            raise KeyError(
                f"term_maps missing {key!r}; have {sorted(term_maps.keys())}"
            )

    # Build signed r matrix: rows = networks, cols = components.
    signed = np.zeros((N, N), dtype=np.float64)
    for i, network in enumerate(ordered):
        ref = term_maps[network.value].astype(np.float64)
        for j in range(N):
            signed[i, j] = np.corrcoef(ref, components[j].astype(np.float64))[0, 1]
    abs_r = np.abs(signed)

    # Hungarian on -|r| to maximize Σ|r|
    row_ind, col_ind = linear_sum_assignment(-abs_r)

    # Report the full matrix at INFO for visibility
    log.info("NeuroSynth × ICA signed correlation matrix:")
    log.info("                   " + "  ".join(f"comp{j}  " for j in range(N)))
    for i, network in enumerate(ordered):
        row = "  ".join(f"{signed[i, j]:+7.3f}" for j in range(N))
        log.info("  %-22s %s", network.value, row)

    # Apply the assignment and sign-flip if needed
    adjusted = components.copy()
    assignment: dict[ICANetwork, int] = {}
    correlations: dict[ICANetwork, float] = {}
    for i, j in zip(row_ind, col_ind):
        network = ordered[i]
        r = float(signed[i, j])
        if r < 0:
            adjusted[j] = -adjusted[j]
            r = -r
        assignment[network] = int(j)
        correlations[network] = r

    # Sanity checks
    for network, r in correlations.items():
        if r < _ASSIGNMENT_ERROR_THRESHOLD:
            raise RuntimeError(
                f"ICA component assigned to {network.value!r} has |r|={r:.3f} "
                f"with its NeuroSynth reference — below error threshold "
                f"{_ASSIGNMENT_ERROR_THRESHOLD}. ICA likely did not recover "
                "this network; investigate before caching."
            )
        if r < _ASSIGNMENT_WARN_THRESHOLD:
            log.warning(
                "ICA component assigned to %s has weak |r|=%.3f (below "
                "warn threshold %.2f). Paper Fig 6B shows r ≈ 0.35–0.70.",
                network.value, r, _ASSIGNMENT_WARN_THRESHOLD,
            )

    return adjusted, assignment, correlations
