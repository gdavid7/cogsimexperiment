"""ICA Network Atlas — derives five canonical brain network masks from TRIBE v2 weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import sklearn
from sklearn.decomposition import FastICA

from cognitive_similarity.models import ICANetwork
from cognitive_similarity.neurosynth_labels import (
    NEUROSYNTH_TERMS,
    compute_label_assignment,
)

log = logging.getLogger(__name__)

# Number of cortical surface vertices in fsaverage5 space
N_VERTICES = 20484
# Number of ICA components (one per canonical network)
N_COMPONENTS = 5
# Shape of the unseen-subject projection layer in best.ckpt
PROJECTION_SHAPE = (2048, N_VERTICES)
# FastICA iteration cap. Empirically the real best.ckpt projection needs
# ~3998 iterations to converge (sklearn 1.8, tol=1e-4, random_state=42);
# the prior max_iter=1000 was silently capping the optimizer mid-descent,
# producing seed-dependent non-converged components. Per TRIBEv2.pdf §5.10
# the paper uses "default parameters" but doesn't report convergence; we
# choose a value well above observed convergence with headroom for
# numerical variation across sklearn versions. If this cap is ever hit
# that's a signal to investigate, not to bump blindly.
FASTICA_MAX_ITER = 10_000

# NeuroSynth term maps cached alongside the ICA masks in the same cache
# directory. Populated by scripts/fetch_neurosynth_maps.py; consumed by
# ICANetworkAtlas at init if present. Layout: .npz keyed by
# ICANetwork.value, each value a (N_VERTICES,) float32 surface map.
NEUROSYNTH_MAPS_FILENAME = "neurosynth_maps.npz"


class ICANetworkAtlas:
    """Derives five ICA network masks from TRIBE v2's unseen-subject projection layer.

    On first use, loads best.ckpt from HuggingFace, extracts the projection
    layer of shape (2048, 20484), runs FastICA(n_components=5), and thresholds
    each component at the top 10% of absolute values to produce binary masks.
    Computed masks are cached locally to avoid recomputation.
    """

    NETWORKS = [
        ICANetwork.PRIMARY_AUDITORY_CORTEX,
        ICANetwork.LANGUAGE_NETWORK,
        ICANetwork.MOTION_DETECTION_MT_PLUS,
        ICANetwork.DEFAULT_MODE_NETWORK,
        ICANetwork.VISUAL_SYSTEM,
    ]

    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        top_percentile: float = 0.10,
        cache_dir: Optional[str] = None,
        use_neurosynth_labels: bool = True,
        _projection_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Loads best.ckpt from HuggingFace, extracts the unseen-subject
        projection layer (shape 2048 × 20484, where 2048 = low_rank_head
        bottleneck per config.yaml), runs FastICA(n_components=5),
        and thresholds each component at top_percentile to produce binary masks.
        Computed masks are cached locally so this only runs once.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier (default: "facebook/tribev2").
        top_percentile:
            Fraction of vertices to include in each binary mask (default: 0.10).
        cache_dir:
            Directory for caching computed masks. Defaults to the current
            working directory.
        use_neurosynth_labels:
            If True (default), apply the §5.10 NeuroSynth-based bipartite
            label assignment when ``neurosynth_maps.npz`` exists in
            ``cache_dir``. If that file is missing, log a WARNING and
            fall back to positional labels (not paper-faithful; run
            scripts/fetch_neurosynth_maps.py to generate the maps).
            Set False for tests with synthetic projections — positional
            labels are used unconditionally.
        _projection_matrix:
            Optional pre-computed projection matrix of shape (2048, 20484).
            When provided, skips HuggingFace loading entirely (useful for tests).
        """
        self._model_id = model_id
        self._top_percentile = top_percentile
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(".")
        self._cache_path = self._cache_dir / "ica_masks.npz"
        self._neurosynth_maps_path = self._cache_dir / NEUROSYNTH_MAPS_FILENAME
        self._use_neurosynth_labels = use_neurosynth_labels

        # components[i] is the full ICA component vector of shape (N_VERTICES,)
        self._components: np.ndarray  # shape (N_COMPONENTS, N_VERTICES)
        # masks[i] is a boolean array of shape (N_VERTICES,)
        self._masks: np.ndarray       # shape (N_COMPONENTS, N_VERTICES), dtype bool
        # FastICA convergence marker; set by _compute_ica, read from cache on load
        self._fastica_n_iter: Optional[int] = None
        # sklearn version that produced the cached components (A3 provenance)
        self._sklearn_version: Optional[str] = None
        # ICANetwork → component-index assignment (A2). Initialized to positional
        # identity; overwritten by NeuroSynth assignment when available.
        self._label_assignment: dict[ICANetwork, int] = {
            n: i for i, n in enumerate(self.NETWORKS)
        }
        # ICANetwork → |Pearson r| with its NeuroSynth reference map; 0.0 when
        # assignment is positional (no NeuroSynth maps available).
        self._label_correlations: dict[ICANetwork, float] = {
            n: 0.0 for n in self.NETWORKS
        }
        # Record how labels were determined so callers + caches can audit.
        # "positional" = NETWORKS list order; "neurosynth" = §5.10 bipartite match.
        self._label_source: str = "positional"

        if _projection_matrix is not None:
            # Bypass HuggingFace — used for testing. Synthetic random-normal
            # matrices have no true independent components so FastICA won't
            # converge; don't let that fail the test path.
            log.debug("ICANetworkAtlas: using provided projection matrix (test mode)")
            self._components, self._masks = self._compute_ica(
                _projection_matrix, strict=False
            )
            # Test path never attempts NeuroSynth labeling — positional only.
        elif self._cache_path.exists():
            log.debug("ICANetworkAtlas: loading masks from cache %s", self._cache_path)
            self._load_cache()
            # If cache predates NeuroSynth labeling and maps are now available,
            # compute assignment on the already-loaded components and re-save.
            if self._use_neurosynth_labels and self._label_source == "positional":
                self._maybe_apply_neurosynth_labels(resave_on_success=True)
        else:
            log.info("ICANetworkAtlas: no cache found — loading model from HuggingFace")
            projection = self._load_projection_from_hf()
            self._components, self._masks = self._compute_ica(projection)
            self._sklearn_version = sklearn.__version__
            if self._use_neurosynth_labels:
                self._maybe_apply_neurosynth_labels(resave_on_success=False)
            self._save_cache()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_mask(self, network: ICANetwork) -> np.ndarray:
        """Returns boolean array of shape [20484] for the given network.

        Binary mask: top ``top_percentile`` of vertices by absolute ICA
        component value.
        """
        idx = self._network_index(network)
        return self._masks[idx].copy()

    def get_vertex_indices(self, network: ICANetwork) -> np.ndarray:
        """Returns integer index array (~2048 vertices) for the given network."""
        return np.where(self.get_mask(network))[0]

    def get_component(self, network: ICANetwork) -> np.ndarray:
        """Returns the full continuous ICA component vector of shape [20484].

        Values represent each vertex's association strength with this network.
        Used for continuous weighting mode in SimilarityEngine.
        """
        idx = self._network_index(network)
        return self._components[idx].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _network_index(self, network: ICANetwork) -> int:
        """Return the component index assigned to *network*.

        Uses the NeuroSynth-derived assignment when one has been computed
        (see ``_maybe_apply_neurosynth_labels``); otherwise falls back to
        the positional ``NETWORKS`` order, which is the legacy behavior.
        """
        try:
            return self._label_assignment[network]
        except KeyError as exc:
            raise ValueError(
                f"Unknown network {network!r}. "
                f"Valid networks: {[n.value for n in self.NETWORKS]}"
            ) from exc

    def _maybe_apply_neurosynth_labels(self, *, resave_on_success: bool) -> None:
        """Attempt to relabel components via the §5.10 NeuroSynth procedure.

        Reads ``<cache_dir>/neurosynth_maps.npz`` (produced by
        ``scripts/fetch_neurosynth_maps.py``). If the file doesn't exist,
        logs a WARNING and leaves positional labels in place. If it does
        exist, computes the bipartite assignment, sign-flips components
        as needed, updates ``_label_assignment`` and ``_label_correlations``,
        and optionally re-saves the cache.
        """
        if not self._neurosynth_maps_path.exists():
            log.warning(
                "NeuroSynth term maps not found at %s — falling back to "
                "POSITIONAL ICA labels. Run scripts/fetch_neurosynth_maps.py "
                "to enable paper-faithful §5.10 labeling.",
                self._neurosynth_maps_path,
            )
            return

        log.info(
            "Applying NeuroSynth §5.10 labels from %s", self._neurosynth_maps_path
        )
        data = np.load(self._neurosynth_maps_path)
        term_maps = {k: data[k] for k in data.files}

        # Compute bipartite assignment; may sign-flip components.
        adjusted, assignment, correlations = compute_label_assignment(
            self._components, term_maps
        )

        # If any components were sign-flipped, rebuild masks from the
        # adjusted components so top-|x| vertex selection uses the same
        # oriented component.
        sign_flipped = [j for j in range(N_COMPONENTS)
                        if not np.array_equal(adjusted[j], self._components[j])]
        if sign_flipped:
            log.info(
                "Sign-flipped components %s to align with NeuroSynth references",
                sign_flipped,
            )
            self._components = adjusted
            threshold = 1.0 - self._top_percentile
            for j in sign_flipped:
                abs_vals = np.abs(self._components[j])
                cutoff = np.quantile(abs_vals, threshold)
                self._masks[j] = abs_vals >= cutoff

        self._label_assignment = assignment
        self._label_correlations = correlations
        self._label_source = "neurosynth"

        log.info("NeuroSynth label assignment:")
        for network in self.NETWORKS:
            log.info(
                "  %-24s → comp %d  |r|=%.3f",
                network.value,
                assignment[network],
                correlations[network],
            )

        if resave_on_success:
            self._save_cache()
            log.info("Re-saved ica_masks.npz with NeuroSynth assignment")

    def _compute_ica(
        self,
        projection: np.ndarray,
        strict: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FastICA on the projection matrix and threshold components.

        Parameters
        ----------
        projection:
            Matrix of shape (2048, 20484).
        strict:
            If True (production path — real TRIBE checkpoint), raise
            RuntimeError when FastICA hits the iteration cap without
            converging. If False (test path — synthetic random-normal
            projections), log a WARNING but continue: random Gaussian
            matrices have no true independent components, so FastICA
            legitimately won't converge on them.

        Returns
        -------
        components:
            Array of shape (5, 20484) — one row per ICA component.
        masks:
            Boolean array of shape (5, 20484) — top ``top_percentile``
            vertices per component.
        """
        log.debug("ICANetworkAtlas: running FastICA(n_components=%d)", N_COMPONENTS)
        ica = FastICA(
            n_components=N_COMPONENTS,
            random_state=42,
            max_iter=FASTICA_MAX_ITER,
        )
        # FastICA expects shape (n_samples, n_features); projection is (2048, 20484)
        # Treat each of the 2048 rows as a sample, 20484 features.
        # The mixing matrix columns are the components over vertices.
        ica.fit(projection)

        # A non-converged fit produces seed-dependent components that silently
        # break the paper's §5.10 interpretation (the unmixing matrix W hasn't
        # stabilized at the algorithm's optimum). Fail loudly on the real
        # checkpoint; only soft-warn on synthetic test matrices.
        if ica.n_iter_ >= FASTICA_MAX_ITER:
            if strict:
                raise RuntimeError(
                    f"FastICA did not converge after {FASTICA_MAX_ITER} iterations "
                    f"(n_iter_={ica.n_iter_}). Components are unstable and component "
                    "ordering becomes seed-dependent. Investigate numerical "
                    "conditioning of the projection matrix before bumping the cap."
                )
            log.warning(
                "FastICA did not converge on the provided projection "
                "(n_iter_=%d); proceeding because strict=False. Expected "
                "for synthetic random-normal test matrices.",
                ica.n_iter_,
            )
        self._fastica_n_iter = int(ica.n_iter_)
        log.info(
            "ICANetworkAtlas: FastICA fit finished in %d iterations (cap %d)",
            ica.n_iter_,
            FASTICA_MAX_ITER,
        )

        # components_ has shape (n_components, n_features) = (5, 20484)
        components = ica.components_.astype(np.float32)  # (5, 20484)

        threshold = 1.0 - self._top_percentile
        masks = np.zeros((N_COMPONENTS, N_VERTICES), dtype=bool)
        for i in range(N_COMPONENTS):
            abs_vals = np.abs(components[i])
            cutoff = np.quantile(abs_vals, threshold)
            masks[i] = abs_vals >= cutoff

        return components, masks

    def _load_projection_from_hf(self) -> np.ndarray:
        """Load best.ckpt from HuggingFace and extract the unseen-subject
        projection layer of shape (2048, 20484).
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to load the TRIBE v2 model. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        log.info("ICANetworkAtlas: downloading best.ckpt from %s", self._model_id)
        ckpt_path = hf_hub_download(repo_id=self._model_id, filename="best.ckpt")

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to load best.ckpt. "
                "Install it with: pip install torch"
            ) from exc

        log.info("ICANetworkAtlas: loading checkpoint from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # The checkpoint may be a raw state dict or wrapped under a key
        state_dict = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]

        projection = self._find_projection_tensor(state_dict)
        return projection

    @staticmethod
    def _find_projection_tensor(state_dict: dict) -> np.ndarray:
        """Find the unseen-subject projection in the checkpoint.

        Per TRIBEv2.pdf §5.3, the subject block is a tensor of shape
        (S, D_bottleneck, N_targets). In the public checkpoint the
        unseen-subject head is stored with S=1 as ``model.predictor.weights``
        of shape (1, 2048, 20484); we squeeze the leading axis to obtain
        the (2048, 20484) matrix the design describes.

        Selection rule: prefer keys ending in ``.predictor.weights`` (the
        paper's subject-block location), falling back to any shape match
        only if no such key exists. This avoids silently picking up an
        unrelated tensor of the same shape that a future TRIBE release or
        fine-tune might add.
        """
        target_2d = PROJECTION_SHAPE
        target_3d = (1,) + target_2d
        preferred_suffix = ".predictor.weights"

        candidates_2d: list[tuple[str, object]] = []
        candidates_3d: list[tuple[str, object]] = []
        for key, tensor in state_dict.items():
            if not hasattr(tensor, "shape"):
                continue
            shape = tuple(tensor.shape)
            if shape == target_2d:
                candidates_2d.append((key, tensor))
            elif shape == target_3d:
                candidates_3d.append((key, tensor))

        if candidates_2d:
            candidates = candidates_2d
            squeeze = False
        elif candidates_3d:
            candidates = candidates_3d
            squeeze = True
        else:
            available = [
                f"{k}: {tuple(v.shape)}"
                for k, v in state_dict.items()
                if hasattr(v, "shape")
            ]
            raise ValueError(
                f"Could not find a projection tensor of shape {target_2d} "
                f"or {target_3d} in the checkpoint. Available tensors:\n"
                + "\n".join(available[:30])
            )

        # Prefer the ``.predictor.weights`` key; if no candidate matches,
        # fall back to the first shape-matching candidate with a warning.
        preferred = [(k, t) for k, t in candidates if k.endswith(preferred_suffix)]
        if preferred:
            if len(preferred) > 1:
                log.warning(
                    "Multiple '.predictor.weights' candidates found: %s. Using the first.",
                    [k for k, _ in preferred],
                )
            key, tensor = preferred[0]
        else:
            log.warning(
                "No key ending in '%s' matched shape %s; falling back to the first "
                "shape-matching tensor. Candidates: %s",
                preferred_suffix,
                target_2d if not squeeze else target_3d,
                [k for k, _ in candidates],
            )
            key, tensor = candidates[0]

        arr = tensor.float().numpy()
        if squeeze:
            arr = arr.squeeze(axis=0)
        log.info(
            "ICANetworkAtlas: using projection tensor '%s' -> shape %s",
            key,
            arr.shape,
        )
        return arr

    def _save_cache(self) -> None:
        """Save components, masks, and A3 provenance to ica_masks.npz.

        Schema
        ------
        components        : (N_COMPONENTS, N_VERTICES) float32 — sign-adjusted
        masks             : (N_COMPONENTS, N_VERTICES) bool
        label_source      : 0-d str — "positional" or "neurosynth"
        label_assignment_keys   : (N_COMPONENTS,) str — ICANetwork.value
        label_assignment_values : (N_COMPONENTS,) int — component index
        label_correlations_keys : (N_COMPONENTS,) str — ICANetwork.value
        label_correlations_values : (N_COMPONENTS,) float — |r| with NeuroSynth
        sklearn_version   : 0-d str — sklearn package version
        fastica_n_iter    : 0-d int — iterations FastICA took to converge
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        keys = [n.value for n in self.NETWORKS]
        np.savez(
            self._cache_path,
            components=self._components,
            masks=self._masks,
            label_source=np.array(self._label_source),
            label_assignment_keys=np.array(keys),
            label_assignment_values=np.array(
                [self._label_assignment[n] for n in self.NETWORKS], dtype=np.int64
            ),
            label_correlations_keys=np.array(keys),
            label_correlations_values=np.array(
                [self._label_correlations[n] for n in self.NETWORKS], dtype=np.float64
            ),
            sklearn_version=np.array(self._sklearn_version or ""),
            fastica_n_iter=np.array(self._fastica_n_iter or -1, dtype=np.int64),
        )
        log.info("ICANetworkAtlas: masks cached to %s", self._cache_path)

    def _load_cache(self) -> None:
        """Load components, masks, and provenance from ica_masks.npz.

        Accepts both the pre-A3 (components+masks only) schema and the
        post-A3 extended schema. Missing fields fall back to defaults
        (positional labels, empty version/n_iter) and log a WARNING so
        a stale cache is visible.
        """
        data = np.load(self._cache_path)
        self._components = data["components"]
        self._masks = data["masks"].astype(bool)

        if "label_source" in data.files:
            self._label_source = str(data["label_source"])
            keys = [str(k) for k in data["label_assignment_keys"].tolist()]
            vals = data["label_assignment_values"].tolist()
            corr_vals = data["label_correlations_values"].tolist()
            self._label_assignment = {
                ICANetwork(k): int(v) for k, v in zip(keys, vals)
            }
            self._label_correlations = {
                ICANetwork(k): float(v) for k, v in zip(keys, corr_vals)
            }
            self._sklearn_version = str(data["sklearn_version"]) or None
            n_iter = int(data["fastica_n_iter"])
            self._fastica_n_iter = n_iter if n_iter >= 0 else None
        else:
            log.warning(
                "Legacy ica_masks.npz (pre-A3 schema) at %s — no provenance. "
                "Using positional labels; regenerate for paper-faithful labeling.",
                self._cache_path,
            )
            # Defaults already set in __init__

        # Sklearn drift warning (A3): if the cache was produced by a
        # different sklearn, FastICA internals may differ even with the
        # same random_state. Not fatal, but the user should know.
        if self._sklearn_version and self._sklearn_version != sklearn.__version__:
            log.warning(
                "ica_masks.npz was produced by sklearn %s; current is %s. "
                "Components may differ if regenerated. Delete the cache to "
                "force a rebuild.",
                self._sklearn_version, sklearn.__version__,
            )

        log.debug(
            "ICANetworkAtlas: loaded %d components from cache (label_source=%s)",
            len(self._components), self._label_source,
        )

    # ------------------------------------------------------------------
    # Factory classmethod for testing / pre-computed matrices
    # ------------------------------------------------------------------

    @classmethod
    def from_projection_matrix(
        cls,
        projection_matrix: np.ndarray,
        top_percentile: float = 0.10,
    ) -> "ICANetworkAtlas":
        """Create an ICANetworkAtlas from a pre-computed projection matrix.

        Useful for testing without HuggingFace access.

        Parameters
        ----------
        projection_matrix:
            Matrix of shape (2048, 20484).
        top_percentile:
            Fraction of vertices to include in each binary mask.
        """
        return cls(
            _projection_matrix=projection_matrix,
            top_percentile=top_percentile,
            use_neurosynth_labels=False,
        )
