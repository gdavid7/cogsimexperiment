"""ICA Network Atlas — derives five canonical brain network masks from TRIBE v2 weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import FastICA

from cognitive_similarity.models import ICANetwork

log = logging.getLogger(__name__)

# Number of cortical surface vertices in fsaverage5 space
N_VERTICES = 20484
# Number of ICA components (one per canonical network)
N_COMPONENTS = 5
# Shape of the unseen-subject projection layer in best.ckpt
PROJECTION_SHAPE = (2048, N_VERTICES)


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
        _projection_matrix:
            Optional pre-computed projection matrix of shape (2048, 20484).
            When provided, skips HuggingFace loading entirely (useful for tests).
        """
        self._model_id = model_id
        self._top_percentile = top_percentile
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(".")
        self._cache_path = self._cache_dir / "ica_masks.npz"

        # components[i] is the full ICA component vector of shape (N_VERTICES,)
        self._components: np.ndarray  # shape (N_COMPONENTS, N_VERTICES)
        # masks[i] is a boolean array of shape (N_VERTICES,)
        self._masks: np.ndarray       # shape (N_COMPONENTS, N_VERTICES), dtype bool

        if _projection_matrix is not None:
            # Bypass HuggingFace — used for testing
            log.debug("ICANetworkAtlas: using provided projection matrix (test mode)")
            self._components, self._masks = self._compute_ica(_projection_matrix)
        elif self._cache_path.exists():
            log.debug("ICANetworkAtlas: loading masks from cache %s", self._cache_path)
            self._load_cache()
        else:
            log.info("ICANetworkAtlas: no cache found — loading model from HuggingFace")
            projection = self._load_projection_from_hf()
            self._components, self._masks = self._compute_ica(projection)
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

    @staticmethod
    def _network_index(network: ICANetwork) -> int:
        return ICANetworkAtlas.NETWORKS.index(network)

    def _compute_ica(
        self, projection: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FastICA on the projection matrix and threshold components.

        Parameters
        ----------
        projection:
            Matrix of shape (2048, 20484).

        Returns
        -------
        components:
            Array of shape (5, 20484) — one row per ICA component.
        masks:
            Boolean array of shape (5, 20484) — top ``top_percentile``
            vertices per component.
        """
        log.debug("ICANetworkAtlas: running FastICA(n_components=%d)", N_COMPONENTS)
        ica = FastICA(n_components=N_COMPONENTS, random_state=42, max_iter=1000)
        # FastICA expects shape (n_samples, n_features); projection is (2048, 20484)
        # Treat each of the 2048 rows as a sample, 20484 features.
        # The mixing matrix columns are the components over vertices.
        ica.fit(projection)
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
        """Search the state dict for a weight tensor of shape (2048, 20484)."""
        target_shape = PROJECTION_SHAPE
        candidates = []
        for key, tensor in state_dict.items():
            if hasattr(tensor, "shape") and tuple(tensor.shape) == target_shape:
                candidates.append((key, tensor))

        if not candidates:
            available = [
                f"{k}: {tuple(v.shape)}"
                for k, v in state_dict.items()
                if hasattr(v, "shape")
            ]
            raise ValueError(
                f"Could not find a tensor of shape {target_shape} in the checkpoint. "
                f"Available tensors:\n" + "\n".join(available[:30])
            )

        if len(candidates) > 1:
            log.warning(
                "Multiple tensors of shape %s found: %s. Using the first.",
                target_shape,
                [k for k, _ in candidates],
            )

        key, tensor = candidates[0]
        log.info("ICANetworkAtlas: using projection tensor '%s' of shape %s", key, target_shape)
        return tensor.float().numpy()

    def _save_cache(self) -> None:
        """Save computed components and masks to ica_masks.npz."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._cache_path,
            components=self._components,
            masks=self._masks,
        )
        log.info("ICANetworkAtlas: masks cached to %s", self._cache_path)

    def _load_cache(self) -> None:
        """Load components and masks from ica_masks.npz."""
        data = np.load(self._cache_path)
        self._components = data["components"]
        self._masks = data["masks"].astype(bool)
        log.debug("ICANetworkAtlas: loaded %d components from cache", len(self._components))

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
        )
