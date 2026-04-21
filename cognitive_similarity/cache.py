"""ResponseCache — content-addressed serialization of Brain_Response tensors."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from cognitive_similarity.models import Stimulus

log = logging.getLogger(__name__)

# Expected shapes for validation on load
_CORTICAL_VERTICES = 20484


class ResponseCache:
    """Content-addressed cache for raw and collapsed brain response tensors.

    Directory layout::

        <cache_dir>/
        └── tensors/
            └── <sha256_hash>/
                ├── raw_cortical.npy      # (T, 20484) float32
                └── collapsed.npy         # (20484,)   float32
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._tensors_dir = self._cache_dir / "tensors"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_collapsed(self, stimulus: Stimulus) -> Optional[np.ndarray]:
        """Returns cached Collapsed_Response [20484] or None."""
        path = self._collapsed_path(stimulus)
        if not path.exists():
            return None
        return self._load_npy(path, expected_shape=(20484,), label="collapsed")

    def put_collapsed(self, stimulus: Stimulus, collapsed: np.ndarray) -> None:
        """Saves collapsed response to <cache_dir>/tensors/<hash>/collapsed.npy"""
        path = self._collapsed_path(stimulus)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, collapsed.astype(np.float32))

    def get_raw(self, stimulus: Stimulus) -> Optional[np.ndarray]:
        """Returns raw_cortical or None."""
        h = self._content_hash(stimulus)
        cortical_path = self._tensors_dir / h / "raw_cortical.npy"

        if not cortical_path.exists():
            return None

        return self._load_npy(
            cortical_path,
            expected_ndim=2,
            expected_last_dim=_CORTICAL_VERTICES,
            label="raw_cortical",
        )

    def put_raw(self, stimulus: Stimulus, raw_cortical: np.ndarray) -> None:
        """Saves raw_cortical to <cache_dir>/tensors/<content_hash>/raw_cortical.npy"""
        h = self._content_hash(stimulus)
        tensor_dir = self._tensors_dir / h
        tensor_dir.mkdir(parents=True, exist_ok=True)
        np.save(tensor_dir / "raw_cortical.npy", raw_cortical.astype(np.float32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_collapsed_by_hash(self, content_hash: str) -> Optional[np.ndarray]:
        """Load collapsed.npy directly by content hash, without needing the stimulus file.

        Used by ValidationSuite which already has content hashes from manifest.json.
        """
        path = self._tensors_dir / content_hash / "collapsed.npy"
        if not path.exists():
            return None
        return self._load_npy(path, expected_shape=(20484,), label="collapsed")

    def _content_hash(self, stimulus: Stimulus) -> str:
        """SHA-256 of the raw bytes of all modality inputs."""
        h = hashlib.sha256()
        for path in [stimulus.video_path, stimulus.audio_path, stimulus.text_path]:
            if path is not None:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
        return h.hexdigest()

    def _collapsed_path(self, stimulus: Stimulus) -> Path:
        h = self._content_hash(stimulus)
        return self._tensors_dir / h / "collapsed.npy"

    def _load_npy(
        self,
        path: Path,
        *,
        expected_shape: Optional[tuple] = None,
        expected_ndim: Optional[int] = None,
        expected_last_dim: Optional[int] = None,
        label: str = "tensor",
    ) -> Optional[np.ndarray]:
        """Load a .npy file, returning None (with WARNING) on any error."""
        try:
            arr = np.load(path)
        except Exception as exc:
            log.warning("Cache file corrupted (%s): %s — %s", label, path, exc)
            return None

        if expected_shape is not None and arr.shape != expected_shape:
            log.warning(
                "Cache file wrong shape (%s): expected %s, got %s — %s",
                label, expected_shape, arr.shape, path,
            )
            return None

        if expected_ndim is not None and arr.ndim != expected_ndim:
            log.warning(
                "Cache file wrong ndim (%s): expected %d, got %d — %s",
                label, expected_ndim, arr.ndim, path,
            )
            return None

        if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
            log.warning(
                "Cache file wrong last dim (%s): expected %d, got %d — %s",
                label, expected_last_dim, arr.shape[-1], path,
            )
            return None

        return arr
