"""Run ValidationSuite locally against a synced IBC cache.

Usage:
    python scripts/validate_ibc.py --cache-dir /path/to/cognitive-similarity-cache

Assumes the cache directory contains:
    - manifest.json      (produced by remote_inference.ipynb Cell 3)
    - tensors/<hash>/raw_cortical.npy for each manifest entry
      (produced by remote_inference.ipynb Cell 4 on Colab)

Materializes collapsed.npy for any stimulus that lacks one, then runs
ValidationSuite against the real HuggingFace-loaded ICA atlas.

Requires: torch, huggingface_hub (for the one-time best.ckpt download
and projection-layer extraction used by ICANetworkAtlas). On first run
this downloads ~676 MB of checkpoint and runs FastICA once; the masks
are cached at <cache-dir>/ica_masks.npz for subsequent runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `cognitive_similarity` importable without a prior `pip install -e .`,
# so the script runs cleanly from a throwaway venv that only has the
# runtime deps (torch, huggingface_hub, sklearn, nilearn, pandas).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cognitive_similarity import CognitiveSimilarity, Stimulus  # noqa: E402
from cognitive_similarity.cache import ResponseCache             # noqa: E402
from cognitive_similarity.collapsing import TemporalCollapser    # noqa: E402
from cognitive_similarity.validation import ValidationSuite      # noqa: E402


def _materialize_collapsed(cache_dir: Path, manifest: list[dict]) -> None:
    """Compute collapsed.npy from raw_cortical.npy for any entry that lacks one.

    Duration is left unset so TemporalCollapser infers it from T * tr_s;
    this keeps the script correct for any stimulus length without
    hardcoding assumptions from the Colab-side MP4 duration.
    """
    collapser = TemporalCollapser()
    for entry in manifest:
        tensor_dir = cache_dir / entry["tensor_dir"]
        collapsed_path = tensor_dir / "collapsed.npy"
        raw_path = tensor_dir / "raw_cortical.npy"
        if collapsed_path.exists():
            print(f"  {entry['stimulus_id']:<25}  (cached)")
            continue
        if not raw_path.exists():
            print(f"  {entry['stimulus_id']:<25}  SKIPPED — missing raw_cortical.npy")
            continue
        raw = np.load(raw_path)
        stim = Stimulus(
            video_path=entry["local_path"],
            stimulus_id=entry["stimulus_id"],
        )
        collapsed = collapser.collapse(raw, stim, tr_s=1.0)
        np.save(collapsed_path, collapsed.astype(np.float32))
        print(f"  {entry['stimulus_id']:<25}  raw {raw.shape} -> collapsed {collapsed.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to the synced cognitive-similarity-cache directory "
             "(contains manifest.json and tensors/).",
    )
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    print(f"[1/3] Cache: {cache_dir}")
    print(f"      {len(manifest)} stimuli in manifest")

    print(f"\n[2/3] Materializing collapsed tensors from raw_cortical...")
    _materialize_collapsed(cache_dir, manifest)

    print(f"\n[3/3] Loading real ICA atlas (downloads best.ckpt from HF on first run)...")
    cs = CognitiveSimilarity(cache_dir=str(cache_dir))
    cache = ResponseCache(str(cache_dir))
    suite = ValidationSuite(
        engine=cs._engine,
        cache=cache,
        manifest_path=str(manifest_path),
    )
    report = suite.run()

    print(f"\n==================== RESULTS ====================")
    print(f"{report.passed}/{report.total} checks passed\n")
    for c in report.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.network.value:<28}  sim_A={c.score_a:+.4f}  sim_B={c.score_b:+.4f}")
        print(f"         {c.description}")

    return 0 if report.passed == report.total else 2


if __name__ == "__main__":
    sys.exit(main())
