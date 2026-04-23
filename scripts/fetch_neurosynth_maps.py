"""Fetch NeuroSynth term maps, project to fsaverage5, cache for ICANetworkAtlas.

Per TRIBEv2.pdf §5.10, ICA component order is arbitrary and the paper
assigns labels by spatial Pearson correlation against five NeuroSynth
keyword maps ("primary auditory", "language", "motion", "default
network", "visual"). This script runs that fetch + projection step
once and caches the result at <cache-dir>/neurosynth_maps.npz; the
next ICANetworkAtlas(...) that sees this file applies the bipartite
label assignment automatically.

Usage:
    python scripts/fetch_neurosynth_maps.py --cache-dir /path/to/cognitive-similarity-cache

First-run cost (measured Apr 2026):
    - NiMARE NeuroSynth v7 download: ~100 MB, ~20 s on residential broadband
    - NiMARE Dataset construction from fetched files: ~90 s
    - MKDAChi2 per term (× 5): ~5 s each
    - vol_to_surf projection to fsaverage5 (× 5): ~2 s each
    - Total: ~3 minutes

Subsequent runs that find neurosynth_maps.npz are a no-op.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make cognitive_similarity importable without `pip install -e .`, matching
# the pattern in scripts/validate_ibc.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cognitive_similarity.ica_atlas import NEUROSYNTH_MAPS_FILENAME  # noqa: E402
from cognitive_similarity.neurosynth_labels import (                 # noqa: E402
    fetch_and_project_term_maps,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Where to write neurosynth_maps.npz (same directory as ica_masks.npz).",
    )
    ap.add_argument(
        "--neurosynth-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the NeuroSynth v7 database files. Defaults to "
            "<cache-dir>/_neurosynth. Large (~100 MB); you can reuse across "
            "projects by pointing this somewhere shared."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download + re-compute even if neurosynth_maps.npz already exists.",
    )
    ap.add_argument(
        "--invalidate-ica-cache",
        action="store_true",
        help=(
            "Delete <cache-dir>/ica_masks.npz after generating the NeuroSynth "
            "maps so the next ICANetworkAtlas(...) re-runs FastICA and applies "
            "the NeuroSynth assignment. Without this flag, if ica_masks.npz "
            "already exists it will be relabeled in-place on next load."
        ),
    )
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir.expanduser().resolve()
    data_dir: Path = (
        args.neurosynth_data_dir.expanduser().resolve()
        if args.neurosynth_data_dir
        else cache_dir / "_neurosynth"
    )
    cache_file = cache_dir / NEUROSYNTH_MAPS_FILENAME

    print(f"Cache dir           : {cache_dir}")
    print(f"NeuroSynth data dir : {data_dir}")
    print(f"Output maps file    : {cache_file}")

    fetch_and_project_term_maps(
        data_dir=data_dir,
        cache_file=cache_file,
        overwrite=args.overwrite,
    )

    if args.invalidate_ica_cache:
        ica_cache = cache_dir / "ica_masks.npz"
        if ica_cache.exists():
            backup = ica_cache.with_suffix(".npz.pre_neurosynth_backup")
            ica_cache.rename(backup)
            print(f"Moved existing ICA cache to {backup}")
        else:
            print("No ica_masks.npz present — nothing to invalidate.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
