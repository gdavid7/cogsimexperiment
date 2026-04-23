"""Run the TRIBEv2.pdf §5.9 Figure 4E contrast-map replication.

Per §5.9 the paper computes predicted contrast maps by selecting each
stimulus's response at t=5 s and subtracting the mean t=5 response across
other categories. Figure 4C/D shows these localize to FFA (face), PPA
(place), EBA (body), VWFA (word). This script reproduces the contrast
computation exactly and validates localization at the ICA-mask level
(see cognitive_similarity/paper_replication.py for caveats re: Glasser).

Usage:
    python scripts/replicate_figure_4e.py --cache-dir <path>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cognitive_similarity.ica_atlas import ICANetworkAtlas      # noqa: E402
from cognitive_similarity.models import ICANetwork              # noqa: E402
from cognitive_similarity.paper_replication import (             # noqa: E402
    replicate_figure_4e,
)

# Ground truth from ibc_exemplars.py: which stimulus ids belong to each
# FaceBody category. Keep in sync when exemplars change.
_FACEBODY_CATEGORIES: dict[str, list[str]] = {
    "face":              ["face_01", "face_02"],
    "place":             ["place_01", "place_02"],
    "body":              ["body_01", "body_02"],
    "written_character": ["written_character_01", "written_character_02"],
}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--localization-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of each category's top-1%% contrast vertices "
            "that must fall inside the VISUAL_SYSTEM ICA mask for the "
            "category to PASS (default 0.5)."
        ),
    )
    args = ap.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    atlas = ICANetworkAtlas(cache_dir=str(cache_dir))

    report = replicate_figure_4e(
        cache_dir=cache_dir,
        atlas=atlas,
        categories_to_stimuli=_FACEBODY_CATEGORIES,
        expected_network=ICANetwork.VISUAL_SYSTEM,
        localization_threshold=args.localization_threshold,
    )

    print(f"\n{'=' * 78}")
    print(
        f"Figure 4E replication — §5.9 contrast maps "
        f"(passing threshold: ≥{args.localization_threshold:.0%} of top-1% in "
        f"{report.expected_network.value})"
    )
    print("=" * 78)
    print(f"{'category':<22} {'peak vtx':>10} {'peak val':>10} "
          + "".join(f" {n.value[:10]:>11}" for n in atlas.NETWORKS))
    print("-" * 78)
    for category, result in report.results.items():
        fractions = result.top_pct_fraction_per_network
        row = "".join(
            f" {fractions.get(n, 0.0):>11.3f}" for n in atlas.NETWORKS
        )
        passed = fractions.get(
            report.expected_network, 0.0
        ) >= args.localization_threshold
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {category:<16} {result.peak_vertex:>10d} "
              f"{result.peak_value:>+10.3f}{row}")

    print()
    print(f"{report.passed}/{report.total} categories localize into "
          f"{report.expected_network.value} above threshold.")
    return 0 if report.passed == report.total else 2


if __name__ == "__main__":
    sys.exit(main())
