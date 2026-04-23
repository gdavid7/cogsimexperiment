"""Run ValidationSuite locally against a synced IBC cache.

Usage:
    python scripts/validate_ibc.py --cache-dir /path/to/cognitive-similarity-cache

Assumes the cache directory contains:
    - manifest.json      (produced by scripts/run_inference_modal.py)
    - tensors/<hash>/raw_cortical.npy for each manifest entry
      (written by the Modal worker to the cogsim-cache Volume, then mirrored
      locally via `modal run --download-to` or `modal volume get`)

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
from cognitive_similarity.ica_atlas import ICANetworkAtlas       # noqa: E402
from cognitive_similarity.models import ICANetwork               # noqa: E402
from cognitive_similarity.validation import ValidationSuite      # noqa: E402

# Total cortical vertex count on fsaverage5 (2 × 10,242 hemispheres).
_N_VERTICES = 20484


def _stimulus_from_manifest(entry: dict) -> Stimulus:
    """Build a minimal Stimulus for local collapsing.

    Only stimulus_id is populated; modality path fields stay None on
    purpose. TemporalCollapser.collapse (collapsing.py) reads only
    stimulus.duration_s — which is inferred from T when None — so no
    file path is needed to collapse the cached raw_cortical.npy.

    Previous revisions filled entry["local_path"], which is the Modal
    container path (e.g. /cache/stimulus_videos/foo.mp4) and does not
    exist on the local Mac; any code that hashed that path would have
    died with FileNotFoundError. G3(a) removes that dormant hazard.
    """
    return Stimulus(stimulus_id=entry["stimulus_id"])


def _materialize_collapsed(cache_dir: Path, manifest: list[dict]) -> None:
    """Compute collapsed.npy from raw_cortical.npy for any entry that lacks one.

    Duration is left unset so TemporalCollapser infers it from T * tr_s;
    this keeps the script correct for any stimulus length without
    hardcoding assumptions from the Colab/Modal-side preprocessing.
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
        stim = _stimulus_from_manifest(entry)
        collapsed = collapser.collapse(raw, stim, tr_s=1.0)
        np.save(collapsed_path, collapsed.astype(np.float32))
        print(f"  {entry['stimulus_id']:<25}  raw {raw.shape} -> collapsed {collapsed.shape}")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over two 1-D vectors (mean-centered, length-normalized)."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if norm == 0.0:
        return 0.0
    return float(np.dot(a_c, b_c) / norm)


def _significance_test(
    atlas: ICANetworkAtlas,
    network: ICANetwork,
    resp_a1: np.ndarray,
    resp_a2: np.ndarray,
    resp_b1: np.ndarray,
    resp_b2: np.ndarray,
    *,
    n_perm: int = 1000,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Test whether Δ = sim(a1,a2) - sim(b1,b2) is statistically significant.

    Two complementary nulls, both computed on the already-materialized
    collapsed cortical responses (no new inference):

    (A) Random-mask permutation. Null: "our ICA mask isn't selecting a
        functionally-specific region — any random same-size vertex subset
        would give a comparable Δ." Implementation: sample `n_perm` random
        vertex subsets of the same size as the network mask, recompute Δ.
        One-sided p = fraction of null Δs ≥ observed Δ. p < 0.05 means the
        mask is beating chance.

    (B) Vertex-bootstrap within the mask. Null: "Δ is a stable estimate
        given these stimuli and mask." Resample mask vertices with
        replacement `n_boot` times, compute the Δ distribution, report
        the 95% CI. CI crossing zero means the effect isn't reliable
        even within the mask we chose.

    Caveat — spatial autocorrelation. (B) resamples vertex indices i.i.d.,
    which treats each of the ~2048 mask vertices as an independent sample.
    Cortical vertices are not independent: neighbors on the fsaverage5
    surface (≈2 mm apart) carry near-identical fMRI signal given the ~6-8 mm
    spatial smoothness of BOLD, so the effective number of independent
    samples is much smaller than the vertex count. Consequence: reported
    CIs are narrower than they should be. TRIBEv2.pdf §5.6 addresses the
    time-dimension analog ("we only keep one TR every 60 seconds"); the
    spatial equivalent is a spin test or Moran spectral randomization
    (see the `neuromaps` library). Current |Δ|s are large enough that this
    doesn't flip any verdict, but do not read these CIs as formal 95%
    intervals. Upgrade to spin tests if these numbers need to be published.
    """
    mask = atlas.get_mask(network)
    mask_idx = np.where(mask)[0]
    mask_size = int(mask.sum())

    sim_a_obs = _pearson(resp_a1[mask], resp_a2[mask])
    sim_b_obs = _pearson(resp_b1[mask], resp_b2[mask])
    delta_obs = sim_a_obs - sim_b_obs

    rng = np.random.default_rng(seed)

    null_deltas = np.empty(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(_N_VERTICES, size=mask_size, replace=False)
        sim_a_rand = _pearson(resp_a1[rand_idx], resp_a2[rand_idx])
        sim_b_rand = _pearson(resp_b1[rand_idx], resp_b2[rand_idx])
        null_deltas[i] = sim_a_rand - sim_b_rand
    perm_p = float((null_deltas >= delta_obs).mean())

    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        resampled = rng.choice(mask_idx, size=mask_size, replace=True)
        sim_a_boot = _pearson(resp_a1[resampled], resp_a2[resampled])
        sim_b_boot = _pearson(resp_b1[resampled], resp_b2[resampled])
        boot_deltas[i] = sim_a_boot - sim_b_boot
    ci_lo, ci_hi = np.quantile(boot_deltas, [0.025, 0.975])

    return {
        "delta": delta_obs,
        "sim_a": sim_a_obs,
        "sim_b": sim_b_obs,
        "perm_p": perm_p,
        "bootstrap_ci": (float(ci_lo), float(ci_hi)),
    }


def _classify(stat: dict) -> str:
    """Combine the permutation p-value and the bootstrap CI into a verdict.

    Four outcomes:
      - FAILED: Δ ≤ 0 or bootstrap CI crosses/excludes positive side.
      - ordering+mask-specific: Δ > 0, CI positive, and the ICA mask beats
        a random same-size vertex subset (perm p < 0.05). The gold standard.
      - ordering only: Δ > 0, CI positive, but the mask is no better than
        random for this contrast — the ordering reflects global TRIBE
        structure rather than network-specific selectivity.
      - unstable: CI spans zero — the point estimate isn't reliable.
    """
    ci_lo, ci_hi = stat["bootstrap_ci"]
    delta = stat["delta"]
    perm_p = stat["perm_p"]
    if delta <= 0 or ci_hi <= 0:
        return "FAILED (wrong direction)"
    if ci_lo <= 0:
        return "unstable (CI spans 0)"
    if perm_p < 0.05:
        return "ordering + mask-specific"
    return "ordering only (mask ≈ random)"


def _run_significance(
    atlas: ICANetworkAtlas,
    cache: ResponseCache,
    manifest: list[dict],
    report,
) -> None:
    """Post-process ValidationSuite results with significance tests per check."""
    id_to_hash = {e["stimulus_id"]: e["content_hash"] for e in manifest}

    def _load(stimulus_id: str) -> np.ndarray | None:
        h = id_to_hash.get(stimulus_id)
        if h is None:
            return None
        return cache.get_collapsed_by_hash(h)

    print(f"\n==================== SIGNIFICANCE ====================")
    print("Null (A): random-mask permutation (n=1000) — does the ICA mask beat a")
    print("          random same-size vertex subset? perm_p < 0.05 → yes.")
    print("Null (B): vertex bootstrap (n=1000) within the mask — 95% CI on Δ.")
    print("          CI excluding zero → Δ is a stable non-zero estimate.\n")

    n_mask_specific = 0
    n_ordering_ok = 0
    for c in report.checks:
        a1, a2 = c.pair_a
        b1, b2 = c.pair_b
        tensors = [_load(sid) for sid in (a1, a2, b1, b2)]
        if any(t is None for t in tensors):
            print(f"  [SKIP] {c.description}  (missing collapsed tensors)")
            continue
        resp_a1, resp_a2, resp_b1, resp_b2 = tensors
        stat = _significance_test(atlas, c.network, resp_a1, resp_a2, resp_b1, resp_b2)
        verdict = _classify(stat)
        if verdict == "ordering + mask-specific":
            n_mask_specific += 1
            n_ordering_ok += 1
        elif verdict == "ordering only (mask ≈ random)":
            n_ordering_ok += 1
        ci_lo, ci_hi = stat["bootstrap_ci"]
        perm_p = stat["perm_p"]
        p_str = "<0.001" if perm_p < 0.001 else (">0.999" if perm_p > 0.999 else f"{perm_p:.3f}")
        print(
            f"  [{verdict:<32}] Δ={stat['delta']:+.3f}   "
            f"perm p={p_str:>6}   "
            f"boot CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        )
        print(f"    {c.description}")

    total_real = sum(1 for c in report.checks if "MISSING" not in c.description)
    print(f"\n  → {n_ordering_ok}/{total_real} checks have positive Δ with stable bootstrap CI.")
    print(f"  → {n_mask_specific}/{total_real} of those also show mask-specific selectivity "
          f"(ICA mask beats random).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to the synced cognitive-similarity-cache directory "
             "(contains manifest.json and tensors/).",
    )
    ap.add_argument(
        "--no-significance",
        action="store_true",
        help="Skip the permutation + bootstrap significance tests.",
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
    print(f"{report.passed}/{report.total} checks passed (ordering test only)\n")
    for c in report.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.network.value:<28}  sim_A={c.score_a:+.4f}  sim_B={c.score_b:+.4f}")
        print(f"         {c.description}")

    if not args.no_significance:
        _run_significance(cs._atlas, cache, manifest, report)

    return 0 if report.passed == report.total else 2


if __name__ == "__main__":
    sys.exit(main())
