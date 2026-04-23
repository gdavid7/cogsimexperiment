"""Run TRIBE v2 inference on IBC stimuli via Modal — autonomous from any Mac.

Usage (from the repo root, after `modal token new` and creating `hf-token` secret):

    modal run scripts/run_inference_modal.py                  # smoke test + full batch
    modal run scripts/run_inference_modal.py --smoke-only     # just the 4-modality smoke test
    modal run scripts/run_inference_modal.py --ids face_01,speech_01  # subset
    modal run scripts/run_inference_modal.py --download-to /path/to/local/cache

Design (see CLAUDE.md Workflow section):
- Modal worker clones IBC public_protocols inside the container, preprocesses
  each exemplar to the appropriate 10-s stimulus file, runs TRIBE v2 cortical
  inference, writes raw_cortical.npy under /cache/tensors/<content_hash>/ on a
  persistent Modal Volume.
- Local entrypoint builds the manifest, farms each exemplar to the worker,
  writes manifest.json to the Volume, optionally downloads everything to a
  local cache directory for downstream validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import modal

# Make scripts/ a package-less module-friendly cwd so ibc_exemplars imports work
# both locally (via the entrypoint) and inside the container (via add_local_dir).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ibc_exemplars import EXEMPLARS, SMOKE_TEST_IDS, Exemplar, by_id  # noqa: E402


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
#
# tribev2 pins its own deps (notably numpy==2.2.6 and a torch range). We install
# it first so pip resolves around those pins, then layer our other deps on top.
# This matches the working recipe that validated against Colab in earlier
# runs (see git history for the old remote_inference.ipynb equivalent).
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "git+https://github.com/facebookresearch/tribev2.git",
    )
    .pip_install(
        "nilearn>=0.10",
        "scikit-learn>=1.4",
        "pandas>=2.0",
        "transformers",
        "huggingface-hub>=0.20",
    )
)

# cognitive_similarity/ package needs to be available inside the container so
# the worker can use StimulusRunner / ResponseCache exactly like the Colab
# notebook did.
IMAGE = IMAGE.add_local_dir(
    str(_REPO_ROOT / "cognitive_similarity"),
    remote_path="/root/cognitive_similarity",
).add_local_dir(
    str(_REPO_ROOT / "scripts"),
    remote_path="/root/scripts",
)

VOLUME = modal.Volume.from_name("cogsim-cache", create_if_missing=True)
CACHE_PATH = "/cache"
IBC_PATH = "/ibc_public_protocols"
# Pinned IBC public_protocols commit (master as of 2026-02-24) so upstream path
# reorgs can't silently invalidate the source_path entries in ibc_exemplars.py.
# Bump this deliberately when ibc_exemplars.py is updated to reference newer
# content. Per CLAUDE.md Evidence-Based Development: the paths listed in
# ibc_exemplars.py were verified against this SHA via raw.githubusercontent.com.
IBC_PROTOCOLS_SHA = "cbbb7715ea677feb8ed29ed1bc4eaeb1a7f56cc6"

app = modal.App("cogsim-inference", image=IMAGE)

MINUTES = 60


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A100",
    volumes={CACHE_PATH: VOLUME},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=20 * MINUTES,
    scaledown_window=5 * MINUTES,
)
class TribeWorker:
    """Container-side worker that loads TRIBE v2 once and serves inferences.

    The @modal.enter lifecycle hook runs once per container start: it clones
    public_protocols (cached across calls via the Volume would be possible, but
    a fresh clone is simpler and only takes a few seconds), logs into
    HuggingFace so the gated Llama 3.2 (used internally by TRIBE's text
    pipeline) is accessible, and loads the cortical model to GPU.
    """

    @modal.enter()
    def setup(self) -> None:
        import os
        import subprocess
        import sys
        sys.path.insert(0, "/root")

        # HF login (secret provides HF_TOKEN in the environment)
        from huggingface_hub import login
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)

        # Clone IBC public_protocols at the pinned commit. --depth=1 can't
        # co-exist with checkout <sha>, so we do a full clone + checkout; the
        # repo is small enough that the extra history cost is negligible
        # compared to the ~30 s TRIBE-load step that dominates cold-start.
        if not os.path.exists(IBC_PATH):
            print(f"Cloning IBC public_protocols @ {IBC_PROTOCOLS_SHA[:8]} -> {IBC_PATH}...")
            subprocess.run(
                ["git", "clone",
                 "https://github.com/individual-brain-charting/public_protocols.git",
                 IBC_PATH],
                check=True,
            )
            subprocess.run(
                ["git", "-C", IBC_PATH, "checkout", IBC_PROTOCOLS_SHA],
                check=True,
            )

        # Preprocessed-stimulus workspaces on the Volume.
        for sub in ("stimulus_videos", "stimulus_audios", "stimulus_texts", "tensors"):
            os.makedirs(f"{CACHE_PATH}/{sub}", exist_ok=True)

        # Load TRIBE v2 cortical model.
        from tribev2 import TribeModel
        print("Loading TRIBE v2 cortical model...")
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=f"{CACHE_PATH}/hf",
            device="cuda",
        )
        print("✓ Model ready")

    @modal.method()
    def infer_one(self, exemplar_dict: dict) -> dict:
        """Preprocess one exemplar and run TRIBE v2 inference.

        Returns a manifest entry dict with the content_hash Tell the local
        driver where to find the raw_cortical.npy on the Volume.
        Idempotent: if raw_cortical.npy already exists for this content hash,
        returns the cached entry without re-running inference.
        """
        import sys
        sys.path.insert(0, "/root")

        import numpy as np

        from cognitive_similarity.cache import ResponseCache
        from cognitive_similarity.models import Stimulus
        from cognitive_similarity.stimulus_runner import StimulusRunner

        # Reconstitute the dataclass from its dict form (Modal serializes args).
        ex = Exemplar(**exemplar_dict)

        stim_path = _preprocess_stimulus(ex, CACHE_PATH, IBC_PATH)

        # Build the Stimulus with the right modality-specific field populated.
        kwargs: dict[str, str] = {"stimulus_id": ex.stimulus_id}
        if ex.modality == "video":
            kwargs["video_path"] = str(stim_path)
        elif ex.modality == "audio":
            kwargs["audio_path"] = str(stim_path)
        else:
            kwargs["text_path"] = str(stim_path)
        stim = Stimulus(**kwargs)

        cache = ResponseCache(CACHE_PATH)
        content_hash = cache._content_hash(stim)
        tensor_dir = Path(CACHE_PATH) / "tensors" / content_hash
        cortical_path = tensor_dir / "raw_cortical.npy"

        manifest_entry = {
            "stimulus_id": ex.stimulus_id,
            "category": ex.category,
            "modality": ex.modality,
            "local_path": str(stim_path),
            "content_hash": content_hash,
            "tensor_dir": f"tensors/{content_hash}",
            "ibc_source": ex.source_path if ex.source_path else "(synthesized)",
        }

        if cortical_path.exists():
            arr = np.load(cortical_path)
            print(f"  ✓ {ex.stimulus_id:<22} cached  cortical={arr.shape}")
            return manifest_entry

        # Fresh inference.
        runner = StimulusRunner(self.model)
        brain_response = runner.run(stim)
        assert brain_response.cortical.shape[1] == 20484, (
            f"Unexpected cortical width for {ex.stimulus_id}: "
            f"{brain_response.cortical.shape}"
        )

        tensor_dir.mkdir(parents=True, exist_ok=True)
        np.save(cortical_path, brain_response.cortical)
        VOLUME.commit()

        print(f"  ✓ {ex.stimulus_id:<22} inferred cortical={brain_response.cortical.shape}")
        return manifest_entry

    @modal.method()
    def write_manifest(self, manifest: list[dict]) -> None:
        """Persist manifest.json to the Volume so local validate_ibc.py can use it."""
        import json as _json
        path = Path(CACHE_PATH) / "manifest.json"
        path.write_text(_json.dumps(manifest, indent=2))
        VOLUME.commit()
        print(f"✓ Wrote {path} ({len(manifest)} entries)")


# ---------------------------------------------------------------------------
# Preprocessing — happens inside the container (runs inside infer_one)
# ---------------------------------------------------------------------------

def _preprocess_stimulus(ex: Exemplar, cache_root: str, ibc_root: str) -> Path:
    """Produce the file TRIBE v2 will read, under cache_root/stimulus_{videos,audios,texts}/.

    Cached across calls — if the output file already exists, short-circuits.
    ffmpeg parameters (fps=10, libx264, yuv420p, no audio) mirror
    tribev2.eventstransforms.CreateVideosFromImages defaults so the encoded
    bytes closely match what TRIBE v2's own helper would produce.
    """
    cache = Path(cache_root)
    if ex.src_kind == "facebody_jpg":
        out = cache / "stimulus_videos" / f"{ex.stimulus_id}.mp4"
        if not out.exists():
            src = Path(ibc_root) / ex.source_path
            _run_ffmpeg(
                ["ffmpeg", "-loop", "1", "-i", str(src),
                 "-t", "10", "-r", "10",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                 "-y", "-loglevel", "error", str(out)],
                out,
            )
        return out

    if ex.src_kind == "biomvt_mp4":
        out = cache / "stimulus_videos" / f"{ex.stimulus_id}.mp4"
        if not out.exists():
            src = Path(ibc_root) / ex.source_path
            # Loop the input (5.9 s) to fill 10 s exactly. libx264 + yuv420p
            # need even width/height (chroma subsampling), but the BioMvt
            # clips are 83×171 — pad to the next even size with black.
            _run_ffmpeg(
                ["ffmpeg", "-stream_loop", "-1", "-i", str(src),
                 "-t", "10", "-r", "10",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                 "-y", "-loglevel", "error", str(out)],
                out,
            )
        return out

    if ex.src_kind == "wav_padded":
        out = cache / "stimulus_audios" / f"{ex.stimulus_id}.wav"
        if not out.exists():
            src = Path(ibc_root) / ex.source_path
            # Loop the 1 s IBC realistic_sounds WAV to fill 10 s. Earlier we
            # used apad=whole_dur=10 (silence-pad); that gave TRIBE 9 s of
            # silence dominating the response, which made audio stimuli
            # correlate MORE with pure silence than with each other
            # (validation check audio_segment > silence failed with 0.75 < 0.91).
            # Looping gives continuous auditory content, matching the trick
            # we already use for BioMvt 5.9 s clips.
            _run_ffmpeg(
                ["ffmpeg", "-stream_loop", "-1", "-i", str(src),
                 "-t", "10",
                 "-ar", "16000", "-ac", "1",
                 "-y", "-loglevel", "error", str(out)],
                out,
            )
        return out

    if ex.src_kind == "synth_silence":
        out = cache / "stimulus_audios" / f"{ex.stimulus_id}.wav"
        if not out.exists():
            _run_ffmpeg(
                ["ffmpeg", "-f", "lavfi",
                 "-i", "anullsrc=r=16000:cl=mono",
                 "-t", "10",
                 "-y", "-loglevel", "error", str(out)],
                out,
            )
        return out

    if ex.src_kind == "text_direct":
        out = cache / "stimulus_texts" / f"{ex.stimulus_id}.txt"
        if not out.exists():
            assert ex.text is not None
            out.write_text(ex.text)
        return out

    raise ValueError(f"Unknown src_kind: {ex.src_kind!r}")


def _run_ffmpeg(cmd: list[str], out: Path) -> None:
    """Run an ffmpeg command; on failure delete the (possibly partial) output.

    Without this cleanup, a crashed ffmpeg run leaves a corrupt file that the
    `if not out.exists()` short-circuit will happily reuse on the next call —
    causing downstream readers (e.g. moviepy inside TRIBE) to hit "moov atom
    not found" on what looks like a cached stimulus.
    """
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        if out.exists():
            out.unlink()
        raise


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _exemplars_for(ids_filter: Optional[str]) -> list[Exemplar]:
    if not ids_filter:
        return list(EXEMPLARS)
    wanted = {s.strip() for s in ids_filter.split(",") if s.strip()}
    return [by_id(s) for s in wanted]


def _as_dict(ex: Exemplar) -> dict:
    """Plain-dict form for Modal serialization (Modal pickles args; dataclasses work
    but dicts are more forward-compatible if the dataclass grows fields)."""
    return {
        "stimulus_id": ex.stimulus_id,
        "modality": ex.modality,
        "src_kind": ex.src_kind,
        "category": ex.category,
        "source_path": ex.source_path,
        "text": ex.text,
    }


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    ids: str = "",
    download_to: str = "",
) -> None:
    """Drive the full Colab-equivalent pipeline from the local Mac.

    1. (Optional) smoke-test one stimulus per modality.
    2. Process all (or --ids-filtered) exemplars.
    3. Write manifest.json to the Volume.
    4. (Optional) mirror the Volume contents to --download-to for local validation.
    """
    worker = TribeWorker()

    if smoke_only:
        print("=== Smoke test only ===")
        to_run = [by_id(sid) for sid in SMOKE_TEST_IDS]
    else:
        to_run = _exemplars_for(ids)
        # Run smoke subset first within the same call so smoke failures abort
        # before we commit to the full batch.
        smoke_subset = [ex for ex in to_run if ex.stimulus_id in SMOKE_TEST_IDS]
        rest = [ex for ex in to_run if ex.stimulus_id not in SMOKE_TEST_IDS]
        to_run = smoke_subset + rest

    print(f"Processing {len(to_run)} exemplar(s) on Modal...")
    manifest: list[dict] = []
    for ex in to_run:
        entry = worker.infer_one.remote(_as_dict(ex))
        manifest.append(entry)

    worker.write_manifest.remote(manifest)
    print(f"\n✓ Done. {len(manifest)} manifest entries written to the cogsim-cache Volume.")

    if download_to:
        _download_volume(download_to)


def _download_volume(local_dir: str) -> None:
    """Pull manifest.json + tensors/ from the Volume to a local directory.

    Invokes `modal volume get` as `python -m modal ...` so we don't depend on
    the `modal` CLI being on PATH (it lives inside whichever venv the user
    installed Modal into, and that venv is by definition the one running
    this local_entrypoint).

    Subtle: `modal volume get <volume> <source> <dest>` writes to
    `<dest>/<source>` — it creates the `source` file/dir INSIDE the given
    destination. So we pass the parent directory, not the full target path.
    Writing to an existing full path makes the CLI error out.
    """
    import shutil
    import subprocess

    dest = Path(local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("manifest.json", "tensors"):
        target = dest / sub
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get",
             "cogsim-cache", sub, str(dest)],
            check=True,
        )
    print(f"✓ Synced Volume to {dest}")
