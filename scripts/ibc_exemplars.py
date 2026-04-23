"""Single source of truth for which IBC stimuli enter the validation pipeline.

Used by both scripts/run_inference_modal.py (to drive inference) and
scripts/validate_ibc.py (to materialize collapsed tensors and validate).

Each exemplar has exactly the `stimulus_id` that ValidationSuite
(cognitive_similarity/validation.py) expects — the 14 entries below cover
the 6 validation checks that survive the Slice 2/3 paper-alignment pass:

  Visual System (4 checks):   FFA, PPA, EBA, VWFA — 8 exemplars
  Primary Auditory (1 check): speech>non_speech   — 3 exemplars
  Language Network (1 check): sentence>word_list  — 3 exemplars

Sources:
- Visual: IBC public_protocols/FaceBody/stimuli/ (TRIBEv2.pdf §5.9).
- Auditory: IBC public_protocols/realistic_sounds/stim/all/ — 1-s
  16 kHz mono PCM WAVs from Santoro 2017's auditory category localizer.
  Per Slice 3 (D1) non_speech_01 uses s2_animal_1.wav to mirror TRIBEv2.pdf
  Figure 5B's "[woof woof]" example ("natural sounds" contrast).
- Language: English translations of IBC RSVPLanguage French sentences
  (first ObjCleft constructions from complex_sentence.csv; a simple-SVO
  paraphrase for sentence_0{1,2}; a scrambled-words list for word_list_01).
  Translation workflow per TRIBEv2.pdf §5.9.

Deliberately dropped in Slice 3 (see git log for rationale):

- silence_01, audio_segment_0{1,2} — validation check audio>silence
  removed (Group B3a): the Santoro clean-sound vs synthesized-silence
  pairwise test diverges from TRIBEv2.pdf §5.9's Algonauts movie-audio
  contrast, and TRIBE's response at t=5 to a silent input reflects
  baseline rather than anything informative about auditory cortex.
- complex_sentence_0{1,2}, simple_sentence_01 — complex>simple removed
  (Group B3a): the paper's complex-vs-simple finding (§2.6) is a
  *magnitude* contrast in Broca; translating that to a pairwise
  similarity ordering requires assumptions the paper doesn't make.
- motion_video_0{1,2}, static_image_01 — motion>static removed
  (Group C2): the paper has no 1-s in-silico protocol for motion, and
  the product's motion cognitive axis is validated indirectly by the
  NeuroSynth-based ICA labeling in Slice 2 (motion_mt_plus component
  matches NeuroSynth "motion" map at |r|=0.47).

Stimulus protocol (per TRIBEv2.pdf §5.9): 1 second of stimulus followed
by 7 seconds of blank (black frame / silence / empty text), totaling 8 s
so TRIBE's 1-Hz output has T≈8 timepoints; the collapser then indexes
cortical_response[5] — the hemodynamic peak at t+5 s per §5.8 / Fig 4A.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Modality = Literal["video", "audio", "text"]
SrcKind = Literal[
    "facebody_jpg",    # IBC FaceBody JPG → 1 s static MP4 + 7 s black frame = 8 s
    "wav_padded",      # IBC realistic_sounds WAV → 1 s signal + 7 s silence = 8 s
    "text_direct",     # English text string → written to .txt (TRIBE TTS handles padding)
]


@dataclass
class Exemplar:
    stimulus_id: str
    modality: Modality
    src_kind: SrcKind
    category: str
    # One of the following is populated, based on src_kind:
    source_path: Optional[str] = None   # relative to IBC public_protocols root
    text: Optional[str] = None          # for modality == "text"


# Visual System checks (FFA, PPA, EBA, VWFA) — 8 exemplars. Source paths
# verified against pinned IBC public_protocols SHA cbbb7715 (Slice 1 G2).
_VISUAL: list[Exemplar] = [
    Exemplar("face_01",               "video", "facebody_jpg", "face",
             source_path="FaceBody/stimuli/adult/adult-1.jpg"),
    Exemplar("face_02",               "video", "facebody_jpg", "face",
             source_path="FaceBody/stimuli/adult/adult-2.jpg"),
    Exemplar("place_01",              "video", "facebody_jpg", "place",
             source_path="FaceBody/stimuli/house/house-1.jpg"),
    Exemplar("place_02",              "video", "facebody_jpg", "place",
             source_path="FaceBody/stimuli/house/house-2.jpg"),
    Exemplar("body_01",               "video", "facebody_jpg", "body",
             source_path="FaceBody/stimuli/body/body-1.jpg"),
    Exemplar("body_02",               "video", "facebody_jpg", "body",
             source_path="FaceBody/stimuli/body/body-2.jpg"),
    Exemplar("written_character_01",  "video", "facebody_jpg", "written_character",
             source_path="FaceBody/stimuli/word/word-1.jpg"),
    Exemplar("written_character_02",  "video", "facebody_jpg", "written_character",
             source_path="FaceBody/stimuli/word/word-2.jpg"),
]


# Primary Auditory Cortex check — speech>non_speech (1 check, 3 exemplars).
# Santoro 2017 realistic_sounds are 1-s isolated category exemplars; we pad
# to 8 s with silence (C1). non_speech_01 uses s2_animal_1 to match §5.9
# Fig 5B's "[woof woof]" example. Ecological caveat: the paper actually
# contrasts Algonauts movie segments with/without speech; isolated Santoro
# clips are a reasonable proxy but less rich.
_AUDITORY: list[Exemplar] = [
    Exemplar("speech_01",        "audio", "wav_padded", "speech",
             source_path="realistic_sounds/stim/all/s2_speech_1.wav"),
    Exemplar("speech_02",        "audio", "wav_padded", "speech",
             source_path="realistic_sounds/stim/all/s2_speech_2.wav"),
    Exemplar("non_speech_01",    "audio", "wav_padded", "non_speech",
             source_path="realistic_sounds/stim/all/s2_animal_1.wav"),
]


# Language Network check — sentence>word_list (1 check, 3 exemplars).
# sentence_0{1,2} are simple active-voice English renderings of IBC
# RSVPLanguage content (verified at pinned commit). word_list_01 is
# deliberately scrambled words, matching the RSVPLanguage word-list
# condition that the paper's §2.6 contrasts against grammatical sentences.
_LANGUAGE: list[Exemplar] = [
    Exemplar("sentence_01",          "text", "text_direct", "sentence",
             text="The sailors get annoyed when they hear the seagulls making noise."),
    Exemplar("sentence_02",          "text", "text_direct", "sentence",
             text="The mayor called a young unemployed man by contacting the job center."),
    Exemplar("word_list_01",         "text", "text_direct", "word_list",
             text="Annoy dictated the tried harass carmine halal the in them."),
]


EXEMPLARS: list[Exemplar] = _VISUAL + _AUDITORY + _LANGUAGE


# Representative stimulus per modality; used by the smoke test before the
# full batch so pipeline failures surface fast and cheap.
SMOKE_TEST_IDS: list[str] = [
    "face_01",           # video (FaceBody still → 1 s + 7 s blank)
    "speech_01",         # audio (1 s WAV + 7 s silence)
    "sentence_01",       # text (English sentence → TTS inside TRIBE)
]


def by_id(stimulus_id: str) -> Exemplar:
    for ex in EXEMPLARS:
        if ex.stimulus_id == stimulus_id:
            return ex
    raise KeyError(f"Unknown stimulus_id: {stimulus_id!r}")
