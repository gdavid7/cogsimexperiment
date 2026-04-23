"""Single source of truth for which IBC stimuli enter the validation pipeline.

Used by both scripts/run_inference_modal.py (to drive inference) and
scripts/validate_ibc.py (to materialize collapsed tensors and validate).

Each exemplar has exactly the `stimulus_id` that ValidationSuite
(cognitive_similarity/validation.py) expects — the 23 entries below cover all
9 validation checks across Visual System, Primary Auditory Cortex, Language
Network, and Motion Detection MT+.

Sources:
- Visual exemplars: IBC public_protocols/FaceBody/stimuli/ (TRIBEv2.pdf §5.9).
- Auditory exemplars: IBC public_protocols/realistic_sounds/stim/all/ — 1-second
  16 kHz mono PCM WAVs, padded to 10 s with silence before inference.
- Language exemplars: English translations of sentences from IBC
  public_protocols/RSVPLanguage/rsvp_language_protocol/generate_inputs/stim_main_session/
  conditions_main_session/{simple,complex}_sentence.csv and word_list.csv.
  Translation from French is the workaround described in TRIBEv2.pdf §5.9.
- MT+ exemplars: IBC public_protocols/archi/protocols/archi_emotional/
  {BioMvt,NonBioMvt}/BioGifMovies/ — biological-motion clips (BioMvt) paired
  with their scrambled-motion controls (NonBioMvt). Using `_s10` variants
  (5.9 s) so the looped 10-s inputs produce T≈10 timepoints (enough for the
  collapser to index the t+5 s hemodynamic peak per TRIBEv2.pdf §5.8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Modality = Literal["video", "audio", "text"]
SrcKind = Literal[
    "facebody_jpg",    # IBC FaceBody JPG → 10 s static MP4 via ffmpeg
    "wav_padded",      # IBC realistic_sounds WAV → 10 s WAV padded with silence
    "synth_silence",   # synthesized 10 s silent WAV
    "biomvt_mp4",      # archi_emotional BioMvt/NonBioMvt MP4 → 10 s looped MP4
    "text_direct",     # English text string → written to .txt file
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


# Visual System checks (FFA, PPA, EBA, VWFA) — 8 exemplars, unchanged from
# the working 4-check validation (see CLAUDE.md Validation Status table).
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


# Primary Auditory Cortex checks — 6 exemplars from realistic_sounds (Santoro
# 2017 category localizer). s2_tools was chosen for non_speech because it's
# the cleanest "no voice / no music / no speech" category; s2_music and
# s2_nature for audio_segment give two distinct non-speech sounds.
_AUDITORY: list[Exemplar] = [
    Exemplar("speech_01",        "audio", "wav_padded", "speech",
             source_path="realistic_sounds/stim/all/s2_speech_1.wav"),
    Exemplar("speech_02",        "audio", "wav_padded", "speech",
             source_path="realistic_sounds/stim/all/s2_speech_2.wav"),
    Exemplar("non_speech_01",    "audio", "wav_padded", "non_speech",
             source_path="realistic_sounds/stim/all/s2_tools_1.wav"),
    Exemplar("audio_segment_01", "audio", "wav_padded", "audio_segment",
             source_path="realistic_sounds/stim/all/s2_music_1.wav"),
    Exemplar("audio_segment_02", "audio", "wav_padded", "audio_segment",
             source_path="realistic_sounds/stim/all/s2_nature_1.wav"),
    Exemplar("silence_01",       "audio", "synth_silence", "silence"),
]


# Language Network checks — 6 English translations. Object-cleft construction
# preserved for "complex_sentence" to match the syntactic-load contrast from
# the RSVPLanguage task (simple SVO vs. object-cleft). word_list is
# deliberately word-by-word and ungrammatical, matching the source's shuffled
# word-list condition.
_LANGUAGE: list[Exemplar] = [
    Exemplar("sentence_01",          "text", "text_direct", "sentence",
             text="The sailors get annoyed when they hear the seagulls making noise."),
    Exemplar("sentence_02",          "text", "text_direct", "sentence",
             text="The mayor called a young unemployed man by contacting the job center."),
    Exemplar("word_list_01",         "text", "text_direct", "word_list",
             text="Annoy dictated the tried harass carmine halal the in them."),
    Exemplar("complex_sentence_01",  "text", "text_direct", "complex_sentence",
             text="It is the sailors that the seagulls annoy by crying."),
    Exemplar("complex_sentence_02",  "text", "text_direct", "complex_sentence",
             text="It is an unemployed man that a mayor called by contacting the job center."),
    Exemplar("simple_sentence_01",   "text", "text_direct", "simple_sentence",
             text="The photograph evokes a bad memory when it appears unexpectedly."),
]


# MT+ check — biological motion vs a true static image. baseballswing and
# dancing2 are two distinct human actions; static_image_01 is the FaceBody
# `scrambled` category (a fixed visual-feature-dense image, no motion) rendered
# as a 10 s static MP4 via the same facebody_jpg pipeline as face/place/body.
#
# Earlier we used NonBioMvt/baseballswing_s10.mp4 as the "static" control, but
# NonBioMvt is not actually static — it's scramble-of-biological-motion, which
# still contains motion and therefore still activates MT+. That made the
# motion>static check fail (0.97 < 0.99) because both sides had motion.
# A genuinely still image is the right control.
_MT_PLUS: list[Exemplar] = [
    Exemplar("motion_video_01", "video", "biomvt_mp4", "motion_video",
             source_path="archi/protocols/archi_emotional/BioMvt/BioGifMovies/baseballswing_s10.mp4"),
    Exemplar("motion_video_02", "video", "biomvt_mp4", "motion_video",
             source_path="archi/protocols/archi_emotional/BioMvt/BioGifMovies/dancing2_s10.mp4"),
    Exemplar("static_image_01", "video", "facebody_jpg", "static_image",
             source_path="FaceBody/stimuli/scrambled/scrambled-1.jpg"),
]


EXEMPLARS: list[Exemplar] = _VISUAL + _AUDITORY + _LANGUAGE + _MT_PLUS


# Representative stimulus per modality; used by the smoke test before the
# full batch so pipeline failures surface fast and cheap.
SMOKE_TEST_IDS: list[str] = [
    "face_01",           # video (FaceBody still → static MP4)
    "speech_01",         # audio (WAV padded to 10 s)
    "sentence_01",       # text (English sentence → TTS inside TRIBE)
    "motion_video_01",   # video (BioMvt clip looped to 10 s)
]


def by_id(stimulus_id: str) -> Exemplar:
    for ex in EXEMPLARS:
        if ex.stimulus_id == stimulus_id:
            return ex
    raise KeyError(f"Unknown stimulus_id: {stimulus_id!r}")
