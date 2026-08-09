"""festvox_core.py -- backend for the FestVox Speech GUI.

Pure Python (no Qt) so every tricky part is unit-testable headless. This is
the glue between the GUI and the REAL synthesis engine:

    99_Tools/festvox/synth_diphone.py

which renders concatenative diphone audio from the FestVox-style DBs built by
99_Tools/festvox/utau2festvox.py (dic/diphone_index.json + wav/). A second
backend drives real Festival/UniSyn synthesis through WSL.

Responsibilities:
  * import the bundled synth_diphone.py (with an optional explicit override)
  * read festvox.json (the toolchain config) for voices / defaults
  * GUI config.json load/save (deep-merged over defaults)
  * DiphoneBackend: text -> audio + REAL per-phone segments; phone-list
    re-render (for phoneme overrides typed in the GUI)
  * timing, F0 targets, unit-take, fault and cached-project data
  * cleaned voice-local dictionaries and guarded generated-voice removal
  * time-stretch DSP for boundary drags (librosa if present, else a numpy
    phase vocoder; `hook` is the seam for swapping in rubberband/SoX/...)
  * indefinite X-X sustain stretching, WAV and project IO
"""
from __future__ import annotations
import copy
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import threading
import uuid
import wave
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List, Mapping, Optional

import numpy as np

FESTVOX_TOOL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FESTVOX_TOOL_DIR.parents[1]
NATIVE_FESTIVAL_RUNTIME = (
    FESTVOX_TOOL_DIR / "native_unisyn" / "build" / "festvox-festival"
)
if str(FESTVOX_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(FESTVOX_TOOL_DIR))

from cache_support import (  # noqa: E402
    BoundedMemoryCache,
    deep_freeze,
    estimate_size_bytes,
    file_change_token,
)
from voice_manifest import (  # noqa: E402
    VoiceCompatibility,
    generated_voice_output_calibration,
    normalize_language_code,
    read_voice_compatibility,
)
from voice_paths import (  # noqa: E402
    canonical_windows_path,
    make_voice_registration,
    migrate_voice_registration,
    windows_to_wsl_path,
    wsl_to_windows_path,
)
import pitch_domain as pitch_domain  # noqa: E402
import special_phones as special_phone_domain  # noqa: E402
import asaxi_frontend as asaxi_frontend_domain  # noqa: E402
import asaxi_duration as asaxi_duration_domain  # noqa: E402
import asaxi_phone_fallback as asaxi_phone_fallback_domain  # noqa: E402
import asaxi_prosody as asaxi_prosody_domain  # noqa: E402
import english_syllables as english_syllable_domain  # noqa: E402
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GENERATED_VOICE_ROOT = str(
    (PROJECT_ROOT / "generated_voices").resolve()
)
VOICE_ICON_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
BUILTIN_FESTIVAL_VOICES = {
    "kal_diphone": "/usr/share/festival/voices/english/kal_diphone",
}


def remove_known_voice_icons(root, keep=None):
    """Remove only supported ``speaker.*`` files from a voice folder."""
    root = Path(root)
    keep_key = os.path.normcase(os.path.abspath(str(keep))) if keep else ""
    removed = []
    for suffix in VOICE_ICON_SUFFIXES:
        candidate = root / ("speaker" + suffix)
        if keep_key and os.path.normcase(os.path.abspath(
                str(candidate))) == keep_key:
            continue
        try:
            candidate.unlink()
            removed.append(str(candidate))
        except FileNotFoundError:
            pass
    return removed

# --------------------------------------------------------------------- config
DEFAULT_PHRASE_PAUSES_MS = {
    "minor": 120,
    "major": 300,
    "sentence": 500,
}
JAPANESE_INLINE_BRACKET_PAUSE_MS = 60


DEFAULT_CONFIG = {
    # which synthesis engine drives Generate:
    #   "diphone"      = synth_diphone.py (pure Python, no Festival needed)
    #   "festival_wsl" = real Festival running inside WSL (Multisyn-capable)
    "engine": "diphone",
    # Festival-over-WSL settings (Options > WSL / Festival settings...)
    "festival_wsl": {
        "wsl_exe": "",              # "" = auto (wsl.exe on PATH)
        "distro": "",               # "" = the default WSL distro
        "festival_bin": "festival", # binary inside WSL
        # "" selects the project-local build under native_unisyn/build.
        "native_festival_bin": "",
        # Keep the native Festival interpreter warm between normal renders.
        # The worker is recycled to bound heap growth and Scheme state.
        "persistent_native_runtime": True,
        "native_runtime_max_jobs": 32,
        "generated_voice_root": DEFAULT_GENERATED_VOICE_ROOT,
        "generated_voice_wsl_root": "",
        "voices": {},               # name -> {dir, scm, voice}; dir may be a
                                    # Windows path (auto-translated to /mnt/…)
                                    # or a WSL path (starts with "/")
        "installed_voices": [],     # names found by Scan (voice.list)
        "default_voice": "",
        "extra_scheme": "",         # optional .scm loaded before every synth
        "timeout_s": 180,
    },
    # path to the toolchain's festvox.json; "" = auto-discover (cwd or next
    # to the bundled synth_diphone.py) exactly like synth_diphone's CLI
    "festvox_config": "",
    # Optional developer override. Normal installs use ../synth_diphone.py.
    "synth_diphone_dir": "",
    # Keep at most this many decoded voice databases resident. Each database
    # also owns a bounded WAV LRU in synth_diphone.py.
    "diphone_voice_cache_limit": 2,
    "diphone_wav_cache_files": 64,
    "diphone_wav_cache_mib": 64,
    "diphone_slice_cache_entries": 512,
    "diphone_slice_cache_mib": 32,
    "sustain_cache_entries": 64,
    "sustain_cache_mib": 32,
    "festival_voice_cache_limit": 8,
    "festival_voice_cache_mib": 64,
    "voice_variant_cache_limit": 16,
    # QUndoCommands can retain full sentence/audio snapshots; bound history.
    "undo_limit": 64,
    # UI label -> synth_diphone language code
    "languages": {"Asaxi": "asaxi", "English": "en", "Japanese": "ja"},
    "default_language": "Asaxi",
    "default_text": "asaxi",
    # extra voicebank DB dirs added from the GUI: name -> path
    # (each must contain dic/diphone_index.json)
    "extra_voicebanks": {},
    "synth_speed": 1.0,
    # Real synth_diphone knobs, passed as thread-safe per-render values.
    "advanced": {"crossfade_ms": 15.0, "edge_fade_ms": 8.0, "half_ms": 150.0},
    # Festival-engine prosody controls (PSOLA retargeting): base pitch and
    # how much the contour falls across the utterance (percent of pitch)
    "pitch_hz": 185.0,
    "pitch_fall_pct": 10.0,
    "output_gain_db": 0.0,
    "vocal_tract_length_ratio": 1.0,
    "chipmunk_range": False,
    "japanese_duration_model": "contextual",
    "japanese_vowel_devoicing": "contextual",
    "japanese_devoicing_renderer": "auto",
    "asaxi_duration_model": "moraic_rules",
    "allow_output_clipping": False,
    # Semantic phrase-break durations.  The renderer may internally divide a
    # break into protected guard/gap segments, but users edit the total time.
    "phrase_pauses_ms": dict(DEFAULT_PHRASE_PAUSES_MS),
    "parameter_mode": "timing",
    "voicebank_list_height": 76,
    "show_curve_linguistic_units": False,
    "follow_playhead": True,
    "follow_spoken_sentence": True,
    "speaker_portrait": "",
    "shortcuts": {},
    # local display image by "engine|voice"; the chosen image is also copied
    # into the selected generated voice folder.
    "voice_portraits": {},
    # cleaned dictionaries installed beside generated voice data. Keys are
    # "engine|voice|language"; values are Windows or WSL paths.
    "voice_dictionaries": {},
    # force a perfectly flat F0 (keep phoneme timing) -- Festival engine
    "monotone": False,
    # independently selectable diagnostic degradations.  These are deliberately
    # opt-in: normal synthesis keeps the best-quality behavior.
    "fault_mode": {
        "disable_phone_timing": False,
        "disable_prosody": False,
        "disable_f0_correction": False,
        "single_pause": False,
        "monotone": False,
        "pitch_glitch": False,
        "no_sustain_stretch": False,
        "legacy_joins": False,
        "bit_depth": 0,
    },
    # per-language pronunciation override dicts: {lang_code: path to .yaml}
    "user_dicts": {},
    # Generated by vocab_forge/build_asaxi_synthesis_dictionary.py. It is
    # loaded lazily and file-change-aware by the Asaxi frontend.
    "asaxi_synthesis_dictionary": str(
        FESTVOX_TOOL_DIR / "dictionaries" / "asaxi_lexicon.json"
    ),
}


# phone classes for fallback timing (mirror of build_festival_voice.py)
_CLS_STOPS = {"p", "t", "k", "b", "d", "g", "ch", "ts", "dz", "jh", "q",
              "dx", "cl"}
_CLS_FRICS = {"f", "v", "s", "z", "sh", "zh", "th", "dh", "hh", "h"}
_CLS_VOWELS = {"a", "e", "i", "o", "u", "aw", "ay", "ey", "ow", "oy", "uw",
               "ax", "er", "ih", "ao", "iy", "uh", "eh", "aa", "ah", "ae"}
_CLS_VOICELESS_STOPS = {"p", "t", "k", "q", "py", "ty", "ky", "cl"}
_CLS_VOICED_STOPS = {"b", "d", "g", "by", "dy", "gy", "dx", "dxy"}
_CLS_VOICELESS_AFFRICATES = {"ch", "ts"}
_CLS_VOICED_AFFRICATES = {"jh", "dz"}
_CLS_VOICELESS_FRICS = {"f", "s", "sh", "th", "h", "hh", "fy", "hy"}
_CLS_VOICED_FRICS = {"v", "z", "zh", "dh", "vy", "zi"}
_CLS_NASALS = {"m", "n", "ng", "nn", "mm", "nng", "xn", "my", "ny", "ngy"}
_CLS_LIQUIDS = {"l", "r", "rr", "ly", "ry", "ri"}
_CLS_GLIDES = {"w", "y", "wi"}
_VOICED_SIBILANTS = {"z", "zh", "zi", "dz", "jh"}
_SIBILANT_SUPPORTIVE_CLASSES = {
    "vowel", "nasal", "liquid", "glide", "fricative_voiced",
}
_SIBILANT_HARD_RISK_CLASSES = {
    "stop_voiceless", "stop_voiced",
    "affricate_voiceless", "affricate_voiced",
}


def phone_context_class(phone: str) -> str:
    """Broad articulatory class used for cross-language OTO context matching."""
    import re as _re
    base = _re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    if base in {"*"}:
        return "wildcard"
    if base in {"pau", "sil", "sp"}:
        return "silence"
    for phones, name in (
            (_CLS_VOWELS, "vowel"),
            (_CLS_VOICELESS_STOPS, "stop_voiceless"),
            (_CLS_VOICED_STOPS, "stop_voiced"),
            (_CLS_VOICELESS_AFFRICATES, "affricate_voiceless"),
            (_CLS_VOICED_AFFRICATES, "affricate_voiced"),
            (_CLS_VOICELESS_FRICS, "fricative_voiceless"),
            (_CLS_VOICED_FRICS, "fricative_voiced"),
            (_CLS_NASALS, "nasal"), (_CLS_LIQUIDS, "liquid"),
            (_CLS_GLIDES, "glide")):
        if base in phones:
            return name
    if base.endswith("y"):
        return phone_context_class(base[:-1])
    return "other"


def context_edge_info(phone: str, edge: str) -> dict:
    """Describe one OTO context-token edge without consulting WAV names."""
    import re as _re
    if edge not in {"left", "right"}:
        raise ValueError("edge must be 'left' or 'right'")
    base = _re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    direct_class = phone_context_class(base)
    if base == "*":
        return {"phone": "*", "class": "wildcard",
                "kind": "wildcard_unknown"}
    if direct_class != "other":
        return {"phone": base, "class": direct_class, "kind": "atomic"}
    for vowel in sorted(_CLS_VOWELS, key=lambda item: (-len(item), item)):
        if not base.endswith(vowel) or len(base) <= len(vowel):
            continue
        onset = base[:-len(vowel)]
        onset_class = phone_context_class(onset)
        if onset_class in {"other", "wildcard", "silence", "vowel"}:
            continue
        edge_phone = onset if edge == "left" else vowel
        return {"phone": edge_phone,
                "class": phone_context_class(edge_phone),
                "kind": "compound_cv"}
    return {"phone": base, "class": "other", "kind": "unclassified"}


def choice_recorded_context(choice, side: str) -> str:
    """Return recorded context across English and Japanese metadata schemas."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    row = choice or {}
    return str(row.get(side + "_context") or
               row.get("recorded_" + side + "_context") or "*")


def choice_context_info(choice, side: str) -> dict:
    """Return directional OTO evidence for a choice's left or right context.

    New banks persist this data.  Recomputing it from the literal OTO context
    keeps older generated banks compatible and repairs their former broad
    ``other`` classification for CV aliases.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    token = choice_recorded_context(choice, side)
    info = context_edge_info(token, "right" if side == "left" else "left")
    kind = str((choice or {}).get(side + "_context_kind") or "")
    edge_phone = str((choice or {}).get(side + "_context_edge") or "")
    stored_class = str((choice or {}).get(side + "_class") or "")
    if kind in {"atomic", "compound_cv", "wildcard_unknown", "unclassified"}:
        info["kind"] = kind
        if edge_phone:
            info["phone"] = edge_phone
        if stored_class:
            info["class"] = stored_class
    info["context"] = token
    info["source"] = str(
        (choice or {}).get(side + "_context_source") or "")
    return info


def _sibilant_context_quality(choice) -> str:
    context_class = choice_context_info(choice, "right")["class"]
    if context_class in _SIBILANT_SUPPORTIVE_CLASSES:
        return "verified_supportive"
    if context_class in {"wildcard", "other"}:
        return "unknown"
    if context_class in _SIBILANT_HARD_RISK_CLASSES:
        return "verified_risky_stop"
    return "verified_risky"


def class_seg_durs(phones, speed: float = 1.0, equal: bool = False):
    """[(phone, dur)] with class-based natural-ish durations divided by
    `speed` -- used for direct phoneme input / Japanese, where no text
    front end supplies durations. Edge paus are included (speed-scaled,
    matching Duration_Stretch semantics in text mode)."""
    speed = max(0.125, float(speed) or 1.0)
    out = []
    for p in phones:
        b = str(p).rstrip("_")
        if p == "pau":
            d = 0.15
        elif equal:
            d = 0.10
        elif b in _CLS_VOWELS:
            d = 0.13
        elif b in _CLS_STOPS:
            d = 0.06
        elif b in _CLS_FRICS:
            d = 0.09
        else:
            d = 0.08
        out.append((str(p), d / speed))
    if out and out[0][0] != "pau":
        out.insert(0, ("pau", 0.15 / speed))
    if out and out[-1][0] != "pau":
        out.append(("pau", 0.15 / speed))
    return out


def parse_utau_dict(path: str, limit: int = 400000) -> dict:
    """Parse an OpenUTAU .yaml phonemizer dictionary into {grapheme_lower:
    [phones]}. Reads ONLY the 'entries:' section (grapheme -> phonemes); the
    'symbols'/'replacements'/'timings' fluff is ignored. No YAML dependency, so
    it streams multi-MB dictionaries (English arpasing.yaml ~= 9 MB / 146k
    entries) line by line. Falls back to a plain 'word  ph ph ph' text format
    when no YAML sections are present."""
    import re as _re
    out = {}
    seen_section = False
    in_entries = False
    cur = None
    gz = _re.compile(r'^\s*-\s*grapheme:\s*(.*?)\s*$')
    pz = _re.compile(r'^\s*phonemes:\s*\[(.*?)\]\s*$')
    inline = _re.compile(
        r'^\s*-\s*\{\s*grapheme:\s*(.*?),\s*phonemes:\s*\[(.*?)\]\s*\}')
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if s and not s[0].isspace() and s.rstrip().endswith(":"):
                key = s.rstrip()[:-1].strip()
                if key in ("symbols", "replacements", "timings", "entries"):
                    seen_section = True
                    in_entries = (key == "entries")
                    cur = None
                    continue
            if seen_section:
                if not in_entries:
                    continue
                mi = inline.match(s)
                if mi:
                    g = mi.group(1).strip().strip('"\'')
                    ph = [p.strip().strip('"\'')
                          for p in mi.group(2).split(",") if p.strip()]
                    if g and ph:
                        out[g.lower()] = ph
                    continue
                mg = gz.match(s)
                if mg:
                    cur = mg.group(1).strip().strip('"\'')
                    continue
                mp = pz.match(s)
                if mp and cur is not None:
                    ph = [p.strip().strip('"\'')
                          for p in mp.group(1).split(",") if p.strip()]
                    if ph:
                        out[cur.lower()] = ph
                    cur = None
            else:
                st = s.strip()
                if st and not st.startswith(("#", ";", "%", "-", "?")):
                    parts = _re.split(r"\s+", st)
                    if len(parts) >= 2 and not st.endswith(":"):
                        out[parts[0].lower()] = parts[1:]
            if len(out) >= limit:
                break
    return out


def cleaned_dictionary_text(entries: dict) -> str:
    """Canonical, deterministic ``word phone phone`` dictionary format."""
    lines = []
    for raw_word, raw_phones in sorted(
            dict(entries or {}).items(), key=lambda item: item[0].casefold()):
        word = " ".join(str(raw_word).strip().split()).lower()
        phones = [str(phone).strip() for phone in (raw_phones or [])
                  if str(phone).strip()]
        if word and phones and " " not in word:
            lines.append(word + " " + " ".join(phones))
    return "\n".join(lines) + ("\n" if lines else "")


def parse_cleaned_dictionary_text(text: str) -> dict:
    out = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "%")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0].lower()] = parts[1:]
    return out


def cleaned_dictionary_filename(source_name: str) -> str:
    import re as _re
    stem = Path(str(source_name or "dictionary")).stem
    stem = _re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "dictionary") + "-cleaned.dict"


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def migrate_registered_voice_paths(cfg: dict) -> dict:
    """Normalize registrations to one Windows path plus derived WSL path.

    WSL-only ``/home/...`` voices remain registered and are labelled legacy;
    migration never copies, moves, removes, or probes source UTAU folders.
    """
    festival = cfg.setdefault("festival_wsl", {})
    voices = festival.setdefault("voices", {})
    windows_roots = []
    migrated = {}
    for name, raw in dict(voices).items():
        if not isinstance(raw, dict):
            continue
        registration = migrate_voice_registration(raw)
        local = str(registration.get("windows_path") or "")
        metadata = {}
        if local:
            windows_roots.append(str(Path(local).parent))
            try:
                candidate = Path(local) / "dic" / "diphone_index.json"
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                metadata = {}
        registration = migrate_voice_registration(
            registration, metadata=metadata
        )
        registration.pop("name", None)
        migrated[str(name)] = registration
    festival["voices"] = migrated

    configured_root = str(festival.get("generated_voice_root") or "").strip()
    if configured_root:
        try:
            festival["generated_voice_root"] = canonical_windows_path(
                configured_root
            )
        except ValueError:
            festival["generated_voice_root"] = DEFAULT_GENERATED_VOICE_ROOT
    elif windows_roots:
        try:
            festival["generated_voice_root"] = os.path.commonpath(
                windows_roots
            )
        except ValueError:
            festival["generated_voice_root"] = DEFAULT_GENERATED_VOICE_ROOT
    else:
        festival["generated_voice_root"] = DEFAULT_GENERATED_VOICE_ROOT
    return cfg


def load_config(path: str = "config.json") -> dict:
    """config.json merged over DEFAULT_CONFIG. Missing file -> defaults.
    Legacy keys from the old Festival-based GUI are dropped silently."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return cfg
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"config.json is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("config.json must contain a JSON object")
    for legacy in ("festival", "voicebanks", "samplerate"):
        data.pop(legacy, None)
    if isinstance(data.get("languages"), list):   # legacy list form
        data.pop("languages")
    _deep_merge(cfg, data)
    if not cfg.get("languages"):
        raise ValueError("config.json: 'languages' must be a non-empty "
                         "{label: code} object")
    cfg["phrase_pauses_ms"] = normalize_phrase_pauses_ms(
        cfg.get("phrase_pauses_ms"))
    return migrate_registered_voice_paths(cfg)


def save_config(cfg: dict, path: str) -> None:
    out = {k: cfg[k] for k in DEFAULT_CONFIG if k in cfg}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------- data
@dataclass
class Segment:
    phone: str
    start: float
    end: float
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Optional linguistic timing role.  This preserves Japanese special-mora
    # identity after canonical N is mapped to a bank alias such as nn or nng.
    timing_role: str = ""

    def __post_init__(self):
        if not str(self.uid or "").strip():
            self.uid = uuid.uuid4().hex

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Synthesis:
    samples: np.ndarray            # float32 mono, [-1, 1]
    sr: int
    segments: List[Segment] = field(default_factory=list)
    text: str = ""
    lang: str = ""                 # synth_diphone code: asaxi / en / ja
    voicebank: str = ""
    phones: List[str] = field(default_factory=list)
    diphones: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    # F0 target points [(abs_time_s, hz)] captured from Festival's Target
    # relation -- lets a phoneme re-render keep the original pitch contour
    targets: List = field(default_factory=list)
    # The unedited contour remains available underneath user overrides.
    generated_targets: List = field(default_factory=list)
    pitch_override: List = field(default_factory=list)
    intonation_blocks: List = field(default_factory=list)
    pitch_mode: str = ""
    unit_overrides: dict = field(default_factory=dict)
    # Actual outgoing unit selected for each segment index. This includes
    # automatic context choices, not only explicit user overrides.
    selected_units: dict = field(default_factory=dict)
    fault_events: List = field(default_factory=list)
    output_bit_depth: int = 0
    phrase_ranges: List = field(default_factory=list)
    # Pitchmarks and unit handoffs from the rendered signal path.  Festival
    # exposes these through TargetCoef/US_map after UniSyn mapping.  They are
    # optional so older projects and non-UniSyn backends remain loadable.
    target_pitchmarks: List = field(default_factory=list)
    splice_records: List = field(default_factory=list)
    # Measured source-frame phase corrections made by the native renderer.
    # These are diagnostic provenance only: they never replace recording
    # choices, phone timing, target F0, or the editable crossover settings.
    frame_trajectory_records: List = field(default_factory=list)
    join_repairs: List = field(default_factory=list)
    # Requested and effective UniSyn source-window geometry.  Festival exposes
    # this per utterance rather than per join, so the GUI previews it at the
    # selected handoff while keeping its sentence-wide scope explicit.
    join_settings: dict = field(default_factory=dict)
    vowel_realizations: List = field(default_factory=list)
    # Continuous excitation control. ``source`` is the measured rendered
    # signal, ``generated`` includes safe linguistic defaults, and a manual
    # override is the final authority just like the Pitch curve.
    source_voicing_targets: List = field(default_factory=list)
    generated_voicing_targets: List = field(default_factory=list)
    voicing_override: List = field(default_factory=list)
    voicing_mode: str = ""
    voicing_diagnostics: List = field(default_factory=list)
    # Apparent tract length is independent of pitch and duration.  The
    # requested value is retained when DSP safety clamps it to a profile
    # boundary; 1.0 is the exact original-voice bypass.
    vocal_tract_requested_ratio: float = 1.0
    vocal_tract_length_ratio: float = 1.0
    chipmunk_range: bool = False
    generated_vocal_tract_targets: List = field(default_factory=list)
    vocal_tract_override: List = field(default_factory=list)
    applied_vocal_tract_targets: List = field(default_factory=list)
    vocal_tract_mode: str = ""
    vocal_tract_diagnostics: List = field(default_factory=list)
    # Effective Japanese production models. Keeping this on the rendered
    # object makes stale/legacy routing visible without reopening project JSON.
    japanese_prosody: dict = field(default_factory=dict)
    automatic_gain_db: float = 0.0
    pre_calibration_active_rms: Optional[float] = None
    output_calibration: dict = field(default_factory=dict)
    warning: Optional[str] = None
    # Source-selection view aligned to ``segments``. Canonical ``cl`` remains
    # visible/editable in Segment.phone while generated UTAU banks source its
    # interval from a bounded hold of the following consonant.
    # These fields are appended to preserve the positional constructor order
    # used by older integrations.
    render_phones: List[str] = field(default_factory=list)
    special_phone_realizations: List = field(default_factory=list)
    # Dictionary-driven Asaxi G2P/accent planning provenance. Kept last for
    # positional compatibility with older Synthesis integrations.
    asaxi_prosody: dict = field(default_factory=dict)
    # Dependency-free English syllable boundaries inferred from the rendered
    # phone stream. This is diagnostic metadata only and does not alter audio.
    english_syllabification: dict = field(default_factory=dict)

    def __post_init__(self):
        if (normalize_language_code(self.lang) == "en"
                and not self.english_syllabification):
            phones = list(self.phones)
            if not phones:
                phones = [str(segment.phone) for segment in self.segments]
            self.english_syllabification = (
                english_syllable_domain.syllabify_english(phones).to_dict()
            )

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sr) if self.sr else 0.0


class BackendError(RuntimeError):
    """Raised with a user-actionable message (shown in a dialog)."""


def resolve_voice_special_phones(
    phones, metadata=None, *, voicebank: str = "", available_diphones=None,
    allow_unverified_inventory: bool = False,
):
    """Return language-neutral canonical/source phone views for one voice.

    Kal and unknown external Festival voices own their native phonesets, so
    their ``cl`` remains literal. Generated UTAU voices advertise (or inherit)
    the structural policy and must contain the C-C hold diphones emitted by
    current builders.
    """
    voice_metadata = dict(metadata or {})
    if str(voicebank) == "kal_diphone":
        # Kal owns a native Festival closure phone rather than a generated
        # UTAU source alias. Ignore any stale mirrored metadata and leave its
        # authored phoneset semantics intact.
        voice_metadata = {}
    resolution = special_phone_domain.resolve_special_phone_sequence(
        phones,
        metadata=voice_metadata,
        available_diphones=available_diphones,
        allow_unverified_inventory=allow_unverified_inventory,
    )
    missing = [
        row for row in resolution.unresolved
        if row.status == "missing_source_diphones"
    ]
    if missing:
        details = "; ".join(
            "%s needs %s" % (
                row.phone,
                ", ".join(row.missing_diphones),
            )
            for row in missing
        )
        raise BackendError(
            "This generated voice predates structural cl support "
            f"({details}). Rebuild it with the current FestVox builder. "
            "The renderer refused to substitute a literal cl OTO alias."
        )
    unavailable = [
        row for row in resolution.unresolved
        if row.status == "inventory_unavailable"
    ]
    if unavailable:
        raise BackendError(
            "This generated voice declares structural cl support, but "
            "its diphone inventory is unavailable. Reload or rebuild the "
            "voice metadata; the renderer will not guess at cl source units."
        )
    unsupported = [
        row for row in resolution.unresolved
        if row.status == "unsupported_mode"
    ]
    if unsupported:
        raise BackendError(
            "Unsupported special-phone realization mode: " +
            ", ".join("%s=%s" % (row.phone, row.mode)
                      for row in unsupported)
        )
    return resolution


def apply_special_phone_display(
    synthesis: Synthesis,
    display_phones,
    render_phones,
    realizations,
) -> Synthesis:
    """Relabel rendered segments without changing their audio or timing."""
    display = [str(phone) for phone in display_phones]
    render = [str(phone) for phone in render_phones]
    synthesis.render_phones = [segment.phone for segment in synthesis.segments]
    synthesis.special_phone_realizations = [
        row.to_dict() if hasattr(row, "to_dict") else dict(row)
        for row in realizations
    ]
    if (
        len(synthesis.segments) == len(display) == len(render)
        and all(segment.phone == phone
                for segment, phone in zip(synthesis.segments, render))
    ):
        for segment, phone in zip(synthesis.segments, display):
            segment.phone = phone
        synthesis.render_phones = render
    elif synthesis.special_phone_realizations:
        transformed = any(
            str(row.get("source_phone") or row.get("phone") or "")
            != str(row.get("phone") or "")
            for row in synthesis.special_phone_realizations
        )
        if transformed:
            raise BackendError(
                "Festival returned a Segment layout that cannot be aligned "
                "with the canonical special-phone sequence. Rendering was "
                "stopped so source phones cannot replace editable cl regions."
            )
        note = (
            "special-phone source rendering completed, but Festival returned "
            "a different Segment count; source/display relabeling was skipped"
        )
        synthesis.warning = (
            synthesis.warning + "; " + note if synthesis.warning else note
        )
    return synthesis


def validate_generated_voice_dir(path) -> Path:
    """Accept only a path-shaped generated DB/voice, never an UTAU source."""
    p = Path(path).expanduser().resolve()
    if p == Path(p.anchor) or len(p.parts) < 3:
        raise BackendError(f"Refusing unsafe voicebank path:\n{p}")
    if (p / "oto.ini").is_file() or next(p.rglob("oto.ini"), None) is not None:
        raise BackendError(
            "Refusing to delete an UTAU source voicebank "
            "(oto.ini found in its directory tree):\n"
            + str(p))
    has_db = (p / "dic" / "diphone_index.json").is_file()
    has_voice = ((p / "festvox").is_dir() and
                 any((p / "festvox").glob("*.scm")))
    if not (has_db or has_voice):
        raise BackendError(
            "Refusing to delete this folder because it does not look like a "
            "generated FestVox voicebank:\n" + str(p))
    return p


def delete_generated_voice_dir(path) -> Path:
    """Delete a validated generated voice directory. UI confirmation is caller-owned."""
    import shutil
    p = validate_generated_voice_dir(path)
    shutil.rmtree(p)
    return p


# ---------------------------------------------------------------------- WAV IO
def read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr, n, ch, sw = (w.getframerate(), w.getnframes(),
                         w.getnchannels(), w.getsampwidth())
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw} bytes")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32), sr


def parse_est_pitchmarks(text: str) -> List[float]:
    """Read the timestamp column from an EST Track pitchmark file."""
    rows = []
    in_data = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not in_data:
            in_data = line == "EST_Header_End"
            continue
        if not line:
            continue
        try:
            value = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        if np.isfinite(value) and value >= 0.0:
            rows.append(value)
    return sorted(set(rows))


def parse_pitchmark_f0_sidecar(text: str) -> dict:
    """Validate the deterministic analyzed-F0 sidecar beside a PM file."""
    try:
        payload = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = []
    previous = -1.0
    for row in payload.get("frames") or ():
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            time = float(row[0])
            value = float(row[1])
        except (TypeError, ValueError):
            continue
        if (not np.isfinite(time) or not np.isfinite(value)
                or time < previous or time < 0.0 or value < 0.0):
            continue
        rows.append((time, value))
        previous = time
    if not rows:
        return {}
    return {
        "f0_source": str(payload.get("f0_source") or "analyzed-f0"),
        "frames": rows,
    }


def pitchmark_f0_track(pitchmarks) -> List[tuple]:
    """Convert adjacent PSOLA pitchmarks to midpoint/F0 display points."""
    marks = np.asarray(list(pitchmarks or []), dtype=np.float64)
    if marks.size < 2:
        return []
    periods = np.diff(marks)
    valid = np.isfinite(periods) & (periods > 1e-6)
    return [
        (float((marks[index] + marks[index + 1]) * .5),
         float(1.0 / periods[index]))
        for index in np.flatnonzero(valid)
    ]


def pitchmark_discontinuities(pitchmarks, ratio_limit: float = 1.35) -> List[dict]:
    """Find isolated source-period jumps that can destabilize UniSyn PSOLA.

    The comparison is local and diagnostic only. It does not reinterpret the
    sentence target F0 and it never writes to a source UTAU bank.
    """
    marks = np.asarray(list(pitchmarks or []), dtype=np.float64)
    if marks.size < 4:
        return []
    periods = np.diff(marks)
    rows = []
    for index, period in enumerate(periods):
        if not np.isfinite(period) or period <= 1e-6:
            continue
        before = periods[max(0, index - 2):index]
        after = periods[index + 1:min(len(periods), index + 3)]
        neighbors = np.concatenate((before, after))
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 1e-6)]
        if neighbors.size < 2:
            continue
        reference = float(np.median(neighbors))
        ratio = max(float(period) / reference, reference / float(period))
        if ratio < float(ratio_limit):
            continue
        rows.append({
            "period_index": int(index),
            "time": float((marks[index] + marks[index + 1]) * .5),
            "period_s": float(period),
            "f0_hz": float(1.0 / period),
            "local_period_s": reference,
            "local_f0_hz": float(1.0 / reference),
            "ratio": float(ratio),
        })
    return rows


def parse_unisyn_render_diagnostics(text: str, segments=None) -> dict:
    """Recover rendered PSOLA epochs and unit handoffs from Festival output.

    ``us_mapping`` maps every target pitchmark to a concatenated source frame.
    A unit handoff is therefore the interval between the final target epoch
    mapped to the incoming unit and the first epoch mapped to the outgoing
    unit.  Its midpoint is the nominal splice sample used by the diagnostic;
    both bounding epochs are retained because PSOLA overlap-add makes the
    handoff a short interval rather than a mathematical hard cut.
    """
    source = str(text or "")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    target_rows = [
        (int(index), float(time))
        for index, time in re.findall(
            rf"\(GUIPM\s+(\d+)\s+({number})\)", source)
    ]
    map_rows = [
        {
            "source_index": int(source_index),
            "source_time": float(source_time),
            "target_index": int(target_index),
            "target_time": float(target_time),
        }
        for source_index, source_time, target_index, target_time in re.findall(
            rf"\(GUIMAP\s+(\d+)\s+({number})\s+(\d+)\s+({number})\)",
            source,
        )
    ]
    unit_rows = [
        {
            "unit_index": int(unit_index),
            "frame_count": int(frame_count),
            "first_source_frame": int(first_frame),
            "last_source_frame_exclusive": int(last_frame),
        }
        for unit_index, frame_count, first_frame, last_frame in re.findall(
            r"\(GUIUFRAME\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\)",
            source,
        )
    ]
    frame_trajectory_rows = [
        {
            "target_index": int(target_index),
            "time": float(time),
            "previous_source_frame": int(previous_source),
            "source_frame": int(source_frame),
            "centre_offset_samples": int(centre_offset),
            "original_correlation": float(original_correlation),
            "corrected_correlation": float(corrected_correlation),
            "area_resampled": bool(int(area_resampled)),
            "contribution_count": int(contribution_count),
            "reason": reason,
        }
        for (target_index, time, previous_source, source_frame,
             centre_offset, original_correlation, corrected_correlation,
             area_resampled, contribution_count, reason)
        in re.findall(
            rf"\(GUIFRAMEFIX\s+(\d+)\s+({number})\s+(\d+)\s+(\d+)\s+"
            rf"(-?\d+)\s+({number})\s+({number})\s+([01])\s+(\d+)\s+"
            r'"([^"]*)"\)',
            source,
        )
    ]
    crossover_rows = [
        {
            "unit_index": int(unit_index),
            "target_handoff_index": int(target_handoff),
            "target_start_index": int(target_start),
            "target_end_index": int(target_end),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "requested_left_ms": float(requested_left),
            "requested_right_ms": float(requested_right),
            "context_cap_ms": float(context_cap),
            "effective_ms": float(effective_ms),
            "minimum_mixture_retention": float(retention),
            "active": bool(int(active)),
            "phone": phone,
            "context": context,
            "reason": reason,
        }
        for (unit_index, target_handoff, target_start, target_end,
             start_time, end_time, requested_left, requested_right,
             context_cap, effective_ms, retention, active, phone, context,
             reason)
        in re.findall(
            rf"\(GUIXOVER\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
            rf"({number})\s+({number})\s+({number})\s+({number})\s+"
            rf"({number})\s+({number})\s+({number})\s+([01])\s+"
            r'"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\)',
            source,
        )
    ]
    target_rows = sorted(set(target_rows))
    map_rows.sort(key=lambda row: (
        row["target_index"], row["source_index"]))
    unit_rows.sort(key=lambda row: row["unit_index"])
    frame_trajectory_rows.sort(key=lambda row: row["target_index"])
    crossover_rows.sort(key=lambda row: row["unit_index"])
    crossover_by_unit = {
        row["unit_index"]: row for row in crossover_rows
    }
    # The real TargetCoef -> SourceCoef map is monotonic. Festival 2.5 closes
    # lmap with one or more final target daughters mapped back to source frame
    # zero; long renders can have two such rows. Reconstruct one monotonic
    # source choice per target and stop at the first trailing wrap. This keeps
    # ordinary many-target-to-one-source stretching intact.
    monotonic_map = []
    last_source_index = -1
    target_indices = sorted(set(row["target_index"] for row in map_rows))
    for target_index in target_indices:
        candidates = sorted(
            (row for row in map_rows
             if row["target_index"] == target_index and
             row["source_index"] >= last_source_index),
            key=lambda row: row["source_index"])
        if not candidates:
            break
        chosen = candidates[0]
        monotonic_map.append(chosen)
        last_source_index = chosen["source_index"]
    map_rows = monotonic_map
    rendered_segments = list(segments or ())
    for row in frame_trajectory_rows:
        when = float(row["time"])
        for segment_index, segment in enumerate(rendered_segments):
            start_time = float(getattr(segment, "start", 0.0))
            end_time = float(getattr(segment, "end", start_time))
            if start_time <= when <= end_time:
                row["segment_index"] = segment_index
                row["phone"] = str(getattr(segment, "phone", ""))
                break
        row["correlation_improvement"] = round(
            float(row["corrected_correlation"]) -
            float(row["original_correlation"]), 9)
        row["correction_kind"] = (
            "source-area-resample"
            if row["area_resampled"] else "phase-reference")
    splices = []
    for incoming in unit_rows[:-1]:
        boundary = int(incoming["last_source_frame_exclusive"])
        left = [row for row in map_rows if row["source_index"] < boundary]
        right = [row for row in map_rows if row["source_index"] >= boundary]
        if not left or not right:
            continue
        final_left = max(left, key=lambda row: row["target_index"])
        first_right = min(right, key=lambda row: row["target_index"])
        if first_right["target_index"] <= final_left["target_index"]:
            continue
        start = float(final_left["target_time"])
        end = float(first_right["target_time"])
        when = (start + end) * 0.5
        segment_index = int(incoming["unit_index"]) + 1
        record = {
            "unit_index": int(incoming["unit_index"]),
            "segment_index": segment_index,
            "time": round(when, 9),
            "handoff_start": round(start, 9),
            "handoff_end": round(end, 9),
            "target_left_pitchmark_index": int(final_left["target_index"]),
            "target_right_pitchmark_index": int(first_right["target_index"]),
            "source_left_frame_index": int(final_left["source_index"]),
            "source_right_frame_index": int(first_right["source_index"]),
            "source_frame_boundary": boundary,
            "position_source": "festival-us-map",
            "estimated": False,
        }
        crossover = crossover_by_unit.get(
            int(incoming["unit_index"]))
        if crossover:
            record.update({
                "crossover_active": bool(crossover["active"]),
                "crossover_requested_left_ms": round(
                    crossover["requested_left_ms"], 6),
                "crossover_requested_right_ms": round(
                    crossover["requested_right_ms"], 6),
                "crossover_context_cap_ms": round(
                    crossover["context_cap_ms"], 6),
                "crossover_effective_ms": round(
                    crossover["effective_ms"], 6),
                "crossover_epoch_intervals": max(
                    0,
                    int(crossover["target_end_index"]) -
                    int(crossover["target_start_index"]),
                ) if crossover["active"] else 0,
                "crossover_start": (
                    round(crossover["start_time"], 9)
                    if crossover["active"] else None
                ),
                "crossover_end": (
                    round(crossover["end_time"], 9)
                    if crossover["active"] else None
                ),
                "crossover_minimum_mixture_retention": round(
                    crossover["minimum_mixture_retention"], 6),
                "crossover_phone": crossover["phone"],
                "crossover_context": crossover["context"],
                "crossover_reason": crossover["reason"],
            })
        if 0 <= segment_index < len(rendered_segments):
            segment = rendered_segments[segment_index]
            start_time = float(getattr(segment, "start", 0.0))
            end_time = float(getattr(segment, "end", start_time))
            if end_time > start_time:
                record["phone_fraction"] = round(max(
                    0.0, min(1.0, (when - start_time) /
                             (end_time - start_time))), 9)
        splices.append(record)
    return {
        "target_pitchmarks": [time for _index, time in target_rows],
        "splice_records": splices,
        "frame_trajectory_records": frame_trajectory_rows,
        "crossover_records": crossover_rows,
    }


def wav_bytes_to_samples(data: bytes):
    """Decode in-memory 16-bit WAV bytes (synth_diphone output) -> float32."""
    import io
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0), sr


def samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import io
    s = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((s * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    with open(path, "wb") as f:
        f.write(samples_to_wav_bytes(samples, sr))


def shared_model_cache_info() -> dict:
    """Report immutable model-profile caches shared by both backends."""
    from asaxi_pitch import (
        pitch_model_cache_info as asaxi_pitch_model_cache_info,
    )
    from japanese_duration import duration_model_cache_info
    from japanese_frontend import japanese_frontend_cache_info
    from japanese_pitch import (
        pitch_model_cache_info as japanese_pitch_model_cache_info,
    )
    from vocal_tract import vocal_tract_model_cache_info

    owners = [
        asaxi_pitch_model_cache_info(),
        duration_model_cache_info(),
        japanese_frontend_cache_info(),
        japanese_pitch_model_cache_info(),
        vocal_tract_model_cache_info(),
    ]
    return {
        "owners": owners,
        "entries": sum(int(row.get("entries", 0)) for row in owners),
        "bytes": sum(int(row.get("bytes", 0)) for row in owners),
        "max_entries": sum(int(row.get("max_entries", 0)) for row in owners),
        "max_bytes": sum(int(row.get("max_bytes", 0)) for row in owners),
    }


def clear_shared_model_caches() -> dict:
    """Clear parsed model profiles without deleting their source files."""
    from asaxi_pitch import (
        clear_pitch_model_cache as clear_asaxi_pitch_model_cache,
    )
    from japanese_duration import clear_duration_model_cache
    from japanese_frontend import clear_japanese_frontend_cache
    from japanese_pitch import (
        clear_pitch_model_cache as clear_japanese_pitch_model_cache,
    )
    from vocal_tract import clear_vocal_tract_model_cache

    removed = [
        clear_asaxi_pitch_model_cache(),
        clear_duration_model_cache(),
        clear_japanese_frontend_cache(),
        clear_japanese_pitch_model_cache(),
        clear_vocal_tract_model_cache(),
    ]
    return {
        "owners": removed,
        "entries": sum(int(row.get("entries", 0)) for row in removed),
        "bytes": sum(int(row.get("bytes", 0)) for row in removed),
    }


def _sustain_cache_size(value) -> int:
    if isinstance(value, tuple) and value and hasattr(value[0], "nbytes"):
        return int(value[0].nbytes) + 32
    return 16


def _voice_metadata_size(value) -> int:
    """Account for the complete published graph, including nested choices."""
    return estimate_size_bytes(value)


_VOICE_RUNTIME_METADATA_OMIT = frozenset({
    # These audit/build graphs are duplicated in unit_alternatives.json.
    # Normal rendering needs the compact index and policy fields, while the
    # alternatives loader owns contextual choices and their provenance.
    "alternatives",
    "alias_metadata",
})


def _runtime_voice_metadata(metadata) -> dict:
    """Keep runtime fields while leaving duplicate builder graphs on disk.

    A current integrated bank can carry tens of megabytes of nested
    ``alternatives`` in ``diphone_index.json``.  Freezing and sizing that
    duplicate graph made it larger than the bounded cache, so every metadata
    query parsed it again.  The separate alternatives API retains the complete
    immutable choice records; the compact diphone ``index`` remains here for
    source-pitchmark diagnostics and backward compatibility.
    """
    if not isinstance(metadata, dict):
        return {}
    return {
        key: value for key, value in metadata.items()
        if key not in _VOICE_RUNTIME_METADATA_OMIT
    }


# ------------------------------------------------------------------------ DSP
def time_stretch(samples: np.ndarray, sr: int, factor: float,
                 hook: Optional[Callable] = None) -> np.ndarray:
    """Stretch audio in time by `factor` (>1 = longer/slower, <1 = shorter),
    preserving pitch and LOUDNESS (no re-normalization). WSOLA: time-domain
    waveform-similarity overlap-add -- clean for speech, no phase smearing.
    `hook(samples, sr, factor)` overrides the algorithm (rubberband/SoX/...).
    """
    x = np.asarray(samples, dtype=np.float32)
    factor = float(factor)
    if hook is not None:
        return np.asarray(hook(x, sr, factor), dtype=np.float32)
    if x.size < 64 or abs(factor - 1.0) < 1e-3:
        return x
    # The waveform editor may request very long previews.  The old 8x clamp
    # made _stretch_one pad the remainder with zeros, which sounded like the
    # vowel abruptly turned into silence.
    factor = max(0.125, factor)
    return _wsola(x, sr, factor)


def _wsola(x: np.ndarray, sr: int, factor: float,
           win_s: float = 0.030, seek_s: float = 0.008) -> np.ndarray:
    """WSOLA time stretch. Output length ~ factor*len(x); unity gain."""
    N = max(64, int(win_s * sr) & ~1)          # even window length
    H = N // 2                                  # synthesis hop (50% overlap)
    Ha = H / factor                             # analysis hop
    seek = max(1, int(seek_s * sr))
    win = np.hanning(N).astype(np.float32)
    n_frames = max(2, int(round((len(x) * factor - N) / H)) + 1)

    out = np.zeros(int(round(len(x) * factor)) + N, dtype=np.float32)
    wsum = np.zeros_like(out)
    prev_tail = None      # natural continuation reference (for similarity)
    pos_f = 0.0
    for j in range(n_frames):
        target = int(round(pos_f))
        lo = max(0, min(target - seek, len(x) - N))
        hi = max(lo + 1, min(target + seek, len(x) - N))
        if prev_tail is None or hi - lo <= 1:
            best = min(max(0, target), max(0, len(x) - N))
        else:
            # choose the start whose first H samples best match the natural
            # continuation of what we already wrote (NORMALIZED xcorr --
            # a plain dot product just gravitates to loud, misaligned spots)
            seg = x[lo:hi + H]
            if len(seg) < H + 1:
                best = lo
            else:
                corr = np.correlate(seg, prev_tail, mode="valid")[:hi - lo]
                sq = np.concatenate(([0.0], np.cumsum(seg.astype(np.float64)
                                                      ** 2)))
                energy = (sq[H:] - sq[:-H])[:hi - lo]
                ncc = corr / (np.sqrt(energy) + 1e-6)
                best = lo + int(np.argmax(ncc))
        frame = x[best:best + N]
        if len(frame) < N:
            frame = np.pad(frame, (0, N - len(frame)))
        o = j * H
        out[o:o + N] += frame * win
        wsum[o:o + N] += win
        prev_tail = x[best + H:best + H + H]
        if len(prev_tail) < H:
            prev_tail = np.pad(prev_tail, (0, H - len(prev_tail)))
        pos_f += Ha
    # smooth floor (no hard threshold -> no reconstruction jump at the edges)
    y = out / np.maximum(wsum, 0.25)
    y = y[:int(round(len(x) * factor))]
    # window-attenuated partial frames at both ends: fade them cleanly
    F = min(H // 2, len(y))
    if F > 4:
        y[:F] *= np.linspace(0.0, 1.0, F, dtype=np.float32)
        y[-F:] *= np.linspace(1.0, 0.0, F, dtype=np.float32)
    return y.astype(np.float32)


def _periodic_loop(source: np.ndarray, sr: int, wanted: int) -> np.ndarray:
    """Fill ``wanted`` samples from a stable, pitch-synchronous source loop."""
    x = np.asarray(source, np.float32)
    if wanted <= 0:
        return np.zeros(0, np.float32)
    if x.size < 32:
        return np.resize(x if x.size else np.zeros(1, np.float32), wanted)
    trim = min(x.size // 4, int(0.08 * sr))
    stable = x[trim:x.size - trim] if x.size - 2 * trim >= 64 else x
    stable = stable - float(np.mean(stable))
    lo = max(2, int(sr / 500.0))
    hi = min(len(stable) // 3, int(sr / 50.0))
    period = max(lo, min(hi, int(sr / 180.0)))
    if hi > lo:
        probe_n = min(len(stable), max(int(0.12 * sr), hi * 4))
        probe = stable[(len(stable) - probe_n) // 2:
                       (len(stable) - probe_n) // 2 + probe_n]
        corr = np.correlate(probe, probe, mode="full")[probe_n - 1:]
        if len(corr) > hi:
            period = lo + int(np.argmax(corr[lo:hi + 1]))
    cycles = max(3, min(12, len(stable) // max(1, period)))
    loop_n = max(period, cycles * period)
    center = len(stable) // 2
    start = max(0, min(len(stable) - loop_n, center - loop_n // 2))
    loop = stable[start:start + loop_n]
    if loop.size < 16:
        loop = stable
    # Integral pitch periods make the repeated boundary naturally continuous;
    # a short blend suppresses the residual mismatch from noisy recordings.
    repeats = int(np.ceil((wanted + loop.size) / max(1, loop.size)))
    out = np.tile(loop, max(1, repeats))[:wanted].astype(np.float32)
    seam = min(max(2, period // 2), loop.size // 4)
    if seam > 1 and loop.size < wanted:
        for at in range(loop.size, wanted, loop.size):
            k = min(seam, at, wanted - at)
            if k > 1:
                blend = np.linspace(0.0, 1.0, k, dtype=np.float32)
                out[at - k:at] = (out[at - k:at] * (1.0 - blend)
                                  + out[at:at + k] * blend)
    return out


def stretch_segment(samples: np.ndarray, sr: int, factor: float,
                    sustain=None, use_sustain: bool = True) -> np.ndarray:
    """Pitch-preserving preview stretch with a true indefinite vowel path.

    Up to roughly 3x, WSOLA keeps speech transitions intact.  Longer voiced
    segments preserve the original attack/release and fill their middle from
    the voicebank's X-X sustain sample (or the segment's stable centre when a
    bank has no sustain).  The returned length is exact and never zero-padded.
    """
    x = np.asarray(samples, np.float32)
    factor = max(0.125, float(factor or 1.0))
    wanted = max(1, int(round(len(x) * factor)))
    if not use_sustain or factor <= 3.0 or x.size < 64:
        y = time_stretch(x, sr, factor)
        if len(y) == wanted:
            return y
        return np.resize(y if len(y) else np.zeros(1, np.float32), wanted)
    edge = min(len(x) // 3, max(16, int(0.055 * sr)))
    if wanted <= edge * 2 + 1:
        return time_stretch(x, sr, factor)[:wanted]
    middle = _periodic_loop(
        np.asarray(sustain, np.float32) if sustain is not None else x,
        sr, wanted - edge * 2)
    y = np.concatenate((x[:edge], middle, x[-edge:])).astype(np.float32)
    fade = min(max(2, int(0.006 * sr)), edge, len(middle) // 2)
    if fade > 1:
        a = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[edge - fade:edge] = (y[edge - fade:edge] * (1.0 - a)
                               + y[edge:edge + fade] * a)
        join = edge + len(middle)
        y[join - fade:join] = (y[join - fade:join] * (1.0 - a)
                               + y[join:join + fade] * a)
    return y[:wanted]


def blend_junctions(chunks, sr: int, ms: float = 4.0) -> np.ndarray:
    """Concatenate per-segment audio chunks, smoothing each junction with a
    short linear cross-blend IN PLACE (total length preserved) so segment
    re-timing doesn't click/pop at the seams."""
    chunks = [np.asarray(c, np.float32) for c in chunks if len(c)]
    if not chunks:
        return np.zeros(1, np.float32)
    y = np.concatenate(chunks)
    n = max(2, int(sr * ms / 1000.0))
    pos = 0
    for c in chunks[:-1]:
        pos += len(c)
        k = min(n, pos, len(y) - pos)
        if k < 2:
            continue
        # remove the step discontinuity: fade half the jump into each side
        jump = float(y[pos] - y[pos - 1])
        y[pos - k:pos] += 0.5 * jump * np.linspace(0.0, 1.0, k,
                                                   dtype=np.float32)
        y[pos:pos + k] -= 0.5 * jump * np.linspace(1.0, 0.0, k,
                                                   dtype=np.float32)
    return y


def remap_targets(targets, old_segments, new_durs):
    """Carry F0 target points from a previous render onto a re-timed phone
    sequence. targets: [(abs_t, hz)]; old_segments: [Segment]; new_durs:
    [float] per NEW segment. When counts match, each target keeps its
    position proportionally within its segment; otherwise positions scale
    with total duration. Returns [(new_abs_t, hz)]."""
    if not targets:
        return []
    old = [(s.start, s.end) for s in old_segments]
    new_starts, t0 = [], 0.0
    for d in new_durs:
        new_starts.append(t0)
        t0 += float(d)
    total_new, total_old = t0, (old[-1][1] if old else 0.0) or 1.0
    out = []
    for t, f0 in targets:
        if len(old) == len(new_durs) and old:
            i = next((k for k, (s, e) in enumerate(old) if s <= t <= e),
                     min(len(old) - 1, max(0, len(old) - 1)))
            s, e = old[i]
            frac = 0.0 if e <= s else (t - s) / (e - s)
            nt = new_starts[i] + frac * new_durs[i]
        else:
            nt = t / total_old * total_new
        out.append((max(0.0, min(total_new, nt)), float(f0)))
    return out


def remap_targets_aligned(targets, old_segments, new_segments):
    """Carry targets across a structural phone edit without global scaling.

    ``remap_targets`` is intentionally positional and falls back to scaling
    the entire utterance when segment counts differ.  That is appropriate for
    a wholesale duration transform, but deleting one editor phone must not
    move F0 targets belonging to every later phone.  Stable segment IDs are
    authoritative here; legacy/id-less data falls back to phone-sequence
    matching.  Targets owned by a deleted segment are discarded.
    """
    points = [(float(time), float(value))
              for time, value in (targets or [])]
    old = list(old_segments or [])
    new = list(new_segments or [])
    if not points or not old or not new:
        return [] if not points or not new else remap_targets(
            points, old, [segment.dur for segment in new])

    old_ids = [str(getattr(segment, "uid", "") or "") for segment in old]
    new_ids = [str(getattr(segment, "uid", "") or "") for segment in new]
    if (len(old) == len(new) and
            ((all(old_ids) and old_ids == new_ids) or
             [str(segment.phone) for segment in old] ==
             [str(segment.phone) for segment in new])):
        return remap_targets(points, old, [segment.dur for segment in new])

    old_counts, new_counts = {}, {}
    for uid in old_ids:
        if uid:
            old_counts[uid] = old_counts.get(uid, 0) + 1
    for uid in new_ids:
        if uid:
            new_counts[uid] = new_counts.get(uid, 0) + 1
    old_unique = {
        uid: index for index, uid in enumerate(old_ids)
        if uid and old_counts[uid] == 1
    }
    new_unique = {
        uid: index for index, uid in enumerate(new_ids)
        if uid and new_counts[uid] == 1
    }
    index_map = {
        old_index: new_unique[uid]
        for uid, old_index in old_unique.items()
        if uid in new_unique
    }
    if not index_map:
        matcher = SequenceMatcher(
            a=[str(segment.phone) for segment in old],
            b=[str(segment.phone) for segment in new],
            autojunk=False,
        )
        for old_start, new_start, size in matcher.get_matching_blocks():
            for offset in range(size):
                index_map[old_start + offset] = new_start + offset
    if not index_map:
        return remap_targets(points, old, [segment.dur for segment in new])

    old_starts = np.asarray(
        [float(segment.start) for segment in old], dtype=np.float64)
    out = []
    for time, value in points:
        old_index = int(np.searchsorted(
            old_starts, float(time), side="right")) - 1
        old_index = max(0, min(len(old) - 1, old_index))
        new_index = index_map.get(old_index)
        if new_index is None:
            continue
        old_segment = old[old_index]
        new_segment = new[new_index]
        old_duration = max(0.0, float(old_segment.dur))
        fraction = (
            0.0 if old_duration <= 1.0e-12 else
            (float(time) - float(old_segment.start)) / old_duration
        )
        fraction = max(0.0, min(1.0, fraction))
        new_time = (
            float(new_segment.start) + fraction * float(new_segment.dur)
        )
        out.append((new_time, value))
    return sorted(out)


def _pause_quartet(total, trailing_guard=0.12, leading_guard=0.08):
    """Four pause durations preserving ``total`` exactly.

    The first two parts belong to the outgoing phrase and the final two parts
    belong to the incoming phrase.  The inner pair splits the freely resizable
    phrase gap while the outer pair retains short splice guards.
    """
    total = max(0.0, float(total))
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    # Keep a useful combined gap even when the source pause is short.
    gap_floor = min(0.04, total / 3.0)
    guard_budget = max(0.0, total - gap_floor)
    desired = max(1e-9, float(trailing_guard) + float(leading_guard))
    scale = min(1.0, guard_budget / desired)
    trailing = max(0.0, float(trailing_guard) * scale)
    leading = max(0.0, float(leading_guard) * scale)
    gap = max(0.0, total - trailing - leading)
    trailing_gap = gap * 0.5
    leading_gap = gap - trailing_gap
    return trailing, trailing_gap, leading_gap, leading


def normalize_internal_pause_runs(seg_durs, trailing_guard: float = 0.12,
                                  leading_guard: float = 0.08):
    """Represent every internal phrase break as two outgoing/incoming pauses.

    Existing four-pause runs are retained verbatim.  Legacy three-pause edits
    preserve both guard durations and split only their former middle gap in
    half.  Other internal runs preserve their total duration while gaining an
    unambiguous two-pause boundary on each side.  Utterance-edge pauses are
    untouched.
    """
    entries = [(str(p), max(0.0, float(d))) for p, d in seg_durs]
    out = []
    index = 0
    while index < len(entries):
        phone, duration = entries[index]
        if phone != "pau":
            out.append((phone, duration))
            index += 1
            continue
        end = index
        while end < len(entries) and entries[end][0] == "pau":
            end += 1
        run = entries[index:end]
        internal = index > 0 and end < len(entries)
        if not internal or len(run) == 4:
            out.extend(run)
        elif len(run) == 3:
            middle = run[1][1]
            out.extend((
                run[0],
                ("pau", middle * 0.5),
                ("pau", middle - middle * 0.5),
                run[2],
            ))
        else:
            total = sum(value for _name, value in run)
            quartet = _pause_quartet(
                total, trailing_guard=trailing_guard,
                leading_guard=leading_guard)
            out.extend(("pau", value) for value in quartet)
        index = end
    return out


def split_internal_pauses(seg_durs, lead_pause: float = 0.12,
                          trailing_pause: float = 0.08):
    """Compatibility name for four-part internal phrase normalization."""
    return normalize_internal_pause_runs(
        seg_durs, trailing_guard=lead_pause,
        leading_guard=trailing_pause)


def split_edge_pauses(seg_durs, guard_pause: float = 0.08):
    """Split lone utterance-edge pauses into independently editable pairs.

    Total duration is preserved. At the leading edge the inner pause keeps
    ``guard_pause`` and the outer pause carries the remainder; the order is
    mirrored at the trailing edge. Existing edge pairs are left unchanged.
    """
    entries = [(str(p), max(0.0, float(d))) for p, d in seg_durs]
    if not entries:
        return entries

    def pair(duration, leading):
        if duration <= 0.0:
            return [("pau", duration)]
        guard = min(max(0.01, float(guard_pause)),
                    max(0.01, duration - 0.01))
        outer = max(0.0, duration - guard)
        values = [("pau", outer), ("pau", guard)]
        return values if leading else list(reversed(values))

    leading_run = 0
    while leading_run < len(entries) and entries[leading_run][0] == "pau":
        leading_run += 1
    if leading_run == 1:
        entries[:1] = pair(entries[0][1], True)

    trailing_run = 0
    while (trailing_run < len(entries) and
           entries[len(entries) - 1 - trailing_run][0] == "pau"):
        trailing_run += 1
    if trailing_run == 1:
        entries[-1:] = pair(entries[-1][1], False)
    return entries


def collapse_pause_runs(seg_durs):
    """Collapse every consecutive pause run to one pause of equal duration."""
    out = []
    for phone, dur in seg_durs or []:
        phone, dur = str(phone), float(dur)
        if phone == "pau" and out and out[-1][0] == "pau":
            out[-1] = ("pau", out[-1][1] + dur)
        else:
            out.append((phone, dur))
    return out


def equalize_phone_durations(seg_durs, phone_dur: float = 0.10):
    """Give every non-pause phone one duration while preserving phrase gaps."""
    d = max(0.01, float(phone_dur))
    return [(str(p), float(old) if str(p) == "pau" else d)
            for p, old in seg_durs]


def text_phrase_chunks(text: str):
    """Split at Western spaced punctuation or Japanese punctuation."""
    import re as _re
    parts = [p.strip() for p in
             _re.split(
                 r"(?<=[.?!,:;])\s+|(?<=[\u3002\uff01\uff1f\u3001\uff0c\uff1a\uff1b])\s*",
                 str(text or "").strip())
             if p.strip()]
    out = []
    for part in parts:
        match = _re.search(
            r"([.?!,:;\u3002\uff01\uff1f\u3001\uff0c\uff1a\uff1b])"
            r"[^.?!,:;\u3002\uff01\uff1f\u3001\uff0c\uff1a\uff1b]*$",
            part)
        mark = match.group(1) if match else "."
        out.append((part, {
            "\u3002": ".", "\uff01": "!", "\uff1f": "?",
            "\u3001": ",", "\uff0c": ",", "\uff1a": ":",
            "\uff1b": ";",
        }.get(mark, mark)))
    return out


def split_document_sentences(text: str):
    """Split UTF text into sentence entries while preserving inline [pau]."""
    import re as _re
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    pieces = _re.split(
        r"(?<=[.!?])(?:[ \t]+|\n+)|(?<=[\u3002\uff01\uff1f])\s*|\n{2,}",
        source)
    out = []
    for piece in pieces:
        clean = " ".join(piece.split())
        if clean:
            out.append(clean)
    if len(out) <= 1:
        lines = [" ".join(line.split()) for line in source.split("\n")
                 if line.strip()]
        if len(lines) > len(out):
            out = lines
    return out


def split_sentence_phrases(text: str):
    """Split one sentence at strong explicit, punctuation, or line breaks.

    One ``[pau]`` is an inline hesitation inside the current logical phrase.
    Two or more consecutive tokens remain an explicit phrase boundary.
    """
    import re as _re
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    source = _re.sub(r"[ \t]+", " ", source).strip()
    if not source:
        return []
    pieces = _re.split(
        r"[ \t]*\n+[ \t]*|(?:[ \t]*\[pau\][ \t]*){2,}|"
        r"(?<=[.?!,:;])\s+|"
        r"(?<=[\u3002\uff01\uff1f\u3001\uff0c\uff1a\uff1b])\s*",
        source,
        flags=_re.IGNORECASE)
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_synthesis_text(text: str, lang: str) -> str:
    """Apply language-local orthographic normalization before a frontend."""
    source = str(text or "")
    if str(lang or "").casefold() != "asaxi":
        return source

    # Native Asaxi is lower case, while a full-cap token is an explicit
    # switch to English G2P for a proper name or borrowed term. Preserve those
    # tokens so both synthesis backends can reach the mixed-language frontend;
    # normalize ordinary and accidentally mixed-case Asaxi words as before.
    return asaxi_frontend_domain.ASAXI_WORD_RE.sub(
        lambda match: (
            match.group(0)
            if asaxi_frontend_domain.is_capitalized_term(match.group(0))
            else match.group(0).lower()
        ),
        source,
    )


def phrase_pause_durations(mark: str, speed: float = 1.0):
    return phrase_pause_durations_with_settings(mark, speed, None)


def normalize_phrase_pauses_ms(value=None):
    source = dict(value or {})
    result = {}
    for key, default in DEFAULT_PHRASE_PAUSES_MS.items():
        try:
            number = int(round(float(source.get(key, default))))
        except (TypeError, ValueError):
            number = default
        result[key] = max(0, min(2000, number))
    return result


def japanese_inline_pause_kinds(text: str):
    """Classify non-terminal Japanese symbols that Open JTalk voices as pau."""
    import re as _re

    source = str(text or "")
    bracket_characters = (
        "「」『』【】〈〉《》（）()［］[]“”\""
    )
    pattern = _re.compile(
        r"\[pau\]|[" + _re.escape(
            bracket_characters + "・▽"
        ) + r"]+",
        _re.IGNORECASE,
    )

    def has_speech(value):
        return any(character.isalnum() for character in value)

    events = []
    for match in pattern.finditer(source):
        if not has_speech(source[:match.start()]) \
                or not has_speech(source[match.end():]):
            continue
        marker = match.group(0)
        if marker.casefold() == "[pau]":
            kind = "explicit"
        elif "▽" in marker:
            kind = "major"
        elif "・" in marker:
            kind = "minor"
        else:
            kind = "inline_bracket"
        events.append(kind)
    return events


def retime_japanese_inline_pauses(
        seg_durs, text: str, speed: float = 1.0, settings=None):
    """Bound Open JTalk's inline bracket pauses without removing prosody.

    The first Festival pass remains authoritative for phones, duration, accent
    and F0.  We only retime a pause when the ordered internal pause runs match
    the ordered non-terminal source symbols exactly.  A mismatch returns the
    original plan so an unrelated pause can never be shortened accidentally.
    """
    entries = [(str(phone), max(0.0, float(duration)))
               for phone, duration in seg_durs]
    runs = []
    index = 0
    while index < len(entries):
        if entries[index][0] != "pau":
            index += 1
            continue
        end = index
        while end < len(entries) and entries[end][0] == "pau":
            end += 1
        if index > 0 and end < len(entries):
            runs.append((index, end))
        index = end

    kinds = japanese_inline_pause_kinds(text)
    if not kinds or len(kinds) != len(runs):
        return entries

    speed = max(0.25, float(speed or 1.0))
    pause_values = normalize_phrase_pauses_ms(settings)
    for kind, (start, end) in zip(kinds, runs):
        if kind == "explicit":
            continue
        old_values = [duration for _phone, duration in entries[start:end]]
        old_total = sum(old_values)
        if kind == "inline_bracket":
            target = min(
                old_total,
                JAPANESE_INLINE_BRACKET_PAUSE_MS / 1000.0 / speed,
            )
        else:
            target = pause_values[kind] / 1000.0 / speed
        minimum = 0.01 * (end - start)
        target = max(minimum, target)
        if old_total > 1e-12:
            weights = [value / old_total for value in old_values]
        else:
            weights = [1.0 / (end - start)] * (end - start)
        replacement = [target * weight for weight in weights]
        correction = target - sum(replacement)
        replacement[-1] += correction
        entries[start:end] = [
            ("pau", max(0.0, value)) for value in replacement
        ]
    return entries


def retime_japanese_synthesis_pauses(
        synthesis: Synthesis, text: str, speed: float = 1.0, settings=None):
    """Retain a seed's F0 alignment while applying inline-pause timing."""
    old_segments = list(synthesis.segments)
    original = [(segment.phone, segment.dur)
                for segment in old_segments]
    retimed = retime_japanese_inline_pauses(
        original, text, speed, settings)
    if retimed == original:
        return synthesis
    durations = [duration for _phone, duration in retimed]
    synthesis.targets = remap_targets(
        synthesis.targets, old_segments, durations)
    synthesis.generated_targets = remap_targets(
        synthesis.generated_targets, old_segments, durations)
    synthesis.segments = segments_from_durations(retimed)
    return synthesis


def phrase_pause_durations_with_settings(
        mark: str, speed: float = 1.0, settings=None):
    speed = max(0.25, float(speed or 1.0))
    values = normalize_phrase_pauses_ms(settings)
    level = ("minor" if mark == "," else
             "major" if mark in {":", ";"} else "sentence")
    total = values[level] / 1000.0 / speed
    return _pause_quartet(
        total,
        trailing_guard=0.12 / speed,
        leading_guard=0.08 / speed)


def phrase_boundary_marks(text: str):
    """Return ordered internal punctuation, pause, and line boundaries."""
    import re as _re
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    marks = []
    pattern = _re.compile(
        r"\[pau\]|[.?!,:;\u3002\uff01\uff1f\u3001\uff0c]|\n+",
        _re.IGNORECASE)
    mapping = {"\u3002": ".", "\uff01": "!", "\uff1f": "?",
               "\u3001": ",", "\uff0c": ","}
    previous_end = -1
    for match in pattern.finditer(source):
        if not source[match.end():].strip():
            continue
        raw = match.group(0)
        if raw.startswith("\n"):
            # A line break immediately following punctuation or [pau] is the
            # same boundary, not an additional silent phrase.
            if (marks and previous_end >= 0 and
                    not source[previous_end:match.start()].strip()):
                previous_end = match.end()
                continue
            mark = "."
        else:
            mark = "." if raw.casefold() == "[pau]" else raw
        marks.append(mapping.get(mark, mark))
        previous_end = match.end()
    return marks


def retime_internal_phrase_pauses(
        seg_durs, text: str, speed: float = 1.0, settings=None):
    """Apply semantic pause totals without changing the phone sequence."""
    entries = [(str(phone), max(0.0, float(duration)))
               for phone, duration in seg_durs]
    marks = phrase_boundary_marks(text)
    boundary_index = 0
    index = 0
    while index < len(entries):
        if entries[index][0] != "pau":
            index += 1
            continue
        end = index
        while end < len(entries) and entries[end][0] == "pau":
            end += 1
        if index > 0 and end < len(entries):
            mark = marks[boundary_index] if boundary_index < len(marks) else "."
            parts = phrase_pause_durations_with_settings(
                mark, speed, settings)
            count = end - index
            if count == 1:
                replacement = (sum(parts),)
            elif count == 2:
                replacement = (
                    parts[0] + parts[1],
                    parts[2] + parts[3],
                )
            elif count == 3:
                replacement = (
                    parts[0], parts[1] + parts[2], parts[3],
                )
            else:
                middle = (parts[1] + parts[2]) / max(1, count - 2)
                replacement = (parts[0],) + \
                    (middle,) * (count - 2) + (parts[3],)
            entries[index:end] = [
                ("pau", max(0.0, float(value)))
                for value in replacement[:count]
            ]
            boundary_index += 1
        index = end
    return entries


def segments_from_durations(entries):
    segments, t = [], 0.0
    for phone, dur in entries:
        segments.append(Segment(str(phone), t, t + float(dur)))
        t += float(dur)
    return segments


def merge_phrase_plans(syntheses, punctuation, speed=1.0,
                       single_pause=False, phrase_pauses_ms=None):
    """Merge separately front-ended phrases into one Segment/F0 plan."""
    merged, targets, offset = [], [], 0.0
    for i, syn in enumerate(syntheses):
        local = [(s.phone, s.dur) for s in syn.segments]
        if not local:
            continue
        leading_drop = 0.0
        if i > 0 and local[0][0] == "pau":
            if single_pause:
                leading_drop = local[0][1]
            else:
                _trailing, _trailing_gap, leading_gap, leading = \
                    phrase_pause_durations_with_settings(
                        punctuation[i - 1], speed, phrase_pauses_ms)
                local[0:1] = [
                    ("pau", leading_gap), ("pau", leading),
                ]
        if i < len(syntheses) - 1 and local[-1][0] == "pau":
            trailing, trailing_gap, leading_gap, leading = \
                phrase_pause_durations_with_settings(
                    punctuation[i], speed, phrase_pauses_ms)
            if single_pause:
                local[-1] = (
                    "pau", trailing + trailing_gap + leading_gap + leading)
            else:
                local[-1:] = [
                    ("pau", trailing), ("pau", trailing_gap),
                ]

        local_targets = remap_targets(
            syn.targets, syn.segments, [d for _, d in local])
        append_local = local
        if leading_drop:
            # Keep the phrase's target positions relative to its spoken onset
            # when the second phrase's leading edge pause is omitted.
            shift = local[0][1]
            append_local = local[1:]
            local_targets = [(max(0.0, t - shift), f) for t, f in local_targets]
        targets.extend((offset + t, f) for t, f in local_targets)
        merged.extend(append_local)
        offset += sum(d for _, d in append_local)

    return merged, targets, segments_from_durations(merged)


PITCH_MIN_HZ = 50.0
PITCH_MAX_HZ = 500.0


def metadata_voice_pitch_hz(metadata):
    """Return an honest voice default, including old headroom manifests."""
    meta = dict(metadata or {})
    source = str(meta.get("default_pitch_source") or "")
    if source == "speaker_median_plus_headroom":
        analysis = dict(meta.get("speaker_pitch_analysis") or {})
        candidates = (
            analysis.get("median_f0_hz"),
            meta.get("automatic_pitch_floor_hz"),
            meta.get("average_pitch_hz"),
        )
    else:
        candidates = (
            meta.get("default_synthesis_pitch_hz"),
            meta.get("average_pitch_hz"),
            dict(meta.get("speaker_pitch_analysis") or {}).get(
                "median_f0_hz"),
        )
    for candidate in candidates:
        try:
            value = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if PITCH_MIN_HZ <= value <= PITCH_MAX_HZ:
            return value
    return None


def intonation_targets(blocks, pitch: float, fall: float = 10.0):
    """Convert punctuation blocks into a restrained piecewise F0 contour."""
    base = min(PITCH_MAX_HZ, max(PITCH_MIN_HZ, float(pitch or 160.0)))
    spread = pitch_domain.fall_percent_to_span_semitones(fall)

    def hz(semitones):
        return pitch_domain.clamp_hz(
            pitch_domain.semitone_offset(base, semitones),
            PITCH_MIN_HZ, PITCH_MAX_HZ)

    def ratio_semitones(multiplier):
        return 12.0 * math.log2(float(multiplier))

    out = []
    for block in blocks or []:
        start = max(0.0, float(block.get("start", 0.0)))
        end = max(start + 0.01, float(block.get("end", start + 0.01)))
        mid = start + (end - start) * 0.55
        kind = str(block.get("kind") or ".")
        if kind == "?":
            vals = (-spread * 0.25, 0.0,
                    max(spread, ratio_semitones(1.12)))
        elif kind == "!":
            vals = (ratio_semitones(1.12), ratio_semitones(1.05),
                    -max(spread, ratio_semitones(1.05)))
        elif kind == ",":
            vals = (spread * 0.25, 0.0, ratio_semitones(1.05))
        elif kind == ":":
            vals = (ratio_semitones(1.03), ratio_semitones(1.02), 0.0)
        elif kind == ";":
            vals = (spread * 0.5, 0.0, -spread * 0.35)
        else:
            vals = (spread, 0.0, -spread)
        out.extend(((start, hz(vals[0])), (mid, hz(vals[1])),
                    (end, hz(vals[2]))))
    return sorted(out)


def overlay_intonation_targets(generated, blocks, pitch: float,
                               fall: float = 10.0):
    """Blend punctuation intonation into phrase edges, preserving the middle.

    English ToBI analyses locate the interrogative contrast principally in a
    phrase-final boundary tone, rather than replacing the utterance's lexical
    pitch accents.  This overlay therefore leaves the central 55-70% of a
    normal phrase untouched and smoothly approaches the requested edge tone.
    """
    source = sorted((float(t), float(f)) for t, f in (generated or []))
    if not source:
        return intonation_targets(blocks, pitch, fall)
    base = min(PITCH_MAX_HZ, max(PITCH_MIN_HZ, float(pitch or 160.0)))
    spread = pitch_domain.fall_percent_to_span_semitones(fall)

    def target_hz(semitones):
        return pitch_domain.clamp_hz(
            pitch_domain.semitone_offset(base, semitones),
            PITCH_MIN_HZ, PITCH_MAX_HZ)

    def ratio_semitones(multiplier):
        return 12.0 * math.log2(float(multiplier))

    def smooth(value):
        value = min(1.0, max(0.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    times = {float(t) for t, _f in source}
    specs = []
    for block in blocks or []:
        start = max(0.0, float(block.get("start", 0.0)))
        end = max(start + 0.01, float(block.get("end", start + 0.01)))
        dur = end - start
        end_zone = min(dur * 0.32, 0.42)
        start_zone = min(dur * 0.16, 0.22)
        kind = str(block.get("kind") or ".")
        # Zero Fall means "preserve the generated statement contour".  The
        # old period/default branch still targeted the global base pitch,
        # which pulled a linguistically lowered Japanese endpoint upward.
        # Expressive punctuation remains active at zero Fall.
        if kind not in {"?", "!", ",", ":", ";"} and spread <= 1.0e-9:
            continue
        times.update((start, start + start_zone, end - end_zone, end))
        specs.append((start, end, start_zone, end_zone, kind))

    if not specs:
        return source

    out = []
    for when in sorted(times):
        value = _sample_f0(source, when, base)
        for start, end, start_zone, end_zone, kind in specs:
            if not (start <= when <= end):
                continue
            end_mix = smooth((when - (end - end_zone)) /
                             max(0.001, end_zone))
            if kind == "?":
                target_end = target_hz(
                    max(spread * 1.7, ratio_semitones(1.12)))
            elif kind == "!":
                target_end = target_hz(
                    -max(spread, ratio_semitones(1.06)))
            elif kind == ",":
                target_end = target_hz(
                    max(spread * 0.65, ratio_semitones(1.04)))
            elif kind == ":":
                target_end = target_hz(spread * 0.18)
            elif kind == ";":
                target_end = target_hz(-spread * 0.25)
            else:
                target_end = target_hz(-spread)
            value = pitch_domain.blend_hz_log(value, target_end, end_mix)
            # Exclamations and continuations can have a restrained initial
            # reset; questions retain their generated onset and differ at H%.
            if kind in ("!", ",", ":") and start_zone > 0:
                start_mix = 1.0 - smooth((when - start) / start_zone)
                start_target = target_hz(
                    ratio_semitones(1.08) if kind == "!"
                    else spread * 0.25)
                value = pitch_domain.blend_hz_log(
                    value, start_target, start_mix)
        out.append((when, min(PITCH_MAX_HZ,
                              max(PITCH_MIN_HZ, float(value)))))
    return out


def intonation_overlay_required(blocks, fall: float = 0.0) -> bool:
    """Whether punctuation/Fall adds anything beyond generated linguistic F0."""
    if pitch_domain.fall_percent_to_span_semitones(fall) > 1.0e-9:
        return bool(blocks)
    return any(
        str(block.get("kind") or ".") in {"?", "!", ",", ":", ";"}
        for block in (blocks or [])
    )


def _sample_f0(points, when: float, default: float = 160.0) -> float:
    return pitch_domain.interpolate_hz_log(points, when, default)


def anchor_phrase_targets(entries, targets, default: float = 160.0,
                          min_hz: float = PITCH_MIN_HZ,
                          max_hz: float = PITCH_MAX_HZ):
    """Anchor F0 at both edges of every voiced phrase.

    Festival commonly places its first and last learned targets near phone
    centers. UniSyn then extrapolates through the remaining half-phones, which
    is both hard to edit accurately and can create an audible edge drift.
    Pauses delimit phrases, so each voiced run gets independent edge anchors.
    """
    pts = [(float(t), max(float(min_hz), min(float(max_hz), float(f))))
           for t, f in (targets or [])]
    if not pts:
        return []
    segs = segments_from_durations(entries)
    additions = []
    for block in phrase_blocks(segs):
        start, end = float(block["start"]), float(block["end"])
        local = [(t, f) for t, f in pts if start <= t <= end]
        source = local or pts
        if not any(abs(t - start) < 1e-7 for t, _f in pts):
            additions.append((start, _sample_f0(source, start, default)))
        if not any(abs(t - end) < 1e-7 for t, _f in pts):
            additions.append((end, _sample_f0(source, end, default)))
    anchored = sorted(pts + additions)
    return anchor_pause_targets(entries, anchored, default, min_hz, max_hz)


def anchor_pause_targets(entries, targets, default: float = 160.0,
                         min_hz: float = PITCH_MIN_HZ,
                         max_hz: float = PITCH_MAX_HZ):
    """Hold F0 through each pause using its neighbouring voiced pitch.

    Leading pauses use the first voiced value, trailing pauses use the final
    voiced value, and multi-pause phrase gaps interpolate one stable control
    per pause. Raw-F0 mode skips this function together with phrase anchors.
    """
    pts = sorted((float(t), float(f)) for t, f in (targets or []))
    if not pts:
        return []
    segs = segments_from_durations(entries)
    pause_runs = []
    run_index = 0
    while run_index < len(segs):
        if segs[run_index].phone != "pau":
            run_index += 1
            continue
        run_end = run_index
        while run_end < len(segs) and segs[run_end].phone == "pau":
            run_end += 1
        pause_runs.append((
            float(segs[run_index].start),
            float(segs[run_end - 1].end),
        ))
        run_index = run_end
    # A re-render may feed the previous pass's pause controls back into this
    # function.  Remove every stale interior point before rebuilding the
    # canonical pause contour, otherwise repeated edits can retain a dense,
    # visibly jagged history. Phrase-edge points at the run boundaries remain.
    pts = [
        (time, value) for time, value in pts
        if not any(
            start + 1.0e-7 < time < end - 1.0e-7
            for start, end in pause_runs
        )
    ]
    existing = [t for t, _f in pts]

    def add(when, value):
        if not any(abs(when - t) < 1e-7 for t in existing):
            pts.append((float(when), min(float(max_hz),
                                         max(float(min_hz), float(value)))))
            existing.append(float(when))

    i = 0
    while i < len(segs):
        if segs[i].phone != "pau":
            i += 1
            continue
        first = i
        while i < len(segs) and segs[i].phone == "pau":
            i += 1
        last = i - 1
        run_start, run_end = segs[first].start, segs[last].end
        before = [(t, f) for t, f in pts if t <= run_start + 1e-7]
        after = [(t, f) for t, f in pts if t >= run_end - 1e-7]
        left = before[-1][1] if before else (
            after[0][1] if after else default)
        right = after[0][1] if after else left
        count = last - first + 1
        for offset, index in enumerate(range(first, last + 1)):
            seg = segs[index]
            frac = (offset / max(1, count - 1)) if count > 1 else 0.5
            value = pitch_domain.blend_hz_log(left, right, frac)
            add(seg.start, left if offset == 0 else value)
            add((seg.start + seg.end) / 2.0, value)
            add(seg.end, right if offset == count - 1 else value)
    return sorted(pts)


def pitch_estimation_faults(entries, targets, base: float = 160.0, rng=None,
                            forced_events=None, forced_index=None,
                            probability: float = 0.18,
                            max_faults: int = 5,
                            min_hz: float = PITCH_MIN_HZ,
                            max_hz: float = PITCH_MAX_HZ):
    """Corrupt F0 on one or more phones and return exact fault events.

    Random mode independently selects each eligible phone, guarantees at least
    one fault, and caps the result. Pinned events carry ``broken_hz`` so a
    later render reproduces the heard plateau instead of deriving a new value
    from the phone index.
    """
    import random
    segs = segments_from_durations(entries)
    candidates = [i for i, seg in enumerate(segs)
                  if seg.phone != "pau" and seg.dur >= 0.02]
    if not candidates:
        return list(targets or []), []
    picker = rng or random.SystemRandom()
    raw_pins = forced_events or []
    if isinstance(raw_pins, dict):
        raw_pins = [raw_pins]
    pins = []
    for raw in raw_pins:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("segment"))
            broken_hz = float(raw.get("broken_hz"))
        except (TypeError, ValueError):
            continue
        if index in candidates:
            pins.append((index, max(float(min_hz),
                                    min(float(max_hz), broken_hz))))
    try:
        forced = int(forced_index) if forced_index is not None else None
    except (TypeError, ValueError):
        forced = None
    pinned = bool(pins or forced in candidates)
    if pins:
        choices = list(dict.fromkeys(index for index, _value in pins))
        exact = {index: value for index, value in pins}
    elif forced in candidates:
        choices = [forced]
        exact = {}
    else:
        chance = max(0.0, min(1.0, float(probability)))
        choices = [index for index in candidates
                   if float(picker.random()) < chance]
        if not choices:
            choices = [picker.choice(candidates)]
        limit = max(1, int(max_faults or 1))
        if len(choices) > limit:
            choices = sorted(picker.sample(choices, limit))
        exact = {}

    clean = [(float(t), float(f)) for t, f in (targets or [])]
    pts = list(clean)
    factors = (0.48, 0.58, 1.65, 1.9)
    events = []
    for idx in sorted(choices):
        seg = segs[idx]
        start_f0 = _sample_f0(clean, seg.start, base)
        end_f0 = _sample_f0(clean, seg.end, base)
        middle_f0 = _sample_f0(
            clean, (seg.start + seg.end) / 2.0, base)
        if idx in exact:
            broken = exact[idx]
        else:
            if forced in candidates:
                stable = idx + sum(ord(char) for char in str(seg.phone))
                factor = factors[stable % len(factors)]
            else:
                factor = picker.choice(factors)
            broken = max(float(min_hz),
                         min(float(max_hz), middle_f0 * factor))
        edge = min(0.015, seg.dur * 0.18)
        pts = [(t, f) for t, f in pts
               if not (seg.start < t < seg.end)]
        pts.extend(((seg.start, start_f0), (seg.start + edge, broken),
                    (seg.end - edge, broken), (seg.end, end_f0)))
        events.append({
            "segment": idx, "phone": seg.phone,
            "broken_hz": float(broken),
            "factor": float(broken / max(1e-9, middle_f0)),
            "pinned": pinned,
        })
    return sorted(set(pts)), events


def pitch_estimation_fault(entries, targets, base: float = 160.0, rng=None,
                           forced_index=None, min_hz: float = PITCH_MIN_HZ,
                           max_hz: float = PITCH_MAX_HZ):
    """Compatibility wrapper for one bounded broken-pitch phone."""
    broken, events = pitch_estimation_faults(
        entries, targets, base=base, rng=rng, forced_index=forced_index,
        probability=0.0, max_faults=1,
        min_hz=min_hz, max_hz=max_hz)
    return broken, (events[0]["segment"] if events else None)


BIT_DEPTH_GAINS = {8: 0.95, 4: 0.72, 2: 0.42, 1: 0.09}


def apply_bit_depth(samples, bits: int):
    """Quantize float audio with explicit low-bit volume compensation."""
    bits = int(bits or 0)
    x = np.asarray(samples, np.float32)
    if bits not in BIT_DEPTH_GAINS:
        return x.copy()
    x = np.clip(x, -1.0, 1.0)
    if bits == 1:
        quantized = np.sign(x)
        quantized[np.abs(x) < 1e-7] = 0.0
    else:
        levels = (2 ** bits) - 1
        quantized = (np.round((x + 1.0) * levels / 2.0)
                     * 2.0 / levels - 1.0)
        quantized[np.abs(x) < 1e-7] = 0.0
    return np.asarray(quantized * BIT_DEPTH_GAINS[bits], np.float32)


def apply_gain_db(samples, gain_db: float):
    """Apply bounded output gain without changing the caller's array."""
    gain_db = min(12.0, max(-60.0, float(gain_db or 0.0)))
    gain = 10.0 ** (gain_db / 20.0)
    return np.asarray(np.clip(np.asarray(samples, np.float32) * gain,
                              -1.0, 1.0), np.float32)


def active_speech_rms(samples, sample_rate: int, segments):
    """Measure one rendered phrase without counting editable pause regions."""
    x = np.asarray(samples, np.float64)
    sr = max(1, int(sample_rate or 0))
    total_energy = 0.0
    sample_count = 0
    for segment in segments or ():
        if str(getattr(segment, "phone", "")).casefold() in {
                "pau", "sil", "sp"}:
            continue
        start = max(0, min(len(x), int(round(float(segment.start) * sr))))
        end = max(start, min(len(x), int(round(float(segment.end) * sr))))
        if end <= start:
            continue
        block = x[start:end]
        total_energy += float(np.dot(block, block))
        sample_count += int(block.size)
    if sample_count <= 0:
        return 0.0, 0
    return float(np.sqrt(total_energy / sample_count)), sample_count


def apply_active_speech_calibration(synthesis: Synthesis, policy):
    """Apply one bounded phrase gain; never normalize individual units.

    The generated-voice manifest opts into this policy. Applying one scalar to
    a completed phrase preserves every contextual level relationship and unit
    choice while keeping different language frontends in a comparable output
    range.
    """
    if getattr(synthesis, "output_calibration", None):
        return synthesis
    settings = dict(policy or {})
    if settings.get("method") != "active_speech_rms":
        return synthesis
    rms, count = active_speech_rms(
        synthesis.samples, synthesis.sr, synthesis.segments
    )
    minimum_seconds = max(
        0.0, float(settings.get("minimum_active_seconds") or 0.0)
    )
    if count < int(round(minimum_seconds * max(1, synthesis.sr))) or rms <= 1e-9:
        synthesis.output_calibration = {
            "schema_version": int(settings.get("schema_version") or 1),
            "method": "active_speech_rms",
            "applied": False,
            "reason": "insufficient_active_audio",
            "active_sample_count": int(count),
        }
        return synthesis
    target_dbfs = float(settings.get("target_dbfs", -20.0))
    measured_dbfs = 20.0 * math.log10(rms)
    requested = target_dbfs - measured_dbfs
    minimum_gain = float(settings.get("minimum_gain_db", -6.0))
    maximum_gain = float(settings.get("maximum_gain_db", 6.0))
    gain_db = max(minimum_gain, min(maximum_gain, requested))
    peak = (float(np.max(np.abs(synthesis.samples)))
            if np.asarray(synthesis.samples).size else 0.0)
    peak_ceiling = max(0.01, min(
        1.0, float(settings.get("peak_ceiling", 0.98))
    ))
    if peak > 1e-9:
        gain_db = min(gain_db, 20.0 * math.log10(peak_ceiling / peak))
    synthesis.pre_calibration_active_rms = float(rms)
    synthesis.automatic_gain_db = float(gain_db)
    synthesis.samples = apply_gain_db(synthesis.samples, gain_db)
    synthesis.output_calibration = {
        "schema_version": int(settings.get("schema_version") or 1),
        "method": "active_speech_rms",
        "applied": True,
        "active_sample_count": int(count),
        "measured_dbfs": round(measured_dbfs, 6),
        "target_dbfs": round(target_dbfs, 6),
        "requested_gain_db": round(requested, 6),
        "applied_gain_db": round(gain_db, 6),
        "achieved_dbfs": round(measured_dbfs + gain_db, 6),
        "policy_source": str(settings.get("policy_source") or
                             "voice_metadata"),
        "scope": "completed_phrase",
    }
    return synthesis


def is_vowel_phone(phone: str) -> bool:
    return str(phone).rstrip("_") in _CLS_VOWELS


def is_timing_nucleus(phone: str, timing_role: str = "") -> bool:
    """Return whether duration editing should use the vowel/rhyme path.

    A Japanese moraic nasal occupies a mora and behaves as its timing nucleus,
    even when an integrated ARPAsing profile renders it with a symbol that is
    otherwise a nasal consonant.  The explicit role avoids globally changing
    English or Asaxi classification for the same bank symbol.
    """
    role = str(timing_role or "").casefold()
    return role in {"vowel", "moraic_nasal", "syllabic_nasal", "rhyme"} \
        or is_vowel_phone(phone)


def remap_unit_overrides(
    old_phones, new_phones, overrides, *,
    old_source_phones=None, new_source_phones=None,
) -> dict:
    """Keep manual diphone takes attached to unchanged phone transitions.

    Override keys are Festival segment indexes. Phone edits can insert or
    remove segments, so copying those integer keys directly can apply a take
    to the wrong transition. Sequence matching on actual source diphones,
    when supplied, keeps unchanged occurrences and drops transitions whose
    canonical spelling stayed the same but whose structural source changed
    (for example ``i-cl`` backed by ``i-s`` becoming ``i-t``).
    """
    from difflib import SequenceMatcher

    old_display = [str(phone) for phone in (old_phones or [])]
    new_display = [str(phone) for phone in (new_phones or [])]
    old_source = [str(phone) for phone in (old_source_phones or [])]
    new_source = [str(phone) for phone in (new_source_phones or [])]
    old = (
        old_source if len(old_source) == len(old_display) else old_display
    )
    new = (
        new_source if len(new_source) == len(new_display) else new_display
    )
    old_pairs = list(zip(old, old[1:]))
    new_pairs = list(zip(new, new[1:]))
    source = {int(index): str(name) for index, name in
              dict(overrides or {}).items()}
    result = {}
    matcher = SequenceMatcher(a=old_pairs, b=new_pairs, autojunk=False)
    for old_start, new_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            old_index = old_start + offset
            if old_index in source:
                result[new_start + offset] = source[old_index]
    return result


def transfer_segment_uids(old_segments, new_segments):
    """Carry editor identities onto a freshly rendered phone sequence.

    A phone label is not an identity: repeated phones and duplicated regions
    need to remain distinct across a re-render.  Matching the full occurrence
    sequence preserves every unchanged occurrence while genuinely inserted
    segments keep the fresh ID created by :class:`Segment`.
    """
    from difflib import SequenceMatcher

    old = list(old_segments or [])
    new = list(new_segments or [])
    matcher = SequenceMatcher(
        a=[str(segment.phone) for segment in old],
        b=[str(segment.phone) for segment in new],
        autojunk=False)
    assigned = set()
    for old_start, new_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            uid = str(old[old_start + offset].uid or "").strip()
            if uid and uid not in assigned:
                new[new_start + offset].uid = uid
                assigned.add(uid)
    return new


def _unit_context_score(choice, outer_left: str, outer_right: str,
                        l_class: str = "*") -> int:
    """Conservative context score mirrored by generated Festival Scheme.

    A non-wildcard mismatch keeps a candidate's absolute score negative, but
    the candidate may still improve on a base whose two contexts are worse.
    Phrase-edge-only improvements are rejected separately below.
    """
    def side(expected, actual, exact, expected_info, edge):
        expected = str(expected or "*")
        if expected == "*":
            return 0
        if expected == str(actual):
            return int(exact)
        wanted = str(expected_info.get("class") or "other")
        actual_class = context_edge_info(actual, edge)["class"]
        return 4 if wanted not in {"wildcard", "other"} and \
            wanted == actual_class else -8

    score = side(choice_recorded_context(choice, "left"), outer_left, 6,
                 choice_context_info(choice, "left"), "right")
    score += side(choice_recorded_context(choice, "right"), outer_right, 7,
                  choice_context_info(choice, "right"), "left")
    expected_class = str(choice.get("l_class") or "*")
    if l_class != "*" and expected_class != "*":
        score += 20 if expected_class == l_class else -100
    return score


def _unit_context_relation(choice, side: str, actual: str) -> str:
    expected = choice_recorded_context(choice, side)
    if expected == "*":
        return "wildcard"
    if expected == str(actual):
        return "exact"
    wanted = str(choice_context_info(choice, side).get("class") or "other")
    actual_edge = "right" if side == "left" else "left"
    actual_class = context_edge_info(actual, actual_edge)["class"]
    if wanted not in {"wildcard", "other"} and wanted == actual_class:
        return "class"
    return "mismatch"


def _unsafe_phrase_edge_shortcut(choice, outer_left: str,
                                 outer_right: str) -> bool:
    """Reject a take whose only improvement is an exact phrase-edge pause."""
    left = _unit_context_relation(choice, "left", outer_left)
    right = _unit_context_relation(choice, "right", outer_right)
    return ((str(outer_left) == "pau" and left == "exact" and
             right == "mismatch") or
            (str(outer_right) == "pau" and right == "exact" and
             left == "mismatch"))


def _base_unit_choice(choices):
    """Return the unnumbered base row even if metadata was reordered."""
    rows = list(choices or [])
    if not rows:
        return None
    explicit = next((choice for choice in rows
                     if str(choice.get("id") or "").lower() == "base"), None)
    if explicit is not None:
        return explicit
    return next((choice for choice in rows
                 if "__u" not in str(choice.get("left_name") or "")), rows[0])


def contextual_unit_choice(choices, outer_left: str, outer_right: str,
                           l_class: str = "*", right_phone: str = ""):
    """Choose a take from directional OTO evidence.

    Ordinary transitions compare every safe take with the base's actual score,
    allowing a matching incoming vowel context to beat a consonant-cluster
    base even when the far edge differs. Diphones ending in a voiced sibilant
    first prefer a verified supportive following context, then an unannotated
    context, and retain base if every recording is verified risky. Manual
    overrides bypass this function.
    """
    rows = [dict(choice) for choice in (choices or [])]
    if not rows:
        return None
    if str(outer_left) not in {"pau", "sil", "*"} and \
            str(outer_right) not in {"pau", "sil", "*"}:
        rows = [choice for choice in rows if "inh" not in {
            token.lower() for token in
            str(choice.get("alias") or "").replace("-", " ").split()
        }]
        if not rows:
            return None
    base = _base_unit_choice(rows)
    if not right_phone:
        index_name = str(rows[0].get("index_name") or "")
        if "-" in index_name:
            right_phone = index_name.split("-", 1)[1]
    right_phone = str(right_phone or "").rstrip("_").lower()

    reason = "best directional OTO context match"
    if right_phone in _VOICED_SIBILANTS:
        supportive = [choice for choice in rows
                      if _sibilant_context_quality(choice) ==
                      "verified_supportive"]
        unknown = [choice for choice in rows
                   if _sibilant_context_quality(choice) == "unknown"]
        if supportive or unknown:
            pool = supportive or unknown
            best = pool[0]
            best_score = _unit_context_score(
                best, outer_left, outer_right, l_class)
            for choice in pool[1:]:
                score = _unit_context_score(
                    choice, outer_left, outer_right, l_class)
                if score > best_score:
                    best, best_score = choice, score
            if supportive:
                context_class = choice_context_info(best, "right")["class"]
                reason = ("verified %s right context preferred for voiced "
                          "sibilant %s" %
                          (context_class.replace("_", " "), right_phone))
            else:
                reason = ("no verified supportive take; unannotated OTO "
                          "context preferred to a verified risky context "
                          "for voiced sibilant %s" % right_phone)
        else:
            best = base
            best_score = max(0, _unit_context_score(
                best, outer_left, outer_right, l_class))
            reason = ("all recorded right contexts are risky for voiced "
                      "sibilant %s; retained base" % right_phone)
    else:
        best = base
        best_score = _unit_context_score(
            best, outer_left, outer_right, l_class)
        for choice in rows:
            if choice is best or _unsafe_phrase_edge_shortcut(
                    choice, outer_left, outer_right):
                continue
            score = _unit_context_score(
                choice, outer_left, outer_right, l_class)
            if score > best_score:
                best, best_score = choice, score
        if best is base:
            reason = ("base retained; no numbered take had stronger safe "
                      "context evidence")
        else:
            reason = ("numbered take has stronger safe context evidence than "
                      "base")
    result = dict(best)
    result["context_score"] = int(best_score)
    result["context_quality"] = _sibilant_context_quality(best) \
        if right_phone in _VOICED_SIBILANTS else "ordinary"
    result["selection_reason"] = reason
    return result


def contextual_unit_overrides(phones, inventory) -> dict:
    """Return explicit safe automatic choices for a Festival phone list."""
    phones = [str(phone) for phone in (phones or [])]
    result = {}
    light_followers = set(_CLS_VOWELS) | {"y"}
    for index in range(max(0, len(phones) - 1)):
        left, right = phones[index], phones[index + 1]
        choices = list((inventory or {}).get("%s-%s" % (left, right)) or [])
        if not choices:
            continue
        outer_left = phones[index - 1] if index else "*"
        outer_right = phones[index + 2] if index + 2 < len(phones) else "*"
        l_class = "*"
        if left.rstrip("_") == "l":
            l_class = ("light" if right.rstrip("_") in light_followers
                       else "dark")
        elif right.rstrip("_") == "l":
            l_class = ("light" if outer_right.rstrip("_") in light_followers
                       else "dark")
        choice = contextual_unit_choice(
            choices, outer_left, outer_right, l_class, right_phone=right)
        if choice and choice.get("left_name"):
            result[index] = str(choice["left_name"])
    return result


def concat_audio(items, gap_s: float = 0.25):
    """Concatenate ``(samples, sample_rate)`` pairs at one sample rate."""
    valid = [(np.asarray(samples, np.float32), int(sr))
             for samples, sr in (items or []) if len(samples) and int(sr) > 0]
    if not valid:
        return np.zeros(1, np.float32), 16000
    target_sr = valid[0][1]
    chunks = []
    for samples, sr in valid:
        if sr != target_sr:
            n = max(1, int(round(len(samples) * target_sr / float(sr))))
            samples = np.interp(
                np.linspace(0.0, 1.0, n, endpoint=False),
                np.linspace(0.0, 1.0, len(samples), endpoint=False),
                samples).astype(np.float32)
        chunks.append(samples)
    gap = np.zeros(max(0, int(round(float(gap_s) * target_sr))), np.float32)
    joined = []
    for index, chunk in enumerate(chunks):
        if index and gap.size:
            joined.append(gap)
        joined.append(chunk)
    return np.concatenate(joined).astype(np.float32), target_sr


def combine_syntheses(syntheses, text: str = "", lang: str = "",
                      single_pause: bool = False):
    """Concatenate independently routed phrase renders into one synthesis.

    Independent phrases naturally retain both edge pauses. Normal output
    preserves their combined duration while exposing exactly four editable
    pause segments at each join: two for the outgoing phrase and two for the
    incoming phrase. ``single_pause`` removes the second phrase's leading
    pause and collapses the remaining run for diagnostic comparison.
    """
    syntheses = [syn for syn in (syntheses or []) if syn is not None]
    if not syntheses:
        return Synthesis(np.zeros(1, np.float32), 16000, [],
                         text=text, lang=lang)
    target_sr = int(syntheses[0].sr)
    samples_out = []
    segments_out = []
    targets = []
    generated = []
    target_pitchmarks = []
    splice_records = []
    frame_trajectory_records = []
    selected = {}
    skipped = []
    warnings = []
    render_phones_out = []
    special_phone_realizations = []
    vowel_realizations = []
    source_voicing = []
    generated_voicing = []
    voicing_override = []
    voicing_diagnostics = []
    calibration_rows = []
    japanese_prosody_rows = []
    asaxi_prosody_rows = []
    join_settings_rows = []
    phrase_ranges = []
    offset = 0.0
    segment_offset = 0

    for phrase_index, syn in enumerate(syntheses):
        if getattr(syn, "join_settings", None):
            join_settings_rows.append(dict(syn.join_settings))
        calibration = dict(getattr(syn, "output_calibration", None) or {})
        calibration_rows.append({
            "phrase_index": phrase_index,
            "automatic_gain_db": round(float(getattr(
                syn, "automatic_gain_db", 0.0)), 6),
            "calibration": calibration,
        })
        japanese_prosody = dict(getattr(
            syn, "japanese_prosody", None) or {})
        if japanese_prosody:
            japanese_prosody_rows.append({
                "phrase_index": phrase_index,
                **japanese_prosody,
            })
        asaxi_prosody = dict(getattr(
            syn, "asaxi_prosody", None) or {})
        if asaxi_prosody:
            asaxi_prosody_rows.append({
                "phrase_index": phrase_index,
                **asaxi_prosody,
            })
        samples = np.asarray(syn.samples, np.float32)
        if int(syn.sr) != target_sr:
            count = max(1, int(round(
                len(samples) * target_sr / float(syn.sr))))
            samples = np.interp(
                np.linspace(0.0, 1.0, count, endpoint=False),
                np.linspace(0.0, 1.0, len(samples), endpoint=False),
                samples).astype(np.float32)
        local_segments = list(syn.segments)
        local_render_phones = list(getattr(syn, "render_phones", ()) or ())
        if len(local_render_phones) != len(local_segments):
            local_render_phones = [
                segment.phone for segment in local_segments
            ]
        local_special_realizations = [
            dict(row) for row in getattr(
                syn, "special_phone_realizations", ()
            )
        ]
        local_targets = list(syn.targets)
        local_generated = list(syn.generated_targets)
        local_source_voicing = list(getattr(
            syn, "source_voicing_targets", ()))
        local_generated_voicing = list(getattr(
            syn, "generated_voicing_targets", ()))
        local_voicing_override = list(getattr(
            syn, "voicing_override", ()))
        local_pitchmarks = [float(value) for value in
                            syn.target_pitchmarks]
        local_splices = [dict(row) for row in syn.splice_records]
        local_trajectories = [dict(row) for row in getattr(
            syn, "frame_trajectory_records", ())]
        selected_units = dict(syn.selected_units)
        removed = 0
        if (single_pause and phrase_index and segments_out and
                segments_out[-1].phone == "pau"):
            trim = 0.0
            for segment in local_segments:
                if segment.phone != "pau":
                    break
                removed += 1
                trim = max(trim, float(segment.end))
            if removed and trim > 0.0:
                removed_pitchmarks = sum(
                    1 for time in local_pitchmarks if float(time) < trim)
                cut = min(len(samples), int(round(trim * target_sr)))
                samples = samples[cut:]
                local_segments = [Segment(
                    segment.phone, max(0.0, float(segment.start) - trim),
                    max(0.0, float(segment.end) - trim),
                    segment.uid, segment.timing_role)
                    for segment in local_segments[removed:]]
                local_render_phones = local_render_phones[removed:]
                adjusted_special = []
                for row in local_special_realizations:
                    adjusted = dict(row)
                    local_index = int(adjusted.get("index", -1)) - removed
                    if local_index < 0:
                        continue
                    adjusted["index"] = local_index
                    adjusted_special.append(adjusted)
                local_special_realizations = adjusted_special
                local_targets = [(float(time) - trim, float(value))
                                 for time, value in local_targets
                                 if float(time) >= trim]
                local_generated = [(float(time) - trim, float(value))
                                   for time, value in local_generated
                                   if float(time) >= trim]
                local_source_voicing = [
                    (float(time) - trim, float(value))
                    for time, value in local_source_voicing
                    if float(time) >= trim
                ]
                local_generated_voicing = [
                    (float(time) - trim, float(value))
                    for time, value in local_generated_voicing
                    if float(time) >= trim
                ]
                local_voicing_override = [
                    (float(time) - trim, float(value))
                    for time, value in local_voicing_override
                    if float(time) >= trim
                ]
                local_pitchmarks = [float(time) - trim
                                    for time in local_pitchmarks
                                    if float(time) >= trim]
                adjusted_splices = []
                for row in local_splices:
                    when = float(row.get("time") or 0.0)
                    if when < trim:
                        continue
                    adjusted = dict(row)
                    for key in ("time", "handoff_start", "handoff_end"):
                        if key in adjusted:
                            adjusted[key] = max(
                                0.0, float(adjusted[key]) - trim)
                    adjusted["segment_index"] = (
                        int(adjusted.get("segment_index", -1)) - removed)
                    adjusted_splices.append(adjusted)
                local_splices = adjusted_splices
                adjusted_trajectories = []
                for row in local_trajectories:
                    when = float(row.get("time") or 0.0)
                    if when < trim:
                        continue
                    adjusted = dict(row)
                    adjusted["time"] = max(0.0, when - trim)
                    if "target_index" in adjusted:
                        adjusted["target_index"] = (
                            int(adjusted["target_index"]) -
                            removed_pitchmarks)
                        if adjusted["target_index"] < 0:
                            continue
                    try:
                        local_segment_index = int(
                            adjusted.get("segment_index", -1)) - removed
                    except (TypeError, ValueError):
                        local_segment_index = -1
                    if local_segment_index >= 0:
                        adjusted["segment_index"] = local_segment_index
                    adjusted_trajectories.append(adjusted)
                local_trajectories = adjusted_trajectories
                selected_units = {
                    int(index) - removed: value
                    for index, value in selected_units.items()
                    if int(index) >= removed}
        start_segment = len(segments_out)
        for segment in local_segments:
            segments_out.append(Segment(
                segment.phone, offset + float(segment.start),
                offset + float(segment.end),
                segment.uid, segment.timing_role))
        render_phones_out.extend(local_render_phones)
        for row in local_special_realizations:
            adjusted = dict(row)
            adjusted["index"] = (
                segment_offset + int(adjusted.get("index", -1))
            )
            adjusted["phrase_index"] = phrase_index
            special_phone_realizations.append(adjusted)
        targets.extend((offset + float(time), float(value))
                       for time, value in local_targets)
        generated.extend((offset + float(time), float(value))
                         for time, value in local_generated)
        source_voicing.extend(
            (offset + float(time), float(value)) for time, value in
            local_source_voicing
        )
        generated_voicing.extend(
            (offset + float(time), float(value)) for time, value in
            local_generated_voicing
        )
        voicing_override.extend(
            (offset + float(time), float(value)) for time, value in
            local_voicing_override
        )
        for row in getattr(syn, "voicing_diagnostics", ()):
            adjusted = dict(row)
            adjusted["phrase_index"] = phrase_index
            voicing_diagnostics.append(adjusted)
        target_index_offset = len(target_pitchmarks)
        target_pitchmarks.extend(offset + float(time)
                                 for time in local_pitchmarks)
        for row in local_splices:
            adjusted = dict(row)
            for key in ("time", "handoff_start", "handoff_end"):
                if key in adjusted:
                    adjusted[key] = offset + float(adjusted[key])
            adjusted["segment_index"] = (
                segment_offset + int(adjusted.get("segment_index", -1)))
            splice_records.append(adjusted)
        for row in local_trajectories:
            adjusted = dict(row)
            adjusted["time"] = offset + float(adjusted.get("time") or 0.0)
            if "target_index" in adjusted:
                phrase_target_index = int(adjusted["target_index"])
                adjusted["phrase_target_index"] = phrase_target_index
                adjusted["target_index"] = (
                    target_index_offset + phrase_target_index)
            try:
                local_segment_index = int(
                    adjusted.get("segment_index", -1))
            except (TypeError, ValueError):
                local_segment_index = -1
            if local_segment_index >= 0:
                adjusted["segment_index"] = (
                    segment_offset + local_segment_index)
            adjusted["phrase_index"] = phrase_index
            frame_trajectory_records.append(adjusted)
        selected.update({segment_offset + int(index): str(value)
                         for index, value in selected_units.items()})
        for row in getattr(syn, "vowel_realizations", ()):
            adjusted = dict(row)
            try:
                local_index = int(adjusted.get("segment_index", -1)) - removed
            except (TypeError, ValueError):
                local_index = -1
            if local_index >= 0:
                adjusted["segment_index"] = segment_offset + local_index
            adjusted["phrase_index"] = phrase_index
            vowel_realizations.append(adjusted)
        skipped.extend(syn.skipped)
        if syn.warning:
            warnings.append(syn.warning)
        samples_out.append(samples)
        duration = len(samples) / float(target_sr)
        phrase_ranges.append({
            "phrase": phrase_index, "segment_start": start_segment,
            "segment_end": max(start_segment, len(segments_out) - 1),
            "start": offset, "end": offset + duration,
            "voicebank": syn.voicebank,
        })
        offset += duration
        segment_offset = len(segments_out)

    old_phones = [segment.phone for segment in segments_out]
    old_render_phones = list(render_phones_out)
    entries = [(segment.phone, segment.dur) for segment in segments_out]
    entries = (collapse_pause_runs(entries) if single_pause else
               normalize_internal_pause_runs(entries))
    if [phone for phone, _duration in entries] != old_phones:
        selected = remap_unit_overrides(
            old_phones, [phone for phone, _duration in entries], selected)
    segments_out = segments_from_durations(entries)
    new_phones = [segment.phone for segment in segments_out]
    old_spoken = [
        (index, old_render_phones[index])
        for index, phone in enumerate(old_phones)
        if phone != "pau"
    ]
    new_spoken = [
        index for index, phone in enumerate(new_phones) if phone != "pau"
    ]
    spoken_index_map = {
        old_index: new_index
        for (old_index, _render_phone), new_index in
        zip(old_spoken, new_spoken)
    }
    render_phones_out = list(new_phones)
    for (_old_index, render_phone), new_index in zip(
            old_spoken, new_spoken):
        render_phones_out[new_index] = render_phone
    remapped_special = []
    for row in special_phone_realizations:
        old_index = int(row.get("index", -1))
        if old_index not in spoken_index_map:
            continue
        adjusted = dict(row)
        adjusted["index"] = spoken_index_map[old_index]
        remapped_special.append(adjusted)
    special_phone_realizations = remapped_special
    for record in splice_records:
        when = float(record.get("time") or 0.0)
        if segments_out:
            record["segment_index"] = min(
                range(len(segments_out)),
                key=lambda index: abs(
                    (segments_out[index].start + segments_out[index].end)
                    * 0.5 - when),
            )
        record["sample"] = int(round(when * target_sr))
    for phrase_range in phrase_ranges:
        touched = [index for index, segment in enumerate(segments_out)
                   if (segment.end > float(phrase_range["start"]) and
                       segment.start < float(phrase_range["end"]))]
        if touched:
            phrase_range["segment_start"] = touched[0]
            phrase_range["segment_end"] = touched[-1]
    calibrated_rows = [row for row in calibration_rows
                       if row["calibration"]]
    if len(syntheses) == 1 and calibrated_rows:
        output_calibration = dict(calibrated_rows[0]["calibration"])
        automatic_gain_db = float(calibrated_rows[0][
            "automatic_gain_db"])
        pre_calibration_rms = getattr(
            syntheses[0], "pre_calibration_active_rms", None)
    elif calibrated_rows:
        output_calibration = {
            "schema_version": 1,
            "method": "per_phrase",
            "applied": True,
            "scope": "phrase_sequence",
            "phrase_count": len(syntheses),
            "calibrated_phrase_count": len(calibrated_rows),
            "phrases": calibration_rows,
        }
        automatic_gain_db = 0.0
        pre_calibration_rms = None
    else:
        output_calibration = {}
        automatic_gain_db = 0.0
        pre_calibration_rms = None
    if len(japanese_prosody_rows) == 1:
        japanese_prosody = dict(japanese_prosody_rows[0])
        japanese_prosody.pop("phrase_index", None)
    elif japanese_prosody_rows:
        first = japanese_prosody_rows[0]
        japanese_prosody = {
            "kind": "japanese_phrase_sequence_prosody",
            "duration_model": first.get("duration_model"),
            "duration_model_id": first.get("duration_model_id"),
            "pitch_model_id": first.get("pitch_model_id"),
            "phrases": japanese_prosody_rows,
        }
    else:
        japanese_prosody = {}
    if len(asaxi_prosody_rows) == 1:
        asaxi_prosody = dict(asaxi_prosody_rows[0])
        asaxi_prosody.pop("phrase_index", None)
    elif asaxi_prosody_rows:
        first = asaxi_prosody_rows[0]
        asaxi_prosody = {
            "schema_version": 1,
            "kind": "asaxi_phrase_sequence_prosody",
            "dictionary_ruleset": first.get("dictionary_ruleset", ""),
            "phrase_count": sum(
                int(row.get("phrase_count", 0))
                for row in asaxi_prosody_rows
            ),
            "word_count": sum(
                int(row.get("word_count", 0))
                for row in asaxi_prosody_rows
            ),
            "mora_count": sum(
                int(row.get("mora_count", 0))
                for row in asaxi_prosody_rows
            ),
            "phrases": asaxi_prosody_rows,
        }
        flattened_moras = []
        for sequence_index, metadata in enumerate(asaxi_prosody_rows):
            if sequence_index >= len(phrase_ranges):
                break
            phrase_range = phrase_ranges[sequence_index]
            first = int(phrase_range["segment_start"])
            last = int(phrase_range["segment_end"])
            final_indexes = list(range(first, last + 1))
            final_phones = [
                str(segments_out[index].phone) for index in final_indexes
            ]
            old_phones = [
                str(phone) for phone in metadata.get(
                    "rendered_phones") or []
            ]
            index_map = {}
            for block in SequenceMatcher(
                    None, old_phones, final_phones,
                    autojunk=False).get_matching_blocks():
                for local_offset in range(block.size):
                    index_map[block.a + local_offset] = \
                        final_indexes[block.b + local_offset]
            for local_mora in metadata.get("moras") or []:
                if not isinstance(local_mora, Mapping):
                    continue
                old_indices = [
                    int(index) for index in
                    local_mora.get("segment_indices") or []
                ]
                if not old_indices or not all(
                        index in index_map for index in old_indices):
                    continue
                row = dict(local_mora)
                new_indices = [index_map[index] for index in old_indices]
                row["mora_index"] = len(flattened_moras)
                row["phrase_sequence_index"] = sequence_index
                row["segment_indices"] = new_indices
                row["start"] = min(
                    float(segments_out[index].start)
                    for index in new_indices)
                row["end"] = max(
                    float(segments_out[index].end)
                    for index in new_indices)
                flattened_moras.append(row)
        asaxi_prosody["moras"] = flattened_moras
        asaxi_prosody["rendered_phones"] = new_phones
    else:
        asaxi_prosody = {}
    join_settings = (
        dict(join_settings_rows[0])
        if join_settings_rows and
        all(row == join_settings_rows[0] for row in join_settings_rows)
        else {
            "scope": "per_phrase",
            "phrases": join_settings_rows,
        } if join_settings_rows else {}
    )
    return Synthesis(
        np.concatenate(samples_out).astype(np.float32), target_sr,
        segments_out, text=text, lang=lang,
        voicebank=" + ".join(dict.fromkeys(
            syn.voicebank for syn in syntheses if syn.voicebank)),
        phones=[phone for syn in syntheses for phone in syn.phones],
        render_phones=render_phones_out,
        special_phone_realizations=special_phone_realizations,
        diphones=[dip for syn in syntheses for dip in syn.diphones],
        skipped=skipped, targets=sorted(targets),
        generated_targets=sorted(generated), selected_units=selected,
        phrase_ranges=phrase_ranges,
        target_pitchmarks=sorted(target_pitchmarks),
        splice_records=splice_records,
        frame_trajectory_records=frame_trajectory_records,
        join_settings=join_settings,
        vowel_realizations=vowel_realizations,
        source_voicing_targets=sorted(source_voicing),
        generated_voicing_targets=sorted(generated_voicing),
        voicing_override=sorted(voicing_override),
        voicing_mode=("curve" if voicing_override else ""),
        voicing_diagnostics=voicing_diagnostics,
        japanese_prosody=japanese_prosody,
        asaxi_prosody=asaxi_prosody,
        automatic_gain_db=automatic_gain_db,
        pre_calibration_active_rms=pre_calibration_rms,
        output_calibration=output_calibration,
        warning="; ".join(dict.fromkeys(warnings)) or None)


def phrase_blocks(segments, text: str = "", kinds=None):
    """Return voiced phrase spans separated by one or more internal pauses."""
    import re as _re
    spans = []
    start = end = None
    for seg in segments:
        if hasattr(seg, "phone"):
            phone, a, b = str(seg.phone), float(seg.start), float(seg.end)
        else:
            phone = str(seg[0]) if seg else ""
            a = float(seg[1]) if len(seg) > 1 else 0.0
            b = float(seg[2]) if len(seg) > 2 else a
        if phone == "pau":
            if start is not None:
                spans.append((start, end))
                start = end = None
            continue
        if start is None:
            start = a
        end = b
    if start is not None:
        spans.append((start, end))

    supplied = list(kinds or [])
    punctuation_map = {
        "。": ".", "．": ".",
        "？": "?", "！": "!",
        "、": ",", "，": ",",
        "：": ":", "；": ";",
    }
    punctuation = [
        punctuation_map.get(mark, mark)
        for mark in _re.findall(r"[.?!,:;。．？！、，：；]", text or "")
    ]
    blocks = []
    for i, (a, b) in enumerate(spans):
        if i < len(supplied):
            kind = str(supplied[i])
        elif i < len(punctuation):
            kind = punctuation[i]
        else:
            kind = "." if i == len(spans) - 1 else ","
        blocks.append({"start": a, "end": b, "kind": kind})
    return blocks


def phrase_playback_spans(segments, phrase_weights):
    """Partition rendered segments into complete logical phrase playback.

    A Festival frontend may insert more acoustic pause-separated spans than
    the text contains punctuation phrases.  Pairing those two lists with
    ``zip`` drops the unmatched tail.  This partition instead groups adjacent
    acoustic spans by the logical phrase weights and assigns every segment to
    exactly one phrase.  At a canonical four-pause boundary the first two
    pauses belong to the outgoing phrase and the final two belong to the
    incoming phrase.  Legacy shorter runs are divided conservatively so a lone
    pause, which may contain the next onset, stays with the incoming phrase.
    """
    rows = list(segments or [])
    weights = []
    for value in phrase_weights or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1.0
        weights.append(max(1.0e-6, number))
    if not rows or not weights:
        return []

    def is_pause(index):
        return str(rows[index].phone) == "pau"

    def duration(index):
        return max(1.0e-6, float(rows[index].end) -
                   float(rows[index].start))

    def weighted_groups(masses, group_count):
        item_count = len(masses)
        if group_count <= 1:
            return [(0, item_count)]
        if item_count < group_count:
            return []
        prefix = [0.0]
        for mass in masses:
            prefix.append(prefix[-1] + max(1.0e-6, float(mass)))
        mass_total = prefix[-1]
        weight_total = sum(weights[:group_count])
        cuts = [0]
        previous = 0
        for group_index in range(1, group_count):
            minimum = previous + 1
            maximum = item_count - (group_count - group_index)
            target = (sum(weights[:group_index]) / weight_total)
            cut = min(
                range(minimum, maximum + 1),
                key=lambda candidate: (
                    abs(prefix[candidate] / mass_total - target),
                    candidate,
                ),
            )
            cuts.append(cut)
            previous = cut
        cuts.append(item_count)
        return list(zip(cuts, cuts[1:]))

    voiced_runs = []
    run_start = None
    for index in range(len(rows)):
        if not is_pause(index):
            if run_start is None:
                run_start = index
        elif run_start is not None:
            voiced_runs.append((run_start, index - 1))
            run_start = None
    if run_start is not None:
        voiced_runs.append((run_start, len(rows) - 1))

    phrase_count = len(weights)
    playback_groups = []
    if len(voiced_runs) >= phrase_count:
        playback_boundaries = []
        for run_index in range(len(voiced_runs) - 1):
            gap_start = voiced_runs[run_index][1] + 1
            next_spoken_start = voiced_runs[run_index + 1][0]
            gap_count = max(0, next_spoken_start - gap_start)
            if gap_count <= 1:
                outgoing_count = 0
            elif gap_count == 2:
                outgoing_count = 1
            else:
                outgoing_count = 2
            playback_boundaries.append(gap_start + outgoing_count)
        chunks = []
        for index, (spoken_start, spoken_end) in enumerate(voiced_runs):
            playback_start = (
                0 if index == 0 else playback_boundaries[index - 1]
            )
            playback_end = (
                playback_boundaries[index] - 1
                if index < len(playback_boundaries) else len(rows) - 1
            )
            mass = sum(duration(item) for item in
                       range(spoken_start, spoken_end + 1))
            chunks.append({
                "start_index": playback_start,
                "end_index": playback_end,
                "spoken_start_index": spoken_start,
                "spoken_end_index": spoken_end,
                "mass": mass,
            })
        groups = weighted_groups(
            [chunk["mass"] for chunk in chunks], phrase_count)
        for first, last in groups:
            playback_groups.append({
                "start_index": chunks[first]["start_index"],
                "end_index": chunks[last - 1]["end_index"],
                "spoken_start_index": chunks[first][
                    "spoken_start_index"],
                "spoken_end_index": chunks[last - 1][
                    "spoken_end_index"],
            })
    else:
        # A stale project can contain more logical phrases than acoustic pause
        # spans.  Segment-boundary partitioning still preserves all samples.
        groups = weighted_groups(
            [duration(index) for index in range(len(rows))], phrase_count)
        for first, last in groups:
            spoken = [index for index in range(first, last)
                      if not is_pause(index)]
            playback_groups.append({
                "start_index": first,
                "end_index": last - 1,
                "spoken_start_index": spoken[0] if spoken else None,
                "spoken_end_index": spoken[-1] if spoken else None,
            })

    result = []
    for group in playback_groups:
        first = int(group["start_index"])
        last = int(group["end_index"])
        row = dict(group)
        row["start"] = float(rows[first].start)
        row["end"] = float(rows[last].end)
        spoken_first = row.get("spoken_start_index")
        spoken_last = row.get("spoken_end_index")
        row["spoken_start"] = (
            float(rows[spoken_first].start)
            if spoken_first is not None else row["start"])
        row["spoken_end"] = (
            float(rows[spoken_last].end)
            if spoken_last is not None else row["end"])
        result.append(row)
    return result


# ------------------------------------------------- synth_diphone import glue
_SD = None
_SD_PATH = None


def import_synth_diphone(cfg: dict):
    """Import synth_diphone.py by file path. Search order: config key
    'synth_diphone_dir', then the bundled FestVox copy."""
    global _SD, _SD_PATH
    here = Path(GUI_DIR)
    cands = []
    if cfg.get("synth_diphone_dir"):
        cands.append(Path(cfg["synth_diphone_dir"]))
    cands += [here.parent, here]
    for d in cands:
        p = Path(d) / "synth_diphone.py"
        if p.is_file():
            if _SD is not None and _SD_PATH == str(p):
                return _SD
            spec = importlib.util.spec_from_file_location("synth_diphone", str(p))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["synth_diphone"] = mod   # so its own imports resolve
            spec.loader.exec_module(mod)
            _SD, _SD_PATH = mod, str(p)
            return mod
    raise BackendError(
        "synth_diphone.py not found.\n\nLooked in:\n  " +
        "\n  ".join(str(Path(d)) for d in cands) +
        "\n\nThe bundled renderer should be at 99_Tools/festvox/"
        "synth_diphone.py. Reinstall that file, or set \"synth_diphone_dir\" "
        "to an explicit developer override.")


# ------------------------------------------------------------- diphone backend
class DiphoneBackend:
    """Drives synth_diphone.py: voice inventory from festvox.json, text or
    phone-list synthesis with real per-phone segment timing."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sd = import_synth_diphone(cfg)
        self.fcfg: dict = {}
        self.fcfg_path: Optional[str] = None
        self._dbs = {}                          # voicebank name -> DiphoneDB
        self._db_fingerprints = {}
        try:
            sustain_entries = max(1, int(cfg.get(
                "sustain_cache_entries", 64)))
            sustain_bytes = max(1, int(cfg.get(
                "sustain_cache_mib", 32))) * 1024 * 1024
        except (TypeError, ValueError):
            sustain_entries, sustain_bytes = 64, 32 * 1024 * 1024
        self._sustains = BoundedMemoryCache(
            "diphone-sustain-audio", max_entries=sustain_entries,
            max_bytes=sustain_bytes, size_func=_sustain_cache_size)
        self._cache_lock = threading.RLock()
        try:
            render_parameters = inspect.signature(
                self.sd.render).parameters
            self._direct_pcm_render = "return_pcm" in render_parameters
            self._render_supports_legacy_joins = (
                "legacy_joins" in render_parameters)
        except (TypeError, ValueError):
            self._direct_pcm_render = False
            self._render_supports_legacy_joins = False
        self.reload_festvox_config()

    # -- festvox.json ---------------------------------------------------------
    def reload_festvox_config(self):
        with self._cache_lock:
            self._clear_databases()
            self._sustains.clear()
            explicit = self.cfg.get("festvox_config") or None
            p = self.sd._find_festvox_config(explicit)
            self.fcfg, self.fcfg_path = {}, None
            if p:
                try:
                    self.fcfg = json.loads(
                        Path(p).read_text(encoding="utf-8"))
                    self.fcfg_path = str(p)
                except (OSError, json.JSONDecodeError) as e:
                    raise BackendError(
                        f"festvox.json unreadable ({p}):\n{e}")

    # -- voice inventory ------------------------------------------------------
    def _voice_dir(self, name: str) -> Optional[Path]:
        voices = self.fcfg.get("voices") or {}
        if name in voices:
            d, _ = self.sd._db_from_config(self.fcfg, name)
            return d
        extra = self.cfg.get("extra_voicebanks") or {}
        if name in extra:
            return Path(extra[name])
        return None

    def voicebanks(self) -> List[dict]:
        """[{name, dir, ok, source}] from festvox.json voices + GUI extras."""
        out, seen = [], set()
        for name in (self.fcfg.get("voices") or {}):
            d = self._voice_dir(name)
            out.append(self._vb_info(name, d, "festvox.json"))
            seen.add(name)
        for name, d in (self.cfg.get("extra_voicebanks") or {}).items():
            if name not in seen:
                out.append(self._vb_info(name, Path(d), "config.json"))
        return out

    @staticmethod
    def _vb_info(name, d, source) -> dict:
        ok = bool(d) and (Path(d) / "dic" / "diphone_index.json").is_file()
        return {"name": name, "dir": str(d) if d else "", "ok": ok,
                "source": source}

    def default_voicebank(self) -> Optional[str]:
        vbs = self.voicebanks()
        want = self.fcfg.get("default_voice")
        for v in vbs:
            if v["name"] == want and v["ok"]:
                return want
        for v in vbs:
            if v["ok"]:
                return v["name"]
        return vbs[0]["name"] if vbs else None

    def add_voicebank_dir(self, path: str) -> str:
        """Register a DB folder chosen in the GUI. Returns its name."""
        p = Path(path)
        if not (p / "dic" / "diphone_index.json").is_file():
            raise BackendError(
                f"Not a diphone DB:\n{p}\n\nExpected dic/diphone_index.json "
                "inside (a folder built by utau2festvox.py).")
        name = p.name or "voicebank"
        self.cfg.setdefault("extra_voicebanks", {})[name] = str(p)
        return name

    def unit_alternatives(self, voicebank: str) -> dict:
        try:
            return dict(self.db(voicebank).alternatives)
        except (BackendError, OSError, ValueError, TypeError):
            return {}

    def voice_pitch_hz(self, voicebank: str):
        """Measured bank pitch when available; pure diphone has no fallback."""
        try:
            database = self.db(voicebank)
            value = metadata_voice_pitch_hz(database.metadata)
            if value is not None:
                return value
        except (BackendError, OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def sustain_sample(self, phone: str, voicebank: str):
        key = (str(voicebank), str(phone).rstrip("_"))
        with self._cache_lock:
            if key in self._sustains:
                return self._sustains[key]
        db = self.db(voicebank)
        dip = "%s-%s" % (key[1], key[1])
        if not db.has(dip):
            with self._cache_lock:
                self._sustains[key] = None
            return None
        try:
            sr, chunk, _mid = db.slice_info(
                dip, half_ms=5000.0, copy_samples=False)
            samples = (np.asarray(chunk, dtype=np.int16).astype(np.float32)
                       / 32768.0)
            value = (samples, int(sr)) if samples.size else None
        except (OSError, KeyError, ValueError):
            value = None
        with self._cache_lock:
            self._sustains[key] = value
        return value

    def install_dictionary(self, voicebank: str, source_name: str,
                           entries: dict) -> str:
        root = self._voice_dir(voicebank)
        if root is None or not root.is_dir():
            raise BackendError("The selected diphone voicebank folder is "
                               "missing or not path-backed.")
        folder = root / "dic"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / cleaned_dictionary_filename(source_name)
        path.write_text(cleaned_dictionary_text(entries), encoding="utf-8")
        return str(path)

    def install_voice_icon(self, voicebank: str, source: str) -> str:
        root = self._voice_dir(voicebank)
        if root is None or not root.is_dir():
            raise BackendError("The selected voicebank has no generated folder.")
        suffix = Path(source).suffix.lower()
        if suffix not in VOICE_ICON_SUFFIXES:
            suffix = ".png"
        target = root / ("speaker" + suffix)
        remove_known_voice_icons(root, keep=target)
        import shutil
        shutil.copy2(str(source), str(target))
        return str(target)

    def remove_voice_icon(self, voicebank: str):
        root = self._voice_dir(voicebank)
        if root is None or not root.is_dir():
            raise BackendError("The selected voicebank has no generated folder.")
        return remove_known_voice_icons(root)

    @staticmethod
    def read_installed_dictionary(path: str) -> dict:
        return parse_cleaned_dictionary_text(
            Path(path).read_text(encoding="utf-8", errors="replace"))

    def voicebank_removal_info(self, name: str) -> dict:
        d = self._voice_dir(name)
        if not d:
            raise BackendError(f"Voicebank '{name}' has no deletable folder.")
        p = Path(d).expanduser().resolve()
        exists = p.exists()
        if exists:
            p = validate_generated_voice_dir(p)
        return {"name": name, "path": str(p), "kind": "windows",
                "exists": exists}

    def uninstall_voicebank(self, name: str, delete_files: bool = True) -> str:
        info = self.voicebank_removal_info(name)
        if delete_files and info["exists"]:
            delete_generated_voice_dir(info["path"])
        self._drop_database(name)
        extras = self.cfg.get("extra_voicebanks") or {}
        if name in extras:
            extras.pop(name, None)
        elif name in (self.fcfg.get("voices") or {}):
            self.fcfg["voices"].pop(name, None)
            if self.fcfg.get("default_voice") == name:
                self.fcfg["default_voice"] = ""
            if self.fcfg_path:
                Path(self.fcfg_path).write_text(
                    json.dumps(self.fcfg, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
        return info["path"]

    def _voice_fingerprint(self, name: str):
        root = self._voice_dir(name)
        if root is None:
            return None
        path = Path(root) / "dic" / "diphone_index.json"
        try:
            stat = path.stat()
        except OSError:
            return None
        return (
            str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size),
            int(getattr(stat, "st_dev", 0)), int(getattr(stat, "st_ino", 0)),
            file_change_token(path, stat),
        )

    def refresh_voice_metadata(self, name: str):
        """Invalidate one in-memory DB only when its index file changed."""
        with self._cache_lock:
            token = self._voice_fingerprint(name)
            if (name in self._dbs and
                    self._db_fingerprints.get(name) != token):
                self._drop_database(name)
            return token

    def _drop_database(self, name: str):
        if not hasattr(self, "_cache_lock"):
            self._cache_lock = threading.RLock()
        if not hasattr(self, "_db_fingerprints"):
            self._db_fingerprints = {}
        with self._cache_lock:
            database = self._dbs.pop(name, None)
            self._db_fingerprints.pop(name, None)
            if database is not None and hasattr(database, "clear_cache"):
                database.clear_cache()
            for key in [key for key in self._sustains if key[0] == name]:
                self._sustains.pop(key, None)

    def _clear_databases(self):
        with self._cache_lock:
            for name in list(self._dbs):
                self._drop_database(name)

    def db(self, voicebank: str):
        with self._cache_lock:
            fingerprint = self._voice_fingerprint(voicebank)
            if (voicebank in self._dbs and
                    self._db_fingerprints.get(voicebank) != fingerprint):
                self._drop_database(voicebank)
            if voicebank in self._dbs:
                cached = self._dbs.pop(voicebank)
                self._dbs[voicebank] = cached
                return cached
            d = self._voice_dir(voicebank)
            if not d or not (Path(d) / "dic" /
                             "diphone_index.json").is_file():
                raise BackendError(
                    f"Voicebank '{voicebank}' has no diphone DB at:\n"
                    f"  {d or '(no path)'}\n\n"
                    "Fix the path in festvox.json (voices/output_root), or "
                    "use Voicebank > Add voicebank folder... to point at a "
                    "DB built by utau2festvox.py.")
            try:
                cache_files = max(0, int(self.cfg.get(
                    "diphone_wav_cache_files", 64)))
                cache_bytes = max(0, int(self.cfg.get(
                    "diphone_wav_cache_mib", 64))) * 1024 * 1024
                slice_entries = max(0, int(self.cfg.get(
                    "diphone_slice_cache_entries", 512)))
                slice_bytes = max(0, int(self.cfg.get(
                    "diphone_slice_cache_mib", 32))) * 1024 * 1024
            except (TypeError, ValueError):
                cache_files, cache_bytes = 64, 64 * 1024 * 1024
                slice_entries, slice_bytes = 512, 32 * 1024 * 1024
            self._dbs[voicebank] = self.sd.DiphoneDB(
                Path(d), cache_max_files=cache_files,
                cache_max_bytes=cache_bytes,
                slice_cache_max_entries=slice_entries,
                slice_cache_max_bytes=slice_bytes)
            self._db_fingerprints[voicebank] = fingerprint
            try:
                limit = max(1, int(self.cfg.get(
                    "diphone_voice_cache_limit", 2)))
            except (TypeError, ValueError):
                limit = 2
            while len(self._dbs) > limit:
                self._drop_database(next(iter(self._dbs)))
            return self._dbs[voicebank]

    def db_size(self, voicebank: str) -> int:
        try:
            return len(self.db(voicebank).index)
        except Exception:
            return 0

    def cache_info(self) -> dict:
        """Return owned in-memory cache usage by user-facing category."""
        with self._cache_lock:
            db_rows = [database.cache_info()
                       for database in self._dbs.values()]
            sustain_bytes = sum(
                int(value[0].nbytes)
                for value in self._sustains.values()
                if isinstance(value, tuple) and len(value) == 2 and
                hasattr(value[0], "nbytes")
            )
            sustain_info = (self._sustains.info()
                            if hasattr(self._sustains, "info") else {})
            audio_bytes = sustain_bytes + sum(
                int(row.get("total_bytes", row.get("bytes", 0)))
                for row in db_rows
            )
            voice_bytes = sum(int(getattr(database, "metadata_bytes", 0))
                              for database in self._dbs.values())
            model = (self.sd.model_cache_info()
                     if hasattr(self.sd, "model_cache_info") else
                     {"entries": 0, "bytes": 0})
            return {
                "audio": {
                    "bytes": audio_bytes,
                    "decoded_files": sum(int(row.get("files", 0))
                                         for row in db_rows),
                    "slices": sum(int(row.get("slices", 0))
                                  for row in db_rows),
                    "sustains": len(self._sustains),
                    "max_sustains": int(sustain_info.get(
                        "max_entries", 0)),
                    "max_sustain_bytes": int(sustain_info.get(
                        "max_bytes", 0)),
                },
                "voice": {
                    "bytes": voice_bytes,
                    "voices": len(self._dbs),
                    "index_entries": sum(len(database.index)
                                         for database in self._dbs.values()),
                },
                "model": dict(model),
                "owners": {name: dict(row) for name, row in
                           zip(self._dbs, db_rows)},
            }

    def clear_application_cache(self, category: str) -> dict:
        """Clear only reproducible memory owned by this backend."""
        category = str(category).casefold()
        if category not in {"audio", "voice", "model", "all"}:
            raise ValueError("cache category must be audio, voice, model, or all")
        before = self.cache_info()
        with self._cache_lock:
            if category in {"voice", "all"}:
                self._clear_databases()
                self._sustains.clear()
            elif category == "audio":
                for database in self._dbs.values():
                    database.clear_cache()
                self._sustains.clear()
            if category in {"model", "all"} and hasattr(
                    self.sd, "clear_model_cache"):
                self.sd.clear_model_cache()
        return {"before": before, "after": self.cache_info()}

    # -- g2p ------------------------------------------------------------------
    def g2p(self, text: str, lang: str) -> List[str]:
        try:
            if lang == "en":
                return self.sd.g2p_english(text)
            if lang in ("ja", "jp"):
                return self.sd.g2p_japanese(text)
            return self.sd.g2p_asaxi(text)
        except (RuntimeError, ValueError) as e:
            raise BackendError(str(e))

    # -- synthesis ------------------------------------------------------------
    def g2p_dict(self, text: str, lang: str, user_dict=None):
        """g2p with per-word overrides from a loaded user dictionary; words not
        in the dict fall back to the normal rule g2p."""
        if not user_dict:
            return self.g2p(text, lang)
        import re as _re
        out = []
        for w in _re.findall(r"\S+", text):
            key = w.lower().strip(".,!?;:'\"()")
            out.extend(user_dict[key] if key in user_dict
                       else self.g2p(w, lang))
        return out

    def synth(self, text: str, lang: str, voicebank: str,
              speed: float = 1.0, pitch=None, fall=None,
              monotone: bool = False, user_dict=None,
              fault_mode=None, pitch_targets=None,
              ground_truth_targets=None, intonation_blocks=None,
              pitch_mode: str = "") -> Synthesis:
        if not text or not text.strip():
            raise BackendError("No text to synthesize.")
        text = normalize_synthesis_text(text, lang)
        chunks = text_phrase_chunks(text)
        if len(chunks) > 1:
            phones = []
            for i, (chunk, _mark) in enumerate(chunks):
                if i:
                    phones.extend(["pau"] if (fault_mode or {}).get(
                        "single_pause") else
                        ["pau", "pau", "pau", "pau"])
                phones.extend(self.g2p_dict(chunk, lang, user_dict))
        else:
            phones = self.g2p_dict(text, lang, user_dict)
        if (fault_mode or {}).get("single_pause"):
            collapsed = []
            for phone in phones:
                if phone == "pau" and collapsed and collapsed[-1] == "pau":
                    continue
                collapsed.append(phone)
            phones = collapsed
        if not phones:
            raise BackendError(
                f"No phonemes derived from {text!r} for language '{lang}'.")
        return self.synth_phones(
            phones, voicebank, speed, text=text, lang=lang,
            fault_mode=fault_mode)

    def synth_phones(self, phones: List[str], voicebank: str,
                     speed: float = 1.0, text: str = "",
                     lang: str = "", seg_durs=None, old_segments=None,
                     prev_targets=None, pitch=None, fall=None,
                     monotone: bool = False, fault_mode=None,
                     pitch_targets=None, ground_truth_targets=None,
                     intonation_blocks=None, pitch_mode: str = "",
                     unit_overrides=None,
                     preserve_pitch_register: bool = False) -> Synthesis:
        """Render an explicit phone list -- the path used when the user edits
        the phoneme fields (e.g. overrides 'r' to 'rr') and re-renders.
        (seg_durs/targets/pitch are accepted for interface parity with the
        Festival backend; concatenative rendering has no PSOLA, so it always
        plays the bank at its recorded pitch and re-times freshly.)"""
        phones = [str(p).strip() for p in phones if str(p).strip()]
        # strip EDGE paus only (render() adds its own); interior "pau" is a
        # legitimate user-inserted pause if the bank has x-pau / pau-y units
        while phones and phones[0] == "pau":
            phones.pop(0)
        while phones and phones[-1] == "pau":
            phones.pop()
        if not phones:
            raise BackendError("Phone list is empty.")
        db = self.db(voicebank)
        resolution = resolve_voice_special_phones(
            phones,
            getattr(db, "metadata", {}),
            voicebank=voicebank,
            available_diphones=getattr(db, "index", {}).keys(),
        )
        render_phones = list(resolution.render_phones)
        adv = self.cfg.get("advanced") or {}
        legacy_joins = bool((fault_mode or {}).get("legacy_joins"))
        join_options = ({"legacy_joins": legacy_joins}
                        if self._render_supports_legacy_joins else {})
        try:
            if self._direct_pcm_render:
                r = self.sd.render(
                    db, render_phones, speed=speed,
                    unit_overrides=unit_overrides,
                    return_pcm=True,
                    encode_wav=False,
                    crossfade_ms=float(adv.get("crossfade_ms", 15.0)),
                    edge_fade_ms=float(adv.get("edge_fade_ms", 8.0)),
                    half_ms=float(adv.get("half_ms", 150.0)),
                    **join_options,
                )
            else:
                # Compatibility for an explicitly configured legacy renderer.
                # The bundled renderer uses per-call values and never mutates
                # process-global synthesis settings.
                with self._cache_lock:
                    self.sd.CROSSFADE_MS = float(
                        adv.get("crossfade_ms", 15.0))
                    self.sd.EDGE_FADE_MS = float(
                        adv.get("edge_fade_ms", 8.0))
                    self.sd.HALF_MS = float(adv.get("half_ms", 150.0))
                    r = self.sd.render(
                        db, render_phones, speed=speed,
                        unit_overrides=unit_overrides,
                        **join_options)
        except ValueError as e:
            raise BackendError(
                f"{e}\n\nThe voicebank has none of the needed diphones. "
                "Check the phoneme spelling against the bank's phone set.")
        if r.get("pcm16") is not None:
            samples = (np.asarray(r["pcm16"], dtype=np.int16)
                       .astype(np.float32) / 32768.0)
            sr = int(r["framerate"])
        else:
            samples, sr = wav_bytes_to_samples(r["wav"])
        segs = [Segment(s["phone"], s["start"], s["end"])
                for s in r.get("segments", [])]
        warn = None
        if r["skipped"]:
            warn = ("missing diphones skipped: " + ", ".join(r["skipped"]))
        invalid_joins = [
            row for row in r.get("splice_records", ())
            if row.get("validation_passed") is False
        ]
        if invalid_joins:
            failure_names = sorted({
                str(name)
                for row in invalid_joins
                for name in (row.get("validation_failures") or ())
                if str(name)
            })
            detail = (" (" + ", ".join(failure_names) + ")"
                      if failure_names else "")
            note = ("%d measured join%s failed acoustic validation%s; "
                    "inspect join diagnostics or compare Legacy joins" %
                    (len(invalid_joins),
                     "" if len(invalid_joins) == 1 else "s", detail))
            warn = (warn + "; " + note) if warn else note
        result = Synthesis(
            samples, sr, segs, text=text, lang=lang,
            voicebank=voicebank, phones=list(phones),
            diphones=list(r["diphones"]),
            skipped=list(r["skipped"]),
            unit_overrides={int(k): str(v) for k, v in
                            dict(unit_overrides or {}).items()},
            selected_units={int(k): str(v) for k, v in
                            dict(r.get("selected_units") or {}).items()},
            splice_records=[dict(row) for row in
                            (r.get("splice_records") or [])],
            warning=warn,
        )
        if resolution.realizations:
            actual = [segment.phone for segment in result.segments]
            render_with_edges = ["pau"] + render_phones + ["pau"]
            display_with_edges = ["pau"] + phones + ["pau"]
            if actual == render_with_edges:
                edge_realizations = []
                for row in resolution.realizations:
                    data = row.to_dict()
                    data["index"] = int(data["index"]) + 1
                    edge_realizations.append(data)
                return apply_special_phone_display(
                    result, display_with_edges, render_with_edges,
                    edge_realizations,
                )
            if actual == render_phones:
                return apply_special_phone_display(
                    result, phones, render_phones, resolution.realizations,
                )
            if any(
                row.source_phone != row.phone
                for row in resolution.realizations
            ):
                raise BackendError(
                    "The Python renderer returned a Segment layout that "
                    "cannot be aligned with the canonical special-phone "
                    "sequence. Rendering was stopped so source phones cannot "
                    "replace editable cl regions."
                )
            result.special_phone_realizations = [
                row.to_dict() for row in resolution.realizations
            ]
            result.render_phones = actual
            note = (
                "special-phone source rendering completed, but the Python "
                "renderer returned a nonstandard Segment layout"
            )
            result.warning = (
                result.warning + "; " + note if result.warning else note
            )
        return result

    def synth_output_dir(self) -> str:
        return str(self.fcfg.get("synth_output_dir") or "")

    def default_speed(self) -> float:
        try:
            return float(self.cfg.get("synth_speed")
                         or self.fcfg.get("synth_speed", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def default_lang_code(self) -> str:
        return str(self.fcfg.get("default_lang") or "asaxi")


# ---------------------------------------------------- Festival via WSL backend
# Real Festival, launched through wsl.exe — this is what gives the GUI
# MULTISYN capability: unit-selection voices built with the FestVox multisyn
# recipes (see 99_Tools/festvox/MULTISYN.md) are ordinary Festival voices and
# cannot run in the pure-Python engine.

_WIN_DRIVE_RE = None  # compiled lazily (re imported at module top? -> local)


def win_to_wsl_path(p: str) -> str:
    """Translate a Windows path to its WSL /mnt/<drive>/ form. WSL/POSIX
    paths pass through unchanged."""
    return windows_to_wsl_path(p)


def parse_segs(path: str) -> List[Segment]:
    """Parse a Festival/Xwaves label file (from utt.save.segs)."""
    segs: List[Segment] = []
    prev = 0.0
    started = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not started:
                if line.strip() == "#":
                    started = True
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                end = float(parts[0])
            except ValueError:
                continue
            phone = parts[-1]
            segs.append(Segment(phone, prev, end))
            prev = end
    return segs


def festival_unit_override_scheme(unit_overrides, entry_count: int) -> str:
    """Pass per-segment unit choices to the UniSyn hook.

    A Segments utterance does not own its Segment relation until synthesis has
    begun, so setting item features before ``utt.synth`` fails in Festival.
    The generated voice hook consumes this process-local indexed list once the
    relation exists.
    """
    import re as _re
    rows = []
    for raw_idx, raw_name in dict(unit_overrides or {}).items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        unit_name = str(raw_name)
        if (idx < 0 or idx >= entry_count or
                not _re.fullmatch(r"[A-Za-z0-9_]+", unit_name)):
            continue
        rows.append('(\"%d\" \"%s\")' % (idx, unit_name))
    if not rows:
        return ""
    return "(set! festvox_gui_unit_variant_overrides '(%s))\n" % " ".join(rows)


class FestivalWSLBackend:
    """Runs the real `festival` binary inside WSL. Voices:
      * "installed" names known to Festival ((voice.list) / site init), and
      * festvox voice DIRECTORIES (multisyn or other) — the folder is put on
        Festival's load-path and its festvox/*.scm loaded, so voices need not
        be installed system-wide. Dirs may be Windows paths (E:\\... ->
        /mnt/e/...) or WSL paths (/home/you/voices/...)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        try:
            voice_entries = max(1, int(cfg.get(
                "festival_voice_cache_limit", 8)))
            voice_bytes = max(2, int(cfg.get(
                "festival_voice_cache_mib", 64))) * 1024 * 1024
            sustain_entries = max(1, int(cfg.get(
                "sustain_cache_entries", 64)))
            sustain_bytes = max(1, int(cfg.get(
                "sustain_cache_mib", 32))) * 1024 * 1024
        except (TypeError, ValueError):
            voice_entries, voice_bytes = 8, 64 * 1024 * 1024
            sustain_entries, sustain_bytes = 64, 32 * 1024 * 1024
        # Runtime metadata is deliberately compact. Contextual alternatives
        # retain richer per-take provenance and therefore receive most of the
        # same total voice-cache budget. A 64 MiB default holds one current
        # integrated Lem inventory without increasing application memory.
        metadata_bytes = max(1, voice_bytes // 4)
        alternatives_bytes = max(1, voice_bytes - metadata_bytes)
        self._alternatives = BoundedMemoryCache(
            "festival-unit-alternatives", max_entries=voice_entries,
            max_bytes=alternatives_bytes, size_func=_voice_metadata_size)
        self._sustains = BoundedMemoryCache(
            "festival-sustain-audio", max_entries=sustain_entries,
            max_bytes=sustain_bytes, size_func=_sustain_cache_size)
        self._voice_metadata = BoundedMemoryCache(
            "festival-voice-metadata", max_entries=voice_entries,
            max_bytes=metadata_bytes, size_func=_voice_metadata_size)
        self._voice_fingerprints = {}
        self._cache_epoch = 0
        self._cache_lock = threading.RLock()
        self._native_server_lock = threading.RLock()
        self._native_server_process = None
        self._native_server_queue = None
        self._native_server_reader = None
        self._native_server_jobs = 0
        self._native_server_binary_stamp = None

    # -- config access ---------------------------------------------------------
    def fcfg(self) -> dict:
        return self.cfg.setdefault("festival_wsl", {})

    def generated_voice_root(self) -> str:
        raw = self.fcfg().get("generated_voice_root") \
            or DEFAULT_GENERATED_VOICE_ROOT
        try:
            root = canonical_windows_path(raw)
        except ValueError:
            root = DEFAULT_GENERATED_VOICE_ROOT
        self.fcfg()["generated_voice_root"] = root
        return root

    def generated_voice_wsl_root(self) -> str:
        raw = str(self.fcfg().get("generated_voice_wsl_root") or "").strip()
        if not raw:
            return ""
        if not raw.startswith("/") or raw == "/":
            raise BackendError(
                "The WSL voice scan root must be an absolute folder below /.")
        root = raw.rstrip("/")
        self.fcfg()["generated_voice_wsl_root"] = root
        return root

    @staticmethod
    def _has_festival_scheme(path: Path) -> bool:
        return any((path / "festvox").glob("*.scm")) or any(path.glob("*.scm"))

    def _install_windows_kal(self) -> tuple[Optional[dict], str]:
        """Mirror the packaged Kal voice into the configured Windows root."""
        root = Path(self.generated_voice_root())
        target = root / "kal_diphone"
        if target.exists() and self._has_festival_scheme(target):
            return self.scan_voice_dir(str(target)), ""
        if target.exists() and any(target.iterdir()):
            return None, ("Kal target exists but is not a Festival voice: " +
                          str(target))
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return None, "Could not create the Windows Kal folder: %s" % exc
        source = BUILTIN_FESTIVAL_VOICES["kal_diphone"]
        runtime_target = win_to_wsl_path(str(target)).rstrip("/")
        try:
            proc = self._run([
                "cp", "-a", "--", source + "/.", runtime_target + "/"
            ], timeout=max(60.0, self._timeout()))
        except BackendError as exc:
            return None, "Could not install Kal on Windows: %s" % exc
        if proc.returncode != 0 or not self._has_festival_scheme(target):
            return None, ("Could not install Kal on Windows: " +
                          ((proc.stderr or proc.stdout or
                            "the copied voice is incomplete").strip()))
        return self.scan_voice_dir(str(target)), ""

    def refresh_voice_roots(self, install_kal: bool = True) -> dict:
        """Discover immediate child voices in configured Windows/WSL roots.

        Only registrations previously created by this scanner are removed.
        A root that cannot be read is non-authoritative, so its registrations
        are retained until a later successful scan.
        """
        report = {"added": [], "updated": [], "removed": [],
                  "warnings": [], "windows_root": self.generated_voice_root(),
                  "wsl_root": ""}
        discovered = {"windows": {}, "wsl": {}}
        successful = {"windows": False, "wsl": False}

        windows_root = Path(report["windows_root"])
        try:
            windows_root.mkdir(parents=True, exist_ok=True)
            successful["windows"] = True
            if install_kal:
                info, warning = self._install_windows_kal()
                if warning:
                    report["warnings"].append(warning)
                if info:
                    discovered["windows"][info["name"]] = info
            for child in sorted(windows_root.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir() or not self._has_festival_scheme(child):
                    continue
                try:
                    info = self.scan_voice_dir(str(child))
                    discovered["windows"][info["name"]] = info
                except (BackendError, OSError, ValueError) as exc:
                    report["warnings"].append("%s: %s" % (child.name, exc))
        except OSError as exc:
            report["warnings"].append(
                "Windows voice root is unavailable: %s" % exc)

        try:
            wsl_root = self.generated_voice_wsl_root()
        except BackendError as exc:
            wsl_root = ""
            report["warnings"].append(str(exc))
        report["wsl_root"] = wsl_root
        if wsl_root:
            proc = self._run([
                "find", wsl_root, "-mindepth", "1", "-maxdepth", "1",
                "-type", "d"
            ])
            if proc.returncode == 0:
                successful["wsl"] = True
                for path in sorted(line.strip() for line in
                                   (proc.stdout or "").splitlines()
                                   if line.strip()):
                    try:
                        info = self.scan_voice_dir_wsl(path)
                        discovered["wsl"][info["name"]] = info
                    except (BackendError, OSError, ValueError) as exc:
                        report["warnings"].append("%s: %s" % (path, exc))
            else:
                report["warnings"].append(
                    "WSL voice root is unavailable: " +
                    ((proc.stderr or proc.stdout or wsl_root).strip()))

        registrations = self.fcfg().setdefault("voices", {})
        for origin in ("windows", "wsl"):
            for name, info in discovered[origin].items():
                if origin == "wsl" and name in discovered["windows"]:
                    report["warnings"].append(
                        "Skipped WSL voice '%s'; the Windows scan root has "
                        "the same name." % name)
                    continue
                current = registrations.get(name)
                if (isinstance(current, dict) and
                        not current.get("auto_discovered")):
                    report["warnings"].append(
                        "Skipped discovered '%s'; a manual registration uses "
                        "that name." % name)
                    continue
                registration = migrate_voice_registration(info)
                registration.pop("name", None)
                registration["auto_discovered"] = True
                registration["scan_origin"] = origin
                changed = current != registration
                registrations[name] = registration
                if current is None:
                    report["added"].append(name)
                elif changed:
                    self.invalidate_voice_metadata(name)
                    report["updated"].append(name)

        for name, registration in list(registrations.items()):
            if not isinstance(registration, dict) or \
                    not registration.get("auto_discovered"):
                continue
            origin = str(registration.get("scan_origin") or "")
            if successful.get(origin) and name not in discovered[origin]:
                registrations.pop(name, None)
                self.invalidate_voice_metadata(name)
                report["removed"].append(name)
        return report

    def validate_generated_voice_location(self, path: str) -> str:
        location = Path(canonical_windows_path(path)).resolve()
        root = Path(self.generated_voice_root()).resolve()
        try:
            location.relative_to(root)
        except ValueError as exc:
            raise BackendError(
                "Select a generated voice inside the configured root:\n"
                f"{root}\n\nChange it in Options > WSL / Festival settings."
            ) from exc
        return str(location)

    def _wsl_exe(self) -> Optional[str]:
        import shutil
        exe = (self.fcfg().get("wsl_exe") or "").strip()
        if exe:
            return exe if os.path.exists(exe) or shutil.which(exe) else None
        return shutil.which("wsl") or shutil.which("wsl.exe")

    def _base_cmd(self) -> List[str]:
        exe = self._wsl_exe()
        if not exe:
            raise BackendError(
                "wsl.exe not found.\n\nInstall WSL (wsl --install) or set the "
                "WSL executable in Options > WSL / Festival settings...")
        cmd = [exe]
        distro = (self.fcfg().get("distro") or "").strip()
        if distro:
            cmd += ["-d", distro]
        return cmd

    def _timeout(self) -> float:
        try:
            return float(self.fcfg().get("timeout_s") or 180)
        except (TypeError, ValueError):
            return 180.0

    # -- plumbing ---------------------------------------------------------------
    def _run(self, args: List[str], timeout: Optional[float] = None):
        import subprocess
        cmd = self._base_cmd() + ["--"] + args
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout or self._timeout())
        except FileNotFoundError as e:
            raise BackendError(f"Could not launch WSL ({cmd[0]}): {e}")
        except subprocess.TimeoutExpired:
            raise BackendError(
                "Festival in WSL timed out after %.0f s.\n(Options > WSL / "
                "Festival settings... to raise the timeout. Multisyn voices "
                "on /mnt/<drive> load slowly — consider keeping them inside "
                "the WSL filesystem.)" % self._timeout())

    def available(self):
        """-> (ok, human message). Checks wsl.exe and festival inside WSL.
        Uses plain argv (no shell) -- wsl.exe mangles complex quoting."""
        try:
            fbin = self.fcfg().get("festival_bin") or "festival"
            proc = self._run([fbin, "--version"],
                             timeout=min(60.0, self._timeout()))
        except BackendError as e:
            return False, str(e)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or "estival" not in out:
            return False, ("festival not found inside WSL "
                           f"(exit {proc.returncode}).\n{out.strip()[:400]}\n\n"
                           "Inside WSL:  sudo apt install festival")
        return True, out.strip().splitlines()[-1][:200]

    def _exchange_dir(self) -> str:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "festvox_gui_wsl")
        os.makedirs(d, exist_ok=True)
        return d

    def _native_runtime_bin(self) -> str:
        """Return the WSL command for the local crossover-aware Festival."""
        configured = str(
            self.fcfg().get("native_festival_bin") or "").strip()
        if configured:
            if configured.startswith("/"):
                return configured
            if os.path.isabs(configured):
                path = Path(configured)
                if not path.is_file():
                    raise BackendError(
                        "Configured native Festival runtime was not found:\n"
                        + str(path))
                return win_to_wsl_path(str(path))
            return configured
        if not NATIVE_FESTIVAL_RUNTIME.is_file():
            builder = (
                FESTVOX_TOOL_DIR / "native_unisyn" /
                "build_wsl_runtime.py"
            )
            raise BackendError(
                "The crossover-aware Festival runtime has not been built.\n"
                "Run:\n"
                f'  py -3.14 "{builder}" --distro '
                f'"{self.fcfg().get("distro") or "Ubuntu"}"\n\n'
                "Its WSL build dependencies are g++, festival-dev, "
                "libestools-dev, and libsystemd-dev. Legacy joins remains "
                "available without this runtime."
            )
        return win_to_wsl_path(str(NATIVE_FESTIVAL_RUNTIME))

    def _default_wsl_distro_name(self) -> str:
        """Resolve WSL's configured default without launching a process."""
        cached = getattr(self, "_default_wsl_distro_cache", None)
        if cached is not None:
            return str(cached)
        name = ""
        try:
            import winreg

            with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Lxss") as root:
                identifier = str(
                    winreg.QueryValueEx(root, "DefaultDistribution")[0])
                with winreg.OpenKey(root, identifier) as distro:
                    name = str(
                        winreg.QueryValueEx(
                            distro, "DistributionName")[0]).strip()
        except (ImportError, OSError, TypeError, ValueError):
            name = ""
        if not name:
            try:
                import subprocess

                completed = subprocess.run(
                    [self._wsl_exe(), "--list", "--quiet"],
                    capture_output=True, timeout=10.0,
                    creationflags=getattr(
                        subprocess, "CREATE_NO_WINDOW", 0),
                )
                raw = bytes(completed.stdout or b"")
                encoding = "utf-16-le" if b"\0" in raw else "utf-8"
                listing = raw.decode(encoding, errors="replace").replace(
                    "\0", "")
                names = [
                    line.strip().lstrip("*").strip()
                    for line in listing.splitlines()
                    if line.strip().lstrip("*").strip()
                ]
                if completed.returncode == 0 and names:
                    # WSL lists the default distribution first.
                    name = names[0]
            except (OSError, subprocess.SubprocessError, TypeError):
                name = ""
        self._default_wsl_distro_cache = name
        return name

    def _wsl_path_on_windows(self, value: str):
        """Map one WSL path to a stat-able Windows path when possible."""
        text = str(value or "").strip().replace("\\", "/")
        mapped = wsl_to_windows_path(text)
        if mapped:
            return Path(mapped)
        if not text.startswith("/"):
            return None
        distro = (
            str(self.fcfg().get("distro") or "").strip() or
            self._default_wsl_distro_name()
        )
        if (not distro or "/" in distro or "\\" in distro or
                distro in {".", ".."}):
            return None
        tail = text.lstrip("/").replace("/", "\\")
        return Path("\\\\wsl.localhost\\" + distro + "\\" + tail)

    def _native_runtime_stamp(self):
        configured = str(
            self.fcfg().get("native_festival_bin") or "").strip()
        if configured:
            local = (
                Path(configured)
                if os.path.isabs(configured) and not configured.startswith("/")
                else self._wsl_path_on_windows(configured)
            )
            if local is not None:
                try:
                    info = local.stat()
                    return (
                        "configured-file", str(local),
                        int(info.st_mtime_ns), int(info.st_size),
                        int(getattr(info, "st_ino", 0)),
                    )
                except OSError:
                    pass
            target = configured
            try:
                if not target.startswith("/"):
                    located = self._run(
                        ["which", target],
                        timeout=min(15.0, self._timeout()))
                    if located.returncode != 0:
                        return ("configured-unavailable", configured)
                    target = str(located.stdout or "").strip().splitlines()[0]
                probed = self._run(
                    ["stat", "--printf=%Y:%s:%i:%y", target],
                    timeout=min(15.0, self._timeout()))
                if probed.returncode == 0:
                    return (
                        "configured-wsl-file", target,
                        str(probed.stdout or "").strip(),
                    )
            except (BackendError, IndexError):
                pass
            return ("configured-unavailable", configured)
        try:
            info = NATIVE_FESTIVAL_RUNTIME.stat()
        except OSError:
            return None
        return (
            "project-file", str(NATIVE_FESTIVAL_RUNTIME),
            int(info.st_mtime_ns), int(info.st_size),
            int(getattr(info, "st_ino", 0)),
        )

    def _stop_native_server_locked(self):
        import subprocess

        process = self._native_server_process
        reader = self._native_server_reader
        self._native_server_process = None
        self._native_server_queue = None
        self._native_server_reader = None
        self._native_server_jobs = 0
        self._native_server_binary_stamp = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                process.wait(timeout=1.5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.25)

    def _stop_native_server(self):
        with self._native_server_lock:
            self._stop_native_server_locked()

    def shutdown(self):
        """Release only the backend's own warm Festival child process."""
        self._stop_native_server()

    def _start_native_server_locked(self):
        import queue
        import subprocess
        import time

        command = (
            self._base_cmd() + ["--", self._native_runtime_bin(), "--server"])
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            raise BackendError(
                "Could not start the persistent native Festival runtime: "
                + str(error)) from error
        output_queue = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    output_queue.put(line.rstrip("\r\n"))
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output, name="FestVoxFestivalOutput", daemon=True)
        reader.start()
        deadline = time.monotonic() + min(30.0, self._timeout())
        startup = []
        while time.monotonic() < deadline:
            try:
                line = output_queue.get(
                    timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                break
            if line == "FESTVOX-NATIVE-SERVER-READY":
                self._native_server_process = process
                self._native_server_queue = output_queue
                self._native_server_reader = reader
                self._native_server_jobs = 0
                self._native_server_binary_stamp = (
                    self._native_runtime_stamp())
                return
            startup.append(line)
        try:
            process.terminate()
        except OSError:
            pass
        raise BackendError(
            "The persistent native Festival runtime did not become ready.\n"
            + ("\n".join(startup[-20:]) or "(no startup output)"))

    def _run_native_server_job(self, scheme_path: str):
        import queue
        import time
        import types
        import uuid

        with self._native_server_lock:
            try:
                maximum_jobs = max(
                    1, min(256, int(self.fcfg().get(
                        "native_runtime_max_jobs", 32))))
            except (TypeError, ValueError):
                maximum_jobs = 32
            changed_binary = (
                self._native_server_process is not None and
                self._native_server_binary_stamp !=
                self._native_runtime_stamp())
            if (changed_binary or
                    self._native_server_jobs >= maximum_jobs):
                self._stop_native_server_locked()
            if (self._native_server_process is None or
                    self._native_server_process.poll() is not None):
                self._stop_native_server_locked()
                self._start_native_server_locked()
            process = self._native_server_process
            output_queue = self._native_server_queue
            token = uuid.uuid4().hex
            try:
                process.stdin.write(
                    token + "\t" + win_to_wsl_path(scheme_path) + "\n")
                process.stdin.flush()
            except (AttributeError, BrokenPipeError, OSError) as error:
                self._stop_native_server_locked()
                raise BackendError(
                    "The persistent native Festival runtime stopped before "
                    "the render began.") from error
            begin = '(GUIJOBBEGIN "%s")' % token
            end_prefix = '(GUIJOBEND "%s" ' % token
            deadline = time.monotonic() + self._timeout()
            output = []
            started = False
            status = None
            while time.monotonic() < deadline:
                try:
                    line = output_queue.get(
                        timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if line is None:
                    break
                if line == begin:
                    started = True
                    continue
                if line.startswith(end_prefix) and line.endswith(")"):
                    try:
                        status = int(line[len(end_prefix):-1])
                    except ValueError:
                        status = 2
                    break
                if started:
                    output.append(line)
            self._native_server_jobs += 1
            if status is None:
                self._stop_native_server_locked()
                raise BackendError(
                    "The persistent native Festival runtime stopped or "
                    "timed out before completing this render.\n" +
                    ("\n".join(output[-20:]) or "(no job output)"))
            return types.SimpleNamespace(
                returncode=status,
                stdout="\n".join(output) + ("\n" if output else ""),
                stderr="",
            )

    def _run_scheme(
            self, scheme: str, festival_bin: Optional[str] = None) -> str:
        """Write scheme to the exchange dir, run festival -b on it inside
        WSL, and return combined output. Raises BackendError on failure."""
        import uuid
        scm = os.path.join(self._exchange_dir(), f"job_{uuid.uuid4().hex}.scm")
        with open(scm, "w", encoding="utf-8", newline="\n") as f:
            f.write(scheme)
        if festival_bin:
            fbin = festival_bin
            native_job = False
        elif "(festvox_us_generate_wave" in scheme:
            fbin = self._native_runtime_bin()
            native_job = True
        else:
            fbin = self.fcfg().get("festival_bin") or "festival"
            native_job = False
        try:
            if (native_job and bool(
                    self.fcfg().get("persistent_native_runtime", True))):
                proc = self._run_native_server_job(scm)
            else:
                proc = self._run([fbin, "-b", win_to_wsl_path(scm)])
        finally:
            try:
                os.remove(scm)
            except OSError:
                pass
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            msg = ("Festival (WSL) failed (exit %d):\n%s"
                   % (proc.returncode, out.strip()[-1200:] or "(no output)"))
            if "not member of PhoneSet" in out:
                msg += ("\n\nThat phone does not exist in the selected "
                        "voice's phone set. Asaxi/Japanese phones belong to "
                        "the Lem bank voices; for other voices (e.g. "
                        "kal_diphone) use their own phones.")
            raise BackendError(msg)
        return out

    # -- voices ------------------------------------------------------------------
    @staticmethod
    def _guess_voice_fn(name: str) -> str:
        name = str(name).strip()
        return name if name.startswith("voice_") else "voice_" + name

    @staticmethod
    def _pick_voice_fns(txt: str):
        """All voice_* functions defined in a scheme text -> (main, en).
        The `_en` variant is the English front end build_festival_voice.py
        generates alongside the native one."""
        import re as _re
        fns = _re.findall(r"\(define\s+\(\s*(voice_[A-Za-z0-9_]+)", txt)
        if not fns:
            m = _re.search(r"proclaim_voice\s+'?([A-Za-z0-9_]+)", txt)
            fns = ["voice_" + m.group(1)] if m else []
        main = next((f for f in fns if not f.endswith("_en")),
                    fns[0] if fns else None)
        en = next((f for f in fns if f.endswith("_en")), None)
        return main, en

    def scan_voice_dir(self, path: str) -> dict:
        """Inspect a festvox/Multisyn voice folder (Windows-visible path)
        -> {name, dir, voice, voice_en, scm}."""
        import glob
        path = os.path.abspath(path)
        fx = os.path.join(path, "festvox")
        search = fx if os.path.isdir(fx) else path
        voice_fn, voice_en, scm_rel = None, None, None
        for scm in sorted(glob.glob(os.path.join(search, "*.scm"))):
            try:
                txt = Path(scm).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            main, en = self._pick_voice_fns(txt)
            if main:
                voice_fn, voice_en = main, en
                scm_rel = os.path.relpath(scm, path).replace("\\", "/")
                break
        name = os.path.basename(path.rstrip("/\\")) or "voice"
        metadata = {}
        try:
            metadata = json.loads(
                (Path(path) / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            pass
        return make_voice_registration(
            windows_path=path,
            name=name,
            voice=voice_fn or self._guess_voice_fn(name),
            voice_en=voice_en,
            scm=scm_rel,
            metadata=metadata,
        )

    def scan_voice_dir_wsl(self, wsl_dir: str) -> dict:
        """Same, for a folder that lives INSIDE the WSL filesystem. Uses only
        plain argv commands (find/cat) -- shell one-liners do not survive
        wsl.exe's argument quoting reliably."""
        import re as _re
        wsl_dir = wsl_dir.strip().rstrip("/")
        proc = self._run(["find", wsl_dir + "/festvox", wsl_dir,
                          "-maxdepth", "1", "-name", "*.scm"])
        cands = [ln.strip() for ln in (proc.stdout or "").splitlines()
                 if ln.strip().endswith(".scm")]
        if not cands:
            detail = (proc.stderr or proc.stdout or "").strip()[:400]
            raise BackendError(
                f"No festvox/*.scm found under {wsl_dir} inside WSL.\n"
                f"{detail}\n\nCheck the path (it must be the voice folder "
                "built by build_festival_voice.py or a festvox voice dir).")
        voice_fn, voice_en, scm_rel = None, None, None
        for scm in sorted(cands)[:12]:
            cat = self._run(["cat", scm])
            if cat.returncode != 0:
                continue
            main, en = self._pick_voice_fns(cat.stdout or "")
            if main:
                voice_fn, voice_en = main, en
                scm_rel = (os.path.relpath(scm, wsl_dir)
                           if scm.startswith(wsl_dir) else scm).replace("\\", "/")
                break
        name = wsl_dir.rstrip("/").split("/")[-1] or "voice"
        return {"name": name, "dir": wsl_dir,
                "voice": voice_fn or self._guess_voice_fn(name),
                "voice_en": voice_en, "scm": scm_rel}

    def add_voice(self, info: dict) -> str:
        name = info["name"]
        registration = migrate_voice_registration(info)
        registration.pop("name", None)
        self.fcfg().setdefault("voices", {})[name] = registration
        self.invalidate_voice_metadata(name)
        return name

    def _voice_fingerprint(self, voicebank: str):
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict):
            return ("registration", repr(voice))
        root = str(voice.get("dir") or "").rstrip("/\\")
        registration = tuple(sorted(
            (str(key), repr(value)) for key, value in voice.items()
        ))
        if not root:
            return ("registration", root, registration)
        stat_root = Path(root)
        fingerprint_kind = "windows-files"
        if root.startswith("/"):
            stat_root = self._wsl_path_on_windows(root)
            if stat_root is None:
                return ("wsl-registration", root, registration)
            try:
                root_stat = stat_root.stat()
            except OSError:
                return ("wsl-registration", root, registration)
            fingerprint_kind = "wsl-files"
            root_identity = (
                int(root_stat.st_mtime_ns), int(root_stat.st_size),
                int(getattr(root_stat, "st_ino", 0)),
            )
        else:
            root_identity = ()
        relatives = [
            "dic/diphone_index.json",
            "dic/unit_alternatives.json",
            "dic/voice_manifest.json",
        ]
        scm = str(voice.get("scm") or "").strip().replace("\\", "/")
        if scm and not scm.startswith("/") and ".." not in Path(scm).parts:
            relatives.append(scm)
            scheme_stem = Path(scm).stem
            database_stem = (
                scheme_stem if scheme_stem.endswith("_diphone")
                else scheme_stem + "_diphone"
            )
            # A persistent Festival worker retains the loaded UniSyn database.
            # Fingerprint the actual runtime payload as well as its JSON
            # metadata so rebuilding a voice in place cannot serve stale units.
            relatives.extend([
                "dic/%s.est" % database_stem,
                "dic/%s_legacy.est" % database_stem,
                "group/%s.group" % database_stem,
            ])
        rows = []
        for relative in relatives:
            path = stat_root / Path(relative)
            try:
                stat = path.stat()
                row = [
                    relative, int(stat.st_mtime_ns), int(stat.st_size),
                    int(getattr(stat, "st_dev", 0)),
                    int(getattr(stat, "st_ino", 0)),
                ]
                # UNC metadata fingerprints must remain a cheap stat path. A
                # digest fallback over a large generated index would undo the
                # persistent renderer's latency benefit.
                if fingerprint_kind == "windows-files":
                    row.append(file_change_token(path, stat))
                rows.append(tuple(row))
            except OSError:
                rows.append((relative, None, None, None, None))
        if fingerprint_kind == "wsl-files":
            return (
                fingerprint_kind, root, str(stat_root), registration,
                root_identity, tuple(rows),
            )
        return (
            fingerprint_kind, str(Path(root).resolve()), registration,
            tuple(rows),
        )

    def refresh_voice_metadata(self, voicebank: str):
        """Refresh one local voice cache only when its metadata changed."""
        name = str(voicebank or "")
        if not name:
            return None
        token = self._voice_fingerprint(name)
        with self._cache_lock:
            previous = self._voice_fingerprints.get(name)
            if previous is not None and previous != token:
                self.invalidate_voice_metadata(name)
            self._voice_fingerprints[name] = token
            return (token, int(self._cache_epoch))

    def invalidate_voice_metadata(self, voicebank: str = ""):
        """Forget generated-bank indexes after a rebuild or re-registration."""
        # A warm interpreter may retain loaded Scheme/database objects from
        # the previous generated tree. Restart it before publishing new
        # metadata so a rebuild under the same name cannot stay stale.
        self._stop_native_server()
        name = str(voicebank or "")
        with self._cache_lock:
            self._cache_epoch += 1
            if not name:
                self._alternatives.clear()
                self._sustains.clear()
                self._voice_metadata.clear()
                self._voice_fingerprints.clear()
                return
            self._alternatives.pop(name, None)
            self._voice_metadata.pop(name, None)
            self._voice_fingerprints.pop(name, None)
            for key in [key for key in self._sustains if key[0] == name]:
                self._sustains.pop(key, None)

    def cache_info(self) -> dict:
        """Return reproducible Festival cache usage; paths are never cleared."""
        with self._cache_lock:
            sustain_info = self._sustains.info()
            metadata_info = self._voice_metadata.info()
            alternatives_info = self._alternatives.info()
            return {
                "audio": {
                    "bytes": int(sustain_info["bytes"]),
                    "sustains": len(self._sustains),
                    "max_sustains": int(sustain_info["max_entries"]),
                    "max_sustain_bytes": int(sustain_info["max_bytes"]),
                },
                "voice": {
                    "bytes": (int(metadata_info["bytes"]) +
                              int(alternatives_info["bytes"])),
                    "voices": len(set(self._voice_metadata) |
                                  set(self._alternatives)),
                    "metadata_entries": len(self._voice_metadata),
                    "alternative_entries": len(self._alternatives),
                    "max_voice_entries": int(
                        metadata_info["max_entries"]),
                    "max_voice_bytes": (int(metadata_info["max_bytes"]) +
                                        int(alternatives_info["max_bytes"])),
                },
                "model": {"bytes": 0, "entries": 0},
            }

    def clear_application_cache(self, category: str) -> dict:
        """Clear in-memory Festival data without deleting generated voices."""
        category = str(category).casefold()
        if category not in {"audio", "voice", "model", "all"}:
            raise ValueError("cache category must be audio, voice, model, or all")
        before = self.cache_info()
        with self._cache_lock:
            if category in {"voice", "all"}:
                self.invalidate_voice_metadata()
            elif category == "audio":
                self._sustains.clear()
        return {"before": before, "after": self.cache_info()}

    def voice_metadata(self, voicebank: str) -> dict:
        """Read a generated voice's runtime index without modifying it."""
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            return {}
        root = str(voice["dir"]).rstrip("/\\")
        for _attempt in range(3):
            stamp = self.refresh_voice_metadata(voicebank)
            with self._cache_lock:
                try:
                    return self._voice_metadata[voicebank]
                except KeyError:
                    pass
            try:
                if root.startswith("/"):
                    proc = self._run(
                        ["cat", root + "/dic/diphone_index.json"])
                    if proc.returncode != 0:
                        return {}
                    metadata = json.loads(proc.stdout or "{}")
                else:
                    metadata = json.loads(
                        (Path(root) / "dic" / "diphone_index.json")
                        .read_text(encoding="utf-8")
                    )
            except (OSError, ValueError, TypeError, BackendError):
                return {}
            if not isinstance(metadata, dict):
                return {}
            published = deep_freeze(_runtime_voice_metadata(metadata))
            if self.refresh_voice_metadata(voicebank) != stamp:
                continue
            with self._cache_lock:
                current = (self._voice_fingerprints.get(str(voicebank)),
                           int(self._cache_epoch))
                if current != stamp:
                    continue
                self._voice_metadata[voicebank] = published
                return published
        return {}

    def japanese_runtime_metadata(self, voicebank: str) -> dict:
        """Return metadata only for the isolated Japanese voice entry point."""
        metadata = self.voice_metadata(voicebank)
        compatibility = self.voice_compatibility(voicebank)
        if not compatibility.supports("ja"):
            return {}
        entry = str(compatibility.voice_entry_points.get("ja") or "")
        if not entry:
            return {}
        result = dict(metadata)
        # Integrated voices retain their primary-language entry point at the
        # top level. A Japanese caller needs the explicitly selected Japanese
        # route, not that primary fallback.
        result["language"] = "ja"
        result["voice_entry_point"] = entry
        return result

    def voice_compatibility(self, voicebank: str):
        """Return inspectable language/phoneset support for one voice."""
        if str(voicebank) in BUILTIN_FESTIVAL_VOICES:
            return VoiceCompatibility(
                metadata_status="current",
                primary_language="en",
                supported_languages=("en",),
                voice_entry_points={"en": "voice_kal_diphone"},
                configuration_id="festival-builtin-kal-diphone",
            )
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        configured = {}
        if isinstance(voice, dict):
            main = str(voice.get("voice") or "")
            english = str(voice.get("voice_en") or "")
            declared_language = normalize_language_code(
                voice.get("language")
            )
            declared_entry = str(voice.get("entry_point") or main)
            if declared_language and declared_entry:
                configured[declared_language] = declared_entry
            elif main.endswith("_ja"):
                configured["ja"] = main
            if english and not declared_language:
                configured["en"] = english
        return read_voice_compatibility(
            self.voice_metadata(voicebank),
            configured_entry_points=configured,
        )

    def unit_alternatives(self, voicebank: str) -> dict:
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            return {}
        d = str(voice["dir"]).rstrip("/\\")
        for _attempt in range(3):
            stamp = self.refresh_voice_metadata(voicebank)
            with self._cache_lock:
                try:
                    return self._alternatives[voicebank]
                except KeyError:
                    pass
            try:
                if d.startswith("/"):
                    proc = self._run(
                        ["cat", d + "/dic/unit_alternatives.json"])
                    if proc.returncode != 0:
                        return {}
                    meta = json.loads(proc.stdout or "{}")
                else:
                    meta = json.loads(
                        (Path(d) / "dic" / "unit_alternatives.json")
                        .read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, BackendError):
                return {}
            result = deep_freeze(
                dict(meta.get("diphones") or meta.get("alternatives") or {}))
            if self.refresh_voice_metadata(voicebank) != stamp:
                continue
            with self._cache_lock:
                current = (self._voice_fingerprints.get(str(voicebank)),
                           int(self._cache_epoch))
                if current != stamp:
                    continue
                self._alternatives[voicebank] = result
                return result
        return {}

    @staticmethod
    def _unit_wav_name(pair: str, choice: dict, metadata: dict) -> str:
        direct = str(choice.get("wav_name") or "").strip()
        if direct:
            return Path(direct).name
        source = str(choice.get("wav") or "").replace("\\", "/")
        if source.startswith("generated/"):
            return Path(source).name
        index_name = str(choice.get("index_name") or "").strip()
        if not index_name:
            left = str(choice.get("left_name") or "").strip()
            right = str(pair or "").split("-", 1)[-1]
            index_name = "%s-%s" % (left, right) if left else str(pair)
        row = (metadata.get("index") or {}).get(index_name)
        if not row and index_name != str(pair):
            row = (metadata.get("index") or {}).get(str(pair))
        if isinstance(row, (list, tuple)) and row:
            return Path(str(row[0])).name
        if isinstance(row, dict):
            return Path(str(row.get("wav") or row.get("wav_name") or "")).name
        return ""

    def unit_pitchmark_diagnostic(self, voicebank: str, pair: str,
                                  choice: dict) -> dict:
        """Load the generated WAV and PM track actually used by UniSyn.

        Only generated voice output is read. The original UTAU bank path in
        provenance metadata is deliberately ignored.
        """
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            raise BackendError(
                "Pitchmark inspection is available for registered generated "
                "Festival voices.")
        root = str(voice["dir"]).rstrip("/\\")
        metadata = self.voice_metadata(voicebank)
        wav_name = self._unit_wav_name(pair, dict(choice or {}), metadata)
        if not wav_name:
            raise BackendError(
                "The generated WAV for this unit is not identified. Rebuild "
                "the voice with current metadata and try again.")
        pm_name = Path(wav_name).stem + ".pm"
        f0_name = Path(wav_name).stem + ".f0.json"
        f0_text = ""

        if root.startswith("/"):
            wav_candidates = (root + "/wav/" + wav_name,
                              root + "/db/wav/" + wav_name)
            pm_candidates = (root + "/pm/" + pm_name,
                             root + "/db/pm/" + pm_name)
            f0_candidates = (root + "/pm/" + f0_name,
                             root + "/db/pm/" + f0_name)
            wav_path = next((path for path in wav_candidates
                             if self._run(["test", "-f", path]).returncode == 0),
                            "")
            pm_path = next((path for path in pm_candidates
                            if self._run(["test", "-f", path]).returncode == 0),
                           "")
            f0_path = next((path for path in f0_candidates
                            if self._run(["test", "-f", path]).returncode == 0),
                           "")
            if not wav_path or not pm_path:
                raise BackendError(
                    "This generated unit has no readable WAV/PM pair. Rebuild "
                    "the voice without --skip-pitchmarks.")
            pm_proc = self._run(["cat", pm_path])
            if pm_proc.returncode != 0:
                raise BackendError("Could not read the generated pitchmark track.")
            local = os.path.join(
                self._exchange_dir(), "unit_pm_%s.wav" % uuid.uuid4().hex)
            try:
                copied = self._run(
                    ["cp", "--", wav_path, win_to_wsl_path(local)])
                if copied.returncode != 0:
                    raise BackendError(
                        copied.stderr or "Could not copy the generated unit WAV.")
                samples, sr = read_wav(local)
            finally:
                try:
                    os.remove(local)
                except OSError:
                    pass
            pm_text = pm_proc.stdout or ""
            if f0_path:
                f0_proc = self._run(["cat", f0_path])
                if f0_proc.returncode == 0:
                    f0_text = f0_proc.stdout or ""
        else:
            root_path = Path(root)
            wav_path = next((path for path in (
                root_path / "wav" / wav_name,
                root_path / "db" / "wav" / wav_name,
            ) if path.is_file()), None)
            pm_path = next((path for path in (
                root_path / "pm" / pm_name,
                root_path / "db" / "pm" / pm_name,
            ) if path.is_file()), None)
            f0_path = next((path for path in (
                root_path / "pm" / f0_name,
                root_path / "db" / "pm" / f0_name,
            ) if path.is_file()), None)
            if wav_path is None or pm_path is None:
                raise BackendError(
                    "This generated unit has no readable WAV/PM pair. Rebuild "
                    "the voice without --skip-pitchmarks.")
            try:
                samples, sr = read_wav(str(wav_path))
                pm_text = pm_path.read_text(
                    encoding="utf-8", errors="replace")
                if f0_path is not None:
                    f0_text = f0_path.read_text(
                        encoding="utf-8", errors="replace")
            except (OSError, ValueError, wave.Error) as error:
                raise BackendError(
                    "Could not read this generated WAV/PM pair: %s" % error
                ) from error

        marks = parse_est_pitchmarks(pm_text)
        if len(marks) < 2:
            raise BackendError(
                "The generated pitchmark track contains fewer than two marks.")
        epoch_track = pitchmark_f0_track(marks)
        analyzed = parse_pitchmark_f0_sidecar(f0_text)
        track = analyzed.get("frames") or epoch_track
        return {
            "voicebank": str(voicebank),
            "pair": str(pair),
            "alias": str(choice.get("alias") or ""),
            "wav_name": wav_name,
            "pm_name": pm_name,
            "samples": np.asarray(samples, np.float32),
            "sr": int(sr),
            "pitchmarks": marks,
            "f0_track": track,
            "epoch_f0_track": epoch_track,
            "f0_track_kind": (
                "analyzed" if analyzed else "epoch-rate"
            ),
            "f0_source": (
                analyzed.get("f0_source") if analyzed
                else "pitchmark-intervals"
            ),
            "discontinuities": pitchmark_discontinuities(marks),
            "source_slice": dict(choice.get("source_slice") or {}),
        }

    def automatic_unit_overrides(self, phones, voicebank: str) -> dict:
        """Choose automatic variants only for voices that need GUI routing.

        Current Japanese voices install a language-specific UniSyn selector
        that understands CV/VCV/CVVC roles, vowel blends, and bank-profiled
        moraic-nasal allophones.  Pre-filling every edge here would turn those
        automatic choices into explicit overrides before that selector runs.
        English/ARPAsing voices retain the existing contextual take picker.
        """
        if self.japanese_runtime_metadata(voicebank):
            return {}
        return contextual_unit_overrides(
            phones, self.unit_alternatives(voicebank))

    def voice_pitch_hz(self, voicebank: str):
        """Read a generated voice's measured pitch without modifying it."""
        if str(voicebank) == "kal_diphone":
            return 110.0
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            return None
        root = str(voice["dir"]).rstrip("/\\")

        def read(path):
            if root.startswith("/"):
                proc = self._run(["cat", path])
                return proc.stdout if proc.returncode == 0 else ""
            try:
                return Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""

        for filename in ("unit_alternatives.json", "diphone_index.json"):
            path = (root + "/dic/" + filename if root.startswith("/") else
                    str(Path(root) / "dic" / filename))
            try:
                meta = json.loads(read(path) or "{}")
                value = metadata_voice_pitch_hz(meta)
                if value is not None:
                    return value
            except (ValueError, TypeError):
                pass
        scm = str(voice.get("scm") or "")
        if scm:
            path = (root + "/" + scm.lstrip("/") if root.startswith("/") else
                    str(Path(root) / Path(scm)))
            match = re.search(r"target_f0_mean\s+([0-9]+(?:\.[0-9]+)?)",
                              read(path))
            if match:
                value = float(match.group(1))
                if 40.0 <= value <= 700.0:
                    return value
        return None

    def sustain_sample(self, phone: str, voicebank: str):
        """Load a generated X-X sustain without touching its source bank."""
        key = (str(voicebank), str(phone).rstrip("_"))
        self.refresh_voice_metadata(voicebank)
        with self._cache_lock:
            if key in self._sustains:
                return self._sustains[key]
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            with self._cache_lock:
                self._sustains[key] = None
            return None
        root = str(voice["dir"]).rstrip("/\\")
        try:
            if root.startswith("/"):
                meta = None
                for candidate in (root + "/dic/diphone_index.json",
                                  root + "/db/dic/diphone_index.json"):
                    meta_proc = self._run(["cat", candidate])
                    if meta_proc.returncode == 0:
                        meta = json.loads(meta_proc.stdout or "{}")
                        break
                if meta is None:
                    raise OSError("diphone metadata is unavailable")
            else:
                candidates = (Path(root) / "dic" / "diphone_index.json",
                              Path(root) / "db" / "dic" /
                              "diphone_index.json")
                source = next((path for path in candidates if path.is_file()),
                              None)
                if source is None:
                    raise OSError("diphone metadata is unavailable")
                meta = json.loads(source.read_text(encoding="utf-8"))
            row = (meta.get("index") or {}).get("%s-%s" % (key[1], key[1]))
            if not row:
                with self._cache_lock:
                    self._sustains[key] = None
                return None
            wav_name, start, _mid, end = row
            if root.startswith("/"):
                import uuid
                local = os.path.join(self._exchange_dir(),
                                     "sustain_%s.wav" % uuid.uuid4().hex)
                try:
                    wav_path = root + "/wav/" + str(wav_name)
                    if self._run(["test", "-f", wav_path]).returncode != 0:
                        wav_path = root + "/db/wav/" + str(wav_name)
                    copied = self._run(
                        ["cp", "--", wav_path, win_to_wsl_path(local)])
                    if copied.returncode != 0:
                        raise OSError(copied.stderr or "WSL copy failed")
                    samples, sr = read_wav(local)
                finally:
                    try:
                        os.remove(local)
                    except OSError:
                        pass
            else:
                wav_path = Path(root) / "wav" / str(wav_name)
                if not wav_path.is_file():
                    wav_path = Path(root) / "db" / "wav" / str(wav_name)
                samples, sr = read_wav(str(wav_path))
            a = max(0, min(len(samples), int(round(float(start) * sr))))
            b = max(a, min(len(samples), int(round(float(end) * sr))))
            value = (samples[a:b].copy(), int(sr)) if b - a >= 32 else None
        except (OSError, ValueError, TypeError, BackendError):
            value = None
        with self._cache_lock:
            self._sustains[key] = value
        return value

    def install_dictionary(self, voicebank: str, source_name: str,
                           entries: dict) -> str:
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            raise BackendError("The selected Festival voice is installed "
                               "system-wide and has no writable voice folder.")
        root = str(voice["dir"]).rstrip("/\\")
        filename = cleaned_dictionary_filename(source_name)
        content = cleaned_dictionary_text(entries)
        if root.startswith("/"):
            import uuid
            local = os.path.join(self._exchange_dir(),
                                 "dict_%s.dict" % uuid.uuid4().hex)
            try:
                Path(local).write_text(content, encoding="utf-8")
                made = self._run(["mkdir", "-p", root + "/dic"])
                if made.returncode != 0:
                    raise BackendError(made.stderr or "Could not create dic")
                target = root + "/dic/" + filename
                copied = self._run(
                    ["cp", "--", win_to_wsl_path(local), target])
                if copied.returncode != 0:
                    raise BackendError(copied.stderr or
                                       "Could not install dictionary")
            finally:
                try:
                    os.remove(local)
                except OSError:
                    pass
            return target
        folder = Path(root) / "dic"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / filename
        target.write_text(content, encoding="utf-8")
        return str(target)

    def install_voice_icon(self, voicebank: str, source: str) -> str:
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            raise BackendError("The selected Festival voice has no voice folder.")
        root = str(voice["dir"]).rstrip("/\\")
        suffix = Path(source).suffix.lower()
        if suffix not in VOICE_ICON_SUFFIXES:
            suffix = ".png"
        target_name = "speaker" + suffix
        if root.startswith("/"):
            target = root + "/" + target_name
            removed = self._run(
                ["rm", "-f", "--"] +
                [root + "/speaker" + ext for ext in VOICE_ICON_SUFFIXES
                 if "speaker" + ext != target_name])
            if removed.returncode != 0:
                raise BackendError(removed.stderr or
                                   "Could not replace the old portrait")
            copied = self._run(["cp", "--", win_to_wsl_path(source), target])
            if copied.returncode != 0:
                raise BackendError(copied.stderr or "Could not install portrait")
            return target
        target = Path(root) / target_name
        remove_known_voice_icons(root, keep=target)
        import shutil
        shutil.copy2(str(source), str(target))
        return str(target)

    def remove_voice_icon(self, voicebank: str):
        voice = (self.fcfg().get("voices") or {}).get(voicebank)
        if not isinstance(voice, dict) or not voice.get("dir"):
            raise BackendError("The selected Festival voice has no voice folder.")
        root = str(voice["dir"]).rstrip("/\\")
        if root.startswith("/"):
            targets = [root + "/speaker" + ext
                       for ext in VOICE_ICON_SUFFIXES]
            removed = self._run(["rm", "-f", "--"] + targets)
            if removed.returncode != 0:
                raise BackendError(removed.stderr or
                                   "Could not remove the portrait")
            return targets
        return remove_known_voice_icons(root)

    def read_installed_dictionary(self, path: str) -> dict:
        if str(path).startswith("/"):
            proc = self._run(["cat", str(path)])
            if proc.returncode != 0:
                raise BackendError(proc.stderr or
                                   "Installed dictionary is unavailable")
            return parse_cleaned_dictionary_text(proc.stdout)
        return parse_cleaned_dictionary_text(
            Path(path).read_text(encoding="utf-8", errors="replace"))

    @staticmethod
    def _normalize_wsl_voice_dir(path: str) -> str:
        import posixpath
        p = posixpath.normpath(str(path or "").strip())
        parts = [part for part in p.split("/") if part]
        if not p.startswith("/") or len(parts) < 3:
            raise BackendError(f"Refusing unsafe WSL voicebank path:\n{p}")
        low = p.lower()
        if "/utau/voice/" in low or low.endswith("/lem_v4bi_civet"):
            raise BackendError(
                "Refusing to delete an UTAU source voicebank:\n" + p)
        return p

    def _validate_wsl_generated_voice_dir(self, path: str) -> str:
        p = self._normalize_wsl_voice_dir(path)
        if self._run(["test", "-f", p + "/oto.ini"]).returncode == 0:
            raise BackendError(
                "Refusing to delete an UTAU source voicebank (oto.ini found):\n"
                + p)
        marker = self._run(["find", p + "/festvox", "-maxdepth", "1",
                            "-type", "f", "-name", "*.scm"])
        if marker.returncode != 0 or not (marker.stdout or "").strip():
            raise BackendError(
                "Refusing to delete this WSL folder because it does not look "
                "like a generated FestVox voicebank:\n" + p)
        return p

    def voicebank_removal_info(self, name: str) -> dict:
        if str(name) in BUILTIN_FESTIVAL_VOICES:
            raise BackendError(
                f"'{name}' is a built-in Festival voice and remains available "
                "by default. It cannot be uninstalled from this manager."
            )
        voice = (self.fcfg().get("voices") or {}).get(name)
        if not isinstance(voice, dict) or not voice.get("dir"):
            installed = {
                str(value) for value in
                (self.fcfg().get("installed_voices") or [])
            }
            if str(name) in installed:
                return {
                    "name": str(name),
                    "path": "Festival registration: " + str(name),
                    "kind": "registration",
                    "exists": False,
                }
            raise BackendError(
                f"'{name}' is supplied by Festival itself and has no registered "
                "voice folder. It cannot be safely uninstalled here.")
        path = str(voice["dir"])
        if path.startswith("/"):
            path = self._normalize_wsl_voice_dir(path)
            kind = "wsl"
            exists = self._run(["test", "-d", path]).returncode == 0
            if exists:
                path = self._validate_wsl_generated_voice_dir(path)
        else:
            local_path = Path(path).expanduser().resolve()
            exists = local_path.exists()
            if exists:
                local_path = validate_generated_voice_dir(local_path)
            path = str(local_path)
            kind = "windows"
        return {"name": name, "path": path, "kind": kind,
                "exists": exists}

    def uninstall_voicebank(self, name: str, delete_files: bool = True) -> str:
        info = self.voicebank_removal_info(name)
        if delete_files and info["exists"] and info["kind"] == "wsl":
            proc = self._run(["rm", "-rf", "--", info["path"]])
            if proc.returncode != 0:
                raise BackendError(
                    "WSL could not delete the voicebank:\n" +
                    ((proc.stderr or proc.stdout or "unknown error").strip()))
        elif delete_files and info["exists"]:
            delete_generated_voice_dir(info["path"])
        self.fcfg().setdefault("voices", {}).pop(name, None)
        self.fcfg()["installed_voices"] = [
            v for v in (self.fcfg().get("installed_voices") or []) if v != name]
        if self.fcfg().get("default_voice") == name:
            self.fcfg()["default_voice"] = ""
        self.invalidate_voice_metadata(name)
        return info["path"]

    def list_installed_voices(self) -> List[str]:
        out = self._run_scheme("(print (voice.list))\n")
        import re as _re
        m = _re.search(r"\(([^()]*)\)", out)
        return [t for t in (m.group(1).split() if m else []) if t.strip()]

    def voicebanks(self) -> List[dict]:
        out = []
        registered_names = set()
        for name, v in (self.fcfg().get("voices") or {}).items():
            # A scan may register the optional Windows mirror, but Kal itself
            # belongs to Festival in WSL.  Never let a stale mirror shadow the
            # authoritative built-in entry.
            if str(name) in BUILTIN_FESTIVAL_VOICES:
                continue
            registered_names.add(str(name))
            d = (v or {}).get("dir") or ""
            if d.startswith("/"):
                ok, note = True, (
                    "Legacy WSL-only path; rebuild into the configured "
                    "Windows generated-voice root when practical"
                )
            else:
                ok = bool(d) and os.path.isdir(d)
                note = "" if ok else "folder not found"
            compatibility = self.voice_compatibility(name)
            if compatibility.reason:
                note = "; ".join(item for item in (
                    note, compatibility.reason
                ) if item)
            out.append({"name": name, "dir": d, "ok": ok,
                        "source": "festival voice dir", "note": note,
                        "metadata_status": compatibility.metadata_status,
                        "primary_language": compatibility.primary_language,
                        "supported_languages": list(
                            compatibility.supported_languages)})
        installed = set(str(name) for name in
                        (self.fcfg().get("installed_voices") or []))
        installed.update(BUILTIN_FESTIVAL_VOICES)
        for name in sorted(installed - registered_names):
            built_in_path = BUILTIN_FESTIVAL_VOICES.get(name, "")
            compatibility = self.voice_compatibility(name)
            out.append({
                "name": name,
                "dir": built_in_path,
                "ok": True,
                "source": ("Festival built-in" if built_in_path else
                           "festival (voice.list)"),
                "note": ("Available by default with Festival" if
                           built_in_path else ""),
                "metadata_status": compatibility.metadata_status,
                "primary_language": compatibility.primary_language,
                "supported_languages": list(
                    compatibility.supported_languages),
            })
        return out

    def default_voicebank(self) -> Optional[str]:
        vbs = self.voicebanks()
        want = self.fcfg().get("default_voice")
        for v in vbs:
            if v["name"] == want:
                return want
        return vbs[0]["name"] if vbs else None

    # -- scheme assembly ----------------------------------------------------------
    def _use_symmetric_join_window(
            self, voicebank: str, explicit_durations=None,
            legacy_joins: bool = False) -> bool:
        """Choose UniSyn's overlap geometry without changing selected units.

        Current generated voices declare an adaptive source-window policy and
        use asymmetric pitchmark-aligned overlap at every duration. One long
        phone therefore cannot alter unrelated joins. Festival's built-in Kal
        database and generated banks predating that metadata retain their
        authored symmetric geometry; real A/B validation found no safe
        improvement from forcing them through the newer policy. Legacy joins
        always retain the historical symmetric path exactly.
        """
        _ = explicit_durations
        if legacy_joins or str(voicebank) in BUILTIN_FESTIVAL_VOICES:
            return True
        metadata = self.voice_metadata(str(voicebank))
        policy = metadata.get("source_window_policy") or {}
        explicit = policy.get("normal_unisyn_window_symmetric")
        if explicit is not None:
            return bool(explicit)
        # Compatibility for current checkpoints created before the explicit
        # flag was added. Only an isolated Japanese adaptive build receives
        # the previously validated asymmetric policy. Integrated and older
        # generated voices stay on the stable authored renderer.
        languages = tuple(str(value).strip().lower() for value in
                          (metadata.get("supported_languages") or ())
                          if str(value).strip())
        japanese_only = languages == ("ja",)
        return not (
            japanese_only
            and str(policy.get("mode") or "").strip().lower() == "adaptive"
        )

    @staticmethod
    def normalize_join_settings(settings=None) -> dict:
        """Return bounded, serializable source-window/crossover controls."""
        row = dict(settings or {})
        mode = str(row.get("mode") or "voice").strip().lower()
        aliases = {
            "default": "voice",
            "phase_aligned": "asymmetric",
            "phase-aligned": "asymmetric",
            "phase aligned": "asymmetric",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"voice", "symmetric", "asymmetric"}:
            mode = "voice"
        try:
            factor = float(row.get("window_factor", 1.0))
        except (TypeError, ValueError):
            factor = 1.0
        if not np.isfinite(factor):
            factor = 1.0
        # Wider windows can smear voiced periods; keep the interactive range
        # deliberately narrow.  Legacy mode bypasses this value entirely.
        factor = max(1.0, min(1.25, factor))
        try:
            crossover_ms = float(row.get("crossover_ms", 40.0))
        except (TypeError, ValueError):
            crossover_ms = 40.0
        if not np.isfinite(crossover_ms):
            crossover_ms = 40.0
        # Milliseconds are authoritative. The native renderer snaps the two
        # edges inward to target pitchmarks and reports the effective period
        # count; a fixed period count would vary in duration with F0.
        crossover_ms = max(0.0, min(100.0, crossover_ms))
        overrides = {}
        raw_overrides = row.get("crossover_overrides") or {}
        if isinstance(raw_overrides, dict):
            iterator = raw_overrides.items()
        elif isinstance(raw_overrides, (list, tuple)):
            iterator = enumerate(raw_overrides)
        else:
            iterator = ()
        for raw_index, raw_value in iterator:
            if not isinstance(raw_value, dict):
                continue
            try:
                unit_index = int(raw_index)
                left_ms = float(raw_value.get("left_ms"))
                right_ms = float(raw_value.get("right_ms"))
            except (TypeError, ValueError):
                continue
            if (unit_index < 0 or not np.isfinite(left_ms) or
                    not np.isfinite(right_ms)):
                continue
            left_ms = max(0.0, min(100.0, left_ms))
            right_ms = max(0.0, min(100.0, right_ms))
            total = left_ms + right_ms
            if total > 100.0:
                scale = 100.0 / total
                left_ms *= scale
                right_ms *= scale
            overrides[str(unit_index)] = {
                "left_ms": round(left_ms, 3),
                "right_ms": round(right_ms, 3),
            }
        return {
            "mode": mode,
            "window_factor": round(factor, 3),
            "crossover_ms": round(crossover_ms, 3),
            "crossover_overrides": overrides,
        }

    def resolve_join_settings(
            self, voicebank: str, *, fault_mode=None,
            explicit_durations=None) -> dict:
        """Resolve requested controls without changing timing or unit choice."""
        faults = dict(fault_mode or {})
        requested = self.normalize_join_settings(
            faults.get("_join_settings"))
        legacy = bool(faults.get("legacy_joins"))
        if legacy:
            symmetric = True
            factor = 1.0
            crossover_ms = 0.0
            crossover_overrides = {}
            source = "legacy-fault"
        else:
            mode = requested["mode"]
            symmetric = (
                self._use_symmetric_join_window(
                    voicebank, explicit_durations, legacy_joins=False)
                if mode == "voice" else mode == "symmetric"
            )
            factor = float(requested["window_factor"])
            crossover_ms = float(requested["crossover_ms"])
            crossover_overrides = dict(
                requested["crossover_overrides"])
            source = ("voice-policy" if mode == "voice" else
                      "manual-window")
        native_requested = (
            crossover_ms > 0.0 or any(
                float(row.get("left_ms") or 0.0) +
                float(row.get("right_ms") or 0.0) > 0.0
                for row in crossover_overrides.values()
            )
        )
        return {
            "scope": "utterance",
            "requested_mode": requested["mode"],
            "requested_window_factor": float(
                requested["window_factor"]),
            "window_symmetric": bool(symmetric),
            "window_factor": float(factor),
            "requested_crossover_ms": float(
                requested["crossover_ms"]),
            "crossover_ms": float(crossover_ms),
            "crossover_overrides": crossover_overrides,
            "runtime": (
                "native-crossover" if native_requested else
                "stock-festival"
            ),
            "source": source,
            "legacy_active": legacy,
            "preserves_unit_selection": True,
            "preserves_phone_timing": True,
            "preserves_f0_targets": True,
        }

    def _voice_preamble(self, voicebank: str, lang: str = "",
                        legacy_joins: bool = False,
                        window_symmetric=None,
                        window_factor: float = 1.0) -> str:
        lines = []
        extra = (self.fcfg().get("extra_scheme") or "").strip()
        if extra:
            lines.append('(load "%s")' % win_to_wsl_path(extra))
        v = (self.fcfg().get("voices") or {}).get(voicebank)
        if str(voicebank) in BUILTIN_FESTIVAL_VOICES:
            language = normalize_language_code(lang)
            if language and language != "en":
                raise BackendError(
                    "The built-in Kal voice supports English only."
                )
            fn = self._guess_voice_fn(voicebank)
        elif isinstance(v, dict) and v.get("dir"):
            d = str(
                v.get("runtime_path") or win_to_wsl_path(v["dir"])
            ).rstrip("/")
            lines += ['(set! load-path (cons "%s/festvox" load-path))' % d,
                      '(set! load-path (cons "%s" load-path))' % d]
            if v.get("scm"):
                scm = v["scm"]
                scm_path = scm if scm.startswith("/") else f"{d}/{scm}"
                # The persistent native worker keeps Scheme definitions and
                # UniSyn database handles alive. Reloading a generated voice
                # here reset its db-name variable on every sentence; grouped
                # voices then reparsed the entire packed audio database for
                # every render. The voice fingerprint below restarts the
                # worker after any metadata/Scheme rebuild, so this guard is
                # both fast and safe from stale generated definitions.
                load_guard = "festvox_gui_voice_loaded_" + uuid.uuid5(
                    uuid.NAMESPACE_URL, scm_path
                ).hex
                lines.extend([
                    "(defvar %s nil)" % load_guard,
                    "(if (not %s)" % load_guard,
                    "    (begin",
                    '      (load "%s")' % scm_path,
                    "      (set! %s t)))" % load_guard,
                ])
            fn = v.get("voice") or self._guess_voice_fn(voicebank)
            compatibility = self.voice_compatibility(voicebank)
            language = normalize_language_code(lang)
            if language and compatibility.is_current:
                if not compatibility.supports(language):
                    raise BackendError(
                        "The selected generated voice does not support "
                        f"language '{language}'. Supported: "
                        + ", ".join(compatibility.supported_languages)
                    )
                fn = compatibility.voice_entry_points.get(language) or fn
            elif language == "en" and v.get("voice_en"):
                # Backward-compatible routing for old generated voices.
                fn = v["voice_en"]
        else:
            fn = self._guess_voice_fn(voicebank)
        lines.append("(defvar festvox_gui_legacy_joins nil)")
        lines.append("(set! festvox_gui_legacy_joins %s)" %
                     ("t" if legacy_joins else "nil"))
        lines.append("(%s)" % fn)
        # Festival's UniSyn synthesis type reads Param.unisyn.*, while many
        # historical voices only set an unused global `window_factor` symbol.
        # Apply this after voice activation so every WSL language/front end
        # receives the same join geometry.  Symmetric=1 is the exact pre-fix
        # analysis-period path exposed by Fault Mode.
        lines.append('(Param.set "unisyn.window_name" "hanning")')
        factor = max(1.0, min(1.25, float(window_factor)))
        factor_text = ("%.3f" % factor).rstrip("0").rstrip(".")
        if "." not in factor_text:
            factor_text += ".0"
        lines.append('(Param.set "unisyn.window_factor" %s)' % factor_text)
        symmetric = (
            self._use_symmetric_join_window(
                voicebank, legacy_joins=legacy_joins)
            if window_symmetric is None else bool(window_symmetric)
        )
        lines.append('(Param.set "unisyn.window_symmetric" %d)' %
                     (1 if symmetric else 0))
        return "\n".join(lines)

    @staticmethod
    def _native_unisyn_synth_type(join_settings: dict) -> str:
        """Scheme hook replacing only UniSyn's final waveform generator."""
        crossover_ms = max(0.0, min(
            100.0, float(join_settings.get("crossover_ms") or 0.0)))
        context_groups = (
            ("vowel", _CLS_VOWELS),
            ("stop_voiceless", _CLS_VOICELESS_STOPS),
            ("stop_voiced", _CLS_VOICED_STOPS),
            ("affricate_voiceless", _CLS_VOICELESS_AFFRICATES),
            ("affricate_voiced", _CLS_VOICED_AFFRICATES),
            ("fricative_voiceless", _CLS_VOICELESS_FRICS),
            ("fricative_voiced", _CLS_VOICED_FRICS),
            ("nasal", _CLS_NASALS),
            ("liquid", _CLS_LIQUIDS),
            ("glide", _CLS_GLIDES),
            ("silence", {"pau", "sil", "sp"}),
        )
        context_rows = {
            (phone, context)
            for context, phones in context_groups
            for phone in phones
        }
        # Open JTalk's moraic nasal is case-sensitive and distinct from /n/.
        context_rows.add(("N", "nasal"))
        context_text = " ".join(
            '("%s" "%s")' % (phone, context)
            for phone, context in sorted(context_rows)
        )
        overrides = []
        for raw_index, row in sorted(
                dict(join_settings.get("crossover_overrides") or {}).items(),
                key=lambda pair: int(pair[0])):
            overrides.append(
                "(%d %.3f %.3f)" % (
                    int(raw_index),
                    float(row["left_ms"]),
                    float(row["right_ms"]),
                )
            )
        override_text = " ".join(overrides)
        return (
            "(set! festvox_gui_join_contexts '(%s))\n" % context_text
            + "(define (festvox_gui_join_context phone rows)\n"
            + "  (cond\n"
            + "    ((null rows) nil)\n"
            + "    ((string-equal phone (car (car rows)))\n"
            + "     (car (cdr (car rows))))\n"
            + "    (t (festvox_gui_join_context phone (cdr rows)))))\n"
            + "(define (festvox_gui_apply_join_contexts segments)\n"
            + "  (if (null segments) nil\n"
            + "      (let ((context (festvox_gui_join_context\n"
            + "                      (item.feat (car segments) \"name\")\n"
            + "                      festvox_gui_join_contexts)))\n"
            + "        (begin\n"
            + "          (if context\n"
            + "              (item.set_feat (car segments)\n"
            + "               \"festvox_join_context_class\" context))\n"
            + "          (festvox_gui_apply_join_contexts"
            + " (cdr segments))))))\n"
            + "(set! festvox_gui_join_overrides '(%s))\n" % override_text
            + "(define (festvox_gui_apply_join_overrides units index)\n"
            + "  (if (null units) nil\n"
            + "      (let ((row (assoc index"
            + " festvox_gui_join_overrides)))\n"
            + "        (begin\n"
            + "          (if row\n"
            + "              (begin\n"
            + "                (item.set_feat (car units)"
            + " \"festvox_join_left_ms\""
            + " (car (cdr row)))\n"
            + "                (item.set_feat (car units)"
            + " \"festvox_join_right_ms\""
            + " (car (cdr (cdr row))))))\n"
            + "          (festvox_gui_apply_join_overrides"
            + " (cdr units) (+ index 1))))))\n"
            + "(defSynthType FestVoxUniSyn\n"
            + "  (defvar UniSyn_module_hooks nil)\n"
            + "  (Param.def \"unisyn.window_name\" \"hanning\")\n"
            + "  (Param.def \"unisyn.window_factor\" 1.0)\n"
            + "  (Parameter.def 'us_sigpr 'lpc)\n"
            + "  (apply_hooks UniSyn_module_hooks utt)\n"
            + "  (festvox_gui_apply_join_contexts"
            + " (utt.relation.items utt 'Segment))\n"
            + "  (us_get_diphones utt)\n"
            + "  (festvox_gui_apply_join_overrides"
            + " (utt.relation.items utt 'Unit) 0)\n"
            + "  (us_unit_concat utt)\n"
            + "  (if (not (member 'f0 (utt.relationnames utt)))\n"
            + "      (targets_to_f0 utt))\n"
            + "  (if (utt.relation.last utt 'Segment)\n"
            + "      (set! pm_end (+ (item.feat"
            + " (utt.relation.last utt 'Segment) \"end\") 0.02))\n"
            + "      (set! pm_end 0.02))\n"
            + "  (us_f0_to_pitchmarks utt 'f0 'TargetCoef pm_end)\n"
            + "  (us_mapping utt 'segment_single)\n"
            + "  (cond\n"
            + "    ((string-equal \"td_psola\""
            + " (Parameter.get 'us_sigpr))\n"
            + "     (us_tdpsola_synthesis utt 'analysis_period))\n"
            + "    (t (festvox_us_generate_wave utt"
            + " (Parameter.get 'us_sigpr))))\n"
            + "  utt)\n"
            + "(Param.set \"festvox.join_crossover_ms\" %.3f)\n"
            % crossover_ms
            + "(Parameter.set 'Synth_Method 'FestVoxUniSyn)\n"
        )

    @staticmethod
    def _dict_addenda(text: str, user_dict) -> str:
        """lex.add.entry lines overriding pronunciation for the words in `text`
        that are present in the loaded user dictionary. Only the utterance's own
        words are injected (cheap), so Festival still supplies full prosody and
        its own g2p for out-of-dictionary words. Fixes e.g. English 'in'
        (ax n -> ih n) from the bank's arpasing.yaml. The syllable is marked
        STRESSED so Festival's postlexical function-word reduction does not
        collapse the dict's full vowels back to schwa -- otherwise 'this'
        (dh ih s) reduces to dh ax s while content words like 'seeing' are
        left alone."""
        if not user_dict or not text:
            return ""
        import re as _re
        seen, lines = set(), []
        # Unicode letters are required for Asaxi romanization. Keep this
        # language-neutral so English contractions and numbered tokens retain
        # the same path while accented Asaxi words are no longer skipped.
        for w in _re.findall(
                r"[^\W_]+(?:[’'.\-][^\W_]+)*", text,
                flags=_re.UNICODE):
            wl = w.lower().strip(".-")
            if not wl or wl in seen:
                continue
            seen.add(wl)
            ph = user_dict.get(wl)
            if ph:
                esc = wl.replace("\\", "\\\\").replace('"', '\\"')
                lines.append("(lex.add.entry '(\"%s\" nil (((%s) 1))))"
                             % (esc, " ".join(ph)))
        if not lines:
            return ""
        # Keep the dictionary's vowels VERBATIM while still running the normal
        # text pipeline (so Festival's full duration/prosody is used -- no
        # phone-path timing tradeoff): switch off the postlexical function-word
        # vowel reduction that was collapsing 'this' dh ih s -> dh ax s even
        # with the addendum. Guarded, so voices lacking the table are unaffected.
        pre = ("(if (symbol-bound? 'postlex_vowel_reduce_table)\n"
               "    (set! postlex_vowel_reduce_table nil))\n")
        return pre + "\n".join(lines) + "\n"

    @staticmethod
    def _pitch_override(pitch, fall, monotone: bool = False,
                        lang: str = "") -> str:
        """Scheme that recenters whatever intonation model the active voice
        uses: DuffInt voices via duffint_params, LR-model voices (the _en
        front end / kal) via int_lr_params' target mean/std.
        monotone=True forces a perfectly flat F0 at `pitch` for EVERY voice
        (Int_Method -> DuffInt with start==end), leaving durations untouched."""
        if monotone:
            p = float(pitch) if pitch else 160.0
            # Force a flat F0 for EVERY voice. DuffInt places no accents, and
            # Int_Targets_Default builds the target line straight from
            # duffint_params. Setting Int_Method alone is NOT enough for the
            # English (_en / kal) voice: its Int_Target_Method stays the F2B LR
            # model (int_lr_params, std 22 Hz) and keeps generating a contour,
            # so we pin the target method to Default and zero any LR spread too.
            return ("(Parameter.set 'Int_Method 'DuffInt)\n"
                    "(Parameter.set 'Int_Target_Method Int_Targets_Default)\n"
                    "(set! duffint_params '((start %.1f) (end %.1f)))\n"
                    "(if (symbol-bound? 'int_lr_params)\n"
                    "    (set! int_lr_params '((target_f0_mean %.1f)"
                    " (target_f0_std 0.0)\n"
                    "                          (model_f0_mean 170)"
                    " (model_f0_std 34))))\n"
                    % (p, p, p))
        if not pitch:
            return ""
        pitch = float(pitch)
        spread_st = pitch_domain.fall_percent_to_span_semitones(fall)
        start_hz = pitch_domain.semitone_offset(pitch, spread_st)
        end_hz = pitch_domain.semitone_offset(pitch, -spread_st)
        lr_st = 12.0 * math.log2(
            1.0 + min(40.0, max(0.0, float(fall or 0.0))) / 100.0)
        lr_std = max(0.0, min(
            pitch * 0.24,
            pitch_domain.semitone_offset(pitch, lr_st) - pitch,
        ))
        if str(lang).lower() in ("en", "eng", "english"):
            # Fall=0 means no additional spread, not monotone English. Scale
            # the loaded voice's own LR variance to the requested mean pitch.
            # SIOD supports ``let`` but not ``let*``, so read all source values
            # in one binding form before replacing the parameter list.
            return (
                "(if (symbol-bound? 'duffint_params)\n"
                "    (set! duffint_params '((start %.1f) (end %.1f))))\n"
                % (start_hz, end_hz)
                + "(if (symbol-bound? 'int_lr_params)\n"
                "    (let ((native-mean (car (cdr (assoc"
                " 'target_f0_mean int_lr_params))))\n"
                "          (native-std (car (cdr (assoc"
                " 'target_f0_std int_lr_params))))\n"
                "          (model-mean (car (cdr (assoc"
                " 'model_f0_mean int_lr_params))))\n"
                "          (model-std (car (cdr (assoc"
                " 'model_f0_std int_lr_params)))))\n"
                "      (let ((scaled-std\n"
                "             (if (> native-mean 0)\n"
                "                 (* native-std (/ %.1f native-mean))\n"
                "                 native-std)))\n"
                "        (set! int_lr_params\n"
                "              (list (list 'target_f0_mean %.1f)\n"
                "                    (list 'target_f0_std\n"
                "                          (if (> scaled-std %.1f)\n"
                "                              scaled-std %.1f))\n"
                "                    (list 'model_f0_mean model-mean)\n"
                "                    (list 'model_f0_std model-std))))))\n"
                % (pitch, pitch, lr_std, lr_std))
        return (
            "(if (symbol-bound? 'duffint_params)\n"
            "    (set! duffint_params '((start %.1f) (end %.1f))))\n"
            % (start_hz, end_hz)
            + "(if (symbol-bound? 'int_lr_params)\n"
            "    (set! int_lr_params '((target_f0_mean %.1f)"
            " (target_f0_std %.1f)\n"
            "                          (model_f0_mean 170)"
            " (model_f0_std 34))))\n"
            % (pitch, lr_std))

    def _synth_scheme(self, body: str, voicebank: str, speed: float,
                      wav_wsl: str, seg_wsl: str, lang: str = "",
                      pitch=None, fall=None, monotone: bool = False,
                      addenda: str = "", legacy_joins: bool = False,
                      explicit_durations=None, join_settings=None) -> str:
        dur_stretch = 1.0 / speed if speed else 1.0
        render_faults = {
            "legacy_joins": bool(legacy_joins),
            "_join_settings": dict(join_settings or {}),
        }
        resolved_joins = self.resolve_join_settings(
            voicebank, fault_mode=render_faults,
            explicit_durations=explicit_durations)
        native_synth = (
            self._native_unisyn_synth_type(resolved_joins)
            if resolved_joins["runtime"] == "native-crossover"
            else ""
        )
        return (
            self._voice_preamble(
                voicebank, lang, legacy_joins=legacy_joins,
                window_symmetric=resolved_joins["window_symmetric"],
                window_factor=resolved_joins["window_factor"]) + "\n"
            + native_synth
            + addenda                                   # user-dict pronunciations
            + self._pitch_override(pitch, fall, monotone, lang=lang)
            + "(Parameter.set 'Duration_Stretch %.4f)\n" % dur_stretch
            + body
            # Preserve UniSyn's rendered pitchmarks and exact frame-map
            # handoffs.  The diagnostic consumes stdout only; synthesis and
            # the generated voice are unchanged.
            + "(define (festvox_gui_dump_target_pm track index count)\n"
            + "  (if (< index count)\n"
            + "      (begin\n"
            + "        (print (list 'GUIPM index"
            + " (track.get_time track index)))\n"
            + "        (festvox_gui_dump_target_pm track (+ index 1)"
            + " count))))\n"
            + "(define (festvox_gui_dump_map_targets source targets)\n"
            + "  (if (null targets) nil\n"
            + "      (begin\n"
            + "        (print (list 'GUIMAP"
            + " (item.feat source \"index\")"
            + " (item.feat source \"end\")"
            + " (item.feat (car targets) \"index\")"
            + " (item.feat (car targets) \"end\")))\n"
            + "        (festvox_gui_dump_map_targets source"
            + " (cdr targets)))))\n"
            + "(define (festvox_gui_dump_map_sources sources)\n"
            + "  (if (null sources) nil\n"
            + "      (begin\n"
            + "        (festvox_gui_dump_map_targets (car sources)"
            + " (item.daughters (car sources)))\n"
            + "        (festvox_gui_dump_map_sources (cdr sources)))))\n"
            + "(define (festvox_gui_dump_unit_frames units index first)\n"
            + "  (if (null units) nil\n"
            + "      (let ((count (item.feat (car units)"
            + " \"num_frames\")))\n"
            + "        (begin\n"
            + "          (print (list 'GUIUFRAME index count first"
            + " (+ first count)))\n"
            + "          (festvox_gui_dump_unit_frames (cdr units)"
            + " (+ index 1) (+ first count))))))\n"
            + "(if (member 'TargetCoef (utt.relationnames u))\n"
            + "    (let ((target-track (item.feat"
            + " (utt.relation.first u 'TargetCoef) \"coefs\")))\n"
            + "      (festvox_gui_dump_target_pm target-track 0"
            + " (track.num_frames target-track))))\n"
            + "(if (and (symbol-bound? 'map_to_relation)\n"
            + "         (member 'SourceCoef (utt.relationnames u))\n"
            + "         (member 'TargetCoef (utt.relationnames u))\n"
            + "         (member 'US_map (utt.relationnames u))\n"
            + "         (member 'Unit (utt.relationnames u)))\n"
            + "    (begin\n"
            + "      (map_to_relation u 'SourceCoef 'TargetCoef 'lmap)\n"
            + "      (festvox_gui_dump_map_sources"
            + " (utt.relation.items u 'lmap))\n"
            + "      (festvox_gui_dump_unit_frames"
            + " (utt.relation.items u 'Unit) 0 0)))\n"
            + '(utt.save.wave u "%s" \'riff)\n' % wav_wsl
            + '(utt.save.segs u "%s")\n' % seg_wsl
            # dump the F0 target points so a later phoneme re-render can
            # keep this utterance's pitch contour (parsed as GUITGT below)
            + "(if (member 'Target (utt.relationnames u))\n"
            + "    (mapcar (lambda (targ)\n"
            + "      (mapcar (lambda (tt)\n"
            + "        (print (list 'GUITGT (item.feat tt \"pos\")"
            + " (item.feat tt \"f0\"))))\n"
             + "        (item.daughters targ)))\n"
             + "      (utt.relation.items u 'Target)))\n"
             # Report the exact outgoing unit selected by UniSyn. The GUI uses
             # this for visible Auto/take labels; explicit overrides use the
             # same feature, so reported and rendered choices cannot diverge.
             + "(define (festvox_gui_dump_units items index)\n"
             + "  (if (null items) nil\n"
             + "      (let ((unit (item.feat (car items)"
             + " \"us_diphone_left\")))\n"
             + "        (print (list 'GUIUNIT index"
             + " (item.name (car items)) unit))\n"
             + "        (festvox_gui_dump_units (cdr items) (+ index 1)))))\n"
             + "(festvox_gui_dump_units (utt.relation.items u 'Segment) 0)\n"
        )

    def _synth_common(self, body: str, voicebank: str, speed: float,
                      text: str, lang: str, phones_used=None,
                      pitch=None, fall=None, monotone: bool = False,
                      addenda: str = "", ground_truth_targets=None,
                      pitch_override=None, intonation_blocks=None,
                      pitch_mode: str = "", unit_overrides=None,
                      fault_mode=None, explicit_durations=None) -> Synthesis:
        import uuid
        if not voicebank:
            raise BackendError("No Festival voice selected.\n"
                               "Voicebank menu > Scan Festival voices (WSL) "
                               "or Add Festival voice folder...")
        ex = self._exchange_dir()
        tag = uuid.uuid4().hex
        wav = os.path.join(ex, f"out_{tag}.wav")
        seg = os.path.join(ex, f"out_{tag}.seg")
        scheme = self._synth_scheme(body, voicebank, speed,
                                    win_to_wsl_path(wav), win_to_wsl_path(seg),
                                    lang=lang, pitch=pitch, fall=fall,
                                    monotone=monotone, addenda=addenda,
                                    legacy_joins=bool(
                                        (fault_mode or {}).get(
                                            "legacy_joins")),
                                    explicit_durations=explicit_durations,
                                    join_settings=dict(
                                        (fault_mode or {}).get(
                                            "_join_settings") or {}))
        resolved_joins = self.resolve_join_settings(
            voicebank, fault_mode=fault_mode,
            explicit_durations=explicit_durations)
        out = self._run_scheme(scheme)
        try:
            if not os.path.exists(wav):
                raise BackendError(
                    "Festival produced no audio.\nOutput:\n"
                    + (out.strip()[-1200:] or "(silent)"))
            samples, sr = read_wav(wav)
            segs = parse_segs(seg) if os.path.exists(seg) else []
        finally:
            for p in (wav, seg):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if not segs:
            segs = [Segment("utt", 0.0, len(samples) / sr)]
        import re as _re
        targets = [(float(a), float(b)) for a, b in
                   _re.findall(r"\(GUITGT ([0-9.eE+-]+) ([0-9.eE+-]+)\)", out)]
        selected_units = {}
        for match in _re.finditer(
                r'\(GUIUNIT ([0-9]+) "[^"]*" (?:"([^"]*)"|0)\)', out):
            if match.group(2):
                selected_units[int(match.group(1))] = match.group(2)
        rendered_diagnostic = parse_unisyn_render_diagnostics(out, segs)
        generated = (list(targets) if ground_truth_targets is None else
                     [(float(t), float(f)) for t, f in ground_truth_targets])
        # UniSyn logs every bank gap it papered over -- surface them just
        # like the diphone engine's "skipped" report (these are the blips
        # or silences you hear where a unit is missing)
        skipped = _re.findall(r"using default diphone \S+ for (\S+)", out)
        warn = None
        low = out.lower()
        if "warning" in low or ("error" in low and "default diphone" not in low):
            warn = "Festival said: " + out.strip()[-300:]
        sample_values = np.asarray(samples, np.float32)
        # Post-render PCM join rewriting was retired after listening tests
        # exposed crackle and broad voiced-period corruption.  Keep the
        # provenance field for project compatibility, but synthesis now
        # returns Festival's waveform unchanged.  Join analysis is read-only.
        join_repairs = []
        return Synthesis(sample_values, sr, segs,
                         text=text, lang=lang, voicebank=voicebank,
                         phones=list(phones_used or
                                     [s.phone for s in segs
                                      if s.phone not in ("pau", "#")]),
                         diphones=[], skipped=skipped, targets=targets,
                         generated_targets=generated,
                         pitch_override=list(pitch_override or []),
                         intonation_blocks=[dict(b) for b in
                                            (intonation_blocks or [])],
                         pitch_mode=str(pitch_mode or ""),
                         unit_overrides={int(k): str(v) for k, v in
                                         dict(unit_overrides or {}).items()},
                         selected_units=selected_units,
                         target_pitchmarks=list(
                             rendered_diagnostic["target_pitchmarks"]),
                         splice_records=[dict(row) for row in
                                         rendered_diagnostic["splice_records"]],
                         frame_trajectory_records=[
                             dict(row) for row in rendered_diagnostic[
                                 "frame_trajectory_records"]],
                         join_repairs=[dict(row) for row in join_repairs],
                         join_settings=dict(resolved_joins),
                         warning=warn)

    def phonemize(self, words, voicebank, lang=""):
        """{word_lower: [phones]} from the active Festival voice (its lexicon +
        letter-to-sound for out-of-dictionary words). Used to g2p the words a
        user dictionary does NOT cover, so dictionary synthesis can run through
        the verbatim phone path -- which skips the postlexical function-word
        reduction that was collapsing e.g. 'this' (dh ih s) to dh ax s."""
        seen = []
        for w in words:
            wl = str(w).lower().strip()
            if wl and wl not in seen:
                seen.append(wl)
        if not seen:
            return {}
        def esc(s):
            return s.replace("\\", "\\\\").replace('"', '\\"')
        body = "".join(
            '(let ((e (lex.lookup "%s" nil)))\n'
            '  (print (list \'GUIPRON "%s"\n'
            '    (apply append (mapcar (lambda (s) (car s)) (car (cddr e)))))))\n'
            % (esc(w), esc(w)) for w in seen)
        out = self._run_scheme(
            self._voice_preamble(voicebank, lang) + "\n" + body)
        import re as _re
        res = {}
        for m in _re.finditer(r'\(GUIPRON "([^"]*)" \(([^)]*)\)\)', out):
            res[m.group(1)] = [p for p in m.group(2).split() if p]
        return res

    def _asaxi_dictionary(self):
        path = self.cfg.get("asaxi_synthesis_dictionary") \
            or asaxi_prosody_domain.DEFAULT_DICTIONARY_PATH
        try:
            return asaxi_prosody_domain.load_dictionary(path)
        except (OSError, ValueError, TypeError) as error:
            raise BackendError(
                "The generated Asaxi synthesis dictionary is unavailable or "
                "invalid. Run "
                "99_Tools/vocab_forge/build_asaxi_synthesis_dictionary.py.\n"
                f"{error}"
            ) from error

    def _capitalized_asaxi_pronunciations(
            self, text, voicebank, user_dict=None):
        """Resolve full-cap Asaxi terms through the voice's English frontend."""

        terms = asaxi_frontend_domain.capitalized_terms_in_text(text)
        if not terms:
            return {}
        explicit = {
            asaxi_frontend_domain.normalize_word(key)
            for key in dict(user_dict or {})
            if str(key).strip()
        }
        resolved = {}
        pending = []
        for term in terms:
            key = asaxi_frontend_domain.normalize_word(term)
            if key in explicit:
                continue
            override = (
                asaxi_frontend_domain
                .CAPITALIZED_ENGLISH_PRONUNCIATION_OVERRIDES
                .get(term.upper())
            )
            if override:
                resolved[key] = tuple(override)
            else:
                pending.append(term)
        if pending:
            english = self.phonemize(pending, voicebank, lang="en")
            missing = []
            for term in pending:
                key = asaxi_frontend_domain.normalize_word(term)
                phones = tuple(
                    str(phone)
                    for phone in english.get(key, ())
                    if str(phone)
                )
                if phones:
                    resolved[key] = phones
                else:
                    missing.append(term)
            if missing:
                raise BackendError(
                    "English G2P could not pronounce capitalized Asaxi "
                    "term(s): "
                    + ", ".join(repr(term) for term in missing)
                    + ". Add a Dictionary pronunciation override."
                )
        return resolved

    def _asaxi_seed(
            self, text, voicebank, speed, pitch, fall, monotone,
            faults, dictionary, pronunciations, user_dict,
            mora_index_offset=0, mora_tone_overrides=None,
            mora_pitch_offsets_cents=None):
        overrides = dict(user_dict or {})
        try:
            plan = asaxi_prosody_domain.analyze_utterance(
                text,
                dictionary,
                phone_overrides=overrides,
                capitalized_pronunciations=(
                    self._capitalized_asaxi_pronunciations(
                        text, voicebank, overrides
                    )
                ),
            )
        except (ValueError, TypeError) as error:
            raise BackendError(
                "Asaxi pronunciation/accent planning failed:\n%s" % error
            ) from error
        override_keys = {
            str(key).strip().casefold()
            for key in overrides
            if str(key).strip()
        }
        phone_fallback = \
            asaxi_phone_fallback_domain.adapt_plan_for_inventory(
                plan,
                self.voice_metadata(voicebank),
                protected_word_indices={
                    word.index
                    for word in plan.words
                    if word.surface.strip().casefold() in override_keys
                },
            )
        plan = phone_fallback.plan
        combined = dict(pronunciations)
        # The dictionary contains lexical headwords, while the Asaxi planner
        # also resolves inflected and morphologically segmented surface forms.
        # Festival cannot repeat that inference on its own, so pass the
        # planner's exact per-utterance pronunciations into the text seed.
        combined.update({
            asaxi_frontend_domain.normalize_word(word.surface):
            list(word.phones)
            for word in plan.words
            if word.surface and word.phones
        })
        # Explicit project/user pronunciations remain the final authority.
        combined.update({
            asaxi_frontend_domain.normalize_word(word): phones
            for word, phones in overrides.items()
        })
        escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
        seed = self._synth_common(
            '(set! u (SynthText "%s"))\n' % escaped,
            voicebank,
            speed,
            text,
            "asaxi",
            pitch=pitch,
            fall=fall,
            monotone=monotone,
            addenda=self._dict_addenda(text, combined),
            fault_mode=faults,
        )
        try:
            duration_plan = asaxi_duration_domain.plan_durations(
                plan,
                seed.segments,
                speed=float(speed or 1.0),
            )
            old_segments = list(seed.segments)
            retimed_segments = []
            cursor = 0.0
            for index, (phone, duration) in enumerate(
                    duration_plan.entries):
                old = old_segments[index]
                end = cursor + float(duration)
                retimed_segments.append(Segment(
                    str(phone),
                    cursor,
                    end,
                    old.uid,
                    old.timing_role,
                ))
                cursor = end
            seed.segments = retimed_segments
            seed.asaxi_prosody = {
                "duration_plan": duration_plan.to_dict(),
                "phone_fallback_model_id": (
                    asaxi_phone_fallback_domain
                    .ASAXI_PHONE_FALLBACK_MODEL_ID
                ),
                "phone_fallback_available_diphone_count": (
                    phone_fallback.available_diphone_count
                ),
                "phone_fallbacks": [
                    record.to_dict()
                    for record in phone_fallback.records
                ],
            }
            global_offsets = {}
            for key, value in dict(
                    mora_pitch_offsets_cents or {}).items():
                try:
                    global_offsets[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            local_offsets = {
                mora.index: global_offsets[
                    int(mora_index_offset) + mora.index]
                for mora in plan.moras
                if int(mora_index_offset) + mora.index in global_offsets
            }
            global_tones = {}
            for key, value in dict(mora_tone_overrides or {}).items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    continue
                tone = str(value or "").strip().upper()
                if tone in {"H", "L"}:
                    global_tones[index] = tone
            local_tones = {
                mora.index: global_tones[
                    int(mora_index_offset) + mora.index]
                for mora in plan.moras
                if int(mora_index_offset) + mora.index in global_tones
            }
            targets, pitch_diagnostics = \
                asaxi_prosody_domain.targets_for_segments(
                plan,
                seed.segments,
                base_pitch_hz=float(pitch or 160.0),
                fall_percent=float(fall or 0.0),
                mora_tone_overrides=local_tones,
                mora_pitch_offsets_cents=local_offsets,
            )
            diagnostics = (
                tuple(phone_fallback.diagnostics)
                + tuple(duration_plan.diagnostics)
                + tuple(pitch_diagnostics)
            )
        except (ValueError, TypeError) as error:
            raise BackendError(
                "Asaxi duration/accent planning could not be aligned to "
                "Festival's phones:\n%s" % error
            ) from error
        seed.targets = list(targets)
        seed.generated_targets = list(targets)
        return seed, plan, diagnostics

    @staticmethod
    def _asaxi_metadata(
            plans, diagnostics, segments=None, duration_plans=None,
            phone_fallbacks=None, pitch_realization=None):
        rows = [plan.to_dict() for plan in plans]
        duration_rows = [
            dict(row) for row in (duration_plans or [])
            if isinstance(row, Mapping)
        ]
        mora_rows = []
        final_diagnostics = list(diagnostics)
        if segments is not None:
            aligned, alignment_diagnostics = \
                asaxi_prosody_domain.rendered_morae(plans, segments)
            mora_rows = [mora.to_dict() for mora in aligned]
            final_diagnostics.extend(alignment_diagnostics)
        deduplicated = []
        seen = set()
        for diagnostic in final_diagnostics:
            key = (
                diagnostic.code,
                diagnostic.message,
                diagnostic.severity,
                diagnostic.word_index,
            )
            if key not in seen:
                seen.add(key)
                deduplicated.append(diagnostic)
        pitch_trace = (
            pitch_realization.trace_dict()
            if pitch_realization is not None
            else {}
        )
        return {
            "schema_version": 1,
            "kind": "asaxi_sentence_prosody",
            "dictionary_ruleset": (
                plans[0].dictionary_ruleset if plans else ""
            ),
            "duration_model": "moraic_rules",
            "duration_model_id": (
                duration_rows[0].get("model_id")
                if duration_rows
                else asaxi_duration_domain.ASAXI_DURATION_MODEL_ID
            ),
            "duration_plans": duration_rows,
            "phone_fallback_model_id": (
                asaxi_phone_fallback_domain.ASAXI_PHONE_FALLBACK_MODEL_ID
            ),
            "phone_fallbacks": [
                dict(row) for row in (phone_fallbacks or [])
                if isinstance(row, Mapping)
            ],
            "pitch_model_id": str(
                pitch_trace.get("model_id") or ""),
            "pitch_model_version": int(
                pitch_trace.get("model_version") or 0),
            "prosody_trace": pitch_trace,
            "phrase_count": len(rows),
            "word_count": sum(len(plan.words) for plan in plans),
            "mora_count": sum(len(plan.moras) for plan in plans),
            "phrases": rows,
            "moras": mora_rows,
            "rendered_phones": [
                str(segment.phone if hasattr(segment, "phone") else segment[0])
                for segment in (segments or [])
            ],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in deduplicated
            ],
        }

    def _synth_asaxi(
            self, text, voicebank, speed, pitch, fall, monotone,
            user_dict, faults, pitch_targets=None,
            ground_truth_targets=None, intonation_blocks=None,
            pitch_mode="", asaxi_tone_overrides=None,
            asaxi_pitch_offsets_cents=None):
        dictionary = self._asaxi_dictionary()
        pronunciations = dictionary.pronunciations()
        chunks = text_phrase_chunks(text)
        seeds = []
        plans = []
        diagnostics = []
        duration_plans = []
        phone_fallbacks = []
        mora_index_offset = 0
        for chunk, _mark in chunks:
            seed, plan, local_diagnostics = self._asaxi_seed(
                chunk, voicebank, speed, pitch, fall, monotone,
                faults, dictionary, pronunciations, user_dict,
                mora_index_offset=mora_index_offset,
                mora_tone_overrides=asaxi_tone_overrides,
                mora_pitch_offsets_cents=asaxi_pitch_offsets_cents,
            )
            seeds.append(seed)
            plans.append(plan)
            diagnostics.extend(local_diagnostics)
            duration_plan = dict(
                getattr(seed, "asaxi_prosody", {}).get(
                    "duration_plan") or {}
            )
            if duration_plan:
                duration_plans.append(duration_plan)
            phone_fallbacks.extend(
                dict(row) for row in
                getattr(seed, "asaxi_prosody", {}).get(
                    "phone_fallbacks", ()
                )
                if isinstance(row, Mapping)
            )
            mora_index_offset += len(plan.moras)

        if len(seeds) > 1:
            entries, generated, old_segments = merge_phrase_plans(
                seeds,
                [mark for _chunk, mark in chunks],
                speed,
                single_pause=bool(faults.get("single_pause")),
                phrase_pauses_ms=self.cfg.get("phrase_pauses_ms"),
            )
        else:
            old_segments = list(seeds[0].segments)
            entries = [
                (segment.phone, segment.dur)
                for segment in old_segments
            ]
            generated = list(seeds[0].targets)
            if faults.get("single_pause"):
                entries = collapse_pause_runs(entries)
            else:
                entries = split_internal_pauses(entries)

        original_segments = old_segments
        original_durations = [
            segment.dur for segment in original_segments
        ]
        if faults.get("disable_phone_timing"):
            entries = equalize_phone_durations(
                entries, phone_dur=0.10 / max(0.25, float(speed or 1.0))
            )
        if [duration for _phone, duration in entries] != original_durations:
            generated = remap_targets(
                generated,
                original_segments,
                [duration for _phone, duration in entries],
            )

        disable_prosody = bool(faults.get("disable_prosody"))
        final_segments = segments_from_durations(entries)
        pitch_realization = None
        if disable_prosody:
            generated = []
        else:
            try:
                pitch_realization, final_pitch_diagnostics = \
                    asaxi_prosody_domain.realize_pitch_for_plans(
                        plans,
                        final_segments,
                        base_pitch_hz=float(pitch or 160.0),
                        fall_percent=float(fall or 0.0),
                        mora_tone_overrides=asaxi_tone_overrides,
                        mora_pitch_offsets_cents=(
                            asaxi_pitch_offsets_cents
                        ),
                    )
            except (ValueError, TypeError) as error:
                raise BackendError(
                    "Asaxi sentence-level pitch planning could not be "
                    "aligned to the final phone timing:\n%s" % error
                ) from error
            generated = list(pitch_realization.targets)
            diagnostics.extend(final_pitch_diagnostics)
        previous_targets = generated
        manual_targets = list(pitch_targets or [])
        blocks = list(intonation_blocks or [])
        result = self.synth_phones(
            [phone for phone, _duration in entries],
            voicebank,
            speed=1.0,
            text=text,
            lang="asaxi",
            seg_durs=entries,
            old_segments=final_segments,
            prev_targets=previous_targets,
            pitch=pitch,
            fall=fall,
            monotone=monotone,
            fault_mode=faults,
            pitch_targets=manual_targets or None,
            ground_truth_targets=(
                list(ground_truth_targets)
                if ground_truth_targets is not None
                else (generated or None)
            ),
            intonation_blocks=blocks or None,
            pitch_mode=pitch_mode,
        )
        result.asaxi_prosody = self._asaxi_metadata(
            plans,
            diagnostics,
            result.segments,
            duration_plans=duration_plans,
            phone_fallbacks=phone_fallbacks,
            pitch_realization=pitch_realization,
        )
        if faults.get("disable_phone_timing"):
            result.asaxi_prosody["duration_fault_override"] = \
                "equal_phone_timing"
        notable = [
            item.message
            for item in diagnostics
            if item.severity in {"warning", "error"}
        ]
        if notable:
            note = "; ".join(notable[:2])
            result.warning = (
                result.warning + "; " + note
                if result.warning else note
            )
        return result

    # -- public synthesis interface (mirrors DiphoneBackend) -----------------------
    def synth(self, text: str, lang: str, voicebank: str,
              speed: float = 1.0, pitch=None, fall=None,
              monotone: bool = False, user_dict=None,
              fault_mode=None, pitch_targets=None,
              ground_truth_targets=None, intonation_blocks=None,
              pitch_mode: str = "",
              asaxi_tone_overrides=None,
              asaxi_pitch_offsets_cents=None) -> Synthesis:
        if not text or not text.strip():
            raise BackendError("No text to synthesize.")
        text = normalize_synthesis_text(text, lang)
        faults = fault_mode or {}
        if normalize_language_code(lang) == "asaxi":
            return self._synth_asaxi(
                text,
                voicebank,
                speed,
                pitch,
                fall,
                monotone,
                user_dict,
                faults,
                pitch_targets=pitch_targets,
                ground_truth_targets=ground_truth_targets,
                intonation_blocks=intonation_blocks,
                pitch_mode=pitch_mode,
                asaxi_tone_overrides=asaxi_tone_overrides,
                asaxi_pitch_offsets_cents=asaxi_pitch_offsets_cents,
            )
        chunks = text_phrase_chunks(text)
        if len(chunks) > 1:
            seeds = []
            for chunk, _mark in chunks:
                esc = chunk.replace("\\", "\\\\").replace('"', '\\"')
                body = '(set! u (SynthText "%s"))\n' % esc
                seed = self._synth_common(
                    body, voicebank, speed, chunk, lang,
                    pitch=pitch, fall=fall, monotone=monotone,
                    addenda=self._dict_addenda(chunk, user_dict),
                    fault_mode=faults)
                if lang in {"ja", "jp"}:
                    seed = retime_japanese_synthesis_pauses(
                        seed, chunk, speed,
                        self.cfg.get("phrase_pauses_ms"))
                seeds.append(seed)
            entries, generated, old_segments = merge_phrase_plans(
                seeds, [mark for _, mark in chunks], speed,
                single_pause=bool(faults.get("single_pause")),
                phrase_pauses_ms=self.cfg.get("phrase_pauses_ms"))
            if faults.get("single_pause"):
                entries = collapse_pause_runs(entries)
                # The target remap below uses this normalized timeline.
                old_segments = segments_from_durations(entries)
            no_learned = (lang == "en" and
                          bool(faults.get("disable_prosody")))
            no_phone_rules = (lang != "en" and
                              bool(faults.get("disable_phone_timing")))
            if no_learned or no_phone_rules:
                entries = equalize_phone_durations(
                    entries, phone_dur=0.10 / max(0.25, speed))
            displayed_ground = remap_targets(
                generated, old_segments, [d for _, d in entries])
            if (displayed_ground and
                    not faults.get("disable_f0_correction")):
                displayed_ground = anchor_phrase_targets(
                    entries, displayed_ground, float(pitch or 160.0))
            return self.synth_phones(
                [p for p, _ in entries], voicebank, speed=1.0,
                text=text, lang=lang, seg_durs=entries,
                old_segments=old_segments,
                prev_targets=([] if no_learned else generated),
                pitch=pitch, fall=fall, monotone=monotone,
                fault_mode=faults,
                ground_truth_targets=(displayed_ground or None))

        esc = text.replace("\\", "\\\\").replace('"', '\\"')
        body = '(set! u (SynthText "%s"))\n' % esc
        seed = self._synth_common(
            body, voicebank, speed, text, lang,
            pitch=pitch, fall=fall, monotone=monotone,
            addenda=self._dict_addenda(text, user_dict),
            fault_mode=faults)
        original = [(s.phone, s.dur) for s in seed.segments]
        entries = (
            retime_japanese_inline_pauses(
                original, text, speed,
                self.cfg.get("phrase_pauses_ms"))
            if lang in {"ja", "jp"} else original
        )
        no_learned_prosody = (lang == "en" and
                              bool(faults.get("disable_prosody")))
        no_phone_rules = (lang != "en" and
                          bool(faults.get("disable_phone_timing")))
        if no_learned_prosody or no_phone_rules:
            entries = equalize_phone_durations(
                entries, phone_dur=0.10 / max(0.25, speed))
        if not faults.get("single_pause"):
            entries = split_internal_pauses(
                entries, lead_pause=0.12 / max(0.25, speed))
        else:
            # Explicit [pau] can acquire adjacent pause segments from a text
            # front end. Fault mode promises one logical pause, regardless of
            # how the front end represented that boundary.
            entries = collapse_pause_runs(entries)
        if (entries == original and not no_learned_prosody
                and faults.get("disable_f0_correction")
                and not faults.get("pitch_glitch")
                and not self.unit_alternatives(voicebank)):
            return seed
        # The durations already include Duration_Stretch from the text pass.
        # Rebuild as explicit Segments so phrase-edge F0 correction, faults and
        # independently resizable pauses also apply to ordinary text mode.
        displayed_ground = remap_targets(
            seed.generated_targets or seed.targets, seed.segments,
            [d for _, d in entries])
        if displayed_ground and not faults.get("disable_f0_correction"):
            displayed_ground = anchor_phrase_targets(
                entries, displayed_ground, float(pitch or 160.0))
        return self.synth_phones(
            [p for p, _ in entries], voicebank, speed=1.0, text=text, lang=lang,
            seg_durs=entries, old_segments=seed.segments,
            prev_targets=([] if no_learned_prosody else seed.targets),
            pitch=pitch, fall=fall,
            monotone=monotone, fault_mode=faults,
            ground_truth_targets=(displayed_ground or None))

    def synth_phones(self, phones: List[str], voicebank: str,
                     speed: float = 1.0, text: str = "",
                     lang: str = "", seg_durs=None, old_segments=None,
                     prev_targets=None, pitch=None, fall=None,
                     monotone: bool = False, fault_mode=None,
                     pitch_targets=None, ground_truth_targets=None,
                     intonation_blocks=None, pitch_mode: str = "",
                     unit_overrides=None,
                     preserve_pitch_register: bool = False) -> Synthesis:
        """Explicit phone list re-render.

        Without timing info: a Festival `Phones` utterance (SayPhones path;
        default durations, monotone F0). WITH `seg_durs` [(phone, dur), ...
        including the edge paus] the utterance is built as a `Segments`
        utterance instead: the given durations are used verbatim and the F0
        contour captured from the previous render (`prev_targets` +
        `old_segments`) is re-applied -- so editing one phoneme keeps the
        original prosody instead of collapsing to monotone."""
        import re as _re

        metadata = self.voice_metadata(voicebank)
        compatibility = self.voice_compatibility(voicebank)
        declared_phones = set(str(item) for item in (
            metadata.get("phones") or ()
        ))
        if compatibility.is_current and declared_phones:
            requested = {
                str(phone).strip() for phone in phones
                if str(phone).strip() not in {"pau", "sil"}
            }
            unknown = sorted(requested - declared_phones)
            if unknown:
                raise BackendError(
                    "These phones are outside the selected voice "
                    f"configuration: {', '.join(unknown)}"
                )

        def _chk(p):
            if not _re.fullmatch(r"[A-Za-z0-9_@:~#]+", p):
                raise BackendError(f"'{p}' is not a valid Festival phone name.")

        if seg_durs:
            entries = [(str(p).strip(), max(0.01, float(d)))
                       for p, d in seg_durs if str(p).strip()]
            if not entries:
                raise BackendError("Phone list is empty.")
            if entries[0][0] != "pau":
                entries.insert(0, ("pau", 0.12))
            if entries[-1][0] != "pau":
                entries.append(("pau", 0.12))
            if (fault_mode or {}).get("single_pause"):
                entries = collapse_pause_runs(entries)
            else:
                entries = split_edge_pauses(
                    entries, guard_pause=0.08 / max(0.25, float(speed or 1.0)))
            for p, _ in entries:
                _chk(p)
            resolution = resolve_voice_special_phones(
                [phone for phone, _duration in entries],
                metadata,
                voicebank=voicebank,
                available_diphones=(metadata.get("index") or {}).keys()
                if metadata.get("index") is not None else None,
            )
            render_entries = list(zip(
                resolution.render_phones,
                [duration for _phone, duration in entries],
            ))
            for p, _ in render_entries:
                _chk(p)
            new_durs = [d for _, d in entries]
            # Asaxi's domain planner has already built an absolute H/L contour
            # around the requested base pitch, including independent mora
            # offsets. Recentring that contour here would spread one mora edit
            # across every other mora in the phrase.
            absolute_generated_targets = (
                normalize_language_code(lang) == "asaxi"
            )
            if monotone:
                # perfectly flat contour at the requested pitch (keep timing)
                pf = float(pitch) if pitch else 160.0
                tgt, t_acc = [], 0.0
                for (ph, dd) in entries:
                    if ph != "pau":
                        tgt.append((t_acc + dd / 2.0, pf))
                    t_acc += dd
            elif pitch_targets:
                total = sum(new_durs)
                tgt = [(max(0.0, min(total, float(t))),
                        max(PITCH_MIN_HZ, min(PITCH_MAX_HZ, float(f0))))
                       for t, f0 in pitch_targets]
            elif intonation_blocks:
                carried = remap_targets(prev_targets or [],
                                        old_segments or [], new_durs)
                if (carried and pitch and not absolute_generated_targets and
                        not preserve_pitch_register):
                    carried = pitch_domain.recenter_targets_log(
                        carried, float(pitch), PITCH_MIN_HZ, PITCH_MAX_HZ)
                tgt = overlay_intonation_targets(
                    carried, intonation_blocks, float(pitch or 160.0),
                    float(fall or 0.0))
            else:
                tgt = remap_targets(prev_targets or [], old_segments or [],
                                    new_durs)
                if (tgt and pitch and not absolute_generated_targets and
                        not preserve_pitch_register):
                    # recenter the carried contour on the requested pitch,
                    # preserving its shape
                    tgt = pitch_domain.recenter_targets_log(
                        tgt, float(pitch), PITCH_MIN_HZ, PITCH_MAX_HZ)
                elif not tgt and pitch:
                    # no contour to carry: synthesize a simple declination from
                    # the Pitch/Fall controls (start high, end low)
                    total = sum(new_durs)
                    spread = pitch_domain.fall_percent_to_span_semitones(
                        fall)
                    t_acc = 0.0
                    for (p, d) in entries:
                        mid = t_acc + d / 2.0
                        t_acc += d
                        if p == "pau":
                            continue
                        frac = mid / total if total > 0 else 0.5
                        tgt.append((mid, pitch_domain.semitone_offset(
                            float(pitch), spread * (1.0 - 2.0 * frac))))
            clean_tgt = list(tgt)
            if clean_tgt and not (fault_mode or {}).get("disable_f0_correction"):
                clean_tgt = anchor_phrase_targets(
                    entries, clean_tgt, float(pitch or 160.0),
                    min_hz=PITCH_MIN_HZ, max_hz=PITCH_MAX_HZ)
            glitch_events = []
            if (fault_mode or {}).get("pitch_glitch"):
                tgt, glitch_events = pitch_estimation_faults(
                    entries, clean_tgt, float(pitch or 160.0),
                    forced_events=(fault_mode or {}).get(
                        "pitch_glitch_pins"),
                    forced_index=(fault_mode or {}).get(
                        "pitch_glitch_segment"),
                    min_hz=PITCH_MIN_HZ, max_hz=PITCH_MAX_HZ)
            else:
                tgt = clean_tgt
            # bucket targets into their segments as (offset f0)
            starts, t0 = [], 0.0
            for d in new_durs:
                starts.append(t0)
                t0 += d
            per_seg = [[] for _ in entries]
            for t, f0 in tgt:
                i = max(0, min(len(entries) - 1,
                               next((k for k in range(len(entries))
                                     if starts[k] <= t < starts[k] + new_durs[k]),
                                    len(entries) - 1)))
                per_seg[i].append((t - starts[i], f0))
            parts = []
            for (p, d), tg in zip(render_entries, per_seg):
                s = "(%s %.5f" % (p, d)
                for off, f0 in tg:
                    s += " (%.5f %.2f)" % (max(0.0, min(d, off)), f0)
                parts.append(s + ")")
            manual_overrides = {int(k): str(v) for k, v in
                                dict(unit_overrides or {}).items()}
            scheme_overrides = self.automatic_unit_overrides(
                [phone for phone, _duration in render_entries], voicebank)
            scheme_overrides.update(manual_overrides)
            override_scheme = festival_unit_override_scheme(
                scheme_overrides, len(entries))
            body = ("(set! u (Utterance Segments (%s)))\n" % " ".join(parts)
                    + override_scheme
                    + "(utt.synth u)\n")
            clean = [p for p, _ in entries[1:-1]]
            published_ground = ground_truth_targets
            if not pitch_targets and not intonation_blocks:
                # No manual overlay is active, so the generated baseline must
                # be the exact contour this explicit pass is about to render.
                # Publishing the pre-recenter text-pass contour here made the
                # first editor gesture replace the whole sentence by a
                # globally shifted F0 curve.
                published_ground = clean_tgt
            elif published_ground is None and glitch_events:
                published_ground = clean_tgt
            result = self._synth_common(
                body, voicebank, speed, text, lang,
                phones_used=[p for p in clean if p != "pau"],
                pitch=pitch, fall=fall, monotone=monotone,
                ground_truth_targets=published_ground,
                pitch_override=pitch_targets,
                intonation_blocks=intonation_blocks,
                pitch_mode=pitch_mode,
                unit_overrides=manual_overrides,
                fault_mode=fault_mode,
                explicit_durations=render_entries)
            apply_special_phone_display(
                result,
                [phone for phone, _duration in entries],
                [phone for phone, _duration in render_entries],
                resolution.realizations,
            )
            if glitch_events:
                for event in glitch_events:
                    row = dict(event)
                    row["kind"] = "pitch_glitch"
                    result.fault_events.append(row)
                phones = [str(event.get("phone") or "?")
                          for event in glitch_events]
                note = "pitch estimate fault%s on %s" % (
                    "s" if len(phones) != 1 else "", ", ".join(phones))
                result.warning = ((result.warning + "; " + note)
                                  if result.warning else note)
            return result

        clean = [p for p in (str(x).strip() for x in phones) if p]
        while clean and clean[0] == "pau":
            clean.pop(0)
        while clean and clean[-1] == "pau":
            clean.pop()
        if not clean:
            raise BackendError("Phone list is empty.")
        for p in clean:
            _chk(p)
        display_sequence = ["pau"] + clean + ["pau"]
        resolution = resolve_voice_special_phones(
            display_sequence,
            metadata,
            voicebank=voicebank,
            available_diphones=(metadata.get("index") or {}).keys()
            if metadata.get("index") is not None else None,
        )
        render_sequence = list(resolution.render_phones)
        for p in render_sequence:
            _chk(p)
        seq = " ".join(render_sequence)
        manual_overrides = {int(k): str(v) for k, v in
                            dict(unit_overrides or {}).items()}
        scheme_overrides = self.automatic_unit_overrides(
            render_sequence, voicebank)
        scheme_overrides.update(manual_overrides)
        body = ("(set! u (Utterance Phones (%s)))\n" % seq
                + festival_unit_override_scheme(
                    scheme_overrides, len(clean) + 2)
                + "(utt.synth u)\n")
        result = self._synth_common(
            body, voicebank, speed, text, lang,
            phones_used=clean, pitch=pitch, fall=fall,
            monotone=monotone,
            ground_truth_targets=ground_truth_targets,
            pitch_override=pitch_targets,
            intonation_blocks=intonation_blocks,
            pitch_mode=pitch_mode,
            unit_overrides=manual_overrides,
            fault_mode=fault_mode,
        )
        return apply_special_phone_display(
            result, display_sequence, render_sequence,
            resolution.realizations,
        )


# ------------------------------------------------------------------- project io
def _project_sentence_json(row: dict) -> dict:
    data = dict(row)
    for segment_key in ("segments", "editor_segments"):
        data[segment_key] = [
            {"id": s.uid, "phone": s.phone,
             "start": float(s.start), "end": float(s.end),
             "timing_role": str(getattr(s, "timing_role", "") or "")}
            if isinstance(s, Segment) else
            {"id": str(s.get("id") or s.get("uid") or uuid.uuid4().hex),
             "phone": str(s.get("phone", "")),
             "start": float(s.get("start", 0.0)),
             "end": float(s.get("end", s.get("start", 0.0))),
             "timing_role": str(s.get("timing_role", "") or "")}
            for s in (data.get(segment_key) or [])]
    for key in ("generated_targets", "pitch_override"):
        data[key] = [[float(t), float(f)] for t, f in (data.get(key) or [])]
    data["timing_factors"] = [float(f) for f in
                              (data.get("timing_factors") or [])]
    data["intonation_blocks"] = [dict(b) for b in
                                  (data.get("intonation_blocks") or [])]
    data["unit_overrides"] = {str(k): str(v) for k, v in
                              dict(data.get("unit_overrides") or {}).items()}
    data["selected_units"] = {str(k): str(v) for k, v in
                              dict(data.get("selected_units") or {}).items()}
    data["render_phones"] = [
        str(phone) for phone in (data.get("render_phones") or [])
    ]
    data["special_phone_realizations"] = [
        dict(item)
        for item in (data.get("special_phone_realizations") or [])
        if isinstance(item, Mapping)
    ]
    data["fault_mode"] = dict(data.get("fault_mode") or {})
    if "japanese_state" in data:
        data["japanese_state"] = copy.deepcopy(
            data.get("japanese_state") or {})
    return data


def save_batch_project(path: str, sentences, active_sentence: int = 0) -> None:
    rows = [_project_sentence_json(dict(row)) for row in (sentences or [])]
    if not rows:
        raise ValueError("A project needs at least one sentence.")
    data = {"version": 3, "active_sentence": max(0, min(
        len(rows) - 1, int(active_sentence))), "sentences": rows}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


PROJECT_MANIFEST_NAME = "project.json"


def project_manifest_path(path) -> Path:
    """Resolve a project directory or manifest/legacy JSON to its JSON file."""
    source = Path(path).expanduser()
    return source / PROJECT_MANIFEST_NAME if source.is_dir() else source


def project_root_path(path) -> Optional[Path]:
    """Return the folder root for a version-4 project path, if recognizable."""
    source = Path(path).expanduser()
    if source.is_dir():
        return source
    if source.name.lower() == PROJECT_MANIFEST_NAME:
        return source.parent
    return None


def prepare_project_folder(folder) -> Path:
    """Validate and create the standard project folder structure."""
    root = Path(folder).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError("The project path is an existing file: %s" % root)
    manifest = root / PROJECT_MANIFEST_NAME
    if root.is_dir():
        if manifest.is_file():
            try:
                with open(manifest, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(
                    "The existing project.json is not a readable FestVox "
                    "project: %s" % error) from error
            recognizable = isinstance(existing, dict) and (
                isinstance(existing.get("sentences"), list) or
                any(key in existing for key in
                    ("text", "segments", "phones", "voicebank")))
            if not recognizable:
                raise ValueError(
                    "Refusing to overwrite an unrelated project.json in: %s"
                    % root)
        else:
            allowed = {"cache", "exports"}
            unexpected = [child.name for child in root.iterdir()
                          if child.name not in allowed]
            if unexpected:
                raise ValueError(
                    "Choose a new or empty project folder. This folder "
                    "already contains unrelated files: %s" %
                    ", ".join(unexpected[:5]))
    root.mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(exist_ok=True)
    (root / "exports").mkdir(exist_ok=True)
    return root


def save_project_folder(folder, sentences, active_sentence: int = 0,
                        settings=None) -> Path:
    """Write a version-4 project folder and return its manifest path.

    Existing non-project folders are rejected so selecting a broad user folder
    cannot scatter project files into it. The manifest replacement is atomic;
    legacy JSON projects are never deleted or modified by this function.
    """
    root = prepare_project_folder(folder)
    manifest = root / PROJECT_MANIFEST_NAME
    rows = [_project_sentence_json(dict(row)) for row in (sentences or [])]
    if not rows:
        raise ValueError("A project needs at least one sentence.")
    data = {
        "version": 4,
        "layout": "folder",
        "active_sentence": max(0, min(len(rows) - 1, int(active_sentence))),
        "sentences": rows,
    }
    if isinstance(settings, dict) and settings:
        data["settings"] = json.loads(json.dumps(settings))
    temporary = root / (PROJECT_MANIFEST_NAME + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(str(temporary), str(manifest))
    return manifest


def save_project(path: str, *, text: str, language: str, lang_code: str,
                 voicebank: str, speed: float, segments: List[Segment],
                 phones: List[str], timing_factors=None,
                 engine: str = "diphone", generated_targets=None,
                 pitch_override=None, intonation_blocks=None,
                 pitch_mode: str = "", unit_overrides=None,
                 source_voicing_targets=None,
                 generated_voicing_targets=None,
                 voicing_override=None, voicing_mode: str = "") -> None:
    data = {
        "text": text, "language": language, "lang_code": lang_code,
        "engine": engine,
        "voicebank": voicebank, "speed": float(speed),
        "phones": list(phones),
        "segments": [{"id": s.uid, "phone": s.phone,
                      "start": s.start, "end": s.end}
                     for s in segments],
        "timing_factors": [float(f) for f in (timing_factors or [])],
        "generated_targets": [[float(t), float(f)] for t, f in
                              (generated_targets or [])],
        "pitch_override": [[float(t), float(f)] for t, f in
                           (pitch_override or [])],
        "intonation_blocks": [dict(b) for b in (intonation_blocks or [])],
        "pitch_mode": str(pitch_mode or ""),
        "source_voicing_targets": [
            [float(t), float(value)] for t, value in
            (source_voicing_targets or [])
        ],
        "generated_voicing_targets": [
            [float(t), float(value)] for t, value in
            (generated_voicing_targets or [])
        ],
        "voicing_override": [[float(t), float(value)] for t, value in
                              (voicing_override or [])],
        "voicing_mode": str(voicing_mode or ""),
        "unit_overrides": {str(k): str(v) for k, v in
                           dict(unit_overrides or {}).items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_project(path: str) -> dict:
    source = Path(path).expanduser()
    manifest = project_manifest_path(source)
    if source.is_dir() and not manifest.is_file():
        raise ValueError(
            "This folder does not contain %s: %s" %
            (PROJECT_MANIFEST_NAME, source))
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    def decode(row):
        row = dict(row)
        for segment_key in ("segments", "editor_segments"):
            row[segment_key] = [
                Segment(s["phone"], s["start"], s["end"],
                        str(s.get("id") or s.get("uid") or ""),
                        str(s.get("timing_role") or ""))
                for s in row.get(segment_key, [])]
        row["unit_overrides"] = {int(k): str(v) for k, v in
                                 dict(row.get("unit_overrides") or {}).items()}
        row["selected_units"] = {int(k): str(v) for k, v in
                                 dict(row.get("selected_units") or {}).items()}
        return row

    if isinstance(data.get("sentences"), list):
        data["sentences"] = [decode(row) for row in data["sentences"]]
    else:
        data = decode(data)
    folder_root = project_root_path(source)
    is_folder = bool(folder_root is not None and
                     (source.is_dir() or int(data.get("version") or 0) >= 4))
    data["_project_manifest"] = str(manifest.resolve())
    data["_project_root"] = str(folder_root.resolve()) if is_folder else ""
    data["_legacy_source"] = "" if is_folder else str(manifest.resolve())
    return data
# end of festvox_core
