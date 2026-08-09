# -*- coding: utf-8 -*-
"""
synth_diphone.py — standalone concatenative diphone synthesis for FestVox.

Renders audio from the FestVox-style diphone database produced by
99_Tools/festvox/utau2festvox.py (dic/diphone_index.json + wav/), using the
Lem 4_Fis3 voice. Pure standard library (wave + array); no Festival runtime
needed — the same index the Festival stub loads drives this renderer.

Two front ends share the voice:
  * Asaxi:   grapheme→phone rules derived from
             "00_Phonemes of the Asaxi Language" (romanization → arpasing).
  * English: CMU dictionary lookup (pip install cmudict), stress stripped —
             the bank is arpasing, i.e. lowercase ARPAbet, so English "just
             works" for testing.

The GUI, command-line renderer, and corpus tools import this copy from the
FestVox tool directory.  Vocab Forge may also consume generated databases,
but it is not a runtime dependency of this module.
"""
import array
from collections import OrderedDict
import json
import re
import sys
import threading
import wave
from pathlib import Path

import asaxi_frontend
from cache_support import estimate_size_bytes, file_change_token, read_only_view
from join_synthesis import adaptive_join_pcm16, legacy_linear_join_pcm16

CROSSFADE_MS = 15          # equal-power join at each diphone seam
EDGE_FADE_MS = 8           # de-click fade at utterance edges
HALF_MS = 150              # max audio kept on each side of the phone
                           # boundary (mid) — trims the recordings' long
                           # vowel sustains to conversational phone lengths
DIPHONE_CACHE_MAX_FILES = 64
DIPHONE_CACHE_MAX_BYTES = 64 * 1024 * 1024
DIPHONE_SLICE_CACHE_MAX_ENTRIES = 512
DIPHONE_SLICE_CACHE_MAX_BYTES = 32 * 1024 * 1024
CMU_MODEL_CACHE_MAX_BYTES = 128 * 1024 * 1024
KANA_MODEL_CACHE_MAX_BYTES = 8 * 1024 * 1024
SYNTH_TEXT_DB_CACHE_MAX_VOICES = 2

_synth_text_db_cache = OrderedDict()
_synth_text_db_cache_lock = threading.RLock()

# ------------------------------------------------------------------ database

class DiphoneDB:
    def __init__(self, root: Path, cache_max_files=DIPHONE_CACHE_MAX_FILES,
                 cache_max_bytes=DIPHONE_CACHE_MAX_BYTES,
                 slice_cache_max_entries=DIPHONE_SLICE_CACHE_MAX_ENTRIES,
                 slice_cache_max_bytes=DIPHONE_SLICE_CACHE_MAX_BYTES):
        self.root = Path(root)
        index_path = self.root / "dic" / "diphone_index.json"
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        self._metadata = meta
        self._index = meta["index"]         # name -> [wav, start, mid, end]
        self._alternatives = meta.get("alternatives") or {}
        # Public metadata is lazily read-only, avoiding a recursive walk of a
        # 10+ MiB index during voice selection.
        self.metadata = read_only_view(meta)
        self.index = read_only_view(self._index)
        self.alternatives = read_only_view(self._alternatives)
        # Cache accounting must not recursively traverse a multi-megabyte
        # index during voice selection. The serialized size is a cheap,
        # stable lower-bound estimate suitable for the GUI usage display.
        self.metadata_bytes = int(index_path.stat().st_size)
        # A voice can reference hundreds of long source recordings.  Keeping
        # every decoded WAV forever made repeated generation look like a leak
        # (the integrated Lem bank can retain roughly 339 MiB per voice).
        # This LRU bounds the working set while preserving hot transition data.
        self.cache_max_files = max(0, int(cache_max_files))
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        self._cache = OrderedDict()         # wav -> (rate, frames, byte_count)
        self._cache_bytes = 0
        self.slice_cache_max_entries = max(0, int(slice_cache_max_entries))
        self.slice_cache_max_bytes = max(0, int(slice_cache_max_bytes))
        self._slice_cache = OrderedDict()
        self._slice_cache_bytes = 0
        self._decode_hits = 0
        self._decode_misses = 0
        self._slice_hits = 0
        self._slice_misses = 0
        self._evictions = 0
        self._stale_invalidations = 0
        self._lock = threading.RLock()

    def has(self, dip):
        return dip in self._index

    def choose(self, dip, outer_left="*", outer_right="*", override=None):
        """Choose a take from directional, OTO-only context evidence."""
        choices = self._alternatives.get(dip) or []
        if override:
            forced = next((c for c in choices
                           if c.get("left_name") == override), None)
            if forced and forced.get("index_name") in self._index:
                return forced["index_name"]
        p1, p2 = dip.split("-", 1)
        p1, p2 = p1.rstrip("_"), p2.rstrip("_")
        light_followers = _VOWELS | {"y"}
        l_class = ("light" if p1 == "l" and p2 in light_followers else
                   "dark" if p1 == "l" else
                   "light" if (p2 == "l" and
                               outer_right.rstrip("_") in light_followers)
                   else "dark" if p2 == "l" else "*")
        def context_score(choice, side, actual, exact):
            expected = choice.get(side + "_context")
            if not expected or expected == "*":
                return 0
            if expected == actual:
                return exact
            expected_info = _choice_context_info(choice, side)
            actual_edge = "right" if side == "left" else "left"
            actual_class = _context_edge_info(actual, actual_edge)["class"]
            wanted = expected_info["class"]
            return 4 if wanted not in {"wildcard", "other"} and \
                wanted == actual_class else -8

        def score(choice):
            value = context_score(choice, "left", outer_left, 6)
            value += context_score(choice, "right", outer_right, 7)
            if l_class != "*" and choice.get("l_class") not in (None, "*"):
                value += 20 if choice.get("l_class") == l_class else -100
            return value

        best = choices[0] if choices else None
        if best and p2 in _VOICED_SIBILANTS:
            supportive = [choice for choice in choices
                          if _sibilant_context_quality(choice) ==
                          "verified_supportive"]
            unknown = [choice for choice in choices
                       if _sibilant_context_quality(choice) == "unknown"]
            pool = supportive or unknown
            if pool:
                best = pool[0]
                best_score = score(best)
                for choice in pool[1:]:
                    choice_score = score(choice)
                    if choice_score > best_score:
                        best, best_score = choice, choice_score
            # If every context is verified risky, retain the base take.
        elif best:
            best_score = max(0, score(best))
            for choice in choices[1:]:
                choice_score = score(choice)
                if choice_score > best_score:
                    best, best_score = choice, choice_score
        selected = (best or {}).get("index_name") or dip
        return selected if selected in self._index else dip

    def _wav_identity(self, wav_name):
        path = self.root / "wav" / wav_name
        stat = path.stat()
        return path, (
            int(stat.st_mtime_ns), int(stat.st_size),
            int(getattr(stat, "st_dev", 0)), int(getattr(stat, "st_ino", 0)),
            file_change_token(path, stat),
        )

    def _load(self, wav_name, identity=None):
        with self._lock:
            path, current_identity = self._wav_identity(wav_name)
            if identity is not None and identity != current_identity:
                identity = current_identity
            cached = self._cache.pop(wav_name, None)
            if cached is not None:
                if cached[3] == current_identity:
                    self._cache[wav_name] = cached
                    self._decode_hits += 1
                    return cached[0], cached[1]
                self._cache_bytes -= cached[2]
                self._stale_invalidations += 1
            self._decode_misses += 1
            with wave.open(str(path), "rb") as w:
                assert w.getsampwidth() == 2, "expected 16-bit wavs"
                frames = array.array("h")
                frames.frombytes(w.readframes(w.getnframes()))
                if w.getnchannels() == 2:       # downmix, just in case
                    frames = array.array(
                        "h", [(frames[i] + frames[i + 1]) // 2
                              for i in range(0, len(frames), 2)])
                rate = w.getframerate()
            byte_count = len(frames) * frames.itemsize
            if (self.cache_max_files > 0 and self.cache_max_bytes > 0 and
                    byte_count <= self.cache_max_bytes):
                self._cache[wav_name] = (
                    rate, frames, byte_count, current_identity)
                self._cache_bytes += byte_count
                while (len(self._cache) > self.cache_max_files or
                       self._cache_bytes > self.cache_max_bytes):
                    _old_name, (_old_rate, _old_frames, old_bytes,
                                _old_identity) = \
                        self._cache.popitem(last=False)
                    self._cache_bytes -= old_bytes
                    self._evictions += 1
            return rate, frames

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
            self._slice_cache.clear()
            self._slice_cache_bytes = 0

    def cache_info(self):
        with self._lock:
            return {
                # Legacy keys continue to describe decoded source WAVs.
                "files": len(self._cache),
                "bytes": self._cache_bytes,
                "max_files": self.cache_max_files,
                "max_bytes": self.cache_max_bytes,
                "slices": len(self._slice_cache),
                "slice_bytes": self._slice_cache_bytes,
                "max_slices": self.slice_cache_max_entries,
                "max_slice_bytes": self.slice_cache_max_bytes,
                "total_bytes": self._cache_bytes + self._slice_cache_bytes,
                "decode_hits": self._decode_hits,
                "decode_misses": self._decode_misses,
                "slice_hits": self._slice_hits,
                "slice_misses": self._slice_misses,
                "evictions": self._evictions,
                "stale_invalidations": self._stale_invalidations,
            }

    def slice(self, dip, half_ms=HALF_MS):
        """Return (framerate, mono int16 array) for a diphone. `half_ms` caps
        how much audio is kept each side of the phone boundary — smaller =
        shorter phones = faster speech (see `speed` in render())."""
        fr, chunk, _ = self.slice_info(dip, half_ms)
        return fr, chunk

    def slice_info(self, dip, half_ms=HALF_MS, *, copy_samples=True):
        """Like slice(), but also returns the offset (in samples, into the
        returned chunk) of the diphone's phone boundary — its `mid` point.

        Public callers receive a defensive sample copy.  The renderer opts
        into the private shared view to avoid copying cached slices.
        """
        wav_name, start, mid, end = self._index[dip]
        _path, wav_identity = self._wav_identity(wav_name)
        prefix = (str(dip), round(float(half_ms), 9))
        key = prefix + (wav_identity,)
        with self._lock:
            cached = self._slice_cache.pop(key, None)
            if cached is not None:
                self._slice_cache[key] = cached
                self._slice_hits += 1
                chunk = (array.array("h", cached[1]) if copy_samples
                         else cached[1])
                return cached[0], chunk, cached[2]
            self._slice_misses += 1
            for stale_key in [candidate for candidate in self._slice_cache
                              if candidate[:2] == prefix]:
                _rate, _chunk, _boundary, old_bytes = \
                    self._slice_cache.pop(stale_key)
                self._slice_cache_bytes -= old_bytes
                self._stale_invalidations += 1
            fr, frames = self._load(wav_name, wav_identity)
            half = half_ms / 1000.0
            s = max(start, mid - half)
            e = min(end, mid + half)
            i0, i1 = int(s * fr), int(e * fr)
            chunk = frames[i0:i1]
            boundary = max(0, min(i1 - i0, int(mid * fr) - i0))
            byte_count = len(chunk) * chunk.itemsize
            if (self.slice_cache_max_entries > 0 and
                    self.slice_cache_max_bytes > 0 and
                    byte_count <= self.slice_cache_max_bytes):
                self._slice_cache[key] = (fr, chunk, boundary, byte_count)
                self._slice_cache_bytes += byte_count
                while (len(self._slice_cache) >
                       self.slice_cache_max_entries or
                       self._slice_cache_bytes >
                       self.slice_cache_max_bytes):
                    _old_key, (_old_rate, _old_chunk, _old_boundary,
                               old_bytes) = self._slice_cache.popitem(
                                   last=False)
                    self._slice_cache_bytes -= old_bytes
                    self._evictions += 1
            returned = array.array("h", chunk) if copy_samples else chunk
            return fr, returned, boundary


def find_db(cfg) -> Path:
    cands = cfg.get("festvox_db") or []
    if isinstance(cands, str):
        cands = [cands]
    for c in cands:
        p = Path(c)
        if (p / "dic" / "diphone_index.json").exists():
            return p
    raise FileNotFoundError(
        "No diphone DB found. Set \"festvox_db\" in config.json to the "
        "festvox_db folder built by 99_Tools/festvox/utau2festvox.py "
        f"(tried: {cands or 'nothing'})")

# ------------------------------------------------------------- Asaxi g2p

# Romanization → bank phones, longest match first. From the phoneme chart
# (Arpa column) in 00_Phonemes of the Asaxi Language.
_ASAXI_RULES = [
    # trigraphs / digraphs
    ("nŋ", ["nng"]), ("nn", ["nn"]), ("mm", ["mm"]),
    ("ch", ["ch"]), ("sh", ["sh"]), ("dh", ["dh"]), ("jh", ["jh"]),
    ("zh", ["zh"]), ("th", ["th"]), ("dz", ["dz"]),
    ("si", ["sh", "i"]),           # ś may be written "si" (chart)
    ("ni", ["ny", "i"]),           # n+i palatalizes (chart: nasal shift)
    # single letters — vowels
    ("å", ["a", "w"]), ("ă", ["a", "y"]),
    ("ë", ["e", "y"]), ("ỏ", ["o", "w"]),
    ("ő", ["o", "y"]), ("ů", ["u", "w"]),
    ("è", ["ax"]), ("ě", ["er"]),
    ("ý", ["ih"]), ("ù", ["u"]), ("á", ["ao"]),
    ("a", ["a"]), ("e", ["e"]), ("i", ["i"]), ("o", ["o"]), ("u", ["u"]),
    # single letters — consonants
    ("ŕ", ["dx"]), ("ń", ["ny"]), ("ś", ["sh"]), ("ŋ", ["ng"]),
    ("'", ["q"]), ("x", ["hh"]), ("c", ["ts"]), ("j", ["y"]),
    ("b", ["b"]), ("d", ["d"]), ("f", ["f"]), ("g", ["g"]), ("h", ["h"]),
    ("k", ["k"]), ("l", ["l"]), ("m", ["m"]), ("n", ["n"]), ("p", ["p"]),
    ("r", ["r"]), ("s", ["s"]), ("t", ["t"]), ("v", ["v"]), ("w", ["w"]),
    ("y", ["y"]), ("z", ["z"]),
]
_STOPS = {"p", "t", "k", "b", "d", "g", "ch", "ts", "dz", "jh"}
# every phone the bank treats as a vowel/nucleus -- these must NEVER be
# collapsed by the gemination rule (that rule is for doubled consonants only).
_VOWELS = {"a", "e", "i", "o", "u", "aw", "ay", "ey", "ow", "oy", "uw",
           "ax", "er", "ih", "ao", "iy", "uh", "eh", "aa", "ah", "ae"}
_VOICED_SIBILANTS = {"z", "zh", "zi", "dz", "jh"}
_SIBILANT_SUPPORTIVE_CLASSES = {
    "vowel", "nasal", "liquid", "glide", "fricative_voiced",
}


def _phone_context_class(phone):
    base = re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    classes = (
        (_VOWELS, "vowel"),
        (set("p t k q py ty ky cl".split()), "stop_voiceless"),
        (set("b d g by dy gy dx dxy".split()), "stop_voiced"),
        ({"ch", "ts"}, "affricate_voiceless"),
        ({"jh", "dz"}, "affricate_voiced"),
        (set("f s sh th h hh fy hy".split()), "fricative_voiceless"),
        (set("v z zh dh vy zi".split()), "fricative_voiced"),
        (set("m n ng nn mm nng xn my ny ngy".split()), "nasal"),
        (set("l r rr ly ry ri".split()), "liquid"),
        (set("w y wi".split()), "glide"),
    )
    if base == "*":
        return "wildcard"
    if base in {"pau", "sil", "sp"}:
        return "silence"
    for phones, label in classes:
        if base in phones:
            return label
    if base.endswith("y"):
        return _phone_context_class(base[:-1])
    return "other"


def _context_edge_info(phone, edge):
    """Classify an OTO token edge; WAV filenames are never evidence."""
    if edge not in {"left", "right"}:
        raise ValueError("edge must be 'left' or 'right'")
    base = re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    direct_class = _phone_context_class(base)
    if base == "*":
        return {"phone": "*", "class": "wildcard",
                "kind": "wildcard_unknown"}
    if direct_class != "other":
        return {"phone": base, "class": direct_class, "kind": "atomic"}
    for vowel in sorted(_VOWELS, key=lambda item: (-len(item), item)):
        if not base.endswith(vowel) or len(base) <= len(vowel):
            continue
        onset = base[:-len(vowel)]
        onset_class = _phone_context_class(onset)
        if onset_class in {"other", "wildcard", "silence", "vowel"}:
            continue
        edge_phone = onset if edge == "left" else vowel
        return {"phone": edge_phone,
                "class": _phone_context_class(edge_phone),
                "kind": "compound_cv"}
    return {"phone": base, "class": "other", "kind": "unclassified"}


def _choice_context_info(choice, side):
    token = str((choice or {}).get(side + "_context") or "*")
    info = _context_edge_info(token, "right" if side == "left" else "left")
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
    return info


def _sibilant_context_quality(choice):
    context_class = _choice_context_info(choice, "right")["class"]
    if context_class in _SIBILANT_SUPPORTIVE_CLASSES:
        return "verified_supportive"
    if context_class in {"wildcard", "other"}:
        return "unknown"
    return "verified_risky"
# acoustically interchangeable vowel fallbacks (bank gaps: e.g. "k i" was
# recorded as "k iy" per arpasing convention)
ALT_VOWELS = {"i": ("iy", "ih"), "u": ("uw", "uh"), "e": ("eh", "ey"),
              "o": ("ow", "ao"), "a": ("aa", "ah", "ax"),
              "iy": ("i",), "uw": ("u",), "eh": ("e",), "ow": ("o",),
              "aa": ("a",), "ah": ("a",), "ax": ("a",), "ih": ("i",)}
_PALATAL = {c + "y" for c in
            "b d g k m n p r t h l v f ng dx".split()}   # Cy units in the bank


def g2p_capitalized_asaxi_term(term: str):
    """Resolve one full-cap Asaxi term through the English frontend.

    The orthography uses full capitals as an explicit foreign/proper-term
    marker. Attested project pronunciations take precedence over the general
    English dictionary, while a project/user dictionary can still override the
    returned phones at the caller boundary.
    """

    key = str(term or "").strip().upper()
    override = (
        asaxi_frontend.CAPITALIZED_ENGLISH_PRONUNCIATION_OVERRIDES.get(key)
    )
    if override:
        return list(override)
    try:
        return g2p_english(term)
    except ValueError as error:
        raise ValueError(
            f"capitalized Asaxi term {term!r} has no English pronunciation; "
            "add a Dictionary pronunciation override"
        ) from error


def g2p_asaxi(text: str):
    """Asaxi text -> bank phones, routing full-cap terms through English G2P."""

    phones = []
    for word in asaxi_frontend.words_in_text(
            text, reject_unsupported_letters=True, preserve_case=True):
        if asaxi_frontend.is_capitalized_term(word):
            phones.extend(g2p_capitalized_asaxi_term(word))
        else:
            phones.extend(asaxi_frontend.g2p_asaxi(word))
    # Keep the historical list return type used by the renderer and callers.
    return phones

# ----------------------------------------------------------- English g2p

_MODEL_CACHE_LOCK = threading.RLock()
_CMU_DICT = None
_CMU_DICT_BYTES = 0
_KANA_TABLE = {}   # kana grapheme -> [phones], lazily loaded from YAML
_KANA_TABLE_BYTES = 0


def _load_cmu_dict():
    global _CMU_DICT, _CMU_DICT_BYTES
    with _MODEL_CACHE_LOCK:
        if _CMU_DICT is not None:
            return _CMU_DICT
        try:
            import cmudict
        except ImportError:
            raise RuntimeError("pip install cmudict  (needed for --lang en)")
        result = cmudict.dict()
        byte_count = estimate_size_bytes(result)
        if byte_count <= CMU_MODEL_CACHE_MAX_BYTES:
            _CMU_DICT = result
            _CMU_DICT_BYTES = byte_count
        return result


def g2p_english(text: str):
    """English text -> bank phones via the CMU dictionary (arpasing)."""
    d = _load_cmu_dict()
    phones, missing = [], []
    for word in re.findall(r"[a-zA-Z']+", text.lower()):
        pron = d.get(word)
        if not pron:
            missing.append(word)
            continue
        phones.extend(re.sub(r"\d", "", p).lower() for p in pron[0])
    if missing:
        raise ValueError(f"not in CMU dictionary: {missing}")
    return phones

# ----------------------------------------------------------- Japanese g2p
# Uses the OpenUTAU phonemizer table (en-jap-mapping.yaml): 1247 hiragana +
# 289 katakana graphemes already map to this bank's phone set. Input may be
# kana OR Hepburn romaji (romaji is normalized to kana first). Kanji is NOT
# handled — that needs a morphological analyzer (see MULTISYN.md, Japanese).

def _load_kana_table():
    global _KANA_TABLE_BYTES
    with _MODEL_CACHE_LOCK:
        if _KANA_TABLE:
            return _KANA_TABLE
        here = Path(__file__).resolve().parent
        for c in (here / "en-jap-mapping.yaml",
                  here.parent / "festvox" / "en-jap-mapping.yaml"):
            if c.is_file():
                text = c.read_text(encoding="utf-8")
                for g, ph in re.findall(
                        r"grapheme:\s*(\S+)\s*\n\s*phonemes:\s*\[([^\]]*)\]",
                        text):
                    if re.search(r"[぀-ゟ゠-ヿ]", g):    # kana only
                        _KANA_TABLE[g] = [
                            x.strip() for x in ph.split(",") if x.strip()
                        ]
                break
        if not _KANA_TABLE:
            raise RuntimeError(
                "en-jap-mapping.yaml not found next to synth_diphone.py "
                "or in ../festvox/ (needed for --lang ja)")
        _KANA_TABLE_BYTES = estimate_size_bytes(_KANA_TABLE)
        if _KANA_TABLE_BYTES > KANA_MODEL_CACHE_MAX_BYTES:
            result = dict(_KANA_TABLE)
            _KANA_TABLE.clear()
            _KANA_TABLE_BYTES = 0
            return result
        return _KANA_TABLE


def clear_model_cache():
    """Clear process-owned pronunciation models; no files are touched."""
    global _CMU_DICT, _CMU_DICT_BYTES, _KANA_TABLE_BYTES
    with _MODEL_CACHE_LOCK:
        removed = {
            "entries": int(_CMU_DICT is not None) + int(bool(_KANA_TABLE)),
            "bytes": _CMU_DICT_BYTES + _KANA_TABLE_BYTES,
        }
        _CMU_DICT = None
        _CMU_DICT_BYTES = 0
        _KANA_TABLE.clear()
        _KANA_TABLE_BYTES = 0
        return removed


def model_cache_info():
    with _MODEL_CACHE_LOCK:
        return {
            "owner": "synth-diphone-pronunciation-models",
            "entries": int(_CMU_DICT is not None) + int(bool(_KANA_TABLE)),
            "bytes": _CMU_DICT_BYTES + _KANA_TABLE_BYTES,
            "max_entries": 2,
            "max_bytes": CMU_MODEL_CACHE_MAX_BYTES +
                         KANA_MODEL_CACHE_MAX_BYTES,
            "cmu_words": len(_CMU_DICT or {}),
            "kana_entries": len(_KANA_TABLE),
        }

# Hepburn romaji -> hiragana (gojūon + dakuten + yōon + sokuon/long vowel).
_ROMAJI = {
 "kya":"きゃ","kyu":"きゅ","kyo":"きょ","sha":"しゃ","shu":"しゅ","sho":"しょ",
 "cha":"ちゃ","chu":"ちゅ","cho":"ちょ","nya":"にゃ","nyu":"にゅ","nyo":"にょ",
 "hya":"ひゃ","hyu":"ひゅ","hyo":"ひょ","mya":"みゃ","myu":"みゅ","myo":"みょ",
 "rya":"りゃ","ryu":"りゅ","ryo":"りょ","gya":"ぎゃ","gyu":"ぎゅ","gyo":"ぎょ",
 "ja":"じゃ","ju":"じゅ","jo":"じょ","bya":"びゃ","byu":"びゅ","byo":"びょ",
 "pya":"ぴゃ","pyu":"ぴゅ","pyo":"ぴょ",
 "shi":"し","chi":"ち","tsu":"つ","dzu":"づ",
 "ka":"か","ki":"き","ku":"く","ke":"け","ko":"こ",
 "sa":"さ","su":"す","se":"せ","so":"そ","si":"し",
 "ta":"た","te":"て","to":"と","ti":"ち","tu":"つ",
 "na":"な","ni":"に","nu":"ぬ","ne":"ね","no":"の",
 "ha":"は","hi":"ひ","fu":"ふ","hu":"ふ","he":"へ","ho":"ほ",
 "ma":"ま","mi":"み","mu":"む","me":"め","mo":"も",
 "ya":"や","yu":"ゆ","yo":"よ",
 "ra":"ら","ri":"り","ru":"る","re":"れ","ro":"ろ",
 "wa":"わ","wo":"を","wi":"うぃ","we":"うぇ",
 "ga":"が","gi":"ぎ","gu":"ぐ","ge":"げ","go":"ご",
 "za":"ざ","zi":"じ","ji":"じ","zu":"ず","ze":"ぜ","zo":"ぞ",
 "da":"だ","di":"ぢ","du":"づ","de":"で","do":"ど",
 "ba":"ば","bi":"び","bu":"ぶ","be":"べ","bo":"ぼ",
 "pa":"ぱ","pi":"ぴ","pu":"ぷ","pe":"ぺ","po":"ぽ",
 "fa":"ふぁ","fi":"ふぃ","fe":"ふぇ","fo":"ふぉ",
 "a":"あ","i":"い","u":"う","e":"え","o":"お","n":"ん",
}

def romaji_to_kana(text: str) -> str:
    """Hepburn romaji -> hiragana. Handles sokuon (double consonant -> っ)
    and long vowels (macron/doubled vowel -> ー)."""
    s = text.lower().strip()
    s = (s.replace("ā","aa").replace("ī","ii").replace("ū","uu")
          .replace("ē","ee").replace("ō","ou").replace("â","aa")
          .replace("î","ii").replace("û","uu").replace("ê","ee").replace("ô","ou"))
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if not ch.isalpha():
            out.append(ch); i += 1; continue
        # sokuon: doubled consonant (except n) -> っ
        if (ch not in "aeimoun" and i + 1 < len(s) and s[i+1] == ch):
            out.append("っ"); i += 1; continue
        # syllabic n before consonant/space/end
        if ch == "n" and (i + 1 >= len(s) or s[i+1] not in "aeiouy"):
            out.append("ん"); i += 1; continue
        for L in (3, 2, 1):                       # longest romaji match
            if s[i:i+L] in _ROMAJI:
                out.append(_ROMAJI[s[i:i+L]]); i += L; break
        else:
            i += 1                                # skip unknown
    return "".join(out)

def g2p_japanese(text: str):
    """Japanese (kana or romaji) -> bank phone list."""
    table = _load_kana_table()
    # romaji if it's mostly ASCII letters
    if re.search(r"[A-Za-z]", text) and not re.search(r"[぀-ゟ゠-ヿ]", text):
        text = romaji_to_kana(text)
    phones, i = [], 0
    while i < len(text):
        if text[i] == "っ":                       # sokuon -> held closure
            phones.append("cl"); i += 1; continue
        if text[i] in "ーｰ" and phones:            # chōonpu -> lengthen (repeat)
            phones.append(phones[-1]); i += 1; continue
        for L in (2, 1):                          # yōon like きゃ are 2 chars
            if text[i:i+L] in table:
                phones.extend(table[text[i:i+L]]); i += L; break
        else:
            i += 1                                # punctuation / unknown
    return phones

# ------------------------------------------------------------- rendering

def _xfade(a: array.array, b: array.array, n: int) -> array.array:
    """The exact pre-fix linear crossfade, retained by Legacy joins."""
    return legacy_linear_join_pcm16(a, b, n)


def render(db: DiphoneDB, phones, out_path=None, speed=1.0,
           unit_overrides=None, *, return_pcm=False, encode_wav=True,
           crossfade_ms=None, edge_fade_ms=None, half_ms=None,
           legacy_joins=False):
    """Phone list -> wav bytes (and optionally a file). Diphone selection:
    pau-p1, p1-p2, ..., pn-pau, preferring the bank's word-final allophones
    ("prev-C_") for a final consonant. Missing diphones are skipped with a
    note in the returned report.

    `speed` sets the pace: >1 is faster, <1 is slower (1.0 = normal). It works
    by scaling how much of each recorded phone is kept — this is concatenative,
    not time-stretch, so it never changes pitch, but slowing below ~1.0 is
    capped by the amount of audio actually recorded per phone.

    The report also carries per-phone timing: "segments" is a gapless list of
    {"phone", "start", "end"} (seconds) covering the whole output, derived
    from each unit's indexed `mid` boundary — used by the GUI editor."""
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    crossfade_ms = float(CROSSFADE_MS if crossfade_ms is None
                         else crossfade_ms)
    edge_fade_ms = float(EDGE_FADE_MS if edge_fade_ms is None
                         else edge_fade_ms)
    base_half_ms = float(HALF_MS if half_ms is None else half_ms)
    slice_half_ms = base_half_ms / speed          # slower => wider window
    seq = ["pau"] + list(phones) + ["pau"]
    picked, skipped, plan = [], [], []
    # plan entries: (diphone, seq_pos_of_first_phone, [(seq_pos, "pre"|"mid")])
    # "mid" = phone onset at the unit's boundary; "pre" = shortly before it
    # (VCV consonant — its true onset inside the sample isn't indexed).
    i = 0
    unit_overrides = {int(k): str(v) for k, v in
                      dict(unit_overrides or {}).items()}
    while i < len(seq) - 1:
        x, y = seq[i], seq[i + 1]
        # VCV fallback: some C+V transitions exist only as "prev CV" units
        # (Japanese-style recording) — "u-ki" covers u->k->i in one slice
        if (i + 2 < len(seq) and not db.has(f"{x}-{y}")
                and db.has(f"{x}-{y}{seq[i + 2]}")):
            picked.append(f"{x}-{y}{seq[i + 2]}")
            plan.append((picked[-1], i, [(i + 1, "pre"), (i + 2, "mid")]))
            i += 2
            continue
        # word-final consonant: the bank records "prev C-" as one unit
        if (y != "pau" and i + 2 < len(seq) and seq[i + 2] == "pau"
                and db.has(f"{x}-{y}_")):
            final_pair = f"{x}-{y}_"
            picked.append(db.choose(
                final_pair, seq[i - 1] if i > 0 else "*", "pau",
                unit_overrides.get(i)))
            plan.append((picked[-1], i, [(i + 1, "mid")]))
            i += 2
            continue
        cands = [f"{x}-{y}"]
        cands += [f"{x}-{y2}" for y2 in ALT_VOWELS.get(y, ())]
        cands += [f"{x2}-{y}" for x2 in ALT_VOWELS.get(x, ())]
        if x == y:
            cands.append(f"{x}-{x}")
        for cand in cands:
            outer_left = seq[i - 1] if i > 0 else "*"
            outer_right = seq[i + 2] if i + 2 < len(seq) else "*"
            selected = db.choose(cand, outer_left, outer_right,
                                 unit_overrides.get(i))
            if db.has(selected):
                picked.append(selected)
                plan.append((selected, i, [(i + 1, "mid")]))
                break
        else:
            # gap fallback: the bank's Japanese-style CV units ("ki") cover
            # C+V as one sample whose mid is the C/V boundary
            if y != "pau" and db.has(f"{x}{y}-{x}{y}"):
                picked.append(f"{x}{y}-{x}{y}")
                plan.append((picked[-1], i, [(i + 1, "mid")]))
            else:
                skipped.append(f"{x}-{y}")
        i += 1

    fr, out = None, array.array("h")
    try:
        expected_f0_hz = float(db.metadata.get("average_pitch_hz") or 0.0)
    except (AttributeError, TypeError, ValueError):
        expected_f0_hz = 0.0
    events = []          # (abs_sample, seq_pos) — phone onsets in the output
    splice_records = []  # exact crossfade intervals between rendered units
    next_pos = 0         # first seq position still awaiting an onset
    for unit_index, (dip, p, marks) in enumerate(plan):
        r, chunk, bnd = db.slice_info(
            dip, half_ms=slice_half_ms, copy_samples=False)
        fr = fr or r
        if not unit_index:
            chunk_start = len(out)
            out.extend(chunk)
        elif legacy_joins:
            n_eff = min(int(crossfade_ms / 1000 * r), len(out), len(chunk))
            chunk_start = len(out) - max(0, n_eff)
            if n_eff:
                splice_records.append({
                    "unit_index": unit_index,
                    "sample": chunk_start + n_eff // 2,
                    "handoff_start_sample": chunk_start,
                    "handoff_end_sample": chunk_start + n_eff,
                    "position_source": "python-legacy-linear-crossfade",
                    "join_method": "legacy-linear-crossfade",
                    "estimated": False,
                })
            _xfade(out, chunk, int(crossfade_ms / 1000 * r))
        else:
            shared_phone = seq[p] if 0 <= p < len(seq) else ""
            shared_class = _phone_context_class(shared_phone)
            closure_may_be_silent = bool(
                str(shared_phone).lower() not in {"dx", "dxy"} and
                shared_class in {
                    "silence", "stop_voiceless", "stop_voiced",
                    "affricate_voiceless", "affricate_voiced",
                })
            decision = adaptive_join_pcm16(
                out, chunk, r,
                expected_f0_hz=expected_f0_hz or None,
                allow_silent_handoff=closure_may_be_silent,
                left_phone=shared_phone,
                right_phone=shared_phone)
            chunk_start = (decision.handoff_start_sample -
                           decision.right_skip_samples)
            record = decision.to_dict()
            record.update({
                "unit_index": unit_index,
                "sample": record.pop("splice_sample"),
                "handoff_start_sample": record.pop(
                    "handoff_start_sample"),
                "handoff_end_sample": record.pop("handoff_end_sample"),
                "position_source": "python-measured-crossfade",
                "join_method": record.pop("method"),
                "estimated": False,
            })
            splice_records.append(record)
        # phones whose pair was skipped never got an onset: they start
        # (approximately) where this chunk starts
        while next_pos <= p:
            events.append((chunk_start, next_pos))
            next_pos += 1
        for pos, kind in marks:
            if pos < next_pos:
                continue
            at = chunk_start + bnd
            if kind == "pre":    # VCV consonant: shortly before the boundary
                at = chunk_start + bnd - min(int(0.090 * r), bnd // 2)
            events.append((at, pos))
            next_pos = pos + 1
    if not out:
        raise ValueError(f"nothing rendered (missing diphones: {skipped})")
    edge = int(edge_fade_ms / 1000 * fr)
    for i in range(min(edge, len(out))):
        out[i] = int(out[i] * i / edge)
        out[-1 - i] = int(out[-1 - i] * i / edge)
    # peak-normalize to -3 dB
    peak = max(1, max(out), -min(out))
    gain = min(4.0, 0.707 * 32767 / peak)
    np = sys.modules.get("numpy")
    if np is not None:
        # The GUI already owns NumPy. Truncation matches Python's int() so
        # this fast path is byte-identical to the dependency-free fallback.
        values = np.frombuffer(out, dtype=np.int16).astype(np.float64)
        normalized = np.clip(
            np.trunc(values * gain), -32768, 32767).astype(np.int16)
        out = array.array("h")
        out.frombytes(normalized.tobytes())
    else:
        out = array.array("h", [
            max(-32768, min(32767, int(sample * gain))) for sample in out
        ])

    # per-phone segments: clamp onsets monotonic, then span onset -> next onset
    segments = []
    last = -1
    onsets = []
    for at, pos in events:
        at = max(at, last + 1)
        if at >= len(out):
            break
        onsets.append((at, seq[pos]))
        last = at
    for k, (at, ph) in enumerate(onsets):
        end = onsets[k + 1][0] if k + 1 < len(onsets) else len(out)
        segments.append({"phone": ph, "start": round(at / fr, 5),
                         "end": round(end / fr, 5)})
    for record in splice_records:
        record["time"] = round(record["sample"] / fr, 9)
        record["handoff_start"] = round(
            record.pop("handoff_start_sample") / fr, 9)
        record["handoff_end"] = round(
            record.pop("handoff_end_sample") / fr, 9)
        record["segment_index"] = min(
            range(len(segments)),
            key=lambda index: abs(
                (segments[index]["start"] + segments[index]["end"]) * 0.5
                - record["time"]),
        ) if segments else -1

    data = None
    if encode_wav or out_path:
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(fr)
            w.writeframes(out.tobytes())
        data = buf.getvalue()
        if out_path:
            Path(out_path).write_bytes(data)
    selected_units = {str(pos): dip.split("-", 1)[0]
                      for dip, pos, _marks in plan}
    result = {"wav": data, "phones": list(phones), "diphones": picked,
              "skipped": skipped, "seconds": round(len(out) / fr, 2),
              "speed": round(speed, 3), "framerate": fr,
              "segments": segments, "selected_units": selected_units,
              "splice_records": splice_records,
              "join_mode": "legacy" if legacy_joins else "measured"}
    if return_pcm:
        # The final buffer belongs to the caller. Cached source slices remain
        # internal and are only read by the renderer.
        result["pcm16"] = out
    return result


def synth_text(cfg, text: str, lang: str = "asaxi", out_path=None, speed=None,
               *, legacy_joins: bool = False):
    """One-call front end used by the CLI and the HTTP API. `speed` (>1 faster,
    <1 slower) falls back to cfg['synth_speed'] then 1.0."""
    db = _cached_synth_text_db(find_db(cfg))
    if lang == "en":
        phones = g2p_english(text)
    elif lang in ("ja", "jp"):
        phones = g2p_japanese(text)
    else:
        phones = g2p_asaxi(text)
    if not phones:
        raise ValueError(f"no phonemes derived from {text!r}")
    if speed is None:
        speed = cfg.get("synth_speed", 1.0)
    return render(
        db, phones, out_path=out_path, speed=speed,
        legacy_joins=bool(legacy_joins))


def _cached_synth_text_db(root):
    """Reuse hot decoded WAVs for CLI/HTTP one-call synthesis."""
    root = Path(root).expanduser().resolve()
    index_path = root / "dic" / "diphone_index.json"
    stat = index_path.stat()
    identity = (
        int(stat.st_mtime_ns), int(stat.st_size),
        file_change_token(index_path, stat),
    )
    key = (str(root), identity)
    with _synth_text_db_cache_lock:
        cached = _synth_text_db_cache.pop(key, None)
        if cached is not None:
            _synth_text_db_cache[key] = cached
            return cached
        for stale in [item for item in _synth_text_db_cache
                      if item[0] == str(root)]:
            old = _synth_text_db_cache.pop(stale)
            old.clear_cache()
        database = DiphoneDB(root)
        _synth_text_db_cache[key] = database
        while len(_synth_text_db_cache) > SYNTH_TEXT_DB_CACHE_MAX_VOICES:
            _old_key, old = _synth_text_db_cache.popitem(last=False)
            old.clear_cache()
        return database


def clear_synth_text_cache():
    with _synth_text_db_cache_lock:
        for database in _synth_text_db_cache.values():
            database.clear_cache()
        _synth_text_db_cache.clear()


def synth_text_cache_info():
    with _synth_text_db_cache_lock:
        return {
            "voices": len(_synth_text_db_cache),
            "max_voices": SYNTH_TEXT_DB_CACHE_MAX_VOICES,
            "databases": [
                {"root": key[0], **database.cache_info()}
                for key, database in _synth_text_db_cache.items()
            ],
        }


# ------------------------------------------------------------- standalone CLI
# Lets you render audio directly from the DB, driven by the same festvox.json
# the builder and GUI use. See GUIDE.md.

def _find_festvox_config(explicit=None):
    import os
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("FESTVOX_CONFIG"):
        cands.append(Path(os.environ["FESTVOX_CONFIG"]))
    here = Path(__file__).resolve().parent
    cands += [Path.cwd() / "festvox.json", here / "festvox.json"]
    for c in cands:
        if c and c.is_file():
            return c
    return None


def _db_from_config(fcfg, voice=None):
    """Resolve a voice key in festvox.json to its built DB directory."""
    voices = fcfg.get("voices") or {}
    key = voice or fcfg.get("default_voice") or (next(iter(voices), None))
    if key and key in voices and voices[key].get("out"):
        return Path(voices[key]["out"]), key
    root = fcfg.get("output_root") or "."
    return (Path(root) / key if key else None), key


def safe_name(text, maxlen=48):
    """Filesystem-safe slug that KEEPS non-ASCII letters (kana, accented
    Asaxi, etc.) so distinct inputs get distinct filenames. Falls back to a
    short hash only when nothing usable remains (e.g. punctuation only)."""
    s = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")[:maxlen]
    s = s.strip("_")
    if not s:
        import hashlib
        s = "u" + hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return s


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render text with a diphone voice (Asaxi or English), "
                    "standalone. DB + output dir come from festvox.json unless "
                    "overridden.")
    ap.add_argument("text")
    ap.add_argument("--lang", default=None, choices=["asaxi", "en", "ja"],
                    help="default: festvox.json 'default_lang', else 'en'")
    ap.add_argument("--voice", default=None,
                    help="voice key in festvox.json (default: default_voice)")
    ap.add_argument("--db", default=None,
                    help="path to a built DB dir (overrides --voice/config)")
    ap.add_argument("--config", default=None, help="path to festvox.json")
    ap.add_argument("--out", default=None, help="explicit output wav path")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: config synth_output_dir "
                         "or current dir); filename auto-generated from text")
    ap.add_argument("--speed", type=float, default=None,
                    help="pace: >1 faster, <1 slower (default: festvox.json "
                         "'synth_speed', else 1.0)")
    ap.add_argument(
        "--legacy-joins", action="store_true",
        help="use the exact pre-measured linear join path for A/B diagnosis")
    a = ap.parse_args()

    fcfg, fp = {}, None
    cf = _find_festvox_config(a.config)
    if cf:
        fcfg = json.loads(cf.read_text(encoding="utf-8"))
        fp = cf

    if a.db:
        db_dir = Path(a.db)
    else:
        db_dir, key = _db_from_config(fcfg, a.voice)
        if not db_dir:
            raise SystemExit("No DB: pass --db, or set voices/output_root in "
                             f"festvox.json (looked at {fp}).")
    cfg = {"festvox_db": [str(db_dir)]}

    lang = a.lang or fcfg.get("default_lang") or "en"
    if a.out:
        out_path = Path(a.out)
    else:
        outdir = Path(a.outdir or fcfg.get("synth_output_dir") or ".")
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{lang}_{safe_name(a.text)}.wav"

    speed = a.speed if a.speed is not None else fcfg.get("synth_speed", 1.0)
    r = synth_text(
        cfg, a.text, lang, out_path=str(out_path), speed=speed,
        legacy_joins=a.legacy_joins)
    print(json.dumps({"out": str(out_path), "db": str(db_dir), "speed": speed,
                      "seconds": r["seconds"], "phones": r["phones"],
                      "skipped": r["skipped"],
                      "join_mode": r.get("join_mode")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    _main()
