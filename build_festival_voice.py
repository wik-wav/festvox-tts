# -*- coding: utf-8 -*-
"""Build one language-scoped Festival/UniSyn voice from a UTAU bank.

The primary Windows command accepts ``--language ja|en|asaxi``, an explicit
``--bank-type``, Windows-visible source/OTO paths, and a Windows output folder.
Festival and EST tools run locally when available or through the selected WSL
distribution. The source UTAU bank is read only; all converted audio,
pitchmarks, indexes, Scheme, and metadata are written beneath ``--output``.

English and Asaxi use the established ARPAsing converter. Japanese uses the
separate CV/VCV/CVVC candidate and assembly path. Every output declares one
language, one alias namespace, one configuration identity, and one voice entry
point. WSL paths are derived at the external-tool boundary and generated
Scheme resolves its root from Festival's load path, so builds are portable.

The older ``--db``/``--utau`` command remains as a compatibility interface.
New builds should use the unified command documented in
``UNIFIED_VOICE_BUILDER.md``.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from speaker_pitch import (
    analyze_speaker_pitch,
    automatic_pitch_metadata,
    pitchmark_bounds,
    recommended_default_pitch_hz,
)
from source_timing import build_source_timing_profile
from source_window import (
    DEFAULT_SOURCE_WINDOW_MS,
    DEFAULT_ZERO_OVERLAP_GUARD_MS,
    SOURCE_WINDOW_MODES,
)
from special_phones import (
    declared_display_phones,
    generated_voice_policy,
    parse_literal_phone_map_specs,
    parse_special_phone_mode_specs,
)
from unisyn_runtime import (
    RUNTIME_AUDIO_STORAGE_MODES,
    apply_runtime_audio_metadata,
    build_grouped_runtime,
    separate_runtime_metadata,
)

from voice_manifest import (
    DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION,
    VoiceConfiguration,
    generated_voice_fields,
    source_recording_bundle_from_paths,
)
from voice_paths import (
    UNIFIED_BUILDER_VERSION,
    VoicePathError,
    validate_build_layout,
    windows_to_wsl_path,
)

# ---------------------------------------------------------------- Asaxi data
# Kept in sync with the bundled synth_diphone.py so Festival's text
# front end produces the same phones as the pure-Python engine.
ASAXI_RULES = [
    ("nŋ", ["nng"]), ("nn", ["nn"]), ("mm", ["mm"]),
    ("ch", ["ch"]), ("sh", ["sh"]), ("dh", ["dh"]), ("jh", ["jh"]),
    ("zh", ["zh"]), ("th", ["th"]), ("dz", ["dz"]),
    ("si", ["sh", "i"]),
    ("ni", ["ny", "i"]),
    ("å", ["a", "w"]), ("ă", ["a", "y"]),
    ("ë", ["e", "y"]), ("ỏ", ["o", "w"]),
    ("ő", ["o", "y"]), ("ů", ["u", "w"]),
    ("è", ["ax"]), ("ě", ["er"]),
    ("ý", ["ih"]), ("ù", ["u"]), ("á", ["ao"]),
    ("a", ["a"]), ("e", ["e"]), ("i", ["i"]), ("o", ["o"]), ("u", ["u"]),
    ("ŕ", ["dx"]), ("ń", ["ny"]), ("ś", ["sh"]), ("ŋ", ["ng"]),
    ("'", ["q"]), ("x", ["hh"]), ("c", ["ts"]), ("j", ["y"]),
    ("b", ["b"]), ("d", ["d"]), ("f", ["f"]), ("g", ["g"]), ("h", ["h"]),
    ("k", ["k"]), ("l", ["l"]), ("m", ["m"]), ("n", ["n"]), ("p", ["p"]),
    ("r", ["r"]), ("s", ["s"]), ("t", ["t"]), ("v", ["v"]), ("w", ["w"]),
    ("y", ["y"]), ("z", ["z"]),
]
STOPS = ["p", "t", "k", "b", "d", "g", "ch", "ts", "dz", "jh", "q", "dx",
         "cl"]
FRICS = ["f", "v", "s", "z", "sh", "zh", "th", "dh", "hh", "h"]
VOWELS = ["a", "e", "i", "o", "u", "aw", "ay", "ey", "ow", "oy", "uw",
          "ax", "er", "ih", "ao", "iy", "uh", "eh", "aa", "ah", "ae"]
PALATALS = [c + "y" for c in
            "b d g k m n p r t h l v f ng dx".split()]
ALT_VOWELS = {"i": "iy", "u": "uw", "e": "eh", "o": "ow", "a": "aa",
              "iy": "i", "uw": "u", "eh": "e", "ow": "o", "aa": "a",
              "ah": "a", "ax": "a", "ih": "i"}
VOICELESS_STOPS = set("p t k q py ty ky cl".split())
VOICED_STOPS = set("b d g by dy gy dx dxy".split())
VOICELESS_AFFRICATES = {"ch", "ts"}
VOICED_AFFRICATES = {"jh", "dz"}
VOICELESS_FRICS = set("f s sh th h hh fy hy".split())
VOICED_FRICS = set("v z zh dh vy zi".split())
NASALS = set("m n ng nn mm nng xn my ny ngy".split())
LIQUIDS = set("l r rr ly ry ri".split())
GLIDES = set("w y wi".split())
VOICED_SIBILANTS = {"z", "zh", "zi", "dz", "jh"}

RE_SPECIALS = r".^$*+?()[]{}\|"

OUTPUT_CALIBRATION_POLICY = dict(
    DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION
)


def rx_escape(s: str) -> str:
    return "".join("\\" + c if c in RE_SPECIALS else c for c in s)


def scm_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --------------------------------------------------------------- build steps
UNIT_VARIANT_RE = re.compile(r"(?:__u\d+(?:__w[lrb])?|__w[lrb])$")


def unit_base_phone(phone: str) -> str:
    return UNIT_VARIANT_RE.sub("", str(phone))


def phone_context_class(phone: str) -> str:
    base = unit_base_phone(phone).rstrip("_").lower()
    if base == "*":
        return "wildcard"
    for values, label in (
            (set(VOWELS), "vowel"),
            (VOICELESS_STOPS, "stop_voiceless"),
            (VOICED_STOPS, "stop_voiced"),
            (VOICELESS_AFFRICATES, "affricate_voiceless"),
            (VOICED_AFFRICATES, "affricate_voiced"),
            (VOICELESS_FRICS, "fricative_voiceless"),
            (VOICED_FRICS, "fricative_voiced"),
            (NASALS, "nasal"), (LIQUIDS, "liquid"), (GLIDES, "glide")):
        if base in values:
            return label
    if base in {"pau", "sil", "sp"}:
        return "silence"
    if base.endswith("y"):
        return phone_context_class(base[:-1])
    return "other"


def context_edge_info(phone: str, edge: str) -> dict:
    """Classify a directional OTO alias edge without using WAV filenames."""
    if edge not in {"left", "right"}:
        raise ValueError("edge must be 'left' or 'right'")
    base = unit_base_phone(phone).rstrip("_").lower()
    direct_class = phone_context_class(base)
    if base == "*":
        return {"phone": "*", "class": "wildcard",
                "kind": "wildcard_unknown"}
    if direct_class != "other":
        return {"phone": base, "class": direct_class, "kind": "atomic"}
    for vowel in sorted(set(VOWELS), key=lambda item: (-len(item), item)):
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


def normalize_alternative_contexts(alternatives: dict) -> dict:
    """Enrich old DB alternatives with the current OTO-only context model."""
    normalized = {}
    for diphone, choices in dict(alternatives or {}).items():
        rows = []
        for source in choices or []:
            choice = dict(source)
            for side, edge in (("left", "right"), ("right", "left")):
                info = context_edge_info(
                    choice.get(side + "_context") or "*", edge)
                choice[side + "_context_edge"] = info["phone"]
                choice[side + "_context_kind"] = info["kind"]
                choice[side + "_class"] = info["class"]
            rows.append(choice)
        normalized[str(diphone)] = rows
    return normalized


def load_db_metadata(db: Path) -> dict:
    p = db / "dic" / "diphone_index.json"
    if not p.is_file():
        sys.exit(f"error: {p} not found (build the DB with utau2festvox.py first)")
    return json.loads(p.read_text(encoding="utf-8"))


def load_index(db: Path) -> dict:
    return load_db_metadata(db)["index"]


def phone_inventory(index: dict):
    # ``cl`` is a canonical structural timing phone, not necessarily a
    # literal source-bank alias. Current generated voices realize it through
    # the following consonant and the generated C-C hold unit, so it must be
    # declared even when no OTO entry happens to be named "cl".
    toks = {"pau", "cl"}
    for dip in index:
        a, b = dip.split("-", 1)
        toks.add(unit_base_phone(a))
        toks.add(unit_base_phone(b))
    return sorted(toks)


def _special_phone_policy(values, literal_phone_maps=None) -> dict:
    try:
        overrides = parse_special_phone_mode_specs(values)
        literal_mappings = parse_literal_phone_map_specs(
            literal_phone_maps
        )
        return generated_voice_policy(
            overrides,
            literal_phone_mappings=literal_mappings,
        )
    except ValueError as exc:
        raise VoicePathError(str(exc)) from exc


def _validate_literal_special_phone_sources(
    policy: dict, index: dict,
) -> None:
    """Require real source edges for each creator-declared literal alias."""
    mappings = dict(policy.get("literal_phone_mappings") or {})
    source_inventory = set()
    for raw_pair in dict(index or {}):
        if "-" not in str(raw_pair):
            continue
        left, right = str(raw_pair).split("-", 1)
        source_inventory.update({
            unit_base_phone(left).rstrip("_"),
            unit_base_phone(right).rstrip("_"),
        })
    for display, settings in sorted(mappings.items()):
        phone = str(dict(settings or {}).get("source_phone") or "")
        if not phone:
            raise VoicePathError(
                f"literal phone mapping {display!r} has no source phone"
            )
        if display in source_inventory:
            raise VoicePathError(
                f"literal display phone {display!r} collides with an "
                "authored source phone. Choose a distinct display name."
            )
        incoming = set()
        outgoing = set()
        for raw_pair, value in dict(index or {}).items():
            if "-" not in str(raw_pair):
                continue
            left, right = str(raw_pair).split("-", 1)
            left = unit_base_phone(left).rstrip("_")
            right = unit_base_phone(right).rstrip("_")
            source_name = str((value or [""])[0])
            if Path(source_name).stem.casefold() in {
                "_silence", "_japanese_silence", "silence", "sil"
            }:
                continue
            if right == phone:
                incoming.add(f"{left}-{right}")
            if left == phone:
                outgoing.add(f"{left}-{right}")
        if not incoming or not outgoing:
            missing = []
            if not incoming:
                missing.append(f"an incoming X-{phone} unit")
            if not outgoing:
                missing.append(f"an outgoing {phone}-X unit")
            raise VoicePathError(
                f"{display}={phone} requires explicitly authored /{phone}/ "
                f"source coverage ({' and '.join(missing)} are missing). "
                f"Structural {phone} remains available separately."
            )


def copy_wavs(index: dict, db: Path, out: Path):
    if (db / "wav").resolve() == (out / "wav").resolve():
        raise ValueError(
            "voice output must be separate from the input diphone database"
        )
    (out / "wav").mkdir(parents=True, exist_ok=True)
    used = sorted({v[0] for v in index.values()})
    for w in used:
        src, dst = db / "wav" / w, out / "wav" / w
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
    return used


def write_runtime_metadata(metadata: dict, index: dict,
                           alternatives: dict, out: Path) -> Path:
    """Install the final unit index beside a generated Festival voice.

    The GUI reads this copy for X-X sustain previews.  Write the post-build
    index because silence installation and onset repair may add or adjust
    rows after the source diphone database was loaded.
    """
    folder = out / "dic"
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata or {})
    payload["context_model"] = "oto_directional_v1"
    payload["index"] = dict(index or {})
    normalized = normalize_alternative_contexts(alternatives)
    payload["alternatives"] = normalized
    payload["source_timing_profile"] = build_source_timing_profile(normalized)
    path = folder / "diphone_index.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def bank_durations(index: dict, phones):
    """Natural per-phone durations measured from the bank itself (the OTO
    geometry survives in the index): phone p's tail half = median(mid-start)
    over units "p-*", head half = median(end-mid) over "*-p"; natural length
    = head + tail, clamped per class. This replaces one-size-fits-all
    defaults so the voice's timing follows how the singer actually spoke."""
    import statistics
    stops, frics, vowels = set(STOPS), set(FRICS), set(VOWELS)
    tails, heads = {}, {}
    for dip, (_, s, m, e) in index.items():
        a, b = dip.split("-", 1)
        a, b = unit_base_phone(a), unit_base_phone(b)
        tails.setdefault(a, []).append(max(0.0, m - s))
        heads.setdefault(b, []).append(max(0.0, e - m))

    def med(xs):
        return statistics.median(xs) if xs else 0.0

    out = {}
    for p in phones:
        base = unit_base_phone(p).rstrip("_")
        nat = med(heads.get(p, [])) + med(tails.get(p, []))
        if p == "pau":
            lo, hi, fallback = 0.10, 0.20, 0.15
        elif base in stops:
            lo, hi, fallback = 0.035, 0.090, 0.060
        elif base in vowels:
            lo, hi, fallback = 0.060, 0.300, 0.130
        elif base in frics:
            lo, hi, fallback = 0.050, 0.160, 0.090
        else:
            lo, hi, fallback = 0.040, 0.180, 0.080
        out[p] = round(min(hi, max(lo, nat)) if nat > 0.001 else fallback, 3)
    return out


def fix_initial_stops(index: dict, mode: str = "closure"):
    """Fix glitchy / spilling phrase-initial obstruents (stops AND fricatives).
    Two different bank problems, handled two ways (mode "closure", default):

    STOPS: the rest-onset entries ('- b', '- k', ...) are aliased to a
    calibration file ('sil.wav') with a bogus blank (-4925 ms), so every
    pau-STOP unit spans ~4.9 s of it -- and that file holds loud sustained
    vowel blobs. TD-PSOLA crushes them into the stop's first half -> the "two
    repeated glitched pitched signals" at phrase start (independent of the
    phone's duration). Fix: repoint pau-STOP to the real _silence.wav unit -> a
    clean silent closure, with the single burst coming from the STOP-vowel unit.

    FRICATIVES (dh, v, s, ...): these ARE real '- dh'/'- v' recordings, but the
    pau/fric boundary is labelled `offset + preutterance`, ~100 ms INTO the
    frication -- so that frication lands in the pau half and spills into the
    pause. Frication actually begins at `offset` (pre-offset is silence). Fix:
    move the boundary back to `offset` and pull the unit start into the leading
    silence, so the pau half is silent (no spill) AND frication fills the
    fricative from its very start (no late onset).

    Both guarded by the existence of a C-vowel onset unit; nasals/approximants
    measured clean and are left alone.
    mode "keep": legacy -- only nudge the labelled join ~12 ms (fixes neither;
    an escape hatch)."""
    stops, frics = set(STOPS), set(FRICS)
    SIL = ["_silence.wav", 0.02, 0.15, 0.28]   # clean unit from ensure_silence_unit
    PAD = 0.030                                 # silent lead-in kept before frication
    n = 0
    for dip, v in list(index.items()):
        a, b = dip.split("-", 1)
        abase, base = unit_base_phone(a), unit_base_phone(b).rstrip("_")
        if abase != "pau" or base not in (stops | frics):
            continue
        # is there a C-onset unit (b-a, dh-i, ...) carrying the real consonant?
        has_onset = any(d != dip
                        and unit_base_phone(d.split("-", 1)[0]) != "pau"
                        and unit_base_phone(d.split("-", 1)[0]).rstrip("_") == base
                        for d in index)
        w, s, m, e = v
        if mode != "closure" or not has_onset:
            m2 = max(s + 0.004, m - 0.012)       # legacy nudge only
            if m2 < m:
                index[dip] = [w, s, round(m2, 6), e]
                n += 1
            continue
        if base in stops or (e - s) > 0.5:
            index[dip] = list(SIL)               # stop / garbage span -> silence
            n += 1
        else:                                     # real fricative: re-align onset
            index[dip] = [w, round(max(0.0, s - PAD), 6), round(s, 6),
                          round(e, 6)]
            n += 1
    return n


SILENCE_PHONES = {"pau", "sil", "sp"}  # phones that must render as silence


def install_silence_units(index: dict, out: Path):
    """Guarantee a real silence unit and make EVERY silence-type phone (pau,
    sil, sp -- rests and boundaries) render as true silence.

    Writes a 0.30 s digital-silence wav (_silence.wav) and points pau-pau at it,
    so it is also the default-diphone fallback for missing units. Then it
    neutralises the bank's silence/calibration file: UTAU banks routinely alias
    their rest / glottal onsets ('- b', 'q b', held
    'n n' ...) to ONE placeholder wav with a bogus blank, so each such unit
    spans seconds of stray audio that TD-PSOLA smears into pause glitches.
    That file is detected as whatever the all-silence diphones point at,
    plus anything obviously named 'sil' /
    'silence' -- and every unit cut from it, together with every all-silence
    pair, is repointed to _silence.wav. Detection by span is deliberately NOT
    used (real held sustains like 'n n' can be seconds long too). Banks that
    have no pau/sil at all still get the _silence.wav fallback, so gaps stay
    quiet instead of glitchy."""
    import wave as _wave
    SIL = ["_silence.wav", 0.02, 0.15, 0.28]

    def bp(p):
        return unit_base_phone(p).rstrip("_")

    # 1) samplerate from any real source wav, then write the silence wav
    sr = 44100
    for entry in index.values():
        if entry[0] != "_silence.wav":
            try:
                with _wave.open(str(out / "wav" / entry[0]), "rb") as w:
                    sr = w.getframerate()
                break
            except OSError:
                continue
    with _wave.open(str(out / "wav" / "_silence.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * 0.30))

    # 2) detect the bank's silence / calibration wav(s)
    sil_wavs = set()
    for dip, v in index.items():
        a, b = dip.split("-", 1)
        if bp(a) in SILENCE_PHONES and bp(b) in SILENCE_PHONES:
            sil_wavs.add(v[0])
        if Path(v[0]).stem.lower() in ("sil", "silence", "_sil", "_silence"):
            sil_wavs.add(v[0])
    sil_wavs.discard("_silence.wav")

    # 3) repoint every unit on those wavs + every all-silence pair to silence
    n = 0
    for dip, v in list(index.items()):
        a, b = dip.split("-", 1)
        both = bp(a) in SILENCE_PHONES and bp(b) in SILENCE_PHONES
        if (v[0] in sil_wavs or both) and list(v) != SIL:
            index[dip] = list(SIL)
            n += 1
    index["pau-pau"] = list(SIL)
    return n


def write_est(index: dict, out: Path, name: str, *, legacy: bool = False):
    (out / "dic").mkdir(parents=True, exist_ok=True)
    suffix = "_legacy" if legacy else ""
    est = out / "dic" / f"{name}_diphone{suffix}.est"
    with est.open("w", encoding="ascii") as f:
        f.write("EST_File index\nDataType ascii\nNumEntries %d\n"
                "IndexName %s_diphone%s\nEST_Header_End\n" % (
                    len(index), name, suffix))
        for dip in sorted(index):
            w, s, m, e = index[dip]
            f.write(f"{dip} {Path(w).stem} {s:.6f} {m:.6f} {e:.6f}\n")
    return est


def _external_command(args, wsl_distro=None):
    """Run EST/Festival locally, or derive WSL paths on Windows."""
    executable = str(args[0])
    if shutil.which(executable):
        return list(map(str, args))
    if os.name != "nt":
        raise FileNotFoundError(executable)
    wsl = shutil.which("wsl") or shutil.which("wsl.exe")
    if not wsl:
        raise FileNotFoundError(executable)
    command = [wsl]
    if str(wsl_distro or "").strip():
        command.extend(["-d", str(wsl_distro).strip()])
    command.append("--")
    command.extend(windows_to_wsl_path(item) for item in args)
    return command


def _run_external(args, *, wsl_distro=None, timeout=None):
    command = _external_command(args, wsl_distro=wsl_distro)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _source_pitchmark_wavs(metadata: dict, source_root=None) -> dict:
    """Map generated WAV names to read-only UTAU recordings for FRQ lookup."""
    if source_root is None:
        return {}
    root = Path(source_root).resolve()
    index = dict(metadata.get("index") or {})
    result = {}
    for choices in dict(metadata.get("alternatives") or {}).values():
        for choice in choices or ():
            index_name = str(choice.get("index_name") or "")
            row = index.get(index_name)
            relative = str(choice.get("wav") or "").replace("\\", "/")
            if not row or not relative:
                continue
            source = (root / Path(relative)).resolve()
            try:
                source.relative_to(root)
            except ValueError:
                continue
            if source.is_file():
                result.setdefault(Path(str(row[0])).name, source)
    return result


def make_pitchmarks(used_wavs, out: Path, f0min: float, f0max: float,
                    verbose=False, wsl_distro=None, *, default_f0=None,
                    source_root=None, metadata=None,
                    f0_estimator="harvest"):
    """Create FRQ/WORLD-guided source epochs for every language route."""
    from japanese_festival import make_pitchmarks as make_source_pitchmarks

    names = tuple(sorted(set(used_wavs)))
    count = make_source_pitchmarks(
        out,
        names,
        f0_min=float(f0min),
        f0_max=float(f0max),
        default_f0=float(
            default_f0 if default_f0 is not None
            else (float(f0min) + float(f0max)) * 0.5
        ),
        source_wavs=_source_pitchmark_wavs(
            dict(metadata or {}),
            Path(source_root) if source_root is not None else None,
        ),
        f0_estimator=str(f0_estimator),
        distro=wsl_distro,
    )
    if verbose:
        for name in names:
            print("  pm", name)
    return count


# ------------------------------------------------------------ scheme writing
def gen_scheme(name: str, out: Path, index: dict, phones, f0: float,
               alternatives=None, language_mode="legacy",
               enable_japanese=False,
               runtime_audio_storage="grouped") -> Path:
    if runtime_audio_storage not in RUNTIME_AUDIO_STORAGE_MODES:
        raise ValueError(
            "runtime_audio_storage must be grouped or separate")
    prefer_grouped = "t" if runtime_audio_storage == "grouped" else "nil"
    default_diphone = ("pau-pau" if "pau-pau" in index else sorted(index)[0])
    vowels = set(VOWELS)

    def phone_def(p):
        if p == "pau":
            return f"   (pau  -  0 - - - 0 0 -)"
        base = unit_base_phone(p).rstrip("_")
        context = phone_context_class(base)
        if context == "vowel":
            return f"   ({p}  +  l 1 1 - 0 0 +)"
        if context in {"stop_voiceless", "stop_voiced"}:
            ctype = "s"
        elif context in {"affricate_voiceless", "affricate_voiced"}:
            ctype = "a"
        elif context in {"fricative_voiceless", "fricative_voiced"}:
            ctype = "f"
        elif context == "nasal":
            ctype = "n"
        elif context == "liquid":
            ctype = "l"
        elif context == "glide":
            ctype = "r"
        else:
            return f"   ({p}  -  0 - - - 0 0 0)"
        voiced = "+" if context in {
            "stop_voiced", "affricate_voiced", "fricative_voiced",
            "nasal", "liquid", "glide",
        } else "-"
        return f"   ({p}  -  0 - - - {ctype} a {voiced})"

    g2p_rules = "\n".join(
        "    (%s %s (%s))" % (scm_str(g), scm_str(rx_escape(g) + ".*"),
                              " ".join(ph))
        for g, ph in ASAXI_RULES)

    alt_pairs = sorted({(a, b) for a, b in ALT_VOWELS.items()
                        if a in phones or b in phones})
    alts = " ".join("(%s %s)" % ab for ab in alt_pairs)

    # timing measured from the bank (OTO geometry) -- see bank_durations()
    bdur = bank_durations(index, phones)
    durs = "\n".join("    (%s %.3f)" % (p, bdur[p]) for p in phones)
    # Structural cl is a timed consonant region, never a PhoneSet silence.
    sil_decl = " ".join(["pau"] + [p for p in ("sil", "sp")
                                    if p in set(phones)])
    variant_rows = []
    context_tokens = set(phones) | {"*"}
    for dip, choices in sorted((alternatives or {}).items()):
        if not choices:
            continue
        encoded_choices = []
        for choice in choices:
            left_name = str(choice.get(
                "left_name", dip.split("-", 1)[0]
            ))
            left_context = str(choice.get("left_context") or "*")
            right_context = str(choice.get("right_context") or "*")
            context_tokens.update((left_context, right_context))
            left_info = context_edge_info(left_context, "right")
            right_info = context_edge_info(right_context, "left")
            left_activation = choice.get("window_left_activation")
            right_activation = choice.get("window_right_activation")
            encoded_choices.append(
                "(%s %s %s %s %s %s %s %s %s %.6f %.6f)" %
                (scm_str(left_name),
                 scm_str(left_context), scm_str(right_context),
                 scm_str(choice.get("l_class", "*")),
                 scm_str(left_info["class"]),
                 scm_str(right_info["class"]),
                 scm_str(choice.get("window_left_name") or left_name),
                 scm_str(choice.get("window_right_name") or left_name),
                 scm_str(choice.get("window_both_name") or left_name),
                 (float(left_activation) if left_activation is not None
                  else 1000000.0),
                 (float(right_activation) if right_activation is not None
                  else 1000000.0)))
        encoded = " ".join(encoded_choices)
        variant_rows.append("  (%s (%s))" % (scm_str(dip), encoded))
    variant_data = "\n".join(variant_rows)
    context_classes = "\n".join(
        "    (%s %s %s)" %
        (scm_str(phone),
         scm_str(context_edge_info(phone, "left")["class"]),
         scm_str(context_edge_info(phone, "right")["class"]))
        for phone in sorted(context_tokens))
    voiced_sibilants = VOICED_SIBILANTS | {
        phone + "_" for phone in VOICED_SIBILANTS}
    scm = f""";; {name}.scm -- REAL Festival voice for the {name} diphone bank.
;; Generated by build_festival_voice.py. UniSyn diphones, TD-PSOLA (psola_sep:
;; pitchmarks + raw wavs). Asaxi text front end ported from synth_diphone.py.

;; The backend prepends the generated voice root before loading this file.
;; Resolve from that entry so the generated tree remains relocatable.
(defvar {name}_dir (car load-path)
  "Location of the {name} voice directory.")

;; ---------------------------------------------------------------- phoneset
(defPhoneSet {name}
  ((vc + -) (vlng s l d a 0) (vheight 1 2 3 0 -) (vfront 1 2 3 0 -)
   (vrnd + - 0) (ctype s f a n l r 0) (cplace l a p b d v g 0) (cvox + - 0))
  (
{chr(10).join(phone_def(p) for p in phones)}
  ))
(PhoneSet.silences '({sil_decl}))

;; -------------------------------------------- context-sensitive OTO takes
;; Each row is (diphone ((unit left right L-class left-class right-class) ...)).
;; us_diphone_left changes only this segment's outgoing unit lookup; phone
;; names and the active phoneset remain untouched.
(set! {name}_unit_variants '(
{variant_data}
  ))
(defvar festvox_gui_unit_variant_overrides nil)

(define ({name}_variant_by_name choices wanted)
  (cond ((null choices) nil)
        ((string-equal (car (car choices)) wanted) (car choices))
        (t ({name}_variant_by_name (cdr choices) wanted))))

(define ({name}_choice_field choice index)
  (if (> index 0)
      ({name}_choice_field (cdr choice) (- index 1))
      (car choice)))

(define ({name}_segment_duration seg)
  (let ((previous (item.prev seg)))
    (if previous
        (- (item.feat seg "end") (item.feat previous "end"))
        (item.feat seg "end"))))

(define ({name}_source_window_name choice seg next)
  (let ((left-long
         (not (< ({name}_segment_duration seg)
                 ({name}_choice_field choice 9)))))
    (let ((right-long
           (not (< ({name}_segment_duration next)
                   ({name}_choice_field choice 10)))))
      (cond ((and left-long right-long)
             ({name}_choice_field choice 8))
            (left-long ({name}_choice_field choice 6))
            (right-long ({name}_choice_field choice 7))
            (t (car choice))))))

(define ({name}_phone_left_edge_class phone)
  (let ((row (assoc_string phone {name}_phone_context_classes)))
    (if row (cadr row) "other")))

(define ({name}_phone_right_edge_class phone)
  (let ((row (assoc_string phone {name}_phone_context_classes)))
    (if row (car (cddr row)) "other")))

(define ({name}_variant_context_score expected actual expected_class
                                      actual_class exact_score)
  (cond ((string-equal expected "*") 0)
        ((string-equal expected actual) exact_score)
        ((and (not (string-equal expected_class "wildcard"))
              (not (string-equal expected_class "other"))
              (string-equal expected_class actual_class)) 4)
        (t -8)))

(define ({name}_variant_score choice outer_left outer_right l_class)
  (+ ({name}_variant_context_score (cadr choice) outer_left
                                     (car (cdr (cdr (cdr (cdr choice)))))
                                     ({name}_phone_right_edge_class outer_left)
                                     6)
     ({name}_variant_context_score (car (cddr choice)) outer_right
                                     (cadr (cdr (cdr (cdr (cdr choice)))))
                                     ({name}_phone_left_edge_class outer_right)
                                     7)
     (if (and (not (string-equal l_class "*"))
              (not (string-equal (cadr (cddr choice)) "*")))
         (if (string-equal (cadr (cddr choice)) l_class) 20 -100)
          0)))

(define ({name}_variant_context_kind expected actual expected_class
                                      actual_class)
  (cond ((string-equal expected "*") 0)
        ((string-equal expected actual) 2)
        ((and (not (string-equal expected_class "wildcard"))
              (not (string-equal expected_class "other"))
              (string-equal expected_class actual_class)) 1)
        (t -1)))

(define ({name}_unsafe_phrase_edge_shortcut choice outer_left outer_right)
  (let ((left_kind
         ({name}_variant_context_kind
          (cadr choice) outer_left
          (car (cdr (cdr (cdr (cdr choice)))))
          ({name}_phone_right_edge_class outer_left))))
    (let ((right_kind
           ({name}_variant_context_kind
            (car (cddr choice)) outer_right
            (cadr (cdr (cdr (cdr (cdr choice)))))
            ({name}_phone_left_edge_class outer_right))))
      (or (and (string-equal outer_left "pau")
               (> left_kind 1) (< right_kind 0))
          (and (string-equal outer_right "pau")
               (> right_kind 1) (< left_kind 0))))))

(define ({name}_best_variant choices outer_left outer_right l_class
                             best best_score)
  (if (null choices)
      best
      (if ({name}_unsafe_phrase_edge_shortcut
           (car choices) outer_left outer_right)
          ({name}_best_variant (cdr choices) outer_left outer_right l_class
                               best best_score)
          (let ((score ({name}_variant_score (car choices)
                                            outer_left outer_right l_class)))
            (if (> score best_score)
                ({name}_best_variant (cdr choices) outer_left outer_right l_class
                                     (car choices) score)
                ({name}_best_variant (cdr choices) outer_left outer_right l_class
                                     best best_score))))))

(define ({name}_sibilant_context_tier choice)
  (let ((class (cadr (cdr (cdr (cdr (cdr choice)))))))
    (cond ((or (string-equal class "vowel")
               (string-equal class "nasal")
               (string-equal class "liquid")
               (string-equal class "glide")
               (string-equal class "fricative_voiced")) 3)
          ((or (string-equal class "wildcard")
               (string-equal class "other")) 2)
          (t 1))))

(define ({name}_max_sibilant_context_tier choices)
  (if (null choices)
      0
      (let ((here ({name}_sibilant_context_tier (car choices))))
        (let ((rest ({name}_max_sibilant_context_tier (cdr choices))))
          (if (> here rest) here rest)))))

(define ({name}_best_variant_at_tier choices outer_left outer_right l_class
                                     wanted_tier best best_score)
  (if (null choices)
      best
      (let ((choice (car choices)))
        (let ((choice_tier ({name}_sibilant_context_tier choice)))
          (if (and (not (> choice_tier wanted_tier))
                   (not (> wanted_tier choice_tier)))
              (let ((score ({name}_variant_score choice outer_left outer_right
                                                  l_class)))
                (if (or (null best) (> score best_score))
                    ({name}_best_variant_at_tier
                     (cdr choices) outer_left outer_right l_class wanted_tier
                     choice score)
                    ({name}_best_variant_at_tier
                     (cdr choices) outer_left outer_right l_class wanted_tier
                     best best_score)))
              ({name}_best_variant_at_tier
               (cdr choices) outer_left outer_right l_class wanted_tier
               best best_score))))))

(define ({name}_automatic_variant choices outer_left outer_right l_class
                                  right_phone)
  (cond ((null choices) nil)
        ((member_string right_phone {name}_voiced_sibilants)
         (let ((tier ({name}_max_sibilant_context_tier choices)))
           (if (> tier 1)
               ({name}_best_variant_at_tier
                choices outer_left outer_right l_class tier nil -100000)
               ;; All following contexts are verified risky: retain base.
               (car choices))))
        (t ({name}_best_variant
            choices outer_left outer_right l_class (car choices)
            ({name}_variant_score (car choices)
                                  outer_left outer_right l_class)))))

(define ({name}_l_class seg next outer_right)
  (cond ((string-equal (item.name seg) "l")
         (if (member_string (item.name next) {name}_l_light_followers)
             "light" "dark"))
        ((string-equal (item.name next) "l")
         (if (member_string outer_right {name}_l_light_followers)
             "light" "dark"))
        (t "*")))

(define ({name}_select_unit_variant seg index)
  (let ((next (item.next seg)))
    (if next
        ;; Keep sequential dependencies explicit with SIOD-portable nested
        ;; bindings; not every Festival build has the extended form.
        (let ((key (string-append (item.name seg) "-" (item.name next))))
          (let ((row (assoc_string key {name}_unit_variants)))
            (let ((choices (if row (cadr row) nil))
                  (prev (item.prev seg))
                  (after (item.next next)))
              (let ((outer_left (if prev (item.name prev) "*"))
                    (outer_right (if after (item.name after) "*")))
                (let ((l_class ({name}_l_class seg next outer_right))
                      (override_row
                       (assoc_string (format nil "%d" index)
                                     festvox_gui_unit_variant_overrides)))
                  (let ((override
                         (if override_row (cadr override_row)
                             (item.feat seg "unit_variant_override"))))
                    (let ((chosen
                           (if (and override
                                    (not (string-equal override "0")))
                               ({name}_variant_by_name choices override)
                               ({name}_automatic_variant
                                choices outer_left outer_right l_class
                                (item.name next)))))
                      (if chosen
                          (item.set_feat seg "us_diphone_left"
                            ({name}_source_window_name
                             chosen seg next)))))))))))))

(define ({name}_select_unit_variant_list segments index)
  (if (null segments)
      nil
      (cons ({name}_select_unit_variant (car segments) index)
            ({name}_select_unit_variant_list (cdr segments) (+ index 1)))))

(define ({name}_select_unit_variants utt)
  ({name}_select_unit_variant_list (utt.relation.items utt 'Segment) 0)
  (set! festvox_gui_unit_variant_overrides nil)
  utt)

;; ------------------------------------------------------- Asaxi g2p (= LTS)
;; rules: (grapheme prefix-regex (phones...)) longest-match first
(set! {name}_g2p_rules '(
{g2p_rules}
  ))
(set! {name}_vowels '({" ".join(sorted(vowels))}))
(set! {name}_phone_context_classes '(
{context_classes}
  ))
(set! {name}_voiced_sibilants '({" ".join(sorted(voiced_sibilants))}))
(set! {name}_l_light_followers '({" ".join(sorted(vowels | {"y"}))}))
(set! {name}_stops '({" ".join(STOPS)}))
(set! {name}_palatals '({" ".join(PALATALS)}))

(define ({name}_find_rule s rules)
  (cond ((null rules) nil)
        ((string-matches s (cadr (car rules))) (car rules))
        (t ({name}_find_rule s (cdr rules)))))

(define ({name}_drop1 s)
  (let ((e (symbolexplode s)))
    (if (or (null e) (null (cdr e))) ""
        (string-after s (format nil "%s" (car e))))))

(define ({name}_g2p_str s phones)
  (if (or (not s) (string-equal s ""))
      (reverse phones)
      (let ((r ({name}_find_rule s {name}_g2p_rules)))
        (if r
            ({name}_g2p_str (string-after s (car r))
                            (append (reverse (car (cddr r))) phones))
            ({name}_g2p_str ({name}_drop1 s) phones)))))

;; gemination: doubled stop -> (cl stop); doubled continuant -> single
(define ({name}_gem phones out)
  (cond ((null phones) (reverse out))
        ((and (not (null out))
              (string-equal (format nil "%s" (car phones))
                            (format nil "%s" (car out)))
              (not (member_string (format nil "%s" (car phones))
                                  {name}_vowels)))
         (if (member_string (format nil "%s" (car phones)) {name}_stops)
             ({name}_gem (cdr phones) (cons (car phones) (cons 'cl (cdr out))))
             ({name}_gem (cdr phones) out)))
        (t ({name}_gem (cdr phones) (cons (car phones) out)))))

;; palatalization: C y V -> Cy V where the bank has the Cy unit
(define ({name}_palatal phones)
  (cond ((null phones) nil)
        ((and (cdr phones) (cdr (cdr phones))
              (string-equal (format nil "%s" (car (cdr phones))) "y")
              (member_string (string-append (format nil "%s" (car phones)) "y")
                             {name}_palatals))
         (cons (intern (string-append (format nil "%s" (car phones)) "y"))
               ({name}_palatal (cdr (cdr phones)))))
        (t (cons (car phones) ({name}_palatal (cdr phones))))))

(define ({name}_g2p word)
  "Asaxi orthography -> phone list (same rules as synth_diphone.py)."
  ({name}_palatal ({name}_gem ({name}_g2p_str (downcase word) nil) nil)))

(define (lex_user_unknown_word word feats)
  (let ((phones ({name}_g2p word)))
    (if (null phones) (set! phones '(pau)))
    (list word nil (list (list (mapcar (lambda (p) (intern (format nil "%s" p)))
                                       phones) 0)))))

(lex.create "{name}")
(lex.set.phoneset "{name}")
(lex.set.lts.method 'function)

;; --------------------------------------------------------------- prosody
(set! {name}_phone_durs '(
{durs}
  ))
(set! {name}_phrase_cart_tree
  '((lisp_token_end_punc in ("?" "." "!" ":" ";" ","))
    ((BB))
    ((n.name is 0)
     ((BB))
     ((NB)))))

(define ({name}_token_to_words token name)
  (list name))

;; --------------------------------------------------------------- UniSyn db
;; The packed group file is the rendering cache. It stores the same indexed
;; units as the separate WAV/PM development database and therefore does not
;; alter contextual selection, timing, F0, or per-occurrence overrides.
(defvar festvox_gui_force_separate_database nil)
(defvar festvox_gui_legacy_joins nil)
(defvar {name}_prefer_grouped_database {prefer_grouped})
(defvar {name}_group_file
  (path-append {name}_dir "group/{name}_diphone.group"))
(set! {name}_separate_db_params
  (list
   (list 'name '{name}_diphone_separate)
   (list 'index_file (path-append {name}_dir "dic/{name}_diphone.est"))
   '(grouped "false")
   (list 'coef_dir (path-append {name}_dir "pm"))
   (list 'sig_dir  (path-append {name}_dir "wav"))
   '(coef_ext ".pm")
   '(sig_ext ".wav")
   '(alternates_left ({alts}))
   '(alternates_right ({alts}))
   (list 'default_diphone "{default_diphone}")))
(set! {name}_legacy_db_params
  (list
   (list 'name '{name}_diphone_legacy)
   (list 'index_file
         (path-append {name}_dir "dic/{name}_diphone_legacy.est"))
   '(grouped "false")
   (list 'coef_dir (path-append {name}_dir "pm"))
   (list 'sig_dir  (path-append {name}_dir "wav"))
   '(coef_ext ".legacy.pm")
   '(sig_ext ".wav")
   '(alternates_left ({alts}))
   '(alternates_right ({alts}))
   (list 'default_diphone "{default_diphone}")))
(set! {name}_grouped_db_params
  (list
   (list 'name '{name}_diphone_grouped)
   (list 'index_file {name}_group_file)
   '(grouped "true")
   '(alternates_left ({alts}))
   '(alternates_right ({alts}))
   (list 'default_diphone "{default_diphone}")))

(define ({name}_runtime_db_params)
  (if (and {name}_prefer_grouped_database
           (not festvox_gui_force_separate_database)
           (probe_file {name}_group_file))
      {name}_grouped_db_params
      {name}_separate_db_params))
(set! {name}_db_name nil)
(set! {name}_legacy_db_name nil)

(define ({name}_active_db_name)
  (if festvox_gui_legacy_joins
      (begin
        (if (null {name}_legacy_db_name)
            (set! {name}_legacy_db_name
                  (us_diphone_init {name}_legacy_db_params)))
        {name}_legacy_db_name)
      (begin
        (if (null {name}_db_name)
            (set! {name}_db_name
                  (us_diphone_init ({name}_runtime_db_params))))
        {name}_db_name)))

(define ({name}_configure_join_windows)
  ;; UniSyn reads these Param values when it extracts and overlap-adds source
  ;; periods.  ARPAsing and integrated voices retain Festival's stable
  ;; symmetric renderer in every language.  Normal mode uses the current
  ;; bridge database; Fault Mode selects the paired pre-fix database.
  (Param.set "unisyn.window_name" "hanning")
  (Param.set "unisyn.window_factor" 1.0)
  (Param.set "unisyn.window_symmetric" 1))

;; ---------------------------------------------------------------- voice
(define (voice_{name})
  "(voice_{name})  Asaxi diphone voice (UniSyn/PSOLA) built from the
UTAU-derived bank by build_festival_voice.py."
  (voice_reset)
  (Parameter.set 'Language '{name})
  (PhoneSet.select '{name})
  (set! token_to_words {name}_token_to_words)
  (set! pos_lex_name nil)
  (lex.select "{name}")
  (require 'phrase)
  (Parameter.set 'Phrase_Method 'cart_tree)
  (set! phrase_cart_tree {name}_phrase_cart_tree)
  (Parameter.set 'Int_Method 'DuffInt)
  (Parameter.set 'Int_Target_Method Int_Targets_Default)
  ;; gentle declination; the GUI's Pitch/Fall controls override these
  (set! duffint_params '((start {f0 * 1.05:.0f}) (end {f0 * 0.88:.0f})))
  (Parameter.set 'Duration_Method 'Default)
  (set! phoneme_durations {name}_phone_durs)
  (set! UniSyn_module_hooks (list {name}_select_unit_variants))
  (set! us_abs_offset 0.0)
  ({name}_configure_join_windows)
  (set! us_rel_offset 0.0)
  (set! us_gain 0.9)
  (Parameter.set 'Synth_Method 'UniSyn)
  (Parameter.set 'us_sigpr 'psola)
  (us_db_select ({name}_active_db_name))
  (set! current-voice '{name})
)

(proclaim_voice
 '{name}
 '((language asaxi)
   (gender female)
   (dialect none)
   (description "Asaxi diphone voice (Lem bank) via UniSyn TD-PSOLA.")))

;; ------------------------------------------------- English front end
;; Real English text processing (CMU lexicon, POS, phrasing, duration and
;; F0 models -- borrowed wholesale from kal_diphone) driving THESE units.
;; The bank is lowercase arpasing, which is the same phone naming the
;; English modules emit; gaps fall back to the pau-pau silence unit.
;; Requires: festvox-kallpc16k, festlex-cmu, festlex-poslex.
(define (voice_{name}_en)
  "(voice_{name}_en)  English (kal front end) over the {name} diphones."
  (voice_kal_diphone)
  ;; Kal selects its narrower `radio` phoneset. Restore this generated bank's
  ;; superset so explicit ARPAsing/Asaxi/Japanese phones such as `q` remain
  ;; legal in English mode while retaining Kal's lexicon and prosody modules.
  (PhoneSet.select '{name})
  ;; Drop kal-specific renaming, but keep this bank's context/take selector.
  (set! UniSyn_module_hooks (list {name}_select_unit_variants))
  ;; kal's F0 model targets a ~105 Hz male voice; recenter it on THIS
  ;; bank's pitch so English comes out at the same pitch as Asaxi (the
  ;; f2b contour SHAPE is kept -- only mean/spread move)
  (set! int_lr_params
        '((target_f0_mean {f0:.0f}) (target_f0_std {f0 * 0.12:.0f})
          (model_f0_mean 170) (model_f0_std 34)))
  ({name}_configure_join_windows)
  (Parameter.set 'Synth_Method 'UniSyn)
  (Parameter.set 'us_sigpr 'psola)
  (us_db_select ({name}_active_db_name))
  (set! current-voice '{name}_en)
)

(proclaim_voice
 '{name}_en
 '((language english)
   (gender female)
   (dialect american)
   (description "English (kal frontend) over the {name} diphone units.")))

;; BEGIN Japanese explicit-phone entry
;; Japanese linguistic analysis, timing and F0 are supplied by the GUI.  This
;; entry point activates the shared ARPAsing phoneset and UniSyn database only.
(define (voice_{name}_ja)
  "(voice_{name}_ja) Japanese explicit-phone frontend over shared units."
  (voice_{name})
  (set! current-voice '{name}_ja)
)

(proclaim_voice
 '{name}_ja
 '((language japanese)
   (gender female)
   (dialect none)
   (description "Japanese explicit phones over shared ARPAsing units.")))
;; END Japanese explicit-phone entry

(provide '{name})
"""
    if not enable_japanese:
        ja_start = scm.index(";; BEGIN Japanese explicit-phone entry\n")
        ja_end = scm.index(";; END Japanese explicit-phone entry\n")
        ja_end += len(";; END Japanese explicit-phone entry\n")
        scm = scm[:ja_start] + scm[ja_end:]
    if language_mode not in {"legacy", "en", "asaxi"}:
        raise ValueError("language_mode must be legacy, en, or asaxi")
    english_marker = ";; ------------------------------------------------- English front end\n"
    provide_marker = f"(provide '{name})"
    if language_mode == "asaxi" and not enable_japanese:
        scm = scm[:scm.index(english_marker)] + provide_marker + "\n"
    elif language_mode == "en" and not enable_japanese:
        voice_marker = ";; ---------------------------------------------------------------- voice\n"
        prefix = scm[:scm.index(voice_marker)]
        english = scm[
            scm.index(english_marker):scm.index(provide_marker)
        ]
        english = english.replace(f"voice_{name}_en", f"voice_{name}")
        english = english.replace(f"'{name}_en", f"'{name}")
        scm = prefix + english + provide_marker + "\n"
    (out / "festvox").mkdir(exist_ok=True)
    p = out / "festvox" / f"{name}.scm"
    p.write_text(scm, encoding="utf-8")
    return p


# ------------------------------------------------------------------- testing
def run_test(out: Path, name: str, festival_bin: str, text: str,
             wsl_distro=None, voice_entry_point=None):
    import uuid

    runtime_out = (
        windows_to_wsl_path(out) if os.name == "nt" else str(out)
    ).rstrip("/")
    entry_point = voice_entry_point or f"voice_{name}"
    tag = uuid.uuid4().hex
    text_wav = out / f".voice_test_text_{tag}.wav"
    text_seg = out / f".voice_test_text_{tag}.seg"
    phones_wav = out / f".voice_test_phones_{tag}.wav"
    scm = f"""(set! load-path (cons "{runtime_out}/festvox" load-path))
(set! load-path (cons "{runtime_out}" load-path))
(load "{runtime_out}/festvox/{name}.scm")
({entry_point})
(set! u1 (SynthText {scm_str(text)}))
(utt.save.wave u1 "{runtime_out}/{text_wav.name}" 'riff)
(utt.save.segs u1 "{runtime_out}/{text_seg.name}")
(set! u2 (Utterance Phones (pau t a k i pau)))
(utt.synth u2)
(utt.save.wave u2 "{runtime_out}/{phones_wav.name}" 'riff)
(print (list 'g2p-taki ({name}_g2p "taki")))
(print (list 'segments (length (utt.relation.items u1 'Segment))))
(print 'VOICE-OK)
"""
    p = out / f".voice_test_{tag}.scm"
    p.write_text(scm, encoding="utf-8")
    try:
        try:
            r = _run_external(
                [festival_bin, "-b", str(p)],
                wsl_distro=wsl_distro,
                timeout=600,
            )
        except FileNotFoundError:
            print("error: Festival is unavailable locally and through WSL")
            return False
    finally:
        p.unlink(missing_ok=True)
    print(r.stdout[-2000:] or "", r.stderr[-2000:] or "", sep="")
    ok = bool(
        r.returncode == 0
        and "VOICE-OK" in (r.stdout or "")
        and text_wav.exists()
        and text_wav.stat().st_size > 44
        and text_seg.exists()
        and phones_wav.exists()
        and phones_wav.stat().st_size > 44
    )
    if ok:
        text_wav.replace(out / "test_text.wav")
        text_seg.replace(out / "test_text.seg")
        phones_wav.replace(out / "test_phones.wav")
    else:
        for artifact in (text_wav, text_seg, phones_wav):
            artifact.unlink(missing_ok=True)
    print("TEST:", "PASS" if ok else "FAIL",
          "->", out / "test_text.wav" if ok else "(see output above)")
    return ok


def run_japanese_test(out: Path, name: str, voice_entry_point: str,
                      festival_bin: str, text: str, base_pitch_hz: float,
                      wsl_distro=None) -> bool:
    """Render the canonical Japanese phone/timing/F0 plan in Festival."""
    import uuid

    from japanese_frontend import analyze_japanese
    from japanese_synthesis import create_synthesis_plan
    from special_phones import resolve_special_phone_sequence

    utterance = analyze_japanese(text, mode="auto")
    plan = create_synthesis_plan(
        utterance,
        runtime_metadata=out,
        base_pitch_hz=base_pitch_hz,
    )
    if not plan.segments:
        print("error: Japanese smoke text produced no canonical segments")
        return False
    runtime_metadata = json.loads(
        (out / "dic" / "diphone_index.json").read_text(encoding="utf-8")
    )
    display_phones = [segment.phone for segment in plan.segments]
    resolution = resolve_special_phone_sequence(
        display_phones,
        metadata=runtime_metadata,
        available_diphones=(
            runtime_metadata.get("index") or {}
        ).keys(),
    )
    if resolution.unresolved:
        for row in resolution.unresolved:
            print(
                "error: unresolved Japanese special phone "
                f"{row.phone!r} at segment {row.index}: {row.status}"
            )
        return False
    render_phones = list(resolution.render_phones)

    starts = []
    cursor = 0.0
    for segment in plan.segments:
        starts.append(cursor)
        cursor += segment.duration
    targets = [[] for _segment in plan.segments]
    for target in plan.f0_targets:
        segment_index = next(
            (
                index for index, (start, segment) in enumerate(
                    zip(starts, plan.segments)
                )
                if start <= target.time < start + segment.duration
            ),
            len(plan.segments) - 1,
        )
        targets[segment_index].append(
            (max(0.0, target.time - starts[segment_index]), target.hz)
        )

    parts = []
    for segment, render_phone, segment_targets in zip(
        plan.segments, render_phones, targets
    ):
        if not re.fullmatch(r"[A-Za-z0-9_@:~#]+", render_phone):
            print(f"error: invalid Japanese Festival phone {render_phone!r}")
            return False
        row = f"({render_phone} {segment.duration:.6f}"
        for offset, hz in segment_targets:
            row += f" ({min(segment.duration, offset):.6f} {hz:.3f})"
        parts.append(row + ")")

    override_rows = []
    for index, unit_name in sorted(plan.unit_overrides.items()):
        position = int(index)
        source_pair_changed = (
            0 <= position + 1 < len(display_phones)
            and (
                display_phones[position], display_phones[position + 1]
            ) != (
                render_phones[position], render_phones[position + 1]
            )
        )
        if (
            not source_pair_changed
            and 0 <= position < len(plan.segments)
            and re.fullmatch(
            r"[A-Za-z0-9_]+", str(unit_name)
            )
        ):
            override_rows.append(f'("{position}" "{unit_name}")')
    overrides = ""
    if override_rows:
        overrides = (
            "(set! festvox_gui_unit_variant_overrides '("
            + " ".join(override_rows)
            + "))\n"
        )

    runtime_out = (
        windows_to_wsl_path(out) if os.name == "nt" else str(out)
    ).rstrip("/")
    tag = uuid.uuid4().hex
    wav = out / f".voice_test_ja_{tag}.wav"
    seg = out / f".voice_test_ja_{tag}.seg"
    scm_path = out / f".voice_test_ja_{tag}.scm"
    scheme = (
        f'(set! load-path (cons "{runtime_out}/festvox" load-path))\n'
        f'(set! load-path (cons "{runtime_out}" load-path))\n'
        f'(load "{runtime_out}/festvox/{name}_ja.scm")\n'
        f'({voice_entry_point})\n'
        + overrides
        + "(set! u (Utterance Segments ("
        + " ".join(parts)
        + ")))\n"
        + "(utt.synth u)\n"
        + f'(utt.save.wave u "{runtime_out}/{wav.name}" \'riff)\n'
        + f'(utt.save.segs u "{runtime_out}/{seg.name}")\n'
        + "(print (list 'JAPANESE-VOICE-OK "
        + "(length (utt.relation.items u 'Segment))))\n"
    )
    scm_path.write_text(scheme, encoding="utf-8", newline="\n")
    try:
        try:
            result = _run_external(
                [festival_bin, "-b", str(scm_path)],
                wsl_distro=wsl_distro,
                timeout=600,
            )
        except FileNotFoundError:
            print("error: Festival is unavailable locally and through WSL")
            return False
    finally:
        scm_path.unlink(missing_ok=True)
    print(result.stdout[-2000:] or "", result.stderr[-2000:] or "", sep="")
    ok = bool(
        result.returncode == 0
        and "JAPANESE-VOICE-OK" in (result.stdout or "")
        and wav.exists()
        and wav.stat().st_size > 44
        and seg.exists()
    )
    if ok:
        wav.replace(out / "test_text.wav")
        seg.replace(out / "test_text.seg")
    else:
        wav.unlink(missing_ok=True)
        seg.unlink(missing_ok=True)
    print("JAPANESE TEST:", "PASS" if ok else "FAIL")
    return ok


def run_utau_conversion(bank: Path, out: Path, name: str,
                        character_yaml=None, prefix_map=None,
                        alias_prefixes=None, alias_suffixes=None,
                        voice_color=None, oto_files=None,
                        phoneme_map=None, source_window_mode="adaptive",
                        source_window_ms=DEFAULT_SOURCE_WINDOW_MS,
                        zero_overlap_guard_ms=
                        DEFAULT_ZERO_OVERLAP_GUARD_MS) -> Path:
    """One-stop path: UTAU bank (audio + oto.ini files) -> DB -> voice.
    Imports utau2festvox.py (same folder as this script) and runs its
    convert() into OUT/db, returning that DB dir."""
    bank = Path(bank).expanduser().resolve()
    requested_oto_scopes = tuple(oto_files or ())
    selected_oto_files = _selected_oto_files(bank, requested_oto_scopes)
    _validate_single_pitch_oto_scope(
        bank, requested_oto_scopes, selected_oto_files
    )

    import importlib.util
    here = Path(__file__).resolve().parent
    src = here / "utau2festvox.py"
    if not src.is_file():
        sys.exit(f"error: utau2festvox.py not found next to this script "
                 f"({here}) -- needed for --utau")
    spec = importlib.util.spec_from_file_location("utau2festvox", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    db = out / "db"
    print(f"converting UTAU bank {bank} -> {db} (utau2festvox)")
    mod.convert(bank, db, name, True,
                character_yaml=character_yaml, prefix_map=prefix_map,
                alias_prefixes=alias_prefixes,
                alias_suffixes=alias_suffixes, voice_color=voice_color,
                oto_files=selected_oto_files, phoneme_profile=phoneme_map,
                source_window_mode=source_window_mode,
                source_window_ms=source_window_ms,
                zero_overlap_guard_ms=zero_overlap_guard_ms)
    return db


def _legacy_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a REAL Festival (UniSyn/PSOLA) voice from a "
                    "utau2festvox diphone DB, or straight from an UTAU "
                    "voicebank (--utau). See the module docstring.")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--db",
                    help="diphone DB dir (wav/ + dic/diphone_index.json), "
                         "e.g. /mnt/e/Portable_Software/FestVox_DBs/asaxi_lem")
    src_group.add_argument("--utau",
                    help="UTAU voicebank dir (audio + root/nested oto.ini): "
                         "runs the "
                         "utau2festvox conversion first, then builds the "
                         "voice -- the whole pipeline in one command")
    ap.add_argument("--out", required=True,
                    help="voice output dir (keep it inside the WSL fs, "
                         "e.g. ~/voices/asaxi_lem)")
    ap.add_argument("--name", default=None,
                    help="voice name -> voice_<name> (default: out dir name)")
    ap.add_argument("--character-yaml", default=None,
                    help="OpenUtau character.yaml path for --utau; preferred "
                         "over prefix.map and auto-detected at bank root")
    ap.add_argument("--prefix-map", default=None,
                    help="legacy prefix.map path for --utau; auto-detected "
                         "at bank root")
    ap.add_argument("--oto", action="append", default=[],
                    help="one oto.ini file or one single-pitch folder inside "
                         "--utau; multiple scopes are refused")
    ap.add_argument("--alias-prefix", action="append", default=[],
                    help="manual UTAU alias prefix to remove; repeatable")
    ap.add_argument("--alias-suffix", action="append", default=[],
                    help="manual UTAU alias suffix to remove; repeatable; "
                         "supports color before or after a pitch tag")
    ap.add_argument("--voice-color", default=None,
                    help="OpenUtau color to build. Default is uncolored; "
                         "use 'all' only to include every declared color")
    ap.add_argument(
        "--special-phone-mode", action="append", default=[],
        metavar="PHONE=MODE",
        help=("override a language-neutral special-phone realization; "
              "repeatable. Generated voices default to "
              "cl=anticipatory_consonant. cl=literal is a compatibility "
              "shorthand for adding cl_literal=cl while retaining the "
              "structural phone."),
    )
    ap.add_argument(
        "--literal-phone-map", action="append", default=[],
        metavar="DISPLAY=SOURCE",
        help=("expose an explicitly authored special source phone under a "
              "distinct canonical display token; repeatable, for example "
              "cl_literal=cl. OTO aliases never enable this automatically."),
    )
    ap.add_argument(
        "--source-window-mode", choices=SOURCE_WINDOW_MODES,
        default="adaptive",
        help=("adaptive bounds normal phones and restores full source halves "
              "only for sufficiently stretched phones; full restores the "
              "legacy index"),
    )
    ap.add_argument(
        "--source-window-ms", type=float,
        default=DEFAULT_SOURCE_WINDOW_MS,
        help="maximum source milliseconds per normal diphone half",
    )
    ap.add_argument(
        "--zero-overlap-guard-ms", type=float,
        default=DEFAULT_ZERO_OVERLAP_GUARD_MS,
        help=("experimental source-cut guard for zero OTO overlap; "
              "disabled by default because it changes recorded geometry"),
    )
    ap.add_argument("--f0min", type=float, default=None,
                    help="min pitch (Hz) for pitchmarking. Default: auto-detect "
                         "from the bank -- crucial so phrase-final pitch decays "
                         "stay tracked (a too-high f0min scatters marks -> warble)")
    ap.add_argument("--f0max", type=float, default=None,
                    help="max pitch (Hz) for pitchmarking. Default: auto-detect")
    ap.add_argument("--f0", type=float, default=None,
                    help="base synthesis pitch (Hz). Default: the bank's median "
                         "(this E3 bank ~ 165)")
    ap.add_argument(
        "--f0-estimator", choices=("harvest", "dio"), default="harvest",
        help=("fallback for source recordings without usable UTAU FRQ data: "
              "Harvest favors voiced coverage; DIO is faster"),
    )
    ap.add_argument("--skip-pm", action="store_true",
                    help="skip pitchmarking (only regenerate scheme/index)")
    ap.add_argument(
        "--runtime-audio-storage",
        choices=RUNTIME_AUDIO_STORAGE_MODES,
        default="grouped",
        help=("grouped packs indexed PSOLA units into one cached UniSyn "
              "file; separate retains development-time per-WAV access"),
    )
    ap.add_argument("--initial-stops", choices=("closure", "keep"),
                    default="closure",
                    help="phrase-initial obstruent (stop+fricative) handling: "
                         "'closure' silences the pau-C half (fixes stop "
                         "double-burst AND fricative spill); 'keep' = legacy nudge")
    ap.add_argument("--test", action="store_true",
                    help="load the voice in festival and synthesize")
    ap.add_argument("--test-text", default="taki")
    ap.add_argument("--festival-bin", default="festival")
    ap.add_argument("--wsl-distro", default="",
                    help="WSL distribution used for EST/Festival on Windows")
    ap.add_argument("--unified-internal", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--language-mode", choices=("legacy", "en", "asaxi"),
                    default="legacy", help=argparse.SUPPRESS)
    ap.add_argument("--phoneme-map", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--enable-japanese", action="store_true",
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        special_phone_policy = _special_phone_policy(
            a.special_phone_mode,
            a.literal_phone_map,
        )
    except VoicePathError as exc:
        ap.error(str(exc))

    out = Path(a.out).expanduser()
    name = re.sub(r"[^A-Za-z0-9_]", "_", a.name or out.name)
    if a.utau:
        bank = Path(a.utau).expanduser().resolve()
        try:
            selected_oto_files = _selected_oto_files(bank, a.oto)
            _validate_single_pitch_oto_scope(
                bank, a.oto, selected_oto_files
            )
            selected_source_recordings = _source_recordings_from_oto_files(
                bank, selected_oto_files
            )
        except VoicePathError as exc:
            ap.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    if a.utau:
        phoneme_map = None
        if a.phoneme_map:
            from arpasing_profile import load_arpasing_profile
            phoneme_map = load_arpasing_profile(a.phoneme_map)
        db = run_utau_conversion(
            Path(a.utau).expanduser(), out, name,
            character_yaml=a.character_yaml, prefix_map=a.prefix_map,
            alias_prefixes=a.alias_prefix,
            alias_suffixes=a.alias_suffix, voice_color=a.voice_color,
            oto_files=(a.oto or None), phoneme_map=phoneme_map,
            source_window_mode=a.source_window_mode,
            source_window_ms=a.source_window_ms,
            zero_overlap_guard_ms=a.zero_overlap_guard_ms)
    else:
        if (a.character_yaml or a.prefix_map or a.oto or a.alias_prefix or
                a.alias_suffix or a.voice_color):
            ap.error("UTAU metadata/affix options require --utau, not --db")
        db = Path(a.db).expanduser()

    metadata = load_db_metadata(db)
    index = metadata["index"]
    alternatives = normalize_alternative_contexts(
        metadata.get("alternatives") or {})
    metadata["context_model"] = "oto_directional_v1"
    metadata["special_phone_realizations"] = special_phone_policy
    try:
        _validate_literal_special_phone_sources(
            special_phone_policy, index
        )
    except VoicePathError as exc:
        ap.error(str(exc))
    phones = phone_inventory(index)
    print(f"{len(index)} diphones, {len(phones)} phone symbols")

    used = copy_wavs(index, db, out)
    n_sil = install_silence_units(index, out)
    used = used + ["_silence.wav"]
    phones = phone_inventory(index)
    metadata["phones"] = declared_display_phones(
        phones, special_phone_policy
    )
    print(f"silence unit installed; {n_sil} calibration/silence-phone units "
          "-> silence (pau/sil/sp boundaries)")
    # fricative onset re-align + any remaining pau-stop closure.
    n_fixed = fix_initial_stops(index, a.initial_stops)
    if n_fixed:
        print(f"fixed {n_fixed} phrase-initial obstruents (mode={a.initial_stops}) "
              "-- clean silent closure (stop double-burst + fricative spill)")
    print(f"wav/: {len(used)} files")
    write_est(index, out, name)
    write_est(index, out, name, legacy=True)
    print(f"dic/{name}_diphone.est written")
    pitch_analysis = analyze_speaker_pitch(
        Path(a.utau).expanduser() if a.utau else db,
        recording_files=(selected_source_recordings if a.utau else None),
    )
    d_min, d_max = pitchmark_bounds(pitch_analysis)
    d_med = float(pitch_analysis.median_f0_hz)
    f0min, f0max, f0 = a.f0min, a.f0max, a.f0
    if f0min is None or f0max is None or f0 is None:
        f0min = a.f0min if a.f0min is not None else d_min
        f0max = a.f0max if a.f0max is not None else d_max
        f0 = a.f0 if a.f0 is not None else d_med
        print(f"auto f0 ({pitch_analysis.source}): median ~{d_med:.0f} Hz -> "
              f"f0min={f0min} f0max={f0max} f0={f0:.0f}")
    if not a.skip_pm:
        n = make_pitchmarks(
            used, out, f0min, f0max, wsl_distro=a.wsl_distro,
            default_f0=f0,
            source_root=(Path(a.utau).expanduser() if a.utau else None),
            metadata=metadata,
            f0_estimator=a.f0_estimator,
        )
        print(f"pm/: {n} new pitchmark files ({len(used)} total)")
    metadata["average_pitch_hz"] = float(f0)
    metadata["f0_min_hz"] = float(f0min)
    metadata["f0_max_hz"] = float(f0max)
    metadata["f0_fallback_estimator"] = str(a.f0_estimator)
    metadata["speaker_pitch_analysis"] = pitch_analysis.to_dict()
    metadata["runtime_audio_storage"] = separate_runtime_metadata(
        requested=a.runtime_audio_storage)
    metadata["source_timing_profile"] = build_source_timing_profile(
        alternatives)
    alternatives_payload = {
        "version": 3,
        "context_model": "oto_directional_v1",
        "diphone_geometry_model": metadata.get(
            "diphone_geometry_model", "legacy_index_geometry"
        ),
        "average_pitch_hz": float(f0),
        "f0_min_hz": float(f0min), "f0_max_hz": float(f0max),
        "f0_fallback_estimator": str(a.f0_estimator),
        "speaker_pitch_analysis": pitch_analysis.to_dict(),
        "source_timing_profile": metadata["source_timing_profile"],
        "special_phone_realizations": special_phone_policy,
        "phones": list(metadata.get("phones") or ()),
        "diphones": alternatives,
    }
    for key in ("subbank_mode", "subbank_note", "alias_metadata",
                "source_subbanks", "oto_files"):
        if key in metadata:
            alternatives_payload[key] = metadata[key]
    (out / "dic" / "unit_alternatives.json").write_text(
        json.dumps(alternatives_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_runtime_metadata(metadata, index, alternatives, out)
    print("dic/diphone_index.json installed for GUI sustain previews")
    scheme_path = gen_scheme(
        name, out, index, phones, f0, alternatives,
        language_mode=a.language_mode,
        enable_japanese=a.enable_japanese,
        runtime_audio_storage=(
            a.runtime_audio_storage if not a.skip_pm else "separate"),
    )
    print(f"festvox/{name}.scm written  (voice_{name})")

    if a.runtime_audio_storage == "grouped" and not a.skip_pm:
        print("packing indexed source audio into the grouped UniSyn cache")
        storage = build_grouped_runtime(
            out,
            voice_name=name,
            scheme_path=scheme_path,
            voice_entry_point=f"voice_{name}",
            festival_bin=a.festival_bin,
            run_external=_run_external,
            wsl_distro=(a.wsl_distro or None),
        )
        metadata["runtime_audio_storage"] = storage
        apply_runtime_audio_metadata(out, storage)
        print(
            "group/%s_diphone.group written (%d bytes)" %
            (name, storage["group_file_bytes"])
        )
    elif a.runtime_audio_storage == "grouped":
        print(
            "grouped runtime deferred because --skip-pm was selected; "
            "the generated Scheme uses separate WAV/PM files"
        )

    if a.test:
        if not run_test(
            out, name, a.festival_bin, a.test_text,
            wsl_distro=a.wsl_distro,
        ):
            sys.exit(1)
    if a.unified_internal:
        print("\nGenerated voice folder:", out)
    else:
        print("\nDone. In the GUI: Voicebank > Add Festival voice by WSL path...")
        print("  ", out)


def _selected_oto_files(samples: Path, values) -> tuple[Path, ...]:
    selected = []
    for raw in values or ():
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            selected.extend(
                item for item in path.rglob("*")
                if item.is_file() and item.name.casefold() == "oto.ini"
            )
        elif path.is_file():
            selected.append(path)
        else:
            raise VoicePathError(f"OTO source not found: {path}")
    if not selected:
        selected = [
            item for item in samples.rglob("*")
            if item.is_file() and item.name.casefold() == "oto.ini"
        ]
    result = []
    for path in selected:
        resolved = path.resolve()
        try:
            resolved.relative_to(samples.resolve())
        except ValueError as exc:
            raise VoicePathError(
                f"Selected OTO is outside the sample folder: {resolved}"
            ) from exc
        result.append(resolved)
    result = sorted(
        dict.fromkeys(result),
        key=lambda path: path.relative_to(samples).as_posix().casefold(),
    )
    if not result:
        raise VoicePathError(f"No oto.ini found under {samples}")
    return tuple(result)


def _validate_single_pitch_oto_scope(
    samples: Path,
    requested_scopes,
    oto_files: tuple[Path, ...],
) -> None:
    """Reject implicit or explicit OTO selections spanning pitch folders.

    Multipitch routing is not part of the stable builder.  A single explicit
    pitch folder may contain several nested ``oto.ini`` files, but selecting
    unrelated folders (or auto-discovering several folders at bank root) must
    fail before conversion rather than silently merging pitches.
    """
    source = samples.resolve()
    scopes = []
    for raw in requested_scopes or ():
        path = Path(raw).expanduser().resolve()
        scopes.append(path)
    unique_scopes = sorted(set(scopes), key=lambda path: str(path).casefold())
    if len(unique_scopes) > 1:
        raise VoicePathError(
            "More than one --oto scope was selected. Build exactly one pitch "
            "at a time by passing one oto.ini or one pitch folder; multipitch "
            "merging is disabled."
        )
    if len(oto_files) > 1 and not unique_scopes:
        relative_parents = sorted({
            path.parent.relative_to(source).as_posix()
            for path in oto_files
        })
        raise VoicePathError(
            "Multiple OTO folders were discovered automatically. Select "
            "exactly one pitch folder with --oto. Found: " +
            ", ".join(relative_parents[:8])
        )
    # UTAU note names use an uppercase note letter.  Keep only the final
    # note-like token from each path/entry header: numbered phoneme takes such
    # as ``a11E3``, ``b1PE3`` and ``E1PE3`` may contain earlier note-shaped
    # fragments, while the actual pitch suffix is the final ``E3`` token.
    pitch_pattern = re.compile(r"([A-G](?:#|b)?-?\d)")

    def final_pitch_tag(value: str):
        matches = tuple(pitch_pattern.finditer(value))
        return matches[-1].group(1).upper() if matches else None

    pitch_tags = set()
    for path in oto_files:
        for component in path.relative_to(source).parts:
            path_tag = final_pitch_tag(component)
            if path_tag:
                pitch_tags.add(path_tag)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            entry_header = line.split(",", 1)[0]
            wav_field, separator, alias_field = entry_header.partition("=")
            for field in ((wav_field, alias_field) if separator
                          else (entry_header,)):
                entry_tag = final_pitch_tag(field)
                if entry_tag:
                    pitch_tags.add(entry_tag)
    if len(pitch_tags) > 1:
        raise VoicePathError(
            "The selected OTO scope contains multiple pitch tags "
            f"({', '.join(sorted(pitch_tags))}). Build one pitch at a time; "
            "multipitch merging is disabled."
        )
    if len(oto_files) <= 1:
        return
    scope = unique_scopes[0]
    if scope.is_file():
        # One explicit file cannot legitimately expand to several OTO files.
        raise VoicePathError(
            "One --oto file unexpectedly expanded to multiple OTO files; "
            "the build was stopped to prevent pitch merging."
        )
    # A directory argument is the user's explicit single-pitch boundary. The
    # detected pitch tags above still catch accidentally selecting a parent
    # that contains named E3/F3-style subbanks.


def _source_recordings_from_oto_files(
    samples: Path,
    oto_files: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return only WAVs referenced by the selected single-pitch OTO scope."""
    from utau2festvox import read_text_fallback

    root = samples.resolve()
    recordings = set()
    for oto_path in oto_files:
        text, _encoding = read_text_fallback(oto_path)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")) or "=" not in line:
                continue
            wav_raw = line.split("=", 1)[0].strip().replace("\\", "/")
            if not wav_raw:
                continue
            source = (oto_path.parent / Path(wav_raw)).resolve()
            try:
                source.relative_to(root)
            except ValueError:
                continue
            if source.is_file() and source.suffix.casefold() == ".wav":
                recordings.add(source)
    return tuple(sorted(
        recordings,
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    ))


def _source_recordings_from_metadata(samples: Path, metadata: dict):
    paths = {}
    for choices in dict(metadata.get("alternatives") or {}).values():
        for choice in choices or ():
            relative = str(choice.get("wav") or "").replace("\\", "/")
            if not relative:
                continue
            path = (samples / Path(relative)).resolve()
            try:
                path.relative_to(samples.resolve())
            except ValueError:
                continue
            if path.is_file():
                paths[path] = None
    return tuple(sorted(
        paths,
        key=lambda path: path.relative_to(samples).as_posix().casefold(),
    ))


def _shared_manifest_payload(metadata: dict) -> dict:
    keys = (
        "voice_manifest_schema_version", "source_bundle_id",
        "configuration_id", "primary_language", "supported_languages",
        "alias_system", "alias_namespace", "canonical_phone_namespace",
        "voice_entry_points", "source_recording_bundle",
        "voice_configuration", "builder_version",
        "front_door_builder_version",
        "speaker_pitch_analysis", "average_pitch_hz",
        "automatic_pitch_floor_hz", "automatic_pitch_headroom_semitones",
        "default_pitch_source", "output_calibration",
        "f0_min_hz", "f0_max_hz",
        "f0_fallback_estimator", "diphone_geometry_model",
        "runtime_audio_storage", "special_phone_realizations",
        "phones",
    )
    return {
        "kind": "generated_festival_voice_manifest",
        **{key: metadata[key] for key in keys if key in metadata},
    }


def _write_shared_manifest(output: Path, metadata: dict) -> Path:
    path = output / "dic" / "voice_manifest.json"
    path.write_text(
        json.dumps(
            _shared_manifest_payload(metadata),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def _install_non_japanese_manifest(
    *,
    samples: Path,
    oto_files,
    output: Path,
    name: str,
    language: str,
    supported_languages=None,
    phoneme_profile=None,
    bank_type: str,
    speaker_pitch_analysis,
    selected_voice_color=None,
    pitch_policy=None,
) -> dict:
    path = output / "dic" / "diphone_index.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    recordings = _source_recordings_from_metadata(samples, metadata)
    metadata_files = tuple(
        path for path in (
            samples / "character.yaml",
            samples / "prefix.map",
            samples / "presamp.ini",
        ) if path.is_file()
    )
    bundle = source_recording_bundle_from_paths(
        samples,
        oto_files=oto_files,
        recording_files=recordings,
        metadata_files=metadata_files,
        speaker_pitch_analysis=speaker_pitch_analysis.to_dict(),
    )
    languages = tuple(dict.fromkeys(supported_languages or (language,)))
    entry_points = {}
    if "asaxi" in languages:
        entry_points["asaxi"] = f"voice_{name}"
    if "en" in languages:
        entry_points["en"] = (
            f"voice_{name}" if languages == ("en",) else f"voice_{name}_en"
        )
    if "ja" in languages:
        entry_points["ja"] = f"voice_{name}_ja"
    source_window_policy = dict(
        metadata.get("source_window_policy") or {}
    )
    source_window_policy.update({
        "normal_unisyn_window_symmetric": True,
        "legacy_unisyn_window_symmetric": True,
        "unisyn_window_policy_reason": (
            "ARPAsing and integrated voices use Festival's stable symmetric "
            "renderer in every enabled language; normal builds retain the "
            "new bridge assets while Legacy joins selects paired pre-fix "
            "assets."
        ),
    })
    metadata["source_window_policy"] = source_window_policy
    policy = {
            "oto_files": [
                path.relative_to(samples).as_posix() for path in oto_files
            ],
            "voice_color": selected_voice_color,
            "multipitch_routing": False,
            "merged_voice_colors": False,
            "source_window_policy": dict(source_window_policy),
    }
    if phoneme_profile is not None:
        policy["phoneme_map_sha256"] = phoneme_profile.source_sha256
    if len(languages) > 1:
        configuration = VoiceConfiguration.arpasing(
            source_bundle_id=bundle.source_bundle_id,
            primary_language=language,
            supported_languages=languages,
            configuration_policy=policy,
            voice_entry_points=entry_points,
            selected_voice_color=selected_voice_color,
        )
    else:
        configuration = VoiceConfiguration.single_language(
            source_bundle_id=bundle.source_bundle_id,
            language=language,
            bank_type=bank_type,
            alias_system=(
                "utau-english-arpasing-v1" if language == "en"
                else "utau-asaxi-arpasing-v1"
            ),
            configuration_policy=policy,
            voice_entry_point=entry_points[language],
            frontend=(
                "festival-kal-english-v1" if language == "en"
                else "festival-asaxi-g2p-v1"
            ),
            duration_model=(
                "festival-kal-duration-v1" if language == "en"
                else "oto-measured-duration-v1"
            ),
            prosody_model=(
                "festival-kal-intonation-v1" if language == "en"
                else "festival-duffint-v1"
            ),
            selected_voice_color=selected_voice_color,
        )
    metadata.update(generated_voice_fields(bundle, configuration))
    metadata.update({
        "kind": "festival_unisyn_runtime_index",
        "language": language,
        "voice_entry_point": entry_points[language],
        "builder_version": UNIFIED_BUILDER_VERSION,
        "front_door_builder_version": UNIFIED_BUILDER_VERSION,
        "phones": declared_display_phones(
            phone_inventory(metadata.get("index") or {}),
            metadata.get("special_phone_realizations") or {},
        ),
        **dict(pitch_policy or {}),
        "output_calibration": dict(OUTPUT_CALIBRATION_POLICY),
    })
    write_runtime_metadata(
        metadata,
        metadata.get("index") or {},
        metadata.get("alternatives") or {},
        output,
    )
    alternatives_path = output / "dic" / "unit_alternatives.json"
    if alternatives_path.is_file():
        alternatives = json.loads(
            alternatives_path.read_text(encoding="utf-8")
        )
        alternatives.update(_shared_manifest_payload(metadata))
        alternatives_path.write_text(
            json.dumps(
                alternatives, ensure_ascii=False, sort_keys=True, indent=2
            ) + "\n",
            encoding="utf-8",
        )
    _write_shared_manifest(output, metadata)
    return metadata


def _mark_japanese_front_door(output: Path) -> dict:
    from japanese_festival import load_japanese_runtime_metadata

    metadata = load_japanese_runtime_metadata(output)
    metadata["front_door_builder_version"] = UNIFIED_BUILDER_VERSION
    path = output / "dic" / "diphone_index.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_shared_manifest(output, metadata)
    return metadata


def _validate_generated_layout(output: Path, metadata: dict) -> None:
    entry_points = dict(metadata.get("voice_entry_points") or {})
    required = (
        output / "dic" / "diphone_index.json",
        output / "dic" / "voice_manifest.json",
        output / "wav",
        output / "festvox",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Generated voice validation failed; missing: "
            + ", ".join(missing)
        )
    if not metadata.get("source_bundle_id") \
            or not metadata.get("configuration_id") or not entry_points:
        raise RuntimeError(
            "Generated voice validation failed; compatibility identity is "
            "incomplete."
        )
    storage = dict(metadata.get("runtime_audio_storage") or {})
    if storage.get("effective") == "grouped":
        group_file = output / str(storage.get("group_file") or "")
        if not group_file.is_file() or group_file.stat().st_size <= 64:
            raise RuntimeError(
                "Generated voice validation failed; grouped runtime audio "
                "is declared but its group file is missing."
            )


def _unified_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build one language-scoped Festival voice from a Windows-visible "
            "UTAU source. WSL paths are derived internally."
        )
    )
    parser.add_argument("--language", required=True,
                        choices=("ja", "en", "asaxi"))
    parser.add_argument("--bank-type", required=True,
                        choices=("cv", "vcv", "cvvc", "arpasing"))
    parser.add_argument("--samples", required=True,
                        help="Windows UTAU sample/bank folder")
    parser.add_argument("--oto", action="append", default=[],
                        help=("exactly one selected oto.ini or one "
                              "single-pitch containing folder"))
    parser.add_argument("--output", "--out", dest="output", required=True,
                        help="Windows generated-voice folder")
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile", default=None,
                        help="Japanese profile JSON")
    parser.add_argument(
        "--enable-language", action="append", default=[],
        choices=("en", "asaxi", "ja"),
        help=("additional frontend for an ARPAsing build; repeatable. "
              "Japanese uses --phoneme-map and explicit GUI timing/F0."),
    )
    parser.add_argument(
        "--phoneme-map", default=None,
        help=("ARPAsing phoneme/grapheme profile YAML. Defaults to the "
              "bundled en-jap mapping."),
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="update known files in an existing generated voice")
    parser.add_argument("--character-yaml", default=None)
    parser.add_argument("--prefix-map", default=None)
    parser.add_argument("--alias-prefix", action="append", default=[])
    parser.add_argument("--alias-suffix", action="append", default=[])
    parser.add_argument("--voice-color", default=None)
    parser.add_argument(
        "--special-phone-mode", action="append", default=[],
        metavar="PHONE=MODE",
        help=("override a language-neutral special-phone realization; "
              "repeatable. cl=literal adds cl_literal without replacing "
              "structural cl."),
    )
    parser.add_argument(
        "--literal-phone-map", action="append", default=[],
        metavar="DISPLAY=SOURCE",
        help=("expose an authored special source under a distinct canonical "
              "token, for example cl_literal=cl; repeatable"),
    )
    parser.add_argument("--f0-min", "--f0min", dest="f0min", type=float)
    parser.add_argument("--f0-max", "--f0max", dest="f0max", type=float)
    parser.add_argument("--f0", type=float)
    parser.add_argument(
        "--f0-estimator",
        choices=("harvest", "dio"),
        default="harvest",
        help=("fallback for any source recording without usable UTAU FRQ "
              "data: Harvest favors voiced coverage; DIO is faster. FRQ "
              "data remains authoritative when present."),
    )
    parser.add_argument(
        "--source-window-mode", choices=SOURCE_WINDOW_MODES,
        default="adaptive",
        help=("adaptive bounds source audio for normal phones and uses the "
              "full selected recording only when edited phone durations can "
              "accommodate it; bounded always caps; full is legacy behavior"),
    )
    parser.add_argument(
        "--source-window-ms", type=float,
        default=DEFAULT_SOURCE_WINDOW_MS,
        help="maximum source milliseconds per diphone half at normal timing",
    )
    parser.add_argument(
        "--zero-overlap-guard-ms", type=float,
        default=DEFAULT_ZERO_OVERLAP_GUARD_MS,
        help=("experimental source-cut guard for zero OTO overlap; "
              "disabled by default because it changes recorded geometry"),
    )
    parser.add_argument("--skip-pm", action="store_true")
    parser.add_argument(
        "--runtime-audio-storage",
        choices=RUNTIME_AUDIO_STORAGE_MODES,
        default="grouped",
        help=("grouped (default) packs all indexed PSOLA source windows "
              "into one deterministic UniSyn cache; separate keeps "
              "per-WAV runtime access for development"),
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--test-text",
        default=None,
        help="smoke text; defaults are language-specific",
    )
    parser.add_argument("--festival-bin", default="festival")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    return parser


def _run_unified(args) -> int:
    special_phone_policy = _special_phone_policy(
        args.special_phone_mode,
        args.literal_phone_map,
    )
    samples, output, _ = validate_build_layout(
        args.samples, args.output, overwrite=args.overwrite
    )
    oto_files = _selected_oto_files(samples, args.oto)
    _validate_single_pitch_oto_scope(samples, args.oto, oto_files)
    selected_source_recordings = _source_recordings_from_oto_files(
        samples, oto_files
    )
    if args.language == "ja" and args.bank_type not in {"cv", "vcv", "cvvc"}:
        raise VoicePathError(
            "Japanese requires an explicit --bank-type cv, vcv, or cvvc."
        )
    if args.language != "ja" and args.bank_type != "arpasing":
        raise VoicePathError(
            "English and Asaxi currently require --bank-type arpasing."
        )
    enabled_languages = tuple(dict.fromkeys(
        [args.language, *args.enable_language]
    ))
    if args.language == "ja" and len(enabled_languages) > 1:
        raise VoicePathError(
            "Standalone Japanese CV/VCV/CVVC builds cannot add ARPAsing "
            "frontends. Use an ARPAsing primary build for shared languages."
        )
    if str(args.voice_color or "").casefold() == "all":
        raise VoicePathError(
            "Merged voice colors are experimental. Select one color or none."
        )
    output.mkdir(parents=True, exist_ok=True)
    pitch_analysis = analyze_speaker_pitch(
        samples, recording_files=selected_source_recordings
    )
    automatic_f0_min, automatic_f0_max = pitchmark_bounds(pitch_analysis)
    pitch_policy = automatic_pitch_metadata(
        pitch_analysis, default_is_automatic=args.f0 is None
    )
    effective_f0 = (
        float(args.f0) if args.f0 is not None
        else recommended_default_pitch_hz(pitch_analysis)
    )
    effective_f0_min = (
        float(args.f0min) if args.f0min is not None else automatic_f0_min
    )
    effective_f0_max = (
        float(args.f0max) if args.f0max is not None else automatic_f0_max
    )
    name = re.sub(r"[^A-Za-z0-9_]", "_", args.name)
    test_text = args.test_text or {
        "ja": "\u305f\u304d",
        "en": "this is a test",
        "asaxi": "taki",
    }[args.language]
    print("[1/5] Validated source, OTO scope, and protected output path")
    print(
        "      Shared speaker pitch: "
        f"{pitch_analysis.median_f0_hz:.1f} Hz "
        f"({pitch_analysis.source}, "
        f"{pitch_analysis.voiced_sample_count} voiced samples)"
    )

    if args.language == "ja":
        if args.character_yaml or args.prefix_map or args.alias_prefix \
                or args.alias_suffix or args.voice_color:
            raise VoicePathError(
                "Japanese affix/color overrides belong in the selected "
                "profile. The stable builder does not merge colors or pitches."
            )
        from dataclasses import replace
        from japanese_candidates import compile_candidate_graph
        from japanese_festival import compile_festival_voice
        from japanese_profiles import infer_bank_profile, load_profile

        source_scope = oto_files[0] if len(oto_files) == 1 else samples
        if args.profile:
            profile = replace(
                load_profile(Path(args.profile)),
                bank_configuration=args.bank_type,
            )
        else:
            profile = infer_bank_profile(
                source_scope,
                bank_configuration=args.bank_type,
                oto_files=oto_files,
            )
        print("[2/5] Compiling Japanese source candidates")
        graph = compile_candidate_graph(
            source_scope,
            profile=profile,
            oto_files=oto_files,
        )
        print("[3/5] Compiling Japanese Festival/UniSyn units")
        build = compile_festival_voice(
            graph,
            output,
            voice_name=name,
            average_pitch_hz=effective_f0,
            pitchmark=not args.skip_pm,
            f0_min=effective_f0_min,
            f0_max=effective_f0_max,
            f0_estimator=args.f0_estimator,
            source_window_mode=args.source_window_mode,
            source_window_ms=args.source_window_ms,
            zero_overlap_guard_ms=args.zero_overlap_guard_ms,
            speaker_pitch_analysis=pitch_analysis.to_dict(),
            wsl_distro=(args.wsl_distro or None),
            runtime_audio_storage=(
                args.runtime_audio_storage
                if not args.skip_pm else "separate"),
        )
        if args.runtime_audio_storage == "grouped" and not args.skip_pm:
            print("      Packing indexed source audio into UniSyn cache")
            storage = build_grouped_runtime(
                output,
                voice_name=name,
                scheme_path=output / "festvox" / f"{name}_ja.scm",
                voice_entry_point=build.voice_entry_point,
                festival_bin=args.festival_bin,
                run_external=_run_external,
                wsl_distro=(args.wsl_distro or None),
            )
            apply_runtime_audio_metadata(output, storage)
        metadata = _mark_japanese_front_door(output)
        metadata["special_phone_realizations"] = special_phone_policy
        _validate_literal_special_phone_sources(
            special_phone_policy, metadata.get("index") or {}
        )
        metadata["phones"] = declared_display_phones(
            metadata.get("phones") or phone_inventory(
                metadata.get("index") or {}
            ),
            special_phone_policy,
        )
        metadata.update(pitch_policy)
        metadata["output_calibration"] = dict(OUTPUT_CALIBRATION_POLICY)
        (output / "dic" / "diphone_index.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        alternatives_path = output / "dic" / "unit_alternatives.json"
        if alternatives_path.is_file():
            alternatives_payload = json.loads(
                alternatives_path.read_text(encoding="utf-8")
            )
            alternatives_payload[
                "special_phone_realizations"
            ] = special_phone_policy
            alternatives_path.write_text(
                json.dumps(
                    alternatives_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        _write_shared_manifest(output, metadata)
        entry_point = build.voice_entry_point
    else:
        from arpasing_profile import (
            DEFAULT_PHONEME_MAP_PATH,
            load_arpasing_profile,
        )
        phoneme_profile_path = Path(
            args.phoneme_map or DEFAULT_PHONEME_MAP_PATH
        )
        phoneme_profile = load_arpasing_profile(phoneme_profile_path)
        print("[2/5] Converting the selected ARPAsing OTO scope")
        legacy_args = [
            "--utau", str(samples), "--out", str(output),
            "--name", name, "--unified-internal",
            "--language-mode", (
                args.language if len(enabled_languages) == 1 else "legacy"
            ),
            "--phoneme-map", str(phoneme_profile_path),
            "--wsl-distro", args.wsl_distro,
            "--source-window-mode", args.source_window_mode,
            "--source-window-ms", str(args.source_window_ms),
            "--zero-overlap-guard-ms", str(args.zero_overlap_guard_ms),
            "--runtime-audio-storage", args.runtime_audio_storage,
        ]
        if "ja" in enabled_languages:
            legacy_args.append("--enable-japanese")
        for scope in args.oto:
            legacy_args.extend(["--oto", str(scope)])
        for flag, value in (
            ("--character-yaml", args.character_yaml),
            ("--prefix-map", args.prefix_map),
            ("--voice-color", args.voice_color),
            ("--f0min", effective_f0_min),
            ("--f0max", effective_f0_max),
            ("--f0", effective_f0),
            ("--f0-estimator", args.f0_estimator),
        ):
            if value is not None and str(value) != "":
                legacy_args.extend([flag, str(value)])
        for value in args.alias_prefix:
            legacy_args.extend(["--alias-prefix", value])
        for value in args.alias_suffix:
            legacy_args.extend(["--alias-suffix", value])
        for value in args.special_phone_mode:
            legacy_args.extend(["--special-phone-mode", value])
        for value in args.literal_phone_map:
            legacy_args.extend(["--literal-phone-map", value])
        if args.skip_pm:
            legacy_args.append("--skip-pm")
        _legacy_main(legacy_args)
        print("[3/5] Installing the language-scoped generated manifest")
        metadata = _install_non_japanese_manifest(
            samples=samples,
            oto_files=oto_files,
            output=output,
            name=name,
            language=args.language,
            supported_languages=enabled_languages,
            phoneme_profile=phoneme_profile,
            bank_type=args.bank_type,
            speaker_pitch_analysis=pitch_analysis,
            selected_voice_color=args.voice_color,
            pitch_policy=pitch_policy,
        )
        entry_point = str(metadata["voice_entry_point"])

    print("[4/5] Validating shared metadata and generated layout")
    _validate_generated_layout(output, metadata)
    if args.test and args.language != "ja" and not args.skip_pm:
        if not run_test(
            output,
            name,
            args.festival_bin,
            test_text,
            wsl_distro=args.wsl_distro,
            voice_entry_point=entry_point,
        ):
            raise RuntimeError("Festival smoke render failed")
    elif args.test and args.skip_pm:
        print("      Audio smoke render skipped because --skip-pm was selected")
    elif args.test:
        if not run_japanese_test(
            output,
            name,
            entry_point,
            args.festival_bin,
            test_text,
            float(metadata.get("average_pitch_hz") or args.f0 or 180.0),
            wsl_distro=args.wsl_distro,
        ):
            raise RuntimeError("Japanese Festival smoke render failed")
    print("[5/5] Build complete")
    print(f"Generated voice: {output}")
    print(f"Runtime path: {windows_to_wsl_path(output)}")
    print(f"Entry point: {entry_point}")
    print("Source UTAU bank remained read only.")
    return 0


def _unified_main(argv=None) -> int:
    parser = _unified_parser()
    args = parser.parse_args(argv)
    try:
        return _run_unified(args)
    except (VoicePathError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error[voice-build]: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error[voice-build]: {exc.code}", file=sys.stderr)
            return 2
        return int(exc.code or 0)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(item in arguments for item in (
        "--language", "--samples", "--output"
    )):
        return _unified_main(arguments)
    _legacy_main(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
