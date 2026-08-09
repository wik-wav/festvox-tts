# -*- coding: utf-8 -*-
"""
utau2festvox.py — convert a configured UTAU voicebank (oto.ini) into a
FestVox-compatible diphone database.

Usage:
    python utau2festvox.py --bank <voicebank dir> [--out <output dir>]
                           [--name asaxi] [--no-copy]
                           [--character-yaml PATH] [--prefix-map PATH]
                           [--alias-prefix TEXT] [--alias-suffix TEXT]
                           [--voice-color NAME]

What it does
  1. Parses root and nested oto.ini files:
        File=Alias,Offset,Consonant,Blank,Preutterance,Overlap
     OpenUtau character.yaml subbanks are preferred over legacy prefix.map
     declarations. Exact declared alias prefixes/suffixes are removed before
     phone mapping, including combined color+pitch forms such as PE3 or E3P.
  2. Converts UTAU's relative-ms geometry into the absolute second-based
     (start, mid, end) triples a FestVox diphone index needs:
        start = Offset + bounded positive Overlap, or a bounded inferred
                guard when Overlap is zero
        mid   = Offset + Preutterance     (the UTAU alignment point is the
                                           phone boundary of the diphone)
        end   = the next matching alias's overlap anchor when available,
                otherwise Offset + Consonant when that is a valid center,
                otherwise the OTO region end
     OpenUtau uses positive Overlap as the fade-in/coexistence duration.
     Its end is therefore a substantially safer left-phone center than the
     raw Offset, which is the beginning of material intended to coexist with
     the preceding unit.  Festival then performs its own pitch-synchronous
     overlap-add at these stable phone centers.
     NOTE on Blank: standard UTAU semantics are NEGATIVE = length measured
     from Offset, POSITIVE = milliseconds trimmed from the file end. (The
     task brief stated the inverse; on this bank's own data the inverse
     produces end-times that overrun the next diphone's offset by seconds,
     so the standard semantics are used. Verified: every produced
     start < mid < end and end - start < 1.2 s.)
     When an OTO region extends beyond the next transition's alignment point,
     its end is clamped to that point. This prevents closure or silence from
     the recorded right context leaking into a different synthesis context.
  3. Maps UTAU aliases to Festival phone names through PHONEME_MAP below,
     sanitizing everything into valid Scheme atoms.
  4. Recovers each take's outer context from adjacent, time-ordered OTO
     aliases in the same recording. Strict CV tokens receive directional edge
     classes; missing transitions stay unknown. WAV filenames are never used
     as phonetic evidence.
  5. Builds the FestVox layout:
        OUT/wav/        copied + renamed source wavs
        OUT/dic/        <name>_diphone.scm  (Scheme index list)
                        <name>_diphone.est  (EST-format index, for UniSyn)
                        diphone_index.json  (fast loader for other tools)
        OUT/festival/   <name>_diphone_stub.scm (minimal voice scaffold)
  6. Writes OUT/conversion_report.txt (unmapped aliases, preserved takes,
     missing/odd wavs).

Requires only the Python standard library (wave module measures the files,
which is what makes negative Blank values computable at all).
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import wave
from pathlib import Path

from source_window import (
    DEFAULT_SOURCE_WINDOW_MS,
    DEFAULT_ZERO_OVERLAP_GUARD_MS,
    SOURCE_WINDOW_MODES,
    build_source_window_plan,
    effective_oto_overlap_ms,
    normalize_source_window_mode,
    normalize_zero_overlap_guard_ms,
    source_window_variant_names,
)
from special_phones import generated_voice_policy

# --------------------------------------------------------------------------
# PHONEME MAPPING — edit this block to control the UTAU→Festival mapping.
#
# Keys are UTAU alias *tokens* (after the pitch suffix, e.g. "F#3", has been
# stripped and the alias split on whitespace). Values are the Festival phone
# names to use in diphone identifiers ("k","a" -> diphone "k-a").
#
#   * "-" is the UTAU silence marker -> Festival "pau".
#   * A standalone token ending in "-" (e.g. "b-") maps to a distinct final
#     variant. A two-token ``V b-`` alias is a V-C-silence triphone and is
#     rejected by convert(); ordinary ``V b`` plus ``b -`` is used instead.
#   * Tokens not listed here fall back to themselves if they are plain
#     ASCII alphanumerics. Numbered recording variants ("aa2", "ah11") share
#     a base phone spelling but remain separate selectable units with context,
#     EXCEPT for symbols explicitly listed here (so å1-style *real* symbols
#     survive as phone symbols if a bank uses them).
#   * Map a token to None to exclude it from the database (breaths etc.).
# --------------------------------------------------------------------------
PHONEME_MAP = {
    "-": "pau",
    # breaths / non-speech: excluded from the diphone index
    "inh": None, "exh": None, "sil": "pau", "BR": None, "br": None,
    # identity mappings, written out for documentation value — the full
    # arpasing set used by the 4_Fis3 bank (vowels, consonants, extras):
    **{p: p for p in (
        "a aa ae ah ao aw ax ay e eh er ey i ih iy o ow oy u uh uw "
        "b by ch d dy dh dx dxy dz f fy g gy h hy hh jh k ky l ly m my "
        "n ny ng ngy nn mm nng xn p py q r ry rr s sh t ty ts th v vy "
        "w y z zh cl si zi shi ri wi".split())},
    # Japanese-style CV units exist in the bank ("ka","byo"...) but are not
    # used for diphone synthesis; they are indexed anyway (harmless) since
    # unlisted ASCII tokens fall through as themselves.
}

PITCH_SUFFIX = re.compile(r"([A-G]#?-?\d+)$")     # trailing UTAU pitch tag
VARIANT_DIGITS = re.compile(r"\d+$")             # recording-take numbering
NOTE_NAME = re.compile(r"([A-Ga-g])([#b]?)(-?\d+)")
L_VOWELS = {
    "a", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "e", "eh",
    "er", "ey", "i", "ih", "iy", "o", "ow", "oy", "u", "uh", "uw",
}
L_LIGHT_FOLLOWERS = L_VOWELS | {"y"}

# Language-neutral context classes.  OTO context can contain symbols from a
# different front end (for example Japanese ``u`` around an English ``dh``),
# so selectors retain the exact spelling and also compare broad articulation.
PHONE_CLASSES = {
    **{p: "vowel" for p in L_VOWELS},
    **{p: "stop_voiceless" for p in
       "p t k q py ty ky cl".split()},
    **{p: "stop_voiced" for p in
       "b d g by dy gy dx dxy".split()},
    **{p: "affricate_voiceless" for p in "ch ts".split()},
    **{p: "affricate_voiced" for p in "jh dz".split()},
    **{p: "fricative_voiceless" for p in
       "f s sh th h hh fy hy".split()},
    **{p: "fricative_voiced" for p in
       "v z zh dh vy zi".split()},
    **{p: "nasal" for p in
       "m n ng nn mm nng xn my ny ngy".split()},
    **{p: "liquid" for p in "l r rr ly ry ri".split()},
    **{p: "glide" for p in "w y wi".split()},
    "pau": "silence", "sil": "silence", "sp": "silence",
}


def read_text_fallback(path: Path, preferred=None):
    """Read UTAU metadata without adding a mandatory YAML dependency."""
    encodings = []
    for encoding in (preferred, "utf-8-sig", "utf-8", "cp932",
                     "shift_jis", "latin-1"):
        if encoding and encoding not in encodings:
            encodings.append(encoding)
    last_error = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except (LookupError, UnicodeDecodeError) as exc:
            last_error = exc
    raise UnicodeError(f"could not decode {path}: {last_error}")


def _strip_yaml_comment(value: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if (char == "#" and quote is None and
                (index == 0 or value[index - 1].isspace())):
            return value[:index].rstrip()
    return value.rstrip()


def _yaml_scalar(value: str) -> str:
    value = _strip_yaml_comment(value).strip()
    if not value or value.lower() in {"null", "~"}:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def _yaml_list(value: str):
    value = _strip_yaml_comment(value).strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    return [_yaml_scalar(item) for item in value[1:-1].split(",")
            if _yaml_scalar(item)]


def parse_character_yaml(path: Path) -> dict:
    """Read the OpenUtau subbank fields needed for alias normalization.

    character.yaml is ordinary YAML, but its subbank records use a small,
    stable scalar/list shape. This parser intentionally supports that shape
    with the standard library so the WSL builder remains dependency-free.
    Unknown character fields are ignored.
    """
    text, file_encoding = read_text_fallback(path)
    result = {"subbanks": [], "text_file_encoding": None,
              "file_encoding": file_encoding}
    current = None
    in_subbanks = False
    in_tone_ranges = False
    for raw_line in text.splitlines():
        without_comment = _strip_yaml_comment(raw_line)
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip())
        line = without_comment.strip()
        if not in_subbanks:
            if line.startswith("text_file_encoding:"):
                result["text_file_encoding"] = _yaml_scalar(
                    line.split(":", 1)[1]) or None
            if line == "subbanks:":
                in_subbanks = True
            continue

        # OpenUtau commonly uses indentationless sequence items immediately
        # under ``subbanks:``. A different top-level key ends this section.
        if indent == 0 and not line.startswith("-") and ":" in line:
            break
        if line.startswith("- "):
            body = line[2:].strip()
            if ":" in body:
                if current is not None:
                    result["subbanks"].append(current)
                current = {"color": "", "prefix": "", "suffix": "",
                           "tone_ranges": []}
                key, value = body.split(":", 1)
                key = key.strip()
                if key in {"color", "prefix", "suffix"}:
                    current[key] = _yaml_scalar(value)
                elif key == "tone_ranges":
                    current["tone_ranges"].extend(_yaml_list(value))
                in_tone_ranges = key == "tone_ranges"
            elif current is not None and in_tone_ranges:
                tone_range = _yaml_scalar(body)
                if tone_range:
                    current["tone_ranges"].append(tone_range)
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"color", "prefix", "suffix"}:
            current[key] = _yaml_scalar(value)
            in_tone_ranges = False
        elif key == "tone_ranges":
            current["tone_ranges"].extend(_yaml_list(value))
            in_tone_ranges = True
        else:
            in_tone_ranges = False
    if current is not None:
        result["subbanks"].append(current)
    return result


def parse_prefix_map(path: Path) -> list:
    """Group legacy ``tone<TAB>prefix<TAB>suffix`` rows by affix pair."""
    text, _encoding = read_text_fallback(path)
    grouped = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        tone, prefix = parts[0].strip(), parts[1]
        suffix = "\t".join(parts[2:]).strip()
        key = (prefix, suffix)
        grouped.setdefault(key, []).append(tone)
    return [{"color": "", "prefix": prefix, "suffix": suffix,
             "tone_ranges": tones}
            for (prefix, suffix), tones in grouped.items()]


def _metadata_spec(row, source, source_path, order):
    return {
        "color": str(row.get("color") or ""),
        "prefix": str(row.get("prefix") or ""),
        "suffix": str(row.get("suffix") or ""),
        "tone_ranges": [str(item) for item in
                        (row.get("tone_ranges") or [])],
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "order": int(order),
    }


def load_alias_metadata(bank: Path, character_yaml=None, prefix_map=None,
                        alias_prefixes=None, alias_suffixes=None) -> dict:
    """Discover OpenUtau metadata in priority order, then manual overrides."""
    warnings = []
    specs = []
    character_explicit = character_yaml is not None
    prefix_explicit = prefix_map is not None
    character_path = (Path(character_yaml).expanduser()
                      if character_explicit else bank / "character.yaml")
    prefix_path = (Path(prefix_map).expanduser()
                   if prefix_explicit else bank / "prefix.map")
    text_file_encoding = None

    if character_path.is_file():
        parsed = parse_character_yaml(character_path)
        text_file_encoding = parsed.get("text_file_encoding")
        if not parsed["subbanks"]:
            message = (f"{character_path} has no readable OpenUtau subbanks; "
                       "expected color/prefix/suffix entries")
            if character_explicit:
                raise SystemExit(message)
            warnings.append(message)
        for order, row in enumerate(parsed["subbanks"]):
            specs.append(_metadata_spec(
                row, "character.yaml", character_path, order))
    elif character_explicit:
        raise SystemExit(f"character.yaml not found: {character_path}")

    if prefix_path.is_file():
        start = len(specs)
        for offset, row in enumerate(parse_prefix_map(prefix_path)):
            spec = _metadata_spec(row, "prefix.map", prefix_path,
                                  start + offset)
            duplicate = any(
                existing["prefix"] == spec["prefix"] and
                existing["suffix"] == spec["suffix"] and
                existing["color"] == spec["color"]
                for existing in specs)
            if not duplicate:
                specs.append(spec)
    elif prefix_explicit:
        raise SystemExit(f"prefix.map not found: {prefix_path}")

    for value in alias_prefixes or ():
        if value:
            specs.append(_metadata_spec(
                {"prefix": value}, "manual --alias-prefix", None,
                len(specs)))
    for value in alias_suffixes or ():
        if value:
            specs.append(_metadata_spec(
                {"suffix": value}, "manual --alias-suffix", None,
                len(specs)))
    return {
        "specs": specs,
        "character_yaml": str(character_path) if character_path.is_file()
        else None,
        "prefix_map": str(prefix_path) if prefix_path.is_file() else None,
        "text_file_encoding": text_file_encoding,
        "manual_prefixes": [str(item) for item in alias_prefixes or ()
                            if item],
        "manual_suffixes": [str(item) for item in alias_suffixes or ()
                            if item],
        "warnings": warnings,
    }


def _strip_alias_prefix(alias: str, prefix: str):
    if not prefix:
        return alias, False
    if alias.startswith(prefix) and len(alias) > len(prefix):
        return alias[len(prefix):].lstrip(), True
    tokens = alias.split()
    for index, token in enumerate(tokens):
        if token == "-":
            continue
        if token.startswith(prefix) and len(token) > len(prefix):
            tokens[index] = token[len(prefix):]
            return " ".join(tokens), True
        break
    return alias, False


def _strip_alias_suffix(alias: str, suffix: str):
    if not suffix:
        return alias, False
    if alias.endswith(suffix) and len(alias) > len(suffix):
        return alias[:-len(suffix)].rstrip(), True
    tokens = alias.split()
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token == "-":
            continue
        if token.endswith(suffix) and len(token) > len(suffix):
            tokens[index] = token[:-len(suffix)]
            return " ".join(tokens), True
        break
    return alias, False


def _strip_affix_pair(alias: str, spec: dict):
    candidate = alias
    changed = False
    if spec["prefix"]:
        candidate, removed = _strip_alias_prefix(candidate, spec["prefix"])
        if not removed:
            return alias, False
        changed = True
    if spec["suffix"]:
        candidate, removed = _strip_alias_suffix(candidate, spec["suffix"])
        if not removed:
            return alias, False
        changed = True
    if not changed or not candidate.strip():
        return alias, False
    return candidate.strip(), True


def _strip_pitch_suffix(alias: str):
    match = PITCH_SUFFIX.search(alias)
    if match and match.end() == len(alias) and match.start() > 0:
        return alias[:match.start()].rstrip(), match.group(1)
    tokens = alias.split()
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] == "-":
            continue
        match = PITCH_SUFFIX.search(tokens[index])
        if match and match.end() == len(tokens[index]) and match.start() > 0:
            pitch = match.group(1)
            tokens[index] = tokens[index][:match.start()]
            return " ".join(tokens), pitch
        break
    return alias, None


def normalize_alias(alias: str, specs: list) -> dict:
    """Remove declared affixes and pitch tags in either suffix order.

    Iteration is deliberate: with a manual color suffix ``P``, both ``ayPE3``
    and ``ayE3P`` reduce to ``ay`` without guessing that every trailing P is
    metadata.
    """
    normalized = alias.strip()
    matches = []
    pitches = []
    ordered_specs = sorted(
        (spec for spec in specs if spec["prefix"] or spec["suffix"]),
        key=lambda spec: (
            0 if spec["source"] == "character.yaml" else
            1 if spec["source"] == "prefix.map" else 2,
            -(len(spec["prefix"]) + len(spec["suffix"])),
            spec["order"]))
    for _iteration in range(12):
        changed = False
        for spec in ordered_specs:
            candidate, removed = _strip_affix_pair(normalized, spec)
            if removed:
                normalized = candidate
                if spec not in matches:
                    matches.append(spec)
                changed = True
                break
        if changed:
            continue
        candidate, pitch = _strip_pitch_suffix(normalized)
        if pitch:
            normalized = candidate
            pitches.append(pitch)
            continue
        break
    return {"alias": normalized.strip(),
            "match": matches[0] if matches else None,
            "matches": matches, "pitch_tags": pitches}


def _note_number(text: str):
    matches = list(NOTE_NAME.finditer(text or ""))
    if not matches:
        return None
    note, accidental, octave = matches[-1].groups()
    pitch_class = {"C": 0, "D": 2, "E": 4, "F": 5,
                   "G": 7, "A": 9, "B": 11}[note.upper()]
    if accidental == "#":
        pitch_class += 1
    elif accidental == "b":
        pitch_class -= 1
    return (int(octave) + 1) * 12 + pitch_class


def resolve_voice_color(specs: list, requested=None):
    character_colors = []
    for spec in specs:
        if spec["source"] == "character.yaml" and \
                spec["color"] not in character_colors:
            character_colors.append(spec["color"])
    if requested is not None and str(requested).casefold() in {"*", "all"}:
        return "*", character_colors
    if requested is None:
        if "" in character_colors:
            return "", character_colors
        return (character_colors[0] if character_colors else None,
                character_colors)
    if str(requested).casefold() in {"default", "normal", "none"}:
        if not character_colors:
            return None, character_colors
        requested = ""
    for color in character_colors:
        if color.casefold() == str(requested).casefold():
            return color, character_colors
    choices = [color or "default" for color in character_colors]
    raise SystemExit(
        f"voice color {requested!r} is not declared by character.yaml; "
        f"available: {choices or ['default']} (or use --voice-color all)")


def choose_default_subbank(specs: list, selected_color):
    candidates = [spec for spec in specs
                  if spec["source"] == "character.yaml" and
                  (selected_color == "*" and spec["color"] == "" or
                   selected_color != "*" and
                   (selected_color is None or
                    spec["color"] == selected_color))]
    if not candidates:
        candidates = [spec for spec in specs
                      if spec["source"] in {"character.yaml", "prefix.map"}]
    pitched = [(number, spec) for spec in candidates
               if (number := _note_number(spec["prefix"] + spec["suffix"]))
               is not None]
    if pitched:
        pitched.sort(key=lambda item: (item[0], item[1]["order"]))
        return pitched[(len(pitched) - 1) // 2][1]
    return candidates[0] if candidates else None


def find_oto_files(bank: Path) -> list:
    return sorted((path for path in bank.rglob("oto.ini") if path.is_file()),
                  key=lambda path: path.relative_to(bank).as_posix().casefold())


def _spec_public(spec):
    if not spec:
        return None
    return {key: spec[key] for key in
            ("color", "prefix", "suffix", "tone_ranges", "source")}


def _public_source_path(value, bank: Path):
    """Keep generated provenance useful without exposing private roots."""
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(bank.resolve()).as_posix()
    except ValueError:
        return path.name


def phone_context_class(phone: str) -> str:
    """Return a stable context class for any bank phone symbol."""
    base = re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    if base == "*":
        return "wildcard"
    if base in PHONE_CLASSES:
        return PHONE_CLASSES[base]
    if base.endswith("y") and base[:-1] in PHONE_CLASSES:
        return PHONE_CLASSES[base[:-1]]
    return "other"


def context_edge_info(phone: str, edge: str) -> dict:
    """Classify the acoustic edge nearest a diphone from OTO alias spelling.

    OTO context tokens can be atomic phones (``zh``), strict CV compounds
    (``zha``), or ``*`` when no adjacent OTO transition exists.  WAV names are
    intentionally never consulted.  ``edge`` names the leftmost or rightmost
    edge of the context token itself.
    """
    if edge not in {"left", "right"}:
        raise ValueError("edge must be 'left' or 'right'")
    base = re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    direct_class = phone_context_class(base)
    if base == "*":
        return {"phone": "*", "class": "wildcard",
                "kind": "wildcard_unknown"}
    if direct_class != "other":
        return {"phone": base, "class": direct_class, "kind": "atomic"}

    # Parse only an unambiguous consonant-onset + vowel-nucleus token.  This
    # covers the bank's ka/zha/ngya/fya/lya/vya-style aliases without turning
    # arbitrary unknown strings into guessed phones.
    for vowel in sorted(L_VOWELS, key=lambda item: (-len(item), item)):
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


def sanitize_scheme(name: str) -> str:
    """Make a string safe as a Scheme atom / filename component:
    ASCII-fold, keep [A-Za-z0-9_], collapse the rest to '_'."""
    folded = unicodedata.normalize("NFKD", name)
    folded = folded.encode("ascii", "ignore").decode("ascii") or "x"
    return re.sub(r"[^A-Za-z0-9_]+", "_", folded).strip("_") or "x"


def map_token(tok: str, report):
    """UTAU alias token -> Festival phone name (or None to skip)."""
    if tok in PHONEME_MAP:
        return PHONEME_MAP[tok]
    if tok.endswith("-") and tok[:-1]:
        base = map_token(tok[:-1], report)          # final allophone: "b-"
        return None if base is None else base + "_"
    stripped = VARIANT_DIGITS.sub("", tok)          # "aa2" -> "aa"
    if stripped != tok and stripped in PHONEME_MAP:
        return PHONEME_MAP[stripped]
    if stripped != tok and stripped.endswith("-"):  # "d-1" -> final "d-"
        return map_token(stripped, report)
    if re.fullmatch(r"[A-Za-z_]+", stripped or tok):
        return stripped or tok                      # plain ASCII: identity
    report["unmapped"].add(tok)
    return None


def parse_oto(path: Path, report, label=None, preferred_encoding=None):
    """Yield entries from one root or nested oto.ini file."""
    label = label or path.name
    text, encoding = read_text_fallback(path, preferred_encoding)
    report["oto_encodings"][label] = encoding
    lines = text.splitlines()
    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        fname, rest = line.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 6:
            report["bad_lines"].append(f"{label}:{ln}")
            continue
        alias = parts[0].strip()
        try:
            offset, consonant, blank, preutt, overlap = map(float, parts[1:6])
        except ValueError:
            report["bad_lines"].append(f"{label}:{ln}")
            continue
        yield {"wav": fname.strip(), "alias": alias, "offset": offset,
               "consonant": consonant, "blank": blank,
               "preutterance": preutt, "overlap": overlap, "line": ln,
               "oto_file": label, "oto_path": path}


def wav_length_ms(path: Path, cache={}):
    """Total duration of a wav in milliseconds (measured, cached)."""
    if path not in cache:
        with wave.open(str(path), "rb") as w:
            cache[path] = w.getnframes() / w.getframerate() * 1000.0
    return cache[path]


def _source_wav(bank: Path, oto_path: Path, wav_name: str):
    relative = Path(wav_name.replace("\\", "/"))
    source = (oto_path.parent / relative).resolve()
    bank_root = bank.resolve()
    try:
        source_relative = source.relative_to(bank_root).as_posix()
    except ValueError:
        return None, None
    return source, source_relative


def _possible_affix_tokens(tokens):
    possible = []
    for token in tokens:
        plain = token.rstrip("-")
        if re.search(r"[a-z][A-Z]", plain) or \
                ("#" in plain and re.search(r"[a-z]", plain)):
            possible.append(token)
    return possible


def _target_wav_name(source_relative: str, owners: dict) -> str:
    source_stem = Path(source_relative).with_suffix("").as_posix()
    candidate = sanitize_scheme(source_stem) + ".wav"
    if candidate in owners and owners[candidate] != source_relative:
        digest = hashlib.sha1(source_relative.encode("utf-8")).hexdigest()[:10]
        candidate = sanitize_scheme(source_stem) + "_" + digest + ".wav"
    owners[candidate] = source_relative
    return candidate


def _same_subbank(left, right):
    if not left or not right:
        return False
    return all(left.get(key) == right.get(key)
               for key in ("color", "prefix", "suffix", "source"))


def _dominant_character_spec(normalized_rows):
    """Infer one OTO file's declared subbank from strong exact agreement."""
    counts = {}
    objects = {}
    total = 0
    for _entry, normalized in normalized_rows:
        match = next((item for item in normalized["matches"]
                      if item["source"] == "character.yaml"), None)
        if not match:
            continue
        key = (match["color"], match["prefix"], match["suffix"])
        counts[key] = counts.get(key, 0) + 1
        objects[key] = match
        total += 1
    if not counts:
        return None
    key, count = max(counts.items(), key=lambda item: item[1])
    if (count < 2 or count / total < 0.95 or
            count / max(1, len(normalized_rows)) < 0.50):
        return None
    return objects[key]


def _unit_order(unit):
    return (unit.get("subbank_rank", 0),
            unit.get("oto_file", "").casefold(), unit["line"])


def _no_units_message(bank, report, metadata, selected_color):
    color = ("all" if selected_color == "*" else
             "default" if selected_color == "" else
             selected_color or "not declared")
    possible = sorted(report["possible_affixes"])
    lines = [
        f"No usable diphone units were produced from {report['oto_entries']} "
        f"entries in {len(report['oto_files'])} oto.ini file(s) under {bank}.",
        f"Selected voice color: {color}.",
    ]
    if report["color_skips"]:
        lines.append(f"{report['color_skips']} entries belonged to another "
                     "declared voice color.")
    if possible:
        lines.append("Possible undeclared alias affixes: " +
                     ", ".join(possible[:20]))
    lines.extend([
        "Metadata lookup order is character.yaml, then prefix.map, then the "
        "ordinary trailing pitch tag.",
        "Point at metadata with --character-yaml PATH or --prefix-map PATH. "
        "For a manual color marker, repeat --alias-prefix TEXT and/or "
        "--alias-suffix TEXT.",
        "Example: --alias-suffix P accepts both ayPE3 and ayE3P; the color "
        "marker and pitch tag are removed iteratively.",
    ])
    if metadata["warnings"]:
        lines.extend("Metadata warning: " + item
                     for item in metadata["warnings"])
    return "\n".join(lines)


def convert(bank: Path, out: Path, name: str, copy_wavs: bool,
            character_yaml=None, prefix_map=None, alias_prefixes=None,
            alias_suffixes=None, voice_color=None, oto_files=None,
            phoneme_profile=None, source_window_mode="adaptive",
            source_window_ms=DEFAULT_SOURCE_WINDOW_MS,
            zero_overlap_guard_ms=DEFAULT_ZERO_OVERLAP_GUARD_MS):
    bank, out = Path(bank).expanduser(), Path(out).expanduser()
    source_window_mode = normalize_source_window_mode(source_window_mode)
    source_window_ms = float(source_window_ms)
    if not 20.0 <= source_window_ms <= 2000.0:
        raise ValueError("source window must be between 20 and 2000 ms")
    zero_overlap_guard_ms = normalize_zero_overlap_guard_ms(
        zero_overlap_guard_ms
    )
    report = {"unmapped": set(), "bad_lines": [], "dupes": 0,
              "missing_wav": set(), "skipped_nonspeech": 0, "singles": 0,
              "context_tail_clamps": 0, "skipped_coda_triphones": 0,
              "oto_overlap_start_centers": 0,
              "oto_inferred_overlap_start_centers": 0,
              "oto_fixed_end_centers": 0,
              "oto_chained_end_centers": 0,
              "bad_wav": set(), "outside_wav": set(), "oto_entries": 0,
              "oto_encodings": {}, "oto_files": [], "empty_oto_files": [],
              "aliases_normalized": 0, "affix_normalized": 0,
              "pitch_normalized": 0, "color_skips": 0,
              "possible_affixes": set(), "oto_affix_inferences": 0,
              "unresolved_affix_skips": 0,
              "profile_mapped_aliases": 0,
              "profile_sequence_skips": [],
              "source_window_bounded_units": 0,
              "source_window_variants": 0,
              "structural_consonant_holds": 0}
    if not bank.is_dir():
        raise SystemExit(f"UTAU voicebank folder not found: {bank}")
    if oto_files is None:
        oto_files = find_oto_files(bank)
    else:
        selected = []
        for raw_path in oto_files:
            path = Path(raw_path).expanduser().resolve()
            try:
                path.relative_to(bank.resolve())
            except ValueError as exc:
                raise SystemExit(
                    f"selected oto.ini is outside the source bank: {path}"
                ) from exc
            if not path.is_file():
                raise SystemExit(f"selected oto.ini not found: {path}")
            selected.append(path)
        oto_files = sorted(
            dict.fromkeys(selected),
            key=lambda path: path.relative_to(bank.resolve())
            .as_posix().casefold(),
        )
    if not oto_files:
        raise SystemExit(f"no oto.ini found in {bank} or its subfolders")
    report["oto_files"] = [path.relative_to(bank).as_posix()
                           for path in oto_files]
    metadata = load_alias_metadata(
        bank, character_yaml=character_yaml, prefix_map=prefix_map,
        alias_prefixes=alias_prefixes, alias_suffixes=alias_suffixes)
    specs = metadata["specs"]
    selected_color, available_colors = resolve_voice_color(specs, voice_color)
    default_subbank = choose_default_subbank(specs, selected_color)

    for d in ("wav", "dic", "festival"):
        (out / d).mkdir(parents=True, exist_ok=True)

    index = {}          # selectable unit name -> (wavfile, start_s, mid_s, end_s)
    units = []          # every valid OTO take before duplicate grouping
    wav_map = {}        # bank-relative source wav -> generated target name
    source_paths = {}   # bank-relative source wav -> absolute read-only path
    target_owners = {}
    for oto_path in oto_files:
        oto_label = oto_path.relative_to(bank).as_posix()
        entries = list(parse_oto(
            oto_path, report, oto_label, metadata["text_file_encoding"]))
        if not entries:
            report["empty_oto_files"].append(oto_label)
        normalized_rows = [(entry, normalize_alias(entry["alias"], specs))
                           for entry in entries]
        dominant_affix = _dominant_character_spec(normalized_rows)
        for e, normalized in normalized_rows:
            report["oto_entries"] += 1
            alias = normalized["alias"]
            match = normalized["match"]
            character_match = next(
                (item for item in normalized["matches"]
                 if item["source"] == "character.yaml"), None)
            source_affix = character_match or dominant_affix or match
            affix_inferred = bool(not character_match and dominant_affix)
            if affix_inferred:
                report["oto_affix_inferences"] += 1
            if alias != e["alias"].strip():
                report["aliases_normalized"] += 1
            if normalized["matches"]:
                report["affix_normalized"] += 1
            if normalized["pitch_tags"]:
                report["pitch_normalized"] += 1
            if (selected_color not in {None, "*"} and source_affix and
                    source_affix["source"] == "character.yaml" and
                    source_affix["color"] != selected_color):
                report["color_skips"] += 1
                continue

            toks = alias.split()
            possible_affixes = _possible_affix_tokens(toks)
            if possible_affixes and not normalized["matches"]:
                report["possible_affixes"].update(possible_affixes)
                if metadata["character_yaml"] or metadata["prefix_map"]:
                    report["unresolved_affix_skips"] += 1
                    continue
            # ``V C-`` is UTAU shorthand for a V-C-silence triphone. It
            # already contains a silent tail; explicit ``C -`` stays valid.
            if (len(toks) == 2 and toks[-1] != "-" and
                    VARIANT_DIGITS.sub("", toks[-1]).endswith("-")):
                report["skipped_coda_triphones"] += 1
                continue
            mapped = []
            profile_used = False
            token_failed = False
            for token in toks:
                # Existing ARPAsing spellings remain authoritative. The
                # profile is a fallback for Japanese graphemes and extensions.
                if re.fullmatch(r"[A-Za-z_]+(?:\d+)?-?", token):
                    phone = map_token(token, report)
                    values = (phone,) if phone is not None else ()
                else:
                    resolved = (phoneme_profile.resolve(token, max_phones=3)
                                if phoneme_profile is not None else None)
                    values = resolved.phonemes if resolved is not None else ()
                    profile_used = profile_used or bool(values)
                    if not values:
                        phone = map_token(token, report)
                        values = (phone,) if phone is not None else ()
                if not values:
                    token_failed = True
                mapped.extend(value for value in values if value is not None)
            if token_failed:
                report["skipped_nonspeech"] += 1
                continue
            if len(mapped) == 1:                # sustain: X-X diphone
                mapped = [mapped[0], mapped[0]]
                report["singles"] += 1
            if len(mapped) != 2:
                report["bad_lines"].append(
                    f"{e['oto_file']}:{e['line']}")
                if profile_used:
                    report["profile_sequence_skips"].append({
                        "oto_file": e["oto_file"], "line": e["line"],
                        "alias": alias, "phones": list(mapped),
                    })
                continue
            p1, p2 = mapped
            if profile_used:
                report["profile_mapped_aliases"] += 1

            src, source_relative = _source_wav(
                bank, e["oto_path"], e["wav"])
            source_reference = f"{e['oto_file']} -> {e['wav']}"
            if src is None:
                report["outside_wav"].add(source_reference)
                continue
            if not src.is_file():
                report["missing_wav"].add(source_reference)
                continue
            source_paths[source_relative] = src
            try:
                total = wav_length_ms(src)
            except (OSError, EOFError, wave.Error):
                report["bad_wav"].add(source_relative)
                continue

            clean = not any(VARIANT_DIGITS.search(token) for token in toks)
            dip = f"{sanitize_scheme(p1)}-{sanitize_scheme(p2)}"
            raw_start = e["offset"]
            mid = e["offset"] + e["preutterance"]
            raw_end = e["offset"] + abs(e["blank"]) if e["blank"] < 0 \
                else total - e["blank"]
            raw_end = min(raw_end, total)

            # In an UTAU renderer, positive overlap is the fade-in span from
            # the raw OTO offset.  The end of that span is a stable point in
            # the left phone and is the closest available analogue to the
            # phone-center cut required by a Festival diphone index.  Keep a
            # 2 ms half-phone floor for malformed or unusually short OTOs.
            overlap_shift = effective_oto_overlap_ms(
                e["preutterance"], e["overlap"],
                zero_overlap_guard_ms=zero_overlap_guard_ms,
            )
            start = raw_start + overlap_shift
            if float(e["overlap"]) > 0.0 and overlap_shift > 0.0:
                left_center_method = "oto_overlap_end"
                report["oto_overlap_start_centers"] += 1
            elif overlap_shift > 0.0:
                left_center_method = "inferred_zero_overlap_guard"
                report["oto_inferred_overlap_start_centers"] += 1
            else:
                left_center_method = "oto_offset_fallback"

            fixed_end = e["offset"] + max(0.0, float(e["consonant"]))
            if mid + 2.0 <= fixed_end <= raw_end:
                end = fixed_end
                right_center_method = "oto_fixed_end"
                report["oto_fixed_end_centers"] += 1
            else:
                end = raw_end
                right_center_method = "oto_region_end_fallback"
            if not (0 <= start < mid < end):
                report["bad_lines"].append(
                    f"{e['oto_file']}:{e['line']}")
                continue

            if default_subbank is None:
                subbank_rank = 0
            elif _same_subbank(source_affix, default_subbank):
                subbank_rank = 0
            elif source_affix and (selected_color in {None, "*"} or
                                   source_affix["color"] == selected_color):
                subbank_rank = 1 + int(source_affix.get("order", 0))
            else:
                subbank_rank = 100000
            subbank = e["oto_path"].parent.relative_to(bank).as_posix() or "."
            units.append({
                "dip": dip, "p1": sanitize_scheme(p1),
                "p2": sanitize_scheme(p2), "wav": source_relative,
                "alias": alias, "raw_alias": e["alias"],
                "line": e["line"], "oto_file": e["oto_file"],
                "source_subbank": subbank, "affix": source_affix,
                "affix_inferred_from_oto": affix_inferred,
                "alias_cleanup_sources": [item["source"] for item in
                                          normalized["matches"]],
                "source_pitch_tags": normalized["pitch_tags"],
                "subbank_rank": subbank_rank, "clean": clean,
                "start": round(start / 1000.0, 6),
                "mid": round(mid / 1000.0, 6),
                "end": round(end / 1000.0, 6),
                "raw_start": round(raw_start / 1000.0, 6),
                "raw_end": round(raw_end / 1000.0, 6),
                "left_center_method": left_center_method,
                "effective_overlap_ms": round(overlap_shift, 6),
                "right_center_method": right_center_method,
                "oto_timing_ms": {
                    "offset": float(e["offset"]),
                    "consonant": float(e["consonant"]),
                    "cutoff": float(e["blank"]),
                    "preutterance": float(e["preutterance"]),
                    "overlap": float(e["overlap"]),
                },
            })

    if not units:
        raise SystemExit(_no_units_message(
            bank, report, metadata, selected_color))

    # Recover the outer recording context from adjacent, time-ordered OTO
    # transitions in the same wav.  For example, the otherwise-identical
    # t-eh takes retain whether they were recorded in z-t-eh-r or s-t-eh-l.
    by_wav = {}
    for unit in units:
        by_wav.setdefault(unit["wav"], []).append(unit)
    for wav_units in by_wav.values():
        ordered = sorted(wav_units, key=lambda u: (u["mid"], u["line"]))
        for position, unit in enumerate(ordered):
            previous_unit = next(
                (candidate for candidate in reversed(ordered[:position])
                 if candidate["mid"] < unit["mid"] - 0.001), None)
            following_unit = next(
                (candidate for candidate in ordered[position + 1:]
                 if candidate["mid"] > unit["mid"] + 0.001), None)
            previous_chains = bool(
                previous_unit and previous_unit["p2"] == unit["p1"])
            following_chains = bool(
                following_unit and following_unit["p1"] == unit["p2"])

            # Some banks omit an internal transition while retaining the OTOs
            # on either side.  In ``ae s`` followed by ``t k``, the immediate
            # preceding OTO still proves that ``s`` occurs before ``t-k``.
            # Use only the adjacent ordered OTO edge; searching farther back
            # can jump over intervening phones and invent a false context.
            unit["left_context"] = (
                previous_unit["p1"] if previous_chains else
                previous_unit["p2"] if previous_unit else "*")
            unit["right_context"] = (
                following_unit["p2"] if following_chains else
                following_unit["p1"] if following_unit else "*")
            unit["left_context_source"] = (
                "adjacent_transition" if previous_chains else
                "adjacent_oto_edge" if previous_unit else "unavailable")
            unit["right_context_source"] = (
                "adjacent_transition" if following_chains else
                "adjacent_oto_edge" if following_unit else "unavailable")
            # If B-C follows A-B in the same recording, both units can meet
            # at B's exact OTO overlap anchor.  That gives UniSyn a coherent
            # center cut instead of joining A-B's region tail to B-C's raw
            # offset.  Never extend outside A-B's declared source region.
            if (following_chains and
                    following_unit["start"] > unit["mid"] + 0.002 and
                    following_unit["start"] <= unit["raw_end"]):
                unit["end"] = round(following_unit["start"], 6)
                unit["tail_clamped"] = True
                unit["right_center_method"] = "next_oto_overlap_end"
                report["context_tail_clamps"] += 1
                report["oto_chained_end_centers"] += 1

    # Keep one compatibility key (p1-p2) plus an arbitrary number of selectable
    # alternatives (p1__uN-p2).  The suffix is an internal unit key, never a
    # phone name, and is applied with Festival's us_diphone_left feature.
    grouped = {}
    for unit in units:
        grouped.setdefault(unit["dip"], []).append(unit)
    alternatives = {}
    for dip, candidates in sorted(grouped.items()):
        unique, seen = [], set()
        for unit in sorted(candidates, key=_unit_order):
            signature = (unit["wav"], unit["start"], unit["mid"], unit["end"])
            if signature not in seen:
                unique.append(unit)
                seen.add(signature)
        clean_units = [unit for unit in unique if unit["clean"]]
        base = min(clean_units or unique, key=_unit_order)
        ordered = [base] + sorted(
            (unit for unit in unique if unit is not base), key=_unit_order)
        report["dupes"] += max(0, len(ordered) - 1)
        choices = []
        for number, unit in enumerate(ordered):
            left_name = (unit["p1"] if number == 0 else
                         "%s__u%d" % (unit["p1"], number))
            index_name = "%s-%s" % (left_name, unit["p2"])
            if unit["wav"] not in wav_map:
                wav_map[unit["wav"]] = _target_wav_name(
                    unit["wav"], target_owners)
            window_plan = build_source_window_plan(
                unit["start"], unit["mid"], unit["end"],
                mode=source_window_mode,
                half_window_ms=source_window_ms,
            )
            window_names = source_window_variant_names(
                left_name, window_plan
            )
            primary = tuple(round(value, 6) for value in
                            window_plan.geometry("base"))
            index[index_name] = (wav_map[unit["wav"]], *primary)
            if primary != (unit["start"], unit["mid"], unit["end"]):
                report["source_window_bounded_units"] += 1
            for kind in ("left", "right", "both"):
                variant_left = window_names[kind]
                if variant_left == left_name:
                    continue
                variant_key = "%s-%s" % (variant_left, unit["p2"])
                geometry = tuple(round(value, 6) for value in
                                 window_plan.geometry(kind))
                if variant_key not in index:
                    report["source_window_variants"] += 1
                index[variant_key] = (wav_map[unit["wav"]], *geometry)
            left_info = context_edge_info(unit["left_context"], "right")
            right_info = context_edge_info(unit["right_context"], "left")
            choice = {
                "id": "base" if number == 0 else "take%d" % number,
                "left_name": left_name, "index_name": index_name,
                "left_context": unit["left_context"],
                "right_context": unit["right_context"],
                "left_context_source": unit["left_context_source"],
                "right_context_source": unit["right_context_source"],
                "left_context_edge": left_info["phone"],
                "right_context_edge": right_info["phone"],
                "left_context_kind": left_info["kind"],
                "right_context_kind": right_info["kind"],
                "left_class": left_info["class"],
                "right_class": right_info["class"],
                "l_class": (
                    "light" if (unit["p1"].rstrip("_") == "l" and
                                unit["p2"].rstrip("_") in L_LIGHT_FOLLOWERS) else
                    "dark" if unit["p1"].rstrip("_") == "l" else
                    "light" if (unit["p2"].rstrip("_") == "l" and
                                unit["right_context"].rstrip("_") in
                                L_LIGHT_FOLLOWERS)
                    else "dark" if unit["p2"].rstrip("_") == "l" else "*"),
                "alias": unit["alias"], "raw_alias": unit["raw_alias"],
                "wav": unit["wav"], "oto_file": unit["oto_file"],
                "oto_line": unit["line"],
                "source_subbank": unit["source_subbank"],
                "source_color": (unit["affix"].get("color")
                                 if unit["affix"] else None),
                "source_prefix": (unit["affix"].get("prefix")
                                  if unit["affix"] else None),
                "source_suffix": (unit["affix"].get("suffix")
                                  if unit["affix"] else None),
                "source_tone_ranges": (unit["affix"].get("tone_ranges")
                                       if unit["affix"] else []),
                "source_affix_source": (unit["affix"].get("source")
                                        if unit["affix"] else None),
                "source_affix_inferred_from_oto":
                    unit["affix_inferred_from_oto"],
                "alias_cleanup_sources": unit["alias_cleanup_sources"],
                "source_pitch_tags": unit["source_pitch_tags"],
                "start": primary[0], "mid": primary[1],
                "end": primary[2],
                "full_start": unit["start"],
                "full_mid": unit["mid"],
                "full_end": unit["end"],
                "source_window": window_plan.to_dict(),
                "window_left_name": window_names["left"],
                "window_right_name": window_names["right"],
                "window_both_name": window_names["both"],
                "window_left_activation": (
                    round(window_plan.left_activation_duration, 6)
                    if window_plan.left_activation_duration is not None
                    else None
                ),
                "window_right_activation": (
                    round(window_plan.right_activation_duration, 6)
                    if window_plan.right_activation_duration is not None
                    else None
                ),
                "raw_start": unit["raw_start"],
                "raw_end": unit["raw_end"],
                "left_center_method": unit["left_center_method"],
                "effective_overlap_ms": unit["effective_overlap_ms"],
                "right_center_method": unit["right_center_method"],
                "oto_timing_ms": dict(unit["oto_timing_ms"]),
            }
            if unit.get("tail_clamped"):
                choice["tail_clamped"] = True
            choices.append(choice)
        alternatives[dip] = choices

    # Structural ``cl`` is one editable phone in the GUI while source
    # selection uses the following consonant twice. Generate a bounded C-C
    # hold from each available C-V transition so this works for every
    # consonant class without relying on a coincidentally named OTO ``cl``.
    hold_candidates = {}
    for unit in units:
        consonant = str(unit["p1"])
        if (
            phone_context_class(consonant) in {
                "vowel", "silence", "wildcard"
            }
            or phone_context_class(unit["p2"]) != "vowel"
            or float(unit["mid"]) - float(unit["start"]) < 0.008
        ):
            continue
        hold_candidates.setdefault(consonant, []).append(unit)
    for consonant, candidates in sorted(hold_candidates.items()):
        diphone = f"{consonant}-{consonant}"
        if diphone in index:
            continue
        unique = []
        seen = set()
        for unit in sorted(candidates, key=_unit_order):
            signature = (
                unit["wav"], unit["start"], unit["mid"], unit["p2"]
            )
            if signature not in seen:
                unique.append(unit)
                seen.add(signature)
        choices = []
        for number, unit in enumerate(unique):
            source_diphone = f"{consonant}-{unit['p2']}"
            source_choice = next(
                (
                    choice for choice in alternatives.get(
                        source_diphone, ())
                    if choice.get("wav") == unit["wav"]
                    and abs(float(choice.get("full_mid", choice["mid"]))
                            - float(unit["mid"])) < 1e-7
                ),
                (alternatives.get(source_diphone) or [{}])[0],
            )
            duration = float(unit["mid"]) - float(unit["start"])
            phone_class = phone_context_class(consonant)
            retained_fraction = (
                0.72 if phone_class.startswith(("stop", "affricate"))
                else 0.90
            )
            hold_start = float(unit["start"])
            hold_end = hold_start + duration * retained_fraction
            if hold_end - hold_start < 0.006:
                continue
            hold_mid = (hold_start + hold_end) * 0.5
            left_name = (
                consonant if not choices else
                f"{consonant}__hold{len(choices)}"
            )
            index_name = f"{left_name}-{consonant}"
            geometry = (
                round(hold_start, 6),
                round(hold_mid, 6),
                round(hold_end, 6),
            )
            index[index_name] = (wav_map[unit["wav"]], *geometry)
            choice = dict(source_choice)
            choice.update({
                "id": (
                    "structural_hold" if not choices else
                    f"structural_hold{len(choices)}"
                ),
                "left_name": left_name,
                "index_name": index_name,
                "role": "structural_consonant_hold",
                "transition_kind": "cc",
                "alias": unit["alias"],
                "raw_alias": unit["raw_alias"],
                "right_context": unit["p2"],
                "right_context_source": "following_cv_vowel",
                "right_context_edge": unit["p2"],
                "right_context_kind": "atomic",
                "right_class": "vowel",
                "start": geometry[0],
                "mid": geometry[1],
                "end": geometry[2],
                "full_start": geometry[0],
                "full_mid": geometry[1],
                "full_end": geometry[2],
                "geometry_method": "bounded_cv_consonant_hold",
                "source_diphone": source_diphone,
                "source_window": {
                    "mode": "structural_consonant_hold",
                    "base": list(geometry),
                },
                "window_left_name": left_name,
                "window_right_name": left_name,
                "window_both_name": left_name,
                "window_left_activation": None,
                "window_right_activation": None,
            })
            choices.append(choice)
        if choices:
            alternatives[diphone] = choices
            report["structural_consonant_holds"] += 1

    # Report the geometry that survived de-duplication and was actually
    # installed, rather than provisional fixed endpoints later replaced by a
    # chained overlap center.
    installed_choices = [
        choice
        for choices in alternatives.values()
        for choice in choices
        if choice.get("role") != "structural_consonant_hold"
    ]
    report["oto_overlap_start_centers"] = sum(
        choice["left_center_method"] == "oto_overlap_end"
        for choice in installed_choices
    )
    report["oto_inferred_overlap_start_centers"] = sum(
        choice["left_center_method"] == "inferred_zero_overlap_guard"
        for choice in installed_choices
    )
    report["oto_fixed_end_centers"] = sum(
        choice["right_center_method"] == "oto_fixed_end"
        for choice in installed_choices
    )
    report["oto_chained_end_centers"] = sum(
        choice["right_center_method"] == "next_oto_overlap_end"
        for choice in installed_choices
    )

    # ---- wav copy --------------------------------------------------------
    if copy_wavs:
        for src_name, tgt_name in sorted(wav_map.items()):
            tgt = out / "wav" / tgt_name
            if not tgt.exists():
                shutil.copyfile(source_paths[src_name], tgt)
    # ---- dic/<name>_diphone.scm (the Scheme index list) -------------------
    scm = out / "dic" / f"{name}_diphone.scm"
    with scm.open("w", encoding="ascii") as f:
        f.write(f";; {name} diphone index - generated by utau2festvox.py\n")
        f.write(";; (diphone wavfile start_s mid_s end_s)\n")
        f.write(f"(set! {sanitize_scheme(name)}_diphone_index\n  '(\n")
        for dip in sorted(index):
            w, s, m, e = index[dip]
            f.write(f'    ("{dip}" "{w}" {s:.6f} {m:.6f} {e:.6f})\n')
        f.write("  ))\n")

    # ---- dic/<name>_diphone.est (EST_File index for UniSyn) ---------------
    est = out / "dic" / f"{name}_diphone.est"
    with est.open("w", encoding="ascii") as f:
        f.write("EST_File index\nDataType ascii\nNumEntries %d\n"
                "IndexName %s_diphone\nEST_Header_End\n"
                % (len(index), sanitize_scheme(name)))
        for dip in sorted(index):
            w, s, m, e = index[dip]
            f.write(f"{dip} {Path(w).stem} {s:.6f} {m:.6f} {e:.6f}\n")

    # ---- machine-readable copy for other tools (vocab_forge synth) --------
    declared_subbanks = []
    for spec in specs:
        if spec["source"] != "character.yaml":
            continue
        public = _spec_public(spec)
        public["included"] = (selected_color == "*" or
                              spec["color"] == selected_color)
        declared_subbanks.append(public)
    alias_metadata = {
        "character_yaml": _public_source_path(
            metadata["character_yaml"], bank
        ),
        "prefix_map": _public_source_path(metadata["prefix_map"], bank),
        "manual_prefixes": metadata["manual_prefixes"],
        "manual_suffixes": metadata["manual_suffixes"],
        "available_voice_colors": available_colors,
        "selected_voice_color": selected_color,
        "default_subbank": _spec_public(default_subbank),
        "runtime_pitch_selection": False,
    }
    payload = {
        "name": name, "samplerate_note": "see wav files",
        "context_model": "oto_directional_v1",
        # Source-window selection is applied after the OTO centers are
        # established; it does not replace or reinterpret that geometry.
        "diphone_geometry_model": "oto_overlap_centers_v3",
        "source_window_policy": {
            "mode": source_window_mode,
            "half_window_ms": round(source_window_ms, 6),
            "adaptive_full_window_threshold": (
                "target phone half-duration must accommodate the full "
                "source half"
            ),
            "context_selection_precedence": (
                "recording-first-window-second"
            ),
            "zero_overlap_guard_ms": round(zero_overlap_guard_ms, 6),
            "zero_overlap_policy": (
                "preserve raw OTO geometry by default; a nonzero guard is an "
                "explicit source-cut experiment"
            ),
        },
        "subbank_mode": "provenance_only",
        "subbank_note": ("Pitch/color provenance is retained; dynamic "
                         "F0-driven subbank selection is not implemented."),
        "alias_metadata": alias_metadata,
        "source_subbanks": declared_subbanks,
        "oto_files": report["oto_files"],
        "special_phone_realizations": generated_voice_policy(),
        "index": index, "alternatives": alternatives,
    }
    if phoneme_profile is not None:
        payload["arpasing_profile"] = phoneme_profile.metadata()
        payload["japanese_phoneme_map"] = phoneme_profile.runtime_map()
        payload["profile_conversion"] = {
            "mapped_alias_count": report["profile_mapped_aliases"],
            "skipped_sequences": report["profile_sequence_skips"],
        }
    (out / "dic" / "diphone_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- festival/ stub ----------------------------------------------------
    (out / "festival" / f"{name}_diphone_stub.scm").write_text(
        f""";; Minimal scaffold for a FestVox UniSyn diphone voice "{name}".
;; Load the index and point us_diphone_init at it:
(load (path-append (pwd) "../dic/{name}_diphone.scm"))
(set! {sanitize_scheme(name)}_db_params
      (list
       (list 'name '{sanitize_scheme(name)})
       (list 'index_file (path-append (pwd) "../dic/{name}_diphone.est"))
       (list 'grouped "false")
       (list 'base_dir (path-append (pwd) ".."))
       (list 'coef_dir "wav")   ;; using raw wav; run make_lpc for LPC coefs
       (list 'sig_dir  "wav")
       (list 'default_diphone "pau-pau")))
;; (us_diphone_init {sanitize_scheme(name)}_db_params)
""", encoding="ascii")

    # ---- report ------------------------------------------------------------
    rep = out / "conversion_report.txt"
    color_label = ("all" if selected_color == "*" else
                   "default" if selected_color == "" else
                   selected_color or "not declared")
    metadata_sources = [
        item for item in (
            _public_source_path(metadata["character_yaml"], bank),
            _public_source_path(metadata["prefix_map"], bank),
        ) if item
    ]
    warning_lines = "".join(
        f"WARNING metadata  : {warning}\n" for warning in metadata["warnings"])
    if report["possible_affixes"]:
        warning_lines += (
            "WARNING affixes   : possible undeclared alias endings " +
            repr(sorted(report["possible_affixes"])[:20]) + "\n"
            "  Point to character.yaml/prefix.map or pass --alias-prefix / "
            "--alias-suffix. A manual P suffix handles ayPE3 and ayE3P.\n")
    rep.write_text(
        "utau2festvox conversion report\n"
        f"diphones indexed : {len(index)}\n"
        f"wav files used   : {len(wav_map)}\n"
        f"oto files read   : {len(oto_files)} "
        f"({len(report['empty_oto_files'])} empty)\n"
        f"oto entries      : {report['oto_entries']}\n"
        f"metadata sources : {metadata_sources or ['none']}\n"
        f"voice color      : {color_label} "
        f"({report['color_skips']} other-color entries skipped)\n"
        f"default subbank  : {_spec_public(default_subbank)}\n"
        f"alias cleanup    : {report['aliases_normalized']} normalized; "
        f"{report['affix_normalized']} declared/manual affixes; "
        f"{report['pitch_normalized']} legacy pitch tags\n"
        f"OTO provenance   : {report['oto_affix_inferences']} entries inherited "
        f"a strongly matched OTO subbank; "
        f"{report['unresolved_affix_skips']} unresolved affix entries skipped\n"
        "pitch subbanks   : provenance retained; dynamic F0 routing is not "
        "implemented\n"
        f"sustain singles  : {report['singles']}\n"
        f"variant takes    : {report['dupes']} alternatives preserved\n"
        f"overlap centers  : {report['oto_overlap_start_centers']} left "
        "centers use OTO overlap ends\n"
        f"inferred overlap : {report['oto_inferred_overlap_start_centers']} "
        f"zero-overlap entries use a bounded {zero_overlap_guard_ms:.1f} ms "
        "source guard\n"
        f"fixed centers    : {report['oto_fixed_end_centers']} right centers "
        "use OTO fixed-region ends\n"
        f"chained centers  : {report['oto_chained_end_centers']} right centers "
        "meet the next matching OTO overlap anchor\n"
        f"context tails    : {report['context_tail_clamps']} centered at the "
        "next matching phone anchor\n"
        f"source windows  : {source_window_mode}, "
        f"{source_window_ms:.1f} ms per normal half; "
        f"{report['source_window_bounded_units']} units bounded, "
        f"{report['source_window_variants']} hidden stretch variants\n"
        f"structural holds: {report['structural_consonant_holds']} C-C units "
        "derived from bounded C-V consonant regions for language-neutral cl\n"
        f"coda triphones   : {report['skipped_coda_triphones']} ignored "
        "(V-C-sil aliases)\n"
        f"non-speech skips : {report['skipped_nonspeech']}\n"
        f"bad oto lines    : {sorted(set(report['bad_lines']))[:20]}\n"
        f"missing wavs     : {sorted(report['missing_wav'])[:20]}\n"
        f"outside wavs     : {sorted(report['outside_wav'])[:20]}\n"
        f"invalid wavs     : {sorted(report['bad_wav'])[:20]}\n"
        f"unmapped tokens  : {sorted(report['unmapped'])}\n"
        + warning_lines,
        encoding="utf-8")
    print(rep.read_text(encoding="utf-8"))
    print(f"wrote {scm}\n      {est}\n      dic/diphone_index.json")
    return {"out": str(out), "diphones": len(alternatives),
            "runtime_units": len(index), "wavs": len(wav_map),
            "unmapped": sorted(report["unmapped"]),
            "selected_voice_color": selected_color,
            "default_subbank": _spec_public(default_subbank)}


# --------------------------------------------------------------------------
# Config file (shared with synth_diphone.py). Located in this order:
#   --config PATH  >  $FESTVOX_CONFIG  >  ./festvox.json  >  <script dir>/festvox.json
# Schema:
#   { "output_root": "<dir where DBs are built>",
#     "voices": { "<key>": {"bank": "<UTAU dir>", "name": "<phoneset>",
#                           "out": "<optional explicit DB dir>",
#                           "copy_wavs": true,
#                           "character_yaml": "<optional path>",
#                           "prefix_map": "<optional path>",
#                           "alias_prefixes": [], "alias_suffixes": [],
#                           "voice_color": "<optional OpenUtau color>"} } }
# A voice builds to  out  if given, else  output_root/<key>.
# --------------------------------------------------------------------------
import os

def find_config(explicit=None):
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("FESTVOX_CONFIG"):
        cands.append(Path(os.environ["FESTVOX_CONFIG"]))
    cands += [Path.cwd() / "festvox.json",
              Path(__file__).resolve().parent / "festvox.json"]
    for c in cands:
        if c and c.is_file():
            return c
    return None


def load_config(explicit=None):
    fp = find_config(explicit)
    if not fp:
        return None, None
    return json.loads(fp.read_text(encoding="utf-8")), fp


def voice_out_dir(cfg, key, spec):
    if spec.get("out"):
        return Path(spec["out"])
    root = cfg.get("output_root") or "."
    return Path(root) / key


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build FestVox diphone DB(s) from UTAU voicebank(s). "
                    "With no --bank, builds voices listed in festvox.json.")
    ap.add_argument("--config", default=None,
                    help="path to festvox.json (default: auto-locate)")
    ap.add_argument("--voice", default=None,
                    help="build only this voice key from the config "
                         "(default: build all)")
    ap.add_argument("--bank", type=Path, default=None,
                    help="one-off: UTAU bank dir (bypasses the config)")
    ap.add_argument("--out", type=Path, default=None,
                    help="one-off: output DB dir (default: <bank>/festvox_db)")
    ap.add_argument("--name", default=None, help="one-off: phoneset name")
    ap.add_argument("--no-copy", action="store_true",
                    help="index only; do not copy wav files")
    ap.add_argument("--character-yaml", default=None,
                    help="OpenUtau character.yaml path; auto-detected at the "
                         "bank root when omitted")
    ap.add_argument("--prefix-map", default=None,
                    help="legacy prefix.map path; used after character.yaml "
                         "and auto-detected at the bank root")
    ap.add_argument("--alias-prefix", action="append", default=[],
                    help="exact manual alias prefix to remove; repeatable")
    ap.add_argument("--alias-suffix", action="append", default=[],
                    help="exact manual alias suffix to remove; repeatable; "
                         "combines with pitch tags in either order")
    ap.add_argument("--voice-color", default=None,
                    help="OpenUtau color to build. Default uses the uncolored "
                         "subbanks; use 'all' only to include every color")
    ap.add_argument(
        "--source-window-mode", choices=SOURCE_WINDOW_MODES,
        default="adaptive",
        help=("adaptive bounds normal phones but restores full source halves "
              "for sufficiently stretched phones; bounded never restores "
              "them; full preserves legacy whole-region indexing"),
    )
    ap.add_argument(
        "--source-window-ms", type=float,
        default=DEFAULT_SOURCE_WINDOW_MS,
        help="maximum source milliseconds on each side for normal phones",
    )
    ap.add_argument(
        "--zero-overlap-guard-ms", type=float,
        default=DEFAULT_ZERO_OVERLAP_GUARD_MS,
        help=("experimental source-cut guard for zero OTO overlap; "
              "disabled by default because it changes recorded geometry"),
    )
    a = ap.parse_args()

    if a.bank:                                   # explicit one-off build
        convert(a.bank, a.out or a.bank / "festvox_db", a.name or "asaxi",
                copy_wavs=not a.no_copy,
                character_yaml=a.character_yaml, prefix_map=a.prefix_map,
                alias_prefixes=a.alias_prefix,
                alias_suffixes=a.alias_suffix, voice_color=a.voice_color,
                source_window_mode=a.source_window_mode,
                source_window_ms=a.source_window_ms,
                zero_overlap_guard_ms=a.zero_overlap_guard_ms)
        sys.exit(0)

    cfg, fp = load_config(a.config)
    if not cfg:
        sys.exit("No festvox.json found and no --bank given. "
                 "Create festvox.json (see GUIDE.md) or pass --bank.")
    print(f"config: {fp}")
    voices = cfg.get("voices") or {}
    keys = [a.voice] if a.voice else list(voices)
    if a.voice and a.voice not in voices:
        sys.exit(f"voice {a.voice!r} not in config (have: {list(voices)})")
    if not keys:
        sys.exit("config has no voices.")
    for key in keys:
        spec = voices[key]
        out = voice_out_dir(cfg, key, spec)
        print(f"\n=== building '{key}' -> {out} ===")
        prefixes = list(spec.get("alias_prefixes") or []) + a.alias_prefix
        suffixes = list(spec.get("alias_suffixes") or []) + a.alias_suffix
        convert(Path(spec["bank"]), out, spec.get("name", "asaxi"),
                copy_wavs=spec.get("copy_wavs", True) and not a.no_copy,
                character_yaml=(a.character_yaml or
                                spec.get("character_yaml")),
                prefix_map=a.prefix_map or spec.get("prefix_map"),
                alias_prefixes=prefixes, alias_suffixes=suffixes,
                voice_color=(a.voice_color if a.voice_color is not None
                             else spec.get("voice_color")),
                source_window_mode=spec.get(
                    "source_window_mode", a.source_window_mode),
                source_window_ms=spec.get(
                    "source_window_ms", a.source_window_ms),
                zero_overlap_guard_ms=spec.get(
                    "zero_overlap_guard_ms", a.zero_overlap_guard_ms))
