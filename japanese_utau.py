#!/usr/bin/env python3
"""Read-only Japanese UTAU OTO inspection and bank classification.

This module is deliberately separate from the production ARPAsing converter.
It is the first phase of Japanese voicebank support: decode metadata strictly,
preserve source evidence, classify aliases, and report uncertainty. It does not
convert audio or modify the source voicebank.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


SCHEMA_VERSION = 1
AUTO_ENCODINGS = ("utf-8", "cp932")
BANK_TYPES = ("auto", "cv", "vcv", "cvvc", "mixed")

_PITCH_SUFFIX = re.compile(
    r"(?P<separator>_?)(?P<take>\d*)(?P<pitch>[A-G](?:#|b)?-?\d+)$",
    re.IGNORECASE,
)
_KANA_ONLY = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u30fc]+$")
_HAS_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_ROMAJI = re.compile(r"[a-z][a-z'_-]*$", re.IGNORECASE)
_ROMAJI_CONTEXTS = {
    "a", "i", "u", "e", "o", "n", "nn", "ng", "m", "N",
}
_SILENCE_ALIASES = {"pau", "sil", "sp", "rest", "r"}
_BREATH_ALIASES = {"br", "bre", "breath", "息", "吸", "吐"}
_GEMINATE_ALIASES = {"っ", "ッ", "q", "cl"}


# Common CVVC banks use RB for a rest/breath transition. It is source
# material, but never a linguistic tapped-r consonant.
_BREATH_ALIASES.add("rb")


class TextDecodeError(UnicodeError):
    """Raised when metadata cannot be decoded without data loss."""


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    path: str
    line: Optional[int] = None
    byte_offset: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.byte_offset is not None:
            result["byte_offset"] = self.byte_offset
        return result


@dataclass(frozen=True)
class DecodedText:
    path: Path
    text: str
    encoding: str
    confidence: float
    ambiguous: bool
    candidate_scores: dict[str, float]
    sha256: str
    byte_length: int
    diagnostics: tuple[Diagnostic, ...] = ()
    raw_bytes: bytes = field(default=b"", repr=False)

    def source_dict(self) -> dict:
        return {
            "path": str(self.path),
            "encoding": self.encoding,
            "encoding_confidence": self.confidence,
            "encoding_ambiguous": self.ambiguous,
            "encoding_candidate_scores": dict(self.candidate_scores),
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }


@dataclass(frozen=True)
class AliasNormalization:
    source_alias: str
    canonical_alias: str
    match_key: str
    analysis_alias: str
    removed_prefixes: tuple[str, ...] = ()
    removed_suffixes: tuple[str, ...] = ()
    pitch_tags: tuple[str, ...] = ()
    alternative_numbers: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "source_alias": self.source_alias,
            "canonical_alias": self.canonical_alias,
            "match_key": self.match_key,
            "analysis_alias": self.analysis_alias,
            "removed_prefixes": list(self.removed_prefixes),
            "removed_suffixes": list(self.removed_suffixes),
            "pitch_tags": list(self.pitch_tags),
            "alternative_numbers": list(self.alternative_numbers),
        }


@dataclass(frozen=True)
class AliasEvidence:
    role: str
    family: Optional[str]
    subtype: str
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "family": self.family,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OtoEntry:
    source_path: Path
    line_number: int
    byte_offset: int
    raw_line: str
    wav_raw: str
    alias_raw: str
    offset: Optional[float]
    consonant: Optional[float]
    cutoff: Optional[float]
    preutterance: Optional[float]
    overlap: Optional[float]
    normalization: AliasNormalization
    evidence: AliasEvidence

    @property
    def timing_valid(self) -> bool:
        return all(value is not None for value in (
            self.offset, self.consonant, self.cutoff,
            self.preutterance, self.overlap,
        ))

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path),
            "line": self.line_number,
            "byte_offset": self.byte_offset,
            "raw_line": self.raw_line,
            "wav_raw": self.wav_raw,
            "alias_raw": self.alias_raw,
            "timing": {
                "offset": self.offset,
                "consonant": self.consonant,
                "cutoff": self.cutoff,
                "preutterance": self.preutterance,
                "overlap": self.overlap,
                "valid": self.timing_valid,
            },
            "normalization": self.normalization.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class OtoDocument:
    source: DecodedText
    entries: tuple[OtoEntry, ...]
    diagnostics: tuple[Diagnostic, ...]

    def to_dict(self, include_entries: bool = False) -> dict:
        result = self.source.source_dict()
        result.update({
            "entry_count": len(self.entries),
            "valid_timing_entries": sum(e.timing_valid for e in self.entries),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        })
        if include_entries:
            result["entries"] = [entry.to_dict() for entry in self.entries]
        return result


@dataclass(frozen=True)
class BankAnalysis:
    source_root: Path
    bank_type: str
    confidence: float
    classification_reason: str
    bank_type_override: Optional[str]
    documents: tuple[OtoDocument, ...]
    role_counts: dict[str, int]
    family_counts: dict[str, int]
    family_shares: dict[str, float]
    metadata_files: dict[str, dict]
    diagnostics: tuple[Diagnostic, ...]
    examples: dict[str, tuple[dict, ...]]

    @property
    def entries(self) -> tuple[OtoEntry, ...]:
        return tuple(entry for doc in self.documents for entry in doc.entries)

    def to_dict(self, include_entries: bool = False) -> dict:
        entries = self.entries
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "japanese_utau_bank_analysis",
            "source_root": str(self.source_root),
            "bank_type": self.bank_type,
            "bank_type_confidence": self.confidence,
            "bank_type_override": self.bank_type_override,
            "classification_reason": self.classification_reason,
            "oto_file_count": len(self.documents),
            "entry_count": len(entries),
            "valid_timing_entries": sum(e.timing_valid for e in entries),
            "role_counts": dict(self.role_counts),
            "family_counts": dict(self.family_counts),
            "family_shares": dict(self.family_shares),
            "metadata_files": dict(self.metadata_files),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "examples": {
                key: list(value) for key, value in self.examples.items()
            },
            "oto_files": [
                doc.to_dict(include_entries=include_entries)
                for doc in self.documents
            ],
        }


def _text_score(text: str) -> float:
    """Score a strict decode using OTO syntax and plausible text content."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.20
    oto_lines = sum("=" in line and line.count(",") >= 5 for line in lines)
    syntax = oto_lines / len(lines)
    japanese = len(_HAS_KANA.findall(text))
    controls = sum(
        unicodedata.category(char) == "Cc" and char not in "\r\n\t"
        for char in text
    )
    score = 0.25 + 0.55 * syntax
    score += min(0.15, japanese / max(1, len(text)) * 3.0)
    score -= min(0.50, controls / max(1, len(text)) * 20.0)
    return round(max(0.0, min(1.0, score)), 4)


def decode_text_file(path: Path, encoding_override: Optional[str] = None) -> DecodedText:
    """Decode metadata strictly; never use replacement characters or Latin-1."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    diagnostics: list[Diagnostic] = []

    if encoding_override:
        try:
            encoding = codecs.lookup(encoding_override).name
        except LookupError as exc:
            raise TextDecodeError(
                f"unknown encoding {encoding_override!r} for {path}"
            ) from exc
        try:
            text = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            raise TextDecodeError(
                f"{path} is not valid {encoding}: byte {exc.start}"
            ) from exc
        return DecodedText(
            path=path,
            text=text,
            encoding=encoding,
            confidence=1.0,
            ambiguous=False,
            candidate_scores={encoding: _text_score(text)},
            sha256=digest,
            byte_length=len(raw),
            diagnostics=(),
            raw_bytes=raw,
        )

    if raw.startswith(codecs.BOM_UTF8):
        text = raw.decode("utf-8-sig", errors="strict")
        return DecodedText(
            path=path,
            text=text,
            encoding="utf-8-sig",
            confidence=1.0,
            ambiguous=False,
            candidate_scores={"utf-8-sig": _text_score(text)},
            sha256=digest,
            byte_length=len(raw),
            diagnostics=(),
            raw_bytes=raw,
        )

    candidates: dict[str, str] = {}
    failures: dict[str, UnicodeDecodeError] = {}
    for encoding in AUTO_ENCODINGS:
        try:
            candidates[encoding] = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            failures[encoding] = exc

    if not candidates:
        detail = ", ".join(
            f"{name} failed at byte {exc.start}"
            for name, exc in failures.items()
        )
        raise TextDecodeError(f"could not decode {path} strictly: {detail}")

    scores = {name: _text_score(text) for name, text in candidates.items()}
    if len(candidates) == 1:
        encoding, text = next(iter(candidates.items()))
        confidence = 0.99
        ambiguous = False
    elif len(set(candidates.values())) == 1:
        # ASCII-only OTO files are byte-identical under both codecs.
        encoding = "utf-8"
        text = candidates[encoding]
        confidence = 1.0
        ambiguous = False
    else:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        encoding = ranked[0][0]
        text = candidates[encoding]
        gap = ranked[0][1] - ranked[1][1]
        confidence = round(min(0.98, 0.58 + max(0.0, gap)), 3)
        ambiguous = gap < 0.12
        if ambiguous:
            diagnostics.append(Diagnostic(
                severity="warning",
                code="encoding_ambiguous",
                message=(
                    "UTF-8 and CP932 both decode strictly with similar scores; "
                    "use --encoding to make the choice explicit"
                ),
                path=str(path),
            ))

    return DecodedText(
        path=path,
        text=text,
        encoding=encoding,
        confidence=confidence,
        ambiguous=ambiguous,
        candidate_scores=scores,
        sha256=digest,
        byte_length=len(raw),
        diagnostics=tuple(diagnostics),
        raw_bytes=raw,
    )


def normalize_alias(
    source_alias: str,
    alias_prefixes: Sequence[str] = (),
    alias_suffixes: Sequence[str] = (),
) -> AliasNormalization:
    """Create separate lossless, canonical, and compatibility-match forms."""
    canonical = unicodedata.normalize("NFC", source_alias)
    match_key = unicodedata.normalize("NFKC", canonical)
    working = match_key.strip()
    prefixes = sorted((str(item) for item in alias_prefixes if item),
                      key=len, reverse=True)
    suffixes = sorted((str(item) for item in alias_suffixes if item),
                      key=len, reverse=True)
    removed_prefixes: list[str] = []
    removed_suffixes: list[str] = []
    pitch_tags: list[str] = []
    alternatives: list[int] = []

    # Affixes and note tags occur in both P+E3 and E3+P orders in real banks.
    for _ in range(12):
        changed = False
        for prefix in prefixes:
            normalized = unicodedata.normalize("NFKC", prefix)
            if normalized and working.startswith(normalized):
                working = working[len(normalized):]
                removed_prefixes.append(prefix)
                changed = True
                break
        if changed:
            continue
        for suffix in suffixes:
            normalized = unicodedata.normalize("NFKC", suffix)
            if normalized and working.endswith(normalized):
                working = working[:-len(normalized)]
                removed_suffixes.append(suffix)
                changed = True
                break
        if changed:
            continue
        # RB1 is a numbered rest/breath alias, not the musical pitch B1.
        # Leave its number for the candidate compiler's alternative parser.
        if re.search(r"(?:^|\s)RB\d+$", working, re.IGNORECASE):
            break
        match = _PITCH_SUFFIX.search(working)
        if match:
            pitch_tags.append(match.group("pitch"))
            if match.group("take"):
                alternatives.append(int(match.group("take")))
            working = working[:match.start()]
            changed = True
        if not changed:
            break

    return AliasNormalization(
        source_alias=source_alias,
        canonical_alias=canonical,
        match_key=match_key,
        analysis_alias=working.strip(),
        removed_prefixes=tuple(removed_prefixes),
        removed_suffixes=tuple(removed_suffixes),
        pitch_tags=tuple(pitch_tags),
        alternative_numbers=tuple(alternatives),
    )


def _is_kana(token: str) -> bool:
    return bool(token and _KANA_ONLY.fullmatch(token))


def _is_romaji_mora(token: str) -> bool:
    token = token.casefold().rstrip("-")
    return bool(_ROMAJI.fullmatch(token)) and (
        token.endswith(("a", "i", "u", "e", "o"))
        or token in {"n", "nn", "ng"}
    )


def _is_context(token: str) -> bool:
    return token in _ROMAJI_CONTEXTS or token.casefold() in {
        item.casefold() for item in _ROMAJI_CONTEXTS
    }


def classify_alias(normalization: AliasNormalization) -> AliasEvidence:
    """Classify one alias from OTO text only; WAV names are never consulted."""
    alias = normalization.analysis_alias
    folded = alias.casefold()
    if not alias:
        return AliasEvidence("unknown", None, "empty", 0.0,
                             ("no alias remains after declared metadata",))
    if folded in _SILENCE_ALIASES:
        return AliasEvidence("silence", None, "silence", 0.98,
                             ("recognized silence alias",))
    if folded in {item.casefold() for item in _BREATH_ALIASES}:
        return AliasEvidence("breath", None, "breath", 0.92,
                             ("recognized breath alias",))
    if alias in _GEMINATE_ALIASES or folded in _GEMINATE_ALIASES:
        return AliasEvidence("special_mora", None, "geminate_closure", 0.92,
                             ("recognized geminate or closure alias",))

    parts = alias.split()
    if len(parts) == 2:
        left, right = parts
        if right == "R" or right.casefold() in {
                item.casefold() for item in _BREATH_ALIASES}:
            return AliasEvidence(
                "breath", None, "context_to_breath", 0.98,
                (
                    ("uppercase R rest alias; preserved as non-speech "
                     "material") if right == "R" else
                    ("context-prefixed rest/breath alias; preserved as "
                     "non-speech material"),
                ),
            )
        if left == "-" and (_is_kana(right) or _is_romaji_mora(right)):
            return AliasEvidence(
                "cv", "cv", "phrase_initial_cv", 0.98,
                ("phrase-start marker followed by a mora",),
            )
        if left == "*" and (_is_kana(right) or _is_romaji_mora(right)):
            return AliasEvidence(
                "cv", "cv", "vowel_blend", 0.99,
                (
                    "asterisk vowel-blend marker followed by a mora; "
                    "the OTO offset is the audible vowel onset",
                ),
            )
        if _is_context(left) and _is_kana(right):
            return AliasEvidence(
                "vcv", "vcv", "vowel_to_mora", 0.99,
                ("vowel or mora-nasal context followed by kana",),
            )
        if _is_context(left) and _is_romaji_mora(right):
            return AliasEvidence(
                "vcv", "vcv", "vowel_to_romaji_mora", 0.84,
                ("vowel context followed by a romaji mora",),
            )
        if (_is_context(left) and _ROMAJI.fullmatch(right)
                and not _is_romaji_mora(right)):
            subtype = "vc_release" if right.endswith("-") else "vc_transition"
            return AliasEvidence(
                "cvvc_vc", "cvvc", subtype, 0.96,
                ("vowel or nasal context followed by a consonant alias",),
            )
        if _is_kana(left) and _ROMAJI.fullmatch(right):
            subtype = "mora_release" if right.endswith("-") else "mora_to_consonant"
            return AliasEvidence(
                "cvvc_vc", "cvvc", subtype, 0.86,
                ("kana mora followed by a consonant alias",),
            )
        return AliasEvidence(
            "unknown", None, "unclassified_pair", 0.20,
            ("two-part alias does not match a safe Japanese role",),
        )

    if len(parts) == 1 and _is_kana(alias):
        subtype = "mora_nasal" if alias in {"ん", "ン"} else "kana_mora"
        return AliasEvidence(
            "cv", "cv", subtype, 0.98,
            ("standalone kana mora",),
        )
    if len(parts) == 1 and _is_romaji_mora(alias):
        return AliasEvidence(
            "cv", "cv", "romaji_mora", 0.76,
            ("standalone romaji mora",),
        )
    return AliasEvidence(
        "unknown", None, "unclassified", 0.10,
        ("phonetic role cannot be inferred safely from the OTO alias",),
    )


def _iter_decoded_lines(decoded: DecodedText) -> Iterable[tuple[int, int, str]]:
    raw = decoded.raw_bytes
    codec = "utf-8" if decoded.encoding == "utf-8-sig" else decoded.encoding
    offset = 0
    chunks = raw.split(b"\n")
    for index, chunk in enumerate(chunks, 1):
        start = offset
        offset += len(chunk) + (1 if index < len(chunks) else 0)
        if chunk.endswith(b"\r"):
            chunk = chunk[:-1]
        if index == 1 and decoded.encoding == "utf-8-sig" \
                and chunk.startswith(codecs.BOM_UTF8):
            chunk = chunk[len(codecs.BOM_UTF8):]
        yield index, start, chunk.decode(codec, errors="strict")


def _parse_number(
    value: str,
    field_name: str,
    path: Path,
    line: int,
    byte_offset: int,
    diagnostics: list[Diagnostic],
) -> Optional[float]:
    try:
        number = float(value.strip())
        if not math.isfinite(number):
            raise ValueError
        return number
    except ValueError:
        diagnostics.append(Diagnostic(
            severity="error",
            code="oto_invalid_number",
            message=f"{field_name} is not a finite number: {value!r}",
            path=str(path),
            line=line,
            byte_offset=byte_offset,
        ))
        return None


def parse_oto_file(
    path: Path,
    encoding_override: Optional[str] = None,
    alias_prefixes: Sequence[str] = (),
    alias_suffixes: Sequence[str] = (),
) -> OtoDocument:
    """Parse one OTO without discarding aliases that have malformed timing."""
    path = Path(path)
    decoded = decode_text_file(path, encoding_override)
    diagnostics = list(decoded.diagnostics)
    entries: list[OtoEntry] = []
    timing_names = ("offset", "consonant", "cutoff", "preutterance", "overlap")

    for line_number, byte_offset, raw_line in _iter_decoded_lines(decoded):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if "=" not in raw_line:
            diagnostics.append(Diagnostic(
                severity="warning",
                code="oto_missing_equals",
                message="non-comment OTO line has no '=' separator",
                path=str(path),
                line=line_number,
                byte_offset=byte_offset,
            ))
            continue

        wav_raw, rest = raw_line.split("=", 1)
        fields = rest.split(",")
        if len(fields) != 6:
            diagnostics.append(Diagnostic(
                severity="error",
                code="oto_field_count",
                message=f"expected 6 comma-separated OTO fields, found {len(fields)}",
                path=str(path),
                line=line_number,
                byte_offset=byte_offset,
            ))
        if not fields:
            continue
        alias_raw = fields[0]
        numeric_raw = fields[1:6]
        numeric_raw.extend([""] * (5 - len(numeric_raw)))
        numbers = [
            _parse_number(value, name, path, line_number, byte_offset, diagnostics)
            for name, value in zip(timing_names, numeric_raw)
        ]
        if not wav_raw.strip():
            diagnostics.append(Diagnostic(
                severity="error",
                code="oto_empty_wav",
                message="OTO entry has an empty WAV field",
                path=str(path),
                line=line_number,
                byte_offset=byte_offset,
            ))
        if not alias_raw.strip():
            diagnostics.append(Diagnostic(
                severity="warning",
                code="oto_empty_alias",
                message="OTO entry has an empty alias",
                path=str(path),
                line=line_number,
                byte_offset=byte_offset,
            ))

        normalization = normalize_alias(
            alias_raw,
            alias_prefixes=alias_prefixes,
            alias_suffixes=alias_suffixes,
        )
        entries.append(OtoEntry(
            source_path=path,
            line_number=line_number,
            byte_offset=byte_offset,
            raw_line=raw_line,
            wav_raw=wav_raw,
            alias_raw=alias_raw,
            offset=numbers[0],
            consonant=numbers[1],
            cutoff=numbers[2],
            preutterance=numbers[3],
            overlap=numbers[4],
            normalization=normalization,
            evidence=classify_alias(normalization),
        ))

    return OtoDocument(
        source=decoded,
        entries=tuple(entries),
        diagnostics=tuple(diagnostics),
    )


def _file_provenance(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
    }


def _discover_metadata(root: Path) -> dict[str, dict]:
    result = {}
    for name in ("character.yaml", "prefix.map", "presamp.ini"):
        path = root / name
        if path.is_file():
            result[name] = _file_provenance(path)
    return result


def _classify_bank(
    family_counts: Counter,
    override: Optional[str],
) -> tuple[str, float, str, dict[str, float]]:
    counts = {name: int(family_counts.get(name, 0))
              for name in ("cv", "vcv", "cvvc")}
    total = sum(counts.values())
    shares = {
        name: round(value / total, 4) if total else 0.0
        for name, value in counts.items()
    }
    if override and override != "auto":
        return override, 1.0, "explicit bank-type override", shares
    if not total:
        return "unknown", 0.0, "no safely classified Japanese aliases", shares

    minimum = max(3, int(total * 0.08))
    if (counts["vcv"] >= minimum and shares["vcv"] >= 0.45
            and counts["vcv"] >= counts["cvvc"] * 2):
        bank_type = "vcv"
        reason = "vowel-to-mora aliases dominate the distinctive evidence"
    elif (counts["cvvc"] >= minimum and shares["cvvc"] >= 0.35
          and counts["cvvc"] >= counts["vcv"] * 2):
        bank_type = "cvvc"
        reason = "VC transition aliases dominate the distinctive evidence"
    elif shares["cv"] >= 0.55:
        bank_type = "cv"
        reason = "standalone and phrase-initial mora aliases dominate"
    else:
        bank_type = "mixed"
        reason = "no one configuration dominates the alias evidence"

    ordered = sorted(shares.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    confidence = round(min(0.99, 0.55 + margin * 0.55
                           + min(total, 200) / 2000.0), 3)
    if bank_type == "mixed":
        confidence = min(confidence, 0.70)
    return bank_type, confidence, reason, shares


def _find_oto_files(source: Path) -> tuple[Path, list[Path]]:
    source = source.expanduser().resolve()
    if source.is_file():
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Japanese UTAU source not found: {source}")
    files = sorted(
        (path for path in source.rglob("*")
         if path.is_file() and path.name.casefold() == "oto.ini"),
        key=lambda path: path.relative_to(source).as_posix().casefold(),
    )
    if not files:
        raise FileNotFoundError(f"no oto.ini files found under {source}")
    return source, files


def analyze_bank(
    source: Path,
    encoding_override: Optional[str] = None,
    bank_type: str = "auto",
    alias_prefixes: Sequence[str] = (),
    alias_suffixes: Sequence[str] = (),
    oto_files: Optional[Sequence[Path]] = None,
) -> BankAnalysis:
    """Analyze a file or bank tree without writing into it."""
    if bank_type not in BANK_TYPES:
        raise ValueError(f"bank_type must be one of {BANK_TYPES}, got {bank_type!r}")
    selected_source = Path(source).expanduser().resolve()
    if oto_files is None:
        root, selected_oto_files = _find_oto_files(selected_source)
    else:
        root = (
            selected_source.parent
            if selected_source.is_file() else selected_source
        )
        if not root.is_dir():
            raise FileNotFoundError(
                f"Japanese UTAU source not found: {selected_source}"
            )
        selected = []
        for value in oto_files:
            path = Path(value).expanduser().resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"selected Japanese OTO is outside the source: {path}"
                ) from exc
            if not path.is_file() or path.name.casefold() != "oto.ini":
                raise FileNotFoundError(
                    f"selected Japanese OTO not found: {path}"
                )
            selected.append(path)
        selected_oto_files = sorted(
            dict.fromkeys(selected),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        if not selected_oto_files:
            raise ValueError("explicit Japanese OTO scope must not be empty")
    documents = tuple(
        parse_oto_file(
            path,
            encoding_override=encoding_override,
            alias_prefixes=alias_prefixes,
            alias_suffixes=alias_suffixes,
        )
        for path in selected_oto_files
    )
    entries = tuple(entry for doc in documents for entry in doc.entries)
    role_counts = Counter(entry.evidence.role for entry in entries)
    family_counts = Counter(
        entry.evidence.family for entry in entries if entry.evidence.family
    )
    override = None if bank_type == "auto" else bank_type
    detected, confidence, reason, shares = _classify_bank(
        family_counts, override
    )
    diagnostics = tuple(item for doc in documents for item in doc.diagnostics)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        role = entry.evidence.role
        if len(grouped[role]) >= 6:
            continue
        grouped[role].append({
            "alias": entry.alias_raw,
            "analysis_alias": entry.normalization.analysis_alias,
            "path": str(entry.source_path),
            "line": entry.line_number,
            "subtype": entry.evidence.subtype,
        })

    return BankAnalysis(
        source_root=root,
        bank_type=detected,
        confidence=confidence,
        classification_reason=reason,
        bank_type_override=override,
        documents=documents,
        role_counts=dict(sorted(role_counts.items())),
        family_counts={name: int(family_counts.get(name, 0))
                       for name in ("cv", "vcv", "cvvc")},
        family_shares=shares,
        metadata_files=_discover_metadata(root),
        diagnostics=diagnostics,
        examples={key: tuple(value) for key, value in sorted(grouped.items())},
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def write_report(
    analysis: BankAnalysis,
    output: Path,
    include_entries: bool = False,
) -> None:
    """Write only to an explicit path outside the source bank."""
    output = Path(output).expanduser().resolve()
    if _is_within(output, analysis.source_root):
        raise ValueError(
            "refusing to write an analysis report inside the source voicebank"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(analysis.to_dict(include_entries), ensure_ascii=False,
                   indent=2) + "\n",
        encoding="utf-8",
    )


def format_summary(analysis: BankAnalysis) -> str:
    entries = analysis.entries
    errors = sum(item.severity == "error" for item in analysis.diagnostics)
    warnings = sum(item.severity == "warning" for item in analysis.diagnostics)
    encodings = Counter(doc.source.encoding for doc in analysis.documents)
    lines = [
        "Japanese UTAU bank analysis",
        f"Source: {analysis.source_root}",
        (f"Type: {analysis.bank_type} "
         f"(confidence {analysis.confidence:.2f})"),
        f"Reason: {analysis.classification_reason}",
        f"OTO files: {len(analysis.documents)}",
        f"Aliases: {len(entries)}",
        "Families: " + ", ".join(
            f"{name}={analysis.family_counts.get(name, 0)}"
            for name in ("cv", "vcv", "cvvc")
        ),
        "Encodings: " + ", ".join(
            f"{name}={count}" for name, count in sorted(encodings.items())
        ),
        f"Diagnostics: {errors} error(s), {warnings} warning(s)",
        "Source files were read only; no voicebank file was changed.",
    ]
    if any(doc.source.ambiguous for doc in analysis.documents):
        lines.append("Encoding ambiguity is present; rerun with --encoding.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Japanese UTAU bank without modifying it. This phase "
            "reports OTO encoding, diagnostics, and CV/VCV/CVVC evidence; "
            "it does not build a Festival voice."
        )
    )
    parser.add_argument("source", type=Path,
                        help="voicebank directory or one oto.ini file")
    parser.add_argument("--encoding", default=None,
                        help="strict source encoding override, e.g. cp932")
    parser.add_argument("--bank-type", choices=BANK_TYPES, default="auto",
                        help="explicit configuration override")
    parser.add_argument("--alias-prefix", action="append", default=[],
                        help="declared alias prefix to remove; repeatable")
    parser.add_argument("--alias-suffix", action="append", default=[],
                        help="declared alias suffix to remove; repeatable")
    parser.add_argument("--json", action="store_true",
                        help="print JSON instead of the human summary")
    parser.add_argument("--include-entries", action="store_true",
                        help="include every parsed alias in JSON output")
    parser.add_argument("--output", type=Path, default=None,
                        help="write UTF-8 JSON outside the source voicebank")
    args = parser.parse_args(argv)

    try:
        analysis = analyze_bank(
            args.source,
            encoding_override=args.encoding,
            bank_type=args.bank_type,
            alias_prefixes=args.alias_prefix,
            alias_suffixes=args.alias_suffix,
        )
        if args.output:
            write_report(analysis, args.output, args.include_entries)
    except (FileNotFoundError, TextDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            analysis.to_dict(args.include_entries),
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print(format_summary(analysis))
        if args.output:
            print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
