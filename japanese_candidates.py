"""Deterministic Phase 2 Japanese UTAU source-candidate compiler.

This module compiles OTO evidence into a lossless, inspectable candidate graph.
It does not slice waveforms, generate a Festival voice, or touch the production
English ARPAsing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import japanese_utau as ju
from japanese_kana_frontend import (
    convert_romaji,
    normalize_kana_reading,
    segment_kana_reading,
)
from japanese_profiles import (
    CANDIDATE_FAMILIES,
    JapaneseAliasOverride,
    JapaneseBankProfile,
    JapaneseSubbank,
    infer_bank_profile,
    load_profile,
    resolve_bank_context,
)
from voice_manifest import SourceRecordingBundle, VoiceConfiguration


CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_STATUS = "phase2-provisional"
CANDIDATE_ROLES = (
    "mora_cv",
    "phrase_start_cv",
    "vowel_blend",
    "vcv_mora",
    "vc_transition",
    "release",
    "special_mora",
    "silence",
    "breath",
    "extra",
    "unresolved",
)
PRIMARY_ROLES_BY_CONFIGURATION = {
    "cv": frozenset({
        "mora_cv", "phrase_start_cv", "vowel_blend", "special_mora",
    }),
    "vcv": frozenset({"vcv_mora", "phrase_start_cv", "special_mora"}),
    "cvvc": frozenset({
        "mora_cv", "phrase_start_cv", "vowel_blend", "vc_transition",
        "release", "special_mora",
    }),
}
STRICT_RUNTIME_EXCLUDED_FAMILIES = {
    # CVVC synthesis is assembled from ordinary CV onsets/morae and recorded
    # VC transitions. VCV aliases remain in the source graph for provenance,
    # but an explicit CVVC request must not make them runtime alternatives.
    "cvvc": frozenset({"vcv"}),
}

DEFAULT_COVERAGE_MORAS = (
    "a", "i", "u", "e", "o",
    "ka", "ki", "ku", "ke", "ko",
    "ga", "gi", "gu", "ge", "go",
    "sa", "shi", "su", "se", "so",
    "za", "ji", "zu", "ze", "zo",
    "ta", "chi", "tsu", "te", "to",
    "da", "de", "do",
    "na", "ni", "nu", "ne", "no",
    "ha", "hi", "fu", "he", "ho",
    "ba", "bi", "bu", "be", "bo",
    "pa", "pi", "pu", "pe", "po",
    "ma", "mi", "mu", "me", "mo",
    "ya", "yu", "yo",
    "ra", "ri", "ru", "re", "ro",
    "wa", "wo", "n",
)

_TRAILING_ALTERNATIVE = re.compile(r"^(?P<base>.*?\D)(?P<number>\d+)$")
_ROMAJI_TOKEN = re.compile(r"[A-Za-z][A-Za-z'_-]*$")
_VOWELS = {"a", "i", "u", "e", "o"}
_NASAL_CONTEXTS = {"n", "nn", "ng", "m", "my", "ny", "ngy", "N"}
_SMALL_VOWELS = {
    "\u3041": "a", "\u3043": "i", "\u3045": "u",
    "\u3047": "e", "\u3049": "o",
}
_SMALL_Y = {"\u3083": "a", "\u3085": "u", "\u3087": "o"}
_PALATALIZED = {
    "k": "ky", "g": "gy", "n": "ny", "h": "hy", "b": "by",
    "p": "py", "m": "my", "r": "ry", "f": "fy", "v": "vy",
}


def runtime_family_policy(profile: JapaneseBankProfile) -> dict[str, object]:
    """Describe which source families an explicit build may select.

    Automatic profile inference remains permissive. The strict rule applies
    only when the caller explicitly requests a configuration, which keeps the
    Phase 2 analyzer able to represent genuinely mixed banks without silently
    dropping any source row.
    """
    excluded = STRICT_RUNTIME_EXCLUDED_FAMILIES.get(
        profile.bank_configuration, frozenset()
    )
    return {
        "mode": "strict" if excluded else "permissive",
        "requested_configuration": profile.bank_configuration,
        "effective_configuration": profile.effective_configuration,
        "excluded_families": sorted(excluded),
        "cvvc_components": (
            ["cv", "cvvc"]
            if profile.bank_configuration == "cvvc" else []
        ),
        "excluded_entries_preserved_for_analysis": True,
    }


def runtime_family_allowed(
    profile: JapaneseBankProfile, family: str
) -> bool:
    excluded = STRICT_RUNTIME_EXCLUDED_FAMILIES.get(
        profile.bank_configuration, frozenset()
    )
    return family not in excluded


@dataclass(frozen=True)
class CandidateDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    action: Optional[str] = None
    source_path: Optional[str] = None
    line: Optional[int] = None
    candidate_id: Optional[str] = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        for key in ("action", "source_path", "line", "candidate_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.details:
            result["details"] = dict(sorted(self.details.items()))
        return result


@dataclass(frozen=True)
class CanonicalCandidateTarget:
    key: str
    kind: str
    mora_id: Optional[str] = None
    mora_reading: Optional[str] = None
    phones: tuple[str, ...] = ()
    left_context: Optional[str] = None
    right_context: Optional[str] = None
    special_mora: Optional[str] = None
    moraic_nasal_allophone: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "key": self.key,
            "kind": self.kind,
            "phones": list(self.phones),
        }
        for key in (
            "mora_id", "mora_reading", "left_context",
            "right_context", "special_mora", "moraic_nasal_allophone",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class CandidateTiming:
    offset: Optional[float]
    consonant: Optional[float]
    cutoff: Optional[float]
    preutterance: Optional[float]
    overlap: Optional[float]
    valid: bool

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "consonant": self.consonant,
            "cutoff": self.cutoff,
            "preutterance": self.preutterance,
            "overlap": self.overlap,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class CandidateSource:
    oto_path: str
    oto_sha256: str
    oto_encoding: str
    line: int
    byte_offset: int
    wav_raw: str
    wav_raw_sha256: str
    wav_path: Optional[str]
    wav_within_bank: bool
    wav_exists: bool
    alias_raw: str
    alias_nfc: str
    alias_match_key: str
    analysis_alias: str
    raw_line_sha256: str
    removed_prefixes: tuple[str, ...] = ()
    removed_suffixes: tuple[str, ...] = ()
    pitch_tags: tuple[str, ...] = ()
    alternative_numbers: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "oto_path": self.oto_path,
            "oto_sha256": self.oto_sha256,
            "oto_encoding": self.oto_encoding,
            "line": self.line,
            "byte_offset": self.byte_offset,
            "wav_raw": self.wav_raw,
            "wav_raw_sha256": self.wav_raw_sha256,
            "wav_path": self.wav_path,
            "wav_within_bank": self.wav_within_bank,
            "wav_exists": self.wav_exists,
            "alias_raw": self.alias_raw,
            "alias_nfc": self.alias_nfc,
            "alias_match_key": self.alias_match_key,
            "analysis_alias": self.analysis_alias,
            "raw_line_sha256": self.raw_line_sha256,
            "removed_prefixes": list(self.removed_prefixes),
            "removed_suffixes": list(self.removed_suffixes),
            "pitch_tags": list(self.pitch_tags),
            "alternative_numbers": list(self.alternative_numbers),
        }


@dataclass(frozen=True)
class JapaneseSourceCandidate:
    candidate_id: str
    role: str
    family: str
    subtype: str
    target: CanonicalCandidateTarget
    source: CandidateSource
    timing: CandidateTiming
    confidence: float
    reasons: tuple[str, ...]
    selectable: bool
    profile_override: bool
    override_key: Optional[str]
    selection_cost: float
    configuration_id: str = ""
    alias_namespace: str = ""
    canonical_phone_namespace: str = ""
    subbank_ids: tuple[str, ...] = ()
    diagnostics: tuple[CandidateDiagnostic, ...] = ()

    def to_dict(self) -> dict:
        result = {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "family": self.family,
            "subtype": self.subtype,
            "target": self.target.to_dict(),
            "source": self.source.to_dict(),
            "timing": self.timing.to_dict(),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "selectable": self.selectable,
            "profile_override": self.profile_override,
            "selection_cost": self.selection_cost,
            "configuration_id": self.configuration_id,
            "alias_namespace": self.alias_namespace,
            "canonical_phone_namespace": self.canonical_phone_namespace,
            "scoped_target_key": (
                f"{self.canonical_phone_namespace}:{self.target.key}"
            ),
            "subbank_ids": list(self.subbank_ids),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }
        if self.override_key is not None:
            result["override_key"] = self.override_key
        return result


@dataclass(frozen=True)
class CandidateGroup:
    target_key: str
    role: str
    candidate_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "role": self.role,
            "candidate_ids": list(self.candidate_ids),
            "alternative_count": len(self.candidate_ids),
        }


@dataclass(frozen=True)
class CoverageReport:
    source_entry_count: int
    candidate_count: int
    all_entries_traceable: bool
    role_counts: dict[str, int]
    family_counts: dict[str, int]
    selectable_counts: dict[str, int]
    unresolved_count: int
    unresolved_rate: float
    unresolved_candidate_ids: tuple[str, ...]
    invalid_timing_count: int
    missing_wav_count: int
    outside_bank_wav_count: int
    candidate_group_count: int
    alternative_group_count: int
    mora_coverage: dict[str, dict[str, int]]
    missing_core_mora_any: tuple[str, ...]
    missing_core_plain_cv: tuple[str, ...]
    missing_core_phrase_start: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "source_entry_count": self.source_entry_count,
            "candidate_count": self.candidate_count,
            "all_entries_traceable": self.all_entries_traceable,
            "role_counts": dict(self.role_counts),
            "family_counts": dict(self.family_counts),
            "selectable_counts": dict(self.selectable_counts),
            "unresolved_count": self.unresolved_count,
            "unresolved_rate": self.unresolved_rate,
            "unresolved_candidate_ids": list(
                self.unresolved_candidate_ids
            ),
            "invalid_timing_count": self.invalid_timing_count,
            "missing_wav_count": self.missing_wav_count,
            "outside_bank_wav_count": self.outside_bank_wav_count,
            "candidate_group_count": self.candidate_group_count,
            "alternative_group_count": self.alternative_group_count,
            "mora_coverage": {
                key: dict(self.mora_coverage[key])
                for key in sorted(self.mora_coverage)
            },
            "missing_core_mora_any": list(self.missing_core_mora_any),
            "missing_core_plain_cv": list(self.missing_core_plain_cv),
            "missing_core_phrase_start": list(
                self.missing_core_phrase_start
            ),
        }


@dataclass(frozen=True)
class JapaneseCandidateGraph:
    profile: JapaneseBankProfile
    source: dict[str, object]
    source_bundle: SourceRecordingBundle
    voice_configuration: Optional[VoiceConfiguration]
    candidates: tuple[JapaneseSourceCandidate, ...]
    groups: tuple[CandidateGroup, ...]
    coverage: CoverageReport
    diagnostics: tuple[CandidateDiagnostic, ...] = ()
    schema_version: int = CANDIDATE_SCHEMA_VERSION
    schema_status: str = CANDIDATE_SCHEMA_STATUS
    _bank_root: Optional[Path] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "kind": "japanese_utau_candidate_graph",
            "language": "ja",
            "profile": self.profile.to_dict(),
            "runtime_family_policy": runtime_family_policy(self.profile),
            "source": dict(self.source),
            "source_recording_bundle": self.source_bundle.to_dict(),
            "voice_configuration": (
                self.voice_configuration.to_dict()
                if self.voice_configuration is not None else None
            ),
            "candidates": [item.to_dict() for item in self.candidates],
            "groups": [item.to_dict() for item in self.groups],
            "coverage": self.coverage.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _MoraTarget:
    mora_id: str
    reading: str
    phones: tuple[str, ...]
    vowel: Optional[str]
    special_mora: Optional[str]
    moraic_nasal_allophone: Optional[str] = None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix() or "."


def _find_oto_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    files = sorted(
        (
            path for path in source.rglob("*")
            if path.is_file() and path.name.casefold() == "oto.ini"
        ),
        key=lambda path: path.relative_to(source).as_posix().casefold(),
    )
    if not files:
        raise FileNotFoundError(f"no oto.ini files found under {source}")
    return tuple(files)


def _validate_explicit_oto_files(
    values: Sequence[Path], bank_root: Path
) -> tuple[Path, ...]:
    """Retain an explicit OTO scope without rediscovering sibling configs."""
    selected = []
    root = bank_root.resolve()
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.name.casefold() != "oto.ini":
            raise FileNotFoundError(f"selected Japanese OTO not found: {path}")
        if not _is_within(path, root):
            raise ValueError(
                f"selected Japanese OTO is outside the source bank: {path}"
            )
        selected.append(path)
    result = tuple(sorted(
        dict.fromkeys(selected),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    ))
    if not result:
        raise ValueError("explicit Japanese OTO scope must not be empty")
    return result


def _encoding_for(
    path: Path,
    bank_root: Path,
    source_scope: Path,
    profile: JapaneseBankProfile,
) -> Optional[str]:
    keys = [_relative(path, bank_root)]
    scope_root = source_scope.parent if source_scope.is_file() else source_scope
    if _is_within(path, scope_root):
        keys.append(_relative(path, scope_root))
    keys.append(path.name)
    for key in keys:
        normalized = key.replace("\\", "/")
        if normalized in profile.encoding_overrides:
            return profile.encoding_overrides[normalized]
    return profile.default_encoding


def _allophone_aliases(
    profile: Optional[JapaneseBankProfile], attribute: str
) -> dict[str, str]:
    if profile is None:
        return {}
    return {
        alias: allophone_id
        for allophone_id, rule in profile.moraic_nasal_allophones.items()
        for alias in getattr(rule, attribute)
    }


def _allophone_for_alias(
    profile: Optional[JapaneseBankProfile], attribute: str, token: str
) -> Optional[str]:
    return _allophone_aliases(profile, attribute).get(token.strip())


def _protected_allophone_tokens(
    profile: Optional[JapaneseBankProfile],
) -> frozenset[str]:
    return frozenset(
        set(_allophone_aliases(profile, "mora_aliases"))
        | set(_allophone_aliases(profile, "context_aliases"))
    )


def _strip_numbered_alternatives(
    alias: str, protected_tokens: Sequence[str] = ()
) -> tuple[str, tuple[int, ...]]:
    numbers: list[int] = []
    protected = set(protected_tokens)
    parts = alias.split()
    cleaned = []
    for part in parts:
        normalized_part = part.rstrip("_")
        if normalized_part in protected:
            cleaned.append(normalized_part)
            continue
        match = _TRAILING_ALTERNATIVE.fullmatch(normalized_part)
        if match and match.group("base"):
            cleaned.append(match.group("base"))
            numbers.append(int(match.group("number")))
        else:
            cleaned.append(normalized_part)
    return " ".join(cleaned), tuple(numbers)


def _parse_mora(
    token: str, profile: Optional[JapaneseBankProfile] = None
) -> Optional[_MoraTarget]:
    token = token.strip()
    if not token:
        return None
    allophone = (
        _allophone_for_alias(profile, "mora_aliases", token)
        or _allophone_for_alias(profile, "context_aliases", token)
    )
    if allophone is not None:
        return _MoraTarget(
            mora_id="moraic_nasal",
            reading="\u3093",
            phones=("N",),
            vowel=None,
            special_mora="moraic_nasal",
            moraic_nasal_allophone=allophone,
        )
    if token in _NASAL_CONTEXTS or token.casefold() in {
        "n", "nn", "ng", "m"
    }:
        return _MoraTarget(
            mora_id="moraic_nasal",
            reading="\u3093",
            phones=("N",),
            vowel=None,
            special_mora="moraic_nasal",
            moraic_nasal_allophone=None,
        )
    reading = token
    if _ROMAJI_TOKEN.fullmatch(token):
        converted = convert_romaji(token)
        if any(item.code == "unsupported_romaji" for item in converted.diagnostics):
            return None
        reading = converted.reading
    moras, diagnostics = segment_kana_reading(reading)
    if diagnostics or len(moras) != 1 or moras[0].unknown:
        return _parse_extended_kana_mora(reading)
    mora = moras[0]
    mora_id = "+".join(mora.phones)
    if mora.special_mora:
        mora_id = mora.special_mora
    return _MoraTarget(
        mora_id=mora_id,
        reading=mora.reading,
        phones=mora.phones,
        vowel=mora.vowel,
        special_mora=mora.special_mora,
    )


def _parse_extended_kana_mora(token: str) -> Optional[_MoraTarget]:
    """Interpret conservative small-kana OTO spellings absent from Phase 1."""
    normalized = normalize_kana_reading(token)
    if len(normalized) != 2:
        return None
    base_moras, base_diagnostics = segment_kana_reading(normalized[0])
    if base_diagnostics or len(base_moras) != 1 or base_moras[0].unknown:
        return None
    base = base_moras[0]
    modifier = normalized[1]
    if modifier in _SMALL_Y:
        vowel = _SMALL_Y[modifier]
        consonant = _PALATALIZED.get(
            base.consonant or "", (base.consonant or "") + "y"
        )
    elif modifier in _SMALL_VOWELS:
        vowel = _SMALL_VOWELS[modifier]
        consonant = base.consonant
        if base.vowel == "i" and vowel != "i" and consonant:
            consonant = _PALATALIZED.get(consonant, consonant)
    else:
        return None
    phones = (consonant, vowel) if consonant else (vowel,)
    return _MoraTarget(
        mora_id="+".join(phones),
        reading=normalized,
        phones=phones,
        vowel=vowel,
        special_mora=None,
    )


def _canonical_context(
    token: str, profile: Optional[JapaneseBankProfile] = None
) -> Optional[str]:
    token = token.strip()
    if not token:
        return None
    if token in _VOWELS:
        return token
    if token in _NASAL_CONTEXTS or token.casefold() in {
        "n", "nn", "ng", "m", "my", "ny", "ngy"
    }:
        return "N"
    if (
        _allophone_for_alias(profile, "context_aliases", token)
        or _allophone_for_alias(profile, "mora_aliases", token)
    ):
        return "N"
    mora = _parse_mora(token, profile)
    if mora is not None:
        if mora.special_mora == "moraic_nasal":
            return "N"
        return mora.vowel
    return None


def _canonical_consonant(token: str, *, release: bool = False) -> Optional[str]:
    value = token.strip()
    if release:
        value = value.rstrip("-")
        if not value:
            return "sil"
    if not value or not re.fullmatch(r"[A-Za-z][A-Za-z']*", value):
        return None
    return value.casefold()


def _is_explicit_cvvc_transition_token(
    token: str, profile: JapaneseBankProfile
) -> bool:
    """Recognize the right side of an explicit CVVC VC/VV alias.

    A right-hand kana or multi-phone romaji mora (``a か`` / ``a ka``) is
    VCV material. A single vowel is a recorded VV transition, while a romaji
    token without a vowel letter is a consonant context such as ``k``, ``sh``,
    or ``n``. Bank-defined moraic-nasal aliases remain morae rather than being
    guessed into consonants.
    """
    value = token.strip()
    if not value or value in _allophone_aliases(profile, "mora_aliases"):
        return False
    folded = value.casefold()
    if folded in _VOWELS:
        return True
    return bool(
        _ROMAJI_TOKEN.fullmatch(value)
        and not any(vowel in folded for vowel in _VOWELS)
    )


def _unresolved_target(alias: str) -> CanonicalCandidateTarget:
    digest = hashlib.sha256(alias.encode("utf-8")).hexdigest()[:20]
    return CanonicalCandidateTarget(
        key=f"unresolved:{digest}",
        kind="unresolved",
    )


def _target_for_role(
    role: str,
    alias: str,
    override: Optional[JapaneseAliasOverride],
    profile: Optional[JapaneseBankProfile] = None,
) -> Optional[CanonicalCandidateTarget]:
    parts = alias.split()
    if role in {
        "mora_cv", "phrase_start_cv", "vowel_blend", "vcv_mora"
    }:
        mora_token = override.mora if override and override.mora else (
            parts[-1] if parts else ""
        )
        mora = _parse_mora(mora_token, profile)
        if mora is None:
            return None
        if role == "vcv_mora":
            left_raw = (
                override.left_context
                if override and override.left_context is not None
                else (parts[0] if len(parts) >= 2 else "")
            )
            left = _canonical_context(left_raw, profile)
            if left is None:
                return None
            key = f"vcv:{left}>{mora.mora_id}"
        elif role == "phrase_start_cv":
            left = None
            key = f"start:{mora.mora_id}"
        elif role == "vowel_blend":
            left = None
            key = f"blend:{mora.mora_id}"
        else:
            left = None
            key = f"cv:{mora.mora_id}"
        return CanonicalCandidateTarget(
            key=key,
            kind=role,
            mora_id=mora.mora_id,
            mora_reading=mora.reading,
            phones=mora.phones,
            left_context=left,
            special_mora=mora.special_mora,
            moraic_nasal_allophone=(
                mora.moraic_nasal_allophone
                or (
                    _allophone_for_alias(
                        profile, "context_aliases", parts[0]
                    ) if role == "vcv_mora" and parts else None
                )
            ),
        )

    if role in {"vc_transition", "release"}:
        left_raw = (
            override.left_context
            if override and override.left_context is not None
            else (parts[0] if parts else "")
        )
        right_raw = (
            override.right_context
            if override and override.right_context is not None
            else (parts[1] if len(parts) >= 2 else "")
        )
        left = _canonical_context(left_raw, profile)
        right = _canonical_consonant(
            right_raw, release=(role == "release")
        )
        right_mora = None
        if right is None and role == "vc_transition":
            right_mora = _parse_mora(right_raw, profile)
            if (
                right_mora is not None
                and right_mora.special_mora is not None
                and right_mora.phones
            ):
                right = right_mora.phones[-1]
        if left is None or right is None:
            return None
        prefix = "release" if role == "release" else "vc"
        return CanonicalCandidateTarget(
            key=f"{prefix}:{left}>{right}",
            kind=role,
            phones=((right,) if right_mora is not None else ()),
            left_context=left,
            right_context=right,
            special_mora=(
                right_mora.special_mora
                if right_mora is not None else None
            ),
            moraic_nasal_allophone=(
                (
                    right_mora.moraic_nasal_allophone
                    if right_mora is not None else None
                ) or _allophone_for_alias(
                    profile, "context_aliases", left_raw
                ) or _allophone_for_alias(
                    profile, "mora_aliases", left_raw
                )
            ),
        )

    if role == "special_mora":
        mora_token = override.mora if override and override.mora else alias
        mora = _parse_mora(mora_token, profile)
        special = mora.special_mora if mora else alias.casefold()
        phones = mora.phones if mora else (special,)
        return CanonicalCandidateTarget(
            key=f"special:{special}",
            kind=role,
            mora_id=(mora.mora_id if mora else special),
            mora_reading=(mora.reading if mora else alias),
            phones=phones,
            special_mora=special,
            moraic_nasal_allophone=(
                mora.moraic_nasal_allophone if mora else None
            ),
        )

    if role in {"silence", "breath", "extra"}:
        key_alias = alias.casefold().strip() or role
        digest = hashlib.sha256(key_alias.encode("utf-8")).hexdigest()[:12]
        return CanonicalCandidateTarget(
            key=f"{role}:{digest}",
            kind=role,
        )
    if role == "unresolved":
        return _unresolved_target(alias)
    return None


def _role_from_evidence(evidence: ju.AliasEvidence) -> str:
    if evidence.role == "cv":
        if evidence.subtype == "phrase_initial_cv":
            return "phrase_start_cv"
        if evidence.subtype == "vowel_blend":
            return "vowel_blend"
        return "mora_cv"
    if evidence.role == "vcv":
        return "vcv_mora"
    if evidence.role == "cvvc_vc":
        if "release" in evidence.subtype:
            return "release"
        return "vc_transition"
    if evidence.role in {"special_mora", "silence", "breath"}:
        return evidence.role
    return "unresolved"


def _secondary_role(
    alias: str, profile: Optional[JapaneseBankProfile] = None
) -> tuple[Optional[str], Optional[str], str]:
    """Recover only roles whose token structure remains unambiguous."""
    parts = alias.split()
    if len(parts) == 1 and parts[0].casefold().startswith("breath_"):
        return "breath", "extra", "named_breath"
    if len(parts) == 1 and _parse_mora(parts[0], profile) is not None:
        return "mora_cv", "cv", "secondary_mora"
    if len(parts) != 2:
        return None, None, ""
    left, right = parts
    if left == "-" and _parse_mora(right, profile) is not None:
        return "phrase_start_cv", "cv", "secondary_phrase_start"
    if left == "*" and _parse_mora(right, profile) is not None:
        return "vowel_blend", "cv", "secondary_vowel_blend"
    if left == "\u30fb" and _parse_mora(right, profile) is not None:
        return "phrase_start_cv", "vcv", "wildcard_vcv_start"
    context = _canonical_context(left, profile)
    if context is None:
        return None, None, ""
    if _parse_mora(right, profile) is not None:
        return "vcv_mora", "vcv", "secondary_vcv_mora"
    if right == "-":
        return "release", "cvvc", "secondary_release"
    if _canonical_consonant(right, release=right.endswith("-")) is not None:
        role = "release" if right.endswith("-") else "vc_transition"
        return role, "cvvc", "secondary_vc"
    return None, None, ""


def _family_for_role(role: str) -> str:
    if role in {"mora_cv", "phrase_start_cv", "vowel_blend"}:
        return "cv"
    if role == "vcv_mora":
        return "vcv"
    if role in {"vc_transition", "release"}:
        return "cvvc"
    return "extra"


def _find_override(
    profile: JapaneseBankProfile,
    entry: ju.OtoEntry,
    oto_relative: str,
) -> tuple[Optional[str], Optional[JapaneseAliasOverride]]:
    keys = (
        f"{oto_relative}:{entry.line_number}",
        entry.alias_raw,
        entry.normalization.canonical_alias,
        entry.normalization.match_key,
    )
    for key in keys:
        override = profile.alias_overrides.get(key)
        if override is not None:
            return key, override
    return None, None


def _safe_wav_source(
    entry: ju.OtoEntry,
    bank_root: Path,
) -> tuple[str, str, Optional[str], bool, bool]:
    raw = entry.wav_raw.strip()
    raw_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    portable = raw.replace("\\", "/")
    raw_path = Path(portable)
    if raw_path.is_absolute() or re.match(r"^[A-Za-z]:", portable):
        return "<absolute-source-path>", raw_digest, None, False, False
    resolved = (entry.source_path.parent / raw_path).resolve()
    if not _is_within(resolved, bank_root):
        return portable, raw_digest, None, False, False
    return (
        portable,
        raw_digest,
        _relative(resolved, bank_root),
        True,
        resolved.is_file(),
    )


def _candidate_id(
    configuration_id: str,
    alias_namespace: str,
    oto_relative: str,
    wav_raw: str,
    alias_nfc: str,
    occurrence: int,
) -> str:
    payload = {
        "configuration_id": configuration_id,
        "alias_namespace": alias_namespace,
        "oto_path": oto_relative,
        "wav_raw": wav_raw,
        "alias_nfc": alias_nfc,
        "occurrence": occurrence,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "jc_" + hashlib.sha256(encoded).hexdigest()[:24]


def _matching_subbanks(
    entry: ju.OtoEntry,
    profile: JapaneseBankProfile,
) -> tuple[str, ...]:
    result = []
    for subbank in profile.subbanks:
        prefix_match = (
            not subbank.prefix
            or subbank.prefix in entry.normalization.removed_prefixes
            or entry.alias_raw.startswith(subbank.prefix)
        )
        suffix_match = (
            not subbank.suffix
            or subbank.suffix in entry.normalization.removed_suffixes
            or entry.alias_raw.endswith(subbank.suffix)
        )
        if prefix_match and suffix_match:
            result.append(subbank.subbank_id)
    return tuple(result)


def _selection_cost(
    role: str,
    family: str,
    confidence: float,
    timing_valid: bool,
    selectable: bool,
    alternatives: tuple[int, ...],
    profile: JapaneseBankProfile,
    override: Optional[JapaneseAliasOverride],
    matching_subbanks: tuple[str, ...],
) -> float:
    cost = 1.0 - confidence
    if override is not None:
        cost -= 100.0 + float(override.priority)
    if not timing_valid:
        cost += 20.0
    if not selectable:
        cost += 1000.0
    effective = profile.effective_configuration
    primary_roles = PRIMARY_ROLES_BY_CONFIGURATION
    if (
        effective in primary_roles
        and role not in primary_roles[effective]
        and role not in {"silence", "breath", "extra", "unresolved"}
    ):
        # A selected alias system controls automatic assembly.  Material from
        # another family remains available as an explicit, visible fallback,
        # but it must not outrank a valid primary-role candidate merely due to
        # a context bonus.  CV is a primary component of CVVC, not a foreign
        # family: incoming CV and outgoing VC halves are both required.
        cost += 20.0
    if alternatives:
        cost += min(alternatives) * 0.01
    if profile.voice_color is not None:
        matching = {
            item.subbank_id: item for item in profile.subbanks
        }
        if not any(
            matching[item].color == profile.voice_color
            for item in matching_subbanks if item in matching
        ):
            cost += 2.0
    return round(cost, 6)


def _sanitize_diagnostic(
    diagnostic: ju.Diagnostic,
    bank_root: Path,
) -> CandidateDiagnostic:
    path = Path(diagnostic.path)
    source_path = (
        _relative(path, bank_root)
        if path.exists() and _is_within(path, bank_root)
        else path.name
    )
    return CandidateDiagnostic(
        code=diagnostic.code,
        message=diagnostic.message,
        severity=diagnostic.severity,
        source_path=source_path,
        line=diagnostic.line,
        details=(
            {"byte_offset": diagnostic.byte_offset}
            if diagnostic.byte_offset is not None else {}
        ),
    )


def _source_manifest(documents, bank_root: Path, profile) -> dict[str, object]:
    oto_files = [
        {
            "path": _relative(document.source.path, bank_root),
            "sha256": document.source.sha256,
            "byte_length": document.source.byte_length,
            "encoding": document.source.encoding,
        }
        for document in documents
    ]
    recording_paths = {}
    for document in documents:
        for entry in document.entries:
            _raw, _raw_digest, relative, within, exists = _safe_wav_source(
                entry, bank_root
            )
            if within and exists and relative:
                recording_paths[relative] = bank_root / Path(relative)
    recording_files = []
    for relative, path in sorted(recording_paths.items()):
        content = path.read_bytes()
        recording_files.append({
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_length": len(content),
        })
    fingerprint_payload = {
        "oto_files": oto_files,
        "recording_files": recording_files,
        "metadata_files": profile.metadata_files,
        "source_scope": profile.source_scope,
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "source_scope": profile.source_scope,
        "fingerprint_sha256": fingerprint,
        "oto_files": oto_files,
        "recording_files": recording_files,
        "metadata_files": {
            key: dict(profile.metadata_files[key])
            for key in sorted(profile.metadata_files)
        },
    }


def _requested_mora_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        mora = _parse_mora(value)
        result.append(mora.mora_id if mora else str(value))
    return tuple(dict.fromkeys(result))


def _coverage(
    source_entry_count: int,
    candidates: Sequence[JapaneseSourceCandidate],
    groups: Sequence[CandidateGroup],
    requested_moras: Sequence[str],
) -> CoverageReport:
    roles = Counter(item.role for item in candidates)
    families = Counter(item.family for item in candidates)
    selectable = Counter(
        item.role for item in candidates if item.selectable
    )
    unresolved = tuple(
        item.candidate_id for item in candidates
        if item.role == "unresolved"
    )
    mora_coverage: dict[str, Counter] = defaultdict(Counter)
    for item in candidates:
        if item.target.mora_id:
            mora_coverage[item.target.mora_id][item.role] += 1
    requested = _requested_mora_ids(requested_moras)
    any_roles = {
        "mora_cv", "phrase_start_cv", "vowel_blend", "vcv_mora"
    }
    missing_any = []
    missing_cv = []
    missing_start = []
    for mora_id in requested:
        counts = mora_coverage.get(mora_id, Counter())
        if not any(counts.get(role, 0) for role in any_roles):
            missing_any.append(mora_id)
        if not counts.get("mora_cv", 0):
            missing_cv.append(mora_id)
        if not counts.get("phrase_start_cv", 0):
            missing_start.append(mora_id)
    return CoverageReport(
        source_entry_count=source_entry_count,
        candidate_count=len(candidates),
        all_entries_traceable=source_entry_count == len(candidates),
        role_counts={key: roles.get(key, 0) for key in CANDIDATE_ROLES},
        family_counts={key: families.get(key, 0)
                       for key in CANDIDATE_FAMILIES},
        selectable_counts={key: selectable.get(key, 0)
                           for key in CANDIDATE_ROLES},
        unresolved_count=len(unresolved),
        unresolved_rate=round(
            len(unresolved) / source_entry_count if source_entry_count else 0.0,
            6,
        ),
        unresolved_candidate_ids=unresolved,
        invalid_timing_count=sum(
            not item.timing.valid for item in candidates
        ),
        missing_wav_count=sum(
            item.source.wav_within_bank and not item.source.wav_exists
            for item in candidates
        ),
        outside_bank_wav_count=sum(
            not item.source.wav_within_bank for item in candidates
        ),
        candidate_group_count=len(groups),
        alternative_group_count=sum(
            len(item.candidate_ids) > 1 for item in groups
        ),
        mora_coverage={
            key: dict(sorted(value.items()))
            for key, value in sorted(mora_coverage.items())
        },
        missing_core_mora_any=tuple(missing_any),
        missing_core_plain_cv=tuple(missing_cv),
        missing_core_phrase_start=tuple(missing_start),
    )


def compile_candidate_graph(
    source: Path,
    *,
    profile: Optional[JapaneseBankProfile] = None,
    requested_moras: Sequence[str] = DEFAULT_COVERAGE_MORAS,
    oto_files: Optional[Sequence[Path]] = None,
) -> JapaneseCandidateGraph:
    """Compile selected OTO entries without writing to the source bank.

    ``oto_files`` is an explicit configuration scope. When omitted, the
    established recursive discovery behavior remains unchanged.
    """
    context = resolve_bank_context(source)
    profile = profile or infer_bank_profile(context.source_scope)
    selected_oto_files = (
        _find_oto_files(context.source_scope)
        if oto_files is None
        else _validate_explicit_oto_files(oto_files, context.bank_root)
    )
    documents = tuple(
        ju.parse_oto_file(
            path,
            encoding_override=_encoding_for(
                path, context.bank_root, context.source_scope, profile
            ),
            alias_prefixes=profile.alias_prefixes,
            alias_suffixes=profile.alias_suffixes,
        )
        for path in selected_oto_files
    )
    diagnostics = [
        _sanitize_diagnostic(item, context.bank_root)
        for document in documents for item in document.diagnostics
    ]
    source_manifest = _source_manifest(
        documents, context.bank_root, profile
    )
    source_bundle = SourceRecordingBundle.from_source_manifest(
        source_manifest
    )
    observed_subbank_ids = tuple(sorted({
        subbank_id
        for document in documents for entry in document.entries
        for subbank_id in _matching_subbanks(entry, profile)
    }))
    selected_subbank_id = (
        observed_subbank_ids[0]
        if len(observed_subbank_ids) == 1 else None
    )
    explicit = profile.bank_configuration in {"cv", "vcv", "cvvc"}
    effective = profile.effective_configuration
    voice_configuration = None
    if effective in {"cv", "vcv", "cvvc"}:
        policy = {
            "source_scope": profile.source_scope,
            "alias_prefixes": list(profile.alias_prefixes),
            "alias_suffixes": list(profile.alias_suffixes),
            "enabled_families": list(profile.enabled_families),
            "alias_overrides": {
                key: profile.alias_overrides[key].to_dict()
                for key in sorted(profile.alias_overrides)
            },
            "moraic_nasal_allophones": {
                key: profile.moraic_nasal_allophones[key].to_dict()
                for key in sorted(profile.moraic_nasal_allophones)
            },
            "observed_subbank_ids": list(observed_subbank_ids),
        }
        voice_configuration = VoiceConfiguration.japanese(
            source_bundle_id=source_bundle.source_bundle_id,
            bank_type=effective,
            configuration_policy=policy,
            selected_subbank_id=selected_subbank_id,
            selected_voice_color=profile.voice_color,
            selection_status=("explicit" if explicit
                              else "analysis-proposal"),
        )
        profile = replace(
            profile,
            source_bundle_id=source_bundle.source_bundle_id,
            configuration_id=voice_configuration.configuration_id,
            alias_system=voice_configuration.alias_system,
            alias_namespace=voice_configuration.alias_namespace,
            canonical_phone_namespace=(
                voice_configuration.canonical_phone_namespace
            ),
        )
    else:
        analysis_payload = {
            "source_bundle_id": source_bundle.source_bundle_id,
            "effective_configuration": effective,
            "source_scope": profile.source_scope,
        }
        analysis_digest = hashlib.sha256(json.dumps(
            analysis_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:24]
        configuration_id = "analysis_" + analysis_digest
        profile = replace(
            profile,
            source_bundle_id=source_bundle.source_bundle_id,
            configuration_id=configuration_id,
            alias_system="utau-japanese-analysis-v1",
            alias_namespace=configuration_id + ".aliases",
            canonical_phone_namespace=configuration_id + ".ja.phones",
        )
    candidates: list[JapaneseSourceCandidate] = []
    occurrence_counts: Counter = Counter()

    for document in documents:
        oto_relative = _relative(document.source.path, context.bank_root)
        for entry in document.entries:
            cleaned_alias, extra_alternatives = _strip_numbered_alternatives(
                entry.normalization.analysis_alias,
                _protected_allophone_tokens(profile),
            )
            evidence = entry.evidence
            if cleaned_alias != entry.normalization.analysis_alias:
                evidence = ju.classify_alias(replace(
                    entry.normalization,
                    analysis_alias=cleaned_alias,
                ))
            override_key, override = _find_override(
                profile, entry, oto_relative
            )
            family_hint = None
            secondary_subtype = ""
            configuration_reinterpretation = ""
            role = override.role if override else _role_from_evidence(evidence)
            alias_parts = cleaned_alias.split()
            if (
                override is None
                and profile.effective_configuration == "cvvc"
                and role == "vcv_mora"
                and len(alias_parts) == 2
                and alias_parts[1] in _allophone_aliases(
                    profile, "context_aliases"
                )
                and alias_parts[1] not in _allophone_aliases(
                    profile, "mora_aliases"
                )
            ):
                # In CVVC OTOs, ``e n`` is a vowel-to-consonant VC.  Moraic
                # nasal recordings use the profile's kana mora aliases such
                # as ``e ん`` / ``e んm``.  The same strings are not globally
                # standardized, so this reinterpretation is profile-scoped.
                role = "vc_transition"
                family_hint = "cvvc"
                secondary_subtype = "profile_nasal_context_vc"
                configuration_reinterpretation = (
                    "profile-defined nasal context interpreted as CVVC VC"
                )
            if override is None and role == "unresolved":
                recovered_role, family_hint, secondary_subtype = (
                    _secondary_role(cleaned_alias, profile)
                )
                if recovered_role is not None:
                    role = recovered_role
            if (
                override is None
                and not configuration_reinterpretation
                and profile.bank_configuration == "cvvc"
                and role == "vcv_mora"
                and len(alias_parts) == 2
            ):
                special_target = _parse_mora(alias_parts[1], profile)
                if _is_explicit_cvvc_transition_token(
                    alias_parts[1], profile
                ):
                    role = "vc_transition"
                    family_hint = "cvvc"
                    secondary_subtype = (
                        "explicit_cvvc_vv_transition"
                        if alias_parts[1].casefold() in _VOWELS
                        else "explicit_cvvc_vc_transition"
                    )
                    configuration_reinterpretation = (
                        "explicit CVVC profile interpreted the two-phone alias "
                        "as a recorded VC/VV transition"
                    )
                elif (
                    special_target is not None
                    and special_target.special_mora is not None
                ):
                    role = "vc_transition"
                    family_hint = "cvvc"
                    secondary_subtype = (
                        "explicit_cvvc_special_mora_transition"
                    )
                    configuration_reinterpretation = (
                        "explicit CVVC profile interpreted the special-mora "
                        "row as an incoming VC transition"
                    )
            if role not in CANDIDATE_ROLES:
                role = "unresolved"
            target = _target_for_role(
                role, cleaned_alias, override, profile
            )
            candidate_diagnostics: list[CandidateDiagnostic] = []
            if configuration_reinterpretation:
                candidate_diagnostics.append(CandidateDiagnostic(
                    code="alias_reinterpreted_for_explicit_cvvc",
                    message=configuration_reinterpretation,
                    severity="info",
                    source_path=oto_relative,
                    line=entry.line_number,
                    details={
                        "alias": entry.alias_raw,
                        "selected_role": role,
                        "selected_family": family_hint or _family_for_role(role),
                    },
                ))
            parts = cleaned_alias.split()
            ambiguous_nasal_tokens = [
                token for token in parts
                if (
                    token.casefold() in {"m", "nn", "ng"}
                    or (token.startswith("\u3093") and token != "\u3093")
                )
                and token not in _protected_allophone_tokens(profile)
            ]
            if ambiguous_nasal_tokens:
                candidate_diagnostics.append(CandidateDiagnostic(
                    code="moraic_nasal_allophone_unconfigured",
                    message=(
                        "This source appears to name a bank-specific moraic "
                        "nasal allophone, but the profile does not define its "
                        "meaning. The source remains preserved."
                    ),
                    action=(
                        "Define mora_aliases, context_aliases, and following_"
                        "phones under moraic_nasal_allophones in the Japanese "
                        "bank profile."
                    ),
                    source_path=oto_relative,
                    line=entry.line_number,
                    details={"tokens": ambiguous_nasal_tokens},
                ))
            if target is None:
                candidate_diagnostics.append(CandidateDiagnostic(
                    code="candidate_target_unresolved",
                    message=(
                        "The alias role was recognizable, but its canonical "
                        "target could not be recovered without guessing."
                    ),
                    action=(
                        "Add an exact alias override with mora or context fields."
                    ),
                    source_path=oto_relative,
                    line=entry.line_number,
                    details={"alias": entry.alias_raw, "proposed_role": role},
                ))
                role = "unresolved"
                target = _unresolved_target(entry.normalization.match_key)
            family = override.family if override and override.family else (
                family_hint or _family_for_role(role)
            )
            if family not in CANDIDATE_FAMILIES:
                family = "extra"

            identity_key = (
                oto_relative,
                entry.wav_raw,
                entry.normalization.canonical_alias,
            )
            occurrence = occurrence_counts[identity_key]
            occurrence_counts[identity_key] += 1
            candidate_id = _candidate_id(
                profile.configuration_id,
                profile.alias_namespace,
                oto_relative,
                entry.wav_raw,
                entry.normalization.canonical_alias,
                occurrence,
            )
            wav_raw, wav_digest, wav_path, wav_within, wav_exists = (
                _safe_wav_source(entry, context.bank_root)
            )
            if not wav_within:
                candidate_diagnostics.append(CandidateDiagnostic(
                    code="source_wav_outside_bank",
                    message=(
                        "The OTO WAV field resolves outside the source bank and "
                        "will not be opened."
                    ),
                    severity="error",
                    action="Use a WAV path contained by the source voicebank.",
                    source_path=oto_relative,
                    line=entry.line_number,
                    candidate_id=candidate_id,
                    details={"wav_raw_sha256": wav_digest},
                ))
            alternatives = tuple(dict.fromkeys(
                entry.normalization.alternative_numbers + extra_alternatives
            ))
            source_item = CandidateSource(
                oto_path=oto_relative,
                oto_sha256=document.source.sha256,
                oto_encoding=document.source.encoding,
                line=entry.line_number,
                byte_offset=entry.byte_offset,
                wav_raw=wav_raw,
                wav_raw_sha256=wav_digest,
                wav_path=wav_path,
                wav_within_bank=wav_within,
                wav_exists=wav_exists,
                alias_raw=entry.alias_raw,
                alias_nfc=entry.normalization.canonical_alias,
                alias_match_key=entry.normalization.match_key,
                analysis_alias=cleaned_alias,
                raw_line_sha256=hashlib.sha256(
                    entry.raw_line.encode("utf-8")
                ).hexdigest(),
                removed_prefixes=entry.normalization.removed_prefixes,
                removed_suffixes=entry.normalization.removed_suffixes,
                pitch_tags=entry.normalization.pitch_tags,
                alternative_numbers=alternatives,
            )
            timing = CandidateTiming(
                offset=entry.offset,
                consonant=entry.consonant,
                cutoff=entry.cutoff,
                preutterance=entry.preutterance,
                overlap=entry.overlap,
                valid=entry.timing_valid,
            )
            subbank_ids = _matching_subbanks(entry, profile)
            family_allowed = runtime_family_allowed(profile, family)
            if not family_allowed:
                candidate_diagnostics.append(CandidateDiagnostic(
                    code="candidate_family_excluded_by_configuration",
                    message=(
                        "The source alias is preserved for analysis but is not "
                        "selectable by the explicitly requested bank type."
                    ),
                    severity="info",
                    action=(
                        "Build with --bank-type vcv if VCV recordings should "
                        "become runtime alternatives."
                    ),
                    source_path=oto_relative,
                    line=entry.line_number,
                    candidate_id=candidate_id,
                    details={
                        "candidate_family": family,
                        "requested_configuration": (
                            profile.bank_configuration
                        ),
                    },
                ))
            enabled = (
                family in profile.enabled_families
                and (override.enabled if override else True)
                and family_allowed
            )
            selectable = (
                enabled and timing.valid and wav_within
                and role != "unresolved"
                and not (
                    role == "breath"
                    and evidence.subtype == "context_to_breath"
                )
            )
            confidence = 1.0 if override else evidence.confidence
            reasons = (
                ("exact profile override",)
                if override else evidence.reasons
            )
            if configuration_reinterpretation:
                reasons = tuple(reasons) + (configuration_reinterpretation,)
            cost = _selection_cost(
                role, family, confidence, timing.valid, selectable,
                alternatives, profile, override, subbank_ids,
            )
            candidates.append(JapaneseSourceCandidate(
                candidate_id=candidate_id,
                role=role,
                family=family,
                subtype=(
                    "profile_override" if override
                    else secondary_subtype or evidence.subtype
                ),
                target=target,
                source=source_item,
                timing=timing,
                confidence=confidence,
                reasons=tuple(reasons),
                selectable=selectable,
                profile_override=override is not None,
                override_key=override_key,
                selection_cost=cost,
                configuration_id=profile.configuration_id,
                alias_namespace=profile.alias_namespace,
                canonical_phone_namespace=(
                    profile.canonical_phone_namespace
                ),
                subbank_ids=subbank_ids,
                diagnostics=tuple(candidate_diagnostics),
            ))

    candidate_lookup = {item.candidate_id: item for item in candidates}
    grouped: dict[str, list[str]] = defaultdict(list)
    role_by_target = {}
    for item in candidates:
        grouped[item.target.key].append(item.candidate_id)
        role_by_target[item.target.key] = item.role
    groups = tuple(
        CandidateGroup(
            target_key=target_key,
            role=role_by_target[target_key],
            candidate_ids=tuple(sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    candidate_lookup[candidate_id].selection_cost,
                    candidate_id,
                ),
            )),
        )
        for target_key, candidate_ids in sorted(grouped.items())
    )
    coverage = _coverage(
        sum(len(document.entries) for document in documents),
        candidates,
        groups,
        requested_moras,
    )
    excluded_count = sum(
        not runtime_family_allowed(profile, item.family)
        for item in candidates
    )
    if excluded_count:
        diagnostics.append(CandidateDiagnostic(
            code="runtime_family_policy_applied",
            message=(
                "Explicit bank-type policy kept nonmatching source aliases "
                "traceable but removed them from runtime selection."
            ),
            severity="info",
            details={
                "excluded_candidate_count": excluded_count,
                "requested_configuration": profile.bank_configuration,
                "excluded_families": sorted(
                    STRICT_RUNTIME_EXCLUDED_FAMILIES.get(
                        profile.bank_configuration, frozenset()
                    )
                ),
            },
        ))
    if not coverage.all_entries_traceable:
        raise RuntimeError(
            "candidate graph is internally inconsistent: source entries were lost"
        )
    return JapaneseCandidateGraph(
        profile=profile,
        source=source_manifest,
        source_bundle=source_bundle,
        voice_configuration=voice_configuration,
        candidates=tuple(candidates),
        groups=groups,
        coverage=coverage,
        diagnostics=tuple(diagnostics),
        _bank_root=context.bank_root,
    )


def candidate_metadata_bytes(graph: JapaneseCandidateGraph) -> bytes:
    return (
        json.dumps(
            graph.to_dict(), ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n"
    ).encode("utf-8")


def write_candidate_metadata(
    graph: JapaneseCandidateGraph,
    output: Path,
) -> None:
    target = Path(output).expanduser().resolve()
    if graph._bank_root is not None and _is_within(
        target, graph._bank_root
    ):
        raise ValueError(
            "refusing to write candidate metadata inside the source voicebank"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(candidate_metadata_bytes(graph))


def format_coverage_summary(graph: JapaneseCandidateGraph) -> str:
    coverage = graph.coverage
    roles = ", ".join(
        f"{key}={value}" for key, value in coverage.role_counts.items()
        if value
    )
    return "\n".join((
        "Japanese UTAU Phase 2 candidate graph",
        f"Source entries: {coverage.source_entry_count}",
        f"Candidates: {coverage.candidate_count}",
        f"Roles: {roles or 'none'}",
        (
            f"Unresolved: {coverage.unresolved_count} "
            f"({coverage.unresolved_rate:.2%})"
        ),
        f"Candidate groups: {coverage.candidate_group_count}",
        f"Alternative groups: {coverage.alternative_group_count}",
        "Source voicebank was read only.",
    ))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile a read-only Phase 2 Japanese UTAU candidate graph."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    profile = load_profile(args.profile) if args.profile else None
    graph = compile_candidate_graph(args.source, profile=profile)
    if args.output:
        write_candidate_metadata(graph, args.output)
    if args.json and not args.output:
        print(candidate_metadata_bytes(graph).decode("utf-8"), end="")
    else:
        print(format_coverage_summary(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
