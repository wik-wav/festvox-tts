"""Deterministic, dependency-free English phone syllabification.

This module operates on the ARPAbet/ARPAsing phone stream already produced by
the active frontend.  It does not perform text-to-phone conversion and it does
not alter synthesis.  Inline phone input may contain a vowel supplied by an
integrated multilingual bank even when that phone is not part of English
ARPAbet. Those declared vowel phones remain nuclei for boundary inference.
Boundaries are inferred with the maximal-onset principle constrained by an
explicit English onset inventory. Optional lexical boundaries can be supplied
by future frontends when they know word spans.

Phone indexes and ``phone_end`` values use Python's usual zero-based, half-open
convention.  Pause phones remain in the utterance phone stream but never become
members of a syllable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 2
FRONTEND_NAME = "festvox-english-maximal-onset"
FRONTEND_VERSION = "2"

BOUNDARY_PHONES = frozenset({"pau", "sil", "sp", "#", "*"})
ENGLISH_VOWEL_PHONES = frozenset({
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw",
})
# Speech-bearing nuclei declared by the default integrated ARPAsing profile.
# ``inh`` is deliberately excluded even though its timing class is "vowel":
# an inhale can be stretched like a vowel but is not a syllable nucleus.
INTEGRATED_VOWEL_PHONES = frozenset({
    "a", "e", "i", "o", "u",
    "rr", "nn", "mm", "nng", "xn",
})
NON_SYLLABIC_TIMING_VOWELS = frozenset({"inh"})
VOWEL_PHONES = ENGLISH_VOWEL_PHONES | INTEGRATED_VOWEL_PHONES
SYLLABIC_CONSONANTS = frozenset({"el", "em", "en"})
NUCLEUS_PHONES = VOWEL_PHONES | SYLLABIC_CONSONANTS

CONSONANT_PHONES = frozenset({
    "b", "ch", "d", "dh", "dx", "f", "g", "hh", "jh", "k", "l",
    "m", "n", "ng", "p", "q", "r", "s", "sh", "t", "th", "v", "w",
    "y", "z", "zh",
})

# Native and established loanword onsets represented by the phone inventories
# used by Kal and generated ARPAsing voices.  Intervocalic clusters choose the
# longest matching suffix; material before it remains in the preceding coda.
_LEGAL_ONSETS = frozenset({
    (),
    *((phone,) for phone in CONSONANT_PHONES if phone not in {"ng", "dx", "q"}),
    ("p", "l"), ("p", "r"), ("p", "y"),
    ("b", "l"), ("b", "r"), ("b", "y"),
    ("t", "r"), ("t", "w"), ("t", "y"),
    ("d", "r"), ("d", "w"), ("d", "y"),
    ("k", "l"), ("k", "r"), ("k", "w"), ("k", "y"),
    ("g", "l"), ("g", "r"), ("g", "w"), ("g", "y"),
    ("f", "l"), ("f", "r"), ("f", "y"),
    ("v", "r"), ("v", "y"),
    ("th", "r"), ("th", "w"),
    ("sh", "r"), ("ch", "r"), ("jh", "r"),
    ("m", "y"), ("n", "y"), ("l", "y"), ("r", "y"), ("hh", "y"),
    ("s", "p"), ("s", "t"), ("s", "k"), ("s", "f"), ("s", "m"),
    ("s", "n"), ("s", "l"), ("s", "w"), ("s", "y"),
    ("s", "p", "l"), ("s", "p", "r"), ("s", "p", "y"),
    ("s", "t", "r"), ("s", "t", "y"),
    ("s", "k", "l"), ("s", "k", "r"), ("s", "k", "w"),
    ("s", "k", "y"),
})

_ALTERNATIVE_SUFFIX_RE = re.compile(r"__u\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class EnglishSyllableDiagnostic:
    code: str
    message: str
    phone_start: int
    phone_end: int
    severity: str = "warning"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "phone_start": int(self.phone_start),
            "phone_end": int(self.phone_end),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object],
    ) -> "EnglishSyllableDiagnostic":
        return cls(
            code=str(value.get("code") or "UNKNOWN"),
            message=str(value.get("message") or ""),
            phone_start=int(value.get("phone_start") or 0),
            phone_end=int(value.get("phone_end") or 0),
            severity=str(value.get("severity") or "warning"),
        )


@dataclass(frozen=True)
class EnglishSyllable:
    index: int
    phone_start: int
    phone_end: int
    phones: tuple[str, ...]
    onset: tuple[str, ...]
    nucleus: tuple[str, ...]
    coda: tuple[str, ...]
    stress: Optional[int]
    boundary_before: str
    confidence: float
    diagnostics: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return "-".join(self.phones)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "phone_start": int(self.phone_start),
            "phone_end": int(self.phone_end),
            "phones": list(self.phones),
            "onset": list(self.onset),
            "nucleus": list(self.nucleus),
            "coda": list(self.coda),
            "stress": self.stress,
            "boundary_before": self.boundary_before,
            "confidence": float(self.confidence),
            "diagnostics": list(self.diagnostics),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EnglishSyllable":
        stress = value.get("stress")
        return cls(
            index=int(value.get("index") or 0),
            phone_start=int(value.get("phone_start") or 0),
            phone_end=int(value.get("phone_end") or 0),
            phones=tuple(str(item) for item in value.get("phones") or ()),
            onset=tuple(str(item) for item in value.get("onset") or ()),
            nucleus=tuple(str(item) for item in value.get("nucleus") or ()),
            coda=tuple(str(item) for item in value.get("coda") or ()),
            stress=(None if stress is None else int(stress)),
            boundary_before=str(
                value.get("boundary_before") or "inferred"),
            confidence=float(value.get("confidence") or 0.0),
            diagnostics=tuple(
                str(item) for item in value.get("diagnostics") or ()),
        )


@dataclass(frozen=True)
class EnglishSyllabification:
    phones: tuple[str, ...]
    normalized_phones: tuple[str, ...]
    syllables: tuple[EnglishSyllable, ...]
    syllable_starts: tuple[int, ...]
    boundaries: tuple[int, ...]
    pause_indices: tuple[int, ...]
    diagnostics: tuple[EnglishSyllableDiagnostic, ...] = ()
    declared_nucleus_phones: tuple[str, ...] = ()
    frontend_name: str = FRONTEND_NAME
    frontend_version: str = FRONTEND_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "frontend_name": self.frontend_name,
            "frontend_version": self.frontend_version,
            "phones": list(self.phones),
            "normalized_phones": list(self.normalized_phones),
            "syllable_starts": list(self.syllable_starts),
            "boundaries": list(self.boundaries),
            "pause_indices": list(self.pause_indices),
            "syllables": [item.to_dict() for item in self.syllables],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "declared_nucleus_phones": list(self.declared_nucleus_phones),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object],
    ) -> "EnglishSyllabification":
        return cls(
            phones=tuple(str(item) for item in value.get("phones") or ()),
            normalized_phones=tuple(
                str(item) for item in value.get("normalized_phones") or ()),
            syllables=tuple(
                EnglishSyllable.from_dict(item)
                for item in value.get("syllables") or ()
                if isinstance(item, Mapping)
            ),
            syllable_starts=tuple(
                int(item) for item in value.get("syllable_starts") or ()),
            boundaries=tuple(
                int(item) for item in value.get("boundaries") or ()),
            pause_indices=tuple(
                int(item) for item in value.get("pause_indices") or ()),
            diagnostics=tuple(
                EnglishSyllableDiagnostic.from_dict(item)
                for item in value.get("diagnostics") or ()
                if isinstance(item, Mapping)
            ),
            declared_nucleus_phones=tuple(
                str(item)
                for item in value.get("declared_nucleus_phones") or ()
            ),
            frontend_name=str(value.get("frontend_name") or FRONTEND_NAME),
            frontend_version=str(
                value.get("frontend_version") or FRONTEND_VERSION),
            schema_version=int(value.get("schema_version") or SCHEMA_VERSION),
        )


def _normalized_inventory(values: Optional[Iterable[str]]) -> frozenset[str]:
    return frozenset(
        _ALTERNATIVE_SUFFIX_RE.sub("", str(value or "").strip())
        .rstrip("_").casefold()
        for value in (values or ())
        if str(value or "").strip()
    )


def profile_nucleus_phones(
    phone_types: Optional[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return speech nuclei explicitly declared by a generated voice profile."""

    nuclei = []
    for symbol, raw_type in (phone_types or {}).items():
        if isinstance(raw_type, Mapping):
            raw_type = raw_type.get("type")
        phone_type = str(raw_type or "").strip().casefold()
        normalized = next(iter(_normalized_inventory([str(symbol)])), "")
        if (
            phone_type in {"vowel", "syllabic", "syllabic_vowel", "nucleus"}
            and normalized
            and normalized not in NON_SYLLABIC_TIMING_VOWELS
            and normalized not in BOUNDARY_PHONES
        ):
            nuclei.append(normalized)
    return tuple(sorted(set(nuclei)))


def normalize_phone(
    phone: str,
    *,
    nucleus_phones: Optional[Iterable[str]] = None,
) -> tuple[str, Optional[int]]:
    """Return a matching symbol and optional lexical-stress digit."""

    value = _ALTERNATIVE_SUFFIX_RE.sub("", str(phone or "").strip())
    value = value.rstrip("_").casefold()
    nuclei = (
        NUCLEUS_PHONES | _normalized_inventory(nucleus_phones)
        if nucleus_phones is not None else NUCLEUS_PHONES
    )
    stress = None
    if len(value) > 1 and value[-1] in "012":
        stress = int(value[-1])
        candidate = value[:-1]
        if candidate in nuclei:
            value = candidate
        else:
            stress = None
    return value, stress


def _longest_legal_onset(cluster: Sequence[str]) -> int:
    maximum = min(3, len(cluster))
    for length in range(maximum, 0, -1):
        if tuple(cluster[-length:]) in _LEGAL_ONSETS:
            return length
    return 0


def _syllabify_chunk(
    original: Sequence[str],
    normalized: Sequence[str],
    stresses: Sequence[Optional[int]],
    start: int,
    end: int,
    boundary_before: str,
    syllable_offset: int,
    nucleus_phones: frozenset[str],
) -> tuple[list[EnglishSyllable], list[EnglishSyllableDiagnostic]]:
    diagnostics: list[EnglishSyllableDiagnostic] = []
    nuclei = [
        index for index in range(start, end)
        if normalized[index] in nucleus_phones
    ]
    unknown = [
        index for index in range(start, end)
        if normalized[index] not in nucleus_phones
        and normalized[index] not in CONSONANT_PHONES
    ]
    for index in unknown:
        diagnostics.append(EnglishSyllableDiagnostic(
            "UNKNOWN_PHONE",
            "Unknown English phone remains representable but lowers "
            "syllabification confidence: %s" % original[index],
            index,
            index + 1,
        ))

    if not nuclei:
        diagnostics.append(EnglishSyllableDiagnostic(
            "NO_NUCLEUS",
            "No English vowel or syllabic consonant was found in this "
            "non-pause span.",
            start,
            end,
        ))
        return [EnglishSyllable(
            index=syllable_offset,
            phone_start=start,
            phone_end=end,
            phones=tuple(original[start:end]),
            onset=tuple(original[start:end]),
            nucleus=(),
            coda=(),
            stress=None,
            boundary_before=boundary_before,
            confidence=0.2,
            diagnostics=("NO_NUCLEUS",),
        )], diagnostics

    starts = [start]
    for left_nucleus, right_nucleus in zip(nuclei, nuclei[1:]):
        cluster = normalized[left_nucleus + 1:right_nucleus]
        onset_length = _longest_legal_onset(cluster)
        starts.append(right_nucleus - onset_length)

    syllables: list[EnglishSyllable] = []
    for local_index, syllable_start in enumerate(starts):
        syllable_end = (
            starts[local_index + 1]
            if local_index + 1 < len(starts) else end
        )
        nucleus_index = next(
            index for index in nuclei
            if syllable_start <= index < syllable_end
        )
        onset_normalized = tuple(
            normalized[syllable_start:nucleus_index])
        flags: list[str] = []
        confidence = 1.0
        if local_index == 0 and onset_normalized not in _LEGAL_ONSETS:
            flags.append("NONCANONICAL_INITIAL_ONSET")
            confidence = min(confidence, 0.82)
        if any(index in unknown
               for index in range(syllable_start, syllable_end)):
            flags.append("UNKNOWN_PHONE")
            confidence = min(confidence, 0.55)
        syllables.append(EnglishSyllable(
            index=syllable_offset + local_index,
            phone_start=syllable_start,
            phone_end=syllable_end,
            phones=tuple(original[syllable_start:syllable_end]),
            onset=tuple(original[syllable_start:nucleus_index]),
            nucleus=(original[nucleus_index],),
            coda=tuple(original[nucleus_index + 1:syllable_end]),
            stress=stresses[nucleus_index],
            boundary_before=(
                boundary_before if local_index == 0 else "inferred"),
            confidence=confidence,
            diagnostics=tuple(flags),
        ))
    return syllables, diagnostics


def syllabify_english(
    phones: Sequence[str],
    *,
    word_boundaries: Optional[Iterable[int]] = None,
    extra_nucleus_phones: Optional[Iterable[str]] = None,
) -> EnglishSyllabification:
    """Infer syllable boundaries in an English phone stream.

    ``word_boundaries`` contains positions *between* phones, so ``3`` means a
    new lexical word begins at phone index 3.  Supplying those positions keeps
    maximal-onset inference inside words.  Without them, the parser operates
    over each pause-delimited phrase and reports that provenance explicitly.

    ``extra_nucleus_phones`` accepts vowel or syllabic symbols declared by a
    voice profile. It affects diagnostic syllabification only and never
    rewrites the phones sent to synthesis.
    """

    original = tuple(str(phone) for phone in phones)
    declared_nuclei = (
        _normalized_inventory(extra_nucleus_phones)
        - NON_SYLLABIC_TIMING_VOWELS
        - BOUNDARY_PHONES
    )
    nucleus_phones = NUCLEUS_PHONES | declared_nuclei
    normalized_rows = tuple(
        normalize_phone(phone, nucleus_phones=nucleus_phones)
        for phone in original
    )
    normalized = tuple(row[0] for row in normalized_rows)
    stresses = tuple(row[1] for row in normalized_rows)
    lexical = {
        int(value) for value in (word_boundaries or ())
        if 0 < int(value) < len(original)
    }
    pause_indices = tuple(
        index for index, phone in enumerate(normalized)
        if phone in BOUNDARY_PHONES
    )

    syllables: list[EnglishSyllable] = []
    diagnostics: list[EnglishSyllableDiagnostic] = []
    span_start: Optional[int] = None
    next_boundary = "utterance"

    def flush(span_end: int) -> None:
        nonlocal span_start, next_boundary
        if span_start is None or span_end <= span_start:
            span_start = None
            return
        rows, issues = _syllabify_chunk(
            original, normalized, stresses,
            span_start, span_end, next_boundary, len(syllables),
            nucleus_phones,
        )
        syllables.extend(rows)
        diagnostics.extend(issues)
        span_start = None

    for index, phone in enumerate(normalized):
        if phone in BOUNDARY_PHONES:
            flush(index)
            next_boundary = "pause"
            continue
        if index in lexical:
            flush(index)
            next_boundary = "word"
        if span_start is None:
            span_start = index
    flush(len(original))

    starts = tuple(item.phone_start for item in syllables)
    boundaries = tuple(starts[1:])
    if original and not syllables:
        diagnostics.append(EnglishSyllableDiagnostic(
            "ONLY_BOUNDARIES",
            "The phone stream contains only pause or boundary phones.",
            0,
            len(original),
            "info",
        ))
    return EnglishSyllabification(
        phones=original,
        normalized_phones=normalized,
        syllables=tuple(syllables),
        syllable_starts=starts,
        boundaries=boundaries,
        pause_indices=pause_indices,
        diagnostics=tuple(diagnostics),
        declared_nucleus_phones=tuple(sorted(declared_nuclei)),
    )


__all__ = [
    "BOUNDARY_PHONES",
    "CONSONANT_PHONES",
    "ENGLISH_VOWEL_PHONES",
    "EnglishSyllable",
    "EnglishSyllableDiagnostic",
    "EnglishSyllabification",
    "FRONTEND_NAME",
    "INTEGRATED_VOWEL_PHONES",
    "NON_SYLLABIC_TIMING_VOWELS",
    "NUCLEUS_PHONES",
    "SCHEMA_VERSION",
    "SYLLABIC_CONSONANTS",
    "VOWEL_PHONES",
    "normalize_phone",
    "profile_nucleus_phones",
    "syllabify_english",
]
