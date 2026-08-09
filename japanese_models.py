"""Canonical Japanese linguistic structures for the Phase 1 frontend.

These structures describe linguistic intent only.  They do not contain UTAU
aliases, Festival units, durations, waveforms, or F0 trajectories.  Accent
nuclei are zero-based mora indexes within an accent phrase; Open JTalk's
one-based values are converted at the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, field
import json
from typing import Any, Mapping, Optional


PROVISIONAL_SCHEMA = "festvox.japanese_utterance.phase1-provisional"
ACCENT_STATES = {"accented", "unaccented", "unknown", "unavailable"}


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _validate_confidence(value: float, owner: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{owner} confidence must be between 0 and 1")


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class JapaneseFrontendDiagnostic(_Serializable):
    code: str
    message: str
    severity: str = "warning"
    action: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    frontend: Optional[str] = None
    confidence: Optional[float] = None
    raw_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"unsupported diagnostic severity: {self.severity}")
        if self.confidence is not None:
            _validate_confidence(self.confidence, "diagnostic")


@dataclass(frozen=True)
class JapanesePhone(_Serializable):
    index: int
    symbol: str
    raw_symbol: Optional[str] = None
    phone_type: str = "consonant"
    phrase_index: Optional[int] = None
    accent_phrase_index: Optional[int] = None
    mora_index: Optional[int] = None
    devoiced: Optional[bool] = None
    is_pause: bool = False
    is_silence: bool = False
    unknown: bool = False
    raw_label: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("phone index cannot be negative")
        if not self.symbol:
            raise ValueError("phone symbol cannot be empty")
        _validate_confidence(self.confidence, "phone")


@dataclass(frozen=True)
class JapaneseMora(_Serializable):
    index: int
    phrase_index: int
    accent_phrase_index: int
    surface: str
    reading: str
    phones: tuple[JapanesePhone, ...]
    consonant: Optional[str] = None
    vowel: Optional[str] = None
    special_mora: Optional[str] = None
    devoiced: Optional[bool] = None
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("mora index cannot be negative")
        if not self.phones:
            raise ValueError("a mora must contain at least one phone")
        _validate_confidence(self.confidence, "mora")

    @property
    def phone_indices(self) -> tuple[int, ...]:
        return tuple(phone.index for phone in self.phones)


@dataclass(frozen=True)
class JapaneseAccentPhrase(_Serializable):
    index: int
    phrase_index: int
    moras: tuple[JapaneseMora, ...]
    accent_state: str = "unknown"
    accent_nucleus: Optional[int] = None
    interrogative: bool = False
    boundary_strength: int = 1
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("accent-phrase index cannot be negative")
        if self.accent_state not in ACCENT_STATES:
            raise ValueError(f"unsupported accent state: {self.accent_state}")
        if self.accent_state == "accented":
            if self.accent_nucleus is None:
                raise ValueError("an accented phrase needs an accent nucleus")
            if not 0 <= self.accent_nucleus < len(self.moras):
                raise ValueError("accent nucleus is outside the accent phrase")
        elif self.accent_nucleus is not None:
            raise ValueError(
                "only an accented phrase may carry an accent nucleus"
            )
        if not 0 <= self.boundary_strength <= 3:
            raise ValueError("boundary strength must be between 0 and 3")
        _validate_confidence(self.confidence, "accent phrase")


@dataclass(frozen=True)
class JapanesePhrase(_Serializable):
    index: int
    surface: str
    normalized_reading: str
    accent_phrases: tuple[JapaneseAccentPhrase, ...]
    punctuation_after: str = ""
    boundary_strength: int = 3
    interrogative: bool = False
    phone_indices: tuple[int, ...] = ()
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("phrase index cannot be negative")
        if not 0 <= self.boundary_strength <= 3:
            raise ValueError("boundary strength must be between 0 and 3")
        _validate_confidence(self.confidence, "phrase")

    @property
    def moras(self) -> tuple[JapaneseMora, ...]:
        return tuple(
            mora
            for accent_phrase in self.accent_phrases
            for mora in accent_phrase.moras
        )


@dataclass(frozen=True)
class JapaneseUtterance(_Serializable):
    source_text: str
    normalized_reading: str
    phrases: tuple[JapanesePhrase, ...]
    phones: tuple[JapanesePhone, ...]
    diagnostics: tuple[JapaneseFrontendDiagnostic, ...]
    frontend_name: str
    frontend_version: Optional[str] = None
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frontend_name:
            raise ValueError("frontend name cannot be empty")
        _validate_confidence(self.confidence, "utterance")

    @property
    def moras(self) -> tuple[JapaneseMora, ...]:
        return tuple(mora for phrase in self.phrases for mora in phrase.moras)

    @property
    def accent_phrases(self) -> tuple[JapaneseAccentPhrase, ...]:
        return tuple(
            accent_phrase
            for phrase in self.phrases
            for accent_phrase in phrase.accent_phrases
        )

    def to_dict(self) -> dict[str, Any]:
        data = _serialize(self)
        return {"schema": PROVISIONAL_SCHEMA, **data}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )
