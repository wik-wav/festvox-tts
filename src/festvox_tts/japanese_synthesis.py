"""Canonical Japanese utterance planning for Festival/UniSyn.

The planner owns the editable structural baseline: mora-first timing, OTO-safe
stretch diagnostics, and a speaker-relative Japanese F0 contour.  General
punctuation intonation and continuous user Pitch points are applied later by
the shared GUI/backend layers, with continuous points remaining final.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from japanese_festival import (
    JAPANESE_FESTIVAL_SCHEMA_VERSION,
    load_japanese_runtime_metadata,
)
from japanese_models import (
    JapaneseMora,
    JapaneseUtterance,
)
from japanese_duration import (
    DEFAULT_MORA_ALLOCATION_SECONDS,
    DEFAULT_MORA_ANCHOR_SECONDS,
    JapaneseDurationPrediction,
    build_duration_contexts,
    load_duration_priors,
    phone_class,
    predict_mora_durations,
)
from japanese_pitch import load_japanese_pitch_model, mora_pitch_contour
import pitch_domain as pitch_domain
from source_timing import profile_half_seconds, source_timing_profile


JAPANESE_SYNTHESIS_PLAN_VERSION = 3
_VOWELS = {"a", "i", "u", "e", "o"}
_STOPS = {"k", "g", "t", "d", "p", "b"}
_AFFRICATES = {"ch", "ts", "j"}
_FRICATIVES = {"s", "sh", "z", "f", "h"}
_SONORANTS = {"m", "n", "r", "w", "y", "N"}
_PITCH_MIN_HZ = 50.0
_PITCH_MAX_HZ = 500.0
_STRUCTURAL_CONTOUR_MODEL = "japanese_speaker_relative_log_f0_v1"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class JapaneseSynthesisDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    mora_index: Optional[int] = None
    candidate_id: Optional[str] = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.mora_index is not None:
            result["mora_index"] = self.mora_index
        if self.candidate_id is not None:
            result["candidate_id"] = self.candidate_id
        if self.details:
            result["details"] = dict(sorted(self.details.items()))
        return result


@dataclass(frozen=True)
class JapanesePlannedSegment:
    index: int
    phone: str
    duration: float
    phrase_index: Optional[int] = None
    accent_phrase_index: Optional[int] = None
    mora_index: Optional[int] = None
    source_phone_index: Optional[int] = None
    pause_role: Optional[str] = None
    timing_role: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "index": self.index,
            "phone": self.phone,
            "duration": self.duration,
        }
        for name in (
            "phrase_index",
            "accent_phrase_index",
            "mora_index",
            "source_phone_index",
            "pause_role",
            "timing_role",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True)
class JapaneseF0Target:
    time: float
    log_f0: float
    baseline_log_f0: float
    phrase_index: int
    accent_phrase_index: int
    mora_index: int
    kind: str
    components_semitones: Mapping[str, float] = field(default_factory=dict)

    @property
    def hz(self) -> float:
        """Festival/PSOLA boundary representation."""
        return round(pitch_domain.log_f0_to_hz(self.log_f0), 3)

    @property
    def semitones_from_baseline(self) -> float:
        return 12.0 * (self.log_f0 - self.baseline_log_f0)

    def to_dict(self) -> dict[str, object]:
        return {
            "time": self.time,
            "log_f0": self.log_f0,
            "baseline_log_f0": self.baseline_log_f0,
            "semitones_from_baseline": self.semitones_from_baseline,
            "hz": self.hz,
            "phrase_index": self.phrase_index,
            "accent_phrase_index": self.accent_phrase_index,
            "mora_index": self.mora_index,
            "kind": self.kind,
            "components_semitones": dict(sorted(
                self.components_semitones.items())),
        }


@dataclass(frozen=True)
class JapanesePhoneTiming:
    segment_index: int
    phone: str
    predicted_duration: float
    source_reference_duration: Optional[float]
    source_profile_reference_duration: Optional[float]
    source_geometry_ratio: Optional[float]
    source_geometry_ratio_bounded: Optional[float]
    source_safe_min: float
    source_safe_max: float
    requested_stretch: Optional[float]
    final_duration: float
    constraint_source: str
    baseline_source: str = "legacy_class_fallback"
    duration_model: str = "legacy"
    duration_model_id: str = "legacy_mora_allocator_v2"
    context_log_ratio: float = 0.0
    context_effects: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_index": self.segment_index,
            "phone": self.phone,
            "predicted_duration": self.predicted_duration,
            "source_reference_duration": self.source_reference_duration,
            "source_profile_reference_duration": (
                self.source_profile_reference_duration
            ),
            "source_geometry_ratio": self.source_geometry_ratio,
            "source_geometry_ratio_bounded": (
                self.source_geometry_ratio_bounded
            ),
            "source_safe_min": self.source_safe_min,
            "source_safe_max": self.source_safe_max,
            "requested_stretch": self.requested_stretch,
            "final_duration": self.final_duration,
            "constraint_source": self.constraint_source,
            "baseline_source": self.baseline_source,
            "duration_model": self.duration_model,
            "duration_model_id": self.duration_model_id,
            "context_log_ratio": self.context_log_ratio,
            "context_effects": dict(sorted(self.context_effects.items())),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class JapaneseMoraTiming:
    mora_index: int
    predicted_mora_duration: float
    phone_allocation: tuple[JapanesePhoneTiming, ...]
    source_safe_min: float
    source_safe_max: float
    requested_stretch: Optional[float]
    final_duration: float
    phrase_final_lengthening: bool
    duration_model: str = "legacy"
    duration_model_id: str = "legacy_mora_allocator_v2"

    def to_dict(self) -> dict[str, object]:
        return {
            "mora_index": self.mora_index,
            "predicted_mora_duration": self.predicted_mora_duration,
            "phone_allocation": [item.to_dict() for item in
                                 self.phone_allocation],
            "source_safe_min": self.source_safe_min,
            "source_safe_max": self.source_safe_max,
            "requested_stretch": self.requested_stretch,
            "final_duration": self.final_duration,
            "phrase_final_lengthening": self.phrase_final_lengthening,
            "duration_model": self.duration_model,
            "duration_model_id": self.duration_model_id,
        }


@dataclass(frozen=True)
class JapaneseSynthesisPlan:
    source_text: str
    normalized_reading: str
    frontend_name: str
    segments: tuple[JapanesePlannedSegment, ...]
    f0_targets: tuple[JapaneseF0Target, ...]
    unit_overrides: Mapping[int, str]
    manual_candidate_overrides: Mapping[int, str]
    diagnostics: tuple[JapaneseSynthesisDiagnostic, ...]
    base_pitch_hz: float
    speed: float
    duration_model: str = "contextual"
    duration_model_id: str = "japanese_contextual_source_residual_v1"
    mora_timings: tuple[JapaneseMoraTiming, ...] = ()
    speaker_low_hz: float = _PITCH_MIN_HZ
    speaker_high_hz: float = _PITCH_MAX_HZ
    contour_model: str = _STRUCTURAL_CONTOUR_MODEL
    pitch_model_id: str = _STRUCTURAL_CONTOUR_MODEL
    schema_version: int = JAPANESE_SYNTHESIS_PLAN_VERSION
    runtime_schema_version: int = JAPANESE_FESTIVAL_SCHEMA_VERSION

    @property
    def phones(self) -> list[str]:
        return [segment.phone for segment in self.segments]

    @property
    def segment_durations(self) -> list[tuple[str, float]]:
        return [
            (segment.phone, segment.duration) for segment in self.segments
        ]

    @property
    def pitch_targets(self) -> list[tuple[float, float]]:
        return [(target.time, target.hz) for target in self.f0_targets]

    def backend_arguments(self) -> dict[str, object]:
        """Arguments for ``FestivalWSLBackend.synth_phones``."""
        return {
            "phones": self.phones,
            "seg_durs": self.segment_durations,
            "pitch_targets": self.pitch_targets,
            "unit_overrides": dict(self.unit_overrides),
            "text": self.source_text,
            "lang": "ja",
            "speed": 1.0,
            "pitch": self.base_pitch_hz,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_schema_version": self.runtime_schema_version,
            "kind": "japanese_explicit_synthesis_plan",
            "language": "ja",
            "source_text": self.source_text,
            "normalized_reading": self.normalized_reading,
            "frontend_name": self.frontend_name,
            "segments": [item.to_dict() for item in self.segments],
            "f0_targets": [item.to_dict() for item in self.f0_targets],
            "unit_overrides": {
                str(key): self.unit_overrides[key]
                for key in sorted(self.unit_overrides)
            },
            "manual_candidate_overrides": {
                str(key): self.manual_candidate_overrides[key]
                for key in sorted(self.manual_candidate_overrides)
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "base_pitch_hz": self.base_pitch_hz,
            "speaker_low_hz": self.speaker_low_hz,
            "speaker_high_hz": self.speaker_high_hz,
            "contour_model": self.contour_model,
            "pitch_model_id": self.pitch_model_id,
            "mora_timings": [item.to_dict() for item in self.mora_timings],
            "speed": self.speed,
            "duration_model": self.duration_model,
            "duration_model_id": self.duration_model_id,
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


def _consonant_budget(
    phone: str,
    *,
    contextual: bool = False,
    allocations: Mapping[str, float] | None = None,
) -> float:
    if contextual:
        # The allocator and predictor share one Japanese phone taxonomy.
        # In particular, ky/gy/py/by remain stops rather than being captured
        # by the generic ``endswith('y')`` sonorant fallback.
        values = dict(DEFAULT_MORA_ALLOCATION_SECONDS)
        values.update(dict(allocations or {}))
        return float(values.get(
            phone_class(phone), values.get("other", 0.040)
        ))
    if phone in _AFFRICATES:
        return 0.050
    if phone in _FRICATIVES:
        return 0.050
    if phone in _STOPS:
        return 0.027 if phone in {"k", "t", "p"} else 0.030
    if phone in _SONORANTS or phone.endswith("y"):
        return 0.036
    return 0.040


def _mora_phone_durations(
    mora: JapaneseMora,
    speed: float,
    *,
    phrase_final: bool,
    phone_symbols: Sequence[str] | None = None,
    timing_multipliers: Mapping[str, object] | None = None,
    contextual: bool = False,
    mora_allocation_seconds: Mapping[str, float] | None = None,
    mora_anchor_seconds: Mapping[str, float] | None = None,
) -> tuple[tuple[str, float], ...]:
    """Allocate one linguistic mora before considering source geometry."""
    phones = tuple(phone_symbols) if phone_symbols is not None else tuple(
        phone.symbol for phone in mora.phones
        if phone.symbol not in {"pau", "sil"}
    )
    if not phones:
        return ()
    contextual_values = dict(DEFAULT_MORA_ALLOCATION_SECONDS)
    contextual_values.update(dict(mora_allocation_seconds or {}))
    contextual_anchors = dict(DEFAULT_MORA_ANCHOR_SECONDS)
    contextual_anchors.update(dict(mora_anchor_seconds or {}))
    if mora.special_mora == "geminate" or phones == ("cl",):
        total = (contextual_anchors["geminate_closure"]
                 if contextual else 0.082)
    elif mora.special_mora == "moraic_nasal" or phones == ("N",):
        total = (contextual_anchors["moraic_nasal"]
                 if contextual else 0.088)
    elif mora.special_mora == "long_vowel":
        total = contextual_anchors["long_vowel"] if contextual else 0.108
    elif mora.devoiced and any(phone in {"i", "u"} for phone in phones):
        total = (contextual_anchors["devoiced_high_vowel"]
                 if contextual else 0.092)
    elif any(phone in _VOWELS for phone in phones) and contextual:
        consonants = [phone for phone in phones if phone not in _VOWELS]
        # The total belongs to the source voice.  Kokoro-derived class values
        # below are only allocation weights inside that total.
        if not consonants:
            total = contextual_anchors["vowel_only"]
        elif any(phone in _FRICATIVES | _AFFRICATES
                 for phone in consonants):
            total = contextual_anchors["obstruent_cv"]
        else:
            total = contextual_anchors["cv"]
    elif any(phone in _VOWELS for phone in phones):
        consonants = [phone for phone in phones if phone not in _VOWELS]
        # Japanese timing is mora-first: adding an onset redistributes a
        # mora's budget rather than appending an English-sized consonant to a
        # full vowel. The 110/118 ms engineering baseline preserves the source
        # voice's slower register while using Kokoro-derived relative shares.
        total = 0.110 if not consonants else 0.118
        if any(phone in _FRICATIVES | _AFFRICATES for phone in consonants):
            total += 0.004
    else:
        total = 0.095
    if phrase_final and mora.special_mora != "geminate":
        total *= 1.12
    total /= speed

    vowels = [index for index, phone in enumerate(phones)
              if phone in _VOWELS]
    if len(phones) == 1 or not vowels:
        share = total / len(phones)
        durations = [share] * len(phones)
    elif contextual:
        # Normalize the corpus-derived class allocations inside the
        # source-speaker mora total.  This keeps stop, fricative, nasal, and
        # palatalized shares distinct without copying corpus milliseconds.
        weights = []
        for phone in phones:
            if phone in _VOWELS:
                weight = contextual_values.get("cv_vowel", 0.061)
            else:
                weight = _consonant_budget(
                    phone, contextual=True, allocations=contextual_values)
            weights.append(max(0.005, float(weight)))
        weight_total = sum(weights)
        durations = [total * weight / weight_total for weight in weights]
    else:
        consonant_indices = [index for index in range(len(phones))
                             if index not in vowels]
        consonant_values = {
            index: _consonant_budget(
                phones[index],
                contextual=contextual,
                allocations=contextual_values,
            ) / speed
            for index in consonant_indices
        }
        minimum_vowels = max(0.020, 0.042 / speed) * len(vowels)
        available = max(0.0, total - minimum_vowels)
        requested = sum(consonant_values.values())
        scale = min(1.0, available / requested) if requested else 1.0
        durations = [0.0] * len(phones)
        for index in consonant_indices:
            durations[index] = consonant_values[index] * scale
        remainder = max(0.010 * len(vowels), total - sum(durations))
        for index in vowels:
            durations[index] = remainder / len(vowels)
    multipliers = dict(timing_multipliers or {})
    weighted = []
    for phone, value in zip(phones, durations):
        try:
            multiplier = float(multipliers.get(phone, 1.0))
        except (TypeError, ValueError):
            multiplier = 1.0
        weighted.append(value * max(0.5, min(2.0, multiplier)))
    weighted_total = sum(weighted)
    if weighted_total > 0:
        durations = [value * total / weighted_total for value in weighted]
    rounded = [round(max(0.010, value), 6) for value in durations]
    correction = round(total - sum(rounded), 6)
    rounded[-1] = round(max(0.010, rounded[-1] + correction), 6)
    return tuple(zip(phones, rounded))


def _flexible_pause(
    boundary_strength: int,
    interrogative: bool,
    punctuation: str = "",
    phrase_pauses_ms: Mapping[str, object] | None = None,
) -> float:
    """Return the editable gap after the protected 80 ms unit-edge guard."""
    defaults = {"minor": 120.0, "major": 300.0, "sentence": 500.0}
    source = dict(phrase_pauses_ms or {})
    values = {}
    for key, default in defaults.items():
        try:
            values[key] = max(0.0, min(
                2000.0, float(source.get(key, default))))
        except (TypeError, ValueError):
            values[key] = default
    mark = str(punctuation or "")[-1:]
    if mark == "," or boundary_strength <= 1:
        level = "minor"
    elif mark in {":", ";"} or boundary_strength == 2:
        level = "major"
    else:
        level = "sentence"
    # Interrogative shape belongs to the ordinary Intonation editor; it does
    # not silently alter the linguistic pause duration here.
    _ = interrogative
    return max(0.0, values[level] / 1000.0 - 0.08)


def _load_runtime(
    value: Mapping[str, object] | Path | str | None,
) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return load_japanese_runtime_metadata(value)


def _speaker_pitch_context(
    runtime: Mapping[str, object], explicit_base: Optional[float], pitch_model
) -> tuple[float, float, float]:
    nested = dict(runtime.get("source_recording_bundle") or {})
    analysis = dict(
        runtime.get("speaker_pitch_analysis")
        or nested.get("speaker_pitch_analysis")
        or {}
    )
    base = float(
        explicit_base
        if explicit_base is not None
        else runtime.get("average_pitch_hz")
        or analysis.get("median_f0_hz")
        or 180.0
    )
    low = float(analysis.get("low_percentile_f0_hz") or base)
    high = float(analysis.get("high_percentile_f0_hz") or base)
    # Recorded FRQ bounds characterize the speaker, not the language's full
    # expressive range. Kokoro-calibrated headroom extends those bounds while
    # retaining the project speaker's own median as the immutable centre.
    low = min(low, pitch_domain.semitone_offset(
        base, -pitch_model.headroom_below_semitones))
    high = max(high, pitch_domain.semitone_offset(
        base, pitch_model.headroom_above_semitones))
    return base, max(_PITCH_MIN_HZ, low), min(_PITCH_MAX_HZ, high)


def _speaker_relative_pitch(
    base: float, semitones: float, low: float, high: float
) -> float:
    if semitones < 0.0:
        capacity = max(0.0, 12.0 * math.log2(base / max(low, 1.0)))
        semitones = -min(abs(semitones), capacity)
    else:
        capacity = max(0.0, 12.0 * math.log2(max(high, 1.0) / base))
        semitones = min(semitones, capacity)
    return pitch_domain.clamp_hz(
        pitch_domain.semitone_offset(base, semitones),
        _PITCH_MIN_HZ, _PITCH_MAX_HZ)


def _speaker_relative_log_f0(
    base: float, semitones: float, low: float, high: float
) -> float:
    return pitch_domain.hz_to_log_f0(
        _speaker_relative_pitch(base, semitones, low, high))


def _speaker_scaled_pitch(
    base: float, multiplier: float, low: float, high: float
) -> float:
    """Backward-compatible multiplier wrapper for optional refinement code."""
    semitones = 12.0 * math.log2(max(0.01, float(multiplier)))
    return _speaker_relative_pitch(base, semitones, low, high)


def _choice_source_half(
    runtime: Mapping[str, object],
    left: str,
    right: str,
    *,
    side: str,
    selected_left_name: Optional[str] = None,
) -> Optional[float]:
    alternatives = runtime.get("alternatives")
    if not isinstance(alternatives, Mapping):
        return None
    rows = alternatives.get(f"{left}-{right}")
    if not isinstance(rows, (list, tuple)):
        return None
    choices = [item for item in rows if isinstance(item, Mapping)]
    if selected_left_name:
        matching = [item for item in choices
                    if str(item.get("left_name") or "") ==
                    selected_left_name]
        if matching:
            choices = matching
    values: list[float] = []
    for choice in choices:
        source_slice = choice.get("source_slice")
        if not isinstance(source_slice, Mapping):
            source_slice = {
                "start": choice.get("start"),
                "phone_boundary": choice.get("mid"),
                "end": choice.get("end"),
            }
        try:
            start = float(source_slice["start"])
            middle = float(source_slice["phone_boundary"])
            end = float(source_slice["end"])
        except (KeyError, TypeError, ValueError):
            continue
        value = middle - start if side == "left" else end - middle
        if 0.001 <= value <= 5.0:
            values.append(value)
    # Automatic unit choice is context-sensitive and not known until Festival
    # runs. The shortest valid half is the non-inflating safety reference.
    # A manual left-unit override filters to that exact choice above.
    return min(values) if values else None


def _source_reference_duration(
    segments: Sequence[JapanesePlannedSegment],
    segment_index: int,
    runtime: Mapping[str, object],
    unit_overrides: Mapping[int, str],
) -> Optional[float]:
    halves: list[float] = []
    if segment_index > 0:
        previous = segments[segment_index - 1].phone
        current = segments[segment_index].phone
        value = _choice_source_half(
            runtime, previous, current, side="right",
            selected_left_name=unit_overrides.get(segment_index - 1),
        )
        if value is not None:
            halves.append(value)
    if segment_index + 1 < len(segments):
        current = segments[segment_index].phone
        following = segments[segment_index + 1].phone
        value = _choice_source_half(
            runtime, current, following, side="left",
            selected_left_name=unit_overrides.get(segment_index),
        )
        if value is not None:
            halves.append(value)
    return sum(halves) if halves else None


@dataclass(frozen=True)
class _SourceTimingReference:
    raw_duration: Optional[float]
    profile_duration: Optional[float]
    sides: tuple[str, ...]


def _source_timing_reference(
    segments: Sequence[JapanesePlannedSegment],
    segment_index: int,
    runtime: Mapping[str, object],
    unit_overrides: Mapping[int, str],
    profile: Mapping[str, object],
) -> _SourceTimingReference:
    """Return occurrence and bank-typical geometry for matching phone sides.

    Pause-neighbour units often include arbitrary recording collars. They are
    still retained in source-slice diagnostics, but cannot establish a spoken
    phone duration and are therefore excluded from both sums here.
    """
    raw_halves: list[float] = []
    profile_halves: list[float] = []
    sides: list[str] = []
    current = segments[segment_index].phone
    silence = {"pau", "sil", "sp"}
    if segment_index > 0:
        previous = segments[segment_index - 1].phone
        if previous not in silence and current not in silence:
            value = _choice_source_half(
                runtime, previous, current, side="right",
                selected_left_name=unit_overrides.get(segment_index - 1),
            )
            expected = profile_half_seconds(profile, current, "incoming")
            if value is not None and expected is not None:
                raw_halves.append(value)
                profile_halves.append(expected)
                sides.append("incoming")
    if segment_index + 1 < len(segments):
        following = segments[segment_index + 1].phone
        if current not in silence and following not in silence:
            value = _choice_source_half(
                runtime, current, following, side="left",
                selected_left_name=unit_overrides.get(segment_index),
            )
            expected = profile_half_seconds(profile, current, "outgoing")
            if value is not None and expected is not None:
                raw_halves.append(value)
                profile_halves.append(expected)
                sides.append("outgoing")
    return _SourceTimingReference(
        raw_duration=sum(raw_halves) if raw_halves else None,
        profile_duration=sum(profile_halves) if profile_halves else None,
        sides=tuple(sides),
    )


def _source_safe_bounds(
    phone: str,
    source_reference: Optional[float],
    *,
    timing_role: Optional[str] = None,
) -> tuple[float, float]:
    if timing_role == "moraic_nasal":
        lower, upper = 0.040, 0.240
    elif phone in _VOWELS:
        lower, upper = 0.020, 0.600
    elif phone == "N":
        lower, upper = 0.040, 0.350
    elif phone == "cl":
        lower, upper = 0.045, 0.300
    elif phone in _FRICATIVES | _AFFRICATES:
        lower, upper = 0.020, 0.260
    elif phone in _STOPS:
        lower, upper = 0.018, 0.220
    elif phone in _SONORANTS or phone.endswith("y"):
        lower, upper = 0.020, 0.240
    else:
        lower, upper = 0.020, 0.260
    if source_reference is None:
        return lower, upper
    if phone in _VOWELS:
        # Long stable vowel tails can be shortened or sustained safely; do
        # not mistake the entire recorded tail for a linguistic minimum.
        return lower, min(0.650, max(upper, source_reference * 1.5))
    lower = max(lower, min(0.060, source_reference * 0.35))
    if timing_role == "moraic_nasal":
        return lower, upper
    upper = min(max(upper, source_reference * 2.25), 0.400)
    return lower, upper


def _apply_source_timing_constraints(
    segments: Sequence[JapanesePlannedSegment],
    mora_segment_indices: Mapping[int, Sequence[int]],
    utterance: JapaneseUtterance,
    runtime: Mapping[str, object],
    unit_overrides: Mapping[int, str],
    phrase_final_moras: set[int],
    diagnostics: list[JapaneseSynthesisDiagnostic],
    *,
    speed: float,
    duration_model: str,
    duration_priors=None,
    duration_priors_error: Exception | None = None,
) -> tuple[
    tuple[JapanesePlannedSegment, ...],
    tuple[JapaneseMoraTiming, ...],
    str,
    str,
]:
    adjusted = list(segments)
    timings: list[JapaneseMoraTiming] = []
    effective_model = duration_model
    model_id = "legacy_mora_allocator_v2"
    priors = duration_priors
    if duration_model == "contextual":
        if priors is not None:
            model_id = priors.model_id
        else:
            effective_model = "legacy"
            diagnostics.append(JapaneseSynthesisDiagnostic(
                code="duration_model_legacy_fallback",
                message=(
                    "The contextual Japanese duration priors could not be "
                    "loaded; the legacy mora allocator was used."
                ),
                details={"error": str(
                    duration_priors_error or "duration priors unavailable"
                )},
            ))
    mora_lookup = {mora.index: mora for mora in utterance.moras}
    timing_profile = source_timing_profile(runtime)
    for mora_index in sorted(mora_segment_indices):
        rows: list[JapanesePhoneTiming] = []
        segment_indices = list(mora_segment_indices[mora_index])
        timing_references = [
            _source_timing_reference(
                adjusted, segment_index, runtime, unit_overrides,
                timing_profile,
            )
            for segment_index in segment_indices
        ]
        references = [item.raw_duration for item in timing_references]
        profile_references = [
            item.profile_duration for item in timing_references
        ]
        predictions: Sequence[JapaneseDurationPrediction | None]
        if effective_model == "contextual" and priors is not None:
            mora = mora_lookup[mora_index]
            contexts = build_duration_contexts(
                utterance,
                mora,
                [adjusted[index].phone for index in segment_indices],
            )
            predictions = predict_mora_durations(
                contexts,
                references,
                [adjusted[index].duration for index in segment_indices],
                speed=speed,
                source_profile_references=profile_references,
                priors=priors,
            )
        else:
            predictions = [None] * len(segment_indices)
        for segment_index, reference, timing_reference, prediction in zip(
                segment_indices, references, timing_references, predictions):
            segment = adjusted[segment_index]
            predicted = (
                float(prediction.predicted_duration)
                if prediction is not None else float(segment.duration)
            )
            safe_min, safe_max = _source_safe_bounds(
                segment.phone, reference, timing_role=segment.timing_role
            )
            final = max(safe_min, min(safe_max, predicted))
            final = round(final, 6)
            if abs(final - predicted) > 1e-7:
                diagnostics.append(JapaneseSynthesisDiagnostic(
                    code=("duration_oto_safe_clamp" if reference is not None
                          else "duration_class_safety_clamp"),
                    message=(
                        "A Japanese phone duration was bounded to a safe "
                        "source stretch; the requested and final values are "
                        "available in mora_timings."
                    ),
                    severity="info",
                    mora_index=mora_index,
                    details={
                        "phone": segment.phone,
                        "predicted_duration": round(predicted, 6),
                        "final_duration": final,
                        "source_safe_min": round(safe_min, 6),
                        "source_safe_max": round(safe_max, 6),
                    },
                ))
            if abs(final - float(segment.duration)) > 1e-7:
                adjusted[segment_index] = replace(segment, duration=final)
            stretch = (predicted / reference
                       if reference is not None and reference > 0.0 else None)
            rows.append(JapanesePhoneTiming(
                segment_index=segment_index,
                phone=segment.phone,
                predicted_duration=round(predicted, 6),
                source_reference_duration=(
                    round(reference, 6) if reference is not None else None
                ),
                source_profile_reference_duration=(
                    prediction.source_profile_reference_duration
                    if prediction is not None else
                    (round(timing_reference.profile_duration, 6)
                     if timing_reference.profile_duration is not None
                     else None)
                ),
                source_geometry_ratio=(
                    prediction.source_geometry_ratio
                    if prediction is not None else None
                ),
                source_geometry_ratio_bounded=(
                    prediction.source_geometry_ratio_bounded
                    if prediction is not None else None
                ),
                source_safe_min=round(safe_min, 6),
                source_safe_max=round(safe_max, 6),
                requested_stretch=(round(stretch, 6)
                                   if stretch is not None else None),
                final_duration=final,
                constraint_source=("profiled_oto_edges"
                                   if reference is not None
                                   else "phone_class"),
                baseline_source=(
                    prediction.baseline_source if prediction is not None
                    else "legacy_class_fallback"
                ),
                duration_model=effective_model,
                duration_model_id=model_id,
                context_log_ratio=(
                    prediction.context_log_ratio if prediction is not None
                    else 0.0
                ),
                context_effects=(
                    prediction.effects if prediction is not None else {}
                ),
                context=(
                    prediction.context.to_dict() if prediction is not None
                    else {}
                ),
            ))
        predicted_total = sum(item.predicted_duration for item in rows)
        final_total = sum(item.final_duration for item in rows)
        reference_total = sum(
            item.source_reference_duration or 0.0 for item in rows
        )
        timings.append(JapaneseMoraTiming(
            mora_index=mora_index,
            predicted_mora_duration=round(predicted_total, 6),
            phone_allocation=tuple(rows),
            source_safe_min=round(sum(item.source_safe_min for item in rows), 6),
            source_safe_max=round(sum(item.source_safe_max for item in rows), 6),
            requested_stretch=(
                round(predicted_total / reference_total, 6)
                if reference_total > 0.0 else None
            ),
            final_duration=round(final_total, 6),
            phrase_final_lengthening=mora_index in phrase_final_moras,
            duration_model=effective_model,
            duration_model_id=model_id,
        ))
    return tuple(adjusted), tuple(timings), effective_model, model_id


def _voiced_segment_for_mora(
    segment_indices: Sequence[int],
    segments: Sequence[JapanesePlannedSegment],
) -> Optional[int]:
    for index in reversed(segment_indices):
        phone = segments[index].phone
        if phone in _VOWELS or phone in {"N"}:
            return index
    return segment_indices[-1] if segment_indices else None


def create_synthesis_plan(
    utterance: JapaneseUtterance,
    *,
    runtime_metadata: Mapping[str, object] | Path | str | None = None,
    manual_candidate_overrides: Mapping[int, str] | None = None,
    base_pitch_hz: Optional[float] = None,
    speed: float = 1.0,
    phrase_pauses_ms: Mapping[str, object] | None = None,
    duration_model: str | None = None,
) -> JapaneseSynthesisPlan:
    """Create explicit Japanese phone timing, baseline F0, and unit choices.

    Manual candidate overrides are keyed by the canonical utterance's global
    mora index.  A multi-edge VCV choice can therefore override both the
    incoming and internal edge of exactly that occurrence.
    """
    if not 0.25 <= float(speed) <= 4.0:
        raise ValueError("Japanese synthesis speed must be between 0.25 and 4")
    runtime = _load_runtime(runtime_metadata)
    requested_duration_model = str(
        duration_model or runtime.get("duration_model") or "contextual"
    ).casefold()
    if requested_duration_model not in {"legacy", "contextual"}:
        raise ValueError("duration_model must be legacy or contextual")
    allocation_priors = None
    allocation_priors_error = None
    if requested_duration_model == "contextual":
        try:
            allocation_priors = load_duration_priors(
                runtime.get("duration_priors_path") or None
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            # The constraint pass emits the actionable fallback diagnostic.
            # Starting with legacy allocation keeps that fallback internally
            # coherent rather than applying half of the contextual model.
            allocation_priors = None
            allocation_priors_error = error
    contextual_allocator = (
        requested_duration_model == "contextual"
        and allocation_priors is not None
    )
    effective_phrase_pauses_ms = phrase_pauses_ms
    if effective_phrase_pauses_ms is None and allocation_priors is not None:
        effective_phrase_pauses_ms = allocation_priors.phrase_pauses_ms
    supported = tuple(str(item) for item in (
        runtime.get("supported_languages") or ()
    ))
    if runtime and runtime.get("language") != "ja" and "ja" not in supported:
        raise ValueError("runtime metadata is not Japanese")
    pitch_model = load_japanese_pitch_model(
        runtime.get("japanese_pitch_model_path") or None)
    base_pitch_hz, speaker_low_hz, speaker_high_hz = _speaker_pitch_context(
        runtime, base_pitch_hz, pitch_model
    )
    if not 50.0 <= float(base_pitch_hz) <= 500.0:
        raise ValueError("base Japanese pitch must be between 50 and 500 Hz")
    overrides = {
        int(key): str(value)
        for key, value in dict(manual_candidate_overrides or {}).items()
    }
    diagnostics: list[JapaneseSynthesisDiagnostic] = []
    segments: list[JapanesePlannedSegment] = []
    mora_segment_indices: dict[int, list[int]] = {}
    phrase_final_moras: set[int] = set()
    mapped_mora_phones: dict[int, tuple[str, ...]] = {}
    japanese_map = runtime.get("japanese_phoneme_map")
    profile_timings = {}
    if isinstance(japanese_map, Mapping):
        from arpasing_profile import map_japanese_mora

        all_moras = [mora for phrase in utterance.phrases
                     for accent in phrase.accent_phrases
                     for mora in accent.moras]
        available = tuple(str(item) for item in runtime.get("phones") or ())
        profile_timings = dict(japanese_map.get("timing_multipliers") or {})
        for position, mora in enumerate(all_moras):
            canonical = tuple(
                phone.symbol for phone in mora.phones
                if phone.symbol not in {"pau", "sil"}
            )
            following = None
            if position + 1 < len(all_moras):
                following = next((
                    phone.symbol for phone in all_moras[position + 1].phones
                    if phone.symbol not in {"pau", "sil"}
                ), None)
            mapped, reason = map_japanese_mora(
                mora.reading, canonical, japanese_map,
                following_phone=following,
                available_phones=available,
            )
            mapped_mora_phones[mora.index] = mapped
            if reason and reason.startswith("profile_target_missing"):
                diagnostics.append(JapaneseSynthesisDiagnostic(
                    code="profile_phone_unavailable",
                    message=("The ARPAsing Japanese profile selected a phone "
                             "that is absent from this generated bank."),
                    mora_index=mora.index,
                    details={"reading": mora.reading, "reason": reason,
                             "phones": list(mapped)},
                ))

    def add_segment(
        phone: str,
        duration: float,
        *,
        phrase_index: Optional[int] = None,
        accent_phrase_index: Optional[int] = None,
        mora_index: Optional[int] = None,
        source_phone_index: Optional[int] = None,
        pause_role: Optional[str] = None,
        timing_role: Optional[str] = None,
    ) -> int:
        index = len(segments)
        segments.append(JapanesePlannedSegment(
            index=index,
            phone=str(phone),
            duration=round(float(duration), 6),
            phrase_index=phrase_index,
            accent_phrase_index=accent_phrase_index,
            mora_index=mora_index,
            source_phone_index=source_phone_index,
            pause_role=pause_role,
            timing_role=timing_role,
        ))
        return index

    # Stable two-part edge pauses prevent the backend from changing indexes.
    add_segment("pau", 0.04 / speed, pause_role="leading_outer")
    add_segment("pau", 0.08 / speed, pause_role="leading_guard")

    for phrase_position, phrase in enumerate(utterance.phrases):
        if phrase.moras:
            phrase_final_moras.add(phrase.moras[-1].index)
        for accent_phrase in phrase.accent_phrases:
            for mora in accent_phrase.moras:
                indices = mora_segment_indices.setdefault(mora.index, [])
                source_phones = [phone for phone in mora.phones
                                 if phone.symbol not in {"pau", "sil"}]
                timing_mora = (
                    replace(mora, devoiced=False)
                    if contextual_allocator else mora
                )
                allocations = _mora_phone_durations(
                    timing_mora, speed,
                    phrase_final=(
                        mora.index in phrase_final_moras
                        if not contextual_allocator else False
                    ),
                    phone_symbols=mapped_mora_phones.get(mora.index),
                    # Integrated ARPAsing maps contain English consonant
                    # timing multipliers. They remain available to the legacy
                    # allocator but are not Japanese linguistic evidence.
                    timing_multipliers=(
                        profile_timings
                        if not contextual_allocator else None
                    ),
                    contextual=contextual_allocator,
                    mora_allocation_seconds=(
                        allocation_priors.mora_allocation_seconds
                        if allocation_priors is not None else None
                    ),
                    mora_anchor_seconds=(
                        allocation_priors.mora_anchor_seconds
                        if allocation_priors is not None else None
                    ),
                )
                for mapped_index, (symbol, duration) in enumerate(allocations):
                    phone = source_phones[
                        min(mapped_index, len(source_phones) - 1)
                    ] if source_phones else None
                    if mora.special_mora == "moraic_nasal":
                        timing_role = "moraic_nasal"
                    elif phone is not None and phone.symbol in _VOWELS:
                        timing_role = "vowel"
                    elif mora.special_mora == "long_vowel":
                        timing_role = "vowel"
                    elif mora.special_mora == "geminate":
                        timing_role = "geminate_closure"
                    else:
                        timing_role = "consonant"
                    indices.append(add_segment(
                        symbol,
                        duration,
                        phrase_index=phrase.index,
                        accent_phrase_index=accent_phrase.index,
                        mora_index=mora.index,
                        source_phone_index=(phone.index if phone else None),
                        timing_role=timing_role,
                    ))
        if phrase_position < len(utterance.phrases) - 1:
            # Guard both spoken edges.  Only the middle pause carries the
            # freely editable linguistic gap, so stretching it cannot smear
            # either adjacent source unit.
            add_segment(
                "pau", 0.08 / speed,
                phrase_index=phrase.index,
                pause_role="phrase_guard_out",
            )
            add_segment(
                "pau",
                _flexible_pause(
                    phrase.boundary_strength,
                    phrase.interrogative,
                    phrase.punctuation_after,
                    effective_phrase_pauses_ms,
                ) / speed,
                phrase_index=phrase.index,
                pause_role="phrase_gap",
            )
            add_segment(
                "pau", 0.08 / speed,
                phrase_index=phrase.index + 1,
                pause_role="phrase_guard_in",
            )

    add_segment("pau", 0.08 / speed, pause_role="trailing_guard")
    add_segment("pau", 0.04 / speed, pause_role="trailing_outer")

    candidate_units = dict(runtime.get("candidate_units") or {})
    unit_overrides: dict[int, str] = {}
    for mora_index in sorted(overrides):
        candidate_id = overrides[mora_index]
        mora_indices = mora_segment_indices.get(mora_index)
        if not mora_indices:
            diagnostics.append(JapaneseSynthesisDiagnostic(
                code="manual_candidate_mora_missing",
                message="Manual candidate refers to a mora absent from the plan.",
                mora_index=mora_index,
                candidate_id=candidate_id,
            ))
            continue
        rows = candidate_units.get(candidate_id)
        if not isinstance(rows, list):
            rows = list(rows) if isinstance(rows, tuple) else None
        if not rows:
            diagnostics.append(JapaneseSynthesisDiagnostic(
                code="manual_candidate_unknown",
                message="Manual candidate is not present in this generated voice.",
                mora_index=mora_index,
                candidate_id=candidate_id,
            ))
            continue
        mora_start = mora_indices[0]
        applied = 0
        for row in rows:
            try:
                edge_index = mora_start + int(row["edge_offset"])
                diphone = str(row["diphone"])
                left_name = str(row["left_name"])
            except (KeyError, TypeError, ValueError):
                continue
            if edge_index < 0 or edge_index + 1 >= len(segments):
                continue
            actual = (
                f"{segments[edge_index].phone}-"
                f"{segments[edge_index + 1].phone}"
            )
            if actual != diphone:
                diagnostics.append(JapaneseSynthesisDiagnostic(
                    code="manual_candidate_context_mismatch",
                    message=(
                        "Candidate edge does not match this mora occurrence "
                        "and was not applied."
                    ),
                    mora_index=mora_index,
                    candidate_id=candidate_id,
                    details={"expected": diphone, "actual": actual},
                ))
                continue
            unit_overrides[edge_index] = left_name
            applied += 1
        if not applied:
            diagnostics.append(JapaneseSynthesisDiagnostic(
                code="manual_candidate_not_applied",
                message="No edge from the manual candidate matched this occurrence.",
                mora_index=mora_index,
                candidate_id=candidate_id,
            ))

    (constrained_segments, mora_timings, effective_duration_model,
     duration_model_id) = _apply_source_timing_constraints(
        segments,
        mora_segment_indices,
        utterance,
        runtime,
        unit_overrides,
        phrase_final_moras,
        diagnostics,
        speed=speed,
        duration_model=requested_duration_model,
        duration_priors=allocation_priors,
        duration_priors_error=allocation_priors_error,
    )
    segments = list(constrained_segments)

    starts: list[float] = []
    cursor = 0.0
    for segment in segments:
        starts.append(cursor)
        cursor += segment.duration

    mora_times = {}
    for mora_index, segment_indices in mora_segment_indices.items():
        if not segment_indices:
            continue
        first = segment_indices[0]
        last = segment_indices[-1]
        mora_times[mora_index] = (
            starts[first] + starts[last] + segments[last].duration
        ) / 2.0

    targets: list[JapaneseF0Target] = []
    for contour in mora_pitch_contour(
            utterance, pitch_model, mora_times_seconds=mora_times):
        segment_index = _voiced_segment_for_mora(
            mora_segment_indices.get(contour.mora_index, ()), segments
        )
        if segment_index is None:
            continue
        time = starts[segment_index] + segments[segment_index].duration / 2.0
        targets.append(JapaneseF0Target(
            time=round(time, 6),
            log_f0=round(_speaker_relative_log_f0(
                base_pitch_hz,
                contour.semitones_from_baseline,
                speaker_low_hz,
                speaker_high_hz,
            ), 12),
            baseline_log_f0=round(
                pitch_domain.hz_to_log_f0(base_pitch_hz), 12),
            phrase_index=contour.phrase_index,
            accent_phrase_index=contour.accent_phrase_index,
            mora_index=contour.mora_index,
            kind=contour.kind,
            components_semitones={
                key: round(value, 6)
                for key, value in contour.components_semitones.items()
            },
        ))

    if any(
        item.accent_state in {"unknown", "unavailable"}
        for item in utterance.accent_phrases
    ):
        diagnostics.append(JapaneseSynthesisDiagnostic(
            code="neutral_accent_baseline",
            message=(
                "One or more accent phrases lack lexical accent; a neutral "
                "structural F0 baseline was used."
            ),
            severity="info",
        ))
    if any(phrase.interrogative for phrase in utterance.phrases):
        diagnostics.append(JapaneseSynthesisDiagnostic(
            code="question_intonation_uses_blocks",
            message=(
                "Interrogative analysis is retained, while question rise is "
                "applied by the shared Intonation-block layer."
            ),
            severity="info",
        ))

    return JapaneseSynthesisPlan(
        source_text=utterance.source_text,
        normalized_reading=utterance.normalized_reading,
        frontend_name=utterance.frontend_name,
        segments=tuple(segments),
        f0_targets=tuple(sorted(targets, key=lambda item: item.time)),
        unit_overrides={key: unit_overrides[key] for key in sorted(unit_overrides)},
        manual_candidate_overrides={
            key: overrides[key] for key in sorted(overrides)
        },
        diagnostics=tuple(diagnostics),
        base_pitch_hz=float(base_pitch_hz),
        speed=float(speed),
        duration_model=effective_duration_model,
        duration_model_id=duration_model_id,
        mora_timings=mora_timings,
        speaker_low_hz=round(float(speaker_low_hz), 6),
        speaker_high_hz=round(float(speaker_high_hz), 6),
        pitch_model_id=pitch_model.model_id,
        contour_model=_STRUCTURAL_CONTOUR_MODEL,
    )


def retime_mora_diagnostics(
    plan: JapaneseSynthesisPlan,
    segments: Sequence[JapanesePlannedSegment],
) -> JapaneseSynthesisPlan:
    """Keep serialized timing diagnostics coherent after a baseline retime."""
    by_index = {segment.index: segment for segment in segments}
    updated_moras: list[JapaneseMoraTiming] = []
    for mora in plan.mora_timings:
        phones: list[JapanesePhoneTiming] = []
        for phone in mora.phone_allocation:
            duration = float(by_index.get(
                phone.segment_index,
                plan.segments[phone.segment_index],
            ).duration)
            reference = phone.source_reference_duration
            phones.append(replace(
                phone,
                requested_stretch=(
                    round(duration / reference, 6)
                    if reference is not None and reference > 0.0 else None
                ),
                final_duration=round(duration, 6),
            ))
        reference_total = sum(
            item.source_reference_duration or 0.0 for item in phones
        )
        final_total = sum(item.final_duration for item in phones)
        updated_moras.append(replace(
            mora,
            phone_allocation=tuple(phones),
            requested_stretch=(
                round(final_total / reference_total, 6)
                if reference_total > 0.0 else None
            ),
            final_duration=round(final_total, 6),
        ))
    return replace(
        plan,
        segments=tuple(segments),
        mora_timings=tuple(updated_moras),
    )
