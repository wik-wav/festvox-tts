"""Rule-based Asaxi phone durations on a mora-timed scaffold.

This is an engineering prior for voices that do not yet have recorded Asaxi
duration data.  It deliberately models approximate mora timing rather than
assigning one duration to every phone:

* consonants and nuclei divide a mora according to broad articulatory class;
* longer onsets partly shorten the following nucleus;
* closed and otherwise complex morae remain slightly longer instead of being
  compressed to exact isochrony;
* syllabic nasals and geminate holds occupy their own mora;
* phrase-final lengthening is concentrated on the nucleus and coda.

The model never changes phone identity or source-unit selection.  It only
returns a replacement duration for each already-rendered Segment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import asaxi_prosody


ASAXI_DURATION_MODEL_ID = "asaxi-moraic-rules-v1"
PAUSE_PHONES = frozenset({"pau", "sil", "sp", "#"})
VOWEL_PHONES = frozenset({
    "a", "e", "i", "o", "u", "ao", "ax", "ih",
    "aa", "ae", "ah", "aw", "ay", "eh", "er", "ey",
    "iy", "ow", "oy", "uh", "uw",
})
DIPHTHONG_PHONES = frozenset({"aw", "ay", "ey", "ow", "oy", "uw", "er"})
VOICELESS_STOPS = frozenset({"p", "t", "k", "q", "py", "ty", "ky"})
VOICED_STOPS = frozenset({"b", "d", "g", "by", "dy", "gy"})
TAPS = frozenset({"dx", "dxy"})
VOICELESS_AFFRICATES = frozenset({"ch", "ts"})
VOICED_AFFRICATES = frozenset({"jh", "dz"})
VOICELESS_FRICATIVES = frozenset({
    "f", "s", "sh", "th", "h", "hh", "fy", "hy",
})
VOICED_FRICATIVES = frozenset({"v", "z", "zh", "dh", "vy", "zi"})
NASALS = frozenset({"m", "n", "ng", "my", "ny", "ngy"})
SYLLABIC_NASALS = frozenset({"mm", "nn", "nng", "xn"})
LIQUIDS = frozenset({"l", "r", "rr", "ly", "ry", "ri"})
GLIDES = frozenset({"w", "y", "wi"})


@dataclass(frozen=True)
class AsaxiDurationConfig:
    """Auditable constants for the provisional rule model."""

    model_id: str = ASAXI_DURATION_MODEL_ID
    base_mora_seconds: float = 0.120
    complexity_compensation: float = 0.28
    vowel_only_factor: float = 0.94
    syllabic_nasal_factor: float = 0.98
    geminate_hold_factor: float = 0.90
    phrase_final_factor: float = 1.14
    minor_boundary_factor: float = 1.07
    minimum_mora_seconds: float = 0.090
    maximum_mora_seconds: float = 0.180


DEFAULT_CONFIG = AsaxiDurationConfig()


@dataclass(frozen=True)
class AsaxiPhoneDuration:
    segment_index: int
    phone: str
    phone_class: str
    role: str
    duration_seconds: float
    mora_index: int
    mora_kind: str
    mora_target_seconds: float
    phrase_final: bool = False
    absorbed_mora_indices: tuple[int, ...] = ()
    modifiers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_index": self.segment_index,
            "phone": self.phone,
            "phone_class": self.phone_class,
            "role": self.role,
            "duration_seconds": round(self.duration_seconds, 6),
            "mora_index": self.mora_index,
            "mora_kind": self.mora_kind,
            "mora_target_seconds": round(self.mora_target_seconds, 6),
            "phrase_final": self.phrase_final,
            "absorbed_mora_indices": list(self.absorbed_mora_indices),
            "modifiers": list(self.modifiers),
        }


@dataclass(frozen=True)
class AsaxiDurationPlan:
    model_id: str
    speed: float
    entries: tuple[tuple[str, float], ...]
    phones: tuple[AsaxiPhoneDuration, ...]
    diagnostics: tuple[asaxi_prosody.AsaxiProsodyDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        spoken = [row for row in self.phones if row.phone not in PAUSE_PHONES]
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "speed": round(self.speed, 6),
            "phone_count": len(spoken),
            "total_spoken_seconds": round(sum(
                row.duration_seconds for row in spoken
            ), 6),
            "phones": [row.to_dict() for row in self.phones],
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


def _base_phone(phone: str) -> str:
    value = re.sub(r"__u\d+$", "", str(phone or "")).rstrip("_").lower()
    return value


def phone_class(phone: str) -> str:
    """Return the broad class used by the duration prior."""

    value = _base_phone(phone)
    if value in PAUSE_PHONES:
        return "pause"
    if value in VOWEL_PHONES:
        return "vowel"
    if value == "cl":
        return "geminate_hold"
    if value in TAPS:
        return "tap"
    if value in VOICELESS_STOPS:
        return "stop_voiceless"
    if value in VOICED_STOPS:
        return "stop_voiced"
    if value in VOICELESS_AFFRICATES:
        return "affricate_voiceless"
    if value in VOICED_AFFRICATES:
        return "affricate_voiced"
    if value in VOICELESS_FRICATIVES:
        return "fricative_voiceless"
    if value in VOICED_FRICATIVES:
        return "fricative_voiced"
    if value in SYLLABIC_NASALS:
        return "syllabic_nasal"
    if value in NASALS:
        return "nasal"
    if value in LIQUIDS:
        return "liquid"
    if value in GLIDES:
        return "glide"
    if value.endswith("y"):
        return phone_class(value[:-1])
    return "other"


def _preferred_seconds(phone: str, role: str) -> float:
    value = _base_phone(phone)
    category = phone_class(value)
    if category == "vowel":
        multiplier = {
            "ih": 0.84,
            "ax": 0.86,
            "u": 0.90,
            "i": 0.92,
            "iy": 0.94,
            "e": 0.98,
            "eh": 0.98,
            "o": 1.02,
            "a": 1.05,
            "aa": 1.05,
            "ah": 1.02,
            "ao": 1.07,
        }.get(value, 1.0)
        if value in DIPHTHONG_PHONES:
            multiplier = max(multiplier, 1.15)
        return 0.090 * multiplier
    return {
        "stop_voiceless": 0.052,
        "stop_voiced": 0.044,
        "tap": 0.030,
        "affricate_voiceless": 0.072,
        "affricate_voiced": 0.064,
        "fricative_voiceless": 0.075,
        "fricative_voiced": 0.065,
        "nasal": 0.060 if role != "coda" else 0.055,
        "syllabic_nasal": 0.118,
        "liquid": 0.050,
        "glide": 0.040,
        "geminate_hold": 0.108,
        "other": 0.055,
    }[category]


def _minimum_seconds(phone: str, role: str) -> float:
    category = phone_class(phone)
    if role == "nucleus":
        return 0.060
    if role == "geminate_hold":
        return 0.075
    return {
        "stop_voiceless": 0.030,
        "stop_voiced": 0.028,
        "tap": 0.022,
        "affricate_voiceless": 0.042,
        "affricate_voiced": 0.038,
        "fricative_voiceless": 0.045,
        "fricative_voiced": 0.040,
        "nasal": 0.034,
        "syllabic_nasal": 0.060,
        "liquid": 0.032,
        "glide": 0.024,
        "geminate_hold": 0.075,
        "vowel": 0.060,
        "other": 0.028,
        "pause": 0.0,
    }[category]


def _roles_for_mora(
    mora: asaxi_prosody.AsaxiRenderedMora,
    phones: Sequence[str],
) -> tuple[str, ...]:
    if mora.kind == "syllabic_nasal":
        return tuple("nucleus" for _phone in phones)
    if mora.kind == "geminate":
        return tuple("geminate_hold" for _phone in phones)
    nuclei = [
        index for index, phone in enumerate(phones)
        if phone_class(phone) in {"vowel", "syllabic_nasal"}
    ]
    if not nuclei:
        return tuple("nonvocalic" for _phone in phones)
    first = nuclei[0]
    last = nuclei[-1]
    return tuple(
        "onset" if index < first
        else "coda" if index > last
        else "nucleus"
        for index in range(len(phones))
    )


def _allocate_with_floors(
    preferred: Sequence[float],
    floors: Sequence[float],
    target: float,
) -> tuple[float, ...]:
    floor_total = sum(floors)
    target = max(float(target), floor_total)
    flexible = [
        max(0.0, float(value) - float(floor))
        for value, floor in zip(preferred, floors)
    ]
    flexible_total = sum(flexible)
    remainder = max(0.0, target - floor_total)
    if flexible_total <= 1.0e-12:
        share = remainder / max(1, len(floors))
        return tuple(float(floor) + share for floor in floors)
    return tuple(
        float(floor) + remainder * value / flexible_total
        for floor, value in zip(floors, flexible)
    )


def _mora_target(
    mora: asaxi_prosody.AsaxiRenderedMora,
    preferred: Sequence[float],
    roles: Sequence[str],
    config: AsaxiDurationConfig,
) -> float:
    if mora.kind == "geminate":
        return config.base_mora_seconds * config.geminate_hold_factor
    if mora.kind == "syllabic_nasal":
        return config.base_mora_seconds * config.syllabic_nasal_factor
    if "nucleus" not in roles:
        return min(
            config.base_mora_seconds,
            max(0.035, sum(preferred)),
        )
    target = config.base_mora_seconds + (
        config.complexity_compensation
        * max(0.0, sum(preferred) - config.base_mora_seconds)
    )
    if all(role == "nucleus" for role in roles):
        target *= config.vowel_only_factor
    return min(
        config.maximum_mora_seconds,
        max(config.minimum_mora_seconds, target),
    )


def _boundary_factor(
    plan: asaxi_prosody.AsaxiProsodyPlan,
    config: AsaxiDurationConfig,
) -> float:
    if plan.boundary_tone == "H-":
        return config.minor_boundary_factor
    if plan.boundary_tone in {"L%", "H%", "LH%"}:
        return config.phrase_final_factor
    return 1.0


def _segment_row(segment) -> tuple[str, float, float]:
    if hasattr(segment, "phone"):
        return (
            str(segment.phone),
            float(segment.start),
            float(segment.end),
        )
    return str(segment[0]), float(segment[1]), float(segment[2])


def plan_durations(
    plan: asaxi_prosody.AsaxiProsodyPlan,
    segments: Iterable,
    *,
    speed: float = 1.0,
    config: AsaxiDurationConfig = DEFAULT_CONFIG,
) -> AsaxiDurationPlan:
    """Return deterministic phone durations aligned to ``segments``.

    Pause and unmatched segment durations are retained from Festival.  This
    lets punctuation and the GUI's four-part phrase pauses remain independent
    from the mora model.
    """

    rows = tuple(_segment_row(segment) for segment in segments)
    rate = max(0.25, min(4.0, float(speed or 1.0)))
    durations = [
        max(0.0, end - start) for _phone, start, end in rows
    ]
    aligned, alignment_diagnostics = asaxi_prosody.rendered_morae(
        plan, rows
    )
    diagnostics = tuple(
        diagnostic for diagnostic in alignment_diagnostics
        if diagnostic.code in {
            "festival_phone_alignment",
            "mora_without_rendered_phone",
        }
    )
    nonempty = [mora for mora in aligned if mora.segment_indices]
    final_index = nonempty[-1].index if nonempty else -1
    final_factor = _boundary_factor(plan, config)
    mutable: dict[int, dict[str, object]] = {}

    for mora in nonempty:
        indices = tuple(int(index) for index in mora.segment_indices)
        phones = tuple(rows[index][0] for index in indices)
        roles = _roles_for_mora(mora, phones)
        preferred = tuple(
            _preferred_seconds(phone, role)
            for phone, role in zip(phones, roles)
        )
        floors = tuple(
            _minimum_seconds(phone, role)
            for phone, role in zip(phones, roles)
        )
        target = _mora_target(mora, preferred, roles, config)
        allocated = list(_allocate_with_floors(
            preferred, floors, target
        ))
        is_final = mora.index == final_index and final_factor > 1.0
        if is_final:
            for index, role in enumerate(roles):
                concentration = {
                    "onset": 0.20,
                    "nucleus": 1.00,
                    "coda": 0.70,
                    "geminate_hold": 1.00,
                    "nonvocalic": 0.60,
                }[role]
                allocated[index] *= (
                    1.0 + (final_factor - 1.0) * concentration
                )
        effective_target = sum(allocated) / rate
        for segment_index, phone, role, duration in zip(
            indices, phones, roles, allocated
        ):
            scaled = max(0.010, float(duration) / rate)
            durations[segment_index] = scaled
            modifiers = (
                ("phrase_final_lengthening",) if is_final else ()
            )
            mutable[segment_index] = {
                "segment_index": segment_index,
                "phone": phone,
                "phone_class": phone_class(phone),
                "role": role,
                "duration_seconds": scaled,
                "mora_index": mora.index,
                "mora_kind": mora.kind,
                "mora_target_seconds": effective_target,
                "phrase_final": is_final,
                "absorbed_mora_indices": (),
                "modifiers": modifiers,
            }

    # A continuant geminate has no separate structural phone.  Its mora is
    # realized by extending the following onset, not by duplicating that phone.
    for position, mora in enumerate(aligned):
        if mora.kind != "geminate" or mora.segment_indices:
            continue
        following = next(
            (
                candidate for candidate in aligned[position + 1:]
                if candidate.segment_indices
            ),
            None,
        )
        if following is None:
            diagnostics += (asaxi_prosody.AsaxiProsodyDiagnostic(
                "unrealized_geminate_duration",
                (
                    f"Geminate mora {mora.text!r} has no following phone "
                    "that can carry its hold."
                ),
                "warning",
                mora.word_index,
            ),)
            continue
        segment_index = int(following.segment_indices[0])
        extension = (
            config.base_mora_seconds
            * config.geminate_hold_factor
            / rate
        )
        durations[segment_index] += extension
        row = mutable.get(segment_index)
        if row is None:
            continue
        absorbed = tuple(row["absorbed_mora_indices"]) + (mora.index,)
        modifiers = tuple(row["modifiers"]) + (
            "continuant_geminate_hold",
        )
        row["duration_seconds"] = durations[segment_index]
        row["absorbed_mora_indices"] = absorbed
        row["modifiers"] = modifiers

    predictions = tuple(
        AsaxiPhoneDuration(**mutable[index])
        for index in sorted(mutable)
    )
    entries = tuple(
        (phone, round(float(duration), 9))
        for (phone, _start, _end), duration in zip(rows, durations)
    )
    if any(
        not math.isfinite(duration) or duration < 0.0
        for _phone, duration in entries
    ):
        raise ValueError("Asaxi duration planning produced an invalid value")
    return AsaxiDurationPlan(
        model_id=config.model_id,
        speed=rate,
        entries=entries,
        phones=predictions,
        diagnostics=diagnostics,
    )
