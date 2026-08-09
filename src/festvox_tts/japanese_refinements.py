"""Optional Phase 5 Japanese baselines, pitch routing, and voice colors.

The production structural plan remains the default.  Open JTalk is used only
for linguistic labels; an HTS trajectory must be supplied explicitly as JSON.
Dynamic source routing is deterministic and never replaces a manual unit edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Mapping, Optional, Protocol, Sequence

from japanese_models import JapaneseUtterance
import pitch_domain as pitch_domain
from japanese_synthesis import (
    JapaneseF0Target,
    JapaneseSynthesisDiagnostic,
    JapaneseSynthesisPlan,
    create_synthesis_plan,
    retime_mora_diagnostics,
)


REFINEMENT_SCHEMA_VERSION = 1
REFINEMENT_SCHEMA_STATUS = "phase5-provisional"
BASELINE_PROVIDERS = ("structural", "openjtalk_labels", "external_hts")
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_RANGE_RE = re.compile(
    r"^\s*([A-Ga-g][#b]?-?\d+)\s*-\s*"
    r"([A-Ga-g][#b]?-?\d+)\s*$"
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")


@dataclass(frozen=True)
class JapaneseBaselineDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.details:
            result["details"] = dict(sorted(self.details.items()))
        return result


@dataclass(frozen=True)
class JapaneseBaselineTrajectory:
    provider_name: str
    phones: tuple[str, ...]
    durations: tuple[float, ...]
    f0_targets: tuple[tuple[float, float], ...]
    provider_version: Optional[str] = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = REFINEMENT_SCHEMA_VERSION
    schema_status: str = REFINEMENT_SCHEMA_STATUS

    def __post_init__(self) -> None:
        if len(self.phones) != len(self.durations):
            raise ValueError("baseline phone and duration counts differ")
        if any(not 0.005 <= float(value) <= 2.0 for value in self.durations):
            raise ValueError("baseline durations must be between 5 ms and 2 s")
        if any(not 50.0 <= float(hz) <= 500.0
               for _time, hz in self.f0_targets):
            raise ValueError("baseline F0 targets must be between 50 and 500 Hz")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "kind": "japanese_optional_baseline_trajectory",
            "language": "ja",
            "provider_name": self.provider_name,
            "phones": list(self.phones),
            "durations": list(self.durations),
            "f0_targets": [list(item) for item in self.f0_targets],
            "provenance": dict(sorted(self.provenance.items())),
        }
        if self.provider_version is not None:
            result["provider_version"] = self.provider_version
        return result

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


@dataclass(frozen=True)
class JapaneseBaselineResult:
    trajectory: Optional[JapaneseBaselineTrajectory]
    diagnostics: tuple[JapaneseBaselineDiagnostic, ...] = ()


class JapaneseBaselineProvider(Protocol):
    name: str

    def provide(
        self,
        utterance: JapaneseUtterance,
        *,
        base_pitch_hz: float = 180.0,
        speed: float = 1.0,
    ) -> JapaneseBaselineResult:
        ...


def trajectory_from_plan(
    plan: JapaneseSynthesisPlan,
    *,
    provider_name: str = "structural",
    provider_version: Optional[str] = None,
    provenance: Mapping[str, object] | None = None,
) -> JapaneseBaselineTrajectory:
    return JapaneseBaselineTrajectory(
        provider_name=provider_name,
        provider_version=provider_version,
        phones=tuple(plan.phones),
        durations=tuple(duration for _phone, duration in plan.segment_durations),
        f0_targets=tuple(plan.pitch_targets),
        provenance=dict(provenance or {}),
    )


class StructuralBaselineProvider:
    name = "structural"

    def provide(
        self,
        utterance: JapaneseUtterance,
        *,
        base_pitch_hz: float = 180.0,
        speed: float = 1.0,
    ) -> JapaneseBaselineResult:
        plan = create_synthesis_plan(
            utterance, base_pitch_hz=base_pitch_hz, speed=speed
        )
        return JapaneseBaselineResult(trajectory_from_plan(plan))


class OpenJTalkLabelBaselineProvider:
    """Optional label-derived structural baseline; no HTS audio is generated."""

    name = "openjtalk_labels"

    def provide(
        self,
        utterance: JapaneseUtterance,
        *,
        base_pitch_hz: float = 180.0,
        speed: float = 1.0,
    ) -> JapaneseBaselineResult:
        try:
            from japanese_frontend import analyze_japanese
            analyzed = analyze_japanese(
                utterance.source_text, mode="openjtalk"
            )
            version = importlib.metadata.version("pyopenjtalk")
        except Exception as error:
            diagnostic = getattr(error, "diagnostic", None)
            message = getattr(diagnostic, "message", None) or str(error)
            return JapaneseBaselineResult(None, (
                JapaneseBaselineDiagnostic(
                    code="openjtalk_baseline_unavailable",
                    message=(
                        "Open JTalk label baseline is unavailable; the "
                        f"structural baseline remains active. {message}"
                    ),
                    severity="info",
                ),
            ))
        plan = create_synthesis_plan(
            analyzed, base_pitch_hz=base_pitch_hz, speed=speed
        )
        if tuple(plan.phones) != tuple(
                create_synthesis_plan(
                    utterance, base_pitch_hz=base_pitch_hz, speed=speed
                ).phones):
            return JapaneseBaselineResult(None, (
                JapaneseBaselineDiagnostic(
                    code="openjtalk_baseline_phone_mismatch",
                    message=(
                        "Open JTalk labels produced a different phone sequence; "
                        "the current edited utterance was left unchanged."
                    ),
                ),
            ))
        return JapaneseBaselineResult(trajectory_from_plan(
            plan,
            provider_name=self.name,
            provider_version=version,
            provenance={
                "source": "pyopenjtalk_fullcontext_labels",
                "waveform_used": False,
                "trajectory_kind": "structural_label_experiment",
            },
        ), (
            JapaneseBaselineDiagnostic(
                code="openjtalk_labels_not_hts_trajectory",
                message=(
                    "Open JTalk supplied linguistic labels only; this is an "
                    "experimental structural baseline, not HTS acoustic output."
                ),
                severity="info",
            ),
        ))


class ExternalHTSTrajectoryProvider:
    """Read an explicit trajectory export without loading an HTS waveform."""

    name = "external_hts"

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def provide(
        self,
        utterance: JapaneseUtterance,
        *,
        base_pitch_hz: float = 180.0,
        speed: float = 1.0,
    ) -> JapaneseBaselineResult:
        del utterance, base_pitch_hz, speed
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return JapaneseBaselineResult(None, (
                JapaneseBaselineDiagnostic(
                    code="external_hts_trajectory_unavailable",
                    message=(
                        "External HTS trajectory could not be read; the "
                        f"structural baseline remains active. {error}"
                    ),
                    severity="info",
                ),
            ))
        if value.get("language") != "ja":
            return JapaneseBaselineResult(None, (
                JapaneseBaselineDiagnostic(
                    code="external_hts_trajectory_not_japanese",
                    message="External trajectory is not marked language='ja'.",
                ),
            ))
        try:
            trajectory = JapaneseBaselineTrajectory(
                provider_name=self.name,
                provider_version=(str(value["provider_version"])
                                  if value.get("provider_version") else None),
                phones=tuple(str(item) for item in value.get("phones") or ()),
                durations=tuple(float(item) for item in
                                value.get("durations") or ()),
                f0_targets=tuple(
                    (float(item[0]), float(item[1]))
                    for item in value.get("f0_targets") or ()
                ),
                provenance={
                    "source": "external_trajectory_json",
                    "waveform_used": False,
                    "input_sha256": __import__("hashlib").sha256(
                        self.path.read_bytes()
                    ).hexdigest(),
                },
            )
        except (TypeError, ValueError, IndexError) as error:
            return JapaneseBaselineResult(None, (
                JapaneseBaselineDiagnostic(
                    code="external_hts_trajectory_invalid",
                    message=f"External trajectory is invalid: {error}",
                ),
            ))
        return JapaneseBaselineResult(trajectory)


def resolve_baseline_provider(
    mode: str,
    *,
    external_path: Path | str | None = None,
) -> JapaneseBaselineProvider:
    normalized = str(mode or "structural").strip().casefold()
    if normalized == "structural":
        return StructuralBaselineProvider()
    if normalized == "openjtalk_labels":
        return OpenJTalkLabelBaselineProvider()
    if normalized == "external_hts":
        return ExternalHTSTrajectoryProvider(external_path or "")
    raise ValueError(f"unknown Japanese baseline provider: {mode!r}")


def apply_baseline_trajectory(
    plan: JapaneseSynthesisPlan,
    trajectory: JapaneseBaselineTrajectory,
    *,
    preserve_structural_f0: bool = False,
) -> JapaneseSynthesisPlan:
    """Apply a compatible baseline while preserving every manual unit edge."""
    if tuple(plan.phones) != trajectory.phones:
        diagnostic = JapaneseSynthesisDiagnostic(
            code="optional_baseline_phone_mismatch",
            message=(
                "Optional baseline phone sequence differs from the edited "
                "utterance and was not applied."
            ),
        )
        return replace(plan, diagnostics=plan.diagnostics + (diagnostic,))
    segments = tuple(
        replace(segment, duration=round(float(duration), 6))
        for segment, duration in zip(plan.segments, trajectory.durations)
    )
    templates = list(plan.f0_targets)
    targets = []
    if preserve_structural_f0:
        old_starts, new_starts = [], []
        old_cursor = new_cursor = 0.0
        for old_segment, new_segment in zip(plan.segments, segments):
            old_starts.append(old_cursor)
            new_starts.append(new_cursor)
            old_cursor += old_segment.duration
            new_cursor += new_segment.duration
        for template in templates:
            segment_index = max(0, min(
                len(plan.segments) - 1,
                next((index for index, start in enumerate(old_starts)
                      if start + plan.segments[index].duration >= template.time),
                     len(plan.segments) - 1),
            ))
            old_segment = plan.segments[segment_index]
            ratio = (template.time - old_starts[segment_index]) / max(
                1e-9, old_segment.duration
            )
            targets.append(replace(
                template,
                time=round(new_starts[segment_index] + max(
                    0.0, min(1.0, ratio)
                ) * segments[segment_index].duration, 6),
            ))
    else:
        for position, (time, hz) in enumerate(trajectory.f0_targets):
            if templates:
                template = min(
                    templates, key=lambda item: (abs(item.time - time), item.time)
                )
                targets.append(replace(
                    template, time=round(float(time), 6),
                    log_f0=round(pitch_domain.hz_to_log_f0(float(hz)), 12),
                    kind=f"{trajectory.provider_name}_baseline",
                    components_semitones={
                        "optional_baseline": round(
                            pitch_domain.semitone_difference(
                                plan.base_pitch_hz, float(hz)), 6)
                    },
                ))
            else:
                targets.append(JapaneseF0Target(
                    time=round(float(time), 6),
                    log_f0=round(pitch_domain.hz_to_log_f0(float(hz)), 12),
                    baseline_log_f0=round(
                        pitch_domain.hz_to_log_f0(plan.base_pitch_hz), 12),
                    phrase_index=-1,
                    accent_phrase_index=-1,
                    mora_index=position,
                    kind=f"{trajectory.provider_name}_baseline",
                    components_semitones={
                        "optional_baseline": round(
                            pitch_domain.semitone_difference(
                                plan.base_pitch_hz, float(hz)), 6)
                    },
                ))
    diagnostic = JapaneseSynthesisDiagnostic(
        code="optional_baseline_applied",
        message=(
            f"Applied optional {trajectory.provider_name} duration/F0 "
            "baseline. Manual units, accent edits, and the continuous F0 "
            "editor remain final."
        ),
        severity="info",
        details={"provider": trajectory.provider_name},
    )
    updated = replace(
        plan,
        segments=segments,
        f0_targets=tuple(sorted(targets, key=lambda item: item.time)),
        diagnostics=plan.diagnostics + (diagnostic,),
    )
    return retime_mora_diagnostics(updated, segments)


def note_to_midi(value: str) -> Optional[float]:
    match = _NOTE_RE.match(str(value).strip())
    if not match:
        return None
    pitch_class = {
        "C": 0, "D": 2, "E": 4, "F": 5,
        "G": 7, "A": 9, "B": 11,
    }[match.group(1).upper()]
    accidental = 1 if match.group(2) == "#" else \
        -1 if match.group(2) == "b" else 0
    octave = int(match.group(3))
    return float(12 * (octave + 1) + pitch_class + accidental)


def tone_range_center(value: str) -> Optional[float]:
    match = _RANGE_RE.match(str(value))
    if match:
        left = note_to_midi(match.group(1))
        right = note_to_midi(match.group(2))
        if left is not None and right is not None:
            return (left + right) / 2.0
    return note_to_midi(str(value).strip())


def hz_to_midi(hz: float) -> float:
    if not 20.0 <= float(hz) <= 20000.0:
        raise ValueError("frequency is outside the supported range")
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


@dataclass(frozen=True)
class JapaneseRoutingPolicy:
    dynamic_pitch: bool = False
    voice_color: Optional[str] = None


def available_voice_colors(runtime: Mapping[str, object]) -> tuple[str, ...]:
    colors = {
        str(row.get("color") or "")
        for row in runtime.get("subbanks") or ()
        if isinstance(row, Mapping) and str(row.get("color") or "")
    }
    return tuple(sorted(colors, key=lambda item: (item.casefold(), item)))


def _subbank_map(runtime: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(row.get("subbank_id") or ""): row
        for row in runtime.get("subbanks") or ()
        if isinstance(row, Mapping) and row.get("subbank_id")
    }


def _choice_colors(
    choice: Mapping[str, object],
    subbanks: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    values = {
        str(subbanks[str(item)].get("color") or "")
        for item in choice.get("subbank_ids") or ()
        if str(item) in subbanks
        and str(subbanks[str(item)].get("color") or "")
    }
    return tuple(sorted(values, key=lambda item: (item.casefold(), item)))


def _choice_pitch_midi(
    choice: Mapping[str, object],
    subbanks: Mapping[str, Mapping[str, object]],
) -> Optional[float]:
    values = [
        note_to_midi(str(item))
        for item in choice.get("source_pitch_tags") or ()
    ]
    values = [item for item in values if item is not None]
    if not values:
        for subbank_id in choice.get("subbank_ids") or ():
            row = subbanks.get(str(subbank_id))
            if row is None:
                continue
            values.extend(
                item for item in (
                    tone_range_center(str(value))
                    for value in row.get("tone_ranges") or ()
                ) if item is not None
            )
    if not values:
        return None
    return sum(values) / len(values)


def _target_hz(plan: JapaneseSynthesisPlan, time: float) -> float:
    targets = sorted(plan.pitch_targets)
    if not targets:
        return float(plan.base_pitch_hz)
    if time <= targets[0][0]:
        return float(targets[0][1])
    if time >= targets[-1][0]:
        return float(targets[-1][1])
    for left, right in zip(targets, targets[1:]):
        if left[0] <= time <= right[0]:
            width = max(1e-9, right[0] - left[0])
            ratio = (time - left[0]) / width
            return left[1] + (right[1] - left[1]) * ratio
    return float(plan.base_pitch_hz)


def route_dynamic_candidates(
    plan: JapaneseSynthesisPlan,
    runtime_metadata: Mapping[str, object],
    policy: JapaneseRoutingPolicy,
) -> JapaneseSynthesisPlan:
    """Route compiled alternatives by F0/color; existing overrides stay final."""
    requested_color = str(policy.voice_color or "").strip()
    if not policy.dynamic_pitch and not requested_color:
        return plan
    if runtime_metadata.get("language") != "ja":
        raise ValueError("runtime metadata is not Japanese")
    alternatives = dict(runtime_metadata.get("alternatives") or {})
    subbanks = _subbank_map(runtime_metadata)
    overrides = {int(key): str(value) for key, value in
                 dict(plan.unit_overrides).items()}
    diagnostics = list(plan.diagnostics)
    routed = 0
    color_fallbacks = 0
    starts = []
    cursor = 0.0
    for segment in plan.segments:
        starts.append(cursor)
        cursor += segment.duration

    for edge_index in range(max(0, len(plan.segments) - 1)):
        if edge_index in overrides:
            continue
        pair = (
            f"{plan.segments[edge_index].phone}-"
            f"{plan.segments[edge_index + 1].phone}"
        )
        choices = [dict(item) for item in alternatives.get(pair) or ()]
        if not choices:
            continue
        pool = choices
        if requested_color:
            exact = [choice for choice in choices
                     if requested_color in _choice_colors(choice, subbanks)]
            if exact:
                pool = exact
            else:
                color_fallbacks += 1
        edge_time = starts[edge_index + 1] + \
            plan.segments[edge_index + 1].duration / 2.0
        target_midi = hz_to_midi(_target_hz(plan, edge_time))

        def key(choice: Mapping[str, object]):
            source_midi = _choice_pitch_midi(choice, subbanks)
            distance = abs(source_midi - target_midi) \
                if policy.dynamic_pitch and source_midi is not None else \
                (10000.0 if policy.dynamic_pitch else 0.0)
            return (
                distance,
                float(choice.get("selection_cost") or 0.0),
                str(choice.get("candidate_id") or choice.get("id") or ""),
                str(choice.get("left_name") or ""),
            )

        selected = min(pool, key=key) if policy.dynamic_pitch else pool[0]
        left_name = str(selected.get("left_name") or "")
        if left_name and left_name != str(choices[0].get("left_name") or ""):
            overrides[edge_index] = left_name
            routed += 1
    if routed:
        diagnostics.append(JapaneseSynthesisDiagnostic(
            code="dynamic_source_routing_applied",
            message=(
                f"Dynamically routed {routed} Japanese unit edges by declared "
                "pitch/color metadata."
            ),
            severity="info",
            details={
                "dynamic_pitch": bool(policy.dynamic_pitch),
                "voice_color": requested_color or "auto",
            },
        ))
    if color_fallbacks:
        diagnostics.append(JapaneseSynthesisDiagnostic(
            code="voice_color_fallback",
            message=(
                f"Requested voice color {requested_color!r} was unavailable "
                f"for {color_fallbacks} unit edges; ordinary candidates were "
                "retained and no alias was discarded."
            ),
            details={"edge_count": color_fallbacks,
                     "voice_color": requested_color},
        ))
    return replace(
        plan,
        unit_overrides={key: overrides[key] for key in sorted(overrides)},
        diagnostics=tuple(diagnostics),
    )
