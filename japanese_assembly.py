"""Inspectable Japanese source assembly and automatic unit selection.

The generated Festival selector works one canonical diphone at a time.  This
module mirrors that deterministic decision in Python and exposes the complete
source contribution for each utterance edge.  It is used by tests, quality
checks, and the Recordings UI; it never writes to a source UTAU bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping, Optional, Sequence

from japanese_candidates import PRIMARY_ROLES_BY_CONFIGURATION
from special_phones import resolve_special_phone_sequence


ASSEMBLY_SCHEMA_VERSION = 1
ASSEMBLY_SCHEMA_STATUS = "stage2-provisional"
_VOWELS = frozenset({"a", "i", "u", "e", "o"})
_SILENCE = frozenset({"pau", "sil"})
_PAIRED_ROLES = frozenset({"phrase_start_cv", "vcv_mora", "release"})


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")


def automatic_context_bonus(
    choice: Mapping[str, object],
    outer_left: str,
    outer_right: str,
    left_phone: str = "",
    right_phone: str = "",
    moraic_nasal_routing: Mapping[str, object] | None = None,
) -> float:
    """Mirror the generated SIOD selector's named context rules."""
    role = str(choice.get("role") or "")
    expected_left = str(choice.get("recorded_left_context") or "*")
    expected_right = str(choice.get("recorded_right_context") or "*")
    edge = str(choice.get("edge_offset") or "0")
    if role == "phrase_start_cv":
        if outer_left != "pau":
            base = -120.0
        elif edge == "-1":
            base = (
                120.0 if expected_right in {"*", outer_right} else -120.0
            )
        else:
            base = 110.0
    elif role == "vcv_mora":
        if edge == "-1":
            base = (
                210.0 if expected_right in {"", "*"}
                else 115.0 if expected_right == outer_right else 10.0
            )
        else:
            base = 115.0 if expected_left == outer_left else 10.0
    elif role == "vc_transition":
        if expected_right == "*":
            base = 110.0
        else:
            base = 120.0 if expected_right == outer_right else -80.0
    elif role == "release":
        if edge == "0":
            base = 100.0 if expected_left == outer_left else 20.0
        else:
            base = 55.0
    elif role == "generated_cv_bridge":
        if expected_right == "*":
            base = 40.0
        else:
            base = 50.0 if expected_right == outer_right else -80.0
    elif role == "vowel_blend":
        base = 85.0
    elif role in {"mora_cv", "special_mora"}:
        base = 60.0
    else:
        base = 0.0

    desired = desired_moraic_nasal_allophone(
        left_phone,
        right_phone,
        outer_right,
        moraic_nasal_routing or {},
    )
    if not desired:
        return base
    source = str(choice.get("moraic_nasal_allophone") or "")
    if source == desired:
        return base + 500.0
    if not source:
        return base - 150.0
    return base - 500.0


def desired_moraic_nasal_allophone(
    left_phone: str,
    right_phone: str,
    outer_right: str,
    routing: Mapping[str, object],
) -> str:
    """Resolve a bank-defined /N/ source identity from following context."""
    if right_phone == "N":
        following = outer_right
    elif left_phone == "N":
        following = right_phone
    else:
        return ""
    by_phone = routing.get("following_phones")
    if isinstance(by_phone, Mapping) and following in by_phone:
        return str(by_phone[following])
    return str(routing.get("default") or "")


def automatic_choice_score(
    choice: Mapping[str, object],
    outer_left: str,
    outer_right: str,
    left_phone: str = "",
    right_phone: str = "",
    moraic_nasal_routing: Mapping[str, object] | None = None,
) -> float:
    return (
        automatic_context_bonus(
            choice,
            outer_left,
            outer_right,
            left_phone,
            right_phone,
            moraic_nasal_routing,
        )
        + 20.0
        - 5.0 * float(choice.get("selection_cost") or 0.0)
    )


def select_automatic_choice(
    choices: Sequence[Mapping[str, object]],
    outer_left: str,
    outer_right: str,
    left_phone: str = "",
    right_phone: str = "",
    moraic_nasal_routing: Mapping[str, object] | None = None,
) -> Optional[Mapping[str, object]]:
    """Select exactly as Festival does, preserving first-row tie breaking."""
    best: Optional[Mapping[str, object]] = None
    best_score = -100000.0
    for choice in choices:
        if (
            str(choice.get("role") or "") == "phrase_start_cv"
            and outer_left != "pau"
        ):
            continue
        score = automatic_choice_score(
            choice,
            outer_left,
            outer_right,
            left_phone,
            right_phone,
            moraic_nasal_routing,
        )
        if best is None or score > best_score:
            best = choice
            best_score = score
    return best


@dataclass(frozen=True)
class JapaneseAssemblyDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    edge_index: Optional[int] = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.edge_index is not None:
            result["edge_index"] = self.edge_index
        if self.details:
            result["details"] = {
                key: self.details[key] for key in sorted(self.details)
            }
        return result


@dataclass(frozen=True)
class JapaneseSourceContribution:
    edge_index: int
    diphone: str
    source_diphone: str
    left_phone: str
    right_phone: str
    left_mora_index: Optional[int]
    right_mora_index: Optional[int]
    target_start: float
    target_boundary: float
    target_end: float
    source_kind: str
    selection_reason: str
    candidate_id: Optional[str] = None
    candidate_edge_offset: Optional[int] = None
    role: Optional[str] = None
    family: Optional[str] = None
    source_alias: Optional[str] = None
    source_wav: Optional[str] = None
    source_oto_path: Optional[str] = None
    source_oto_line: Optional[int] = None
    source_start: Optional[float] = None
    source_phone_boundary: Optional[float] = None
    source_end: Optional[float] = None
    shared_anchor: Optional[float] = None
    geometry_method: Optional[str] = None
    moraic_nasal_allophone: Optional[str] = None
    oto_timing_ms: Mapping[str, object] = field(default_factory=dict)
    source_components: tuple[Mapping[str, object], ...] = ()
    alternative_candidate_ids: tuple[str, ...] = ()
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "edge_index": self.edge_index,
            "diphone": self.diphone,
            "source_diphone": self.source_diphone,
            "linguistic_phones": [self.left_phone, self.right_phone],
            "mora_indices": [self.left_mora_index, self.right_mora_index],
            "target_interval": {
                "start": self.target_start,
                "phone_boundary": self.target_boundary,
                "end": self.target_end,
            },
            "source_kind": self.source_kind,
            "selection_reason": self.selection_reason,
            "alternative_candidate_ids": list(self.alternative_candidate_ids),
        }
        optional = {
            "candidate_id": self.candidate_id,
            "candidate_edge_offset": self.candidate_edge_offset,
            "role": self.role,
            "family": self.family,
            "source_alias": self.source_alias,
            "source_wav": self.source_wav,
            "source_oto_path": self.source_oto_path,
            "source_oto_line": self.source_oto_line,
            "shared_anchor": self.shared_anchor,
            "geometry_method": self.geometry_method,
            "moraic_nasal_allophone": self.moraic_nasal_allophone,
            "fallback_reason": self.fallback_reason,
        }
        for key, value in optional.items():
            if value is not None:
                result[key] = value
        if self.source_start is not None:
            result["source_slice"] = {
                "start": self.source_start,
                "phone_boundary": self.source_phone_boundary,
                "end": self.source_end,
            }
        if self.oto_timing_ms:
            result["oto_timing_ms"] = {
                key: self.oto_timing_ms[key]
                for key in sorted(self.oto_timing_ms)
            }
        if self.source_components:
            result["source_components"] = [
                dict(component) for component in self.source_components
            ]
        return result


@dataclass(frozen=True)
class JapaneseSourceContributionPlan:
    bank_type: str
    phones: tuple[str, ...]
    contributions: tuple[JapaneseSourceContribution, ...]
    diagnostics: tuple[JapaneseAssemblyDiagnostic, ...]
    schema_version: int = ASSEMBLY_SCHEMA_VERSION
    schema_status: str = ASSEMBLY_SCHEMA_STATUS

    @property
    def hidden_silence_count(self) -> int:
        return sum(
            item.source_kind == "hidden_silence_fallback"
            for item in self.contributions
        )

    @property
    def fallback_count(self) -> int:
        return sum(item.fallback_reason is not None for item in self.contributions)

    @property
    def all_spoken_edges_sourced(self) -> bool:
        return not any(
            item.source_kind in {"missing", "hidden_silence_fallback"}
            for item in self.contributions
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "kind": "japanese_source_contribution_plan",
            "bank_type": self.bank_type,
            "phones": list(self.phones),
            "all_spoken_edges_sourced": self.all_spoken_edges_sourced,
            "hidden_silence_count": self.hidden_silence_count,
            "fallback_count": self.fallback_count,
            "contributions": [item.to_dict() for item in self.contributions],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


def _bank_type(runtime: Mapping[str, object]) -> str:
    configuration = runtime.get("voice_configuration")
    if isinstance(configuration, Mapping):
        value = str(configuration.get("bank_type") or "").casefold()
        if value in PRIMARY_ROLES_BY_CONFIGURATION:
            return value
    value = str(runtime.get("alias_system") or "").casefold()
    return value if value in PRIMARY_ROLES_BY_CONFIGURATION else "unknown"


def _target_geometry(plan) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    centers: list[float] = []
    cursor = 0.0
    for segment in plan.segments:
        starts.append(cursor)
        centers.append(cursor + float(segment.duration) / 2.0)
        cursor += float(segment.duration)
    return starts, centers


def _selected_choice(
    choices: Sequence[Mapping[str, object]],
    wanted: Optional[str],
    wanted_reason: str,
    outer_left: str,
    outer_right: str,
    left_phone: str,
    right_phone: str,
    moraic_nasal_routing: Mapping[str, object],
) -> tuple[Optional[Mapping[str, object]], str]:
    if wanted:
        for choice in choices:
            if str(choice.get("left_name") or "") == wanted:
                return choice, wanted_reason
    return select_automatic_choice(
        choices,
        outer_left,
        outer_right,
        left_phone,
        right_phone,
        moraic_nasal_routing,
    ), "automatic"


def create_source_contribution_plan(
    plan,
    runtime_metadata: Mapping[str, object],
    *,
    selected_units: Mapping[int, str] | None = None,
) -> JapaneseSourceContributionPlan:
    """Resolve every canonical edge to its exact recording and source slice."""
    supported = tuple(str(item) for item in (
        runtime_metadata.get("supported_languages") or ()
    ))
    if runtime_metadata.get("language") != "ja" and "ja" not in supported:
        raise ValueError("runtime metadata is not Japanese")
    phones = tuple(str(phone) for phone in plan.phones)
    starts, centers = _target_geometry(plan)
    alternatives = dict(runtime_metadata.get("alternatives") or {})
    index = dict(runtime_metadata.get("index") or {})
    special_resolution = resolve_special_phone_sequence(
        phones,
        metadata=runtime_metadata,
        available_diphones=index,
    )
    source_phones = special_resolution.render_phones
    selected = {
        int(key): str(value)
        for key, value in dict(selected_units or {}).items()
    }
    overrides = {
        int(key): str(value)
        for key, value in dict(getattr(plan, "unit_overrides", {}) or {}).items()
    }
    bank_type = _bank_type(runtime_metadata)
    primary = PRIMARY_ROLES_BY_CONFIGURATION.get(bank_type, frozenset())
    contributions: list[JapaneseSourceContribution] = []
    diagnostics: list[JapaneseAssemblyDiagnostic] = []
    for unresolved in special_resolution.unresolved:
        diagnostics.append(JapaneseAssemblyDiagnostic(
            code="special_phone_without_source",
            message=(
                f"{unresolved.phone} could not be resolved to a source "
                f"phone ({unresolved.status})."
            ),
            severity="error",
            edge_index=max(0, int(unresolved.index) - 1),
            details={
                "missing_diphones": list(unresolved.missing_diphones),
                "required_diphones": list(unresolved.required_diphones),
            },
        ))

    for edge_index, (left, right) in enumerate(zip(phones, phones[1:])):
        diphone = f"{left}-{right}"
        source_left = source_phones[edge_index]
        source_right = source_phones[edge_index + 1]
        source_diphone = f"{source_left}-{source_right}"
        choices = tuple(alternatives.get(source_diphone) or ())
        if edge_index in overrides:
            wanted = overrides[edge_index]
            wanted_reason = "manual_override"
        elif edge_index in selected:
            wanted = selected[edge_index]
            wanted_reason = "runtime_selected"
        else:
            wanted = None
            wanted_reason = "automatic"
        choice, selection_reason = _selected_choice(
            choices,
            wanted,
            wanted_reason,
            source_phones[edge_index - 1] if edge_index else "*",
            (
                source_phones[edge_index + 2]
                if edge_index + 2 < len(source_phones) else "*"
            ),
            source_left,
            source_right,
            dict(runtime_metadata.get("moraic_nasal_routing") or {}),
        )
        left_segment = plan.segments[edge_index]
        right_segment = plan.segments[edge_index + 1]
        target_values = {
            "edge_index": edge_index,
            "diphone": diphone,
            "source_diphone": source_diphone,
            "left_phone": left,
            "right_phone": right,
            "left_mora_index": left_segment.mora_index,
            "right_mora_index": right_segment.mora_index,
            "target_start": round(centers[edge_index], 6),
            "target_boundary": round(starts[edge_index + 1], 6),
            "target_end": round(centers[edge_index + 1], 6),
            "alternative_candidate_ids": tuple(
                str(item.get("candidate_id") or item.get("id") or "")
                for item in choices
            ),
        }
        if choice is None:
            row = index.get(source_diphone)
            intentional_pause = left in _SILENCE and right in _SILENCE
            if intentional_pause:
                source_kind = "intentional_pause"
                reason = "protected pause edge"
            elif isinstance(row, (list, tuple)) and row:
                source_kind = "hidden_silence_fallback"
                reason = "generated silence index replaced a spoken transition"
            else:
                source_kind = "missing"
                reason = "no recording or generated fallback exists"
            contribution = JapaneseSourceContribution(
                **target_values,
                source_kind=source_kind,
                selection_reason=(
                    "structural" if intentional_pause else "missing"
                ),
                fallback_reason=(
                    None if intentional_pause else reason
                ),
            )
            contributions.append(contribution)
            if source_kind in {"hidden_silence_fallback", "missing"}:
                diagnostics.append(JapaneseAssemblyDiagnostic(
                    code="spoken_edge_without_source",
                    message=(
                        f"{diphone} has no audible source contribution; "
                        "Festival would substitute its default silence."
                    ),
                    severity="error",
                    edge_index=edge_index,
                    details={"reason": reason},
                ))
            continue

        role = str(choice.get("role") or "")
        fallback_reason = (
            str(choice.get("fallback_reason"))
            if choice.get("fallback_reason") else None
        )
        if fallback_reason is not None:
            diagnostics.append(JapaneseAssemblyDiagnostic(
                code="generated_transition_fallback",
                message=f"{diphone} uses a generated audible transition bridge.",
                severity="warning",
                edge_index=edge_index,
                details={"bank_type": bank_type, "role": role},
            ))
        elif selection_reason != "manual_override" and role == "vowel_blend":
            fallback_reason = (
                "No exact vowel transition was available; used the bank's "
                "explicit * V vowel-blend source."
            )
            diagnostics.append(JapaneseAssemblyDiagnostic(
                code="vowel_blend_fallback",
                message=f"{diphone} uses an explicit * V blend source.",
                severity="warning",
                edge_index=edge_index,
                details={"bank_type": bank_type, "role": role},
            ))
        elif (
            selection_reason != "manual_override"
            and role == "structural_consonant_hold"
        ):
            pass
        elif (
            selection_reason != "manual_override"
            and role == "vcv_mora"
            and str(choice.get("geometry_method") or "")
            == "oto_preutterance_vcv_vowel"
        ):
            # A one-phone VCV source is the exact recorded V-V or V-N
            # transition.  CVVC has no separate consonant VC+CV pair for
            # this edge, so this is structural material rather than a
            # cross-configuration fallback.
            pass
        elif selection_reason != "manual_override" and primary and role not in primary:
            fallback_reason = (
                f"{role} is fallback material for explicit {bank_type} assembly"
            )
            diagnostics.append(JapaneseAssemblyDiagnostic(
                code="cross_configuration_fallback",
                message=f"{diphone} uses {role} fallback material.",
                severity="warning",
                edge_index=edge_index,
                details={"bank_type": bank_type, "role": role},
            ))
        source_slice = choice.get("source_slice")
        if not isinstance(source_slice, Mapping):
            source_slice = {
                "start": choice.get("start"),
                "phone_boundary": choice.get("mid"),
                "end": choice.get("end"),
            }
        oto_timing = choice.get("oto_timing_ms")
        if not isinstance(oto_timing, Mapping):
            oto_timing = {}
        source_components = choice.get("source_components")
        if not isinstance(source_components, (list, tuple)):
            source_components = ()
        contribution = JapaneseSourceContribution(
            **target_values,
            source_kind=(
                "generated_fallback"
                if role == "generated_cv_bridge" else "recording"
            ),
            selection_reason=selection_reason,
            candidate_id=str(
                choice.get("candidate_id") or choice.get("id") or ""
            ),
            candidate_edge_offset=int(choice.get("edge_offset") or 0),
            role=role,
            family=str(choice.get("family") or ""),
            source_alias=str(choice.get("alias") or ""),
            source_wav=str(choice.get("wav") or ""),
            source_oto_path=str(choice.get("oto_file") or ""),
            source_oto_line=int(choice.get("oto_line") or 0),
            source_start=(
                float(source_slice["start"])
                if source_slice.get("start") is not None else None
            ),
            source_phone_boundary=(
                float(source_slice["phone_boundary"])
                if source_slice.get("phone_boundary") is not None else None
            ),
            source_end=(
                float(source_slice["end"])
                if source_slice.get("end") is not None else None
            ),
            shared_anchor=(
                float(choice["shared_anchor"])
                if choice.get("shared_anchor") is not None else None
            ),
            geometry_method=str(choice.get("geometry_method") or ""),
            moraic_nasal_allophone=(
                str(choice.get("moraic_nasal_allophone"))
                if choice.get("moraic_nasal_allophone") else None
            ),
            oto_timing_ms=dict(oto_timing),
            source_components=tuple(
                dict(component) for component in source_components
                if isinstance(component, Mapping)
            ),
            fallback_reason=fallback_reason,
        )
        contributions.append(contribution)

        if contribution.shared_anchor is not None:
            endpoint = (
                contribution.source_end
                if int(choice.get("edge_offset") or 0) == -1
                else contribution.source_start
            )
            if endpoint is None or abs(endpoint - contribution.shared_anchor) > 1e-5:
                diagnostics.append(JapaneseAssemblyDiagnostic(
                    code="shared_anchor_mismatch",
                    message=(
                        f"{diphone} does not meet its declared phone-center "
                        "anchor."
                    ),
                    severity="error",
                    edge_index=edge_index,
                ))

    for left, right in zip(contributions, contributions[1:]):
        if (
            left.candidate_id
            and left.candidate_id == right.candidate_id
            and left.role in _PAIRED_ROLES
            and right.role == left.role
            and left.candidate_edge_offset == -1
            and right.candidate_edge_offset == 0
            and left.shared_anchor is not None
            and right.shared_anchor is not None
        ):
            if left.source_end is None or right.source_start is None:
                continue
            difference = round(right.source_start - left.source_end, 6)
            if abs(difference) > 1e-5:
                code = "paired_source_gap" if difference > 0 else "duplicate_consonant_overlap"
                diagnostics.append(JapaneseAssemblyDiagnostic(
                    code=code,
                    message=(
                        "Adjacent halves from one UTAU alias do not share one "
                        "phone-center boundary."
                    ),
                    severity="error",
                    edge_index=right.edge_index,
                    details={"source_delta_seconds": difference},
                ))
        elif (
            left.role in _PAIRED_ROLES
            and left.role == right.role
            and left.right_phone == right.left_phone
            and left.candidate_edge_offset == -1
            and right.candidate_edge_offset == 0
            and left.shared_anchor is not None
            and right.shared_anchor is not None
            and left.candidate_id != right.candidate_id
        ):
            diagnostics.append(JapaneseAssemblyDiagnostic(
                code="paired_candidate_mismatch",
                message=(
                    "Two halves of one contextual source role selected "
                    "different aliases."
                ),
                severity="error",
                edge_index=right.edge_index,
                details={
                    "left_candidate_id": left.candidate_id or "",
                    "right_candidate_id": right.candidate_id or "",
                },
            ))

    return JapaneseSourceContributionPlan(
        bank_type=bank_type,
        phones=phones,
        contributions=tuple(contributions),
        diagnostics=tuple(diagnostics),
    )
