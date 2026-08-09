"""Japanese linguistic editing state shared by the GUI and project files.

The module deliberately contains no Qt code.  It turns the immutable Phase 1
utterance into an editable overlay, applies that overlay before the Phase 3
synthesis planner, and keeps the existing continuous F0 curve as the final
authority.  Source-bank analysis is read-only; profile writes are explicitly
guarded by :mod:`japanese_profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import json
import math
from pathlib import Path
from typing import Mapping, Optional

import japanese_candidates as jc
import japanese_profiles as jp
import pitch_domain as pitch_domain
from japanese_models import (
    JapaneseAccentPhrase,
    JapaneseFrontendDiagnostic,
    JapaneseMora,
    JapanesePhone,
    JapanesePhrase,
    JapaneseUtterance,
)
from japanese_synthesis import (
    JapaneseF0Target,
    JapaneseSynthesisDiagnostic,
    JapaneseSynthesisPlan,
    create_synthesis_plan,
)


JAPANESE_EDIT_SCHEMA_VERSION = 1
JAPANESE_EDIT_SCHEMA_STATUS = "phase4-provisional"
PITCH_OFFSET_MIN_CENTS = -600
PITCH_OFFSET_MAX_CENTS = 600
PITCH_MIN_HZ = 50.0
PITCH_MAX_HZ = 500.0
MORA_VOICING_MIN = 0.0
MORA_VOICING_MAX = 1.0


def _integer_map(value: object, converter) -> dict:
    result = {}
    for key, item in dict(value or {}).items():
        try:
            index = int(key)
            result[index] = converter(item)
        except (TypeError, ValueError):
            continue
    return result


def new_edit_state(
    utterance: Optional[JapaneseUtterance] = None,
    *,
    frontend_mode: str = "auto",
) -> dict:
    """Return a serializable Japanese sentence overlay.

    Accent phrase and mora indexes are the global zero-based indexes from the
    canonical utterance.  The schema remains explicitly provisional.
    """
    return {
        "schema_version": JAPANESE_EDIT_SCHEMA_VERSION,
        "schema_status": JAPANESE_EDIT_SCHEMA_STATUS,
        "frontend_mode": str(frontend_mode or "auto"),
        "utterance": utterance.to_dict() if utterance is not None else None,
        "accent_overrides": {},
        "accent_phrase_boundaries": {},
        "phrase_overrides": {},
        "mora_pitch_offsets_cents": {},
        "mora_voicing_overrides": {},
        "manual_candidate_overrides": {},
        "continuous_pitch_authority": "pitch_override",
        "last_plan": None,
        "bank_analysis": None,
        "profile_path": "",
        "needs_voice_rebuild": False,
        "baseline_provider": "structural",
        "external_hts_trajectory": "",
        "dynamic_multipitch": False,
        "voice_color": "",
    }


def normalize_edit_state(value: object) -> dict:
    """Migrate missing/older Japanese project data without touching English.

    Phase 4 is the first committed project schema.  Earlier experimental rows
    used ``accent_edits`` and ``mora_pitch_offsets``; both are accepted as a
    compatibility layer and normalized to the current names.
    """
    source = copy.deepcopy(value) if isinstance(value, Mapping) else {}
    state = new_edit_state(frontend_mode=str(source.get(
        "frontend_mode", "auto"
    )))
    utterance = source.get("utterance")
    state["utterance"] = copy.deepcopy(utterance) \
        if isinstance(utterance, Mapping) else None

    accent_source = source.get("accent_overrides")
    if accent_source is None:
        accent_source = source.get("accent_edits")
    state["accent_overrides"] = {
        str(index): dict(item)
        for index, item in _integer_map(
            accent_source,
            lambda row: dict(row) if isinstance(row, Mapping) else {},
        ).items()
    }
    boundary_rows = _integer_map(
        source.get("accent_phrase_boundaries"),
        lambda row: list(row) if isinstance(row, (list, tuple)) else [],
    )
    state["accent_phrase_boundaries"] = {
        str(phrase_index): sorted({
            int(mora_index)
            for mora_index in values
            if isinstance(mora_index, (int, float, str))
            and str(mora_index).lstrip("-").isdigit()
            and int(mora_index) >= 0
        })
        for phrase_index, values in boundary_rows.items()
    }
    state["phrase_overrides"] = {
        str(index): dict(item)
        for index, item in _integer_map(
            source.get("phrase_overrides"),
            lambda row: dict(row) if isinstance(row, Mapping) else {},
        ).items()
    }

    pitch_source = source.get("mora_pitch_offsets_cents")
    if pitch_source is None:
        pitch_source = source.get("mora_pitch_offsets")
    offsets = _integer_map(pitch_source, float)
    state["mora_pitch_offsets_cents"] = {
        str(index): int(round(max(
            PITCH_OFFSET_MIN_CENTS,
            min(PITCH_OFFSET_MAX_CENTS, value),
        )))
        for index, value in offsets.items()
        if math.isfinite(value) and abs(value) >= 0.5
    }
    voicing = _integer_map(source.get("mora_voicing_overrides"), float)
    state["mora_voicing_overrides"] = {
        str(index): round(max(
            MORA_VOICING_MIN,
            min(MORA_VOICING_MAX, value),
        ), 6)
        for index, value in voicing.items()
        if math.isfinite(value)
    }
    state["manual_candidate_overrides"] = {
        str(index): str(candidate_id)
        for index, candidate_id in _integer_map(
            source.get("manual_candidate_overrides"), str
        ).items()
        if candidate_id
    }
    state["continuous_pitch_authority"] = "pitch_override"
    last_plan = source.get("last_plan")
    state["last_plan"] = copy.deepcopy(last_plan) \
        if isinstance(last_plan, Mapping) else None
    analysis = source.get("bank_analysis")
    state["bank_analysis"] = copy.deepcopy(analysis) \
        if isinstance(analysis, Mapping) else None
    state["profile_path"] = str(source.get("profile_path") or "")
    state["needs_voice_rebuild"] = bool(
        source.get("needs_voice_rebuild", False)
    )
    baseline = str(source.get("baseline_provider") or "structural")
    state["baseline_provider"] = baseline \
        if baseline in {"structural", "openjtalk_labels", "external_hts"} \
        else "structural"
    state["external_hts_trajectory"] = str(
        source.get("external_hts_trajectory") or ""
    )
    state["dynamic_multipitch"] = bool(
        source.get("dynamic_multipitch", False)
    )
    state["voice_color"] = str(source.get("voice_color") or "")
    return state


def reconcile_analyzed_utterance(
    value: object,
    utterance: JapaneseUtterance,
) -> dict:
    """Attach analysis while keeping occurrence edits only for identical text.

    Accent, mora, and candidate overrides are indexed into one utterance.  A
    newly typed sentence must therefore start with a clean linguistic overlay.
    Bank-analysis and profile settings belong to the voice and remain useful.
    """
    state = normalize_edit_state(value)
    current = state.get("utterance")
    if isinstance(current, Mapping) and str(current.get("source_text") or "") \
            == utterance.source_text:
        state["utterance"] = utterance.to_dict()
        return state

    fresh = new_edit_state(
        utterance, frontend_mode=str(state.get("frontend_mode") or "auto")
    )
    for key in (
        "bank_analysis", "profile_path", "needs_voice_rebuild",
        "baseline_provider", "external_hts_trajectory",
        "dynamic_multipitch", "voice_color",
    ):
        fresh[key] = copy.deepcopy(state.get(key))
    return fresh


def _diagnostic(value: Mapping[str, object]) -> JapaneseFrontendDiagnostic:
    return JapaneseFrontendDiagnostic(
        code=str(value.get("code") or "project_diagnostic"),
        message=str(value.get("message") or ""),
        severity=str(value.get("severity") or "warning"),
        action=(str(value["action"]) if value.get("action") else None),
        source_start=(int(value["source_start"])
                      if value.get("source_start") is not None else None),
        source_end=(int(value["source_end"])
                    if value.get("source_end") is not None else None),
        frontend=(str(value["frontend"])
                  if value.get("frontend") else None),
        confidence=(float(value["confidence"])
                    if value.get("confidence") is not None else None),
        raw_data=dict(value.get("raw_data") or {}),
    )


def utterance_from_dict(value: Mapping[str, object]) -> JapaneseUtterance:
    """Decode the provisional Phase 1 model stored in a project row."""
    raw = dict(value or {})
    phone_rows = list(raw.get("phones") or [])
    phones = tuple(JapanesePhone(
        index=int(row.get("index", position)),
        symbol=str(row.get("symbol") or "unknown"),
        raw_symbol=(str(row["raw_symbol"])
                    if row.get("raw_symbol") is not None else None),
        phone_type=str(row.get("phone_type") or "consonant"),
        phrase_index=(int(row["phrase_index"])
                      if row.get("phrase_index") is not None else None),
        accent_phrase_index=(int(row["accent_phrase_index"])
                             if row.get("accent_phrase_index") is not None
                             else None),
        mora_index=(int(row["mora_index"])
                    if row.get("mora_index") is not None else None),
        devoiced=(bool(row["devoiced"])
                  if row.get("devoiced") is not None else None),
        is_pause=bool(row.get("is_pause", False)),
        is_silence=bool(row.get("is_silence", False)),
        unknown=bool(row.get("unknown", False)),
        raw_label=(str(row["raw_label"])
                   if row.get("raw_label") is not None else None),
        confidence=float(row.get("confidence", 1.0)),
    ) for position, row in enumerate(phone_rows))
    by_phone_index = {phone.index: phone for phone in phones}

    phrases = []
    for phrase_row in raw.get("phrases") or []:
        accent_phrases = []
        for accent_row in phrase_row.get("accent_phrases") or []:
            moras = []
            for mora_row in accent_row.get("moras") or []:
                mora_phones = tuple(
                    by_phone_index.get(int(phone_row.get("index", -1)))
                    for phone_row in mora_row.get("phones") or []
                    if int(phone_row.get("index", -1)) in by_phone_index
                )
                if not mora_phones:
                    raise ValueError("stored Japanese mora has no known phones")
                moras.append(JapaneseMora(
                    index=int(mora_row.get("index", len(moras))),
                    phrase_index=int(mora_row.get(
                        "phrase_index", phrase_row.get("index", 0)
                    )),
                    accent_phrase_index=int(mora_row.get(
                        "accent_phrase_index", accent_row.get("index", 0)
                    )),
                    surface=str(mora_row.get("surface") or ""),
                    reading=str(mora_row.get("reading") or ""),
                    phones=mora_phones,
                    consonant=(str(mora_row["consonant"])
                               if mora_row.get("consonant") is not None
                               else None),
                    vowel=(str(mora_row["vowel"])
                           if mora_row.get("vowel") is not None else None),
                    special_mora=(str(mora_row["special_mora"])
                                  if mora_row.get("special_mora") is not None
                                  else None),
                    devoiced=(bool(mora_row["devoiced"])
                              if mora_row.get("devoiced") is not None
                              else None),
                    confidence=float(mora_row.get("confidence", 1.0)),
                    provenance=dict(mora_row.get("provenance") or {}),
                ))
            accent_phrases.append(JapaneseAccentPhrase(
                index=int(accent_row.get("index", len(accent_phrases))),
                phrase_index=int(accent_row.get(
                    "phrase_index", phrase_row.get("index", 0)
                )),
                moras=tuple(moras),
                accent_state=str(accent_row.get("accent_state") or "unknown"),
                accent_nucleus=(int(accent_row["accent_nucleus"])
                                if accent_row.get("accent_nucleus") is not None
                                else None),
                interrogative=bool(accent_row.get("interrogative", False)),
                boundary_strength=int(accent_row.get("boundary_strength", 1)),
                confidence=float(accent_row.get("confidence", 1.0)),
                provenance=dict(accent_row.get("provenance") or {}),
            ))
        phrases.append(JapanesePhrase(
            index=int(phrase_row.get("index", len(phrases))),
            surface=str(phrase_row.get("surface") or ""),
            normalized_reading=str(
                phrase_row.get("normalized_reading") or ""
            ),
            accent_phrases=tuple(accent_phrases),
            punctuation_after=str(phrase_row.get("punctuation_after") or ""),
            boundary_strength=int(phrase_row.get("boundary_strength", 3)),
            interrogative=bool(phrase_row.get("interrogative", False)),
            phone_indices=tuple(int(item) for item in
                                (phrase_row.get("phone_indices") or ())),
            confidence=float(phrase_row.get("confidence", 1.0)),
            provenance=dict(phrase_row.get("provenance") or {}),
        ))
    return JapaneseUtterance(
        source_text=str(raw.get("source_text") or ""),
        normalized_reading=str(raw.get("normalized_reading") or ""),
        phrases=tuple(phrases),
        phones=phones,
        diagnostics=tuple(_diagnostic(item) for item in
                          (raw.get("diagnostics") or ())),
        frontend_name=str(raw.get("frontend_name") or "project"),
        frontend_version=(str(raw["frontend_version"])
                          if raw.get("frontend_version") is not None else None),
        confidence=float(raw.get("confidence", 1.0)),
        provenance=dict(raw.get("provenance") or {}),
    )


def _apply_accent_phrase_boundaries(
    utterance: JapaneseUtterance,
    boundaries_by_phrase: Mapping[int, tuple[int, ...]],
) -> JapaneseUtterance:
    """Rebuild accent phrases from global mora-start indexes.

    A boundary value is the first mora of the following accent phrase.  The
    first mora of a linguistic phrase is implicit and therefore never stored.
    Unchanged phrases retain their analyzed indexes and accent state.  A split
    or merge receives a deterministic synthetic index based on its first mora;
    it keeps an accent only when exactly one analyzed nucleus falls inside the
    new group.
    """
    if not boundaries_by_phrase:
        return utterance
    original_indexes = [item.index for item in utterance.accent_phrases]
    synthetic_base = max(original_indexes, default=-1) + 1
    phone_updates: dict[int, JapanesePhone] = {}
    rebuilt_phrases = []

    for phrase in utterance.phrases:
        moras = list(phrase.moras)
        if len(moras) < 2 or phrase.index not in boundaries_by_phrase:
            rebuilt_phrases.append(phrase)
            continue
        valid_starts = sorted({
            int(index) for index in boundaries_by_phrase[phrase.index]
            if any(mora.index == int(index) for mora in moras[1:])
        })
        starts = [0] + [
            position for position, mora in enumerate(moras)
            if mora.index in valid_starts
        ] + [len(moras)]
        starts = sorted(set(starts))
        old_by_mora = {
            mora.index: accent
            for accent in phrase.accent_phrases
            for mora in accent.moras
        }
        old_nuclei = {
            accent.moras[accent.accent_nucleus].index: accent
            for accent in phrase.accent_phrases
            if accent.accent_state == "accented"
            and accent.accent_nucleus is not None
            and 0 <= accent.accent_nucleus < len(accent.moras)
        }
        rebuilt_accents = []
        for first, last in zip(starts, starts[1:]):
            group = moras[first:last]
            if not group:
                continue
            exact = next((
                accent for accent in phrase.accent_phrases
                if tuple(mora.index for mora in accent.moras)
                == tuple(mora.index for mora in group)
            ), None)
            accent_index = (
                exact.index if exact is not None
                else synthetic_base + group[0].index
            )
            rebuilt_phones = []
            rebuilt_moras = []
            for mora in group:
                mora_phones = []
                for phone in mora.phones:
                    changed = replace(
                        phone, accent_phrase_index=accent_index)
                    phone_updates[changed.index] = changed
                    rebuilt_phones.append(changed)
                rebuilt_moras.append(replace(
                    mora,
                    accent_phrase_index=accent_index,
                    phones=tuple(rebuilt_phones[-len(mora_phones):]),
                ))

            if exact is not None:
                accent_state = exact.accent_state
                nucleus = exact.accent_nucleus
                interrogative = exact.interrogative
                boundary_strength = exact.boundary_strength
                confidence = exact.confidence
                provenance = dict(exact.provenance)
            else:
                nuclei = [
                    mora.index for mora in group
                    if mora.index in old_nuclei
                ]
                old_accents = {
                    old_by_mora[mora.index].index for mora in group
                    if mora.index in old_by_mora
                }
                old_rows = [
                    accent for accent in phrase.accent_phrases
                    if accent.index in old_accents
                ]
                if len(nuclei) == 1:
                    accent_state = "accented"
                    nucleus = next(
                        position for position, mora in enumerate(group)
                        if mora.index == nuclei[0]
                    )
                elif old_rows and all(
                        row.accent_state == "unaccented"
                        for row in old_rows):
                    accent_state = "unaccented"
                    nucleus = None
                else:
                    accent_state = "unknown"
                    nucleus = None
                interrogative = any(row.interrogative for row in old_rows)
                boundary_strength = (
                    old_rows[-1].boundary_strength if old_rows else 1
                )
                confidence = min(
                    (row.confidence for row in old_rows), default=1.0
                )
                provenance = {
                    "source": "manual_accent_phrase_structure",
                    "merged_analyzed_phrase_indexes": sorted(old_accents),
                }
            rebuilt_accents.append(JapaneseAccentPhrase(
                index=accent_index,
                phrase_index=phrase.index,
                moras=tuple(rebuilt_moras),
                accent_state=accent_state,
                accent_nucleus=nucleus,
                interrogative=interrogative,
                boundary_strength=boundary_strength,
                confidence=confidence,
                provenance=provenance,
            ))
        rebuilt_phrases.append(replace(
            phrase, accent_phrases=tuple(rebuilt_accents)
        ))

    phones = tuple(phone_updates.get(phone.index, phone)
                   for phone in utterance.phones)
    return replace(
        utterance, phrases=tuple(rebuilt_phrases), phones=phones
    )


def apply_linguistic_edits(
    utterance: JapaneseUtterance,
    state: Mapping[str, object],
) -> JapaneseUtterance:
    normalized = normalize_edit_state(state)
    utterance = _apply_accent_phrase_boundaries(
        utterance,
        _integer_map(
            normalized["accent_phrase_boundaries"],
            lambda values: tuple(int(value) for value in values),
        ),
    )
    accent_overrides = _integer_map(
        normalized["accent_overrides"], lambda item: dict(item)
    )
    phrase_overrides = _integer_map(
        normalized["phrase_overrides"], lambda item: dict(item)
    )
    phrases = []
    for phrase in utterance.phrases:
        edited_accents = []
        for accent_phrase in phrase.accent_phrases:
            override = accent_overrides.get(accent_phrase.index, {})
            accent_state = str(override.get(
                "accent_state", accent_phrase.accent_state
            ))
            nucleus = override.get(
                "accent_nucleus", accent_phrase.accent_nucleus
            )
            if accent_state != "accented":
                nucleus = None
            elif nucleus is None and accent_phrase.moras:
                nucleus = 0
            nucleus = int(nucleus) if nucleus is not None else None
            if nucleus is not None:
                nucleus = max(0, min(len(accent_phrase.moras) - 1, nucleus))
            edited_accents.append(replace(
                accent_phrase,
                accent_state=accent_state,
                accent_nucleus=nucleus,
                interrogative=bool(override.get(
                    "interrogative", accent_phrase.interrogative
                )),
                boundary_strength=max(0, min(3, int(override.get(
                    "boundary_strength", accent_phrase.boundary_strength
                )))),
            ))
        phrase_override = phrase_overrides.get(phrase.index, {})
        phrases.append(replace(
            phrase,
            accent_phrases=tuple(edited_accents),
            interrogative=bool(phrase_override.get(
                "interrogative", phrase.interrogative
            )),
            boundary_strength=max(0, min(3, int(phrase_override.get(
                "boundary_strength", phrase.boundary_strength
            )))),
        ))
    return replace(utterance, phrases=tuple(phrases))


def apply_mora_pitch_offsets(
    plan: JapaneseSynthesisPlan,
    offsets_cents: Mapping[object, object],
) -> JapaneseSynthesisPlan:
    offsets = _integer_map(offsets_cents, float)
    targets = []
    for target in plan.f0_targets:
        cents = max(PITCH_OFFSET_MIN_CENTS, min(
            PITCH_OFFSET_MAX_CENTS,
            float(offsets.get(target.mora_index, 0.0)),
        ))
        shifted_log_f0 = target.log_f0 + cents / 1200.0
        shifted_hz = pitch_domain.clamp_hz(
            pitch_domain.log_f0_to_hz(shifted_log_f0),
            PITCH_MIN_HZ, PITCH_MAX_HZ)
        targets.append(replace(
            target,
            log_f0=round(pitch_domain.hz_to_log_f0(shifted_hz), 12),
            components_semitones={
                **dict(target.components_semitones),
                "manual_mora_offset": round(cents / 100.0, 6),
            },
            kind=(target.kind if abs(cents) < 0.5 else
                  target.kind + "+mora_offset"),
        ))
    return replace(plan, f0_targets=tuple(targets))


def create_edited_plan(
    utterance: JapaneseUtterance,
    state: Mapping[str, object],
    *,
    runtime_metadata: Mapping[str, object] | Path | str | None = None,
    base_pitch_hz: float = 180.0,
    speed: float = 1.0,
    phrase_pauses_ms: Mapping[str, object] | None = None,
    allow_experimental_routing: bool = False,
    duration_model: str | None = None,
) -> JapaneseSynthesisPlan:
    normalized = normalize_edit_state(state)
    edited = apply_linguistic_edits(utterance, normalized)
    plan = create_synthesis_plan(
        edited,
        runtime_metadata=runtime_metadata,
        manual_candidate_overrides=_integer_map(
            normalized["manual_candidate_overrides"], str
        ),
        base_pitch_hz=base_pitch_hz,
        speed=speed,
        phrase_pauses_ms=phrase_pauses_ms,
        duration_model=duration_model,
    )
    from japanese_refinements import (
        JapaneseRoutingPolicy,
        apply_baseline_trajectory,
        resolve_baseline_provider,
        route_dynamic_candidates,
    )

    baseline_mode = str(normalized.get("baseline_provider") or "structural")
    if baseline_mode != "structural":
        provider = resolve_baseline_provider(
            baseline_mode,
            external_path=normalized.get("external_hts_trajectory") or None,
        )
        result = provider.provide(
            edited, base_pitch_hz=base_pitch_hz, speed=speed
        )
        if result.trajectory is not None:
            plan = apply_baseline_trajectory(
                plan, result.trajectory,
                preserve_structural_f0=bool(
                    normalized["accent_overrides"]
                    or normalized["phrase_overrides"]
                ),
            )
        if result.diagnostics:
            plan = replace(plan, diagnostics=plan.diagnostics + tuple(
                JapaneseSynthesisDiagnostic(
                    code=item.code,
                    message=item.message,
                    severity=item.severity,
                    details=item.details,
                ) for item in result.diagnostics
            ))

    plan = apply_mora_pitch_offsets(
        plan, normalized["mora_pitch_offsets_cents"]
    )
    dynamic = bool(normalized.get("dynamic_multipitch"))
    color = str(normalized.get("voice_color") or "")
    if (dynamic or color) and not allow_experimental_routing:
        plan = replace(plan, diagnostics=plan.diagnostics + (
            JapaneseSynthesisDiagnostic(
                code="experimental_routing_disabled",
                message=(
                    "Stored multipitch or voice-color routing is disabled in "
                    "the stable workflow. Build that pitch or color as a "
                    "separate generated voice configuration."
                ),
                severity="info",
            ),
        ))
    elif dynamic or color:
        if isinstance(runtime_metadata, Mapping):
            runtime = dict(runtime_metadata)
        else:
            try:
                runtime = json.loads(
                    Path(runtime_metadata).read_text(encoding="utf-8")
                ) if runtime_metadata is not None else {}
            except (OSError, TypeError, ValueError):
                runtime = {}
        if runtime:
            plan = route_dynamic_candidates(
                plan, runtime,
                JapaneseRoutingPolicy(
                    dynamic_pitch=dynamic,
                    voice_color=color or None,
                ),
            )
        else:
            plan = replace(plan, diagnostics=plan.diagnostics + (
                JapaneseSynthesisDiagnostic(
                    code="dynamic_routing_metadata_unavailable",
                    message=(
                        "Dynamic Japanese pitch/color routing was requested, "
                        "but this generated voice has no compatible metadata."
                    ),
                ),
            ))
    return plan


def final_pitch_targets(
    generated_targets,
    continuous_override,
):
    """Resolve editor precedence; continuous points are always final."""
    override = [(float(time), float(hz))
                for time, hz in (continuous_override or [])]
    if override:
        return override
    return [(float(time), float(hz))
            for time, hz in (generated_targets or [])]


def invalidation_for_edit(kind: str) -> str:
    if str(kind) in {"profile", "alias_override", "bank_configuration"}:
        return "rebuild"
    if str(kind) in {
        "accent", "accent_structure", "question", "phrase_boundary",
        "mora_pitch", "mora_voicing",
        "candidate_override", "baseline_provider", "dynamic_multipitch",
        "voice_color", "external_hts_trajectory",
    }:
        return "rerender"
    return "none"


@dataclass(frozen=True)
class JapaneseBankAnalysis:
    source_path: str
    profile: jp.JapaneseBankProfile
    graph: jc.JapaneseCandidateGraph

    @property
    def unresolved(self):
        return tuple(candidate for candidate in self.graph.candidates
                     if candidate.role == "unresolved")

    def to_state_dict(self) -> dict:
        coverage = self.graph.coverage
        return {
            "schema_version": 1,
            "source_path": self.source_path,
            "source_scope": self.profile.source_scope,
            "configuration": self.profile.effective_configuration,
            "inference_confidence": self.profile.inference_confidence,
            "source_entry_count": coverage.source_entry_count,
            "candidate_count": coverage.candidate_count,
            "all_entries_traceable": coverage.all_entries_traceable,
            "unresolved_count": coverage.unresolved_count,
            "unresolved_rate": coverage.unresolved_rate,
            "role_counts": dict(coverage.role_counts),
            "family_counts": dict(coverage.family_counts),
            "missing_core_mora_any": list(coverage.missing_core_mora_any),
            "unresolved": [
                {
                    "candidate_id": item.candidate_id,
                    "alias": item.source.alias_raw,
                    "match_key": item.source.alias_match_key,
                    "oto_path": item.source.oto_path,
                    "oto_line": item.source.line,
                    "wav": item.source.wav_path,
                    "override_key": item.override_key or
                                    item.source.alias_match_key,
                    "reasons": list(item.reasons),
                }
                for item in self.unresolved
            ],
            "profile": self.profile.to_dict(),
        }


def analyze_bank(
    source: Path | str,
    *,
    profile: Optional[jp.JapaneseBankProfile] = None,
) -> JapaneseBankAnalysis:
    """Analyze a source bank without opening files for write access."""
    source_path = Path(source).expanduser().resolve()
    inferred = profile or jp.infer_bank_profile(source_path)
    graph = jc.compile_candidate_graph(source_path, profile=inferred)
    return JapaneseBankAnalysis(
        source_path=str(source_path), profile=inferred, graph=graph
    )


def profile_with_override(
    profile: jp.JapaneseBankProfile,
    key: str,
    override: jp.JapaneseAliasOverride,
) -> jp.JapaneseBankProfile:
    overrides = dict(profile.alias_overrides)
    overrides[str(key)] = override
    return replace(profile, alias_overrides=overrides)


def write_analysis_profile(
    analysis: JapaneseBankAnalysis,
    output: Path | str,
) -> None:
    jp.write_profile(
        analysis.profile,
        Path(output),
        source_root=analysis.graph._bank_root,
    )
