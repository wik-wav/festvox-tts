"""Versioned sentence-level log-F0 realization for Asaxi.

The Asaxi dictionary and morphology planner produce categorical mora-level
H/L instructions.  This module is the separate phonetic realization layer:
it places those instructions in a phrase-local register, carries target state
between phrases, realizes boundary tones over time, and approaches goals with
a duration-sensitive critically damped tracker.

No cumulative sentence-time or phrase-index frequency offset is permitted.
Later phrases may differ in *shape*, but the contextual components are
mean-centred within each phrase.  This preserves linguistic development
without the global frequency drift rejected during listening validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from cache_support import FileIdentityCache, deep_freeze
import pitch_domain


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "profiles"
    / "asaxi_pitch_model_v1.json"
)
_PITCH_MODEL_CACHE = FileIdentityCache(
    "asaxi-pitch-model", max_entries=4, max_bytes=8 * 1024 * 1024
)


@dataclass(frozen=True)
class AsaxiPitchModel:
    model_id: str
    model_version: int
    tone_goals_semitones: Mapping[str, float]
    declination_total_semitones: float
    later_phrase_declination_delta_semitones: float
    later_phrase_contrast_scale: float
    later_phrase_boundary_delta_semitones: float
    maximum_contextual_phrase_index: int
    boundary_goals_semitones: Mapping[str, float]
    boundary_reset_strengths: Mapping[str, float]
    boundary_minimum_seconds: float
    boundary_preferred_seconds: float
    boundary_maximum_seconds: float
    statement_fall_max_semitones: float
    response_time_seconds: float
    sample_step_seconds: float
    maximum_slew_semitones_per_second: float
    minimum_f0_hz: float
    maximum_f0_hz: float
    analysis_evidence: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "tone_goals_semitones",
            "boundary_goals_semitones",
            "boundary_reset_strengths",
            "analysis_evidence",
            "provenance",
        ):
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))

    def tone_goal(self, tone: str) -> float:
        key = str(tone or "").strip().upper()
        if key not in self.tone_goals_semitones:
            raise KeyError(f"Asaxi pitch tone is missing: {key}")
        return float(self.tone_goals_semitones[key])

    def boundary_goal(self, name: str) -> float:
        if name not in self.boundary_goals_semitones:
            raise KeyError(f"Asaxi boundary goal is missing: {name}")
        return float(self.boundary_goals_semitones[name])

    def reset_strength(self, boundary_tone: str) -> float:
        return float(self.boundary_reset_strengths.get(
            str(boundary_tone or ""), 0.0
        ))


@dataclass(frozen=True)
class AsaxiPitchRealization:
    targets: tuple[tuple[float, float], ...]
    trace: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", deep_freeze(self.trace))

    def trace_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy of the diagnostic trace."""
        return json.loads(json.dumps(self.trace, ensure_ascii=False))


@dataclass(frozen=True)
class _Goal:
    time: float
    semitones: float
    kind: str
    phrase_index: int
    mora_index: int | None = None


def _number_map(value: object, name: str) -> dict[str, float]:
    try:
        rows = dict(value or {})
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an object") from error
    try:
        return {str(key): float(item) for key, item in rows.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be numeric") from error


def _read_asaxi_pitch_model(source: Path) -> AsaxiPitchModel:
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Asaxi pitch model schema")
    if str(data.get("language") or "") != "asaxi":
        raise ValueError("Asaxi pitch model language must be asaxi")

    tones = _number_map(
        data.get("tone_goals_semitones"), "tone_goals_semitones")
    if not {"H", "L"}.issubset(tones):
        raise ValueError("Asaxi pitch model must define H and L goals")
    if not tones["H"] > tones["L"]:
        raise ValueError("Asaxi H must be higher than L")

    phrase = dict(data.get("phrase_model") or {})
    legacy_drift_keys = {
        "utterance_declination_semitones_per_second",
        "maximum_utterance_declination_semitones",
        "phrase_register_drop_semitones",
        "maximum_phrase_register_drops",
        "sentence_frequency_drift_semitones",
    }
    active_drift = {
        key: phrase[key] for key in legacy_drift_keys
        if key in phrase and float(phrase[key]) != 0.0
    }
    if active_drift:
        raise ValueError(
            "Asaxi pitch profiles may not contain cumulative frequency "
            "drift; use mean-centred phrase-shape parameters"
        )
    declination = float(phrase.get(
        "declination_total_semitones", 0.0))
    later_declination = float(phrase.get(
        "later_phrase_declination_delta_semitones", 0.0))
    later_contrast = float(phrase.get(
        "later_phrase_contrast_scale", 0.0))
    later_boundary = float(phrase.get(
        "later_phrase_boundary_delta_semitones", 0.0))
    maximum_context = int(phrase.get(
        "maximum_contextual_phrase_index", 0))
    if not -4.0 <= declination <= 2.0:
        raise ValueError("Asaxi phrase declination must be -4..2 semitones")
    if not -2.0 <= later_declination <= 2.0:
        raise ValueError("Asaxi later-phrase decline must be -2..2")
    if not -0.6 <= later_contrast <= 0.6:
        raise ValueError("Asaxi later-phrase contrast must be -0.6..0.6")
    if not -3.0 <= later_boundary <= 3.0:
        raise ValueError("Asaxi later-phrase boundary delta must be -3..3")
    if not 0 <= maximum_context <= 8:
        raise ValueError("Asaxi contextual phrase index cap must be 0..8")

    boundary_goals = _number_map(
        data.get("boundary_goals_semitones"),
        "boundary_goals_semitones",
    )
    required_boundaries = {"L%", "H-", "H%", "LH%_low", "LH%_high"}
    missing_boundaries = sorted(required_boundaries.difference(
        boundary_goals))
    if missing_boundaries:
        raise ValueError(
            "Asaxi boundary goals missing: " + ", ".join(missing_boundaries)
        )
    boundary = dict(data.get("boundary_model") or {})
    reset_strengths = _number_map(
        boundary.get("reset_strengths"), "boundary reset strengths")
    for name, strength in reset_strengths.items():
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                f"Asaxi boundary reset strength {name} must be 0..1")
    minimum_region = float(boundary.get(
        "minimum_region_seconds", 0.0))
    preferred_region = float(boundary.get(
        "preferred_region_seconds", 0.0))
    maximum_region = float(boundary.get(
        "maximum_region_seconds", 0.0))
    if not (
        0.04 <= minimum_region
        <= preferred_region
        <= maximum_region
        <= 0.8
    ):
        raise ValueError("Asaxi boundary-region durations are inconsistent")
    fall_max = float(boundary.get(
        "statement_fall_max_semitones", 0.0))
    if not 0.0 <= fall_max <= 6.0:
        raise ValueError("Asaxi statement Fall span must be 0..6 semitones")

    approximation = dict(data.get("target_approximation") or {})
    response = float(approximation.get(
        "response_time_seconds", 0.0))
    sample_step = float(approximation.get(
        "sample_step_seconds", 0.0))
    maximum_slew = float(approximation.get(
        "maximum_slew_semitones_per_second", 0.0))
    if not 0.02 <= response <= 0.3:
        raise ValueError("Asaxi pitch response time must be 0.02..0.3 s")
    if not 0.008 <= sample_step <= 0.08:
        raise ValueError("Asaxi pitch sample step must be 0.008..0.08 s")
    if not 12.0 <= maximum_slew <= 120.0:
        raise ValueError("Asaxi maximum pitch slew must be 12..120 st/s")

    safe = dict(data.get("safe_f0_hz") or {})
    minimum_f0 = float(safe.get("minimum") or 0.0)
    maximum_f0 = float(safe.get("maximum") or 0.0)
    if not 20.0 <= minimum_f0 < maximum_f0 <= 1000.0:
        raise ValueError("Asaxi safe F0 range is invalid")

    return AsaxiPitchModel(
        model_id=str(data.get("model_id") or ""),
        model_version=int(data.get("model_version") or 0),
        tone_goals_semitones=tones,
        declination_total_semitones=declination,
        later_phrase_declination_delta_semitones=later_declination,
        later_phrase_contrast_scale=later_contrast,
        later_phrase_boundary_delta_semitones=later_boundary,
        maximum_contextual_phrase_index=maximum_context,
        boundary_goals_semitones=boundary_goals,
        boundary_reset_strengths=reset_strengths,
        boundary_minimum_seconds=minimum_region,
        boundary_preferred_seconds=preferred_region,
        boundary_maximum_seconds=maximum_region,
        statement_fall_max_semitones=fall_max,
        response_time_seconds=response,
        sample_step_seconds=sample_step,
        maximum_slew_semitones_per_second=maximum_slew,
        minimum_f0_hz=minimum_f0,
        maximum_f0_hz=maximum_f0,
        analysis_evidence=dict(data.get("analysis_evidence") or {}),
        provenance=dict(data.get("provenance") or {}),
    )


def load_asaxi_pitch_model(
    path: Path | str | None = None,
) -> AsaxiPitchModel:
    """Load an immutable pitch profile with automatic file invalidation."""
    source = Path(path) if path is not None else DEFAULT_MODEL_PATH
    return _PITCH_MODEL_CACHE.get(source, _read_asaxi_pitch_model)


def pitch_model_cache_info() -> dict[str, int | str]:
    return _PITCH_MODEL_CACHE.info()


def clear_pitch_model_cache() -> dict[str, int | str]:
    return _PITCH_MODEL_CACHE.clear()


def _parse_tone_overrides(value: object) -> dict[int, str]:
    result: dict[int, str] = {}
    for key, item in dict(value or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        tone = str(item or "").strip().upper()
        if tone in {"H", "L"}:
            result[index] = tone
    return result


def _parse_pitch_offsets(value: object) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, item in dict(value or {}).items():
        try:
            index = int(key)
            cents = max(-1200.0, min(1200.0, float(item)))
        except (TypeError, ValueError):
            continue
        if abs(cents) > 1.0e-9:
            result[index] = cents
    return result


def _interpolate_goal(goals: Sequence[_Goal], when: float) -> float:
    if not goals:
        return 0.0
    if when <= goals[0].time:
        return goals[0].semitones
    if when >= goals[-1].time:
        return goals[-1].semitones
    low = 0
    high = len(goals) - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if goals[middle].time <= when:
            low = middle
        else:
            high = middle
    left = goals[low]
    right = goals[high]
    width = max(1.0e-9, right.time - left.time)
    mix = (when - left.time) / width
    return (
        left.semitones * (1.0 - mix)
        + right.semitones * mix
    )


def _tracker_step(
    value: float,
    velocity: float,
    goal: float,
    elapsed: float,
    model: AsaxiPitchModel,
) -> tuple[float, float]:
    """Advance an exact critically damped constant-target response."""
    dt = max(0.0, float(elapsed))
    if dt <= 0.0:
        return value, velocity
    omega = 2.0 / max(1.0e-6, model.response_time_seconds)
    displacement = value - goal
    state = velocity + omega * displacement
    decay = math.exp(-omega * dt)
    next_displacement = (displacement + state * dt) * decay
    next_velocity = (velocity - omega * state * dt) * decay
    next_value = goal + next_displacement
    maximum_change = model.maximum_slew_semitones_per_second * dt
    change = max(-maximum_change, min(maximum_change, next_value - value))
    if abs(change - (next_value - value)) > 1.0e-12:
        next_value = value + change
        next_velocity = change / dt
    return next_value, next_velocity


def _manual_window(mora, when: float) -> float:
    if mora.start is None or mora.end is None:
        return 0.0
    start = float(mora.start)
    end = float(mora.end)
    if when < start or when > end or end <= start:
        return 0.0
    # Keep the edit at full strength across the perceptual centre of the
    # mora. The automatic linguistic anchor remains at 58%, while this short
    # plateau makes both the editor's visual midpoint and the exact target
    # sample authoritative.
    plateau_start = start + (end - start) * 0.42
    plateau_end = start + (end - start) * 0.68
    if when < plateau_start:
        progress = (when - start) / max(1.0e-9, plateau_start - start)
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    if when <= plateau_end:
        return 1.0
    progress = (when - plateau_end) / max(1.0e-9, end - plateau_end)
    return 0.5 + 0.5 * math.cos(math.pi * progress)


def _sample_times(
    start: float,
    end: float,
    step: float,
    exact: Sequence[float],
) -> tuple[float, ...]:
    values = {round(float(start), 6), round(float(end), 6)}
    count = max(0, int(math.floor((end - start) / step)))
    values.update(
        round(start + index * step, 6)
        for index in range(1, count + 1)
        if start < start + index * step < end
    )
    values.update(
        round(float(value), 6)
        for value in exact
        if start <= float(value) <= end
    )
    return tuple(sorted(values))


def realize_pitch(
    plans: Sequence[object],
    rendered_moras: Sequence[object],
    *,
    base_pitch_hz: float = 160.0,
    fall_percent: float = 18.0,
    mora_tone_overrides: Mapping[object, object] | None = None,
    mora_pitch_offsets_cents: Mapping[object, object] | None = None,
    model: AsaxiPitchModel | None = None,
) -> AsaxiPitchRealization:
    """Realize final-timing Asaxi moras as one continuous F0 plan."""
    pitch_model = model or load_asaxi_pitch_model()
    base = pitch_domain.clamp_hz(
        float(base_pitch_hz or 160.0),
        pitch_model.minimum_f0_hz,
        pitch_model.maximum_f0_hz,
    )
    plan_rows = tuple(plans)
    aligned = tuple(rendered_moras)
    tone_overrides = _parse_tone_overrides(mora_tone_overrides)
    pitch_offsets = _parse_pitch_offsets(mora_pitch_offsets_cents)

    phrase_groups: dict[int, list[object]] = {}
    for mora in aligned:
        if mora.start is None or mora.end is None:
            continue
        phrase_groups.setdefault(int(mora.phrase_index), []).append(mora)

    mora_records: list[dict[str, object]] = []
    phrase_goals: dict[int, list[_Goal]] = {}
    manual_deltas: dict[int, float] = {}
    boundary_records: list[dict[str, object]] = []
    for phrase_index, plan in enumerate(plan_rows):
        moras = phrase_groups.get(phrase_index, [])
        accentable = [mora for mora in moras if bool(mora.accentable)]
        if not moras or not accentable:
            continue
        contextual_index = min(
            phrase_index, pitch_model.maximum_contextual_phrase_index)
        tone_values = [
            pitch_model.tone_goal(mora.pitch) for mora in accentable
        ]
        tone_mean = sum(tone_values) / max(1, len(tone_values))
        phrase_goal_rows: list[_Goal] = []
        for local_index, mora in enumerate(accentable):
            progress = local_index / max(1, len(accentable) - 1)
            centered_progress = (
                progress - 0.5 if len(accentable) > 1 else 0.0
            )
            centered_boundary = (
                (1.0 if local_index == len(accentable) - 1 else 0.0)
                - (1.0 / len(accentable))
            )
            lexical_goal = pitch_model.tone_goal(mora.pitch)
            components = {
                "lexical_tone": lexical_goal,
                "phrase_declination": (
                    pitch_model.declination_total_semitones
                    * centered_progress
                ),
                "later_phrase_declination_shape": (
                    pitch_model.later_phrase_declination_delta_semitones
                    * contextual_index
                    * centered_progress
                ),
                "later_phrase_contrast_shape": (
                    pitch_model.later_phrase_contrast_scale
                    * contextual_index
                    * (lexical_goal - tone_mean)
                ),
                "later_phrase_boundary_shape": (
                    pitch_model.later_phrase_boundary_delta_semitones
                    * contextual_index
                    * centered_boundary
                ),
                "downstep_state": 0.0,
                "focus_contribution": 0.0,
                "discourse_offset": 0.0,
            }
            automatic_goal = sum(components.values())
            target_time = float(mora.start) + (
                float(mora.end) - float(mora.start)
            ) * 0.58
            selected_tone = tone_overrides.get(
                int(mora.index), str(mora.pitch))
            manual_tone = (
                pitch_model.tone_goal(selected_tone) - lexical_goal
            )
            manual_cents = pitch_offsets.get(int(mora.index), 0.0)
            manual_delta = manual_tone + manual_cents / 100.0
            manual_deltas[int(mora.index)] = manual_delta
            phrase_goal_rows.append(_Goal(
                time=target_time,
                semitones=automatic_goal,
                kind="mora",
                phrase_index=phrase_index,
                mora_index=int(mora.index),
            ))
            mora_records.append({
                "sentence_index": 0,
                "phrase_index": phrase_index,
                "word_index": int(mora.word_index),
                "mora_index": int(mora.index),
                "local_mora_index": int(mora.local_mora_index),
                "word": str(mora.word),
                "mora": str(mora.text),
                "target_time_seconds": round(target_time, 6),
                "lexical_tone": str(mora.lexical_pitch),
                "utterance_tone": str(mora.pitch),
                "selected_tone": selected_tone,
                "speech_act": (
                    "question" if bool(getattr(plan, "interrogative", False))
                    else "directive" if bool(
                        getattr(plan, "directive", False))
                    else "statement"
                ),
                "boundary_tone": str(
                    getattr(plan, "boundary_tone", "")),
                "speaker_center_hz": round(base, 6),
                "speaker_span_semitones": round(
                    pitch_model.tone_goal("H")
                    - pitch_model.tone_goal("L"),
                    6,
                ),
                "components_semitones": {
                    key: round(value, 6)
                    for key, value in components.items()
                },
                "automatic_goal_semitones": round(automatic_goal, 6),
                "manual_tone_contribution_semitones": round(
                    manual_tone, 6),
                "manual_cents_contribution_semitones": round(
                    manual_cents / 100.0, 6),
                "manual_total_semitones": round(manual_delta, 6),
                "voicing_status": "not_classified",
            })

        phrase_start = float(moras[0].start)
        phrase_end = float(moras[-1].end)
        first_goal = phrase_goal_rows[0].semitones
        phrase_goal_rows.append(_Goal(
            time=phrase_start,
            semitones=first_goal,
            kind="phrase_onset",
            phrase_index=phrase_index,
        ))
        duration = max(0.0, phrase_end - phrase_start)
        region = max(
            pitch_model.boundary_minimum_seconds,
            min(
                pitch_model.boundary_maximum_seconds,
                min(pitch_model.boundary_preferred_seconds,
                    duration * 0.45),
            ),
        )
        region = min(region, max(0.01, duration))
        boundary_start = max(phrase_start, phrase_end - region)
        lexical_at_boundary = _interpolate_goal(
            sorted(phrase_goal_rows, key=lambda item: item.time),
            boundary_start,
        )
        boundary_tone = str(getattr(plan, "boundary_tone", "L%"))
        phrase_goal_rows.append(_Goal(
            time=boundary_start,
            semitones=lexical_at_boundary,
            kind="boundary_region_start",
            phrase_index=phrase_index,
        ))
        event_rows = [{
            "phrase_index": phrase_index,
            "boundary_tone": boundary_tone,
            "kind": "boundary_region_start",
            "time_seconds": round(boundary_start, 6),
            "goal_semitones": round(lexical_at_boundary, 6),
        }]
        boundary_end = max(boundary_start, phrase_end - min(0.01, region * 0.08))
        if boundary_tone == "LH%":
            # The high goal begins around the middle of the region rather
            # than at its final sample. A short final mora can therefore
            # approach the rise instead of receiving an impossible endpoint
            # command that the target tracker never has time to realize.
            low_time = boundary_start + region * 0.20
            high_time = boundary_start + region * 0.48
            low_goal = pitch_model.boundary_goal("LH%_low")
            high_goal = pitch_model.boundary_goal("LH%_high")
            phrase_goal_rows = [
                goal for goal in phrase_goal_rows
                if not (
                    goal.kind == "mora"
                    and goal.time > high_time
                )
            ]
            phrase_goal_rows.extend((
                _Goal(
                    low_time, low_goal, "boundary_low",
                    phrase_index,
                ),
                _Goal(
                    high_time, high_goal, "boundary_high",
                    phrase_index,
                ),
                _Goal(
                    boundary_end, high_goal, "boundary_high_hold",
                    phrase_index,
                ),
            ))
            event_rows.extend((
                {
                    "phrase_index": phrase_index,
                    "boundary_tone": boundary_tone,
                    "kind": "boundary_low",
                    "time_seconds": round(low_time, 6),
                    "goal_semitones": round(low_goal, 6),
                },
                {
                    "phrase_index": phrase_index,
                    "boundary_tone": boundary_tone,
                    "kind": "boundary_high",
                    "time_seconds": round(high_time, 6),
                    "goal_semitones": round(high_goal, 6),
                },
                {
                    "phrase_index": phrase_index,
                    "boundary_tone": boundary_tone,
                    "kind": "boundary_high_hold",
                    "time_seconds": round(boundary_end, 6),
                    "goal_semitones": round(high_goal, 6),
                },
            ))
        else:
            boundary_goal = pitch_model.boundary_goal(
                boundary_tone if boundary_tone in {"L%", "H-", "H%"}
                else "L%"
            )
            if boundary_tone == "L%":
                fall_fraction = max(
                    0.0, min(40.0, float(fall_percent or 0.0))
                ) / 40.0
                boundary_goal -= (
                    pitch_model.statement_fall_max_semitones
                    * fall_fraction
                )
            phrase_goal_rows.append(_Goal(
                boundary_end,
                boundary_goal,
                "boundary_goal",
                phrase_index,
            ))
            event_rows.append({
                "phrase_index": phrase_index,
                "boundary_tone": boundary_tone,
                "kind": "boundary_goal",
                "time_seconds": round(boundary_end, 6),
                "goal_semitones": round(boundary_goal, 6),
            })
        boundary_records.extend(event_rows)

        # Later entries at the same timestamp are intentional authority, as in
        # Festival's target relation. Boundary events therefore win over a
        # lexical target only when both are truly co-timed.
        unique_goals: dict[float, _Goal] = {}
        for goal in sorted(
            phrase_goal_rows,
            key=lambda item: (
                item.time,
                0 if item.kind == "mora" else 1,
            ),
        ):
            unique_goals[round(goal.time, 6)] = goal
        phrase_goals[phrase_index] = [
            unique_goals[key] for key in sorted(unique_goals)
        ]

    trajectory: list[dict[str, object]] = []
    phrase_states: list[dict[str, object]] = []
    target_rows: dict[float, float] = {}
    state_value = 0.0
    state_velocity = 0.0
    state_time: float | None = None
    previous_goal = 0.0
    previous_boundary = ""
    for phrase_index, plan in enumerate(plan_rows):
        moras = phrase_groups.get(phrase_index, [])
        goals = phrase_goals.get(phrase_index, [])
        if not moras or not goals:
            continue
        phrase_start = float(moras[0].start)
        phrase_end = float(moras[-1].end)
        first_desired = _interpolate_goal(goals, phrase_start)
        carry_value = state_value
        carry_velocity = state_velocity
        if state_time is None:
            state_value = first_desired
            state_velocity = 0.0
            reset_strength = 1.0
        else:
            state_value, state_velocity = _tracker_step(
                state_value,
                state_velocity,
                previous_goal,
                max(0.0, phrase_start - state_time),
                pitch_model,
            )
            carry_value = state_value
            carry_velocity = state_velocity
            reset_strength = pitch_model.reset_strength(previous_boundary)
            state_value += reset_strength * (first_desired - state_value)
            state_velocity *= 1.0 - reset_strength
        state_time = phrase_start
        phrase_states.append({
            "phrase_index": phrase_index,
            "previous_boundary_tone": previous_boundary,
            "reset_strength": round(reset_strength, 6),
            "carry_in_semitones": round(carry_value, 6),
            "carry_in_slope_semitones_per_second": round(
                carry_velocity, 6),
            "starting_goal_semitones": round(first_desired, 6),
            "state_after_reset_semitones": round(state_value, 6),
            "state_after_reset_slope_semitones_per_second": round(
                state_velocity, 6),
        })
        exact_times = [goal.time for goal in goals]
        exact_times.extend(
            float(mora.start)
            + (float(mora.end) - float(mora.start)) * 0.58
            for mora in moras
            if bool(mora.accentable)
        )
        mora_cursor = 0
        for when in _sample_times(
            phrase_start,
            phrase_end,
            pitch_model.sample_step_seconds,
            exact_times,
        ):
            desired = _interpolate_goal(goals, when)
            elapsed = max(0.0, when - float(state_time))
            state_value, state_velocity = _tracker_step(
                state_value,
                state_velocity,
                desired,
                elapsed,
                pitch_model,
            )
            state_time = when
            manual = 0.0
            active_mora = None
            while (
                mora_cursor + 1 < len(moras)
                and when > float(moras[mora_cursor].end)
            ):
                mora_cursor += 1
            mora = moras[mora_cursor]
            if float(mora.start) <= when <= float(mora.end):
                active_mora = int(mora.index)
                manual = manual_deltas.get(
                    active_mora, 0.0
                ) * _manual_window(mora, when)
            realized = state_value + manual
            frequency = pitch_domain.clamp_hz(
                pitch_domain.semitone_offset(base, realized),
                pitch_model.minimum_f0_hz,
                pitch_model.maximum_f0_hz,
            )
            rounded_time = round(when, 6)
            target_rows[rounded_time] = round(frequency, 3)
            trajectory.append({
                "time_seconds": rounded_time,
                "phrase_index": phrase_index,
                "mora_index": active_mora,
                "desired_semitones": round(desired, 6),
                "automatic_realized_semitones": round(state_value, 6),
                "manual_contribution_semitones": round(manual, 6),
                "final_semitones": round(realized, 6),
                "final_f0_hz": round(frequency, 3),
                "state_slope_semitones_per_second": round(
                    state_velocity, 6),
                "voicing_status": "latent_curve_render_target",
            })
        previous_goal = _interpolate_goal(goals, phrase_end)
        previous_boundary = str(getattr(plan, "boundary_tone", ""))

    trace = {
        "schema_version": 1,
        "kind": "asaxi_prosody_trace",
        "model_id": pitch_model.model_id,
        "model_version": pitch_model.model_version,
        "base_pitch_hz": round(base, 6),
        "fall_percent": round(
            max(0.0, min(40.0, float(fall_percent or 0.0))), 6),
        "frequency_domain": "speaker-relative semitones over log2 F0",
        "cumulative_frequency_drift": "disabled",
        "target_approximation": {
            "kind": "critically_damped_rate_limited",
            "response_time_seconds": pitch_model.response_time_seconds,
            "sample_step_seconds": pitch_model.sample_step_seconds,
            "maximum_slew_semitones_per_second": (
                pitch_model.maximum_slew_semitones_per_second
            ),
        },
        "phrase_states": phrase_states,
        "mora_goals": mora_records,
        "boundary_events": boundary_records,
        "trajectory": trajectory,
        "analysis_evidence": dict(pitch_model.analysis_evidence),
        "provenance": dict(pitch_model.provenance),
    }
    return AsaxiPitchRealization(
        targets=tuple(sorted(target_rows.items())),
        trace=trace,
    )
