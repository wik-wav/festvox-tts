"""Japanese high-vowel realization using shared source-filter voicing.

Festival TD-PSOLA remains responsible for phone timing and F0.  It cannot
remove periodic excitation from a recorded vowel, so this module applies the
continuous excitation control after UniSyn rendering.  The generic renderer
separates harmonic and aperiodic residuals, uses one continuous deterministic
noise source when a strongly voiced recording has insufficient stochastic
excitation, and preserves their shared vocal-tract envelope.  It never resets
random phase independently at each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from source_filter_voicing import (
    curve_for_regions,
    transform_voicing,
)


@dataclass(frozen=True)
class JapaneseVowelRealizationDecision:
    segment_index: int
    mora_index: int
    phone: str
    requested: bool
    strategy: str
    reason: str
    target_duration: float
    target_voicing: float | None = None
    automatic_target_voicing: float | None = None
    manual_mora_override: float | None = None
    prediction_reasons: tuple[str, ...] = ()
    source_voicing_mean: float | None = None
    periodicity_before: float | None = None
    periodicity_after: float | None = None
    spectral_envelope_distance: float | None = None
    level_step_db: float | None = None
    expected_level_step_db: float | None = None
    source_was_naturally_aperiodic: bool = False
    applied: bool = False

    def to_dict(self) -> dict[str, object]:
        # NumPy comparisons used by quality gates may yield scalar subclasses
        # (notably np.bool_). Keep runtime/project diagnostics ordinary JSON.
        return {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class JapaneseMoraVoicingPrediction:
    """One inspectable linguistic prediction before waveform realization."""

    mora_index: int
    phones: tuple[str, ...]
    segment_indices: tuple[int, ...]
    eligible: bool
    automatic_voicing: float
    final_voicing: float
    manual_override: float | None
    reasons: tuple[str, ...]
    context: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "mora_index": self.mora_index,
            "phones": list(self.phones),
            "segment_indices": list(self.segment_indices),
            "eligible": self.eligible,
            "automatic_voicing": self.automatic_voicing,
            "final_voicing": self.final_voicing,
            "manual_override": self.manual_override,
            "overridden": self.manual_override is not None,
            "reasons": list(self.reasons),
            "context": dict(self.context),
        }


def _rms(samples) -> float:
    values = np.asarray(samples, np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def periodicity_score(samples, sample_rate: int, *,
                      minimum_f0: float = 60.0,
                      maximum_f0: float = 450.0) -> float | None:
    """Return overlap-normalized autocorrelation periodicity."""
    values = np.asarray(samples, np.float64)
    if values.size < max(24, int(sample_rate / maximum_f0) * 3):
        return None
    values = values - float(np.mean(values))
    if _rms(values) < 1e-5:
        return None
    values *= np.hanning(values.size)
    minimum_lag = max(2, int(math.floor(sample_rate / maximum_f0)))
    maximum_lag = min(
        values.size // 2, int(math.ceil(sample_rate / minimum_f0))
    )
    scores = []
    for lag in range(minimum_lag, maximum_lag + 1):
        left, right = values[:-lag], values[lag:]
        denominator = math.sqrt(
            float(np.dot(left, left)) * float(np.dot(right, right))
        ) + 1e-12
        scores.append(float(np.dot(left, right)) / denominator)
    return max(0.0, min(1.0, max(scores, default=0.0)))


def _smoothed_log_envelope(samples, points: int = 96) -> np.ndarray:
    values = np.asarray(samples, np.float64)
    if values.size < 8:
        return np.zeros(points, np.float64)
    nfft = 1 << max(6, int(math.ceil(math.log2(values.size))))
    magnitude = np.abs(np.fft.rfft(
        (values - float(np.mean(values))) * np.hanning(values.size), nfft
    )) + 1e-9
    log_magnitude = np.log(magnitude)
    width = max(5, int(round(len(log_magnitude) / 32.0)))
    if width % 2 == 0:
        width += 1
    half = width // 2
    smoothed = np.convolve(
        np.pad(log_magnitude, (half, half), mode="edge"),
        np.ones(width, np.float64) / width,
        mode="valid",
    )
    normalized = smoothed - float(np.mean(smoothed))
    return np.interp(
        np.linspace(0.0, 1.0, points),
        np.linspace(0.0, 1.0, len(normalized)), normalized,
    )


def spectral_envelope_distance(left, right, sample_rate: int = 16000,
                               minimum_frequency: float = 250.0) -> float:
    """Compare tract-scale log envelopes without the glottal/DC band.

    The lowest bins are dominated by DC, F0, and glottal tilt.  A de-voicing
    transform is expected to change that excitation region, so including it
    would misreport successful removal of periodic energy as a formant change.
    The retained band still includes the vowel resonances and spectral tilt.
    """
    a = _smoothed_log_envelope(left)
    b = _smoothed_log_envelope(right)
    if not len(a) or not len(b):
        return 0.0
    nyquist = max(1.0, float(sample_rate) / 2.0)
    first = min(
        len(a) - 1,
        max(0, int(math.ceil(
            max(0.0, float(minimum_frequency)) / nyquist * (len(a) - 1)
        ))),
    )
    a = a[first:] - float(np.mean(a[first:]))
    b = b[first:] - float(np.mean(b[first:]))
    return float(np.sqrt(np.mean((a - b) ** 2)))


_VOICELESS = {
    "k", "ky", "p", "py", "t", "ch", "ts", "s", "sh", "f", "h",
    "hy", "cl",
}
_FRICATIVE_OR_AFFRICATE = {"ch", "ts", "s", "sh", "f", "h", "hy"}


def _mora_override_map(value) -> dict[int, float]:
    result = {}
    for key, item in dict(value or {}).items():
        try:
            index = int(key)
            degree = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(degree):
            result[index] = max(0.0, min(1.0, degree))
    return result


def predict_mora_voicing(
    plan,
    mora_voicing_overrides: Mapping[object, object] | None = None,
    *,
    target_voicing: float = 0.16,
) -> tuple[JapaneseMoraVoicingPrediction, ...]:
    """Predict a continuous voiced-excitation degree for each mora.

    The duration planner already carries Open JTalk and phonological context
    per phone.  Reusing those named fields keeps pitch, duration, and voicing
    structurally separate while avoiding a second opaque text analysis.
    Values are voiced proportions: 1.0 is the recorded voiced realization and
    0.0 is maximally aperiodic.  The renderer may still reject an acoustically
    unsafe request and records that result separately.
    """
    overrides = _mora_override_map(mora_voicing_overrides)
    speed = max(0.25, min(4.0, float(getattr(plan, "speed", 1.0))))
    floor = max(0.04, min(0.92, float(target_voicing)))
    predictions = []
    for mora in getattr(plan, "mora_timings", ()):
        allocations = list(getattr(mora, "phone_allocation", ()))
        vowels = [row for row in allocations
                  if str(getattr(row, "phone", "")) in
                  {"a", "i", "u", "e", "o"}]
        phones = tuple(str(getattr(row, "phone", "")) for row in vowels)
        indices = tuple(int(getattr(row, "segment_index")) for row in vowels)
        primary = next((row for row in vowels
                        if str(getattr(row, "phone", "")) in {"i", "u"}),
                       vowels[0] if vowels else None)
        context = dict(getattr(primary, "context", {}) or {}) \
            if primary is not None else {}
        effects = dict(getattr(primary, "context_effects", {}) or {}) \
            if primary is not None else {}
        special = str(context.get("special_mora") or "")
        reasons: list[str] = []
        eligible = bool(vowels)
        automatic = 1.0
        if primary is None:
            reasons.append("mora has no editable vowel interval")
        elif str(getattr(primary, "phone", "")) not in {"i", "u"}:
            reasons.append("automatic devoicing is limited to high vowels")
        elif special == "long_vowel":
            reasons.append("long-vowel mora blocks automatic devoicing")
        elif (context.get("previous_mora_devoiced") or
              "consecutive_devoicing_avoidance" in effects):
            reasons.append("consecutive devoicing is suppressed")
        elif "devoiced_high_vowel" not in effects:
            reasons.append("no voiceless high-vowel environment was found")
        else:
            automatic = floor
            reasons.append("voiceless high-vowel environment")
            if context.get("openjtalk_devoiced") is True:
                automatic -= 0.03
                reasons.append("Open JTalk marks the vowel devoiced")
            previous = str(context.get("previous_phone") or "")
            following = str(context.get("following_phone") or "")
            if previous in _VOICELESS and following in _VOICELESS:
                automatic -= 0.02
                reasons.append("both neighboring consonants are voiceless")
            if ({previous, following} & _FRICATIVE_OR_AFFRICATE):
                automatic -= 0.02
                reasons.append("fricative or affricate context strengthens it")
            if "cl" in {previous, following}:
                automatic -= 0.03
                reasons.append("geminate closure strengthens devoicing")
            if context.get("accent_distance") == 0:
                automatic += 0.16
                reasons.append("accent nucleus preserves voicing")
            if context.get("accent_phrase_final"):
                automatic += 0.10
                reasons.append("accent-phrase edge preserves voicing")
            if (context.get("phrase_final") or
                    ("following_phone" in context and not following)):
                automatic += 0.16
                reasons.append("phrase or pause boundary preserves voicing")
            rate_shift = max(-0.10, min(
                0.10, -0.10 * math.log2(speed)
            ))
            if abs(rate_shift) >= 0.005:
                automatic += rate_shift
                reasons.append(
                    "faster rate strengthens devoicing" if rate_shift < 0
                    else "slower rate preserves voicing"
                )
            automatic = max(0.06, min(0.92, automatic))
        index = int(getattr(mora, "mora_index"))
        manual = overrides.get(index) if eligible else None
        final = manual if manual is not None else automatic
        if manual is not None:
            reasons.append("explicit mora voicing override is final")
        predictions.append(JapaneseMoraVoicingPrediction(
            mora_index=index,
            phones=phones,
            segment_indices=indices,
            eligible=eligible,
            automatic_voicing=round(float(automatic), 6),
            final_voicing=round(float(final), 6),
            manual_override=(round(float(manual), 6)
                             if manual is not None else None),
            reasons=tuple(reasons),
            context={
                "speed": round(speed, 6),
                "special_mora": special or None,
                "previous_phone": context.get("previous_phone"),
                "following_phone": context.get("following_phone"),
                "accent_distance": context.get("accent_distance"),
                "accent_phrase_final": bool(
                    context.get("accent_phrase_final", False)),
                "phrase_final": bool(context.get("phrase_final", False)),
                "openjtalk_devoiced": context.get("openjtalk_devoiced"),
            },
        ))
    return tuple(predictions)


def requested_realizations(
    plan,
    mora_voicing_overrides: Mapping[object, object] | None = None,
    *,
    target_voicing: float = 0.16,
) -> tuple[JapaneseVowelRealizationDecision, ...]:
    predictions = predict_mora_voicing(
        plan, mora_voicing_overrides, target_voicing=target_voicing
    )
    timing_by_mora = {
        int(getattr(mora, "mora_index")): mora
        for mora in getattr(plan, "mora_timings", ())
    }
    rows = []
    for prediction in predictions:
        if (not prediction.eligible or
                (prediction.manual_override is None and
                 prediction.automatic_voicing >= 0.995)):
            continue
        timing = timing_by_mora.get(prediction.mora_index)
        by_index = {
            int(getattr(phone, "segment_index")): phone
            for phone in getattr(timing, "phone_allocation", ())
        }
        for segment_index in prediction.segment_indices:
            phone = by_index.get(segment_index)
            if phone is None:
                continue
            rows.append(JapaneseVowelRealizationDecision(
                segment_index=segment_index,
                mora_index=prediction.mora_index,
                phone=str(getattr(phone, "phone", "")),
                requested=prediction.final_voicing < 0.995,
                strategy="pending",
                reason=("; ".join(prediction.reasons) or
                        "mora voicing prediction"),
                target_duration=float(getattr(phone, "final_duration", 0.0)),
                target_voicing=prediction.final_voicing,
                automatic_target_voicing=prediction.automatic_voicing,
                manual_mora_override=prediction.manual_override,
                prediction_reasons=prediction.reasons,
            ))
    return tuple(rows)


def _map_plan_segments(plan, synthesis) -> Mapping[int, int]:
    planned = [str(item.phone) for item in getattr(plan, "segments", ())]
    rendered = [str(item.phone) for item in getattr(synthesis, "segments", ())]
    if planned == rendered:
        return {index: index for index in range(len(planned))}
    from difflib import SequenceMatcher
    result = {}
    for left, right, size in SequenceMatcher(
            a=planned, b=rendered, autojunk=False).get_matching_blocks():
        for offset in range(size):
            result[left + offset] = right + offset
    return result


def _curve_mean(points: Sequence[Sequence[float]], start: float,
                end: float) -> float | None:
    values = [float(value) for time, value in points
              if start <= float(time) <= end]
    return float(np.mean(values)) if values else None


def _segment_parts(segment) -> tuple[str, float, float]:
    if isinstance(segment, Mapping):
        return (
            str(segment.get("phone") or segment.get("name") or ""),
            float(segment.get("start") or 0.0),
            float(segment.get("end") or 0.0),
        )
    return (
        str(getattr(segment, "phone", getattr(segment, "name", ""))),
        float(getattr(segment, "start", 0.0)),
        float(getattr(segment, "end", 0.0)),
    )


def _is_pause(phone: str) -> bool:
    return str(phone).casefold() in {"pau", "sil", "sp"}


def hold_first_post_phrase_pause(
    points: Sequence[Sequence[float]],
    segments: Sequence[object],
) -> list[tuple[float, float]]:
    """Hold the preceding control value through the first pause in a run.

    This is a generated-curve display convention, not a request to synthesize
    voiced silence. Automatic rendering protects true pause samples. An
    explicit continuous user curve remains final even inside a region labelled
    ``pau`` because edge-labelled pauses can contain audible source speech.
    """
    curve = sorted((float(time), float(value)) for time, value in points)
    if not curve:
        return []
    source_x = np.asarray([row[0] for row in curve], np.float64)
    source_y = np.asarray([row[1] for row in curve], np.float64)
    held = []
    previous_was_speech = False
    in_pause_run = False
    for segment in segments:
        phone, start, end = _segment_parts(segment)
        if not _is_pause(phone):
            previous_was_speech = True
            in_pause_run = False
            continue
        if previous_was_speech and not in_pause_run and end > start:
            before = np.flatnonzero(source_x < start - 1.0e-9)
            value = (float(source_y[before[-1]]) if before.size else
                     float(np.interp(start, source_x, source_y)))
            held.append((start, end, value))
        in_pause_run = True
        previous_was_speech = False
    if not held:
        return [(round(time, 6), round(value, 6)) for time, value in curve]
    result = {float(time): float(value) for time, value in curve}
    for start, end, value in held:
        for time in list(result):
            if start <= time < end:
                result[time] = value
        result[start] = value
        edge = max(start, end - min(1.0e-6, (end - start) * 1.0e-4))
        result[edge] = value
        result[end] = float(np.interp(end, source_x, source_y))
    return [(round(time, 6), round(result[time], 6))
            for time in sorted(result)]


def mask_voicing_targets_in_pauses(
    targets: Sequence[Sequence[float]],
    source_curve: Sequence[Sequence[float]],
    segments: Sequence[object],
) -> list[tuple[float, float]]:
    """Force pause targets back to their measured source values."""
    target = {float(time): float(value) for time, value in targets}
    source = sorted((float(time), float(value))
                    for time, value in source_curve)
    if not source:
        return sorted(target.items())
    source_x = np.asarray([row[0] for row in source], np.float64)
    source_y = np.asarray([row[1] for row in source], np.float64)
    pause_spans = [
        (start, end) for phone, start, end in
        (_segment_parts(segment) for segment in segments)
        if _is_pause(phone) and end > start
    ]
    for start, end in pause_spans:
        for time in list(target):
            if start <= time <= end:
                target[time] = float(np.interp(time, source_x, source_y))
        target[start] = float(np.interp(start, source_x, source_y))
        target[end] = float(np.interp(end, source_x, source_y))
    return [(round(time, 6), round(target[time], 6))
            for time in sorted(target)]


def restore_pause_samples(
    rendered: Sequence[float],
    source: Sequence[float],
    sample_rate: int,
    segments: Sequence[object],
    *,
    ramp_seconds: float = 0.004,
) -> np.ndarray:
    """Restore pause runs exactly, with short ramps in adjacent speech."""
    output = np.asarray(rendered, np.float64).copy()
    original = np.asarray(source, np.float64).reshape(-1)
    if output.shape != original.shape:
        raise ValueError("pause mask source and rendered samples disagree")
    spans = sorted(
        (max(0.0, start), max(start, end))
        for phone, start, end in
        (_segment_parts(segment) for segment in segments)
        if _is_pause(phone) and end > start
    )
    runs = []
    for start, end in spans:
        if runs and start <= runs[-1][1] + 1.0e-7:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    ramp = max(0, int(round(float(ramp_seconds) * int(sample_rate))))
    for start, end in runs:
        first = max(0, min(len(output), int(round(start * sample_rate))))
        last = max(first, min(len(output), int(round(end * sample_rate))))
        if first > 0 and ramp:
            left = max(0, first - ramp)
            weight = np.linspace(0.0, 1.0, first - left, endpoint=False)
            output[left:first] = (
                output[left:first] * (1.0 - weight)
                + original[left:first] * weight
            )
        output[first:last] = original[first:last]
        if last < len(output) and ramp:
            right = min(len(output), last + ramp)
            weight = np.linspace(1.0, 0.0, right - last, endpoint=False)
            output[last:right] = (
                output[last:right] * (1.0 - weight)
                + original[last:right] * weight
            )
    return np.asarray(output, np.float32)


def _pause_safe_result(result, source, sample_rate, segments):
    return replace(
        result,
        samples=restore_pause_samples(
            result.samples, source, sample_rate, segments
        ),
    )


def _segment_samples(synthesis, segment_index: int, samples=None):
    values = np.asarray(
        synthesis.samples if samples is None else samples, np.float32
    )
    segment = synthesis.segments[segment_index]
    start = max(0, min(len(values), int(round(segment.start * synthesis.sr))))
    end = max(start, min(len(values), int(round(segment.end * synthesis.sr))))
    edge = min((end - start) // 5, int(round(synthesis.sr * 0.006)))
    if end - start > edge * 2:
        start += edge
        end -= edge
    return values[start:end]


def initialize_voicing_metadata(synthesis):
    """Attach a measured ground-truth curve without changing the waveform."""
    analysis = transform_voicing(synthesis.samples, synthesis.sr)
    display_curve = hold_first_post_phrase_pause(
        analysis.source_curve, synthesis.segments
    )
    synthesis.generated_voicing_targets = list(display_curve)
    synthesis.source_voicing_targets = list(display_curve)
    synthesis.voicing_override = []
    synthesis.voicing_mode = ""
    synthesis.voicing_diagnostics = [analysis.diagnostic_dict(
        include_frames=False)]
    return synthesis


def apply_voicing_override(synthesis, targets):
    """Apply a user-authored curve to any language; user points are final."""
    source_samples = np.asarray(synthesis.samples, np.float32).copy()
    analysis = transform_voicing(source_samples, synthesis.sr)
    # Do not infer editability from the phone label. Source-unit overlap can
    # leave real speech inside an adjacent ``pau`` span; restoring those
    # samples made that audible tail impossible to edit. Pause protection is
    # retained for automatic/mora-only processing below, while an explicit
    # continuous curve is the final authority everywhere on the timeline.
    result = transform_voicing(source_samples, synthesis.sr, targets)
    display_curve = hold_first_post_phrase_pause(
        analysis.source_curve, synthesis.segments
    )
    synthesis.samples = result.samples
    synthesis.source_voicing_targets = list(display_curve)
    synthesis.generated_voicing_targets = list(display_curve)
    synthesis.voicing_override = [
        (float(time), float(value)) for time, value in targets
    ]
    synthesis.voicing_mode = "curve"
    synthesis.voicing_diagnostics = [result.diagnostic_dict(
        include_frames=False)]
    return synthesis


def apply_vowel_realizations(
    synthesis,
    plan,
    *,
    mode: str = "contextual",
    renderer: str = "auto",
    voicing_override: Sequence[Sequence[float]] | None = None,
    mora_voicing_overrides: Mapping[object, object] | None = None,
    natural_voicing_threshold: float = 0.34,
    target_voicing: float = 0.16,
    minimum_periodicity_drop: float = 0.06,
    maximum_envelope_distance: float = 0.62,
    maximum_level_step_db: float = 3.0,
):
    """Apply automatic Japanese targets or a final manual voicing curve."""
    mode = str(mode).casefold()
    renderer = str(renderer).casefold()
    # Old projects may contain this value. It now names the replacement
    # source-filter implementation; the rejected randomized renderer is gone.
    if renderer == "mixed_excitation":
        renderer = "source_filter"
    if renderer == "automatic":
        renderer = "auto"
    if renderer not in {
        "auto", "source_filter", "residual", "shortened_voiced",
        "natural_source",
    }:
        raise ValueError("unsupported Japanese devoicing renderer")

    source_samples = np.asarray(synthesis.samples, np.float32).copy()
    analysis = transform_voicing(source_samples, synthesis.sr)
    source_curve = list(analysis.source_curve)
    display_source_curve = hold_first_post_phrase_pause(
        source_curve, synthesis.segments
    )
    mapping = _map_plan_segments(plan, synthesis)
    requests = requested_realizations(
        plan, mora_voicing_overrides, target_voicing=target_voicing
    ) if mode != "legacy" else ()
    automatic_requests = requested_realizations(
        plan, target_voicing=target_voicing
    ) if mode != "legacy" else ()
    automatic_by_segment = {
        int(item.segment_index): item for item in automatic_requests
        if item.requested
    }
    decisions: list[JapaneseVowelRealizationDecision] = []
    candidate_regions: list[dict[str, float]] = []
    automatic_candidate_regions: list[dict[str, float]] = []
    request_rows = []

    for request in automatic_requests:
        if not request.requested:
            continue
        rendered_index = mapping.get(request.segment_index)
        if rendered_index is None or not 0 <= rendered_index < len(
                synthesis.segments):
            continue
        segment = synthesis.segments[rendered_index]
        mean_voicing = _curve_mean(
            source_curve, float(segment.start), float(segment.end)
        )
        if (mean_voicing is None or
                mean_voicing > natural_voicing_threshold):
            automatic_candidate_regions.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "segment_index": float(rendered_index),
                "target_voicing": float(
                    request.target_voicing
                    if request.target_voicing is not None
                    else target_voicing),
            })

    for request in requests:
        rendered_index = mapping.get(request.segment_index)
        if rendered_index is None or not 0 <= rendered_index < len(
                synthesis.segments):
            decisions.append(JapaneseVowelRealizationDecision(
                **{**request.__dict__,
                   "strategy": "shortened_voiced_fallback",
                   "reason": "rendered segment mapping unavailable"}
            ))
            continue
        segment = synthesis.segments[rendered_index]
        mean_voicing = _curve_mean(
            source_curve, float(segment.start), float(segment.end)
        )
        before_source = _segment_samples(synthesis, rendered_index)
        before_periodicity = periodicity_score(before_source, synthesis.sr)
        row = (request, rendered_index, segment, mean_voicing,
               before_periodicity)
        request_rows.append(row)
        requested_target = float(
            request.target_voicing
            if request.target_voicing is not None else target_voicing
        )
        if not request.requested:
            applied = (mean_voicing is None or
                       mean_voicing >= min(0.92, requested_target - 0.08))
            decisions.append(JapaneseVowelRealizationDecision(
                **{**request.__dict__, "segment_index": rendered_index,
                   "strategy": ("manual_voiced_source" if applied else
                                "manual_revoicing_unavailable"),
                   "reason": (
                       "manual mora override suppresses automatic devoicing"
                       if applied else
                       "the source interval is already aperiodic; this "
                       "renderer cannot invent a stable periodic source"
                   ),
                   "source_voicing_mean": mean_voicing,
                   "periodicity_before": before_periodicity,
                   "periodicity_after": before_periodicity,
                   "spectral_envelope_distance": 0.0,
                   "level_step_db": 0.0,
                   "applied": applied}
            ))
            continue
        if (mean_voicing is not None and
                mean_voicing <= natural_voicing_threshold and
                (request.manual_mora_override is None or
                 requested_target <= mean_voicing + 0.08)):
            decisions.append(JapaneseVowelRealizationDecision(
                **{**request.__dict__, "segment_index": rendered_index,
                   "strategy": "naturally_devoiced_source",
                   "reason": "rendered source interval is already aperiodic",
                   "target_voicing": mean_voicing,
                   "source_voicing_mean": mean_voicing,
                   "periodicity_before": before_periodicity,
                   "periodicity_after": before_periodicity,
                   "spectral_envelope_distance": 0.0,
                   "level_step_db": 0.0,
                   "source_was_naturally_aperiodic": True,
                   "applied": True}
            ))
        elif renderer in {"shortened_voiced", "natural_source"}:
            decisions.append(JapaneseVowelRealizationDecision(
                **{**request.__dict__, "segment_index": rendered_index,
                   "strategy": "shortened_voiced_fallback",
                   "reason": ("no matching naturally aperiodic source; "
                              "excitation transformation disabled"),
                   "target_voicing": mean_voicing,
                   "source_voicing_mean": mean_voicing,
                   "periodicity_before": before_periodicity,
                   "periodicity_after": before_periodicity}
            ))
        else:
            candidate_regions.append({
                "start": float(segment.start), "end": float(segment.end),
                "segment_index": float(rendered_index),
                "target_voicing": requested_target,
            })

    automatic_curve = curve_for_regions(
        source_curve, automatic_candidate_regions,
        target_voicing=target_voicing
    )
    automatic_safe_curve = mask_voicing_targets_in_pauses(
        automatic_curve, source_curve, synthesis.segments
    )
    # Evaluate the automatic proposal independently from a user override.
    # The former remains the dashed generated reference after re-rendering;
    # the latter is the final editable authority and must never overwrite it.
    automatic_result = (_pause_safe_result(
        transform_voicing(
            source_samples, synthesis.sr, automatic_safe_curve
        ),
        source_samples, synthesis.sr, synthesis.segments,
    ) if automatic_candidate_regions else analysis)
    mora_curve = curve_for_regions(
        source_curve, candidate_regions, target_voicing=target_voicing
    )
    requested_curve = (
        list(voicing_override) if voicing_override else
        mora_curve if mora_voicing_overrides else automatic_curve
    )
    if voicing_override:
        requested_safe_curve = list(requested_curve)
        result = transform_voicing(
            source_samples, synthesis.sr, requested_safe_curve
        )
    else:
        requested_safe_curve = mask_voicing_targets_in_pauses(
            requested_curve if requested_curve else source_curve,
            source_curve, synthesis.segments,
        )
        result = (_pause_safe_result(
            transform_voicing(
                source_samples, synthesis.sr, requested_safe_curve
            ),
            source_samples, synthesis.sr, synthesis.segments,
        ) if mora_voicing_overrides else automatic_result)

    def region_quality(rendered_result, segment_index, segment,
                       periodicity_before):
        before_source = _segment_samples(
            synthesis, segment_index, source_samples
        )
        after_source = _segment_samples(
            synthesis, segment_index, rendered_result.samples
        )
        periodicity_after = periodicity_score(after_source, synthesis.sr)
        envelope_distance = spectral_envelope_distance(
            before_source, after_source, synthesis.sr
        )
        before_rms, after_rms = _rms(before_source), _rms(after_source)
        level_step = 20.0 * math.log10(
            (after_rms + 1e-9) / (before_rms + 1e-9)
        )
        frames = [item for item in rendered_result.frame_diagnostics
                  if float(segment.start) <= item.time <= float(segment.end)]
        transformed_frames = sum(item.applied for item in frames)
        expected_level_step = (
            float(np.median([item.gain_db for item in frames if item.applied]))
            if transformed_frames else 0.0
        )
        periodicity_pass = (
            periodicity_before is not None and periodicity_after is not None
            and periodicity_after <=
            periodicity_before - minimum_periodicity_drop
        )
        quality_pass = (
            transformed_frames > 0
            and periodicity_pass
            and envelope_distance <= maximum_envelope_distance
            and abs(level_step - expected_level_step)
            <= maximum_level_step_db
            and np.all(np.isfinite(after_source))
        )
        return {
            "periodicity_after": periodicity_after,
            "envelope_distance": envelope_distance,
            "level_step": level_step,
            "expected_level_step": expected_level_step,
            "transformed_frames": transformed_frames,
            "quality_pass": quality_pass,
        }

    automatic_accepted_regions: list[dict[str, float]] = []
    for automatic_request in automatic_requests:
        if not automatic_request.requested:
            continue
        rendered_index = mapping.get(automatic_request.segment_index)
        if rendered_index is None or not 0 <= rendered_index < len(
                synthesis.segments):
            continue
        segment = synthesis.segments[rendered_index]
        source_region = _segment_samples(
            synthesis, rendered_index, source_samples)
        before = periodicity_score(source_region, synthesis.sr)
        mean_voicing = _curve_mean(
            source_curve, float(segment.start), float(segment.end))
        if (mean_voicing is not None and
                mean_voicing <= natural_voicing_threshold):
            continue
        automatic_metrics = region_quality(
            automatic_result, rendered_index, segment, before)
        if automatic_metrics["quality_pass"]:
            automatic_accepted_regions.append({
                "start": float(segment.start), "end": float(segment.end),
                "target_voicing": float(
                    automatic_request.target_voicing
                    if automatic_request.target_voicing is not None
                    else target_voicing),
            })

    accepted_regions: list[dict[str, float]] = []
    for request, rendered_index, segment, mean_voicing, before in request_rows:
        if any(item.segment_index == rendered_index for item in decisions):
            continue
        automatic_request = automatic_by_segment.get(request.segment_index)
        automatic_metrics = (
            region_quality(
                automatic_result, rendered_index, segment, before
            ) if automatic_request is not None else None
        )
        metrics = (
            region_quality(result, rendered_index, segment, before)
            if (voicing_override or mora_voicing_overrides)
            else automatic_metrics
        )
        if metrics is None:
            metrics = region_quality(
                result, rendered_index, segment, before
            )
        after = metrics["periodicity_after"]
        envelope_distance = metrics["envelope_distance"]
        level_step = metrics["level_step"]
        expected_level_step = metrics["expected_level_step"]
        transformed_frames = metrics["transformed_frames"]
        quality_pass = metrics["quality_pass"]
        if voicing_override or request.manual_mora_override is not None:
            # Manual points are final. Individual unsupported frames remain
            # visible in the diagnostics instead of silently fabricating a
            # harmonic or noise source.
            quality_pass = transformed_frames > 0
        if quality_pass:
            accepted_regions.append({
                "start": float(segment.start), "end": float(segment.end),
                "target_voicing": float(
                    request.target_voicing
                    if request.target_voicing is not None
                    else target_voicing),
            })
            strategy = (
                "manual_source_filter_voicing" if voicing_override else
                "manual_mora_source_filter_voicing"
                if request.manual_mora_override is not None else
                "source_filter_residual_devoiced"
            )
            reason = (
                "harmonic and continuous stochastic excitation were remixed "
                "through the preserved vocal-tract envelope"
            )
        else:
            strategy = "shortened_voiced_fallback"
            reason = (
                "source-filter conversion failed its periodicity, envelope, "
                "level, or measured-noise gate"
            )
        decisions.append(JapaneseVowelRealizationDecision(
            **{**request.__dict__, "segment_index": rendered_index,
               "strategy": strategy, "reason": reason,
               "source_voicing_mean": mean_voicing,
               "periodicity_before": before,
               "periodicity_after": after,
               "spectral_envelope_distance": envelope_distance,
               "level_step_db": level_step,
               "expected_level_step_db": expected_level_step,
               "applied": quality_pass}
        ))

    generated_curve = curve_for_regions(
        source_curve, automatic_accepted_regions,
        target_voicing=target_voicing
    )
    if voicing_override:
        synthesis.samples = result.samples
        voicing_mode = "curve"
    else:
        # Re-run with only accepted regions so a failed token is bit-identical
        # to its shortened voiced fallback instead of inheriting neighboring
        # STFT frames from a rejected conversion.
        final_regions = (accepted_regions if mora_voicing_overrides else
                         automatic_accepted_regions)
        final_curve = mask_voicing_targets_in_pauses(
            curve_for_regions(
                source_curve, final_regions,
                target_voicing=target_voicing,
            ),
            source_curve, synthesis.segments
        )
        final_result = (_pause_safe_result(
            transform_voicing(source_samples, synthesis.sr, final_curve),
            source_samples, synthesis.sr, synthesis.segments,
        ) if final_regions else analysis)
        synthesis.samples = final_result.samples
        result = final_result
        voicing_mode = ""

    synthesis.source_voicing_targets = display_source_curve
    synthesis.generated_voicing_targets = hold_first_post_phrase_pause(
        generated_curve or source_curve, synthesis.segments
    )
    synthesis.voicing_override = [
        (float(time), float(value)) for time, value in (voicing_override or ())
    ]
    synthesis.voicing_mode = voicing_mode
    synthesis.voicing_diagnostics = [
        result.diagnostic_dict(include_frames=False),
        {
            "kind": "japanese_mora_voicing_predictions",
            "decisions": [item.to_dict() for item in predict_mora_voicing(
                plan,
                mora_voicing_overrides,
                target_voicing=target_voicing,
            )],
            "continuous_curve_final": bool(voicing_override),
        },
    ]
    synthesis.vowel_realizations = [item.to_dict() for item in decisions]
    return synthesis
