"""Pitch-synchronous, read-only diagnostics for concatenative joins.

The rendered waveform is never normalized or modified.  Each known handoff
is measured against changes immediately inside the incoming and outgoing
units, so a real splice novelty can be distinguished from ordinary vowel,
diphthong, stop, or noise movement.  The module deliberately keeps component
metrics visible; ``severity_score`` is only a configurable sorting aid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
import math
from typing import Mapping, Sequence

import numpy as np

import diphone_loudness as loudness


LOGGER = logging.getLogger(__name__)
JOIN_DISCONTINUITY_SCHEMA_VERSION = 6
_SILENCE_PHONES = frozenset({"pau", "sil", "sp", "#"})
_VOWEL_PHONES = frozenset({
    "a", "i", "u", "e", "o", "aa", "ae", "ah", "ao", "aw", "ax",
    "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
})
_SONORANT_PHONES = frozenset({
    "m", "n", "ng", "nng", "mm", "nn", "l", "r", "w", "y", "j",
})
_VOICED_OBSTRUENT_PHONES = frozenset({
    "b", "d", "g", "v", "z", "zh", "dh", "jh", "dz", "dx",
})
_NOISE_OR_CLOSURE_PHONES = frozenset({
    "p", "t", "k", "q", "cl", "f", "s", "sh", "th", "h", "hh",
    "ch", "ts", "x", "br", "brth",
})
_EXPECTED_BURST_PHONES = frozenset({
    "p", "b", "t", "d", "k", "g", "q", "cl", "ch", "jh", "ts",
    "dz", "dx",
})
_STOP_OR_CLOSURE_PHONES = frozenset({
    "p", "b", "t", "d", "k", "g", "q", "cl", "ch", "jh", "ts",
    "dz",
})
_EPS = 1.0e-12


@dataclass(frozen=True)
class JoinAnalysisConfig:
    """Deterministic analysis windows, thresholds, and ranking weights."""

    period_context_count: int = 5
    unvoiced_context_count: int = 7
    unvoiced_frame_ms: float = 12.0
    unvoiced_hop_ms: float = 8.0
    immediate_context_ms: float = 12.0
    broadband_impulse_frame_scales_ms: tuple[float, ...] = (
        0.5, 1.0, 2.0, 3.0)
    broadband_impulse_hop_ms: float = 0.125
    broadband_impulse_scan_ms: float = 8.0
    broadband_impulse_context_ms: float = 24.0
    broadband_min_frequency_hz: float = 300.0
    broadband_max_frequency_fraction: float = 0.94
    broadband_band_count: int = 7
    broadband_flatness_floor: float = 0.12
    broadband_flatness_full: float = 0.55
    broadband_tilt_tolerance_db_per_octave: float = 3.5
    broadband_tilt_reject_db_per_octave: float = 9.0
    broadband_uniformity_floor_db: float = 4.0
    broadband_uniformity_reject_db: float = 10.0
    broadband_energy_ratio_gate: float = 1.5
    broadband_energy_ratio_full: float = 4.0
    broadband_floor_novelty_gate: float = 5.0
    broadband_floor_novelty_full: float = 12.0
    broadband_impulse_score: float = 0.30
    broadband_impulse_novelty: float = 4.0
    f0_min_hz: float = 45.0
    f0_max_hz: float = 700.0
    periodicity_threshold: float = 0.48
    minimum_rms: float = 2.0e-4
    voiced_minimum_rms: float = 1.5e-3
    minimum_voicing_confidence: float = 0.60
    full_voicing_periodicity: float = 0.78
    full_voicing_rms: float = 8.0e-3
    level_gate_floor_rms: float = 3.0e-4
    level_gate_full_rms: float = 1.2e-2
    content_dropout_frame_ms: float = 4.0
    content_dropout_hop_ms: float = 1.0
    content_dropout_context_ms: float = 24.0
    content_dropout_reference_rms: float = 1.5e-3
    content_dropout_attenuation_reference_db: float = 4.0
    maximum_phase_lag_cycles: float = 0.25
    spectral_coefficients: int = 12
    spectral_fft_size: int = 512
    voiced_harmonic_floor: float = 0.01
    level_step_db: float = 3.0
    novelty_threshold: float = 3.5
    sample_jump_novelty: float = 5.0
    slope_jump_novelty: float = 5.0
    sample_jump_relative_scale: float = 4.0
    slope_jump_relative_scale: float = 4.0
    sample_jump_absolute_floor: float = 0.006
    sample_jump_absolute_full: float = 0.03
    f0_step_semitones: float = 0.75
    phase_mismatch: float = 0.30
    period_shape_mismatch: float = 0.18
    spectral_step_floor: float = 0.08
    spectral_slope_floor: float = 0.025
    spectral_slope_novelty_scale_floor: float = 2.5e-4
    spectral_step_novelty: float = 3.5
    spectral_slope_novelty: float = 3.5
    formant_count: int = 4
    formant_lpc_order: int = 16
    formant_preemphasis: float = 0.97
    formant_min_hz: float = 120.0
    formant_max_hz: float = 5000.0
    formant_min_bandwidth_hz: float = 25.0
    formant_max_bandwidth_hz: float = 900.0
    formant_min_prominence_db: float = 1.0
    formant_min_tracking_confidence: float = 0.55
    formant_frequency_jump_fraction: float = 0.12
    formant_slope_break_fraction: float = 0.08
    formant_bandwidth_jump_fraction: float = 0.45
    formant_prominence_jump_db: float = 6.0
    formant_balance_jump: float = 0.35
    formant_balance_slope_break: float = 0.15
    formant_novelty_threshold: float = 3.5
    classification_score: float = 2.5
    novelty_support_scale: float = 0.25
    local_normality_discount: float = 0.95
    severity_weights: tuple[tuple[str, float], ...] = field(default_factory=lambda: (
        ("LEVEL_STEP", 1.0),
        ("SAMPLE_DISCONTINUITY", 1.15),
        ("BROADBAND_IMPULSE", 1.25),
        ("CONTENT_DROPOUT", 1.25),
        ("PHASE_MISMATCH", 1.0),
        ("F0_STEP", 0.9),
        ("PERIOD_SHAPE_MISMATCH", 0.95),
        ("SPECTRAL_STEP", 1.0),
        ("SPECTRAL_TRAJECTORY_BREAK", 1.0),
        ("SPECTRAL_ENVELOPE_BREAK", 1.0),
        ("FORMANT_FREQUENCY_BREAK", 1.05),
        ("FORMANT_TRAJECTORY_BREAK", 1.05),
        ("FORMANT_BALANCE_BREAK", 1.05),
        ("FORMANT_PROMINENCE_BREAK", 0.9),
        ("UNVOICED_SPECTRAL_BREAK", 1.0),
    ))

    def __post_init__(self):
        if not 2 <= int(self.period_context_count) <= 12:
            raise ValueError("period_context_count must be between 2 and 12")
        if not 3 <= int(self.unvoiced_context_count) <= 16:
            raise ValueError("unvoiced_context_count must be between 3 and 16")
        if not 4.0 <= float(self.unvoiced_frame_ms) <= 40.0:
            raise ValueError("unvoiced_frame_ms must be between 4 and 40 ms")
        if not 0.5 <= float(self.unvoiced_hop_ms) <= float(self.unvoiced_frame_ms):
            raise ValueError("unvoiced_hop_ms must fit inside its frame")
        scales = tuple(float(value)
                       for value in self.broadband_impulse_frame_scales_ms)
        if (not scales or len(scales) > 8 or
                any(value < 0.5 or value > 6.0 for value in scales)):
            raise ValueError(
                "broadband impulse frame scales must contain 0.5..6 ms values")
        if not 0.05 <= float(self.broadband_impulse_hop_ms) <= min(scales):
            raise ValueError("broadband impulse hop must fit inside its frame")
        if not 1.0 <= float(self.broadband_impulse_scan_ms) <= 40.0:
            raise ValueError("broadband impulse scan must be between 1 and 40 ms")
        if float(self.broadband_impulse_context_ms) < max(scales) * 2.0:
            raise ValueError("broadband impulse context is too short")
        if not 100.0 <= float(self.broadband_min_frequency_hz) <= 3000.0:
            raise ValueError("broadband minimum frequency is invalid")
        if not 0.5 <= float(self.broadband_max_frequency_fraction) < 1.0:
            raise ValueError("broadband maximum frequency fraction is invalid")
        if not 4 <= int(self.broadband_band_count) <= 12:
            raise ValueError("broadband band count must be between 4 and 12")
        if not (0.0 <= float(self.broadband_flatness_floor) <
                float(self.broadband_flatness_full) <= 1.0):
            raise ValueError("broadband flatness bounds are invalid")
        if not (0.0 <= float(self.broadband_tilt_tolerance_db_per_octave) <
                float(self.broadband_tilt_reject_db_per_octave)):
            raise ValueError("broadband tilt bounds are invalid")
        if not (0.0 <= float(self.broadband_uniformity_floor_db) <
                float(self.broadband_uniformity_reject_db)):
            raise ValueError("broadband uniformity bounds are invalid")
        if not (1.0 <= float(self.broadband_energy_ratio_gate) <
                float(self.broadband_energy_ratio_full)):
            raise ValueError("broadband energy ratio bounds are invalid")
        if not (0.0 <= float(self.broadband_floor_novelty_gate) <
                float(self.broadband_floor_novelty_full)):
            raise ValueError("broadband floor novelty bounds are invalid")
        if float(self.broadband_impulse_score) <= 0.0:
            raise ValueError("broadband impulse score must be positive")
        if float(self.broadband_impulse_novelty) <= 0.0:
            raise ValueError("broadband impulse novelty must be positive")
        if not 20.0 <= float(self.f0_min_hz) < float(self.f0_max_hz):
            raise ValueError("invalid F0 analysis range")
        if not 0.0 < float(self.maximum_phase_lag_cycles) <= 0.5:
            raise ValueError("maximum_phase_lag_cycles must be in (0, 0.5]")
        if not 4 <= int(self.spectral_coefficients) <= 32:
            raise ValueError("spectral_coefficients must be between 4 and 32")
        if int(self.spectral_fft_size) < 128:
            raise ValueError("spectral_fft_size must be at least 128")
        if not 1.0e-6 <= float(self.voiced_harmonic_floor) <= 0.2:
            raise ValueError("voiced_harmonic_floor must be between 1e-6 and 0.2")
        if not float(self.minimum_rms) < float(self.voiced_minimum_rms):
            raise ValueError("voiced_minimum_rms must exceed minimum_rms")
        if not (float(self.periodicity_threshold) <
                float(self.full_voicing_periodicity) <= 1.0):
            raise ValueError("full_voicing_periodicity must exceed the threshold")
        if not (float(self.voiced_minimum_rms) <
                float(self.full_voicing_rms)):
            raise ValueError("full_voicing_rms must exceed voiced_minimum_rms")
        if not (0.0 < float(self.minimum_voicing_confidence) <= 1.0):
            raise ValueError("minimum_voicing_confidence must be in (0, 1]")
        if not (0.0 <= float(self.level_gate_floor_rms) <
                float(self.level_gate_full_rms)):
            raise ValueError("level energy gate bounds are invalid")
        if not (
            0.0 <= float(self.sample_jump_absolute_floor) <
            float(self.sample_jump_absolute_full) <= 1.0
        ):
            raise ValueError("absolute sample-jump gate bounds are invalid")
        if float(self.spectral_slope_floor) < 0.0:
            raise ValueError("spectral_slope_floor must be non-negative")
        if float(self.spectral_slope_novelty_scale_floor) <= 0.0:
            raise ValueError("spectral slope novelty scale floor must be positive")
        if not 2 <= int(self.formant_count) <= 6:
            raise ValueError("formant_count must be between 2 and 6")
        if not 6 <= int(self.formant_lpc_order) <= 40:
            raise ValueError("formant_lpc_order must be between 6 and 40")
        if not 0.0 <= float(self.formant_preemphasis) < 1.0:
            raise ValueError("formant_preemphasis must be in [0, 1)")
        if not (50.0 <= float(self.formant_min_hz) <
                float(self.formant_max_hz)):
            raise ValueError("formant frequency limits are invalid")
        if not (5.0 <= float(self.formant_min_bandwidth_hz) <
                float(self.formant_max_bandwidth_hz)):
            raise ValueError("formant bandwidth limits are invalid")
        if float(self.formant_min_prominence_db) < 0.0:
            raise ValueError("formant prominence floor must be non-negative")
        if not 0.0 < float(self.formant_min_tracking_confidence) <= 1.0:
            raise ValueError("formant tracking confidence must be in (0, 1]")
        for name in (
                "formant_frequency_jump_fraction",
                "formant_slope_break_fraction",
                "formant_bandwidth_jump_fraction",
                "formant_prominence_jump_db",
                "formant_balance_jump",
                "formant_balance_slope_break",
                "formant_novelty_threshold"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.classification_score) <= 0.0:
            raise ValueError("classification_score must be positive")
        for name in (
                "content_dropout_frame_ms", "content_dropout_hop_ms",
                "content_dropout_context_ms",
                "content_dropout_reference_rms",
                "content_dropout_attenuation_reference_db"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.novelty_support_scale) <= 0.0:
            raise ValueError("novelty_support_scale must be positive")
        if not 0.0 <= float(self.local_normality_discount) < 1.0:
            raise ValueError("local_normality_discount must be in [0, 1)")

    @property
    def weights(self) -> dict[str, float]:
        return {str(name): float(value)
                for name, value in self.severity_weights}


def _finite(value: object, digits: int = 7) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return 0.0
    return float(math.sqrt(max(0.0, float(np.mean(values * values)))))


def _db_ratio(right: float, left: float) -> float | None:
    if right <= _EPS or left <= _EPS:
        return None
    return 20.0 * math.log10(right / left)


def _smoothstep(value: float, lower: float, upper: float) -> float:
    """Continuous 0..1 gate with no discontinuity at either endpoint."""
    if upper <= lower:
        return float(value >= upper)
    position = max(0.0, min(1.0, (float(value) - lower) / (upper - lower)))
    return position * position * (3.0 - 2.0 * position)


def _phone_key(phone: str) -> str:
    value = str(phone or "").strip().lower()
    while value and value[-1].isdigit():
        value = value[:-1]
    return value


def _phone_voicing_prior(phone: str) -> tuple[float, str]:
    """Return a soft linguistic prior; acoustics remain authoritative."""
    key = _phone_key(phone)
    if key in _SILENCE_PHONES:
        return 0.0, "silence"
    if key in _VOWEL_PHONES:
        return 0.98, "vowel"
    if key in _SONORANT_PHONES or key == "N".lower():
        return 0.88, "sonorant"
    if key in _VOICED_OBSTRUENT_PHONES:
        return 0.48, "voiced-obstruent"
    if key in _NOISE_OR_CLOSURE_PHONES:
        return 0.15, "noise-or-closure"
    return 0.62, "unknown"


def _expected_broadband_burst(phone: str) -> bool:
    """Return whether a short full-band release can be linguistically normal."""
    key = _phone_key(phone)
    if key in _EXPECTED_BURST_PHONES:
        return True
    # Japanese palatalized stops may be represented as ky/gy/by/py rather
    # than separate consonant + glide phones.
    return len(key) == 2 and key.endswith("y") and key[0] in "pkgbtd"


def _expected_low_energy_handoff(phone: str) -> bool:
    """Return whether silence at the phone centre can be linguistic content."""
    key = _phone_key(phone)
    if (key in _SILENCE_PHONES or key in _STOP_OR_CLOSURE_PHONES or
            key == "brth"):
        return True
    # Palatalized stops and affricates can carry a real closure before release.
    return len(key) == 2 and key.endswith("y") and key[0] in "pkgbtd"


def _rms_frames(samples: np.ndarray, start: int, end: int,
                frame: int, hop: int) -> list[dict[str, float]]:
    start = max(0, int(start))
    end = min(len(samples), int(end))
    frame = max(2, int(frame))
    hop = max(1, int(hop))
    if end <= start:
        return []
    if end - start <= frame:
        return [{"sample": float((start + end) * 0.5),
                 "rms": _rms(samples[start:end])}]
    positions = list(range(start, end - frame + 1, hop))
    final = end - frame
    if positions[-1] != final:
        positions.append(final)
    return [{"sample": float(position + frame * 0.5),
             "rms": _rms(samples[position:position + frame])}
            for position in positions]


def _content_dropout_metrics(samples: np.ndarray, sample_rate: int,
                             splice: Mapping, when: float, phone: str,
                             config: "JoinAnalysisConfig", *,
                             voiced: bool = False,
                             period_samples: int | None = None) -> dict:
    """Measure whether the rendered handoff erased expected local content.

    The reference frames are wholly outside the declared handoff.  A short
    frame scan inside the handoff catches a cancellation notch that an average
    over the complete overlap would conceal.  Stop closures and pauses retain
    the raw measurement but are excluded from severity ranking.
    """
    fixed_frame = max(2, int(round(
        sample_rate * config.content_dropout_frame_ms / 1000.0)))
    frame = (max(fixed_frame, int(period_samples))
             if voiced and period_samples else fixed_frame)
    hop = (max(1, frame // 4) if voiced else max(1, int(round(
        sample_rate * config.content_dropout_hop_ms / 1000.0))))
    context = max(frame, int(round(
        sample_rate * config.content_dropout_context_ms / 1000.0)),
        3 * frame)
    center = int(round(float(when) * sample_rate))
    try:
        start = int(round(float(splice.get("handoff_start", when)) *
                          sample_rate))
        end = int(round(float(splice.get("handoff_end", when)) *
                        sample_rate))
    except (TypeError, ValueError):
        start = end = center
    if end < start:
        start, end = end, start
    declared_start = max(0, min(len(samples), start))
    declared_end = max(declared_start, min(len(samples), end))
    analysis_start, analysis_end = declared_start, declared_end
    if analysis_end - analysis_start < frame:
        analysis_start = center - frame // 2
        analysis_end = analysis_start + frame
    analysis_start = max(0, min(len(samples), analysis_start))
    analysis_end = max(analysis_start, min(len(samples), analysis_end))

    handoff_frames = _rms_frames(
        samples, analysis_start, analysis_end, frame, hop)
    left_frames = _rms_frames(
        samples, max(0, declared_start - context), declared_start,
        frame, hop)
    right_frames = _rms_frames(
        samples, declared_end, min(len(samples), declared_end + context),
        frame, hop)
    handoff_values = [row["rms"] for row in handoff_frames]
    handoff_rms = _rms(samples[analysis_start:analysis_end])
    median_rms = (float(np.median(handoff_values))
                  if handoff_values else handoff_rms)
    minimum_rms = (float(np.percentile(handoff_values, 20.0))
                   if handoff_values else handoff_rms)
    left_reference_rms = (float(np.median(
        [row["rms"] for row in left_frames])) if left_frames else 0.0)
    right_reference_rms = (float(np.median(
        [row["rms"] for row in right_frames])) if right_frames else 0.0)
    reference_rms = math.sqrt(max(
        0.0, left_reference_rms * right_reference_rms))
    if reference_rms <= _EPS:
        retention = 1.0 if minimum_rms <= _EPS else 0.0
        attenuation = 0.0
    else:
        median_retention = median_rms / reference_rms
        sustained_retention = minimum_rms / reference_rms
        retention = min(median_retention, sustained_retention)
        attenuation = max(0.0, min(
            120.0, 20.0 * math.log10(
                reference_rms / max(_EPS, minimum_rms))))
    expected = _expected_low_energy_handoff(phone)
    eligible = bool(
        not expected and
        left_reference_rms >= config.content_dropout_reference_rms and
        right_reference_rms >= config.content_dropout_reference_rms)
    return {
        "handoff_start_sample": declared_start,
        "handoff_end_sample": declared_end,
        "analysis_start_sample": analysis_start,
        "analysis_end_sample": analysis_end,
        "frame_samples": frame,
        "pitch_synchronous": bool(voiced and period_samples),
        "handoff_rms": handoff_rms,
        "median_handoff_frame_rms": median_rms,
        "minimum_handoff_frame_rms": minimum_rms,
        "left_reference_rms": left_reference_rms,
        "right_reference_rms": right_reference_rms,
        "reference_rms": reference_rms,
        "retention_ratio": retention,
        "attenuation_db": attenuation,
        "expected": expected,
        "eligible": eligible,
        "reason": (
            "pause-or-stop-closure" if expected else
            "audible-reference" if eligible else
            "reference-below-analysis-floor"),
        "frames": {
            "handoff": handoff_frames,
            "left_reference": left_frames,
            "right_reference": right_frames,
        },
    }


def _voicing_confidence(periodicity: float, rms: float, phone: str,
                        config: "JoinAnalysisConfig") -> tuple[float, dict]:
    periodic = _smoothstep(
        periodicity, config.periodicity_threshold,
        config.full_voicing_periodicity)
    energy = _smoothstep(
        rms, config.voiced_minimum_rms, config.full_voicing_rms)
    prior, prior_class = _phone_voicing_prior(phone)
    # A strong periodic signal can overcome a contradictory label, but the
    # prior lowers confidence for closures, bursts, aspiration, and noise.
    confidence = periodic * energy * (0.75 + 0.25 * prior)
    return confidence, {
        "periodicity_support": periodic,
        "energy_support": energy,
        "phone_prior": prior,
        "phone_prior_class": prior_class,
    }


def _continuous_component(
    raw_value: float | None,
    raw_reference: float,
    novelty: float | None,
    novelty_reference: float,
    *,
    energy_gate: float = 1.0,
    novelty_support_scale: float = 0.25,
    local_normality_discount: float = 0.95,
) -> dict:
    """Keep absolute and local-unusual evidence separate, then rank smoothly.

    A large raw change is weaker evidence when equally large changes are normal
    immediately inside both source units.  The raw metric remains untouched;
    only its ranking contribution receives this continuous local-normality
    discount.  Missing novelty evidence never suppresses the absolute score.
    """
    raw = max(0.0, float(raw_value or 0.0))
    absolute_score = raw / max(_EPS, float(raw_reference))
    novelty_score = max(0.0, float(novelty or 0.0)) / max(
        _EPS, float(novelty_reference))
    support = absolute_score / (absolute_score + float(novelty_support_scale))
    unusual_score = novelty_score * support
    novelty_gate = (1.0 if novelty is None else
                    _smoothstep(novelty_score, 0.0, 1.0))
    normality_discount = (0.0 if novelty is None else
                          max(0.0, min(0.999,
                              float(local_normality_discount))) *
                          support * (1.0 - novelty_gate))
    ranking_absolute_score = absolute_score * (1.0 - normality_discount)
    gate = max(0.0, min(1.0, float(energy_gate)))
    combined = gate * math.sqrt(
        ranking_absolute_score * ranking_absolute_score +
        unusual_score * unusual_score)
    return {
        "raw_value": raw,
        "raw_reference": float(raw_reference),
        "absolute_score": absolute_score,
        "local_novelty": None if novelty is None else max(0.0, float(novelty)),
        "novelty_reference": float(novelty_reference),
        "novelty_score": novelty_score,
        "novelty_support": support,
        "novelty_gate": novelty_gate,
        "local_normality_discount": normality_discount,
        "ranking_absolute_score": ranking_absolute_score,
        "locally_unusual_score": unusual_score,
        "energy_gate": gate,
        "combined_score": combined,
    }


def _robust_baseline(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray([float(value) for value in values
                       if math.isfinite(float(value))], dtype=np.float64)
    if not data.size:
        return 0.0, 0.0
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return median, mad


def _novelty(value: float | None, baseline: Sequence[float], *,
             scale_floor: float = 0.0) -> tuple[float | None, dict]:
    if value is None or not math.isfinite(float(value)):
        return None, {"median": None, "mad": None, "count": 0}
    clean = [float(item) for item in baseline if math.isfinite(float(item))]
    median, mad = _robust_baseline(clean)
    # The small relative floor avoids infinities for perfectly stationary test
    # tones while still making a new non-zero boundary event conspicuous.
    scale = max(1.4826 * mad,
                0.025 * max(abs(float(value)), abs(median), 1.0e-4),
                float(scale_floor),
                1.0e-7)
    score = max(0.0, (float(value) - median) / scale)
    return score, {
        "median": _finite(median), "mad": _finite(mad),
        "count": len(clean),
    }


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    count = max(2, int(count))
    if values.size == count:
        return values.copy()
    if values.size < 2:
        return np.zeros(count, dtype=np.float64)
    return np.interp(
        np.linspace(0.0, 1.0, count, endpoint=False),
        np.linspace(0.0, 1.0, values.size, endpoint=False),
        values,
    )


def _normalise_period(values: np.ndarray, count: int) -> np.ndarray | None:
    frame = _resample(np.asarray(values, dtype=np.float64), count)
    frame -= float(np.mean(frame))
    energy = _rms(frame)
    if energy <= _EPS:
        return None
    return frame / energy


def _period_correlations(left: np.ndarray, right: np.ndarray,
                         maximum_lag_cycles: float) -> dict | None:
    count = max(64, min(512, int(round((len(left) + len(right)) * 0.5))))
    a = _normalise_period(left, count)
    b = _normalise_period(right, count)
    if a is None or b is None:
        return None
    zero = float(np.mean(a * b))
    lag_limit = max(1, int(round(count * float(maximum_lag_cycles))))
    lags = np.arange(-lag_limit, lag_limit + 1, dtype=np.int64)
    indices = ((np.arange(count, dtype=np.int64)[None, :] - lags[:, None]) %
               count)
    scores = np.asarray(b[indices] @ a / count, dtype=np.float64)
    best_offset = int(np.argmax(scores))
    if float(scores[best_offset]) > zero + 1.0e-12:
        best_lag = int(lags[best_offset])
        best = float(scores[best_offset])
        best_wave = np.roll(b, best_lag)
    else:
        best_lag = 0
        best = zero
        best_wave = b.copy()
    error = float(math.sqrt(max(0.0, np.mean((a - best_wave) ** 2))) / 2.0)
    return {
        "zero": max(-1.0, min(1.0, zero)),
        "best": max(-1.0, min(1.0, best)),
        "lag": int(best_lag),
        "lag_cycles": float(best_lag) / count,
        "shape_error": error,
        "left_normalised": a,
        "right_normalised": b,
        "right_aligned": best_wave,
    }


def _lag_periodicity(values: np.ndarray, period: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    period = int(period)
    if period < 2 or values.size < period * 2:
        return 0.0
    a = values[:-period] - float(np.mean(values[:-period]))
    b = values[period:] - float(np.mean(values[period:]))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator <= _EPS:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b)) / denominator))


def _estimate_period(values: np.ndarray, sample_rate: int,
                     minimum_hz: float, maximum_hz: float) -> tuple[int | None,
                                                                    float]:
    """Autocorrelation fallback, always confined to one side of a splice."""
    frame = np.asarray(values, dtype=np.float64)
    if frame.size < 32 or _rms(frame) <= _EPS:
        return None, 0.0
    frame = (frame - float(np.mean(frame))) * np.hanning(frame.size)
    minimum_lag = max(2, int(math.floor(sample_rate / maximum_hz)))
    maximum_lag = min(frame.size // 2,
                      int(math.ceil(sample_rate / minimum_hz)))
    if maximum_lag <= minimum_lag:
        return None, 0.0
    size = 1 << int(math.ceil(math.log2(frame.size * 2 - 1)))
    spectrum = np.fft.rfft(frame, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:frame.size]
    lags = np.arange(minimum_lag, maximum_lag + 1, dtype=np.int64)
    squared_prefix = np.concatenate((
        np.zeros(1, dtype=np.float64),
        np.cumsum(frame * frame, dtype=np.float64),
    ))
    left_energy = squared_prefix[frame.size - lags]
    right_energy = squared_prefix[-1] - squared_prefix[lags]
    denominator = np.sqrt(np.maximum(0.0, left_energy * right_energy))
    scores = np.divide(
        correlation[lags], denominator,
        out=np.zeros_like(denominator), where=denominator > _EPS)
    best_offset = int(np.argmax(scores))
    return int(lags[best_offset]), float(scores[best_offset])


def _sample_bounds(first: float, last: float, sample_rate: int,
                   sample_count: int) -> tuple[float, float] | None:
    # Keep target-epoch positions fractional.  Rounding a 66.67-sample pitch
    # period to 66 or 67 samples changes the apparent harmonic shape even when
    # the underlying waveform is continuous.  _frames interpolates these
    # bounds without ever sampling across the splice.
    start = max(0.0, float(first) * sample_rate)
    end = min(float(sample_count), float(last) * sample_rate)
    return (start, end) if end - start >= 3 else None


def _period_bounds_from_marks(marks: np.ndarray, splice_time: float,
                              sample_rate: int, sample_count: int,
                              count: int) -> tuple[list[tuple[float, float]],
                                                   list[tuple[float, float]]]:
    if len(marks) < 2:
        return [], []
    # Binary searches keep long-utterance diagnostics local. No period in
    # either list crosses the splice.
    left_end = int(np.searchsorted(marks, splice_time, side="right")) - 1
    left_times = [
        (float(marks[index]), float(marks[index + 1]))
        for index in range(max(0, left_end - count), max(0, left_end))
    ]
    right_start = int(np.searchsorted(marks, splice_time, side="left"))
    right_times = [
        (float(marks[index]), float(marks[index + 1]))
        for index in range(right_start,
                           min(len(marks) - 1, right_start + count))
    ]
    left = [_sample_bounds(a, b, sample_rate, sample_count)
            for a, b in left_times]
    right = [_sample_bounds(a, b, sample_rate, sample_count)
             for a, b in right_times]
    return ([item for item in left if item],
            [item for item in right if item])


def _regular_period_bounds(splice_sample: int, left_period: int,
                           right_period: int, sample_count: int,
                           count: int) -> tuple[list[tuple[int, int]],
                                                list[tuple[int, int]]]:
    left = []
    for offset in range(count, 0, -1):
        a = splice_sample - offset * left_period
        b = a + left_period
        if 0 <= a < b <= splice_sample:
            left.append((a, b))
    right = []
    for offset in range(count):
        a = splice_sample + offset * right_period
        b = a + right_period
        if splice_sample <= a < b <= sample_count:
            right.append((a, b))
    return left, right


def _fixed_frame_bounds(splice_sample: int, sample_rate: int,
                        sample_count: int, frame_ms: float, hop_ms: float,
                        count: int) -> tuple[list[tuple[int, int]],
                                             list[tuple[int, int]]]:
    frame = max(8, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    left = []
    for offset in range(count - 1, -1, -1):
        end = splice_sample - offset * hop
        start = end - frame
        if 0 <= start < end <= splice_sample:
            left.append((start, end))
    right = []
    for offset in range(count):
        start = splice_sample + offset * hop
        end = start + frame
        if splice_sample <= start < end <= sample_count:
            right.append((start, end))
    return left, right


def _broadband_frame_feature(values: np.ndarray, sample_rate: int,
                             config: JoinAnalysisConfig) -> dict | None:
    """Describe whether a very short frame contains impulse-like broadband energy.

    Median energy in several frequency bands is deliberate here. A voiced frame
    has narrow harmonic peaks, while a click raises the spectral floor between
    those harmonics across nearly the whole spectrum. Band medians therefore
    expose the click without mistaking a strong fundamental for broadband
    energy. Absolute level is retained separately from this spectral shape.
    """
    raw = np.asarray(values, dtype=np.float64)
    if raw.size < 8:
        return None
    centered = raw - float(np.mean(raw))
    frame_rms = _rms(centered)
    peak = float(np.max(np.abs(centered)))
    window = np.hanning(raw.size)
    fft_size = max(
        256,
        int(config.spectral_fft_size),
        1 << int(math.ceil(math.log2(max(8, raw.size * 2)))),
    )
    spectrum = np.fft.rfft(centered * window, fft_size)
    window_energy = max(_EPS, float(np.sum(window * window)))
    power = np.abs(spectrum) ** 2 / window_energy
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    lower = float(config.broadband_min_frequency_hz)
    upper = (sample_rate * 0.5 *
             float(config.broadband_max_frequency_fraction))
    usable = (frequencies >= lower) & (frequencies <= upper)
    if int(np.count_nonzero(usable)) < int(config.broadband_band_count) * 2:
        return None
    usable_power = power[usable]
    maximum_power = float(np.max(usable_power))
    if frame_rms <= _EPS or maximum_power <= _EPS:
        return {
            "rms": frame_rms,
            "temporal_crest": 0.0,
            "bin_flatness": 0.0,
            "band_flatness": 0.0,
            "tilt_db_per_octave": 0.0,
            "band_uniformity_db": 0.0,
            "broadband_floor": 0.0,
            "shape_score": 0.0,
        }

    edges = np.linspace(
        lower, upper, int(config.broadband_band_count) + 1,
        dtype=np.float64)
    band_power = []
    band_centers = []
    for first, last in zip(edges, edges[1:]):
        selected = power[(frequencies >= first) & (frequencies < last)]
        if not selected.size:
            return None
        # The median estimates the floor between sparse voice harmonics.
        band_power.append(float(np.median(selected)))
        band_centers.append(math.sqrt(max(_EPS, first * last)))
    bands = np.asarray(band_power, dtype=np.float64)
    band_peak = max(_EPS, float(np.max(bands)))
    band_floor = max(band_peak * 1.0e-12, _EPS)
    safe_bands = np.maximum(bands, band_floor)
    band_flatness = float(
        math.exp(float(np.mean(np.log(safe_bands)))) /
        max(_EPS, float(np.mean(safe_bands))))
    safe_bins = np.maximum(usable_power, max(maximum_power * 1.0e-12, _EPS))
    bin_flatness = float(
        math.exp(float(np.mean(np.log(safe_bins)))) /
        max(_EPS, float(np.mean(safe_bins))))
    band_db = 10.0 * np.log10(safe_bands / band_peak)
    octave = np.log2(np.asarray(band_centers, dtype=np.float64) / 1000.0)
    octave -= float(np.mean(octave))
    denominator = float(np.dot(octave, octave))
    tilt = (float(np.dot(octave, band_db - float(np.mean(band_db))) /
                  denominator) if denominator > _EPS else 0.0)
    uniformity = float(np.std(band_db))

    flatness_support = _smoothstep(
        band_flatness, config.broadband_flatness_floor,
        config.broadband_flatness_full)
    tilt_support = 1.0 - _smoothstep(
        abs(tilt), config.broadband_tilt_tolerance_db_per_octave,
        config.broadband_tilt_reject_db_per_octave)
    uniformity_support = 1.0 - _smoothstep(
        uniformity, config.broadband_uniformity_floor_db,
        config.broadband_uniformity_reject_db)
    shape_score = (max(0.0, flatness_support) *
                   max(0.0, tilt_support) *
                   max(0.0, uniformity_support)) ** (1.0 / 3.0)
    return {
        "rms": frame_rms,
        "temporal_crest": peak / max(frame_rms, _EPS),
        "bin_flatness": bin_flatness,
        "band_flatness": band_flatness,
        "tilt_db_per_octave": tilt,
        "band_uniformity_db": uniformity,
        "broadband_floor": math.sqrt(max(0.0, float(np.min(bands)))),
        "shape_score": shape_score,
    }


def _broadband_impulse_scale_metrics(samples: np.ndarray, sample_rate: int,
                                     splice: Mapping, splice_time: float,
                                     config: JoinAnalysisConfig,
                                     frame_ms: float) -> dict:
    """Scan one short-time resolution for a transient flat-spectrum event."""
    frame = max(8, int(round(
        sample_rate * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(
        sample_rate * config.broadband_impulse_hop_ms / 1000.0)))
    fallback_half = config.broadband_impulse_scan_ms / 2000.0
    try:
        scan_start = float(splice.get("handoff_start"))
        scan_end = float(splice.get("handoff_end"))
        if (not math.isfinite(scan_start) or not math.isfinite(scan_end) or
                scan_end <= scan_start):
            raise ValueError
    except (TypeError, ValueError):
        scan_start = splice_time - fallback_half
        scan_end = splice_time + fallback_half
    # Include a half-frame at either edge: a discontinuity reported exactly at
    # a collar boundary must still be centered in at least one analysis frame.
    half_frame_s = frame / (2.0 * sample_rate)
    scan_start = max(half_frame_s, scan_start - half_frame_s)
    scan_end = min(len(samples) / sample_rate - half_frame_s,
                   scan_end + half_frame_s)
    if scan_end <= scan_start:
        return {
            "available": False, "event_time": None, "score": None,
            "novelty": None, "baseline": {
                "median": None, "mad": None, "count": 0}, "plot": {},
        }

    def frame_at(center_sample: int) -> dict | None:
        first = int(center_sample) - frame // 2
        last = first + frame
        if first < 0 or last > len(samples):
            return None
        feature = _broadband_frame_feature(
            samples[first:last], sample_rate, config)
        if feature is None:
            return None
        feature.update({
            "center_sample": int(center_sample),
            "time": float(center_sample) / sample_rate,
        })
        return feature

    first_scan = int(math.ceil(scan_start * sample_rate))
    last_scan = int(math.floor(scan_end * sample_rate))
    scan_centers = list(range(first_scan, last_scan + 1, hop))
    exact = int(round(splice_time * sample_rate))
    scan_centers.extend((first_scan, exact, last_scan))
    scan_frames = [frame_at(center) for center in sorted(set(scan_centers))]
    scan_frames = [item for item in scan_frames if item is not None]

    context = max(frame * 2, int(round(
        sample_rate * config.broadband_impulse_context_ms / 1000.0)))
    baseline_hop = max(hop, frame // 2)
    left_last = first_scan - frame
    right_first = last_scan + frame
    baseline_centers = list(range(
        max(frame // 2, left_last - context), left_last + 1, baseline_hop))
    baseline_centers.extend(range(
        right_first,
        min(len(samples) - frame // 2, right_first + context) + 1,
        baseline_hop))
    baseline_frames = [frame_at(center) for center in baseline_centers]
    baseline_frames = [item for item in baseline_frames if item is not None]
    strengths = [float(item["broadband_floor"])
                 for item in baseline_frames]
    baseline_flatness = [float(item["band_flatness"])
                         for item in baseline_frames]
    baseline_tilt_consistency = [
        1.0 / (1.0 + abs(float(item["tilt_db_per_octave"])))
        for item in baseline_frames]
    baseline_uniformity = [
        1.0 / (1.0 + float(item["band_uniformity_db"]))
        for item in baseline_frames]
    median, mad = _robust_baseline(strengths)
    local_rms = _rms(samples[max(0, first_scan - context):min(
        len(samples), last_scan + context)])
    baseline_scale = max(median + 1.4826 * mad,
                         local_rms * 1.0e-7, _EPS)

    for item in scan_frames:
        strength = float(item["broadband_floor"])
        ratio = strength / baseline_scale
        ratio_support = _smoothstep(
            ratio, config.broadband_energy_ratio_gate,
            config.broadband_energy_ratio_full)
        flatness_novelty, _unused = _novelty(
            float(item["band_flatness"]), baseline_flatness)
        tilt_consistency = 1.0 / (
            1.0 + abs(float(item["tilt_db_per_octave"])))
        tilt_novelty, _unused = _novelty(
            tilt_consistency, baseline_tilt_consistency)
        uniformity_consistency = 1.0 / (
            1.0 + float(item["band_uniformity_db"]))
        uniformity_novelty, _unused = _novelty(
            uniformity_consistency, baseline_uniformity)
        relative_supports = sorted((
            _smoothstep(float(flatness_novelty or 0.0), 2.5, 8.0),
            _smoothstep(float(tilt_novelty or 0.0), 2.5, 8.0),
            _smoothstep(float(uniformity_novelty or 0.0), 2.5, 8.0),
        ), reverse=True)
        # Require two independent signs of spectral flattening. A single
        # noisy band can change one statistic, but a full-spectrum impulse
        # raises flatness while reducing both tilt and inter-band spread.
        relative_shape = math.sqrt(
            relative_supports[0] * relative_supports[1])
        absolute_shape = float(item["shape_score"])
        item["strength"] = strength
        item["local_energy_ratio"] = ratio
        item["ratio_support"] = ratio_support
        item["absolute_shape_score"] = absolute_shape
        item["relative_shape_score"] = relative_shape
        item["flatness_novelty"] = flatness_novelty
        item["tilt_flattening_novelty"] = tilt_novelty
        item["uniformity_novelty"] = uniformity_novelty
        item["novelty"], _unused = _novelty(
            strength, strengths, scale_floor=local_rms * 1.0e-7)
        floor_novelty_support = _smoothstep(
            float(item["novelty"] or 0.0),
            config.broadband_floor_novelty_gate,
            config.broadband_floor_novelty_full)
        item["floor_novelty_support"] = floor_novelty_support
        # A sudden rise in spectral-floor energy is useful corroborating
        # evidence, but is not itself evidence of a flat, broadband shape.
        # Keep it as a ranking influence only after an amplitude-independent
        # absolute or locally relative shape test has succeeded.
        shape_evidence = max(absolute_shape, relative_shape)
        novelty_rank_support = 0.75 + 0.25 * floor_novelty_support
        item["event_score"] = float(
            shape_evidence * ratio_support * novelty_rank_support)
    if not scan_frames:
        return {
            "available": False, "event_time": None, "score": None,
            "novelty": None, "baseline": {
                "median": _finite(median), "mad": _finite(mad),
                "count": len(strengths)}, "plot": {},
        }
    event = max(scan_frames, key=lambda item: (
        float(item["event_score"]), float(item["novelty"] or 0.0),
        float(item["strength"]), -abs(float(item["time"]) - splice_time)))
    energy_gate = _smoothstep(
        float(event["rms"]), config.minimum_rms,
        config.voiced_minimum_rms)
    return {
        "available": True,
        "frame_ms": float(frame_ms),
        "event_time": float(event["time"]),
        "event_sample": int(event["center_sample"]),
        "score": float(event["event_score"]),
        "novelty": float(event["novelty"] or 0.0),
        "energy_gate": energy_gate,
        "rms": float(event["rms"]),
        "temporal_crest": float(event["temporal_crest"]),
        "bin_flatness": float(event["bin_flatness"]),
        "band_flatness": float(event["band_flatness"]),
        "tilt_db_per_octave": float(event["tilt_db_per_octave"]),
        "band_uniformity_db": float(event["band_uniformity_db"]),
        "absolute_shape_score": float(event["absolute_shape_score"]),
        "relative_shape_score": float(event["relative_shape_score"]),
        "flatness_novelty": float(event["flatness_novelty"] or 0.0),
        "tilt_flattening_novelty": float(
            event["tilt_flattening_novelty"] or 0.0),
        "uniformity_novelty": float(event["uniformity_novelty"] or 0.0),
        "floor_novelty_support": float(event["floor_novelty_support"]),
        "broadband_floor": float(event["broadband_floor"]),
        "local_energy_ratio": float(event["local_energy_ratio"]),
        "scan_start": scan_start,
        "scan_end": scan_end,
        "baseline": {
            "median": _finite(median), "mad": _finite(mad),
            "count": len(strengths),
        },
        "plot": {
            "frame_ms": float(frame_ms),
            "scan_times": [float(item["time"]) for item in scan_frames],
            "event_scores": [float(item["event_score"])
                             for item in scan_frames],
            "broadband_strengths": [float(item["strength"])
                                    for item in scan_frames],
            "event_time": float(event["time"]),
        },
    }


def _broadband_impulse_metrics(samples: np.ndarray, sample_rate: int,
                               splice: Mapping, splice_time: float,
                               config: JoinAnalysisConfig) -> dict:
    """Use several short-time scales so one-sample and wider cracks survive."""
    measured = [
        _broadband_impulse_scale_metrics(
            samples, sample_rate, splice, splice_time, config, frame_ms)
        for frame_ms in sorted(set(
            float(value) for value
            in config.broadband_impulse_frame_scales_ms))
    ]
    available = [item for item in measured if item.get("available")]
    if not available:
        return {
            "available": False, "event_time": None, "score": None,
            "novelty": None, "baseline": {
                "median": None, "mad": None, "count": 0}, "plot": {},
            "scales": measured,
        }

    def evidence(item: dict) -> float:
        component = _continuous_component(
            item.get("score"), config.broadband_impulse_score,
            item.get("novelty"), config.broadband_impulse_novelty,
            energy_gate=float(item.get("energy_gate") or 0.0),
            novelty_support_scale=config.novelty_support_scale,
            local_normality_discount=config.local_normality_discount)
        return float(component["combined_score"])

    best = dict(max(available, key=lambda item: (
        evidence(item), float(item.get("score") or 0.0),
        float(item.get("novelty") or 0.0), -float(item.get("frame_ms") or 0.0))))
    best["scale_evidence"] = evidence(best)
    best["scales"] = [{
        "frame_ms": _finite(item.get("frame_ms"), 4),
        "event_time": _finite(item.get("event_time"), 9),
        "score": _finite(item.get("score")),
        "novelty": _finite(item.get("novelty")),
        "energy_gate": _finite(item.get("energy_gate")),
        "evidence": _finite(evidence(item)),
    } for item in available]
    best["plot"] = dict(best.get("plot") or {})
    best["plot"]["tested_scales"] = best["scales"]
    return best


def _frames(samples: np.ndarray, bounds: Sequence[tuple[float, float]],
            sample_rate: int) -> list[dict]:
    rows = []
    for start, end in bounds:
        start_f, end_f = float(start), float(end)
        if end_f <= start_f:
            continue
        count = max(3, int(round(end_f - start_f)))
        positions = np.linspace(start_f, end_f, count, endpoint=False)
        lower = np.clip(np.floor(positions).astype(np.int64),
                        0, max(0, len(samples) - 1))
        upper = np.minimum(lower + 1, max(0, len(samples) - 1))
        fraction = positions - lower
        frame = (np.asarray(samples[lower], dtype=np.float64) *
                 (1.0 - fraction) +
                 np.asarray(samples[upper], dtype=np.float64) * fraction)
        rows.append({
            "start": start_f, "end": end_f,
            "time": (start_f + end_f) * 0.5 / sample_rate,
            "period_s": (end_f - start_f) / sample_rate,
            "f0_hz": sample_rate / (end_f - start_f),
            "rms": _rms(frame),
            "samples": np.asarray(frame, dtype=np.float64),
        })
    return rows


def _cepstral_feature(values: np.ndarray, coefficient_count: int,
                       fft_size: int, *, periodic: bool = False,
                       harmonic_floor: float = 0.01) -> np.ndarray:
    frame = np.asarray(values, dtype=np.float64)
    if frame.size < 4 or _rms(frame) <= _EPS:
        return np.zeros(coefficient_count, dtype=np.float64)
    frame = frame - float(np.mean(frame))
    if periodic:
        # A complete pitch period is a cyclic signal.  Resampling one cycle
        # to a fixed-length rectangular DFT makes harmonic magnitudes
        # insensitive to circular phase/cut position and to absolute F0.  A
        # Hann window here would turn a pure phase offset into a false timbre
        # discontinuity.  Bin 0 and the overall peak are removed, so this is
        # a compact log-harmonic-shape feature rather than another level test.
        size = max(128, int(fft_size))
        frame = _resample(frame, size)
        magnitude = np.abs(np.fft.rfft(frame, size))
        harmonics = magnitude[1:coefficient_count + 1]
        if harmonics.size < coefficient_count:
            harmonics = np.pad(
                harmonics, (0, coefficient_count - harmonics.size))
        peak = max(_EPS, float(np.max(harmonics)))
        return np.log(np.maximum(
            harmonics / peak, float(harmonic_floor))) / 8.0
    else:
        frame *= np.hanning(frame.size)
        size = max(int(fft_size),
                   1 << int(math.ceil(math.log2(frame.size))))
    magnitude = np.abs(np.fft.rfft(frame, size))
    peak = max(_EPS, float(np.max(magnitude)))
    # Dividing before the logarithm and excluding c0 make the feature
    # amplitude-independent; RMS remains a separate metric.
    log_magnitude = np.log(np.maximum(magnitude / peak, 1.0e-8))
    cepstrum = np.fft.irfft(log_magnitude, size)
    return np.asarray(cepstrum[1:coefficient_count + 1], dtype=np.float64)


def _feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return 0.0
    return float(math.sqrt(max(0.0, float(np.mean((a - b) ** 2)))))


def _trajectory(frames: Sequence[dict], splice_time: float,
                config: JoinAnalysisConfig, *, periodic: bool) -> dict | None:
    if len(frames) < 2:
        return None
    times = np.asarray([float(frame["time"]) - splice_time
                        for frame in frames], dtype=np.float64)
    features = np.asarray([
        _cepstral_feature(frame["samples"], config.spectral_coefficients,
                          config.spectral_fft_size, periodic=periodic,
                          harmonic_floor=config.voiced_harmonic_floor)
        for frame in frames
    ], dtype=np.float64)
    deltas = np.diff(times)
    usable = np.abs(deltas) > 1.0e-9
    increments = (np.diff(features, axis=0)[usable] /
                  deltas[usable, None])
    pairwise_slopes = []
    for first in range(len(times) - 1):
        for second in range(first + 1, len(times)):
            delta = float(times[second] - times[first])
            if abs(delta) > 1.0e-9:
                pairwise_slopes.append(
                    (features[second] - features[first]) / delta)
    if pairwise_slopes:
        # Theil-Sen-style component medians use every frame separation and
        # do not let one stochastic unvoiced frame dictate an extrapolation.
        slope = np.median(np.asarray(pairwise_slopes), axis=0)
        intercept = np.median(
            features - times[:, None] * slope[None, :], axis=0)
    else:
        design = np.column_stack((
            times, np.ones(len(times), dtype=np.float64)))
        slope, intercept = np.linalg.lstsq(design, features, rcond=None)[0]
    return {
        "times": times,
        "features": features,
        "slope": slope,
        "intercept": intercept,
        "increments": increments,
    }


def _spectral_metrics(left_frames: Sequence[dict], right_frames: Sequence[dict],
                      splice_time: float, reference_period: float,
                      config: JoinAnalysisConfig, *, periodic: bool) -> dict:
    left = _trajectory(left_frames, splice_time, config,
                       periodic=periodic)
    right = _trajectory(right_frames, splice_time, config,
                        periodic=periodic)
    if left is None or right is None:
        return {
            "step": None, "step_novelty": None,
            "slope_break": None, "slope_novelty": None,
            "flux": None, "flux_novelty": None,
            "baseline_step": {"median": None, "mad": None, "count": 0},
            "baseline_slope": {"median": None, "mad": None, "count": 0},
            "plot": {},
        }
    step = _feature_distance(left["intercept"], right["intercept"])
    period = max(1.0e-5, float(reference_period))
    slope_break = _feature_distance(
        left["slope"] * period, right["slope"] * period)
    adjacent = []
    slope_changes = []
    for trajectory in (left, right):
        features = trajectory["features"]
        times = trajectory["times"]
        increments = []
        for index in range(len(features) - 1):
            distance = _feature_distance(features[index], features[index + 1])
            adjacent.append(distance)
            delta = max(1.0e-6, float(times[index + 1] - times[index]))
            increments.append((features[index + 1] - features[index]) /
                              delta * period)
        fitted_increment = trajectory["slope"] * period
        slope_changes.extend(
            _feature_distance(increment, fitted_increment)
            for increment in increments)
        slope_changes.extend(
            _feature_distance(increments[index], increments[index + 1])
            for index in range(len(increments) - 1))
    step_novelty, step_baseline = _novelty(step, adjacent)
    slope_novelty, slope_baseline = _novelty(
        slope_break, slope_changes,
        scale_floor=config.spectral_slope_novelty_scale_floor)
    flux = _feature_distance(
        left["features"][-1], right["features"][0])
    flux_novelty, flux_baseline = _novelty(flux, adjacent)

    gap = right["intercept"] - left["intercept"]
    norm = float(np.linalg.norm(gap))
    if norm <= _EPS:
        direction = np.zeros_like(gap)
        direction[0] = 1.0
    else:
        direction = gap / norm
    left_projection = left["features"] @ direction
    right_projection = right["features"] @ direction
    return {
        "step": step, "step_novelty": step_novelty,
        "slope_break": slope_break, "slope_novelty": slope_novelty,
        "flux": flux, "flux_novelty": flux_novelty,
        "baseline_step": step_baseline,
        "baseline_slope": slope_baseline,
        "baseline_flux": flux_baseline,
        "plot": {
            "left_times": [float(value + splice_time)
                           for value in left["times"]],
            "right_times": [float(value + splice_time)
                            for value in right["times"]],
            "left_projection": [float(value) for value in left_projection],
            "right_projection": [float(value) for value in right_projection],
            "left_intercept": float(left["intercept"] @ direction),
            "right_intercept": float(right["intercept"] @ direction),
            "left_slope": float(left["slope"] @ direction),
            "right_slope": float(right["slope"] @ direction),
        },
    }


def _lpc_coefficients(values: np.ndarray, order: int) -> np.ndarray | None:
    """Autocorrelation LPC with Levinson recursion and stability checks."""
    signal = np.asarray(values, dtype=np.float64)
    order = min(int(order), max(2, signal.size // 4))
    if signal.size < order * 2 + 1 or _rms(signal) <= _EPS:
        return None
    autocorrelation = np.asarray([
        float(np.dot(signal[:signal.size - lag], signal[lag:]))
        for lag in range(order + 1)
    ], dtype=np.float64)
    if autocorrelation[0] <= _EPS:
        return None
    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[0] = 1.0
    error = float(autocorrelation[0])
    for index in range(1, order + 1):
        residual = float(autocorrelation[index])
        if index > 1:
            residual += float(np.dot(
                coefficients[1:index],
                autocorrelation[index - 1:0:-1],
            ))
        reflection = max(-0.995, min(0.995, -residual / max(error, _EPS)))
        previous = coefficients.copy()
        coefficients[index] = reflection
        for offset in range(1, index):
            coefficients[offset] = (
                previous[offset] + reflection * previous[index - offset]
            )
        error *= max(1.0e-6, 1.0 - reflection * reflection)
        if not math.isfinite(error) or error <= _EPS:
            return None
    return coefficients


def _smoothed_spectral_envelope(values: np.ndarray, sample_rate: int,
                                fft_size: int) -> tuple[np.ndarray, np.ndarray]:
    frame = np.asarray(values, dtype=np.float64)
    if frame.size < 8:
        return np.asarray([], np.float64), np.asarray([], np.float64)
    frame = frame - float(np.mean(frame))
    size = max(int(fft_size), 1 << int(math.ceil(math.log2(frame.size * 4))))
    magnitude = np.abs(np.fft.rfft(frame * np.hanning(frame.size), size))
    log_db = 20.0 * np.log10(np.maximum(magnitude, 1.0e-10))
    bin_hz = sample_rate / float(size)
    width = max(3, int(round(100.0 / max(bin_hz, 1.0e-6))))
    if width % 2 == 0:
        width += 1
    smooth = np.convolve(log_db, np.ones(width) / width, mode="same")
    smooth -= float(np.max(smooth))
    return np.fft.rfftfreq(size, 1.0 / sample_rate), smooth


def _estimate_formants(values: np.ndarray, sample_rate: int,
                       config: JoinAnalysisConfig) -> dict:
    """Estimate F1..F4 and retain rejected candidates for diagnostics."""
    source = np.asarray(values, dtype=np.float64)
    result = {"formants": [], "rejected": [], "confidence": 0.0}
    if source.size < max(48, config.formant_lpc_order * 3):
        result["rejected"].append("frame_too_short")
        return result
    if _rms(source) < config.voiced_minimum_rms:
        result["rejected"].append("below_formant_energy_floor")
        return result
    centered = source - float(np.mean(source))
    emphasized = centered.copy()
    emphasized[1:] -= float(config.formant_preemphasis) * centered[:-1]
    emphasized *= np.hamming(emphasized.size)
    coefficients = _lpc_coefficients(emphasized, config.formant_lpc_order)
    if coefficients is None:
        result["rejected"].append("unstable_lpc")
        return result
    frequencies, envelope = _smoothed_spectral_envelope(
        centered, sample_rate, max(2048, config.spectral_fft_size * 4))
    if not len(frequencies):
        result["rejected"].append("spectral_envelope_unavailable")
        return result
    power = np.power(10.0, envelope / 10.0)
    usable_total = float(np.sum(power[
        (frequencies >= config.formant_min_hz) &
        (frequencies <= min(config.formant_max_hz, sample_rate * 0.48))
    ])) + _EPS
    candidates = []
    for root in np.roots(coefficients):
        if np.imag(root) <= 0.0 or abs(root) <= _EPS:
            continue
        frequency = math.atan2(float(np.imag(root)), float(np.real(root))) \
            * sample_rate / (2.0 * math.pi)
        bandwidth = -sample_rate / math.pi * math.log(max(abs(root), _EPS))
        reason = None
        if not config.formant_min_hz <= frequency <= min(
                config.formant_max_hz, sample_rate * 0.48):
            reason = "frequency_out_of_range"
        elif not config.formant_min_bandwidth_hz <= bandwidth <= \
                config.formant_max_bandwidth_hz:
            reason = "bandwidth_out_of_range"
        index = int(np.argmin(np.abs(frequencies - frequency)))
        radius = max(2, int(round(280.0 / max(
            frequencies[1] - frequencies[0], 1.0e-6))))
        lower = max(0, index - radius)
        upper = min(len(envelope), index + radius + 1)
        neighborhood = np.concatenate((
            envelope[lower:max(lower, index - radius // 3)],
            envelope[min(upper, index + radius // 3 + 1):upper],
        ))
        prominence = float(envelope[index] - np.median(neighborhood)) \
            if neighborhood.size else 0.0
        band_radius = max(50.0, min(450.0, bandwidth * 0.75))
        band = np.abs(frequencies - frequency) <= band_radius
        energy = float(np.sum(power[band]) / usable_total)
        if reason is None and prominence < config.formant_min_prominence_db:
            reason = "insufficient_prominence"
        row = {
            "frequency_hz": float(frequency),
            "bandwidth_hz": float(bandwidth),
            "prominence_db": float(prominence),
            "normalized_energy": float(max(0.0, energy)),
        }
        if reason:
            result["rejected"].append({**row, "reason": reason})
            continue
        prominence_support = _smoothstep(
            prominence, config.formant_min_prominence_db,
            config.formant_min_prominence_db + 10.0)
        bandwidth_support = 1.0 - _smoothstep(
            bandwidth, config.formant_max_bandwidth_hz * 0.55,
            config.formant_max_bandwidth_hz)
        row["confidence"] = float(max(
            0.0, min(1.0, 0.65 * prominence_support +
                     0.35 * bandwidth_support)))
        candidates.append(row)
    candidates.sort(key=lambda item: item["frequency_hz"])
    # Explicit regions prevent a missing weak F1 from silently relabeling F2
    # as F1. Candidates close to a region boundary remain visible but are not
    # forced into either trajectory.
    boundaries = np.asarray((1100.0, 2100.0, 3100.0, 4100.0, 5100.0),
                            dtype=np.float64)
    assignments = {}
    for candidate in candidates:
        frequency = float(candidate["frequency_hz"])
        boundary_distance = (float(np.min(np.abs(boundaries - frequency)))
                             if boundaries.size else math.inf)
        if boundary_distance < 75.0:
            result["rejected"].append({
                **candidate, "reason": "ambiguous_formant_region"})
            continue
        track_index = min(int(config.formant_count) - 1,
                          int(np.searchsorted(boundaries, frequency)))
        candidate = {**candidate, "track_index": track_index,
                     "name": f"F{track_index + 1}"}
        previous = assignments.get(track_index)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            if previous is not None:
                result["rejected"].append({
                    **previous, "reason": "duplicate_formant_region"})
            assignments[track_index] = candidate
        else:
            result["rejected"].append({
                **candidate, "reason": "duplicate_formant_region"})
    result["formants"] = [assignments[index] for index in sorted(assignments)]
    result["confidence"] = (float(np.mean([
        item["confidence"] for item in result["formants"]
    ])) if result["formants"] else 0.0)
    return result


def _formant_observations(frames: Sequence[dict], sample_rate: int,
                          config: JoinAnalysisConfig) -> list[dict]:
    observations = []
    for index, frame in enumerate(frames):
        # A single pitch cycle is often too short for stable LPC. Concatenate
        # up to three adjacent periods from this side only; no window can cross
        # the splice because ``frames`` is already side-separated.
        first = max(0, index - 1)
        last = min(len(frames), index + 2)
        samples = np.concatenate([
            np.asarray(item["samples"], dtype=np.float64)
            for item in frames[first:last]
        ])
        estimate = _estimate_formants(samples, sample_rate, config)
        observations.append({
            "time": float(frame["time"]),
            "formants": estimate["formants"],
            "rejected": estimate["rejected"],
            "confidence": estimate["confidence"],
        })
    return observations


def _scalar_trajectory(times: Sequence[float], values: Sequence[float],
                       splice_time: float) -> dict | None:
    if len(values) < 2 or len(times) != len(values):
        return None
    x = np.asarray(times, dtype=np.float64) - float(splice_time)
    y = np.asarray(values, dtype=np.float64)
    slopes = []
    for first in range(len(x) - 1):
        for second in range(first + 1, len(x)):
            delta = float(x[second] - x[first])
            if abs(delta) > 1.0e-9:
                slopes.append((y[second] - y[first]) / delta)
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - x * slope))
    return {"times": x, "values": y, "slope": slope,
            "intercept": intercept}


def _vector_trajectory(times: Sequence[float], values: Sequence[np.ndarray],
                       splice_time: float) -> dict | None:
    if len(values) < 2 or len(times) != len(values):
        return None
    x = np.asarray(times, dtype=np.float64) - float(splice_time)
    y = np.asarray(values, dtype=np.float64)
    slopes = []
    for first in range(len(x) - 1):
        for second in range(first + 1, len(x)):
            delta = float(x[second] - x[first])
            if abs(delta) > 1.0e-9:
                slopes.append((y[second] - y[first]) / delta)
    slope = np.median(np.asarray(slopes), axis=0) if slopes else np.zeros(y.shape[1])
    intercept = np.median(y - x[:, None] * slope[None, :], axis=0)
    return {"times": x, "values": y, "slope": slope,
            "intercept": intercept}


def _formant_series(observations: Sequence[dict], formant_index: int,
                    key: str) -> tuple[list[float], list[float]]:
    times, values = [], []
    for observation in observations:
        formants = observation.get("formants") or []
        formant = next((item for item in formants
                        if int(item.get("track_index", -1)) == formant_index),
                       None)
        if formant is None:
            continue
        value = formant.get(key)
        if value is not None and math.isfinite(float(value)):
            times.append(float(observation["time"]))
            values.append(float(value))
    return times, values


def _adjacent_normalized(values: Sequence[float], *, logarithmic=False) \
        -> list[float]:
    result = []
    for left, right in zip(values, values[1:]):
        if logarithmic:
            result.append(abs(math.log((right + _EPS) / (left + _EPS))))
        else:
            result.append(abs(right - left) / max(_EPS, (abs(left) + abs(right)) * .5))
    return result


def _formant_metrics(left_frames: Sequence[dict], right_frames: Sequence[dict],
                     sample_rate: int, splice_time: float,
                     reference_period: float, config: JoinAnalysisConfig,
                     *, eligible: bool) -> dict:
    empty = {
        "available": False, "reason": "formants_require_voiced_context",
        "tracking_confidence": 0.0, "per_formant": [],
        "measured_track_count": 0, "classification_track_count": 0,
        "frequency_jump_normalized": None, "frequency_jump_novelty": None,
        "slope_break": None, "slope_break_novelty": None,
        "bandwidth_jump": None, "prominence_jump": None,
        "balance_jump": None, "balance_novelty": None,
        "balance_slope_break": None, "balance_slope_novelty": None,
        "plot": {}, "baselines": {},
    }
    if not eligible:
        return empty
    left_observations = _formant_observations(left_frames, sample_rate, config)
    right_observations = _formant_observations(right_frames, sample_rate, config)
    per_formant = []
    frequency_values = []
    frequency_novelties = []
    slope_values = []
    slope_novelties = []
    bandwidth_values = []
    prominence_values = []
    frequency_baselines = []
    slope_baselines = []
    trajectory_plot = []
    for formant_index in range(int(config.formant_count)):
        series = {}
        for side, observations in (("left", left_observations),
                                   ("right", right_observations)):
            for key in ("frequency_hz", "bandwidth_hz", "prominence_db",
                        "normalized_energy"):
                times, values = _formant_series(observations, formant_index, key)
                series[(side, key)] = (times, values,
                                       _scalar_trajectory(times, values, splice_time))
        left_frequency = series[("left", "frequency_hz")][2]
        right_frequency = series[("right", "frequency_hz")][2]
        if left_frequency is None or right_frequency is None:
            per_formant.append({
                "name": f"F{formant_index + 1}", "available": False,
                "reason": "missing_or_ambiguous_track",
            })
            continue
        left_f = float(left_frequency["intercept"])
        right_f = float(right_frequency["intercept"])
        maximum_formant = min(
            float(config.formant_max_hz), sample_rate * 0.5
        )
        if not (
            math.isfinite(left_f) and math.isfinite(right_f)
            and config.formant_min_hz <= left_f <= maximum_formant
            and config.formant_min_hz <= right_f <= maximum_formant
        ):
            per_formant.append({
                "name": f"F{formant_index + 1}", "available": False,
                "reason": "trajectory_extrapolation_out_of_range",
            })
            continue
        frequency_jump_hz = right_f - left_f
        frequency_normalized = abs(frequency_jump_hz) / ((left_f + right_f) * .5)
        adjacent_frequency = (
            _adjacent_normalized(series[("left", "frequency_hz")][1]) +
            _adjacent_normalized(series[("right", "frequency_hz")][1]))
        frequency_novelty, frequency_baseline = _novelty(
            frequency_normalized, adjacent_frequency, scale_floor=0.005)
        slope_break = abs(left_frequency["slope"] - right_frequency["slope"]) \
            * max(1.0e-5, reference_period) / ((left_f + right_f) * .5)
        local_slopes = []
        for side in ("left", "right"):
            times, values, trajectory = series[(side, "frequency_hz")]
            if trajectory is None:
                continue
            for first, second in zip(range(len(times)), range(1, len(times))):
                delta = max(1.0e-6, abs(times[second] - times[first]))
                step_slope = (values[second] - values[first]) / delta
                local_slopes.append(
                    abs(step_slope - trajectory["slope"])
                    * reference_period / max(_EPS, trajectory["intercept"])
                )
        slope_novelty, slope_baseline = _novelty(
            slope_break, local_slopes, scale_floor=0.003)
        bandwidth_jump = prominence_jump = energy_jump = None
        left_bw = series[("left", "bandwidth_hz")][2]
        right_bw = series[("right", "bandwidth_hz")][2]
        if left_bw and right_bw:
            right_bandwidth = float(right_bw["intercept"])
            left_bandwidth = float(left_bw["intercept"])
            if (
                config.formant_min_bandwidth_hz
                <= left_bandwidth
                <= config.formant_max_bandwidth_hz
                and config.formant_min_bandwidth_hz
                <= right_bandwidth
                <= config.formant_max_bandwidth_hz
            ):
                bandwidth_jump = abs(math.log(
                    right_bandwidth / left_bandwidth))
        left_prom = series[("left", "prominence_db")][2]
        right_prom = series[("right", "prominence_db")][2]
        if left_prom and right_prom:
            prominence_jump = abs(right_prom["intercept"] - left_prom["intercept"])
        left_energy = series[("left", "normalized_energy")][2]
        right_energy = series[("right", "normalized_energy")][2]
        if left_energy and right_energy:
            energy_jump = abs(math.log(
                (max(0.0, right_energy["intercept"]) + _EPS) /
                (max(0.0, left_energy["intercept"]) + _EPS)))
        confidence_values = []
        for observations in (left_observations, right_observations):
            confidence_values.extend(
                float(formant.get("confidence", 0.0))
                for item in observations
                for formant in item["formants"]
                if int(formant.get("track_index", -1)) == formant_index
            )
        confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
        per_formant.append({
            "name": f"F{formant_index + 1}", "available": True,
            "left_frequency_hz": left_f, "right_frequency_hz": right_f,
            "frequency_jump_hz": frequency_jump_hz,
            "frequency_jump_normalized": frequency_normalized,
            "frequency_jump_novelty": frequency_novelty,
            "frequency_slope_break": slope_break,
            "frequency_slope_break_novelty": slope_novelty,
            "bandwidth_jump_log_ratio": bandwidth_jump,
            "prominence_jump_db": prominence_jump,
            "normalized_energy_jump_log_ratio": energy_jump,
            "tracking_confidence": confidence,
            "eligible_for_classification": bool(
                confidence >= config.formant_min_tracking_confidence),
        })
        if confidence >= config.formant_min_tracking_confidence:
            frequency_values.append(frequency_normalized)
            if frequency_novelty is not None:
                frequency_novelties.append(frequency_novelty)
            slope_values.append(slope_break)
            if slope_novelty is not None:
                slope_novelties.append(slope_novelty)
            if bandwidth_jump is not None:
                bandwidth_values.append(bandwidth_jump)
            if prominence_jump is not None:
                prominence_values.append(prominence_jump)
        frequency_baselines.append(frequency_baseline)
        slope_baselines.append(slope_baseline)
        trajectory_plot.append({
            "name": f"F{formant_index + 1}",
            "left_times": series[("left", "frequency_hz")][0],
            "left_values": series[("left", "frequency_hz")][1],
            "right_times": series[("right", "frequency_hz")][0],
            "right_values": series[("right", "frequency_hz")][1],
            "left_intercept": left_f,
            "right_intercept": right_f,
            "left_slope": left_frequency["slope"],
            "right_slope": right_frequency["slope"],
        })

    def balance_rows(observations):
        times, vectors = [], []
        for observation in observations:
            formants = observation.get("formants") or []
            if len(formants) < 2:
                continue
            energies = np.zeros(int(config.formant_count), np.float64)
            for item in formants:
                track = int(item.get("track_index", -1))
                if 0 <= track < len(energies):
                    energies[track] = float(
                        item.get("normalized_energy") or 0.0)
            if np.count_nonzero(energies) < 2 or np.sum(energies) <= _EPS:
                continue
            energies = energies / float(np.sum(energies))
            vectors.append(np.log(energies + 1.0e-6))
            times.append(float(observation["time"]))
        return times, vectors, _vector_trajectory(times, vectors, splice_time)

    left_balance = balance_rows(left_observations)
    right_balance = balance_rows(right_observations)
    balance_jump = balance_novelty = None
    balance_slope = balance_slope_novelty = None
    balance_baseline = slope_balance_baseline = {
        "median": None, "mad": None, "count": 0}
    if left_balance[2] is not None and right_balance[2] is not None:
        balance_jump = _feature_distance(
            left_balance[2]["intercept"], right_balance[2]["intercept"])
        adjacent_balance = []
        local_balance_slopes = []
        for times, vectors, trajectory in (left_balance, right_balance):
            adjacent_balance.extend(
                _feature_distance(a, b) for a, b in zip(vectors, vectors[1:]))
            local_balance_slopes.extend(
                _feature_distance(
                    (vectors[index + 1] - vectors[index]) * reference_period /
                    max(1.0e-6, times[index + 1] - times[index]),
                    trajectory["slope"] * reference_period,
                ) for index in range(len(vectors) - 1))
        balance_novelty, balance_baseline = _novelty(
            balance_jump, adjacent_balance, scale_floor=0.02)
        balance_slope = _feature_distance(
            left_balance[2]["slope"] * reference_period,
            right_balance[2]["slope"] * reference_period)
        balance_slope_novelty, slope_balance_baseline = _novelty(
            balance_slope, local_balance_slopes, scale_floor=0.01)

    left_last = np.concatenate([
        np.asarray(item["samples"], np.float64) for item in left_frames[-3:]
    ]) if left_frames else np.asarray([], np.float64)
    right_first = np.concatenate([
        np.asarray(item["samples"], np.float64) for item in right_frames[:3]
    ]) if right_frames else np.asarray([], np.float64)
    envelope_frequencies, left_envelope = _smoothed_spectral_envelope(
        left_last, sample_rate, max(2048, config.spectral_fft_size * 4))
    right_frequencies, right_envelope = _smoothed_spectral_envelope(
        right_first, sample_rate, max(2048, config.spectral_fft_size * 4))
    usable_count = min(len(envelope_frequencies), len(right_frequencies),
                       len(left_envelope), len(right_envelope))
    measured_confidences = [float(item.get("tracking_confidence") or 0.0)
                            for item in per_formant
                            if item.get("available")]
    classification_confidences = [
        float(item.get("tracking_confidence") or 0.0)
        for item in per_formant
        if item.get("available") and item.get("eligible_for_classification")
    ]
    # Availability describes what can be shown to a human.  Automatic issue
    # classification remains restricted to the stronger tracks accumulated
    # above, so a visible low-confidence F3/F4 cannot create a false defect.
    available = len(measured_confidences) >= 2
    return {
        "available": available,
        "reason": ("available" if available else
                   "fewer_than_two_stable_formant_tracks"),
        "tracking_confidence": (float(np.mean(measured_confidences))
                                if measured_confidences else 0.0),
        "measured_track_count": len(measured_confidences),
        "classification_track_count": len(classification_confidences),
        "per_formant": per_formant,
        "frequency_jump_normalized": max(frequency_values, default=None),
        "frequency_jump_novelty": max(frequency_novelties, default=None),
        "slope_break": max(slope_values, default=None),
        "slope_break_novelty": max(slope_novelties, default=None),
        "bandwidth_jump": max(bandwidth_values, default=None),
        "prominence_jump": max(prominence_values, default=None),
        "balance_jump": balance_jump,
        "balance_novelty": balance_novelty,
        "balance_slope_break": balance_slope,
        "balance_slope_novelty": balance_slope_novelty,
        "baselines": {
            "frequency": frequency_baselines,
            "slope": slope_baselines,
            "balance": balance_baseline,
            "balance_slope": slope_balance_baseline,
        },
        "plot": {
            "tracks": trajectory_plot,
            "left_rejected": [item for observation in left_observations
                              for item in observation.get("rejected") or []],
            "right_rejected": [item for observation in right_observations
                               for item in observation.get("rejected") or []],
            "balance": {
                "left_times": left_balance[0],
                "left_values": [value.tolist() for value in left_balance[1]],
                "right_times": right_balance[0],
                "right_values": [value.tolist() for value in right_balance[1]],
            },
            "spectral_envelopes": {
                "frequencies_hz": [float(value) for value in
                                   envelope_frequencies[:usable_count]],
                "left_db": [float(value) for value in
                            left_envelope[:usable_count]],
                "right_db": [float(value) for value in
                             right_envelope[:usable_count]],
            },
        },
    }


def _adjacent_level_steps(frames: Sequence[dict]) -> list[float]:
    result = []
    for first, second in zip(frames, frames[1:]):
        step = _db_ratio(float(second["rms"]), float(first["rms"]))
        if step is not None:
            result.append(abs(step))
    return result


def _adjacent_f0_steps(frames: Sequence[dict]) -> list[float]:
    result = []
    for first, second in zip(frames, frames[1:]):
        a, b = float(first["f0_hz"]), float(second["f0_hz"])
        if a > _EPS and b > _EPS:
            result.append(abs(12.0 * math.log2(b / a)))
    return result


def _adjacent_shape_mismatches(frames: Sequence[dict], config) -> list[float]:
    result = []
    for first, second in zip(frames, frames[1:]):
        comparison = _period_correlations(
            first["samples"], second["samples"],
            config.maximum_phase_lag_cycles)
        if comparison is not None:
            result.append(max(0.0, 1.0 - comparison["best"]))
    return result


def _source_units(rows: Sequence[dict], selected: Mapping[int, str],
                  alternatives: Mapping[str, Sequence[Mapping]]) -> tuple[list,
                                                                           dict]:
    units = []
    choices_by_edge = {}
    for index in range(max(0, len(rows) - 1)):
        left = str(rows[index]["phone"])
        right = str(rows[index + 1]["phone"])
        pair = f"{left}-{right}"
        unit_name = str(selected.get(index, left))
        choice = loudness._choice_for(pair, unit_name, alternatives)
        choices_by_edge[index] = choice
        units.append({
            "index": index, "pair": pair,
            "start": rows[index]["center"], "end": rows[index + 1]["center"],
            "selected_unit": unit_name,
            "alias": str(choice.get("alias") or ""),
            "wav": str(choice.get("wav") or choice.get("wav_name") or ""),
            "oto_timing_ms": dict(choice.get("oto_timing_ms") or {}),
            "join_conditioning": dict(choice.get("join_conditioning") or {}),
        })
    return units, choices_by_edge


def _splice_rows(rows: Sequence[dict], records: Sequence[Mapping]) -> list[dict]:
    internal = range(1, max(1, len(rows) - 1))
    supplied = []
    for source in records or ():
        record = dict(source)
        try:
            index = int(record.get("segment_index"))
        except (TypeError, ValueError):
            index = -1
        if index not in internal and len(rows) > 2:
            try:
                when = float(record.get("time"))
            except (TypeError, ValueError):
                continue
            index = min(internal, key=lambda candidate: abs(
                float(rows[candidate]["center"]) - when))
        if index not in internal:
            continue
        if record.get("phone_fraction") is not None:
            try:
                fraction = max(0.0, min(1.0,
                    float(record["phone_fraction"])))
                when = (float(rows[index]["start"]) + fraction *
                        (float(rows[index]["end"]) -
                         float(rows[index]["start"])))
            except (TypeError, ValueError):
                when = float(record.get("time") or rows[index]["center"])
        else:
            when = float(record.get("time") or rows[index]["center"])
        record.update({"segment_index": index, "time": when})
        supplied.append(record)
    if supplied:
        return sorted(supplied, key=lambda row: float(row["time"]))
    return [{
        "segment_index": index,
        "time": float(rows[index]["center"]),
        "handoff_start": float(rows[index]["center"]),
        "handoff_end": float(rows[index]["center"]),
        "position_source": "estimated-phone-center",
        "estimated": True,
    } for index in internal]


def _collars(incoming: Mapping, outgoing: Mapping, when: float) -> dict:
    incoming_conditioning = dict(incoming.get("join_conditioning") or {})
    outgoing_conditioning = dict(outgoing.get("join_conditioning") or {})
    timing = dict(outgoing.get("oto_timing_ms") or {})
    try:
        declared = float(timing.get("overlap") or 0.0)
        left = float(incoming_conditioning.get("effective_end_collar_ms")
                     or declared)
        right = float(outgoing_conditioning.get("effective_start_collar_ms")
                      or declared)
    except (TypeError, ValueError):
        declared = left = right = 0.0
    return {
        "declared_oto_overlap_ms": max(0.0, declared),
        "incoming_collar_ms": max(0.0, left),
        "outgoing_collar_ms": max(0.0, right),
        "overlap_start": when - max(0.0, left) / 1000.0,
        "overlap_end": when + max(0.0, right) / 1000.0,
    }


def _recommendation(label: str) -> str:
    return {
        "LEVEL_STEP": "Adjust local gain over one or two periods.",
        "SAMPLE_DISCONTINUITY": "Move the cut or use a pitch-synchronous overlap.",
        "BROADBAND_IMPULSE": "Inspect the measured impulse time; move the cut or use a phase-compatible pitch-synchronous overlap.",
        "CONTENT_DROPOUT": "Preserve the source collar and avoid destructive cancellation; align phase or shorten the overlap.",
        "PHASE_MISMATCH": "Align pitch phase before considering a crossfade.",
        "F0_STEP": "Choose a closer source period or remap the splice epochs.",
        "PERIOD_SHAPE_MISMATCH": "Choose a timbrally closer unit or move the splice.",
        "SPECTRAL_STEP": "Choose a closer source context or move the splice point.",
        "SPECTRAL_ENVELOPE_BREAK": "Choose a closer source context or move the splice point.",
        "SPECTRAL_TRAJECTORY_BREAK": "Prefer a source whose timbral trajectory continues the incoming unit.",
        "FORMANT_FREQUENCY_BREAK": "Move the cut or select a source with matching formant positions.",
        "FORMANT_TRAJECTORY_BREAK": "Select a source whose vowel-colour movement continues across the handoff.",
        "FORMANT_BALANCE_BREAK": "Select a source with a closer formant-energy balance; do not apply global gain.",
        "FORMANT_PROMINENCE_BREAK": "Inspect bandwidth and prominence before changing source context.",
        "UNVOICED_SPECTRAL_BREAK": "Move the cut or use a short noise-compatible overlap.",
        "INSUFFICIENT_CONTEXT": "Inspect a longer render around this join.",
    }.get(label, "No automatic repair is recommended.")


class JoinDiscontinuityAnalyzer:
    """Analyze every known rendered unit handoff without changing audio."""

    def __init__(
        self,
        samples: object,
        sample_rate: int,
        segments: Sequence[object],
        *,
        splice_records: Sequence[Mapping] | None = None,
        target_pitchmarks: Sequence[float] | None = None,
        pitchmarks: Sequence[float] | None = None,
        selected_units: Mapping[int, str] | None = None,
        alternatives: Mapping[str, Sequence[Mapping]] | None = None,
        config: JoinAnalysisConfig | Mapping | None = None,
        window_ms: float = loudness.DEFAULT_JOIN_WINDOW_MS,
        hop_ms: float = loudness.DEFAULT_JOIN_HOP_MS,
        flag_step_lu: float | None = None,
        minimum_audible_lkfs: float = loudness.DEFAULT_MIN_AUDIBLE_LKFS,
        include_curves: bool = True,
        compute_k_weighted_level: bool = True,
    ):
        self.samples = loudness._mono(samples)
        if not self.samples.size:
            raise ValueError("audio samples are empty")
        self.sample_rate = int(sample_rate)
        if self.sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz")
        self.rows = loudness._segments(segments)
        self.splice_records = [dict(row) for row in (splice_records or ())]
        marks = target_pitchmarks if target_pitchmarks is not None else pitchmarks
        mark_values = () if marks is None else marks
        self.pitchmarks = np.asarray(sorted(set(
            float(value) for value in mark_values
            if math.isfinite(float(value)) and float(value) >= 0.0
        )), dtype=np.float64)
        self.selected = {int(key): str(value) for key, value in
                         dict(selected_units or {}).items()}
        self.alternatives = dict(alternatives or {})
        if config is None:
            resolved = JoinAnalysisConfig()
        elif isinstance(config, JoinAnalysisConfig):
            resolved = config
        else:
            resolved = JoinAnalysisConfig(**dict(config))
        if flag_step_lu is not None:
            resolved = replace(resolved, level_step_db=float(flag_step_lu))
        self.config = resolved
        self.window_ms = float(window_ms)
        self.hop_ms = float(hop_ms)
        self.minimum_audible_lkfs = float(minimum_audible_lkfs)
        self.include_curves = bool(include_curves)
        self.compute_k_weighted_level = bool(compute_k_weighted_level)

    def _join(self, splice: Mapping, units: Sequence[dict],
              choices: Mapping[int, Mapping], weighted: np.ndarray) -> dict:
        config = self.config
        index = int(splice["segment_index"])
        when = max(0.0, min(len(self.samples) / self.sample_rate,
                            float(splice["time"])))
        sample = max(2, min(len(self.samples) - 2,
                            int(round(when * self.sample_rate))))
        phone = str(self.rows[index]["phone"])
        phone_context = [
            str(self.rows[position]["phone"])
            for position in range(max(0, index - 1),
                                  min(len(self.rows), index + 2))
        ]
        expected_burst_context = _expected_broadband_burst(phone)
        phone_context_string = " ".join(phone_context)

        context_samples = max(16, int(round(
            self.sample_rate * max(60.0, config.immediate_context_ms)
            / 1000.0)))
        left_context = self.samples[max(0, sample - context_samples):sample]
        right_context = self.samples[sample:min(
            len(self.samples), sample + context_samples)]
        left_estimate, left_periodicity = _estimate_period(
            left_context, self.sample_rate, config.f0_min_hz,
            config.f0_max_hz)
        right_estimate, right_periodicity = _estimate_period(
            right_context, self.sample_rate, config.f0_min_hz,
            config.f0_max_hz)

        left_bounds, right_bounds = _period_bounds_from_marks(
            self.pitchmarks, when, self.sample_rate, len(self.samples),
            config.period_context_count)
        period_source = "rendered-target-pitchmarks" if (
            left_bounds and right_bounds) else "separate-side-autocorrelation"
        if not left_bounds or not right_bounds:
            if left_estimate and right_estimate:
                left_bounds, right_bounds = _regular_period_bounds(
                    sample, left_estimate, right_estimate, len(self.samples),
                    config.period_context_count)

        period_left = (left_bounds[-1][1] - left_bounds[-1][0]
                       if left_bounds else left_estimate)
        period_right = (right_bounds[0][1] - right_bounds[0][0]
                        if right_bounds else right_estimate)
        if period_left:
            left_periodicity = _lag_periodicity(
                left_context, int(period_left))
        if period_right:
            right_periodicity = _lag_periodicity(
                right_context, int(period_right))

        left_window = max(8, int(round(
            period_left or int(.01 * self.sample_rate))))
        right_window = max(8, int(round(
            period_right or int(.01 * self.sample_rate))))
        provisional_left = _rms(left_context[-left_window:])
        provisional_right = _rms(right_context[:right_window])
        left_voicing_confidence, left_voicing_evidence = _voicing_confidence(
            left_periodicity, provisional_left, phone, config)
        right_voicing_confidence, right_voicing_evidence = _voicing_confidence(
            right_periodicity, provisional_right, phone, config)
        left_voiced_eligible = bool(
            provisional_left >= config.voiced_minimum_rms and
            left_periodicity >= config.periodicity_threshold and
            left_voicing_confidence >= config.minimum_voicing_confidence)
        right_voiced_eligible = bool(
            provisional_right >= config.voiced_minimum_rms and
            right_periodicity >= config.periodicity_threshold and
            right_voicing_confidence >= config.minimum_voicing_confidence)
        if max(provisional_left, provisional_right) < config.minimum_rms:
            voicing = "silent"
        elif (left_voiced_eligible and right_voiced_eligible and
              period_left and period_right):
            voicing = "voiced"
        elif left_voiced_eligible != right_voiced_eligible:
            voicing = "mixed"
        else:
            voicing = "unvoiced"

        if voicing == "voiced" and left_bounds and right_bounds:
            left_frames = _frames(self.samples, left_bounds, self.sample_rate)
            right_frames = _frames(self.samples, right_bounds, self.sample_rate)
        else:
            fixed_left, fixed_right = _fixed_frame_bounds(
                sample, self.sample_rate, len(self.samples),
                config.unvoiced_frame_ms, config.unvoiced_hop_ms,
                config.unvoiced_context_count)
            left_frames = _frames(self.samples, fixed_left, self.sample_rate)
            right_frames = _frames(self.samples, fixed_right, self.sample_rate)

        final_left = left_frames[-1] if left_frames else None
        first_right = right_frames[0] if right_frames else None
        left_rms = float(final_left["rms"]) if final_left else 0.0
        right_rms = float(first_right["rms"]) if first_right else 0.0
        level_step = _db_ratio(right_rms, left_rms)
        level_change = abs(level_step) if level_step is not None else None
        level_baseline_values = (_adjacent_level_steps(left_frames) +
                                 _adjacent_level_steps(right_frames))
        level_novelty, level_baseline = _novelty(
            level_change, level_baseline_values)

        signed_jump = float(self.samples[sample] - self.samples[sample - 1])
        sample_jump = abs(signed_jump)
        left_slope = float(self.samples[sample - 1] - self.samples[sample - 2])
        right_slope = float(self.samples[sample + 1] - self.samples[sample])
        # The cross-splice first difference is the impulse a click detector
        # sees. Compare it independently with the derivatives wholly inside
        # each unit; comparing only left_slope with right_slope misses a DC
        # step because both within-unit derivatives can still match.
        slope_jump = max(abs(signed_jump - left_slope),
                         abs(right_slope - signed_jump))
        immediate = max(8, int(round(
            self.sample_rate * config.immediate_context_ms / 1000.0)))
        left_local = self.samples[max(0, sample - immediate):sample]
        right_local = self.samples[sample:min(len(self.samples), sample + immediate)]
        first_differences = np.concatenate((
            np.abs(np.diff(left_local)), np.abs(np.diff(right_local))))
        second_differences = np.concatenate((
            np.abs(np.diff(left_local, n=2)),
            np.abs(np.diff(right_local, n=2))))
        sample_novelty, sample_baseline = _novelty(
            sample_jump, first_differences.tolist())
        slope_novelty, slope_baseline = _novelty(
            slope_jump, second_differences.tolist())
        second_derivative = abs(float(
            self.samples[sample + 1] - 2.0 * self.samples[sample]
            + self.samples[sample - 1]))
        second_novelty, second_baseline = _novelty(
            second_derivative, second_differences.tolist())
        broadband = _broadband_impulse_metrics(
            self.samples, self.sample_rate, splice, when, config)
        content_period = (int(round((period_left + period_right) * 0.5))
                          if period_left and period_right else
                          int(period_left or period_right or 0))
        content = _content_dropout_metrics(
            self.samples, self.sample_rate, splice, when, phone, config,
            voiced=(voicing == "voiced"),
            period_samples=content_period or None)

        left_f0 = right_f0 = f0_step = f0_novelty = None
        phase = None
        phase_mismatch = shape_mismatch = shape_novelty = None
        f0_baseline = {"median": None, "mad": None, "count": 0}
        shape_baseline = {"median": None, "mad": None, "count": 0}
        if voicing == "voiced" and final_left and first_right:
            left_f0 = float(final_left["f0_hz"])
            right_f0 = float(first_right["f0_hz"])
            f0_step = 12.0 * math.log2(right_f0 / left_f0)
            f0_novelty, f0_baseline = _novelty(
                abs(f0_step), _adjacent_f0_steps(left_frames) +
                _adjacent_f0_steps(right_frames))
            phase = _period_correlations(
                final_left["samples"], first_right["samples"],
                config.maximum_phase_lag_cycles)
            if phase is not None:
                phase_mismatch = max(0.0, phase["best"] - phase["zero"])
                shape_mismatch = max(0.0, 1.0 - phase["best"])
                shape_novelty, shape_baseline = _novelty(
                    shape_mismatch,
                    _adjacent_shape_mismatches(left_frames, config) +
                    _adjacent_shape_mismatches(right_frames, config))

        reference_period = (1.0 / max(_EPS, (left_f0 + right_f0) * 0.5)
                            if left_f0 and right_f0 else
                            config.unvoiced_hop_ms / 1000.0)
        spectral = _spectral_metrics(
            left_frames, right_frames, when, reference_period, config,
            periodic=(voicing == "voiced"))
        formants = _formant_metrics(
            left_frames, right_frames, self.sample_rate, when,
            reference_period, config,
            eligible=(voicing == "voiced" and left_voiced_eligible and
                      right_voiced_eligible),
        )

        short_left, short_right = _fixed_frame_bounds(
            sample, self.sample_rate, len(self.samples), 8.0, 8.0, 1)
        medium_left, medium_right = _fixed_frame_bounds(
            sample, self.sample_rate, len(self.samples), 30.0, 30.0, 1)
        def boundary_spectral_step(a_bounds, b_bounds):
            if not a_bounds or not b_bounds:
                return None
            a = _cepstral_feature(
                self.samples[slice(*a_bounds[-1])],
                config.spectral_coefficients, config.spectral_fft_size)
            b = _cepstral_feature(
                self.samples[slice(*b_bounds[0])],
                config.spectral_coefficients, config.spectral_fft_size)
            return _feature_distance(a, b)
        short_spectral = boundary_spectral_step(short_left, short_right)
        medium_spectral = boundary_spectral_step(medium_left, medium_right)

        before_lkfs = loudness._interval_level_lkfs(
            weighted, self.sample_rate, when - 0.020, when - 0.003)
        after_lkfs = loudness._interval_level_lkfs(
            weighted, self.sample_rate, when + 0.003, when + 0.020)
        loudness_floor_met = max(
            before_lkfs if before_lkfs is not None else -180.0,
            after_lkfs if after_lkfs is not None else -180.0,
        ) >= self.minimum_audible_lkfs
        above_analysis_floor = bool(
            max(left_rms, right_rms) >= config.minimum_rms and
            loudness_floor_met)

        # A dB ratio becomes unstable as either period approaches silence.
        # Preserve that raw ratio, but smoothly suppress only its ranking
        # contribution when the geometric mean energy is near the floor.
        level_energy_rms = math.sqrt(max(0.0, left_rms * right_rms))
        level_energy_gate = _smoothstep(
            level_energy_rms, config.level_gate_floor_rms,
            config.level_gate_full_rms)
        spectral_energy_gate = _smoothstep(
            max(left_rms, right_rms), config.minimum_rms,
            config.voiced_minimum_rms)
        formant_energy_gate = spectral_energy_gate * min(
            left_voicing_confidence, right_voicing_confidence)
        # A pause center intentionally transitions between silence/breath and
        # speech. Preserve its raw spectral measurements, but do not rank that
        # expected onset as a timbral splice defect. Click evidence remains
        # active, so a genuine sample impulse at a pause is still visible.
        if _phone_key(phone) in _SILENCE_PHONES:
            spectral_energy_gate = 0.0
        local_signal_rms = max(
            _rms(left_local), _rms(right_local), config.minimum_rms)
        event_energy_gate = _smoothstep(
            max(local_signal_rms, sample_jump),
            config.minimum_rms * 0.25, config.minimum_rms * 2.0)

        components = {}
        components["CONTENT_DROPOUT"] = _continuous_component(
            content["attenuation_db"],
            config.content_dropout_attenuation_reference_db,
            None,
            1.0,
            energy_gate=1.0 if content["eligible"] else 0.0,
            novelty_support_scale=config.novelty_support_scale,
            local_normality_discount=config.local_normality_discount,
        )
        if level_change is not None:
            components["LEVEL_STEP"] = _continuous_component(
                level_change, config.level_step_db, level_novelty,
                config.novelty_threshold, energy_gate=level_energy_gate,
                novelty_support_scale=config.novelty_support_scale,
                local_normality_discount=config.local_normality_discount)

        gain_ratio = (right_rms / left_rms
                      if left_rms > _EPS and right_rms > _EPS else 1.0)
        gain_compensated_jump = sample_jump
        gain_compensated_slope_jump = slope_jump
        gain_compensation_used = bool(
            level_energy_gate >= 0.5 and 0.25 <= gain_ratio <= 4.0)
        if gain_compensation_used:
            compensated_cross = float(
                self.samples[sample] - gain_ratio * self.samples[sample - 1])
            compensated_left_slope = left_slope * gain_ratio
            gain_compensated_jump = abs(compensated_cross)
            gain_compensated_slope_jump = max(
                abs(compensated_cross - compensated_left_slope),
                abs(right_slope - compensated_cross))
        scoring_sample_jump = min(sample_jump, gain_compensated_jump)
        scoring_slope_jump = min(slope_jump, gain_compensated_slope_jump)
        scoring_sample_novelty, _unused_sample_baseline = _novelty(
            scoring_sample_jump, first_differences.tolist())
        scoring_slope_novelty, _unused_slope_baseline = _novelty(
            scoring_slope_jump, second_differences.tolist())

        sample_reference = max(
            float(sample_baseline.get("median") or 0.0),
            local_signal_rms * 1.0e-4, _EPS)
        slope_reference = max(
            float(slope_baseline.get("median") or 0.0),
            local_signal_rms * 1.0e-4, _EPS)
        sample_relative = scoring_sample_jump / sample_reference
        slope_relative = scoring_slope_jump / slope_reference
        immediate_absolute_gate = _smoothstep(
            max(scoring_sample_jump, scoring_slope_jump),
            config.sample_jump_absolute_floor,
            config.sample_jump_absolute_full,
        )
        immediate_energy_gate = event_energy_gate * immediate_absolute_gate
        sample_component = _continuous_component(
            sample_relative, config.sample_jump_relative_scale,
            scoring_sample_novelty, config.sample_jump_novelty,
            energy_gate=immediate_energy_gate,
            novelty_support_scale=config.novelty_support_scale,
            local_normality_discount=config.local_normality_discount)
        slope_component = _continuous_component(
            slope_relative, config.slope_jump_relative_scale,
            scoring_slope_novelty, config.slope_jump_novelty,
            energy_gate=immediate_energy_gate,
            novelty_support_scale=config.novelty_support_scale,
            local_normality_discount=config.local_normality_discount)
        immediate_component = dict(max(
            (sample_component, slope_component),
            key=lambda item: float(item["combined_score"])))
        immediate_component.update({
            "sample_relative_to_local_changes": sample_relative,
            "slope_relative_to_local_changes": slope_relative,
            "sample_local_reference": sample_reference,
            "slope_local_reference": slope_reference,
            "absolute_click_gate": immediate_absolute_gate,
            "gain_compensation_used": gain_compensation_used,
            "gain_ratio": gain_ratio,
            "gain_compensated_sample_jump": gain_compensated_jump,
            "gain_compensated_slope_jump": gain_compensated_slope_jump,
            "sample_component": sample_component,
            "slope_component": slope_component,
        })
        components["SAMPLE_DISCONTINUITY"] = immediate_component
        if broadband["available"]:
            components["BROADBAND_IMPULSE"] = _continuous_component(
                broadband["score"], config.broadband_impulse_score,
                broadband["novelty"], config.broadband_impulse_novelty,
                energy_gate=broadband["energy_gate"],
                novelty_support_scale=config.novelty_support_scale,
                local_normality_discount=config.local_normality_discount)

        if voicing == "voiced":
            if phase is not None and phase_mismatch is not None:
                components["PHASE_MISMATCH"] = _continuous_component(
                    phase_mismatch, config.phase_mismatch, None, 1.0,
                    energy_gate=_smoothstep(phase["best"], 0.45, 0.85),
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)
            if f0_step is not None:
                components["F0_STEP"] = _continuous_component(
                    abs(f0_step), config.f0_step_semitones, f0_novelty,
                    config.novelty_threshold,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)
            if shape_mismatch is not None:
                components["PERIOD_SHAPE_MISMATCH"] = _continuous_component(
                    shape_mismatch, config.period_shape_mismatch,
                    shape_novelty, config.novelty_threshold,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)
        spectral_step = spectral["step"]
        spectral_step_novelty = spectral["step_novelty"]
        spectral_slope_novelty = spectral["slope_novelty"]
        spectral_step_component = (
            _continuous_component(
                spectral_step, config.spectral_step_floor,
                spectral_step_novelty, config.spectral_step_novelty,
                energy_gate=spectral_energy_gate,
                novelty_support_scale=config.novelty_support_scale,
                local_normality_discount=config.local_normality_discount)
            if spectral_step is not None else None)
        spectral_slope_component = (
            _continuous_component(
                spectral["slope_break"], config.spectral_slope_floor,
                spectral_slope_novelty, config.spectral_slope_novelty,
                energy_gate=spectral_energy_gate,
                novelty_support_scale=config.novelty_support_scale,
                local_normality_discount=config.local_normality_discount)
            if spectral["slope_break"] is not None else None)
        if voicing in ("unvoiced", "mixed", "silent"):
            available = [item for item in (
                spectral_step_component, spectral_slope_component) if item]
            if available:
                spectral_component = dict(max(
                    available, key=lambda item: item["combined_score"]))
                spectral_component.update({
                    "value_component": spectral_step_component,
                    "slope_component": spectral_slope_component,
                })
                components["UNVOICED_SPECTRAL_BREAK"] = spectral_component
        else:
            if spectral_step_component:
                components["SPECTRAL_STEP"] = spectral_step_component
            if spectral_slope_component:
                components["SPECTRAL_TRAJECTORY_BREAK"] = (
                    spectral_slope_component)
        if formants["available"]:
            if formants["frequency_jump_normalized"] is not None:
                components["FORMANT_FREQUENCY_BREAK"] = _continuous_component(
                    formants["frequency_jump_normalized"],
                    config.formant_frequency_jump_fraction,
                    formants["frequency_jump_novelty"],
                    config.formant_novelty_threshold,
                    energy_gate=formant_energy_gate,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)
            if formants["slope_break"] is not None:
                components["FORMANT_TRAJECTORY_BREAK"] = _continuous_component(
                    formants["slope_break"],
                    config.formant_slope_break_fraction,
                    formants["slope_break_novelty"],
                    config.formant_novelty_threshold,
                    energy_gate=formant_energy_gate,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)
            balance_candidates = []
            if formants["balance_jump"] is not None:
                balance_candidates.append(_continuous_component(
                    formants["balance_jump"], config.formant_balance_jump,
                    formants["balance_novelty"],
                    config.formant_novelty_threshold,
                    energy_gate=formant_energy_gate,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount))
            if formants["balance_slope_break"] is not None:
                balance_candidates.append(_continuous_component(
                    formants["balance_slope_break"],
                    config.formant_balance_slope_break,
                    formants["balance_slope_novelty"],
                    config.formant_novelty_threshold,
                    energy_gate=formant_energy_gate,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount))
            if balance_candidates:
                components["FORMANT_BALANCE_BREAK"] = dict(max(
                    balance_candidates,
                    key=lambda item: item["combined_score"]))
            prominence_values = [
                max(
                    float(item.get("prominence_jump_db") or 0.0) /
                    config.formant_prominence_jump_db,
                    float(item.get("bandwidth_jump_log_ratio") or 0.0) /
                    config.formant_bandwidth_jump_fraction,
                ) for item in formants["per_formant"]
                if item.get("available") and
                item.get("eligible_for_classification")
            ]
            if prominence_values:
                components["FORMANT_PROMINENCE_BREAK"] = _continuous_component(
                    max(prominence_values), 1.0, None, 1.0,
                    energy_gate=formant_energy_gate,
                    novelty_support_scale=config.novelty_support_scale,
                    local_normality_discount=config.local_normality_discount)

        insufficient = (len(left_local) < 4 or len(right_local) < 4 or
                        len(left_frames) < 2 or len(right_frames) < 2)
        weighted_candidates = {}
        for name, component in components.items():
            weight = config.weights.get(name, 1.0)
            component["weight"] = float(weight)
            component["weighted_score"] = (
                float(component["combined_score"]) * float(weight))
            weighted_candidates[name] = component["weighted_score"]
        severity = math.sqrt(sum(
            float(value) * float(value)
            for value in weighted_candidates.values()))
        top_component = (max(weighted_candidates,
                             key=lambda name: weighted_candidates[name])
                         if weighted_candidates else None)
        top_score = (float(weighted_candidates[top_component])
                     if top_component else 0.0)
        if insufficient:
            dominant = "INSUFFICIENT_CONTEXT"
        elif top_component and top_score >= config.classification_score:
            dominant = top_component
        else:
            dominant = "OK"
        issues = [name for name, _value in sorted(
            weighted_candidates.items(), key=lambda item: (-item[1], item[0]))
                  if _value >= config.classification_score]
        # Backward-compatible wording remains visible while the explicit full
        # envelope label makes clear that this is not a raw harmonic test.
        if "SPECTRAL_STEP" in issues:
            issues.append("SPECTRAL_ENVELOPE_BREAK")
        if top_component:
            top = components[top_component]
            evidence_text = (
                f"{top_component.replace('_', ' ').title()} ranked highest: "
                f"absolute {float(top['absolute_score']):.2f}x reference, "
                f"local novelty {float(top['novelty_score']):.2f}x reference, "
                f"energy gate {float(top['energy_gate']):.2f}, "
                f"weighted score {top_score:.2f}."
            )
        else:
            evidence_text = "No measurable discontinuity component was available."
        if dominant == "OK":
            classification_reason = (
                evidence_text + " Below the uncalibrated display cutoff; "
                "the non-zero score is retained for ranking."
            )
        elif dominant == "INSUFFICIENT_CONTEXT":
            classification_reason = (
                "Not enough non-crossing context for a reliable label. " +
                evidence_text)
        else:
            classification_reason = evidence_text
        if expected_burst_context and "BROADBAND_IMPULSE" in issues:
            contextual_interpretation = "EXPECTED_BURST_CONTEXT_REVIEW"
            contextual_note = (
                f"Phone context '{phone_context_string}' can contain a legitimate "
                "stop or affricate release; "
                "inspect whether this broadband event is singular, aligned, "
                "and appropriate rather than automatically treating it as a defect.")
        elif "BROADBAND_IMPULSE" in issues:
            contextual_interpretation = "UNEXPECTED_BROADBAND_EVENT"
            contextual_note = (
                "The full-band transient is unusual for the local signal and "
                "is not explained by a stop/affricate phone label.")
        else:
            contextual_interpretation = "NO_BROADBAND_EVENT"
            contextual_note = "No broadband impulse crossed the display threshold."

        incoming = choices.get(index - 1, {})
        outgoing = choices.get(index, {})
        collar = _collars(incoming, outgoing, when)
        period_plot = {}
        if final_left and first_right:
            display_count = max(64, min(512, int(round(
                (len(final_left["samples"]) + len(first_right["samples"]))
                * 0.5))))
            period_plot = {
                "phase": [float(value) for value in
                          np.linspace(0.0, 1.0, display_count,
                                      endpoint=False)],
                "left_raw": [float(value) for value in
                             _resample(final_left["samples"], display_count)],
                "right_raw": [float(value) for value in
                              _resample(first_right["samples"], display_count)],
            }
            if phase is not None:
                period_plot.update({
                    "left_normalised": [float(value) for value in
                                        phase["left_normalised"]],
                    "right_normalised": [float(value) for value in
                                         phase["right_normalised"]],
                    "right_best_aligned": [float(value) for value in
                                           phase["right_aligned"]],
                })
        local_marks = [float(mark) for mark in self.pitchmarks
                       if when - 0.12 <= mark <= when + 0.12]
        plot = {
            "context_start": max(0.0, when - max(0.04, 3.0 * reference_period)),
            "context_end": min(len(self.samples) / self.sample_rate,
                               when + max(0.04, 3.0 * reference_period)),
            "pitchmarks": local_marks,
            "periods": period_plot,
            "period_rms": [{
                "time": _finite(frame["time"]),
                "rms": _finite(frame["rms"]),
                "side": side,
            } for side, frames in (("left", left_frames),
                                   ("right", right_frames))
              for frame in frames],
            "local_f0": ([{
                "time": _finite(frame["time"]),
                "f0_hz": _finite(frame["f0_hz"]),
                "side": side,
            } for side, frames in (("left", left_frames),
                                   ("right", right_frames))
              for frame in frames] if voicing == "voiced" else []),
            "spectral_trajectory": {
                key: ([float(value) for value in value]
                      if isinstance(value, list) else _finite(value))
                for key, value in spectral["plot"].items()
            },
            "formants": formants["plot"],
            "broadband_impulse": broadband["plot"],
            "content_preservation": content["frames"],
            "phone_context": phone_context,
        }

        row = {
            "segment_index": index,
            "phone": phone,
            "phone_context": phone_context,
            "phone_context_string": phone_context_string,
            "splice_sample": sample,
            "splice_time_seconds": _finite(when, 9),
            "time": _finite(when, 9),
            "position_source": str(splice.get("position_source") or
                                   "estimated-phone-center"),
            "position_estimated": bool(splice.get("estimated", True)),
            "handoff_start": _finite(splice.get("handoff_start", when), 9),
            "handoff_end": _finite(splice.get("handoff_end", when), 9),
            "voicing": voicing,
            "left_voicing_confidence": _finite(left_voicing_confidence),
            "right_voicing_confidence": _finite(right_voicing_confidence),
            "left_voiced_eligible": left_voiced_eligible,
            "right_voiced_eligible": right_voiced_eligible,
            "left_voicing_evidence": left_voicing_evidence,
            "right_voicing_evidence": right_voicing_evidence,
            "left_periodicity": _finite(left_periodicity),
            "right_periodicity": _finite(right_periodicity),
            "period_source": period_source,
            "left_rms": _finite(left_rms),
            "right_rms": _finite(right_rms),
            "level_step_db": _finite(level_step),
            "level_step_novelty": _finite(level_novelty),
            "sample_value_jump": _finite(sample_jump),
            "sample_value_jump_signed": _finite(signed_jump),
            "sample_jump": _finite(sample_jump),
            "sample_jump_novelty": _finite(sample_novelty),
            "cross_splice_slope": _finite(signed_jump),
            "left_local_slope": _finite(left_slope),
            "right_local_slope": _finite(right_slope),
            "slope_jump": _finite(slope_jump),
            "slope_jump_novelty": _finite(slope_novelty),
            "gain_compensation_used_for_click_ranking": gain_compensation_used,
            "gain_ratio_for_click_ranking": _finite(gain_ratio),
            "gain_compensated_sample_jump": _finite(
                gain_compensated_jump),
            "gain_compensated_slope_jump": _finite(
                gain_compensated_slope_jump),
            "second_derivative_novelty": _finite(second_novelty),
            "broadband_impulse_time_seconds": _finite(
                broadband.get("event_time"), 9),
            "broadband_impulse_sample": broadband.get("event_sample"),
            "broadband_impulse_frame_ms": _finite(
                broadband.get("frame_ms"), 4),
            "broadband_impulse_scale_evidence": _finite(
                broadband.get("scale_evidence")),
            "broadband_impulse_tested_scales": broadband.get("scales", []),
            "broadband_impulse_score": _finite(broadband.get("score")),
            "broadband_impulse_novelty": _finite(
                broadband.get("novelty")),
            "broadband_impulse_energy_gate": _finite(
                broadband.get("energy_gate")),
            "broadband_impulse_rms": _finite(broadband.get("rms")),
            "broadband_temporal_crest": _finite(
                broadband.get("temporal_crest")),
            "broadband_bin_spectral_flatness": _finite(
                broadband.get("bin_flatness")),
            "broadband_band_spectral_flatness": _finite(
                broadband.get("band_flatness")),
            "broadband_spectral_tilt_db_per_octave": _finite(
                broadband.get("tilt_db_per_octave")),
            "broadband_band_uniformity_db": _finite(
                broadband.get("band_uniformity_db")),
            "broadband_absolute_shape_score": _finite(
                broadband.get("absolute_shape_score")),
            "broadband_relative_shape_score": _finite(
                broadband.get("relative_shape_score")),
            "broadband_flatness_novelty": _finite(
                broadband.get("flatness_novelty")),
            "broadband_tilt_flattening_novelty": _finite(
                broadband.get("tilt_flattening_novelty")),
            "broadband_uniformity_novelty": _finite(
                broadband.get("uniformity_novelty")),
            "broadband_floor_novelty_support": _finite(
                broadband.get("floor_novelty_support")),
            "broadband_context_may_be_expected": expected_burst_context,
            "broadband_context_interpretation": contextual_interpretation,
            "broadband_context_note": contextual_note,
            "broadband_floor_energy": _finite(
                broadband.get("broadband_floor")),
            "broadband_local_energy_ratio": _finite(
                broadband.get("local_energy_ratio")),
            "broadband_scan_start": _finite(
                broadband.get("scan_start"), 9),
            "broadband_scan_end": _finite(
                broadband.get("scan_end"), 9),
            "content_handoff_start_sample": int(
                content["handoff_start_sample"]),
            "content_handoff_end_sample": int(
                content["handoff_end_sample"]),
            "content_analysis_start_sample": int(
                content["analysis_start_sample"]),
            "content_analysis_end_sample": int(
                content["analysis_end_sample"]),
            "content_frame_samples": int(content["frame_samples"]),
            "content_pitch_synchronous": bool(
                content["pitch_synchronous"]),
            "content_handoff_rms": _finite(content["handoff_rms"]),
            "content_median_frame_rms": _finite(
                content["median_handoff_frame_rms"]),
            "content_minimum_frame_rms": _finite(
                content["minimum_handoff_frame_rms"]),
            "content_left_reference_rms": _finite(
                content["left_reference_rms"]),
            "content_right_reference_rms": _finite(
                content["right_reference_rms"]),
            "content_reference_rms": _finite(content["reference_rms"]),
            "content_retention_ratio": _finite(
                content["retention_ratio"]),
            "content_attenuation_db": _finite(
                content["attenuation_db"]),
            "content_dropout_expected": bool(content["expected"]),
            "content_dropout_eligible": bool(content["eligible"]),
            "content_dropout_reason": str(content["reason"]),
            "left_period_samples": int(period_left) if period_left else None,
            "right_period_samples": int(period_right) if period_right else None,
            "left_period_seconds": _finite(
                period_left / self.sample_rate if period_left else None, 9),
            "right_period_seconds": _finite(
                period_right / self.sample_rate if period_right else None, 9),
            "left_f0_hz": _finite(left_f0),
            "right_f0_hz": _finite(right_f0),
            "f0_step_semitones": _finite(f0_step),
            "f0_step_cents": _finite(f0_step * 100.0
                                     if f0_step is not None else None),
            "f0_step_novelty": _finite(f0_novelty),
            "zero_lag_period_correlation": _finite(
                phase["zero"] if phase else None),
            "best_lag_period_correlation": _finite(
                phase["best"] if phase else None),
            "best_phase_offset_samples": (
                int(phase["lag"]) if phase else None),
            "best_phase_offset_cycles": _finite(
                phase["lag_cycles"] if phase else None),
            "phase_mismatch": _finite(phase_mismatch),
            "period_shape_mismatch": _finite(shape_mismatch),
            "period_waveform_error": _finite(
                phase["shape_error"] if phase else None),
            "period_shape_novelty": _finite(shape_novelty),
            "spectral_step": _finite(spectral_step),
            "spectral_step_novelty": _finite(spectral_step_novelty),
            "spectral_slope_break": _finite(spectral["slope_break"]),
            "spectral_slope_break_novelty": _finite(
                spectral_slope_novelty),
            "spectral_flux": _finite(spectral["flux"]),
            "spectral_flux_novelty": _finite(spectral["flux_novelty"]),
            "spectral_envelope_step": _finite(spectral_step),
            "spectral_envelope_novelty": _finite(spectral_step_novelty),
            "spectral_envelope_slope_break": _finite(
                spectral["slope_break"]),
            "spectral_envelope_slope_novelty": _finite(
                spectral_slope_novelty),
            "formants_available": bool(formants["available"]),
            "formants_unavailable_reason": str(formants["reason"]),
            "formant_tracking_confidence": _finite(
                formants["tracking_confidence"]),
            "formant_measured_track_count": int(
                formants["measured_track_count"]),
            "formant_classification_track_count": int(
                formants["classification_track_count"]),
            "formant_tracks": formants["per_formant"],
            "formant_frequency_jump_normalized": _finite(
                formants["frequency_jump_normalized"]),
            "formant_frequency_jump_novelty": _finite(
                formants["frequency_jump_novelty"]),
            "formant_slope_break": _finite(formants["slope_break"]),
            "formant_slope_break_novelty": _finite(
                formants["slope_break_novelty"]),
            "formant_bandwidth_jump": _finite(formants["bandwidth_jump"]),
            "formant_prominence_jump": _finite(formants["prominence_jump"]),
            "formant_balance_jump": _finite(formants["balance_jump"]),
            "formant_balance_novelty": _finite(
                formants["balance_novelty"]),
            "formant_balance_slope_break": _finite(
                formants["balance_slope_break"]),
            "formant_balance_slope_break_novelty": _finite(
                formants["balance_slope_novelty"]),
            "short_window_spectral_step": _finite(short_spectral),
            "medium_window_spectral_step": _finite(medium_spectral),
            "short_medium_spectral_difference": _finite(
                abs(short_spectral - medium_spectral)
                if short_spectral is not None and medium_spectral is not None
                else None),
            "dominant_issue": dominant,
            "issues": issues,
            "severity_score": _finite(severity),
            "severity_components": components,
            "ranking_top_component": top_component,
            "ranking_top_component_score": _finite(top_score),
            "severity_is_calibrated": False,
            "flagged": dominant not in ("OK", "INSUFFICIENT_CONTEXT"),
            "repair_recommendation": _recommendation(dominant),
            "classification_reason": classification_reason,
            "incoming_pair": units[index - 1]["pair"],
            "outgoing_pair": units[index]["pair"],
            "incoming_unit": units[index - 1]["selected_unit"],
            "outgoing_unit": units[index]["selected_unit"],
            "incoming_wav": units[index - 1]["wav"],
            "outgoing_wav": units[index]["wav"],
            "same_source_wav": bool(
                units[index - 1]["wav"] and
                units[index - 1]["wav"] == units[index]["wav"]),
            "before_lkfs": _finite(before_lkfs, 4),
            "after_lkfs": _finite(after_lkfs, 4),
            "step_lu": _finite(
                (after_lkfs - before_lkfs)
                if before_lkfs is not None and after_lkfs is not None
                else level_step, 4),
            "absolute_step_lu": _finite(
                abs((after_lkfs - before_lkfs)
                    if before_lkfs is not None and after_lkfs is not None
                    else (level_step or 0.0)), 4),
            "above_analysis_floor": above_analysis_floor,
            "level_energy_rms": _finite(level_energy_rms),
            "level_energy_gate": _finite(level_energy_gate),
            "local_baselines": {
                "level_step_db": level_baseline,
                "sample_difference": sample_baseline,
                "slope_difference": slope_baseline,
                "second_difference": second_baseline,
                "broadband_impulse_strength": broadband["baseline"],
                "f0_step": f0_baseline,
                "period_shape": shape_baseline,
                "spectral_step": spectral["baseline_step"],
                "spectral_slope": spectral["baseline_slope"],
                "spectral_flux": spectral.get("baseline_flux"),
                "formants": formants["baselines"],
            },
            "plot_data": plot,
        }
        # Keep renderer-owned crossover evidence beside the measured acoustic
        # metrics. The join editor can then show requested milliseconds and
        # the pitchmark-snapped/context-capped result without conflating them.
        for key, value in splice.items():
            if key == "unit_index" or str(key).startswith("crossover_"):
                row[key] = value
        for formant_index in range(int(config.formant_count)):
            prefix = f"f{formant_index + 1}"
            detail = (formants["per_formant"][formant_index]
                      if formant_index < len(formants["per_formant"]) else {})
            row[f"{prefix}_available"] = bool(detail.get("available"))
            for output_name, source_name in (
                    ("frequency_jump_hz", "frequency_jump_hz"),
                    ("frequency_jump_normalized", "frequency_jump_normalized"),
                    ("frequency_jump_novelty", "frequency_jump_novelty"),
                    ("slope_break", "frequency_slope_break"),
                    ("slope_break_novelty", "frequency_slope_break_novelty"),
                    ("bandwidth_jump", "bandwidth_jump_log_ratio"),
                    ("prominence_jump_db", "prominence_jump_db"),
                    ("energy_jump", "normalized_energy_jump_log_ratio"),
                    ("tracking_confidence", "tracking_confidence")):
                row[f"{prefix}_{output_name}"] = _finite(
                    detail.get(source_name))
        row.update({key: _finite(value, 6) if isinstance(value, float) else value
                    for key, value in collar.items()})
        LOGGER.info(
            "Join %.6f s (%s) -> %s, severity %.3f; %s",
            when, phone, dominant, severity, row["classification_reason"])
        return row

    def analyze(self) -> dict[str, object]:
        # Candidate repair searches do not rank on LUFS. They may disable this
        # whole-signal IIR pass and use an approximate raw-signal floor, while
        # normal/public reports retain exact K-weighted levels by default.
        weighted = (loudness.k_weight(self.samples, self.sample_rate)
                    if self.compute_k_weighted_level else
                    np.asarray(self.samples, dtype=np.float64))
        if self.include_curves:
            join_curve = loudness._loudness_curve_from_weighted(
                weighted, self.sample_rate, window_ms=self.window_ms,
                hop_ms=self.hop_ms)
            momentary = loudness._loudness_curve_from_weighted(
                weighted, self.sample_rate, window_ms=400.0, hop_ms=100.0)
        else:
            join_curve = []
            momentary = []
        units, choices = _source_units(
            self.rows, self.selected, self.alternatives)
        splices = _splice_rows(self.rows, self.splice_records)
        joins = [self._join(splice, units, choices, weighted)
                 for splice in splices
                 if 0 < int(splice["segment_index"]) < len(units)]
        ranking = sorted(range(len(joins)), key=lambda index: (
            -float(joins[index].get("severity_score") or 0.0),
            float(joins[index].get("time") or 0.0)))
        for rank, join_index in enumerate(ranking, 1):
            joins[join_index]["severity_rank"] = rank
        flagged = [row for row in joins if row["flagged"]]
        analyzed_steps = [float(row["absolute_step_lu"] or 0.0)
                          for row in joins
                          if row["above_analysis_floor"]]
        return {
            "schema_version": JOIN_DISCONTINUITY_SCHEMA_VERSION,
            "method": "continuous-pitch-synchronous-formant-content-v6",
            "sample_rate": self.sample_rate,
            "duration": _finite(len(self.samples) / self.sample_rate, 9),
            "analysis_config": asdict(self.config),
            "severity_weights": self.config.weights,
            "join_curve": join_curve,
            "momentary_curve": momentary,
            "analysis_curves_included": self.include_curves,
            "k_weighted_level_computed": self.compute_k_weighted_level,
            "target_pitchmarks": [float(value) for value in self.pitchmarks],
            "segments": self.rows,
            "units": units,
            "joins": joins,
            "ranking": [int(joins[index]["segment_index"])
                        for index in ranking],
            "ranking_is_calibrated": False,
            "ranking_note": (
                "Severity is a configurable sorting aid, not a validated "
                "perceptual score. Raw and local-novelty components remain "
                "authoritative."),
            "summary": {
                "join_count": len(joins),
                "above_analysis_floor_join_count": sum(
                    bool(row["above_analysis_floor"]) for row in joins),
                "flagged_join_count": len(flagged),
                "expected_burst_context_flag_count": sum(
                    bool(row.get("broadband_context_may_be_expected")) and
                    "BROADBAND_IMPULSE" in row.get("issues", ())
                    for row in joins),
                "unexpected_broadband_event_count": sum(
                    not bool(row.get("broadband_context_may_be_expected")) and
                    "BROADBAND_IMPULSE" in row.get("issues", ())
                    for row in joins),
                "unexpected_content_dropout_count": sum(
                    not bool(row.get("content_dropout_expected")) and
                    "CONTENT_DROPOUT" in row.get("issues", ())
                    for row in joins),
                "expected_low_energy_handoff_count": sum(
                    bool(row.get("content_dropout_expected"))
                    for row in joins),
                "exact_splice_count": sum(
                    not bool(row["position_estimated"]) for row in joins),
                "estimated_splice_count": sum(
                    bool(row["position_estimated"]) for row in joins),
                "flag_threshold_lu": _finite(self.config.level_step_db, 4),
                "classification_score": _finite(
                    self.config.classification_score, 4),
                "median_analyzed_step_lu": _finite(
                    float(np.median(analyzed_steps)) if analyzed_steps else 0.0,
                    4),
                "maximum_analyzed_step_lu": _finite(
                    max(analyzed_steps, default=0.0), 4),
                "maximum_severity": _finite(max(
                    (float(row["severity_score"] or 0.0) for row in joins),
                    default=0.0), 4),
                "dominant_issue_counts": {
                    label: sum(row["dominant_issue"] == label for row in joins)
                    for label in sorted(set(
                        str(row["dominant_issue"]) for row in joins))
                },
            },
        }
