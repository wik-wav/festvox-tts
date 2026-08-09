"""Measured, pitch-aware joins for the dependency-free diphone renderer.

The Festival/UniSyn renderer performs its own pitch-synchronous overlap-add.
This module gives the standalone renderer the same basic invariants without
adding NumPy or another mandatory dependency: estimate each side separately,
search a small source-local neighbourhood, align periodic waveforms, apply a
bounded gradual gain correction, and overlap with complementary Hann ramps.

The functions operate on signed 16-bit PCM.  Analysis windows never cross the
join; contextual unit selection happens before this module is called.
"""

from __future__ import annotations

import array
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Optional, Sequence


_EPSILON = 1e-12
JOIN_SYNTHESIS_CONDITIONING_VERSION = 13


class JoinConstraintError(ValueError):
    """A source-local join constraint has no physically valid solution."""

    def __init__(self, code: str, **details: int) -> None:
        super().__init__(code)
        self.code = str(code)
        self.details = {
            str(key): int(value) for key, value in sorted(details.items())
        }

_APERIODIC_PHONE_HINTS = frozenset({
    "f", "v", "s", "z", "sh", "zh", "th", "dh", "h", "hh", "x",
    "ch", "jh", "ts", "dz",
})
_SILENCE_OR_CLOSURE_PHONE_HINTS = frozenset({
    "pau", "sil", "sp", "cl", "q", "brth",
})


@dataclass(frozen=True)
class JoinSynthesisConfig:
    """Conservative defaults for speech joins at common sample rates."""

    crossover_ms: float = 40.0
    # Retained for compatibility with serialized callers from version 12.
    # Duration is no longer derived from this pitch-dependent count.
    overlap_periods: float = 3.0
    minimum_overlap_ms: float = 7.0
    maximum_overlap_ms: float = 100.0
    unvoiced_overlap_ms: float = 10.0
    search_ms: float = 12.0
    f0_min_hz: float = 45.0
    f0_max_hz: float = 700.0
    voiced_correlation_minimum: float = 0.48
    minimum_rms_pcm: float = 96.0
    gain_minimum: float = 0.80
    gain_maximum: float = 1.25
    period_weight: float = 1.0
    level_weight: float = 0.34
    correlation_weight: float = 2.4
    spectral_weight: float = 0.55
    boundary_weight: float = 0.03
    content_loss_weight: float = 8.0
    content_reference_minimum_rms_pcm: float = 64.0
    content_reference_fraction: float = 0.02
    minimum_source_energy_retention: float = 0.30
    minimum_crossfade_energy_retention: float = 0.45
    local_retention_frame_ms: float = 1.0
    minimum_local_crossfade_energy_retention: float = 0.30
    validation_novelty_limit: float = 8.0
    validation_level_step_limit_db: float = 6.0
    validation_f0_step_limit_semitones: float = 1.5
    validation_best_period_correlation_minimum: float = 0.45
    validation_period_shape_mismatch_limit: float = 0.75
    validation_spectral_envelope_distance_limit: float = 1.35
    validation_failure_weight: float = 6.0


@dataclass(frozen=True)
class JoinSynthesisDecision:
    method: str
    voiced: bool
    splice_sample: int
    handoff_start_sample: int
    handoff_end_sample: int
    overlap_samples: int
    left_trim_samples: int
    right_skip_samples: int
    left_period_samples: Optional[int]
    right_period_samples: Optional[int]
    left_f0_hz: Optional[float]
    right_f0_hz: Optional[float]
    f0_step_semitones: Optional[float]
    left_rms: float
    right_rms: float
    level_step_db: float
    gain_ratio: float
    zero_lag_correlation: Optional[float]
    best_lag_correlation: Optional[float]
    phase_offset_samples: Optional[int]
    phase_offset_cycles: Optional[float]
    spectral_distance: float
    join_cost: float
    impulse_novelty: float
    validation_passed: bool
    content_gate_active: bool = False
    source_reference_rms: float = 0.0
    source_energy_retention: float = 1.0
    crossfade_energy_retention: float = 1.0
    content_attenuation_db: float = 0.0
    content_preservation_passed: bool = True
    phase_mix_energy_retention: float = 1.0
    local_crossfade_energy_retention: float = 1.0
    left_source_energy_retention: float = 1.0
    right_source_energy_retention: float = 1.0
    left_silence_allowed: bool = False
    right_silence_allowed: bool = False
    content_fallback_used: bool = False
    acoustic_validation_passed: bool = True
    acoustic_validation_gate_active: bool = False
    level_validation_passed: bool = True
    f0_validation_passed: bool = True
    period_correlation_validation_passed: bool = True
    period_shape_validation_passed: bool = True
    spectral_validation_passed: bool = True
    period_shape_mismatch: Optional[float] = None
    validation_failures: tuple[str, ...] = ()
    legacy_fallback_used: bool = False
    voicing_hint_reason: Optional[str] = None
    left_source_content_present: bool = True
    right_source_content_present: bool = True
    right_indexed_length_samples: Optional[int] = None
    right_skip_limit_samples: Optional[int] = None
    right_indexed_tail_samples: Optional[int] = None
    right_skip_limit_applied: bool = False
    requested_crossover_ms: float = 0.0
    effective_crossover_ms: float = 0.0
    crossover_period_count: Optional[int] = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values))


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(_EPSILON, numerator) /
                             max(_EPSILON, denominator))


def _normalize_phone_hint(phone: Optional[str]) -> str:
    value = str(phone or "").strip().lower()
    while value and value[-1].isdigit():
        value = value[:-1]
    return value


def _phone_hint_is_aperiodic(phone: Optional[str]) -> bool:
    return _normalize_phone_hint(phone) in _APERIODIC_PHONE_HINTS


def _phone_hint_allows_silence(phone: Optional[str]) -> bool:
    return _normalize_phone_hint(phone) in _SILENCE_OR_CLOSURE_PHONE_HINTS


def _normalized_correlation(left: Sequence[float],
                            right: Sequence[float]) -> float:
    count = min(len(left), len(right))
    if count < 3:
        return 0.0
    left_mean = _mean(left[:count])
    right_mean = _mean(right[:count])
    numerator = 0.0
    left_power = 0.0
    right_power = 0.0
    for index in range(count):
        a = float(left[index]) - left_mean
        b = float(right[index]) - right_mean
        numerator += a * b
        left_power += a * a
        right_power += b * b
    denominator = math.sqrt(left_power * right_power)
    return numerator / denominator if denominator > _EPSILON else 0.0


def _resample(values: Sequence[float], count: int) -> tuple[float, ...]:
    if count <= 0 or not values:
        return ()
    if len(values) == 1:
        return (float(values[0]),) * count
    if count == 1:
        return (float(values[0]),)
    scale = (len(values) - 1) / float(count - 1)
    output = []
    for index in range(count):
        position = index * scale
        first = int(position)
        fraction = position - first
        second = min(len(values) - 1, first + 1)
        output.append(float(values[first]) * (1.0 - fraction) +
                      float(values[second]) * fraction)
    return tuple(output)


def _normalized_period(values: Sequence[float], count: int
                       ) -> tuple[float, ...]:
    samples = _resample(values, count)
    if not samples:
        return ()
    average = _mean(samples)
    centered = tuple(value - average for value in samples)
    energy = _rms(centered)
    if energy <= _EPSILON:
        return ()
    return tuple(value / energy for value in centered)


def _period_match_metrics(
    left: Sequence[float],
    right: Sequence[float],
    left_period: Optional[int],
    right_period: Optional[int],
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[float]]:
    """Compare complete periods without allowing either window to cross.

    The shape metric is normalized RMS error after the best circular phase
    alignment.  Unit-RMS, zero-mean periods make it amplitude independent.
    """
    if not left_period or not right_period:
        return None, None, None, None
    if len(left) < left_period or len(right) < right_period:
        return None, None, None, None
    common = max(8, int(round((left_period + right_period) * 0.5)))
    outgoing = _normalized_period(left[-left_period:], common)
    incoming = _normalized_period(right[:right_period], common)
    if not outgoing or not incoming:
        return None, None, None, None
    zero = _normalized_correlation(outgoing, incoming)
    best = zero
    best_lag = 0
    best_values = incoming
    maximum_lag = max(1, common // 3)
    for lag in range(-maximum_lag, maximum_lag + 1):
        shifted = incoming[-lag:] + incoming[:-lag] if lag else incoming
        score = _normalized_correlation(outgoing, shifted)
        if score > best:
            best = score
            best_lag = lag
            best_values = shifted
    error = math.sqrt(_mean(tuple(
        (a - b) ** 2 for a, b in zip(outgoing, best_values))))
    # Two unrelated unit-RMS signals have an expected difference RMS of
    # sqrt(2), so this scale is easier to configure than raw PCM error.
    shape_mismatch = error / math.sqrt(2.0)
    return zero, best, best_lag, shape_mismatch


def _spectral_shape(values: Sequence[float], bands: int = 12
                    ) -> tuple[float, ...]:
    """Compact amplitude-independent log spectral envelope.

    A full half-spectrum is pooled into broad linear-frequency bands.  This
    intentionally discards individual harmonic detail while retaining the
    coarse source/filter shape that should remain reasonably continuous at a
    bridge.  Subtracting the mean log-band value removes overall energy so the
    level gate remains independent.
    """
    if len(values) < 8:
        return ()
    # Twelve broad bands do not benefit from resolving every harmonic.  A
    # fixed upper bound of 64 samples leaves at least two DFT bins per band
    # while keeping the local-cut search practical for large UTAU banks.
    count = min(64, len(values))
    samples = _resample(values, count)
    mean = _mean(samples)
    windowed = tuple(
        (sample - mean) * (0.5 - 0.5 * math.cos(
            2.0 * math.pi * index / max(1, count - 1)))
        for index, sample in enumerate(samples)
    )
    bin_powers = []
    for frequency_bin in range(1, count // 2 + 1):
        real = 0.0
        imaginary = 0.0
        for index, sample in enumerate(windowed):
            phase = 2.0 * math.pi * frequency_bin * index / count
            real += sample * math.cos(phase)
            imaginary -= sample * math.sin(phase)
        bin_powers.append(real * real + imaginary * imaginary)
    band_values = []
    for band in range(bands):
        first = int(round(band * len(bin_powers) / bands))
        last = int(round((band + 1) * len(bin_powers) / bands))
        selected = bin_powers[first:max(first + 1, last)]
        band_values.append(0.5 * math.log(max(_EPSILON, _mean(selected))))
    average = _mean(band_values)
    return tuple(value - average for value in band_values)


def _spectral_distance(left: Sequence[float],
                       right: Sequence[float]) -> float:
    a = _spectral_shape(left)
    b = _spectral_shape(right)
    if not a or len(a) != len(b):
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def _period_candidates(sample_rate: int, expected_f0_hz: Optional[float],
                       config: JoinSynthesisConfig) -> range:
    if expected_f0_hz and config.f0_min_hz <= expected_f0_hz <= \
            config.f0_max_hz:
        expected = sample_rate / expected_f0_hz
        minimum = max(2, int(round(expected * 0.68)))
        maximum = max(minimum + 1, int(round(expected * 1.47)))
    else:
        minimum = max(2, int(sample_rate / config.f0_max_hz))
        maximum = max(minimum + 1, int(sample_rate / config.f0_min_hz))
    return range(minimum, maximum + 1)


def _estimate_period(values: Sequence[int], sample_rate: int,
                     expected_f0_hz: Optional[float],
                     config: JoinSynthesisConfig) -> tuple[Optional[int], float]:
    """Return a local period and periodicity; ``values`` is one join side."""
    if not values or _rms(values) < config.minimum_rms_pcm:
        return None, 0.0
    maximum_window = max(32, int(round(sample_rate * 0.060)))
    samples = tuple(float(value) for value in values[-maximum_window:])
    best_period = None
    best_score = -1.0
    for period in _period_candidates(sample_rate, expected_f0_hz, config):
        available = len(samples) - period
        if available < max(12, period):
            continue
        count = min(available, max(period * 3, 32))
        score = _normalized_correlation(
            samples[-count:], samples[-count - period:-period])
        if score > best_score:
            best_score = score
            best_period = period
    if best_period is None or best_score < config.voiced_correlation_minimum:
        return None, max(0.0, best_score)
    return best_period, min(1.0, best_score)


def _start_period(values: Sequence[int], sample_rate: int,
                  expected_f0_hz: Optional[float],
                  config: JoinSynthesisConfig) -> tuple[Optional[int], float]:
    maximum_window = max(32, int(round(sample_rate * 0.060)))
    head = tuple(values[:maximum_window])
    # Reversing lets the same strictly-contained tail estimator inspect the
    # first periods without ever crossing the splice.
    return _estimate_period(tuple(reversed(head)), sample_rate,
                            expected_f0_hz, config)


def _phase_offsets(period: int, maximum: int) -> tuple[int, ...]:
    if maximum <= 0:
        return (0,)
    step = max(1, int(round(period / 10.0)))
    values = list(range(0, maximum + 1, step))
    if values[-1] != maximum:
        values.append(maximum)
    return tuple(values)


def _source_boundary_novelty(values: Sequence[int], boundary: int) -> float:
    if boundary <= 0 or boundary >= len(values):
        return 0.0
    first = max(1, boundary - 96)
    last = min(len(values), boundary + 96)
    differences = [abs(int(values[index]) - int(values[index - 1]))
                   for index in range(first, last)]
    baseline = statistics.median(differences) if differences else 1.0
    jump = abs(int(values[boundary]) - int(values[boundary - 1]))
    return jump / max(1.0, float(baseline))


def _join_descriptor(values: Sequence[int], start: int, overlap: int,
                     external_boundary: int) -> dict[str, object]:
    samples = tuple(float(value) for value in
                    values[start:start + overlap])
    mean = _mean(samples)
    centered = tuple(value - mean for value in samples)
    return {
        "samples": samples,
        "centered": centered,
        "power": sum(value * value for value in centered),
        "rms": _rms(samples),
        "spectrum": _spectral_shape(samples),
        "boundary": _source_boundary_novelty(values, external_boundary),
    }


def _descriptor_correlation(left: dict[str, object],
                            right: dict[str, object]) -> float:
    a = left["centered"]
    b = right["centered"]
    denominator = math.sqrt(float(left["power"]) * float(right["power"]))
    if denominator <= _EPSILON:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denominator


def _descriptor_spectral_distance(left: dict[str, object],
                                  right: dict[str, object]) -> float:
    a = left["spectrum"]
    b = right["spectrum"]
    if not a or len(a) != len(b):
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def _acoustic_validation_metrics(
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    left_rms: float,
    right_rms: float,
    spectral_distance: float,
    left_period: Optional[int],
    right_period: Optional[int],
    gate_active: bool,
    config: JoinSynthesisConfig,
) -> dict[str, object]:
    level_step = abs(_db_ratio(right_rms, left_rms))
    f0_step = (abs(12.0 * math.log2(left_period / right_period))
               if left_period and right_period else None)
    zero_correlation, best_correlation, best_lag, shape_mismatch = (
        _period_match_metrics(
            left_samples, right_samples, left_period, right_period))

    level_passed = (not gate_active or
                    level_step <= config.validation_level_step_limit_db)
    f0_passed = (not gate_active or f0_step is None or
                 f0_step <= config.validation_f0_step_limit_semitones)
    period_correlation_passed = (
        not gate_active or left_period is None or right_period is None or
        (best_correlation is not None and best_correlation >=
         config.validation_best_period_correlation_minimum)
    )
    period_shape_passed = (
        not gate_active or left_period is None or right_period is None or
        (shape_mismatch is not None and shape_mismatch <=
         config.validation_period_shape_mismatch_limit)
    )
    spectral_passed = (
        not gate_active or spectral_distance <=
        config.validation_spectral_envelope_distance_limit
    )
    failures = []
    if not level_passed:
        failures.append("LEVEL_MISMATCH")
    if not f0_passed:
        failures.append("F0_MISMATCH")
    if not period_correlation_passed:
        failures.append("PERIOD_CORRELATION")
    if not period_shape_passed:
        failures.append("PERIOD_SHAPE_MISMATCH")
    if not spectral_passed:
        failures.append("SPECTRAL_ENVELOPE_MISMATCH")
    return {
        "acoustic_validation_gate_active": gate_active,
        "acoustic_validation_passed": not failures,
        "level_validation_passed": level_passed,
        "f0_validation_passed": f0_passed,
        "period_correlation_validation_passed":
            period_correlation_passed,
        "period_shape_validation_passed": period_shape_passed,
        "spectral_validation_passed": spectral_passed,
        "validation_failures": tuple(failures),
        "absolute_level_step_db": level_step,
        "absolute_f0_step_semitones": f0_step,
        "spectral_distance": spectral_distance,
        "period_zero_lag_correlation": zero_correlation,
        "period_best_lag_correlation": best_correlation,
        "period_best_lag_samples": best_lag,
        "period_shape_mismatch": shape_mismatch,
    }


def _acoustic_validation_penalty(metrics: dict[str, object],
                                 config: JoinSynthesisConfig) -> float:
    """Rank rejected candidates without obscuring the individual gates."""
    if bool(metrics["acoustic_validation_passed"]):
        return 0.0
    penalty = 0.0
    level = float(metrics["absolute_level_step_db"])
    if level > config.validation_level_step_limit_db:
        penalty += level / max(_EPSILON,
                               config.validation_level_step_limit_db) - 1.0
    f0_step = metrics["absolute_f0_step_semitones"]
    if f0_step is not None and float(f0_step) > \
            config.validation_f0_step_limit_semitones:
        penalty += (float(f0_step) /
                    max(_EPSILON,
                        config.validation_f0_step_limit_semitones) - 1.0)
    correlation = metrics["period_best_lag_correlation"]
    if correlation is not None:
        penalty += max(0.0,
                       config.validation_best_period_correlation_minimum -
                       float(correlation))
    shape = metrics["period_shape_mismatch"]
    if shape is not None:
        penalty += max(
            0.0, float(shape) -
            config.validation_period_shape_mismatch_limit)
    return penalty + max(
        0.0,
        float(metrics.get("spectral_distance", 0.0)) -
        config.validation_spectral_envelope_distance_limit,
    )


def _crossfade_energy_retention(left_values: Sequence[int],
                                right_values: Sequence[int],
                                gain: float) -> float:
    """Return retained overlap energy relative to its weighted sources.

    The comparison uses only the middle half of the crossfade, where both
    sources contribute materially.  Its denominator is the root-sum-square
    energy of the independently weighted sources.  Consequently an ordinary
    silence-to-speech fade retains approximately 1.0, while destructive phase
    cancellation cannot make a join appear cleaner merely by erasing it.
    """
    count = min(len(left_values), len(right_values))
    if count <= 0:
        return 1.0
    first = count // 4 if count >= 8 else 0
    last = count - first if count >= 8 else count
    actual_power = 0.0
    reference_power = 0.0
    used = 0
    for index in range(first, last):
        progress = index / float(max(1, count - 1))
        incoming_weight = 0.5 - 0.5 * math.cos(math.pi * progress)
        outgoing_weight = 1.0 - incoming_weight
        gradual_gain = gain + (1.0 - gain) * progress
        outgoing = float(left_values[index]) * outgoing_weight
        incoming = (float(right_values[index]) * incoming_weight *
                    gradual_gain)
        actual_power += (outgoing + incoming) ** 2
        reference_power += outgoing ** 2 + incoming ** 2
        used += 1
    if used <= 0 or reference_power <= _EPSILON:
        return 1.0
    return math.sqrt(actual_power / reference_power)


def _crossfade_local_energy_retention(
    left_values: Sequence[int],
    right_values: Sequence[int],
    gain: float,
    sample_rate: int,
    frame_ms: float,
) -> float:
    """Return a low-percentile short-frame anti-cancellation measure.

    Aggregate overlap RMS can conceal a very narrow null at the center of an
    otherwise energetic voiced crossfade.  Evaluate short overlapping frames
    in the mixed middle half and return a robust low percentile.  The caller
    enables this gate only when both sides are expected to contain speech, so
    real stop closures and silence remain exempt.
    """
    count = min(len(left_values), len(right_values))
    if count <= 0:
        return 1.0
    frame = max(4, int(round(sample_rate * max(0.25, frame_ms) / 1000.0)))
    frame = min(frame, count)
    first = count // 4
    last = count - first
    if last - first < frame:
        first = 0
        last = count
    hop = max(1, frame // 4)
    starts = list(range(first, max(first + 1, last - frame + 1), hop))
    final_start = max(first, last - frame)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    ratios = []
    for start in starts:
        actual_power = 0.0
        reference_power = 0.0
        for index in range(start, min(count, start + frame)):
            progress = index / float(max(1, count - 1))
            incoming_weight = 0.5 - 0.5 * math.cos(math.pi * progress)
            outgoing_weight = 1.0 - incoming_weight
            gradual_gain = gain + (1.0 - gain) * progress
            outgoing = float(left_values[index]) * outgoing_weight
            incoming = (float(right_values[index]) * incoming_weight *
                        gradual_gain)
            actual_power += (outgoing + incoming) ** 2
            reference_power += outgoing ** 2 + incoming ** 2
        if reference_power > _EPSILON:
            ratios.append(math.sqrt(actual_power / reference_power))
    if not ratios:
        return 1.0
    ratios.sort()
    return ratios[int(round(0.20 * max(0, len(ratios) - 1)))]


def _retention_deficit(value: float, minimum: float) -> float:
    if minimum <= _EPSILON or value >= minimum:
        return 0.0
    return (minimum - max(0.0, value)) / minimum


def _content_floor_rms(values: Sequence[float],
                       frame_samples: Optional[int] = None) -> float:
    """Robust short-frame energy floor used to expose erased subregions."""
    count = len(values)
    if count <= 0:
        return 0.0
    frame = (max(8, min(count, int(frame_samples)))
             if frame_samples else max(8, min(count, count // 4)))
    if count <= frame:
        return _rms(values)
    hop = max(1, frame // 2)
    starts = list(range(0, count - frame + 1, hop))
    if starts[-1] != count - frame:
        starts.append(count - frame)
    energies = sorted(_rms(values[start:start + frame]) for start in starts)
    return energies[int(round(0.20 * max(0, len(energies) - 1)))]


def _boundary_novelty(left: Sequence[int], right: Sequence[int],
                      left_start: int, right_start: int,
                      overlap: int) -> float:
    local_differences = []
    first = max(1, left_start - 64)
    last = min(len(left), left_start + overlap)
    local_differences.extend(abs(int(left[i]) - int(left[i - 1]))
                             for i in range(first, last))
    first = max(1, right_start)
    last = min(len(right), right_start + overlap + 64)
    local_differences.extend(abs(int(right[i]) - int(right[i - 1]))
                             for i in range(first, last))
    baseline = statistics.median(local_differences) if local_differences else 1.0
    baseline = max(1.0, float(baseline))
    start_jump = (abs(int(left[left_start]) - int(left[left_start - 1]))
                  if left_start > 0 else 0.0)
    right_end = right_start + overlap
    end_jump = (abs(int(right[right_end]) - int(right[right_end - 1]))
                if right_end < len(right) else 0.0)
    return max(start_jump, end_jump) / baseline


def _candidate_cost(left: Sequence[int], right: Sequence[int],
                    left_start: int, right_start: int, overlap: int,
                    left_period: Optional[int], right_period: Optional[int],
                    config: JoinSynthesisConfig) -> tuple[float, dict[str, float]]:
    a = left[left_start:left_start + overlap]
    b = right[right_start:right_start + overlap]
    correlation = _normalized_correlation(a, b)
    left_rms = _rms(a)
    right_rms = _rms(b)
    level = abs(math.log(max(_EPSILON, left_rms) /
                         max(_EPSILON, right_rms)))
    period = (abs(math.log(left_period / right_period))
              if left_period and right_period else 0.0)
    spectral = _spectral_distance(a, b)
    boundary = _boundary_novelty(
        left, right, left_start, right_start, overlap)
    cost = (config.period_weight * period +
            config.level_weight * level +
            config.correlation_weight * (1.0 - correlation) +
            config.spectral_weight * spectral +
            config.boundary_weight * min(12.0, boundary))
    return cost, {
        "correlation": correlation,
        "left_rms": left_rms,
        "right_rms": right_rms,
        "spectral": spectral,
        "boundary": boundary,
    }


def _best_local_join(left: Sequence[int], right: Sequence[int],
                     overlap: int, sample_rate: int,
                     left_period: Optional[int],
                     right_period: Optional[int],
                     config: JoinSynthesisConfig,
                     allow_silent_left: bool = False,
                     allow_silent_right: bool = False,
                     enforce_acoustic_similarity: bool = True,
                     right_skip_limit: Optional[int] = None,
                     ) -> tuple[int, int, float, dict[str, object]]:
    search = max(0, int(round(config.search_ms * sample_rate / 1000.0)))
    left_maximum = min(search, max(0, len(left) - overlap - 1))
    right_maximum = min(search, max(0, len(right) - overlap - 1))
    if right_skip_limit is not None:
        right_maximum = min(right_maximum, max(0, int(right_skip_limit)))
    reference_period = left_period or right_period or max(2, overlap // 2)
    left_trims = _phase_offsets(reference_period, left_maximum)
    right_skips = _phase_offsets(reference_period, right_maximum)
    left_rows = []
    for left_trim in left_trims:
        left_start = len(left) - overlap - left_trim
        if left_start >= 1:
            left_rows.append((
                left_trim, left_start,
                _join_descriptor(left, left_start, overlap, left_start),
            ))
    right_rows = [
        (right_skip, _join_descriptor(
            right, right_skip, overlap, right_skip + overlap))
        for right_skip in right_skips
    ]
    period_cost = (abs(math.log(left_period / right_period))
                   if left_period and right_period else 0.0)
    nominal_left = left[max(0, len(left) - overlap):]
    nominal_right = right[:overlap]
    content_frame = (int(round((left_period + right_period) * 0.5))
                     if left_period and right_period else
                     max(8, int(round(sample_rate * 0.004))))
    nominal_left_rms = _content_floor_rms(nominal_left, content_frame)
    nominal_right_rms = _content_floor_rms(nominal_right, content_frame)
    # Reference the undisturbed source immediately outside the proposed
    # overlap as well as the annotated collars.  Otherwise a zeroed collar
    # could define its own zero baseline and falsely pass validation.
    outer_left_rms = _content_floor_rms(left[
        max(0, len(left) - 2 * overlap):max(0, len(left) - overlap)],
        content_frame)
    outer_right_rms = _content_floor_rms(right[
        min(len(right), overlap):min(len(right), 2 * overlap)],
        content_frame)
    left_reference_rms = max(nominal_left_rms, outer_left_rms)
    right_reference_rms = max(nominal_right_rms, outer_right_rms)
    local_unit_rms = max(_rms(left), _rms(right))
    audibility_threshold = max(
        config.content_reference_minimum_rms_pcm,
        config.content_reference_fraction * local_unit_rms)
    left_reference_active = left_reference_rms >= audibility_threshold
    right_reference_active = right_reference_rms >= audibility_threshold
    left_source_required = left_reference_active and not allow_silent_left
    right_source_required = right_reference_active and not allow_silent_right
    active_reference_rms = tuple(
        value for value, active in (
            (left_reference_rms, left_reference_active),
            (right_reference_rms, right_reference_active)) if active)
    source_reference_rms = max(active_reference_rms, default=0.0)
    left_missing_source_content = bool(
        not allow_silent_left and
        left_reference_rms < audibility_threshold)
    right_missing_source_content = bool(
        not allow_silent_right and
        right_reference_rms < audibility_threshold)
    for _trim, _start, descriptor in left_rows:
        descriptor["content_rms"] = _content_floor_rms(
            descriptor["samples"], content_frame)
    for _skip, descriptor in right_rows:
        descriptor["content_rms"] = _content_floor_rms(
            descriptor["samples"], content_frame)
    candidates: list[dict[str, object]] = []
    for left_trim, left_start, left_descriptor in left_rows:
        for right_skip, right_descriptor in right_rows:
            correlation = _descriptor_correlation(
                left_descriptor, right_descriptor)
            left_rms = float(left_descriptor["rms"])
            right_rms = float(right_descriptor["rms"])
            level = abs(math.log(max(_EPSILON, left_rms) /
                                 max(_EPSILON, right_rms)))
            spectral = _descriptor_spectral_distance(
                left_descriptor, right_descriptor)
            boundary = max(float(left_descriptor["boundary"]),
                           float(right_descriptor["boundary"]))
            gain = (left_rms / right_rms
                    if right_rms > _EPSILON else 1.0)
            gain = max(config.gain_minimum,
                       min(config.gain_maximum, gain))
            left_content_rms = float(left_descriptor["content_rms"])
            right_content_rms = float(right_descriptor["content_rms"])
            left_retention = (left_content_rms / left_reference_rms
                              if left_reference_active else 1.0)
            right_retention = (right_content_rms / right_reference_rms
                               if right_reference_active else 1.0)
            required_retentions = []
            if left_source_required:
                required_retentions.append(left_retention)
            if right_source_required:
                required_retentions.append(right_retention)
            source_retention = (
                0.0
                if left_missing_source_content or right_missing_source_content
                else min(required_retentions, default=1.0)
            )
            phase_gate_active = bool(
                max(left_rms, right_rms) >= audibility_threshold)
            source_gate_active = bool(
                left_source_required or right_source_required)
            source_feasible = bool(
                not source_gate_active or source_retention >=
                config.minimum_source_energy_retention)
            source_feasible = bool(
                source_feasible and
                not left_missing_source_content and
                not right_missing_source_content)
            acoustic_gate_active = bool(
                enforce_acoustic_similarity and
                left_rms >= audibility_threshold and
                right_rms >= audibility_threshold and
                not allow_silent_left and not allow_silent_right)
            source_content_penalty = config.content_loss_weight * (
                _retention_deficit(
                    source_retention,
                    config.minimum_source_energy_retention)
            )
            cost = (config.period_weight * period_cost +
                    config.level_weight * level +
                    config.correlation_weight * (1.0 - correlation) +
                    config.spectral_weight * spectral +
                    config.boundary_weight * min(12.0, boundary) +
                    source_content_penalty)
            if boundary > config.validation_novelty_limit:
                cost += 4.0 * (
                    boundary - config.validation_novelty_limit)
            # Prefer the annotated cut when acoustic costs are effectively
            # tied; source movement is a repair, not a new duration model.
            cost += 0.12 * (left_trim + right_skip) / max(1, search)
            metrics = {
                "correlation": correlation,
                "left_rms": left_rms,
                "right_rms": right_rms,
                "spectral": spectral,
                "boundary": boundary,
                "source_reference_rms": source_reference_rms,
                "source_energy_retention": source_retention,
                "left_source_energy_retention": left_retention,
                "right_source_energy_retention": right_retention,
                "left_silence_allowed": float(allow_silent_left),
                "right_silence_allowed": float(allow_silent_right),
                "content_audibility_threshold": audibility_threshold,
                "left_missing_source_content": float(
                    left_missing_source_content),
                "right_missing_source_content": float(
                    right_missing_source_content),
            }
            candidates.append({
                "base_cost": cost,
                "movement": left_trim + right_skip,
                "left_trim": left_trim,
                "right_skip": right_skip,
                "source_feasible": source_feasible,
                "source_gate_active": source_gate_active,
                "phase_gate_active": phase_gate_active,
                "gain": gain,
                "source_content_penalty": source_content_penalty,
                "acoustic_gate_active": acoustic_gate_active,
                "left_samples": left_descriptor["samples"],
                "right_samples": right_descriptor["samples"],
                "metrics": metrics,
            })

    if not candidates:
        left_start = max(0, len(left) - overlap)
        cost, metrics = _candidate_cost(
            left, right, left_start, 0, overlap,
            left_period, right_period, config)
        left_rms = float(metrics["left_rms"])
        right_rms = float(metrics["right_rms"])
        local_unit_rms = max(_rms(left), _rms(right))
        threshold = max(
            config.content_reference_minimum_rms_pcm,
            config.content_reference_fraction * local_unit_rms)
        acoustic_metrics = _acoustic_validation_metrics(
            left[left_start:left_start + overlap], right[:overlap],
            left_rms, right_rms, float(metrics["spectral"]),
            left_period, right_period,
            bool(left_rms >= threshold and right_rms >= threshold and
                 not allow_silent_left and not allow_silent_right),
            config)
        metrics.update({
            "content_gate_active": 0.0,
            "source_reference_rms": 0.0,
            "source_energy_retention": 1.0,
            "left_source_energy_retention": 1.0,
            "right_source_energy_retention": 1.0,
            "crossfade_energy_retention": 1.0,
            "phase_mix_energy_retention": 1.0,
            "local_crossfade_energy_retention": 1.0,
            "content_feasible": 1.0,
            "left_silence_allowed": float(allow_silent_left),
            "right_silence_allowed": float(allow_silent_right),
            "content_audibility_threshold": threshold,
        })
        metrics.update(acoustic_metrics)
        return left_start, 0, cost, metrics

    def evaluate(candidate: dict[str, object]
                 ) -> tuple[int, float, int, int, int, dict[str, object]]:
        metrics = dict(candidate["metrics"])
        phase_mix_retention = _crossfade_energy_retention(
            candidate["left_samples"], candidate["right_samples"],
            float(candidate["gain"]))
        local_mix_retention = _crossfade_local_energy_retention(
            candidate["left_samples"], candidate["right_samples"],
            float(candidate["gain"]), sample_rate,
            config.local_retention_frame_ms)
        source_retention = float(metrics["source_energy_retention"])
        crossfade_retention = min(source_retention, phase_mix_retention)
        content_gate_active = bool(
            candidate["source_gate_active"] or
            candidate["phase_gate_active"] or
            metrics.get("left_missing_source_content", 0.0) or
            metrics.get("right_missing_source_content", 0.0))
        content_feasible = bool(
            candidate["source_feasible"] and (
                not candidate["phase_gate_active"] or
                (
                    phase_mix_retention >=
                    config.minimum_crossfade_energy_retention and
                    local_mix_retention >=
                    config.minimum_local_crossfade_energy_retention
                )
            )
        )
        phase_content_penalty = config.content_loss_weight * (
            _retention_deficit(
                phase_mix_retention,
                config.minimum_crossfade_energy_retention) +
            _retention_deficit(
                local_mix_retention,
                config.minimum_local_crossfade_energy_retention)
        )
        metrics.update({
            "content_gate_active": float(content_gate_active),
            "crossfade_energy_retention": crossfade_retention,
            "phase_mix_energy_retention": phase_mix_retention,
            "local_crossfade_energy_retention": local_mix_retention,
            "content_penalty": (
                float(candidate["source_content_penalty"]) +
                phase_content_penalty
            ),
            "content_feasible": float(content_feasible),
        })
        acoustic_metrics = _acoustic_validation_metrics(
            candidate["left_samples"], candidate["right_samples"],
            float(metrics["left_rms"]), float(metrics["right_rms"]),
            float(metrics["spectral"]), left_period, right_period,
            bool(candidate["acoustic_gate_active"]), config)
        acoustic_feasible = bool(
            acoustic_metrics["acoustic_validation_passed"])
        validation_penalty = (
            config.validation_failure_weight *
            _acoustic_validation_penalty(acoustic_metrics, config)
        )
        metrics["validation_penalty"] = validation_penalty
        metrics.update(acoustic_metrics)
        return (
            0 if content_feasible and acoustic_feasible else 1,
            float(candidate["base_cost"]) + phase_content_penalty +
            validation_penalty,
            int(candidate["movement"]),
            int(candidate["left_trim"]),
            int(candidate["right_skip"]),
            metrics,
        )

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            float(candidates[index]["base_cost"]),
            int(candidates[index]["movement"]),
            int(candidates[index]["left_trim"]),
            int(candidates[index]["right_skip"]),
        ),
    )
    evaluated: list[tuple[
        int, float, int, int, int, dict[str, object]
    ]] = []
    evaluated_indexes: set[int] = set()
    # Every fully valid candidate outranks every rejected candidate. Among
    # valid candidates the validation penalty is zero, so the first passing
    # row in cheap-cost order is exactly the result of the exhaustive search.
    for index in order:
        if not bool(candidates[index]["source_feasible"]):
            continue
        row = evaluate(candidates[index])
        evaluated.append(row)
        evaluated_indexes.add(index)
        if row[0] == 0:
            return (
                len(left) - overlap - row[3], row[4], row[1], row[5]
            )

    # If no candidate passes, retain the old exhaustive ranking so the least
    # severe rejected collar is reported before the caller applies the exact
    # Legacy overlap on the same selected units.
    for index in order:
        if index not in evaluated_indexes:
            evaluated.append(evaluate(candidates[index]))
    best = min(evaluated, key=lambda row: row[:5])
    return len(left) - overlap - best[3], best[4], best[1], best[5]


def _crossfade(left_values: Sequence[int], right_values: Sequence[int],
               gain: float) -> array.array:
    count = min(len(left_values), len(right_values))
    output = array.array("h")
    for index in range(count):
        progress = index / float(max(1, count - 1))
        incoming = 0.5 - 0.5 * math.cos(math.pi * progress)
        outgoing = 1.0 - incoming
        gradual_gain = gain + (1.0 - gain) * progress
        value = (float(left_values[index]) * outgoing +
                 float(right_values[index]) * incoming * gradual_gain)
        output.append(max(-32768, min(32767, int(round(value)))))
    return output


def _impulse_novelty(values: Sequence[int], boundaries: Sequence[int]) -> float:
    differences = [abs(int(values[index]) - int(values[index - 1]))
                   for index in range(1, len(values))]
    if not differences:
        return 0.0
    scores = []
    for boundary in boundaries:
        if boundary <= 0 or boundary >= len(values):
            continue
        local = differences[max(0, boundary - 96):
                            min(len(differences), boundary + 96)]
        baseline = statistics.median(local) if local else 1.0
        scores.append(differences[boundary - 1] / max(1.0, baseline))
    return max(scores, default=0.0)


def adaptive_join_pcm16(
    left: array.array,
    right: Sequence[int],
    sample_rate: int,
    *,
    expected_f0_hz: Optional[float] = None,
    allow_silent_handoff: bool = False,
    allow_silent_left: Optional[bool] = None,
    allow_silent_right: Optional[bool] = None,
    left_phone: Optional[str] = None,
    right_phone: Optional[str] = None,
    enforce_acoustic_similarity: bool = True,
    right_indexed_length_samples: Optional[int] = None,
    minimum_right_indexed_tail_samples: int = 0,
    config: Optional[JoinSynthesisConfig] = None,
) -> JoinSynthesisDecision:
    """Append ``right`` to ``left`` with a measured pitch-aware overlap.

    Candidate windows are wholly inside their source units.  The function
    mutates ``left`` only after analysis has selected a join, and it reports
    every source trim/skip so callers can map phone boundaries accurately.
    """
    config = config or JoinSynthesisConfig()
    if allow_silent_left is None:
        allow_silent_left = bool(
            allow_silent_handoff or _phone_hint_allows_silence(left_phone))
    if allow_silent_right is None:
        allow_silent_right = bool(
            allow_silent_handoff or _phone_hint_allows_silence(right_phone))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not left or not right:
        original_length = len(left)
        left.extend(right)
        return JoinSynthesisDecision(
            "append", False, original_length, original_length,
            original_length, 0, 0, 0, None, None, None, None, None,
            0.0, _rms(right), 0.0, 1.0, None, None, None, None,
            0.0, 0.0, 0.0, True)

    left_period, left_periodicity = _estimate_period(
        left, sample_rate, expected_f0_hz, config)
    right_period, right_periodicity = _start_period(
        right, sample_rate, expected_f0_hz, config)
    hinted_aperiodic = bool(
        _phone_hint_is_aperiodic(left_phone) or
        _phone_hint_is_aperiodic(right_phone) or
        _phone_hint_allows_silence(left_phone) or
        _phone_hint_allows_silence(right_phone))
    if hinted_aperiodic:
        left_period = None
        right_period = None
    voiced = bool(left_period and right_period and
                  min(left_periodicity, right_periodicity) >=
                  config.voiced_correlation_minimum)
    voicing_hint_reason = (
        "phone-context-forced-aperiodic" if hinted_aperiodic else
        "periodicity-supported" if voiced else
        "periodicity-insufficient")
    if voiced:
        common_period = int(round((left_period + right_period) * 0.5))
        requested_ms = max(
            config.minimum_overlap_ms,
            min(config.maximum_overlap_ms, config.crossover_ms))
        requested_samples = int(round(
            sample_rate * requested_ms / 1000.0))
        # Choose a perceptual time span first, then snap inward to complete
        # periods. Period count is an outcome and rises naturally with F0.
        crossover_period_count = max(
            1, requested_samples // max(1, common_period))
        overlap = common_period * crossover_period_count
    else:
        requested_ms = float(config.unvoiced_overlap_ms)
        crossover_period_count = None
        overlap = int(round(sample_rate * config.unvoiced_overlap_ms / 1000.0))
    minimum = int(round(sample_rate * config.minimum_overlap_ms / 1000.0))
    maximum = int(round(sample_rate * config.maximum_overlap_ms / 1000.0))
    overlap = min(
        len(left), len(right),
        max(2, minimum, min(maximum, overlap)),
    )

    indexed_length: Optional[int] = None
    right_skip_limit: Optional[int] = None
    if right_indexed_length_samples is not None:
        indexed_length = max(
            0, min(len(right), int(right_indexed_length_samples)))
        # The Festival midpoint lies halfway through the overlap.  Reserve
        # the requested indexed phone tail beyond that point before searching
        # for a phase-aligned right cut, otherwise a valid source can be
        # consumed by the local search and disappear from the generated DB.
        required_tail = max(0, int(minimum_right_indexed_tail_samples))
        maximum_tail_safe_overlap = 2 * max(
            0, indexed_length - required_tail)
        if maximum_tail_safe_overlap < minimum:
            maximum_right_skip = (
                indexed_length - (minimum // 2) - required_tail)
            raise JoinConstraintError(
                "right_indexed_region_too_short",
                indexed_length_samples=indexed_length,
                overlap_samples=minimum,
                required_tail_samples=required_tail,
                maximum_right_skip_samples=maximum_right_skip,
                sample_rate=sample_rate,
            )
        if overlap > maximum_tail_safe_overlap:
            if voiced:
                common_period = int(round(
                    (left_period + right_period) * 0.5))
                safe_period_count = (
                    maximum_tail_safe_overlap // max(1, common_period))
                if safe_period_count <= 0:
                    maximum_right_skip = (
                        indexed_length - (minimum // 2) - required_tail)
                    raise JoinConstraintError(
                        "right_indexed_region_too_short",
                        indexed_length_samples=indexed_length,
                        overlap_samples=minimum,
                        required_tail_samples=required_tail,
                        maximum_right_skip_samples=maximum_right_skip,
                        sample_rate=sample_rate,
                    )
                crossover_period_count = safe_period_count
                overlap = common_period * safe_period_count
            else:
                overlap = maximum_tail_safe_overlap
        right_skip_limit = indexed_length - (overlap // 2) - required_tail
        if right_skip_limit < 0:
            raise JoinConstraintError(
                "right_indexed_region_too_short",
                indexed_length_samples=indexed_length,
                overlap_samples=overlap,
                required_tail_samples=required_tail,
                maximum_right_skip_samples=right_skip_limit,
                sample_rate=sample_rate,
            )

    original_left_length = len(left)
    left_start, right_start, join_cost, metrics = _best_local_join(
        left, right, overlap, sample_rate, left_period, right_period, config,
        allow_silent_left=bool(allow_silent_left),
        allow_silent_right=bool(allow_silent_right),
        enforce_acoustic_similarity=bool(enforce_acoustic_similarity),
        right_skip_limit=right_skip_limit)
    outgoing = tuple(left[left_start:left_start + overlap])
    incoming = tuple(right[right_start:right_start + overlap])
    left_rms = float(metrics["left_rms"])
    right_rms = float(metrics["right_rms"])
    gain = (left_rms / right_rms if right_rms > _EPSILON else 1.0)
    gain = max(config.gain_minimum, min(config.gain_maximum, gain))

    zero_correlation = (metrics.get("period_zero_lag_correlation")
                        if voiced else None)
    best_correlation = (metrics.get("period_best_lag_correlation")
                        if voiced else None)
    phase_offset = (metrics.get("period_best_lag_samples")
                    if voiced else None)
    period_shape_mismatch = (metrics.get("period_shape_mismatch")
                             if voiced else None)
    phase_cycles = (
        float(phase_offset) / float(max(1, int(round(
            (left_period + right_period) * 0.5))))
        if voiced and phase_offset is not None else None)

    mixed = _crossfade(outgoing, incoming, gain)
    content_gate_active = bool(metrics.get("content_gate_active", 0.0))
    source_reference_rms = float(metrics.get("source_reference_rms", 0.0))
    source_retention = float(metrics.get("source_energy_retention", 1.0))
    crossfade_retention = float(
        metrics.get("crossfade_energy_retention", 1.0))
    phase_mix_retention = float(
        metrics.get("phase_mix_energy_retention", crossfade_retention))
    local_mix_retention = float(
        metrics.get("local_crossfade_energy_retention", 1.0))
    left_source_retention = float(
        metrics.get("left_source_energy_retention", source_retention))
    right_source_retention = float(
        metrics.get("right_source_energy_retention", source_retention))
    content_preservation_passed = bool(
        not content_gate_active or (
            source_retention >= config.minimum_source_energy_retention and
            phase_mix_retention >=
            config.minimum_crossfade_energy_retention and
            local_mix_retention >=
            config.minimum_local_crossfade_energy_retention and
            not bool(metrics.get("left_missing_source_content", 0.0)) and
            not bool(metrics.get("right_missing_source_content", 0.0))
        )
    )
    content_attenuation_db = max(0.0, -_db_ratio(
        min(source_retention, crossfade_retention, local_mix_retention), 1.0))
    content_fallback_used = not content_preservation_passed
    acoustic_validation_passed = bool(
        metrics.get("acoustic_validation_passed", True))
    legacy_fallback_used = bool(
        content_fallback_used or not acoustic_validation_passed)
    if legacy_fallback_used:
        # Never ship a candidate that passed the aggregate cost by erasing
        # content or violating an explicit acoustic gate.  Return to the
        # annotated cuts before applying the conservative pre-fix overlap, so
        # a failed repair cannot alter contextual units or their timing.
        left_start = max(0, original_left_length - overlap)
        right_start = 0
        outgoing = tuple(left[left_start:left_start + overlap])
        incoming = tuple(right[:overlap])
        fallback = array.array("h", outgoing)
        legacy_linear_join_pcm16(fallback, incoming, overlap)
        mixed = fallback
    preview_prefix = tuple(left[max(0, left_start - 128):left_start])
    right_after = right_start + overlap
    preview_suffix = tuple(right[right_after:right_after + 128])
    preview = preview_prefix + tuple(mixed) + preview_suffix
    first_boundary = len(preview_prefix)
    second_boundary = first_boundary + len(mixed)
    impulse_novelty = _impulse_novelty(
        preview, (first_boundary, second_boundary))

    validation_failures = list(
        metrics.get("validation_failures", ()))
    if bool(metrics.get("left_missing_source_content", 0.0)):
        validation_failures.append("MISSING_LEFT_SOURCE_CONTENT")
    if bool(metrics.get("right_missing_source_content", 0.0)):
        validation_failures.append("MISSING_RIGHT_SOURCE_CONTENT")
    if not content_preservation_passed:
        validation_failures.append("CONTENT_RETENTION")
    if impulse_novelty > config.validation_novelty_limit:
        validation_failures.append("IMPULSE_NOVELTY")

    del left[left_start:]
    left.extend(mixed)
    left.extend(right[right_after:])

    left_f0 = sample_rate / left_period if left_period else None
    right_f0 = sample_rate / right_period if right_period else None
    f0_step = (12.0 * math.log2(right_f0 / left_f0)
               if left_f0 and right_f0 else None)
    handoff_start = left_start
    handoff_end = left_start + overlap
    return JoinSynthesisDecision(
        method=("legacy-linear-content-fallback" if content_fallback_used
                else "legacy-linear-validation-fallback"
                if legacy_fallback_used
                else "pitch-synchronous-raised-cosine" if voiced else
                "unvoiced-raised-cosine"),
        voiced=voiced,
        splice_sample=handoff_start + overlap // 2,
        handoff_start_sample=handoff_start,
        handoff_end_sample=handoff_end,
        overlap_samples=overlap,
        left_trim_samples=max(
            0, original_left_length - (left_start + overlap)),
        right_skip_samples=right_start,
        left_period_samples=left_period,
        right_period_samples=right_period,
        left_f0_hz=left_f0,
        right_f0_hz=right_f0,
        f0_step_semitones=f0_step,
        left_rms=left_rms,
        right_rms=right_rms,
        level_step_db=_db_ratio(right_rms, left_rms),
        gain_ratio=gain,
        zero_lag_correlation=(float(zero_correlation)
                              if zero_correlation is not None else None),
        best_lag_correlation=(float(best_correlation)
                              if best_correlation is not None else None),
        phase_offset_samples=(int(phase_offset)
                              if phase_offset is not None else None),
        phase_offset_cycles=phase_cycles,
        spectral_distance=float(metrics["spectral"]),
        join_cost=join_cost,
        impulse_novelty=impulse_novelty,
        validation_passed=not validation_failures,
        content_gate_active=content_gate_active,
        source_reference_rms=source_reference_rms,
        source_energy_retention=source_retention,
        crossfade_energy_retention=crossfade_retention,
        content_attenuation_db=content_attenuation_db,
        content_preservation_passed=content_preservation_passed,
        phase_mix_energy_retention=phase_mix_retention,
        local_crossfade_energy_retention=local_mix_retention,
        left_source_energy_retention=left_source_retention,
        right_source_energy_retention=right_source_retention,
        left_silence_allowed=bool(allow_silent_left),
        right_silence_allowed=bool(allow_silent_right),
        content_fallback_used=content_fallback_used,
        acoustic_validation_passed=acoustic_validation_passed,
        acoustic_validation_gate_active=bool(
            metrics.get("acoustic_validation_gate_active", False)),
        level_validation_passed=bool(
            metrics.get("level_validation_passed", True)),
        f0_validation_passed=bool(
            metrics.get("f0_validation_passed", True)),
        period_correlation_validation_passed=bool(
            metrics.get("period_correlation_validation_passed", True)),
        period_shape_validation_passed=bool(
            metrics.get("period_shape_validation_passed", True)),
        spectral_validation_passed=bool(
            metrics.get("spectral_validation_passed", True)),
        period_shape_mismatch=(float(period_shape_mismatch)
                               if period_shape_mismatch is not None else None),
        validation_failures=tuple(validation_failures),
        legacy_fallback_used=legacy_fallback_used,
        voicing_hint_reason=voicing_hint_reason,
        left_source_content_present=not bool(
            metrics.get("left_missing_source_content", 0.0)),
        right_source_content_present=not bool(
            metrics.get("right_missing_source_content", 0.0)),
        right_indexed_length_samples=(
            indexed_length),
        right_skip_limit_samples=right_skip_limit,
        right_indexed_tail_samples=(
            indexed_length - right_start - overlap // 2
            if indexed_length is not None else None),
        right_skip_limit_applied=right_skip_limit is not None,
        requested_crossover_ms=float(requested_ms),
        effective_crossover_ms=(
            1000.0 * overlap / float(sample_rate)),
        crossover_period_count=crossover_period_count,
    )


def legacy_linear_join_pcm16(left: array.array, right: Sequence[int],
                             samples: int) -> array.array:
    """Exact pre-fix standalone join, retained for Fault Mode comparison."""
    count = min(max(0, int(samples)), len(left), len(right))
    if count <= 0:
        left.extend(right)
        return left
    for index in range(count):
        progress = index / count
        left[len(left) - count + index] = int(
            left[len(left) - count + index] * (1.0 - progress) +
            int(right[index]) * progress)
    left.extend(right[count:])
    return left
