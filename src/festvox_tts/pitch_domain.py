"""Shared log-F0 and semitone operations for multilingual prosody.

All linguistic pitch arithmetic belongs here.  Festival, PSOLA, displays, and
saved legacy targets still exchange hertz, but interpolation, offsets,
recentring, and blends are performed in log frequency.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


MINIMUM_F0_HZ = 1e-3


def hz_to_log_f0(hz: float) -> float:
    value = float(hz)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("F0 must be finite and positive")
    return math.log2(value)


def log_f0_to_hz(log_f0: float) -> float:
    value = float(log_f0)
    if not math.isfinite(value):
        raise ValueError("log F0 must be finite")
    return 2.0 ** value


def hz_to_semitone_number(hz: float) -> float:
    return 12.0 * hz_to_log_f0(hz)


def semitone_number_to_hz(semitones: float) -> float:
    return log_f0_to_hz(float(semitones) / 12.0)


def semitone_offset(base_hz: float, semitones: float) -> float:
    return log_f0_to_hz(
        hz_to_log_f0(base_hz) + float(semitones) / 12.0
    )


def semitone_difference(left_hz: float, right_hz: float) -> float:
    """Return right minus left in semitones."""
    return 12.0 * (hz_to_log_f0(right_hz) - hz_to_log_f0(left_hz))


def clamp_hz(hz: float, minimum_hz: float, maximum_hz: float) -> float:
    lower = float(minimum_hz)
    upper = float(maximum_hz)
    if lower <= 0.0 or upper < lower:
        raise ValueError("invalid F0 bounds")
    return max(lower, min(upper, float(hz)))


def blend_hz_log(left_hz: float, right_hz: float, mix: float) -> float:
    amount = max(0.0, min(1.0, float(mix)))
    value = (
        hz_to_log_f0(left_hz) * (1.0 - amount)
        + hz_to_log_f0(right_hz) * amount
    )
    return log_f0_to_hz(value)


def interpolate_hz_log(
    points: Sequence[Sequence[float]],
    when: float,
    default_hz: float = 160.0,
) -> float:
    rows = sorted((float(row[0]), float(row[1])) for row in (points or ()))
    if not rows:
        return float(default_hz)
    position = float(when)
    if position <= rows[0][0]:
        return rows[0][1]
    if position >= rows[-1][0]:
        return rows[-1][1]
    for (left_time, left_hz), (right_time, right_hz) in zip(rows, rows[1:]):
        if left_time <= position <= right_time:
            span = max(1e-12, right_time - left_time)
            return blend_hz_log(
                left_hz, right_hz, (position - left_time) / span
            )
    return rows[-1][1]


def geometric_mean_hz(values: Iterable[float]) -> float:
    logs = [hz_to_log_f0(value) for value in values if float(value) > 0.0]
    if not logs:
        raise ValueError("at least one positive F0 is required")
    return log_f0_to_hz(sum(logs) / len(logs))


def recenter_targets_log(
    targets: Sequence[Sequence[float]],
    center_hz: float,
    minimum_hz: float = 50.0,
    maximum_hz: float = 500.0,
) -> list[tuple[float, float]]:
    rows = [(float(time), float(hz)) for time, hz in (targets or ())]
    if not rows:
        return []
    current = geometric_mean_hz(hz for _time, hz in rows)
    shift = semitone_difference(current, center_hz)
    return [
        (time, clamp_hz(semitone_offset(hz, shift), minimum_hz, maximum_hz))
        for time, hz in rows
    ]


def fall_percent_to_span_semitones(fall_percent: float) -> float:
    """Map the legacy 0..40 Fall control to a symmetric log-F0 half-span."""
    fraction = max(0.0, min(40.0, float(fall_percent or 0.0))) / 200.0
    return 12.0 * math.log2(1.0 + fraction)
