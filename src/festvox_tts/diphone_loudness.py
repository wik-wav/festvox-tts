"""Time-resolved loudness and unit-join diagnostics for rendered speech.

The broadcast loudness standards use a 400 ms momentary window.  That curve
is useful for the overall utterance but too slow to reveal a diphone handoff,
so this module also exposes a 20 ms K-weighted join curve.  Both curves share
the ITU-R BS.1770 K-weighting filter; only the diagnostic window differs.

The analyzer never changes audio.  It describes the target phone regions,
the center-to-center diphone spans mapped over them, and the measured step at
each phone-center join.  Returned values are plain JSON-compatible objects so
the same implementation can drive tests, reports, and the GUI debug dialog.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

try:  # Optional acceleration for long diagnostic renders.
    from scipy.signal import lfilter as _scipy_lfilter
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment
    _scipy_lfilter = None


LOUDNESS_DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_JOIN_WINDOW_MS = 20.0
DEFAULT_JOIN_HOP_MS = 2.5
DEFAULT_JOIN_GUARD_MS = 3.0
DEFAULT_JOIN_SPAN_MS = 20.0
DEFAULT_FLAG_STEP_LU = 2.5
DEFAULT_MIN_AUDIBLE_LKFS = -55.0
_SILENCE_PHONES = frozenset({"pau", "sil", "sp"})


def _mono(samples: object) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.mean(values, axis=1, dtype=np.float64)
    raise ValueError("audio samples must be mono or frame-by-channel")


def _biquad_coefficients(kind: str, sample_rate: int) -> tuple[np.ndarray,
                                                                    np.ndarray]:
    """Return the two BS.1770 K-weighting stages at any practical rate.

    The constants are the analogue parameters that yield the published
    48 kHz coefficients after the bilinear transform.  This keeps the debug
    curve stable for the 16, 22.05, 44.1 and 48 kHz files used by the project.
    """
    if sample_rate < 8000:
        raise ValueError("loudness analysis requires at least 8 kHz audio")
    if kind == "shelf":
        frequency = 1681.974450955533
        q = 0.7071752369554196
        gain_db = 3.999843853973347
        k = math.tan(math.pi * frequency / float(sample_rate))
        vh = 10.0 ** (gain_db / 20.0)
        vb = vh ** 0.4996667741545416
        a0 = 1.0 + k / q + k * k
        b = np.asarray((
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ), dtype=np.float64)
        a = np.asarray((
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / q + k * k) / a0,
        ), dtype=np.float64)
        return b, a
    if kind == "highpass":
        frequency = 38.13547087602444
        q = 0.5003270373238773
        k = math.tan(math.pi * frequency / float(sample_rate))
        a0 = 1.0 + k / q + k * k
        b = np.asarray((1.0 / a0, -2.0 / a0, 1.0 / a0),
                       dtype=np.float64)
        a = np.asarray((
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / q + k * k) / a0,
        ), dtype=np.float64)
        return b, a
    raise ValueError(f"unknown K-weighting stage: {kind}")


def _filter_biquad(samples: np.ndarray, b: np.ndarray,
                   a: np.ndarray) -> np.ndarray:
    if _scipy_lfilter is not None:
        return np.asarray(
            _scipy_lfilter(b, a, np.asarray(samples, dtype=np.float64)),
            dtype=np.float64)
    result = np.empty(len(samples), dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = (float(value) for value in b)
    _a0, a1, a2 = (float(value) for value in a)
    for index, value in enumerate(np.asarray(samples, dtype=np.float64)):
        current = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        result[index] = current
        x2, x1 = x1, float(value)
        y2, y1 = y1, current
    return result


def k_weight(samples: object, sample_rate: int) -> np.ndarray:
    """Return a mono signal filtered by the BS.1770 K-weighting stages."""
    values = _mono(samples)
    if not len(values):
        return values.copy()
    for stage in ("shelf", "highpass"):
        b, a = _biquad_coefficients(stage, int(sample_rate))
        values = _filter_biquad(values, b, a)
    return values


def loudness_curve(samples: object, sample_rate: int, *,
                   window_ms: float = DEFAULT_JOIN_WINDOW_MS,
                   hop_ms: float = DEFAULT_JOIN_HOP_MS) -> dict[str, object]:
    """Measure a centered K-weighted loudness curve.

    Values use the BS.1770 channel-energy calibration and are reported as
    LKFS-like diagnostic levels.  A 400 ms window is standard momentary
    loudness; the default 20 ms window is intentionally join-local and is not
    presented as an EBU Mode meter.
    """
    rate = int(sample_rate)
    if rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    if not 5.0 <= float(window_ms) <= 4000.0:
        raise ValueError("loudness window must be between 5 and 4000 ms")
    if not 0.5 <= float(hop_ms) <= float(window_ms):
        raise ValueError("loudness hop must be between 0.5 ms and the window")
    weighted = k_weight(samples, rate)
    return _loudness_curve_from_weighted(
        weighted, rate, window_ms=float(window_ms), hop_ms=float(hop_ms)
    )


def _loudness_curve_from_weighted(
    weighted: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float,
    hop_ms: float,
) -> dict[str, object]:
    """Build a curve from audio already passed through K-weighting."""
    rate = int(sample_rate)
    if not len(weighted):
        return {
            "window_ms": float(window_ms), "hop_ms": float(hop_ms),
            "times": [], "levels_lkfs": [],
        }
    window = max(1, int(round(rate * float(window_ms) / 1000.0)))
    hop = max(1, int(round(rate * float(hop_ms) / 1000.0)))
    half = window // 2
    squared = weighted * weighted
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(squared)))
    centers = np.arange(0, len(weighted), hop, dtype=np.int64)
    starts = np.maximum(0, centers - half)
    ends = np.minimum(len(weighted), starts + window)
    starts = np.maximum(0, ends - window)
    counts = np.maximum(1, ends - starts)
    energies = (cumulative[ends] - cumulative[starts]) / counts
    levels = -0.691 + 10.0 * np.log10(np.maximum(energies, 1e-18))
    return {
        "window_ms": round(float(window_ms), 6),
        "hop_ms": round(float(hop_ms), 6),
        "times": [round(float(value) / rate, 6) for value in centers],
        "levels_lkfs": [round(float(value), 6) for value in levels],
    }


def _interval_level_lkfs(weighted: np.ndarray, sample_rate: int,
                         first_s: float, last_s: float) -> float | None:
    first = max(0, int(math.floor(float(first_s) * sample_rate)))
    last = min(len(weighted), int(math.ceil(float(last_s) * sample_rate)))
    if last - first < max(8, int(round(0.005 * sample_rate))):
        return None
    energy = float(np.mean(np.square(weighted[first:last])))
    if not math.isfinite(energy) or energy <= 1e-18:
        return -180.691
    return -0.691 + 10.0 * math.log10(energy)


def _segment_value(segment: object, key: str, default: object = None) -> object:
    if isinstance(segment, Mapping):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _segments(segments: Sequence[object]) -> list[dict[str, object]]:
    result = []
    previous = 0.0
    for index, segment in enumerate(segments):
        phone = str(_segment_value(segment, "phone", "") or "")
        start = float(_segment_value(segment, "start", previous) or 0.0)
        end = float(_segment_value(segment, "end", start) or start)
        if end < start:
            raise ValueError("segment end precedes its start")
        result.append({
            "index": index, "phone": phone,
            "start": round(start, 6), "end": round(end, 6),
            "center": round((start + end) * 0.5, 6),
        })
        previous = end
    return result


def _choice_for(pair: str, unit_name: str,
                alternatives: Mapping[str, Sequence[Mapping[str, object]]]
                ) -> dict[str, object]:
    choices = list(alternatives.get(pair) or ())
    if not choices:
        return {}
    selected = next((
        choice for choice in choices
        if str(choice.get("left_name") or pair.split("-", 1)[0]) == unit_name
    ), choices[0])
    return dict(selected)


def _fit_join_level(times: np.ndarray, levels: np.ndarray, join_time: float,
                    side: str, guard_s: float, span_s: float) -> float | None:
    if side == "left":
        mask = ((times >= join_time - span_s) &
                (times <= join_time - guard_s))
    else:
        mask = ((times >= join_time + guard_s) &
                (times <= join_time + span_s))
    local_x = times[mask] - join_time
    local_y = levels[mask]
    finite = np.isfinite(local_y)
    local_x = local_x[finite]
    local_y = local_y[finite]
    if len(local_y) < 2:
        return None
    slope, intercept = np.polyfit(local_x, local_y, 1)
    _ = slope
    return float(intercept)


def _analyze_rendered_join_levels_legacy(
    samples: object,
    sample_rate: int,
    segments: Sequence[object],
    *,
    selected_units: Mapping[int, str] | None = None,
    alternatives: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    window_ms: float = DEFAULT_JOIN_WINDOW_MS,
    hop_ms: float = DEFAULT_JOIN_HOP_MS,
    flag_step_lu: float = DEFAULT_FLAG_STEP_LU,
    minimum_audible_lkfs: float = DEFAULT_MIN_AUDIBLE_LKFS,
) -> dict[str, object]:
    """Describe rendered phones, diphone spans, and loudness discontinuities."""
    rows = _segments(segments)
    selected = {int(key): str(value) for key, value in
                dict(selected_units or {}).items()}
    inventory = dict(alternatives or {})
    mono = _mono(samples)
    weighted = k_weight(mono, int(sample_rate))
    join_curve = _loudness_curve_from_weighted(
        weighted, int(sample_rate), window_ms=float(window_ms),
        hop_ms=float(hop_ms))
    momentary = _loudness_curve_from_weighted(
        weighted, int(sample_rate), window_ms=400.0, hop_ms=100.0)

    units = []
    choices_by_edge: dict[int, dict[str, object]] = {}
    for index in range(max(0, len(rows) - 1)):
        left = str(rows[index]["phone"])
        right = str(rows[index + 1]["phone"])
        pair = f"{left}-{right}"
        unit_name = selected.get(index, left)
        choice = _choice_for(pair, unit_name, inventory)
        choices_by_edge[index] = choice
        units.append({
            "index": index,
            "pair": pair,
            "start": rows[index]["center"],
            "end": rows[index + 1]["center"],
            "selected_unit": unit_name,
            "alias": str(choice.get("alias") or ""),
            "wav": str(choice.get("wav") or choice.get("wav_name") or ""),
            "oto_timing_ms": dict(choice.get("oto_timing_ms") or {}),
            "join_conditioning": dict(choice.get("join_conditioning") or {}),
        })

    joins = []
    guard_s = DEFAULT_JOIN_GUARD_MS / 1000.0
    span_s = max(float(window_ms), DEFAULT_JOIN_SPAN_MS) / 1000.0
    for index in range(1, max(1, len(rows) - 1)):
        phone = str(rows[index]["phone"])
        if phone in _SILENCE_PHONES:
            continue
        join_time = float(rows[index]["center"])
        before = _interval_level_lkfs(
            weighted, int(sample_rate), join_time - span_s,
            join_time - guard_s)
        after = _interval_level_lkfs(
            weighted, int(sample_rate), join_time + guard_s,
            join_time + span_s)
        if before is None or after is None:
            continue
        step = after - before
        incoming = choices_by_edge.get(index - 1, {})
        outgoing = choices_by_edge.get(index, {})
        incoming_conditioning = dict(
            incoming.get("join_conditioning") or {})
        outgoing_conditioning = dict(
            outgoing.get("join_conditioning") or {})
        timing = dict(outgoing.get("oto_timing_ms") or {})
        try:
            declared_ms = float(timing.get("overlap") or 0.0)
            left_collar_ms = float(
                incoming_conditioning.get("effective_end_collar_ms")
                or declared_ms)
            right_collar_ms = float(
                outgoing_conditioning.get("effective_start_collar_ms")
                or declared_ms)
        except (TypeError, ValueError):
            declared_ms = left_collar_ms = right_collar_ms = 0.0
        audible = max(before, after) >= float(minimum_audible_lkfs)
        flagged = audible and abs(step) >= float(flag_step_lu)
        joins.append({
            "segment_index": index,
            "phone": phone,
            "time": round(join_time, 6),
            "incoming_pair": units[index - 1]["pair"],
            "outgoing_pair": units[index]["pair"],
            "incoming_unit": units[index - 1]["selected_unit"],
            "outgoing_unit": units[index]["selected_unit"],
            "incoming_wav": units[index - 1]["wav"],
            "outgoing_wav": units[index]["wav"],
            "before_lkfs": round(before, 4),
            "after_lkfs": round(after, 4),
            "step_lu": round(step, 4),
            "absolute_step_lu": round(abs(step), 4),
            "audible": bool(audible),
            "flagged": bool(flagged),
            "declared_oto_overlap_ms": round(max(0.0, declared_ms), 4),
            "incoming_collar_ms": round(max(0.0, left_collar_ms), 4),
            "outgoing_collar_ms": round(max(0.0, right_collar_ms), 4),
            "overlap_start": round(
                join_time - max(0.0, left_collar_ms) / 1000.0, 6),
            "overlap_end": round(
                join_time + max(0.0, right_collar_ms) / 1000.0, 6),
        })

    audible_steps = [float(row["absolute_step_lu"]) for row in joins
                     if row["audible"]]
    flagged = [row for row in joins if row["flagged"]]
    return {
        "schema_version": LOUDNESS_DIAGNOSTIC_SCHEMA_VERSION,
        "method": "bs1770-k-weighted-phone-center-joins-v1",
        "sample_rate": int(sample_rate),
        "duration": round(len(mono) / float(sample_rate), 6),
        "join_curve": join_curve,
        "momentary_curve": momentary,
        "segments": rows,
        "units": units,
        "joins": joins,
        "summary": {
            "join_count": len(joins),
            "audible_join_count": len(audible_steps),
            "flagged_join_count": len(flagged),
            "flag_threshold_lu": round(float(flag_step_lu), 4),
            "median_audible_step_lu": round(
                float(np.median(audible_steps)) if audible_steps else 0.0, 4),
            "maximum_audible_step_lu": round(
                max(audible_steps, default=0.0), 4),
        },
    }


def analyze_rendered_joins(
    samples: object,
    sample_rate: int,
    segments: Sequence[object],
    **kwargs,
) -> dict[str, object]:
    """Backward-compatible entry point for the full discontinuity analyzer."""
    from join_discontinuity import JoinDiscontinuityAnalyzer

    return JoinDiscontinuityAnalyzer(
        samples, sample_rate, segments, **kwargs
    ).analyze()
