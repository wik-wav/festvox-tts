"""Rendered-waveform validation for the vocal-tract parameter.

This diagnostic deliberately measures the output waveform, not the internal
target envelope. It writes comparable WAV files, structured JSON, and a Qt
PNG containing spectrograms, tracked formants, and actual-versus-requested
spectral envelopes. The input recording is opened read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
import wave

import numpy as np

from formant_analysis import (
    AnalysisSegment,
    AudioData,
    FormantSegment,
    analyze_segment,
    read_audio,
)
from join_spectrogram import (
    SPECTROGRAM_FLOOR_DB,
    _load_diagnostic_font,
    _spectrogram_rgb,
    spectrogram_db,
)
from vocal_tract import (
    load_vocal_tract_range,
    ratio_to_formant_multiplier,
    transform_vocal_tract,
)


# Stable, voiced bodies from the read-only project-speaker bank.  The
# intervals deliberately avoid OTO transitions and permit the exact same
# five-vowel audit to be reproduced without committing source recordings.
SOURCE_VOWEL_FIXTURES = {
    "a": ("_a_a_a_a_a_a.wav", 0.853, 4.530),
    "i": ("_i_i_i_i_i_i.wav", 0.853, 1.7380045351473923),
    "u": ("_u_u_u_u_u_u.wav", 0.853, 3.1270068027210884),
    "e": ("_le_dxye_pye_hhye_bye_kye.wav", 3.1595,
          3.5292743764172334),
    "o": ("_va_vi_vyu_vye_vyo_vi.wav", 3.22556,
          3.6020408163265305),
}


@dataclass(frozen=True)
class VocalTractValidationPoint:
    ratio: float
    requested_formant_multiplier: float
    measured_formants_hz: tuple[float | None, ...]
    independent_formants_hz: tuple[float | None, ...]
    measured_formant_multipliers: tuple[float | None, ...]
    formant_error_hz: tuple[float | None, ...]
    formant_ratio_error: tuple[float | None, ...]
    formant_error_cents: tuple[float | None, ...]
    median_absolute_formant_error_hz: float | None
    median_absolute_formant_ratio_error: float | None
    median_absolute_formant_error_cents: float | None
    formant_measurement_method: str
    f0_hz: float | None
    f0_drift_semitones: float | None
    duration_samples: int
    duration_drift_samples: int
    envelope_target_rmse_db: float
    envelope_change_rms_db: float
    accepted_formant_frames: int
    rejected_formant_frames: int
    peak: float
    clipped_sample_count: int
    real_time_factor: float
    wav_file: str


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, np.float64).reshape(-1)
    pcm = np.asarray(
        np.rint(np.clip(values, -1.0, 1.0) * 32767.0), np.int16
    )
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(int(sample_rate))
        target.writeframes(pcm.tobytes())


def average_cepstral_envelope_db(
    samples: Sequence[float],
    sample_rate: int,
    *,
    fft_size: int = 4096,
    hop_size: int = 1024,
    quefrency_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable average log-spectrum envelope for visual comparison."""
    values = np.asarray(samples, np.float64).reshape(-1)
    if not values.size:
        raise ValueError("audio samples are empty")
    if values.size < fft_size:
        values = np.pad(values, (0, fft_size - values.size))
    window = np.hanning(fft_size)
    rows: list[np.ndarray] = []
    frame_rms: list[float] = []
    for start in range(0, values.size - fft_size + 1, hop_size):
        frame = values[start:start + fft_size]
        frame_rms.append(float(np.sqrt(np.mean(frame * frame))))
        rows.append(20.0 * np.log10(np.maximum(
            np.abs(np.fft.rfft(frame * window)), 1.0e-10
        )))
    levels = np.asarray(frame_rms, np.float64)
    threshold = max(1.0e-7, float(np.max(levels, initial=0.0)) * 0.08)
    accepted = [row for row, level in zip(rows, levels) if level >= threshold]
    if not accepted:
        accepted = rows
    mean_log_spectrum = np.mean(np.asarray(accepted), axis=0)
    full_length = (mean_log_spectrum.size - 1) * 2
    # Keep the smoothing time constant stable across sample rates. A fixed
    # number of cepstral bins accidentally followed individual harmonics at
    # lower rates and made the validation disagree with its own spectrogram.
    requested_order = (
        int(quefrency_order) if quefrency_order is not None
        else int(round(sample_rate * 0.0023))
    )
    order = max(4, min(requested_order, full_length // 4))
    cepstrum = np.fft.irfft(mean_log_spectrum, full_length)
    lifter = np.zeros(full_length, np.float64)
    lifter[:order + 1] = 1.0
    lifter[-order:] = 1.0
    envelope = np.fft.rfft(cepstrum * lifter, full_length).real
    frequencies = np.fft.rfftfreq(full_length, 1.0 / sample_rate)
    return frequencies, np.asarray(envelope, np.float64)


def _target_envelope(
    frequencies: np.ndarray,
    identity_envelope: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    queries = frequencies * float(ratio)
    target = np.interp(
        np.minimum(queries, frequencies[-1]),
        frequencies,
        identity_envelope,
    )
    valid = (
        (frequencies >= 150.0)
        & (frequencies <= 6000.0)
        & (queries <= frequencies[-1])
    )
    return target, valid


def _shape_rmse(first: np.ndarray, second: np.ndarray,
                mask: np.ndarray) -> float:
    difference = np.asarray(first[mask] - second[mask], np.float64)
    if not difference.size:
        return 0.0
    difference -= float(np.median(difference))
    return float(np.sqrt(np.mean(difference * difference)))


def identity_anchored_formants(
    frequencies: Sequence[float],
    envelope_db: Sequence[float],
    identity_formants_hz: Sequence[float | None],
    requested_multiplier: float,
    *,
    maximum_hz: float = 6000.0,
) -> tuple[float | None, ...]:
    """Pair final-envelope peaks with the source speaker's F1-F4 identities.

    An ordinary independent tracker can relabel a strongly shifted F2 as F1,
    especially for /i/ and /u/.  Validation has stronger information: each
    transformed WAV has an identity rendering from the same samples and a
    requested uniform frequency multiplier.  We therefore search only the
    non-overlapping geometric neighborhood of each expected resonance.  The
    selected values still come from the final rendered envelope; the expected
    locations merely establish correspondence and cannot manufacture a peak.
    Independent frame tracks remain in the report and plot as a cross-check.
    """
    x = np.asarray(frequencies, np.float64).reshape(-1)
    y = np.asarray(envelope_db, np.float64).reshape(-1)
    if x.size != y.size or x.size < 5:
        return tuple(None for _ in identity_formants_hz)
    multiplier = float(requested_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("requested formant multiplier must be positive")
    if abs(multiplier - 1.0) <= 1.0e-12:
        return tuple(
            float(value) if value is not None else None
            for value in identity_formants_hz
        )

    expected = [
        float(value) * multiplier if value is not None else None
        for value in identity_formants_hz
    ]
    frequency_step = max(1.0e-9, float(np.median(np.diff(x))))
    trend_half = max(2, int(round(600.0 / frequency_step)))
    trend_kernel = np.ones(trend_half * 2 + 1, np.float64)
    trend_kernel /= float(trend_kernel.size)
    padded = np.pad(y, (trend_half, trend_half), mode="reflect")
    residual = y - np.convolve(padded, trend_kernel, mode="valid")
    result: list[float | None] = []
    for index, center in enumerate(expected):
        if center is None or center < 100.0 or center > maximum_hz * 1.08:
            result.append(None)
            continue
        lower = max(120.0, center * 0.60)
        upper = min(float(maximum_hz), center * 1.45)
        if index > 0 and expected[index - 1] is not None:
            lower = max(lower, math.sqrt(expected[index - 1] * center))
        if index + 1 < len(expected) and expected[index + 1] is not None:
            upper = min(upper, math.sqrt(center * expected[index + 1]))
        band = np.flatnonzero((x >= lower) & (x <= upper))
        if band.size < 3:
            result.append(None)
            continue
        local = band[
            (residual[band] >= residual[np.maximum(0, band - 1)])
            & (residual[band] > residual[np.minimum(residual.size - 1,
                                                    band + 1)])
        ]
        if not local.size:
            local = band
        distance_scale = max(120.0, center * 0.18)
        scores = (
            residual[local]
            - 2.5 * np.abs(x[local] - center) / distance_scale
        )
        result.append(float(x[local[int(np.argmax(scores))]]))
    return tuple(result)


def _analyze_output(
    samples: np.ndarray,
    sample_rate: int,
    source_path: Path,
    ratio: float,
    vowel: str,
) -> FormantSegment:
    duration = samples.size / float(sample_rate)
    edge = min(0.08, duration * 0.12)
    segment = AnalysisSegment(
        segment_id=f"vocal-tract-validation:{ratio:.5f}",
        speaker_id="rendered_validation",
        audio_path=source_path,
        start_seconds=edge,
        end_seconds=max(edge + 0.02, duration - edge),
        vowel=vowel,
        phone=vowel,
        source_corpus="rendered_vocal_tract_validation",
    )
    return analyze_segment(
        segment,
        audio=AudioData(
            samples=np.asarray(samples, np.float64),
            sample_rate=int(sample_rate),
            channels=1,
            source_path=source_path,
        ),
    )


def _safe_log_ratio(value: float | None, reference: float | None,
                    expected: float = 1.0) -> float | None:
    if value is None or reference is None:
        return None
    if value <= 0.0 or reference <= 0.0 or expected <= 0.0:
        return None
    return 1200.0 * math.log2((float(value) / float(reference)) / expected)


def validate_vocal_tract_recording(
    input_path: Path | str,
    output_directory: Path | str,
    *,
    ratios: Sequence[float] = (0.65, 0.94, 1.0, 1.06, 1.50),
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    vowel: str = "e",
    title: str = "Rendered /e/ vocal-tract validation",
    render_plot: bool = True,
) -> tuple[dict[str, object], Path | None]:
    source_path = Path(input_path)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    audio = read_audio(source_path)
    first = max(0, int(round(float(start_seconds) * audio.sample_rate)))
    last = (
        audio.samples.size
        if end_seconds is None
        else min(audio.samples.size,
                 int(round(float(end_seconds) * audio.sample_rate)))
    )
    if last - first < max(512, int(round(audio.sample_rate * 0.12))):
        raise ValueError("validation interval is too short")
    source = np.asarray(audio.samples[first:last], np.float64)
    unique_ratios = tuple(dict.fromkeys(float(value) for value in ratios))
    if 1.0 not in unique_ratios:
        unique_ratios += (1.0,)
    rendered: dict[float, np.ndarray] = {}
    transforms = {}
    analyses: dict[float, FormantSegment] = {}
    envelopes: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    wav_paths: dict[float, Path] = {}
    duration = source.size / float(audio.sample_rate)
    segments = [{"phone": vowel, "start": 0.0, "end": duration}]
    for ratio in unique_ratios:
        result = transform_vocal_tract(
            source,
            audio.sample_rate,
            ratio,
            chipmunk_range=True,
            segments=segments,
        )
        rendered[ratio] = np.asarray(result.samples, np.float64)
        transforms[ratio] = result
        analyses[ratio] = _analyze_output(
            rendered[ratio], audio.sample_rate, source_path, ratio, vowel
        )
        envelopes[ratio] = average_cepstral_envelope_db(
            rendered[ratio], audio.sample_rate
        )
        wav_path = output_root / f"ratio_{ratio:.3f}.wav"
        _write_pcm16(wav_path, rendered[ratio], audio.sample_rate)
        wav_paths[ratio] = wav_path

    identity_analysis = analyses[1.0]
    identity_formants = identity_analysis.median_formants_hz
    identity_f0 = identity_analysis.median_f0_hz
    frequencies, identity_envelope = envelopes[1.0]
    points: list[VocalTractValidationPoint] = []
    for ratio in unique_ratios:
        analysis = analyses[ratio]
        result = transforms[ratio]
        output_frequencies, output_envelope = envelopes[ratio]
        if not np.array_equal(output_frequencies, frequencies):
            output_envelope = np.interp(
                frequencies, output_frequencies, output_envelope
            )
        target, valid = _target_envelope(
            frequencies, identity_envelope, ratio
        )
        expected_multiplier = ratio_to_formant_multiplier(ratio)
        measured_formants = identity_anchored_formants(
            frequencies,
            output_envelope,
            identity_formants,
            expected_multiplier,
        )
        multipliers: list[float | None] = []
        errors_hz: list[float | None] = []
        ratio_errors: list[float | None] = []
        errors: list[float | None] = []
        for measured, reference in zip(
                measured_formants, identity_formants):
            multiplier = (
                float(measured) / float(reference)
                if measured is not None and reference not in (None, 0.0)
                else None
            )
            multipliers.append(multiplier)
            errors_hz.append(
                float(measured) - float(reference) * expected_multiplier
                if measured is not None and reference is not None else None
            )
            ratio_errors.append(
                multiplier - expected_multiplier
                if multiplier is not None else None
            )
            errors.append(_safe_log_ratio(
                measured, reference, expected_multiplier
            ))
        finite_errors_hz = [abs(value) for value in errors_hz
                            if value is not None and math.isfinite(value)]
        finite_ratio_errors = [abs(value) for value in ratio_errors
                               if value is not None and math.isfinite(value)]
        finite_errors = [abs(value) for value in errors
                         if value is not None and math.isfinite(value)]
        f0_drift = (
            12.0 * math.log2(float(analysis.median_f0_hz) /
                             float(identity_f0))
            if analysis.median_f0_hz and identity_f0
            else None
        )
        points.append(VocalTractValidationPoint(
            ratio=ratio,
            requested_formant_multiplier=expected_multiplier,
            measured_formants_hz=tuple(measured_formants),
            independent_formants_hz=tuple(analysis.median_formants_hz),
            measured_formant_multipliers=tuple(multipliers),
            formant_error_hz=tuple(errors_hz),
            formant_ratio_error=tuple(ratio_errors),
            formant_error_cents=tuple(errors),
            median_absolute_formant_error_hz=(
                float(np.median(finite_errors_hz))
                if finite_errors_hz else None
            ),
            median_absolute_formant_ratio_error=(
                float(np.median(finite_ratio_errors))
                if finite_ratio_errors else None
            ),
            median_absolute_formant_error_cents=(
                float(np.median(finite_errors)) if finite_errors else None
            ),
            formant_measurement_method=(
                "identity_anchored_final_cepstral_envelope_peaks_v1"
            ),
            f0_hz=analysis.median_f0_hz,
            f0_drift_semitones=f0_drift,
            duration_samples=int(rendered[ratio].size),
            duration_drift_samples=int(rendered[ratio].size - source.size),
            envelope_target_rmse_db=_shape_rmse(
                output_envelope, target, valid
            ),
            envelope_change_rms_db=_shape_rmse(
                output_envelope, identity_envelope, valid
            ),
            accepted_formant_frames=analysis.accepted_frame_count,
            rejected_formant_frames=analysis.rejected_frame_count,
            peak=float(np.max(np.abs(rendered[ratio]), initial=0.0)),
            clipped_sample_count=int(np.count_nonzero(
                np.abs(rendered[ratio]) > 1.0
            )),
            real_time_factor=float(result.real_time_factor),
            wav_file=wav_paths[ratio].name,
        ))

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "rendered_vocal_tract_validation",
        "method": transforms[unique_ratios[0]].method,
        "title": title,
        "input": {
            "file_name": source_path.name,
            "vowel": vowel,
            "sample_rate": audio.sample_rate,
            "start_seconds": float(start_seconds),
            "end_seconds": float(last / audio.sample_rate),
            "duration_samples": int(source.size),
        },
        "identity_exact": bool(np.array_equal(rendered[1.0],
                                               source.astype(np.float32))),
        "points": [asdict(point) for point in points],
        "interpretation": {
            "ratio_below_one": "shorter apparent tract; formants rise",
            "ratio_above_one": "longer apparent tract; formants fall",
            "solid_envelope": "measured from final rendered waveform",
            "dashed_envelope": "requested frequency-warp target",
            "measured_formants": (
                "final-waveform cepstral-envelope peaks paired to identity "
                "F1-F4 by the requested uniform frequency scale"
            ),
            "independent_formants": (
                "ordinary frame tracker retained as a non-authoritative "
                "cross-check; it may relabel resonances at expanded ratios"
            ),
            "formant_tracks": (
                "independent accepted output-waveform estimates; not "
                "internal targets"
            ),
        },
    }
    json_path = output_root / "vocal_tract_validation.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_path = output_root / "vocal_tract_validation.png"
    if render_plot:
        render_vocal_tract_validation_plot(
            plot_path,
            rendered,
            audio.sample_rate,
            analyses,
            envelopes,
            report,
        )
    else:
        plot_path = None
    return report, plot_path


def _qimage_from_rgb(QtGui: object, rgb: np.ndarray):
    values = np.ascontiguousarray(rgb, np.uint8)
    return QtGui.QImage(
        values.data,
        values.shape[1],
        values.shape[0],
        values.strides[0],
        QtGui.QImage.Format_RGB888,
    ).copy()


def _draw_polyline(QtCore: object, QtGui: object, painter: object,
                   points: Sequence[tuple[float, float]]) -> None:
    if len(points) < 2:
        return
    painter.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(float(x), float(y)) for x, y in points
    ]))


def formant_view_spectrogram_db(
    decibels: np.ndarray,
    frequencies: np.ndarray,
    *,
    smoothing_hz: float = 240.0,
) -> np.ndarray:
    """Suppress the harmonic comb for a human-readable resonance view.

    Smoothing is applied to spectral power only in the diagnostic image. The
    measured audio and the formant estimator continue to use unsmoothed data.
    """
    values = np.asarray(decibels, np.float64)
    frequency = np.asarray(frequencies, np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != frequency.size:
        raise ValueError("spectrogram and frequency dimensions disagree")
    if frequency.size < 3:
        return values.copy()
    bin_hz = max(1.0e-9, float(np.median(np.diff(frequency))))
    width = max(3, int(round(float(smoothing_hz) / bin_hz)))
    if width % 2 == 0:
        width += 1
    width = min(width, max(3, frequency.size // 2 * 2 - 1))
    kernel = np.hanning(width)
    if float(np.sum(kernel)) <= 0.0:
        kernel = np.ones(width, np.float64)
    kernel /= float(np.sum(kernel))
    power = 10.0 ** (values / 10.0)
    half = width // 2
    padded = np.pad(power, ((half, half), (0, 0)), mode="edge")
    smoothed = np.empty_like(power)
    for frame_index in range(power.shape[1]):
        smoothed[:, frame_index] = np.convolve(
            padded[:, frame_index], kernel, mode="valid"
        )
    if smoothed.shape[1] >= 3:
        smoothed[:, 1:-1] = (
            smoothed[:, :-2]
            + 2.0 * smoothed[:, 1:-1]
            + smoothed[:, 2:]
        ) / 4.0
    result = 10.0 * np.log10(np.maximum(smoothed, 1.0e-12))
    result -= float(np.max(result, initial=0.0))
    return np.clip(result, SPECTROGRAM_FLOOR_DB, 0.0)


def envelope_display_range(
    curves: Sequence[np.ndarray], *, minimum_span_db: float = 40.0
) -> tuple[float, float, float]:
    """Return a padded, tick-aligned range containing every plotted curve.

    The validation graph compares measured and requested envelopes. A fixed
    viewport clipped valid cepstral valleys and peaks, making an otherwise
    correct final-waveform trace look as though it missed its target.
    """
    finite = []
    for curve in curves:
        values = np.asarray(curve, np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return -30.0, 10.0, 10.0
    values = np.concatenate(finite)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(float(minimum_span_db), upper - lower)
    padding = max(3.0, span * 0.08)
    lower -= padding
    upper += padding
    requested_step = max(5.0, (upper - lower) / 6.0)
    tick_step = 5.0 * (2.0 ** math.ceil(math.log2(requested_step / 5.0)))
    lower = math.floor(lower / tick_step) * tick_step
    upper = math.ceil(upper / tick_step) * tick_step
    if upper <= lower:
        upper = lower + tick_step
    return float(lower), float(upper), float(tick_step)


def render_vocal_tract_validation_plot(
    path: Path | str,
    rendered: dict[float, np.ndarray],
    sample_rate: int,
    analyses: dict[float, FormantSegment],
    envelopes: dict[float, tuple[np.ndarray, np.ndarray]],
    report: dict[str, object],
    *,
    width: int = 1900,
    height: int = 1250,
) -> Path:
    """Render an inspectable final-waveform formant-shift comparison."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font_family = _load_diagnostic_font(QtGui)
    image = QtGui.QImage(
        width, height, QtGui.QImage.Format_ARGB32_Premultiplied
    )
    image.fill(QtGui.QColor("#f5f6f7"))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    try:
        title_font = QtGui.QFont(font_family, 15)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#202124"))
        painter.drawText(34, 38, str(report.get("title") or "Validation"))
        body_font = QtGui.QFont(font_family, 9)
        painter.setFont(body_font)
        painter.drawText(
            34, 64,
            "Measured from final WAV output. Harmonic spacing should stay fixed; "
            "colored formant tracks and the broad spectral envelope should move.",
        )
        painter.drawText(
            34, 84,
            "Ratio < 1: shorter / brighter (formants up). Ratio > 1: longer / "
            "darker (formants down). F0 and duration are held constant.",
        )

        legend_font = QtGui.QFont(font_family, 8)
        painter.setFont(legend_font)
        legend_labels = (
            ("F1", "#ffcb48"), ("F2", "#42d6a4"),
            ("F3", "#df80ff"), ("F4", "#62c8ff"),
        )
        legend_x = width - 590
        painter.setPen(QtGui.QColor("#4b5056"))
        painter.drawText(legend_x, 106, "Tracked output formants:")
        offset = 145
        for label, color in legend_labels:
            painter.setPen(QtGui.QPen(QtGui.QColor("#202124"), 5))
            painter.drawLine(legend_x + offset, 101,
                             legend_x + offset + 26, 101)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 3))
            painter.drawLine(legend_x + offset, 101,
                             legend_x + offset + 26, 101)
            painter.setPen(QtGui.QColor("#30343a"))
            painter.drawText(legend_x + offset + 31, 106, label)
            offset += 72

        requested = [float(point["ratio"])
                     for point in report.get("points", [])]
        preferred = [min(requested), 1.0, max(requested)]
        ratios = list(dict.fromkeys(preferred))
        left = 70.0
        right = float(width - 35)
        panel_gap = 24.0
        panel_width = (
            right - left - panel_gap * (len(ratios) - 1)
        ) / len(ratios)
        spec_top = 142.0
        spec_height = 390.0
        maximum_hz = 6000.0
        formant_colors = ("#ffcb48", "#42d6a4", "#df80ff", "#62c8ff")

        point_by_ratio = {float(point["ratio"]): point
                          for point in report.get("points", [])}
        for panel_index, ratio in enumerate(ratios):
            panel_left = left + panel_index * (panel_width + panel_gap)
            panel_rect = QtCore.QRectF(
                panel_left, spec_top, panel_width, spec_height
            )
            _times, frequencies, decibels = spectrogram_db(
                rendered[ratio], sample_rate, fft_size=2048, hop_size=128
            )
            frequency_mask = frequencies <= maximum_hz
            display_frequencies = frequencies[frequency_mask]
            display_decibels = formant_view_spectrogram_db(
                decibels[frequency_mask, :], display_frequencies
            )
            spectral_rgb = _spectrogram_rgb(display_decibels)
            spectral_image = _qimage_from_rgb(
                QtGui, np.flipud(spectral_rgb)
            )
            painter.drawImage(panel_rect, spectral_image)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4f5358"), 1))
            painter.drawRect(panel_rect)
            duration = rendered[ratio].size / float(sample_rate)
            analysis = analyses[ratio]
            painter.save()
            painter.setClipRect(panel_rect)
            for formant_index, color in enumerate(formant_colors):
                groups: list[list[tuple[float, float]]] = [[]]
                for frame in analysis.frames:
                    tracked = frame.tracked_formants_hz
                    value = (
                        tracked[formant_index]
                        if formant_index < len(tracked)
                        else frame.formants_hz[formant_index]
                    )
                    if (not frame.accepted or value is None
                            or value > maximum_hz):
                        if groups[-1]:
                            groups.append([])
                        continue
                    x = panel_left + (
                        frame.frame_time_seconds / max(duration, 1.0e-9)
                    ) * panel_width
                    y = spec_top + spec_height * (
                        1.0 - float(value) / maximum_hz
                    )
                    groups[-1].append((x, y))
                for points in groups:
                    if len(points) < 2:
                        continue
                    painter.setPen(QtGui.QPen(
                        QtGui.QColor(25, 28, 32, 210), 6,
                        QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                        QtCore.Qt.RoundJoin,
                    ))
                    _draw_polyline(QtCore, QtGui, painter, points)
                    painter.setPen(QtGui.QPen(
                        QtGui.QColor(color), 3,
                        QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                        QtCore.Qt.RoundJoin,
                    ))
                    _draw_polyline(QtCore, QtGui, painter, points)
            painter.restore()

            point = point_by_ratio[ratio]
            painter.setFont(QtGui.QFont(font_family, 10))
            painter.setPen(QtGui.QColor("#202124"))
            direction = "shorter" if ratio < 1.0 else (
                "longer" if ratio > 1.0 else "identity"
            )
            painter.drawText(
                QtCore.QRectF(panel_left, 116, panel_width, 22),
                QtCore.Qt.AlignCenter,
                f"{direction}: ratio {ratio:.2f}  |  requested formants "
                f"x{float(point['requested_formant_multiplier']):.3f}",
            )
            formants = point.get("measured_formants_hz") or ()
            labels = ", ".join(
                f"F{i + 1} {float(value):.0f}"
                for i, value in enumerate(formants) if value is not None
            )
            painter.setFont(body_font)
            painter.drawText(
                QtCore.QRectF(panel_left, spec_top + spec_height + 6,
                              panel_width, 42),
                QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                labels,
            )
            painter.drawText(
                QtCore.QRectF(panel_left, spec_top + spec_height + 43,
                              panel_width, 38),
                QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                f"F0 drift {float(point.get('f0_drift_semitones') or 0):+.4f} st  |  "
                f"duration drift {int(point.get('duration_drift_samples') or 0)} samples  |  "
                f"target error {float(point.get('envelope_target_rmse_db') or 0):.2f} dB",
            )

        base_frequencies, base_envelope = envelopes[1.0]
        display_mask = base_frequencies <= 6000.0
        prepared_envelopes = []
        for ratio in ratios:
            frequencies, envelope = envelopes[ratio]
            if not np.array_equal(frequencies, base_frequencies):
                envelope = np.interp(base_frequencies, frequencies, envelope)
            target, valid = _target_envelope(
                base_frequencies, base_envelope, ratio
            )
            common = display_mask & valid
            # Remove one constant energy term; resonance shape remains.
            offset = float(np.median(envelope[common] - target[common]))
            reference_level = float(np.median(base_envelope[common]))
            prepared_envelopes.append((
                ratio,
                common,
                envelope - offset - reference_level,
                target - reference_level,
            ))
        graph_min_db, graph_max_db, graph_tick_db = envelope_display_range(
            [row[2][row[1]] for row in prepared_envelopes]
            + [row[3][row[1]] for row in prepared_envelopes]
        )

        graph_left = 95.0
        graph_top = 690.0
        graph_width = float(width - 155)
        graph_height = 330.0
        painter.fillRect(
            QtCore.QRectF(graph_left, graph_top, graph_width, graph_height),
            QtGui.QColor("#ffffff"),
        )
        for tick_hz in range(0, 6001, 1000):
            x = graph_left + graph_width * tick_hz / 6000.0
            painter.setPen(QtGui.QPen(QtGui.QColor("#e2e5e8"), 1))
            painter.drawLine(QtCore.QPointF(x, graph_top),
                             QtCore.QPointF(x, graph_top + graph_height))
            painter.setPen(QtGui.QColor("#555b62"))
            painter.drawText(int(x - 22), int(graph_top + graph_height + 22),
                             f"{tick_hz}")
        first_tick = math.ceil(graph_min_db / graph_tick_db) * graph_tick_db
        ticks_db = np.arange(
            first_tick, graph_max_db + graph_tick_db * 0.5, graph_tick_db
        )
        for tick_db in ticks_db:
            y = graph_top + graph_height * (
                graph_max_db - float(tick_db)
            ) / (graph_max_db - graph_min_db)
            painter.setPen(QtGui.QPen(QtGui.QColor("#e2e5e8"), 1))
            painter.drawLine(QtCore.QPointF(graph_left, y),
                             QtCore.QPointF(graph_left + graph_width, y))
            painter.setPen(QtGui.QColor("#555b62"))
            painter.drawText(43, int(y + 4), f"{float(tick_db):+.0f}")
        painter.setPen(QtGui.QPen(QtGui.QColor("#4f5358"), 1))
        painter.drawRect(QtCore.QRectF(
            graph_left, graph_top, graph_width, graph_height
        ))
        painter.setFont(QtGui.QFont(font_family, 10))
        painter.drawText(95, 668,
                         "Final-waveform cepstral envelopes (energy offset removed)")
        painter.setFont(body_font)
        painter.drawText(int(graph_left + graph_width / 2 - 70),
                         int(graph_top + graph_height + 43), "Frequency (Hz)")
        palette = {
            min(ratios): "#177bc1",
            1.0: "#555b62",
            max(ratios): "#c43d52",
        }
        legend_x = graph_left + 20
        painter.save()
        painter.setClipRect(QtCore.QRectF(
            graph_left, graph_top, graph_width, graph_height
        ))
        for ratio, common, measured, target in prepared_envelopes:
            color = QtGui.QColor(palette[ratio])
            measured_points = []
            target_points = []
            for frequency, actual, wanted, usable in zip(
                    base_frequencies, measured, target, common):
                if not usable:
                    continue
                x = graph_left + graph_width * float(frequency) / 6000.0
                y1 = graph_top + graph_height * (
                    graph_max_db - float(actual)
                ) / (graph_max_db - graph_min_db)
                y2 = graph_top + graph_height * (
                    graph_max_db - float(wanted)
                ) / (graph_max_db - graph_min_db)
                measured_points.append((x, y1))
                target_points.append((x, y2))
            painter.setPen(QtGui.QPen(color, 3))
            _draw_polyline(QtCore, QtGui, painter, measured_points)
            dashed = QtGui.QPen(color, 1.5, QtCore.Qt.DashLine)
            painter.setPen(dashed)
            _draw_polyline(QtCore, QtGui, painter, target_points)
        painter.restore()
        for line_index, ratio in enumerate(ratios):
            color = QtGui.QColor(palette[ratio])
            painter.setPen(QtGui.QPen(color, 3))
            y = 1094 + line_index * 24
            painter.drawLine(QtCore.QPointF(legend_x, y),
                             QtCore.QPointF(legend_x + 34, y))
            painter.setPen(QtGui.QColor("#30343a"))
            painter.drawText(int(legend_x + 44), int(y + 4),
                             f"ratio {ratio:.2f}: solid measured, dashed requested")

        painter.setFont(body_font)
        painter.setPen(QtGui.QColor("#30343a"))
        painter.drawText(
            QtCore.QRectF(720, 1080, width - 760, 112),
            QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap,
            "How to read this image: the top panels are frequency-smoothed formant "
            "views; smoothing changes only this image, not the audio. Dark-outlined "
            "F1-F4 curves are confidence-aware tracked output estimates. In the "
            "lower graph, a solid line lying on its "
            "same-color dashed line means the final rendered waveform follows the "
            "requested source/filter warp; this is not an internal target-only plot.",
        )
    finally:
        painter.end()
    if not image.save(str(destination), "PNG"):
        raise RuntimeError(f"could not save validation image: {destination}")
    del app
    return destination


def default_validation_ratios() -> tuple[float, ...]:
    """Return profile-derived realistic and expanded sweep points."""
    profile = load_vocal_tract_range()
    values = (
        profile.expanded_min_ratio,
        math.sqrt(profile.expanded_min_ratio *
                  profile.realistic_min_ratio),
        profile.realistic_min_ratio,
        math.sqrt(profile.realistic_min_ratio * profile.identity_ratio),
        profile.identity_ratio,
        math.sqrt(profile.identity_ratio * profile.realistic_max_ratio),
        profile.realistic_max_ratio,
        math.sqrt(profile.realistic_max_ratio *
                  profile.expanded_max_ratio),
        profile.expanded_max_ratio,
    )
    return tuple(dict.fromkeys(round(float(value), 8) for value in values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_point_values(points: Sequence[Mapping[str, object]],
                         key: str, *, absolute: bool = False) -> list[float]:
    values = []
    for point in points:
        value = point.get(key)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        values.append(abs(number) if absolute else number)
    return values


def _aggregate_validation_points(
    points: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize transparent final-waveform metrics for one audit slice."""
    rows = list(points)
    accepted = sum(int(row.get("accepted_formant_frames") or 0)
                   for row in rows)
    rejected = sum(int(row.get("rejected_formant_frames") or 0)
                   for row in rows)
    identity_rows = [row for row in rows
                     if row.get("vowel_identity_preserved") is not None]

    def median(key: str, *, absolute: bool = False) -> float | None:
        values = _finite_point_values(rows, key, absolute=absolute)
        return float(np.median(values)) if values else None

    return {
        "point_count": len(rows),
        "median_absolute_formant_error_hz": median(
            "median_absolute_formant_error_hz", absolute=True),
        "median_absolute_formant_ratio_error": median(
            "median_absolute_formant_ratio_error", absolute=True),
        "median_absolute_formant_error_cents": median(
            "median_absolute_formant_error_cents", absolute=True),
        "median_absolute_f0_drift_semitones": median(
            "f0_drift_semitones", absolute=True),
        "maximum_absolute_duration_drift_samples": max(
            (abs(int(row.get("duration_drift_samples") or 0))
             for row in rows), default=0),
        "median_envelope_target_rmse_db": median(
            "envelope_target_rmse_db", absolute=True),
        "formant_tracking_accepted_frames": accepted,
        "formant_tracking_rejected_frames": rejected,
        "formant_tracking_failure_rate": (
            float(rejected / (accepted + rejected))
            if accepted + rejected else None),
        "clipped_sample_count": sum(
            int(row.get("clipped_sample_count") or 0) for row in rows),
        "median_peak": median("peak", absolute=True),
        "median_real_time_factor": median("real_time_factor"),
        "vowel_identity_check_count": len(identity_rows),
        "vowel_identity_preserved_count": sum(
            bool(row.get("vowel_identity_preserved"))
            for row in identity_rows),
        "median_compensated_f1_f2_distance_semitones": median(
            "compensated_f1_f2_identity_distance_semitones",
            absolute=True),
    }


def _annotate_vowel_identity(
    rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Classify intended vowel shape after undoing the requested global warp.

    Multiplying transformed formants by the tract ratio removes the intended
    first-order 1/ratio shift.  Nearest-centroid classification therefore
    measures whether F1/F2 vowel shape survived, rather than penalizing the
    requested apparent-size change itself.
    """
    centroids: dict[str, tuple[float, float]] = {}
    for vowel, row in rows.items():
        identity = next((point for point in row.get("points", ())
                         if abs(float(point.get("ratio") or 0.0) - 1.0)
                         <= 1.0e-8), None)
        formants = list((identity or {}).get("measured_formants_hz") or ())
        if len(formants) >= 2 and all(
                value is not None and float(value) > 0.0
                for value in formants[:2]):
            centroids[str(vowel)] = (
                math.log2(float(formants[0])),
                math.log2(float(formants[1])),
            )

    centroid_pairs = list(centroids.items())
    minimum_centroid_distance = min((
        12.0 * math.sqrt(
            ((left[0] - right[0]) ** 2 +
             (left[1] - right[1]) ** 2) / 2.0
        )
        for index, (_left_name, left) in enumerate(centroid_pairs)
        for _right_name, right in centroid_pairs[index + 1:]
    ), default=None)
    centroids_are_distinct = bool(
        minimum_centroid_distance is not None and
        minimum_centroid_distance >= 0.1
    )

    checks = []
    for vowel, row in rows.items():
        for point in row.get("points", ()):
            formants = list(point.get("measured_formants_hz") or ())
            ratio = float(point.get("ratio") or 0.0)
            if (len(formants) < 2 or ratio <= 0.0 or
                    not centroids_are_distinct or
                    any(value is None or float(value) <= 0.0
                        for value in formants[:2])):
                point["vowel_identity_preserved"] = None
                point["nearest_identity_vowel"] = None
                point["compensated_f1_f2_identity_distance_semitones"] = None
                continue
            compensated = (
                math.log2(float(formants[0]) * ratio),
                math.log2(float(formants[1]) * ratio),
            )
            distances = {
                name: 12.0 * math.sqrt(
                    ((compensated[0] - center[0]) ** 2 +
                     (compensated[1] - center[1]) ** 2) / 2.0
                )
                for name, center in centroids.items()
            }
            nearest = min(distances, key=distances.get)
            preserved = nearest == str(vowel)
            point["nearest_identity_vowel"] = nearest
            point["vowel_identity_preserved"] = preserved
            point["compensated_f1_f2_identity_distance_semitones"] = round(
                distances[str(vowel)], 9)
            checks.append(preserved)
    return {
        "method": "nearest_identity_centroid_after_expected_ratio_compensation",
        "centroid_vowels": sorted(centroids),
        "minimum_centroid_distance_semitones": minimum_centroid_distance,
        "status": ("measured" if centroids_are_distinct else
                   "insufficient_identity_centroid_separation"),
        "check_count": len(checks),
        "preserved_count": sum(checks),
        "preservation_rate": (
            float(sum(checks) / len(checks)) if checks else None),
    }


def validate_source_vowel_suite(
    source_root: Path | str,
    output_directory: Path | str,
    *,
    ratios: Sequence[float] | None = None,
    render_plots: bool = True,
    fixtures: Mapping[str, tuple[str, float, float]] | None = None,
) -> dict[str, object]:
    """Run the final-waveform Stage B audit on all five source vowels.

    Only the named source WAVs are opened, and their hashes are checked again
    after every transformation.  All generated data is written beneath the
    requested output directory.
    """
    source = Path(source_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("validation output must not be inside the source bank")
    if not source.is_dir():
        raise FileNotFoundError(f"source bank folder does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    profile = load_vocal_tract_range()
    sweep = tuple(ratios or default_validation_ratios())
    rows: dict[str, object] = {}
    source_hashes: dict[str, str] = {}
    all_points: list[dict[str, object]] = []
    fixture_rows = dict(fixtures or SOURCE_VOWEL_FIXTURES)
    if set(fixture_rows) != set("aeiou"):
        raise ValueError("source-vowel suite requires a, e, i, o, and u")
    for vowel in "aeiou":
        file_name, start, end = fixture_rows[vowel]
        path = (source / file_name).resolve()
        try:
            path.relative_to(source)
        except ValueError as error:
            raise ValueError(f"source-vowel path escaped the bank: {file_name}") \
                from error
        if not path.is_file():
            raise FileNotFoundError(
                f"source-vowel recording is missing: {file_name}")
        before = _sha256(path)
        report, plot = validate_vocal_tract_recording(
            path,
            output / vowel,
            ratios=sweep,
            start_seconds=start,
            end_seconds=end,
            vowel=vowel,
            title=f"Rendered /{vowel}/ source-vowel tract validation",
            render_plot=render_plots,
        )
        after = _sha256(path)
        if after != before:
            raise RuntimeError(
                f"read-only source recording changed during validation: "
                f"{file_name}")
        source_hashes[file_name] = before
        points = list(report.get("points") or ())
        all_points.extend(points)
        rows[vowel] = {
            "source_file": file_name,
            "source_sha256": before,
            "identity_exact": bool(report.get("identity_exact")),
            "report": f"{vowel}/vocal_tract_validation.json",
            "plot": (f"{vowel}/{plot.name}" if plot is not None else None),
            "points": points,
        }

    vowel_identity = _annotate_vowel_identity(rows)
    global_metrics = _aggregate_validation_points(all_points)
    metrics_by_vowel = {
        vowel: _aggregate_validation_points(row["points"])
        for vowel, row in rows.items()
    }
    metrics_by_range: dict[str, dict[str, object]] = {}
    for name, selected in {
        "identity": [
            row for row in all_points
            if abs(float(row.get("ratio") or 0.0) - 1.0) <= 1.0e-8
        ],
        "realistic": [
            row for row in all_points
            if (abs(float(row.get("ratio") or 0.0) - 1.0) > 1.0e-8 and
                profile.realistic_min_ratio <=
                float(row.get("ratio") or 0.0) <=
                profile.realistic_max_ratio)
        ],
        "expanded": [
            row for row in all_points
            if (float(row.get("ratio") or 0.0) <
                profile.realistic_min_ratio or
                float(row.get("ratio") or 0.0) >
                profile.realistic_max_ratio)
        ],
    }.items():
        metrics_by_range[name] = _aggregate_validation_points(selected)

    f0_ranges = {
        "low_below_140_hz": [],
        "mid_140_to_220_hz": [],
        "high_220_hz_and_above": [],
    }
    for row in all_points:
        f0_hz = row.get("f0_hz")
        if f0_hz is None:
            continue
        value = float(f0_hz)
        if value < 140.0:
            f0_ranges["low_below_140_hz"].append(row)
        elif value < 220.0:
            f0_ranges["mid_140_to_220_hz"].append(row)
        else:
            f0_ranges["high_220_hz_and_above"].append(row)
    metrics_by_f0_range = {
        name: _aggregate_validation_points(selected)
        for name, selected in f0_ranges.items()
    }

    finite_formant_errors = [
        abs(float(row["median_absolute_formant_error_cents"]))
        for row in all_points
        if row.get("median_absolute_formant_error_cents") is not None
    ]
    f0_drifts = [
        abs(float(row["f0_drift_semitones"]))
        for row in all_points if row.get("f0_drift_semitones") is not None
    ]
    direction_checks = []
    for row in all_points:
        ratio = float(row["ratio"])
        if abs(ratio - 1.0) <= 1.0e-10:
            continue
        multipliers = [
            float(value) for value in
            (row.get("measured_formant_multipliers") or ())
            if value is not None and math.isfinite(float(value))
        ]
        if not multipliers:
            direction_checks.append(False)
            continue
        measured = float(np.median(multipliers))
        direction_checks.append(
            measured > 1.0 if ratio < 1.0 else measured < 1.0)
    maximum_f0_drift = max(f0_drifts, default=0.0)
    maximum_duration_drift = max(
        (abs(int(row.get("duration_drift_samples") or 0))
         for row in all_points), default=0)
    clipped = sum(int(row.get("clipped_sample_count") or 0)
                  for row in all_points)
    all_identity = all(bool(row["identity_exact"])
                       for row in rows.values())
    realistic_identity = metrics_by_range["realistic"]
    realistic_identity_checks_pass = bool(
        not realistic_identity["vowel_identity_check_count"] or
        realistic_identity["vowel_identity_preserved_count"] ==
        realistic_identity["vowel_identity_check_count"]
    )
    passed = bool(
        len(rows) == 5
        and all_identity
        and maximum_duration_drift == 0
        and maximum_f0_drift <= 0.08
        and clipped == 0
        and direction_checks
        and all(direction_checks)
        and realistic_identity_checks_pass
    )
    result = {
        "schema_version": 1,
        "kind": "prompt20_stage_b_source_vowel_validation",
        "passed": passed,
        "source_bank_write_performed": False,
        "source_hashes_unchanged": True,
        "ratios": list(sweep),
        "summary": {
            "vowel_count": len(rows),
            "point_count": len(all_points),
            "identity_exact_for_all_vowels": all_identity,
            "maximum_duration_drift_samples": maximum_duration_drift,
            "maximum_absolute_f0_drift_semitones": maximum_f0_drift,
            "clipped_sample_count": clipped,
            "direction_check_count": len(direction_checks),
            "direction_check_pass_count": sum(direction_checks),
            "median_absolute_formant_error_cents": (
                float(np.median(finite_formant_errors))
                if finite_formant_errors else None
            ),
            "maximum_median_absolute_formant_error_cents": (
                max(finite_formant_errors) if finite_formant_errors else None
            ),
            "median_real_time_factor": float(np.median([
                float(row.get("real_time_factor") or 0.0)
                for row in all_points
            ])),
            "median_absolute_formant_error_hz":
                global_metrics["median_absolute_formant_error_hz"],
            "median_absolute_formant_ratio_error":
                global_metrics["median_absolute_formant_ratio_error"],
            "formant_tracking_failure_rate":
                global_metrics["formant_tracking_failure_rate"],
            "vowel_identity_check_count":
                vowel_identity["check_count"],
            "vowel_identity_preserved_count":
                vowel_identity["preserved_count"],
            "realistic_vowel_identity_check_count":
                realistic_identity["vowel_identity_check_count"],
            "realistic_vowel_identity_preserved_count":
                realistic_identity["vowel_identity_preserved_count"],
            "expanded_vowel_identity_warning_count": (
                metrics_by_range["expanded"]["vowel_identity_check_count"] -
                metrics_by_range["expanded"][
                    "vowel_identity_preserved_count"]
            ),
        },
        "vowel_identity": vowel_identity,
        "metrics_global": global_metrics,
        "metrics_by_vowel": metrics_by_vowel,
        "metrics_by_source_speaker": {
            "project_source_speaker": global_metrics,
        },
        "metrics_by_f0_range": metrics_by_f0_range,
        "metrics_by_range": metrics_by_range,
        "metrics_by_phonation": {
            "modal": global_metrics,
            "creaky": {
                "point_count": 0,
                "status": (
                    "not present in the stable source-vowel sweep; current-"
                    "model creak fixtures and join metrics are in the "
                    "separate listening manifest"
                ),
            },
        },
        "source_files": source_hashes,
        "vowels": rows,
        "limitations": [
            "Formant correspondence is anchored to each identity rendering; "
            "the independent tracker remains a cross-check.",
            "Acoustic naturalness and speaker-identity preservation require "
            "human listening.",
            "The uniform spectral-envelope warp is an acoustic approximation, "
            "not an anatomical larynx or vocal-tract simulator.",
            "Nearest-centroid vowel identity is a structural warning metric. "
            "The realistic range is gated; expanded-range warnings remain "
            "visible and require human intelligibility listening.",
        ],
    }
    (output / "stage_b_source_vowels.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate vocal-tract formant shifts on rendered WAV output"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument(
        "--source-root", type=Path,
        help="run the five-vowel project-speaker Stage B suite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--vowel", default="e")
    parser.add_argument("--title", default="Rendered /e/ vocal-tract validation")
    parser.add_argument("--ratios", type=float, nargs="+")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_root is not None:
        report = validate_source_vowel_suite(
            args.source_root,
            args.output,
            ratios=args.ratios,
            render_plots=not args.no_plots,
        )
        print(json.dumps({
            "json": str(args.output / "stage_b_source_vowels.json"),
            "passed": report["passed"],
            "vowel_count": report["summary"]["vowel_count"],
            "point_count": report["summary"]["point_count"],
        }, ensure_ascii=False))
        return 0 if report["passed"] else 1
    report, plot = validate_vocal_tract_recording(
        args.input,
        args.output,
        ratios=args.ratios or default_validation_ratios(),
        start_seconds=args.start,
        end_seconds=args.end,
        vowel=args.vowel,
        title=args.title,
        render_plot=not args.no_plots,
    )
    print(json.dumps({
        "json": str(args.output / "vocal_tract_validation.json"),
        "plot": str(plot) if plot is not None else None,
        "point_count": len(report["points"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
