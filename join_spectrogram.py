"""Deterministic waveform/spectrogram audit images for rendered joins.

This is a diagnostic only.  It does not alter audio, unit choices, pitchmarks,
or splice positions.  Rendering uses the Qt dependency already required by the
GUI and NumPy's FFT, so no plotting or signal-processing package is added.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import wave

import numpy as np


SPECTROGRAM_FLOOR_DB = -84.0
_EXPECTED_BURST_PHONES = frozenset({
    "p", "b", "t", "d", "k", "g", "q", "cl", "ch", "jh", "ts",
    "dz", "dx",
})


def _load_diagnostic_font(QtGui: object) -> str:
    """Load a concrete font file when Qt's system-family lookup is broken."""
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        fonts / "meiryo.ttc",
        fonts / "YuGothR.ttc",
        fonts / "segoeui.ttf",
        fonts / "arial.ttf",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(candidate))
        if font_id < 0:
            continue
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return str(families[0])
    return "Arial"


def spectrogram_db(
    samples: object,
    sample_rate: int,
    *,
    fft_size: int = 1024,
    hop_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame times, frequencies, and peak-relative STFT decibels."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    sample_rate = int(sample_rate)
    fft_size = int(fft_size)
    hop_size = int(hop_size)
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    if fft_size < 128 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a power of two at least 128")
    if hop_size < 1 or hop_size > fft_size:
        raise ValueError("hop_size must be between 1 and fft_size")
    if values.size < fft_size:
        values = np.pad(values, (0, fft_size - values.size))
    frame_count = 1 + int(math.ceil((values.size - fft_size) / hop_size))
    needed = (frame_count - 1) * hop_size + fft_size
    if needed > values.size:
        values = np.pad(values, (0, needed - values.size))
    window = np.hanning(fft_size)
    result = np.empty((fft_size // 2 + 1, frame_count), dtype=np.float32)
    for frame_index in range(frame_count):
        first = frame_index * hop_size
        frame = values[first:first + fft_size] * window
        result[:, frame_index] = np.abs(np.fft.rfft(frame))
    peak = max(float(np.max(result)), 1.0e-12)
    decibels = 20.0 * np.log10(np.maximum(result, peak * 1.0e-8) / peak)
    decibels = np.clip(decibels, SPECTROGRAM_FLOOR_DB, 0.0)
    times = (
        np.arange(frame_count, dtype=np.float64) * hop_size
        + fft_size * 0.5
    ) / sample_rate
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return times, frequencies, decibels


def _spectrogram_rgb(decibels: np.ndarray) -> np.ndarray:
    level = np.clip(
        (np.asarray(decibels, dtype=np.float64) - SPECTROGRAM_FLOOR_DB)
        / -SPECTROGRAM_FLOOR_DB,
        0.0,
        1.0,
    )
    stops = np.asarray((
        (249, 250, 253),
        (196, 218, 250),
        (65, 111, 224),
        (210, 43, 177),
        (247, 34, 47),
    ), dtype=np.float64)
    scaled = level * (len(stops) - 1)
    lower = np.minimum(scaled.astype(np.int32), len(stops) - 2)
    fraction = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[lower + 1] * fraction
    return np.asarray(np.rint(rgb), dtype=np.uint8)


def _join_rows(diagnostic: object) -> list[dict]:
    if not isinstance(diagnostic, dict):
        return []
    rows = diagnostic.get("joins") or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _segment_rows(diagnostic: object) -> list[dict]:
    if not isinstance(diagnostic, dict):
        return []
    rows = diagnostic.get("segments") or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _phone_sequence_text(diagnostic: object) -> str:
    """Return the complete rendered phone string used by the image footer."""
    phones = [str(row.get("phone") or "?")
              for row in _segment_rows(diagnostic)]
    return " ".join(phones)


def _is_expected_burst_phone(phone: object) -> bool:
    key = str(phone or "").strip().lower()
    while key and key[-1].isdigit():
        key = key[:-1]
    return (key in _EXPECTED_BURST_PHONES or
            (len(key) == 2 and key.endswith("y") and key[0] in "pkgbtd"))


def _join_marker_time(row: dict) -> float:
    """Point a crackle marker at the measured event, not the nominal splice."""
    splice_time = float(
        row.get("time") or row.get("splice_time_seconds") or 0.0)
    issues = {str(value) for value in (row.get("issues") or ())}
    if (str(row.get("dominant_issue") or "") == "BROADBAND_IMPULSE" or
            "BROADBAND_IMPULSE" in issues):
        try:
            event_time = float(row["broadband_impulse_time_seconds"])
        except (KeyError, TypeError, ValueError):
            return splice_time
        if math.isfinite(event_time):
            return event_time
    return splice_time


def render_join_spectrogram(
    samples: object,
    sample_rate: int,
    output_path: Path | str,
    *,
    diagnostic: dict | None = None,
    title: str = "Rendered join acoustic audit",
    width: int = 1800,
    height: int = 1000,
    fft_size: int = 512,
    hop_size: int = 32,
) -> Path:
    """Save a waveform/STFT image with join and measured-event markers."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not values.size:
        raise ValueError("audio samples are empty")
    sample_rate = int(sample_rate)
    duration = values.size / float(sample_rate)
    _times, frequencies, decibels = spectrogram_db(
        values, sample_rate, fft_size=fft_size, hop_size=hop_size,
    )
    rgb = _spectrogram_rgb(decibels)
    rgb = np.ascontiguousarray(np.flipud(rgb))
    spectral_image = QtGui.QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QtGui.QImage.Format_RGB888,
    ).copy()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font_family = _load_diagnostic_font(QtGui)
    image = QtGui.QImage(
        int(width), int(height), QtGui.QImage.Format_ARGB32_Premultiplied
    )
    image.fill(QtGui.QColor("#f7f7f5"))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    try:
        margin_left = 96
        margin_right = 28
        plot_width = width - margin_left - margin_right
        wave_top = 82
        wave_height = max(130, min(230, int(round(height * 0.23))))
        spec_top = wave_top + wave_height + 58
        # Reserve a real footer for phone context, time labels, and the
        # diagnostic disclaimer.  The old fixed 500 px STFT fell off-canvas
        # when callers requested a compact image.
        spec_height = max(160, height - spec_top - 160)
        waveform_rect = QtCore.QRectF(
            margin_left, wave_top, plot_width, wave_height
        )
        spectrum_rect = QtCore.QRectF(
            margin_left, spec_top, plot_width, spec_height
        )

        title_font = QtGui.QFont(font_family)
        title_font.setPointSize(15)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#202124"))
        painter.drawText(24, 38, str(title))

        summary = dict((diagnostic or {}).get("summary") or {})
        detail = (
            f"{sample_rate} Hz | {duration:.3f} s | "
            f"{len(_join_rows(diagnostic or {}))} joins | "
            f"{int(summary.get('flagged_join_count') or 0)} flagged"
        )
        body_font = QtGui.QFont(font_family)
        body_font.setPointSize(9)
        painter.setFont(body_font)
        painter.drawText(24, 62, detail)

        legend_rect = QtCore.QRectF(width - 870, 8, 840, 68)
        painter.fillRect(legend_rect, QtGui.QColor(255, 255, 255, 235))
        painter.setPen(QtGui.QPen(QtGui.QColor("#b8bdc3"), 1))
        painter.drawRect(legend_rect)
        legend_font = QtGui.QFont(font_family)
        legend_font.setPointSize(8)
        painter.setFont(legend_font)
        painter.setPen(QtGui.QColor("#30343a"))
        painter.drawText(
            QtCore.QRectF(width - 854, 12, 810, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "Blue waveform = amplitude  |  Spectrogram: white -> blue -> pink -> red = stronger energy",
        )
        painter.drawText(
            QtCore.QRectF(width - 854, 31, 810, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "Red = unexplained issue  |  Violet = review a stop/affricate event  |  Amber = measured join  |  Bar = handoff span",
        )
        painter.drawText(
            QtCore.QRectF(width - 854, 50, 810, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "Phone strip/sequence gives context; blue k/t/p-like stops may legitimately contain a broadband release burst",
        )

        painter.fillRect(waveform_rect, QtGui.QColor("#ffffff"))
        painter.fillRect(spectrum_rect, QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#9aa0a6"), 1))
        painter.drawRect(waveform_rect)
        painter.drawRect(spectrum_rect)
        middle_y = wave_top + wave_height * 0.5
        painter.setPen(QtGui.QPen(QtGui.QColor("#d0d4d8"), 1))
        painter.drawLine(
            margin_left, int(middle_y), margin_left + plot_width, int(middle_y)
        )

        peak = max(float(np.max(np.abs(values))), 1.0e-9)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2457d6"), 1))
        for pixel in range(plot_width):
            first = int(pixel * values.size / plot_width)
            last = max(first + 1, int((pixel + 1) * values.size / plot_width))
            block = values[first:min(last, values.size)]
            low = float(np.min(block)) / peak
            high = float(np.max(block)) / peak
            y1 = middle_y - high * wave_height * 0.47
            y2 = middle_y - low * wave_height * 0.47
            painter.drawLine(
                margin_left + pixel, int(round(y1)),
                margin_left + pixel, int(round(y2)),
            )

        painter.drawImage(spectrum_rect, spectral_image)

        grid_pen = QtGui.QPen(QtGui.QColor(70, 70, 70, 75), 1)
        painter.setPen(grid_pen)
        for fraction in np.linspace(0.0, 1.0, 9):
            x = margin_left + int(round(fraction * plot_width))
            painter.drawLine(x, wave_top, x, wave_top + wave_height)
            painter.drawLine(x, spec_top, x, spec_top + spec_height)
            painter.setPen(QtGui.QColor("#4c5156"))
            painter.drawText(x - 16, spec_top + spec_height + 83,
                             f"{duration * fraction:.2f}")
            painter.setPen(grid_pen)
        painter.drawText(
            margin_left + plot_width // 2 - 30,
            spec_top + spec_height + 105,
            "Time (s)",
        )
        nyquist = float(frequencies[-1])
        for frequency in np.linspace(0.0, nyquist, 6):
            fraction = frequency / max(nyquist, 1.0)
            y = spec_top + spec_height - int(round(fraction * spec_height))
            painter.drawLine(margin_left, y, margin_left + plot_width, y)
            painter.setPen(QtGui.QColor("#4c5156"))
            painter.drawText(18, y + 4, f"{frequency / 1000.0:.1f} kHz")
            painter.setPen(grid_pen)

        joins = _join_rows(diagnostic or {})
        marker_base_y = spec_top - 3
        for row in joins:
            splice_when = float(
                row.get("time") or row.get("splice_time_seconds") or 0)
            marker_when = _join_marker_time(row)
            if marker_when < 0.0 or marker_when > duration:
                continue
            x = margin_left + int(round(marker_when / duration * plot_width))
            flagged = bool(row.get("flagged"))
            expected_burst = bool(
                row.get("broadband_context_may_be_expected") and
                "BROADBAND_IMPULSE" in (row.get("issues") or ()))
            color = QtGui.QColor(
                "#7651a8" if flagged and expected_burst else
                "#d93025" if flagged else "#f29900")
            color.setAlpha(235 if flagged else 170)
            size = 8 if flagged else 4
            triangle = QtGui.QPolygonF((
                QtCore.QPointF(x - size, marker_base_y - size * 1.4),
                QtCore.QPointF(x + size, marker_base_y - size * 1.4),
                QtCore.QPointF(x, marker_base_y),
            ))
            painter.setPen(QtGui.QPen(color, 1))
            painter.setBrush(QtGui.QBrush(color))
            painter.drawPolygon(triangle)

            # When Festival reports the overlap handoff collar, show its
            # extent in the marker rail rather than painting over the STFT.
            if flagged:
                try:
                    handoff_start = float(row["handoff_start"])
                    handoff_end = float(row["handoff_end"])
                except (KeyError, TypeError, ValueError):
                    handoff_start = handoff_end = splice_when
                x1 = margin_left + int(round(
                    max(0.0, min(duration, handoff_start))
                    / duration * plot_width
                ))
                x2 = margin_left + int(round(
                    max(0.0, min(duration, handoff_end))
                    / duration * plot_width
                ))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(color, 2))
                painter.drawLine(x1, marker_base_y - 2, x2, marker_base_y - 2)

        phone_top = spec_top + spec_height + 5
        phone_height = 28
        painter.setPen(QtGui.QColor("#4c5156"))
        painter.drawText(18, phone_top + 19, "Phones")
        phone_font = QtGui.QFont(font_family)
        phone_font.setPointSize(7)
        painter.setFont(phone_font)
        metrics = QtGui.QFontMetrics(phone_font)
        segments = _segment_rows(diagnostic or {})
        for segment_index, segment in enumerate(segments):
            try:
                start = max(0.0, min(duration, float(segment["start"])))
                end = max(start, min(duration, float(segment["end"])))
            except (KeyError, TypeError, ValueError):
                continue
            x1 = margin_left + int(round(start / duration * plot_width))
            x2 = margin_left + int(round(end / duration * plot_width))
            region_width = max(1, x2 - x1)
            phone = str(segment.get("phone") or "?")
            if _is_expected_burst_phone(phone):
                fill = QtGui.QColor("#d9edf2")
            else:
                fill = QtGui.QColor(
                    "#f0f1f3" if segment_index % 2 == 0 else "#e7e9ec")
            rect = QtCore.QRectF(x1, phone_top, region_width, phone_height)
            painter.fillRect(rect, fill)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor("#9aa0a6"), 1))
            painter.drawRect(rect)
            if region_width >= 8:
                label = metrics.elidedText(
                    phone, QtCore.Qt.ElideRight, max(1, region_width - 4))
                painter.setPen(QtGui.QColor("#30343a"))
                painter.drawText(
                    rect.adjusted(2, 0, -2, 0),
                    QtCore.Qt.AlignCenter,
                    label,
                )

        sequence = _phone_sequence_text(diagnostic or {})
        if sequence:
            sequence_font = QtGui.QFont(font_family)
            sequence_font.setPointSize(7)
            painter.setFont(sequence_font)
            sequence_metrics = QtGui.QFontMetrics(sequence_font)
            sequence_label = sequence_metrics.elidedText(
                "Sequence: " + sequence,
                QtCore.Qt.ElideRight,
                max(1, plot_width),
            )
            painter.setPen(QtGui.QColor("#30343a"))
            painter.drawText(
                QtCore.QRectF(margin_left, phone_top + phone_height + 3,
                              plot_width, 18),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                sequence_label,
            )

        painter.setPen(QtGui.QColor("#202124"))
        painter.drawText(margin_left, wave_top - 10, "Waveform")
        painter.drawText(margin_left, spec_top - 10,
                         "STFT magnitude (peak-relative dB)")
        painter.setPen(QtGui.QColor("#5f6368"))
        painter.drawText(
            margin_left,
            height - 28,
            "Red: unexplained issue | Violet: event in expected burst context | Amber: measured join | "
            "Phone context distinguishes legitimate consonant releases. Diagnostic only; audio was not modified.",
        )
    finally:
        painter.end()
        app.processEvents()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(destination), "PNG"):
        raise OSError(f"could not save spectrogram image: {destination}")
    return destination


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("the spectrogram CLI currently expects 16-bit PCM")
    values = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return values, sample_rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a waveform/spectrogram join audit PNG."
    )
    parser.add_argument("--wav", required=True)
    parser.add_argument("--json", default="", help="join diagnostic JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Rendered join acoustic audit")
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=32)
    args = parser.parse_args(argv)
    samples, sample_rate = _read_pcm_wav(Path(args.wav))
    diagnostic = None
    if args.json:
        diagnostic = json.loads(Path(args.json).read_text(encoding="utf-8"))
    render_join_spectrogram(
        samples, sample_rate, args.output,
        diagnostic=diagnostic, title=args.title,
        fft_size=args.fft_size, hop_size=args.hop_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
