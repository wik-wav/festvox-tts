"""Measured spectrogram validation for continuous source-filter voicing.

The image compares the same rendered interval at its measured voicing, at a
partial setting, and at zero.  It is deliberately separate from listening
claims: narrow harmonic-ridge contrast, autocorrelation periodicity, and the
smoothed vocal-tract envelope are reported independently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence
import wave

import numpy as np

from japanese_devoicing import periodicity_score
from join_spectrogram import (
    SPECTROGRAM_FLOOR_DB,
    _load_diagnostic_font,
    _spectrogram_rgb,
    spectrogram_db,
)
from source_filter_voicing import transform_voicing


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("voicing validation expects 16-bit PCM WAV input")
    values = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return np.asarray(values, np.float32), sample_rate


def _write_pcm_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(samples, np.float64).reshape(-1)
    pcm = np.asarray(
        np.rint(np.clip(values, -1.0, 32767.0 / 32768.0) * 32768.0),
        dtype="<i2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _safe_output(output_dir: Path) -> None:
    lowered = {part.casefold() for part in output_dir.parts}
    if "utau" in lowered and "voice" in lowered:
        raise ValueError("Refusing to write validation inside an UTAU bank")


def _rms(values: np.ndarray) -> float:
    samples = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def _best_periodic_crop(samples: np.ndarray, sample_rate: int,
                        seconds: float = 0.22) -> tuple[int, int]:
    values = np.asarray(samples, np.float64)
    width = min(len(values), max(512, int(round(seconds * sample_rate))))
    if len(values) <= width:
        return 0, len(values)
    step = max(1, int(round(0.025 * sample_rate)))
    peak_rms = 1e-9
    rows = []
    for start in range(0, len(values) - width + 1, step):
        region = values[start:start + width]
        level = _rms(region)
        peak_rms = max(peak_rms, level)
        rows.append((start, level, periodicity_score(region, sample_rate)))
    ranked = [
        ((float(periodicity or 0.0) ** 2) * math.sqrt(level / peak_rms), start)
        for start, level, periodicity in rows
    ]
    start = max(ranked)[1] if ranked else 0
    return int(start), int(start + width)


def _estimate_f0(samples: np.ndarray, sample_rate: int,
                 minimum_hz: float = 60.0,
                 maximum_hz: float = 450.0) -> float | None:
    values = np.asarray(samples, np.float64)
    values = values - float(np.mean(values))
    if values.size < 128 or _rms(values) < 1e-6:
        return None
    values *= np.hanning(values.size)
    first = max(2, int(math.floor(sample_rate / maximum_hz)))
    last = min(values.size // 2, int(math.ceil(sample_rate / minimum_hz)))
    scores = []
    for lag in range(first, last + 1):
        left, right = values[:-lag], values[lag:]
        denominator = math.sqrt(
            float(np.dot(left, left)) * float(np.dot(right, right))
        ) + 1e-12
        scores.append(float(np.dot(left, right)) / denominator)
    if not scores:
        return None
    offset = int(np.argmax(scores))
    if scores[offset] < 0.22:
        return None
    threshold = max(0.22, 0.90 * float(scores[offset]))
    for candidate in range(1, len(scores) - 1):
        if (scores[candidate] >= threshold
                and scores[candidate] >= scores[candidate - 1]
                and scores[candidate] >= scores[candidate + 1]):
            offset = candidate
            break
    return float(sample_rate / (first + offset))


def _mean_spectrum(samples: np.ndarray, sample_rate: int,
                   fft_size: int = 2048,
                   hop_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, np.float64)
    if values.size < fft_size:
        values = np.pad(values, (0, fft_size - values.size))
    count = 1 + max(0, (len(values) - fft_size) // hop_size)
    window = np.blackman(fft_size)
    power = np.zeros(fft_size // 2 + 1, np.float64)
    for index in range(count):
        start = index * hop_size
        spectrum = np.fft.rfft(values[start:start + fft_size] * window)
        power += np.abs(spectrum) ** 2
    power /= max(1, count)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return frequencies, 10.0 * np.log10(np.maximum(power, 1e-16))


def _harmonic_ridge_contrast(samples: np.ndarray, sample_rate: int,
                             reference_f0: float | None) -> float | None:
    if not reference_f0 or reference_f0 <= 0.0:
        return None
    frequencies, decibels = _mean_spectrum(samples, sample_rate)
    bin_hz = frequencies[1] - frequencies[0]
    contrasts = []
    maximum = min(6000.0, float(frequencies[-1]) - reference_f0)
    for harmonic in range(2, int(maximum / reference_f0) + 1):
        frequency = harmonic * reference_f0
        center = int(round(frequency / bin_hz))
        between = int(round((frequency + 0.5 * reference_f0) / bin_hz))
        radius = max(1, int(round(0.08 * reference_f0 / bin_hz)))
        if between + radius >= len(decibels):
            break
        ridge = float(np.max(decibels[center - radius:center + radius + 1]))
        valley = float(np.median(
            decibels[between - radius:between + radius + 1]
        ))
        contrasts.append(ridge - valley)
    return float(np.median(contrasts)) if contrasts else None


def _smooth_spectrum(decibels: np.ndarray, width: int) -> np.ndarray:
    width = max(5, int(width))
    if width % 2 == 0:
        width += 1
    width = min(width, max(5, len(decibels) // 3 * 2 + 1))
    half = width // 2
    return np.convolve(
        np.pad(decibels, (half, half), mode="edge"),
        np.ones(width, np.float64) / width,
        mode="valid",
    )


def _tract_envelope_metrics(left: np.ndarray, right: np.ndarray,
                            sample_rate: int,
                            reference_f0: float | None) -> tuple[float, float]:
    frequencies, left_db = _mean_spectrum(left, sample_rate)
    _right_frequencies, right_db = _mean_spectrum(right, sample_rate)
    bin_hz = max(1e-9, float(frequencies[1] - frequencies[0]))
    smoothing_hz = max(300.0, 2.5 * float(reference_f0 or 120.0))
    width = int(round(smoothing_hz / bin_hz))
    left_envelope = _smooth_spectrum(left_db, width)
    right_envelope = _smooth_spectrum(right_db, width)
    selected = (frequencies >= 250.0) & (frequencies <= 6500.0)
    left_envelope = left_envelope[selected]
    right_envelope = right_envelope[selected]
    left_envelope -= float(np.mean(left_envelope))
    right_envelope -= float(np.mean(right_envelope))
    correlation = float(np.corrcoef(left_envelope, right_envelope)[0, 1])
    distance = float(np.sqrt(np.mean(
        (left_envelope - right_envelope) ** 2
    )))
    return correlation, distance


def analyze_voicing_variants(source: np.ndarray, partial: np.ndarray,
                             zero: np.ndarray, sample_rate: int,
                             crop: tuple[int, int]) -> dict[str, object]:
    start, end = crop
    regions = {
        "source": np.asarray(source[start:end], np.float64),
        "partial": np.asarray(partial[start:end], np.float64),
        "zero": np.asarray(zero[start:end], np.float64),
    }
    f0 = _estimate_f0(regions["source"], sample_rate)
    rows = {}
    for name, values in regions.items():
        rows[name] = {
            "periodicity": round(float(
                periodicity_score(values, sample_rate) or 0.0
            ), 6),
            "harmonic_ridge_contrast_db": (
                round(float(contrast), 6)
                if (contrast := _harmonic_ridge_contrast(
                    values, sample_rate, f0
                )) is not None else None
            ),
            "rms": round(_rms(values), 8),
        }
    correlation, distance = _tract_envelope_metrics(
        regions["source"], regions["zero"], sample_rate, f0
    )
    source_contrast = rows["source"]["harmonic_ridge_contrast_db"]
    zero_contrast = rows["zero"]["harmonic_ridge_contrast_db"]
    contrast_drop = (
        float(source_contrast) - float(zero_contrast)
        if source_contrast is not None and zero_contrast is not None else None
    )
    contrast_ratio = (
        float(zero_contrast) / max(float(source_contrast), 1e-9)
        if source_contrast is not None and zero_contrast is not None else None
    )
    return {
        "crop_start_seconds": round(start / float(sample_rate), 6),
        "crop_end_seconds": round(end / float(sample_rate), 6),
        "source_f0_hz": round(float(f0), 4) if f0 else None,
        "variants": rows,
        "zero_vs_source": {
            "harmonic_ridge_contrast_drop_db": (
                round(contrast_drop, 6) if contrast_drop is not None else None
            ),
            "harmonic_ridge_contrast_ratio": (
                round(contrast_ratio, 6) if contrast_ratio is not None else None
            ),
            "tract_envelope_correlation": round(correlation, 6),
            "tract_envelope_distance_db": round(distance, 6),
            "periodicity_removed": bool(
                rows["zero"]["periodicity"] <= 0.30
                and rows["zero"]["periodicity"]
                <= rows["source"]["periodicity"] - 0.20
            ),
            "harmonic_ridges_reduced": bool(
                contrast_drop is not None and contrast_ratio is not None
                and contrast_drop >= 1.0 and contrast_ratio <= 0.65
            ),
            "tract_envelope_retained": bool(correlation >= 0.75),
        },
    }


def _panel_image(values: np.ndarray, sample_rate: int,
                 maximum_frequency: float = 6500.0):
    _times, frequencies, decibels = spectrogram_db(
        values, sample_rate, fft_size=1024, hop_size=24
    )
    keep = frequencies <= min(maximum_frequency, frequencies[-1])
    rgb = _spectrogram_rgb(decibels[keep])
    return np.ascontiguousarray(np.flipud(rgb)), frequencies[keep]


def render_voicing_validation_image(
    variants: dict[str, np.ndarray],
    sample_rate: int,
    crop: tuple[int, int],
    metrics: dict[str, object],
    output_path: Path | str,
    *,
    title: str = "Source-filter voicing validation",
    width: int = 1800,
    height: int = 930,
) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font_family = _load_diagnostic_font(QtGui)
    image = QtGui.QImage(
        int(width), int(height), QtGui.QImage.Format_ARGB32_Premultiplied
    )
    image.fill(QtGui.QColor("#f4f5f7"))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    try:
        title_font = QtGui.QFont(font_family, 16)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#202124"))
        painter.drawText(28, 38, title)

        body = QtGui.QFont(font_family, 9)
        painter.setFont(body)
        crop_seconds = (
            (crop[1] - crop[0]) / float(sample_rate)
        )
        painter.drawText(
            28, 64,
            f"Matched {crop_seconds:.3f} s interval | 0-6.5 kHz | "
            "1024-sample STFT | all panels use the same synthesized source",
        )

        legend = QtCore.QRectF(28, 78, width - 56, 58)
        painter.fillRect(legend, QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#b6bac1"), 1))
        painter.drawRect(legend)
        painter.setPen(QtGui.QColor("#30343a"))
        painter.drawText(
            legend.adjusted(14, 7, -14, -29),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "How to read this: regularly spaced narrow horizontal ridges are voiced harmonics; "
            "broad horizontal energy bands are vocal-tract/formant resonances.",
        )
        painter.drawText(
            legend.adjusted(14, 29, -14, -7),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "At Voicing 0.00 the ridges should disappear into diffuse noise while the broad formant bands remain. "
            "White -> blue -> pink -> red means increasing energy.",
        )

        names = (
            ("source", "Measured source"),
            ("partial", "Voicing 0.50"),
            ("zero", "Voicing 0.00"),
        )
        margin = 28
        gap = 24
        panel_width = (width - 2 * margin - 2 * gap) / 3.0
        top = 190
        panel_height = 510
        start, end = crop
        metrics_rows = dict(metrics.get("variants") or {})
        for index, (key, label) in enumerate(names):
            x = margin + index * (panel_width + gap)
            values = np.asarray(variants[key][start:end], np.float64)
            rgb, _frequencies = _panel_image(values, sample_rate)
            qimage = QtGui.QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                QtGui.QImage.Format_RGB888,
            ).copy()
            rect = QtCore.QRectF(x, top, panel_width, panel_height)
            painter.fillRect(rect, QtGui.QColor("#ffffff"))
            painter.drawImage(rect, qimage)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4b5057"), 1))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect)

            heading = QtGui.QFont(font_family, 11)
            heading.setBold(True)
            painter.setFont(heading)
            painter.setPen(QtGui.QColor("#202124"))
            painter.drawText(int(x), top - 27, label)
            row = dict(metrics_rows.get(key) or {})
            painter.setFont(body)
            painter.drawText(
                int(x), top + panel_height + 22,
                f"Periodicity {row.get('periodicity')}   |   "
                f"harmonic contrast {row.get('harmonic_ridge_contrast_db')} dB",
            )
            for tick, frequency in enumerate((0, 1000, 2000, 3000, 4000, 5000, 6000)):
                fraction = frequency / 6500.0
                y = top + panel_height - fraction * panel_height
                painter.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30, 55), 1))
                painter.drawLine(int(x), int(y), int(x + panel_width), int(y))
                if index == 0:
                    painter.setPen(QtGui.QColor("#4c5156"))
                    painter.drawText(int(x + 4), int(y - 3), f"{frequency / 1000:.0f}k")

        comparison = dict(metrics.get("zero_vs_source") or {})
        result_rect = QtCore.QRectF(28, 760, width - 56, 122)
        painter.fillRect(result_rect, QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#b6bac1"), 1))
        painter.drawRect(result_rect)
        result_font = QtGui.QFont(font_family, 10)
        result_font.setBold(True)
        painter.setFont(result_font)
        painter.setPen(QtGui.QColor("#202124"))
        verdicts = (
            f"Periodicity removed: {comparison.get('periodicity_removed')}   |   "
            f"Harmonic ridges reduced: {comparison.get('harmonic_ridges_reduced')}   |   "
            f"Tract envelope retained: {comparison.get('tract_envelope_retained')}"
        )
        painter.drawText(44, 790, verdicts)
        painter.setFont(body)
        painter.drawText(
            44, 817,
            f"Harmonic-ridge contrast drop: "
            f"{comparison.get('harmonic_ridge_contrast_drop_db')} dB   |   "
            f"Tract-envelope correlation: "
            f"{comparison.get('tract_envelope_correlation')}   |   "
            f"distance: {comparison.get('tract_envelope_distance_db')} dB",
        )
        painter.setPen(QtGui.QColor("#5f6368"))
        painter.drawText(
            QtCore.QRectF(44, 835, width - 88, 39),
            QtCore.Qt.TextWordWrap,
            "This is an objective source/filter check, not a naturalness judgment. "
            "A successful zero endpoint removes periodic ridges without erasing the broad resonance pattern; "
            "the companion WAV files remain necessary for human listening.",
        )
    finally:
        painter.end()
        app.processEvents()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(destination), "PNG"):
        raise OSError(f"could not save voicing validation image: {destination}")
    return destination


def generate_voicing_validation(source_wav: Path | str,
                                output_dir: Path | str,
                                *, prefix: str = "voicing") -> dict[str, object]:
    source_path = Path(source_wav).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    _safe_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source, sample_rate = _read_pcm_wav(source_path)
    duration = len(source) / float(sample_rate)
    partial_result = transform_voicing(
        source, sample_rate, [(0.0, 0.50), (duration, 0.50)]
    )
    zero_result = transform_voicing(
        source, sample_rate, [(0.0, 0.0), (duration, 0.0)]
    )
    variants = {
        "source": np.asarray(source, np.float32),
        "partial": np.asarray(partial_result.samples, np.float32),
        "zero": np.asarray(zero_result.samples, np.float32),
    }
    crop = _best_periodic_crop(source, sample_rate)
    metrics = analyze_voicing_variants(
        variants["source"], variants["partial"], variants["zero"],
        sample_rate, crop,
    )
    filenames = {
        "source": f"{prefix}__source.wav",
        "partial": f"{prefix}__voicing_050.wav",
        "zero": f"{prefix}__voicing_000.wav",
        "image": f"{prefix}__spectrogram.png",
        "json": f"{prefix}__spectrogram.json",
    }
    for key in ("source", "partial", "zero"):
        _write_pcm_wav(output_root / filenames[key], variants[key], sample_rate)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "source_filter_voicing_spectrogram_validation",
        "source_wav": source_path.name,
        "sample_rate": sample_rate,
        "outputs": filenames,
        **metrics,
        "partial_transform": partial_result.diagnostic_dict(
            include_frames=False
        ),
        "zero_transform": zero_result.diagnostic_dict(
            include_frames=False
        ),
        "naturalness_verified": False,
    }
    render_voicing_validation_image(
        variants, sample_rate, crop, payload,
        output_root / filenames["image"],
    )
    (output_root / filenames["json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render measured source/partial/zero voicing spectrograms"
    )
    parser.add_argument("source_wav", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prefix", default="voicing")
    args = parser.parse_args(argv)
    result = generate_voicing_validation(
        args.source_wav, args.output_dir, prefix=args.prefix
    )
    comparison = dict(result.get("zero_vs_source") or {})
    print(json.dumps(comparison, sort_keys=True))
    return 0 if all((
        comparison.get("periodicity_removed"),
        comparison.get("harmonic_ridges_reduced"),
        comparison.get("tract_envelope_retained"),
    )) else 2


if __name__ == "__main__":
    raise SystemExit(main())
