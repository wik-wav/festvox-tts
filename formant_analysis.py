"""Offline F0-adaptive formant and reference-voice-space analysis.

Stage A deliberately has no production transformation code.  It measures
final audio with two independent estimators:

* an iterative cepstral true-envelope approximation informed by Roebel and
  Rodet's max-and-resmooth procedure;
* dynamically ordered Burg LPC roots and bandwidths.

Every frame is retained.  Low-confidence frames carry explicit rejection
reasons instead of disappearing from aggregate statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Iterable, Mapping, Sequence
import wave

import numpy as np

from kokoro_reference import (
    SilverUtteranceAlignment,
    sha256_file,
)


FORMANT_ANALYSIS_VERSION = "prompt20-formants-v1"
REFERENCE_VOICE_SPACE_VERSION = 1
_VOWELS = {"a", "i", "u", "e", "o"}


@dataclass(frozen=True)
class FormantAnalysisConfig:
    frame_step_seconds: float = 0.010
    minimum_frame_seconds: float = 0.040
    maximum_frame_seconds: float = 0.080
    periods_per_frame: float = 4.5
    f0_min_hz: float = 55.0
    f0_max_hz: float = 500.0
    minimum_voicing_confidence: float = 0.42
    maximum_creak_confidence: float = 0.72
    true_envelope_tolerance_db: float = 2.0
    true_envelope_iterations: int = 24
    minimum_formant_spacing_hz: float = 150.0
    maximum_analysis_hz: float = 6000.0
    preemphasis: float = 0.97
    stable_trim_fraction: float = 0.20
    maximum_frames_per_segment: int = 80


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    source_path: Path


@dataclass(frozen=True)
class AnalysisSegment:
    segment_id: str
    speaker_id: str
    audio_path: Path
    start_seconds: float
    end_seconds: float
    vowel: str = "unknown"
    phone: str = "unknown"
    transcript: str = ""
    partition: str = "source"
    recording_style: str = "unknown"
    source_corpus: str = "project"
    label_confidence: float = 1.0
    long_vowel: bool = False
    adjacent_moraic_nasal: bool = False
    phrase_final: bool = False
    probable_devoicing: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_manifest_dict(self, root: Path | None = None) -> dict[str, object]:
        path = self.audio_path
        if root is not None:
            try:
                path_text = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                path_text = path.name
        else:
            path_text = path.name
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "audio_path": path_text,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "vowel": self.vowel,
            "phone": self.phone,
            "transcript": self.transcript,
            "partition": self.partition,
            "recording_style": self.recording_style,
            "source_corpus": self.source_corpus,
            "label_confidence": self.label_confidence,
            "long_vowel": self.long_vowel,
            "adjacent_moraic_nasal": self.adjacent_moraic_nasal,
            "phrase_final": self.phrase_final,
            "probable_devoicing": self.probable_devoicing,
            "metadata": dict(self.metadata),
        }


@dataclass
class FormantFrame:
    segment_id: str
    speaker_id: str
    source_corpus: str
    partition: str
    vowel: str
    phone: str
    frame_time_seconds: float
    sample_rate: int
    f0_hz: float | None
    f0_confidence: float
    frame_power_db: float
    voicing_confidence: float
    creak_confidence: float
    cepstral_peak_prominence_db: float
    spectral_tilt_db_per_octave: float | None
    true_envelope_order: int
    lpc_order: int
    formants_hz: list[float | None]
    bandwidths_hz: list[float | None]
    formant_confidences: list[float]
    lpc_formants_hz: list[float | None]
    lpc_bandwidths_hz: list[float | None]
    estimator_disagreement_hz: list[float | None]
    formant_dispersion_hz: float | None
    apparent_vocal_tract_length_cm: float | None
    envelope_frequencies_hz: list[float]
    envelope_db: list[float]
    accepted: bool
    rejection_reasons: list[str]
    tracked_formants_hz: list[float | None] = field(default_factory=list)
    tracking_confidences: list[float] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "source_corpus": self.source_corpus,
            "partition": self.partition,
            "vowel": self.vowel,
            "phone": self.phone,
            "frame_time_seconds": self.frame_time_seconds,
            "sample_rate": self.sample_rate,
            "f0_hz": self.f0_hz,
            "f0_confidence": self.f0_confidence,
            "frame_power_db": self.frame_power_db,
            "voicing_confidence": self.voicing_confidence,
            "creak_confidence": self.creak_confidence,
            "cepstral_peak_prominence_db": self.cepstral_peak_prominence_db,
            "spectral_tilt_db_per_octave": self.spectral_tilt_db_per_octave,
            "true_envelope_order": self.true_envelope_order,
            "lpc_order": self.lpc_order,
            "formant_dispersion_hz": self.formant_dispersion_hz,
            "apparent_vocal_tract_length_cm": self.apparent_vocal_tract_length_cm,
            "accepted": self.accepted,
            "rejection_reasons": ";".join(self.rejection_reasons),
            "spectral_envelope": json.dumps({
                "frequencies_hz": self.envelope_frequencies_hz,
                "db": self.envelope_db,
            }, separators=(",", ":")),
        }
        for index in range(4):
            number = index + 1
            row[f"f{number}_hz"] = self.formants_hz[index]
            row[f"tracked_f{number}_hz"] = (
                self.tracked_formants_hz[index]
                if index < len(self.tracked_formants_hz) else None
            )
            row[f"f{number}_tracking_confidence"] = (
                self.tracking_confidences[index]
                if index < len(self.tracking_confidences) else 0.0
            )
            row[f"f{number}_bandwidth_hz"] = self.bandwidths_hz[index]
            row[f"f{number}_confidence"] = self.formant_confidences[index]
            row[f"lpc_f{number}_hz"] = self.lpc_formants_hz[index]
            row[f"lpc_f{number}_bandwidth_hz"] = self.lpc_bandwidths_hz[index]
            row[f"f{number}_estimator_disagreement_hz"] = \
                self.estimator_disagreement_hz[index]
        return row


@dataclass
class FormantSegment:
    segment: AnalysisSegment
    frames: list[FormantFrame]
    accepted_frame_count: int
    rejected_frame_count: int
    stable_start_seconds: float
    stable_end_seconds: float
    median_f0_hz: float | None
    median_formants_hz: list[float | None]
    median_bandwidths_hz: list[float | None]
    median_formant_confidences: list[float]
    median_formant_dispersion_hz: float | None
    apparent_vocal_tract_length_cm: float | None
    median_spectral_tilt_db_per_octave: float | None
    accepted: bool
    rejection_reasons: list[str]

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "segment_id": self.segment.segment_id,
            "speaker_id": self.segment.speaker_id,
            "source_corpus": self.segment.source_corpus,
            "partition": self.segment.partition,
            "recording_style": self.segment.recording_style,
            "vowel": self.segment.vowel,
            "phone": self.segment.phone,
            "transcript": self.segment.transcript,
            "start_seconds": self.segment.start_seconds,
            "end_seconds": self.segment.end_seconds,
            "stable_start_seconds": self.stable_start_seconds,
            "stable_end_seconds": self.stable_end_seconds,
            "duration_seconds": (
                self.segment.end_seconds - self.segment.start_seconds
            ),
            "label_confidence": self.segment.label_confidence,
            "long_vowel": self.segment.long_vowel,
            "adjacent_moraic_nasal": self.segment.adjacent_moraic_nasal,
            "phrase_final": self.segment.phrase_final,
            "probable_devoicing": self.segment.probable_devoicing,
            "accepted_frame_count": self.accepted_frame_count,
            "rejected_frame_count": self.rejected_frame_count,
            "median_f0_hz": self.median_f0_hz,
            "median_formant_dispersion_hz": self.median_formant_dispersion_hz,
            "apparent_vocal_tract_length_cm": (
                self.apparent_vocal_tract_length_cm
            ),
            "median_spectral_tilt_db_per_octave": (
                self.median_spectral_tilt_db_per_octave
            ),
            "accepted": self.accepted,
            "rejection_reasons": ";".join(self.rejection_reasons),
        }
        for index in range(4):
            number = index + 1
            row[f"median_f{number}_hz"] = self.median_formants_hz[index]
            row[f"median_f{number}_bandwidth_hz"] = \
                self.median_bandwidths_hz[index]
            row[f"median_f{number}_confidence"] = \
                self.median_formant_confidences[index]
        return row


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = (
            octets[:, 0].astype(np.int32)
            | (octets[:, 1].astype(np.int32) << 8)
            | (octets[:, 2].astype(np.int32) << 16)
        )
        integers = np.where(integers & 0x800000, integers - 0x1000000,
                            integers)
        values = integers.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return np.asarray(values, dtype=np.float64)


def _read_wav(path: Path) -> AudioData:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        compression = source.getcomptype()
        if compression != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {compression}")
        raw = source.readframes(source.getnframes())
    return AudioData(
        samples=_decode_pcm(raw, sample_width, channels),
        sample_rate=sample_rate,
        channels=channels,
        source_path=path,
    )


def _find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if executable:
        return executable
    candidate = Path(
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\ffmpeg.EXE"
    )
    return str(candidate) if candidate.is_file() else None


def read_audio(
    path: Path | str, *, expected_sample_rate: int | None = None
) -> AudioData:
    source_path = Path(path).resolve()
    if source_path.suffix.casefold() == ".wav":
        return _read_wav(source_path)
    executable = _find_ffmpeg()
    if not executable:
        raise RuntimeError(
            f"decoding {source_path.suffix} requires ffmpeg on PATH"
        )
    sample_rate = int(expected_sample_rate or 22050)
    process = subprocess.run(
        [
            executable, "-v", "error", "-i", str(source_path),
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
            "-ar", str(sample_rate), "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"ffmpeg could not decode {source_path.name}: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )
    samples = np.frombuffer(process.stdout, dtype="<i2").astype(np.float64)
    samples /= 32768.0
    return AudioData(samples, sample_rate, 1, source_path)


def _frame_power_db(frame: np.ndarray) -> float:
    return 10.0 * math.log10(max(1e-12, float(np.mean(frame * frame))))


def estimate_f0(
    frame: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    minimum_hz: float = 55.0,
    maximum_hz: float = 500.0,
) -> tuple[float | None, float, float]:
    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    values = values - float(np.mean(values)) if values.size else values
    energy = float(np.dot(values, values))
    if values.size < 8 or energy < 1e-10:
        return None, 0.0, 0.0
    windowed = values * np.hanning(values.size)
    nfft = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(windowed, nfft)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), nfft)
    correlation = correlation[:values.size]
    minimum_lag = max(1, int(sample_rate / maximum_hz))
    maximum_lag = min(values.size - 2, int(sample_rate / minimum_hz))
    if maximum_lag <= minimum_lag:
        return None, 0.0, 0.0
    lags = np.arange(minimum_lag, maximum_lag + 1)
    scores = np.empty(lags.size, dtype=np.float64)
    for position, lag in enumerate(lags):
        left = windowed[:-lag]
        right = windowed[lag:]
        denominator = math.sqrt(max(
            1e-18, float(np.dot(left, left) * np.dot(right, right))
        ))
        scores[position] = float(np.dot(left, right)) / denominator
    local_peaks = np.flatnonzero(
        (scores >= np.r_[scores[0], scores[:-1]])
        & (scores >= np.r_[scores[1:], scores[-1]])
    )
    if local_peaks.size:
        strong = local_peaks[scores[local_peaks] >= max(
            0.28, 0.82 * float(np.max(scores[local_peaks]))
        )]
        selected = int(strong[0] if strong.size else
                       local_peaks[np.argmax(scores[local_peaks])])
    else:
        selected = int(np.argmax(scores))
    lag = float(lags[selected])
    if 0 < selected < scores.size - 1:
        left, center, right = scores[selected - 1:selected + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            lag += 0.5 * (left - right) / denominator
    confidence = float(np.clip(scores[selected], 0.0, 1.0))
    # Adjacent samples describe the same autocorrelation peak and exact
    # integer multiples describe the same pulse train. Neither is evidence
    # of creak or an alternate F0 hypothesis.
    selected_lag = max(1.0, float(lags[selected]))
    alternatives = []
    for candidate in local_peaks:
        if int(candidate) == selected:
            continue
        candidate_lag = max(1.0, float(lags[candidate]))
        ratio = max(candidate_lag, selected_lag) / min(
            candidate_lag, selected_lag
        )
        nearest_multiple = max(1.0, round(ratio))
        if abs(ratio - nearest_multiple) / nearest_multiple <= 0.08:
            continue
        alternatives.append(float(scores[candidate]))
    second = max(alternatives, default=0.0)
    ambiguity = float(np.clip(second / max(confidence, 1e-9), 0.0, 1.0))
    return sample_rate / max(lag, 1e-9), confidence, ambiguity


def _cepstral_peak_prominence(
    frame: np.ndarray, sample_rate: int, config: FormantAnalysisConfig
) -> float:
    nfft = 1 << max(10, (frame.size * 2 - 1).bit_length())
    magnitude = np.maximum(
        1e-12, np.abs(np.fft.rfft(frame * np.hanning(frame.size), nfft))
    )
    cepstrum = np.fft.irfft(np.log(magnitude), nfft)
    lower = max(1, int(sample_rate / config.f0_max_hz))
    upper = min(cepstrum.size - 1, int(sample_rate / config.f0_min_hz))
    if upper <= lower:
        return 0.0
    region = cepstrum[lower:upper + 1]
    baseline = float(np.median(region))
    scale = max(1e-9, float(np.median(np.abs(region - baseline))))
    return float((np.max(region) - baseline) / scale)


def _spectral_tilt(
    frame: np.ndarray, sample_rate: int
) -> float | None:
    nfft = 1 << max(10, (frame.size - 1).bit_length())
    magnitude = np.maximum(
        1e-12, np.abs(np.fft.rfft(frame * np.hanning(frame.size), nfft))
    )
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    mask = (frequencies >= 300.0) & (frequencies <= min(5000.0,
                                                        sample_rate * 0.45))
    if np.count_nonzero(mask) < 8:
        return None
    x = np.log2(frequencies[mask] / 300.0)
    y = 20.0 * np.log10(magnitude[mask])
    return float(np.polyfit(x, y, 1)[0])


def _cepstral_smooth(log_magnitude: np.ndarray, order: int) -> np.ndarray:
    full_length = (log_magnitude.size - 1) * 2
    cepstrum = np.fft.irfft(log_magnitude, full_length)
    keep = min(order, full_length // 2 - 1)
    lifter = np.zeros(full_length, dtype=np.float64)
    if keep > 0:
        positions = np.arange(keep + 1, dtype=np.float64)
        taper = 0.54 + 0.46 * np.cos(math.pi * positions / (keep + 1.0))
        lifter[:keep + 1] = taper
        lifter[-keep:] = taper[1:][::-1]
    lifter[0] = 1.0
    return np.fft.rfft(cepstrum * lifter, full_length).real


def true_envelope(
    frame: Sequence[float] | np.ndarray,
    sample_rate: int,
    f0_hz: float | None,
    config: FormantAnalysisConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    cfg = config or FormantAnalysisConfig()
    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    nfft = 1 << max(11, (values.size * 2 - 1).bit_length())
    magnitude = np.maximum(
        1e-10, np.abs(np.fft.rfft(values * np.hanning(values.size), nfft))
    )
    log_magnitude = 20.0 * np.log10(magnitude)
    reference_f0 = float(f0_hz or 180.0)
    order = int(round(sample_rate / (2.0 * max(55.0, reference_f0))))
    order = max(12, min(96, order, nfft // 16))
    adjusted = log_magnitude.copy()
    envelope = _cepstral_smooth(adjusted, order)
    iterations = 0
    for iterations in range(1, cfg.true_envelope_iterations + 1):
        peak_error = float(np.max(log_magnitude - envelope))
        if peak_error <= cfg.true_envelope_tolerance_db:
            break
        adjusted = np.maximum(adjusted, envelope)
        envelope = _cepstral_smooth(adjusted, order)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    return frequencies, envelope, order, iterations


def _peak_candidates(
    frequencies: np.ndarray,
    envelope_db: np.ndarray,
    sample_rate: int,
    config: FormantAnalysisConfig,
) -> list[tuple[float, float, float]]:
    maximum = min(config.maximum_analysis_hz, sample_rate * 0.46)
    mask = (frequencies >= 120.0) & (frequencies <= maximum)
    indices = np.flatnonzero(mask)
    if indices.size < 5:
        return []
    smooth_width = max(1, int(round(35.0 /
                                    (frequencies[1] - frequencies[0]))))
    kernel = np.ones(smooth_width * 2 + 1, dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = np.convolve(envelope_db, kernel, mode="same")
    # A formant can be a broad shoulder rather than a strict maximum on the
    # glottal source's downward spectral slope.  Remove only a much broader
    # trend before finding resonance-shaped curvature; retain the original
    # true-envelope level for diagnostics.
    trend_half = max(2, int(round(650.0 /
                                (frequencies[1] - frequencies[0]))))
    trend_kernel = np.ones(trend_half * 2 + 1, dtype=np.float64)
    trend_kernel /= trend_kernel.sum()
    padded = np.pad(smoothed, (trend_half, trend_half), mode="reflect")
    trend = np.convolve(padded, trend_kernel, mode="valid")
    residual = smoothed - trend
    local = indices[
        (residual[indices] >= residual[indices - 1])
        & (residual[indices] > residual[indices + 1])
    ]
    candidates: list[tuple[float, float, float]] = []
    span = max(2, int(round(180.0 /
                            (frequencies[1] - frequencies[0]))))
    for index in local:
        left = max(indices[0], index - span)
        right = min(indices[-1], index + span)
        valley = max(float(np.min(residual[left:index + 1])),
                     float(np.min(residual[index:right + 1])))
        prominence = float(residual[index] - valley)
        level = float(smoothed[index])
        candidates.append((float(frequencies[index]), prominence, level))
    return candidates


def _select_true_formants(
    candidates: Sequence[tuple[float, float, float]],
    sample_rate: int,
    config: FormantAnalysisConfig,
    anchors: Sequence[float | None] | None = None,
) -> tuple[list[float | None], list[float | None], list[float]]:
    upper = min(config.maximum_analysis_hz, sample_rate * 0.46)
    # Deliberately overlapping search supports transformed and child-like
    # tracts. The earlier adult-only ceilings made an upward-shifted F2/F3
    # impossible to observe and the 2.8 kHz F4 floor made a lengthened F4
    # impossible. Ordering and LPC/temporal evidence disambiguate the overlap.
    bands = (
        (120.0, min(1800.0, upper)),
        (350.0, min(3800.0, upper)),
        (900.0, min(5200.0, upper)),
        (1600.0, upper),
    )
    selected: list[float | None] = []
    bandwidths: list[float | None] = []
    confidences: list[float] = []
    previous = 0.0
    for formant_index, (lower, higher) in enumerate(bands):
        valid = [item for item in candidates
                 if max(lower, previous + config.minimum_formant_spacing_hz)
                 <= item[0] <= higher]
        anchor = (anchors[formant_index]
                  if anchors is not None and formant_index < len(anchors)
                  else None)
        if valid and anchor is not None:
            radius = max(260.0, 0.22 * float(anchor))
            anchored = [item for item in valid
                        if abs(item[0] - float(anchor)) <= radius]
            # Never relabel a distant higher resonance merely because an LPC
            # anchor has no matching envelope peak. The pairing stage can keep
            # the LPC estimate, with an explicit missing-crosscheck flag.
            valid = anchored
        if not valid:
            selected.append(None)
            bandwidths.append(None)
            confidences.append(0.0)
            continue
        if anchor is None:
            frequency, prominence, _level = max(
                valid, key=lambda item: (item[1], -item[0])
            )
        else:
            frequency, prominence, _level = min(
                valid,
                key=lambda item: (
                    abs(item[0] - float(anchor)), -item[1], item[0]
                ),
            )
        selected.append(frequency)
        bandwidths.append(max(35.0, min(700.0, 260.0 / max(prominence, 0.5))))
        confidences.append(float(np.clip(prominence / 2.5, 0.0, 1.0)))
        previous = frequency
    return selected, bandwidths, confidences


def _burg_coefficients(values: np.ndarray, order: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size <= order + 2:
        raise ValueError("frame is too short for requested Burg order")
    forward = signal[1:].copy()
    backward = signal[:-1].copy()
    coefficients = np.array([1.0], dtype=np.float64)
    for _index in range(order):
        denominator = float(np.dot(forward, forward) +
                            np.dot(backward, backward))
        if denominator <= 1e-14:
            break
        reflection = float(-2.0 * np.dot(backward, forward) / denominator)
        reflection = max(-0.999, min(0.999, reflection))
        coefficients = (
            np.r_[coefficients, 0.0]
            + reflection * np.r_[0.0, coefficients[::-1]]
        )
        next_forward = forward + reflection * backward
        next_backward = backward + reflection * forward
        if next_forward.size <= 2:
            break
        forward = next_forward[1:]
        backward = next_backward[:-1]
    return coefficients


def burg_formants(
    frame: Sequence[float] | np.ndarray,
    sample_rate: int,
    f0_hz: float | None,
    config: FormantAnalysisConfig | None = None,
) -> tuple[list[float | None], list[float | None], int]:
    cfg = config or FormantAnalysisConfig()
    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    emphasized = values.copy()
    if emphasized.size > 1:
        emphasized[1:] -= cfg.preemphasis * values[:-1]
    sample_rate_order = int(round(2.0 + sample_rate / 1000.0))
    f0_order_limit = int(round(sample_rate /
                               (2.25 * max(55.0, float(f0_hz or 180.0)))))
    order = max(10, min(48, sample_rate_order, f0_order_limit,
                        emphasized.size // 4))
    coefficients = _burg_coefficients(emphasized * np.hanning(emphasized.size),
                                      order)
    roots = np.roots(coefficients)
    roots = roots[np.imag(roots) >= 0.0]
    frequencies = np.angle(roots) * sample_rate / (2.0 * math.pi)
    bandwidths = -sample_rate / math.pi * np.log(np.maximum(
        1e-12, np.abs(roots)
    ))
    rows = sorted(
        (float(frequency), float(bandwidth))
        for frequency, bandwidth in zip(frequencies, bandwidths)
        if (90.0 <= frequency <= min(cfg.maximum_analysis_hz,
                                     sample_rate * 0.46)
            and 20.0 <= bandwidth <= 900.0)
    )
    selected: list[float | None] = []
    selected_bandwidths: list[float | None] = []
    previous = 0.0
    for frequency, bandwidth in rows:
        if frequency < previous + cfg.minimum_formant_spacing_hz:
            continue
        selected.append(frequency)
        selected_bandwidths.append(bandwidth)
        previous = frequency
        if len(selected) == 4:
            break
    while len(selected) < 4:
        selected.append(None)
        selected_bandwidths.append(None)
    return selected, selected_bandwidths, order


def _pair_estimators(
    true_formants: Sequence[float | None],
    true_bandwidths: Sequence[float | None],
    true_confidences: Sequence[float],
    lpc_formants: Sequence[float | None],
    lpc_bandwidths: Sequence[float | None],
) -> tuple[list[float | None], list[float | None], list[float],
           list[float | None], list[str]]:
    formants: list[float | None] = []
    bandwidths: list[float | None] = []
    confidences: list[float] = []
    disagreement: list[float | None] = []
    flags: list[str] = []
    for index in range(4):
        true_value = true_formants[index]
        lpc_value = lpc_formants[index]
        if true_value is not None and lpc_value is not None:
            difference = abs(true_value - lpc_value)
            tolerance = max(180.0, true_value * 0.16)
            agreement = math.exp(-difference / tolerance)
            # Agreement with an independently fitted all-pole model is useful
            # evidence even when a broad true-envelope shoulder has modest
            # peak prominence.  Keep both raw estimates and disagreement.
            confidence = 0.40 * float(true_confidences[index]) + \
                0.60 * agreement
            formants.append(float(true_value))
            bandwidths.append(true_bandwidths[index])
            confidences.append(confidence)
            disagreement.append(difference)
            if difference > tolerance:
                flags.append(f"f{index + 1}_lpc_true_envelope_disagreement")
        elif true_value is not None:
            formants.append(float(true_value))
            bandwidths.append(true_bandwidths[index])
            confidences.append(float(true_confidences[index]) * 0.45)
            disagreement.append(None)
            flags.append(f"f{index + 1}_missing_lpc_crosscheck")
        elif lpc_value is not None:
            formants.append(float(lpc_value))
            bandwidths.append(lpc_bandwidths[index])
            confidences.append(0.25)
            disagreement.append(None)
            flags.append(f"f{index + 1}_lpc_fallback")
        else:
            formants.append(None)
            bandwidths.append(None)
            confidences.append(0.0)
            disagreement.append(None)
            flags.append(f"f{index + 1}_unavailable")
    return formants, bandwidths, confidences, disagreement, flags


def _dispersion(formants: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in formants if value is not None]
    if len(finite) < 3:
        return None
    positions = np.arange(1, len(finite) + 1, dtype=np.float64)
    return float(np.polyfit(positions, np.asarray(finite), 1)[0])


def _apparent_vtl_cm(dispersion_hz: float | None) -> float | None:
    if dispersion_hz is None or dispersion_hz <= 0.0:
        return None
    return 34300.0 / (2.0 * dispersion_hz)


def analyze_frame(
    frame: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    segment: AnalysisSegment,
    frame_time_seconds: float,
    config: FormantAnalysisConfig | None = None,
) -> FormantFrame:
    cfg = config or FormantAnalysisConfig()
    values = np.asarray(frame, dtype=np.float64).reshape(-1)
    power_db = _frame_power_db(values)
    f0_hz, f0_confidence, f0_ambiguity = estimate_f0(
        values, sample_rate,
        minimum_hz=cfg.f0_min_hz,
        maximum_hz=cfg.f0_max_hz,
    )
    cpp = _cepstral_peak_prominence(values, sample_rate, cfg)
    voicing = float(np.clip(
        0.65 * f0_confidence + 0.35 * min(1.0, cpp / 8.0), 0.0, 1.0
    ))
    creak = float(np.clip(
        0.65 * f0_ambiguity + 0.35 * max(0.0, 0.55 - f0_confidence) / 0.55,
        0.0, 1.0,
    ))
    frequencies, envelope, envelope_order, _iterations = true_envelope(
        values, sample_rate, f0_hz, cfg
    )
    try:
        lpc_values, lpc_bandwidths, lpc_order = burg_formants(
            values, sample_rate, f0_hz, cfg
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        lpc_values = [None] * 4
        lpc_bandwidths = [None] * 4
        lpc_order = 0
    candidates = _peak_candidates(frequencies, envelope, sample_rate, cfg)
    true_values, true_bandwidths, true_confidences = _select_true_formants(
        candidates, sample_rate, cfg, anchors=lpc_values
    )
    (formants, bandwidths, confidences, disagreement,
     estimator_flags) = _pair_estimators(
        true_values, true_bandwidths, true_confidences,
        lpc_values, lpc_bandwidths,
    )
    reasons: list[str] = []
    if power_db < -55.0:
        reasons.append("frame_power_too_low")
    if f0_hz is None:
        reasons.append("f0_unavailable")
    if voicing < cfg.minimum_voicing_confidence:
        reasons.append("devoiced_or_unreliably_voiced")
    if segment.probable_devoicing:
        reasons.append("labelled_probable_devoicing")
    if creak > cfg.maximum_creak_confidence:
        reasons.append("creak_or_period_ambiguity")
    if f0_hz is not None and f0_hz > 320.0:
        reasons.append("high_f0_sparse_harmonic_sampling")
    reasons.extend(estimator_flags)
    for index, (value, bandwidth, confidence) in enumerate(zip(
            formants, bandwidths, confidences), 1):
        if value is None:
            continue
        if value > sample_rate * 0.44:
            reasons.append(f"f{index}_near_nyquist")
        if bandwidth is None or not 25.0 <= bandwidth <= 750.0:
            reasons.append(f"f{index}_implausible_bandwidth")
        if confidence < 0.18:
            reasons.append(f"f{index}_low_confidence")
    critical = {
        "frame_power_too_low", "f0_unavailable",
        "devoiced_or_unreliably_voiced", "labelled_probable_devoicing",
        "creak_or_period_ambiguity",
    }
    accepted = (
        not any(reason in critical for reason in reasons)
        and sum(value is not None for value in formants) >= 3
        and sum(value >= 0.18 for value in confidences) >= 2
    )
    dispersion = _dispersion(formants)
    downsample_indices = np.linspace(
        0, frequencies.size - 1, min(128, frequencies.size), dtype=int
    )
    return FormantFrame(
        segment_id=segment.segment_id,
        speaker_id=segment.speaker_id,
        source_corpus=segment.source_corpus,
        partition=segment.partition,
        vowel=segment.vowel,
        phone=segment.phone,
        frame_time_seconds=round(float(frame_time_seconds), 6),
        sample_rate=sample_rate,
        f0_hz=round(float(f0_hz), 6) if f0_hz is not None else None,
        f0_confidence=round(float(f0_confidence), 6),
        frame_power_db=round(power_db, 6),
        voicing_confidence=round(voicing, 6),
        creak_confidence=round(creak, 6),
        cepstral_peak_prominence_db=round(cpp, 6),
        spectral_tilt_db_per_octave=(
            round(float(tilt), 6)
            if (tilt := _spectral_tilt(values, sample_rate)) is not None
            else None
        ),
        true_envelope_order=envelope_order,
        lpc_order=lpc_order,
        formants_hz=[round(value, 6) if value is not None else None
                     for value in formants],
        bandwidths_hz=[round(value, 6) if value is not None else None
                       for value in bandwidths],
        formant_confidences=[round(float(value), 6) for value in confidences],
        lpc_formants_hz=[round(value, 6) if value is not None else None
                         for value in lpc_values],
        lpc_bandwidths_hz=[round(value, 6) if value is not None else None
                           for value in lpc_bandwidths],
        estimator_disagreement_hz=[
            round(value, 6) if value is not None else None
            for value in disagreement
        ],
        formant_dispersion_hz=(round(dispersion, 6)
                              if dispersion is not None else None),
        apparent_vocal_tract_length_cm=(
            round(float(vtl), 6)
            if (vtl := _apparent_vtl_cm(dispersion)) is not None else None
        ),
        envelope_frequencies_hz=[
            round(float(frequencies[index]), 3) for index in downsample_indices
        ],
        envelope_db=[
            round(float(envelope[index]), 4) for index in downsample_indices
        ],
        accepted=accepted,
        rejection_reasons=sorted(set(reasons)),
        tracked_formants_hz=[
            round(value, 6) if value is not None else None
            for value in formants
        ],
        tracking_confidences=[
            round(float(value), 6) for value in confidences
        ],
    )


def _finite_median(values: Iterable[float | None]) -> float | None:
    finite = np.asarray(
        [float(value) for value in values
         if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.median(finite)) if finite.size else None


def smooth_formant_track(
    values: Sequence[float | None],
    confidences: Sequence[float] | None = None,
    *,
    radius: int = 2,
    local_outlier_cents: float = 280.0,
    segment_outlier_cents: float = 600.0,
) -> tuple[list[float | None], list[float], list[bool]]:
    """Track one formant without discarding the raw frame estimates.

    Formant frequency is perceptually logarithmic. A Hampel-like local gate
    removes single-frame shoulder/harmonic substitutions, while the segment
    median catches short alternate tracks that are locally self-consistent.
    Short gaps are interpolated before a confidence-weighted 50 ms triangular
    smoother. This is intended for the stable interior of one labelled vowel,
    not for unconstrained whole-utterance formant tracking.
    """
    count = len(values)
    raw = np.full(count, np.nan, np.float64)
    for index, value in enumerate(values):
        if value is not None and math.isfinite(float(value)) and value > 0.0:
            raw[index] = float(value)
    supplied_confidence = list(confidences or ())
    confidence = np.asarray([
        max(0.0, min(1.0, float(supplied_confidence[index])))
        if index < len(supplied_confidence) else 0.5
        for index in range(count)
    ], np.float64)
    finite = np.isfinite(raw)
    if np.count_nonzero(finite) < 3:
        return (
            [float(value) if math.isfinite(value) else None for value in raw],
            [float(value) if ok else 0.0
             for value, ok in zip(confidence, finite)],
            [False] * count,
        )
    log_values = np.log2(raw)
    segment_center = float(np.nanmedian(log_values))
    outliers = np.zeros(count, dtype=bool)
    radius = max(1, int(radius))
    for index in np.flatnonzero(finite):
        first = max(0, index - radius)
        last = min(count, index + radius + 1)
        neighborhood = log_values[first:last]
        local = float(np.nanmedian(neighborhood))
        local_difference = abs(1200.0 * (log_values[index] - local))
        segment_difference = abs(1200.0 * (
            log_values[index] - segment_center
        ))
        outliers[index] = (
            local_difference > float(local_outlier_cents)
            or segment_difference > float(segment_outlier_cents)
        )
    good = finite & ~outliers
    if np.count_nonzero(good) < 2:
        good = finite.copy()
        outliers[:] = False
    positions = np.arange(count, dtype=np.float64)
    reconstructed = np.full(count, np.nan, np.float64)
    reconstructed[good] = log_values[good]
    # Fill only interior gaps of at most two frames. Long unreliable spans stay
    # absent instead of being presented as measured data.
    good_indices = np.flatnonzero(good)
    for left, right in zip(good_indices, good_indices[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= 2:
            reconstructed[left + 1:right] = np.interp(
                positions[left + 1:right],
                [float(left), float(right)],
                [log_values[left], log_values[right]],
            )
    reconstructed[good] = log_values[good]
    tracking_confidence = confidence.copy()
    tracking_confidence[outliers] *= 0.25
    for index in range(count):
        if np.isfinite(reconstructed[index]) and not good[index]:
            tracking_confidence[index] = max(
                0.15, tracking_confidence[index]
            )
    kernel = np.asarray([1, 2, 3, 2, 1], np.float64)
    smoothed = np.full(count, np.nan, np.float64)
    for index in range(count):
        if not np.isfinite(reconstructed[index]):
            continue
        first = max(0, index - 2)
        last = min(count, index + 3)
        kernel_first = 2 - (index - first)
        kernel_last = kernel_first + (last - first)
        usable = np.isfinite(reconstructed[first:last])
        weights = kernel[kernel_first:kernel_last] * np.maximum(
            0.10, tracking_confidence[first:last]
        )
        weights *= usable
        if float(np.sum(weights)) > 0.0:
            smoothed[index] = float(np.sum(
                reconstructed[first:last] * weights
            ) / np.sum(weights))
    result = [
        float(2.0 ** value) if math.isfinite(value) else None
        for value in smoothed
    ]
    return result, [float(value) for value in tracking_confidence], [
        bool(value) for value in outliers
    ]


def _track_formant_frames(
    frames: Sequence[FormantFrame],
    config: FormantAnalysisConfig,
) -> None:
    """Populate temporally tracked F1-F4 fields in-place."""
    tracks: list[list[float | None]] = []
    track_confidences: list[list[float]] = []
    outlier_masks: list[list[bool]] = []
    for formant_index in range(4):
        values = [frame.formants_hz[formant_index] for frame in frames]
        confidences = [frame.formant_confidences[formant_index]
                       for frame in frames]
        track, confidence, outliers = smooth_formant_track(
            values, confidences
        )
        tracks.append(track)
        track_confidences.append(confidence)
        outlier_masks.append(outliers)
    for frame_index, frame in enumerate(frames):
        tracked = [tracks[index][frame_index] for index in range(4)]
        tracked_confidence = [
            track_confidences[index][frame_index] for index in range(4)
        ]
        # Do not display a crossing as a valid tract configuration. Retain the
        # raw estimates and flag the lower-confidence tracked observation.
        previous = None
        for formant_index, value in enumerate(tracked):
            if value is None:
                continue
            if (previous is not None and
                    value < previous + config.minimum_formant_spacing_hz):
                tracked[formant_index] = None
                tracked_confidence[formant_index] = 0.0
                frame.rejection_reasons.append(
                    f"f{formant_index + 1}_tracking_order_conflict"
                )
                continue
            previous = value
            if outlier_masks[formant_index][frame_index]:
                frame.rejection_reasons.append(
                    f"f{formant_index + 1}_raw_tracking_outlier"
                )
        frame.tracked_formants_hz = [
            round(float(value), 6) if value is not None else None
            for value in tracked
        ]
        frame.tracking_confidences = [
            round(float(value), 6) for value in tracked_confidence
        ]
        frame.rejection_reasons = sorted(set(frame.rejection_reasons))


def analyze_segment(
    segment: AnalysisSegment,
    *,
    audio: AudioData | None = None,
    config: FormantAnalysisConfig | None = None,
) -> FormantSegment:
    cfg = config or FormantAnalysisConfig()
    data = audio or read_audio(
        segment.audio_path,
        expected_sample_rate=(22050 if segment.audio_path.suffix.casefold()
                              == ".flac" else None),
    )
    duration = data.samples.size / float(data.sample_rate)
    start = max(0.0, min(duration, float(segment.start_seconds)))
    end = max(start, min(duration, float(segment.end_seconds)))
    reasons: list[str] = []
    if end - start < cfg.minimum_frame_seconds:
        reasons.append("stable_vowel_interval_too_short")
    trim = min(
        (end - start) * cfg.stable_trim_fraction,
        max(0.0, (end - start - cfg.minimum_frame_seconds) / 2.0),
    )
    stable_start = start + trim
    stable_end = end - trim
    center_start = int(round(stable_start * data.sample_rate))
    center_end = int(round(stable_end * data.sample_rate))
    region = data.samples[center_start:center_end]
    initial_window = int(round(0.060 * data.sample_rate))
    center = region.size // 2
    initial = region[max(0, center - initial_window // 2):
                     min(region.size, center + initial_window // 2)]
    initial_f0, _confidence, _ambiguity = estimate_f0(
        initial, data.sample_rate,
        minimum_hz=cfg.f0_min_hz,
        maximum_hz=cfg.f0_max_hz,
    )
    frame_seconds = max(
        cfg.minimum_frame_seconds,
        cfg.periods_per_frame / max(cfg.f0_min_hz, float(initial_f0 or 180.0)),
    )
    frame_seconds = min(cfg.maximum_frame_seconds, frame_seconds)
    frame_length = max(16, int(round(frame_seconds * data.sample_rate)))
    half = frame_length // 2
    first_center = center_start + half
    last_center = center_end - (frame_length - half)
    if last_center < first_center:
        centers = np.asarray([(center_start + center_end) // 2], dtype=int)
    else:
        step = max(1, int(round(cfg.frame_step_seconds * data.sample_rate)))
        centers = np.arange(first_center, last_center + 1, step, dtype=int)
    if centers.size > cfg.maximum_frames_per_segment:
        positions = np.linspace(0, centers.size - 1,
                                cfg.maximum_frames_per_segment, dtype=int)
        centers = centers[positions]
    frames: list[FormantFrame] = []
    for frame_center in centers:
        frame_start = frame_center - half
        frame = np.zeros(frame_length, dtype=np.float64)
        source_start = max(0, frame_start)
        source_end = min(data.samples.size, frame_start + frame_length)
        destination_start = source_start - frame_start
        frame[destination_start:destination_start + source_end - source_start] = \
            data.samples[source_start:source_end]
        frames.append(analyze_frame(
            frame,
            data.sample_rate,
            segment=segment,
            frame_time_seconds=frame_center / float(data.sample_rate),
            config=cfg,
        ))
    _track_formant_frames(frames, cfg)
    accepted_frames = [frame for frame in frames if frame.accepted]
    # Preserve raw frame jumps as diagnostics. The tracked fields carry the
    # continuity-corrected trajectory, so one bad shoulder pick no longer
    # invalidates two otherwise usable frames.
    for previous, current in zip(accepted_frames, accepted_frames[1:]):
        delta_time = max(1e-6, current.frame_time_seconds -
                         previous.frame_time_seconds)
        for index, (left, right) in enumerate(zip(
                previous.formants_hz, current.formants_hz), 1):
            if left is None or right is None:
                continue
            threshold = max(320.0, 0.20 * min(left, right)) * \
                max(1.0, delta_time / 0.010)
            if abs(right - left) > threshold:
                flag = f"f{index}_implausible_trajectory_jump"
                previous.rejection_reasons.append(flag)
                current.rejection_reasons.append(flag)
                previous.rejection_reasons = sorted(set(
                    previous.rejection_reasons))
                current.rejection_reasons = sorted(set(
                    current.rejection_reasons))
    median_formants = [
        _finite_median(
            (frame.tracked_formants_hz[index]
             if index < len(frame.tracked_formants_hz)
             else frame.formants_hz[index])
            for frame in accepted_frames
        )
        for index in range(4)
    ]
    median_bandwidths = [
        _finite_median(frame.bandwidths_hz[index] for frame in accepted_frames)
        for index in range(4)
    ]
    median_confidences = [
        _finite_median(frame.formant_confidences[index]
                       for frame in accepted_frames) or 0.0
        for index in range(4)
    ]
    minimum_accepted = max(2, math.ceil(len(frames) * 0.25))
    accepted = len(accepted_frames) >= minimum_accepted
    if not accepted:
        reasons.append("insufficient_reliable_formant_frames")
    dispersion = _dispersion(median_formants)
    return FormantSegment(
        segment=segment,
        frames=frames,
        accepted_frame_count=len(accepted_frames),
        rejected_frame_count=len(frames) - len(accepted_frames),
        stable_start_seconds=round(stable_start, 6),
        stable_end_seconds=round(stable_end, 6),
        median_f0_hz=(
            round(float(value), 6)
            if (value := _finite_median(
                frame.f0_hz for frame in accepted_frames
            )) is not None else None
        ),
        median_formants_hz=[
            round(value, 6) if value is not None else None
            for value in median_formants
        ],
        median_bandwidths_hz=[
            round(value, 6) if value is not None else None
            for value in median_bandwidths
        ],
        median_formant_confidences=[
            round(float(value), 6) for value in median_confidences
        ],
        median_formant_dispersion_hz=(
            round(dispersion, 6) if dispersion is not None else None
        ),
        apparent_vocal_tract_length_cm=(
            round(float(vtl), 6)
            if (vtl := _apparent_vtl_cm(dispersion)) is not None else None
        ),
        median_spectral_tilt_db_per_octave=(
            round(float(value), 6)
            if (value := _finite_median(
                frame.spectral_tilt_db_per_octave
                for frame in accepted_frames
            )) is not None else None
        ),
        accepted=accepted,
        rejection_reasons=sorted(set(reasons)),
    )


def analyze_segments(
    segments: Sequence[AnalysisSegment],
    *,
    config: FormantAnalysisConfig | None = None,
) -> tuple[FormantSegment, ...]:
    cfg = config or FormantAnalysisConfig()
    cache: dict[Path, AudioData] = {}
    results: list[FormantSegment] = []
    for segment in segments:
        path = segment.audio_path.resolve()
        if path not in cache:
            cache[path] = read_audio(
                path,
                expected_sample_rate=(22050 if path.suffix.casefold() == ".flac"
                                      else None),
            )
        results.append(analyze_segment(segment, audio=cache[path], config=cfg))
    return tuple(results)


def segments_from_kokoro_alignment(
    record,
    alignment: SilverUtteranceAlignment,
    audio_path: Path | str,
) -> tuple[AnalysisSegment, ...]:
    phones = list(alignment.phones)
    result: list[AnalysisSegment] = []
    for index, phone in enumerate(phones):
        if phone.phone not in _VOWELS:
            continue
        previous = phones[index - 1].phone if index else "pau"
        following = phones[index + 1].phone if index + 1 < len(phones) else "pau"
        result.append(AnalysisSegment(
            segment_id=f"kokoro:{record.utterance_id}:{phone.index}",
            speaker_id="kokoro_xlarge_speaker",
            audio_path=Path(audio_path).resolve(),
            start_seconds=phone.start_seconds,
            end_seconds=phone.end_seconds,
            vowel=phone.phone,
            phone=phone.phone,
            transcript=record.transcript,
            partition=record.partition,
            recording_style="audiobook",
            source_corpus="Kokoro-Speech-Dataset-v1.3-xlarge",
            label_confidence=phone.confidence,
            long_vowel=phone.long_vowel,
            adjacent_moraic_nasal=(previous == "N" or following == "N"),
            phrase_final=following == "pau",
            probable_devoicing=phone.probable_devoicing,
            metadata={
                "alignment_method": alignment.method,
                "utterance_alignment_confidence": alignment.confidence,
                "mora_index": phone.mora_index,
                "phrase_index": phone.phrase_index,
            },
        ))
    return tuple(result)


def source_voice_segments(
    source_root: Path | str,
    diphone_index_path: Path | str,
    *,
    speaker_id: str = "project_source_speaker",
    maximum_per_vowel: int = 48,
) -> tuple[AnalysisSegment, ...]:
    root = Path(source_root).resolve()
    payload = json.loads(Path(diphone_index_path).read_text(encoding="utf-8"))
    alternatives = payload.get("alternatives") or {}
    candidates: list[AnalysisSegment] = []
    seen: set[tuple[str, float, float, str]] = set()
    for diphone, choices in sorted(alternatives.items()):
        if "-" not in diphone or not isinstance(choices, list):
            continue
        left, right = diphone.split("-", 1)
        for choice in choices:
            try:
                relative = str(choice["wav"])
                start = float(choice["start"])
                middle = float(choice["mid"])
                end = float(choice["end"])
            except (KeyError, TypeError, ValueError):
                continue
            audio_path = (root / relative).resolve()
            try:
                audio_path.relative_to(root)
            except ValueError:
                continue
            if not audio_path.is_file():
                continue
            for vowel, part_start, part_end, side in (
                (left, start, middle, "left"),
                (right, middle, end, "right"),
            ):
                if vowel not in _VOWELS or part_end - part_start < 0.045:
                    continue
                signature = (relative, round(part_start, 4),
                             round(part_end, 4), vowel)
                if signature in seen:
                    continue
                seen.add(signature)
                digest = hashlib.sha256(
                    repr(signature).encode("utf-8")
                ).hexdigest()[:12]
                candidates.append(AnalysisSegment(
                    segment_id=f"source:{vowel}:{digest}",
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    start_seconds=part_start,
                    end_seconds=part_end,
                    vowel=vowel,
                    phone=vowel,
                    partition="source",
                    recording_style="UTAU_reclist",
                    source_corpus="project_source_voicebank",
                    metadata={
                        "diphone": diphone,
                        "unit_side": side,
                        "source_wav": relative,
                        "oto_line": choice.get("oto_line"),
                    },
                ))
    selected: list[AnalysisSegment] = []
    for vowel in sorted(_VOWELS):
        rows = sorted(
            (item for item in candidates if item.vowel == vowel),
            key=lambda item: item.segment_id,
        )
        if len(rows) > maximum_per_vowel:
            positions = np.linspace(0, len(rows) - 1,
                                    maximum_per_vowel, dtype=int)
            rows = [rows[index] for index in positions]
        selected.extend(rows)
    return tuple(selected)


def supplied_reference_segments(
    reference_root: Path | str,
) -> tuple[AnalysisSegment, ...]:
    root = Path(reference_root).resolve()
    results: list[AnalysisSegment] = []
    for path in sorted(root.glob("*.wav"), key=lambda item: item.name.casefold()):
        audio = read_audio(path)
        duration = audio.samples.size / float(audio.sample_rate)
        name = path.stem.casefold()
        pieces = 10 if "sweep" in name or "change" in name else 1
        for index in range(pieces):
            start = duration * index / pieces
            end = duration * (index + 1) / pieces
            results.append(AnalysisSegment(
                segment_id=f"provided:{path.stem}:{index:02d}",
                speaker_id=f"provided_{path.stem}",
                audio_path=path,
                start_seconds=start,
                end_seconds=end,
                vowel="e",
                phone="e",
                partition="reference",
                recording_style="provided_formant_shift_reference",
                source_corpus="prompt20_supplied_references",
                metadata={
                    "reference_file": path.name,
                    "reference_vowel": "e",
                    "trajectory_position": round((index + 0.5) / pieces, 4),
                },
            ))
    return tuple(results)


def _group_summary(rows: Sequence[FormantSegment]) -> dict[str, object]:
    accepted = [row for row in rows if row.accepted]
    result: dict[str, object] = {
        "segment_count": len(rows),
        "accepted_segment_count": len(accepted),
        "rejected_segment_count": len(rows) - len(accepted),
        "accepted_frame_count": sum(row.accepted_frame_count for row in rows),
        "rejected_frame_count": sum(row.rejected_frame_count for row in rows),
        "median_f0_hz": _finite_median(row.median_f0_hz for row in accepted),
        "median_formant_dispersion_hz": _finite_median(
            row.median_formant_dispersion_hz for row in accepted
        ),
        "median_apparent_vocal_tract_length_cm": _finite_median(
            row.apparent_vocal_tract_length_cm for row in accepted
        ),
        "median_spectral_tilt_db_per_octave": _finite_median(
            row.median_spectral_tilt_db_per_octave for row in accepted
        ),
    }
    for index in range(4):
        result[f"median_f{index + 1}_hz"] = _finite_median(
            row.median_formants_hz[index] for row in accepted
        )
        result[f"median_f{index + 1}_bandwidth_hz"] = _finite_median(
            row.median_bandwidths_hz[index] for row in accepted
        )
        result[f"median_f{index + 1}_confidence"] = _finite_median(
            row.median_formant_confidences[index] for row in accepted
        )
    rejection_counts: dict[str, int] = {}
    for row in rows:
        for reason in row.rejection_reasons:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        for frame in row.frames:
            for reason in frame.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    result["rejection_reason_counts"] = dict(sorted(rejection_counts.items()))
    return result


def speaker_formant_summary(
    segments: Sequence[FormantSegment],
) -> dict[str, object]:
    speakers = sorted({row.segment.speaker_id for row in segments})
    payload: dict[str, object] = {
        "analysis_version": FORMANT_ANALYSIS_VERSION,
        "kind": "speaker_formant_summary",
        "speakers": {},
    }
    target = payload["speakers"]
    assert isinstance(target, dict)
    for speaker in speakers:
        rows = [row for row in segments if row.segment.speaker_id == speaker]
        target[speaker] = {
            "all": _group_summary(rows),
            "vowels": {
                vowel: _group_summary([
                    row for row in rows if row.segment.vowel == vowel
                ])
                for vowel in sorted(_VOWELS | {"unknown"})
                if any(row.segment.vowel == vowel for row in rows)
            },
        }
    return payload


def _percentile(values: Sequence[float], percentage: float) -> float | None:
    finite = np.sort(np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=np.float64,
    ))
    return float(np.percentile(finite, percentage)) if finite.size else None


def _bootstrap_median_interval(
    values: Sequence[float], *, iterations: int = 400
) -> dict[str, float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=np.float64)
    if not finite.size:
        return None
    generator = np.random.default_rng(20260716)
    medians = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        medians[index] = np.median(generator.choice(
            finite, size=finite.size, replace=True
        ))
    return {
        "median": round(float(np.median(finite)), 8),
        "bootstrap_p025": round(float(np.percentile(medians, 2.5)), 8),
        "bootstrap_p975": round(float(np.percentile(medians, 97.5)), 8),
        "sample_count": int(finite.size),
    }


def derive_reference_voice_space(
    segments: Sequence[FormantSegment],
    *,
    source_speaker_id: str = "project_source_speaker",
    reference_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    accepted = sorted(
        (row for row in segments if row.accepted),
        key=lambda row: (
            row.segment.speaker_id,
            row.segment.vowel,
            row.segment.segment_id,
        ),
    )
    source = [row for row in accepted
              if row.segment.speaker_id == source_speaker_id]
    if not source:
        raise ValueError("source speaker has no accepted formant segments")
    source_by_vowel = {
        vowel: _finite_median(
            row.median_formant_dispersion_hz for row in source
            if row.segment.vowel == vowel
        )
        for vowel in sorted(_VOWELS)
    }
    source_global = _finite_median(
        row.median_formant_dispersion_hz for row in source
    )
    if source_global is None:
        raise ValueError("source speaker has no valid formant dispersion")
    provided = [row for row in accepted if row.segment.source_corpus ==
                "prompt20_supplied_references" and
                row.median_formant_dispersion_hz is not None]
    provided_by_speaker: dict[str, float] = {}
    for speaker in sorted({row.segment.speaker_id for row in provided}):
        value = _finite_median(
            row.median_formant_dispersion_hz for row in provided
            if row.segment.speaker_id == speaker
        )
        if value is not None:
            provided_by_speaker[speaker] = value
    neutral_provided = [
        value for speaker, value in provided_by_speaker.items()
        if "neutral" in speaker and "change" not in speaker
    ]
    provided_baseline = _finite_median(neutral_provided)
    provided_ratios = {
        speaker: provided_baseline / value
        for speaker, value in provided_by_speaker.items()
        if provided_baseline is not None and value > 0.0
    }
    realistic_provided_ratios = [
        ratio for speaker, ratio in provided_ratios.items()
        if "sweep" not in speaker
    ]
    expanded_provided_ratios = list(provided_ratios.values())
    per_vowel: dict[str, dict[str, object]] = {}
    all_reference_ratios: list[float] = []
    for vowel in sorted(_VOWELS):
        baseline = source_by_vowel[vowel]
        ratios: list[float] = []
        if baseline is not None:
            comparison_rows = [
                row for row in accepted
                if (row.segment.speaker_id != source_speaker_id and
                    row.segment.source_corpus !=
                    "prompt20_supplied_references" and
                    row.segment.vowel == vowel and
                    row.median_formant_dispersion_hz is not None)
            ]
            for speaker in sorted({row.segment.speaker_id
                                   for row in comparison_rows}):
                speaker_median = _finite_median(
                    row.median_formant_dispersion_hz
                    for row in comparison_rows
                    if row.segment.speaker_id == speaker
                )
                if speaker_median is not None and speaker_median > 0.0:
                    ratios.append(baseline / speaker_median)
        # The supplied neutral/high/low/sweep recordings are all /e/.  Keep
        # that evidence attached to /e/ instead of projecting one vowel's
        # dispersion ratios onto the other four vowel configurations.
        supplied_vowel_ratios = (
            realistic_provided_ratios if vowel == "e" else []
        )
        ratios.extend(supplied_vowel_ratios)
        all_reference_ratios.extend(ratios)
        per_vowel[vowel] = {
            "source_dispersion_hz": baseline,
            "reference_ratio_statistics": _bootstrap_median_interval(ratios),
            "p05_ratio": _percentile(ratios, 5.0),
            "p10_ratio": _percentile(ratios, 10.0),
            "p90_ratio": _percentile(ratios, 90.0),
            "p95_ratio": _percentile(ratios, 95.0),
            "reference_segment_count": len(ratios),
            "supplied_reference_vowel": "e" if supplied_vowel_ratios else None,
            "supplied_reference_ratio_count": len(supplied_vowel_ratios),
        }
    global_p10 = _percentile(all_reference_ratios, 10.0) or 1.0
    global_p90 = _percentile(all_reference_ratios, 90.0) or 1.0
    lower_candidates = [float(row["p10_ratio"]) for row in per_vowel.values()
                        if row["p10_ratio"] is not None]
    upper_candidates = [float(row["p90_ratio"]) for row in per_vowel.values()
                        if row["p90_ratio"] is not None]
    realistic_min = max(0.5, min(1.0, max(lower_candidates)
                                 if lower_candidates else global_p10))
    realistic_max = min(1.8, max(1.0, min(upper_candidates)
                                 if upper_candidates else global_p90))
    if realistic_max - realistic_min < 0.06:
        median_ratio = float(np.median(all_reference_ratios)) \
            if all_reference_ratios else 1.0
        robust_spread = float(np.median(np.abs(
            np.asarray(all_reference_ratios or [1.0]) - median_ratio
        ))) * 1.4826
        half = max(0.03, min(0.12, robust_spread))
        realistic_min = max(0.75, min(1.0, median_ratio - half))
        realistic_max = min(1.30, max(1.0, median_ratio + half))
    source_formants = [
        value for row in source for value in row.median_formants_hz
        if value is not None
    ]
    sample_rates = [frame.sample_rate for row in source for frame in row.frames]
    nyquist = min(sample_rates) / 2.0 if sample_rates else 8000.0
    highest = _percentile(source_formants, 98.0) or 4500.0
    lowest_f1 = _percentile([
        row.median_formants_hz[0] for row in source
        if row.median_formants_hz[0] is not None
    ], 2.0) or 250.0
    numerical_min = max(0.55, highest / max(1.0, nyquist * 0.88))
    numerical_max = min(1.60, lowest_f1 / 90.0)
    expanded_evidence = all_reference_ratios + expanded_provided_ratios
    evidence_min = _percentile(expanded_evidence, 2.5) or realistic_min
    evidence_max = _percentile(expanded_evidence, 97.5) or realistic_max
    expanded_min = max(numerical_min, min(realistic_min, evidence_min))
    expanded_max = min(numerical_max, max(realistic_max, evidence_max))
    total_frames = sum(len(row.frames) for row in segments)
    accepted_frames = sum(row.accepted_frame_count for row in segments)
    return {
        "model_version": REFERENCE_VOICE_SPACE_VERSION,
        "analysis_version": FORMANT_ANALYSIS_VERSION,
        "kind": "reference_voice_space",
        "source_speaker_id": source_speaker_id,
        "reference_dataset": [
            "prompt20_supplied_references",
            "Kokoro-Speech-Dataset-v1.3-xlarge",
        ],
        "reference_file_hashes": dict(sorted(
            (reference_hashes or {}).items()
        )),
        "formant_estimator": {
            "primary": "f0_adaptive_iterative_true_envelope",
            "crosscheck": "dynamic_order_burg_lpc",
            "true_envelope_tolerance_db": 2.0,
        },
        "sample_rate": sorted(set(sample_rates)),
        "identity_vocal_tract_ratio": 1.0,
        "realistic_min_ratio": round(realistic_min, 8),
        "realistic_max_ratio": round(realistic_max, 8),
        "expanded_min_ratio": round(expanded_min, 8),
        "expanded_max_ratio": round(expanded_max, 8),
        "per_vowel_limits": per_vowel,
        "confidence_statistics": {
            "global_reference_ratio": _bootstrap_median_interval(
                all_reference_ratios
            ),
            "provided_neutral_dispersion_hz": provided_baseline,
            "provided_speaker_ratios": dict(sorted(provided_ratios.items())),
            "provided_ratio_statistics": _bootstrap_median_interval(
                expanded_provided_ratios
            ),
            "source_segment_count": len(source),
            "reference_segment_count": len(accepted) - len(source),
            "accepted_frame_count": accepted_frames,
            "total_frame_count": total_frames,
        },
        "excluded_measurement_count": total_frames - accepted_frames,
        "range_basis": (
            "intersection of robust per-vowel speaker-median ratios; the "
            "supplied /e/ neutral/high/low/change ratios constrain /e/, and "
            "the supplied /e/ sweep informs expanded bounds, which also enforce "
            "source F4/Nyquist and low-F1 analysis limits"
        ),
        "population_claim": False,
        "provenance": {
            "provided_data_primary": True,
            "kokoro_boundaries_are_silver": True,
            "anatomical_length_claim": False,
            "uniform_tube_formula_is_diagnostic_only": True,
            "supplied_reference_vowel": "e",
            "supplied_ratios_are_e_vowel_only": True,
        },
    }


def write_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_json(path: Path | str, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
