"""Reference-bounded vocal-tract resonance transformation.

The canonical control is ``target_length / source_length``.  A value below
one raises resonances; a value above one lowers them.  The renderer leaves the
excitation phase and frame timing in place and warps only an F0-adaptive true
spectral envelope.  It is an acoustic uniform-tube approximation, not an
anatomical larynx model or a binary gender classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from cache_support import FileIdentityCache
from formant_analysis import estimate_f0


REFERENCE_PROFILE_PATH = (
    Path(__file__).resolve().parent
    / "profiles"
    / "reference_voice_space_v1.json"
)
_VOCAL_TRACT_RANGE_CACHE = FileIdentityCache(
    "vocal-tract-range", max_entries=4, max_bytes=4 * 1024 * 1024
)


@dataclass(frozen=True)
class VocalTractRange:
    identity_ratio: float
    realistic_min_ratio: float
    realistic_max_ratio: float
    expanded_min_ratio: float
    expanded_max_ratio: float
    model_version: int
    analysis_version: str
    profile_path: str = field(compare=False, default="")

    def bounds(self, chipmunk_range: bool = False) -> tuple[float, float]:
        return (
            (self.expanded_min_ratio, self.expanded_max_ratio)
            if chipmunk_range
            else (self.realistic_min_ratio, self.realistic_max_ratio)
        )

    def clamp(self, ratio: float, chipmunk_range: bool = False) -> float:
        lower, upper = self.bounds(chipmunk_range)
        return max(lower, min(upper, float(ratio)))


@dataclass(frozen=True)
class VocalTractTransformConfig:
    frame_seconds: float = 0.040
    hop_seconds: float = 0.010
    maximum_envelope_gain_db: float = 30.0
    nyquist_taper_start: float = 0.90
    invalid_band_taper_fraction: float = 0.10
    minimum_frame_rms: float = 1e-5
    preserve_frame_rms: bool = True
    cepstral_lifter_taper_fraction: float = 0.25
    minimum_cepstral_order: int = 12
    maximum_cepstral_order: int = 128
    # Retained for compatibility with early Stage B callers. The corrected
    # renderer no longer performs iterative true-envelope peak lifting.
    runtime_true_envelope_iterations: int = 12
    minimum_strength: float = 0.02


@dataclass(frozen=True)
class VocalTractFrameDiagnostic:
    time: float
    phone: str
    requested_ratio: float
    applied_ratio: float
    formant_shift_semitones: float
    strength: float
    f0_hz: float | None
    f0_confidence: float
    f0_ambiguity: float
    envelope_order: int
    envelope_iterations: int
    maximum_gain_db: float
    invalid_band_fraction: float
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class VocalTractTransformResult:
    samples: np.ndarray = field(compare=False, repr=False)
    requested_ratio: float
    applied_ratio: float
    requested_targets: tuple[tuple[float, float], ...]
    applied_targets: tuple[tuple[float, float], ...]
    chipmunk_range: bool
    profile_model_version: int
    frame_diagnostics: tuple[VocalTractFrameDiagnostic, ...]
    modified_frame_count: int
    skipped_frame_count: int
    input_peak: float
    output_peak: float
    duration_samples: int
    sample_rate: int
    processing_seconds: float
    identity_bypass: bool
    method: str = "f0_adaptive_cepstral_source_filter_warp_v2"
    schema_version: int = 1

    @property
    def real_time_factor(self) -> float:
        if self.identity_bypass or self.duration_samples <= 0:
            return 0.0
        duration_seconds = self.duration_samples / float(self.sample_rate)
        return self.processing_seconds / duration_seconds

    def diagnostic_dict(self, include_frames: bool = True) -> dict[str, object]:
        applied_values = [value for _time, value in self.applied_targets]
        result = {
            "schema_version": self.schema_version,
            "method": self.method,
            "requested_ratio": self.requested_ratio,
            "applied_ratio": self.applied_ratio,
            "requested_targets": [list(point)
                                  for point in self.requested_targets],
            "applied_targets": [list(point)
                                for point in self.applied_targets],
            "applied_ratio_min": min(applied_values, default=1.0),
            "applied_ratio_max": max(applied_values, default=1.0),
            "formant_frequency_multiplier": ratio_to_formant_multiplier(
                self.applied_ratio
            ),
            "formant_shift_semitones": ratio_to_formant_semitones(
                self.applied_ratio
            ),
            "chipmunk_range": self.chipmunk_range,
            "profile_model_version": self.profile_model_version,
            "modified_frame_count": self.modified_frame_count,
            "skipped_frame_count": self.skipped_frame_count,
            "input_peak": self.input_peak,
            "output_peak": self.output_peak,
            "duration_samples": self.duration_samples,
            "sample_rate": self.sample_rate,
            "processing_seconds": self.processing_seconds,
            "real_time_factor": self.real_time_factor,
            "identity_bypass": self.identity_bypass,
        }
        if include_frames:
            result["frames"] = [row.to_dict()
                                for row in self.frame_diagnostics]
        return result


def _read_vocal_tract_range(source: Path) -> VocalTractRange:
    data = json.loads(source.read_text(encoding="utf-8"))
    required = (
        "identity_vocal_tract_ratio",
        "realistic_min_ratio",
        "realistic_max_ratio",
        "expanded_min_ratio",
        "expanded_max_ratio",
        "model_version",
        "analysis_version",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("vocal-tract profile is missing: " + ", ".join(missing))
    result = VocalTractRange(
        identity_ratio=float(data["identity_vocal_tract_ratio"]),
        realistic_min_ratio=float(data["realistic_min_ratio"]),
        realistic_max_ratio=float(data["realistic_max_ratio"]),
        expanded_min_ratio=float(data["expanded_min_ratio"]),
        expanded_max_ratio=float(data["expanded_max_ratio"]),
        model_version=int(data["model_version"]),
        analysis_version=str(data["analysis_version"]),
        profile_path=str(source),
    )
    if not (
        0.5 <= result.expanded_min_ratio
        <= result.realistic_min_ratio
        <= result.identity_ratio
        <= result.realistic_max_ratio
        <= result.expanded_max_ratio
        <= 1.8
    ):
        raise ValueError("vocal-tract profile bounds are inconsistent")
    return result


def load_vocal_tract_range(path: Path | str | None = None) -> VocalTractRange:
    """Load an immutable range profile with automatic file invalidation."""
    source = Path(path) if path is not None else REFERENCE_PROFILE_PATH
    return _VOCAL_TRACT_RANGE_CACHE.get(source, _read_vocal_tract_range)


def vocal_tract_model_cache_info() -> dict[str, int | str]:
    return _VOCAL_TRACT_RANGE_CACHE.info()


def clear_vocal_tract_model_cache() -> dict[str, int | str]:
    return _VOCAL_TRACT_RANGE_CACHE.clear()


def ratio_to_formant_multiplier(ratio: float) -> float:
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("vocal-tract ratio must be finite and positive")
    return 1.0 / value


def ratio_to_formant_semitones(ratio: float) -> float:
    return -12.0 * math.log2(float(ratio))


def ratio_from_formant_semitones(semitones: float) -> float:
    return 2.0 ** (-float(semitones) / 12.0)


def normalize_ratio_targets(
    targets: Sequence[Sequence[float]] | None,
    duration_seconds: float,
    fallback_ratio: float,
    tract_range: VocalTractRange,
    chipmunk_range: bool = False,
) -> tuple[tuple[tuple[float, float], ...],
           tuple[tuple[float, float], ...]]:
    """Validate a time curve and return requested and safety-clamped points.

    Ratios are interpolated logarithmically by the renderer. Duplicate times
    use the final supplied value, matching the other editable parameter tracks.
    """
    duration = max(0.0, float(duration_seconds))
    rows: dict[float, float] = {}
    for raw in targets or ():
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            when, ratio = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(when) or not math.isfinite(ratio)
                or ratio <= 0.0):
            continue
        rows[max(0.0, min(duration, when))] = ratio
    if not rows:
        rows = {0.0: float(fallback_ratio)}
        if duration > 0.0:
            rows[duration] = float(fallback_ratio)
    requested = tuple(sorted((float(time), float(value))
                             for time, value in rows.items()))
    applied = tuple((time, tract_range.clamp(value, chipmunk_range))
                    for time, value in requested)
    return requested, applied


def sample_ratio_targets(
    targets: Sequence[Sequence[float]],
    when: float,
    default: float = 1.0,
) -> float:
    """Sample a tract-ratio curve with smooth log-domain interpolation."""
    rows = [(float(row[0]), float(row[1])) for row in targets
            if len(row) >= 2 and float(row[1]) > 0.0]
    if not rows:
        return float(default)
    rows.sort()
    times = np.asarray([row[0] for row in rows], np.float64)
    logs = np.log(np.asarray([row[1] for row in rows], np.float64))
    return float(np.exp(np.interp(float(when), times, logs)))


def ratio_curve_summary(
    targets: Sequence[Sequence[float]],
    default: float = 1.0,
) -> float:
    """Return a stable scalar summary for legacy project/status fields."""
    values = [float(row[1]) for row in targets
              if len(row) >= 2 and float(row[1]) > 0.0]
    return (float(np.exp(np.median(np.log(values))))
            if values else float(default))


def ratio_curves_close(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    tolerance: float = 5e-4,
) -> bool:
    """Compare audible curves independently of their control-point grids."""
    times = sorted({float(row[0]) for rows in (first, second)
                    for row in rows if len(row) >= 2})
    if not times:
        times = [0.0]
    return all(
        abs(math.log(sample_ratio_targets(first, when, 1.0))
            - math.log(sample_ratio_targets(second, when, 1.0)))
        <= float(tolerance)
        for when in times
    )


def control_position_to_ratio(
    position: float,
    tract_range: VocalTractRange,
    chipmunk_range: bool = False,
) -> float:
    """Map 0..1 to a log-ratio control with identity exactly at 0.5."""
    value = max(0.0, min(1.0, float(position)))
    lower, upper = tract_range.bounds(chipmunk_range)
    if value <= 0.5:
        fraction = value / 0.5
        ratio = math.exp(
            math.log(upper) * (1.0 - fraction)
            + math.log(tract_range.identity_ratio) * fraction
        )
    else:
        fraction = (value - 0.5) / 0.5
        ratio = math.exp(
            math.log(tract_range.identity_ratio) * (1.0 - fraction)
            + math.log(lower) * fraction
        )
    return float(ratio)


def ratio_to_control_position(
    ratio: float,
    tract_range: VocalTractRange,
    chipmunk_range: bool = False,
) -> float:
    value = tract_range.clamp(ratio, chipmunk_range)
    lower, upper = tract_range.bounds(chipmunk_range)
    identity = tract_range.identity_ratio
    if value >= identity:
        denominator = math.log(upper) - math.log(identity)
        fraction = ((math.log(value) - math.log(identity)) / denominator
                    if abs(denominator) > 1e-12 else 0.0)
        return 0.5 * (1.0 - fraction)
    denominator = math.log(identity) - math.log(lower)
    fraction = ((math.log(identity) - math.log(value)) / denominator
                if abs(denominator) > 1e-12 else 0.0)
    return 0.5 + 0.5 * fraction


_VOWELS = {
    "a", "i", "u", "e", "o", "aa", "ae", "ah", "ao", "aw", "ax",
    "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}
_SONORANTS = {
    "m", "n", "ng", "N", "nn", "nng", "mm", "r", "l", "w", "y",
}
_VOICED_FRICATIVES = {"v", "z", "zh", "dh", "j"}
_UNVOICED_FRICATIVES = {"f", "h", "s", "sh", "th", "x", "hy"}
_STOPS = {"b", "d", "g", "k", "p", "t", "q", "cl", "dx", "ch", "ts"}
_SILENCE = {"", "pau", "sil", "#", "sp"}


def phone_warp_strength(phone: str, periodicity: float = 0.0) -> float:
    symbol = str(phone or "").split("_")[0]
    if symbol in _SILENCE:
        return 0.0
    if symbol in _VOWELS:
        return 1.0
    if symbol in _SONORANTS:
        return 0.88
    if symbol in _VOICED_FRICATIVES:
        return 0.48
    if symbol in _UNVOICED_FRICATIVES:
        return 0.10
    if symbol in _STOPS:
        return 0.08 if symbol in {"k", "p", "t", "q", "cl", "ch", "ts"} \
            else 0.22
    return 0.72 if float(periodicity) >= 0.45 else 0.14


def _segment_phone_at(segments: Sequence[object] | None, when: float) -> str:
    if not segments:
        return ""
    for row in segments:
        if isinstance(row, Mapping):
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start))
            phone = str(row.get("phone") or "")
        else:
            start = float(getattr(row, "start", 0.0))
            end = float(getattr(row, "end", start))
            phone = str(getattr(row, "phone", "") or "")
        if start <= when < end or abs(when - end) < 1e-9:
            return phone
    return ""


def _frame_strength(
    segments: Sequence[object] | None,
    when: float,
    frame_seconds: float,
    periodicity: float,
) -> tuple[str, float]:
    if not segments:
        return "", phone_warp_strength("unknown", periodicity)
    probes = (when - frame_seconds * 0.22, when, when + frame_seconds * 0.22)
    phones = [_segment_phone_at(segments, max(0.0, probe)) for probe in probes]
    strengths = [phone_warp_strength(phone, periodicity) for phone in phones]
    phone = phones[1] or next((value for value in phones if value), "")
    return phone, float(sum(strengths) / len(strengths))


def _warped_envelope_gain_db(
    frequencies: np.ndarray,
    envelope_db: np.ndarray,
    ratio: float,
    strength: float,
    config: VocalTractTransformConfig,
) -> tuple[np.ndarray, float]:
    source = np.asarray(envelope_db, np.float64)
    freqs = np.asarray(frequencies, np.float64)
    queries = freqs * float(ratio)
    target = np.interp(
        np.minimum(queries, freqs[-1]), freqs, source,
        left=float(source[0]), right=float(source[-1]),
    )
    gain = target - source

    invalid = queries > freqs[-1]
    invalid_fraction = float(np.count_nonzero(invalid) / max(1, gain.size))
    if np.any(invalid):
        valid_end = freqs[-1] / max(float(ratio), 1e-9)
        fade_start = valid_end * (1.0 - config.invalid_band_taper_fraction)
        fade = np.clip(
            (valid_end - freqs) / max(1.0, valid_end - fade_start),
            0.0, 1.0,
        )
        gain *= fade

    nyquist_start = freqs[-1] * config.nyquist_taper_start
    top_fade = np.clip(
        (freqs[-1] - freqs) / max(1.0, freqs[-1] - nyquist_start),
        0.0, 1.0,
    )
    gain *= top_fade
    gain = np.clip(
        gain,
        -config.maximum_envelope_gain_db,
        config.maximum_envelope_gain_db,
    )
    # Both operands are already smooth cepstral envelopes. A second bin-domain
    # smoother would blur narrow formants and was the main reason the original
    # Stage B transform barely moved measured resonances.
    return gain * max(0.0, min(1.0, strength)), invalid_fraction


def _cepstral_envelope_db_from_spectrum(
    spectrum: np.ndarray,
    sample_rate: int,
    f0_hz: float | None,
    config: VocalTractTransformConfig,
) -> tuple[np.ndarray, int]:
    """Separate a smooth tract envelope from one windowed FFT spectrum.

    The spectrum used here is exactly the spectrum later resynthesized.  The
    previous implementation estimated an envelope from a differently windowed
    and differently sized FFT, so ``target / source`` did not reconstruct the
    requested filter.  The F0-adaptive low-quefrency lifter excludes the pulse
    train while retaining broad vocal-tract resonances.
    """
    values = np.asarray(spectrum, np.complex128).reshape(-1)
    full_length = max(2, (values.size - 1) * 2)
    reference_f0 = max(55.0, float(f0_hz or 180.0))
    order = int(round(float(sample_rate) / (2.0 * reference_f0)))
    order = max(
        int(config.minimum_cepstral_order),
        min(
            int(config.maximum_cepstral_order),
            order,
            max(2, full_length // 4),
        ),
    )
    log_magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(values), 1.0e-10)
    )
    cepstrum = np.fft.irfft(log_magnitude_db, full_length)
    lifter = np.zeros(full_length, dtype=np.float64)
    taper_count = max(
        1,
        min(
            order,
            int(round(order * config.cepstral_lifter_taper_fraction)),
        ),
    )
    flat_end = order - taper_count
    positive = np.ones(order + 1, dtype=np.float64)
    positions = np.arange(1, taper_count + 1, dtype=np.float64)
    positive[flat_end + 1:order + 1] = 0.5 * (
        1.0 + np.cos(math.pi * positions / float(taper_count))
    )
    lifter[:order + 1] = positive
    lifter[-order:] = positive[1:][::-1]
    envelope_db = np.fft.rfft(cepstrum * lifter, full_length).real
    return np.asarray(envelope_db, np.float64), int(order)


def transform_vocal_tract(
    samples: Sequence[float],
    sample_rate: int,
    vocal_tract_length_ratio: float = 1.0,
    *,
    chipmunk_range: bool = False,
    ratio_targets: Sequence[Sequence[float]] | None = None,
    segments: Sequence[object] | None = None,
    tract_range: VocalTractRange | None = None,
    config: VocalTractTransformConfig | None = None,
) -> VocalTractTransformResult:
    """Warp the final waveform's tract envelope without resampling it."""
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    values = np.asarray(samples, np.float64).reshape(-1)
    profile = tract_range or load_vocal_tract_range()
    requested = float(vocal_tract_length_ratio)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("vocal-tract ratio must be finite and positive")
    duration_seconds = values.size / float(sample_rate)
    requested_targets, applied_targets = normalize_ratio_targets(
        ratio_targets, duration_seconds, requested, profile, chipmunk_range
    )
    requested_summary = ratio_curve_summary(requested_targets, requested)
    ratio = ratio_curve_summary(applied_targets, profile.identity_ratio)
    input_peak = float(np.max(np.abs(values))) if values.size else 0.0
    identity_curve = all(
        abs(value - profile.identity_ratio) <= 1e-12
        for _time, value in applied_targets
    )
    if not values.size or identity_curve:
        output = np.asarray(values, np.float32).copy()
        return VocalTractTransformResult(
            samples=output,
            requested_ratio=requested_summary,
            applied_ratio=ratio,
            requested_targets=requested_targets,
            applied_targets=applied_targets,
            chipmunk_range=bool(chipmunk_range),
            profile_model_version=profile.model_version,
            frame_diagnostics=(),
            modified_frame_count=0,
            skipped_frame_count=0,
            input_peak=input_peak,
            output_peak=input_peak,
            duration_samples=int(values.size),
            sample_rate=int(sample_rate),
            processing_seconds=0.0,
            identity_bypass=True,
        )

    cfg = config or VocalTractTransformConfig()
    started = time.perf_counter()
    frame_length = max(128, int(round(cfg.frame_seconds * sample_rate)))
    if frame_length % 2:
        frame_length += 1
    hop = max(16, int(round(cfg.hop_seconds * sample_rate)))
    half = frame_length // 2
    nfft = 1 << max(9, (frame_length - 1).bit_length())
    window = np.sqrt(np.maximum(np.hanning(frame_length), 1e-8))
    pad_mode = "reflect" if values.size > 1 else "edge"
    padded = np.pad(values, (half, half), mode=pad_mode)
    output = np.zeros_like(padded)
    weights = np.zeros_like(padded)
    centers = np.arange(0, values.size, hop, dtype=int)
    if not centers.size or centers[-1] != values.size - 1:
        centers = np.append(centers, values.size - 1)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    diagnostics: list[VocalTractFrameDiagnostic] = []
    modified = skipped = 0

    for center in centers:
        frame = padded[int(center):int(center) + frame_length]
        frame_time = float(center) / float(sample_rate)
        requested_frame_ratio = sample_ratio_targets(
            requested_targets, frame_time, requested_summary
        )
        frame_ratio = sample_ratio_targets(
            applied_targets, frame_time, ratio
        )
        windowed = frame * window
        spectrum = np.fft.rfft(windowed, nfft)
        rms = float(np.sqrt(np.mean(frame * frame)))
        f0_hz, f0_confidence, f0_ambiguity = estimate_f0(frame, sample_rate)
        phone, strength = _frame_strength(
            segments, frame_time, cfg.frame_seconds, f0_confidence
        )
        applied = False
        reason = "applied"
        order = iterations = 0
        maximum_gain = invalid_fraction = 0.0
        transformed = spectrum
        if rms < cfg.minimum_frame_rms:
            reason = "near_silence"
            skipped += 1
        elif strength < cfg.minimum_strength:
            reason = "protected_phone"
            skipped += 1
        else:
            envelope, order = _cepstral_envelope_db_from_spectrum(
                spectrum, sample_rate, f0_hz, cfg
            )
            iterations = 0
            gain_db, invalid_fraction = _warped_envelope_gain_db(
                frequencies, envelope, frame_ratio, strength, cfg
            )
            maximum_gain = float(np.max(np.abs(gain_db)))
            transformed = spectrum * (10.0 ** (gain_db / 20.0))
            if cfg.preserve_frame_rms:
                source_frame = np.fft.irfft(spectrum, nfft)[:frame_length]
                target_frame = np.fft.irfft(transformed, nfft)[:frame_length]
                source_rms = float(np.sqrt(np.mean(source_frame ** 2)))
                target_rms = float(np.sqrt(np.mean(target_frame ** 2)))
                if source_rms > 1e-10 and target_rms > 1e-10:
                    transformed *= source_rms / target_rms
            if np.all(np.isfinite(transformed)):
                applied = True
                modified += 1
            else:
                transformed = spectrum
                reason = "non_finite_reconstruction"
                skipped += 1
        rendered = np.fft.irfft(transformed, nfft)[:frame_length]
        start = int(center)
        output[start:start + frame_length] += rendered * window
        weights[start:start + frame_length] += window * window
        diagnostics.append(VocalTractFrameDiagnostic(
            time=round(frame_time, 6),
            phone=phone,
            requested_ratio=round(requested_frame_ratio, 7),
            applied_ratio=round(frame_ratio, 7),
            formant_shift_semitones=round(
                ratio_to_formant_semitones(frame_ratio), 5),
            strength=round(float(strength), 6),
            f0_hz=(round(float(f0_hz), 4) if f0_hz is not None else None),
            f0_confidence=round(float(f0_confidence), 6),
            f0_ambiguity=round(float(f0_ambiguity), 6),
            envelope_order=int(order),
            envelope_iterations=int(iterations),
            maximum_gain_db=round(maximum_gain, 6),
            invalid_band_fraction=round(invalid_fraction, 6),
            applied=applied,
            reason=reason,
        ))

    reconstructed = np.divide(
        output, weights, out=padded.copy(), where=weights > 1e-10
    )[half:half + values.size]
    if not np.all(np.isfinite(reconstructed)):
        reconstructed = values.copy()
        modified = 0
        skipped = len(centers)
    elapsed = time.perf_counter() - started
    output_peak = (float(np.max(np.abs(reconstructed)))
                   if reconstructed.size else 0.0)
    return VocalTractTransformResult(
        samples=np.asarray(reconstructed, np.float32),
        requested_ratio=requested_summary,
        applied_ratio=ratio,
        requested_targets=requested_targets,
        applied_targets=applied_targets,
        chipmunk_range=bool(chipmunk_range),
        profile_model_version=profile.model_version,
        frame_diagnostics=tuple(diagnostics),
        modified_frame_count=int(modified),
        skipped_frame_count=int(skipped),
        input_peak=round(input_peak, 9),
        output_peak=round(output_peak, 9),
        duration_samples=int(values.size),
        sample_rate=int(sample_rate),
        processing_seconds=round(float(elapsed), 6),
        identity_bypass=False,
    )
