"""Deterministic source-filter voicing control for rendered speech.

The renderer uses a reconstructive short-time model::

    speech = (harmonic excitation + aperiodic excitation) * tract envelope

The smooth spectral envelope is divided out before a pitch-scaled harmonic
mask separates the residual.  Both residual components are then passed back
through the *same* envelope.  At the measured source voicing value the two
scale factors are exactly one, so the analysis/synthesis path reconstructs the
input instead of replacing it with a vocoder approximation.

This is intentionally smaller than a full glottal LF-model estimator.  It
implements the engineering constraints shared by source-filter models and the
SVLN/DSM family of methods: deterministic and stochastic excitation remain
distinct, noise is not allowed to dominate low frequencies, and complementary
overlap windows integrate changing excitation over time.  A strongly voiced
recording contains too little trustworthy noise to support a zero-voicing
endpoint by residual amplification alone.  In that case a single continuous,
deterministic noise excitation supplies the stochastic source and the measured
spectral envelope supplies the vocal-tract filter.  This avoids preserving the
comb-shaped, phase-coherent leakage that otherwise sounds like growl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SourceFilterVoicingConfig:
    frame_seconds: float = 0.032
    hop_seconds: float = 0.008
    minimum_f0_hz: float = 55.0
    maximum_f0_hz: float = 500.0
    minimum_periodicity: float = 0.24
    harmonic_half_width_f0: float = 0.34
    envelope_smoothing_f0: float = 1.8
    minimum_aperiodic_fraction: float = 0.020
    minimum_harmonic_fraction: float = 0.010
    maximum_component_gain: float = 4.0
    unchanged_tolerance: float = 0.008
    low_noise_cutoff_hz: float = 500.0
    low_noise_transition_hz: float = 320.0
    low_noise_floor_db: float = -15.0
    residual_noise_blend_power: float = 1.35
    stochastic_shape_smoothing_f0: float = 2.5
    maximum_stochastic_shape_db: float = 15.0
    harmonic_envelope_weight: float = 0.88
    zero_voicing_gain_db: float = -15.0
    devoicing_gain_power: float = 1.15
    stochastic_glottal_shelf_db: float = -2.0
    stochastic_pre_formant_valley_db: float = -7.5

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VoicingFrameDiagnostic:
    time: float
    source_voicing: float
    target_voicing: float
    periodicity: float
    f0_hz: float | None
    harmonic_fraction: float
    aperiodic_fraction: float
    stochastic_fraction: float
    stochastic_excitation: bool
    gain_db: float
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class VoicingTransformResult:
    samples: np.ndarray = field(compare=False, repr=False)
    source_curve: tuple[tuple[float, float], ...]
    target_curve: tuple[tuple[float, float], ...]
    frame_diagnostics: tuple[VoicingFrameDiagnostic, ...]
    modified_frame_count: int
    skipped_frame_count: int
    reconstruction_nrmse: float
    schema_version: int = 2
    method: str = "continuous_stochastic_source_filter_v2"

    @property
    def applied(self) -> bool:
        return self.modified_frame_count > 0

    def diagnostic_dict(self, *, include_frames: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "method": self.method,
            "modified_frame_count": self.modified_frame_count,
            "skipped_frame_count": self.skipped_frame_count,
            "reconstruction_nrmse": self.reconstruction_nrmse,
            "source_curve": [list(item) for item in self.source_curve],
            "target_curve": [list(item) for item in self.target_curve],
        }
        if include_frames:
            result["frames"] = [item.to_dict() for item in self.frame_diagnostics]
        return result


@dataclass
class _FrameAnalysis:
    spectrum: np.ndarray
    envelope: np.ndarray
    harmonic: np.ndarray
    aperiodic: np.ndarray
    stochastic: np.ndarray
    periodicity: float
    f0_hz: float | None
    harmonic_fraction: float
    aperiodic_fraction: float
    stochastic_fraction: float
    source_voicing: float


def _rms(values: np.ndarray) -> float:
    data = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(data * data))) if data.size else 0.0


def _normalized_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, np.float64)
    candidate = np.asarray(candidate, np.float64)
    denominator = _rms(reference) + 1e-12
    return _rms(candidate - reference) / denominator


def _interp_curve(points: Sequence[Sequence[float]] | None,
                  times: np.ndarray,
                  defaults: np.ndarray) -> np.ndarray:
    if not points:
        return np.asarray(defaults, np.float64).copy()
    cleaned = sorted(
        (float(item[0]), max(0.0, min(1.0, float(item[1]))))
        for item in points if len(item) >= 2
        and math.isfinite(float(item[0])) and math.isfinite(float(item[1]))
    )
    if not cleaned:
        return np.asarray(defaults, np.float64).copy()
    xs = np.asarray([item[0] for item in cleaned], np.float64)
    ys = np.asarray([item[1] for item in cleaned], np.float64)
    return np.interp(times, xs, ys, left=ys[0], right=ys[-1])


def _estimate_f0(frame: np.ndarray, sample_rate: int,
                 config: SourceFilterVoicingConfig) -> tuple[float | None, float]:
    centered = np.asarray(frame, np.float64) - float(np.mean(frame))
    if centered.size < 24 or _rms(centered) < 1e-6:
        return None, 0.0
    windowed = centered * np.hanning(centered.size)
    minimum_lag = max(2, int(math.floor(
        sample_rate / config.maximum_f0_hz)))
    maximum_lag = min(
        centered.size // 2,
        int(math.ceil(sample_rate / config.minimum_f0_hz)),
    )
    if maximum_lag <= minimum_lag:
        return None, 0.0
    scores = []
    for lag in range(minimum_lag, maximum_lag + 1):
        left = windowed[:-lag]
        right = windowed[lag:]
        denominator = math.sqrt(
            float(np.dot(left, left)) * float(np.dot(right, right))
        ) + 1e-12
        scores.append(float(np.dot(left, right)) / denominator)
    best_offset = int(np.argmax(scores)) if scores else 0
    # A two-period lag often has a marginally higher normalized correlation
    # than the true period in a short vowel frame. Prefer the earliest strong
    # local peak when it is within 90% of the global peak; this avoids an
    # octave-down harmonic mask without blindly selecting a weak upper
    # harmonic.
    if scores:
        threshold = max(
            config.minimum_periodicity,
            0.90 * float(scores[best_offset]),
        )
        for offset in range(1, len(scores) - 1):
            if (scores[offset] >= threshold
                    and scores[offset] >= scores[offset - 1]
                    and scores[offset] >= scores[offset + 1]):
                best_offset = offset
                break
    best_lag = minimum_lag + best_offset
    score = max(0.0, min(1.0, scores[best_offset] if scores else 0.0))
    if score < config.minimum_periodicity:
        return None, score
    # Parabolic interpolation reduces staircase movement in the mask without
    # allowing an analysis window to leave this frame.
    lag = float(best_lag)
    if 0 < best_offset < len(scores) - 1:
        before, peak, after = scores[best_offset - 1:best_offset + 2]
        denominator = before - 2.0 * peak + after
        if abs(denominator) > 1e-12:
            lag += max(-0.5, min(0.5, 0.5 * (before - after) / denominator))
    return float(sample_rate / lag), score


def _smooth_log_envelope(magnitude: np.ndarray, f0_hz: float,
                         bin_hz: float,
                         config: SourceFilterVoicingConfig) -> np.ndarray:
    log_magnitude = np.log(np.maximum(np.asarray(magnitude, np.float64), 1e-10))
    width = max(5, int(round(
        config.envelope_smoothing_f0 * f0_hz / max(bin_hz, 1e-9)
    )))
    if width % 2 == 0:
        width += 1
    width = min(width, max(5, len(log_magnitude) // 3 * 2 + 1))
    half = width // 2
    padded = np.pad(log_magnitude, (half, half), mode="edge")
    smoothed = np.convolve(
        padded, np.ones(width, np.float64) / width, mode="valid"
    )
    # A moving log average alone fills real valleys between sparse harmonics,
    # which makes a noise excitation unnaturally forceful.  Interpolate the
    # local harmonic peaks to retain the F0-synchronous estimate of the tract
    # envelope, then keep a small amount of the broad estimate for stability
    # where high-frequency peaks become noise dominated.
    peak_frequencies = []
    peak_logs = []
    nyquist = (len(log_magnitude) - 1) * bin_hz
    for harmonic in range(1, int(nyquist / f0_hz) + 1):
        center_hz = harmonic * f0_hz
        first = max(0, int(math.floor(
            (center_hz - 0.36 * f0_hz) / bin_hz
        )))
        last = min(len(log_magnitude), int(math.ceil(
            (center_hz + 0.36 * f0_hz) / bin_hz
        )) + 1)
        if last <= first:
            continue
        peak_index = first + int(np.argmax(log_magnitude[first:last]))
        peak_frequencies.append(peak_index * bin_hz)
        peak_logs.append(float(log_magnitude[peak_index]))
    if len(peak_logs) >= 3:
        frequencies = np.arange(len(log_magnitude), dtype=np.float64) * bin_hz
        peak_envelope = np.interp(
            frequencies,
            np.asarray(peak_frequencies, np.float64),
            np.asarray(peak_logs, np.float64),
            left=peak_logs[0], right=peak_logs[-1],
        )
        weight = max(0.0, min(1.0, config.harmonic_envelope_weight))
        smoothed = weight * peak_envelope + (1.0 - weight) * smoothed
    return np.exp(smoothed)


def _harmonic_mask(frequencies: np.ndarray, f0_hz: float,
                   periodicity: float,
                   config: SourceFilterVoicingConfig) -> np.ndarray:
    frequencies = np.asarray(frequencies, np.float64)
    harmonic_number = np.maximum(1.0, np.rint(frequencies / f0_hz))
    distance = np.abs(frequencies - harmonic_number * f0_hz)
    width = max(1.0, config.harmonic_half_width_f0 * f0_hz)
    mask = np.zeros_like(frequencies)
    inside = distance < width
    mask[inside] = 0.5 * (
        1.0 + np.cos(np.pi * distance[inside] / width)
    )
    # Aspiration/noise should not be boosted into DC or sub-glottal rumble.
    mask[frequencies < max(config.low_noise_cutoff_hz, 0.45 * f0_hz)] = 1.0
    # Periodicity is already a hard gate for entering this branch. Scaling
    # the mask by that score would leak every harmonic into the noise
    # component and make the curve report a mostly periodic vowel as weakly
    # voiced. Keep periodicity as a separate confidence measurement.
    _ = periodicity
    return mask


def _continuous_noise(values: np.ndarray, sample_rate: int,
                      length: int) -> np.ndarray:
    """Return a stable noise source shared by every overlapping frame.

    Independent random phases per STFT frame produce flutter and can leave
    periodic beating after overlap-add.  One source stream gives neighboring
    frames compatible phase while a content-derived seed makes repeat renders
    byte-for-byte deterministic.
    """
    source = np.asarray(values, np.float32)
    digest = hashlib.blake2b(
        source.tobytes(order="C")
        + int(sample_rate).to_bytes(4, "little", signed=False),
        digest_size=16,
        person=b"festvox-voice-v2",
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    generator = np.random.Generator(np.random.PCG64(seed))
    noise = generator.standard_normal(int(length)).astype(np.float64)
    noise -= float(np.mean(noise)) if noise.size else 0.0
    return noise


def _stochastic_taper(frequencies: np.ndarray,
                      config: SourceFilterVoicingConfig) -> np.ndarray:
    """Suppress DC/rumble without putting a hard spectral edge in the noise."""
    frequencies = np.asarray(frequencies, np.float64)
    cutoff = max(0.0, float(config.low_noise_cutoff_hz))
    width = max(1.0, float(config.low_noise_transition_hz))
    low = max(0.0, cutoff - width)
    position = np.clip((frequencies - low) / width, 0.0, 1.0)
    floor = 10.0 ** (float(config.low_noise_floor_db) / 20.0)
    taper = floor + (1.0 - floor) * (
        0.5 - 0.5 * np.cos(np.pi * position)
    )
    taper[frequencies >= cutoff] = 1.0
    if taper.size:
        taper[0] = 0.0
    return taper


def _stochastic_residual(
    noise_frame: np.ndarray,
    analysis_window: np.ndarray,
    nfft: int,
    envelope: np.ndarray,
    frequencies: np.ndarray,
    residual: np.ndarray,
    harmonic_mask: np.ndarray,
    f0_hz: float,
    reference_energy: float,
    config: SourceFilterVoicingConfig,
) -> np.ndarray:
    """Model the aperiodic excitation as filtered continuous noise.

    Only phase is taken from the shared noise stream.  Its residual-domain
    magnitude is broadband (with a smooth low-frequency taper); applying the
    same speech envelope afterward preserves the vowel resonances without
    preserving a harmonic comb.  The component is energy-matched to the
    measured residual, with a small floor for nearly pure periodic sources.
    """
    spectrum = np.fft.rfft(
        np.asarray(noise_frame, np.float64) * analysis_window, nfft
    )
    phase = spectrum / np.maximum(np.abs(spectrum), 1e-12)
    # Preserve only the slow excitation tilt measured between harmonics.  The
    # harmonic bins themselves are excluded; interpolating across them and
    # smoothing over several F0 intervals prevents the old comb leakage from
    # returning in the stochastic endpoint.
    residual_magnitude = np.abs(np.asarray(residual, np.complex128))
    valid = (
        (np.asarray(harmonic_mask, np.float64) <= 0.12)
        & (frequencies >= max(1.0, config.low_noise_cutoff_hz))
        & np.isfinite(residual_magnitude)
        & (residual_magnitude > 1e-12)
    )
    if int(np.count_nonzero(valid)) >= 8:
        interpolated = np.interp(
            frequencies, frequencies[valid],
            np.log(np.maximum(residual_magnitude[valid], 1e-12)),
        )
        bin_hz = max(1e-9, float(frequencies[1] - frequencies[0])) \
            if len(frequencies) > 1 else 1.0
        width = max(5, int(round(
            config.stochastic_shape_smoothing_f0 * f0_hz / bin_hz
        )))
        if width % 2 == 0:
            width += 1
        width = min(width, max(5, len(interpolated) // 3 * 2 + 1))
        half = width // 2
        smooth = np.convolve(
            np.pad(interpolated, (half, half), mode="edge"),
            np.ones(width, np.float64) / width,
            mode="valid",
        )
        active = frequencies >= max(1.0, config.low_noise_cutoff_hz)
        center = float(np.median(smooth[active])) if np.any(active) else 0.0
        bound = math.log(10.0) * config.maximum_stochastic_shape_db / 20.0
        shape = np.exp(np.clip(smooth - center, -bound, bound))
    else:
        shape = np.ones_like(frequencies)
    # Real voiced/devoiced controls retain F1 but show less stochastic energy
    # in the glottal-to-F1 valley. Anchor the correction to the frame's own
    # first broad tract peak rather than hard-coding the /e/ reference band.
    search = (
        (frequencies >= max(220.0, 1.35 * f0_hz))
        & (frequencies <= min(1200.0, frequencies[-1]))
    )
    if np.any(search):
        indices = np.flatnonzero(search)
        f1_hz = float(frequencies[
            indices[int(np.argmax(envelope[indices]))]
        ])
        anchor_frequencies = np.asarray((
            0.0,
            0.35 * f1_hz,
            0.58 * f1_hz,
            0.84 * f1_hz,
            f1_hz,
            frequencies[-1],
        ), np.float64)
        anchor_gains = np.asarray((
            config.stochastic_glottal_shelf_db,
            config.stochastic_glottal_shelf_db,
            config.stochastic_pre_formant_valley_db,
            config.stochastic_pre_formant_valley_db,
            0.0,
            0.0,
        ), np.float64)
        pre_formant_shape = 10.0 ** (
            np.interp(frequencies, anchor_frequencies, anchor_gains) / 20.0
        )
    else:
        pre_formant_shape = np.ones_like(frequencies)
    candidate = (
        phase * shape * pre_formant_shape
        * _stochastic_taper(frequencies, config)
    )
    filtered_energy = float(np.sum(np.abs(candidate * envelope) ** 2))
    if filtered_energy <= 1e-18 or reference_energy <= 1e-18:
        return np.zeros_like(candidate)
    return candidate * math.sqrt(reference_energy / filtered_energy)


def _analyze_frame(frame: np.ndarray, sample_rate: int, nfft: int,
                   analysis_window: np.ndarray, noise_frame: np.ndarray,
                   config: SourceFilterVoicingConfig) -> _FrameAnalysis:
    windowed = np.asarray(frame, np.float64) * analysis_window
    spectrum = np.fft.rfft(windowed, nfft)
    f0_hz, periodicity = _estimate_f0(frame, sample_rate, config)
    if f0_hz is None:
        ones = np.ones_like(spectrum.real)
        return _FrameAnalysis(
            spectrum=spectrum,
            envelope=ones,
            harmonic=np.zeros_like(spectrum),
            aperiodic=spectrum.copy(),
            stochastic=spectrum.copy(),
            periodicity=periodicity,
            f0_hz=None,
            harmonic_fraction=0.0,
            aperiodic_fraction=1.0,
            stochastic_fraction=1.0,
            source_voicing=0.0,
        )
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    envelope = _smooth_log_envelope(
        np.abs(spectrum), f0_hz, sample_rate / nfft, config
    )
    residual = spectrum / np.maximum(envelope, 1e-10)
    mask = _harmonic_mask(frequencies, f0_hz, periodicity, config)
    harmonic = residual * mask
    aperiodic = residual - harmonic
    filtered_harmonic = harmonic * envelope
    filtered_aperiodic = aperiodic * envelope
    h_energy = float(np.sum(np.abs(filtered_harmonic) ** 2))
    n_energy = float(np.sum(np.abs(filtered_aperiodic) ** 2))
    total = h_energy + n_energy + 1e-18
    stochastic_energy = max(
        n_energy,
        config.minimum_aperiodic_fraction * total,
    )
    stochastic = _stochastic_residual(
        noise_frame, analysis_window, nfft, envelope, frequencies,
        residual, mask, f0_hz, stochastic_energy, config,
    )
    h_fraction = max(0.0, min(1.0, h_energy / total))
    n_fraction = max(0.0, min(1.0, n_energy / total))
    modeled_total = h_energy + stochastic_energy + 1e-18
    stochastic_fraction = max(
        0.0, min(1.0, stochastic_energy / modeled_total)
    )
    return _FrameAnalysis(
        spectrum=spectrum,
        envelope=envelope,
        harmonic=harmonic,
        aperiodic=aperiodic,
        stochastic=stochastic,
        periodicity=periodicity,
        f0_hz=f0_hz,
        harmonic_fraction=h_fraction,
        aperiodic_fraction=n_fraction,
        stochastic_fraction=stochastic_fraction,
        source_voicing=h_fraction,
    )


def transform_voicing(
    samples: Sequence[float],
    sample_rate: int,
    target_curve: Sequence[Sequence[float]] | None = None,
    *,
    config: SourceFilterVoicingConfig | None = None,
) -> VoicingTransformResult:
    """Analyze and optionally render a continuous 0..1 voicing curve.

    ``1`` favors the measured harmonic residual and ``0`` favors the measured
    aperiodic residual.  The renderer only creates a component when the source
    contains enough evidence for it; this prevents nearly periodic vowels from
    turning into amplified numerical residue.
    """
    cfg = config or SourceFilterVoicingConfig()
    values = np.asarray(samples, np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not values.size:
        return VoicingTransformResult(
            samples=np.asarray([], np.float32), source_curve=(), target_curve=(),
            frame_diagnostics=(), modified_frame_count=0,
            skipped_frame_count=0, reconstruction_nrmse=0.0,
        )
    frame_length = max(64, int(round(cfg.frame_seconds * sample_rate)))
    if frame_length % 2:
        frame_length += 1
    hop = max(8, int(round(cfg.hop_seconds * sample_rate)))
    half = frame_length // 2
    nfft = 1 << int(math.ceil(math.log2(frame_length)))
    window = np.sqrt(np.maximum(np.hanning(frame_length), 1e-8))
    pad_mode = "reflect" if values.size > 1 else "edge"
    padded = np.pad(values, (half, half), mode=pad_mode)
    noise = _continuous_noise(values, sample_rate, len(padded))
    centers = np.arange(0, values.size, hop, dtype=int)
    if not len(centers) or centers[-1] != values.size - 1:
        centers = np.append(centers, values.size - 1)
    times = centers.astype(np.float64) / float(sample_rate)
    analyses = [
        _analyze_frame(
            padded[int(center):int(center) + frame_length],
            sample_rate, nfft, window,
            noise[int(center):int(center) + frame_length], cfg,
        )
        for center in centers
    ]
    source_values = np.asarray(
        [item.source_voicing for item in analyses], np.float64
    )
    targets = _interp_curve(target_curve, times, source_values)
    source_curve = tuple(
        (round(float(time), 6), round(float(value), 6))
        for time, value in zip(times, source_values)
    )
    rendered_curve = tuple(
        (round(float(time), 6), round(float(value), 6))
        for time, value in zip(times, targets)
    )
    # Analysis-only is exact and avoids needless floating-point churn.
    if target_curve is None:
        diagnostics = tuple(
            VoicingFrameDiagnostic(
                time=round(float(time), 6),
                source_voicing=round(float(item.source_voicing), 6),
                target_voicing=round(float(item.source_voicing), 6),
                periodicity=round(float(item.periodicity), 6),
                f0_hz=(round(float(item.f0_hz), 4)
                       if item.f0_hz is not None else None),
                harmonic_fraction=round(float(item.harmonic_fraction), 6),
                aperiodic_fraction=round(float(item.aperiodic_fraction), 6),
                stochastic_fraction=round(
                    float(item.stochastic_fraction), 6),
                stochastic_excitation=False,
                gain_db=0.0,
                applied=False,
                reason="analysis_only",
            )
            for time, item in zip(times, analyses)
        )
        return VoicingTransformResult(
            samples=np.asarray(values, np.float32),
            source_curve=source_curve, target_curve=rendered_curve,
            frame_diagnostics=diagnostics, modified_frame_count=0,
            skipped_frame_count=0, reconstruction_nrmse=0.0,
        )

    output = np.zeros_like(padded)
    weights = np.zeros_like(padded)
    diagnostics: list[VoicingFrameDiagnostic] = []
    modified = skipped = 0
    for center, time, target, analysis in zip(
            centers, times, targets, analyses):
        source = float(analysis.source_voicing)
        requested = max(0.0, min(1.0, float(target)))
        spectrum = analysis.spectrum
        applied = False
        applied_gain_db = 0.0
        reason = "unchanged"
        if abs(requested - source) > cfg.unchanged_tolerance:
            if analysis.f0_hz is None and requested > source:
                skipped += 1
                reason = "insufficient_harmonic_source"
            elif (requested > source and analysis.harmonic_fraction <
                  cfg.minimum_harmonic_fraction):
                skipped += 1
                reason = "insufficient_harmonic_source"
            else:
                harmonic_scale = math.sqrt(
                    requested / max(source, 1e-9)
                ) if requested > 0.0 else 0.0
                noise_scale = math.sqrt(
                    (1.0 - requested) / max(
                        1.0 - source, cfg.minimum_aperiodic_fraction
                    )
                ) if requested < 1.0 else 0.0
                harmonic_scale = min(
                    cfg.maximum_component_gain, harmonic_scale
                )
                noise_scale = min(cfg.maximum_component_gain, noise_scale)
                # Near the measured operating point retain the recorded
                # aperiodic residual.  Toward zero progressively replace it
                # with a continuous stochastic excitation.  At exactly zero
                # no phase-coherent residual remains to produce vocal-fry or
                # growl, while every point still uses the same tract envelope.
                stochastic_weight = 0.0
                if requested < source and source > 1e-9:
                    stochastic_weight = min(
                        1.0,
                        max(0.0, (source - requested) / source)
                        ** cfg.residual_noise_blend_power,
                    )
                applied_gain_db = (
                    cfg.zero_voicing_gain_db
                    * stochastic_weight ** cfg.devoicing_gain_power
                )
                aperiodic = (
                    (1.0 - stochastic_weight) * analysis.aperiodic
                    + stochastic_weight * analysis.stochastic
                )
                spectrum = analysis.envelope * (
                    harmonic_scale * analysis.harmonic
                    + noise_scale * aperiodic
                )
                source_time = np.fft.irfft(analysis.spectrum, nfft)[:frame_length]
                target_time = np.fft.irfft(spectrum, nfft)[:frame_length]
                source_rms = _rms(source_time)
                target_rms = _rms(target_time)
                if source_rms > 1e-9 and target_rms > 1e-9:
                    spectrum *= (
                        source_rms / target_rms
                        * 10.0 ** (applied_gain_db / 20.0)
                    )
                if np.all(np.isfinite(spectrum)):
                    applied = True
                    modified += 1
                    reason = (
                        "source_filter_stochastic_mix"
                        if stochastic_weight > 0.0
                        else "source_filter_measured_mix"
                    )
                else:
                    spectrum = analysis.spectrum
                    skipped += 1
                    reason = "non_finite_reconstruction"
        frame_time = np.fft.irfft(spectrum, nfft)[:frame_length]
        start = int(center)
        output[start:start + frame_length] += frame_time * window
        weights[start:start + frame_length] += window * window
        diagnostics.append(VoicingFrameDiagnostic(
            time=round(float(time), 6),
            source_voicing=round(source, 6),
            target_voicing=round(requested, 6),
            periodicity=round(float(analysis.periodicity), 6),
            f0_hz=(round(float(analysis.f0_hz), 4)
                   if analysis.f0_hz is not None else None),
            harmonic_fraction=round(float(analysis.harmonic_fraction), 6),
            aperiodic_fraction=round(float(analysis.aperiodic_fraction), 6),
            stochastic_fraction=round(
                float(analysis.stochastic_fraction), 6),
            stochastic_excitation=(
                applied and requested < source
            ),
            gain_db=round(float(applied_gain_db), 6),
            applied=applied,
            reason=reason,
        ))
    reconstructed = np.divide(
        output, weights, out=padded.copy(), where=weights > 1e-10
    )[half:half + values.size]
    # The OLA identity error is measured separately from the requested change
    # by reconstructing all original spectra with the same windows.
    identity = np.zeros_like(padded)
    identity_weight = np.zeros_like(padded)
    for center, analysis in zip(centers, analyses):
        frame_time = np.fft.irfft(
            analysis.spectrum, nfft)[:frame_length]
        start = int(center)
        identity[start:start + frame_length] += frame_time * window
        identity_weight[start:start + frame_length] += window * window
    identity = np.divide(
        identity, identity_weight, out=padded.copy(),
        where=identity_weight > 1e-10,
    )[half:half + values.size]
    nrmse = _normalized_error(values, identity)
    if not np.all(np.isfinite(reconstructed)):
        reconstructed = values.copy()
        modified = 0
        skipped = len(analyses)
    return VoicingTransformResult(
        samples=np.asarray(reconstructed, np.float32),
        source_curve=source_curve,
        target_curve=rendered_curve,
        frame_diagnostics=tuple(diagnostics),
        modified_frame_count=modified,
        skipped_frame_count=skipped,
        reconstruction_nrmse=round(float(nrmse), 9),
    )


def simplify_curve(points: Sequence[Sequence[float]],
                   tolerance: float = 0.015) -> list[tuple[float, float]]:
    """Remove nearly collinear points while preserving extrema and endpoints."""
    source = [(float(item[0]), float(item[1])) for item in points]
    if len(source) <= 2:
        return source
    kept = [source[0]]
    for index in range(1, len(source) - 1):
        previous = kept[-1]
        current = source[index]
        following = source[index + 1]
        span = following[0] - previous[0]
        expected = previous[1] if abs(span) < 1e-12 else (
            previous[1] + (following[1] - previous[1])
            * (current[0] - previous[0]) / span
        )
        if abs(current[1] - expected) > tolerance:
            kept.append(current)
    kept.append(source[-1])
    return [(round(time, 6), round(value, 6)) for time, value in kept]


def curve_for_regions(
    source_curve: Sequence[Sequence[float]],
    regions: Sequence[Mapping[str, float]],
    *,
    target_voicing: float = 0.16,
    minimum_ramp_seconds: float = 0.006,
) -> list[tuple[float, float]]:
    """Create a smooth automatic curve with source-valued region edges.

    A region may provide its own ``target_voicing``.  The optional field keeps
    per-mora linguistic decisions independent while preserving the original
    one-target behavior for existing callers.
    """
    curve = [(float(item[0]), float(item[1])) for item in source_curve]
    if not curve:
        return []
    result: list[tuple[float, float]] = []
    floor = max(0.0, min(1.0, float(target_voicing)))
    for time, source in curve:
        target = source
        for row in regions:
            start = float(row["start"])
            end = float(row["end"])
            if not start <= time <= end or end <= start:
                continue
            ramp = min(
                (end - start) * 0.32,
                max(minimum_ramp_seconds, (end - start) * 0.18),
            )
            if ramp <= 1e-9:
                local = 1.0
            else:
                local = min(1.0, (time - start) / ramp,
                            (end - time) / ramp)
                local = 0.5 - 0.5 * math.cos(math.pi * max(0.0, local))
            region_floor = max(0.0, min(
                1.0, float(row.get("target_voicing", floor))
            ))
            proposal = (
                source * (1.0 - local)
                + min(source, region_floor) * local
            )
            target = min(target, proposal)
        result.append((round(time, 6), round(target, 6)))
    # Frame centers seldom land exactly on linguistic boundaries. Explicit
    # source-valued anchors prevent interpolation from beginning the change
    # in an adjacent consonant or leaving a residual change after the vowel.
    source_x = np.asarray([item[0] for item in curve], np.float64)
    source_y = np.asarray([item[1] for item in curve], np.float64)
    for row in regions:
        for boundary in (float(row["start"]), float(row["end"])):
            value = float(np.interp(boundary, source_x, source_y))
            result.append((round(boundary, 6), round(value, 6)))
    deduplicated = {}
    for time, value in sorted(result):
        deduplicated[float(time)] = float(value)
    return [(time, deduplicated[time]) for time in sorted(deduplicated)]
