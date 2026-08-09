"""Deterministic source-speaker pitch analysis shared by every builder.

UTAU ``FREQ0003`` files are preferred because they describe the source
recordings directly.  When they are absent or unusable, short deterministic
windows from the source WAV inventory are estimated with autocorrelation.
All serialized paths are relative to the sample root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import statistics
import struct
from typing import Mapping, Sequence
import wave


FRQ_MAGIC = b"FREQ0003"
MIN_F0_HZ = 60.0
MAX_F0_HZ = 800.0
DEFAULT_MEDIAN_F0_HZ = 185.0
DEFAULT_LOW_F0_HZ = 140.0
DEFAULT_HIGH_F0_HZ = 260.0
# The generated voice's reference/default pitch must describe the selected
# source bank. Linguistic contours create their own excursions around that
# reference; pre-transposing the manifest made an E3 bank report about 202 Hz.
AUTOMATIC_CONTOUR_HEADROOM_SEMITONES = 0.0


@dataclass(frozen=True)
class PitchDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    path: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.path:
            result["path"] = self.path
        if self.details:
            result["details"] = {
                key: self.details[key] for key in sorted(self.details)
            }
        return result


@dataclass(frozen=True)
class SpeakerPitchStatistics:
    median_f0_hz: float
    low_percentile_f0_hz: float
    high_percentile_f0_hz: float
    voiced_sample_count: int
    source: str
    files_used: tuple[str, ...] = ()
    diagnostics: tuple[PitchDiagnostic, ...] = ()
    percentile_definition: str = "p10_p90"

    def to_dict(self) -> dict[str, object]:
        return {
            "median_f0_hz": round(float(self.median_f0_hz), 6),
            "low_percentile_f0_hz": round(
                float(self.low_percentile_f0_hz), 6
            ),
            "high_percentile_f0_hz": round(
                float(self.high_percentile_f0_hz), 6
            ),
            "voiced_sample_count": int(self.voiced_sample_count),
            "source": self.source,
            "files_used": list(self.files_used),
            "percentile_definition": self.percentile_definition,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _valid_f0(value: float) -> bool:
    return math.isfinite(value) and MIN_F0_HZ < value < MAX_F0_HZ


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _recording_scope(
    root: Path,
    recording_files: Sequence[Path | str] | None,
) -> tuple[Path, ...] | None:
    if recording_files is None:
        return None
    selected = []
    for raw in recording_files:
        path = Path(raw).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"pitch-analysis recording is outside the sample root: {path}"
            ) from exc
        if path.is_file() and path.suffix.casefold() == ".wav":
            selected.append(path)
    return tuple(sorted(
        set(selected), key=lambda path: _relative(path, root).casefold()
    ))


def _matching_frq_files(recording_files: Sequence[Path]) -> tuple[Path, ...]:
    result = []
    for wav_path in recording_files:
        candidates = (
            wav_path.with_name(wav_path.stem + ".frq"),
            wav_path.with_name(wav_path.name + ".frq"),
            wav_path.with_name(wav_path.name.replace(".", "_") + ".frq"),
        )
        result.extend(path for path in candidates if path.is_file())
    return tuple(sorted(set(result), key=lambda path: str(path).casefold()))


def _frq_statistics(
    root: Path,
    frq_scope: Sequence[Path] | None = None,
) -> tuple[SpeakerPitchStatistics | None, tuple[PitchDiagnostic, ...]]:
    diagnostics: list[PitchDiagnostic] = []
    representatives: list[float] = []
    frame_values: list[float] = []
    files_used: list[str] = []
    if frq_scope is None:
        frq_files = sorted(
            (path for path in root.rglob("*")
             if path.is_file() and path.suffix.casefold() == ".frq"),
            key=lambda path: _relative(path, root).casefold(),
        )
    else:
        frq_files = sorted(
            set(frq_scope), key=lambda path: _relative(path, root).casefold()
        )
    if not frq_files:
        diagnostics.append(PitchDiagnostic(
            code="frq_not_found",
            message="No UTAU FREQ0003 files were found under the sample root.",
            severity="info",
        ))
        return None, tuple(diagnostics)

    for path in frq_files:
        relative = _relative(path, root)
        try:
            data = path.read_bytes()
        except OSError as exc:
            diagnostics.append(PitchDiagnostic(
                code="frq_read_failed",
                message=f"Could not read FRQ data: {exc}",
                path=relative,
            ))
            continue
        if len(data) < 40 or data[:8] != FRQ_MAGIC:
            diagnostics.append(PitchDiagnostic(
                code="frq_invalid_header",
                message="The file is not a complete UTAU FREQ0003 file.",
                path=relative,
            ))
            continue
        try:
            average_f0 = struct.unpack_from("<d", data, 12)[0]
            declared_count = struct.unpack_from("<i", data, 36)[0]
        except struct.error:
            diagnostics.append(PitchDiagnostic(
                code="frq_invalid_header",
                message="The FREQ0003 header could not be decoded.",
                path=relative,
            ))
            continue
        if declared_count < 0 or declared_count > 10_000_000:
            diagnostics.append(PitchDiagnostic(
                code="frq_invalid_frame_count",
                message="The FREQ0003 frame count is outside safe limits.",
                path=relative,
                details={"declared_count": declared_count},
            ))
            continue
        expected = 40 + declared_count * 16
        if len(data) < expected:
            diagnostics.append(PitchDiagnostic(
                code="frq_truncated",
                message="The FREQ0003 frame table is truncated.",
                path=relative,
                details={"declared_bytes": expected, "actual_bytes": len(data)},
            ))
            continue

        values: list[float] = []
        for index in range(declared_count):
            try:
                f0 = struct.unpack_from("<d", data, 40 + index * 16)[0]
            except struct.error:
                values = []
                break
            if _valid_f0(f0):
                values.append(float(f0))
        if not values:
            diagnostics.append(PitchDiagnostic(
                code="frq_no_voiced_frames",
                message="The FRQ file contains no valid voiced F0 frames.",
                path=relative,
            ))
            continue
        if _valid_f0(average_f0):
            representative = float(average_f0)
        else:
            representative = float(statistics.median(values))
            diagnostics.append(PitchDiagnostic(
                code="frq_invalid_average",
                message=(
                    "The FRQ header average was invalid; the median of its "
                    "valid voiced frames was used for this recording."
                ),
                path=relative,
            ))
        representatives.append(representative)
        frame_values.extend(values)
        files_used.append(relative)

    if len(files_used) < 3:
        diagnostics.append(PitchDiagnostic(
            code="frq_insufficient_files",
            message=(
                "Fewer than three usable FRQ files were found; source WAV "
                "pitch estimation will be used instead."
            ),
            severity="info",
            details={"usable_files": len(files_used)},
        ))
        return None, tuple(diagnostics)

    return SpeakerPitchStatistics(
        median_f0_hz=round(statistics.median(representatives), 6),
        low_percentile_f0_hz=round(_percentile(frame_values, 0.10), 6),
        high_percentile_f0_hz=round(_percentile(frame_values, 0.90), 6),
        voiced_sample_count=len(frame_values),
        source="frq",
        files_used=tuple(files_used),
        diagnostics=tuple(diagnostics),
    ), tuple(diagnostics)


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width == 1:
        samples = [float(value - 128) for value in raw]
    elif sample_width == 2:
        count = len(raw) // 2
        samples = [float(value) for value in struct.unpack(
            "<" + "h" * count, raw[:count * 2]
        )]
    elif sample_width == 3:
        samples = []
        for offset in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[offset:offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            samples.append(float(value))
    elif sample_width == 4:
        count = len(raw) // 4
        samples = [float(value) for value in struct.unpack(
            "<" + "i" * count, raw[:count * 4]
        )]
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if channels <= 1:
        return samples
    frames = len(samples) // channels
    return [
        sum(samples[index * channels:(index + 1) * channels]) / channels
        for index in range(frames)
    ]


def _estimate_window_f0(samples: Sequence[float], sample_rate: int) -> float:
    if len(samples) < 64 or sample_rate <= 0:
        return 0.0
    step = max(1, int(round(sample_rate / 8000.0)))
    reduced = [float(value) for value in samples[::step]]
    reduced_rate = sample_rate / step
    mean = sum(reduced) / len(reduced)
    centered = [value - mean for value in reduced]
    energy = sum(value * value for value in centered)
    if energy <= 1e-9:
        return 0.0
    minimum_lag = max(1, int(reduced_rate / 700.0))
    maximum_lag = min(
        int(reduced_rate / MIN_F0_HZ), len(centered) // 2
    )
    correlations: list[tuple[int, float]] = []
    for lag in range(minimum_lag, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        numerator = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(
            sum(value * value for value in left)
            * sum(value * value for value in right)
        )
        correlations.append((lag, numerator / denominator if denominator else 0.0))
    if not correlations:
        return 0.0
    best = max(value for _lag, value in correlations)
    if best < 0.45:
        return 0.0
    # Prefer the first near-equal peak, avoiding octave-low choices when a
    # periodic signal has strong peaks at several multiples of its period.
    lag = next(
        item_lag for item_lag, value in correlations
        if value >= best * 0.985
    )
    f0 = reduced_rate / lag
    return float(f0) if _valid_f0(f0) else 0.0


def _sample_paths(paths: Sequence[Path], maximum: int = 64) -> tuple[Path, ...]:
    if len(paths) <= maximum:
        return tuple(paths)
    indexes = {
        round(index * (len(paths) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return tuple(paths[index] for index in sorted(indexes))


def _waveform_statistics(
    root: Path,
    wav_scope: Sequence[Path] | None = None,
) -> tuple[SpeakerPitchStatistics | None, tuple[PitchDiagnostic, ...]]:
    diagnostics: list[PitchDiagnostic] = []
    if wav_scope is None:
        all_wavs = sorted(
            (path for path in root.rglob("*")
             if path.is_file() and path.suffix.casefold() == ".wav"),
            key=lambda path: _relative(path, root).casefold(),
        )
    else:
        all_wavs = sorted(
            set(wav_scope), key=lambda path: _relative(path, root).casefold()
        )
    selected = _sample_paths(all_wavs)
    if not selected:
        diagnostics.append(PitchDiagnostic(
            code="waveform_not_found",
            message="No source WAV files were found for pitch estimation.",
        ))
        return None, tuple(diagnostics)
    if len(selected) < len(all_wavs):
        diagnostics.append(PitchDiagnostic(
            code="waveform_sampled_inventory",
            message="A deterministic spread of source WAV files was analyzed.",
            severity="info",
            details={"available_files": len(all_wavs), "sampled_files": len(selected)},
        ))

    estimates: list[float] = []
    files_used: list[str] = []
    for path in selected:
        relative = _relative(path, root)
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getcomptype() != "NONE":
                    raise ValueError("compressed WAV is unsupported")
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate()
                total = handle.getnframes()
                window_frames = min(total, max(64, int(rate * 0.18)))
                starts = sorted({
                    max(0, min(total - window_frames,
                               int(total * fraction - window_frames / 2)))
                    for fraction in (0.25, 0.50, 0.75)
                })
                file_estimates = []
                for start in starts:
                    handle.setpos(start)
                    raw = handle.readframes(window_frames)
                    estimate = _estimate_window_f0(
                        _decode_pcm(raw, width, channels), rate
                    )
                    if estimate:
                        file_estimates.append(estimate)
        except (OSError, EOFError, wave.Error, ValueError, struct.error) as exc:
            diagnostics.append(PitchDiagnostic(
                code="waveform_estimation_failed",
                message=f"Could not estimate pitch from this WAV: {exc}",
                path=relative,
            ))
            continue
        if file_estimates:
            estimates.extend(file_estimates)
            files_used.append(relative)

    if len(estimates) < 3:
        diagnostics.append(PitchDiagnostic(
            code="waveform_insufficient_voicing",
            message=(
                "Fewer than three voiced waveform windows were usable; a "
                "conservative fixed pitch range will be used."
            ),
            severity="info",
            details={"voiced_windows": len(estimates)},
        ))
        return None, tuple(diagnostics)

    return SpeakerPitchStatistics(
        median_f0_hz=round(statistics.median(estimates), 6),
        low_percentile_f0_hz=round(_percentile(estimates, 0.10), 6),
        high_percentile_f0_hz=round(_percentile(estimates, 0.90), 6),
        voiced_sample_count=len(estimates),
        source="waveform_estimation",
        files_used=tuple(files_used),
        diagnostics=tuple(diagnostics),
    ), tuple(diagnostics)


def analyze_speaker_pitch(
    sample_root: Path | str,
    *,
    recording_files: Sequence[Path | str] | None = None,
) -> SpeakerPitchStatistics:
    """Analyze source recordings without writing into their bank.

    ``recording_files`` constrains both waveform and adjacent FRQ discovery.
    Builders use this to prevent one selected pitch from borrowing pitch
    statistics from another subbank under the same source root.
    """
    root = Path(sample_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source sample root not found: {root}")
    wav_scope = _recording_scope(root, recording_files)
    frq_scope = (
        _matching_frq_files(wav_scope) if wav_scope is not None else None
    )
    frq, frq_diagnostics = _frq_statistics(root, frq_scope)
    if frq is not None:
        return frq
    waveform, waveform_diagnostics = _waveform_statistics(root, wav_scope)
    combined = tuple(frq_diagnostics) + tuple(waveform_diagnostics)
    if waveform is not None:
        return SpeakerPitchStatistics(
            median_f0_hz=waveform.median_f0_hz,
            low_percentile_f0_hz=waveform.low_percentile_f0_hz,
            high_percentile_f0_hz=waveform.high_percentile_f0_hz,
            voiced_sample_count=waveform.voiced_sample_count,
            source=waveform.source,
            files_used=waveform.files_used,
            diagnostics=combined,
        )
    return SpeakerPitchStatistics(
        median_f0_hz=DEFAULT_MEDIAN_F0_HZ,
        low_percentile_f0_hz=DEFAULT_LOW_F0_HZ,
        high_percentile_f0_hz=DEFAULT_HIGH_F0_HZ,
        voiced_sample_count=0,
        source="fallback",
        diagnostics=combined + (PitchDiagnostic(
            code="speaker_pitch_fallback",
            message=(
                "No reliable FRQ or waveform pitch evidence was available; "
                "the legacy conservative speaker range was used."
            ),
        ),),
    )


def pitchmark_bounds(
    statistics: SpeakerPitchStatistics,
) -> tuple[float, float]:
    """Return broad EST tracking bounds while retaining English behavior."""
    median = float(statistics.median_f0_hz)
    low = min(median * 0.62, statistics.low_percentile_f0_hz * 0.85)
    high = max(median * 1.70, statistics.high_percentile_f0_hz * 1.20)
    low = max(30.0, min(low, 950.0))
    high = max(low + 20.0, min(high, 1000.0))
    return round(low, 1), round(high, 1)


def recommended_default_pitch_hz(
    statistics: SpeakerPitchStatistics,
    *,
    headroom_semitones: float = AUTOMATIC_CONTOUR_HEADROOM_SEMITONES,
) -> float:
    """Return the source median, unless an explicit offset is requested.

    Builders call this with zero headroom so ``average_pitch_hz`` and the GUI
    default remain honest source-bank metadata. The optional argument is kept
    for callers that intentionally request a transposed synthesis baseline.
    """
    headroom = max(0.0, min(12.0, float(headroom_semitones)))
    median = max(40.0, min(700.0, float(statistics.median_f0_hz)))
    return round(min(700.0, median * (2.0 ** (headroom / 12.0))), 6)


def automatic_pitch_metadata(
    statistics: SpeakerPitchStatistics,
    *,
    default_is_automatic: bool,
    headroom_semitones: float = AUTOMATIC_CONTOUR_HEADROOM_SEMITONES,
) -> dict[str, object]:
    """Return the serializable pitch policy shared by all entry points."""
    return {
        "automatic_pitch_floor_hz": round(
            float(statistics.median_f0_hz), 6
        ),
        "automatic_pitch_headroom_semitones": round(
            max(0.0, min(12.0, float(headroom_semitones))), 6
        ),
        "default_pitch_source": (
            ("speaker_median_plus_headroom"
             if float(headroom_semitones) > 0.0 else "speaker_median")
            if default_is_automatic else "builder_override"
        ),
    }
