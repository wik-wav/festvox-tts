"""Phase 3 Japanese UTAU candidate-to-Festival compiler.

This module consumes the language-neutral candidate graph from Phase 2 and
creates a separate Japanese UniSyn voice.  It never imports or mutates the
ARPAsing converter, its phone map, or the generated English voice entry point.

UTAU OTO geometry exposes an authoritative phone boundary at
``offset + preutterance``. Consonant-bearing recordings are split around one
bounded phone-center anchor so adjacent diphones cannot replay the consonant.
Missing pure-CV transitions receive visible generated-output bridges instead
of Festival's hidden default silence. These assumptions are recorded in exact
source-contribution metadata; they remain a listening baseline, not a
naturalness claim.
"""

from __future__ import annotations

import argparse
import array
import bisect
from dataclasses import dataclass, field, replace
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import wave
from typing import Iterable, Mapping, Optional, Sequence

from japanese_candidates import (
    JapaneseCandidateGraph,
    JapaneseSourceCandidate,
    runtime_family_allowed,
    runtime_family_policy,
)
from join_synthesis import (
    JOIN_SYNTHESIS_CONDITIONING_VERSION,
    JoinConstraintError,
    JoinSynthesisConfig,
    adaptive_join_pcm16,
)
from source_timing import build_source_timing_profile
from special_phones import generated_voice_policy
from source_window import (
    DEFAULT_SOURCE_WINDOW_MS,
    DEFAULT_ZERO_OVERLAP_GUARD_MS,
    SOURCE_WINDOW_MODES,
    build_source_window_plan,
    effective_oto_overlap_ms,
    normalize_source_window_mode,
    normalize_zero_overlap_guard_ms,
    source_window_variant_names,
)
from unisyn_runtime import (
    RUNTIME_AUDIO_STORAGE_MODES,
    separate_runtime_metadata,
)
from voice_manifest import generated_voice_fields


JAPANESE_FESTIVAL_SCHEMA_VERSION = 1
JAPANESE_FESTIVAL_SCHEMA_STATUS = "phase3-provisional"
JAPANESE_FESTIVAL_BUILDER_VERSION = "3.0"
VOWEL_BLEND_LONG_DURATION_FACTOR = 1.5

_SCHEME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_@:~#]*\Z")
_VOWELS = {"a", "i", "u", "e", "o"}
_SILENCE_PHONES = {"pau", "sil"}
_NONLEXICAL_PHONES = _SILENCE_PHONES | {"cl"}
_STOP_OR_AFFRICATE_PHONES = {
    "p", "b", "t", "d", "k", "g", "q", "ch", "jh", "ts", "dz",
    "py", "by", "ty", "dy", "ky", "gy",
}
_VOICELESS_STOPS = {"p", "t", "k", "q", "py", "ty", "ky", "cl"}
_VOICED_STOPS = {"b", "d", "g", "by", "dy", "gy", "dx", "dxy"}
_VOICELESS_AFFRICATES = {"ch", "ts"}
_VOICED_AFFRICATES = {"j", "jh", "dz"}
_VOICELESS_FRICATIVES = {"f", "s", "sh", "h", "hy", "fy"}
_VOICED_FRICATIVES = {"v", "z", "zh"}
_NASALS = {"N", "m", "my", "n", "ny", "ng", "ngy"}
_LIQUIDS = {"r", "ry"}
_GLIDES = {"w", "y"}
F0_FALLBACK_ESTIMATORS = ("harvest", "dio")
_DEFAULT_DURATIONS = {
    "pau": 0.12,
    "sil": 0.12,
    "cl": 0.08,
    "N": 0.11,
    "a": 0.12,
    "i": 0.11,
    "u": 0.11,
    "e": 0.12,
    "o": 0.12,
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _scheme_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _safe_token(value: str, owner: str = "phone") -> str:
    token = str(value)
    if not _SCHEME_TOKEN.fullmatch(token):
        raise ValueError(f"{owner} is not a safe Festival token: {token!r}")
    return token


def _safe_voice_name(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
    token = token.strip("_") or "japanese_utau"
    if token[0].isdigit():
        token = "j_" + token
    return _safe_token(token, "voice name")


def _portable_runtime_path(path: Path) -> str:
    """Return a Festival-readable path without changing output ownership."""
    resolved = path.resolve()
    text = str(resolved)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match:
        tail = match.group(2).replace("\\", "/")
        return f"/mnt/{match.group(1).lower()}/{tail}"
    return text.replace("\\", "/")


def _wav_duration(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
        if sample_rate <= 0:
            raise wave.Error("sample rate is zero")
        return frames / float(sample_rate), sample_rate


def _copy_name(source_relative: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(source_relative).stem).strip("_")
    stem = stem[:40] or "source"
    digest = hashlib.sha256(source_relative.encode("utf-8")).hexdigest()[:12]
    return f"j_{stem}_{digest}.wav"


@dataclass(frozen=True)
class JapaneseBuildDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    candidate_id: Optional[str] = None
    source_path: Optional[str] = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.candidate_id is not None:
            result["candidate_id"] = self.candidate_id
        if self.source_path is not None:
            result["source_path"] = self.source_path
        if self.details:
            result["details"] = dict(sorted(self.details.items()))
        return result


@dataclass(frozen=True)
class JapaneseCompiledUnit:
    candidate_id: str
    edge_index: int
    edge_offset: int
    diphone: str
    left_phone: str
    right_phone: str
    left_name: str
    index_name: str
    wav_name: str
    start: float
    midpoint: float
    end: float
    role: str
    family: str
    selection_cost: float
    geometry_method: str
    source_path: str
    source_alias: str
    source_oto_path: str
    source_oto_line: int
    shared_anchor: Optional[float]
    oto_offset_ms: float
    oto_consonant_ms: float
    oto_cutoff_ms: float
    oto_preutterance_ms: float
    oto_overlap_ms: float
    effective_overlap_ms: float = 0.0
    overlap_method: str = "oto"
    recorded_left_context: str = "*"
    recorded_right_context: str = "*"
    moraic_nasal_allophone: str = ""
    source_pitch_tags: tuple[str, ...] = ()
    subbank_ids: tuple[str, ...] = ()
    source_components: tuple[Mapping[str, object], ...] = ()
    source_window: Mapping[str, object] = field(default_factory=dict)
    window_left_name: str = ""
    window_right_name: str = ""
    window_both_name: str = ""
    window_left_activation: Optional[float] = None
    window_right_activation: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "edge_index": self.edge_index,
            "edge_offset": self.edge_offset,
            "diphone": self.diphone,
            "left_phone": self.left_phone,
            "right_phone": self.right_phone,
            "left_name": self.left_name,
            "index_name": self.index_name,
            "wav_name": self.wav_name,
            "start": self.start,
            "midpoint": self.midpoint,
            "end": self.end,
            "role": self.role,
            "family": self.family,
            "selection_cost": self.selection_cost,
            "geometry_method": self.geometry_method,
            "source_path": self.source_path,
            "source_alias": self.source_alias,
            "source_oto_path": self.source_oto_path,
            "source_oto_line": self.source_oto_line,
            "shared_anchor": self.shared_anchor,
            "oto_timing_ms": {
                "offset": self.oto_offset_ms,
                "consonant": self.oto_consonant_ms,
                "cutoff": self.oto_cutoff_ms,
                "preutterance": self.oto_preutterance_ms,
                "overlap": self.oto_overlap_ms,
            },
            "effective_overlap_ms": self.effective_overlap_ms,
            "overlap_method": self.overlap_method,
            "recorded_left_context": self.recorded_left_context,
            "recorded_right_context": self.recorded_right_context,
            "moraic_nasal_allophone": self.moraic_nasal_allophone,
            "source_pitch_tags": list(self.source_pitch_tags),
            "subbank_ids": list(self.subbank_ids),
            "source_components": [
                dict(component) for component in self.source_components
            ],
            "source_window": dict(self.source_window),
            "window_left_name": self.window_left_name or self.left_name,
            "window_right_name": self.window_right_name or self.left_name,
            "window_both_name": self.window_both_name or self.left_name,
            "window_left_activation": self.window_left_activation,
            "window_right_activation": self.window_right_activation,
        }


@dataclass(frozen=True)
class JapaneseFestivalBuild:
    voice_name: str
    voice_entry_point: str
    phones: tuple[str, ...]
    units: tuple[JapaneseCompiledUnit, ...]
    index: Mapping[str, tuple[str, float, float, float]]
    alternatives: Mapping[str, tuple[Mapping[str, object], ...]]
    candidate_units: Mapping[str, tuple[Mapping[str, object], ...]]
    voice_manifest: Mapping[str, object]
    diagnostics: tuple[JapaneseBuildDiagnostic, ...]
    average_pitch_hz: float
    source_candidate_count: int
    selectable_candidate_count: int
    compiled_candidate_count: int
    output_relative_files: tuple[str, ...]
    source_window_policy: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = JAPANESE_FESTIVAL_SCHEMA_VERSION
    schema_status: str = JAPANESE_FESTIVAL_SCHEMA_STATUS
    builder_version: str = JAPANESE_FESTIVAL_BUILDER_VERSION
    _output_root: Optional[Path] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "builder_version": self.builder_version,
            "kind": "japanese_festival_unisyn_build",
            "language": "ja",
            "voice_name": self.voice_name,
            "voice_entry_point": self.voice_entry_point,
            "phones": list(self.phones),
            "units": [item.to_dict() for item in self.units],
            "index": {
                key: list(self.index[key]) for key in sorted(self.index)
            },
            "alternatives": {
                key: [dict(item) for item in self.alternatives[key]]
                for key in sorted(self.alternatives)
            },
            "candidate_units": {
                key: [dict(item) for item in self.candidate_units[key]]
                for key in sorted(self.candidate_units)
            },
            "average_pitch_hz": self.average_pitch_hz,
            "source_candidate_count": self.source_candidate_count,
            "selectable_candidate_count": self.selectable_candidate_count,
            "compiled_candidate_count": self.compiled_candidate_count,
            "output_relative_files": list(self.output_relative_files),
            "source_window_policy": dict(self.source_window_policy),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }
        result.update(dict(self.voice_manifest))
        return result

    def metadata_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


@dataclass(frozen=True)
class _EdgeProposal:
    left: str
    right: str
    start_ms: float
    midpoint_ms: float
    end_ms: float
    edge_offset: int
    method: str
    shared_anchor_ms: Optional[float] = None
    effective_overlap_ms: float = 0.0
    overlap_method: str = "oto"


@dataclass(frozen=True)
class _BridgeHalf:
    candidate: JapaneseSourceCandidate
    wav_name: str
    start_ms: float
    end_ms: float
    purpose: str
    preferred_crossfade_ms: Optional[float] = None
    effective_overlap_ms: Optional[float] = None
    overlap_method: str = ""
    indexed_end_ms: Optional[float] = None

    def source_component(self) -> Mapping[str, object]:
        result = {
            "purpose": self.purpose,
            "candidate_id": self.candidate.candidate_id,
            "role": self.candidate.role,
            "family": self.candidate.family,
            "alias": self.candidate.source.alias_raw,
            "wav": self.candidate.source.wav_path or "",
            "oto_file": self.candidate.source.oto_path,
            "oto_line": self.candidate.source.line,
            "source_slice": {
                "start": round(self.start_ms / 1000.0, 6),
                "end": round(self.end_ms / 1000.0, 6),
            },
            "oto_timing_ms": self.candidate.timing.to_dict(),
        }
        if self.preferred_crossfade_ms is not None:
            result["preferred_crossfade_ms"] = round(
                self.preferred_crossfade_ms, 6
            )
        if self.effective_overlap_ms is not None:
            result["effective_overlap_ms"] = round(
                self.effective_overlap_ms, 6
            )
        if self.overlap_method:
            result["overlap_method"] = self.overlap_method
        if self.indexed_end_ms is not None:
            result["indexed_source_end"] = round(
                self.indexed_end_ms / 1000.0, 6
            )
        return result


@dataclass(frozen=True)
class _BridgeHalfPoolSet:
    """Best generic bridge halves plus consonant onsets by following phone."""

    left_best: Mapping[str, _BridgeHalf]
    left_continuity: Mapping[str, tuple[_BridgeHalf, ...]]
    right_best: Mapping[str, _BridgeHalf]
    right_by_context: Mapping[str, Mapping[str, _BridgeHalf]]


def _decode_pcm_mono(
    data: bytes, sample_width: int, channels: int
) -> tuple[float, ...]:
    if sample_width == 1:
        values = [(value - 128) / 128.0 for value in data]
    elif sample_width == 2:
        count = len(data) // 2
        values = [value / 32768.0 for value in struct.unpack(
            f"<{count}h", data[:count * 2]
        )]
    elif sample_width == 3:
        values = []
        for offset in range(0, len(data) - 2, 3):
            value = int.from_bytes(
                data[offset:offset + 3], "little", signed=False
            )
            if value & 0x800000:
                value -= 0x1000000
            values.append(value / 8388608.0)
    elif sample_width == 4:
        count = len(data) // 4
        values = [value / 2147483648.0 for value in struct.unpack(
            f"<{count}i", data[:count * 4]
        )]
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if channels <= 1:
        return tuple(values)
    return tuple(
        sum(values[offset:offset + channels]) / channels
        for offset in range(0, len(values) - channels + 1, channels)
    )


def _read_pcm_mono(path: Path) -> tuple[tuple[float, ...], int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {path.name}")
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        data = handle.readframes(handle.getnframes())
    return _decode_pcm_mono(data, width, channels), rate


@functools.lru_cache(maxsize=128)
def _read_pcm_mono_slice(
    path_text: str, start_ms: float, end_ms: float
) -> tuple[tuple[float, ...], int]:
    """Read only the bounded source frames needed by one bridge half."""
    path = Path(path_text)
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {path.name}")
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        frame_count = int(handle.getnframes())
        first = max(0, min(
            frame_count, int(round(float(start_ms) * rate / 1000.0))
        ))
        last = max(first, min(
            frame_count, int(round(float(end_ms) * rate / 1000.0))
        ))
        handle.setpos(first)
        data = handle.readframes(last - first)
    return _decode_pcm_mono(data, width, channels), rate


def _resample_linear(
    samples: Sequence[float], source_rate: int, target_rate: int
) -> tuple[float, ...]:
    if source_rate == target_rate or not samples:
        return tuple(samples)
    count = max(1, int(round(len(samples) * target_rate / source_rate)))
    if count == 1 or len(samples) == 1:
        return (float(samples[0]),) * count
    scale = (len(samples) - 1) / (count - 1)
    result = []
    for index in range(count):
        position = index * scale
        first = int(position)
        second = min(len(samples) - 1, first + 1)
        fraction = position - first
        result.append(
            float(samples[first]) * (1.0 - fraction)
            + float(samples[second]) * fraction
        )
    return tuple(result)


def _bridge_clip(
    wav_root: Path, half: _BridgeHalf, target_rate: Optional[int] = None
) -> tuple[tuple[float, ...], int]:
    path = (wav_root / half.wav_name).resolve()
    clip, rate = _read_pcm_mono_slice(
        str(path), round(float(half.start_ms), 6),
        round(float(half.end_ms), 6),
    )
    if target_rate is not None and rate != target_rate:
        clip = _resample_linear(clip, rate, target_rate)
        rate = target_rate
    return tuple(clip), rate


def _write_pcm16_mono(path: Path, samples: Sequence[float], rate: int) -> None:
    frames = bytearray()
    for sample in samples:
        bounded = max(-1.0, min(1.0, float(sample)))
        frames.extend(struct.pack(
            "<h", max(-32768, min(32767, int(round(bounded * 32767.0))))
        ))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def _legacy_bridge_name(output_name: str) -> str:
    """Return the paired pre-fix bridge WAV name used by Fault Mode."""
    path = Path(output_name)
    return f"_legacy_{path.stem.lstrip('_')}{path.suffix}"


def _bridge_overlap_samples(
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    rate: int,
    requested_overlap_ms: float,
) -> int:
    return min(
        max(1, int(round(rate * requested_overlap_ms / 1000.0))),
        max(1, len(left_samples) // 3),
        max(1, len(right_samples) // 3),
    )


def _legacy_linear_bridge(
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    rate: int,
    requested_overlap_ms: float,
) -> tuple[tuple[float, ...], float, int]:
    """Reproduce the exact pre-fix generated-bridge overlap."""
    overlap = _bridge_overlap_samples(
        left_samples, right_samples, rate, requested_overlap_ms)
    mixed = []
    for index in range(overlap):
        fraction = (index + 1) / (overlap + 1)
        mixed.append(
            left_samples[-overlap + index] * (1.0 - fraction)
            + right_samples[index] * fraction
        )
    output = (
        tuple(left_samples[:-overlap])
        + tuple(mixed)
        + tuple(right_samples[overlap:])
    )
    midpoint = (len(left_samples) - overlap / 2.0) / rate
    return output, midpoint, overlap


def _pcm16(samples: Sequence[float]) -> array.array:
    return array.array(
        "h",
        (
            max(-32768, min(32767, int(round(
                max(-1.0, min(1.0, float(sample))) * 32767.0
            ))))
            for sample in samples
        ),
    )


def _measured_bridge(
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    rate: int,
    requested_overlap_ms: float,
    expected_f0_hz: Optional[float],
    allow_silent_left: bool = False,
    allow_silent_right: bool = False,
    left_phone: Optional[str] = None,
    right_phone: Optional[str] = None,
    right_indexed_length_samples: Optional[int] = None,
) -> tuple[tuple[float, ...], Mapping[str, object]]:
    """Join fixed source halves without changing either selected recording.

    The source search is local to each selected half.  It may move the cut by
    at most 12 ms, but it never substitutes another contextual candidate.
    Voiced joins use three pitch periods; unvoiced joins use a short fixed
    window.  Festival then performs its normal target-F0 PSOLA mapping.
    """
    left_pcm = _pcm16(left_samples)
    right_pcm = _pcm16(right_samples)
    decision = adaptive_join_pcm16(
        left_pcm,
        right_pcm,
        rate,
        expected_f0_hz=expected_f0_hz,
        allow_silent_left=allow_silent_left,
        allow_silent_right=allow_silent_right,
        left_phone=left_phone,
        right_phone=right_phone,
        enforce_acoustic_similarity=(
            str(left_phone or "") == str(right_phone or "")
        ),
        right_indexed_length_samples=right_indexed_length_samples,
        minimum_right_indexed_tail_samples=int(math.ceil(rate * 0.002)),
        config=JoinSynthesisConfig(
            overlap_periods=3.0,
            minimum_overlap_ms=4.0,
            maximum_overlap_ms=max(
                8.0, min(32.0, float(requested_overlap_ms))
            ),
            unvoiced_overlap_ms=max(
                6.0, min(12.0, float(requested_overlap_ms))
            ),
            search_ms=12.0,
        ),
    )
    return (
        tuple(float(value) / 32768.0 for value in left_pcm),
        decision.to_dict(),
    )


def _render_bounded_bridge(
    wav_root: Path,
    output_name: str,
    left: Optional[_BridgeHalf],
    right: Optional[_BridgeHalf],
    expected_f0_hz: Optional[float] = None,
    allow_silent_left: bool = False,
    allow_silent_right: bool = False,
    left_phone: Optional[str] = None,
    right_phone: Optional[str] = None,
    require_validated_conditioning: bool = False,
) -> tuple[float, float, float, tuple[Mapping[str, object], ...]]:
    if left is None and right is None:
        raise ValueError("a bridge requires at least one recording half")
    if left is not None:
        left_samples, rate = _bridge_clip(wav_root, left)
    else:
        right_probe, rate = _bridge_clip(wav_root, right)  # type: ignore[arg-type]
        left_samples = (0.0,) * int(round(rate * 0.04))
        right_samples = right_probe
    if right is not None and not (left is None):
        right_samples, _ = _bridge_clip(wav_root, right, rate)
    elif right is None:
        right_samples = (0.0,) * int(round(rate * 0.04))
    if len(left_samples) < int(rate * 0.004):
        raise ValueError("left bridge half is shorter than 4 ms")
    if len(right_samples) < int(rate * 0.004):
        raise ValueError("right bridge half is shorter than 4 ms")
    requested_overlap_ms = 8.0
    if right is not None and right.preferred_crossfade_ms is not None:
        requested_overlap_ms = max(
            4.0, min(60.0, float(right.preferred_crossfade_ms))
        )
    legacy_right_samples = right_samples
    if (
        right is not None and right.indexed_end_ms is not None and
        right.indexed_end_ms < right.end_ms
    ):
        legacy_right_samples, _ = _bridge_clip(
            wav_root,
            replace(right, end_ms=right.indexed_end_ms),
            rate,
        )
    legacy_output, legacy_midpoint, legacy_overlap = _legacy_linear_bridge(
        left_samples, legacy_right_samples, rate, requested_overlap_ms)
    legacy_crossfade_start_sample = len(left_samples) - legacy_overlap
    legacy_crossfade_end_sample = len(left_samples)
    legacy_midpoint_sample = len(left_samples) - legacy_overlap / 2.0
    legacy_end_sample = len(legacy_output)
    right_indexed_length_samples = None
    if right is not None and right.indexed_end_ms is not None:
        right_indexed_length_samples = int(math.floor(max(
            0.0, right.indexed_end_ms - right.start_ms
        ) * rate / 1000.0))
    output, conditioning = _measured_bridge(
        left_samples, right_samples, rate, requested_overlap_ms,
        expected_f0_hz,
        allow_silent_left=allow_silent_left,
        allow_silent_right=allow_silent_right,
        left_phone=left_phone,
        right_phone=right_phone,
        right_indexed_length_samples=right_indexed_length_samples)
    hard_failures = {
        str(value) for value in conditioning.get(
            "validation_failures", ())
    } & {
        "MISSING_LEFT_SOURCE_CONTENT",
        "MISSING_RIGHT_SOURCE_CONTENT",
    }
    if hard_failures:
        raise ValueError(
            "generated bridge has no expected source content: " +
            ", ".join(sorted(hard_failures))
        )
    if require_validated_conditioning and (
        not bool(conditioning.get("validation_passed", False)) or
        not bool(conditioning.get("content_preservation_passed", False)) or
        bool(conditioning.get("legacy_fallback_used", False))
    ):
        raise JoinConstraintError(
            "continuity_companion_validation_failed",
            validation_passed=int(bool(
                conditioning.get("validation_passed", False))),
            content_preservation_passed=int(bool(
                conditioning.get("content_preservation_passed", False))),
            legacy_fallback_used=int(bool(
                conditioning.get("legacy_fallback_used", False))),
        )
    midpoint = float(conditioning["splice_sample"]) / rate
    crossfade_start = float(conditioning["handoff_start_sample"]) / rate
    crossfade_end = float(conditioning["handoff_end_sample"]) / rate
    wav_end = len(output) / rate
    end = wav_end
    if right is not None and right.indexed_end_ms is not None:
        used_start_ms = (
            right.start_ms +
            float(conditioning["right_skip_samples"]) * 1000.0 / rate
        )
        end = min(
            wav_end,
            crossfade_start + max(
                0.0, right.indexed_end_ms - used_start_ms
            ) / 1000.0,
        )
    if not 0.002 <= midpoint <= end - 0.002:
        raise ValueError("generated bridge has no valid phone boundary")
    _write_pcm16_mono(wav_root / output_name, output, rate)
    legacy_name = _legacy_bridge_name(output_name)
    _write_pcm16_mono(wav_root / legacy_name, legacy_output, rate)
    components_list = []
    for side, half in (("left", left), ("right", right)):
        if half is None:
            continue
        component = dict(half.source_component())
        if side == "left":
            output_start = 0.0
            output_end = crossfade_end
        else:
            output_start = crossfade_start
            output_end = wav_end
        component["output_slice"] = {
            "start": output_start,
            "end": output_end,
        }
        if side == "right" and wav_end > end + 1e-9:
            component["analysis_guard"] = {
                "indexed_end": end,
                "wav_end": wav_end,
            }
        component["crossfade"] = {
            "side": side,
            "start": crossfade_start,
            "end": crossfade_end,
        }
        if side == "left":
            legacy_output_start = 0.0
            legacy_output_end = legacy_crossfade_end_sample / rate
        else:
            legacy_output_start = legacy_crossfade_start_sample / rate
            legacy_output_end = legacy_end_sample / rate
        component["legacy_output_slice"] = {
            "start": legacy_output_start,
            "end": legacy_output_end,
        }
        component["legacy_crossfade"] = {
            "side": side,
            "start": legacy_crossfade_start_sample / rate,
            "end": legacy_crossfade_end_sample / rate,
        }
        if side == "left":
            used_start_ms = half.start_ms
            used_end_ms = max(
                used_start_ms,
                half.end_ms
                - float(conditioning["left_trim_samples"]) * 1000.0 / rate,
            )
        else:
            used_start_ms = min(
                half.end_ms,
                half.start_ms
                + float(conditioning["right_skip_samples"]) * 1000.0 / rate,
            )
            used_end_ms = half.end_ms
        component["used_source_slice"] = {
            "start": used_start_ms / 1000.0,
            "end": used_end_ms / 1000.0,
        }
        component["join_conditioning"] = {
            **dict(conditioning),
            "requested_overlap_ms": float(requested_overlap_ms),
            "legacy_wav_name": legacy_name,
            "legacy_midpoint_seconds": legacy_midpoint,
            "legacy_overlap_samples": legacy_overlap,
            "legacy_geometry": {
                "sample_rate": rate,
                "start_sample": 0,
                "midpoint_sample": legacy_midpoint_sample,
                "end_sample": legacy_end_sample,
            },
        }
        components_list.append(component)
    components = tuple(components_list)
    return 0.0, midpoint, end, components


def _candidate_region(
    candidate: JapaneseSourceCandidate,
    wav_duration_ms: float,
) -> Optional[tuple[float, float, float, float, float]]:
    timing = candidate.timing
    if not timing.valid or any(
        value is None
        for value in (
            timing.offset,
            timing.consonant,
            timing.cutoff,
            timing.preutterance,
            timing.overlap,
        )
    ):
        return None
    offset = float(timing.offset)
    consonant_end = offset + max(0.0, float(timing.consonant))
    preutterance = offset + float(timing.preutterance)
    overlap = offset + max(0.0, float(timing.overlap))
    cutoff = float(timing.cutoff)
    end = offset + abs(cutoff) if cutoff < 0.0 else wav_duration_ms - cutoff
    end = min(wav_duration_ms, end)
    return offset, overlap, preutterance, consonant_end, end


def _ensure_edge(
    left: str,
    right: str,
    start: float,
    midpoint: float,
    end: float,
    edge_offset: int,
    method: str,
    shared_anchor: Optional[float] = None,
    effective_overlap_ms: float = 0.0,
    overlap_method: str = "oto",
) -> Optional[_EdgeProposal]:
    epsilon = 2.0
    start = max(0.0, float(start))
    end = max(start, float(end))
    midpoint = max(start + epsilon, min(end - epsilon, float(midpoint)))
    if not start + epsilon <= midpoint <= end - epsilon:
        return None
    _safe_token(left)
    _safe_token(right)
    return _EdgeProposal(
        left=left,
        right=right,
        start_ms=start,
        midpoint_ms=midpoint,
        end_ms=end,
        edge_offset=edge_offset,
        method=method,
        shared_anchor_ms=(
            round(float(shared_anchor), 6)
            if shared_anchor is not None else None
        ),
        effective_overlap_ms=round(float(effective_overlap_ms), 6),
        overlap_method=str(overlap_method),
    )


def _center_between(left: float, right: float) -> Optional[float]:
    """Return a phone-center anchor with two milliseconds on either side."""
    if right - left < 4.0:
        return None
    return (float(left) + float(right)) / 2.0


def candidate_edge_proposals(
    candidate: JapaneseSourceCandidate,
    wav_duration_seconds: float,
    *,
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
) -> tuple[_EdgeProposal, ...]:
    """Compile one Phase 2 candidate into zero or more phone-edge units."""
    normalize_zero_overlap_guard_ms(zero_overlap_guard_ms)
    region = _candidate_region(candidate, wav_duration_seconds * 1000.0)
    if region is None:
        return ()
    start, overlap, preutterance, consonant_end, end = region
    # Recorded edge geometry remains authoritative. The zero-overlap guard is
    # applied only when a generated bridge extracts a consonant onset; moving
    # this direct phone-center anchor perturbs the following vowel handoff.
    overlap_shift = effective_oto_overlap_ms(
        candidate.timing.preutterance or 0.0,
        candidate.timing.overlap or 0.0,
        zero_overlap_guard_ms=0.0,
    )
    overlap = start + overlap_shift
    overlap_method = (
        "oto_overlap_end"
        if float(candidate.timing.overlap or 0.0) > 0.0
        and overlap_shift > 0.0
        else "inferred_zero_overlap_guard"
        if overlap_shift > 0.0
        else "oto_offset_fallback"
    )
    target = candidate.target
    phones = tuple(target.phones)
    proposals: list[_EdgeProposal] = []

    def add(
        left: str,
        right: str,
        edge_start: float,
        midpoint: float,
        edge_end: float,
        edge_offset: int,
        method: str,
        shared_anchor: Optional[float] = None,
    ) -> None:
        proposal = _ensure_edge(
            left,
            right,
            edge_start,
            midpoint,
            edge_end,
            edge_offset,
            method,
            shared_anchor,
            overlap_shift,
            overlap_method,
        )
        if proposal is not None:
            proposals.append(proposal)

    if candidate.role == "mora_cv":
        if len(phones) == 1:
            add(
                phones[0], phones[0], start, preutterance, end, 0,
                "oto_preutterance_sustain",
            )
        elif len(phones) >= 2:
            onset = max(start + 2.0, min(preutterance - 2.0, overlap))
            consonant_center = _center_between(onset, preutterance)
            if consonant_center is not None:
                add(
                    phones[-2], phones[-1], consonant_center,
                    preutterance, end, 0, "oto_centered_cv",
                    consonant_center,
                )
    elif candidate.role == "vowel_blend":
        # ``* V`` CV-bank aliases are not phrase starts.  Their OTO offset is
        # the audible target-vowel onset; preutterance marks a stable point
        # inside that vowel and overlap requests the blend span.
        if len(phones) == 1 and phones[0] in _VOWELS:
            add(
                phones[0], phones[0], start, preutterance, end, 0,
                "oto_asterisk_vowel_blend", preutterance,
            )
    elif candidate.role == "phrase_start_cv":
        if len(phones) == 1:
            vowel_center = preutterance
            sustain_boundary = _center_between(vowel_center, end)
            add(
                "pau", phones[0], start, vowel_center,
                sustain_boundary if sustain_boundary is not None else end,
                -1, "oto_preutterance_phrase_start_vowel",
                None,
            )
            # A ``- V`` VCV alias contains the phrase-start transition only.
            # Treating its vowel tail as a second V-V unit lets leading
            # silence leak into medial positions and duplicates the vowel.
            # Sustains come from ordinary CV/VCV vowel aliases instead.
        elif len(phones) >= 2:
            consonant = phones[-2]
            vowel = phones[-1]
            onset = max(start + 2.0, min(preutterance - 2.0, overlap))
            consonant_center = _center_between(onset, preutterance)
            if consonant_center is not None:
                add(
                    "pau", consonant, start, onset, consonant_center, -1,
                    "oto_centered_phrase_start_left", consonant_center,
                )
                add(
                    consonant, vowel, consonant_center, preutterance,
                    end, 0, "oto_centered_phrase_start_right",
                    consonant_center,
                )
    elif candidate.role == "vcv_mora":
        left = target.left_context
        if left and len(phones) == 1:
            add(
                left, phones[0], start, preutterance, end, -1,
                "oto_preutterance_vcv_vowel",
            )
        elif left and len(phones) >= 2:
            consonant = phones[-2]
            vowel = phones[-1]
            onset = max(start + 2.0, min(preutterance - 2.0, overlap))
            consonant_center = _center_between(onset, preutterance)
            if consonant_center is not None:
                add(
                    left, consonant, start, onset, consonant_center, -1,
                    "oto_centered_vcv_left", consonant_center,
                )
                add(
                    consonant, vowel, consonant_center, preutterance,
                    end, 0, "oto_centered_vcv_right",
                    consonant_center,
                )
    elif candidate.role == "vc_transition":
        if target.left_context and target.right_context:
            consonant_extent = max(
                preutterance + 4.0,
                min(end, max(preutterance + 4.0, consonant_end)),
            )
            consonant_extent = min(end, consonant_extent)
            consonant_center = _center_between(
                preutterance, consonant_extent
            )
            if consonant_center is not None:
                add(
                    target.left_context,
                    target.right_context,
                    start,
                    preutterance,
                    consonant_center,
                    -1,
                    "oto_centered_vc",
                    consonant_center,
                )
            # Canonical gemination is V-cl-C, but cl is a language-neutral
            # timing operator rather than a source-bank alias. Runtime source
            # selection realizes that sequence as V-C-C, so no V-cl unit is
            # compiled from the OTO's coincidentally named closure aliases.
    elif candidate.role == "release":
        if target.left_context and target.right_context:
            if target.right_context == "sil":
                add(
                    target.left_context,
                    "pau",
                    start,
                    preutterance,
                    end,
                    -1,
                    "oto_preutterance_vowel_release",
                )
                return tuple(proposals)
            release_end = max(
                preutterance + 4.0, min(end - 2.0, consonant_end)
            )
            consonant_center = _center_between(preutterance, release_end)
            if consonant_center is not None:
                add(
                    target.left_context,
                    target.right_context,
                    start,
                    preutterance,
                    consonant_center,
                    -1,
                    "oto_centered_release_left",
                    consonant_center,
                )
                add(
                    target.right_context,
                    "pau",
                    consonant_center,
                    release_end,
                    end,
                    0,
                    "oto_centered_release_right",
                    consonant_center,
                )
    elif candidate.role == "special_mora" and phones:
        add(
            phones[-1], phones[-1], start, preutterance, end, 0,
            "oto_preutterance_special_mora",
        )
    return tuple(proposals)


def _collect_bridge_half_pools(
    raw_units: Sequence[tuple[
        JapaneseSourceCandidate, _EdgeProposal, str
    ]],
    *,
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
) -> _BridgeHalfPoolSet:
    left_rows: dict[str, list[tuple[tuple[object, ...], _BridgeHalf]]] = {}
    left_continuity_rows: dict[
        str, list[tuple[tuple[object, ...], _BridgeHalf]]
    ] = {}
    right_rows: dict[str, list[tuple[tuple[object, ...], _BridgeHalf]]] = {}
    right_context_rows: dict[
        tuple[str, str], list[tuple[tuple[object, ...], _BridgeHalf]]
    ] = {}
    stable_phones = _VOWELS | {"N"}
    for candidate, proposal, wav_name in raw_units:
        # Phrase-start recordings may contain leading silence.  They remain
        # direct alternatives on pau-* edges but never seed medial bridges.
        if candidate.role == "phrase_start_cv":
            continue
        if (
            candidate.role == "vowel_blend"
            and proposal.right in _VOWELS
            and proposal.end_ms - proposal.start_ms >= 8.0
        ):
            blend_ms = max(
                4.0,
                min(60.0, abs(float(candidate.timing.overlap or 8.0))),
            )
            right_end = min(
                proposal.end_ms,
                proposal.start_ms + max(80.0, blend_ms + 8.0),
            )
            half = _BridgeHalf(
                candidate=candidate,
                wav_name=wav_name,
                start_ms=proposal.start_ms,
                end_ms=right_end,
                purpose="right_vowel_blend",
                preferred_crossfade_ms=blend_ms,
            )
            key = (
                -1,
                candidate.selection_cost,
                -(right_end - proposal.start_ms),
                candidate.candidate_id,
                proposal.method,
            )
            right_rows.setdefault(proposal.right, []).append((key, half))

            # ``* V`` recordings are intended to be neutral vowel glue when
            # the bank has no recorded VV/VC transition.  Reuse the same
            # recording for the stable outgoing side as well as the incoming
            # onset side.  Otherwise adjacent generated bridges can switch
            # from ``* V`` to an arbitrary CV vowel tail at their shared V,
            # producing an avoidable level, phase, and timbre discontinuity.
            # Start at preutterance (or the final 80 ms) so the outgoing half
            # excludes the vowel attack and retains a complete stable tail.
            # This mirrors the generic stable-vowel geometry; the only
            # difference is explicit source provenance for continuity ties.
            stable_end = proposal.end_ms
            stable_start = max(
                proposal.midpoint_ms,
                stable_end - 80.0,
            )
            if stable_end - stable_start >= 8.0:
                stable_half = _BridgeHalf(
                    candidate=candidate,
                    wav_name=wav_name,
                    start_ms=stable_start,
                    end_ms=stable_end,
                    purpose="left_stable_phone",
                )
                stable_key = (
                    -1,
                    candidate.selection_cost,
                    -(stable_end - stable_start),
                    candidate.candidate_id,
                    proposal.method,
                )
                left_continuity_rows.setdefault(proposal.right, []).append(
                    (stable_key, stable_half)
                )
            continue
        if (
            proposal.right in stable_phones
            and proposal.end_ms - proposal.midpoint_ms >= 8.0
        ):
            clip_start = max(
                proposal.midpoint_ms, proposal.end_ms - 80.0
            )
            half = _BridgeHalf(
                candidate=candidate,
                wav_name=wav_name,
                start_ms=clip_start,
                end_ms=proposal.end_ms,
                purpose="left_stable_phone",
            )
            key = (
                0,
                candidate.selection_cost,
                -(proposal.end_ms - clip_start),
                candidate.candidate_id,
                proposal.method,
            )
            left_rows.setdefault(proposal.right, []).append((key, half))

            right_half = _BridgeHalf(
                candidate=candidate,
                wav_name=wav_name,
                start_ms=proposal.midpoint_ms,
                end_ms=min(
                    proposal.end_ms, proposal.midpoint_ms + 80.0
                ),
                purpose="right_phone_onset",
            )
            right_key = (
                0,
                candidate.selection_cost,
                -(right_half.end_ms - right_half.start_ms),
                candidate.candidate_id,
                proposal.method,
            )
            right_rows.setdefault(proposal.right, []).append(
                (right_key, right_half)
            )

        if (
            proposal.right not in _NONLEXICAL_PHONES
            and proposal.end_ms - proposal.midpoint_ms >= 4.0
        ):
            right_end = min(
                proposal.end_ms, proposal.midpoint_ms + 80.0
            )
            half = _BridgeHalf(
                candidate=candidate,
                wav_name=wav_name,
                start_ms=proposal.midpoint_ms,
                end_ms=right_end,
                purpose="right_phone_onset",
            )
            key = (
                0,
                candidate.selection_cost,
                -(right_end - proposal.midpoint_ms),
                candidate.candidate_id,
                proposal.method,
            )
            right_rows.setdefault(proposal.right, []).append((key, half))

        if (
            proposal.left not in stable_phones | _NONLEXICAL_PHONES
            and proposal.right in stable_phones
        ):
            region = _candidate_region(candidate, proposal.end_ms + 1.0)
            if region is None:
                continue
            start, overlap, preutterance, _consonant_end, _end = region
            overlap = start + effective_oto_overlap_ms(
                candidate.timing.preutterance or 0.0,
                candidate.timing.overlap or 0.0,
                zero_overlap_guard_ms=zero_overlap_guard_ms,
            )
            onset = max(start + 2.0, min(preutterance - 2.0, overlap))
            if proposal.midpoint_ms - onset < 4.0:
                continue
            half = _BridgeHalf(
                candidate=candidate,
                wav_name=wav_name,
                start_ms=onset,
                end_ms=proposal.midpoint_ms,
                purpose="right_consonant_onset",
                effective_overlap_ms=max(0.0, overlap - start),
                overlap_method=(
                    "oto_overlap_end"
                    if float(candidate.timing.overlap or 0.0) > 0.0
                    else "inferred_zero_overlap_guard"
                    if overlap > start
                    else "oto_offset_fallback"
                ),
                indexed_end_ms=proposal.start_ms,
            )
            key = (
                0,
                candidate.selection_cost,
                -(proposal.midpoint_ms - onset),
                candidate.candidate_id,
                proposal.method,
            )
            right_rows.setdefault(proposal.left, []).append((key, half))
            # This consonant onset extends through the following CV unit's
            # phone boundary. Retain that following phone so a generated V-C
            # or pau-C bridge and the selected C-V transition share the full
            # recorded consonant interval instead of merely touching at one
            # source sample.
            context = str(proposal.right or "")
            if context and context not in _NONLEXICAL_PHONES:
                right_context_rows.setdefault(
                    (proposal.left, context), []).append((key, half))

    left_best = {
            phone: sorted(rows, key=lambda row: row[0])[0][1]
            for phone, rows in left_rows.items()
        }
    right_best = {
            phone: sorted(rows, key=lambda row: row[0])[0][1]
            for phone, rows in right_rows.items()
        }
    right_by_context: dict[str, dict[str, _BridgeHalf]] = {}
    for (phone, context), rows in sorted(right_context_rows.items()):
        right_by_context.setdefault(phone, {})[context] = sorted(
            rows, key=lambda row: row[0]
        )[0][1]
    return _BridgeHalfPoolSet(
        left_best=left_best,
        left_continuity={
            phone: tuple(
                row[1] for row in sorted(rows, key=lambda row: row[0])
            )
            for phone, rows in sorted(left_continuity_rows.items())
        },
        right_best=right_best,
        right_by_context={
            phone: {
                context: contexts[context]
                for context in sorted(contexts)
            }
            for phone, contexts in sorted(right_by_context.items())
        },
    )


def _bridge_half_pools(
    raw_units: Sequence[tuple[
        JapaneseSourceCandidate, _EdgeProposal, str
    ]],
    *,
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
) -> tuple[dict[str, _BridgeHalf], dict[str, _BridgeHalf]]:
    """Compatibility view used by existing analysis/tests."""
    pools = _collect_bridge_half_pools(
        raw_units, zero_overlap_guard_ms=zero_overlap_guard_ms
    )
    return dict(pools.left_best), dict(pools.right_best)


def _generated_bridge_id(
    configuration_id: str,
    diphone: str,
    left: Optional[_BridgeHalf],
    right: Optional[_BridgeHalf],
    recorded_right_context: str,
    expected_f0_hz: Optional[float],
) -> tuple[str, str]:
    value = {
        "conditioning_version": JOIN_SYNTHESIS_CONDITIONING_VERSION,
        "configuration_id": configuration_id,
        "diphone": diphone,
        "recorded_right_context": recorded_right_context,
        "expected_f0_hz": (
            round(float(expected_f0_hz), 6)
            if expected_f0_hz is not None else None),
        "left": (
            [left.candidate.candidate_id, left.start_ms, left.end_ms]
            if left is not None else ["synthetic_pause"]
        ),
        "right": (
            [right.candidate.candidate_id, right.start_ms, right.end_ms]
            if right is not None else ["synthetic_pause"]
        ),
    }
    digest = hashlib.sha256(_json_bytes(value)).hexdigest()
    return f"jfb_{digest[:24]}", f"_jfb_{digest[:16]}.wav"


def _add_generated_bridge(
    *,
    output_root: Path,
    configuration_id: str,
    left_phone: str,
    right_phone: str,
    left: Optional[_BridgeHalf],
    right: Optional[_BridgeHalf],
    recorded_right_context: str,
    units: list[JapaneseCompiledUnit],
    index: dict[str, tuple[str, float, float, float]],
    alternatives: dict[str, tuple[Mapping[str, object], ...]],
    expected_f0_hz: Optional[float],
    continuity_group_id: str = "",
    continuity_companion: bool = False,
    structural_hold: bool = False,
    failures: Optional[list[dict[str, object]]] = None,
) -> Optional[str]:
    diphone = f"{left_phone}-{right_phone}"
    candidate_id, wav_name = _generated_bridge_id(
        configuration_id, diphone, left, right, recorded_right_context,
        expected_f0_hz,
    )
    try:
        start, midpoint, end, components = _render_bounded_bridge(
            output_root / "wav", wav_name, left, right,
            expected_f0_hz=expected_f0_hz,
            allow_silent_left=(
                left_phone in _SILENCE_PHONES or
                structural_hold and left_phone in
                _STOP_OR_AFFRICATE_PHONES
            ),
            allow_silent_right=(
                right_phone in _SILENCE_PHONES or
                right_phone in _STOP_OR_AFFRICATE_PHONES),
            left_phone=left_phone,
            right_phone=right_phone,
            require_validated_conditioning=continuity_companion,
        )
    except (OSError, ValueError, wave.Error) as error:
        if failures is not None:
            message = str(error)
            if isinstance(error, JoinConstraintError):
                reason = error.code
                stage = (
                    "join_validation"
                    if error.code == "continuity_companion_validation_failed"
                    else "join_search"
                )
                details: Mapping[str, object] = error.details
            elif "shorter than 4 ms" in message:
                reason = "source_half_too_short"
                stage = "source_geometry"
                details = {}
            elif "no valid phone boundary" in message:
                reason = "indexed_phone_tail_too_short"
                stage = "index_geometry"
                details = {}
            elif "no expected source content" in message:
                reason = "source_content_missing"
                stage = "join_validation"
                details = {}
            elif isinstance(error, wave.Error):
                reason = "source_wave_invalid"
                stage = "source_read"
                details = {}
            elif isinstance(error, OSError):
                reason = "source_read_failed"
                stage = "source_read"
                details = {}
            else:
                reason = "bridge_render_failed"
                stage = "bridge_render"
                details = {}
            failures.append({
                "bridge_candidate_id": candidate_id,
                "diphone": diphone,
                "recorded_right_context": recorded_right_context,
                "left_source_candidate_id": (
                    left.candidate.candidate_id if left is not None else ""
                ),
                "right_source_candidate_id": (
                    right.candidate.candidate_id if right is not None else ""
                ),
                "code": reason,
                "stage": stage,
                "details": dict(details),
            })
        return None
    number = len(alternatives.get(diphone, ()))
    left_name = left_phone if number == 0 else (
        f"{left_phone}__j{candidate_id[4:14]}e1"
    )
    _safe_token(left_name, "unit name")
    index_name = f"{left_name}-{right_phone}"
    unit = JapaneseCompiledUnit(
        candidate_id=candidate_id,
        edge_index=number,
        edge_offset=-1,
        diphone=diphone,
        left_phone=left_phone,
        right_phone=right_phone,
        left_name=left_name,
        index_name=index_name,
        wav_name=wav_name,
        start=round(start, 6),
        midpoint=round(midpoint, 6),
        end=round(end, 6),
        role=(
            "structural_consonant_hold"
            if structural_hold else "generated_cv_bridge"
        ),
        family="cv",
        selection_cost=0.0,
        geometry_method=(
            "generated_bounded_consonant_hold"
            if structural_hold else "generated_bounded_cv_bridge"
        ),
        source_path=f"generated/{wav_name}",
        source_alias=(
            f"[{left_phone} structural hold]"
            if structural_hold else
            f"[{left_phone} {right_phone} CV fallback]"
        ),
        source_oto_path="",
        source_oto_line=0,
        shared_anchor=None,
        oto_offset_ms=0.0,
        oto_consonant_ms=0.0,
        oto_cutoff_ms=0.0,
        oto_preutterance_ms=round(midpoint * 1000.0, 6),
        oto_overlap_ms=round(max(0.0, midpoint * 1000.0 - 8.0), 6),
        effective_overlap_ms=8.0,
        overlap_method="generated_bridge_crossfade",
        recorded_left_context=left_phone,
        recorded_right_context=recorded_right_context,
        moraic_nasal_allophone=str(
            (
                right.candidate.target.moraic_nasal_allophone
                if right_phone == "N" and right is not None else
                left.candidate.target.moraic_nasal_allophone
                if left_phone == "N" and left is not None else ""
            ) or ""
        ),
        source_components=components,
    )
    units.append(unit)
    index[index_name] = (wav_name, unit.start, unit.midpoint, unit.end)
    choice = _choice_payload(unit)
    choice["continuity_group_id"] = continuity_group_id
    choice["fallback_reason"] = (
        "No authored consonant hold was available; generated a bounded "
        "C-C unit from the consonant portion of a normal C-V source."
        if structural_hold else
        "No matching recorded VC/VV transition was available; generated a "
        "bounded bridge from a stable left phone and the next CV onset."
    )
    alternatives[diphone] = tuple(alternatives.get(diphone, ())) + (choice,)
    return wav_name


def _compile_generated_bridges(
    *,
    output_root: Path,
    configuration_id: str,
    raw_units: Sequence[tuple[
        JapaneseSourceCandidate, _EdgeProposal, str
    ]],
    phones: set[str],
    units: list[JapaneseCompiledUnit],
    index: dict[str, tuple[str, float, float, float]],
    alternatives: dict[str, tuple[Mapping[str, object], ...]],
    expected_f0_hz: Optional[float],
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
) -> tuple[set[str], tuple[Mapping[str, object], ...]]:
    # A bridge needs only a short source slice. Keep those slices cached for
    # this build because the same consonant onset is reused across left-phone
    # contexts, then release memory before returning to a long-lived GUI.
    _read_pcm_mono_slice.cache_clear()
    pools = _collect_bridge_half_pools(
        raw_units,
        zero_overlap_guard_ms=zero_overlap_guard_ms,
    )
    left_pool = pools.left_best
    right_pool = pools.right_best
    generated: set[str] = set()
    failures: list[dict[str, object]] = []
    stable = sorted((set(_VOWELS) | {"N"}) & phones)
    spoken = sorted(phones - _NONLEXICAL_PHONES)

    def add(
        left_phone: str,
        right_phone: str,
        left_half: Optional[_BridgeHalf],
        right_half: Optional[_BridgeHalf],
        recorded_right_context: str = "*",
        continuity_group_id: str = "",
        continuity_companion: bool = False,
        structural_hold: bool = False,
    ) -> Optional[str]:
        if left_half is None and left_phone != "pau":
            bridge_candidate_id, _ = _generated_bridge_id(
                configuration_id,
                f"{left_phone}-{right_phone}",
                left_half,
                right_half,
                recorded_right_context,
                expected_f0_hz,
            )
            failures.append({
                "bridge_candidate_id": bridge_candidate_id,
                "diphone": f"{left_phone}-{right_phone}",
                "recorded_right_context": recorded_right_context,
                "left_source_candidate_id": "",
                "right_source_candidate_id": (
                    right_half.candidate.candidate_id
                    if right_half is not None else ""
                ),
                "code": "left_source_half_unavailable",
                "stage": "source_selection",
                "details": {},
            })
            return None
        if right_half is None and right_phone != "pau":
            bridge_candidate_id, _ = _generated_bridge_id(
                configuration_id,
                f"{left_phone}-{right_phone}",
                left_half,
                right_half,
                recorded_right_context,
                expected_f0_hz,
            )
            failures.append({
                "bridge_candidate_id": bridge_candidate_id,
                "diphone": f"{left_phone}-{right_phone}",
                "recorded_right_context": recorded_right_context,
                "left_source_candidate_id": (
                    left_half.candidate.candidate_id
                    if left_half is not None else ""
                ),
                "right_source_candidate_id": "",
                "code": "right_source_half_unavailable",
                "stage": "source_selection",
                "details": {},
            })
            return None
        wav_name = _add_generated_bridge(
            output_root=output_root,
            configuration_id=configuration_id,
            left_phone=left_phone,
            right_phone=right_phone,
            left=left_half,
            right=right_half,
            recorded_right_context=recorded_right_context,
            units=units,
            index=index,
            alternatives=alternatives,
            expected_f0_hz=expected_f0_hz,
            continuity_group_id=continuity_group_id,
            continuity_companion=continuity_companion,
            structural_hold=structural_hold,
            failures=failures,
        )
        if wav_name is not None:
            generated.add(wav_name)
        return wav_name

    def add_with_continuity_companions(
        left_phone: str,
        right_phone: str,
        left_half: Optional[_BridgeHalf],
        right_half: Optional[_BridgeHalf],
        recorded_right_context: str = "*",
    ) -> None:
        """Keep the historical base first, then add source-matched ties."""
        continuity_group_id, _ = _generated_bridge_id(
            configuration_id,
            f"{left_phone}-{right_phone}",
            left_half,
            right_half,
            recorded_right_context,
            expected_f0_hz,
        )
        base_wav = add(
            left_phone, right_phone, left_half, right_half,
            recorded_right_context,
            continuity_group_id,
        )
        if base_wav is None or left_half is None:
            return
        for companion in pools.left_continuity.get(left_phone, ()):
            if (
                companion.candidate.candidate_id
                == left_half.candidate.candidate_id
                and companion.start_ms == left_half.start_ms
                and companion.end_ms == left_half.end_ms
            ):
                continue
            add(
                left_phone, right_phone, companion, right_half,
                recorded_right_context,
                continuity_group_id,
                True,
            )

    for left_phone in stable:
        for right_phone in spoken:
            diphone = f"{left_phone}-{right_phone}"
            if diphone not in alternatives:
                add_with_continuity_companions(
                    left_phone, right_phone,
                    left_pool.get(left_phone), right_pool.get(right_phone),
                )
                for context, contextual_half in sorted(
                    pools.right_by_context.get(right_phone, {}).items()
                ):
                    add_with_continuity_companions(
                        left_phone, right_phone,
                        left_pool.get(left_phone), contextual_half, context,
                    )
        if f"{left_phone}-pau" not in alternatives:
            add_with_continuity_companions(
                left_phone, "pau", left_pool.get(left_phone), None
            )

    for right_phone in spoken:
        if f"pau-{right_phone}" not in alternatives:
            add("pau", right_phone, None, right_pool.get(right_phone))
            for context, contextual_half in sorted(
                pools.right_by_context.get(right_phone, {}).items()
            ):
                add(
                    "pau", right_phone, None, contextual_half, context,
                )

    consonants = sorted(set(spoken) - set(_VOWELS) - {"N"})
    # Structural cl is rendered as the following consonant twice, so every
    # consonant needs a bounded C-C hold that avoids replaying the full CV.
    for consonant in consonants:
        diphone = f"{consonant}-{consonant}"
        if diphone in alternatives:
            continue
        source_halves = [
            ("*", right_pool.get(consonant)),
            *sorted(
                pools.right_by_context.get(consonant, {}).items()
            ),
        ]
        seen_halves = set()
        for context, half in source_halves:
            if half is None:
                continue
            signature = (
                half.candidate.candidate_id,
                round(half.start_ms, 6),
                round(half.end_ms, 6),
            )
            if signature in seen_halves:
                continue
            seen_halves.add(signature)
            add(
                consonant,
                consonant,
                half,
                half,
                str(context),
                structural_hold=True,
            )
    _read_pcm_mono_slice.cache_clear()
    finalized_failures = []
    for row in failures:
        finalized = dict(row)
        finalized["failure_id"] = (
            "jbf_" + hashlib.sha256(_json_bytes(finalized)).hexdigest()[:20]
        )
        finalized_failures.append(finalized)
    return generated, tuple(sorted(
        finalized_failures,
        key=lambda row: (
            str(row["diphone"]),
            str(row["recorded_right_context"]),
            str(row["left_source_candidate_id"]),
            str(row["right_source_candidate_id"]),
            str(row["code"]),
            str(row["failure_id"]),
        ),
    ))


def _write_silence(path: Path, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * 0.30))


def _phone_definition(phone: str) -> str:
    if phone in {"pau", "sil"}:
        return f"   ({phone} - 0 - - - 0 0 -)"
    if phone in _VOWELS:
        return f"   ({phone} + l 2 1 - 0 0 +)"
    if phone in _VOICELESS_STOPS | _VOICED_STOPS:
        ctype = "s"
    elif phone in _VOICELESS_AFFRICATES | _VOICED_AFFRICATES:
        ctype = "a"
    elif phone in _VOICELESS_FRICATIVES | _VOICED_FRICATIVES:
        ctype = "f"
    elif phone in _NASALS:
        ctype = "n"
    elif phone in _LIQUIDS:
        ctype = "l"
    elif phone in _GLIDES:
        ctype = "r"
    else:
        return f"   ({phone} - 0 - - - 0 0 0)"
    voiced = "+" if phone in (
        _VOICED_STOPS | _VOICED_AFFRICATES | _VOICED_FRICATIVES |
        _NASALS | _LIQUIDS | _GLIDES
    ) else "-"
    return f"   ({phone} - 0 - - - {ctype} a {voiced})"


def _phone_duration(phone: str) -> float:
    if phone in _DEFAULT_DURATIONS:
        return _DEFAULT_DURATIONS[phone]
    if phone in {"k", "g", "t", "d", "p", "b"}:
        return 0.055
    if phone in {"s", "sh", "z", "j", "f", "h", "ch", "ts"}:
        return 0.085
    if phone in {"m", "n", "r", "w", "y"} or phone.endswith("y"):
        return 0.075
    return 0.08


def _moraic_nasal_routing(profile) -> dict[str, object]:
    following: dict[str, str] = {}
    default = ""
    for allophone_id, rule in sorted(
        profile.moraic_nasal_allophones.items()
    ):
        for phone in rule.following_phones:
            following[str(phone)] = str(allophone_id)
        if rule.default:
            default = str(allophone_id)
    return {
        "following_phones": {
            key: following[key] for key in sorted(following)
        },
        "default": default,
    }


def _est_floor_seconds(value: float) -> float:
    """Quantize without allowing EST's six decimals to pass a sample bound."""
    return math.floor(max(0.0, float(value)) * 1_000_000.0 + 1e-9) / 1_000_000.0


def _bounded_est_geometry(
    wav_path: Path,
    start: float,
    midpoint: float,
    end: float,
) -> tuple[float, float, float]:
    """Validate and quantize one EST row against its actual WAV frames."""
    with wave.open(str(wav_path), "rb") as handle:
        frame_count = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
    if sample_rate <= 0:
        raise wave.Error(f"sample rate is zero: {wav_path.name}")
    duration = frame_count / float(sample_rate)
    tolerance = 1.0 / sample_rate
    values = (float(start), float(midpoint), float(end))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite EST geometry for {wav_path.name}")
    if (
        start < -tolerance
        or end > duration + tolerance
        or midpoint < start - tolerance
        or midpoint > end + tolerance
    ):
        raise ValueError(
            f"EST geometry exceeds {wav_path.name}: "
            f"{start:.9f}/{midpoint:.9f}/{end:.9f} vs {duration:.9f}s"
        )
    bounded_start = max(0.0, min(float(start), duration))
    bounded_end = max(bounded_start, min(float(end), duration))
    bounded_midpoint = max(
        bounded_start, min(float(midpoint), bounded_end)
    )
    quantized_start = _est_floor_seconds(bounded_start)
    quantized_midpoint = max(
        quantized_start, _est_floor_seconds(bounded_midpoint)
    )
    quantized_end = max(
        quantized_midpoint, _est_floor_seconds(bounded_end)
    )
    if quantized_end * sample_rate > frame_count + 1e-7:
        raise ValueError(f"EST endpoint exceeds {wav_path.name} samples")
    return quantized_start, quantized_midpoint, quantized_end


def _legacy_bridge_geometry(
    unit: JapaneseCompiledUnit,
    legacy_wav_path: Path,
) -> tuple[float, float, float]:
    """Recover the paired linear bridge's sample-domain EST geometry."""
    geometry: Optional[Mapping[str, object]] = None
    for component in unit.source_components:
        conditioning = component.get("join_conditioning")
        if not isinstance(conditioning, Mapping):
            continue
        candidate = conditioning.get("legacy_geometry")
        if isinstance(candidate, Mapping):
            geometry = candidate
            break
    if geometry is None:
        raise ValueError(
            f"generated bridge {unit.wav_name} has no Legacy geometry"
        )
    try:
        metadata_rate = int(geometry["sample_rate"])
        start_sample = float(geometry["start_sample"])
        midpoint_sample = float(geometry["midpoint_sample"])
        end_sample = float(geometry["end_sample"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"generated bridge {unit.wav_name} has invalid Legacy geometry"
        ) from error
    with wave.open(str(legacy_wav_path), "rb") as handle:
        actual_rate = int(handle.getframerate())
        actual_frames = int(handle.getnframes())
    if metadata_rate != actual_rate or abs(end_sample - actual_frames) > 1e-7:
        raise ValueError(
            f"Legacy geometry does not match {legacy_wav_path.name}"
        )
    return _bounded_est_geometry(
        legacy_wav_path,
        start_sample / actual_rate,
        midpoint_sample / actual_rate,
        end_sample / actual_rate,
    )


def _write_est(
    output: Path,
    voice_name: str,
    index: Mapping[str, tuple[str, float, float, float]],
    *,
    legacy: bool = False,
) -> Path:
    suffix = "_legacy" if legacy else ""
    path = output / "dic" / f"{voice_name}_ja_diphone{suffix}.est"
    lines = [
        "EST_File index",
        "DataType ascii",
        f"NumEntries {len(index)}",
        f"IndexName {voice_name}_ja_diphone{suffix}",
        "EST_Header_End",
    ]
    for key in sorted(index):
        wav_name, start, midpoint, end = index[key]
        lines.append(
            f"{key} {Path(wav_name).stem} "
            f"{start:.6f} {midpoint:.6f} {end:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_scheme(
    output: Path,
    voice_name: str,
    phones: Sequence[str],
    alternatives: Mapping[str, Sequence[Mapping[str, object]]],
    average_pitch_hz: float,
    moraic_nasal_routing: Mapping[str, object],
    runtime_audio_storage: str,
) -> Path:
    if runtime_audio_storage not in RUNTIME_AUDIO_STORAGE_MODES:
        raise ValueError(
            "runtime_audio_storage must be grouped or separate")
    prefer_grouped = "t" if runtime_audio_storage == "grouped" else "nil"
    symbol = f"{voice_name}_ja"
    definitions = "\n".join(_phone_definition(phone) for phone in phones)
    durations = "\n".join(
        f"    ({phone} {_phone_duration(phone):.3f})" for phone in phones
    )
    variant_rows = []
    for diphone in sorted(alternatives):
        encoded_choices = []
        for choice in alternatives[diphone]:
            left_name = str(choice["left_name"])
            left_activation = choice.get("window_left_activation")
            right_activation = choice.get("window_right_activation")
            encoded_choices.append(
                "(%s %s %s %s %s %s %.6f %s %s %s %s %.6f %.6f %s %s %s %.6f %s)" % (
                _scheme_string(str(choice["left_name"])),
                _scheme_string(str(choice["candidate_id"])),
                _scheme_string(str(choice["role"])),
                _scheme_string(str(choice["recorded_left_context"])),
                _scheme_string(str(choice["recorded_right_context"])),
                _scheme_string(str(choice["edge_offset"])),
                float(choice["selection_cost"]),
                _scheme_string(str(
                    choice.get("moraic_nasal_allophone") or ""
                )),
                _scheme_string(str(
                    choice.get("window_left_name") or left_name
                )),
                _scheme_string(str(
                    choice.get("window_right_name") or left_name
                )),
                _scheme_string(str(
                    choice.get("window_both_name") or left_name
                )),
                (float(left_activation) if left_activation is not None
                 else 1000000.0),
                (float(right_activation) if right_activation is not None
                 else 1000000.0),
                _scheme_string(str(
                    choice.get("left_source_candidate_id") or ""
                )),
                _scheme_string(str(
                    choice.get("right_source_candidate_id") or ""
                )),
                _scheme_string(str(
                    choice.get("left_source_role") or ""
                )),
                float(choice.get("vowel_blend_activation_seconds")
                      or 1000000.0),
                _scheme_string(str(
                    choice.get("continuity_group_id") or ""
                )),
            ))
        choices = " ".join(encoded_choices)
        variant_rows.append(
            f"  ({_scheme_string(diphone)} ({choices}))"
        )
    variants = "\n".join(variant_rows)
    nasal_routes = " ".join(
        "(%s %s)" % (_scheme_string(str(phone)), _scheme_string(str(name)))
        for phone, name in sorted(
            dict(moraic_nasal_routing.get("following_phones") or {}).items()
        )
    )
    nasal_default = _scheme_string(str(
        moraic_nasal_routing.get("default") or ""
    ))
    silence = " ".join(
        phone for phone in ("pau", "sil") if phone in phones
    )
    default_diphone = "pau-pau"
    scheme = f""";; {voice_name}_ja.scm -- Japanese-only UniSyn voice.
;; Generated by japanese_festival.py (Phase 3).  It loads no English frontend
;; resources and exposes only the explicit Japanese voice entry point.

;; The backend prepends the generated voice root to load-path before loading
;; this file.  Resolving from that entry keeps otherwise identical builds
;; byte-for-byte deterministic across destination directories.
(defvar {symbol}_dir (car load-path))

(defPhoneSet {symbol}
  ((vc + -) (vlng s l d a 0) (vheight 1 2 3 0 -) (vfront 1 2 3 0 -)
   (vrnd + - 0) (ctype s f a n l r 0) (cplace l a p b d v g 0)
   (cvox + - 0))
  (
{definitions}
  ))
(PhoneSet.silences '({silence}))

;; Choice row fields are left-unit, candidate-id, role, recorded contexts,
;; edge/cost/nasal data, source-window variants and activation thresholds,
;; left/right source candidate IDs, left source role, duration-gated neutral
;; vowel blending, and the explicit generated-bridge companion group.
(set! {symbol}_unit_variants '(
{variants}
  ))
(set! {symbol}_moraic_nasal_routes '({nasal_routes}))
(set! {symbol}_moraic_nasal_default {nasal_default})
(defvar festvox_gui_unit_variant_overrides nil)

(define ({symbol}_variant_by_left choices wanted)
  (cond ((null choices) nil)
        ((string-equal (car (car choices)) wanted) (car choices))
        (t ({symbol}_variant_by_left (cdr choices) wanted))))

(define ({symbol}_choice_role choice)
  (car (cdr (cdr choice))))
(define ({symbol}_choice_left_context choice)
  (car (cdr (cdr (cdr choice)))))
(define ({symbol}_choice_right_context choice)
  (car (cdr (cdr (cdr (cdr choice))))))
(define ({symbol}_choice_edge_offset choice)
  (car (cdr (cdr (cdr (cdr (cdr choice)))))))
(define ({symbol}_choice_cost choice)
  (car (cdr (cdr (cdr (cdr (cdr (cdr choice))))))))
(define ({symbol}_choice_nasal_allophone choice)
  (car (cdr (cdr (cdr (cdr (cdr (cdr (cdr choice)))))))))

(define ({symbol}_choice_field choice index)
  (if (> index 0)
      ({symbol}_choice_field (cdr choice) (- index 1))
      (car choice)))

(define ({symbol}_choice_left_source choice)
  ({symbol}_choice_field choice 13))
(define ({symbol}_choice_right_source choice)
  ({symbol}_choice_field choice 14))
(define ({symbol}_choice_left_source_role choice)
  ({symbol}_choice_field choice 15))
(define ({symbol}_choice_blend_activation choice)
  ({symbol}_choice_field choice 16))
(define ({symbol}_choice_continuity_group choice)
  ({symbol}_choice_field choice 17))

(define ({symbol}_phrase_boundary_phone phone)
  (or (string-equal phone "pau")
      (string-equal phone "sil")
      (string-equal phone "sp")
      (string-equal phone "*")))

(define ({symbol}_choice_continuity
         choice prior_right_source outer_left seg enabled)
  (if (not enabled)
      0
      (cond
       ((and (not (string-equal prior_right_source ""))
             (not (string-equal prior_right_source "0"))
             (string-equal ({symbol}_choice_left_source choice)
                           prior_right_source))
        2)
       ((and (not ({symbol}_phrase_boundary_phone outer_left))
             (string-equal ({symbol}_choice_left_source_role choice)
                           "vowel_blend")
             (not (< ({symbol}_segment_duration seg)
                     ({symbol}_choice_blend_activation choice))))
        1)
       (t 0))))

(define ({symbol}_segment_duration seg)
  (let ((previous (item.prev seg)))
    (if previous
        (- (item.feat seg "end") (item.feat previous "end"))
        (item.feat seg "end"))))

(define ({symbol}_source_window_name choice seg next)
  (let ((left-long
         (not (< ({symbol}_segment_duration seg)
                 ({symbol}_choice_field choice 11)))))
    (let ((right-long
           (not (< ({symbol}_segment_duration next)
                   ({symbol}_choice_field choice 12)))))
      (cond ((and left-long right-long)
             ({symbol}_choice_field choice 10))
            (left-long ({symbol}_choice_field choice 8))
            (right-long ({symbol}_choice_field choice 9))
            (t (car choice))))))

(define ({symbol}_desired_nasal_allophone
         left_phone right_phone outer_right)
  (let ((following
         (cond ((string-equal right_phone "N") outer_right)
               ((string-equal left_phone "N") right_phone)
               (t ""))))
    (if (string-equal following "")
        ""
        (let ((row (assoc_string following
                                 {symbol}_moraic_nasal_routes)))
          (if row (cadr row) {symbol}_moraic_nasal_default)))))

(define ({symbol}_nasal_allophone_bonus choice desired)
  (let ((source ({symbol}_choice_nasal_allophone choice)))
    (cond ((string-equal desired "") 0)
          ((string-equal source desired) 500)
          ((string-equal source "") -150)
          (t -500))))

(define ({symbol}_choice_eligible choice outer_left)
  (let ((role ({symbol}_choice_role choice)))
    (if (and (string-equal role "phrase_start_cv")
             (not (string-equal outer_left "pau")))
        nil
        t)))

(define ({symbol}_context_bonus choice outer_left outer_right)
  (let ((role ({symbol}_choice_role choice)))
    (let ((expected_left ({symbol}_choice_left_context choice)))
      (let ((expected_right ({symbol}_choice_right_context choice)))
        (let ((edge ({symbol}_choice_edge_offset choice)))
          (cond
           ((string-equal role "phrase_start_cv")
            (if (not (string-equal outer_left "pau"))
                -120
                (if (string-equal edge "-1")
                    (if (or (string-equal expected_right "*")
                            (string-equal expected_right outer_right))
                        120 -120)
                    110)))
           ((string-equal role "vcv_mora")
            (if (string-equal edge "-1")
                (if (string-equal expected_right "") 210
                    (if (string-equal expected_right "*") 210
                        (if (string-equal expected_right outer_right) 115 10)))
                (if (string-equal expected_left outer_left) 115 10)))
           ((string-equal role "vc_transition")
            (if (string-equal expected_right "*")
                110
                (if (string-equal expected_right outer_right) 120 -80)))
           ((string-equal role "release")
            (if (string-equal edge "0")
                (if (string-equal expected_left outer_left) 100 20)
                55))
           ((string-equal role "generated_cv_bridge")
            (if (string-equal expected_right "*")
                40
                (if (string-equal expected_right outer_right) 50 -80)))
           ((string-equal role "structural_consonant_hold")
            (if (string-equal expected_right "*")
                75
                (if (string-equal expected_right outer_right) 120 -60)))
           ((string-equal role "vowel_blend") 85)
           ((string-equal role "mora_cv") 60)
           ((string-equal role "special_mora") 60)
           (t 0)))))))

(define ({symbol}_choice_score choice outer_left outer_right desired)
  (+ ({symbol}_context_bonus choice outer_left outer_right)
     ({symbol}_nasal_allophone_bonus choice desired)
     (- 20 (* 5 ({symbol}_choice_cost choice)))))

(define ({symbol}_best_choice
         choices outer_left outer_right desired prior_right_source
         seg continuity_enabled best best_score best_continuity)
  (if (null choices)
      best
      (if (not ({symbol}_choice_eligible (car choices) outer_left))
          ({symbol}_best_choice (cdr choices) outer_left outer_right
                                desired prior_right_source
                                seg continuity_enabled
                                best best_score best_continuity)
          (let ((score ({symbol}_choice_score
                        (car choices) outer_left outer_right desired)))
            (let ((continuity
                   ({symbol}_choice_continuity
                    (car choices) prior_right_source outer_left seg
                    continuity_enabled)))
              (if (or (null best)
                      (> score best_score)
                      (and (not (or (> score best_score)
                                    (< score best_score)))
                           (not (string-equal
                                 ({symbol}_choice_continuity_group
                                  (car choices))
                                 ""))
                           (string-equal
                            ({symbol}_choice_continuity_group
                             (car choices))
                            ({symbol}_choice_continuity_group best))
                           (> continuity best_continuity)))
                  ({symbol}_best_choice
                   (cdr choices) outer_left outer_right desired
                   prior_right_source seg continuity_enabled
                   (car choices) score continuity)
                  ({symbol}_best_choice
                   (cdr choices) outer_left outer_right desired
                   prior_right_source seg continuity_enabled
                   best best_score
                   best_continuity)))))))

(define ({symbol}_select_one seg index)
  (item.set_feat seg "selected_right_source_candidate_id" "")
  (let ((next (item.next seg)))
    (if next
        (let ((key (string-append (item.name seg) "-" (item.name next))))
          (let ((row (assoc_string key {symbol}_unit_variants)))
            (let ((choices (if row (cadr row) nil)))
              (let ((previous (item.prev seg)))
                (let ((following (item.next next)))
                  (let ((outer-left
                         (if previous (item.name previous) "*")))
                    (let ((outer-right
                           (if following (item.name following) "*")))
                      (let ((desired
                             ({symbol}_desired_nasal_allophone
                              (item.name seg) (item.name next)
                              outer-right)))
                        (let ((prior-right-source
                               (if previous
                                   (item.feat
                                    previous
                                    "selected_right_source_candidate_id")
                                   "")))
                          (let ((override-row
                                 (assoc_string
                                  (format nil "%d" index)
                                  festvox_gui_unit_variant_overrides)))
                            (let ((wanted
                                   (if override-row (cadr override-row)
                                       (item.feat seg
                                                  "unit_variant_override"))))
                              (let ((chosen
                                     (if (and wanted
                                              (not (string-equal wanted "0")))
                                         ({symbol}_variant_by_left
                                          choices wanted)
                                         ({symbol}_best_choice
                                          choices outer-left outer-right
                                          desired prior-right-source
                                          seg
                                          t
                                          nil -100000 -1))))
                                (if chosen
                                    (begin
                                      (item.set_feat
                                       seg
                                       "selected_right_source_candidate_id"
                                       ({symbol}_choice_right_source chosen))
                                      (item.set_feat seg "us_diphone_left"
                                        ({symbol}_source_window_name
                                         chosen seg next)))))))))))))))))))

(define ({symbol}_select_list segments index)
  (if (null segments)
      nil
      (begin
        ({symbol}_select_one (car segments) index)
        ({symbol}_select_list (cdr segments) (+ index 1)))))

(define ({symbol}_select_units utt)
  ({symbol}_select_list (utt.relation.items utt 'Segment) 0)
  (set! festvox_gui_unit_variant_overrides nil)
  utt)

(set! {symbol}_phone_durs '(
{durations}
  ))

(defvar festvox_gui_force_separate_database nil)
(defvar festvox_gui_legacy_joins nil)
(defvar {symbol}_prefer_grouped_database {prefer_grouped})
(defvar {symbol}_group_file
  (path-append {symbol}_dir "group/{voice_name}_diphone.group"))
(set! {symbol}_separate_db_params
  (list
   (list 'name '{symbol}_diphone_separate)
   (list 'index_file
         (path-append {symbol}_dir "dic/{voice_name}_ja_diphone.est"))
   '(grouped "false")
   (list 'coef_dir (path-append {symbol}_dir "pm"))
   (list 'sig_dir (path-append {symbol}_dir "wav"))
   '(coef_ext ".pm")
   '(sig_ext ".wav")
   (list 'default_diphone "{default_diphone}")))
(set! {symbol}_legacy_db_params
  (list
   (list 'name '{symbol}_diphone_legacy)
   (list 'index_file
         (path-append {symbol}_dir
                      "dic/{voice_name}_ja_diphone_legacy.est"))
   '(grouped "false")
   (list 'coef_dir (path-append {symbol}_dir "pm"))
   (list 'sig_dir (path-append {symbol}_dir "wav"))
   '(coef_ext ".legacy.pm")
   '(sig_ext ".wav")
   (list 'default_diphone "{default_diphone}")))
(set! {symbol}_grouped_db_params
  (list
   (list 'name '{symbol}_diphone_grouped)
   (list 'index_file {symbol}_group_file)
   '(grouped "true")
   (list 'default_diphone "{default_diphone}")))

(define ({symbol}_runtime_db_params)
  (if (and {symbol}_prefer_grouped_database
           (not festvox_gui_force_separate_database)
           (probe_file {symbol}_group_file))
      {symbol}_grouped_db_params
      {symbol}_separate_db_params))
(set! {symbol}_db_name nil)
(set! {symbol}_legacy_db_name nil)

(define ({symbol}_active_db_name)
  (if festvox_gui_legacy_joins
      (begin
        (if (null {symbol}_legacy_db_name)
            (set! {symbol}_legacy_db_name
                  (us_diphone_init {symbol}_legacy_db_params)))
        {symbol}_legacy_db_name)
      (begin
        (if (null {symbol}_db_name)
            (set! {symbol}_db_name
                  (us_diphone_init ({symbol}_runtime_db_params))))
        {symbol}_db_name)))

(define ({symbol}_configure_join_windows)
  ;; Festival's UniSyn implementation reads Param.unisyn.*, not the legacy
  ;; global `window_factor` variable.  Keep the generated voice's standalone
  ;; default on Festival's stable symmetric analysis-period geometry.  The GUI
  ;; may opt a metadata-qualified Japanese-only bank into asymmetric windows
  ;; after voice activation; integrated and legacy paths stay symmetric.
  (Param.set "unisyn.window_name" "hanning")
  (Param.set "unisyn.window_factor" 1.0)
  (Param.set "unisyn.window_symmetric" 1))

(define (voice_{symbol})
  "Japanese UTAU-derived UniSyn voice generated by japanese_festival.py."
  (voice_reset)
  (Parameter.set 'Language 'japanese)
  (PhoneSet.select '{symbol})
  (Parameter.set 'Int_Method 'DuffInt)
  (Parameter.set 'Int_Target_Method Int_Targets_Default)
  (set! duffint_params
        '((start {average_pitch_hz * 1.03:.2f})
          (end {average_pitch_hz * 0.94:.2f})))
  (Parameter.set 'Duration_Method 'Default)
  (set! phoneme_durations {symbol}_phone_durs)
  (set! UniSyn_module_hooks (list {symbol}_select_units))
  (set! us_abs_offset 0.0)
  ({symbol}_configure_join_windows)
  (set! us_rel_offset 0.0)
  (set! us_gain 0.9)
  (Parameter.set 'Synth_Method 'UniSyn)
  (Parameter.set 'us_sigpr 'psola)
  (us_db_select ({symbol}_active_db_name))
  (set! current-voice '{symbol}))

(proclaim_voice
 '{symbol}
 '((language japanese)
   (gender unknown)
   (dialect none)
   (description "Japanese UTAU-derived UniSyn voice.")))

(provide '{symbol})
"""
    path = output / "festvox" / f"{voice_name}_ja.scm"
    path.write_text(scheme, encoding="utf-8", newline="\n")
    return path


def _tool_command(
    tool: str,
    arguments: Sequence[str],
    *,
    distro: Optional[str],
) -> list[str]:
    if os.name != "nt":
        return [tool, *arguments]
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        raise RuntimeError("wsl.exe is required to run Festival tools on Windows")
    command = [wsl]
    if distro:
        command.extend(["-d", distro])
    command.extend(["--", tool])
    command.extend(arguments)
    return command


def _tool_path(path: Path) -> str:
    return _portable_runtime_path(path) if os.name == "nt" else str(path)


@dataclass(frozen=True)
class _F0Guide:
    times: tuple[float, ...]
    values: tuple[float, ...]
    provenance: str


def _find_frq_for_wav(wav_path: Path) -> Optional[Path]:
    names = (
        wav_path.stem + ".frq",
        wav_path.name + ".frq",
        wav_path.name.replace(".", "_") + ".frq",
    )
    for name in names:
        candidate = wav_path.with_name(name)
        if candidate.is_file():
            return candidate
    try:
        entries = {
            path.name.casefold(): path
            for path in wav_path.parent.iterdir()
            if path.is_file()
        }
    except OSError:
        return None
    for name in names:
        candidate = entries.get(name.casefold())
        if candidate is not None:
            return candidate
    return None


def _read_frq_guide(wav_path: Path) -> Optional[_F0Guide]:
    """Read an adjacent UTAU FREQ0003 contour without modifying the bank."""
    frq_path = _find_frq_for_wav(wav_path)
    if frq_path is None:
        return None
    try:
        data = frq_path.read_bytes()
        _, sample_rate = _wav_duration(wav_path)
    except (OSError, EOFError, wave.Error):
        return None
    if len(data) < 40 or data[:8] != b"FREQ0003" or sample_rate <= 0:
        return None
    try:
        hop_samples = struct.unpack_from("<i", data, 8)[0]
        frame_count = struct.unpack_from("<i", data, 36)[0]
    except struct.error:
        return None
    if (
        hop_samples <= 0
        or frame_count <= 0
        or frame_count > 10_000_000
        or len(data) < 40 + frame_count * 16
    ):
        return None
    times = []
    values = []
    for index in range(frame_count):
        try:
            value = float(struct.unpack_from(
                "<d", data, 40 + index * 16
            )[0])
        except struct.error:
            return None
        times.append(index * hop_samples / float(sample_rate))
        values.append(value if math.isfinite(value) and value > 0.0 else 0.0)
    return _F0Guide(tuple(times), tuple(values), "utau-frq")


def _guide_value(guide: _F0Guide, time_s: float) -> float:
    if not guide.times:
        return 0.0
    index = bisect.bisect_left(guide.times, float(time_s))
    if index <= 0:
        return guide.values[0]
    if index >= len(guide.times):
        return guide.values[-1]
    first_time = guide.times[index - 1]
    second_time = guide.times[index]
    first = guide.values[index - 1]
    second = guide.values[index]
    if first <= 0.0 or second <= 0.0 or second_time <= first_time:
        return 0.0
    fraction = (time_s - first_time) / (second_time - first_time)
    return first * (1.0 - fraction) + second * fraction


def _component_pitch_guide(
    components: Sequence[Mapping[str, object]],
    source_root: Path,
    duration: float,
    cache: dict[str, Optional[_F0Guide]],
    *,
    output_slice_key: str = "output_slice",
    crossfade_key: str = "crossfade",
) -> Optional[_F0Guide]:
    resolved = []
    for component in components:
        relative = str(component.get("wav") or "")
        source_slice = component.get("source_slice")
        output_slice = component.get(output_slice_key)
        crossfade = component.get(crossfade_key)
        if not relative or not isinstance(source_slice, Mapping):
            continue
        if not isinstance(output_slice, Mapping):
            continue
        source = (source_root / Path(relative)).resolve()
        if not _is_within(source, source_root) or not source.is_file():
            continue
        if relative not in cache:
            cache[relative] = _read_frq_guide(source)
        guide = cache[relative]
        if guide is None:
            continue
        try:
            source_start = float(source_slice["start"])
            output_start = float(output_slice["start"])
            output_end = float(output_slice["end"])
        except (KeyError, TypeError, ValueError):
            continue
        resolved.append((
            guide,
            source_start,
            output_start,
            output_end,
            dict(crossfade) if isinstance(crossfade, Mapping) else {},
        ))
    if not resolved:
        return None

    step = 0.005
    frame_count = max(2, int(math.ceil(duration / step)) + 1)
    times = []
    values = []
    for index in range(frame_count):
        output_time = min(duration, index * step)
        weighted = []
        for guide, source_start, output_start, output_end, fade in resolved:
            if output_time < output_start or output_time > output_end:
                continue
            value = _guide_value(
                guide, source_start + output_time - output_start
            )
            if value <= 0.0:
                continue
            weight = 1.0
            try:
                fade_start = float(fade["start"])
                fade_end = float(fade["end"])
                side = str(fade["side"])
            except (KeyError, TypeError, ValueError):
                fade_start = fade_end = 0.0
                side = ""
            if fade_end > fade_start and fade_start <= output_time <= fade_end:
                fraction = (
                    (output_time - fade_start) / (fade_end - fade_start)
                )
                weight = 1.0 - fraction if side == "left" else fraction
            weighted.append((value, max(0.0, weight)))
        total_weight = sum(weight for _, weight in weighted)
        if total_weight > 0.0:
            value = sum(value * weight for value, weight in weighted) / total_weight
        elif weighted:
            value = statistics.median(value for value, _ in weighted)
        else:
            value = 0.0
        times.append(output_time)
        values.append(float(value))
    return _F0Guide(tuple(times), tuple(values), "utau-frq-components")


def _pitch_guides_for_units(
    units: Sequence[JapaneseCompiledUnit],
    source_root: Optional[Path],
    output: Path,
) -> dict[str, _F0Guide]:
    if source_root is None:
        return {}
    guides: dict[str, _F0Guide] = {}
    cache: dict[str, Optional[_F0Guide]] = {}
    for unit in units:
        if unit.wav_name in guides:
            continue
        if unit.source_components:
            try:
                duration, _ = _wav_duration(output / "wav" / unit.wav_name)
            except (OSError, EOFError, wave.Error):
                continue
            guide = _component_pitch_guide(
                unit.source_components, source_root, duration, cache
            )
            legacy_name = _legacy_bridge_name(unit.wav_name)
            legacy_path = output / "wav" / legacy_name
            if (
                legacy_path.is_file()
                and any(
                    isinstance(component.get("legacy_output_slice"), Mapping)
                    for component in unit.source_components
                )
            ):
                try:
                    legacy_duration, _ = _wav_duration(legacy_path)
                except (OSError, EOFError, wave.Error):
                    legacy_duration = 0.0
                if legacy_duration > 0.0:
                    legacy_guide = _component_pitch_guide(
                        unit.source_components,
                        source_root,
                        legacy_duration,
                        cache,
                        output_slice_key="legacy_output_slice",
                        crossfade_key="legacy_crossfade",
                    )
                    if legacy_guide is not None:
                        guides[legacy_name] = legacy_guide
        else:
            relative = unit.source_path
            source = (source_root / Path(relative)).resolve()
            if (
                not relative
                or not _is_within(source, source_root)
                or not source.is_file()
            ):
                continue
            if relative not in cache:
                cache[relative] = _read_frq_guide(source)
            guide = cache[relative]
        if guide is not None:
            guides[unit.wav_name] = guide
    return guides


def _world_f0_guide(
    samples: Sequence[float],
    sample_rate: int,
    *,
    estimator: str,
    f0_min: float,
    f0_max: float,
) -> _F0Guide:
    """Estimate speech F0 with WORLD, imported only for FRQ-less banks."""
    estimator = str(estimator or "harvest").strip().lower()
    if estimator not in F0_FALLBACK_ESTIMATORS:
        raise ValueError(
            "F0 estimator must be one of: "
            + ", ".join(F0_FALLBACK_ESTIMATORS)
        )
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API.*")
            import numpy as np
            import pyworld
    except ImportError as error:
        raise RuntimeError(
            "This UTAU recording has no usable FRQ contour. Install the "
            "optional WORLD fallback with `python -m pip install "
            "pyworld==0.3.5 \"setuptools<81\"`, or rebuild with "
            "--skip-pitchmarks only for metadata inspection."
        ) from error

    analysis_rate = max(16000, int(sample_rate))
    analysis = (
        tuple(samples) if analysis_rate == int(sample_rate)
        else _resample_linear(samples, int(sample_rate), analysis_rate)
    )
    waveform_values = np.ascontiguousarray(analysis, dtype=np.float64)
    if estimator == "harvest":
        raw_f0, times = pyworld.harvest(
            waveform_values,
            analysis_rate,
            f0_floor=float(f0_min),
            f0_ceil=float(f0_max),
            frame_period=5.0,
        )
    else:
        raw_f0, times = pyworld.dio(
            waveform_values,
            analysis_rate,
            f0_floor=float(f0_min),
            f0_ceil=float(f0_max),
            frame_period=5.0,
        )
    refined = pyworld.stonemask(
        waveform_values, raw_f0, times, analysis_rate
    )
    values = tuple(
        float(value)
        if math.isfinite(float(value)) and f0_min <= value <= f0_max
        else 0.0
        for value in refined
    )
    provenance = "world-%s-stonemask" % estimator
    if not any(value > 0.0 for value in values):
        # Fricatives, closures, breaths, and other valid source units can be
        # energetic while remaining entirely unvoiced. PSOLA still needs
        # traversal epochs, but the diagnostic contour must not invent F0.
        provenance += "-unvoiced"
    return _F0Guide(
        tuple(float(value) for value in times),
        values,
        provenance,
    )


def _sanitize_f0_guide(
    guide: _F0Guide, f0_min: float, f0_max: float
) -> _F0Guide:
    """Reject invalid frames and repair only isolated octave/gap errors.

    This deliberately does not smooth a sustained bad region into something
    plausible.  A run of bad estimates stays bad/unvoiced so the generated
    diagnostic contour exposes it and the user can replace the FRQ data or
    select another fallback estimator.
    """
    values = [
        value if f0_min <= value <= f0_max else 0.0
        for value in guide.values
    ]
    original = tuple(values)
    for index, value in enumerate(original):
        if value <= 0.0:
            continue
        neighbours = [
            original[other]
            for other in range(max(0, index - 2), min(len(original), index + 3))
            if other != index and original[other] > 0.0
        ]
        if len(neighbours) < 2:
            continue
        reference = statistics.median(neighbours)
        candidates = [
            candidate for candidate in (value * 0.5, value, value * 2.0)
            if f0_min <= candidate <= f0_max
        ]
        corrected = min(
            candidates,
            key=lambda candidate: abs(math.log2(candidate / reference)),
        )
        if (
            corrected != value
            and abs(math.log2(value / reference))
            - abs(math.log2(corrected / reference)) >= 0.35
        ):
            values[index] = corrected

    index = 0
    while index < len(values):
        if values[index] > 0.0:
            index += 1
            continue
        first = index
        while index < len(values) and values[index] <= 0.0:
            index += 1
        last = index
        if first == 0 or last >= len(values):
            continue
        if guide.times[last] - guide.times[first - 1] > 0.035:
            continue
        left = values[first - 1]
        right = values[last]
        if left <= 0.0 or right <= 0.0:
            continue
        if max(left, right) / min(left, right) > 1.5:
            continue
        span = last - first + 1
        for offset, position in enumerate(range(first, last), start=1):
            fraction = offset / span
            values[position] = left * (1.0 - fraction) + right * fraction
    return _F0Guide(guide.times, tuple(values), guide.provenance)


def _centered_average(
    samples: Sequence[float], radius: int
) -> tuple[float, ...]:
    if radius <= 0 or not samples:
        return tuple(float(value) for value in samples)
    prefix = [0.0]
    running = 0.0
    for value in samples:
        running += float(value)
        prefix.append(running)
    count = len(samples)
    result = []
    for index in range(count):
        first = max(0, index - radius)
        last = min(count, index + radius + 1)
        result.append((prefix[last] - prefix[first]) / (last - first))
    return tuple(result)


def _negative_zero_crossings(
    samples: Sequence[float], sample_rate: int
) -> tuple[float, ...]:
    if len(samples) < 2 or sample_rate <= 0:
        return ()
    target_rate = min(sample_rate, 16000)
    analysis = _resample_linear(samples, sample_rate, target_rate)
    low_pass = _centered_average(
        analysis, max(1, int(round(target_rate / 1200.0)))
    )
    baseline = _centered_average(
        low_pass, max(1, int(round(target_rate / 80.0)))
    )
    filtered = tuple(
        value - baseline[index] for index, value in enumerate(low_pass)
    )
    crossings = []
    for index in range(1, len(filtered)):
        before = filtered[index - 1]
        after = filtered[index]
        if before < 0.0 or after >= 0.0:
            continue
        denominator = before - after
        fraction = before / denominator if denominator else 0.0
        crossings.append((index - 1 + fraction) / target_rate)
    return tuple(crossings)


def _nearest_crossing(
    crossings: Sequence[float], target: float, distance: float
) -> Optional[float]:
    if not crossings:
        return None
    index = bisect.bisect_left(crossings, target)
    candidates = []
    if index < len(crossings):
        candidates.append(crossings[index])
    if index > 0:
        candidates.append(crossings[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - target))
    return nearest if abs(nearest - target) <= distance else None


def _generate_pitchmarks_legacy(
    samples: Sequence[float],
    sample_rate: int,
    guide: _F0Guide,
    *,
    default_f0: float,
    f0_min: float,
    f0_max: float,
) -> tuple[float, ...]:
    duration = len(samples) / float(sample_rate)
    if duration <= 0.0:
        return ()
    crossings = _negative_zero_crossings(samples, sample_rate)
    default_f0 = max(f0_min, min(f0_max, float(default_f0)))
    min_period = 1.0 / f0_max
    max_period = 1.0 / f0_min

    initial_f0 = _guide_value(guide, 0.0)
    initial_period = 1.0 / (initial_f0 if initial_f0 > 0.0 else default_f0)
    first = initial_period * 0.5
    if initial_f0 > 0.0:
        crossing = _nearest_crossing(crossings, first, initial_period * 0.45)
        if crossing is not None:
            first = crossing
    first = max(min_period * 0.5, min(first, duration))
    marks = [first]
    previous_period = initial_period
    previous_voiced = initial_f0 > 0.0
    while marks[-1] < duration:
        probe_time = marks[-1] + previous_period * 0.5
        f0 = _guide_value(guide, probe_time)
        voiced = f0 > 0.0
        period = 1.0 / (f0 if voiced else default_f0)
        period = max(min_period, min(max_period, period))
        if voiced and previous_voiced:
            period = max(
                previous_period * 0.80,
                min(previous_period * 1.25, period),
            )
        target = marks[-1] + period
        if voiced:
            crossing = _nearest_crossing(crossings, target, period * 0.40)
            if crossing is not None:
                phase_shift = max(
                    -period * 0.12,
                    min(period * 0.12, crossing - target),
                )
                target += phase_shift
        interval = target - marks[-1]
        interval = max(min_period * 0.88, min(max_period * 1.12, interval))
        target = marks[-1] + interval
        if target >= duration:
            break
        marks.append(target)
        previous_period = interval
        previous_voiced = voiced
    return tuple(marks)


def _excitation_residual(
    samples: Sequence[float], sample_rate: int
) -> tuple[tuple[float, ...], int, int]:
    """Return a low-frequency, polarity-invariant excitation proxy.

    A zero crossing is not a reliable glottal phase reference.  A lightly
    smoothed pre-emphasis residual gives a much sharper closure-like event
    while remaining deterministic and dependency-free.  The epoch search uses
    residual magnitude: independently choosing a signed lobe for each bounded
    source clip can put neighboring clips on opposite phase conventions, and
    microphone polarity may legitimately differ between recordings.
    """
    if len(samples) < 3 or sample_rate <= 0:
        return (), max(1, sample_rate), 1
    target_rate = min(sample_rate, 16000)
    analysis = _resample_linear(samples, sample_rate, target_rate)
    smooth = _centered_average(
        analysis, max(1, int(round(target_rate / 5000.0))))
    baseline = _centered_average(
        smooth, max(1, int(round(target_rate / 70.0))))
    centered = tuple(value - baseline[index]
                     for index, value in enumerate(smooth))
    residual = [0.0]
    residual.extend(
        centered[index] - 0.96 * centered[index - 1]
        for index in range(1, len(centered))
    )
    # Zero denotes absolute residual magnitude in _nearest_excitation_epoch.
    return tuple(residual), target_rate, 0


def _nearest_excitation_epoch(
    residual: Sequence[float],
    residual_rate: int,
    target: float,
    radius: float,
    polarity: int,
) -> Optional[float]:
    if not residual or residual_rate <= 0 or radius <= 0.0:
        return None
    center = int(round(target * residual_rate))
    extent = max(1, int(round(radius * residual_rate)))
    first = max(1, center - extent)
    last = min(len(residual) - 1, center + extent)
    if first > last:
        return None
    local = [abs(float(residual[index])) for index in range(first, last + 1)]
    floor = statistics.median(local) if local else 0.0
    def score(index: int) -> float:
        value = float(residual[index])
        return abs(value) if polarity == 0 else polarity * value

    peak = max((score(index)
                for index in range(first, last + 1)), default=0.0)
    if peak <= max(1e-8, floor * 1.6):
        return None
    distance_penalty = peak * 0.18 / max(1, extent)
    best = max(
        range(first, last + 1),
        key=lambda index: (
            score(index) -
            abs(index - center) * distance_penalty,
            -abs(index - center),
            -index,
        ),
    )
    return best / float(residual_rate)


def _generate_pitchmarks_with_diagnostics(
    samples: Sequence[float],
    sample_rate: int,
    guide: _F0Guide,
    *,
    default_f0: float,
    f0_min: float,
    f0_max: float,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Generate epochs at one consistent excitation phase per recording."""
    duration = len(samples) / float(sample_rate)
    if duration <= 0.0:
        return (), {
            "phase_reference": "none",
            "aligned_epoch_count": 0,
            "median_epoch_shift_ms": 0.0,
            "maximum_epoch_shift_ms": 0.0,
        }
    residual, residual_rate, polarity = _excitation_residual(
        samples, sample_rate)
    default_f0 = max(f0_min, min(f0_max, float(default_f0)))
    min_period = 1.0 / f0_max
    max_period = 1.0 / f0_min

    initial_f0 = _guide_value(guide, 0.0)
    initial_period = 1.0 / (initial_f0 if initial_f0 > 0.0 else default_f0)
    provisional = initial_period * 0.5
    first = provisional
    shifts = []
    if initial_f0 > 0.0:
        epoch = _nearest_excitation_epoch(
            residual, residual_rate, provisional,
            # The first provisional mark has no established phase. Search a
            # full half-period once; later marks remain tightly constrained.
            min(initial_period * 0.52, 0.012), polarity)
        if epoch is not None:
            first = epoch
            shifts.append(first - provisional)
    first = max(min_period * 0.5, min(first, duration))
    marks = [first]
    previous_period = initial_period
    previous_voiced = initial_f0 > 0.0
    while marks[-1] < duration:
        probe_time = marks[-1] + previous_period * 0.5
        f0 = _guide_value(guide, probe_time)
        voiced = f0 > 0.0
        period = 1.0 / (f0 if voiced else default_f0)
        period = max(min_period, min(max_period, period))
        if voiced and previous_voiced:
            period = max(
                previous_period * 0.80,
                min(previous_period * 1.25, period),
            )
        provisional = marks[-1] + period
        target = provisional
        if voiced:
            epoch = _nearest_excitation_epoch(
                residual, residual_rate, provisional,
                min(period * 0.28, 0.0045), polarity)
            if epoch is not None:
                interval = epoch - marks[-1]
                if period * 0.78 <= interval <= period * 1.30:
                    target = epoch
                    shifts.append(target - provisional)
        interval = target - marks[-1]
        interval = max(min_period * 0.88, min(max_period * 1.12, interval))
        target = marks[-1] + interval
        if target >= duration:
            break
        marks.append(target)
        previous_period = interval
        previous_voiced = voiced
    absolute_shifts = [abs(value) * 1000.0 for value in shifts]
    return tuple(marks), {
        "phase_reference": (
            "absolute-excitation-residual" if polarity == 0 else
            "positive-excitation-residual" if polarity > 0 else
            "negative-excitation-residual"),
        "aligned_epoch_count": len(shifts),
        "median_epoch_shift_ms": round(
            statistics.median(absolute_shifts), 6)
            if absolute_shifts else 0.0,
        "maximum_epoch_shift_ms": round(
            max(absolute_shifts), 6) if absolute_shifts else 0.0,
    }


def _generate_pitchmarks(
    samples: Sequence[float],
    sample_rate: int,
    guide: _F0Guide,
    *,
    default_f0: float,
    f0_min: float,
    f0_max: float,
) -> tuple[float, ...]:
    marks, _diagnostics = _generate_pitchmarks_with_diagnostics(
        samples, sample_rate, guide,
        default_f0=default_f0, f0_min=f0_min, f0_max=f0_max)
    return marks


def _write_est_pitchmarks(path: Path, marks: Sequence[float]) -> None:
    lines = [
        "EST_File Track",
        "DataType ascii",
        f"NumFrames {len(marks)}",
        "NumChannels 0",
        "NumAuxChannels 0",
        "EqualSpace 0",
        "BreaksPresent true",
        "EST_Header_End",
    ]
    lines.extend(f"{mark:.6f}\t1 \t" for mark in marks)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_est_pitchmarks(path: Path) -> tuple[float, ...]:
    marks: list[float] = []
    in_data = False
    for line in path.read_text(encoding="ascii").splitlines():
        if line == "EST_Header_End":
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        marks.append(float(line.split()[0]))
    return tuple(marks)


def _write_f0_guide(path: Path, guide: _F0Guide) -> None:
    """Persist the exact analyzed contour used to place source epochs."""
    frames = [
        [round(float(time), 6), round(float(value), 6)]
        for time, value in zip(guide.times, guide.values)
    ]
    path.write_bytes(_json_bytes({
        "schema_version": 1,
        "f0_source": guide.provenance,
        "unvoiced_value_hz": 0.0,
        "frames": frames,
    }))


def make_pitchmarks(
    output: Path,
    wav_names: Iterable[str],
    *,
    f0_min: float = 80.0,
    f0_max: float = 500.0,
    default_f0: float = 180.0,
    source_root: Optional[Path] = None,
    source_wavs: Optional[Mapping[str, Path]] = None,
    units: Sequence[JapaneseCompiledUnit] = (),
    f0_estimator: str = "harvest",
    distro: Optional[str] = "Ubuntu",
) -> int:
    """Create waveform-aligned EST pitchmarks without writing to source banks.

    UTAU FRQ tracks are authoritative when present. WORLD Harvest (default) or
    DIO+StoneMask supplies an F0 contour for recordings without FRQ data. The
    EST ``pitchmark`` program is intentionally not used because it expects a
    laryngograph waveform rather than ordinary speech.
    """
    if not 30.0 <= f0_min < f0_max <= 1000.0:
        raise ValueError("pitchmark range must satisfy 30 <= min < max <= 1000")
    if not f0_min <= default_f0 <= f0_max:
        raise ValueError("default pitch must be within the pitchmark range")
    f0_estimator = str(f0_estimator or "harvest").strip().lower()
    if f0_estimator not in F0_FALLBACK_ESTIMATORS:
        raise ValueError(
            "F0 estimator must be one of: "
            + ", ".join(F0_FALLBACK_ESTIMATORS)
        )
    # Kept in the public signature for callers that also pass a Festival WSL
    # distribution. WORLD analysis itself runs in this Python process.
    _ = distro
    pm_dir = output / "pm"
    pm_dir.mkdir(parents=True, exist_ok=True)
    guides = _pitch_guides_for_units(units, source_root, output)
    for wav_name, source_wav in sorted(
        dict(source_wavs or {}).items(), key=lambda item: str(item[0])
    ):
        normalized_name = Path(str(wav_name)).name
        if normalized_name in guides:
            continue
        source_path = Path(source_wav).resolve()
        if not source_path.is_file():
            continue
        guide = _read_frq_guide(source_path)
        if guide is not None:
            guides[normalized_name] = guide
    manifest = {}
    count = 0
    for wav_name in sorted(set(wav_names)):
        source = output / "wav" / wav_name
        destination = pm_dir / (Path(wav_name).stem + ".pm")
        legacy_destination = pm_dir / (
            Path(wav_name).stem + ".legacy.pm")
        try:
            samples, sample_rate = _read_pcm_mono(source)
        except (OSError, ValueError, EOFError, wave.Error) as error:
            raise RuntimeError(
                f"could not read {wav_name} for pitchmarking: {error}"
            ) from error
        guide = guides.get(wav_name)
        peak = max((abs(value) for value in samples), default=0.0)
        if guide is None and peak > 1e-5:
            guide = _world_f0_guide(
                samples,
                sample_rate,
                estimator=f0_estimator,
                f0_min=f0_min,
                f0_max=f0_max,
            )
        if guide is None:
            guide = _F0Guide((0.0,), (0.0,), "unvoiced-default")
        guide = _sanitize_f0_guide(guide, f0_min, f0_max)
        legacy_marks = _generate_pitchmarks_legacy(
            samples,
            sample_rate,
            guide,
            default_f0=default_f0,
            f0_min=f0_min,
            f0_max=f0_max,
        )
        # A real-bank A/B audit rejected the experimental residual-epoch
        # tracker: it was internally periodic but did not preserve one stable
        # phase convention across independent UTAU recordings.  The proven
        # negative-going low-pass crossing remains the source epoch.  Join
        # improvement comes from Festival's asymmetric period windows, which
        # align these epochs without changing unit identity or source audio.
        marks = legacy_marks
        phase_diagnostics = {
            "phase_reference": "negative-zero-crossing",
            "aligned_epoch_count": len(marks),
            "median_epoch_shift_ms": 0.0,
            "maximum_epoch_shift_ms": 0.0,
        }
        _write_est_pitchmarks(destination, marks)
        _write_est_pitchmarks(legacy_destination, legacy_marks)
        f0_name = Path(wav_name).stem + ".f0.json"
        _write_f0_guide(pm_dir / f0_name, guide)
        manifest[wav_name] = {
            "f0_source": guide.provenance,
            "f0_file": f0_name,
            "pitchmark_count": len(marks),
            "legacy_pitchmark_file": legacy_destination.name,
            "legacy_pitchmark_count": len(legacy_marks),
            **phase_diagnostics,
        }
        count += 1
    settings = (
        "source-pitchmarks-v5 method=frq-or-world-negative-zero-crossing "
        f"min={f0_min:.3f} max={f0_max:.3f} "
        f"default={default_f0:.3f} fallback={f0_estimator} "
        "join_window=asymmetric-analysis-period "
        "legacy=negative-zero-crossing+symmetric-analysis-period\n"
    )
    (pm_dir / ".pm_settings").write_text(settings, encoding="ascii")
    (pm_dir / "pitchmark_sources.json").write_bytes(_json_bytes({
        "schema_version": 2,
        "method": "shared-frq-guided-negative-zero-crossing-v3",
        "join_window": "asymmetric-analysis-period",
        "legacy_method": "negative-zero-crossing-v3",
        "legacy_join_window": "symmetric-analysis-period",
        "fallback_estimator": f0_estimator,
        "units": manifest,
    }))
    return count


def _choice_payload(unit: JapaneseCompiledUnit) -> dict[str, object]:
    if unit.left_phone in _VOWELS and unit.right_phone in _VOWELS:
        transition_kind = "vv"
    elif unit.left_phone in _VOWELS:
        transition_kind = "vc"
    elif unit.right_phone in _VOWELS:
        transition_kind = "cv"
    else:
        transition_kind = "other"
    left_source_candidate_id = unit.candidate_id
    right_source_candidate_id = unit.candidate_id
    left_source_role = unit.role
    if unit.source_components:
        left_source_candidate_id = ""
        right_source_candidate_id = ""
        left_source_role = ""
        for component in unit.source_components:
            crossfade = component.get("crossfade")
            side = (
                str(crossfade.get("side") or "")
                if isinstance(crossfade, Mapping) else ""
            )
            source_candidate_id = str(
                component.get("candidate_id") or ""
            )
            if side == "left":
                left_source_candidate_id = source_candidate_id
                left_source_role = str(component.get("role") or "")
            elif side == "right":
                right_source_candidate_id = source_candidate_id
    return {
        "id": unit.candidate_id,
        "candidate_id": unit.candidate_id,
        "diphone": unit.diphone,
        "left_name": unit.left_name,
        "index_name": unit.index_name,
        "wav_name": unit.wav_name,
        "edge_index": unit.edge_index,
        "edge_offset": unit.edge_offset,
        "role": unit.role,
        # ``family=vcv`` describes the UTAU source alias convention. The
        # compiled Festival unit is still a diphone, so a vowel-to-vowel edge
        # is explicitly VV rather than being mislabeled as VCV.
        "transition_kind": transition_kind,
        "family": unit.family,
        "selection_cost": unit.selection_cost,
        "alias": unit.source_alias,
        "wav": unit.source_path,
        "oto_file": unit.source_oto_path,
        "oto_line": unit.source_oto_line,
        "source_pitch_tags": list(unit.source_pitch_tags),
        "subbank_ids": list(unit.subbank_ids),
        "geometry_method": unit.geometry_method,
        "source_slice": {
            "start": unit.start,
            "phone_boundary": unit.midpoint,
            "end": unit.end,
        },
        "shared_anchor": unit.shared_anchor,
        "oto_timing_ms": {
            "offset": unit.oto_offset_ms,
            "consonant": unit.oto_consonant_ms,
            "cutoff": unit.oto_cutoff_ms,
            "preutterance": unit.oto_preutterance_ms,
            "overlap": unit.oto_overlap_ms,
        },
        "effective_overlap_ms": unit.effective_overlap_ms,
        "overlap_method": unit.overlap_method,
        "recorded_left_context": unit.recorded_left_context,
        "recorded_right_context": unit.recorded_right_context,
        "moraic_nasal_allophone": unit.moraic_nasal_allophone,
        "left_source_candidate_id": left_source_candidate_id,
        "right_source_candidate_id": right_source_candidate_id,
        "left_source_role": left_source_role,
        "vowel_blend_activation_seconds": round(
            _phone_duration(unit.left_phone)
            * VOWEL_BLEND_LONG_DURATION_FACTOR,
            6,
        ) if (
            unit.left_phone in _VOWELS
            and left_source_role == "vowel_blend"
        ) else 1000000.0,
        "source_components": [
            dict(component) for component in unit.source_components
        ],
        "source_window": dict(unit.source_window),
        "window_left_name": unit.window_left_name or unit.left_name,
        "window_right_name": unit.window_right_name or unit.left_name,
        "window_both_name": unit.window_both_name or unit.left_name,
        "window_left_activation": unit.window_left_activation,
        "window_right_activation": unit.window_right_activation,
    }


def _apply_generated_bridge_pitchmark_guards(
    output: Path,
    units: list[JapaneseCompiledUnit],
    index: dict[str, tuple[str, float, float, float]],
    alternatives: dict[str, tuple[Mapping[str, object], ...]],
) -> dict[str, object]:
    """Give each generated voiced-left bridge one real preceding epoch.

    Festival chooses the pitchmark nearest an EST start, then loads the
    preceding pitchmark as source-window context.  A start at zero therefore
    has no preceding epoch.  Finalizing against the emitted PM track avoids a
    phase-dependent estimate: the normal index begins exactly at PM 1, while
    the separately constructed Legacy index remains at its pre-fix geometry.
    """
    applied: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    updated_units: list[JapaneseCompiledUnit] = []
    for unit in units:
        left_component_index = next((
            position for position, component in enumerate(
                unit.source_components
            )
            if component.get("purpose") == "left_stable_phone"
        ), None)
        if unit.role != "generated_cv_bridge" or left_component_index is None:
            updated_units.append(unit)
            continue

        pm_path = output / "pm" / (Path(unit.wav_name).stem + ".pm")
        reason = ""
        try:
            marks = _read_est_pitchmarks(pm_path)
        except (OSError, ValueError, IndexError):
            marks = ()
            reason = "pitchmark_track_unreadable"
        if len(marks) < 2:
            reason = reason or "fewer_than_two_pitchmarks"
        indexed_start = float(marks[1]) if len(marks) >= 2 else 0.0
        if not reason and indexed_start > unit.midpoint - 0.002:
            reason = "second_pitchmark_reaches_phone_boundary"
        if reason:
            unavailable.append({
                "candidate_id": unit.candidate_id,
                "diphone": unit.diphone,
                "reason": reason,
            })
            updated_units.append(unit)
            continue

        components = [dict(component) for component in unit.source_components]
        left_component = components[left_component_index]
        source_slice = left_component.get("source_slice")
        if isinstance(source_slice, Mapping):
            left_component["indexed_source_start"] = round(
                float(source_slice.get("start", 0.0)) + indexed_start,
                6,
            )
        left_component["analysis_guard"] = {
            "wav_start": 0.0,
            "indexed_start": round(indexed_start, 6),
            "pitchmark_index": 1,
        }
        guarded = replace(
            unit,
            start=round(indexed_start, 6),
            source_components=tuple(components),
        )
        updated_units.append(guarded)
        index[guarded.index_name] = (
            guarded.wav_name,
            guarded.start,
            guarded.midpoint,
            guarded.end,
        )
        choices = []
        for choice in alternatives.get(guarded.diphone, ()):
            if str(choice.get("candidate_id") or "") != guarded.candidate_id:
                choices.append(choice)
                continue
            replacement_choice = _choice_payload(guarded)
            for key, value in choice.items():
                replacement_choice.setdefault(key, value)
            choices.append(replacement_choice)
        alternatives[guarded.diphone] = tuple(choices)
        applied.append({
            "candidate_id": guarded.candidate_id,
            "diphone": guarded.diphone,
            "indexed_start": guarded.start,
            "pitchmark_index": 1,
        })

    units[:] = updated_units
    return {
        "method": "normal-generated-bridge-start-at-second-pitchmark",
        "eligible_count": len(applied) + len(unavailable),
        "applied_count": len(applied),
        "unavailable_count": len(unavailable),
        "applied": sorted(
            applied, key=lambda row: (row["diphone"], row["candidate_id"])
        ),
        "unavailable": sorted(
            unavailable,
            key=lambda row: (
                row["diphone"], row["candidate_id"], row["reason"]
            ),
        ),
    }


def _generated_bridge_validation_summary(
    units: Sequence[JapaneseCompiledUnit],
) -> dict[str, object]:
    """Return a deterministic, path-private audit of generated bridges."""
    rows: list[dict[str, object]] = []
    failure_counts: dict[str, int] = {}
    for unit in sorted(
        (item for item in units if item.role == "generated_cv_bridge"),
        key=lambda item: (item.diphone, item.candidate_id),
    ):
        conditioning: Optional[Mapping[str, object]] = None
        for component in unit.source_components:
            candidate = component.get("join_conditioning")
            if isinstance(candidate, Mapping):
                conditioning = candidate
                break

        if conditioning is None:
            failures = ("VALIDATION_METADATA_MISSING",)
            validation_passed = False
            legacy_fallback_used = False
            gate_active = False
            content_passed = False
        else:
            failures = tuple(sorted({
                str(value) for value in conditioning.get(
                    "validation_failures", ())
                if str(value)
            }))
            validation_passed = bool(
                conditioning.get("validation_passed", not failures)
            ) and not failures
            legacy_fallback_used = bool(
                conditioning.get("legacy_fallback_used", False)
            )
            gate_active = bool(
                conditioning.get("acoustic_validation_gate_active", False)
            )
            content_passed = bool(
                conditioning.get("content_preservation_passed", True)
            )
            if not validation_passed and not failures:
                failures = ("VALIDATION_FAILED_UNSPECIFIED",)

        for failure in failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        rows.append({
            "candidate_id": unit.candidate_id,
            "diphone": unit.diphone,
            "validation_passed": validation_passed,
            "validation_failures": list(failures),
            "legacy_fallback_used": legacy_fallback_used,
            "acoustic_validation_gate_active": gate_active,
            "content_preservation_passed": content_passed,
        })

    return {
        "schema_version": 1,
        "conditioning_version": JOIN_SYNTHESIS_CONDITIONING_VERSION,
        "candidate_count": len(rows),
        "passed_count": sum(
            1 for row in rows if row["validation_passed"]
        ),
        "failed_count": sum(
            1 for row in rows if not row["validation_passed"]
        ),
        "legacy_fallback_count": sum(
            1 for row in rows if row["legacy_fallback_used"]
        ),
        "failure_counts": {
            key: failure_counts[key] for key in sorted(failure_counts)
        },
        "candidates": rows,
    }


def _base_role_priority(
    candidate: JapaneseSourceCandidate,
    proposal: _EdgeProposal,
    bank_type: str,
) -> int:
    """Keep the EST base row acoustically safe if a selector hook is absent."""
    role = candidate.role
    if role == "phrase_start_cv":
        return 0 if proposal.left == "pau" else 90
    if role == "vc_transition":
        return 0 if bank_type == "cvvc" else 25
    if role == "vcv_mora":
        if proposal.left in _VOWELS and proposal.right in _VOWELS:
            return 0
        return 0 if bank_type == "vcv" else 20
    if role in {"mora_cv", "special_mora"}:
        return 1
    if role == "vowel_blend":
        return 10
    if role == "release":
        return 80
    return 50


def compile_festival_voice(
    graph: JapaneseCandidateGraph,
    output: Path | str,
    *,
    voice_name: str = "japanese_utau",
    average_pitch_hz: float = 180.0,
    pitchmark: bool = True,
    f0_min: float = 80.0,
    f0_max: float = 500.0,
    f0_estimator: str = "harvest",
    source_window_mode: str = "adaptive",
    source_window_ms: float = DEFAULT_SOURCE_WINDOW_MS,
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
    speaker_pitch_analysis: Optional[Mapping[str, object]] = None,
    wsl_distro: Optional[str] = "Ubuntu",
    runtime_audio_storage: str = "grouped",
) -> JapaneseFestivalBuild:
    """Compile a Phase 2 graph into an isolated generated Japanese voice."""
    if runtime_audio_storage not in RUNTIME_AUDIO_STORAGE_MODES:
        raise ValueError(
            "runtime_audio_storage must be grouped or separate")
    source_window_mode = normalize_source_window_mode(source_window_mode)
    zero_overlap_guard_ms = normalize_zero_overlap_guard_ms(
        zero_overlap_guard_ms
    )
    source_window_ms = float(source_window_ms)
    if not math.isfinite(source_window_ms) or not 20.0 <= source_window_ms <= 2000.0:
        raise ValueError("source window must be between 20 and 2000 ms")
    if (
        graph.profile.bank_configuration not in {"cv", "vcv", "cvvc"}
        or graph.voice_configuration is None
        or graph.voice_configuration.selection_status != "explicit"
    ):
        raise ValueError(
            "Japanese Festival builds require an explicit bank type: cv, "
            "vcv, or cvvc. Analyzer inference is a recommendation only."
        )
    if str(graph.profile.voice_color or "").casefold() == "all":
        raise ValueError(
            "The stable Japanese builder accepts one source configuration "
            "at a time; voice_color='all' is not supported."
        )
    observed_subbanks = sorted({
        subbank_id
        for candidate in graph.candidates
        for subbank_id in candidate.subbank_ids
    })
    if len(observed_subbanks) > 1:
        raise ValueError(
            "The selected source contains multiple pitch or voice-color "
            "subbanks. Build one subbank folder as one voice configuration."
        )
    source_root = graph._bank_root.resolve() if graph._bank_root else None
    if source_root is None or not source_root.is_dir():
        raise ValueError("candidate graph has no readable source-bank root")
    output_root = Path(output).expanduser().resolve()
    if _is_within(output_root, source_root):
        raise ValueError("refusing to build inside the source UTAU voicebank")
    name = _safe_voice_name(voice_name)
    voice_entry_point = f"voice_{name}_ja"
    voice_configuration = graph.voice_configuration.with_entry_point(
        "ja", voice_entry_point
    )
    source_bundle = (
        replace(
            graph.source_bundle,
            speaker_pitch_analysis=dict(speaker_pitch_analysis),
        )
        if speaker_pitch_analysis else graph.source_bundle
    )
    manifest_fields = generated_voice_fields(source_bundle, voice_configuration)
    if not 40.0 <= float(average_pitch_hz) <= 700.0:
        raise ValueError("average pitch must be between 40 and 700 Hz")
    for folder in ("wav", "pm", "dic", "festvox"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    diagnostics: list[JapaneseBuildDiagnostic] = []
    raw_units: list[tuple[JapaneseSourceCandidate, _EdgeProposal, str]] = []
    copied_sources: dict[str, str] = {}
    sample_rates: list[int] = []
    selectable_count = 0
    configuration_excluded_count = 0

    for candidate in graph.candidates:
        if not runtime_family_allowed(graph.profile, candidate.family):
            configuration_excluded_count += 1
            continue
        if not candidate.selectable:
            continue
        selectable_count += 1
        if candidate.role in {"breath", "silence", "extra"}:
            diagnostics.append(JapaneseBuildDiagnostic(
                code="nonlinguistic_candidate_not_phone",
                message=(
                    "Candidate is preserved in the source graph but is not "
                    "a canonical linguistic phone unit."
                ),
                severity="info",
                candidate_id=candidate.candidate_id,
                source_path=candidate.source.wav_path,
                details={"role": candidate.role},
            ))
            continue
        source_relative = candidate.source.wav_path
        if not source_relative or not candidate.source.wav_within_bank:
            diagnostics.append(JapaneseBuildDiagnostic(
                code="candidate_source_unavailable",
                message="Selectable candidate has no safe source WAV.",
                candidate_id=candidate.candidate_id,
                source_path=source_relative,
            ))
            continue
        source = (source_root / Path(source_relative)).resolve()
        if not _is_within(source, source_root) or not source.is_file():
            diagnostics.append(JapaneseBuildDiagnostic(
                code="candidate_source_missing",
                message="Candidate source WAV is missing or outside the bank.",
                candidate_id=candidate.candidate_id,
                source_path=source_relative,
            ))
            continue
        try:
            duration, sample_rate = _wav_duration(source)
        except (OSError, EOFError, wave.Error) as error:
            diagnostics.append(JapaneseBuildDiagnostic(
                code="source_wav_unreadable",
                message=f"Source WAV could not be read: {error}",
                candidate_id=candidate.candidate_id,
                source_path=source_relative,
            ))
            continue
        proposals = candidate_edge_proposals(
            candidate,
            duration,
            zero_overlap_guard_ms=zero_overlap_guard_ms,
        )
        if not proposals:
            diagnostics.append(JapaneseBuildDiagnostic(
                code="candidate_geometry_unusable",
                message="Candidate did not yield a valid Festival unit edge.",
                candidate_id=candidate.candidate_id,
                source_path=source_relative,
                details={"role": candidate.role},
            ))
            continue
        first_source_use = source_relative not in copied_sources
        wav_name = copied_sources.setdefault(
            source_relative, _copy_name(source_relative)
        )
        if first_source_use:
            target = output_root / "wav" / wav_name
            source_digest = hashlib.sha256(source.read_bytes()).digest()
            target_digest = (
                hashlib.sha256(target.read_bytes()).digest()
                if target.is_file() else None
            )
            if source_digest != target_digest:
                shutil.copyfile(source, target)
        sample_rates.append(sample_rate)
        for proposal in proposals:
            raw_units.append((candidate, proposal, wav_name))

    if configuration_excluded_count:
        diagnostics.append(JapaneseBuildDiagnostic(
            code="strict_runtime_family_policy_applied",
            message=(
                "The explicit bank type preserved nonmatching source rows in "
                "analysis metadata but excluded them from Festival units."
            ),
            severity="info",
            details={
                "excluded_candidate_count": configuration_excluded_count,
                "runtime_family_policy": runtime_family_policy(
                    graph.profile
                ),
            },
        ))

    grouped: dict[str, list[tuple[JapaneseSourceCandidate, _EdgeProposal, str]]] = {}
    for row in raw_units:
        proposal = row[1]
        grouped.setdefault(f"{proposal.left}-{proposal.right}", []).append(row)

    units: list[JapaneseCompiledUnit] = []
    index: dict[str, tuple[str, float, float, float]] = {}
    alternatives: dict[str, tuple[Mapping[str, object], ...]] = {}
    candidate_units: dict[str, list[Mapping[str, object]]] = {}
    for diphone in sorted(grouped):
        rows = sorted(
            grouped[diphone],
            key=lambda row: (
                _base_role_priority(
                    row[0], row[1], graph.profile.effective_configuration
                ),
                row[0].selection_cost,
                row[0].candidate_id,
                row[1].edge_offset,
                row[1].method,
            ),
        )
        choices = []
        # Keep every stable candidate ID addressable even when two OTO rows
        # point at byte-identical geometry.  Their audio may be shared, but a
        # user-selected source row must not disappear at the runtime boundary.
        for number, (candidate, proposal, wav_name) in enumerate(rows):
            left_name = proposal.left if number == 0 else (
                f"{proposal.left}__j{candidate.candidate_id[3:13]}e"
                f"{proposal.edge_offset + 2}"
            )
            _safe_token(left_name, "unit name")
            index_name = f"{left_name}-{proposal.right}"
            # Paired CVVC/VCV halves declare a source phone-center anchor
            # which must remain inside the primary window.  A short global
            # window may otherwise clip that anchor and make the two halves
            # geometrically inconsistent.
            unit_window_ms = source_window_ms
            if proposal.shared_anchor_ms is not None:
                unit_window_ms = max(
                    unit_window_ms,
                    abs(
                        proposal.shared_anchor_ms
                        - proposal.midpoint_ms
                    ),
                )
            window_plan = build_source_window_plan(
                proposal.start_ms / 1000.0,
                proposal.midpoint_ms / 1000.0,
                proposal.end_ms / 1000.0,
                mode=source_window_mode,
                half_window_ms=unit_window_ms,
            )
            window_names = source_window_variant_names(
                left_name, window_plan
            )
            primary_start, primary_midpoint, primary_end = \
                window_plan.geometry("base")
            unit = JapaneseCompiledUnit(
                candidate_id=candidate.candidate_id,
                edge_index=number,
                edge_offset=proposal.edge_offset,
                diphone=diphone,
                left_phone=proposal.left,
                right_phone=proposal.right,
                left_name=left_name,
                index_name=index_name,
                wav_name=wav_name,
                start=round(primary_start, 6),
                midpoint=round(primary_midpoint, 6),
                end=round(primary_end, 6),
                role=candidate.role,
                family=candidate.family,
                selection_cost=candidate.selection_cost,
                geometry_method=proposal.method,
                source_path=str(candidate.source.wav_path),
                source_alias=candidate.source.alias_raw,
                source_oto_path=candidate.source.oto_path,
                source_oto_line=candidate.source.line,
                shared_anchor=(
                    round(proposal.shared_anchor_ms / 1000.0, 6)
                    if proposal.shared_anchor_ms is not None else None
                ),
                oto_offset_ms=float(candidate.timing.offset),
                oto_consonant_ms=float(candidate.timing.consonant),
                oto_cutoff_ms=float(candidate.timing.cutoff),
                oto_preutterance_ms=float(candidate.timing.preutterance),
                oto_overlap_ms=float(candidate.timing.overlap),
                effective_overlap_ms=proposal.effective_overlap_ms,
                overlap_method=proposal.overlap_method,
                recorded_left_context=(
                    "pau" if candidate.role == "phrase_start_cv"
                    else str(candidate.target.left_context or "*")
                ),
                recorded_right_context=(
                    str(candidate.target.phones[-1])
                    if (
                        candidate.role in {"vcv_mora", "phrase_start_cv"}
                        and proposal.edge_offset == -1
                        and len(candidate.target.phones) >= 2
                    )
                    else "pau" if proposal.right == "pau" else "*"
                ),
                moraic_nasal_allophone=str(
                    candidate.target.moraic_nasal_allophone or ""
                ),
                source_pitch_tags=candidate.source.pitch_tags,
                subbank_ids=candidate.subbank_ids,
                source_window=window_plan.to_dict(),
                window_left_name=window_names["left"],
                window_right_name=window_names["right"],
                window_both_name=window_names["both"],
                window_left_activation=(
                    round(window_plan.left_activation_duration, 6)
                    if window_plan.left_activation_duration is not None
                    else None
                ),
                window_right_activation=(
                    round(window_plan.right_activation_duration, 6)
                    if window_plan.right_activation_duration is not None
                    else None
                ),
            )
            units.append(unit)
            index[index_name] = (
                wav_name, unit.start, unit.midpoint, unit.end
            )
            for kind in ("left", "right", "both"):
                variant_left = window_names[kind]
                if variant_left == left_name:
                    continue
                variant_key = f"{variant_left}-{proposal.right}"
                start, midpoint, end = window_plan.geometry(kind)
                index[variant_key] = (
                    wav_name,
                    round(start, 6),
                    round(midpoint, 6),
                    round(end, 6),
                )
            choice = _choice_payload(unit)
            choices.append(choice)
            candidate_units.setdefault(unit.candidate_id, []).append({
                "diphone": unit.diphone,
                "left_name": unit.left_name,
                "index_name": unit.index_name,
                "edge_offset": unit.edge_offset,
                "edge_index": unit.edge_index,
            })
        alternatives[diphone] = tuple(choices)

    phones = {"pau", "sil", "cl"}
    for unit in units:
        phones.update((unit.left_phone, unit.right_phone))
    phones = {_safe_token(phone) for phone in phones}

    generated_bridge_names, unavailable_bridge_failures = (
        _compile_generated_bridges(
            output_root=output_root,
            configuration_id=voice_configuration.configuration_id,
            raw_units=raw_units,
            phones=phones,
            units=units,
            index=index,
            alternatives=alternatives,
            expected_f0_hz=float(average_pitch_hz),
            zero_overlap_guard_ms=zero_overlap_guard_ms,
        )
    )
    if generated_bridge_names:
        diagnostics.append(JapaneseBuildDiagnostic(
            code="generated_cv_transition_fallbacks",
            message=(
                "Generated bounded audible bridges for missing recorded VC/VV "
                "transitions; every use remains visible in runtime metadata."
            ),
            severity="info",
            details={"count": len(generated_bridge_names)},
        ))
    if unavailable_bridge_failures:
        diagnostics.append(JapaneseBuildDiagnostic(
            code="generated_transition_source_unavailable",
            message=(
                "Some theoretical transition bridges could not be generated "
                "because the selected bank lacks a usable source half."
            ),
            severity="warning",
            details={
                "count": len(unavailable_bridge_failures),
                "failures": [
                    dict(row) for row in unavailable_bridge_failures
                ],
            },
        ))

    silence_name = "_japanese_silence.wav"
    silence_rate = min(sample_rates) if sample_rates else 44100
    _write_silence(output_root / "wav" / silence_name, silence_rate)
    silence_geometry = (silence_name, 0.02, 0.15, 0.28)
    for left, right in (
        ("pau", "pau"), ("sil", "sil"), ("pau", "sil"),
        ("sil", "pau"),
    ):
        index.setdefault(f"{left}-{right}", silence_geometry)

    source_phones = tuple(sorted(phones))

    legacy_bridge_names = {
        name: _legacy_bridge_name(name)
        for name in sorted(generated_bridge_names)
    }
    generated_units_by_wav = {
        unit.wav_name: unit
        for unit in units
        if unit.wav_name in legacy_bridge_names
        and unit.role in {
            "generated_cv_bridge",
            "structural_consonant_hold",
        }
    }
    legacy_index: dict[str, tuple[str, float, float, float]] = {}
    for key, value in index.items():
        normal_wav_name, start, midpoint, end = value
        legacy_wav_name = legacy_bridge_names.get(
            normal_wav_name, normal_wav_name
        )
        legacy_wav_path = output_root / "wav" / legacy_wav_name
        generated_unit = generated_units_by_wav.get(normal_wav_name)
        if generated_unit is not None:
            start, midpoint, end = _legacy_bridge_geometry(
                generated_unit, legacy_wav_path
            )
        else:
            try:
                start, midpoint, end = _bounded_est_geometry(
                    legacy_wav_path, start, midpoint, end
                )
            except ValueError as error:
                raise ValueError(
                    f"legacy EST row {key!r}: {error}"
                ) from error
        legacy_index[key] = (
            legacy_wav_name, start, midpoint, end
        )

    wav_names = tuple(sorted(
        set(copied_sources.values())
        | generated_bridge_names
        | set(legacy_bridge_names.values())
        | {silence_name}
    ))
    bridge_pitchmark_guards: dict[str, object] = {
        "method": "normal-generated-bridge-start-at-second-pitchmark",
        "eligible_count": 0,
        "applied_count": 0,
        "unavailable_count": 0,
        "applied": [],
        "unavailable": [],
    }
    if pitchmark:
        make_pitchmarks(
            output_root,
            wav_names,
            f0_min=f0_min,
            f0_max=f0_max,
            default_f0=float(average_pitch_hz),
            source_root=source_root,
            units=tuple(units),
            f0_estimator=f0_estimator,
            distro=wsl_distro,
        )
        bridge_pitchmark_guards = _apply_generated_bridge_pitchmark_guards(
            output_root, units, index, alternatives
        )
        if int(bridge_pitchmark_guards["applied_count"]):
            diagnostics.append(JapaneseBuildDiagnostic(
                code="generated_bridge_pitchmark_guards",
                message=(
                    "Generated bridges with a stable left phone begin at "
                    "their second source pitchmark so UniSyn retains one "
                    "preceding analysis epoch."
                ),
                severity="info",
                details={
                    "eligible_count": bridge_pitchmark_guards[
                        "eligible_count"
                    ],
                    "applied_count": bridge_pitchmark_guards[
                        "applied_count"
                    ],
                },
            ))
        if int(bridge_pitchmark_guards["unavailable_count"]):
            diagnostics.append(JapaneseBuildDiagnostic(
                code="generated_bridge_pitchmark_guard_unavailable",
                message=(
                    "Some generated bridges lacked enough indexed source "
                    "context for a preceding UniSyn pitchmark."
                ),
                severity="warning",
                details={
                    "count": bridge_pitchmark_guards["unavailable_count"],
                    "candidates": bridge_pitchmark_guards["unavailable"],
                },
            ))
    else:
        diagnostics.append(JapaneseBuildDiagnostic(
            code="pitchmarks_not_generated",
            message="Pitchmark generation was explicitly skipped.",
            severity="info",
        ))

    sorted_units = tuple(sorted(
        units, key=lambda unit: (unit.diphone, unit.edge_index, unit.candidate_id)
    ))
    sorted_candidate_units = {
        key: tuple(sorted(
            value,
            key=lambda item: (
                int(item["edge_offset"]),
                str(item["diphone"]),
                str(item["left_name"]),
            ),
        ))
        for key, value in sorted(candidate_units.items())
    }
    generated_bridge_validation = _generated_bridge_validation_summary(
        sorted_units
    )
    structural_hold_count = sum(
        unit.role == "structural_consonant_hold" for unit in sorted_units
    )
    generated_cv_bridge_count = sum(
        unit.role == "generated_cv_bridge" for unit in sorted_units
    )

    _write_est(output_root, name, index)
    _write_est(output_root, name, legacy_index, legacy=True)
    _write_scheme(
        output_root,
        name,
        source_phones,
        alternatives,
        float(average_pitch_hz),
        _moraic_nasal_routing(graph.profile),
        runtime_audio_storage,
    )

    runtime_subbanks = [
        item.to_dict() for item in sorted(
            graph.profile.subbanks,
            key=lambda item: (item.order, item.subbank_id),
        )
    ]
    runtime_colors = sorted({
        str(item.get("color") or "") for item in runtime_subbanks
        if str(item.get("color") or "")
    }, key=lambda item: (item.casefold(), item))
    source_window_policy = {
        "mode": source_window_mode,
        "half_window_ms": round(source_window_ms, 6),
        "normal_unisyn_window_symmetric": False,
        "legacy_unisyn_window_symmetric": True,
        "unisyn_window_policy_reason": (
            "Japanese-only adaptive bridge voices may use pitchmark-aligned "
            "asymmetric windows; Legacy joins restores paired pre-fix assets "
            "with Festival's symmetric renderer."
        ),
        "adaptive_full_window_threshold": (
            "target phone half-duration must accommodate the full source half"
        ),
        "context_selection_precedence": "recording-first-window-second",
        "zero_overlap_guard_ms": round(zero_overlap_guard_ms, 6),
        "zero_overlap_policy": (
            "preserve raw OTO geometry by default; a nonzero guard is an "
            "explicit source-cut experiment"
        ),
    }
    runtime_metadata = {
        **manifest_fields,
        "schema_version": JAPANESE_FESTIVAL_SCHEMA_VERSION,
        "schema_status": JAPANESE_FESTIVAL_SCHEMA_STATUS,
        "builder_version": JAPANESE_FESTIVAL_BUILDER_VERSION,
        "kind": "japanese_festival_runtime_index",
        "language": "ja",
        "voice_name": name,
        "voice_entry_point": voice_entry_point,
        "average_pitch_hz": float(average_pitch_hz),
        "f0_min_hz": float(f0_min),
        "f0_max_hz": float(f0_max),
        "f0_fallback_estimator": str(f0_estimator),
        "special_phone_realizations": generated_voice_policy(),
        "phones": list(source_phones),
        "index": {key: list(index[key]) for key in sorted(index)},
        "alternatives": {
            key: [dict(item) for item in alternatives[key]]
            for key in sorted(alternatives)
        },
        "candidate_units": {
            key: [dict(item) for item in sorted_candidate_units[key]]
            for key in sorted(sorted_candidate_units)
        },
        "subbanks": runtime_subbanks,
        "available_voice_colors": runtime_colors,
        "selected_voice_color": graph.profile.voice_color,
        "runtime_family_policy": runtime_family_policy(graph.profile),
        "configuration_excluded_candidate_count": (
            configuration_excluded_count
        ),
        "moraic_nasal_routing": _moraic_nasal_routing(graph.profile),
        "moraic_nasal_allophones": {
            key: graph.profile.moraic_nasal_allophones[key].to_dict()
            for key in sorted(graph.profile.moraic_nasal_allophones)
        },
        "generated_cv_bridge_count": generated_cv_bridge_count,
        "structural_consonant_hold_count": structural_hold_count,
        "generated_cv_bridge_source_failures": len(
            unavailable_bridge_failures),
        "generated_bridge_validation": generated_bridge_validation,
        "generated_bridge_pitchmark_guards": bridge_pitchmark_guards,
        "join_databases": {
            "default": {
                "index_file": f"dic/{name}_ja_diphone.est",
                "source_bridge_method": (
                    "measured-pitch-synchronous-raised-cosine"
                ),
            },
            "legacy": {
                "index_file": f"dic/{name}_ja_diphone_legacy.est",
                "source_bridge_method": "fixed-linear-overlap",
                "generated_bridge_wavs": {
                    key: legacy_bridge_names[key]
                    for key in sorted(legacy_bridge_names)
                },
            },
        },
        "source_window_policy": source_window_policy,
        "runtime_audio_storage": separate_runtime_metadata(
            requested=runtime_audio_storage),
    }
    runtime_metadata["source_timing_profile"] = build_source_timing_profile(
        runtime_metadata["alternatives"]
    )
    (output_root / "dic" / "diphone_index.json").write_bytes(
        _json_bytes(runtime_metadata)
    )
    (output_root / "dic" / "unit_alternatives.json").write_bytes(
        _json_bytes({
            **manifest_fields,
            "schema_version": JAPANESE_FESTIVAL_SCHEMA_VERSION,
            "language": "ja",
            "diphones": runtime_metadata["alternatives"],
            "source_timing_profile": runtime_metadata[
                "source_timing_profile"
            ],
            "candidate_units": runtime_metadata["candidate_units"],
            "subbanks": runtime_metadata["subbanks"],
            "available_voice_colors": runtime_colors,
            "selected_voice_color": graph.profile.voice_color,
            "runtime_family_policy": runtime_metadata[
                "runtime_family_policy"
            ],
            "configuration_excluded_candidate_count": (
                configuration_excluded_count
            ),
            "moraic_nasal_routing": runtime_metadata[
                "moraic_nasal_routing"
            ],
            "moraic_nasal_allophones": runtime_metadata[
                "moraic_nasal_allophones"
            ],
            "special_phone_realizations": runtime_metadata[
                "special_phone_realizations"
            ],
            "phones": runtime_metadata["phones"],
            "structural_consonant_hold_count": structural_hold_count,
            "average_pitch_hz": float(average_pitch_hz),
            "f0_min_hz": float(f0_min),
            "f0_max_hz": float(f0_max),
            "f0_fallback_estimator": str(f0_estimator),
            "source_window_policy": source_window_policy,
            "generated_bridge_validation": runtime_metadata[
                "generated_bridge_validation"
            ],
            "generated_bridge_pitchmark_guards": runtime_metadata[
                "generated_bridge_pitchmark_guards"
            ],
            "runtime_audio_storage": runtime_metadata[
                "runtime_audio_storage"
            ],
        })
    )

    compiled_ids = {
        unit.candidate_id for unit in units
        if not unit.candidate_id.startswith("jfb_")
    }
    relative_files = tuple(sorted(
        {
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*") if path.is_file()
        }
        | {"dic/japanese_build_report.json"}
    ))
    build = JapaneseFestivalBuild(
        voice_name=name,
        voice_entry_point=voice_entry_point,
        phones=source_phones,
        units=sorted_units,
        index={key: index[key] for key in sorted(index)},
        alternatives={
            key: alternatives[key] for key in sorted(alternatives)
        },
        candidate_units=sorted_candidate_units,
        voice_manifest=manifest_fields,
        diagnostics=tuple(diagnostics),
        average_pitch_hz=float(average_pitch_hz),
        source_candidate_count=len(graph.candidates),
        selectable_candidate_count=selectable_count,
        compiled_candidate_count=len(compiled_ids),
        output_relative_files=relative_files,
        source_window_policy=source_window_policy,
        _output_root=output_root,
    )
    report_path = output_root / "dic" / "japanese_build_report.json"
    report_path.write_bytes(build.metadata_bytes())
    return build


def load_japanese_runtime_metadata(path: Path | str) -> dict[str, object]:
    source = Path(path)
    voice_root = source if source.is_dir() else source.parent.parent
    if source.is_dir():
        source = source / "dic" / "diphone_index.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("language") == "ja":
        if int(data.get("schema_version", 0)) != JAPANESE_FESTIVAL_SCHEMA_VERSION:
            raise ValueError("unsupported Japanese Festival metadata version")
        return data

    # Integrated ARPAsing builds deliberately keep one shared unit index with
    # the primary-language identity.  Adapt it only when the separately
    # versioned manifest explicitly declares a Japanese entry point; the
    # presence of Japanese-looking aliases alone is not authority.
    manifest_path = voice_root / "dic" / "voice_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            "runtime metadata is not a Japanese voice index"
        ) from error
    languages = {
        str(value).casefold() for value in
        (manifest.get("supported_languages") or ())
    }
    entry_points = dict(manifest.get("voice_entry_points") or {})
    japanese_entry = str(entry_points.get("ja") or "")
    if (data.get("kind") != "festival_unisyn_runtime_index" or
            "ja" not in languages or not japanese_entry):
        raise ValueError("runtime metadata is not a Japanese voice index")
    adapted = dict(data)
    adapted.update({
        "language": "ja",
        "schema_version": JAPANESE_FESTIVAL_SCHEMA_VERSION,
        "voice_name": str(data.get("name") or voice_root.name),
        "voice_entry_point": japanese_entry,
        "voice_scm": "festvox/%s.scm" % str(
            data.get("name") or voice_root.name),
        "metadata_adapter": "integrated-arpasing-japanese-v1",
        "shared_runtime_index_language": str(data.get("language") or ""),
        "supported_languages": sorted(languages),
    })
    return adapted


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a read-only Phase 2 Japanese UTAU candidate graph into "
            "a separate Festival/UniSyn voice."
        )
    )
    parser.add_argument("source", type=Path, help="source UTAU bank or subbank")
    parser.add_argument("output", type=Path, help="generated voice directory")
    parser.add_argument("--name", default="japanese_utau", help="voice name")
    parser.add_argument("--profile", type=Path, help="Phase 2 profile JSON")
    parser.add_argument(
        "--bank-type", choices=("cv", "vcv", "cvvc"), required=True,
        help=(
            "explicit alias system for this build; analyzer inference is "
            "never used as build authority"
        ),
    )
    parser.add_argument("--pitch", type=float, default=180.0)
    parser.add_argument("--f0-min", type=float, default=80.0)
    parser.add_argument("--f0-max", type=float, default=500.0)
    parser.add_argument(
        "--f0-estimator",
        choices=F0_FALLBACK_ESTIMATORS,
        default="harvest",
        help=(
            "fallback for recordings without UTAU FRQ data; Harvest is the "
            "quality default, DIO+StoneMask is faster"
        ),
    )
    parser.add_argument(
        "--source-window-mode", choices=SOURCE_WINDOW_MODES,
        default="adaptive",
    )
    parser.add_argument(
        "--source-window-ms", type=float,
        default=DEFAULT_SOURCE_WINDOW_MS,
    )
    parser.add_argument(
        "--zero-overlap-guard-ms", type=float,
        default=DEFAULT_ZERO_OVERLAP_GUARD_MS,
        help=("experimental source-cut guard for zero OTO overlap; "
              "disabled by default because it changes recorded geometry"),
    )
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--skip-pitchmarks", action="store_true")
    parser.add_argument(
        "--runtime-audio-storage",
        choices=RUNTIME_AUDIO_STORAGE_MODES,
        default="separate",
        help=("standalone compiler storage preference; use the unified "
              "builder to create the grouped runtime cache"),
    )
    args = parser.parse_args(argv)

    from japanese_candidates import compile_candidate_graph
    from japanese_profiles import infer_bank_profile, load_profile

    if args.profile:
        profile = replace(
            load_profile(args.profile),
            bank_configuration=args.bank_type,
        )
    else:
        profile = infer_bank_profile(
            args.source, bank_configuration=args.bank_type
        )
    graph = compile_candidate_graph(args.source, profile=profile)
    build = compile_festival_voice(
        graph,
        args.output,
        voice_name=args.name,
        average_pitch_hz=args.pitch,
        pitchmark=not args.skip_pitchmarks,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        f0_estimator=args.f0_estimator,
        source_window_mode=args.source_window_mode,
        source_window_ms=args.source_window_ms,
        zero_overlap_guard_ms=args.zero_overlap_guard_ms,
        wsl_distro=(args.wsl_distro or None),
        runtime_audio_storage=args.runtime_audio_storage,
    )
    print(
        f"Built {build.voice_entry_point}: {len(build.units)} units from "
        f"{build.compiled_candidate_count}/"
        f"{build.selectable_candidate_count} selectable candidates."
    )
    print(f"Output: {args.output}")
    if build.diagnostics:
        counts: dict[str, int] = {}
        for diagnostic in build.diagnostics:
            counts[diagnostic.code] = counts.get(diagnostic.code, 0) + 1
        print("Diagnostics: " + ", ".join(
            f"{key}={counts[key]}" for key in sorted(counts)
        ))
    print("Source UTAU bank was read only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
