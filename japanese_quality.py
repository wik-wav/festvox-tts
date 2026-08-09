"""Deterministic acoustic join diagnostics for generated Japanese voices.

Only generated voice copies are inspected.  The analyzer reads small windows
around adjacent UniSyn unit edges, reports transparent signal measurements,
and can reuse content-addressed cache entries.  It never writes to a source
UTAU bank and does not claim to measure perceived naturalness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import wave
from typing import Mapping, Optional, Sequence

from japanese_assembly import select_automatic_choice


QUALITY_SCHEMA_VERSION = 1
QUALITY_SCHEMA_STATUS = "phase5-provisional"
DEFAULT_WINDOW_SECONDS = 0.04
_UTAU_SOURCE_MARKERS = ("oto.ini", "character.yaml", "prefix.map")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_non_source_output(path: Path | str, purpose: str) -> Path:
    """Reject output paths nested in a directory marked as a UTAU bank."""
    resolved = Path(path).expanduser().resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    for directory in (current, *current.parents):
        if any((directory / marker).is_file()
               for marker in _UTAU_SOURCE_MARKERS):
            raise ValueError(
                f"Refusing to place {purpose} inside a source UTAU bank: "
                f"{resolved}"
            )
    return resolved


def _decode_pcm(data: bytes, sample_width: int, channels: int) -> tuple[float, ...]:
    if sample_width == 1:
        values = [(item - 128) / 128.0 for item in data]
    elif sample_width == 2:
        count = len(data) // 2
        values = [item / 32768.0 for item in struct.unpack(
            "<%dh" % count, data[:count * 2]
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
        values = [item / 2147483648.0 for item in struct.unpack(
            "<%di" % count, data[:count * 4]
        )]
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if channels <= 1:
        return tuple(values)
    frames = []
    for offset in range(0, len(values) - channels + 1, channels):
        frames.append(sum(values[offset:offset + channels]) / channels)
    return tuple(frames)


def _read_window(
    path: Path,
    position_seconds: float,
    *,
    before_seconds: float,
    after_seconds: float,
) -> tuple[tuple[float, ...], int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {path.name}")
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        frame_count = int(handle.getnframes())
        center = max(0, min(frame_count, int(round(position_seconds * rate))))
        first = max(0, center - int(round(before_seconds * rate)))
        last = min(frame_count, center + int(round(after_seconds * rate)))
        handle.setpos(first)
        data = handle.readframes(max(0, last - first))
    return _decode_pcm(data, width, channels), rate


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _zcr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    crossings = sum(
        1 for left, right in zip(values, values[1:])
        if (left < 0.0 <= right) or (left >= 0.0 > right)
    )
    return crossings / (len(values) - 1)


def _roughness(values: Sequence[float]) -> float:
    level = _rms(values)
    if len(values) < 2 or level < 1e-9:
        return 0.0
    return _rms(tuple(
        right - left for left, right in zip(values, values[1:])
    )) / level


def _pitch_hz(values: Sequence[float], sample_rate: int) -> Optional[float]:
    if len(values) < 8 or _rms(values) < 0.008 or _zcr(values) > 0.20:
        return None
    crossings = [
        index for index in range(1, len(values))
        if values[index - 1] <= 0.0 < values[index]
    ]
    periods = [
        right - left for left, right in zip(crossings, crossings[1:])
        if sample_rate / 500.0 <= right - left <= sample_rate / 50.0
    ]
    if len(periods) < 2:
        return None
    return sample_rate / statistics.median(periods)


@dataclass(frozen=True)
class JapaneseJoinMetric:
    join_index: int
    shared_phone: str
    left_diphone: str
    right_diphone: str
    left_candidate_id: str
    right_candidate_id: str
    left_wav: str
    right_wav: str
    amplitude_step: float
    rms_mismatch_db: float
    roughness_mismatch: float
    zero_crossing_mismatch: float
    pitch_mismatch_cents: Optional[float]
    clipping_ratio: float
    risk_score: float
    rating: str
    cache_key: str

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "join_index": self.join_index,
            "shared_phone": self.shared_phone,
            "left_diphone": self.left_diphone,
            "right_diphone": self.right_diphone,
            "left_candidate_id": self.left_candidate_id,
            "right_candidate_id": self.right_candidate_id,
            "left_wav": self.left_wav,
            "right_wav": self.right_wav,
            "amplitude_step": self.amplitude_step,
            "rms_mismatch_db": self.rms_mismatch_db,
            "roughness_mismatch": self.roughness_mismatch,
            "zero_crossing_mismatch": self.zero_crossing_mismatch,
            "clipping_ratio": self.clipping_ratio,
            "risk_score": self.risk_score,
            "rating": self.rating,
            "cache_key": self.cache_key,
        }
        if self.pitch_mismatch_cents is not None:
            result["pitch_mismatch_cents"] = self.pitch_mismatch_cents
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "JapaneseJoinMetric":
        return cls(
            join_index=int(value["join_index"]),
            shared_phone=str(value["shared_phone"]),
            left_diphone=str(value["left_diphone"]),
            right_diphone=str(value["right_diphone"]),
            left_candidate_id=str(value["left_candidate_id"]),
            right_candidate_id=str(value["right_candidate_id"]),
            left_wav=str(value["left_wav"]),
            right_wav=str(value["right_wav"]),
            amplitude_step=float(value["amplitude_step"]),
            rms_mismatch_db=float(value["rms_mismatch_db"]),
            roughness_mismatch=float(value["roughness_mismatch"]),
            zero_crossing_mismatch=float(value["zero_crossing_mismatch"]),
            pitch_mismatch_cents=(
                float(value["pitch_mismatch_cents"])
                if value.get("pitch_mismatch_cents") is not None else None
            ),
            clipping_ratio=float(value["clipping_ratio"]),
            risk_score=float(value["risk_score"]),
            rating=str(value["rating"]),
            cache_key=str(value["cache_key"]),
        )


@dataclass(frozen=True)
class JapaneseJoinDiagnostic:
    code: str
    message: str
    join_index: Optional[int] = None
    severity: str = "warning"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.join_index is not None:
            result["join_index"] = self.join_index
        return result


@dataclass(frozen=True)
class JapaneseJoinQualityReport:
    metrics: tuple[JapaneseJoinMetric, ...]
    diagnostics: tuple[JapaneseJoinDiagnostic, ...]
    requested_join_count: int
    cache_hits: int = 0
    schema_version: int = QUALITY_SCHEMA_VERSION
    schema_status: str = QUALITY_SCHEMA_STATUS

    def to_dict(self) -> dict[str, object]:
        ratings = {"good": 0, "review": 0, "poor": 0}
        for metric in self.metrics:
            ratings[metric.rating] = ratings.get(metric.rating, 0) + 1
        return {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "kind": "japanese_generated_voice_join_quality",
            "requested_join_count": self.requested_join_count,
            "analyzed_join_count": len(self.metrics),
            "cache_hits": self.cache_hits,
            "rating_counts": ratings,
            "metrics": [item.to_dict() for item in self.metrics],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


class JapaneseQualityCache:
    """Timestamp-free, content-addressed cache for generated join metrics."""

    def __init__(self, directory: Path | str):
        self.directory = _require_non_source_output(
            directory, "the Japanese quality cache"
        )

    def _path(self, key: str) -> Path:
        return self.directory / "japanese-join-v1" / f"{key}.json"

    def get(self, key: str) -> Optional[JapaneseJoinMetric]:
        try:
            value = json.loads(self._path(key).read_text(encoding="utf-8"))
            if value.get("cache_key") != key:
                return None
            return JapaneseJoinMetric.from_dict(value)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def put(self, metric: JapaneseJoinMetric) -> None:
        target = self._path(metric.cache_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(_json_bytes(metric.to_dict()))
        temporary.replace(target)


def _metric_key(
    left_path: Path,
    right_path: Path,
    left_position: float,
    right_position: float,
    window_seconds: float,
    file_hashes: Optional[dict[Path, str]] = None,
) -> str:
    hashes = file_hashes if file_hashes is not None else {}
    left_resolved = left_path.resolve()
    right_resolved = right_path.resolve()
    if left_resolved not in hashes:
        hashes[left_resolved] = _sha256(left_resolved)
    if right_resolved not in hashes:
        hashes[right_resolved] = _sha256(right_resolved)
    value = {
        "schema": QUALITY_SCHEMA_VERSION,
        "left_sha256": hashes[left_resolved],
        "right_sha256": hashes[right_resolved],
        "left_position": round(left_position, 6),
        "right_position": round(right_position, 6),
        "window_seconds": round(window_seconds, 6),
    }
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def measure_join(
    *,
    join_index: int,
    shared_phone: str,
    left_diphone: str,
    right_diphone: str,
    left_candidate_id: str,
    right_candidate_id: str,
    left_path: Path,
    right_path: Path,
    left_position: float,
    right_position: float,
    generated_voice_root: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    cache: Optional[JapaneseQualityCache] = None,
    file_hashes: Optional[dict[Path, str]] = None,
) -> tuple[JapaneseJoinMetric, bool]:
    """Measure one generated-unit boundary and return ``(metric, cache_hit)``."""
    root = generated_voice_root.resolve()
    left_path = left_path.resolve()
    right_path = right_path.resolve()
    left_name = _relative(left_path, root)
    right_name = _relative(right_path, root)
    key = _metric_key(
        left_path, right_path, left_position, right_position, window_seconds,
        file_hashes,
    )
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached, True

    left, left_rate = _read_window(
        left_path, left_position,
        before_seconds=window_seconds, after_seconds=0.0,
    )
    right, right_rate = _read_window(
        right_path, right_position,
        before_seconds=0.0, after_seconds=window_seconds,
    )
    if left_rate != right_rate:
        raise ValueError(
            f"sample-rate mismatch: {left_rate} versus {right_rate}"
        )
    if not left or not right:
        raise ValueError("join window is empty")

    left_rms = _rms(left)
    right_rms = _rms(right)
    amplitude_step = abs(left[-1] - right[0])
    rms_mismatch = abs(20.0 * math.log10(
        max(left_rms, 1e-9) / max(right_rms, 1e-9)
    ))
    roughness_mismatch = abs(_roughness(left) - _roughness(right))
    zcr_mismatch = abs(_zcr(left) - _zcr(right))
    left_pitch = _pitch_hz(left, left_rate)
    right_pitch = _pitch_hz(right, right_rate)
    pitch_mismatch = None
    if left_pitch is not None and right_pitch is not None:
        pitch_mismatch = abs(1200.0 * math.log2(left_pitch / right_pitch))
    clipped = sum(abs(value) >= 0.999 for value in left + right)
    clipping_ratio = clipped / (len(left) + len(right))

    score = (
        min(1.0, amplitude_step / 0.30) * 26.0
        + min(1.0, rms_mismatch / 18.0) * 22.0
        + min(1.0, roughness_mismatch / 1.25) * 16.0
        + min(1.0, zcr_mismatch / 0.20) * 12.0
        + (min(1.0, pitch_mismatch / 600.0) * 18.0
           if pitch_mismatch is not None else 0.0)
        + min(1.0, clipping_ratio / 0.01) * 30.0
    )
    score = round(min(100.0, score), 3)
    rating = "good" if score < 25.0 else "review" if score < 55.0 else "poor"
    metric = JapaneseJoinMetric(
        join_index=int(join_index),
        shared_phone=str(shared_phone),
        left_diphone=str(left_diphone),
        right_diphone=str(right_diphone),
        left_candidate_id=str(left_candidate_id),
        right_candidate_id=str(right_candidate_id),
        left_wav=left_name,
        right_wav=right_name,
        amplitude_step=round(amplitude_step, 6),
        rms_mismatch_db=round(rms_mismatch, 3),
        roughness_mismatch=round(roughness_mismatch, 6),
        zero_crossing_mismatch=round(zcr_mismatch, 6),
        pitch_mismatch_cents=(
            round(pitch_mismatch, 3) if pitch_mismatch is not None else None
        ),
        clipping_ratio=round(clipping_ratio, 8),
        risk_score=score,
        rating=rating,
        cache_key=key,
    )
    if cache is not None:
        cache.put(metric)
    return metric, False


def _choice_for_edge(
    edge_index: int,
    pair: str,
    runtime: Mapping[str, object],
    selected_units: Mapping[int, str],
    unit_overrides: Mapping[int, str],
    outer_left: str,
    outer_right: str,
) -> Optional[Mapping[str, object]]:
    alternatives = dict(runtime.get("alternatives") or {})
    choices = list(alternatives.get(pair) or ())
    if not choices:
        return None
    wanted = unit_overrides.get(edge_index) or selected_units.get(edge_index)
    if wanted:
        for choice in choices:
            if str(choice.get("left_name") or "") == str(wanted):
                return choice
    return select_automatic_choice(choices, outer_left, outer_right)


def _choice_geometry(
    choice: Mapping[str, object],
    runtime: Mapping[str, object],
    voice_root: Path,
) -> tuple[Path, float, float]:
    index = dict(runtime.get("index") or {})
    row = index.get(str(choice.get("index_name") or ""))
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        raise ValueError("candidate has no generated index geometry")
    path = voice_root / "wav" / str(row[0])
    return path, float(row[1]), float(row[3])


def analyze_plan_joins(
    plan,
    runtime_metadata: Mapping[str, object],
    generated_voice_root: Path | str,
    *,
    selected_units: Mapping[int, str] | None = None,
    cache_directory: Path | str | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> JapaneseJoinQualityReport:
    """Analyze adjacent selected units from an explicit Japanese plan.

    ``generated_voice_root`` must be the generated Festival voice directory.
    Source paths in provenance are intentionally ignored.
    """
    if not 0.005 <= float(window_seconds) <= 0.20:
        raise ValueError("join window must be between 5 and 200 ms")
    root = Path(generated_voice_root).resolve()
    if runtime_metadata.get("language") != "ja":
        raise ValueError("runtime metadata is not Japanese")
    selected = {int(key): str(value) for key, value in dict(
        selected_units or {}
    ).items()}
    overrides = {int(key): str(value) for key, value in dict(
        getattr(plan, "unit_overrides", {}) or {}
    ).items()}
    cache = JapaneseQualityCache(cache_directory) \
        if cache_directory is not None else None
    metrics: list[JapaneseJoinMetric] = []
    diagnostics: list[JapaneseJoinDiagnostic] = []
    hits = 0
    file_hashes: dict[Path, str] = {}
    phones = list(getattr(plan, "phones", ()))
    requested = max(0, len(phones) - 2)
    for join_index in range(requested):
        left_pair = f"{phones[join_index]}-{phones[join_index + 1]}"
        right_pair = f"{phones[join_index + 1]}-{phones[join_index + 2]}"
        left_choice = _choice_for_edge(
            join_index, left_pair, runtime_metadata, selected, overrides,
            phones[join_index - 1] if join_index else "*",
            phones[join_index + 2]
            if join_index + 2 < len(phones) else "*",
        )
        right_choice = _choice_for_edge(
            join_index + 1, right_pair, runtime_metadata, selected, overrides,
            phones[join_index],
            phones[join_index + 3]
            if join_index + 3 < len(phones) else "*",
        )
        if left_choice is None or right_choice is None:
            diagnostics.append(JapaneseJoinDiagnostic(
                code="join_choice_unavailable",
                message=(
                    "No generated candidate geometry is available for one "
                    "side of this join."
                ),
                join_index=join_index,
                severity="info",
            ))
            continue
        try:
            left_path, _left_start, left_end = _choice_geometry(
                left_choice, runtime_metadata, root
            )
            right_path, right_start, _right_end = _choice_geometry(
                right_choice, runtime_metadata, root
            )
            metric, hit = measure_join(
                join_index=join_index,
                shared_phone=phones[join_index + 1],
                left_diphone=left_pair,
                right_diphone=right_pair,
                left_candidate_id=str(
                    left_choice.get("candidate_id") or left_choice.get("id")
                    or "unknown"
                ),
                right_candidate_id=str(
                    right_choice.get("candidate_id") or right_choice.get("id")
                    or "unknown"
                ),
                left_path=left_path,
                right_path=right_path,
                left_position=left_end,
                right_position=right_start,
                generated_voice_root=root,
                window_seconds=float(window_seconds),
                cache=cache,
                file_hashes=file_hashes,
            )
            metrics.append(metric)
            hits += int(hit)
        except (OSError, ValueError, wave.Error) as error:
            diagnostics.append(JapaneseJoinDiagnostic(
                code="join_measurement_failed",
                message=str(error),
                join_index=join_index,
            ))
    return JapaneseJoinQualityReport(
        metrics=tuple(metrics),
        diagnostics=tuple(diagnostics),
        requested_join_count=requested,
        cache_hits=hits,
    )


@dataclass(frozen=True)
class _SerializedPlanView:
    phones: tuple[str, ...]
    unit_overrides: Mapping[int, str]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure joins in a generated Japanese Festival voice."
    )
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("--selected-units", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-ms", type=float, default=40.0)
    args = parser.parse_args(argv)
    from japanese_festival import load_japanese_runtime_metadata

    plan_value = json.loads(args.plan_json.read_text(encoding="utf-8"))
    if plan_value.get("language") != "ja":
        raise ValueError("plan JSON is not Japanese")
    plan = _SerializedPlanView(
        phones=tuple(str(row.get("phone") or "")
                     for row in plan_value.get("segments") or ()),
        unit_overrides={
            int(key): str(value) for key, value in
            dict(plan_value.get("unit_overrides") or {}).items()
        },
    )
    selected = {}
    if args.selected_units:
        selected = {
            int(key): str(value) for key, value in dict(json.loads(
                args.selected_units.read_text(encoding="utf-8")
            )).items()
        }
    report = analyze_plan_joins(
        plan,
        load_japanese_runtime_metadata(args.voice_dir),
        args.voice_dir,
        selected_units=selected,
        cache_directory=args.cache,
        window_seconds=float(args.window_ms) / 1000.0,
    )
    if args.output:
        output = _require_non_source_output(
            args.output, "the Japanese quality report"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(report.to_json_bytes())
    print(report.to_json_bytes().decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
