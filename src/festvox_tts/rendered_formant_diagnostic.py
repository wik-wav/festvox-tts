"""Final-waveform formant tracks and potential discontinuity flags.

This module is diagnostic only.  It never changes rendered samples.  Formant
tracks are estimated separately inside phone spans so analysis windows do not
cross known phone boundaries.  Exact splice-local evidence from the existing
join analyzer is retained when available; otherwise boundary flags are marked
as estimated and should be interpreted as prompts for inspection, not proof of
a faulty join.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from formant_analysis import (
    AnalysisSegment,
    AudioData,
    FormantAnalysisConfig,
    analyze_segment,
)


RENDERED_FORMANT_DIAGNOSTIC_VERSION = "prompt20-rendered-formants-v1"


@dataclass(frozen=True)
class RenderedFormantDiagnosticConfig:
    frame_step_seconds: float = 0.010
    minimum_phone_seconds: float = 0.045
    maximum_total_frames: int = 3000
    internal_jump_cents: float = 280.0
    boundary_jump_cents: float = 350.0
    novelty_threshold: float = 3.5
    exact_join_fraction_threshold: float = 0.12


def _segment_fields(segment, index: int) -> tuple[str, float, float]:
    if isinstance(segment, Mapping):
        phone = segment.get("phone", segment.get("name", "unknown"))
        start = segment.get("start", 0.0)
        end = segment.get("end", start)
    else:
        phone = getattr(segment, "phone", getattr(segment, "name", "unknown"))
        start = getattr(segment, "start", 0.0)
        end = getattr(segment, "end", start)
    return str(phone or f"phone_{index}"), float(start), float(end)


def _finite_track(frame, index: int) -> float | None:
    values = frame.tracked_formants_hz or frame.formants_hz
    if index >= len(values):
        return None
    value = values[index]
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return None
    return float(value)


def _cents(left: float, right: float) -> float:
    return 1200.0 * math.log2(float(right) / float(left))


def _robust_novelty(values: Sequence[float]) -> list[float]:
    data = np.asarray(values, np.float64)
    if not data.size:
        return []
    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    scale = max(30.0, 1.4826 * mad)
    return [max(0.0, (float(value) - center) / scale) for value in data]


def _frame_row(frame) -> dict[str, object]:
    return {
        "time": float(frame.frame_time_seconds),
        "f0_hz": frame.f0_hz,
        "voicing_confidence": float(frame.voicing_confidence),
        "formants_hz": [
            _finite_track(frame, index) for index in range(4)
        ],
        "tracking_confidences": [
            float(frame.tracking_confidences[index])
            if index < len(frame.tracking_confidences) else 0.0
            for index in range(4)
        ],
        "accepted": bool(frame.accepted),
        "rejection_reasons": list(frame.rejection_reasons),
    }


def _jump_details(left, right) -> tuple[list[dict[str, float]], float]:
    details = []
    maximum = 0.0
    for index in range(4):
        first = _finite_track(left, index)
        second = _finite_track(right, index)
        if first is None or second is None:
            continue
        delta = _cents(first, second)
        maximum = max(maximum, abs(delta))
        details.append({
            "formant": index + 1,
            "left_hz": first,
            "right_hz": second,
            "delta_cents": delta,
        })
    return details, maximum


def analyze_rendered_formants(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    segments: Sequence[object],
    *,
    join_diagnostic: Mapping[str, object] | None = None,
    config: RenderedFormantDiagnosticConfig | None = None,
    analysis_config: FormantAnalysisConfig | None = None,
) -> dict[str, object]:
    """Analyze the final rendered waveform without modifying it."""
    cfg = config or RenderedFormantDiagnosticConfig()
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    audio_samples = np.asarray(samples, np.float64).reshape(-1)
    if not np.all(np.isfinite(audio_samples)):
        raise ValueError("rendered samples contain non-finite values")
    duration = audio_samples.size / float(sr)
    adaptive_step = max(
        float(cfg.frame_step_seconds),
        duration / max(1, int(cfg.maximum_total_frames)),
    )
    base_analysis = analysis_config or FormantAnalysisConfig()
    formant_config = replace(
        base_analysis,
        frame_step_seconds=adaptive_step,
        stable_trim_fraction=0.0,
        maximum_frames_per_segment=max(
            base_analysis.maximum_frames_per_segment, 240
        ),
    )
    audio = AudioData(
        samples=audio_samples,
        sample_rate=sr,
        channels=1,
        source_path=Path("rendered-result.wav"),
    )
    phone_rows = []
    analyzed = {}
    accepted_frame_count = 0
    rejected_frame_count = 0
    for index, segment in enumerate(segments):
        phone, raw_start, raw_end = _segment_fields(segment, index)
        start = max(0.0, min(duration, raw_start))
        end = max(start, min(duration, raw_end))
        row = {
            "index": index,
            "phone": phone,
            "start": start,
            "end": end,
            "analyzed": False,
            "frames": [],
            "rejection_reasons": [],
        }
        if phone.casefold() in {"pau", "sil", "sp"}:
            row["rejection_reasons"] = ["pause_or_silence"]
            phone_rows.append(row)
            continue
        if end - start < float(cfg.minimum_phone_seconds):
            row["rejection_reasons"] = ["phone_too_short_for_contained_window"]
            phone_rows.append(row)
            continue
        analysis_segment = AnalysisSegment(
            segment_id=f"rendered:{index}",
            speaker_id="rendered_voice",
            audio_path=Path("rendered-result.wav"),
            start_seconds=start,
            end_seconds=end,
            vowel=phone if phone.casefold() in {"a", "i", "u", "e", "o"}
            else "unknown",
            phone=phone,
            partition="rendered",
            recording_style="synthesized_result",
            source_corpus="current_project",
        )
        result = analyze_segment(
            analysis_segment, audio=audio, config=formant_config
        )
        row["analyzed"] = True
        row["frames"] = [_frame_row(frame) for frame in result.frames]
        row["accepted_frame_count"] = result.accepted_frame_count
        row["rejected_frame_count"] = result.rejected_frame_count
        row["rejection_reasons"] = list(result.rejection_reasons)
        phone_rows.append(row)
        analyzed[index] = result
        accepted_frame_count += result.accepted_frame_count
        rejected_frame_count += result.rejected_frame_count

    jumps = []
    for phone_index, result in analyzed.items():
        frames = [frame for frame in result.frames if frame.accepted]
        if len(frames) < 2:
            continue
        pairs = list(zip(frames, frames[1:]))
        pair_rows = [_jump_details(left, right) for left, right in pairs]
        novelties = _robust_novelty([maximum for _details, maximum in pair_rows])
        for (left, right), (details, maximum), novelty in zip(
                pairs, pair_rows, novelties):
            if not details or maximum < cfg.internal_jump_cents:
                continue
            if novelty < cfg.novelty_threshold and maximum < (
                    cfg.internal_jump_cents * 2.0):
                continue
            jumps.append({
                "time": (left.frame_time_seconds + right.frame_time_seconds) / 2.0,
                "kind": "INTERNAL_FORMANT_JUMP",
                "phone_index": phone_index,
                "left_phone": phone_rows[phone_index]["phone"],
                "right_phone": phone_rows[phone_index]["phone"],
                "boundary_index": None,
                "exact_splice_evidence": False,
                "max_delta_cents": maximum,
                "novelty": novelty,
                "severity": max(
                    maximum / cfg.internal_jump_cents,
                    novelty / cfg.novelty_threshold,
                ),
                "formants": details,
                "interpretation": "potential abrupt within-phone trajectory",
            })

    exact_boundaries = set()
    for join in (join_diagnostic or {}).get("joins", ()):
        if not join.get("formants_available"):
            continue
        fraction = join.get("formant_frequency_jump_normalized")
        novelty = join.get("formant_frequency_jump_novelty")
        fraction_value = float(fraction) if fraction is not None else 0.0
        novelty_value = float(novelty) if novelty is not None else 0.0
        if (fraction_value < cfg.exact_join_fraction_threshold and
                novelty_value < cfg.novelty_threshold):
            continue
        boundary_index = int(join.get("segment_index", 0))
        exact_boundaries.add(boundary_index)
        left_phone = str(join.get("left_phone") or
                         join.get("phone_before") or "?")
        right_phone = str(join.get("right_phone") or
                          join.get("phone_after") or join.get("phone") or "?")
        jumps.append({
            "time": float(join.get("time") or 0.0),
            "kind": "EXACT_SPLICE_FORMANT_JUMP",
            "phone_index": boundary_index,
            "left_phone": left_phone,
            "right_phone": right_phone,
            "boundary_index": boundary_index,
            "exact_splice_evidence": True,
            "max_delta_cents": None,
            "frequency_jump_fraction": fraction_value,
            "novelty": novelty_value,
            "severity": max(
                fraction_value / cfg.exact_join_fraction_threshold,
                novelty_value / cfg.novelty_threshold,
            ),
            "formants": list(join.get("formant_tracks") or ()),
            "interpretation": "potential exact rendered-splice formant jump",
        })

    for right_index in range(1, len(phone_rows)):
        if right_index in exact_boundaries:
            continue
        left_result = analyzed.get(right_index - 1)
        right_result = analyzed.get(right_index)
        if left_result is None or right_result is None:
            continue
        left_frames = [frame for frame in left_result.frames if frame.accepted]
        right_frames = [frame for frame in right_result.frames if frame.accepted]
        if not left_frames or not right_frames:
            continue
        details, maximum = _jump_details(left_frames[-1], right_frames[0])
        if not details or maximum < cfg.boundary_jump_cents:
            continue
        boundary_time = float(phone_rows[right_index]["start"])
        jumps.append({
            "time": boundary_time,
            "kind": "ESTIMATED_PHONE_BOUNDARY_FORMANT_JUMP",
            "phone_index": right_index,
            "left_phone": phone_rows[right_index - 1]["phone"],
            "right_phone": phone_rows[right_index]["phone"],
            "boundary_index": right_index,
            "exact_splice_evidence": False,
            "max_delta_cents": maximum,
            "novelty": None,
            "severity": maximum / cfg.boundary_jump_cents,
            "formants": details,
            "interpretation": (
                "potential phone-boundary jump; windows are contained on "
                "their respective sides"
            ),
        })

    jumps.sort(key=lambda row: (-float(row["severity"]), float(row["time"])))
    for rank, row in enumerate(jumps, 1):
        row["rank"] = rank
    return {
        "kind": "rendered_formant_diagnostic",
        "version": RENDERED_FORMANT_DIAGNOSTIC_VERSION,
        "sample_rate": sr,
        "duration_seconds": duration,
        "analysis_frame_step_seconds": adaptive_step,
        "accepted_frame_count": accepted_frame_count,
        "rejected_frame_count": rejected_frame_count,
        "phone_count": len(phone_rows),
        "analyzed_phone_count": len(analyzed),
        "potential_jump_count": len(jumps),
        "phones": phone_rows,
        "jumps": jumps,
        "interpretation": {
            "tracks": "confidence-filtered formants measured from final audio",
            "jumps": "potential issues for inspection, not proof of a bad join",
            "spectrogram": "frequency-smoothed display only; audio is unchanged",
        },
    }
