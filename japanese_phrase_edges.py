"""Read-only acoustic checks for Japanese phrase-edge timing.

Festival segment boundaries describe the requested linguistic timeline.  A
pause-to-vowel diphone can nevertheless contain audible vowel energy before
that boundary, and a phrase-final unit can continue into the following pause.
This module measures that acoustic extension without changing unit choices or
waveforms.  Source and synthesis must be measured with the same settings; the
difference between them is the useful perceptual-timing diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PhraseEdgeConfig:
    frame_ms: float = 8.0
    hop_ms: float = 2.0
    context_ms: float = 140.0
    search_ms: float = 85.0
    guard_ms: float = 6.0
    sustained_ms: float = 10.0
    threshold_fraction: float = 0.34
    minimum_contrast_db: float = 5.0


@dataclass(frozen=True)
class AcousticPhraseEdge:
    edge: str
    logical_boundary_seconds: float
    acoustic_boundary_seconds: float | None
    extension_ms: float | None
    pause_rms_db: float | None
    active_rms_db: float | None
    contrast_db: float | None
    threshold_db: float | None
    confidence: float
    available: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        def rounded(value, digits=6):
            return round(float(value), digits) if value is not None else None

        return {
            "edge": self.edge,
            "logical_boundary_seconds": rounded(
                self.logical_boundary_seconds),
            "acoustic_boundary_seconds": rounded(
                self.acoustic_boundary_seconds),
            "extension_ms": rounded(self.extension_ms, 3),
            "pause_rms_db": rounded(self.pause_rms_db, 3),
            "active_rms_db": rounded(self.active_rms_db, 3),
            "contrast_db": rounded(self.contrast_db, 3),
            "threshold_db": rounded(self.threshold_db, 3),
            "confidence": rounded(self.confidence, 4),
            "available": self.available,
            "reason": self.reason,
        }


def _frame_rms(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    config: PhraseEdgeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0 or sample_rate <= 0:
        return np.zeros(0), np.zeros(0)
    frame = max(4, int(round(config.frame_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(config.hop_ms * sample_rate / 1000.0)))
    first = max(0, int(math.floor(start_seconds * sample_rate)))
    last = min(values.size, int(math.ceil(end_seconds * sample_rate)))
    if last - first < frame:
        return np.zeros(0), np.zeros(0)
    offsets = np.arange(first, last - frame + 1, hop, dtype=np.int64)
    times = (offsets + frame / 2.0) / float(sample_rate)
    rms = np.empty(offsets.size, dtype=np.float64)
    window = np.hanning(frame)
    normalization = max(1e-12, float(np.sum(window * window)))
    for index, offset in enumerate(offsets):
        chunk = values[offset:offset + frame]
        rms[index] = math.sqrt(
            max(0.0, float(np.sum(np.square(chunk * window))) / normalization)
        )
    return times, 20.0 * np.log10(np.maximum(rms, 1e-9))


def _runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return ()
    result = []
    start = previous = int(indexes[0])
    for raw in indexes[1:]:
        value = int(raw)
        if value != previous + 1:
            result.append((start, previous + 1))
            start = value
        previous = value
    result.append((start, previous + 1))
    return tuple(result)


def detect_acoustic_phrase_edge(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    logical_boundary_seconds: float,
    *,
    edge: str,
    config: PhraseEdgeConfig | None = None,
) -> AcousticPhraseEdge:
    """Locate sustained activity crossing a phrase-initial/final boundary.

    ``extension_ms`` is positive when synthesis extends beyond its logical
    region: activity before an initial boundary or after a final boundary.
    Negative values indicate delayed onset or early offset.  Pitch periods are
    intentionally not used, so vowel, consonant, and devoiced edges share one
    deterministic measurement.
    """
    if edge not in {"initial", "final"}:
        raise ValueError("edge must be 'initial' or 'final'")
    settings = config or PhraseEdgeConfig()
    boundary = float(logical_boundary_seconds)
    context = settings.context_ms / 1000.0
    guard = settings.guard_ms / 1000.0
    times, levels = _frame_rms(
        samples, sample_rate, boundary - context, boundary + context,
        settings,
    )
    unavailable = lambda reason: AcousticPhraseEdge(
        edge=edge,
        logical_boundary_seconds=boundary,
        acoustic_boundary_seconds=None,
        extension_ms=None,
        pause_rms_db=None,
        active_rms_db=None,
        contrast_db=None,
        threshold_db=None,
        confidence=0.0,
        available=False,
        reason=reason,
    )
    if times.size < 4:
        return unavailable("insufficient_samples")
    if edge == "initial":
        pause_values = levels[times <= boundary - guard]
        active_values = levels[times >= boundary + guard]
    else:
        active_values = levels[times <= boundary - guard]
        pause_values = levels[times >= boundary + guard]
    if pause_values.size < 2 or active_values.size < 2:
        return unavailable("insufficient_two_sided_context")
    pause_db = float(np.percentile(pause_values, 50.0))
    active_db = float(np.percentile(active_values, 75.0))
    contrast = active_db - pause_db
    if not math.isfinite(contrast) or contrast < settings.minimum_contrast_db:
        row = unavailable("insufficient_speech_pause_contrast")
        return AcousticPhraseEdge(
            **{**row.__dict__, "pause_rms_db": pause_db,
               "active_rms_db": active_db, "contrast_db": contrast}
        )
    threshold = pause_db + max(
        3.0, settings.threshold_fraction * contrast)
    threshold = min(threshold, active_db - 1.5)
    search = settings.search_ms / 1000.0
    relevant = (times >= boundary - search) & (times <= boundary + search)
    runs = _runs((levels >= threshold) & relevant)
    minimum_frames = max(
        1, int(math.ceil(settings.sustained_ms / settings.hop_ms)))
    candidates = []
    for start, end in runs:
        if end - start < minimum_frames:
            continue
        run_times = times[start:end]
        # The activity must reach the logical boundary. This rejects the tail
        # of the previous/next phrase when the pause itself is short.
        if edge == "initial" and run_times[-1] < boundary:
            continue
        if edge == "final" and run_times[0] > boundary:
            continue
        candidates.append((start, end))
    if not candidates:
        row = unavailable("no_sustained_activity_crossing_boundary")
        return AcousticPhraseEdge(
            **{**row.__dict__, "pause_rms_db": pause_db,
               "active_rms_db": active_db, "contrast_db": contrast,
               "threshold_db": threshold}
        )
    frame_half = settings.frame_ms / 2000.0
    if edge == "initial":
        start, _end = min(candidates, key=lambda item: item[0])
        acoustic = float(times[start] - frame_half)
        extension_ms = (boundary - acoustic) * 1000.0
    else:
        _start, end = max(candidates, key=lambda item: item[1])
        acoustic = float(times[end - 1] + frame_half)
        extension_ms = (acoustic - boundary) * 1000.0
    confidence = min(1.0, max(0.0, (contrast - 3.0) / 24.0))
    return AcousticPhraseEdge(
        edge=edge,
        logical_boundary_seconds=boundary,
        acoustic_boundary_seconds=acoustic,
        extension_ms=extension_ms,
        pause_rms_db=pause_db,
        active_rms_db=active_db,
        contrast_db=contrast,
        threshold_db=threshold,
        confidence=confidence,
        available=True,
    )


def compare_phrase_edges(
    source: AcousticPhraseEdge,
    synthesis: AcousticPhraseEdge,
) -> dict[str, object]:
    excess = None
    if source.extension_ms is not None and synthesis.extension_ms is not None:
        excess = synthesis.extension_ms - source.extension_ms
    return {
        "edge": source.edge,
        "source": source.to_dict(),
        "synthesis": synthesis.to_dict(),
        "synthesis_excess_extension_ms": (
            round(float(excess), 3) if excess is not None else None
        ),
        "interpretation": (
            "positive means the rendered speech extends farther into its "
            "pause than the matched source phrase"
        ),
    }
