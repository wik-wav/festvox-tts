# -*- coding: utf-8 -*-
"""Asaxi mora-level voicing and breathiness prediction/rendering.

The linguistic rules are intentionally narrow:

* a vowel may devoice between voiceless obstruents;
* ``x`` after a voiceless consonant is aspirating;
* the documented interjection ``ox`` has a breathy coda.

Breathiness never means replacing the vowel with noise. It lowers the
harmonic share conservatively while retaining the measured vocal-tract
envelope through the shared source-filter renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import asaxi_editing
from source_filter_voicing import curve_for_regions, transform_voicing


VOWEL_PHONES = frozenset(
    "a e i o u aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw".split()
)
VOICELESS_OBSTRUENTS = frozenset(
    "p t k ch ts s sh f h th q cl".split()
)
PAUSE_PHONES = frozenset({"pau", "sil", "#"})
AUTOMATIC_DEVOICED_VOICING = 0.18
BREATHINESS_HARMONIC_DEPTH = 0.48


@dataclass(frozen=True)
class AsaxiMoraPhonation:
    mora_index: int
    phrase_index: int
    text: str
    word: str
    segment_indices: tuple[int, ...]
    vowel_segment_indices: tuple[int, ...]
    start: float | None
    end: float | None
    eligible: bool
    automatic_voicing: float
    automatic_effective_voicing: float
    final_voicing: float
    automatic_breathiness: float
    final_breathiness: float
    voicing_overridden: bool
    breathiness_overridden: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mora_index": self.mora_index,
            "phrase_index": self.phrase_index,
            "text": self.text,
            "word": self.word,
            "segment_indices": list(self.segment_indices),
            "vowel_segment_indices": list(self.vowel_segment_indices),
            "start": self.start,
            "end": self.end,
            "eligible": self.eligible,
            "automatic_voicing": self.automatic_voicing,
            "automatic_effective_voicing": self.automatic_effective_voicing,
            "final_voicing": self.final_voicing,
            "automatic_breathiness": self.automatic_breathiness,
            "final_breathiness": self.final_breathiness,
            "voicing_overridden": self.voicing_overridden,
            "breathiness_overridden": self.breathiness_overridden,
            "reasons": list(self.reasons),
        }


def _override_map(value: object) -> dict[int, float]:
    result: dict[int, float] = {}
    if not isinstance(value, Mapping):
        return result
    for key, amount in value.items():
        try:
            result[int(key)] = max(0.0, min(1.0, float(amount)))
        except (TypeError, ValueError):
            continue
    return result


def _segment_phone(segment) -> str:
    return str(segment.phone if hasattr(segment, "phone") else segment[0])


def _segment_times(segment) -> tuple[float, float]:
    if hasattr(segment, "start"):
        return float(segment.start), float(segment.end)
    return float(segment[1]), float(segment[2])


def _neighbor_phone(segments, index: int, direction: int) -> str:
    cursor = int(index) + int(direction)
    while 0 <= cursor < len(segments):
        phone = _segment_phone(segments[cursor])
        if phone not in PAUSE_PHONES:
            return phone
        return ""
    return ""


def predict_mora_phonation(
    metadata: Mapping[str, object],
    segments,
    *,
    voicing_overrides: Mapping[object, object] | None = None,
    breathiness_overrides: Mapping[object, object] | None = None,
) -> tuple[AsaxiMoraPhonation, ...]:
    """Return deterministic language decisions on the rendered timeline."""

    segment_rows = list(segments or [])
    voicing = _override_map(voicing_overrides)
    breathiness = _override_map(breathiness_overrides)
    result: list[AsaxiMoraPhonation] = []
    for position, row in enumerate(asaxi_editing.mora_rows(metadata)):
        try:
            mora_index = int(row.get("mora_index", position))
            phrase_index = int(row.get("phrase_index", 0))
        except (TypeError, ValueError):
            continue
        indices = []
        for value in row.get("segment_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(segment_rows):
                indices.append(index)
        indices = sorted(set(indices))
        vowel_indices = [
            index for index in indices
            if _segment_phone(segment_rows[index]) in VOWEL_PHONES
        ]
        reasons: list[str] = []
        automatic_voicing = 1.0
        automatic_breathiness = 0.0
        if vowel_indices:
            left = _neighbor_phone(segment_rows, vowel_indices[0], -1)
            right = _neighbor_phone(segment_rows, vowel_indices[-1], 1)
            if left in VOICELESS_OBSTRUENTS and \
                    right in VOICELESS_OBSTRUENTS:
                automatic_voicing = AUTOMATIC_DEVOICED_VOICING
                reasons.append(
                    f"vowel between voiceless {left} and {right}"
                )

            phones = tuple(str(phone) for phone in row.get("phones") or [])
            text = str(row.get("text") or "")
            word = str(row.get("word") or "")
            if "hh" in phones:
                hh_position = phones.index("hh")
                preceding = phones[hh_position - 1] if hh_position else left
                if preceding in VOICELESS_OBSTRUENTS:
                    automatic_breathiness = max(
                        automatic_breathiness, 0.55
                    )
                    reasons.append("x marks aspiration after a voiceless phone")
                else:
                    automatic_breathiness = max(
                        automatic_breathiness, 0.22
                    )
                    reasons.append("voiced glottal x adds light breathiness")
            if word == "ox" or (text.endswith("x") and word == "ox"):
                automatic_breathiness = max(
                    automatic_breathiness, 0.72
                )
                reasons.append("ox has a documented breathy-sigh coda")
        else:
            text = str(row.get("text") or "")
            word = str(row.get("word") or "")
            reasons.append("mora has no aligned vowel-bearing segment")

        final_breathiness = breathiness.get(mora_index, automatic_breathiness)
        breathy_ceiling = (
            1.0 - final_breathiness * BREATHINESS_HARMONIC_DEPTH
        )
        automatic_effective_voicing = min(
            automatic_voicing,
            1.0 - automatic_breathiness * BREATHINESS_HARMONIC_DEPTH,
        )
        # Mora voicing is the one user-facing phonation authority. Legacy
        # breathiness values are migrated into this scalar by asaxi_editing.
        final_voicing = voicing.get(
            mora_index, min(automatic_voicing, breathy_ceiling))
        if vowel_indices:
            starts_ends = [
                _segment_times(segment_rows[index])
                for index in vowel_indices
            ]
            start = min(item[0] for item in starts_ends)
            end = max(item[1] for item in starts_ends)
        else:
            start = end = None
        result.append(AsaxiMoraPhonation(
            mora_index=mora_index,
            phrase_index=phrase_index,
            text=text,
            word=word,
            segment_indices=tuple(indices),
            vowel_segment_indices=tuple(vowel_indices),
            start=start,
            end=end,
            eligible=bool(vowel_indices),
            automatic_voicing=round(automatic_voicing, 6),
            automatic_effective_voicing=round(
                max(0.0, min(1.0, automatic_effective_voicing)), 6),
            final_voicing=round(max(0.0, min(1.0, final_voicing)), 6),
            automatic_breathiness=round(automatic_breathiness, 6),
            final_breathiness=round(final_breathiness, 6),
            voicing_overridden=mora_index in voicing,
            breathiness_overridden=mora_index in breathiness,
            reasons=tuple(reasons),
        ))
    return tuple(result)


def _regions(predictions, *, automatic: bool) -> list[dict[str, float]]:
    result = []
    for item in predictions:
        if not item.eligible or item.start is None or item.end is None:
            continue
        if automatic:
            value = item.automatic_effective_voicing
        else:
            value = item.final_voicing
        result.append({
            "start": float(item.start),
            "end": float(item.end),
            "target_voicing": max(0.0, min(1.0, float(value))),
        })
    return result


def mora_voicing_curve(
    metadata: Mapping[str, object],
    segments,
    source_curve: Sequence[Sequence[float]],
    *,
    voicing_overrides: Mapping[object, object] | None = None,
) -> tuple[list[tuple[float, float]], tuple[AsaxiMoraPhonation, ...]]:
    """Preview the effective block edits without re-analyzing the waveform."""

    predictions = predict_mora_phonation(
        metadata,
        segments,
        voicing_overrides=voicing_overrides,
    )
    return (
        curve_for_regions(
            source_curve, _regions(predictions, automatic=False)),
        predictions,
    )


def apply_phonation(
    synthesis,
    metadata: Mapping[str, object],
    *,
    voicing_overrides: Mapping[object, object] | None = None,
    breathiness_overrides: Mapping[object, object] | None = None,
    continuous_voicing_override: Sequence[Sequence[float]] | None = None,
):
    """Apply Asaxi defaults and mora overlays through the shared renderer."""

    source = np.asarray(synthesis.samples, np.float32).copy()
    analysis = transform_voicing(source, synthesis.sr)
    predictions = predict_mora_phonation(
        metadata,
        synthesis.segments,
        voicing_overrides=voicing_overrides,
        breathiness_overrides=breathiness_overrides,
    )
    generated_curve = curve_for_regions(
        analysis.source_curve, _regions(predictions, automatic=True)
    )
    mora_curve = curve_for_regions(
        analysis.source_curve, _regions(predictions, automatic=False)
    )
    if continuous_voicing_override:
        final_curve = [
            (float(time), float(value))
            for time, value in continuous_voicing_override
        ]
        mode = "curve"
    else:
        final_curve = mora_curve
        mode = ""
    result = transform_voicing(source, synthesis.sr, final_curve)
    synthesis.samples = result.samples
    synthesis.source_voicing_targets = list(analysis.source_curve)
    # The dashed GUI curve is the generated linguistic baseline after block
    # edits. The untouched source analysis remains separately available.
    synthesis.generated_voicing_targets = list(mora_curve)
    synthesis.voicing_override = (
        list(final_curve) if continuous_voicing_override else []
    )
    synthesis.voicing_mode = mode
    synthesis.voicing_diagnostics = [
        {
            "kind": "asaxi_mora_phonation",
            "automatic_curve": list(generated_curve),
            "predictions": [item.to_dict() for item in predictions],
        },
        result.diagnostic_dict(include_frames=False),
    ]
    synthesis.vowel_realizations = [
        {
            "language": "asaxi",
            "mora_index": item.mora_index,
            "segment_indices": list(item.vowel_segment_indices),
            "automatic_voicing": item.automatic_voicing,
            "final_voicing": item.final_voicing,
            "automatic_breathiness": item.automatic_breathiness,
            "final_breathiness": item.final_breathiness,
            "reasons": list(item.reasons),
        }
        for item in predictions if item.eligible
    ]
    enriched = dict(metadata or {})
    enriched["mora_phonation_predictions"] = [
        item.to_dict() for item in predictions
    ]
    synthesis.asaxi_prosody = enriched
    return synthesis
