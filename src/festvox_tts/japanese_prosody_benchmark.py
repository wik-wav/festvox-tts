"""Held-out final-plan timing and pitch audit for Japanese synthesis.

This benchmark deliberately evaluates the output of ``create_synthesis_plan``
rather than the abstract duration predictor.  It therefore catches routing,
bank-phone mapping, OTO-geometry, source-safety, and model-selection mistakes
that a model-only score cannot see.  Kokoro boundaries are silver references,
not manually corrected phonetic labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from formant_analysis import estimate_f0, read_audio
from japanese_duration import phone_class
from japanese_duration_corpus import align_phone_sequences, normalize_label_phone
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
from japanese_models import JapaneseUtterance
from japanese_synthesis import JapaneseSynthesisPlan, create_synthesis_plan
from kokoro_reference import (
    SilverPhoneAlignment,
    SilverUtteranceAlignment,
    load_alignment,
    load_selection,
    refine_phrase_pauses,
)


SCHEMA_VERSION = 1
_SILENCE = {"sil", "pau", "sp"}
_VOWELS = {"a", "i", "u", "e", "o"}
_COLORS = {
    "vowel": "#3f78b5",
    "stop": "#c95b4b",
    "fricative": "#a66ab0",
    "affricate": "#d38b35",
    "nasal": "#4f9567",
    "approximant": "#5c8e9e",
    "moraic_nasal": "#42836a",
    "geminate_closure": "#8a6a52",
    "other": "#777777",
}
LEGACY_PITCH_MODEL_ID = "speaker_relative_phrase_accent_v2"


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    finite = np.asarray([
        float(value) for value in values if math.isfinite(float(value))
    ], dtype=np.float64)
    return float(np.percentile(finite, percentile)) if finite.size else None


@dataclass(frozen=True)
class PlannedPhoneTiming:
    phone: str
    start_seconds: float
    end_seconds: float
    source_phone_index: int
    mora_index: int
    phrase_index: int
    phone_class: str
    phenomena: tuple[str, ...] = ()
    timing_decisions: tuple[Mapping[str, object], ...] = ()

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class AlignedPlanPhone:
    utterance_id: str
    partition: str
    reference_index: int
    plan_index: int
    phone: str
    phone_class: str
    mora_index: int
    phrase_index: int
    reference_start_seconds: float
    reference_end_seconds: float
    predicted_start_seconds: float
    predicted_end_seconds: float
    reference_duration_seconds: float
    predicted_duration_seconds: float
    signed_duration_error_seconds: float
    absolute_duration_error_seconds: float
    duration_ratio: float
    boundary_drift_seconds: float
    reference_confidence: float
    reference_rejected: bool
    phenomena: tuple[str, ...]
    timing_decisions: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "utterance_id": self.utterance_id,
            "partition": self.partition,
            "reference_index": self.reference_index,
            "plan_index": self.plan_index,
            "phone": self.phone,
            "phone_class": self.phone_class,
            "mora_index": self.mora_index,
            "phrase_index": self.phrase_index,
            "reference_start_seconds": round(self.reference_start_seconds, 6),
            "reference_end_seconds": round(self.reference_end_seconds, 6),
            "predicted_start_seconds": round(self.predicted_start_seconds, 6),
            "predicted_end_seconds": round(self.predicted_end_seconds, 6),
            "reference_duration_seconds": round(
                self.reference_duration_seconds, 6),
            "predicted_duration_seconds": round(
                self.predicted_duration_seconds, 6),
            "signed_duration_error_ms": round(
                self.signed_duration_error_seconds * 1000.0, 6),
            "absolute_duration_error_ms": round(
                self.absolute_duration_error_seconds * 1000.0, 6),
            "duration_ratio": round(self.duration_ratio, 6),
            "boundary_drift_ms": round(self.boundary_drift_seconds * 1000.0, 6),
            "reference_confidence": round(self.reference_confidence, 6),
            "reference_rejected": self.reference_rejected,
            "phenomena": list(self.phenomena),
            "timing_decisions": [dict(item) for item in
                                 self.timing_decisions],
        }


def _mora_phenomena(utterance: JapaneseUtterance) -> dict[int, tuple[str, ...]]:
    result = {}
    phrase_moras = {
        phrase.index: tuple(phrase.moras) for phrase in utterance.phrases
    }
    for mora in utterance.moras:
        names = []
        if mora.special_mora:
            names.append(str(mora.special_mora))
        if mora.devoiced:
            names.append("devoiced_high_vowel")
        members = phrase_moras.get(mora.phrase_index, ())
        position = next((
            index for index, item in enumerate(members)
            if item.index == mora.index
        ), None)
        vowel_only = bool(
            mora.vowel and not mora.consonant
            and mora.special_mora not in {"moraic_nasal", "geminate"}
        )
        if position == 0:
            names.append("phrase_initial")
            if vowel_only:
                names.append("phrase_initial_vowel")
        if position is not None and position == len(members) - 1:
            names.append("phrase_final")
            if vowel_only:
                names.append("phrase_final_vowel")
        morphology = dict(mora.provenance.get("morphology") or {})
        role = str(morphology.get("grammatical_role") or "")
        if role:
            names.append(f"grammar:{role}")
        node_position = morphology.get("mora_position_in_node_zero_based")
        node_count = morphology.get("mora_count_in_node")
        if node_position == 0:
            names.append("grammatical_node_initial")
        if (node_position is not None and node_count is not None
                and int(node_position) == int(node_count) - 1):
            names.append("grammatical_node_final")
        result[mora.index] = tuple(sorted(set(names)))
    return result


def final_plan_phone_timings(
    utterance: JapaneseUtterance,
    plan: JapaneseSynthesisPlan,
) -> tuple[PlannedPhoneTiming, ...]:
    """Recover canonical linguistic phones from the final bank-phone plan."""
    phone_by_index = {phone.index: phone for phone in utterance.phones}
    timing_by_segment = {
        allocation.segment_index: allocation
        for mora in plan.mora_timings
        for allocation in mora.phone_allocation
    }
    phenomena = _mora_phenomena(utterance)
    rows: list[PlannedPhoneTiming] = []
    cursor = 0.0
    for segment in plan.segments:
        start = cursor
        cursor += float(segment.duration)
        source_index = segment.source_phone_index
        if source_index is None or source_index not in phone_by_index:
            continue
        source = phone_by_index[source_index]
        canonical = str(source.symbol)
        if canonical in _SILENCE or segment.phone == "pau":
            continue
        mora_index = int(segment.mora_index if segment.mora_index is not None
                         else source.mora_index if source.mora_index is not None
                         else -1)
        phrase_index = int(segment.phrase_index if segment.phrase_index is not None
                           else source.phrase_index if source.phrase_index is not None
                           else -1)
        timing = timing_by_segment.get(segment.index)
        timing_decisions = ()
        if timing is not None:
            timing_decisions = ({
                "segment_index": timing.segment_index,
                "rendered_phone": segment.phone,
                "duration_model": timing.duration_model,
                "duration_model_id": timing.duration_model_id,
                "baseline_source": timing.baseline_source,
                "context_log_ratio": timing.context_log_ratio,
                "context_effects": dict(sorted(
                    timing.context_effects.items())),
                "source_geometry_ratio": timing.source_geometry_ratio,
                "source_geometry_ratio_bounded": (
                    timing.source_geometry_ratio_bounded
                ),
                "source_safe_min": timing.source_safe_min,
                "source_safe_max": timing.source_safe_max,
                "constraint_source": timing.constraint_source,
                "requested_stretch": timing.requested_stretch,
                "final_duration": timing.final_duration,
            },)
        row = PlannedPhoneTiming(
            phone=canonical,
            start_seconds=start,
            end_seconds=cursor,
            source_phone_index=int(source_index),
            mora_index=mora_index,
            phrase_index=phrase_index,
            phone_class=(
                "moraic_nasal" if segment.timing_role == "moraic_nasal"
                else "geminate_closure" if segment.timing_role == "geminate"
                else phone_class(canonical)
            ),
            phenomena=phenomena.get(mora_index, ()),
            timing_decisions=timing_decisions,
        )
        if rows and rows[-1].source_phone_index == row.source_phone_index:
            rows[-1] = replace(
                rows[-1],
                end_seconds=row.end_seconds,
                timing_decisions=(
                    rows[-1].timing_decisions + row.timing_decisions
                ),
            )
            continue
        # Kokoro writes a long vowel as one ``o:`` interval while Open JTalk
        # represents it as two mora phones. Merge only that explicit second
        # long-vowel mora; ordinary adjacent equal vowels remain distinct.
        if (rows and "long_vowel" in row.phenomena
                and rows[-1].phone == row.phone
                and rows[-1].end_seconds <= row.start_seconds + 1e-9):
            merged = tuple(sorted(set(rows[-1].phenomena + row.phenomena)))
            rows[-1] = replace(
                rows[-1], end_seconds=row.end_seconds,
                phenomena=merged,
                timing_decisions=(
                    rows[-1].timing_decisions + row.timing_decisions
                ),
            )
            continue
        rows.append(row)
    return tuple(rows)


def _reference_phones(
    alignment: SilverUtteranceAlignment,
) -> tuple[SilverPhoneAlignment, ...]:
    return tuple(phone for phone in alignment.phones
                 if normalize_label_phone(phone.phone)[0] not in _SILENCE)


def normalize_kokoro_frontend_text(text: str) -> str:
    """Remove corpus token spacing before asking Open JTalk for prosody.

    Kokoro metadata separates Japanese lexical tokens with ASCII spaces.
    Open JTalk treats those spaces as pause boundaries, while ordinary written
    Japanese does not contain them.  Keeping the spaces would benchmark an
    artificial phrase reset after almost every word.
    """
    return re.sub(r"\s+", "", str(text or ""))


def align_final_plan(
    utterance_id: str,
    partition: str,
    alignment: SilverUtteranceAlignment,
    planned: Sequence[PlannedPhoneTiming],
) -> tuple[tuple[AlignedPlanPhone, ...], dict[str, object]]:
    """Align timelines at the first equal phone, then retain cumulative drift."""
    reference = _reference_phones(alignment)
    result = align_phone_sequences(
        [phone.phone for phone in reference],
        [phone.phone for phone in planned],
    )
    equal_pairs = []
    for reference_index, plan_index in result.pairs:
        if reference_index is None or plan_index is None:
            continue
        left = normalize_label_phone(reference[reference_index].phone)[0]
        right = normalize_label_phone(planned[plan_index].phone)[0]
        if left == right:
            equal_pairs.append((reference_index, plan_index))
    if not equal_pairs:
        return (), {
            "alignment_cost": result.cost,
            "matched_phone_count": 0,
            "reference_phone_count": len(reference),
            "planned_phone_count": len(planned),
            "match_fraction": 0.0,
            "diagnostics": list(result.diagnostics) + ["no_equal_phone_pair"],
        }
    first_reference, first_plan = equal_pairs[0]
    reference_origin = reference[first_reference].start_seconds
    plan_origin = planned[first_plan].start_seconds
    rows = []
    for reference_index, plan_index in equal_pairs:
        left = reference[reference_index]
        right = planned[plan_index]
        reference_start = left.start_seconds - reference_origin
        reference_end = left.end_seconds - reference_origin
        predicted_start = right.start_seconds - plan_origin
        predicted_end = right.end_seconds - plan_origin
        reference_duration = max(1e-9, left.duration_seconds)
        predicted_duration = max(1e-9, right.duration_seconds)
        rows.append(AlignedPlanPhone(
            utterance_id=utterance_id,
            partition=partition,
            reference_index=reference_index,
            plan_index=plan_index,
            phone=normalize_label_phone(left.phone)[0],
            phone_class=right.phone_class,
            mora_index=right.mora_index,
            phrase_index=right.phrase_index,
            reference_start_seconds=reference_start,
            reference_end_seconds=reference_end,
            predicted_start_seconds=predicted_start,
            predicted_end_seconds=predicted_end,
            reference_duration_seconds=reference_duration,
            predicted_duration_seconds=predicted_duration,
            signed_duration_error_seconds=predicted_duration - reference_duration,
            absolute_duration_error_seconds=abs(
                predicted_duration - reference_duration),
            duration_ratio=predicted_duration / reference_duration,
            boundary_drift_seconds=predicted_end - reference_end,
            reference_confidence=float(left.confidence),
            reference_rejected=bool(left.rejection_reasons),
            phenomena=tuple(sorted(set(right.phenomena + (
                ("long_vowel",) if left.long_vowel else ()
            )))),
            timing_decisions=right.timing_decisions,
        ))
    return tuple(rows), {
        "alignment_cost": result.cost,
        "matched_phone_count": len(rows),
        "reference_phone_count": len(reference),
        "planned_phone_count": len(planned),
        "match_fraction": round(
            len(rows) / max(1, max(len(reference), len(planned))), 6),
        "first_reference_phone_index": first_reference,
        "first_planned_phone_index": first_plan,
        "timeline_origin": "first_equal_phone_start",
        "diagnostics": list(result.diagnostics),
    }


def summarize_duration_rows(
    rows: Sequence[AlignedPlanPhone],
) -> dict[str, object]:
    accepted = [row for row in rows if not row.reference_rejected]

    def summary(group: Sequence[AlignedPlanPhone]) -> dict[str, object]:
        ratios = [row.duration_ratio for row in group]
        absolute = [row.absolute_duration_error_seconds * 1000.0
                    for row in group]
        signed = [row.signed_duration_error_seconds * 1000.0
                  for row in group]
        drift = [abs(row.boundary_drift_seconds) * 1000.0 for row in group]
        normalized_residuals = []
        by_utterance = defaultdict(list)
        for row in group:
            by_utterance[row.utterance_id].append(math.log(max(
                1e-9, row.predicted_duration_seconds
            ) / max(1e-9, row.reference_duration_seconds)))
        for values in by_utterance.values():
            center = float(statistics.median(values))
            normalized_residuals.extend(value - center for value in values)
        return {
            "count": len(group),
            "median_duration_ratio": _median(ratios),
            "median_signed_error_ms": _median(signed),
            "mean_absolute_error_ms": (
                float(np.mean(absolute)) if absolute else None),
            "median_absolute_error_ms": _median(absolute),
            "p90_absolute_error_ms": _percentile(absolute, 90.0),
            "median_absolute_boundary_drift_ms": _median(drift),
            "rate_normalized_log_rmse": (
                float(np.sqrt(np.mean(np.square(normalized_residuals))))
                if normalized_residuals else None
            ),
        }

    by_class = defaultdict(list)
    by_phenomenon = defaultdict(list)
    by_partition = defaultdict(list)
    for row in accepted:
        by_class[row.phone_class].append(row)
        by_partition[row.partition].append(row)
        for name in row.phenomena:
            by_phenomenon[name].append(row)
    return {
        "all_accepted": summary(accepted),
        "all_including_rejected": summary(rows),
        "rejected_reference_phone_count": len(rows) - len(accepted),
        "by_phone_class": {
            key: summary(value) for key, value in sorted(by_class.items())
        },
        "by_phenomenon": {
            key: summary(value) for key, value in sorted(by_phenomenon.items())
        },
        "by_partition": {
            key: summary(value) for key, value in sorted(by_partition.items())
        },
    }


def _median_f0_interval(
    samples: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> tuple[float | None, float]:
    start = max(0, int(round(start_seconds * sample_rate)))
    end = min(len(samples), int(round(end_seconds * sample_rate)))
    if end - start < int(round(0.018 * sample_rate)):
        return None, 0.0
    duration = (end - start) / sample_rate
    window_seconds = min(0.050, max(0.020, duration * 0.72))
    window = max(16, int(round(window_seconds * sample_rate)))
    centers = np.linspace(start + window // 2, end - (window - window // 2), 3)
    estimates = []
    confidences = []
    for center in centers:
        left = int(round(center)) - window // 2
        frame = samples[left:left + window]
        if len(frame) != window:
            continue
        f0, confidence, ambiguity = estimate_f0(
            frame, sample_rate, minimum_hz=60.0, maximum_hz=420.0)
        if f0 is None or confidence < 0.38 or ambiguity > 0.78:
            continue
        estimates.append(float(f0))
        confidences.append(float(confidence))
    return _median(estimates), float(_median(confidences) or 0.0)


def legacy_pitch_semitones(
    utterance: JapaneseUtterance,
) -> dict[int, float]:
    """Reproduce the pre-Prompt-20 phrase-reset contour for A/B scoring."""
    output = {}
    for phrase in utterance.phrases:
        phrase_moras = tuple(phrase.moras)
        phrase_position = 0
        accented_before = 0
        for accent_position, accent in enumerate(phrase.accent_phrases):
            local_count = len(accent.moras)
            for local_index, mora in enumerate(accent.moras):
                if accent.accent_state == "accented":
                    nucleus = int(accent.accent_nucleus or 0)
                    if nucleus == 0:
                        lexical = 1.70 if local_index == 0 else -0.90
                    elif local_index == 0:
                        lexical = -1.00
                    elif local_index <= nucleus:
                        lexical = 1.65 if local_index == nucleus else 1.25
                    else:
                        lexical = -0.90
                elif accent.accent_state == "unaccented":
                    lexical = (-0.90 if local_index == 0 and
                               len(accent.moras) > 1 else 1.00)
                else:
                    lexical = -0.30 if local_index == 0 else 0.20
                phrase_progress = phrase_position / max(
                    1, len(phrase_moras) - 1)
                local_progress = local_index / max(1, local_count - 1)
                value = (
                    lexical
                    + 0.55 * (1.0 - phrase_progress)
                    - 0.80 * phrase_progress
                    - 0.15 * local_progress
                    + (0.35 if accent_position > 0 and local_index == 0
                       else 0.0)
                    - 0.45 * min(accented_before, 2)
                    + ((-0.25 - 0.08 * phrase.boundary_strength)
                       if phrase_position == len(phrase_moras) - 1 else 0.0)
                )
                output[mora.index] = round(value, 9)
                phrase_position += 1
            if accent.accent_state == "accented":
                accented_before += 1
    return output


def filter_pitch_observation_outliers(
    rows: Sequence[Mapping[str, object]],
    *,
    utterance_id: str,
    partition: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Reject only robust utterance-level octave/pulse-rate outliers."""
    copied = [dict(row) for row in rows]
    observed_log_center = _median(
        math.log2(float(row["observed_f0_hz"])) for row in copied)
    if observed_log_center is None:
        return (), ()
    offsets = [
        12.0 * (math.log2(float(row["observed_f0_hz"]))
                - observed_log_center)
        for row in copied
    ]
    offset_median = float(statistics.median(offsets)) if offsets else 0.0
    offset_mad = float(statistics.median(
        abs(value - offset_median) for value in offsets
    )) if offsets else 0.0
    # Natural phrase ranges can be broad, so only reject isolated values far
    # outside both a six-semitone floor and the utterance's robust spread.
    outlier_limit = max(6.0, min(9.0, 4.0 * 1.4826 * offset_mad))
    retained = []
    rejected = []
    for row, offset in zip(copied, offsets):
        if abs(offset - offset_median) <= outlier_limit:
            retained.append(row)
            continue
        rejected.append({
            "utterance_id": utterance_id,
            "partition": partition,
            "mora_index": row.get("mora_index"),
            "observed_f0_hz": row["observed_f0_hz"],
            "offset_from_utterance_median_semitones": round(offset, 6),
            "robust_outlier_limit_semitones": round(outlier_limit, 6),
            "reason": "probable_octave_or_pulse_rate_error",
        })
    return tuple(retained), tuple(rejected)


def _linear_slope(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    if len(rows) < 4:
        return None
    x = np.asarray([float(row["time_seconds"]) for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    if float(np.ptp(x)) < 0.1:
        return None
    return float(np.polyfit(x, y, 1)[0])


def summarize_pitch_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_utterance = defaultdict(list)
    by_phrase = defaultdict(list)
    for row in rows:
        by_utterance[str(row["utterance_id"])].append(row)
        by_phrase[(str(row["utterance_id"]), int(row["phrase_index"]))].append(row)
    utterance_metrics = []
    for utterance_id, values in sorted(by_utterance.items()):
        observed = [float(row["observed_semitones_centered"]) for row in values]
        predicted = [float(row["predicted_semitones_centered"]) for row in values]
        correlation = None
        if len(values) >= 3 and np.std(observed) > 1e-6 and np.std(predicted) > 1e-6:
            correlation = float(np.corrcoef(observed, predicted)[0, 1])
        observed_slope = _linear_slope(values, "observed_semitones_centered")
        predicted_slope = _linear_slope(values, "predicted_semitones_centered")
        utterance_metrics.append({
            "utterance_id": utterance_id,
            "mora_count": len(values),
            "mae_semitones": float(np.mean([
                abs(left - right) for left, right in zip(observed, predicted)
            ])),
            "contour_correlation": correlation,
            "observed_declination_semitones_per_second": observed_slope,
            "predicted_declination_semitones_per_second": predicted_slope,
            "declination_error_semitones_per_second": (
                abs(predicted_slope - observed_slope)
                if observed_slope is not None and predicted_slope is not None
                else None
            ),
        })
    range_errors = []
    for values in by_phrase.values():
        if len(values) < 3:
            continue
        observed = [float(row["observed_semitones_centered"]) for row in values]
        predicted = [float(row["predicted_semitones_centered"]) for row in values]
        observed_range = float(np.percentile(observed, 90) - np.percentile(observed, 10))
        predicted_range = float(np.percentile(predicted, 90) - np.percentile(predicted, 10))
        range_errors.append(abs(predicted_range - observed_range))
    return {
        "mora_observation_count": len(rows),
        "utterance_count": len(utterance_metrics),
        "mean_absolute_error_semitones": (
            float(np.mean([row["absolute_error_semitones"] for row in rows]))
            if rows else None
        ),
        "median_contour_correlation": _median(
            row["contour_correlation"] for row in utterance_metrics
            if row["contour_correlation"] is not None),
        "median_phrase_range_error_semitones": _median(range_errors),
        "median_declination_error_semitones_per_second": _median(
            row["declination_error_semitones_per_second"]
            for row in utterance_metrics
            if row["declination_error_semitones_per_second"] is not None),
        "utterances": utterance_metrics,
        "normalization": (
            "Observed log-F0 and predicted targets are independently centered "
            "per utterance; Kokoro speaker register is not copied."
        ),
    }


def _pitch_rows_with_origin(
    utterance_id: str,
    partition: str,
    utterance: JapaneseUtterance,
    plan: JapaneseSynthesisPlan,
    aligned: Sequence[AlignedPlanPhone],
    alignment: SilverUtteranceAlignment,
    audio_path: Path,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    reference = _reference_phones(alignment)
    if not aligned:
        return (), ()
    first_reference_index = aligned[0].reference_index
    source_origin = reference[first_reference_index].start_seconds
    data = read_audio(audio_path, expected_sample_rate=alignment.sample_rate)
    targets = {target.mora_index: target for target in plan.f0_targets}
    target_time_origin = min(
        (target.time for target in plan.f0_targets), default=0.0)
    legacy_targets = legacy_pitch_semitones(utterance)
    accent_by_mora = {}
    for phrase in utterance.phrases:
        for accent in phrase.accent_phrases:
            for local_index, mora in enumerate(accent.moras):
                accent_by_mora[mora.index] = (
                    accent.accent_state, accent.accent_nucleus, local_index)
    by_mora = defaultdict(list)
    for row in aligned:
        if row.phone in _VOWELS and not row.reference_rejected:
            by_mora[row.mora_index].append(row)
    provisional = []
    for mora_index, phones in sorted(by_mora.items()):
        target = targets.get(mora_index)
        legacy_target = legacy_targets.get(mora_index)
        if target is None or legacy_target is None:
            continue
        start = min(row.reference_start_seconds for row in phones)
        end = max(row.reference_end_seconds for row in phones)
        observed, confidence = _median_f0_interval(
            data.samples, data.sample_rate,
            source_origin + start, source_origin + end)
        if observed is None:
            continue
        state, nucleus, local = accent_by_mora.get(
            mora_index, (None, None, None))
        provisional.append({
            "utterance_id": utterance_id,
            "partition": partition,
            "mora_index": mora_index,
            "phrase_index": target.phrase_index,
            "time_seconds": round((start + end) / 2.0, 6),
            "observed_f0_hz": round(observed, 6),
            "observed_f0_confidence": round(confidence, 6),
            "predicted_semitones_raw": target.semitones_from_baseline,
            "pitch_model_id": plan.pitch_model_id,
            "pitch_target_kind": target.kind,
            "predicted_target_time_seconds": target.time,
            "predicted_target_elapsed_seconds": round(
                max(0.0, target.time - target_time_origin), 6),
            "pitch_components_semitones": dict(sorted(
                target.components_semitones.items())),
            "predicted_log_f0": target.log_f0,
            "baseline_log_f0": target.baseline_log_f0,
            "legacy_predicted_semitones_raw": legacy_target,
            "accent_state": state,
            "accent_nucleus": nucleus,
            "accent_local_index": local,
        })
    provisional, rejected = filter_pitch_observation_outliers(
        provisional, utterance_id=utterance_id, partition=partition)
    observed_center = _median(
        math.log2(float(row["observed_f0_hz"])) for row in provisional)
    predicted_center = _median(
        float(row["predicted_semitones_raw"]) for row in provisional)
    legacy_center = _median(
        float(row["legacy_predicted_semitones_raw"])
        for row in provisional)
    if (observed_center is None or predicted_center is None
            or legacy_center is None):
        return (), rejected
    output = []
    for row in provisional:
        observed = 12.0 * (
            math.log2(float(row["observed_f0_hz"])) - observed_center)
        predicted = float(row["predicted_semitones_raw"]) - predicted_center
        legacy_predicted = (
            float(row["legacy_predicted_semitones_raw"]) - legacy_center)
        output.append({
            **row,
            "observed_semitones_centered": round(observed, 6),
            "predicted_semitones_centered": round(predicted, 6),
            "absolute_error_semitones": round(abs(predicted - observed), 6),
            "legacy_predicted_semitones_centered": round(
                legacy_predicted, 6),
            "legacy_absolute_error_semitones": round(
                abs(legacy_predicted - observed), 6),
        })
    return tuple(output), rejected


def _safe_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _duration_svg(path: Path, rows: Sequence[AlignedPlanPhone]) -> None:
    by_utterance = defaultdict(list)
    for row in rows:
        if not row.reference_rejected:
            by_utterance[row.utterance_id].append(row)
    ranked = sorted(
        by_utterance.items(),
        key=lambda item: -float(np.mean([
            row.absolute_duration_error_seconds for row in item[1]
        ])),
    )[:6]
    width = 1320
    panel_height = 130
    height = 135 + panel_height * max(1, len(ranked))
    body = [
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#252525}'
        '.title{font-size:22px;font-weight:600}.note{font-size:12px;fill:#555}'
        '.phone{font-size:10px;fill:white;font-weight:600}'
        '.axis{stroke:#444;stroke-width:1}.drift{fill:none;stroke:#222;stroke-width:1.5}'
        '</style>',
        '<text x="30" y="34" class="title">Held-out final-plan phone timing</text>',
        '<text x="30" y="57" class="note">Top track: Kokoro silver boundary. Bottom: final contextual plan. Timelines meet at the first equal phone.</text>',
        '<text x="30" y="76" class="note">Colors are phone classes; the black line is cumulative boundary drift. Rejected silver phones are omitted.</text>',
    ]
    left, right = 230.0, width - 35.0
    for panel, (utterance_id, values) in enumerate(ranked):
        y = 105 + panel * panel_height
        maximum = max(max(row.reference_end_seconds, row.predicted_end_seconds)
                      for row in values)
        maximum = max(0.1, maximum)
        scale = (right - left) / maximum
        body.append(f'<text x="30" y="{y + 20}" class="note">{html.escape(utterance_id)}</text>')
        body.append(f'<line x1="{left}" y1="{y + 94}" x2="{right}" y2="{y + 94}" class="axis"/>')
        for row in values:
            color = _COLORS.get(row.phone_class, _COLORS["other"])
            for start, end, yy in (
                    (row.reference_start_seconds, row.reference_end_seconds, y + 8),
                    (row.predicted_start_seconds, row.predicted_end_seconds, y + 42)):
                x = left + start * scale
                w = max(1.0, (end - start) * scale)
                body.append(
                    f'<rect x="{x:.2f}" y="{yy}" width="{w:.2f}" height="25" '
                    f'fill="{color}" stroke="#fff" stroke-width="0.5"/>')
                if w >= 18:
                    body.append(
                        f'<text x="{x + w / 2:.2f}" y="{yy + 17}" '
                        f'text-anchor="middle" class="phone">{html.escape(row.phone)}</text>')
        points = []
        maximum_drift = max(0.020, max(abs(row.boundary_drift_seconds)
                                      for row in values))
        for row in values:
            x = left + row.reference_end_seconds * scale
            yy = y + 94 - 24.0 * row.boundary_drift_seconds / maximum_drift
            points.append(f'{x:.2f},{yy:.2f}')
        body.append(f'<polyline points="{" ".join(points)}" class="drift"/>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(body) + '</svg>\n', encoding="utf-8")


def _pitch_svg(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    by_utterance = defaultdict(list)
    for row in rows:
        by_utterance[str(row["utterance_id"])].append(row)
    ranked = sorted(
        by_utterance.items(),
        key=lambda item: -float(np.mean([
            row["absolute_error_semitones"] for row in item[1]
        ])),
    )[:5]
    width = 1320
    panel_height = 145
    height = 135 + panel_height * max(1, len(ranked))
    left, right = 230.0, width - 35.0
    body = [
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#252525}'
        '.title{font-size:22px;font-weight:600}.note{font-size:12px;fill:#555}'
        '.grid{stroke:#ddd;stroke-width:1}.observed{fill:none;stroke:#2878b8;stroke-width:2}'
        '.predicted{fill:none;stroke:#c65b3e;stroke-width:2}.dotO{fill:#2878b8}.dotP{fill:#c65b3e}'
        '</style>',
        '<text x="30" y="34" class="title">Held-out speaker-normalized Japanese F0</text>',
        '<text x="30" y="57" class="note">Blue: Kokoro observed log-F0. Orange: production structural targets. Each utterance is median-centered.</text>',
        '<text x="30" y="76" class="note">This tests contour shape without copying Kokoro speaker register; gaps are unvoiced or low-confidence morae.</text>',
    ]
    for panel, (utterance_id, values) in enumerate(ranked):
        values = sorted(values, key=lambda row: float(row["time_seconds"]))
        y = 105 + panel * panel_height
        maximum_time = max(0.1, max(float(row["time_seconds"]) for row in values))
        all_pitch = [float(row[key]) for row in values for key in (
            "observed_semitones_centered", "predicted_semitones_centered")]
        limit = max(3.0, _percentile([abs(value) for value in all_pitch], 95) or 3.0)
        body.append(f'<text x="30" y="{y + 20}" class="note">{html.escape(utterance_id)}</text>')
        for semitone in (-limit, 0.0, limit):
            yy = y + 70 - semitone / limit * 48
            body.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>')
        observed_points = []
        predicted_points = []
        for row in values:
            x = left + float(row["time_seconds"]) / maximum_time * (right - left)
            observed_y = y + 70 - float(row["observed_semitones_centered"]) / limit * 48
            predicted_y = y + 70 - float(row["predicted_semitones_centered"]) / limit * 48
            observed_points.append(f'{x:.2f},{observed_y:.2f}')
            predicted_points.append(f'{x:.2f},{predicted_y:.2f}')
            body.append(f'<circle cx="{x:.2f}" cy="{observed_y:.2f}" r="2.5" class="dotO"/>')
            body.append(f'<circle cx="{x:.2f}" cy="{predicted_y:.2f}" r="2.5" class="dotP"/>')
        body.append(f'<polyline points="{" ".join(observed_points)}" class="observed"/>')
        body.append(f'<polyline points="{" ".join(predicted_points)}" class="predicted"/>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(body) + '</svg>\n', encoding="utf-8")


def _markdown(report: Mapping[str, object]) -> str:
    duration = dict(report["duration"])
    contextual = dict(duration["contextual"]["summary"])["all_accepted"]
    legacy = dict(duration["legacy"]["summary"])["all_accepted"]
    pitch = dict(dict(report["pitch"])["contextual"])["summary"]
    legacy_pitch = dict(dict(report["pitch"])["legacy"])["summary"]
    lines = [
        "# Japanese Prosody Ground-Truth Audit",
        "",
        "Kokoro-Align boundaries are silver references, not hand-corrected labels. "
        "All timing comparisons use the final synthesis plan and align at the first equal phone.",
        "",
        "## Active Models",
        "",
        f"- Duration: `{report['models']['duration_model_id']}`",
        f"- Pitch: `{report['models']['pitch_model_id']}`",
        f"- Frontend: `{report['models']['frontend_name']}`",
        "",
        "## Timing",
        "",
        f"- Contextual MAE: {contextual['mean_absolute_error_ms']:.2f} ms",
        f"- Legacy MAE: {legacy['mean_absolute_error_ms']:.2f} ms",
        f"- Contextual median ratio: {contextual['median_duration_ratio']:.3f}",
        f"- Legacy median ratio: {legacy['median_duration_ratio']:.3f}",
        "",
        "## Pitch",
        "",
        f"- Speaker-normalized F0 MAE: {pitch['mean_absolute_error_semitones']:.3f} semitones",
        f"- Legacy F0 MAE: {legacy_pitch['mean_absolute_error_semitones']:.3f} semitones",
        f"- Median contour correlation: {pitch['median_contour_correlation']}",
        f"- Legacy contour correlation: {legacy_pitch['median_contour_correlation']}",
        f"- Median phrase-range error: {pitch['median_phrase_range_error_semitones']} semitones",
        f"- Median declination error: {pitch['median_declination_error_semitones_per_second']} semitones/s",
        "",
        "The benchmark never copies Kokoro's raw speaker register. Acoustic naturalness still requires listening.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    *,
    selection_path: Path | str,
    alignments_dir: Path | str,
    audio_dir: Path | str,
    voice_dir: Path | str,
    output_dir: Path | str,
    partitions: Sequence[str] = ("validation", "test"),
    frontend_mode: str = "openjtalk",
    maximum_records: int | None = None,
) -> dict[str, object]:
    selection_source = Path(selection_path)
    alignment_root = Path(alignments_dir)
    audio_root = Path(audio_dir)
    output_root = Path(output_dir)
    runtime = load_japanese_runtime_metadata(voice_dir)
    wanted = {str(value) for value in partitions}
    records = [record for record in load_selection(selection_source)
               if record.partition in wanted]
    if maximum_records is not None:
        records = records[:max(0, int(maximum_records))]
    if not records:
        raise ValueError("no Kokoro records matched the requested partitions")

    duration_rows = {"contextual": [], "legacy": []}
    alignment_reports = {"contextual": [], "legacy": []}
    pitch_rows = []
    pitch_rejections = []
    model_ids = set()
    pitch_ids = set()
    frontend_names = set()
    skipped = []
    for record in records:
        alignment_path = alignment_root / f"{record.utterance_id}.json"
        audio_candidates = sorted(audio_root.glob(f"{record.utterance_id}.*"))
        if not alignment_path.is_file() or not audio_candidates:
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "alignment_or_audio_missing",
            })
            continue
        alignment = load_alignment(alignment_path)
        if not alignment.accepted:
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "alignment_not_accepted",
            })
            continue
        source_audio = read_audio(
            audio_candidates[0], expected_sample_rate=alignment.sample_rate)
        alignment = refine_phrase_pauses(
            alignment, source_audio.samples, source_audio.sample_rate)
        frontend_text = normalize_kokoro_frontend_text(record.transcript)
        utterance = analyze_japanese(frontend_text, mode=frontend_mode)
        frontend_names.add(utterance.frontend_name)
        contextual_plan = None
        contextual_rows = ()
        for mode in ("contextual", "legacy"):
            plan = create_synthesis_plan(
                utterance,
                runtime_metadata=runtime,
                duration_model=mode,
            )
            if mode == "contextual":
                contextual_plan = plan
                model_ids.add(plan.duration_model_id)
                pitch_ids.add(plan.pitch_model_id)
            planned = final_plan_phone_timings(utterance, plan)
            aligned, alignment_report = align_final_plan(
                record.utterance_id, record.partition, alignment, planned)
            alignment_reports[mode].append({
                "utterance_id": record.utterance_id,
                "partition": record.partition,
                **alignment_report,
            })
            duration_rows[mode].extend(aligned)
            if mode == "contextual":
                contextual_rows = aligned
        if contextual_plan is not None and contextual_rows:
            accepted_pitch, rejected_pitch = _pitch_rows_with_origin(
                record.utterance_id,
                record.partition,
                utterance,
                contextual_plan,
                contextual_rows,
                alignment,
                audio_candidates[0],
            )
            pitch_rows.extend(accepted_pitch)
            pitch_rejections.extend(rejected_pitch)
    if len(model_ids) != 1 or len(pitch_ids) != 1:
        raise RuntimeError(
            "benchmark observed inconsistent production model IDs: "
            f"duration={sorted(model_ids)}, pitch={sorted(pitch_ids)}")
    if frontend_mode == "openjtalk" and frontend_names != {"openjtalk"}:
        raise RuntimeError(
            "Open JTalk was requested but the benchmark silently used: "
            + ", ".join(sorted(frontend_names)))

    duration_payload = {}
    for mode in ("contextual", "legacy"):
        rows = tuple(duration_rows[mode])
        duration_payload[mode] = {
            "summary": summarize_duration_rows(rows),
            "alignments": sorted(
                alignment_reports[mode], key=lambda row: row["utterance_id"]),
            "phones": [row.to_dict() for row in rows],
        }
    pitch_summary = summarize_pitch_rows(pitch_rows)
    legacy_pitch_rows = [{
        **row,
        "predicted_semitones_centered": row[
            "legacy_predicted_semitones_centered"],
        "absolute_error_semitones": row[
            "legacy_absolute_error_semitones"],
    } for row in pitch_rows]
    legacy_pitch_summary = summarize_pitch_rows(legacy_pitch_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "japanese_final_plan_kokoro_prosody_benchmark",
        "reference_status": (
            "Kokoro-Align acoustic boundaries are silver references and remain "
            "subject to alignment error."
        ),
        "partitions": sorted(wanted),
        "record_count_requested": len(records),
        "record_count_evaluated": len(records) - len(skipped),
        "skipped": skipped,
        "models": {
            "duration_model": "contextual",
            "duration_model_id": next(iter(model_ids)),
            "pitch_model_id": next(iter(pitch_ids)),
            "frontend_name": ",".join(sorted(frontend_names)),
            "silent_fallback_allowed": False,
        },
        "duration": duration_payload,
        "pitch": {
            "contextual": {
                "model_id": next(iter(pitch_ids)),
                "summary": pitch_summary,
                "moras": pitch_rows,
            },
            "legacy": {
                "model_id": LEGACY_PITCH_MODEL_ID,
                "summary": legacy_pitch_summary,
            },
            "rejections": pitch_rejections,
        },
        "provenance": {
            "selection_sha256": _sha256(selection_source),
            "alignment_method": "kokoro_align_ctc_20221201_acoustic_refinement_v1",
            "timeline_alignment": "first_equal_phone_start",
            "kokoro_token_spacing_removed_before_openjtalk": True,
            "corpus_speaker_absolute_rate_copied": False,
            "corpus_speaker_register_copied": False,
            "deterministic_wall_clock_fields": False,
        },
        "limitations": [
            "Kokoro phone boundaries are silver rather than hand-labelled.",
            "F0 is estimated only inside matched vowel intervals with adequate confidence.",
            "Probable octave or pulse-rate outliers are retained as explicit pitch rejections.",
            "The audit validates plans and target contours, not perceived naturalness.",
        ],
    }
    _safe_json(output_root / "japanese_prosody_benchmark.json", report)
    (output_root / "japanese_prosody_benchmark.md").write_text(
        _markdown(report), encoding="utf-8")
    _duration_svg(
        output_root / "timing_alignment.svg",
        duration_rows["contextual"],
    )
    _pitch_svg(output_root / "pitch_alignment.svg", pitch_rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare final Japanese synthesis plans with held-out Kokoro")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--alignments", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", action="append",
                        choices=("train", "validation", "test"))
    parser.add_argument("--frontend", default="openjtalk",
                        choices=("openjtalk", "auto", "kana"))
    parser.add_argument("--maximum-records", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        selection_path=args.selection,
        alignments_dir=args.alignments,
        audio_dir=args.audio,
        voice_dir=args.voice,
        output_dir=args.output,
        partitions=args.partition or ("validation", "test"),
        frontend_mode=args.frontend,
        maximum_records=args.maximum_records,
    )
    print(json.dumps({
        "models": report["models"],
        "records": report["record_count_evaluated"],
        "contextual": report["duration"]["contextual"]["summary"]["all_accepted"],
        "legacy": report["duration"]["legacy"]["summary"]["all_accepted"],
        "pitch": report["pitch"]["contextual"]["summary"],
        "legacy_pitch": report["pitch"]["legacy"]["summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
