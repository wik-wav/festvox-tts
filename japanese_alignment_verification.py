"""Render inspectable Kokoro-source versus Festival-synthesis alignments.

The numeric Prompt 20 benchmark aligns timelines at the first equal phone.
This companion audit makes that same alignment visible on the actual source
and rendered waveforms. Kokoro boundaries remain silver references.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import textwrap
from typing import Mapping, Sequence

import numpy as np

from formant_analysis import read_audio
from japanese_devoicing import apply_vowel_realizations
from japanese_duration_ab import _backend, _safe_output
from japanese_duration_corpus import normalize_label_phone
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
from japanese_prosody_ab import _gui_core, build_pitch_systems
from japanese_prosody_benchmark import (
    AlignedPlanPhone,
    align_final_plan,
    final_plan_phone_timings,
    normalize_kokoro_frontend_text,
)
from japanese_phrase_edges import (
    compare_phrase_edges,
    detect_acoustic_phrase_edge,
)
from japanese_synthesis import create_synthesis_plan
from kokoro_reference import (
    SilverUtteranceAlignment,
    load_alignment,
    load_selection,
    refine_phrase_pauses,
)


SCHEMA_VERSION = 1
_SILENCE = {"sil", "pau", "sp"}


def _reference_phones(
    alignment: SilverUtteranceAlignment,
) -> tuple[object, ...]:
    return tuple(
        phone for phone in alignment.phones
        if normalize_label_phone(phone.phone)[0] not in _SILENCE
    )


def canonical_phone_sequences(
    alignment: SilverUtteranceAlignment,
    planned: Sequence[object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the two linguistic sequences used by the visual audit."""
    reference = tuple(
        normalize_label_phone(phone.phone)[0]
        for phone in _reference_phones(alignment)
    )
    synthesis = tuple(
        normalize_label_phone(phone.phone)[0] for phone in planned
    )
    return reference, synthesis


@dataclass(frozen=True)
class PhraseAlignment:
    reference_phrase_index: int
    synthesis_phrase_index: int
    rows: tuple[AlignedPlanPhone, ...]

    @property
    def reference_start_seconds(self) -> float:
        return self.rows[0].reference_start_seconds

    @property
    def reference_end_seconds(self) -> float:
        return self.rows[-1].reference_end_seconds

    @property
    def synthesis_start_seconds(self) -> float:
        return self.rows[0].predicted_start_seconds

    @property
    def synthesis_end_seconds(self) -> float:
        return self.rows[-1].predicted_end_seconds

    def localized_rows(self) -> tuple[AlignedPlanPhone, ...]:
        """Align this phrase at its own first matched phone."""
        source_origin = self.reference_start_seconds
        synthesis_origin = self.synthesis_start_seconds
        localized = []
        for row in self.rows:
            reference_start = row.reference_start_seconds - source_origin
            reference_end = row.reference_end_seconds - source_origin
            predicted_start = row.predicted_start_seconds - synthesis_origin
            predicted_end = row.predicted_end_seconds - synthesis_origin
            localized.append(replace(
                row,
                reference_start_seconds=reference_start,
                reference_end_seconds=reference_end,
                predicted_start_seconds=predicted_start,
                predicted_end_seconds=predicted_end,
                boundary_drift_seconds=predicted_end - reference_end,
            ))
        return tuple(localized)

    def to_metrics(self) -> dict[str, object]:
        localized = self.localized_rows()
        source_duration = max(
            0.0, self.reference_end_seconds - self.reference_start_seconds)
        synthesis_duration = max(
            0.0, self.synthesis_end_seconds - self.synthesis_start_seconds)
        accepted = [row for row in localized if not row.reference_rejected]
        duration_errors = [
            row.absolute_duration_error_seconds * 1000.0 for row in accepted
        ]
        drift = [
            abs(row.boundary_drift_seconds) * 1000.0 for row in accepted
        ]
        return {
            "reference_phrase_index": self.reference_phrase_index,
            "synthesis_phrase_index": self.synthesis_phrase_index,
            "phone_count": len(self.rows),
            "source_duration_seconds": round(source_duration, 6),
            "synthesis_duration_seconds": round(synthesis_duration, 6),
            "synthesis_to_source_ratio": (
                round(synthesis_duration / source_duration, 6)
                if source_duration > 1e-9 else None
            ),
            "phone_duration_mae_ms": (
                round(float(statistics.fmean(duration_errors)), 6)
                if duration_errors else None
            ),
            "median_absolute_local_boundary_drift_ms": (
                round(_finite_median(drift), 6) if drift else None
            ),
            "phrase_end_drift_ms": round(
                (synthesis_duration - source_duration) * 1000.0, 6),
        }


def split_phrase_alignments(
    aligned: Sequence[AlignedPlanPhone],
    alignment: SilverUtteranceAlignment,
) -> tuple[PhraseAlignment, ...]:
    """Group exact phone matches into one-to-one punctuation phrases.

    A source phrase may not map to two synthesis phrases (or vice versa). Such
    a disagreement is a linguistic segmentation mismatch and must be visible,
    rather than being hidden by a local timing origin.
    """
    reference = _reference_phones(alignment)
    groups: list[tuple[tuple[int, int], list[AlignedPlanPhone]]] = []
    source_to_synthesis: dict[int, int] = {}
    synthesis_to_source: dict[int, int] = {}
    for row in aligned:
        source_phrase = int(reference[row.reference_index].phrase_index)
        synthesis_phrase = int(row.phrase_index)
        if (source_phrase in source_to_synthesis and
                source_to_synthesis[source_phrase] != synthesis_phrase):
            raise ValueError("one source phrase maps to multiple synthesis phrases")
        if (synthesis_phrase in synthesis_to_source and
                synthesis_to_source[synthesis_phrase] != source_phrase):
            raise ValueError("one synthesis phrase maps to multiple source phrases")
        source_to_synthesis[source_phrase] = synthesis_phrase
        synthesis_to_source[synthesis_phrase] = source_phrase
        key = (source_phrase, synthesis_phrase)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(row)
    return tuple(
        PhraseAlignment(source, synthesis, tuple(rows))
        for (source, synthesis), rows in groups if rows
    )


def summarize_phrase_alignment(
    phrases: Sequence[PhraseAlignment],
) -> dict[str, object]:
    phrase_rows = tuple(phrases)
    if not phrase_rows:
        return {"phrase_count": 0, "phrases": [], "pauses": []}
    source_active = sum(
        phrase.reference_end_seconds - phrase.reference_start_seconds
        for phrase in phrase_rows)
    synthesis_active = sum(
        phrase.synthesis_end_seconds - phrase.synthesis_start_seconds
        for phrase in phrase_rows)
    source_span = (
        phrase_rows[-1].reference_end_seconds
        - phrase_rows[0].reference_start_seconds)
    synthesis_span = (
        phrase_rows[-1].synthesis_end_seconds
        - phrase_rows[0].synthesis_start_seconds)
    pauses = []
    for index, (left, right) in enumerate(zip(phrase_rows, phrase_rows[1:])):
        source_pause = max(
            0.0, right.reference_start_seconds - left.reference_end_seconds)
        synthesis_pause = max(
            0.0, right.synthesis_start_seconds - left.synthesis_end_seconds)
        pauses.append({
            "after_phrase_index": index,
            "source_duration_seconds": round(source_pause, 6),
            "synthesis_duration_seconds": round(synthesis_pause, 6),
            "error_ms": round((synthesis_pause - source_pause) * 1000.0, 6),
        })
    source_pause_total = sum(
        float(row["source_duration_seconds"]) for row in pauses)
    synthesis_pause_total = sum(
        float(row["synthesis_duration_seconds"]) for row in pauses)
    return {
        "phrase_count": len(phrase_rows),
        "source_matched_span_seconds": round(source_span, 6),
        "synthesis_matched_span_seconds": round(synthesis_span, 6),
        "total_rate_ratio": (
            round(synthesis_span / source_span, 6)
            if source_span > 1e-9 else None
        ),
        "source_active_phrase_seconds": round(source_active, 6),
        "synthesis_active_phrase_seconds": round(synthesis_active, 6),
        "active_speech_ratio": (
            round(synthesis_active / source_active, 6)
            if source_active > 1e-9 else None
        ),
        "source_interphrase_pause_seconds": round(source_pause_total, 6),
        "synthesis_interphrase_pause_seconds": round(
            synthesis_pause_total, 6),
        "interphrase_pause_error_ms": round(
            (synthesis_pause_total - source_pause_total) * 1000.0, 6),
        "phrases": [phrase.to_metrics() for phrase in phrase_rows],
        "pauses": pauses,
    }


def analyze_phrase_edge_acoustics(
    *,
    phrases: Sequence[PhraseAlignment],
    source_samples: Sequence[float] | np.ndarray,
    source_sample_rate: int,
    synthesis_samples: Sequence[float] | np.ndarray,
    synthesis_sample_rate: int,
    source_origin_seconds: float,
    synthesis_origin_seconds: float,
    utterance: object | None = None,
) -> tuple[dict[str, object], ...]:
    """Measure audible activity outside each phrase's logical boundaries."""
    results = []
    mora_by_index = {
        int(mora.index): mora for mora in getattr(utterance, "moras", ())
    }
    for phrase_number, phrase in enumerate(phrases, start=1):
        first_mora = phrase.rows[0].mora_index
        last_mora = phrase.rows[-1].mora_index
        first_model_mora = mora_by_index.get(first_mora)
        last_model_mora = mora_by_index.get(last_mora)
        first_morphology = dict(
            getattr(first_model_mora, "provenance", {}).get("morphology")
            or {})
        last_morphology = dict(
            getattr(last_model_mora, "provenance", {}).get("morphology")
            or {})
        first_rows = tuple(
            row for row in phrase.rows if row.mora_index == first_mora)
        last_rows = tuple(
            row for row in phrase.rows if row.mora_index == last_mora)
        source_initial_boundary = (
            source_origin_seconds + phrase.reference_start_seconds)
        synth_initial_boundary = (
            synthesis_origin_seconds + phrase.synthesis_start_seconds)
        source_final_boundary = (
            source_origin_seconds + phrase.reference_end_seconds)
        synth_final_boundary = (
            synthesis_origin_seconds + phrase.synthesis_end_seconds)
        source_initial = detect_acoustic_phrase_edge(
            source_samples, source_sample_rate, source_initial_boundary,
            edge="initial")
        synth_initial = detect_acoustic_phrase_edge(
            synthesis_samples, synthesis_sample_rate, synth_initial_boundary,
            edge="initial")
        source_final = detect_acoustic_phrase_edge(
            source_samples, source_sample_rate, source_final_boundary,
            edge="final")
        synth_final = detect_acoustic_phrase_edge(
            synthesis_samples, synthesis_sample_rate, synth_final_boundary,
            edge="final")
        initial = compare_phrase_edges(source_initial, synth_initial)
        final = compare_phrase_edges(source_final, synth_final)

        def effective_initial(rows, origin, detected, synthesis=False):
            if detected.acoustic_boundary_seconds is None:
                return None
            end = (rows[-1].predicted_end_seconds if synthesis
                   else rows[-1].reference_end_seconds) + origin
            return max(0.0, end - detected.acoustic_boundary_seconds)

        def effective_final(rows, origin, detected, synthesis=False):
            if detected.acoustic_boundary_seconds is None:
                return None
            start = (rows[0].predicted_start_seconds if synthesis
                     else rows[0].reference_start_seconds) + origin
            return max(0.0, detected.acoustic_boundary_seconds - start)

        source_initial_mora = effective_initial(
            first_rows, source_origin_seconds, source_initial)
        synth_initial_mora = effective_initial(
            first_rows, synthesis_origin_seconds, synth_initial,
            synthesis=True)
        source_final_mora = effective_final(
            last_rows, source_origin_seconds, source_final)
        synth_final_mora = effective_final(
            last_rows, synthesis_origin_seconds, synth_final,
            synthesis=True)

        def duration_payload(source_value, synth_value):
            return {
                "source_seconds": (
                    round(source_value, 6)
                    if source_value is not None else None),
                "synthesis_seconds": (
                    round(synth_value, 6)
                    if synth_value is not None else None),
                "synthesis_excess_ms": (
                    round((synth_value - source_value) * 1000.0, 3)
                    if source_value is not None and synth_value is not None
                    else None
                ),
            }

        results.append({
            "phrase_number_one_based": phrase_number,
            "initial_phone": phrase.rows[0].phone,
            "initial_phone_class": phrase.rows[0].phone_class,
            "initial_surface": first_morphology.get("string"),
            "initial_grammatical_role": first_morphology.get(
                "grammatical_role"),
            "initial_edge_type": (
                "vowel_initial" if phrase.rows[0].phone_class == "vowel"
                else "consonant_initial"
            ),
            "final_phone": phrase.rows[-1].phone,
            "final_phone_class": phrase.rows[-1].phone_class,
            "final_surface": last_morphology.get("string"),
            "final_grammatical_role": last_morphology.get(
                "grammatical_role"),
            "initial": initial,
            "final": final,
            "effective_initial_mora_duration": duration_payload(
                source_initial_mora, synth_initial_mora),
            "effective_final_mora_duration": duration_payload(
                source_final_mora, synth_final_mora),
        })
    return tuple(results)


def summarize_edge_acoustics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        category = str(row.get("initial_edge_type") or "unknown")
        initial = dict(row.get("initial") or {})
        effective = dict(row.get("effective_initial_mora_duration") or {})
        bucket = grouped.setdefault(category, {
            "edge": [], "effective_mora": []})
        edge_value = initial.get("synthesis_excess_extension_ms")
        mora_value = effective.get("synthesis_excess_ms")
        if edge_value is not None:
            bucket["edge"].append(float(edge_value))
        if mora_value is not None:
            bucket["effective_mora"].append(float(mora_value))
    return {
        category: {
            "count": len(values["edge"]),
            "median_synthesis_excess_extension_ms": round(
                _finite_median(values["edge"]), 3),
            "median_effective_first_mora_excess_ms": round(
                _finite_median(values["effective_mora"]), 3),
        }
        for category, values in sorted(grouped.items())
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _finite_median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values
              if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else 0.0


def select_alignment_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    per_partition: int = 5,
    max_phrases: int | None = 2,
) -> tuple[dict[str, object], ...]:
    """Select deterministic best/median/worst timing examples per partition."""
    count = max(1, int(per_partition))
    phrase_limit = (None if max_phrases is None
                    else max(1, int(max_phrases)))
    selected: list[dict[str, object]] = []
    partitions = sorted({str(row["partition"]) for row in candidates})
    for partition in partitions:
        rows = sorted(
            (dict(row) for row in candidates
             if str(row["partition"]) == partition
             and (phrase_limit is None
                  or int(row.get("phrase_count", 1)) <= phrase_limit)),
            key=lambda row: (
                float(row["median_absolute_boundary_drift_ms"]),
                str(row["utterance_id"]),
            ),
        )
        if not rows:
            continue
        take = min(count, len(rows))
        if take == 1:
            indexes = [len(rows) // 2]
            tiers = ["median"]
        else:
            indexes = [int(round(index * (len(rows) - 1) / (take - 1)))
                       for index in range(take)]
            tiers = (["best", "worst"] if take == 2 else
                     ["best"] + ["intermediate"] * (take - 2) + ["worst"])
            if take % 2 == 1:
                tiers[take // 2] = "median"
        used = set()
        for index, tier in zip(indexes, tiers):
            if index in used:
                continue
            used.add(index)
            row = dict(rows[index])
            row["selection_tier"] = tier
            selected.append(row)
    return tuple(selected)


def waveform_envelope(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    origin_seconds: float,
    start_seconds: float,
    end_seconds: float,
    columns: int = 2600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel min/max peaks so short transients remain visible."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0 or sample_rate <= 0 or end_seconds <= start_seconds:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    absolute_start = max(0.0, float(start_seconds) + float(origin_seconds))
    absolute_end = min(
        values.size / float(sample_rate),
        float(end_seconds) + float(origin_seconds),
    )
    first = max(0, int(math.floor(absolute_start * sample_rate)))
    last = min(values.size, int(math.ceil(absolute_end * sample_rate)))
    if last <= first:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    window = values[first:last]
    bucket_count = min(max(1, int(columns)), window.size)
    edges = np.linspace(0, window.size, bucket_count + 1, dtype=np.int64)
    low = np.empty(bucket_count, dtype=np.float64)
    high = np.empty(bucket_count, dtype=np.float64)
    centers = np.empty(bucket_count, dtype=np.float64)
    for index in range(bucket_count):
        left = int(edges[index])
        right = max(left + 1, int(edges[index + 1]))
        chunk = window[left:right]
        low[index] = float(np.min(chunk))
        high[index] = float(np.max(chunk))
        centers[index] = (
            (first + 0.5 * (left + right - 1)) / float(sample_rate)
            - float(origin_seconds)
        )
    return centers, low, high


def _normalized_peaks(samples: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    divisor = max(1e-9, peak)
    return samples / divisor, peak


def _nice_tick_step(span: float, target: int = 10) -> float:
    raw = max(1e-6, float(span) / max(1, int(target)))
    power = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / power
    multiple = 1.0 if scaled <= 1.0 else 2.0 if scaled <= 2.0 else \
        5.0 if scaled <= 5.0 else 10.0
    return multiple * power


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    color: str = "#202124",
    bold: bool = False,
    anchor: str = "start",
    line_height: float | None = None,
) -> str:
    lines = str(value).splitlines() or [""]
    spacing = float(line_height or size * 1.25)
    spans = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else f"{spacing:.1f}"
        spans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    weight = "700" if bold else "400"
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" '
        f'font-family="Yu Gothic UI, Meiryo, Noto Sans CJK JP, Arial, sans-serif" '
        f'font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
        + "".join(spans) + "</text>"
    )


def _svg_phone_boundaries(
    rows: Sequence[AlignedPlanPhone],
    *,
    source: bool,
    top: float,
    bottom: float,
    map_x,
) -> list[str]:
    color = "#c43b3b" if source else "#2f62c5"
    dash = "" if source else ' stroke-dasharray="7 5"'
    elements = []
    for index, row in enumerate(rows):
        start = (row.reference_start_seconds if source
                 else row.predicted_start_seconds)
        end = (row.reference_end_seconds if source
               else row.predicted_end_seconds)
        x = map_x(start)
        elements.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
            f'y2="{bottom:.1f}" stroke="{color}" stroke-width="1.2"{dash}/>'
        )
        if index == len(rows) - 1:
            x_end = map_x(end)
            elements.append(
                f'<line x1="{x_end:.1f}" y1="{top:.1f}" '
                f'x2="{x_end:.1f}" y2="{bottom:.1f}" stroke="{color}" '
                f'stroke-width="1.2"{dash}/>'
            )
        width = map_x(end) - x
        if width >= 22.0:
            label_y = top + (31.0 if index % 2 == 0 else 56.0)
            elements.append(_svg_text(
                x + width / 2.0, label_y, row.phone,
                size=13, anchor="middle"))
    return elements


def render_alignment_plot(
    path: Path | str,
    *,
    source_samples: Sequence[float] | np.ndarray,
    source_sample_rate: int,
    synthesis_samples: Sequence[float] | np.ndarray,
    synthesis_sample_rate: int,
    source_origin_seconds: float,
    synthesis_origin_seconds: float,
    aligned_phones: Sequence[AlignedPlanPhone],
    title: str,
    subtitle: str,
    annotation_lines: Sequence[str] = (),
    fit_to_rows: bool = False,
    timeline_label: str = "Seconds after first equal phone start",
) -> Path:
    """Draw matched target/synthesis timelines around a correspondence panel."""
    output = Path(path)
    if output.suffix.casefold() != ".svg":
        raise ValueError("alignment plot output must use the .svg extension")
    rows = tuple(aligned_phones)
    if not rows:
        raise ValueError("alignment plot requires at least one matched phone")
    source = np.asarray(source_samples, dtype=np.float64).reshape(-1)
    synthesis = np.asarray(synthesis_samples, dtype=np.float64).reshape(-1)
    source_norm, source_peak = _normalized_peaks(source)
    synthesis_norm, synthesis_peak = _normalized_peaks(synthesis)
    if fit_to_rows:
        left = min(
            rows[0].reference_start_seconds,
            rows[0].predicted_start_seconds,
        )
        right = max(
            rows[-1].reference_end_seconds,
            rows[-1].predicted_end_seconds,
        )
    else:
        left = min(
            -float(source_origin_seconds),
            -float(synthesis_origin_seconds),
            rows[0].reference_start_seconds,
            rows[0].predicted_start_seconds,
        )
        right = max(
            source.size / float(source_sample_rate) - source_origin_seconds,
            synthesis.size / float(synthesis_sample_rate)
            - synthesis_origin_seconds,
            rows[-1].reference_end_seconds,
            rows[-1].predicted_end_seconds,
        )
    margin = max(0.03, min(0.15, 0.03 * max(1.0, right - left)))
    left -= margin
    right += margin
    sx, slo, shi = waveform_envelope(
        source_norm, source_sample_rate,
        origin_seconds=source_origin_seconds,
        start_seconds=left,
        end_seconds=right,
    )
    yx, ylo, yhi = waveform_envelope(
        synthesis_norm, synthesis_sample_rate,
        origin_seconds=synthesis_origin_seconds,
        start_seconds=left,
        end_seconds=right,
    )

    wrapped_annotations = []
    for annotation in annotation_lines:
        wrapped_annotations.extend(textwrap.wrap(
            str(annotation), width=150, break_long_words=True,
            replace_whitespace=False, drop_whitespace=True,
        ) or [""])
    panel_shift = (14.0 + 21.0 * len(wrapped_annotations)
                   if wrapped_annotations else 0.0)
    width = int(max(2480, min(3200, 340 + 110 * len(rows))))
    height = int(round(1335 + panel_shift))
    plot_left = 235.0
    plot_right = width - 45.0
    plot_width = plot_right - plot_left
    panels = {
        "source_wave": (135.0 + panel_shift, 405.0 + panel_shift),
        "source_phones": (425.0 + panel_shift, 493.0 + panel_shift),
        "correspondence": (493.0 + panel_shift, 720.0 + panel_shift),
        "synth_phones": (720.0 + panel_shift, 788.0 + panel_shift),
        "synth_wave": (808.0 + panel_shift, 1078.0 + panel_shift),
    }

    def map_x(value: float) -> float:
        return plot_left + ((float(value) - left) / (right - left)) * plot_width

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="#f5f4ef"/>',
        _svg_text(width / 2.0, 42.0, title, size=28, bold=True,
                  anchor="middle"),
        _svg_text(width / 2.0, 78.0, subtitle, size=16, color="#444444",
                  anchor="middle"),
    ]
    for index, annotation in enumerate(wrapped_annotations):
        elements.append(_svg_text(
            25.0, 108.0 + index * 21.0, annotation,
            size=14, color="#343434"))
    for name, (top, bottom) in panels.items():
        fill = "#fffdf8" if name != "correspondence" else "#f5f4ef"
        elements.append(
            f'<rect x="{plot_left:.1f}" y="{top:.1f}" '
            f'width="{plot_width:.1f}" height="{bottom - top:.1f}" '
            f'fill="{fill}"/>'
        )

    tick_step = _nice_tick_step(right - left)
    tick = math.ceil(left / tick_step) * tick_step
    while tick <= right + 1e-9:
        x = map_x(tick)
        for name in ("source_wave", "source_phones",
                     "synth_phones", "synth_wave"):
            top, bottom = panels[name]
            elements.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
                f'y2="{bottom:.1f}" stroke="#d7d5ce" stroke-width="1"/>'
            )
        elements.append(_svg_text(
            x, panels["source_wave"][0] - 10.0, f"{tick:.2f}", size=13,
            color="#555555", anchor="middle"))
        elements.append(_svg_text(
            x, panels["synth_wave"][1] + 23.0, f"{tick:.2f}", size=13,
            color="#555555", anchor="middle"))
        tick += tick_step
    zero_x = map_x(0.0)
    for top, bottom in panels.values():
        elements.append(
            f'<line x1="{zero_x:.1f}" y1="{top:.1f}" '
            f'x2="{zero_x:.1f}" y2="{bottom:.1f}" '
            'stroke="#202124" stroke-width="1.8"/>'
        )

    def waveform_path(times, low_values, high_values, top, bottom, color):
        center = 0.5 * (top + bottom)
        scale = 0.43 * (bottom - top)
        commands = []
        for time_value, low_value, high_value in zip(
                times, low_values, high_values):
            x = map_x(float(time_value))
            y1 = center - float(high_value) * scale
            y2 = center - float(low_value) * scale
            commands.append(f"M{x:.1f},{y1:.1f}V{y2:.1f}")
        return (
            f'<path d="{"".join(commands)}" fill="none" stroke="{color}" '
            'stroke-width="1.25"/>'
            f'<line x1="{plot_left:.1f}" y1="{center:.1f}" '
            f'x2="{plot_right:.1f}" y2="{center:.1f}" '
            'stroke="#5c6261" stroke-width="0.8"/>'
        )

    elements.append(waveform_path(
        sx, slo, shi, *panels["source_wave"], "#197b72"))
    elements.append(waveform_path(
        yx, ylo, yhi, *panels["synth_wave"], "#4868b1"))
    elements.append(_svg_text(
        12.0, panels["source_wave"][0] + 105.0,
        "Target waveform\nKokoro silver reference", size=16, bold=True))
    elements.append(_svg_text(
        12.0, panels["synth_wave"][0] + 105.0,
        "Synthesized waveform\nFestival render",
        size=16, bold=True))

    source_y, source_bottom = panels["source_phones"]
    synth_y, synth_bottom = panels["synth_phones"]
    elements.append(_svg_text(
        218.0, source_y + 41.0, "Target phones", size=15,
        bold=True, anchor="end"))
    elements.append(_svg_text(
        218.0, synth_y + 41.0, "Synthesized phones", size=15,
        bold=True, anchor="end"))
    elements.append(_svg_text(
        218.0, panels["correspondence"][0] + 35.0,
        "Phone duration delta\n(synth - target)", size=13,
        bold=True, anchor="end"))
    boundary_pairs = []
    delta_labels = []
    for index, row in enumerate(rows):
        source_x = map_x(row.reference_start_seconds)
        source_width = max(
            1.0, map_x(row.reference_end_seconds) - source_x)
        synth_x = map_x(row.predicted_start_seconds)
        synth_width = max(
            1.0, map_x(row.predicted_end_seconds) - synth_x)
        elements.append(
            f'<rect x="{source_x:.1f}" y="{source_y:.1f}" '
            f'width="{source_width:.1f}" height="{source_bottom - source_y:.1f}" '
            'fill="#f2c8c2" '
            'stroke="#c43b3b" stroke-width="1"/>'
        )
        elements.append(
            f'<rect x="{synth_x:.1f}" y="{synth_y:.1f}" '
            f'width="{synth_width:.1f}" height="{synth_bottom - synth_y:.1f}" '
            'fill="#c6d5f0" '
            'stroke="#2f62c5" stroke-width="1"/>'
        )
        if source_width >= 20.0:
            elements.append(_svg_text(
                source_x + source_width / 2.0, source_y + 42.0,
                row.phone, size=12, anchor="middle"))
        if synth_width >= 20.0:
            elements.append(_svg_text(
                synth_x + synth_width / 2.0, synth_y + 42.0,
                row.phone, size=12, anchor="middle"))
        boundary_pairs.append((source_x, synth_x))
        delta_ms = ((row.predicted_duration_seconds
                     - row.reference_duration_seconds) * 1000.0)
        delta_x = 0.5 * (
            source_x + source_width / 2.0
            + synth_x + synth_width / 2.0)
        delta_y = (panels["correspondence"][0] + 55.0
                   + (index % 4) * 39.0)
        delta_labels.append(_svg_text(
            delta_x, delta_y, f"{delta_ms:+.0f} ms",
            size=15, color="#772f39", bold=True, anchor="middle",
        ))
    boundary_pairs.append((
        map_x(rows[-1].reference_end_seconds),
        map_x(rows[-1].predicted_end_seconds),
    ))
    for source_x, synth_x in boundary_pairs:
        elements.append(
            f'<line x1="{source_x:.1f}" y1="{source_bottom:.1f}" '
            f'x2="{synth_x:.1f}" y2="{synth_y:.1f}" '
            'stroke="#df4b5b" stroke-width="1.6" opacity="0.82"/>'
        )
    elements.extend(delta_labels)
    elements.append(_svg_text(
        (plot_left + plot_right) / 2.0, panels["synth_wave"][1] + 55.0,
        timeline_label, size=15,
        bold=True, anchor="middle"))
    elements.append(_svg_text(
        25.0, panels["synth_wave"][1] + 94.0,
        "Reading guide: black line = first equal phone start (t=0); each red "
        "connector joins an identified target boundary to the corresponding "
        "synthesized boundary.",
        size=14, color="#343434"))
    elements.append(_svg_text(
        25.0, panels["synth_wave"][1] + 126.0,
        "Phone deltas are synthesized duration minus target duration. "
        "Waveforms are peak-normalized independently for display only "
        f"(source peak {source_peak:.4f}, synth peak {synthesis_peak:.4f}).",
        size=14, color="#343434"))
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return output


def _overview(paths: Sequence[Path], output: Path) -> Path | None:
    if not paths:
        return None
    if output.suffix.casefold() != ".svg":
        raise ValueError("alignment overview output must use .svg")
    columns = 2
    rows = int(math.ceil(len(paths) / columns))
    tile_width = 1200
    tile_height = 735
    header = 100
    width = columns * tile_width
    height = header + rows * tile_height
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="#f5f4ef"/>',
        _svg_text(
            width / 2.0, 38.0,
            "Prompt 20 held-out waveform alignment audit",
            size=24, bold=True, anchor="middle"),
        _svg_text(
            width / 2.0, 72.0,
            "Best, median, and worst boundary-drift examples per partition",
            size=18, anchor="middle"),
    ]
    for index, path in enumerate(paths):
        row = index // columns
        column = index % columns
        x = column * tile_width + 10
        y = header + row * tile_height
        elements.append(_svg_text(
            x + (tile_width - 20) / 2.0, y + 28.0,
            path.stem, size=15, bold=True, anchor="middle"))
        relative = html.escape(path.name, quote=True)
        elements.append(
            f'<image x="{x:.1f}" y="{y + 40:.1f}" '
            f'width="{tile_width - 20}" height="{tile_height - 50}" '
            f'preserveAspectRatio="xMidYMid meet" href="{relative}" '
            f'xlink:href="{relative}"/>'
        )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return output


@dataclass
class _PreparedCandidate:
    utterance_id: str
    partition: str
    transcript: str
    frontend_text: str
    corpus_reading: str
    synthesis_reading: str
    source_path: Path
    source_sha256: str
    utterance: object
    alignment: object
    plan: object
    planned: tuple
    aligned: tuple[AlignedPlanPhone, ...]
    phrases: tuple[PhraseAlignment, ...]
    source_phones: tuple[str, ...]
    synthesis_phones: tuple[str, ...]
    alignment_report: Mapping[str, object]
    median_absolute_boundary_drift_ms: float


def render_alignment_verification(
    *,
    selection_path: Path | str,
    alignments_dir: Path | str,
    audio_dir: Path | str,
    voice_dir: Path | str,
    output_dir: Path | str,
    partitions: Sequence[str] = ("validation", "test"),
    frontend_mode: str = "openjtalk",
    wsl_distro: str = "",
    per_partition: int = 5,
    max_phrases: int = 2,
    base_pitch_hz: float | None = None,
    fall_percent: float = 18.0,
    backend=None,
) -> dict[str, object]:
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    _safe_output(voice_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = load_japanese_runtime_metadata(voice_root)
    renderer = backend or _backend(voice_root, runtime, wsl_distro)
    core = _gui_core()
    pitch = float(base_pitch_hz or runtime.get("average_pitch_hz") or 180.0)
    wanted = {str(value) for value in partitions}
    phrase_limit = max(1, int(max_phrases))
    alignment_root = Path(alignments_dir)
    audio_root = Path(audio_dir)
    prepared: dict[str, _PreparedCandidate] = {}
    candidate_rows = []
    skipped = []
    for record in load_selection(selection_path):
        if record.partition not in wanted:
            continue
        alignment_path = alignment_root / f"{record.utterance_id}.json"
        audio_candidates = sorted(audio_root.glob(f"{record.utterance_id}.*"))
        if not alignment_path.is_file() or not audio_candidates:
            skipped.append({"utterance_id": record.utterance_id,
                            "reason": "alignment_or_audio_missing"})
            continue
        alignment = load_alignment(alignment_path)
        if not alignment.accepted:
            skipped.append({"utterance_id": record.utterance_id,
                            "reason": "alignment_not_accepted"})
            continue
        source_path = audio_candidates[0].resolve()
        source_audio = read_audio(
            source_path, expected_sample_rate=alignment.sample_rate)
        alignment = refine_phrase_pauses(
            alignment, source_audio.samples, source_audio.sample_rate)
        frontend_text = normalize_kokoro_frontend_text(record.transcript)
        utterance = analyze_japanese(frontend_text, mode=frontend_mode)
        plan = create_synthesis_plan(
            utterance, runtime_metadata=runtime,
            duration_model="contextual", base_pitch_hz=pitch,
        )
        planned = final_plan_phone_timings(utterance, plan)
        source_phones, synthesis_phones = canonical_phone_sequences(
            alignment, planned)
        if source_phones != synthesis_phones:
            mismatch_index = next((
                index for index, (left, right) in enumerate(zip(
                    source_phones, synthesis_phones)) if left != right
            ), min(len(source_phones), len(synthesis_phones)))
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "linguistic_phone_sequence_mismatch",
                "source_phone_count": len(source_phones),
                "synthesis_phone_count": len(synthesis_phones),
                "first_mismatch_index": mismatch_index,
                "source_phone": (
                    source_phones[mismatch_index]
                    if mismatch_index < len(source_phones) else None
                ),
                "synthesis_phone": (
                    synthesis_phones[mismatch_index]
                    if mismatch_index < len(synthesis_phones) else None
                ),
            })
            continue
        aligned, report = align_final_plan(
            record.utterance_id, record.partition, alignment, planned)
        if (len(aligned) != len(source_phones) or
                float(report.get("match_fraction") or 0.0) != 1.0):
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "exact_alignment_invariant_failed",
                **dict(report),
            })
            continue
        try:
            phrases = split_phrase_alignments(aligned, alignment)
        except ValueError as error:
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "phrase_segmentation_mismatch",
                "detail": str(error),
            })
            continue
        if len(phrases) > phrase_limit:
            skipped.append({
                "utterance_id": record.utterance_id,
                "reason": "too_many_phrases_for_visual_audit",
                "phrase_count": len(phrases),
                "maximum_phrase_count": phrase_limit,
            })
            continue
        accepted = [row for row in aligned if not row.reference_rejected]
        if (len(accepted) < 3 or
                report.get("first_reference_phone_index") is None):
            skipped.append({"utterance_id": record.utterance_id,
                            "reason": "insufficient_matched_phone_context"})
            continue
        drift = _finite_median([
            abs(row.boundary_drift_seconds) * 1000.0 for row in accepted
        ])
        item = _PreparedCandidate(
            utterance_id=record.utterance_id,
            partition=record.partition,
            transcript=str(record.transcript),
            frontend_text=frontend_text,
            corpus_reading=str(record.reading),
            synthesis_reading=str(utterance.normalized_reading),
            source_path=source_path,
            source_sha256=_sha256(source_path),
            utterance=utterance,
            alignment=alignment,
            plan=plan,
            planned=tuple(planned),
            aligned=tuple(aligned),
            phrases=phrases,
            source_phones=source_phones,
            synthesis_phones=synthesis_phones,
            alignment_report=dict(report),
            median_absolute_boundary_drift_ms=drift,
        )
        prepared[item.utterance_id] = item
        candidate_rows.append({
            "utterance_id": item.utterance_id,
            "partition": item.partition,
            "median_absolute_boundary_drift_ms": drift,
            "match_fraction": float(report.get("match_fraction") or 0.0),
            "phrase_count": len(item.phrases),
        })

    selected_rows = select_alignment_candidates(
        candidate_rows, per_partition=per_partition,
        max_phrases=phrase_limit)
    examples = []
    plot_paths = []
    failures = []
    for selection in selected_rows:
        item = prepared[str(selection["utterance_id"])]
        before_hash = _sha256(item.source_path)
        systems = build_pitch_systems(
            item.utterance, item.plan, base_pitch_hz=pitch,
            fall_percent=fall_percent)
        details = systems["contextual_pitch"]
        try:
            rendered = renderer.synth_phones(
                item.plan.phones,
                "japanese_duration_ab",
                speed=1.0,
                text=item.frontend_text,
                lang="ja",
                seg_durs=item.plan.segment_durations,
                pitch=pitch,
                fall=fall_percent,
                pitch_targets=details["render_targets"],
                ground_truth_targets=details["raw_targets"],
                intonation_blocks=details["intonation_blocks"],
                pitch_mode="intonation",
                unit_overrides=item.plan.unit_overrides,
            )
            apply_vowel_realizations(
                rendered, item.plan, mode="contextual", renderer="auto")
            wav_name = f"{item.partition}__{item.utterance_id}.wav"
            plot_name = f"{item.partition}__{item.utterance_id}.svg"
            core.write_wav(
                str(output_root / wav_name), rendered.samples, rendered.sr)
            audio = read_audio(item.source_path, expected_sample_rate=22050)
            reference = _reference_phones(item.alignment)
            first_reference = int(
                item.alignment_report["first_reference_phone_index"])
            first_plan = int(item.alignment_report["first_planned_phone_index"])
            source_origin = float(reference[first_reference].start_seconds)
            synthesis_origin = float(item.planned[first_plan].start_seconds)
            accepted = [row for row in item.aligned
                        if not row.reference_rejected]
            phone_mae = float(statistics.fmean(
                row.absolute_duration_error_seconds * 1000.0
                for row in accepted))
            subtitle = (
                f"{item.partition} / {selection['selection_tier']} | "
                f"exactly matched {len(item.aligned)} phones | "
                f"phone MAE {phone_mae:.1f} ms | median absolute boundary "
                f"drift {item.median_absolute_boundary_drift_ms:.1f} ms"
            )
            timing_summary = summarize_phrase_alignment(item.phrases)
            edge_acoustics = analyze_phrase_edge_acoustics(
                phrases=item.phrases,
                source_samples=audio.samples,
                source_sample_rate=audio.sample_rate,
                synthesis_samples=rendered.samples,
                synthesis_sample_rate=rendered.sr,
                source_origin_seconds=source_origin,
                synthesis_origin_seconds=synthesis_origin,
                utterance=item.utterance,
            )
            edge_summary = summarize_edge_acoustics(edge_acoustics)
            annotations = (
                f"Text: {item.transcript}",
                f"Corpus reading: {item.corpus_reading}",
                f"Synthesis reading: {item.synthesis_reading}",
                "Linguistic phones: exact canonical match (" +
                " ".join(item.source_phones) + ")",
                (
                    "Timing split: total ratio "
                    f"{timing_summary['total_rate_ratio']:.3f}; active speech "
                    f"{timing_summary['active_speech_ratio']:.3f}; "
                    "inter-phrase pause error "
                    f"{timing_summary['interphrase_pause_error_ms']:.1f} ms"
                ),
                "Acoustic phrase edges: " + "; ".join(
                    f"{name} n={values['count']}, edge lead "
                    f"{values['median_synthesis_excess_extension_ms']:.1f} ms, "
                    "effective first-mora excess "
                    f"{values['median_effective_first_mora_excess_ms']:.1f} ms"
                    for name, values in edge_summary.items()
                ),
            )
            plot = render_alignment_plot(
                output_root / plot_name,
                source_samples=audio.samples,
                source_sample_rate=audio.sample_rate,
                synthesis_samples=rendered.samples,
                synthesis_sample_rate=rendered.sr,
                source_origin_seconds=source_origin,
                synthesis_origin_seconds=synthesis_origin,
                aligned_phones=item.aligned,
                title=f"Held-out waveform alignment: {item.utterance_id}",
                subtitle=subtitle,
                annotation_lines=annotations,
                fit_to_rows=True,
            )
            plot_paths.append(plot)
            phrase_plots = []
            for phrase_number, phrase in enumerate(item.phrases, start=1):
                local_rows = phrase.localized_rows()
                metrics = phrase.to_metrics()
                phrase_plot_name = (
                    f"{item.partition}__{item.utterance_id}__"
                    f"phrase_{phrase_number:02d}.svg"
                )
                phrase_plot = render_alignment_plot(
                    output_root / phrase_plot_name,
                    source_samples=audio.samples,
                    source_sample_rate=audio.sample_rate,
                    synthesis_samples=rendered.samples,
                    synthesis_sample_rate=rendered.sr,
                    source_origin_seconds=(
                        source_origin + phrase.reference_start_seconds),
                    synthesis_origin_seconds=(
                        synthesis_origin + phrase.synthesis_start_seconds),
                    aligned_phones=local_rows,
                    title=(
                        f"Phrase-local alignment {phrase_number}: "
                        f"{item.utterance_id}"
                    ),
                    subtitle=(
                        f"{metrics['phone_count']} phones | source "
                        f"{metrics['source_duration_seconds']:.3f} s | "
                        f"synthesis {metrics['synthesis_duration_seconds']:.3f} s | "
                        f"ratio {metrics['synthesis_to_source_ratio']:.3f} | "
                        "end drift "
                        f"{metrics['phrase_end_drift_ms']:.1f} ms"
                    ),
                    annotation_lines=annotations[:4] + (
                        "Acoustic edges: initial synth excess "
                        f"{edge_acoustics[phrase_number - 1]['initial'].get('synthesis_excess_extension_ms')} ms; "
                        "effective first-mora excess "
                        f"{edge_acoustics[phrase_number - 1]['effective_initial_mora_duration'].get('synthesis_excess_ms')} ms; "
                        "final synth excess "
                        f"{edge_acoustics[phrase_number - 1]['final'].get('synthesis_excess_extension_ms')} ms",
                    ),
                    fit_to_rows=True,
                    timeline_label=(
                        "Seconds after this phrase's first matched phone"
                    ),
                )
                phrase_plots.append({
                    "plot": phrase_plot.name,
                    **metrics,
                })
            after_hash = _sha256(item.source_path)
            if before_hash != after_hash or after_hash != item.source_sha256:
                raise RuntimeError("source recording hash changed during audit")
            examples.append({
                **dict(selection),
                "source_audio_file": item.source_path.name,
                "source_sha256": item.source_sha256,
                "synthesis_wav": wav_name,
                "plot": plot_name,
                "phrase_plots": phrase_plots,
                "source_sample_rate": audio.sample_rate,
                "synthesis_sample_rate": rendered.sr,
                "source_origin_seconds": round(source_origin, 6),
                "synthesis_origin_seconds": round(synthesis_origin, 6),
                "matched_phone_count": len(item.aligned),
                "linguistic_phone_sequence_exact": True,
                "transcript": item.transcript,
                "corpus_reading": item.corpus_reading,
                "synthesis_reading": item.synthesis_reading,
                "source_phones": list(item.source_phones),
                "synthesis_phones": list(item.synthesis_phones),
                "phone_duration_mae_ms": round(phone_mae, 6),
                "phrase_timing": timing_summary,
                "phrase_edge_acoustics": list(edge_acoustics),
                "phrase_edge_summary": edge_summary,
                "duration_model_id": item.plan.duration_model_id,
                "pitch_model_id": item.plan.pitch_model_id,
                "source_hash_unchanged": True,
                "phones": [row.to_dict() for row in item.aligned],
            })
        except Exception as error:
            failures.append({
                "utterance_id": item.utterance_id,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    overview = _overview(
        plot_paths, output_root / "alignment_verification_overview.svg")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prompt20_source_synthesis_waveform_alignment_audit",
        "timeline_alignment": {
            "global": "first_equal_phone_start",
            "phrase_local": "each_phrase_first_equal_phone_start",
        },
        "selection_method": (
            "exact linguistic phone sequences only; deterministic "
            "timing-error quantiles per partition; no example exceeds the "
            "configured phrase limit"
        ),
        "maximum_phrase_count": phrase_limit,
        "partitions": sorted(wanted),
        "candidate_count": len(candidate_rows),
        "example_count": len(examples),
        "render_failure_count": len(failures),
        "source_bank_or_corpus_write_performed": False,
        "boundaries_are_silver_reference": True,
        "acoustic_naturalness_verified": False,
        "overview_plot": overview.name if overview else None,
        "skipped": skipped,
        "failures": failures,
        "examples": examples,
    }
    _safe_json(output_root / "alignment_verification.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Kokoro-source versus Festival waveform alignments")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--alignments", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", action="append",
                        choices=("train", "validation", "test"))
    parser.add_argument("--frontend", default="openjtalk",
                        choices=("openjtalk", "auto", "kana"))
    parser.add_argument("--wsl-distro", default="")
    parser.add_argument("--per-partition", type=int, default=5)
    parser.add_argument("--max-phrases", type=int, default=2)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--fall", type=float, default=18.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = render_alignment_verification(
        selection_path=args.selection,
        alignments_dir=args.alignments,
        audio_dir=args.audio,
        voice_dir=args.voice,
        output_dir=args.output,
        partitions=args.partition or ("validation", "test"),
        frontend_mode=args.frontend,
        wsl_distro=args.wsl_distro,
        per_partition=args.per_partition,
        max_phrases=args.max_phrases,
        base_pitch_hz=args.pitch,
        fall_percent=args.fall,
    )
    print(f"Rendered {report['example_count']} source/synthesis alignment "
          f"images; failures: {report['render_failure_count']}.")
    print("Kokoro boundaries are silver reference; naturalness is unverified.")
    return 1 if report["render_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
