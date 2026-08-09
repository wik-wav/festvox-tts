"""Small dependency-free SVG diagnostics for Prompt 20 Stage A."""

from __future__ import annotations

from collections import defaultdict
import html
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from formant_analysis import FormantFrame, FormantSegment


_COLORS = {
    "a": "#c94f43",
    "i": "#3f77b5",
    "u": "#5b9b57",
    "e": "#9a5aa0",
    "o": "#d18c35",
    "unknown": "#777777",
}


def _svg_document(title: str, body: str, *, explanation: str = "") -> str:
    escaped_title = html.escape(title)
    escaped_explanation = html.escape(explanation)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="720" viewBox="0 0 1120 720">
<rect width="1120" height="720" fill="#fafafa"/>
<style>
text {{ font-family: Segoe UI, Arial, sans-serif; fill: #252525; }}
.title {{ font-size: 22px; font-weight: 600; }}
.axis {{ stroke: #454545; stroke-width: 1; }}
.grid {{ stroke: #dedede; stroke-width: 1; }}
.tick {{ font-size: 12px; fill: #555; }}
.note {{ font-size: 13px; fill: #444; }}
.legend {{ font-size: 12px; }}
</style>
<text x="70" y="38" class="title">{escaped_title}</text>
<text x="70" y="64" class="note">{escaped_explanation}</text>
{body}
</svg>\n'''


def _write(path: Path | str, title: str, body: str, explanation: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _svg_document(title, body, explanation=explanation), encoding="utf-8"
    )
    return destination


def _scale(value: float, lower: float, upper: float,
           screen_lower: float, screen_upper: float) -> float:
    if upper <= lower:
        return (screen_lower + screen_upper) / 2.0
    ratio = (value - lower) / (upper - lower)
    return screen_lower + ratio * (screen_upper - screen_lower)


def _finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def _axes(x_min: float, x_max: float, y_min: float, y_max: float,
          *, x_label: str, y_label: str, reverse_x=False,
          reverse_y=False) -> str:
    left, right, top, bottom = 90.0, 1060.0, 95.0, 650.0
    lines = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        xv = x_max - fraction * (x_max - x_min) if reverse_x else \
            x_min + fraction * (x_max - x_min)
        yv = y_max - fraction * (y_max - y_min) if reverse_y else \
            y_min + fraction * (y_max - y_min)
        lines.extend([
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" class="grid"/>',
            f'<text x="{x:.2f}" y="670" text-anchor="middle" class="tick">{xv:.0f}</text>',
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>',
            f'<text x="80" y="{y + 4:.2f}" text-anchor="end" class="tick">{yv:.0f}</text>',
        ])
    lines.append(
        f'<text x="575" y="704" text-anchor="middle" class="note">{html.escape(x_label)}</text>'
    )
    lines.append(
        f'<text x="24" y="370" text-anchor="middle" class="note" transform="rotate(-90 24 370)">{html.escape(y_label)}</text>'
    )
    return "\n".join(lines)


def write_f1_f2_vowel_space(path: Path | str,
                            segments: Sequence[FormantSegment]) -> Path:
    rows = [row for row in segments if row.accepted and
            row.segment.vowel in _COLORS and
            row.median_formants_hz[0] is not None and
            row.median_formants_hz[1] is not None]
    f1 = _finite(row.median_formants_hz[0] for row in rows)
    f2 = _finite(row.median_formants_hz[1] for row in rows)
    if not rows:
        return _write(path, "F1-F2 vowel space", "",
                      "No accepted labelled-vowel measurements.")
    x_min, x_max = min(f2) * 0.9, max(f2) * 1.1
    y_min, y_max = min(f1) * 0.9, max(f1) * 1.1
    body = [_axes(x_min, x_max, y_min, y_max,
                  x_label="F2 (Hz, high to low)",
                  y_label="F1 (Hz, high to low)",
                  reverse_x=True, reverse_y=True)]
    for row in rows:
        x = _scale(float(row.median_formants_hz[1]), x_max, x_min, 90, 1060)
        y = _scale(float(row.median_formants_hz[0]), y_max, y_min, 95, 650)
        color = _COLORS[row.segment.vowel]
        body.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" fill-opacity="0.62"><title>{html.escape(row.segment.segment_id)}</title></circle>'
        )
    for index, vowel in enumerate("aiueo"):
        body.append(
            f'<circle cx="{910 + (index % 3) * 70}" cy="{82 + (index // 3) * 18}" r="5" fill="{_COLORS[vowel]}"/>'
            f'<text x="{920 + (index % 3) * 70}" y="{86 + (index // 3) * 18}" class="legend">/{vowel}/</text>'
        )
    return _write(
        path, "F1-F2 vowel space", "\n".join(body),
        "Each point is one accepted stable vowel segment; axes follow the conventional reversed vowel-space orientation.",
    )


def write_group_metric_plot(
    path: Path | str,
    title: str,
    groups: Mapping[str, Sequence[float]],
    *,
    y_label: str,
    explanation: str,
) -> Path:
    cleaned = {key: _finite(values) for key, values in groups.items()}
    cleaned = {key: values for key, values in sorted(cleaned.items()) if values}
    all_values = [value for values in cleaned.values() for value in values]
    if not all_values:
        return _write(path, title, "", "No accepted measurements. " + explanation)
    y_min, y_max = min(all_values), max(all_values)
    margin = max(1.0, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin
    left, right, top, bottom = 90.0, 1060.0, 95.0, 650.0
    body = [_axes(0.0, max(1.0, len(cleaned) - 1.0), y_min, y_max,
                  x_label="Group", y_label=y_label)]
    keys = list(cleaned)
    for group_index, key in enumerate(keys):
        x = left if len(keys) == 1 else _scale(
            group_index, 0, len(keys) - 1, left + 30, right - 30
        )
        values = cleaned[key]
        for value_index, value in enumerate(values):
            jitter = ((value_index * 37) % 17 - 8) * 1.15
            y = _scale(value, y_min, y_max, bottom, top)
            body.append(
                f'<circle cx="{x + jitter:.2f}" cy="{y:.2f}" r="3" fill="#527ea8" fill-opacity="0.42"/>'
            )
        median = float(np.median(values))
        y = _scale(median, y_min, y_max, bottom, top)
        body.extend([
            f'<line x1="{x - 25:.2f}" y1="{y:.2f}" x2="{x + 25:.2f}" y2="{y:.2f}" stroke="#b43d35" stroke-width="3"/>',
            f'<text x="{x:.2f}" y="686" text-anchor="middle" class="tick">{html.escape(key[:22])}</text>',
        ])
    return _write(path, title, "\n".join(body), explanation)


def write_f0_failure_plot(path: Path | str,
                          frames: Sequence[FormantFrame]) -> Path:
    rows = [frame for frame in frames if frame.f0_hz is not None]
    if not rows:
        return _write(path, "Estimator reliability versus F0", "",
                      "No frames with an F0 estimate.")
    x_min = min(float(row.f0_hz) for row in rows)
    x_max = max(float(row.f0_hz) for row in rows)
    body = [_axes(x_min, x_max, 0.0, 1.0, x_label="F0 (Hz)",
                  y_label="Accepted (1) / rejected (0)")]
    for index, row in enumerate(rows):
        x = _scale(float(row.f0_hz), x_min, x_max, 90, 1060)
        value = 1.0 if row.accepted else 0.0
        jitter = ((index * 29) % 19 - 9) * 0.004
        y = _scale(value + jitter, 0.0, 1.0, 650, 95)
        color = "#3d8b62" if row.accepted else "#bc4c43"
        body.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" fill-opacity="0.48"><title>{html.escape(";".join(row.rejection_reasons))}</title></circle>'
        )
    return _write(
        path, "Estimator reliability versus F0", "\n".join(body),
        "Green frames passed all reliability gates; red frames remain in the CSV with their rejection reasons.",
    )


def write_alignment_plot(path: Path | str, samples: Sequence[float],
                         sample_rate: int, phones: Sequence[object],
                         *, title: str) -> Path:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    width = 970
    if values.size:
        bins = min(width, values.size)
        edges = np.linspace(0, values.size, bins + 1, dtype=int)
        peaks = [float(np.max(np.abs(values[edges[i]:edges[i + 1]]),
                                   initial=0.0)) for i in range(bins)]
    else:
        peaks = []
    maximum = max(peaks, default=1.0) or 1.0
    body = ['<rect x="90" y="130" width="970" height="420" fill="#eeeeee"/>']
    for index, peak in enumerate(peaks):
        x = 90 + index * 970 / max(1, len(peaks) - 1)
        half = 190 * peak / maximum
        body.append(
            f'<line x1="{x:.2f}" y1="{340 - half:.2f}" x2="{x:.2f}" y2="{340 + half:.2f}" stroke="#447db4" stroke-width="1"/>'
        )
    duration = values.size / max(1.0, float(sample_rate))
    for phone in phones:
        start = float(getattr(phone, "start_seconds"))
        x = _scale(start, 0.0, max(duration, 1e-9), 90, 1060)
        confidence = float(getattr(phone, "confidence", 0.0))
        color = "#2f7658" if confidence >= 0.6 else "#c27b2d"
        label = html.escape(str(getattr(phone, "raw_phone", "?")))
        body.extend([
            f'<polygon points="{x - 5:.2f},112 {x + 5:.2f},112 {x:.2f},124" fill="{color}"/>',
            f'<line x1="{x:.2f}" y1="125" x2="{x:.2f}" y2="555" stroke="{color}" stroke-width="1" stroke-dasharray="3 3"/>',
            f'<text x="{x + 3:.2f}" y="575" class="tick">{label}</text>',
        ])
    return _write(
        path, title, "\n".join(body),
        "Blue is the decoded waveform. Green boundary triangles have confidence >= 0.60; orange boundaries are lower-confidence silver labels.",
    )


def write_stage_a_plot_suite(output: Path | str,
                             segments: Sequence[FormantSegment]) -> list[Path]:
    root = Path(output)
    accepted = [row for row in segments if row.accepted]
    frames = [frame for row in segments for frame in row.frames]
    paths = [write_f1_f2_vowel_space(root / "f1_f2_vowel_space.svg", segments)]
    groups = {
        f"/{vowel}/ F{number}": [
            row.median_formants_hz[number - 1] for row in accepted
            if row.segment.vowel == vowel
        ]
        for vowel in "aiueo" for number in range(1, 5)
    }
    paths.append(write_group_metric_plot(
        root / "f1_f4_by_vowel.svg", "F1 through F4 by vowel", groups,
        y_label="Frequency (Hz)",
        explanation="Dots are accepted segment medians; red bars are group medians.",
    ))
    paths.append(write_group_metric_plot(
        root / "dispersion_by_speaker.svg", "Formant dispersion by speaker",
        {speaker: [row.median_formant_dispersion_hz for row in accepted
                   if row.segment.speaker_id == speaker]
         for speaker in sorted({row.segment.speaker_id for row in accepted})},
        y_label="Mean adjacent-formant spacing (Hz)",
        explanation="Speaker-relative spacing is an acoustic diagnostic, not anatomical tract length.",
    ))
    source = [row.median_formant_dispersion_hz for row in accepted
              if row.segment.speaker_id == "project_source_speaker" and
              row.median_formant_dispersion_hz is not None]
    source_median = float(np.median(source)) if source else 1.0
    paths.append(write_group_metric_plot(
        root / "formants_relative_to_source.svg",
        "Resonance spacing relative to source",
        {speaker: [source_median / float(row.median_formant_dispersion_hz)
                   for row in accepted
                   if row.segment.speaker_id == speaker and
                   row.median_formant_dispersion_hz]
         for speaker in sorted({row.segment.speaker_id for row in accepted})},
        y_label="Apparent tract ratio (source dispersion / reference dispersion)",
        explanation="Values above 1 correspond to lower reference resonance spacing under the simplified uniform model.",
    ))
    paths.append(write_group_metric_plot(
        root / "f0_vs_formant_estimates.svg", "F0 distribution by vowel",
        {vowel: [row.median_f0_hz for row in accepted
                 if row.segment.vowel == vowel] for vowel in "aiueo"},
        y_label="F0 (Hz)",
        explanation="Use with the reliability plot to inspect sparse-harmonic estimator behavior.",
    ))
    paths.append(write_f0_failure_plot(root / "estimator_failures_vs_f0.svg",
                                       frames))
    paths.append(write_group_metric_plot(
        root / "phrase_medial_vs_final.svg", "Phrase-medial versus final vowels",
        {"medial": [row.median_formant_dispersion_hz for row in accepted
                    if not row.segment.phrase_final],
         "final": [row.median_formant_dispersion_hz for row in accepted
                   if row.segment.phrase_final]},
        y_label="Formant dispersion (Hz)",
        explanation="Phrase-final status comes from aligned phone and pause context.",
    ))
    paths.append(write_group_metric_plot(
        root / "modal_vs_creaky.svg", "Modal versus irregular frames",
        {"modal": [frame.creak_confidence for frame in frames
                   if frame.creak_confidence < 0.45],
         "irregular": [frame.creak_confidence for frame in frames
                       if frame.creak_confidence >= 0.45]},
        y_label="Creak / periodicity-ambiguity confidence",
        explanation="Irregular frames remain visible and are excluded when they exceed the analysis threshold.",
    ))
    paths.append(write_group_metric_plot(
        root / "short_vs_long.svg", "Short versus long vowels",
        {"short": [row.median_formant_dispersion_hz for row in accepted
                   if not row.segment.long_vowel],
         "long": [row.median_formant_dispersion_hz for row in accepted
                  if row.segment.long_vowel]},
        y_label="Formant dispersion (Hz)",
        explanation="Long-vowel identity is structural metadata, not inferred from raw duration alone.",
    ))
    paths.append(write_group_metric_plot(
        root / "control_range_distribution.svg",
        "Distributions used for tract-range derivation",
        {vowel: [row.median_formant_dispersion_hz for row in accepted
                 if row.segment.vowel == vowel] for vowel in "aiueo"},
        y_label="Formant dispersion (Hz)",
        explanation="Runtime bounds use robust per-vowel intersections rather than raw extrema or a binary gender preset.",
    ))
    return paths
