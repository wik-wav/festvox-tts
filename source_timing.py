"""Deterministic source-unit timing summaries for generated voices.

UTAU aliases often contain a held vowel tail or a long silence collar.  Those
regions are useful source material for UniSyn but are not, by themselves,
linguistic phone-duration targets.  This module records the typical incoming
and outgoing diphone-half geometry for each phone so an occurrence can retain
the bank's relative timing without treating a sustained note as natural
speech duration.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


SOURCE_TIMING_PROFILE_SCHEMA_VERSION = 1
_SILENCE_PHONES = {"pau", "sil", "sp"}


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, object]:
    cleaned = sorted(
        float(value) for value in values
        if math.isfinite(float(value)) and 0.001 <= float(value) <= 2.0
    )
    if not cleaned:
        return {
            "count": 0,
            "median_seconds": None,
            "mad_seconds": None,
            "p10_seconds": None,
            "p90_seconds": None,
        }
    median = float(statistics.median(cleaned))
    mad = float(statistics.median(abs(value - median) for value in cleaned))
    return {
        "count": len(cleaned),
        "median_seconds": round(median, 9),
        "mad_seconds": round(mad, 9),
        "p10_seconds": round(_quantile(cleaned, 0.10), 9),
        "p90_seconds": round(_quantile(cleaned, 0.90), 9),
    }


def _source_slice(row: Mapping[str, object]) \
        -> tuple[float, float, float] | None:
    source = row.get("source_slice")
    if not isinstance(source, Mapping):
        source = {
            "start": row.get("start"),
            "phone_boundary": row.get("mid"),
            "end": row.get("end"),
        }
    try:
        start = float(source["start"])
        middle = float(source["phone_boundary"])
        end = float(source["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (start, middle, end)):
        return None
    if not start < middle < end:
        return None
    return start, middle, end


def build_source_timing_profile(
    alternatives: Mapping[str, object],
) -> dict[str, object]:
    """Summarize non-silence diphone halves without reading source WAVs.

    ``incoming`` is the right half of ``previous-current``. ``outgoing`` is
    the left half of ``current-following``. Silence-neighbour edges are
    excluded because their collars frequently encode arbitrary recording
    padding rather than speech timing.
    """
    incoming: dict[str, list[float]] = {}
    outgoing: dict[str, list[float]] = {}
    rejected = 0
    accepted = 0
    for diphone in sorted(alternatives):
        parts = str(diphone).split("-", 1)
        if len(parts) != 2:
            rejected += 1
            continue
        left, right = parts
        rows = alternatives[diphone]
        if not isinstance(rows, (list, tuple)):
            rejected += 1
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                rejected += 1
                continue
            source = _source_slice(row)
            if source is None:
                rejected += 1
                continue
            start, middle, end = source
            if left not in _SILENCE_PHONES and right not in _SILENCE_PHONES:
                outgoing.setdefault(left, []).append(middle - start)
                incoming.setdefault(right, []).append(end - middle)
                accepted += 1

    phones = sorted(set(incoming) | set(outgoing))
    return {
        "schema_version": SOURCE_TIMING_PROFILE_SCHEMA_VERSION,
        "method": "median_non_silence_diphone_halves_v1",
        "silence_neighbor_edges_excluded": True,
        "accepted_unit_count": accepted,
        "rejected_unit_count": rejected,
        "phones": {
            phone: {
                "incoming": _summary(incoming.get(phone, ())),
                "outgoing": _summary(outgoing.get(phone, ())),
            }
            for phone in phones
        },
    }


def source_timing_profile(
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Return serialized profile data or derive it for an older voice."""
    profile = runtime.get("source_timing_profile")
    if (isinstance(profile, Mapping)
            and int(profile.get("schema_version") or 0) ==
            SOURCE_TIMING_PROFILE_SCHEMA_VERSION
            and isinstance(profile.get("phones"), Mapping)):
        return dict(profile)
    alternatives = runtime.get("alternatives") or runtime.get("diphones")
    if not isinstance(alternatives, Mapping):
        alternatives = {}
    return build_source_timing_profile(alternatives)


def profile_half_seconds(
    profile: Mapping[str, object],
    phone: str,
    side: str,
) -> float | None:
    """Read one robust half-duration, returning ``None`` when unavailable."""
    phones = profile.get("phones")
    if not isinstance(phones, Mapping):
        return None
    row = phones.get(str(phone))
    if not isinstance(row, Mapping):
        return None
    summary = row.get(str(side))
    if not isinstance(summary, Mapping):
        return None
    try:
        value = float(summary["median_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.001 else None
